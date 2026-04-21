#include "../../include/backend/tensor_cuda.hpp"
#include <cuda/cmath>
#include <curand_kernel.h>

static constexpr u32 TILE = 16;
static constexpr u32 N_THREADS = 256;

// ── Functors
// ────────────────────────────────────────────────────────────────── Small
// device-callable structs passed as template parameters to generic kernels so
// the compiler inlines the operation and generates a single fused kernel per
// op.

struct AddOp {
    __device__ f32 operator()(f32 a, f32 b) const { return a + b; }
};
struct SubOp {
    __device__ f32 operator()(f32 a, f32 b) const { return a - b; }
};
struct MulOp {
    __device__ f32 operator()(f32 a, f32 b) const { return a * b; }
};
struct DivOp {
    __device__ f32 operator()(f32 a, f32 b) const { return a / b; }
};
struct EqualOp {
    __device__ f32 operator()(f32 a, f32 b) const { return a == b; }
};

struct ReluFwdOp {
    __device__ f32 operator()(f32 x) const { return x > 0.0f ? x : 0.0f; }
};
struct ReluBackwardOp {
    __device__ f32 operator()(f32 grad, f32 input) const {
        return input > 0.0f ? grad : 0.0f;
    }
};
struct ExpOp {
    __device__ f32 operator()(f32 x) const { return expf(x); }
};
struct LogOp {
    __device__ f32 operator()(f32 x) const { return logf(x); }
};
struct SqrtOp {
    __device__ f32 operator()(f32 x) const { return sqrtf(x); }
};

// ── Kernels
// ───────────────────────────────────────────────────────────────────

// ---- fill / clear --------------------------------------------------------

__global__ void tensor_fill(u64 tensor_size, f32 *tensor_data, f32 value) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < tensor_size)
        tensor_data[workIdx] = value;
}

// ---- activations (relu, exp) ---------------------------------------------

// Fast path: both tensors contiguous, flat index maps directly to memory.
template <typename Op>
__global__ void elementwise_unary(u64 size, f32 *out, const f32 *a, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx]);
}

// Strided path: workIdx is a logical row-major index decomposed via
// out_contig's contiguous strides. out_actual and a_meta carry the real
// (potentially non-contiguous) strides used to compute physical offsets.
template <typename Op>
__global__ void
elementwise_unary_strided(TensorMeta out_contig, TensorMeta out_actual,
                          TensorMeta a_meta, f32 *out, const f32 *a, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset]);
}

// ---- elementwise binary (add / sub / mul / div) --------------------------

template <typename Op>
__global__ void elementwise_binary(u64 size, f32 *out, f32 *a, f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b[workIdx]);
}

// Broadcast binary op: each thread handles one output element.
// out_contig has contiguous row-major strides of the output shape — used by
// offset_from() to decompose workIdx into per-dim indices. out_actual carries
// out's real (possibly non-contiguous) strides so the write lands at the
// correct physical address. a_meta/b_meta use stride=0 on broadcast dims.
template <typename Op>
__global__ void
elementwise_broadcast(const TensorMeta out_contig, const TensorMeta out_actual,
                      const TensorMeta a_meta, const TensorMeta b_meta,
                      f32 *out, const f32 *a, const f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    u64 b_offset = out_contig.offset_from(workIdx, b_meta);
    out[out_offset] = op(a[a_offset], b[b_offset]);
}

// ---- scalar operations ---------------------------------------------------

// Fast path (both contiguous).
template <typename Op>
__global__ void elementwise_binary(u64 size, f32 *out, const f32 *a, f32 b,
                                   Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b);
}

// Strided path for scalar ops when out or a is non-contiguous.
template <typename Op>
__global__ void elementwise_scalar_strided(TensorMeta out_contig,
                                           TensorMeta out_actual,
                                           TensorMeta a_meta, f32 *out,
                                           const f32 *a, f32 scalar, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset], scalar);
}

// ---- copy (strided, D2D) -------------------------------------------------

// Stride-aware D2D copy kernel. Same decomposition trick as the other strided
// kernels: out_contig decomposes workIdx, out_actual and src_meta give the
// physical write/read offsets.
__global__ void tensor_copy_strided(TensorMeta out_contig,
                                    TensorMeta out_actual, TensorMeta src_meta,
                                    f32 *dst, const f32 *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 dst_offset = out_contig.offset_from(workIdx, out_actual);
    u64 src_offset = out_contig.offset_from(workIdx, src_meta);
    dst[dst_offset] = src[src_offset];
}

// ---- matrix multiply -----------------------------------------------------

__global__ void mat_mul(TensorMeta out_meta, TensorMeta a_meta,
                        TensorMeta b_meta, f32 *out, f32 *a, f32 *b) {
    u64 myCol = threadIdx.x + blockIdx.x * blockDim.x;
    u64 myRow = threadIdx.y + blockIdx.y * blockDim.y;
    if (myRow >= out_meta.rows() || myCol >= out_meta.cols())
        return;
    f32 value = 0;
    for (u32 i = 0; i < a_meta.cols(); i++)
        value += a[a_meta.at(myRow, i)] * b[b_meta.at(i, myCol)];
    out[out_meta.at(myRow, myCol)] = value;
}

// Tiled matmul using shared memory. Each thread block computes a TILE×TILE
// output tile. The K dimension is split into TILE-wide strips; each strip is
// loaded cooperatively into shared memory (a_tile and b_tile) before the dot
// product is accumulated. __syncthreads() ensures all threads have written
// their tile element before anyone reads from it.
// blocks: x=row_tiles (up to 2^31-1), y=col_tiles (≤65535, always small).
// This avoids the 65535 grid.y limit for tall matrices like unfolded conv
// inputs.
__global__ void mat_mul_tiled(TensorMeta out_meta, TensorMeta a_meta,
                              TensorMeta b_meta, f32 *out, f32 *a, f32 *b,
                              f32 beta) {
    u64 myRow = threadIdx.y + blockIdx.x * blockDim.y;
    u64 myCol = threadIdx.x + blockIdx.y * blockDim.x;

    __shared__ f32 a_tile[TILE * TILE];
    __shared__ f32 b_tile[TILE * TILE];

    bool in_bounds = myRow < out_meta.rows() && myCol < out_meta.cols();
    u32 n_tiles = cuda::ceil_div(a_meta.cols(), TILE);

    f32 value = 0.0f;
    for (u32 n = 0; n < n_tiles; n++) {
        u32 col_a = n * TILE + threadIdx.x;
        u32 row_b = n * TILE + threadIdx.y;

        a_tile[threadIdx.y * TILE + threadIdx.x] =
            (myRow < out_meta.rows() && col_a < a_meta.cols())
                ? a[a_meta.at(myRow, col_a)]
                : 0.0f;

        b_tile[threadIdx.y * TILE + threadIdx.x] =
            (row_b < b_meta.rows() && myCol < b_meta.cols())
                ? b[b_meta.at(row_b, myCol)]
                : 0.0f;

        __syncthreads();

        for (u32 i = 0; i < TILE; i++)
            value +=
                a_tile[threadIdx.y * TILE + i] * b_tile[i * TILE + threadIdx.x];

        __syncthreads();
    }

    if (in_bounds)
        out[out_meta.at(myRow, myCol)] =
            value + beta * out[out_meta.at(myRow, myCol)];
}

// ---- reduction (sum, max, argmax) ----------------------------------------

// One step of a parallel tree reduction for sum.
// Each block reduces N_THREADS consecutive elements to a single value using
// shared memory. The result is written to out[blockIdx.x], so after this kernel
// the problem size shrinks from `size` to `gridDim.x` (number of blocks).
// Call repeatedly until gridDim.x == 1 to get the global sum.
__global__ void tensor_sum_step(u64 size, f32 *out, f32 *tensor) {
    __shared__ f32 partial[N_THREADS];
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    partial[threadIdx.x] = (workIdx < size) ? tensor[workIdx] : 0.0f;
    __syncthreads();
    for (u32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = partial[0];
}

__global__ void welford_mean_var(f32 *mean, f32 *var, const f32 *src,
                                 u64 stride_dim, TensorMeta meta_contig,
                                 TensorMeta meta_src, u64 other_dims_size) {
    __shared__ u32 partial_n[N_THREADS];
    __shared__ f32 partial_mu[N_THREADS];
    __shared__ f32 partial_M2[N_THREADS];

    u32 c = blockIdx.x;
    u64 base = c * stride_dim;
    f32 mu = 0.0f, M2 = 0.0f;
    u32 n = 0;

    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x) {
        f32 x = src[base + meta_contig.offset_from(i, meta_src)];
        n++;
        f32 delta = x - mu;
        mu += delta / n;
        M2 += delta * (delta - delta / n);
    }

    partial_n[threadIdx.x] = n;
    partial_mu[threadIdx.x] = mu;
    partial_M2[threadIdx.x] = M2;
    __syncthreads();

    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u32 na = partial_n[threadIdx.x], nb = partial_n[threadIdx.x + s];
            f32 ua = partial_mu[threadIdx.x], ub = partial_mu[threadIdx.x + s];
            f32 ma = partial_M2[threadIdx.x], mb = partial_M2[threadIdx.x + s];
            u32 nc = na + nb;
            f32 delta = ub - ua;
            partial_n[threadIdx.x] = nc;
            partial_mu[threadIdx.x] = ua + delta * ((f32)nb / nc);
            partial_M2[threadIdx.x] =
                ma + mb + delta * delta * ((f32)na * nb / nc);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mean[c] = partial_mu[0];
        var[c] = partial_M2[0] / (f32)(partial_n[0] - 1); // Bessel-corrected
    }
}

// Each thread owns one output element (one position in the reduced tensor).
// offset_from maps that output position to the first element along the
// reduction dim in the input. The loop then walks along that dim by stepping
// tensor_meta.stride[dim] in flat memory, summing all values.
__global__ void tensor_sum(TensorMeta out_meta, TensorMeta tensor_meta,
                           f32 *out, f32 *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    f32 sum = 0.0f;
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++)
        sum += tensor[base_offset + i * tensor_meta.stride[dim]];
    out[workIdx] = sum;
}

// Same parallel tree reduction as tensor_sum_step but uses fmaxf instead of
// addition. Identity element is -FLT_MAX so out-of-bounds threads don't affect
// the result.
__global__ void tensor_max_step(u64 size, f32 *out, f32 *tensor) {
    __shared__ f32 partial[N_THREADS];
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    partial[threadIdx.x] = (workIdx < size) ? tensor[workIdx] : -__FLT_MAX__;
    __syncthreads();
    for (u32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = partial[0];
}

__global__ void tensor_max(TensorMeta out_meta, TensorMeta tensor_meta,
                           f32 *out, f32 *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    f32 max_val = -__FLT_MAX__;
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++) {
        f32 val = tensor[base_offset + i * tensor_meta.stride[dim]];
        if (val > max_val)
            max_val = val;
    }
    out[workIdx] = max_val;
}

// Same structure as tensor_max (each thread owns one output slot, walks the
// reduction dim via stride). Also tracks the index i of the max value and
// stores it as a float in out[].
__global__ void tensor_argmax_kernel(TensorMeta out_meta,
                                     TensorMeta tensor_meta, f32 *out,
                                     f32 *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    f32 max_val = -__FLT_MAX__;
    u32 max_idx = 0;
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++) {
        f32 val = tensor[base_offset + i * tensor_meta.stride[dim]];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    out[workIdx] = (f32)max_idx;
}

// ---- scattering ----------------------------------------------------------
__global__ void tensor_scatter_add(TensorMeta out_meta,
                                   TensorMeta out_base_meta,
                                   TensorMeta indices_meta, TensorMeta src_meta,
                                   TensorMeta src_contig, f32 *out,
                                   f32 *indices, f32 *src, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= src_contig.size)
        return;

    u64 src_offset = src_contig.offset_from(workIdx, src_meta);
    u64 out_base_offset = src_contig.offset_from(workIdx, out_base_meta);
    u64 indices_offset = src_contig.offset_from(workIdx, indices_meta);
    out_base_offset += (u32)indices[indices_offset] * out_meta.stride[dim];

    atomicAdd(&out[out_base_offset], src[src_offset]);
}

// ---- indexing ------------------------------------------------------------

// Each thread handles one output element. The flat output index (workIdx) is
// decomposed into per-dimension coords using dst_meta strides. For every dim
// except the selected one the coord maps directly to the source. For dim, the
// coord is used as an index into the indices[] array to get the source row.
__global__ void tensor_index_select(TensorMeta dst_meta, TensorMeta src_meta,
                                    f32 *dst, const f32 *src,
                                    const u32 *indices, u32 n_indices,
                                    u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= dst_meta.size)
        return;
    u64 remaining = workIdx;
    u64 offset = 0;
    for (u32 i = 0; i < src_meta.ndim; i++) {
        u64 idx_i = remaining / dst_meta.stride[i];
        remaining -= idx_i * dst_meta.stride[i];
        offset += (i == dim) ? indices[idx_i] * src_meta.stride[i]
                             : idx_i * src_meta.stride[i];
    }
    dst[workIdx] = src[offset];
}

// dst shape: [N, L_h, L_w, C, kH, kW]
// src_virtual carries virtual strides that map each dst dimension directly to
// its contribution in the src address:
//   dim 0 (N)   → src.stride[N]
//   dim 1 (L_h) → stride_h * src.stride[H]
//   dim 2 (L_w) → stride_w * src.stride[W]
//   dim 3 (C)   → src.stride[C]
//   dim 4 (kH)  → src.stride[H]
//   dim 5 (kW)  → src.stride[W]
// The padding is absorbed into the src pointer itself (shifted back by
// pad_h*stride[H] + pad_w*stride[W]) so that kh=0/kw=0 maps to position
// (-pad_h, -pad_w) in the image. The bounds check below discards those
// accesses as zero before any invalid read occurs.

__global__ void tensor_unfold2d(TensorMeta dst_meta, // for index decomposition
                                TensorMeta dst_meta_contig, // for writing
                                TensorMeta src_meta, Unfold2dParams params,
                                f32 *dst, const f32 *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= dst_meta.size)
        return;

    // Decompose using CONTIGUOUS strides
    u64 remaining = workIdx;
    u32 idx[6];
    for (u32 i = 0; i < 6; i++) {
        idx[i] = remaining / dst_meta_contig.stride[i];
        remaining -= idx[i] * dst_meta_contig.stride[i];
    }

    u32 n = idx[0];
    u32 lh = idx[1];
    u32 lw = idx[2];
    u32 c = idx[3];
    u32 kh = idx[4];
    u32 kw = idx[5];

    i32 h = lh * params.stride_h + kh - params.pad_h;
    i32 w = lw * params.stride_w + kw - params.pad_w;

    // Compute destination offset using ACTUAL strides
    u64 dst_offset = n * dst_meta.stride[0] + lh * dst_meta.stride[1] +
                     lw * dst_meta.stride[2] + c * dst_meta.stride[3] +
                     kh * dst_meta.stride[4] + kw * dst_meta.stride[5];

    if (h < 0 || h >= src_meta.shape[2] || w < 0 || w >= src_meta.shape[3]) {
        dst[dst_offset] = params.pad_constant;
        return;
    }

    u64 src_offset = n * src_meta.stride[0] + c * src_meta.stride[1] +
                     h * src_meta.stride[2] + w * src_meta.stride[3];

    dst[dst_offset] = src[src_offset];
}

// Each thread owns one element of col [N, L_h, L_w, C, kH, kW].
// Decomposes its workIdx into (n,lh,lw,c,kh,kw), computes the spatial (h,w),
// and atomicAdd's its value into dst[n,c,h,w]. atomicAdd is required because
// overlapping windows (stride < kernel size) map multiple col entries to the
// same dst pixel.
__global__ void tensor_fold2d(TensorMeta col_meta, TensorMeta col_meta_contig,
                              TensorMeta dst_meta, Unfold2dParams params,
                              const f32 *col, f32 *dst) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= col_meta.size)
        return;

    u64 remaining = workIdx;
    u32 idx[6];
    for (u32 i = 0; i < 6; i++) {
        idx[i] = remaining / col_meta_contig.stride[i];
        remaining -= idx[i] * col_meta_contig.stride[i];
    }
    u32 n = idx[0];
    u32 lh = idx[1];
    u32 lw = idx[2];
    u32 c = idx[3];
    u32 kh = idx[4];
    u32 kw = idx[5];

    i32 h = (i32)(lh * params.stride_h + kh) - (i32)params.pad_h;
    i32 w = (i32)(lw * params.stride_w + kw) - (i32)params.pad_w;
    u64 col_offset = n * col_meta.stride[0] + lh * col_meta.stride[1] +
                     lw * col_meta.stride[2] + c * col_meta.stride[3] +
                     kh * col_meta.stride[4] + kw * col_meta.stride[5];

    if (h < 0 || h >= (i32)dst_meta.shape[2] || w < 0 ||
        w >= (i32)dst_meta.shape[3])
        return;

    u64 dst_off = (u64)n * dst_meta.stride[0] + (u64)c * dst_meta.stride[1] +
                  (u64)h * dst_meta.stride[2] + (u64)w * dst_meta.stride[3];
    atomicAdd(&dst[dst_off], col[col_offset]);
}

// ---- comparison ----------------------------------------------------------

// Each thread checks one element pair. All threads that find a mismatch write
// the same value (0), so races between them are harmless.
__global__ void tensor_check_close(u64 size, u32 *ok, const f32 *a,
                                   const f32 *b, f32 tol) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size && fabsf(a[workIdx] - b[workIdx]) > tol)
        *ok = 0;
}

// ── Template dispatch helpers
// ─────────────────────────────────────────────────

// Builds a TensorMeta whose strides are contiguous row-major (i.e. the
// strides you'd get from a freshly allocated tensor of the same shape).
// Used as the decomposition base in offset_from() so workIdx → per-dim
// indices is always a simple radix decomposition regardless of out's actual
// layout.
static TensorMeta make_contig_meta(const Tensor *t) {
    TensorMeta m(t);
    tensor_compute_strides(m.stride, m.shape, m.ndim);
    return m;
}

template <typename Op>
static void cuda_elementwise_unary(Tensor *out, const Tensor *a, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        elementwise_unary<<<blocks, threads>>>(out->size, out->data, a->data,
                                               op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise_unary_strided<<<blocks, threads>>>(
        out_contig, out_actual, a_meta, out->data, a->data, op);
}

// Fast path: all three tensors same shape and contiguous — plain flat
// kernel. Strided path: builds a contiguous decomposition meta for out
// (out_contig) and passes out's actual strides (out_actual) so the write
// goes to the right slot. a_meta/b_meta are broadcast-expanded (stride=0 on
// dims where size==1).
template <typename Op>
static void cuda_elementwise_binary(Tensor *out, const Tensor *a,
                                    const Tensor *b, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(out) &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        elementwise_binary<<<blocks, threads>>>(out->size, out->data, a->data,
                                                b->data, op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a, out->shape, out->ndim);
    TensorMeta b_meta(b, out->shape, out->ndim);
    elementwise_broadcast<<<blocks, threads>>>(out_contig, out_actual, a_meta,
                                               b_meta, out->data, a->data,
                                               b->data, op);
}

template <typename Op>
static void cuda_elementwise_scalar(Tensor *out, const Tensor *a, f32 scalar,
                                    Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        elementwise_binary<<<blocks, threads>>>(out->size, out->data, a->data,
                                                scalar, op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise_scalar_strided<<<blocks, threads>>>(
        out_contig, out_actual, a_meta, out->data, a->data, scalar, op);
}

// ── TensorMeta
// ────────────────────────────────────────────────────────────────

TensorMeta::TensorMeta(const Tensor *t) : size(t->size), ndim(t->ndim) {
    memcpy(shape, t->shape, ndim * sizeof(u32));
    memcpy(stride, t->stride, ndim * sizeof(u64));
}

TensorMeta::TensorMeta(const Tensor *t, const u32 *bcast_shape, u32 bcast_ndim)
    : size(t->size), ndim(bcast_ndim) {
    expanded_shape(t, bcast_ndim, shape);
    expanded_stride(t, bcast_ndim, stride);
}

// ── Host functions
// ────────────────────────────────────────────────────────────

// ---- memory management (alloc / free / transfers)
// ------------------------

void tensor_cuda_alloc(Tensor *tensor) {
    cudaMalloc(&tensor->data, sizeof(f32) * tensor->size);
    cudaMemset(tensor->data, 0, sizeof(f32) * tensor->size);
}

void tensor_cuda_free(Tensor *tensor) { cudaFree(tensor->data); }

Tensor *tensor_cuda_to_gpu(const Tensor *t_cpu) {
    Tensor *t_gpu = new Tensor(t_cpu->ndim, t_cpu->shape, true);
    memcpy(t_gpu->stride, t_cpu->stride, t_cpu->ndim * sizeof(u64));
    cudaMemcpy(t_gpu->data, t_cpu->data, t_cpu->size * sizeof(f32),
               cudaMemcpyHostToDevice);
    return t_gpu;
}

Tensor *tensor_cuda_to_cpu(const Tensor *t_gpu) {
    Tensor *t_cpu = new Tensor(t_gpu->ndim, t_gpu->shape, false);
    memcpy(t_cpu->stride, t_gpu->stride, t_gpu->ndim * sizeof(u64));
    cudaMemcpy(t_cpu->data, t_gpu->data, t_gpu->size * sizeof(f32),
               cudaMemcpyDeviceToHost);
    return t_cpu;
}

Tensor *tensor_cuda_copy(const Tensor *t_gpu) {
    Tensor *copy = new Tensor(t_gpu->ndim, t_gpu->shape, true);
    memcpy(copy->stride, t_gpu->stride, t_gpu->ndim * sizeof(u64));
    cudaMemcpy(copy->data, t_gpu->data, t_gpu->size * sizeof(f32),
               cudaMemcpyDeviceToDevice);
    return copy;
}

// ---- copy (into existing tensor) ----------------------------------------

// D2D strided copy: handles non-contiguous dst and/or src on the same device.
//
// Two paths:
//   dst contiguous — iterate over src elements using src's ndim/strides, write
//   dst at flat index workIdx. Passing src_contig as both out_contig and
//   out_actual makes offset_from return workIdx for the dst address. This is
//   the correct path when dst and src have different ndims (e.g. FlattenOp
//   copies a 4D non-contiguous pool output into a 2D contiguous flat tensor).
//
//   dst non-contiguous — iterate using dst's ndim (same ndim as src in this
//   codebase; this path is taken by tensor_contiguous which always preserves
//   ndim).
static void cuda_copy_d2d_strided(Tensor *dst, const Tensor *src) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst->size, (u64)threads);
    if (tensor_is_contiguous(dst)) {
        TensorMeta src_contig = make_contig_meta(src);
        TensorMeta src_actual(src);
        tensor_copy_strided<<<blocks, threads>>>(
            src_contig, src_contig, src_actual, dst->data, src->data);
    } else {
        TensorMeta dst_contig = make_contig_meta(dst);
        TensorMeta dst_actual(dst);
        TensorMeta src_meta(src);
        tensor_copy_strided<<<blocks, threads>>>(
            dst_contig, dst_actual, src_meta, dst->data, src->data);
    }
}

// Selects cudaMemcpyKind from the two on_gpu flags packed into a 2-bit key:
//   bit1 = dst->on_gpu, bit0 = src->on_gpu → 00=H2H, 01=D2H, 10=H2D,
//   11=D2D.
// For D2D with non-contiguous tensors the stride-aware kernel is used
// instead of cudaMemcpy (which would copy the flat buffer in the wrong
// order).
void tensor_cuda_copy(Tensor *dst, const Tensor *src) {
    if (dst->on_gpu && src->on_gpu &&
        (!tensor_is_contiguous(dst) || !tensor_is_contiguous(src))) {
        cuda_copy_d2d_strided(dst, src);
        return;
    }
    cudaMemcpyKind kind;
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        kind = cudaMemcpyHostToHost;
        break;
    case 0b01:
        kind = cudaMemcpyDeviceToHost;
        break;
    case 0b10:
        kind = cudaMemcpyHostToDevice;
        break;
    case 0b11:
        kind = cudaMemcpyDeviceToDevice;
        break;
    }
    cudaMemcpy(dst->data, src->data, src->size * sizeof(f32), kind);
}

// ---- fill / clear
// --------------------------------------------------------

void tensor_cuda_fill(Tensor *tensor, f32 value) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(tensor->size, (u64)threads);
    tensor_fill<<<blocks, threads>>>(tensor->size, tensor->data, value);
}

void tensor_cuda_clear(Tensor *tensor) {
    cudaMemset(tensor->data, 0, sizeof(f32) * tensor->size);
}

// ---- activations (relu, exp)
// ---------------------------------------------

void tensor_cuda_relu(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, ReluFwdOp{});
}

void tensor_cuda_exp(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, ExpOp{});
}

void tensor_cuda_log(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, LogOp{});
}

void tensor_cuda_sqrt(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, SqrtOp{});
}

// ---- elementwise binary (add / sub / mul / div)
// -------------------------- Validation is done by the dispatcher in
// tensor.cpp before calling here.

void tensor_cuda_add(Tensor *out, const Tensor *a, const Tensor *b) {
    cuda_elementwise_binary(out, a, b, AddOp{});
}

void tensor_cuda_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    cuda_elementwise_binary(out, a, b, SubOp{});
}

void tensor_cuda_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    cuda_elementwise_binary(out, a, b, MulOp{});
}

void tensor_cuda_div(Tensor *out, const Tensor *a, const Tensor *b) {
    cuda_elementwise_binary(out, a, b, DivOp{});
}

void tensor_cuda_equal(Tensor *out, const Tensor *a, const Tensor *b) {
    cuda_elementwise_binary(out, a, b, EqualOp{});
}
void tensor_cuda_relu_backward(Tensor *out, const Tensor *grad,
                               const Tensor *in) {
    cuda_elementwise_binary(out, grad, in, ReluBackwardOp{});
}

// ---- scalar operations
// ---------------------------------------------------

void tensor_cuda_add(Tensor *out, const Tensor *tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, AddOp{});
}

void tensor_cuda_sub(Tensor *out, const Tensor *tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, SubOp{});
}

void tensor_cuda_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, MulOp{});
}

void tensor_cuda_div(Tensor *out, const Tensor *tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, DivOp{});
}

// ---- matrix multiply
// -----------------------------------------------------

void tensor_cuda_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                         b32 clear_out) {
    u32 threads = TILE;
    u32 row_tiles =
        cuda::ceil_div(out->shape[0], threads); // goes into x (large limit)
    u32 col_tiles =
        cuda::ceil_div(out->shape[1], threads); // goes into y (≤65535)
    dim3 threadsPerBlock(threads, threads);
    dim3 blocks(row_tiles, col_tiles);
    TensorMeta out_meta(out);
    TensorMeta a_meta(a);
    TensorMeta b_meta(b);
    f32 beta = clear_out ? 0.0f : 1.0f;
    mat_mul_tiled<<<blocks, threadsPerBlock>>>(
        out_meta, a_meta, b_meta, out->data, a->data, b->data, beta);
}

// ---- reduction (sum, max, argmax)
// ----------------------------------------

// Global reduction via repeated tensor_sum_step launches.
// Each launch reduces the current size by N_THREADS× (one output per
// block). Intermediate results are stored in temporary GPU tensors. Loop
// continues until blocks==1, meaning a single block produced the final
// scalar in out.
void tensor_cuda_sum(Tensor *out, const Tensor *tensor) {
    u32 threads = N_THREADS;
    u32 blocks = 0;
    Tensor *cur = tensor_view(tensor);
    Tensor *next = nullptr;
    while (true) {
        blocks = cuda::ceil_div(cur->size, (u64)threads);
        u32 shape[] = {blocks};
        next = blocks == 1 ? tensor_view(out) : new Tensor(1, shape, true);
        tensor_sum_step<<<blocks, threads>>>(cur->size, next->data, cur->data);
        delete cur;
        if (blocks == 1) {
            delete next;
            break;
        }
        cur = next;
    }
}

void tensor_cuda_sum(Tensor *out, const Tensor *tensor, u32 dim) {
    if (out->size == 1) {
        tensor_cuda_sum(out, tensor);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_sum<<<blocks, threads>>>(out_meta, tensor_meta, out->data,
                                    tensor->data, dim);
}

void tensor_cuda_welford_mean_var(Tensor *mean, Tensor *var, const Tensor *src,
                                  u32 dim) {
    u64 other_dims_size = src->size / src->shape[dim];

    TensorMeta meta_contig(src);
    TensorMeta meta_src(src);
    meta_contig.ndim = meta_src.ndim = src->ndim - 1;
    meta_contig.size = meta_src.size = other_dims_size;
    for (u32 d = 0, o = 0; d < src->ndim; d++) {
        if (d == dim)
            continue;
        meta_contig.shape[o] = src->shape[d];
        meta_src.shape[o] = src->shape[d];
        meta_src.stride[o] = src->stride[d];
        o++;
    }
    tensor_compute_strides(meta_contig.stride, meta_contig.shape,
                           meta_contig.ndim);

    welford_mean_var<<<src->shape[dim], N_THREADS>>>(
        mean->data, var->data, src->data, src->stride[dim], meta_contig,
        meta_src, other_dims_size);
}

void tensor_cuda_max(Tensor *out, const Tensor *tensor) {
    u32 threads = N_THREADS;
    u32 blocks = 0;
    Tensor *cur = tensor_view(tensor);
    Tensor *next = nullptr;
    while (true) {
        blocks = cuda::ceil_div(cur->size, (u64)threads);
        u32 shape[] = {blocks};
        next = blocks == 1 ? tensor_view(out) : new Tensor(1, shape, true);
        tensor_max_step<<<blocks, threads>>>(cur->size, next->data, cur->data);
        delete cur;
        if (blocks == 1) {
            delete next;
            break;
        }
        cur = next;
    }
}

void tensor_cuda_max(Tensor *out, const Tensor *tensor, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_max<<<blocks, threads>>>(out_meta, tensor_meta, out->data,
                                    tensor->data, dim);
}

void tensor_cuda_argmax(Tensor *out, const Tensor *tensor, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_argmax_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out->data,
                                              tensor->data, dim);
}

// ---- scattering ----------------------------------------------------------
void tensor_cuda_scatter_add(Tensor *out, const Tensor *src,
                             const Tensor *indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(src->size, u64(threads));

    TensorMeta out_meta(out);
    TensorMeta out_base_meta(out);
    out_base_meta.stride[dim] = 0;
    TensorMeta indices_meta(indices);
    TensorMeta src_meta(src);
    TensorMeta src_contig = make_contig_meta(src);

    tensor_scatter_add<<<blocks, threads>>>(
        out_meta, out_base_meta, indices_meta, src_meta, src_contig, out->data,
        indices->data, src->data, dim);
}

// ---- initializing --------------------------------------------------------

void tensor_cuda_he_init(Tensor *tensor) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    u32 in_features = tensor->shape[ROW_DIM(tensor)];
    float stddev = sqrtf(2.0f / in_features);
    curandGenerateNormal(gen, tensor->data, tensor->size, 0.0f, stddev);
    curandDestroyGenerator(gen);
}

// ---- indexing
// ------------------------------------------------------------

void tensor_cuda_index_select(Tensor *dst, const Tensor *src,
                              const u32 *indices, u32 n_indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst->size, u64(threads));
    TensorMeta dst_meta(dst);
    TensorMeta src_meta(src);
    u32 *indices_gpu;
    cudaMalloc(&indices_gpu, n_indices * sizeof(u32));
    cudaMemcpy(indices_gpu, indices, n_indices * sizeof(u32),
               cudaMemcpyHostToDevice);
    tensor_index_select<<<blocks, threads>>>(
        dst_meta, src_meta, dst->data, src->data, indices_gpu, n_indices, dim);
    cudaFree(indices_gpu);
}

void tensor_cuda_unfold2d(Tensor *dst, const Tensor *src,
                          Unfold2dParams params) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst->size, u64(threads));
    u32 N = src->shape[0];
    u32 C = src->shape[1];
    u32 H = src->shape[2];
    u32 W = src->shape[3];

    params.compute_output_size(H, W);
    u32 L = params.L_h * params.L_w;

    u32 shape[MAX_NDIM] = {N, params.L_h, params.L_w,
                           C, params.k_h, params.k_w};
    tensor_reshape(dst, shape, 6);

    TensorMeta dst_meta = TensorMeta(dst);
    TensorMeta dst_meta_contig = make_contig_meta(dst);

    TensorMeta src_meta(src);

    tensor_unfold2d<<<blocks, threads>>>(dst_meta, dst_meta_contig, src_meta,
                                         params, dst->data, src->data);
    u32 shape_[MAX_NDIM] = {N, L, C * params.k_h * params.k_w};
    tensor_reshape(dst, shape_, 3);
}

void tensor_cuda_fold2d(Tensor *dst, const Tensor *col, Unfold2dParams params) {
    u32 N = dst->shape[0];
    u32 C = dst->shape[1];
    u32 H = dst->shape[2];
    u32 W = dst->shape[3];
    params.compute_output_size(H, W);

    // reshape col to 6D so stride decomposition maps to (n,lh,lw,c,kh,kw)
    Tensor *col6 = tensor_view(col);
    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w,
                            C, params.k_h, params.k_w};
    tensor_reshape(col6, shape6, 6);

    TensorMeta col_meta(col6);
    TensorMeta col_meta_contig = make_contig_meta(col6);
    TensorMeta dst_meta(dst);

    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(col6->size, u64(threads));
    tensor_fold2d<<<blocks, threads>>>(col_meta, col_meta_contig, dst_meta,
                                       params, col6->data, dst->data);

    delete col6;
}

// ---- comparison
// ----------------------------------------------------------

b32 tensor_cuda_equals(const Tensor *a, const Tensor *b, f32 tol) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(a->size, (u64)threads);
    u32 *ok_gpu;
    cudaMalloc(&ok_gpu, sizeof(u32));
    cudaMemset(ok_gpu, 1, sizeof(u32));
    tensor_check_close<<<blocks, threads>>>(a->size, ok_gpu, a->data, b->data,
                                            tol);
    u32 ok_cpu;
    cudaMemcpy(&ok_cpu, ok_gpu, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaFree(ok_gpu);
    return (b32)ok_cpu;
}
