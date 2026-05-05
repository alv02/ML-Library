#include "../../include/backend/tensor_cuda.hpp"
#include <cuda/cmath>
#include <curand_kernel.h>

static constexpr u32 TILE = 32;
static constexpr u32 N_THREADS = 512;

// ── Functors ─────────────────────────────────────────────────────────────────

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

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void tensor_fill(u64 tensor_size, f32 *tensor_data, f32 value) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < tensor_size)
        tensor_data[workIdx] = value;
}

template <typename Op>
__global__ void elementwise_unary(u64 size, f32 *out, const f32 *a, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx]);
}

template <typename Op>
__global__ void elementwise_unary(TensorMeta out_contig, TensorMeta out_actual,
                                  TensorMeta a_meta, f32 *out, const f32 *a,
                                  Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset]);
}

template <typename Op>
__global__ void elementwise_binary(u64 size, f32 *out, f32 *a, f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b[workIdx]);
}

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

template <typename Op>
__global__ void elementwise_binary(u64 size, f32 *out, const f32 *a, f32 b,
                                   Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b);
}

template <typename Op>
__global__ void elementwise(TensorMeta out_contig, TensorMeta out_actual,
                            TensorMeta a_meta, f32 *out, const f32 *a,
                            f32 scalar, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset], scalar);
}

__global__ void tensor_contiguous(TensorMeta contig_meta,
                                  TensorMeta actual_meta, f32 *dst, f32 *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= contig_meta.size)
        return;
    u64 src_offset = contig_meta.offset_from(workIdx, actual_meta);
    dst[workIdx] = src[src_offset];
}

__global__ void tensor_copy(TensorMeta out_contig, TensorMeta out_actual,
                            TensorMeta src_meta, f32 *dst, const f32 *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 dst_offset = out_contig.offset_from(workIdx, out_actual);
    u64 src_offset = out_contig.offset_from(workIdx, src_meta);
    dst[dst_offset] = src[src_offset];
}

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

// 2×2 thread coarsening: 16×16 block (256 threads), each thread owns a 2×2
// output subregion. Keeps the same 32×32 shared memory tile as before, giving
// 4× more FMAs per thread and 6 blocks/SM (100% warp occupancy on SM 12.x).
__global__ void mat_mul_tiled(TensorMeta out_meta, TensorMeta a_meta,
                              TensorMeta b_meta, f32 *out, f32 *a, f32 *b,
                              f32 beta) {
    constexpr u32 HTILE = TILE / 2;  // 16 — thread block side length

    // Output coordinates for this thread's 2×2 subregion
    u64 row0 = threadIdx.y * 2 + blockIdx.x * TILE;
    u64 row1 = row0 + 1;
    u64 col0 = threadIdx.x * 2 + blockIdx.y * TILE;
    u64 col1 = col0 + 1;

    __shared__ f32 a_tile[TILE * TILE];
    __shared__ f32 b_tile[TILE * TILE];

    u32 n_tiles = cuda::ceil_div(a_meta.cols(), TILE);

    // Linear id for coalesced tile loading (256 threads → 1024 elements, 4 each)
    u32 tid = threadIdx.y * HTILE + threadIdx.x;

    f32 v00 = 0, v01 = 0, v10 = 0, v11 = 0;

    for (u32 n = 0; n < n_tiles; n++) {
        // Load a_tile and b_tile — 4 elements per thread, coalesced
        for (u32 i = 0; i < 4; i++) {
            u32 flat = tid + i * (HTILE * HTILE);
            u32 ar = flat / TILE, ac = flat % TILE;
            u64 grow = blockIdx.x * TILE + ar, gcol = n * TILE + ac;
            a_tile[flat] = (grow < a_meta.rows() && gcol < a_meta.cols())
                               ? a[a_meta.at(grow, gcol)]
                               : 0.0f;

            u32 br = flat / TILE, bc = flat % TILE;
            u64 brow = n * TILE + br, bcol = blockIdx.y * TILE + bc;
            b_tile[flat] = (brow < b_meta.rows() && bcol < b_meta.cols())
                               ? b[b_meta.at(brow, bcol)]
                               : 0.0f;
        }

        __syncthreads();

        for (u32 k = 0; k < TILE; k++) {
            f32 a0 = a_tile[threadIdx.y * 2 * TILE + k];
            f32 a1 = a_tile[(threadIdx.y * 2 + 1) * TILE + k];
            f32 b0 = b_tile[k * TILE + threadIdx.x * 2];
            f32 b1 = b_tile[k * TILE + threadIdx.x * 2 + 1];
            v00 += a0 * b0;  v01 += a0 * b1;
            v10 += a1 * b0;  v11 += a1 * b1;
        }

        __syncthreads();
    }

    u64 rows = out_meta.rows(), cols = out_meta.cols();
    if (row0 < rows && col0 < cols)
        out[out_meta.at(row0, col0)] = v00 + beta * out[out_meta.at(row0, col0)];
    if (row0 < rows && col1 < cols)
        out[out_meta.at(row0, col1)] = v01 + beta * out[out_meta.at(row0, col1)];
    if (row1 < rows && col0 < cols)
        out[out_meta.at(row1, col0)] = v10 + beta * out[out_meta.at(row1, col0)];
    if (row1 < rows && col1 < cols)
        out[out_meta.at(row1, col1)] = v11 + beta * out[out_meta.at(row1, col1)];
}

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
        var[c] = partial_M2[0] / (f32)(partial_n[0] - 1);
    }
}

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

__global__ void tensor_unfold2d(TensorMeta dst_meta, TensorMeta dst_meta_contig,
                                TensorMeta src_meta, Unfold2dParams params,
                                f32 *dst, const f32 *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= dst_meta.size)
        return;

    u64 remaining = workIdx;
    u32 idx[6];
    for (u32 i = 0; i < 6; i++) {
        idx[i] = remaining / dst_meta_contig.stride[i];
        remaining -= idx[i] * dst_meta_contig.stride[i];
    }

    u32 n = idx[0], lh = idx[1], lw = idx[2], c = idx[3], kh = idx[4],
        kw = idx[5];

    i32 h = lh * params.stride_h + kh - params.pad_h;
    i32 w = lw * params.stride_w + kw - params.pad_w;

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
    u32 n = idx[0], lh = idx[1], lw = idx[2], c = idx[3], kh = idx[4],
        kw = idx[5];

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

__global__ void tensor_check_close(u64 size, u32 *ok, const f32 *a,
                                   const f32 *b, f32 tol) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size && fabsf(a[workIdx] - b[workIdx]) > tol)
        *ok = 0;
}

// ── Template dispatch helpers
// ─────────────────────────────────────────────────

static TensorMeta make_contig_meta(const TensorImpl &t) {
    TensorMeta m(t);
    tensor_compute_strides(m.stride, m.shape, m.ndim);
    return m;
}

template <typename Op>
static void cuda_elementwise_unary(TensorImpl &out, const TensorImpl &a,
                                   Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        elementwise_unary<<<blocks, threads>>>(out.numel(), out.data(),
                                               a.data(), op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise_unary<<<blocks, threads>>>(out_contig, out_actual, a_meta,
                                           out.data(), a.data(), op);
}

template <typename Op>
static void cuda_elementwise_binary(TensorImpl &out, const TensorImpl &a,
                                    const TensorImpl &b, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(out) &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        elementwise_binary<<<blocks, threads>>>(out.numel(), out.data(),
                                                a.data(), b.data(), op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a, out.shape, out.ndim);
    TensorMeta b_meta(b, out.shape, out.ndim);
    elementwise_broadcast<<<blocks, threads>>>(out_contig, out_actual, a_meta,
                                               b_meta, out.data(), a.data(),
                                               b.data(), op);
}

template <typename Op>
static void cuda_elementwise_scalar(TensorImpl &out, const TensorImpl &a,
                                    f32 scalar, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        elementwise_binary<<<blocks, threads>>>(out.numel(), out.data(),
                                                a.data(), scalar, op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise<<<blocks, threads>>>(out_contig, out_actual, a_meta, out.data(),
                                     a.data(), scalar, op);
}

// ── TensorMeta
// ────────────────────────────────────────────────────────────────

TensorMeta::TensorMeta(const TensorImpl &t) : size(t.numel()), ndim(t.ndim) {
    memcpy(shape, t.shape, ndim * sizeof(u32));
    memcpy(stride, t.stride, ndim * sizeof(u64));
}

TensorMeta::TensorMeta(const TensorImpl &t, const u32 *bcast_shape,
                       u32 bcast_ndim)
    : size(t.numel()), ndim(bcast_ndim) {
    expanded_shape(t, bcast_ndim, shape);
    expanded_stride(t, bcast_ndim, stride);
}

// ── Host functions
// ────────────────────────────────────────────────────────────

// ---- memory management (transfers)
// -------------------------------------------
void tensor_cuda_copy(TensorImpl &dst, const TensorImpl &src) {
    if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        cudaMemcpyAsync(dst.data(), src.data(), src.numel() * sizeof(f32),
                        cudaMemcpyDeviceToDevice, 0);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), (u64)threads);
    TensorMeta out_contig = make_contig_meta(dst);
    TensorMeta out_actual(dst);
    TensorMeta src_meta(src);
    tensor_copy<<<blocks, threads>>>(out_contig, out_actual, src_meta,
                                     dst.data(), src.data());
}

void tensor_cuda_contiguous(TensorImpl &t, CudaMemArena *arena) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(t.numel(), (u64)threads);

    TensorMeta src_meta(t);
    TensorMeta src_contig = make_contig_meta(t);

    Tensor temp_t = Tensor::make(t.ndim, t.shape, true, arena);
    TensorImpl &temp = temp_t.impl();

    tensor_contiguous<<<blocks, threads>>>(src_contig, src_meta, temp.data(),
                                           t.data());

    cudaMemcpyAsync(t.data(), temp.data(), t.numel() * sizeof(f32),
                    cudaMemcpyDeviceToDevice, 0);
    tensor_compute_strides(t.stride, t.shape, t.ndim);
}

// ---- fill / clear --------------------------------------------------------

void tensor_cuda_fill(TensorImpl &tensor, f32 value) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(tensor.numel(), (u64)threads);
    tensor_fill<<<blocks, threads>>>(tensor.numel(), tensor.data(), value);
}

void tensor_cuda_clear(TensorImpl &tensor) {
    cudaMemsetAsync(tensor.data(), 0, sizeof(f32) * tensor.numel(), 0);
}

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cuda_relu(TensorImpl &dst, const TensorImpl &src) {
    cuda_elementwise_unary(dst, src, ReluFwdOp{});
}

void tensor_cuda_exp(TensorImpl &dst, const TensorImpl &src) {
    cuda_elementwise_unary(dst, src, ExpOp{});
}

void tensor_cuda_log(TensorImpl &dst, const TensorImpl &src) {
    cuda_elementwise_unary(dst, src, LogOp{});
}

void tensor_cuda_sqrt(TensorImpl &dst, const TensorImpl &src) {
    cuda_elementwise_unary(dst, src, SqrtOp{});
}

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cuda_add(TensorImpl &out, const TensorImpl &a,
                     const TensorImpl &b) {
    cuda_elementwise_binary(out, a, b, AddOp{});
}

void tensor_cuda_sub(TensorImpl &out, const TensorImpl &a,
                     const TensorImpl &b) {
    cuda_elementwise_binary(out, a, b, SubOp{});
}

void tensor_cuda_mul(TensorImpl &out, const TensorImpl &a,
                     const TensorImpl &b) {
    cuda_elementwise_binary(out, a, b, MulOp{});
}

void tensor_cuda_div(TensorImpl &out, const TensorImpl &a,
                     const TensorImpl &b) {
    cuda_elementwise_binary(out, a, b, DivOp{});
}

void tensor_cuda_equal(TensorImpl &out, const TensorImpl &a,
                       const TensorImpl &b) {
    cuda_elementwise_binary(out, a, b, EqualOp{});
}

void tensor_cuda_relu_backward(TensorImpl &out, const TensorImpl &grad,
                               const TensorImpl &in) {
    cuda_elementwise_binary(out, grad, in, ReluBackwardOp{});
}

// ---- scalar operations ---------------------------------------------------

void tensor_cuda_add(TensorImpl &out, const TensorImpl &tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, AddOp{});
}

void tensor_cuda_sub(TensorImpl &out, const TensorImpl &tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, SubOp{});
}

void tensor_cuda_mul(TensorImpl &out, const TensorImpl &tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, MulOp{});
}

void tensor_cuda_div(TensorImpl &out, const TensorImpl &tensor, f32 scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, DivOp{});
}

// ---- matrix multiply -----------------------------------------------------

void tensor_cuda_mat_mul(TensorImpl &out, const TensorImpl &a,
                         const TensorImpl &b, b32 clear_out) {
    constexpr u32 HTILE = TILE / 2;
    u32 row_tiles = cuda::ceil_div(out.shape[0], TILE);
    u32 col_tiles = cuda::ceil_div(out.shape[1], TILE);
    dim3 threadsPerBlock(HTILE, HTILE);  // 16×16 = 256 threads
    dim3 blocks(row_tiles, col_tiles);
    TensorMeta out_meta(out);
    TensorMeta a_meta(a);
    TensorMeta b_meta(b);
    f32 beta = clear_out ? 0.0f : 1.0f;
    mat_mul_tiled<<<blocks, threadsPerBlock>>>(
        out_meta, a_meta, b_meta, out.data(), a.data(), b.data(), beta);
}

// ---- reduction (sum, max, argmax) ----------------------------------------

// Global reduction using repeated kernel launches — each reduces the current
// buffer by N_THREADS×. Keeps a Tensor alive for each intermediate buffer
// (prev_temp) while reading from its data pointer (cur_data).
void tensor_cuda_sum(TensorImpl &out, const TensorImpl &tensor) {
    u32 threads = N_THREADS;
    const f32 *cur_data = tensor.data();
    u64 cur_size = tensor.numel();
    Tensor prev_temp; // keeps previous intermediate buffer alive

    while (true) {
        u32 blocks = cuda::ceil_div(cur_size, (u64)threads);
        if (blocks == 1) {
            tensor_sum_step<<<1, threads>>>(cur_size, out.data(),
                                            const_cast<f32 *>(cur_data));
            break;
        }
        u32 shape[] = {blocks};
        Tensor next = Tensor::make(1, shape, true);
        tensor_sum_step<<<blocks, threads>>>(cur_size, next->data(),
                                             const_cast<f32 *>(cur_data));
        cur_data = next->data();
        cur_size = blocks;
        prev_temp = std::move(next); // old prev_temp freed, next kept alive
    }
}

void tensor_cuda_sum(TensorImpl &out, const TensorImpl &tensor, u32 dim) {
    if (out.numel() == 1) {
        tensor_cuda_sum(out, tensor);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_sum<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                    tensor.data(), dim);
}

void tensor_cuda_welford_mean_var(TensorImpl &mean, TensorImpl &var,
                                  const TensorImpl &src, u32 dim) {
    u64 other_dims_size = src.numel() / src.shape[dim];

    TensorMeta meta_contig(src);
    TensorMeta meta_src(src);
    meta_contig.ndim = meta_src.ndim = src.ndim - 1;
    meta_contig.size = meta_src.size = other_dims_size;
    for (u32 d = 0, o = 0; d < src.ndim; d++) {
        if (d == dim)
            continue;
        meta_contig.shape[o] = src.shape[d];
        meta_src.shape[o] = src.shape[d];
        meta_src.stride[o] = src.stride[d];
        o++;
    }
    tensor_compute_strides(meta_contig.stride, meta_contig.shape,
                           meta_contig.ndim);

    welford_mean_var<<<src.shape[dim], N_THREADS>>>(
        mean.data(), var.data(), src.data(), src.stride[dim], meta_contig,
        meta_src, other_dims_size);
}

void tensor_cuda_max(TensorImpl &out, const TensorImpl &tensor) {
    u32 threads = N_THREADS;
    const f32 *cur_data = tensor.data();
    u64 cur_size = tensor.numel();
    Tensor prev_temp;

    while (true) {
        u32 blocks = cuda::ceil_div(cur_size, (u64)threads);
        if (blocks == 1) {
            tensor_max_step<<<1, threads>>>(cur_size, out.data(),
                                            const_cast<f32 *>(cur_data));
            break;
        }
        u32 shape[] = {blocks};
        Tensor next = Tensor::make(1, shape, true);
        tensor_max_step<<<blocks, threads>>>(cur_size, next->data(),
                                             const_cast<f32 *>(cur_data));
        cur_data = next->data();
        cur_size = blocks;
        prev_temp = std::move(next);
    }
}

void tensor_cuda_max(TensorImpl &out, const TensorImpl &tensor, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_max<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                    tensor.data(), dim);
}

void tensor_cuda_argmax(TensorImpl &out, const TensorImpl &tensor, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_argmax_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                              tensor.data(), dim);
}

// ---- scattering ----------------------------------------------------------

void tensor_cuda_scatter_add(TensorImpl &out, const TensorImpl &src,
                             const TensorImpl &indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(src.numel(), u64(threads));

    TensorMeta out_meta(out);
    TensorMeta out_base_meta(out);
    out_base_meta.stride[dim] = 0;
    TensorMeta indices_meta(indices);
    TensorMeta src_meta(src);
    TensorMeta src_contig = make_contig_meta(src);

    tensor_scatter_add<<<blocks, threads>>>(
        out_meta, out_base_meta, indices_meta, src_meta, src_contig, out.data(),
        indices.data(), src.data(), dim);
}

// ---- initializing --------------------------------------------------------

void tensor_cuda_he_init(TensorImpl &tensor) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    u32 in_features = tensor.shape[ROW_DIM(tensor)];
    float stddev = sqrtf(2.0f / in_features);
    curandGenerateNormal(gen, tensor.data(), tensor.numel(), 0.0f, stddev);
    curandDestroyGenerator(gen);
}

// ---- indexing ------------------------------------------------------------

void tensor_cuda_index_select(TensorImpl &dst, const TensorImpl &src,
                              const u32 *indices, u32 n_indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), u64(threads));
    TensorMeta dst_meta(dst);
    TensorMeta src_meta(src);
    u32 *indices_gpu;
    cudaMallocAsync(&indices_gpu, n_indices * sizeof(u32), 0);
    cudaMemcpyAsync(indices_gpu, indices, n_indices * sizeof(u32),
                    cudaMemcpyHostToDevice, 0);
    tensor_index_select<<<blocks, threads>>>(dst_meta, src_meta, dst.data(),
                                             src.data(), indices_gpu, n_indices,
                                             dim);
    cudaFreeAsync(indices_gpu, 0);
}

// ---- spatial / patch operations ------------------------------------------

void tensor_cuda_unfold2d(TensorImpl &dst, const TensorImpl &src,
                          Unfold2dParams params) {
    u32 N = src.shape[0];
    u32 C = src.shape[1];
    u32 H = src.shape[2];
    u32 W = src.shape[3];

    params.compute_output_size(H, W);
    u32 L = params.L_h * params.L_w;

    u32 shape[MAX_NDIM] = {N, params.L_h, params.L_w,
                           C, params.k_h, params.k_w};
    tensor_reshape(dst, shape, 6);

    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), u64(threads));

    TensorMeta dst_meta(dst);
    TensorMeta dst_meta_contig = make_contig_meta(dst);
    TensorMeta src_meta(src);

    tensor_unfold2d<<<blocks, threads>>>(dst_meta, dst_meta_contig, src_meta,
                                         params, dst.data(), src.data());

    u32 shape_[MAX_NDIM] = {N, L, C * params.k_h * params.k_w};
    tensor_reshape(dst, shape_, 3);
}

void tensor_cuda_fold2d(TensorImpl &dst, const TensorImpl &col,
                        Unfold2dParams params) {
    u32 N = dst.shape[0];
    u32 C = dst.shape[1];
    u32 H = dst.shape[2];
    u32 W = dst.shape[3];
    params.compute_output_size(H, W);

    // Build 6D TensorMeta for col directly without creating a view tensor
    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w,
                            C, params.k_h, params.k_w};
    u64 stride6[MAX_NDIM];
    tensor_compute_strides(stride6, shape6, 6);

    TensorMeta col_meta;
    col_meta.ndim = 6;
    col_meta.size = col.numel();
    memcpy(col_meta.shape, shape6, 6 * sizeof(u32));
    memcpy(col_meta.stride, stride6, 6 * sizeof(u64));

    TensorMeta col_meta_contig = col_meta; // already contiguous
    TensorMeta dst_meta(dst);

    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(col.numel(), u64(threads));
    tensor_fold2d<<<blocks, threads>>>(col_meta, col_meta_contig, dst_meta,
                                       params, col.data(), dst.data());
}

// ---- comparison ----------------------------------------------------------

b32 tensor_cuda_equals(const TensorImpl &a, const TensorImpl &b, f32 tol) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(a.numel(), (u64)threads);
    u32 *ok_gpu;
    cudaMalloc(&ok_gpu, sizeof(u32));
    cudaMemset(ok_gpu, 1, sizeof(u32));
    tensor_check_close<<<blocks, threads>>>(a.numel(), ok_gpu, a.data(),
                                            b.data(), tol);
    u32 ok_cpu;
    cudaMemcpy(&ok_cpu, ok_gpu, sizeof(u32), cudaMemcpyDeviceToHost);
    cudaFree(ok_gpu);
    return (b32)ok_cpu;
}
