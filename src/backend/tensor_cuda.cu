#include "../../include/backend/tensor_cuda.hpp"
#include <cuda/cmath>
#include <curand_kernel.h>

static constexpr u32 TILE = 16;
static constexpr u32 N_THREADS = 256;

// ── Functors ──────────────────────────────────────────────────────────────────
// Small device-callable structs passed as template parameters to generic kernels
// so the compiler inlines the operation and generates a single fused kernel per op.

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
__global__ void elementwise_unary_strided(TensorMeta out_contig,
                                          TensorMeta out_actual,
                                          TensorMeta a_meta,
                                          f32 *out, const f32 *a, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset   = out_contig.offset_from(workIdx, a_meta);
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
__global__ void elementwise_broadcast(const TensorMeta out_contig,
                                      const TensorMeta out_actual,
                                      const TensorMeta a_meta,
                                      const TensorMeta b_meta, f32 *out,
                                      const f32 *a, const f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset   = out_contig.offset_from(workIdx, a_meta);
    u64 b_offset   = out_contig.offset_from(workIdx, b_meta);
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
                                           TensorMeta a_meta,
                                           f32 *out, const f32 *a,
                                           f32 scalar, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset   = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset], scalar);
}

// ---- copy (strided, D2D) -------------------------------------------------

// Stride-aware D2D copy kernel. Same decomposition trick as the other strided
// kernels: out_contig decomposes workIdx, out_actual and src_meta give the
// physical write/read offsets.
__global__ void tensor_copy_strided(TensorMeta out_contig, TensorMeta out_actual,
                                    TensorMeta src_meta,
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
__global__ void mat_mul_tiled(TensorMeta out_meta, TensorMeta a_meta,
                              TensorMeta b_meta, f32 *out, f32 *a, f32 *b) {
    u64 myCol = threadIdx.x + blockIdx.x * blockDim.x;
    u64 myRow = threadIdx.y + blockIdx.y * blockDim.y;

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
        out[out_meta.at(myRow, myCol)] = value;
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
// addition. Identity element is -FLT_MAX so out-of-bounds threads don't affect the result.
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

// ── Template dispatch helpers
// ─────────────────────────────────────────────────

// Builds a TensorMeta whose strides are contiguous row-major (i.e. the strides
// you'd get from a freshly allocated tensor of the same shape). Used as the
// decomposition base in offset_from() so workIdx → per-dim indices is always
// a simple radix decomposition regardless of out's actual layout.
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
        elementwise_unary<<<blocks, threads>>>(out->size, out->data, a->data, op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise_unary_strided<<<blocks, threads>>>(out_contig, out_actual,
                                                   a_meta, out->data, a->data, op);
}

// Fast path: all three tensors same shape and contiguous — plain flat kernel.
// Strided path: builds a contiguous decomposition meta for out (out_contig) and
// passes out's actual strides (out_actual) so the write goes to the right slot.
// a_meta/b_meta are broadcast-expanded (stride=0 on dims where size==1).
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
    elementwise_broadcast<<<blocks, threads>>>(out_contig, out_actual,
                                               a_meta, b_meta,
                                               out->data, a->data, b->data, op);
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
    elementwise_scalar_strided<<<blocks, threads>>>(out_contig, out_actual,
                                                    a_meta, out->data, a->data,
                                                    scalar, op);
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

// ---- memory management (alloc / free / transfers) ------------------------

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
static void cuda_copy_d2d_strided(Tensor *dst, const Tensor *src) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst->size, (u64)threads);
    TensorMeta dst_contig = make_contig_meta(dst);
    TensorMeta dst_actual(dst);
    TensorMeta src_meta(src);
    tensor_copy_strided<<<blocks, threads>>>(dst_contig, dst_actual, src_meta,
                                             dst->data, src->data);
}

// Selects cudaMemcpyKind from the two on_gpu flags packed into a 2-bit key:
//   bit1 = dst->on_gpu, bit0 = src->on_gpu → 00=H2H, 01=D2H, 10=H2D, 11=D2D.
// For D2D with non-contiguous tensors the stride-aware kernel is used instead
// of cudaMemcpy (which would copy the flat buffer in the wrong order).
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

// ---- fill / clear --------------------------------------------------------

void tensor_cuda_fill(Tensor *tensor, f32 value) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(tensor->size, (u64)threads);
    tensor_fill<<<blocks, threads>>>(tensor->size, tensor->data, value);
}

void tensor_cuda_clear(Tensor *tensor) {
    cudaMemset(tensor->data, 0, sizeof(f32) * tensor->size);
}

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cuda_relu(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, ReluFwdOp{});
}

void tensor_cuda_exp(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, ExpOp{});
}

void tensor_cuda_log(Tensor *dst, const Tensor *src) {
    cuda_elementwise_unary(dst, src, LogOp{});
}

// ---- elementwise binary (add / sub / mul / div) --------------------------
// Validation is done by the dispatcher in tensor.cpp before calling here.

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

// ---- scalar operations ---------------------------------------------------

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

// ---- matrix multiply -----------------------------------------------------

void tensor_cuda_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                         b32 clear_out) {
    u32 threads = TILE;
    u32 blocks_x = cuda::ceil_div(out->shape[1], threads);
    u32 blocks_y = cuda::ceil_div(out->shape[0], threads);
    dim3 threadsPerBlock(threads, threads);
    dim3 blocks(blocks_x, blocks_y);
    TensorMeta out_meta(out);
    TensorMeta a_meta(a);
    TensorMeta b_meta(b);
    mat_mul_tiled<<<blocks, threadsPerBlock>>>(out_meta, a_meta, b_meta,
                                               out->data, a->data, b->data);
}

// ---- reduction (sum, max, argmax) ----------------------------------------

// Global reduction via repeated tensor_sum_step launches.
// Each launch reduces the current size by N_THREADS× (one output per block).
// Intermediate results are stored in temporary GPU tensors. Loop continues
// until blocks==1, meaning a single block produced the final scalar in out.
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

// ---- indexing ------------------------------------------------------------

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
