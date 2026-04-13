#include "../../include/backend/tensor_cuda.hpp"
#include <cuda/cmath>

static constexpr u32 TILE = 16;
static constexpr u32 N_THREADS = 256;

// ---- Functors ---------------------------------------------------------------

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

// ---- Kernels ----------------------------------------------------------------

__global__ void tensor_fill(u64 tensor_size, f32 *tensor_data, f32 value) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < tensor_size)
        tensor_data[workIdx] = value;
}

template <typename Op>
__global__ void elementwise(u64 size, f32 *out, f32 *a, f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b[workIdx]);
}

template <typename Op>
__global__ void elementwise_broadcast(const TensorMeta out_meta,
                                      const TensorMeta a_meta,
                                      const TensorMeta b_meta, f32 *out,
                                      const f32 *a, const f32 *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;

    u64 remaining = workIdx;
    u64 a_offset = 0, b_offset = 0;
    for (u32 i = 0; i < out_meta.ndim; i++) {
        u64 idx_i = remaining / out_meta.stride[i];
        remaining -= idx_i * out_meta.stride[i];
        a_offset += idx_i * a_meta.stride[i];
        b_offset += idx_i * b_meta.stride[i];
    }
    out[workIdx] = op(a[a_offset], b[b_offset]);
}

template <typename Op>
__global__ void elementwise(u64 size, f32 *out, const f32 *a, f32 b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b);
}

__global__ void mat_mul(TensorMeta out_meta, TensorMeta a_meta,
                        TensorMeta b_meta, f32 *out, f32 *a, f32 *b) {
    // Use myCol with x since x moves the fastest
    u64 myCol = threadIdx.x + blockIdx.x * blockDim.x;
    u64 myRow = threadIdx.y + blockIdx.y * blockDim.y;

    if (myRow >= out_meta.rows() || myCol >= out_meta.cols())
        return;

    f32 value = 0;
    for (u32 i = 0; i < a_meta.cols(); i++)
        value += a[a_meta.at(myRow, i)] * b[b_meta.at(i, myCol)];

    out[out_meta.at(myRow, myCol)] = value;
}

__global__ void mat_mul_tiled(TensorMeta out_meta, TensorMeta a_meta,
                              TensorMeta b_meta, f32 *out, f32 *a, f32 *b) {
    // Use myCol with x since x moves the fastest
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

        // Load tile with bounds check — out-of-range threads contribute 0
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

__global__ void tensor_sum_step(u64 size, f32 *out, f32 *tensor) {
    __shared__ f32 partial_sum[N_THREADS];

    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;

    partial_sum[threadIdx.x] = (workIdx < size) ? tensor[workIdx] : 0.0f;
    __syncthreads();

    for (u32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[blockIdx.x] = partial_sum[0];
}
__global__ void tensor_sum(TensorMeta out_meta, TensorMeta tensor_meta,
                           f32 *out, f32 *tensor, u32 dim) {}

// ---- Template dispatch ------------------------------------------------------

template <typename Op>
static void cuda_elementwise_binary(Tensor *out, const Tensor *a,
                                    const Tensor *b, Op op) {
    u32 threads = 256;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);

    if (tensor_shape_eq(a, b) && tensor_is_contiguous(a) &&
        tensor_is_contiguous(b)) {
        elementwise<<<blocks, threads>>>(out->size, out->data, a->data, b->data,
                                         op);
        return;
    }

    TensorMeta out_meta(out);
    TensorMeta a_meta(a, out->shape, out->ndim);
    TensorMeta b_meta(b, out->shape, out->ndim);
    elementwise_broadcast<<<blocks, threads>>>(out_meta, a_meta, b_meta,
                                               out->data, a->data, b->data, op);
    return;
}

// Tensor Meta
TensorMeta::TensorMeta(const Tensor *t) : size(t->size), ndim(t->ndim) {
    memcpy(shape, t->shape, ndim * sizeof(u32));
    memcpy(stride, t->stride, ndim * sizeof(u64));
}

TensorMeta::TensorMeta(const Tensor *t, const u32 *bcast_shape, u32 bcast_ndim)
    : size(t->size), ndim(bcast_ndim) {
    expanded_shape(t, bcast_shape, bcast_ndim, shape);
    expanded_stride(t, bcast_shape, bcast_ndim, stride);
}

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

void tensor_cuda_alloc(Tensor *tensor) {
    cudaMalloc(&tensor->data, sizeof(f32) * tensor->size);
    cudaMemset(tensor->data, 0, sizeof(f32) * tensor->size);
}
void tensor_cuda_free(Tensor *tensor) { cudaFree(tensor->data); }
void tensor_cuda_clear(Tensor *tensor) {
    cudaMemset(tensor->data, 0, sizeof(f32) * tensor->size);
}
// Wrapper functions for kernels

void tensor_cuda_fill(Tensor *tensor, f32 value) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(tensor->size, (u64)threads);

    tensor_fill<<<blocks, threads>>>(tensor->size, tensor->data, value);
}
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

// Scalar functions
void tensor_cuda_add(Tensor *out, const Tensor *tensor, f32 scalar) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);

    elementwise<<<blocks, threads>>>(out->size, out->data, tensor->data, scalar,
                                     AddOp{});
}

void tensor_cuda_sub(Tensor *out, const Tensor *tensor, f32 scalar) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);

    elementwise<<<blocks, threads>>>(out->size, out->data, tensor->data, scalar,
                                     SubOp{});
}
void tensor_cuda_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);

    elementwise<<<blocks, threads>>>(out->size, out->data, tensor->data, scalar,
                                     MulOp{});
}

void tensor_cuda_div(Tensor *out, const Tensor *tensor, f32 scalar) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out->size, (u64)threads);

    elementwise<<<blocks, threads>>>(out->size, out->data, tensor->data, scalar,
                                     DivOp{});
}

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

void tensor_cuda_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
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
void tensor_cuda_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
                     b32 clear_out) {
    if (out->size == 1) {
        tensor_cuda_sum(out, tensor, clear_out);
    } else {
        printf("Not supported yet");
    }
}
