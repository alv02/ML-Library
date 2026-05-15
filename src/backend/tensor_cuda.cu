#include "../../include/backend/tensor_cuda.hpp"
#include <atomic>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <curand_kernel.h>

static constexpr u32 TILE = 32;
static constexpr u32 N_THREADS = 512;

// ── Device helpers ───────────────────────────────────────────────────────────

template <typename T> __device__ T cuda_type_lowest() { return T(0); }
template <> __device__ float cuda_type_lowest<float>() { return -__FLT_MAX__; }
template <> __device__ unsigned int cuda_type_lowest<unsigned int>() {
    return 0;
}

// ── Functors ─────────────────────────────────────────────────────────────────

// Generic arithmetic functors (template operator() works for any T)
struct AddOp {
    template <typename T> __device__ T operator()(T a, T b) const {
        return a + b;
    }
};
struct SubOp {
    template <typename T> __device__ T operator()(T a, T b) const {
        return a - b;
    }
};
struct MulOp {
    template <typename T> __device__ T operator()(T a, T b) const {
        return a * b;
    }
};
struct DivOp {
    template <typename T> __device__ T operator()(T a, T b) const {
        return a / b;
    }
};
struct EqualOp {
    template <typename T> __device__ T operator()(T a, T b) const {
        return a == b ? T(1) : T(0);
    }
};

// f32-only functors
struct ReluFwdOp {
    __device__ f32 operator()(f32 x) const { return x > 0.0f ? x : 0.0f; }
};
// GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
struct GeluFwdOp {
    __device__ f32 operator()(f32 x) const {
        constexpr f32 c = 0.7978845608f;
        constexpr f32 k = 0.044715f;
        return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
    }
};

struct ReluBackwardOp {
    __device__ f32 operator()(f32 grad, f32 input) const {
        return input > 0.0f ? grad : 0.0f;
    }
};
struct GeluBackwardOp {
    __device__ f32 operator()(f32 g, f32 x) const {
        constexpr f32 c = 0.7978845608f;
        constexpr f32 k = 0.044715f;
        f32 t = tanhf(c * (x + k * x * x * x));
        f32 dtdx = (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        return g * 0.5f * (1.0f + t + x * dtdx);
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

template <typename T>
__global__ void fill_kernel(u64 tensor_size, T *tensor_data, T value) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < tensor_size)
        tensor_data[workIdx] = value;
}

template <typename T>
__global__ void arange_kernel(u64 tensor_size, T *tensor_data) {
    u64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < tensor_size)
        tensor_data[i] = (T)i;
}

template <typename T, typename Op>
__global__ void elementwise_unary(u64 size, T *out, const T *a, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx]);
}

template <typename T, typename Op>
__global__ void elementwise_unary(TensorMeta out_contig, TensorMeta out_actual,
                                  TensorMeta a_meta, T *out, const T *a,
                                  Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset]);
}

template <typename T, typename Op>
__global__ void elementwise_binary(u64 size, T *out, T *a, T *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b[workIdx]);
}

template <typename T, typename Op>
__global__ void
elementwise_broadcast(const TensorMeta out_contig, const TensorMeta out_actual,
                      const TensorMeta a_meta, const TensorMeta b_meta, T *out,
                      const T *a, const T *b, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    u64 b_offset = out_contig.offset_from(workIdx, b_meta);
    out[out_offset] = op(a[a_offset], b[b_offset]);
}

// out flat, a flat, b broadcasts — 1× offset_from
template <typename T, typename Op>
__global__ void elementwise_bcast_b(u64 size, T *out, const T *a,
                                     TensorMeta contig_meta, TensorMeta b_meta,
                                     const T *b, Op op) {
    u64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    out[i] = op(a[i], b[contig_meta.offset_from(i, b_meta)]);
}

// out flat, a broadcasts, b flat — 1× offset_from
template <typename T, typename Op>
__global__ void elementwise_bcast_a(u64 size, T *out,
                                     TensorMeta contig_meta, TensorMeta a_meta,
                                     const T *a, const T *b, Op op) {
    u64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    out[i] = op(a[contig_meta.offset_from(i, a_meta)], b[i]);
}

template <typename T, typename Op>
__global__ void elementwise_binary_scalar(u64 size, T *out, const T *a, T b,
                                          Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx < size)
        out[workIdx] = op(a[workIdx], b);
}

template <typename T, typename Op>
__global__ void elementwise_scalar(TensorMeta out_contig, TensorMeta out_actual,
                                   TensorMeta a_meta, T *out, const T *a,
                                   T scalar, Op op) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u64 a_offset = out_contig.offset_from(workIdx, a_meta);
    out[out_offset] = op(a[a_offset], scalar);
}

template <typename T>
__global__ void contiguous_kernel(TensorMeta contig_meta,
                                  TensorMeta actual_meta, T *dst, T *src) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= contig_meta.size)
        return;
    u64 src_offset = contig_meta.offset_from(workIdx, actual_meta);
    dst[workIdx] = src[src_offset];
}

// dst flat, src uses offset_from (1× call — most common: copy to new contiguous tensor)
template <typename T>
__global__ void copy_kernel_dst_flat(u64 size, T *dst,
                                      TensorMeta contig_meta, TensorMeta src_meta,
                                      const T *src) {
    u64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    dst[i] = src[contig_meta.offset_from(i, src_meta)];
}

// src flat, dst uses offset_from (1× call — writing into a non-contiguous view)
template <typename T>
__global__ void copy_kernel_src_flat(u64 size, T *dst,
                                      TensorMeta contig_meta, TensorMeta dst_meta,
                                      const T *src) {
    u64 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;
    dst[contig_meta.offset_from(i, dst_meta)] = src[i];
}

// General: both dst and src may be non-contiguous (2× offset_from)
template <typename T>
__global__ void copy_kernel(TensorMeta out_contig, TensorMeta out_actual,
                            TensorMeta src_meta, T *dst, const T *src) {
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

__global__ void mat_mul_tiled(TensorMeta out_meta, TensorMeta a_meta,
                              TensorMeta b_meta, f32 *out, f32 *a, f32 *b,
                              f32 beta) {
    constexpr u32 HTILE = TILE / 2;

    u64 row0 = threadIdx.y * 2 + blockIdx.x * TILE;
    u64 row1 = row0 + 1;
    u64 col0 = threadIdx.x * 2 + blockIdx.y * TILE;
    u64 col1 = col0 + 1;

    __shared__ f32 a_tile[TILE * TILE];
    __shared__ f32 b_tile[TILE * TILE];

    u32 n_tiles = cuda::ceil_div(a_meta.cols(), TILE);
    u32 tid = threadIdx.y * HTILE + threadIdx.x;

    f32 v00 = 0, v01 = 0, v10 = 0, v11 = 0;

    for (u32 n = 0; n < n_tiles; n++) {
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
            v00 += a0 * b0;
            v01 += a0 * b1;
            v10 += a1 * b0;
            v11 += a1 * b1;
        }

        __syncthreads();
    }

    u64 rows = out_meta.rows(), cols = out_meta.cols();
    if (row0 < rows && col0 < cols)
        out[out_meta.at(row0, col0)] =
            v00 + beta * out[out_meta.at(row0, col0)];
    if (row0 < rows && col1 < cols)
        out[out_meta.at(row0, col1)] =
            v01 + beta * out[out_meta.at(row0, col1)];
    if (row1 < rows && col0 < cols)
        out[out_meta.at(row1, col0)] =
            v10 + beta * out[out_meta.at(row1, col0)];
    if (row1 < rows && col1 < cols)
        out[out_meta.at(row1, col1)] =
            v11 + beta * out[out_meta.at(row1, col1)];
}

// ── Sum kernels ───────────────────────────────────────────────────────────────

// Global: tree reduction per block; call repeatedly (multi-pass) until 1 block.
template <typename T>
__global__ void tensor_sum_step(u64 size, T *out, T *tensor) {
    __shared__ T partial[N_THREADS];
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    partial[threadIdx.x] = (workIdx < size) ? tensor[workIdx] : T(0);
    __syncthreads();
    for (u32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = partial[0];
}

// Dim: one thread per output element, sequential loop over the reduced dim.
template <typename T>
__global__ void tensor_sum_dim_kernel(TensorMeta out_meta, TensorMeta tensor_meta,
                                      T *out, const T *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base = out_meta.offset_from(workIdx, tensor_meta);
    T sum = T(0);
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++)
        sum += tensor[base + i * tensor_meta.stride[dim]];
    out[workIdx] = sum;
}

// Skip: one block per element of the kept dim, tree reduction over all others.
__global__ void tensor_sum_skip_kernel(f32 *out, const f32 *src,
                                       u64 stride_dim, TensorMeta meta_contig,
                                       TensorMeta meta_src, u64 other_dims_size) {
    __shared__ f32 partial[N_THREADS];
    u32 c = blockIdx.x;
    u64 base = (u64)c * stride_dim;

    f32 sum = 0.0f;
    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x)
        sum += src[base + meta_contig.offset_from(i, meta_src)];

    partial[threadIdx.x] = sum;
    __syncthreads();
    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[c] = partial[0];
}

// ── Welford kernels ───────────────────────────────────────────────────────────

// Global: single block iterates all elements then tree-merges Welford state.
__global__ void welford_global_kernel(u64 size, f32 *mean, f32 *M2,
                                      const f32 *src) {
    __shared__ f32 s_mu[N_THREADS];
    __shared__ f32 s_M2[N_THREADS];
    __shared__ u32 s_n[N_THREADS];

    f32 mu = 0.0f, m2 = 0.0f;
    u32 n = 0;
    for (u64 i = threadIdx.x; i < size; i += blockDim.x) {
        f32 x = src[i];
        n++;
        f32 delta = x - mu;
        mu += delta / (f32)n;
        m2 += delta * (x - mu);
    }
    s_mu[threadIdx.x] = mu;
    s_M2[threadIdx.x] = m2;
    s_n[threadIdx.x] = n;
    __syncthreads();

    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u32 na = s_n[threadIdx.x], nb = s_n[threadIdx.x + s];
            u32 nc = na + nb;
            if (nc > 0) {
                f32 d = s_mu[threadIdx.x + s] - s_mu[threadIdx.x];
                s_mu[threadIdx.x] += d * ((f32)nb / nc);
                s_M2[threadIdx.x] += s_M2[threadIdx.x + s] +
                                     d * d * ((f32)na * nb / nc);
            }
            s_n[threadIdx.x] = nc;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean[0] = s_mu[0];
        M2[0] = s_M2[0];
    }
}

// Dim: one thread per output element, sequential Welford loop over the reduced dim.
__global__ void welford_dim_kernel(TensorMeta out_meta, TensorMeta tensor_meta,
                                   f32 *mean, f32 *M2, const f32 *src, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base = out_meta.offset_from(workIdx, tensor_meta);
    u64 stride = tensor_meta.stride[dim];
    u32 count = tensor_meta.shape[dim];

    f32 mu = 0.0f, m2 = 0.0f;
    u32 n = 0;
    for (u32 i = 0; i < count; i++) {
        f32 x = src[base + i * stride];
        n++;
        f32 delta = x - mu;
        mu += delta / (f32)n;
        m2 += delta * (x - mu);
    }
    mean[workIdx] = mu;
    M2[workIdx] = m2;
}

// Skip: one block per element of the kept dim, Welford tree over all others.
__global__ void welford_mean_M2(f32 *mean, f32 *M2, const f32 *src,
                                u64 stride_dim, TensorMeta meta_contig,
                                TensorMeta meta_src, u64 other_dims_size) {
    __shared__ u32 s_n[N_THREADS];
    __shared__ f32 s_mu[N_THREADS];
    __shared__ f32 s_M2[N_THREADS];

    u32 c = blockIdx.x;
    u64 base = (u64)c * stride_dim;
    f32 mu = 0.0f, m2 = 0.0f;
    u32 n = 0;

    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x) {
        f32 x = src[base + meta_contig.offset_from(i, meta_src)];
        n++;
        f32 delta = x - mu;
        mu += delta / (f32)n;
        m2 += delta * (x - mu);
    }
    s_n[threadIdx.x] = n;
    s_mu[threadIdx.x] = mu;
    s_M2[threadIdx.x] = m2;
    __syncthreads();

    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            u32 na = s_n[threadIdx.x], nb = s_n[threadIdx.x + s];
            u32 nc = na + nb;
            if (nc > 0) {
                f32 d = s_mu[threadIdx.x + s] - s_mu[threadIdx.x];
                s_mu[threadIdx.x] += d * ((f32)nb / nc);
                s_M2[threadIdx.x] += s_M2[threadIdx.x + s] +
                                     d * d * ((f32)na * nb / nc);
            }
            s_n[threadIdx.x] = nc;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        mean[c] = s_mu[0];
        M2[c] = s_M2[0];
    }
}

template <typename T>
__global__ void tensor_max_step(u64 size, T *out, T *tensor) {
    __shared__ T partial[N_THREADS];
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    partial[threadIdx.x] =
        (workIdx < size) ? tensor[workIdx] : cuda_type_lowest<T>();
    __syncthreads();
    for (u32 stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (partial[threadIdx.x + stride] > partial[threadIdx.x])
                partial[threadIdx.x] = partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = partial[0];
}

template <typename T>
__global__ void tensor_max_kernel(TensorMeta out_meta, TensorMeta tensor_meta,
                                  T *out, T *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    T max_val = cuda_type_lowest<T>();
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++) {
        T val = tensor[base_offset + i * tensor_meta.stride[dim]];
        if (val > max_val)
            max_val = val;
    }
    out[workIdx] = max_val;
}

template <typename T>
__global__ void tensor_argmax_kernel(TensorMeta out_meta,
                                     TensorMeta tensor_meta, u32 *out,
                                     T *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    T max_val = cuda_type_lowest<T>();
    u32 max_idx = 0;
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++) {
        T val = tensor[base_offset + i * tensor_meta.stride[dim]];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    out[workIdx] = max_idx;
}

template <typename T>
__global__ void
tensor_unfold2d_kernel(TensorMeta dst_meta, TensorMeta dst_meta_contig,
                       TensorMeta src_meta, Unfold2dParams params, T *dst,
                       const T *src) {
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

    if (h < 0 || h >= (i32)src_meta.shape[2] || w < 0 ||
        w >= (i32)src_meta.shape[3]) {
        dst[dst_offset] = T(params.pad_constant);
        return;
    }

    u64 src_offset = n * src_meta.stride[0] + c * src_meta.stride[1] +
                     h * src_meta.stride[2] + w * src_meta.stride[3];

    dst[dst_offset] = src[src_offset];
}

template <typename T>
__global__ void tensor_fold2d_kernel(TensorMeta col_meta,
                                     TensorMeta col_meta_contig,
                                     TensorMeta dst_meta, Unfold2dParams params,
                                     const T *col, T *dst) {
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

template <typename T>
__global__ void scatter_add_kernel(TensorMeta out_meta, TensorMeta src_meta,
                                   TensorMeta indices_meta,
                                   TensorMeta out_base_meta,
                                   TensorMeta src_contig, T *out, const T *src,
                                   const u32 *indices, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= src_contig.size)
        return;

    u64 src_offset = src_contig.offset_from(workIdx, src_meta);
    u64 out_base_offset = src_contig.offset_from(workIdx, out_base_meta);
    u64 indices_offset = src_contig.offset_from(workIdx, indices_meta);
    out_base_offset += (u64)indices[indices_offset] * out_meta.stride[dim];

    atomicAdd(&out[out_base_offset], src[src_offset]);
}

// ── Template dispatch helpers
// ─────────────────────────────────────────────────

template <typename T>
static TensorMeta make_contig_meta(const TensorImpl<T> &t) {
    TensorMeta m(t);
    tensor_compute_strides(m.stride, m.shape, m.ndim);
    return m;
}

template <typename T, typename Op>
static void cuda_elementwise_unary(TensorImpl<T> &out, const TensorImpl<T> &a,
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

template <typename T, typename Op>
static void cuda_elementwise_binary(TensorImpl<T> &out, const TensorImpl<T> &a,
                                    const TensorImpl<T> &b, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);

    // Fast path: all contiguous, same shape — no offset_from
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(out) &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        elementwise_binary<<<blocks, threads>>>(out.numel(), out.data(),
                                                a.data(), b.data(), op);
        return;
    }

    TensorMeta contig_meta = make_contig_meta(out);

    if (tensor_is_contiguous(out)) {
        bool a_flat = tensor_is_contiguous(a) && tensor_shape_eq(a, out);
        bool b_flat = tensor_is_contiguous(b) && tensor_shape_eq(b, out);

        if (a_flat) {
            TensorMeta b_meta(b, out.shape, out.ndim);
            elementwise_bcast_b<<<blocks, threads>>>(out.numel(), out.data(),
                                                      a.data(), contig_meta,
                                                      b_meta, b.data(), op);
            return;
        }
        if (b_flat) {
            TensorMeta a_meta(a, out.shape, out.ndim);
            elementwise_bcast_a<<<blocks, threads>>>(out.numel(), out.data(),
                                                      contig_meta, a_meta,
                                                      a.data(), b.data(), op);
            return;
        }
        // Both broadcast, out flat: pass contig_meta for out_actual so
        // the out offset_from becomes an identity (offset_from(i, contig)==i).
        TensorMeta a_meta(a, out.shape, out.ndim);
        TensorMeta b_meta(b, out.shape, out.ndim);
        elementwise_broadcast<<<blocks, threads>>>(contig_meta, contig_meta,
                                                    a_meta, b_meta,
                                                    out.data(), a.data(),
                                                    b.data(), op);
        return;
    }

    // General: out non-contiguous
    TensorMeta out_actual(out);
    TensorMeta a_meta(a, out.shape, out.ndim);
    TensorMeta b_meta(b, out.shape, out.ndim);
    elementwise_broadcast<<<blocks, threads>>>(contig_meta, out_actual, a_meta,
                                               b_meta, out.data(), a.data(),
                                               b.data(), op);
}

template <typename T, typename Op>
static void cuda_elementwise_scalar(TensorImpl<T> &out, const TensorImpl<T> &a,
                                    T scalar, Op op) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        elementwise_binary_scalar<<<blocks, threads>>>(out.numel(), out.data(),
                                                       a.data(), scalar, op);
        return;
    }
    TensorMeta out_contig = make_contig_meta(out);
    TensorMeta out_actual(out);
    TensorMeta a_meta(a);
    elementwise_scalar<<<blocks, threads>>>(out_contig, out_actual, a_meta,
                                            out.data(), a.data(), scalar, op);
}

// ── Host functions
// ────────────────────────────────────────────────────────────

// ---- memory management ---------------------------------------------------

template <typename T>
void tensor_cuda_copy(TensorImpl<T> &dst, const TensorImpl<T> &src) {
    if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        cudaMemcpyAsync(dst.data(), src.data(), src.numel() * sizeof(T),
                        cudaMemcpyDeviceToDevice, 0);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), (u64)threads);
    TensorMeta contig_meta = make_contig_meta(dst);
    if (tensor_is_contiguous(dst)) {
        TensorMeta src_meta(src);
        copy_kernel_dst_flat<<<blocks, threads>>>(dst.numel(), dst.data(),
                                                   contig_meta, src_meta,
                                                   src.data());
        return;
    }
    if (tensor_is_contiguous(src)) {
        TensorMeta dst_actual(dst);
        copy_kernel_src_flat<<<blocks, threads>>>(dst.numel(), dst.data(),
                                                   contig_meta, dst_actual,
                                                   src.data());
        return;
    }
    TensorMeta dst_actual(dst);
    TensorMeta src_meta(src);
    copy_kernel<<<blocks, threads>>>(contig_meta, dst_actual, src_meta,
                                     dst.data(), src.data());
}

template <typename T>
void tensor_cuda_contiguous(TensorImpl<T> &t) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(t.numel(), (u64)threads);

    TensorMeta src_meta(t);
    TensorMeta src_contig = make_contig_meta(t);

    Tensor<T> temp_t = Tensor<T>::make(t.ndim, t.shape, true);
    TensorImpl<T> &temp = temp_t.impl();

    contiguous_kernel<<<blocks, threads>>>(src_contig, src_meta, temp.data(),
                                           t.data());

    cudaMemcpyAsync(t.data(), temp.data(), t.numel() * sizeof(T),
                    cudaMemcpyDeviceToDevice, 0);
    tensor_compute_strides(t.stride, t.shape, t.ndim);
}

// ---- fill / clear --------------------------------------------------------

template <typename T> void tensor_cuda_fill(TensorImpl<T> &tensor, T value) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(tensor.numel(), (u64)threads);
    fill_kernel<<<blocks, threads>>>(tensor.numel(), tensor.data(), value);
}

template <typename T> void tensor_cuda_clear(TensorImpl<T> &tensor) {
    cudaMemsetAsync(tensor.data(), 0, sizeof(T) * tensor.numel(), 0);
}

template <typename T> void tensor_cuda_arange(TensorImpl<T> &out) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), (u64)threads);
    arange_kernel<<<blocks, threads>>>(out.numel(), out.data());
}

// ---- activations — f32 only ----------------------------------------------

void tensor_cuda_relu(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    cuda_elementwise_unary(dst, src, ReluFwdOp{});
}

void tensor_cuda_gelu(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    cuda_elementwise_unary(dst, src, GeluFwdOp{});
}

void tensor_cuda_exp(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    cuda_elementwise_unary(dst, src, ExpOp{});
}

void tensor_cuda_log(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    cuda_elementwise_unary(dst, src, LogOp{});
}

void tensor_cuda_sqrt(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    cuda_elementwise_unary(dst, src, SqrtOp{});
}

// ---- elementwise binary --------------------------------------------------

template <typename T>
void tensor_cuda_add(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b) {
    cuda_elementwise_binary(out, a, b, AddOp{});
}

template <typename T>
void tensor_cuda_sub(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b) {
    cuda_elementwise_binary(out, a, b, SubOp{});
}

template <typename T>
void tensor_cuda_mul(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b) {
    cuda_elementwise_binary(out, a, b, MulOp{});
}

template <typename T>
void tensor_cuda_div(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b) {
    cuda_elementwise_binary(out, a, b, DivOp{});
}

template <typename T>
void tensor_cuda_equal(TensorImpl<T> &out, const TensorImpl<T> &a,
                       const TensorImpl<T> &b) {
    cuda_elementwise_binary(out, a, b, EqualOp{});
}

void tensor_cuda_relu_backward(TensorImpl<f32> &out,
                               const TensorImpl<f32> &grad,
                               const TensorImpl<f32> &in) {
    cuda_elementwise_binary(out, grad, in, ReluBackwardOp{});
}

void tensor_cuda_gelu_backward(TensorImpl<f32> &out,
                               const TensorImpl<f32> &grad,
                               const TensorImpl<f32> &input) {
    cuda_elementwise_binary(out, grad, input, GeluBackwardOp{});
}

// ---- scalar operations ---------------------------------------------------

template <typename T>
void tensor_cuda_add(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                     T scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, AddOp{});
}

template <typename T>
void tensor_cuda_sub(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                     T scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, SubOp{});
}

template <typename T>
void tensor_cuda_mul(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                     T scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, MulOp{});
}

template <typename T>
void tensor_cuda_div(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                     T scalar) {
    cuda_elementwise_scalar(out, tensor, scalar, DivOp{});
}

// ---- matrix multiply — f32 only -----------------------------------------

void tensor_cuda_mat_mul(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                         const TensorImpl<f32> &b, b32 clear_out) {
    constexpr u32 HTILE = TILE / 2;
    u32 row_tiles = cuda::ceil_div(out.shape[0], TILE);
    u32 col_tiles = cuda::ceil_div(out.shape[1], TILE);
    dim3 threadsPerBlock(HTILE, HTILE);
    dim3 blocks(row_tiles, col_tiles);
    TensorMeta out_meta(out);
    TensorMeta a_meta(a);
    TensorMeta b_meta(b);
    f32 beta = clear_out ? 0.0f : 1.0f;
    mat_mul_tiled<<<blocks, threadsPerBlock>>>(
        out_meta, a_meta, b_meta, out.data(), a.data(), b.data(), beta);
}

// ---- cuBLAS matrix multiply — f32 only ----------------------------------

static cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        cublasCreate(&h);
        return h;
    }();
    return handle;
}

void tensor_cuda_mat_mul_cublas(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                const TensorImpl<f32> &b, b32 clear_out) {
    int M = (int)a.shape[0], K = (int)a.shape[1], N = (int)b.shape[1];
    float alpha = 1.0f;
    float beta = clear_out ? 0.0f : 1.0f;

    cublasOperation_t op_b = (b.stride[1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_b = (b.stride[1] == 1) ? (int)b.stride[0] : (int)b.stride[1];

    cublasOperation_t op_a = (a.stride[1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_a = (a.stride[1] == 1) ? (int)a.stride[0] : (int)a.stride[1];

    int ldc = (int)out.stride[0];

    cublasSgemm(cublas_handle(), op_b, op_a, N, M, K, &alpha, b.data(), ld_b,
                a.data(), ld_a, &beta, out.data(), ldc);
}

void tensor_cuda_mat_mul_batched(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                 const TensorImpl<f32> &b, b32 clear_out) {
    u32 nd = a.ndim;
    int M = (int)a.shape[nd - 2], K = (int)a.shape[nd - 1],
        N = (int)b.shape[nd - 1];
    u32 batch_ndim = nd - 2;
    int batch = 1;
    for (u32 i = 0; i < batch_ndim; i++)
        batch *= (int)a.shape[i];
    float alpha = 1.0f, beta = clear_out ? 0.0f : 1.0f;

    cublasOperation_t op_b =
        (b.stride[nd - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_b =
        (int)((b.stride[nd - 1] == 1) ? b.stride[nd - 2] : b.stride[nd - 1]);
    cublasOperation_t op_a =
        (a.stride[nd - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_a =
        (int)((a.stride[nd - 1] == 1) ? a.stride[nd - 2] : a.stride[nd - 1]);
    int ld_c = (int)out.stride[nd - 2];

    // Check if the batch dims are contiguous (stride[d] == stride[d+1]*shape[d+1])
    // so we can use the fast StridedBatched path. Otherwise use per-pointer path.
    bool batch_contiguous = true;
    for (u32 d = 0; d + 1 < batch_ndim; d++) {
        if (a.stride[d] != a.stride[d + 1] * a.shape[d + 1] ||
            b.stride[d] != b.stride[d + 1] * b.shape[d + 1] ||
            out.stride[d] != out.stride[d + 1] * out.shape[d + 1]) {
            batch_contiguous = false;
            break;
        }
    }
    // Also check the innermost batch dim stride matches what StridedBatched expects.
    // For a single batch dim (nd==3) this is always fine.

    if (batch_contiguous || batch_ndim == 1) {
        long long sa = (batch > 1) ? (long long)a.stride[nd - 3] : (long long)M * K;
        long long sb = (batch > 1) ? (long long)b.stride[nd - 3] : (long long)K * N;
        long long sc = (batch > 1) ? (long long)out.stride[nd - 3] : (long long)M * N;
        cublasSgemmStridedBatched(cublas_handle(), op_b, op_a, N, M, K, &alpha,
                                  b.data(), ld_b, sb, a.data(), ld_a, sa, &beta,
                                  out.data(), ld_c, sc, batch);
    } else {
        // Non-contiguous batch dims: make contiguous D2D copies then use
        // StridedBatched, avoiding 3 synchronous H2D memcpys (pointer-array path).
        u32 a_shape[MAX_NDIM], b_shape[MAX_NDIM];
        memcpy(a_shape, a.shape, nd * sizeof(u32));
        memcpy(b_shape, b.shape, nd * sizeof(u32));
        Tensor<f32> ta = Tensor<f32>::make(nd, a_shape, true);
        Tensor<f32> tb = Tensor<f32>::make(nd, b_shape, true);
        tensor_cuda_copy(*ta, a);
        tensor_cuda_copy(*tb, b);

        // After making contiguous, stride[nd-1] == 1 for both — always CUBLAS_OP_N.
        int ld_a_c = (int)ta->stride[nd - 2];
        int ld_b_c = (int)tb->stride[nd - 2];
        int ld_c_c = (int)out.stride[nd - 2];
        long long sa = (long long)ta->stride[nd - 3];
        long long sb = (long long)tb->stride[nd - 3];
        long long sc = (long long)out.stride[nd - 3];
        cublasSgemmStridedBatched(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, tb->data(), ld_b_c, sb,
                                  ta->data(), ld_a_c, sa, &beta,
                                  out.data(), ld_c_c, sc, batch);
    }
}

// ---- reduction -----------------------------------------------------------

// Helper for skip-dim kernels: builds metas for "all dims except dim".
static void build_skip_meta(u64 &stride_dim, u64 &other_dims_size,
                             TensorMeta &meta_contig, TensorMeta &meta_src,
                             const TensorImpl<f32> &t, u32 dim) {
    stride_dim = t.stride[dim];
    other_dims_size = t.numel() / t.shape[dim];
    meta_contig.ndim = meta_src.ndim = t.ndim - 1;
    meta_contig.size = meta_src.size = other_dims_size;
    for (u32 d = 0, o = 0; d < t.ndim; d++) {
        if (d == dim) continue;
        meta_contig.shape[o] = t.shape[d];
        meta_src.shape[o]    = t.shape[d];
        meta_src.stride[o]   = t.stride[d];
        o++;
    }
    tensor_compute_strides(meta_contig.stride, meta_contig.shape,
                           meta_contig.ndim);
}

// Global sum: multi-pass tree reduction → scalar. Requires contiguous input.
template <typename T>
void tensor_cuda_sum_global(TensorImpl<T> &out, const TensorImpl<T> &tensor) {
    u32 threads = N_THREADS;
    const T *cur_data = tensor.data();
    u64 cur_size = tensor.numel();
    Tensor<T> prev_temp;

    while (true) {
        u32 blocks = cuda::ceil_div(cur_size, (u64)threads);
        if (blocks == 1) {
            tensor_sum_step<<<1, threads>>>(cur_size, out.data(),
                                            const_cast<T *>(cur_data));
            break;
        }
        u32 shape[] = {blocks};
        Tensor<T> next = Tensor<T>::make(1, shape, true);
        tensor_sum_step<<<blocks, threads>>>(cur_size, next->data(),
                                             const_cast<T *>(cur_data));
        cur_data = next->data();
        cur_size = blocks;
        prev_temp = std::move(next);
    }
}

// Dim sum: one thread per output element, sequential loop over reduced dim.
template <typename T>
void tensor_cuda_sum_dim(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                         u32 dim) {
    if (out.numel() == 1) {
        tensor_cuda_sum_global(out, tensor);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_sum_dim_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                               tensor.data(), dim);
}

// Skip sum: reduce all dims except dim. One block per kept element.
void tensor_cuda_sum_skip(TensorImpl<f32> &out, const TensorImpl<f32> &src,
                          u32 dim) {
    u64 stride_dim, other_dims_size;
    TensorMeta meta_contig, meta_src;
    build_skip_meta(stride_dim, other_dims_size, meta_contig, meta_src, src, dim);
    u32 C = src.shape[dim];
    tensor_sum_skip_kernel<<<C, N_THREADS>>>(out.data(), src.data(), stride_dim,
                                              meta_contig, meta_src,
                                              other_dims_size);
}

// Global welford: single block grid-strides all elements. Requires contiguous input.
void tensor_cuda_welford_global(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                                const TensorImpl<f32> &src) {
    welford_global_kernel<<<1, N_THREADS>>>(src.numel(), mean.data(), M2.data(),
                                             src.data());
}

// Dim welford: one thread per output element, sequential Welford over reduced dim.
void tensor_cuda_welford_dim(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                             const TensorImpl<f32> &src, u32 dim) {
    if (mean.numel() == 1) {
        tensor_cuda_welford_global(mean, M2, src);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(mean.numel(), u64(threads));
    TensorMeta out_meta(mean);
    TensorMeta tensor_meta(src);
    welford_dim_kernel<<<blocks, threads>>>(out_meta, tensor_meta, mean.data(),
                                            M2.data(), src.data(), dim);
}

// Skip welford: reduce all dims except dim. One block per kept element.
void tensor_cuda_welford_skip(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                               const TensorImpl<f32> &src, u32 dim) {
    u64 stride_dim, other_dims_size;
    TensorMeta meta_contig, meta_src;
    build_skip_meta(stride_dim, other_dims_size, meta_contig, meta_src, src, dim);
    u32 C = src.shape[dim];
    welford_mean_M2<<<C, N_THREADS>>>(mean.data(), M2.data(), src.data(),
                                       stride_dim, meta_contig, meta_src,
                                       other_dims_size);
}

template <typename T>
void tensor_cuda_max(TensorImpl<T> &out, const TensorImpl<T> &tensor) {
    u32 threads = N_THREADS;
    const T *cur_data = tensor.data();
    u64 cur_size = tensor.numel();
    Tensor<T> prev_temp;

    while (true) {
        u32 blocks = cuda::ceil_div(cur_size, (u64)threads);
        if (blocks == 1) {
            tensor_max_step<<<1, threads>>>(cur_size, out.data(),
                                            const_cast<T *>(cur_data));
            break;
        }
        u32 shape[] = {blocks};
        Tensor<T> next = Tensor<T>::make(1, shape, true);
        tensor_max_step<<<blocks, threads>>>(cur_size, next->data(),
                                             const_cast<T *>(cur_data));
        cur_data = next->data();
        cur_size = blocks;
        prev_temp = std::move(next);
    }
}

template <typename T>
void tensor_cuda_max(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_max_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                           tensor.data(), dim);
}

template <typename T>
void tensor_cuda_argmax(TensorImpl<u32> &out, const TensorImpl<T> &tensor,
                        u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_argmax_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                              tensor.data(), dim);
}

// ---- scattering ----------------------------------------------------------

template <typename T>
void tensor_cuda_scatter_add(TensorImpl<T> &out, const TensorImpl<T> &src,
                             const TensorImpl<u32> &indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(src.numel(), u64(threads));

    TensorMeta out_meta(out);
    TensorMeta src_meta(src);
    TensorMeta src_contig = make_contig_meta(src);
    TensorMeta out_base_meta(out);
    out_base_meta.stride[dim] = 0;
    TensorMeta indices_meta(indices);

    scatter_add_kernel<<<blocks, threads>>>(
        out_meta, src_meta, indices_meta, out_base_meta, src_contig, out.data(),
        src.data(), indices.data(), dim);
}

// ---- indexing ------------------------------------------------------------

template <typename T>
__global__ void gather_kernel(TensorMeta dst_meta, TensorMeta src_meta,
                              TensorMeta indices_meta, TensorMeta src_base_meta,
                              TensorMeta dst_contig, T *dst, const T *src,
                              const u32 *indices, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= dst_contig.size)
        return;

    u64 dst_offset = dst_contig.offset_from(workIdx, dst_meta);
    u64 src_base_offset = dst_contig.offset_from(workIdx, src_base_meta);
    u64 indices_offset = dst_contig.offset_from(workIdx, indices_meta);
    src_base_offset += (u64)indices[indices_offset] * src_meta.stride[dim];

    dst[dst_offset] = src[src_base_offset];
}

template <typename T>
void tensor_cuda_gather(TensorImpl<T> &dst, const TensorImpl<T> &src,
                        const TensorImpl<u32> &indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), u64(threads));

    TensorMeta dst_meta(dst);
    TensorMeta dst_contig = make_contig_meta(dst);
    TensorMeta src_meta(src);
    TensorMeta src_base_meta(src);
    src_base_meta.stride[dim] = 0;
    TensorMeta indices_meta(indices);

    gather_kernel<<<blocks, threads>>>(dst_meta, src_meta, indices_meta,
                                       src_base_meta, dst_contig, dst.data(),
                                       src.data(), indices.data(), dim);
}

// ---- initializing — f32 only
// --------------------------------------------

__global__ void causal_mask_kernel(f32 *out, TensorMeta out_contig,
                                   TensorMeta out_actual, u32 rows, u32 cols) {
    u64 workIdx = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (workIdx >= out_contig.size)
        return;
    u64 out_offset = out_contig.offset_from(workIdx, out_actual);
    u32 i = (workIdx % ((u64)rows * cols)) / cols;
    u32 j = workIdx % cols;
    out[out_offset] = (j <= i) ? 0.0f : -1e9f;
}

void tensor_cuda_causal_mask(TensorImpl<f32> &t) {
    u32 rows = t.shape[t.ndim - 2];
    u32 cols = t.shape[t.ndim - 1];
    TensorMeta out_contig = make_contig_meta(t);
    TensorMeta out_actual(t);
    u32 blocks = cuda::ceil_div(t.numel(), (u64)N_THREADS);
    causal_mask_kernel<<<blocks, N_THREADS>>>(t.data(), out_contig, out_actual,
                                              rows, cols);
}

void tensor_cuda_he_init(TensorImpl<f32> &tensor, float std) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    static u64 seed_counter = 0;
    curandSetPseudoRandomGeneratorSeed(
        gen, (u64)time(nullptr) ^ (seed_counter++ * 6364136223846793005ULL));
    u32 in_features = tensor.shape[COL_DIM(tensor)];
    float stddev = (std > 0.0f) ? std : sqrtf(2.0f / in_features);
    curandGenerateNormal(gen, tensor.data(), tensor.numel(), 0.0f, stddev);
    curandDestroyGenerator(gen);
}

__global__ void dropout_mask_kernel(f32 *mask, u64 n, f32 p, f32 scale) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    mask[i] = (mask[i] >= p) ? scale : 0.0f;
}

void tensor_cuda_dropout_mask(TensorImpl<f32> &mask, f32 p) {
    static std::atomic<u64> seed_counter{0};
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed_counter.fetch_add(1) ^
                                                0xdeadbeefULL);
    curandGenerateUniform(gen, mask.data(), mask.numel());
    curandDestroyGenerator(gen);

    f32 scale = 1.0f / (1.0f - p);
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(mask.numel(), (u64)threads);
    dropout_mask_kernel<<<blocks, threads>>>(mask.data(), mask.numel(), p,
                                             scale);
}

// ---- spatial / patch operations
// ------------------------------------------

template <typename T>
void tensor_cuda_unfold2d(TensorImpl<T> &dst, const TensorImpl<T> &src,
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

    tensor_unfold2d_kernel<<<blocks, threads>>>(
        dst_meta, dst_meta_contig, src_meta, params, dst.data(), src.data());

    u32 shape_[MAX_NDIM] = {N, L, C * params.k_h * params.k_w};
    tensor_reshape(dst, shape_, 3);
}

template <typename T>
void tensor_cuda_fold2d(TensorImpl<T> &dst, const TensorImpl<T> &col,
                        Unfold2dParams params) {
    u32 N = dst.shape[0];
    u32 C = dst.shape[1];
    u32 H = dst.shape[2];
    u32 W = dst.shape[3];
    params.compute_output_size(H, W);

    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w,
                            C, params.k_h, params.k_w};
    u64 stride6[MAX_NDIM];
    tensor_compute_strides(stride6, shape6, 6);

    TensorMeta col_meta;
    col_meta.ndim = 6;
    col_meta.size = col.numel();
    memcpy(col_meta.shape, shape6, 6 * sizeof(u32));
    memcpy(col_meta.stride, stride6, 6 * sizeof(u64));

    TensorMeta col_meta_contig = col_meta;
    TensorMeta dst_meta(dst);

    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(col.numel(), u64(threads));
    tensor_fold2d_kernel<<<blocks, threads>>>(
        col_meta, col_meta_contig, dst_meta, params, col.data(), dst.data());
}

// ---- comparison — f32 only
// ----------------------------------------------

b32 tensor_cuda_equals(const TensorImpl<f32> &a, const TensorImpl<f32> &b,
                       f32 tol) {
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

// ---- fused batch norm — f32 only ----------------------------------------

__global__ void bn_fwd_normalize(f32 *out, f32 *xhat, const f32 *inp,
                                 const f32 *mean, const f32 *m2,
                                 const f32 *gamma, const f32 *beta,
                                 f32 count_inv, f32 eps, u64 stride_c,
                                 TensorMeta meta_contig, TensorMeta meta_src,
                                 u64 other_dims_size) {
    u32 c = blockIdx.x;
    f32 mu = mean[c];
    f32 inv_std = rsqrtf(m2[c] * count_inv + eps);
    f32 g = gamma[c];
    f32 b = beta[c];
    u64 base = (u64)c * stride_c;

    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x) {
        u64 off = base + meta_contig.offset_from(i, meta_src);
        f32 xh = (inp[off] - mu) * inv_std;
        xhat[off] = xh;
        out[off] = fmaf(g, xh, b);
    }
}

__global__ void bn_bwd(f32 *dx, f32 *d_gamma, f32 *d_beta, const f32 *grad,
                       const f32 *xhat, const f32 *gamma, const f32 *var,
                       f32 m_inv, f32 eps, u64 stride_c, TensorMeta meta_contig,
                       TensorMeta meta_src, u64 other_dims_size) {
    u32 c = blockIdx.x;
    f32 g = gamma[c];
    f32 inv_std = rsqrtf(var[c] + eps);
    u64 base = (u64)c * stride_c;

    __shared__ f32 s_sum_grad[N_THREADS];
    __shared__ f32 s_sum_grad_xhat[N_THREADS];

    f32 sum_grad = 0.f, sum_grad_xhat = 0.f;

    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x) {
        u64 off = base + meta_contig.offset_from(i, meta_src);
        f32 gv = grad[off];
        f32 xh = xhat[off];
        sum_grad += gv;
        sum_grad_xhat += gv * xh;
    }

    s_sum_grad[threadIdx.x] = sum_grad;
    s_sum_grad_xhat[threadIdx.x] = sum_grad_xhat;
    __syncthreads();

    for (u32 s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_grad[threadIdx.x] += s_sum_grad[threadIdx.x + s];
            s_sum_grad_xhat[threadIdx.x] += s_sum_grad_xhat[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_gamma[c] += s_sum_grad_xhat[0];
        d_beta[c] += s_sum_grad[0];
    }
    __syncthreads();

    f32 mean_grad = s_sum_grad[0] * m_inv;
    f32 mean_grad_xhat = s_sum_grad_xhat[0] * m_inv;
    f32 coeff = g * inv_std;

    for (u64 i = threadIdx.x; i < other_dims_size; i += blockDim.x) {
        u64 off = base + meta_contig.offset_from(i, meta_src);
        dx[off] += coeff * (grad[off] - mean_grad - xhat[off] * mean_grad_xhat);
    }
}

static void bn_build_meta(u64 &stride_c, u64 &other_dims_size,
                          TensorMeta &meta_contig, TensorMeta &meta_src,
                          const TensorImpl<f32> &t) {
    stride_c = t.stride[1];
    other_dims_size = t.numel() / t.shape[1];
    meta_contig.ndim = meta_src.ndim = t.ndim - 1;
    meta_contig.size = meta_src.size = other_dims_size;
    for (u32 d = 0, o = 0; d < t.ndim; d++) {
        if (d == 1)
            continue;
        meta_contig.shape[o] = t.shape[d];
        meta_src.shape[o] = t.shape[d];
        meta_src.stride[o] = t.stride[d];
        o++;
    }
    tensor_compute_strides(meta_contig.stride, meta_contig.shape,
                           meta_contig.ndim);
}

void tensor_cuda_bn_fwd_normalize(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                                  const TensorImpl<f32> &inp,
                                  const TensorImpl<f32> &mean,
                                  const TensorImpl<f32> &m2,
                                  const TensorImpl<f32> &gamma,
                                  const TensorImpl<f32> &beta, u64 count,
                                  f32 eps) {
    u64 stride_c, other_dims_size;
    TensorMeta meta_contig, meta_src;
    bn_build_meta(stride_c, other_dims_size, meta_contig, meta_src, inp);
    u32 C = inp.shape[1];
    bn_fwd_normalize<<<C, N_THREADS>>>(out.data(), xhat.data(), inp.data(),
                                       mean.data(), m2.data(), gamma.data(),
                                       beta.data(), 1.0f / count, eps, stride_c,
                                       meta_contig, meta_src, other_dims_size);
}
void tensor_cuda_bn_bwd(TensorImpl<f32> &dx, TensorImpl<f32> &d_gamma,
                        TensorImpl<f32> &d_beta, const TensorImpl<f32> &grad,
                        const TensorImpl<f32> &xhat,
                        const TensorImpl<f32> &gamma,
                        const TensorImpl<f32> &var, f32 m, f32 eps) {
    u64 stride_c, other_dims_size;
    TensorMeta meta_contig, meta_src;
    bn_build_meta(stride_c, other_dims_size, meta_contig, meta_src, grad);
    u32 C = grad.shape[1];
    bn_bwd<<<C, N_THREADS>>>(dx.data(), d_gamma.data(), d_beta.data(),
                             grad.data(), xhat.data(), gamma.data(), var.data(),
                             1.0f / m, eps, stride_c, meta_contig, meta_src,
                             other_dims_size);
}

// ── Fused softmax kernels ─────────────────────────────────────────────────────

__global__ void softmax_fwd_kernel(f32 *out, const f32 *inp, u64 D) {
    u64 base = (u64)blockIdx.x * D;
    __shared__ f32 s[N_THREADS];

    f32 vmax = -__FLT_MAX__;
    for (u64 i = threadIdx.x; i < D; i += blockDim.x)
        vmax = fmaxf(vmax, inp[base + i]);
    s[threadIdx.x] = vmax;
    __syncthreads();
    for (u32 half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + half]);
        __syncthreads();
    }
    f32 row_max = s[0];

    f32 sum = 0.f;
    for (u64 i = threadIdx.x; i < D; i += blockDim.x) {
        f32 e = expf(inp[base + i] - row_max);
        out[base + i] = e;
        sum += e;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (u32 half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half)
            s[threadIdx.x] += s[threadIdx.x + half];
        __syncthreads();
    }
    f32 inv_sum = 1.f / s[0];

    for (u64 i = threadIdx.x; i < D; i += blockDim.x)
        out[base + i] *= inv_sum;
}

__global__ void softmax_bwd_kernel(f32 *dx, const f32 *s, const f32 *grad,
                                    u64 D) {
    u64 base = (u64)blockIdx.x * D;
    __shared__ f32 sh[N_THREADS];

    f32 dot = 0.f;
    for (u64 i = threadIdx.x; i < D; i += blockDim.x)
        dot += s[base + i] * grad[base + i];
    sh[threadIdx.x] = dot;
    __syncthreads();
    for (u32 half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half)
            sh[threadIdx.x] += sh[threadIdx.x + half];
        __syncthreads();
    }
    dot = sh[0];

    for (u64 i = threadIdx.x; i < D; i += blockDim.x)
        dx[base + i] += s[base + i] * (grad[base + i] - dot);
}

void tensor_cuda_softmax_fwd(TensorImpl<f32> &out, const TensorImpl<f32> &inp) {
    u64 D = inp.shape[inp.ndim - 1];
    u64 n_rows = inp.numel() / D;
    softmax_fwd_kernel<<<(u32)n_rows, N_THREADS>>>(out.data(), inp.data(), D);
}

void tensor_cuda_softmax_bwd(TensorImpl<f32> &dx, const TensorImpl<f32> &s,
                              const TensorImpl<f32> &grad) {
    u64 D = s.shape[s.ndim - 1];
    u64 n_rows = s.numel() / D;
    softmax_bwd_kernel<<<(u32)n_rows, N_THREADS>>>(dx.data(), s.data(),
                                                    grad.data(), D);
}

// ── Fused layer norm kernels ──────────────────────────────────────────────────

__global__ void ln_fwd_kernel(f32 *out, f32 *xhat, f32 *inv_std_out,
                               const f32 *inp, const f32 *gamma,
                               const f32 *beta, f32 eps, u64 D) {
    u64 base = (u64)blockIdx.x * D;
    __shared__ f32 s_mu[N_THREADS], s_M2[N_THREADS];
    __shared__ u32 s_n[N_THREADS];

    f32 mu = 0.f, m2 = 0.f;
    u32 n = 0;
    for (u64 i = threadIdx.x; i < D; i += blockDim.x) {
        f32 x = inp[base + i];
        n++;
        f32 delta = x - mu;
        mu += delta / (f32)n;
        m2 += delta * (x - mu);
    }
    s_mu[threadIdx.x] = mu;
    s_M2[threadIdx.x] = m2;
    s_n[threadIdx.x] = n;
    __syncthreads();

    for (u32 half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            u32 na = s_n[threadIdx.x], nb = s_n[threadIdx.x + half];
            u32 nc = na + nb;
            if (nc > 0) {
                f32 d = s_mu[threadIdx.x + half] - s_mu[threadIdx.x];
                s_mu[threadIdx.x] += d * ((f32)nb / nc);
                s_M2[threadIdx.x] += s_M2[threadIdx.x + half] +
                                     d * d * ((f32)na * (f32)nb / (f32)nc);
            }
            s_n[threadIdx.x] = nc;
        }
        __syncthreads();
    }

    f32 row_mu  = s_mu[0];
    f32 inv_std = rsqrtf(s_M2[0] / (f32)D + eps);
    if (threadIdx.x == 0)
        inv_std_out[blockIdx.x] = inv_std;

    for (u64 i = threadIdx.x; i < D; i += blockDim.x) {
        f32 xh = (inp[base + i] - row_mu) * inv_std;
        xhat[base + i] = xh;
        out[base + i] = fmaf(gamma[i], xh, beta[i]);
    }
}

__global__ void ln_bwd_kernel(f32 *dx, const f32 *grad, const f32 *xhat,
                               const f32 *inv_std, const f32 *gamma, u64 D) {
    u64 base = (u64)blockIdx.x * D;
    f32 isd = inv_std[blockIdx.x];
    __shared__ f32 s_gn[N_THREADS], s_gnx[N_THREADS];

    f32 sum_gn = 0.f, sum_gnx = 0.f;
    for (u64 i = threadIdx.x; i < D; i += blockDim.x) {
        f32 gn = grad[base + i] * gamma[i];
        f32 xh = xhat[base + i];
        sum_gn  += gn;
        sum_gnx += gn * xh;
    }
    s_gn[threadIdx.x]  = sum_gn;
    s_gnx[threadIdx.x] = sum_gnx;
    __syncthreads();

    for (u32 half = blockDim.x / 2; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            s_gn[threadIdx.x]  += s_gn[threadIdx.x + half];
            s_gnx[threadIdx.x] += s_gnx[threadIdx.x + half];
        }
        __syncthreads();
    }

    f32 mean_gn  = s_gn[0]  / (f32)D;
    f32 mean_gnx = s_gnx[0] / (f32)D;

    for (u64 i = threadIdx.x; i < D; i += blockDim.x) {
        f32 gn = grad[base + i] * gamma[i];
        f32 xh = xhat[base + i];
        dx[base + i] += (gn - mean_gn - xh * mean_gnx) * isd;
    }
}

void tensor_cuda_ln_fwd(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                         TensorImpl<f32> &inv_std, const TensorImpl<f32> &inp,
                         const TensorImpl<f32> &gamma,
                         const TensorImpl<f32> &beta, f32 eps) {
    u64 D = inp.shape[inp.ndim - 1];
    u64 n_rows = inp.numel() / D;
    ln_fwd_kernel<<<(u32)n_rows, N_THREADS>>>(out.data(), xhat.data(),
                                               inv_std.data(), inp.data(),
                                               gamma.data(), beta.data(),
                                               eps, D);
}

void tensor_cuda_ln_bwd(TensorImpl<f32> &dx, const TensorImpl<f32> &grad,
                         const TensorImpl<f32> &xhat,
                         const TensorImpl<f32> &inv_std,
                         const TensorImpl<f32> &gamma) {
    u64 D = grad.shape[grad.ndim - 1];
    u64 n_rows = grad.numel() / D;
    ln_bwd_kernel<<<(u32)n_rows, N_THREADS>>>(dx.data(), grad.data(),
                                               xhat.data(), inv_std.data(),
                                               gamma.data(), D);
}

// ── Explicit instantiations
// ───────────────────────────────────────────────────

#define INST(T)                                                                \
    template void tensor_cuda_copy(TensorImpl<T> &, const TensorImpl<T> &);    \
    template void tensor_cuda_contiguous(TensorImpl<T> &);                      \
    template void tensor_cuda_fill(TensorImpl<T> &, T);                        \
    template void tensor_cuda_clear(TensorImpl<T> &);                          \
    template void tensor_cuda_arange(TensorImpl<T> &);                         \
    template void tensor_cuda_add(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  const TensorImpl<T> &);                      \
    template void tensor_cuda_sub(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  const TensorImpl<T> &);                      \
    template void tensor_cuda_mul(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  const TensorImpl<T> &);                      \
    template void tensor_cuda_div(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  const TensorImpl<T> &);                      \
    template void tensor_cuda_equal(TensorImpl<T> &, const TensorImpl<T> &,    \
                                    const TensorImpl<T> &);                    \
    template void tensor_cuda_add(TensorImpl<T> &, const TensorImpl<T> &, T);  \
    template void tensor_cuda_sub(TensorImpl<T> &, const TensorImpl<T> &, T);  \
    template void tensor_cuda_mul(TensorImpl<T> &, const TensorImpl<T> &, T);  \
    template void tensor_cuda_div(TensorImpl<T> &, const TensorImpl<T> &, T);  \
    template void tensor_cuda_sum_global(TensorImpl<T> &,                       \
                                         const TensorImpl<T> &);               \
    template void tensor_cuda_sum_dim(TensorImpl<T> &, const TensorImpl<T> &,  \
                                      u32);                                    \
    template void tensor_cuda_max(TensorImpl<T> &, const TensorImpl<T> &);     \
    template void tensor_cuda_max(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  u32);                                        \
    template void tensor_cuda_argmax(TensorImpl<u32> &, const TensorImpl<T> &, \
                                     u32);                                     \
    template void tensor_cuda_scatter_add(                                     \
        TensorImpl<T> &, const TensorImpl<T> &, const TensorImpl<u32> &, u32); \
    template void tensor_cuda_gather(TensorImpl<T> &, const TensorImpl<T> &,   \
                                     const TensorImpl<u32> &, u32);            \
    template void tensor_cuda_unfold2d(TensorImpl<T> &, const TensorImpl<T> &, \
                                       Unfold2dParams);                        \
    template void tensor_cuda_fold2d(TensorImpl<T> &, const TensorImpl<T> &,   \
                                     Unfold2dParams);

INST(f32)
INST(u32)
