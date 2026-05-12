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
        f32 t    = tanhf(c * (x + k * x * x * x));
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

__global__ void welford_mean_var(f32 *mean, f32 *m2, const f32 *src,
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
        m2[c] = partial_M2[0];
    }
}

template <typename T>
__global__ void tensor_sum_kernel(TensorMeta out_meta, TensorMeta tensor_meta,
                                  T *out, T *tensor, u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= out_meta.size)
        return;
    u64 base_offset = out_meta.offset_from(workIdx, tensor_meta);
    T sum = T(0);
    for (u32 i = 0; i < tensor_meta.shape[dim]; i++)
        sum += tensor[base_offset + i * tensor_meta.stride[dim]];
    out[workIdx] = sum;
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
tensor_scatter_add_kernel(TensorMeta out_meta, TensorMeta out_base_meta,
                          TensorMeta indices_meta, TensorMeta src_meta,
                          TensorMeta src_contig, T *out, u32 *indices, T *src,
                          u32 dim) {
    u64 workIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIdx >= src_contig.size)
        return;

    u64 src_offset = src_contig.offset_from(workIdx, src_meta);
    u64 out_base_offset = src_contig.offset_from(workIdx, out_base_meta);
    u64 indices_offset = src_contig.offset_from(workIdx, indices_meta);
    out_base_offset += indices[indices_offset] * out_meta.stride[dim];

    atomicAdd(&out[out_base_offset], src[src_offset]);
}

template <typename T>
__global__ void tensor_index_select_kernel(TensorMeta dst_meta,
                                           TensorMeta src_meta, T *dst,
                                           const T *src, const u32 *indices,
                                           u32 n_indices, u32 dim) {
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

template <typename T>
__global__ void tensor_index_select_tensor_kernel(TensorMeta dst_meta,
                                                  TensorMeta src_meta, T *dst,
                                                  const T *src,
                                                  const u32 *indices, u32 dim) {
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
    TensorMeta out_contig = make_contig_meta(dst);
    TensorMeta out_actual(dst);
    TensorMeta src_meta(src);
    copy_kernel<<<blocks, threads>>>(out_contig, out_actual, src_meta,
                                     dst.data(), src.data());
}

template <typename T>
void tensor_cuda_contiguous(TensorImpl<T> &t, CudaMemArena *arena) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(t.numel(), (u64)threads);

    TensorMeta src_meta(t);
    TensorMeta src_contig = make_contig_meta(t);

    Tensor<T> temp_t = Tensor<T>::make(t.ndim, t.shape, true, arena);
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
    int M = (int)a.shape[nd - 2], K = (int)a.shape[nd - 1], N = (int)b.shape[nd - 1];
    int batch = (int)(a.numel() / ((u64)M * K));
    float alpha = 1.0f, beta = clear_out ? 0.0f : 1.0f;

    // Same transpose detection as the 2D path, but using the last two strides.
    cublasOperation_t op_b = (b.stride[nd - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_b = (int)((b.stride[nd - 1] == 1) ? b.stride[nd - 2] : b.stride[nd - 1]);
    cublasOperation_t op_a = (a.stride[nd - 1] == 1) ? CUBLAS_OP_N : CUBLAS_OP_T;
    int ld_a = (int)((a.stride[nd - 1] == 1) ? a.stride[nd - 2] : a.stride[nd - 1]);
    int ld_c = (int)out.stride[nd - 2];

    // Stride between successive batch slices (in elements).
    long long sb = (batch > 1) ? (long long)b.stride[nd - 3] : (long long)K * N;
    long long sa = (batch > 1) ? (long long)a.stride[nd - 3] : (long long)M * K;
    long long sc = (batch > 1) ? (long long)out.stride[nd - 3] : (long long)M * N;

    // cuBLAS is column-major, so pass (B, A) to compute C = A @ B in row-major.
    cublasSgemmStridedBatched(cublas_handle(), op_b, op_a, N, M, K,
                              &alpha, b.data(), ld_b, sb,
                              a.data(), ld_a, sa,
                              &beta, out.data(), ld_c, sc, batch);
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

__global__ void bn_bwd_fused(f32 *dx, f32 *d_gamma, f32 *d_beta,
                             const f32 *grad, const f32 *xhat, const f32 *gamma,
                             const f32 *var, f32 m_inv, f32 eps, u64 stride_c,
                             TensorMeta meta_contig, TensorMeta meta_src,
                             u64 other_dims_size) {
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
                                  const TensorImpl<f32> &beta, f32 count,
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
    bn_bwd_fused<<<C, N_THREADS>>>(dx.data(), d_gamma.data(), d_beta.data(),
                                   grad.data(), xhat.data(), gamma.data(),
                                   var.data(), 1.0f / m, eps, stride_c,
                                   meta_contig, meta_src, other_dims_size);
}

// ---- reduction -----------------------------------------------------------

template <typename T>
void tensor_cuda_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor) {
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

template <typename T>
void tensor_cuda_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim) {
    if (out.numel() == 1) {
        tensor_cuda_sum(out, tensor);
        return;
    }
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(out.numel(), u64(threads));
    TensorMeta out_meta(out);
    TensorMeta tensor_meta(tensor);
    tensor_sum_kernel<<<blocks, threads>>>(out_meta, tensor_meta, out.data(),
                                           tensor.data(), dim);
}

void tensor_cuda_welford_mean_var(TensorImpl<f32> &mean, TensorImpl<f32> &m2,
                                  const TensorImpl<f32> &src, u32 dim) {
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
        mean.data(), m2.data(), src.data(), src.stride[dim], meta_contig,
        meta_src, other_dims_size);
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
    TensorMeta out_base_meta(out);
    out_base_meta.stride[dim] = 0;
    TensorMeta indices_meta(indices);
    TensorMeta src_meta(src);
    TensorMeta src_contig = make_contig_meta(src);

    tensor_scatter_add_kernel<<<blocks, threads>>>(
        out_meta, out_base_meta, indices_meta, src_meta, src_contig, out.data(),
        indices.data(), src.data(), dim);
}

// ---- initializing — f32 only --------------------------------------------

void tensor_cuda_he_init(TensorImpl<f32> &tensor) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    static u64 seed_counter = 0;
    curandSetPseudoRandomGeneratorSeed(gen, (u64)time(nullptr) ^ (seed_counter++ * 6364136223846793005ULL));
    u32 in_features = tensor.shape[COL_DIM(tensor)];
    float stddev = sqrtf(2.0f / in_features);
    curandGenerateNormal(gen, tensor.data(), tensor.numel(), 0.0f, stddev);
    curandDestroyGenerator(gen);
}

__global__ void dropout_mask_kernel(f32 *mask, u64 n, f32 p, f32 scale) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mask[i] = (mask[i] >= p) ? scale : 0.0f;
}

void tensor_cuda_dropout_mask(TensorImpl<f32> &mask, f32 p) {
    static std::atomic<u64> seed_counter{0};
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed_counter.fetch_add(1) ^ 0xdeadbeefULL);
    curandGenerateUniform(gen, mask.data(), mask.numel());
    curandDestroyGenerator(gen);

    f32 scale = 1.0f / (1.0f - p);
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(mask.numel(), (u64)threads);
    dropout_mask_kernel<<<blocks, threads>>>(mask.data(), mask.numel(), p, scale);
}

// ---- indexing ------------------------------------------------------------

template <typename T>
void tensor_cuda_index_select(TensorImpl<T> &dst, const TensorImpl<T> &src,
                              const u32 *indices, u32 n_indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), u64(threads));
    TensorMeta dst_meta(dst);
    TensorMeta src_meta(src);
    u32 *indices_gpu;
    cudaMallocAsync(&indices_gpu, n_indices * sizeof(u32), 0);
    cudaMemcpyAsync(indices_gpu, indices, n_indices * sizeof(u32),
                    cudaMemcpyHostToDevice, 0);
    tensor_index_select_kernel<<<blocks, threads>>>(
        dst_meta, src_meta, dst.data(), src.data(), indices_gpu, n_indices,
        dim);
    cudaFreeAsync(indices_gpu, 0);
}

template <typename T>
void tensor_cuda_index_select(TensorImpl<T> &dst, const TensorImpl<T> &src,
                              const TensorImpl<u32> &indices, u32 dim) {
    u32 threads = N_THREADS;
    u32 blocks = cuda::ceil_div(dst.numel(), u64(threads));
    TensorMeta dst_meta(dst);
    TensorMeta src_meta(src);
    tensor_index_select_tensor_kernel<<<blocks, threads>>>(
        dst_meta, src_meta, dst.data(), src.data(), indices.data(), dim);
}

// ---- spatial / patch operations ------------------------------------------

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

// ---- comparison — f32 only ----------------------------------------------

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

// ── Explicit instantiations
// ───────────────────────────────────────────────────

#define INST(T)                                                                \
    template void tensor_cuda_copy(TensorImpl<T> &, const TensorImpl<T> &);    \
    template void tensor_cuda_contiguous(TensorImpl<T> &, CudaMemArena *);     \
    template void tensor_cuda_fill(TensorImpl<T> &, T);                        \
    template void tensor_cuda_clear(TensorImpl<T> &);                          \
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
    template void tensor_cuda_sum(TensorImpl<T> &, const TensorImpl<T> &);     \
    template void tensor_cuda_sum(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  u32);                                        \
    template void tensor_cuda_max(TensorImpl<T> &, const TensorImpl<T> &);     \
    template void tensor_cuda_max(TensorImpl<T> &, const TensorImpl<T> &,      \
                                  u32);                                        \
    template void tensor_cuda_argmax(TensorImpl<u32> &, const TensorImpl<T> &, \
                                     u32);                                     \
    template void tensor_cuda_scatter_add(                                     \
        TensorImpl<T> &, const TensorImpl<T> &, const TensorImpl<u32> &, u32); \
    template void tensor_cuda_index_select(                                    \
        TensorImpl<T> &, const TensorImpl<T> &, const u32 *, u32, u32);        \
    template void tensor_cuda_index_select(                                    \
        TensorImpl<T> &, const TensorImpl<T> &, const TensorImpl<u32> &, u32); \
    template void tensor_cuda_unfold2d(TensorImpl<T> &, const TensorImpl<T> &, \
                                       Unfold2dParams);                        \
    template void tensor_cuda_fold2d(TensorImpl<T> &, const TensorImpl<T> &,   \
                                     Unfold2dParams);

INST(f32)
INST(u32)
