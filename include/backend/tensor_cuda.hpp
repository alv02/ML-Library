#ifndef TENSOR_CUDA_HPP
#define TENSOR_CUDA_HPP

#include "../tensor.hpp"

#ifdef __CUDACC__
#define ML_DEVICE __device__
#else
#define ML_DEVICE
#endif

// GPU-side mirror of a Tensor's shape and stride metadata.
struct TensorMeta {
    u64 size;
    u32 ndim;
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];

    TensorMeta() = default;
    template <typename T>
    TensorMeta(const TensorImpl<T> &t) : size(t.numel()), ndim(t.ndim) {
        memcpy(shape, t.shape, ndim * sizeof(u32));
        memcpy(stride, t.stride, ndim * sizeof(u64));
    }
    template <typename T>
    TensorMeta(const TensorImpl<T> &t, const u32 *bcast_shape, u32 bcast_ndim)
        : size(t.numel()), ndim(bcast_ndim) {
        expanded_shape(t, bcast_ndim, shape);
        expanded_stride(t, bcast_ndim, stride);
    }

    ML_DEVICE u32 rows() const { return shape[ndim - 2]; }
    ML_DEVICE u32 cols() const { return shape[ndim - 1]; }
    ML_DEVICE u64 at(u64 row, u64 col) const {
        return row * stride[ndim - 2] + col * stride[ndim - 1];
    }
    ML_DEVICE u64 offset_from(u64 flat_idx, const TensorMeta &src) const {
        u64 remaining = flat_idx;
        u64 offset = 0;
        for (u32 i = 0; i < ndim; i++) {
            u64 idx_i = remaining / stride[i];
            remaining -= idx_i * stride[i];
            offset += idx_i * src.stride[i];
        }
        return offset;
    }
};

// ---- memory management ---------------------------------------------------

template <typename T>
void tensor_cuda_copy(TensorImpl<T> &dst, const TensorImpl<T> &src);
template <typename T>
void tensor_cuda_contiguous(TensorImpl<T> &t, CudaMemArena *arena = nullptr);

// ---- fill / clear / arange -----------------------------------------------

template <typename T> void tensor_cuda_fill(TensorImpl<T> &tensor, T value);
template <typename T> void tensor_cuda_clear(TensorImpl<T> &tensor);
template <typename T> void tensor_cuda_arange(TensorImpl<T> &out);

// ---- activations — f32 only ----------------------------------------------

void tensor_cuda_relu(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cuda_gelu(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cuda_gelu_backward(TensorImpl<f32> &out, const TensorImpl<f32> &grad,
                               const TensorImpl<f32> &input);
void tensor_cuda_exp(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cuda_log(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cuda_sqrt(TensorImpl<f32> &dst, const TensorImpl<f32> &src);

// ---- elementwise binary --------------------------------------------------

template <typename T>
void tensor_cuda_add(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b);
template <typename T>
void tensor_cuda_sub(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b);
template <typename T>
void tensor_cuda_mul(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b);
template <typename T>
void tensor_cuda_div(TensorImpl<T> &out, const TensorImpl<T> &a,
                     const TensorImpl<T> &b);
template <typename T>
void tensor_cuda_equal(TensorImpl<T> &out, const TensorImpl<T> &a,
                       const TensorImpl<T> &b);
void tensor_cuda_relu_backward(TensorImpl<f32> &out,
                               const TensorImpl<f32> &grad,
                               const TensorImpl<f32> &in);
void tensor_cuda_dropout_mask(TensorImpl<f32> &mask, f32 p);

// ---- scalar operations ---------------------------------------------------

template <typename T>
void tensor_cuda_add(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar);
template <typename T>
void tensor_cuda_sub(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar);
template <typename T>
void tensor_cuda_mul(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar);
template <typename T>
void tensor_cuda_div(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar);

// ---- matrix multiply — f32 only -----------------------------------------

void tensor_cuda_mat_mul(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                         const TensorImpl<f32> &b, b32 clear_out);
void tensor_cuda_mat_mul_cublas(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                const TensorImpl<f32> &b, b32 clear_out);
void tensor_cuda_mat_mul_batched(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                 const TensorImpl<f32> &b, b32 clear_out);

// ---- fused batch norm — f32 only ----------------------------------------

void tensor_cuda_bn_fwd_normalize(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                                  const TensorImpl<f32> &inp,
                                  const TensorImpl<f32> &mean,
                                  const TensorImpl<f32> &m2,
                                  const TensorImpl<f32> &gamma,
                                  const TensorImpl<f32> &beta, f32 count,
                                  f32 eps);
void tensor_cuda_bn_bwd(TensorImpl<f32> &dx, TensorImpl<f32> &d_gamma,
                        TensorImpl<f32> &d_beta, const TensorImpl<f32> &grad,
                        const TensorImpl<f32> &xhat,
                        const TensorImpl<f32> &gamma,
                        const TensorImpl<f32> &var, f32 m, f32 eps);

// ---- reduction -----------------------------------------------------------

template <typename T>
void tensor_cuda_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor);
template <typename T>
void tensor_cuda_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim);
void tensor_cuda_welford_mean_var(TensorImpl<f32> &mean, TensorImpl<f32> &m2,
                                  const TensorImpl<f32> &src, u32 dim);
template <typename T>
void tensor_cuda_max(TensorImpl<T> &out, const TensorImpl<T> &tensor);
template <typename T>
void tensor_cuda_max(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim);
template <typename T>
void tensor_cuda_argmax(TensorImpl<u32> &out, const TensorImpl<T> &tensor,
                        u32 dim);

// ---- scattering ----------------------------------------------------------

template <typename T>
void tensor_cuda_scatter_add(TensorImpl<T> &out, const TensorImpl<T> &src,
                             const TensorImpl<u32> &indices, u32 dim);

// ---- initializing — f32 only --------------------------------------------

void tensor_cuda_he_init(TensorImpl<f32> &tensor);

// ---- indexing ------------------------------------------------------------

template <typename T>
void tensor_cuda_gather(TensorImpl<T> &dst, const TensorImpl<T> &src,
                        const TensorImpl<u32> &indices, u32 dim);

// ---- spatial -------------------------------------------------------------

template <typename T>
void tensor_cuda_unfold2d(TensorImpl<T> &dst, const TensorImpl<T> &src,
                          Unfold2dParams params);
template <typename T>
void tensor_cuda_fold2d(TensorImpl<T> &dst, const TensorImpl<T> &col,
                        Unfold2dParams params);

// ---- comparison — f32 only ----------------------------------------------

b32 tensor_cuda_equals(const TensorImpl<f32> &a, const TensorImpl<f32> &b,
                       f32 tol);

#endif // TENSOR_CUDA_HPP
