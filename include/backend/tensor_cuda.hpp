#ifndef TENSOR_CUDA_HPP
#define TENSOR_CUDA_HPP

#include "../tensor.hpp"

#ifdef __CUDACC__
#define ML_DEVICE __device__
#else
#define ML_DEVICE
#endif

struct TensorMeta {
    u64 size;
    u32 ndim;
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];

    // Simple copy
    TensorMeta(const Tensor *t);
    // Broadcast-expanded — zeros out strides on broadcast dims
    TensorMeta(const Tensor *t, const u32 *bcast_shape, u32 bcast_ndim);

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

// ---- memory management (alloc / free / transfers) ------------------------

void tensor_cuda_alloc(Tensor *tensor);
void tensor_cuda_free(Tensor *tensor);
Tensor *tensor_cuda_to_gpu(const Tensor *t_cpu);
Tensor *tensor_cuda_to_cpu(const Tensor *t_gpu);
Tensor *tensor_cuda_copy(const Tensor *t_gpu);

// ---- copy (into existing tensor) ----------------------------------------

void tensor_cuda_copy(Tensor *dst, const Tensor *src);

// ---- fill / clear --------------------------------------------------------

void tensor_cuda_fill(Tensor *tensor, f32 value);
void tensor_cuda_clear(Tensor *tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cuda_relu(Tensor *dst, const Tensor *src);
void tensor_cuda_exp(Tensor *dst, const Tensor *src);
void tensor_cuda_log(Tensor *dst, const Tensor *src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cuda_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_div(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_equal(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_relu_backward(Tensor *out, const Tensor *grad,
                               const Tensor *in);

// ---- scalar operations ---------------------------------------------------

void tensor_cuda_add(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_sub(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_div(Tensor *out, const Tensor *tensor, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cuda_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                         b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

void tensor_cuda_sum(Tensor *out, const Tensor *tensor);
void tensor_cuda_sum(Tensor *out, const Tensor *tensor, u32 dim);
void tensor_cuda_max(Tensor *out, const Tensor *tensor);
void tensor_cuda_max(Tensor *out, const Tensor *tensor, u32 dim);
void tensor_cuda_argmax(Tensor *out, const Tensor *tensor, u32 dim);

// ---- initializing --------------------------------------------------------

void tensor_cuda_he_init(Tensor *tensor);

// ---- indexing ------------------------------------------------------------

void tensor_cuda_index_select(Tensor *dst, const Tensor *src,
                              const u32 *indices, u32 n_indices, u32 dim);

#endif // TENSOR_CUDA_HPP
