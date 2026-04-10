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
};

Tensor *tensor_to_gpu(const Tensor *t_cpu);
Tensor *tensor_to_cpu(const Tensor *t_gpu);
void tensor_cuda_alloc(Tensor *tensor);
void tensor_cuda_free(Tensor *tensor);

void tensor_cuda_fill(Tensor *tensor, f32 value);
void tensor_cuda_clear(Tensor *tensor);

void tensor_cuda_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_div(Tensor *out, const Tensor *a, const Tensor *b);

void tensor_cuda_add(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_sub(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_div(Tensor *out, const Tensor *tensor, f32 scalar);

void tensor_cuda_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                         b32 clear_out);

void tensor_cuda_sum(Tensor *out, const Tensor *tensor, b32 clear_out);
void tensor_cuda_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
                     b32 clear_out);

#endif // TENSOR_CUDA_HPP
