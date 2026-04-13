#ifndef TENSOR_HPP
#define TENSOR_HPP

#define MAX_NDIM 8
#include "base.hpp"

#define ROW_DIM(t) ((t)->ndim - 2)
#define COL_DIM(t) ((t)->ndim - 1)

struct Tensor {
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];
    f32 *data;
    u32 ndim;
    u64 size;
    b32 on_gpu;
    b32 owns_data;

    Tensor(u32 ndim, const u32 *shape, b32 on_gpu); // allocating
    Tensor(const Tensor *src); // view — shares data, owns_data=false
    ~Tensor();

    template <typename... Dims> f32 &operator()(Dims... dims) {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data[offset];
    }
    template <typename... Dims> const f32 &operator()(Dims... dims) const {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data[offset];
    }
};
Tensor *tensor_load(const char *filename, b32 on_gpu);
Tensor *tensor_to_gpu(const Tensor *t);
Tensor *tensor_to_cpu(const Tensor *t);
void tensor_fill(Tensor *tensor, f32 value);

b32 tensor_is_contiguous(const Tensor *t);
b32 tensor_shape_eq(const Tensor *a, const Tensor *b);

u32 broadcast_shape(const Tensor *a, const Tensor *b, u32 *expanded_shape);
void expanded_shape(const Tensor *t, const u32 *expanded_shape,
                    u32 expanded_ndim, u32 *t_expanded_shape);

void expanded_stride(const Tensor *t, const u32 *expanded_shape,
                     u32 expanded_ndim, u64 *t_expanded_stride);
b32 tensor_transpose(Tensor *tensor, u32 dim0, u32 dim1);
void tensor_clear(Tensor *tensor);
void tensor_mul(Tensor *out, const Tensor *tensor, f32 scalar);
Tensor *tensor_mul(const Tensor *tensor, f32 scalar);
b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_add(const Tensor *a, const Tensor *b);
Tensor *tensor_add(const Tensor *a, f32 scalar);
b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_sub(const Tensor *a, const Tensor *b);
b32 tensor_mul(Tensor *out, const Tensor *a,
               const Tensor *b); // Element wise
Tensor *tensor_mul(const Tensor *a, const Tensor *b);
b32 tensor_div(Tensor *out, const Tensor *a,
               const Tensor *b); // Element wise
Tensor *tensor_div(const Tensor *a, const Tensor *b);

b32 tensor_div(Tensor *out, const Tensor *a, f32 scalar); // Element wise
Tensor *tensor_div(const Tensor *a, f32 scalar);

b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                   b32 clear_out = true);
Tensor *tensor_mat_mul(const Tensor *a, const Tensor *b);

b32 tensor_sum(Tensor *out, const Tensor *tensor, b32 clear_out = true);
b32 tensor_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim = true,
               b32 clear_out = true);
Tensor *tensor_sum(const Tensor *tensor);
Tensor *tensor_sum(const Tensor *tensor, u32 dim, b32 keep_dim = true);

Tensor *tensor_view(const Tensor *src);
Tensor *tensor_create_like(const Tensor *src);
b32 tensor_reshape(Tensor *tensor, const u32 *shape, u32 ndim);

void tensor_print(const Tensor *tensor);

#endif
