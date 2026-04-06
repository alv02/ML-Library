#ifndef TENSOR_H
#define TENSOR_H

#include "../util/arena.h"
#include "../util/base.h"

#define ROW_DIM(t) ((t)->ndim - 2)
#define COL_DIM(t) ((t)->ndim - 1)
#define ROW_DIM_T(t) ((t)->ndim - 1)
#define COL_DIM_T(t) ((t)->ndim - 2)

struct Tensor {
    u32 *shape;
    u64 *stride;
    f32 *data;
    u32 ndim; // you'll need this constantly
    u64 size; // total elements — saves recomputing
    b32 on_gpu;

    template <typename... Dims> f32 &operator[](Dims... dims) {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data[offset];
    }
};

Tensor *tensor_create(mem_arena *arena, u32 ndim, u32 *shape, b32 on_gpu);
Tensor *tensor_load(mem_arena *arena, const char *filename, b32 on_gpu);

b32 tensor_shape_eq(const Tensor *a, const Tensor *b);

b32 tensor_transpose(Tensor *tensor, u32 dim0, u32 dim1);
void tensor_clear(Tensor *tensor);
void tensor_scale(Tensor *tensor, f32 scale);
b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
b32 tensor_mul(Tensor *out, const Tensor *a, const Tensor *b); // Element wise
b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b); // Element wise

b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b);

f32 tensor_sum(Tensor *tensor);
b32 tensor_sum(Tensor *out, const Tensor *a, u32 dim);

Tensor *tensor_view(mem_arena *arena, Tensor *src);
Tensor *tensor_create_like(mem_arena *arena, const Tensor *src);

void tensor_print(const Tensor *tensor);

#endif
