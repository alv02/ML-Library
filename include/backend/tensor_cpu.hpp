#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include "../tensor.hpp"

void tensor_cpu_fill(Tensor *tensor, f32 value);
void tensor_cpu_clear(Tensor *tensor);

void tensor_cpu_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_div(Tensor *out, const Tensor *a, const Tensor *b);

void tensor_cpu_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cpu_div(Tensor *out, const Tensor *a, f32 scalar);
void tensor_cpu_add(Tensor *out, const Tensor *a, f32 scalar);

void tensor_cpu_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                        b32 clear_out);

void tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out);
void tensor_cpu_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
                    b32 clear_out);

#endif // TENSOR_CPU_HPP
