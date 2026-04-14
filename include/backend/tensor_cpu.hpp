#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include "../tensor.hpp"

// ---- copy ----------------------------------------------------------------

void tensor_cpu_copy(Tensor *dst, const Tensor *src);

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(Tensor *tensor, f32 value);
void tensor_cpu_clear(Tensor *tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cpu_relu(Tensor *dst, const Tensor *src);
void tensor_cpu_exp(Tensor *dst, const Tensor *src);
void tensor_cpu_log(Tensor *dst, const Tensor *src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cpu_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_div(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_relu_backward(Tensor *out, const Tensor *grad,
                              const Tensor *in);

// ---- scalar operations ---------------------------------------------------

void tensor_cpu_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cpu_div(Tensor *out, const Tensor *a, f32 scalar);
void tensor_cpu_add(Tensor *out, const Tensor *a, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cpu_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                        b32 clear_out);

// ---- reduction (sum, max) ------------------------------------------------

void tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out);
void tensor_cpu_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 clear_out);
void tensor_cpu_max(Tensor *out, const Tensor *tensor);
void tensor_cpu_max(Tensor *out, const Tensor *tensor, u32 dim);

// ---- intializing ---------------------------------------------------------
void tensor_cpu_he_init(Tensor *tensor);

#endif // TENSOR_CPU_HPP
