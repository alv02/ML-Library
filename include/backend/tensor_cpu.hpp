#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include "../tensor.hpp"

void tensor_cpu_copy(Tensor *dst, const Tensor *src);
void tensor_cpu_contigous(Tensor *t);

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(Tensor *tensor, f32 value);
void tensor_cpu_clear(Tensor *tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cpu_relu(Tensor *dst, const Tensor *src);
void tensor_cpu_exp(Tensor *dst, const Tensor *src);
void tensor_cpu_log(Tensor *dst, const Tensor *src);
void tensor_cpu_sqrt(Tensor *dst, const Tensor *src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cpu_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_div(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_equal(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cpu_relu_backward(Tensor *out, const Tensor *grad,
                              const Tensor *in);

// ---- scalar operations ---------------------------------------------------

void tensor_cpu_add(Tensor *out, const Tensor *a, f32 scalar);
void tensor_cpu_sub(Tensor *out, const Tensor *a, f32 scalar);
void tensor_cpu_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cpu_div(Tensor *out, const Tensor *a, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cpu_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                        b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

// Reduces the entire tensor to a scalar using Kahan compensated summation.
void tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out);
// Reduces along dim using the stride=0 trick: out_strides[dim]=0 maps all
// positions along that axis to the same output slot so they accumulate there.
void tensor_cpu_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 clear_out);
// Reduces the entire tensor to a scalar (maximum value).
void tensor_cpu_max(Tensor *out, const Tensor *tensor);
// Reduces along dim using the same stride=0 trick as tensor_cpu_sum.
void tensor_cpu_max(Tensor *out, const Tensor *tensor, u32 dim);
// Returns the index (as f32) of the max value along dim.
void tensor_cpu_argmax(Tensor *out, const Tensor *tensor, u32 dim);

// ---- welford mean+var ----------------------------------------------------

void tensor_cpu_welford_mean_var(Tensor *mean, Tensor *var, const Tensor *src,
                                  u32 dim);

// ---- scattering ----------------------------------------------------------
void tensor_cpu_scatter_add(Tensor *out, const Tensor *src,
                            const Tensor *indices, u32 dim);

// ---- initializing --------------------------------------------------------

void tensor_cpu_he_init(Tensor *tensor);

// ---- indexing ------------------------------------------------------------

void tensor_cpu_index_select(Tensor *dst, const Tensor *src, const u32 *indices,
                             u32 n_indices, u32 dim);

// ---- spatial / patch operations ------------------------------------------

void tensor_cpu_unfold2d(Tensor *dst, const Tensor *src, Unfold2dParams params);
void tensor_cpu_fold2d(Tensor *dst, const Tensor *col, Unfold2dParams params);

// ---- comparison ----------------------------------------------------------

b32 tensor_cpu_equals(const Tensor *a, const Tensor *b, f32 tol);

#endif // TENSOR_CPU_HPP
