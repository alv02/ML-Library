#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include "../tensor.hpp"

void tensor_cpu_copy(TensorImpl &dst, const TensorImpl &src);
void tensor_cpu_contigous(TensorImpl &t);

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(TensorImpl &tensor, f32 value);
void tensor_cpu_clear(TensorImpl &tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cpu_relu(TensorImpl &dst, const TensorImpl &src);
void tensor_cpu_exp(TensorImpl &dst, const TensorImpl &src);
void tensor_cpu_log(TensorImpl &dst, const TensorImpl &src);
void tensor_cpu_sqrt(TensorImpl &dst, const TensorImpl &src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cpu_add(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cpu_sub(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cpu_mul(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cpu_div(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cpu_equal(TensorImpl &out, const TensorImpl &a,
                      const TensorImpl &b);
void tensor_cpu_relu_backward(TensorImpl &out, const TensorImpl &grad,
                              const TensorImpl &in);

// ---- scalar operations ---------------------------------------------------

void tensor_cpu_add(TensorImpl &out, const TensorImpl &a, f32 scalar);
void tensor_cpu_sub(TensorImpl &out, const TensorImpl &a, f32 scalar);
void tensor_cpu_mul(TensorImpl &out, const TensorImpl &tensor, f32 scalar);
void tensor_cpu_div(TensorImpl &out, const TensorImpl &a, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cpu_mat_mul(TensorImpl &out, const TensorImpl &a,
                        const TensorImpl &b, b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

// Reduces the entire tensor to a scalar using Kahan compensated summation.
void tensor_cpu_sum(TensorImpl &out, const TensorImpl &tensor, b32 clear_out);
// Reduces along dim using the stride=0 trick: out_strides[dim]=0 maps all
// positions along that axis to the same output slot so they accumulate there.
void tensor_cpu_sum(TensorImpl &out, const TensorImpl &tensor, u32 dim,
                    b32 clear_out);
// Reduces the entire tensor to a scalar (maximum value).
void tensor_cpu_max(TensorImpl &out, const TensorImpl &tensor);
// Reduces along dim using the same stride=0 trick as tensor_cpu_sum.
void tensor_cpu_max(TensorImpl &out, const TensorImpl &tensor, u32 dim);
// Returns the index (as f32) of the max value along dim.
void tensor_cpu_argmax(TensorImpl &out, const TensorImpl &tensor, u32 dim);

// ---- welford mean+var ----------------------------------------------------

void tensor_cpu_welford_mean_var(TensorImpl &mean, TensorImpl &var,
                                 const TensorImpl &src, u32 dim);

// ---- scattering ----------------------------------------------------------
void tensor_cpu_scatter_add(TensorImpl &out, const TensorImpl &src,
                            const TensorImpl &indices, u32 dim);

// ---- initializing --------------------------------------------------------

void tensor_cpu_he_init(TensorImpl &tensor);

// ---- indexing ------------------------------------------------------------

void tensor_cpu_index_select(TensorImpl &dst, const TensorImpl &src,
                             const u32 *indices, u32 n_indices, u32 dim);

// ---- spatial / patch operations ------------------------------------------

void tensor_cpu_unfold2d(TensorImpl &dst, const TensorImpl &src,
                         Unfold2dParams params);
void tensor_cpu_fold2d(TensorImpl &dst, const TensorImpl &col,
                       Unfold2dParams params);

// ---- comparison ----------------------------------------------------------

b32 tensor_cpu_equals(const TensorImpl &a, const TensorImpl &b, f32 tol);

#endif // TENSOR_CPU_HPP
