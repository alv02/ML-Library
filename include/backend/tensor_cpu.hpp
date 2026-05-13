#ifndef TENSOR_CPU_HPP
#define TENSOR_CPU_HPP

#include "../tensor.hpp"

template <typename T>
void tensor_cpu_copy(TensorImpl<T> &dst, const TensorImpl<T> &src);
template <typename T> void tensor_cpu_contigous(TensorImpl<T> &t);

// ---- fill / clear / arange -----------------------------------------------

template <typename T> void tensor_cpu_fill(TensorImpl<T> &tensor, T value);
template <typename T> void tensor_cpu_clear(TensorImpl<T> &tensor);
template <typename T> void tensor_cpu_arange(TensorImpl<T> &out);

// ---- activations (relu, exp) — f32 only ---------------------------------

void tensor_cpu_relu(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cpu_gelu(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cpu_gelu_backward(TensorImpl<f32> &out, const TensorImpl<f32> &grad,
                              const TensorImpl<f32> &input);
void tensor_cpu_exp(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cpu_log(TensorImpl<f32> &dst, const TensorImpl<f32> &src);
void tensor_cpu_sqrt(TensorImpl<f32> &dst, const TensorImpl<f32> &src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

template <typename T>
void tensor_cpu_add(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b);
template <typename T>
void tensor_cpu_sub(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b);
template <typename T>
void tensor_cpu_mul(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b);
template <typename T>
void tensor_cpu_div(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b);
template <typename T>
void tensor_cpu_equal(TensorImpl<T> &out, const TensorImpl<T> &a,
                      const TensorImpl<T> &b);
void tensor_cpu_relu_backward(TensorImpl<f32> &out, const TensorImpl<f32> &grad,
                              const TensorImpl<f32> &in);
void tensor_cpu_dropout_mask(TensorImpl<f32> &mask, f32 p);

// ---- scalar operations ---------------------------------------------------

template <typename T>
void tensor_cpu_add(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar);
template <typename T>
void tensor_cpu_sub(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar);
template <typename T>
void tensor_cpu_mul(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar);
template <typename T>
void tensor_cpu_div(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar);

// ---- matrix multiply — f32 only -----------------------------------------

void tensor_cpu_mat_mul(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                        const TensorImpl<f32> &b, b32 clear_out);
void tensor_cpu_mat_mul_batched(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                const TensorImpl<f32> &b, b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

template <typename T>
void tensor_cpu_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                    b32 clear_out);
template <typename T>
void tensor_cpu_sum(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim,
                    b32 clear_out);
template <typename T>
void tensor_cpu_max(TensorImpl<T> &out, const TensorImpl<T> &tensor);
template <typename T>
void tensor_cpu_max(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim);
template <typename T>
void tensor_cpu_argmax(TensorImpl<u32> &out, const TensorImpl<T> &tensor,
                       u32 dim);

// ---- welford mean+m2 — f32 only ----------------------------------------

void tensor_cpu_welford_mean_var(TensorImpl<f32> &mean, TensorImpl<f32> &m2,
                                 const TensorImpl<f32> &src, u32 dim);

// ---- fused batch norm — f32 only ----------------------------------------

void tensor_cpu_bn_fwd_normalize(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                                 const TensorImpl<f32> &inp,
                                 const TensorImpl<f32> &mean,
                                 const TensorImpl<f32> &m2,
                                 const TensorImpl<f32> &gamma,
                                 const TensorImpl<f32> &beta, f32 count,
                                 f32 eps);

void tensor_cpu_bn_bwd(TensorImpl<f32> &dx, TensorImpl<f32> &d_gamma,
                       TensorImpl<f32> &d_beta, const TensorImpl<f32> &grad,
                       const TensorImpl<f32> &xhat,
                       const TensorImpl<f32> &gamma, const TensorImpl<f32> &var,
                       f32 m, f32 eps);

// ---- scattering ----------------------------------------------------------

template <typename T>
void tensor_cpu_scatter_add(TensorImpl<T> &out, const TensorImpl<T> &src,
                            const TensorImpl<u32> &indices, u32 dim);

// ---- initializing — f32 only --------------------------------------------

void tensor_cpu_he_init(TensorImpl<f32> &tensor);

// ---- indexing ------------------------------------------------------------

template <typename T>
void tensor_cpu_gather(TensorImpl<T> &dst, const TensorImpl<T> &src,
                       const TensorImpl<u32> &indices, u32 dim);

// ---- spatial / patch operations ------------------------------------------

template <typename T>
void tensor_cpu_unfold2d(TensorImpl<T> &dst, const TensorImpl<T> &src,
                         Unfold2dParams params);
template <typename T>
void tensor_cpu_fold2d(TensorImpl<T> &dst, const TensorImpl<T> &col,
                       Unfold2dParams params);

// ---- comparison — f32 only ----------------------------------------------

b32 tensor_cpu_equals(const TensorImpl<f32> &a, const TensorImpl<f32> &b,
                      f32 tol);

#endif // TENSOR_CPU_HPP
