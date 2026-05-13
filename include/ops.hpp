#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

// Works for 2D and ND: leading dims are treated as batch, last two are
// [M,K]×[K,N]→[M,N].
Var mat_mul(Var a, Var b, CudaMemArena *arena = nullptr);
// Zero-copy shape reinterpretation (copies only if input is non-contiguous).
// For f32: differentiable (backward reshapes grad back). For other T: metadata
// only.
template <typename T = f32>
Var_<T> reshape(Var_<T> a, const u32 *shape, u32 ndim,
                CudaMemArena *arena = nullptr);
// Zero-copy dim swap. For f32: backward swaps the same dims. For other T:
// metadata only.
template <typename T = f32>
Var_<T> transpose(Var_<T> a, u32 d0, u32 d1, CudaMemArena *arena = nullptr);
// Insert a size-1 dim at position `dim`. For f32: backward squeezes it out.
template <typename T = f32>
Var_<T> unsqueeze(Var_<T> a, u32 dim, CudaMemArena *arena = nullptr);
// Expand to target shape via stride-0 views (numpy left-alignment). For f32:
// backward sums over all broadcast dims. For other T: metadata only.
template <typename T = f32>
Var_<T> broadcast_to(Var_<T> a, const u32 *shape, u32 ndim,
                     CudaMemArena *arena = nullptr);
// Differentiable gather. Backward accumulates via scatter_add.
Var gather(Var src, TensorU32 indices, u32 dim, CudaMemArena *arena = nullptr);
Var add(Var a, Var b, CudaMemArena *arena = nullptr);
// Elementwise multiply by a constant scalar — gradient is grad*scalar.
Var mul(Var a, f32 scalar, CudaMemArena *arena = nullptr);
// Elementwise multiply.
Var mul(Var a, Var b, CudaMemArena *arena = nullptr);
// Softmax over the given dimension (negative indices count from the end,
// default -1 = last).
Var softmax(Var a, i32 dim = -1, CudaMemArena *arena = nullptr);
Var relu(Var a, CudaMemArena *arena = nullptr);
Var gelu(Var a, CudaMemArena *arena = nullptr);

// input  [N, C_in, H, W]
// weight [C_in*kH*kW, C_out]  (caller passes transpose(W) where layer W is
// [C_out, C_in*kH*kW]) output [N, C_out, L_h, L_w]
Var conv2d(Var input, Var weight, Unfold2dParams params,
           CudaMemArena *arena = nullptr);

// input [N, C, H, W] → output [N, C, L_h, L_w]
Var max_pool2d(Var input, Unfold2dParams params, CudaMemArena *arena = nullptr);

// input [..., D], gamma [D], beta [D] — normalizes over last dimension
Var layer_norm(Var input, Var gamma, Var beta, f32 eps = 1e-5f,
               CudaMemArena *arena = nullptr);

Var batch_norm(Var input, Var gamma, Var beta, Tensor<f32> running_mean,
               Tensor<f32> running_var, bool training = true,
               f32 momentum = 0.1f, f32 eps = 1e-5f,
               CudaMemArena *arena = nullptr);

// training=true: zeros p fraction of elements and scales remainder by 1/(1-p).
// training=false or p==0: identity (no-op, returns input directly).
Var dropout(Var a, f32 p, bool training = true, CudaMemArena *arena = nullptr);

Var mse_loss(Var pred, Var target, CudaMemArena *arena = nullptr);

// Fused softmax + cross-entropy. logits/targets [N, C], output scalar.
Var cross_entropy_with_logits(Var logits, Var targets,
                              CudaMemArena *arena = nullptr);

#endif
