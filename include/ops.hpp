#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

Var mat_mul(Var a, Var b, CudaMemArena *arena = nullptr);
Var reshape(Var a, const u32 *shape, u32 ndim, CudaMemArena *arena = nullptr);
Var transpose(Var a, u32 d0, u32 d1, CudaMemArena *arena = nullptr);
Var unsqueeze(Var a, u32 dim, CudaMemArena *arena = nullptr);
Var broadcast_to(Var a, const u32 *shape, u32 ndim, CudaMemArena *arena = nullptr);
// Differentiable gather. Backward accumulates via scatter_add.
Var gather(Var src, TensorU32 indices, u32 dim, CudaMemArena *arena = nullptr);
Var add(Var a, Var b, CudaMemArena *arena = nullptr);
// Elementwise multiply by a constant scalar.
Var mul(Var a, f32 scalar, CudaMemArena *arena = nullptr);
Var mul(Var a, Var b, CudaMemArena *arena = nullptr);
// Softmax over the given dimension (negative indices count from the end, default -1 = last).
Var softmax(Var a, i32 dim = -1, CudaMemArena *arena = nullptr);
Var relu(Var a, CudaMemArena *arena = nullptr);
Var gelu(Var a, CudaMemArena *arena = nullptr);

// input [N,C_in,H,W], weight [C_in*kH*kW, C_out] → output [N,C_out,L_h,L_w]
Var conv2d(Var input, Var weight, Unfold2dParams params,
           CudaMemArena *arena = nullptr);

// input [N,C,H,W] → output [N,C,L_h,L_w]
Var max_pool2d(Var input, Unfold2dParams params, CudaMemArena *arena = nullptr);

// input [...,D], gamma [D], beta [D] — normalizes over last dimension
Var layer_norm(Var input, Var gamma, Var beta, f32 eps = 1e-5f,
               CudaMemArena *arena = nullptr);

Var batch_norm(Var input, Var gamma, Var beta, Tensor<f32> running_mean,
               Tensor<f32> running_var, bool training = true,
               f32 momentum = 0.1f, f32 eps = 1e-5f,
               CudaMemArena *arena = nullptr);

// training=true: zeros p fraction of elements and scales remainder by 1/(1-p).
Var dropout(Var a, f32 p, bool training = true, CudaMemArena *arena = nullptr);

Var mse_loss(Var pred, Var target, CudaMemArena *arena = nullptr);

// Fused softmax + cross-entropy. logits/targets [N,C], output scalar.
Var cross_entropy_with_logits(Var logits, Var targets,
                              CudaMemArena *arena = nullptr);

#endif
