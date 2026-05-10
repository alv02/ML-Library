#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

Var mat_mul(Var a, Var b, CudaMemArena *arena = nullptr);
Var add(Var a, Var b, CudaMemArena *arena = nullptr);
Var relu(Var a, CudaMemArena *arena = nullptr);

// input  [N, C_in, H, W]
// weight [C_in*kH*kW, C_out]  (pre-transposed, Wt convention)
// output [N, C_out, L_h, L_w]
Var conv2d(Var input, Var weight, Unfold2dParams params,
           CudaMemArena *arena = nullptr);

// input [N, C, H, W] → output [N, C, L_h, L_w]
Var max_pool2d(Var input, Unfold2dParams params, CudaMemArena *arena = nullptr);

// [N, ...] → [N, C*H*W]
Var flatten(Var input, CudaMemArena *arena = nullptr);

// input [N, C, ...], gamma/beta [C], running_mean/var shape [1,C,1,...,1] matching input ndim
// Normalizes per channel (dim 1) over all other dims — works for dense (N,C) and conv (N,C,H,W) etc.
// training=true  → normalize with biased batch variance (divide by N);
//                  updates running_var with unbiased variance (divide by N-1)
//                  via EMA: running = (1-momentum)*running + momentum*batch
// training=false → normalize with running stats, no grad_fn
// momentum: weight given to the incoming batch estimate (PyTorch convention, default 0.1)
Var batch_norm(Var input, Var gamma, Var beta,
               Tensor running_mean, Tensor running_var,
               bool training = true, f32 momentum = 0.1f, f32 eps = 1e-5f,
               CudaMemArena *arena = nullptr);

// weight [vocab_size, d_model], indices [B, T] plain Tensor (not differentiable)
// → out [B, T, d_model]
Var embedding(Var weight, Tensor indices, CudaMemArena *arena = nullptr);

Var mse_loss(Var pred, Var target, CudaMemArena *arena = nullptr);

// Fused softmax + cross-entropy. logits/targets [N, C], output scalar.
Var cross_entropy_with_logits(Var logits, Var targets,
                              CudaMemArena *arena = nullptr);

#endif
