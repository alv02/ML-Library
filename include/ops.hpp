#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

Var mat_mul(Var a, Var b);
Var reshape(Var a, const u32 *shape, u32 ndim);
Var transpose(Var a, u32 d0, u32 d1);
Var unsqueeze(Var a, u32 dim);
Var broadcast_to(Var a, const u32 *shape, u32 ndim);
Var gather(Var src, TensorU32 indices, u32 dim);
Var add(Var a, Var b);
Var mul(Var a, f32 scalar);
Var mul(Var a, Var b);
Var softmax(Var a, i32 dim = -1);
Var relu(Var a);
Var gelu(Var a);

Var conv2d(Var input, Var weight, Unfold2dParams params);
Var max_pool2d(Var input, Unfold2dParams params);
Var layer_norm(Var input, Var gamma, Var beta, f32 eps = 1e-5f);
Var batch_norm(Var input, Var gamma, Var beta, Tensor<f32> running_mean,
               Tensor<f32> running_var, bool training = true,
               f32 momentum = 0.1f, f32 eps = 1e-5f);
Var dropout(Var a, f32 p, bool training = true);
Var mse_loss(Var pred, Var target);
Var cross_entropy_with_logits(Var logits, Var targets);
Var cross_entropy_with_logits(Var logits, TensorU32 targets);

#endif
