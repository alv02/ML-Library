#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

Var mat_mul(Var a, Var b);
Var add(Var a, Var b);
Var relu(Var a);

// input  [N, C_in, H, W]
// weight [C_in*kH*kW, C_out]  (pre-transposed, Wt convention)
// output [N, C_out, L_h, L_w]
Var conv2d(Var input, Var weight, Unfold2dParams params);

// input [N, C, H, W] → output [N, C, L_h, L_w]
Var max_pool2d(Var input, Unfold2dParams params);

// [N, ...] → [N, C*H*W]
Var flatten(Var input);

// input [N, C, H, W], gamma/beta [C]
Var batch_norm(Var input, Var gamma, Var beta, f32 eps = 1e-5f);

Var mse_loss(Var pred, Var target);

// Fused softmax + cross-entropy. logits/targets [N, C], output scalar.
Var cross_entropy_with_logits(Var logits, Var targets);

#endif
