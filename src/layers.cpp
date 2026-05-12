#include "../include/layers.hpp"

// ── Linear
// ────────────────────────────────────────────────────────────────────

Linear::Linear(u32 in_features, u32 out_features, bool on_gpu,
               CudaMemArena *perm_arena) {
    u32 w_shape[2] = {out_features, in_features};
    W = Var(Tensor<f32>::make(2, w_shape, on_gpu, perm_arena),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_he_init(W->data);

    u32 b_shape[2] = {1, out_features};
    b = Var(tensor_zeros<f32>(2, b_shape, on_gpu, perm_arena),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
}

Var Linear::forward(Var input, CudaMemArena *arena) {
    return add(mat_mul(input, transpose(W, 0, 1, arena), arena), b, arena);
}

// ── Conv2d
// ────────────────────────────────────────────────────────────────────

Conv2d::Conv2d(u32 C_in, u32 C_out, Unfold2dParams params, bool on_gpu,
               CudaMemArena *perm_arena)
    : params(params) {
    u32 w_shape[2] = {C_out, C_in * params.k_h * params.k_w};
    W = Var(Tensor<f32>::make(2, w_shape, on_gpu, perm_arena),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_he_init(W->data);

    u32 b_shape[4] = {1, C_out, 1, 1};
    b = Var(tensor_zeros<f32>(4, b_shape, on_gpu, perm_arena),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
}

Var Conv2d::forward(Var input, CudaMemArena *arena) {
    return add(conv2d(input, transpose(W, 0, 1, arena), params, arena), b, arena);
}

// ── BatchNorm2d
// ───────────────────────────────────────────────────────────────

BatchNorm2d::BatchNorm2d(u32 C, bool on_gpu, CudaMemArena *perm_arena,
                         f32 momentum, f32 eps)
    : momentum(momentum), eps(eps) {
    u32 g_shape[1] = {C};
    gamma = Var(Tensor<f32>::make(1, g_shape, on_gpu, perm_arena),
                FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_fill(gamma->data, 1.0f);

    beta = Var(tensor_zeros<f32>(1, g_shape, on_gpu, perm_arena),
               FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    u32 stat_shape[4] = {1, C, 1, 1};
    running_mean = Tensor<f32>::make(4, stat_shape, on_gpu, perm_arena);
    tensor_fill(running_mean, 0.0f);
    running_var = Tensor<f32>::make(4, stat_shape, on_gpu, perm_arena);
    tensor_fill(running_var, 1.0f);
}

Var BatchNorm2d::forward(Var input, CudaMemArena *arena) {
    return batch_norm(input, gamma, beta, running_mean, running_var, training,
                      momentum, eps, arena);
}

// ── ResBlock
// ──────────────────────────────────────────────────────────────────

ResBlock::ResBlock(u32 C_in, u32 C_out, u32 stride, bool on_gpu,
                   CudaMemArena *perm_arena) {
    // residual path: Conv(stride) → BN → ReLU → Conv(1) → BN
    // ReLU is NOT added here — it's applied after the skip addition in
    // forward()
    residual.add<Conv2d>(C_in, C_out, Unfold2dParams(3, stride, 1), on_gpu,
                         perm_arena);
    residual.add<BatchNorm2d>(C_out, on_gpu, perm_arena);
    residual.add<ReLU>();
    residual.add<Conv2d>(C_out, C_out, Unfold2dParams(3, 1, 1), on_gpu,
                         perm_arena);
    residual.add<BatchNorm2d>(C_out, on_gpu, perm_arena);

    // projection shortcut: 1×1 Conv(stride) → BN to match spatial dims and
    // channels
    has_proj = (C_in != C_out || stride != 1);
    if (has_proj) {
        proj.add<Conv2d>(C_in, C_out, Unfold2dParams(1, stride, 0), on_gpu,
                         perm_arena);
        proj.add<BatchNorm2d>(C_out, on_gpu, perm_arena);
    }
}

Var ResBlock::forward(Var input, CudaMemArena *arena) {
    Var skip = has_proj ? proj(input, arena) : input;
    Var out = residual(input, arena);
    out = add(out, skip, arena);
    return relu(out, arena);
}

std::vector<Var> ResBlock::parameters() {
    auto p = residual.parameters();
    auto s = proj.parameters();
    p.insert(p.end(), s.begin(), s.end());
    return p;
}

void ResBlock::train(bool mode) {
    Layer::train(mode);
    residual.train(mode);
    proj.train(mode);
}
