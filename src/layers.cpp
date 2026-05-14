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
    return add(conv2d(input, transpose(W, 0, 1, arena), params, arena), b,
               arena);
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

// ── LayerNorm ─────────────────────────────────────────────────────────────────

LayerNorm::LayerNorm(u32 d_model, bool on_gpu, CudaMemArena *perm_arena) {
    u32 shape[1] = {d_model};
    gamma = Var(Tensor<f32>::make(1, shape, on_gpu, perm_arena),
                FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_fill(gamma->data, 1.0f);

    beta = Var(Tensor<f32>::make(1, shape, on_gpu, perm_arena),
               FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_fill(beta->data, 0.0f);
}

Var LayerNorm::forward(Var input, CudaMemArena *arena) {
    return layer_norm(input, gamma, beta, 1e-5f, arena);
}

// ── EmbeddingLayer ────────────────────────────────────────────────────────────

EmbeddingLayer::EmbeddingLayer(u32 vocab_size, u32 d_model, bool on_gpu,
                               CudaMemArena *perm_arena) {
    u32 w_shape[2] = {vocab_size, d_model};
    weight = Var(Tensor<f32>::make(2, w_shape, on_gpu, perm_arena),
                 FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_he_init(weight->data);
}

Var EmbeddingLayer::forward(TensorU32 indices, CudaMemArena *arena) {
    u32 D        = weight->data->shape[1];
    u32 idx_ndim = indices->ndim;
    u32 numel    = indices->numel();

    // [*] → [numel] → [numel, 1] → [numel, D] (stride-0 broadcast)
    TensorU32 idx = tensor_view<u32>(indices);
    u32 flat_shape[1] = {numel};
    tensor_reshape(idx, flat_shape, 1, arena);
    tensor_unsqueeze(idx.impl(), 1);
    u32 bcast_shape[2] = {numel, D};
    tensor_broadcast_to(idx.impl(), bcast_shape, 2);

    Var out_flat = gather(weight, idx, 0, arena);

    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, indices->shape, idx_ndim * sizeof(u32));
    out_shape[idx_ndim] = D;
    return reshape(out_flat, out_shape, idx_ndim + 1, arena);
}

// ── PositionalEmbeddingLayer ──────────────────────────────────────────────────

PositionalEmbeddingLayer::PositionalEmbeddingLayer(u32 max_seq_len, u32 d_model,
                                                   bool on_gpu,
                                                   CudaMemArena *perm_arena) {
    u32 w_shape[2] = {max_seq_len, d_model};
    weight = Var(Tensor<f32>::make(2, w_shape, on_gpu, perm_arena),
                 FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
    tensor_he_init(weight->data);

    u32 p_shape[1] = {max_seq_len};
    positions = TensorU32::make(1, p_shape, on_gpu);
    tensor_arange(positions);
}

Var PositionalEmbeddingLayer::forward(u32 T, CudaMemArena *arena) {
    u32 D = weight->data->shape[1];

    // [T] → [T, 1] → [T, D] (stride-0 broadcast)
    TensorU32 pos = tensor_view<u32>(positions);
    pos->shape[0] = T;
    tensor_unsqueeze(pos.impl(), 1);
    u32 bcast_shape[2] = {T, D};
    tensor_broadcast_to(pos.impl(), bcast_shape, 2);

    Var out_flat = gather(weight, pos, 0, arena);
    u32 out_shape[2] = {T, D};
    return reshape(out_flat, out_shape, 2, arena);
}

// ── InputEmbedding ────────────────────────────────────────────────────────────

InputEmbedding::InputEmbedding(u32 vocab_size, u32 max_seq_len, u32 d_model,
                               f32 dropout_p, bool on_gpu,
                               CudaMemArena *perm_arena)
    : tok_emb(vocab_size, d_model, on_gpu, perm_arena),
      pos_emb(max_seq_len, d_model, on_gpu, perm_arena),
      dropout_p(dropout_p) {}

Var InputEmbedding::forward(TensorU32 tokens, CudaMemArena *arena) {
    u32 T = tokens->shape[1];
    Var x = add(tok_emb.forward(tokens, arena), pos_emb.forward(T, arena), arena);
    return dropout(x, dropout_p, training, arena);
}

std::vector<Var> InputEmbedding::parameters() {
    auto p = tok_emb.parameters();
    auto q = pos_emb.parameters();
    p.insert(p.end(), q.begin(), q.end());
    return p;
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

MultiHeadAttention::MultiHeadAttention(u32 d_model, u32 n_heads, u32 max_seq_len,
                                       f32 dropout_p, bool on_gpu,
                                       CudaMemArena *perm_arena)
    : n_heads(n_heads), d_head(d_model / n_heads), dropout_p(dropout_p),
      q_proj(d_model, d_model, on_gpu, perm_arena),
      k_proj(d_model, d_model, on_gpu, perm_arena),
      v_proj(d_model, d_model, on_gpu, perm_arena),
      out_proj(d_model, d_model, on_gpu, perm_arena) {
    u32 mask_shape[2] = {max_seq_len, max_seq_len};
    causal_mask = Tensor<f32>::make(2, mask_shape, on_gpu, perm_arena);
    tensor_causal_mask(causal_mask);
}

Var MultiHeadAttention::forward(Var x, CudaMemArena *arena) {
    const u32 B      = x->data->shape[0];
    const u32 T      = x->data->shape[1];
    const u32 d_model = x->data->shape[2];

    Var Q = q_proj.forward(x, arena);
    Var K = k_proj.forward(x, arena);
    Var V = v_proj.forward(x, arena);

    // [B, T, d_model] → [B, T, n_heads, d_head] → [B, n_heads, T, d_head]
    u32 head_shape[4] = {B, T, n_heads, d_head};
    Q = transpose(reshape(Q, head_shape, 4, arena), 1, 2, arena);
    K = transpose(reshape(K, head_shape, 4, arena), 1, 2, arena);
    V = transpose(reshape(V, head_shape, 4, arena), 1, 2, arena);

    // scores [B, n_heads, T, T]
    Var scores = mul(mat_mul(Q, transpose(K, 2, 3, arena), arena),
                     1.0f / sqrtf((f32)d_head), arena);

    // Causal mask: [T, T] strided view into the precomputed [max_T, max_T] mask
    Tensor<f32> mask_view = tensor_view(causal_mask);
    mask_view->shape[0] = T;
    mask_view->shape[1] = T;
    scores = add(scores, Var(mask_view, FV_FLAG_NONE), arena);

    Var attn = dropout(softmax(scores, -1, arena), dropout_p, training, arena);

    // [B, n_heads, T, d_head] → [B, T, n_heads, d_head] → [B, T, d_model]
    Var out = transpose(mat_mul(attn, V, arena), 1, 2, arena);
    u32 out_shape[3] = {B, T, d_model};
    return out_proj.forward(reshape(out, out_shape, 3, arena), arena);
}

std::vector<Var> MultiHeadAttention::parameters() {
    auto p = q_proj.parameters();
    auto k = k_proj.parameters();
    auto v = v_proj.parameters();
    auto o = out_proj.parameters();
    p.insert(p.end(), k.begin(), k.end());
    p.insert(p.end(), v.begin(), v.end());
    p.insert(p.end(), o.begin(), o.end());
    return p;
}

// ── TransformerBlock ─────────────────────────────────────────────────────────

TransformerBlock::TransformerBlock(u32 d_model, u32 n_heads, u32 max_seq_len,
                                   f32 dropout_p, bool on_gpu,
                                   CudaMemArena *perm_arena)
    : ln1(d_model, on_gpu, perm_arena),
      ln2(d_model, on_gpu, perm_arena),
      mha(d_model, n_heads, max_seq_len, dropout_p, on_gpu, perm_arena),
      mlp_fc(d_model, 4 * d_model, on_gpu, perm_arena),
      mlp_proj(4 * d_model, d_model, on_gpu, perm_arena),
      dropout_p(dropout_p) {}

Var TransformerBlock::forward(Var x, CudaMemArena *arena) {
    Var h = add(x, dropout(mha.forward(ln1.forward(x, arena), arena),
                           dropout_p, training, arena), arena);
    Var m = mlp_proj.forward(gelu(mlp_fc.forward(ln2.forward(h, arena), arena), arena), arena);
    return add(h, dropout(m, dropout_p, training, arena), arena);
}

std::vector<Var> TransformerBlock::parameters() {
    auto p = ln1.parameters();
    auto a = ln2.parameters();
    auto b = mha.parameters();
    auto c = mlp_fc.parameters();
    auto d = mlp_proj.parameters();
    p.insert(p.end(), a.begin(), a.end());
    p.insert(p.end(), b.begin(), b.end());
    p.insert(p.end(), c.begin(), c.end());
    p.insert(p.end(), d.begin(), d.end());
    return p;
}

void TransformerBlock::train(bool mode) {
    Layer::train(mode);
    ln1.train(mode);
    ln2.train(mode);
    mha.train(mode);
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
