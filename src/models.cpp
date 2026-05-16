#include "../include/models.hpp"

// ── GPTModel ──────────────────────────────────────────────────────────────────

GPTModel::GPTModel(u32 vocab_size, u32 d_model, u32 n_heads, u32 n_layers,
                   u32 max_seq_len, f32 dropout_p, bool on_gpu)
    : embedding(vocab_size, max_seq_len, d_model, dropout_p, on_gpu, 0.02f),
      ln_f(d_model, on_gpu),
      lm_head(d_model, vocab_size, on_gpu, 0.02f) {
    blocks.reserve(n_layers);
    for (u32 i = 0; i < n_layers; i++)
        blocks.emplace_back(d_model, n_heads, max_seq_len, dropout_p, on_gpu, 0.02f);
}

Var GPTModel::forward(TensorU32 tokens) {
    Var x = embedding.forward(tokens);
    for (auto &block : blocks)
        x = block.forward(x);
    return lm_head.forward(ln_f.forward(x));
}

std::vector<Var> GPTModel::parameters() {
    auto p = embedding.parameters();
    for (auto &block : blocks) {
        auto bp = block.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto lnp = ln_f.parameters();
    auto lhp = lm_head.parameters();
    p.insert(p.end(), lnp.begin(), lnp.end());
    p.insert(p.end(), lhp.begin(), lhp.end());
    return p;
}

void GPTModel::train(bool mode) {
    training = mode;
    embedding.train(mode);
    for (auto &block : blocks)
        block.train(mode);
    ln_f.train(mode);
}

Sequential make_mlp(u32 in_features, const std::vector<u32> &sizes,
                    bool on_gpu) {
    Sequential model;
    for (u32 i = 0; i < (u32)sizes.size(); i++) {
        model.add<Linear>(in_features, sizes[i], on_gpu);
        if (i < (u32)sizes.size() - 1)
            model.add<ReLU>();
        in_features = sizes[i];
    }
    return model;
}

Sequential make_cnn(u32 C_in, u32 H, u32 W, bool on_gpu,
                    const std::vector<conv_layer_params> &conv_layers,
                    const std::vector<u32> &dense_sizes) {
    Sequential model;
    u32 H_cur = H, W_cur = W, C_cur = C_in;

    for (const auto &spec : conv_layers) {
        model.add<Conv2d>(C_cur, spec.C_out, spec.params, on_gpu);
        if (spec.bn)
            model.add<BatchNorm2d>(spec.C_out, on_gpu);
        model.add<ReLU>();
        if (spec.pool)
            model.add<MaxPool2d>(spec.pool_params);

        Unfold2dParams p = spec.params;
        p.compute_output_size(H_cur, W_cur);
        H_cur = p.L_h;
        W_cur = p.L_w;

        if (spec.pool) {
            Unfold2dParams pp = spec.pool_params;
            pp.compute_output_size(H_cur, W_cur);
            H_cur = pp.L_h;
            W_cur = pp.L_w;
        }

        C_cur = spec.C_out;
    }

    u32 flat = C_cur * H_cur * W_cur;
    u32 flat_shape[2] = {0, flat};
    model.add<Reshape>(2, flat_shape);
    for (u32 i = 0; i < (u32)dense_sizes.size(); i++) {
        model.add<Linear>(flat, dense_sizes[i], on_gpu);
        if (i < (u32)dense_sizes.size() - 1)
            model.add<ReLU>();
        flat = dense_sizes[i];
    }

    return model;
}

Sequential make_resnet(u32 num_classes, bool on_gpu,
                       const std::vector<u32> &stage_blocks) {
    const u32 channels[] = {64, 128, 256, 512};
    Sequential model;

    // Stem: Conv(3→64, k=3, s=1, p=1) → BN → ReLU
    // No MaxPool — CIFAR-10 images are 32×32, halving early loses too much detail.
    model.add<Conv2d>(3, 64, Unfold2dParams(3, 1, 1), on_gpu);
    model.add<BatchNorm2d>(64, on_gpu);
    model.add<ReLU>();

    // Residual stages. Stage 0 keeps spatial size (stride=1), stages 1-3 halve it.
    u32 C_in = 64;
    for (u32 s = 0; s < (u32)stage_blocks.size(); s++) {
        u32 C_out = channels[s];
        u32 first_stride = (s == 0) ? 1 : 2;
        for (u32 b = 0; b < stage_blocks[s]; b++) {
            model.add<ResBlock>(C_in, C_out, (b == 0) ? first_stride : 1u, on_gpu);
            C_in = C_out;
        }
    }

    // Global average pool: after 3 stride-2 stages on 32×32, spatial size is 4×4.
    // MaxPool(4,4) collapses it to 1×1 → [N, 512, 1, 1] → [N, 512].
    model.add<MaxPool2d>(Unfold2dParams(4, 4, 0));
    u32 resnet_flat[2] = {0, channels[stage_blocks.size() - 1]};
    model.add<Reshape>(2, resnet_flat);
    model.add<Linear>(channels[stage_blocks.size() - 1], num_classes, on_gpu);

    return model;
}
