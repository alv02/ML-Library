#include "../include/models.hpp"

// ── linear_model ─────────────────────────────────────────────────────────────

linear_model::linear_model(u32 n_features, bool on_gpu) {
    u32 w_shape[2] = {n_features, 1};
    W = Var(Tensor::make(2, w_shape, on_gpu),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    u32 b_shape[1] = {1};
    b = Var(Tensor::make(1, b_shape, on_gpu),
            FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
}

Var linear_model::predict(Var X) { return add(mat_mul(X, W), b); }

Var linear_model::forward(Var X, Var y) { return mse_loss(predict(X), y); }

// ── nn_model ─────────────────────────────────────────────────────────────────

nn_model::nn_model(u32 in_features, const std::vector<u32> &layer_sizes,
                   bool on_gpu) {
    for (u32 out : layer_sizes) {
        u32 w_shape[2] = {in_features, out};
        Var w(Tensor::make(2, w_shape, on_gpu),
              FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
        tensor_he_init(w->data);
        Wt.push_back(w);

        u32 b_shape[2] = {1, out};
        b.push_back(Var(Tensor::make(2, b_shape, on_gpu),
                        FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        in_features = out;
    }
}

std::vector<Var> nn_model::parameters() const {
    std::vector<Var> p;
    for (auto &w : Wt)  p.push_back(w);
    for (auto &bi : b)  p.push_back(bi);
    return p;
}

Var nn_model::predict(Var X) {
    Var cur = X;
    for (u32 l = 0; l < (u32)Wt.size(); l++) {
        cur = add(mat_mul(cur, Wt[l]), b[l]);
        if (l < (u32)Wt.size() - 1)
            cur = relu(cur);
    }
    return cur;
}

Var nn_model::forward(Var X, Var y) {
    return cross_entropy_with_logits(predict(X), y);
}

// ── cnn_model ─────────────────────────────────────────────────────────────────

cnn_model::cnn_model(u32 C_in, u32 H, u32 W, bool on_gpu,
                     const std::vector<conv_layer_params> &conv_layers,
                     const std::vector<u32> &dense_sizes)
    : conv_specs(conv_layers) {

    u32 H_cur = H, W_cur = W, C_cur = C_in;

    for (const auto &spec : conv_layers) {
        u32 C_out = spec.C_out;

        u32 k_shape[2] = {C_cur * spec.params.k_h * spec.params.k_w, C_out};
        Var kernel(Tensor::make(2, k_shape, on_gpu),
                   FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
        tensor_he_init(kernel->data);
        kernels.push_back(kernel);

        u32 b_shape[4] = {1, C_out, 1, 1};
        conv_b.push_back(Var(Tensor::make(4, b_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        if (spec.bn) {
            u32 g_shape[1] = {C_out};
            Var gamma(Tensor::make(1, g_shape, on_gpu),
                      FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
            tensor_fill(gamma->data, 1.0f);
            bn_gamma.push_back(gamma);
            bn_beta.push_back(Var(Tensor::make(1, g_shape, on_gpu),
                                  FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));
        }

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

        C_cur = C_out;
    }

    u32 flat = C_cur * H_cur * W_cur;
    for (u32 out : dense_sizes) {
        u32 w_shape[2] = {flat, out};
        Var w(Tensor::make(2, w_shape, on_gpu),
              FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);
        tensor_he_init(w->data);
        dense_Wt.push_back(w);

        u32 b_shape[2] = {1, out};
        dense_b.push_back(Var(Tensor::make(2, b_shape, on_gpu),
                               FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        flat = out;
    }
}

std::vector<Var> cnn_model::parameters() const {
    std::vector<Var> p;
    for (auto &k : kernels)  p.push_back(k);
    for (auto &b : conv_b)   p.push_back(b);
    for (auto &g : bn_gamma) p.push_back(g);
    for (auto &b : bn_beta)  p.push_back(b);
    for (auto &w : dense_Wt) p.push_back(w);
    for (auto &b : dense_b)  p.push_back(b);
    return p;
}

Var cnn_model::predict(Var X) {
    Var cur = X;
    u32 bn_idx = 0;

    for (u32 l = 0; l < (u32)kernels.size(); l++) {
        cur = conv2d(cur, kernels[l], conv_specs[l].params);
        cur = add(cur, conv_b[l]);
        if (conv_specs[l].bn) {
            cur = batch_norm(cur, bn_gamma[bn_idx], bn_beta[bn_idx]);
            bn_idx++;
        }
        cur = relu(cur);
        if (conv_specs[l].pool)
            cur = max_pool2d(cur, conv_specs[l].pool_params);
    }

    cur = flatten(cur);

    for (u32 l = 0; l < (u32)dense_Wt.size(); l++) {
        cur = add(mat_mul(cur, dense_Wt[l]), dense_b[l]);
        if (l < (u32)dense_Wt.size() - 1)
            cur = relu(cur);
    }

    return cur;
}

Var cnn_model::forward(Var X, Var y) {
    return cross_entropy_with_logits(predict(X), y);
}
