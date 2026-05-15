#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "autograd.hpp"
#include "ops.hpp"
#include <memory>
#include <vector>

// ── Layer base
// ────────────────────────────────────────────────────────────────

struct Layer {
    bool training = true;

    virtual Var forward(Var input) = 0;
    virtual std::vector<Var> parameters() { return {}; }
    virtual void train(bool mode = true) { training = mode; }
    void eval() { train(false); }
    Var operator()(Var input) { return forward(input); }
    virtual ~Layer() = default;
};

// ── Linear
// ────────────────────────────────────────────────────────────────────

struct Linear : Layer {
    Var W, b;

    Linear(u32 in_features, u32 out_features, bool on_gpu, float init_std = 0.0f);
    Var forward(Var input) override;
    std::vector<Var> parameters() override { return {W, b}; }
};

// ── ReLU
// ──────────────────────────────────────────────────────────────────────

struct ReLU : Layer {
    Var forward(Var input) override { return relu(input); }
};

// ── Reshape
// ─────────────────────────────────────────────────────────────────── shape[i]
// == 0 means "copy dim i from input at runtime"

struct Reshape : Layer {
    u32 ndim;
    u32 shape[MAX_NDIM];

    Reshape(u32 ndim, const u32 *shape) : ndim(ndim) {
        memcpy(this->shape, shape, ndim * sizeof(u32));
    }

    Var forward(Var input) override {
        u32 resolved[MAX_NDIM];
        for (u32 i = 0; i < ndim; i++)
            resolved[i] = shape[i] ? shape[i] : input->data->shape[i];
        return reshape(input, resolved, ndim);
    }
};

// ── Conv2d
// ────────────────────────────────────────────────────────────────────

struct Conv2d : Layer {
    Var W, b;
    Unfold2dParams params;

    Conv2d(u32 C_in, u32 C_out, Unfold2dParams params, bool on_gpu);
    Var forward(Var input) override;
    std::vector<Var> parameters() override { return {W, b}; }
};

// ── MaxPool2d
// ─────────────────────────────────────────────────────────────────

struct MaxPool2d : Layer {
    Unfold2dParams params;

    MaxPool2d(Unfold2dParams params) : params(params) {}
    Var forward(Var input) override { return max_pool2d(input, params); }
};

// ── BatchNorm2d
// ───────────────────────────────────────────────────────────────

struct BatchNorm2d : Layer {
    Var gamma, beta;
    Tensor<f32> running_mean, running_var;
    f32 momentum, eps;

    BatchNorm2d(u32 C, bool on_gpu, f32 momentum = 0.1f, f32 eps = 1e-5f);
    Var forward(Var input) override;
    std::vector<Var> parameters() override { return {gamma, beta}; }
};

// ── LayerNorm
// ───────────────────────────────────────────────────────────────── Normalizes
// over the last dimension. gamma [d_model] init 1, beta [d_model] init 0.

struct LayerNorm : Layer {
    Var gamma, beta;

    LayerNorm(u32 d_model, bool on_gpu);
    Var forward(Var input) override;
    std::vector<Var> parameters() override { return {gamma, beta}; }
};

// ── EmbeddingLayer
// ──────────────────────────────────────────────────────────── weight
// [vocab_size, d_model], indices [*] → out [*, d_model]

struct EmbeddingLayer {
    Var weight;

    EmbeddingLayer(u32 vocab_size, u32 d_model, bool on_gpu, float init_std = 0.0f);
    Var forward(TensorU32 indices);
    std::vector<Var> parameters() { return {weight}; }
};

// ── PositionalEmbeddingLayer
// ────────────────────────────────────────────────── weight [max_seq_len,
// d_model]; forward(T) returns [T, d_model].

struct PositionalEmbeddingLayer {
    Var weight;
    TensorU32 positions;

    PositionalEmbeddingLayer(u32 max_seq_len, u32 d_model, bool on_gpu,
                             float init_std = 0.0f);
    Var forward(u32 T);
    std::vector<Var> parameters() { return {weight}; }
};

// ── InputEmbedding
// ──────────────────────────────────────────────────────────── tok_emb(tokens)
// + pos_emb(T) + dropout → [B, T, d_model]

struct InputEmbedding {
    EmbeddingLayer tok_emb;
    PositionalEmbeddingLayer pos_emb;
    f32 dropout_p;
    bool training = true;

    InputEmbedding(u32 vocab_size, u32 max_seq_len, u32 d_model, f32 dropout_p,
                   bool on_gpu, float init_std = 0.0f);
    Var forward(TensorU32 tokens);
    std::vector<Var> parameters();
    void train(bool mode = true) { training = mode; }
    void eval() { train(false); }
};

// ── Sequential
// ────────────────────────────────────────────────────────────────

struct Sequential : Layer {
    std::vector<std::unique_ptr<Layer>> layers;

    Sequential() = default;
    Sequential(Sequential &&) = default;
    Sequential &operator=(Sequential &&) = default;

    template <typename T, typename... Args> Sequential &add(Args &&...args) {
        layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;
    }

    Sequential &add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
        return *this;
    }

    Var forward(Var input) override {
        Var cur = input;
        for (auto &l : layers)
            cur = l->forward(cur);
        return cur;
    }

    std::vector<Var> parameters() override {
        std::vector<Var> params;
        for (auto &l : layers) {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    void train(bool mode = true) override {
        Layer::train(mode);
        for (auto &l : layers)
            l->train(mode);
    }
};

// ── MultiHeadAttention
// ──────────────────────────────────────────────────────── GPT-style causal
// self-attention. dropout_p=0 disables attn dropout.

struct MultiHeadAttention {
    u32 n_heads, d_head;
    Var W_q, W_k, W_v, W_o;
    Tensor<f32>
        causal_mask; // [max_seq_len, max_seq_len], 0 on/below diag, -1e9 above
    f32 dropout_p;
    bool training = true;

    MultiHeadAttention(u32 d_model, u32 n_heads, u32 max_seq_len, f32 dropout_p,
                       bool on_gpu, float init_std = 0.0f);
    Var forward(Var x);
    std::vector<Var> parameters();
    void train(bool mode = true) { training = mode; }
    void eval() { train(false); }
};

// ── TransformerBlock
// ────────────────────────────────────────────────────────── Pre-norm GPT-style
// block: x + MHA(LN(x)), then x + MLP(LN(x)). MLP = Linear(d_model→4*d_model) →
// GELU → Linear(4*d_model→d_model).

struct TransformerBlock : Layer {
    LayerNorm ln1, ln2;
    MultiHeadAttention mha;
    Linear mlp_fc, mlp_proj;
    f32 dropout_p;

    TransformerBlock(u32 d_model, u32 n_heads, u32 max_seq_len, f32 dropout_p,
                     bool on_gpu, float init_std = 0.0f);
    Var forward(Var x) override;
    std::vector<Var> parameters() override;
    void train(bool mode = true) override;
};

// ── ResBlock
// ──────────────────────────────────────────────────────────────────

struct ResBlock : Layer {
    Sequential residual;
    Sequential proj;
    bool has_proj;

    ResBlock(u32 C_in, u32 C_out, u32 stride, bool on_gpu);

    Var forward(Var input) override;
    std::vector<Var> parameters() override;
    void train(bool mode = true) override;
};

#endif
