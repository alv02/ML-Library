#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "autograd.hpp"
#include "ops.hpp"
#include <memory>
#include <vector>

// ── Layer base ────────────────────────────────────────────────────────────────

struct Layer {
    bool training = true;

    virtual Var forward(Var input, CudaMemArena *arena = nullptr) = 0;
    virtual std::vector<Var> parameters() { return {}; }
    virtual void train(bool mode = true) { training = mode; }
    void eval() { train(false); }
    Var operator()(Var input, CudaMemArena *arena = nullptr) {
        return forward(input, arena);
    }
    virtual ~Layer() = default;
};

// ── Linear ────────────────────────────────────────────────────────────────────

struct Linear : Layer {
    Var W, b;

    Linear(u32 in_features, u32 out_features, bool on_gpu,
           CudaMemArena *perm_arena = nullptr);
    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override { return {W, b}; }
};

// ── ReLU ──────────────────────────────────────────────────────────────────────

struct ReLU : Layer {
    Var forward(Var input, CudaMemArena *arena = nullptr) override {
        return relu(input, arena);
    }
};

// ── Reshape ───────────────────────────────────────────────────────────────────
// shape[i] == 0 means "copy dim i from input at runtime"

struct Reshape : Layer {
    u32 ndim;
    u32 shape[MAX_NDIM];

    Reshape(u32 ndim, const u32 *shape) : ndim(ndim) {
        memcpy(this->shape, shape, ndim * sizeof(u32));
    }

    Var forward(Var input, CudaMemArena *arena = nullptr) override {
        u32 resolved[MAX_NDIM];
        for (u32 i = 0; i < ndim; i++)
            resolved[i] = shape[i] ? shape[i] : input->data->shape[i];
        return reshape(input, resolved, ndim, arena);
    }
};

// ── Conv2d ────────────────────────────────────────────────────────────────────

struct Conv2d : Layer {
    Var W, b;
    Unfold2dParams params;

    Conv2d(u32 C_in, u32 C_out, Unfold2dParams params, bool on_gpu,
           CudaMemArena *perm_arena = nullptr);
    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override { return {W, b}; }
};

// ── MaxPool2d ─────────────────────────────────────────────────────────────────

struct MaxPool2d : Layer {
    Unfold2dParams params;

    MaxPool2d(Unfold2dParams params) : params(params) {}
    Var forward(Var input, CudaMemArena *arena = nullptr) override {
        return max_pool2d(input, params, arena);
    }
};

// ── BatchNorm2d ───────────────────────────────────────────────────────────────

struct BatchNorm2d : Layer {
    Var gamma, beta;
    Tensor<f32> running_mean, running_var;
    f32 momentum, eps;

    BatchNorm2d(u32 C, bool on_gpu, CudaMemArena *perm_arena = nullptr,
                f32 momentum = 0.1f, f32 eps = 1e-5f);
    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override { return {gamma, beta}; }
};

// ── EmbeddingLayer ────────────────────────────────────────────────────────────
// weight [vocab_size, d_model], indices [*] → out [*, d_model]

struct EmbeddingLayer {
    Var weight;

    EmbeddingLayer(u32 vocab_size, u32 d_model, bool on_gpu,
                   CudaMemArena *perm_arena = nullptr);
    Var forward(TensorU32 indices, CudaMemArena *arena = nullptr);
    std::vector<Var> parameters() { return {weight}; }
};

// ── PositionalEmbeddingLayer ──────────────────────────────────────────────────
// weight [max_seq_len, d_model]; forward(T) returns [T, d_model].

struct PositionalEmbeddingLayer {
    Var weight;
    TensorU32 positions;

    PositionalEmbeddingLayer(u32 max_seq_len, u32 d_model, bool on_gpu,
                             CudaMemArena *perm_arena = nullptr);
    Var forward(u32 T, CudaMemArena *arena = nullptr);
    std::vector<Var> parameters() { return {weight}; }
};

// ── InputEmbedding ────────────────────────────────────────────────────────────
// tok_emb(tokens) + pos_emb(T) + dropout → [B, T, d_model]

struct InputEmbedding {
    EmbeddingLayer tok_emb;
    PositionalEmbeddingLayer pos_emb;
    f32 dropout_p;
    bool training = true;

    InputEmbedding(u32 vocab_size, u32 max_seq_len, u32 d_model, f32 dropout_p,
                   bool on_gpu, CudaMemArena *perm_arena = nullptr);
    Var forward(TensorU32 tokens, CudaMemArena *arena = nullptr);
    std::vector<Var> parameters();
    void train(bool mode = true) { training = mode; }
    void eval() { train(false); }
};

// ── Sequential ────────────────────────────────────────────────────────────────

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

    Var forward(Var input, CudaMemArena *arena = nullptr) override {
        Var cur = input;
        for (auto &l : layers)
            cur = l->forward(cur, arena);
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

// ── ResBlock ──────────────────────────────────────────────────────────────────

struct ResBlock : Layer {
    Sequential residual;
    Sequential proj;
    bool has_proj;

    ResBlock(u32 C_in, u32 C_out, u32 stride, bool on_gpu,
             CudaMemArena *perm_arena = nullptr);

    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override;
    void train(bool mode = true) override;
};

#endif
