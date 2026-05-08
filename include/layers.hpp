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

// ── Flatten ───────────────────────────────────────────────────────────────────

struct Flatten : Layer {
    Var forward(Var input, CudaMemArena *arena = nullptr) override {
        return flatten(input, arena);
    }
};

// ── Conv2d ────────────────────────────────────────────────────────────────────
// W shape: [C_in * kH * kW, C_out]  (Wt convention matching the conv2d op)
// b shape: [1, C_out, 1, 1]

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
// gamma/beta [C], running_mean/var [1, C, 1, 1]
// Uses this->training to switch between batch stats and running stats.

struct BatchNorm2d : Layer {
    Var gamma, beta;
    Tensor running_mean, running_var;
    f32 momentum, eps;

    BatchNorm2d(u32 C, bool on_gpu, CudaMemArena *perm_arena = nullptr,
                f32 momentum = 0.1f, f32 eps = 1e-5f);
    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override { return {gamma, beta}; }
};

// ── Sequential ────────────────────────────────────────────────────────────────

struct Sequential : Layer {
    std::vector<std::unique_ptr<Layer>> layers;

    Sequential() = default;
    Sequential(Sequential &&) = default;
    Sequential &operator=(Sequential &&) = default;

    // add<T>(constructor args...) — builds a T in place
    template <typename T, typename... Args>
    Sequential &add(Args &&...args) {
        layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;
    }

    // add(unique_ptr) — for when the layer is already constructed
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

    // Cascades the training flag to all child layers.
    void train(bool mode = true) override {
        Layer::train(mode);
        for (auto &l : layers)
            l->train(mode);
    }
};

// ── ResBlock ──────────────────────────────────────────────────────────────────
// Standard residual block: Conv→BN→ReLU→Conv→BN, then add skip, then ReLU.
// When C_in != C_out or stride != 1 a 1×1 projection shortcut is used.
// Sequential is complete before ResBlock so it can be used as a member type.

struct ResBlock : Layer {
    Sequential residual;  // Conv→BN→ReLU→Conv→BN (no trailing ReLU — after add)
    Sequential proj;      // empty = identity shortcut, or 1×1 Conv→BN projection
    bool has_proj;

    ResBlock(u32 C_in, u32 C_out, u32 stride, bool on_gpu,
             CudaMemArena *perm_arena = nullptr);

    Var forward(Var input, CudaMemArena *arena = nullptr) override;
    std::vector<Var> parameters() override;
    void train(bool mode = true) override;
};

#endif
