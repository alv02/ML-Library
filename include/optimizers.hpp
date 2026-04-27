#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "autograd.hpp"
#include "tensor.hpp"
#include <unordered_map>
#include <vector>

// ── DataLoader ────────────────────────────────────────────────────────────────

struct DataLoader {
    Tensor X, y;
    u32 batch_size;
    u32 n_samples;
    u32 cursor;
    std::vector<u32> indices;

    DataLoader(Tensor X, Tensor y, u32 batch_size);
    void shuffle();
    bool next(Tensor &X_batch, Tensor &y_batch);
};

// ── sgd ───────────────────────────────────────────────────────────────────────

struct sgd {
    f32 lr;
    f32 lambda; // L2 weight decay
    f32 mu;     // momentum coefficient (0 = plain SGD)
    std::vector<Var> params;
    std::unordered_map<VarImpl *, Tensor> velocity;

    sgd(std::vector<Var> params, f32 lr, f32 lambda = 0.0f, f32 mu = 0.0f);
    void step();
    void zero_grad();
};

#endif
