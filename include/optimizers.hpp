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
    bool next(Tensor &X_batch, Tensor &y_batch,
              CudaMemArena *arena = nullptr);
};

// ── sgd ───────────────────────────────────────────────────────────────────────

struct sgd {
    f32 lr;
    f32 lambda; // L2 weight decay
    f32 mu;     // momentum coefficient (0 = plain SGD)
    std::vector<Var> params;
    std::unordered_map<VarImpl *, Tensor> velocity;
    CudaMemArena *perm_arena; // for persistent velocity tensors

    sgd(std::vector<Var> params, f32 lr, f32 lambda = 0.0f, f32 mu = 0.0f,
        CudaMemArena *perm_arena = nullptr);
    void step(CudaMemArena *arena = nullptr);
    void zero_grad();
    void set_lr(f32 new_lr) { lr = new_lr; }
};

#endif
