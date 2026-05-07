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

// ── MultiStepLR ───────────────────────────────────────────────────────────────
// Multiplies lr by gamma each time epoch (0-indexed) hits a milestone.

struct MultiStepLR {
    sgd &optimizer;
    std::vector<int> milestones;
    f32 gamma;
    f32 base_lr;

    MultiStepLR(sgd &optimizer, std::vector<int> milestones, f32 gamma = 0.1f);
    void step(int epoch);
};

// ── ReduceLROnPlateau ─────────────────────────────────────────────────────────
// Reduces lr by factor when loss has not improved by min_delta for patience
// consecutive epochs. Stops reducing once lr reaches min_lr.

struct ReduceLROnPlateau {
    sgd &optimizer;
    f32 factor;
    int patience;
    f32 min_lr;
    f32 min_delta;

    f32 best_loss = 1e9f;
    int no_improve = 0;

    ReduceLROnPlateau(sgd &optimizer, f32 factor = 0.1f, int patience = 10,
                      f32 min_lr = 1e-6f, f32 min_delta = 1e-4f);
    void step(f32 loss, int epoch);
};

// ── EarlyStopping ─────────────────────────────────────────────────────────────
// Returns true (stop signal) when loss has not improved by min_delta for
// patience consecutive epochs.

struct EarlyStopping {
    int patience;
    f32 min_delta;

    f32 best_loss = 1e9f;
    int no_improve = 0;

    EarlyStopping(int patience, f32 min_delta = 1e-4f);
    bool step(f32 loss, int epoch);
};

#endif
