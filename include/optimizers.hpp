#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "autograd.hpp"
#include "layers.hpp"
#include "tensor.hpp"
#include <unordered_map>
#include <vector>

// ── DataLoader
// ────────────────────────────────────────────────────────────────

struct DataLoader {
    Tensor<f32> X, y;
    u32 batch_size;
    u32 n_samples;
    u32 cursor;
    std::vector<u32> indices;

    DataLoader(Tensor<f32> X, Tensor<f32> y, u32 batch_size);
    void shuffle();
    bool next(Tensor<f32> &X_batch, Tensor<f32> &y_batch);
};

// ── Optimizer (base)
// ──────────────────────────────────────────────────────────────

struct Optimizer {
    f32 lr;
    std::vector<Var> params;

    Optimizer(std::vector<Var> params, f32 lr);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad() = 0;
    void set_lr(f32 new_lr) { lr = new_lr; }
};

// ── sgd
// ───────────────────────────────────────────────────────────────────────

struct sgd : Optimizer {
    f32 lambda; // L2 weight decay
    f32 mu;     // momentum coefficient (0 = plain SGD)
    std::unordered_map<VarImpl *, Tensor<f32>> velocity;

    sgd(std::vector<Var> params, f32 lr, f32 lambda = 0.0f, f32 mu = 0.0f);
    sgd(Layer &model, f32 lr, f32 lambda = 0.0f, f32 mu = 0.0f);
    void step() override;
    void zero_grad() override;
};

// ── AdamW
// ─────────────────────────────────────────────────────────────────────

struct AdamW : Optimizer {
    f32 beta1;
    f32 beta2;
    f32 eps;
    f32 lambda;                              // decoupled weight decay
    u32 t;                                   // step counter for bias correction
    std::unordered_map<VarImpl *, Tensor<f32>> m; // first moment
    std::unordered_map<VarImpl *, Tensor<f32>> v; // second moment

    AdamW(std::vector<Var> params, f32 lr = 1e-3f, f32 beta1 = 0.9f,
          f32 beta2 = 0.999f, f32 eps = 1e-8f, f32 lambda = 0.0f);
    AdamW(Layer &model, f32 lr = 1e-3f, f32 beta1 = 0.9f, f32 beta2 = 0.999f,
          f32 eps = 1e-8f, f32 lambda = 0.0f);
    void step() override;
    void zero_grad() override;
};

// ── CosineAnnealingLR
// ──────────────────────────────────────────────────────── Linear warmup for
// warmup_steps, then cosine decay from max_lr down to min_lr over total_steps.
// Call step(current_step) once per optimizer step (1-indexed).

struct CosineAnnealingLR {
    Optimizer &optimizer;
    u32 warmup_steps;
    u32 total_steps;
    f32 max_lr;
    f32 min_lr;

    CosineAnnealingLR(Optimizer &optimizer, u32 warmup_steps, u32 total_steps,
                      f32 min_lr = 0.0f);
    f32 get_lr(u32 current_step) const;
    void step(u32 current_step);
};

// ── MultiStepLR
// ─────────────────────────────────────────────────────────────── Multiplies lr
// by gamma each time epoch (0-indexed) hits a milestone.

struct MultiStepLR {
    Optimizer &optimizer;
    std::vector<int> milestones;
    f32 gamma;
    f32 base_lr;

    MultiStepLR(Optimizer &optimizer, std::vector<int> milestones,
                f32 gamma = 0.1f);
    void step(int epoch);
};

// ── ReduceLROnPlateau
// ───────────────────────────────────────────────────────── Reduces lr by
// factor when loss has not improved by min_delta for patience consecutive
// epochs. Stops reducing once lr reaches min_lr.

struct ReduceLROnPlateau {
    Optimizer &optimizer;
    f32 factor;
    int patience;
    f32 min_lr;
    f32 min_delta;

    f32 best_loss = 1e9f;
    int no_improve = 0;

    ReduceLROnPlateau(Optimizer &optimizer, f32 factor = 0.1f,
                      int patience = 10, f32 min_lr = 1e-6f,
                      f32 min_delta = 1e-4f);
    void step(f32 loss, int epoch);
};

// ── EarlyStopping
// ───────────────────────────────────────────────────────────── Returns true
// (stop signal) when loss has not improved by min_delta for patience
// consecutive epochs.

struct EarlyStopping {
    int patience;
    f32 min_delta;

    f32 best_loss = 1e9f;
    int no_improve = 0;

    EarlyStopping(int patience, f32 min_delta = 1e-4f);
    bool step(f32 loss, int epoch);
};

#endif
