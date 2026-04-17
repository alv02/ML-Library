#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "autograd.hpp"
#include "tensor.hpp"
#include <vector>

// ── DataLoader
// ──────────────────────────────────────────────────────────────── Shuffles the
// dataset each epoch and yields mini-batches via next(). When batch_size ==
// n_samples this degenerates to full-batch GD.

struct DataLoader {
    const Tensor *X;
    const Tensor *y;
    u32 batch_size;
    u32 n_samples;
    u32 cursor;
    std::vector<u32> indices;

    DataLoader(const Tensor *X, const Tensor *y, u32 batch_size);

    // Fisher-Yates shuffle — call once at the start of each epoch.
    void shuffle();

    // Allocates and returns the next batch. Returns false when the epoch is
    // done. The caller owns the returned tensors and must delete them.
    bool next(Tensor *&X_batch, Tensor *&y_batch);
};

// ── sgd
// ───────────────────────────────────────────────────────────────────────

struct sgd {
    f32 lr;
    f32 lambda;
    Graph *graph; // non-owning

    sgd(f32 learning_rate, f32 lambda = 0.0f);
    ~sgd();
    void step();
    void zero_grad();
    void set_graph(Graph *graph);
};

#endif
