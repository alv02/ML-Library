#include "../include/optimizers.hpp"
#include <algorithm>
#include <cstdlib>
#include <numeric>

// ── DataLoader
// ────────────────────────────────────────────────────────────────

DataLoader::DataLoader(const Tensor *X, const Tensor *y, u32 batch_size)
    : X(X), y(y), batch_size(batch_size), cursor(0) {
    n_samples = X->shape[0];
    indices.resize(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
}

void DataLoader::shuffle() {
    cursor = 0;
    for (u32 i = n_samples - 1; i > 0; i--) {
        u32 j = (u32)rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
}

bool DataLoader::next(Tensor *&X_batch, Tensor *&y_batch) {
    if (cursor >= n_samples)
        return false;

    u32 end = std::min(cursor + batch_size, n_samples);
    u32 n = end - cursor;

    X_batch = tensor_index_select(X, indices.data() + cursor, n, 0);
    y_batch = tensor_index_select(y, indices.data() + cursor, n, 0);
    cursor = end;
    return true;
}

// ── sgd
// ───────────────────────────────────────────────────────────────────────

sgd::sgd(f32 learning_rate) : lr(learning_rate), graph(nullptr) {}

sgd::~sgd() {}

void sgd::step() {
    if (!graph) {
        printf("sgd: no graph set\n");
        return;
    }
    for (function_var *fv : graph->nodes) {
        if ((fv->flags & FV_FLAG_PARAMETER) && fv->grad) {
            Tensor *tmp = tensor_mul(fv->grad, lr);
            tensor_sub(fv->val, fv->val, tmp);
            delete tmp;
        }
    }
}

void sgd::zero_grad() {
    if (!graph) {
        printf("sgd: no graph set\n");
        return;
    }
    for (function_var *fv : graph->nodes) {
        if ((fv->flags & FV_FLAG_REQUIERES_GRAD) && fv->grad)
            tensor_clear(fv->grad);
    }
}

void sgd::set_graph(Graph *graph) { this->graph = graph; }
