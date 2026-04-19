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

sgd::sgd(f32 learning_rate, f32 lambda, f32 mu)
    : lr(learning_rate), lambda(lambda), mu(mu), graph(nullptr) {}

sgd::~sgd() {
    for (auto &[fv, v] : velocity)
        delete v;
}

void sgd::step() {
    if (!graph) {
        printf("sgd: no graph set\n");
        return;
    }
    for (function_var *fv : graph->nodes) {
        if ((fv->flags & FV_FLAG_PARAMETER) && fv->grad) {
            if (lambda > 0.0f) {
                Tensor *reg = tensor_mul(fv->val, lambda);
                tensor_add(fv->grad, fv->grad, reg);
                delete reg;
            }

            if (mu > 0.0f) {
                Tensor *&v = velocity[fv];
                if (!v) {
                    v = tensor_create_like(fv->val);
                    tensor_clear(v);
                }
                // v = mu * v + grad
                tensor_mul(v, v, mu);
                tensor_add(v, v, fv->grad);
                // param -= lr * v
                Tensor *tmp = tensor_mul(v, lr);
                tensor_sub(fv->val, fv->val, tmp);
                delete tmp;
            } else {
                Tensor *tmp = tensor_mul(fv->grad, lr);
                tensor_sub(fv->val, fv->val, tmp);
                delete tmp;
            }
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
