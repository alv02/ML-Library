#include "../include/optimizers.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>

// ── DataLoader ────────────────────────────────────────────────────────────────

DataLoader::DataLoader(Tensor X, Tensor y, u32 batch_size)
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

bool DataLoader::next(Tensor &X_batch, Tensor &y_batch, CudaMemArena *arena) {
    if (cursor >= n_samples)
        return false;
    u32 end = std::min(cursor + batch_size, n_samples);
    u32 n = end - cursor;
    X_batch = tensor_index_select(X, indices.data() + cursor, n, 0, arena);
    y_batch = tensor_index_select(y, indices.data() + cursor, n, 0, arena);
    cursor = end;
    return true;
}

// ── sgd ───────────────────────────────────────────────────────────────────────

sgd::sgd(std::vector<Var> params, f32 lr, f32 lambda, f32 mu,
         CudaMemArena *perm_arena)
    : lr(lr), lambda(lambda), mu(mu), params(std::move(params)),
      perm_arena(perm_arena) {
    // Pre-allocate parameter gradients on perm_arena so backward never
    // allocates them on the batch arena (which is cleared every iteration).
    for (auto &p : this->params)
        p->grad = tensor_create_like(p->data, perm_arena);
}

void sgd::step(CudaMemArena *arena) {
    for (auto &p : params) {
        if (!p->grad.defined())
            continue;

        if (lambda > 0.0f) {
            Tensor reg = tensor_mul(p->data, lambda, arena);
            tensor_add(p->grad, p->grad, reg);
        }

        if (mu > 0.0f) {
            Tensor &v = velocity[p.impl_.get()];
            if (!v.defined()) {
                v = tensor_create_like(p->data, perm_arena);
                tensor_clear(v);
            }
            // v = mu*v + grad
            tensor_mul(v, v, mu);
            tensor_add(v, v, p->grad);
            // param -= lr * v
            Tensor delta = tensor_mul(v, lr, arena);
            tensor_sub(p->data, p->data, delta);
        } else {
            Tensor delta = tensor_mul(p->grad, lr, arena);
            tensor_sub(p->data, p->data, delta);
        }
    }
}

void sgd::zero_grad() {
    for (auto &p : params) {
        if (p->grad.defined())
            tensor_clear(p->grad);
    }
}

// ── MultiStepLR ───────────────────────────────────────────────────────────────

MultiStepLR::MultiStepLR(sgd &optimizer, std::vector<int> milestones, f32 gamma)
    : optimizer(optimizer), milestones(std::move(milestones)), gamma(gamma),
      base_lr(optimizer.lr) {
    std::sort(this->milestones.begin(), this->milestones.end());
}

void MultiStepLR::step(int epoch) {
    for (int m : milestones) {
        if (epoch == m) {
            f32 old_lr = optimizer.lr;
            f32 new_lr = old_lr * gamma;
            optimizer.set_lr(new_lr);
            printf("MultiStepLR: epoch %d — lr %.2e → %.2e\n", epoch + 1, old_lr,
                   new_lr);
            break;
        }
    }
}

// ── ReduceLROnPlateau ─────────────────────────────────────────────────────────

ReduceLROnPlateau::ReduceLROnPlateau(sgd &optimizer, f32 factor, int patience,
                                     f32 min_lr, f32 min_delta)
    : optimizer(optimizer), factor(factor), patience(patience), min_lr(min_lr),
      min_delta(min_delta) {}

void ReduceLROnPlateau::step(f32 loss, int epoch) {
    if (best_loss - loss > min_delta) {
        best_loss = loss;
        no_improve = 0;
    } else {
        no_improve++;
    }
    if (no_improve >= patience && optimizer.lr > min_lr) {
        f32 old_lr = optimizer.lr;
        f32 new_lr = std::max(old_lr * factor, min_lr);
        optimizer.set_lr(new_lr);
        no_improve = 0;
        printf("ReduceLROnPlateau: epoch %d — lr %.2e → %.2e\n", epoch + 1, old_lr,
               new_lr);
    }
}

// ── EarlyStopping ─────────────────────────────────────────────────────────────

EarlyStopping::EarlyStopping(int patience, f32 min_delta)
    : patience(patience), min_delta(min_delta) {}

bool EarlyStopping::step(f32 loss, int epoch) {
    if (best_loss - loss > min_delta) {
        best_loss = loss;
        no_improve = 0;
    } else {
        no_improve++;
    }
    if (no_improve >= patience) {
        printf("EarlyStopping: no improvement for %d epochs, stopping at epoch %d\n",
               patience, epoch + 1);
        return true;
    }
    return false;
}

