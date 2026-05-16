#include "../include/optimizers.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

// ── DataLoader
// ────────────────────────────────────────────────────────────────

DataLoader::DataLoader(Tensor<f32> X, Tensor<f32> y, u32 batch_size)
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

// Build a broadcast index tensor that selects `n` rows from a tensor with the
// given ndim/shape. shape[0] = n, remaining dims broadcast via stride=0.
// The storage holds only n u32 elements; it is placed on the same device as the
// source so gather's same-device check passes.
static TensorU32 gather_row_idx(const u32 *row_ids, u32 n, u32 ndim,
                                const u32 *ref_shape, bool on_gpu) {
    // Always stage the indices on CPU first.
    TensorU32 cpu_idx = TensorU32::make(1, &n, false);
    memcpy(cpu_idx->data(), row_ids, n * sizeof(u32));

    TensorU32 idx;
    if (on_gpu) {
        idx = TensorU32::make(1, &n, true);
        cudaMemcpyAsync(idx->data(), cpu_idx->data(), n * sizeof(u32),
                        cudaMemcpyHostToDevice);
    } else {
        idx = cpu_idx;
    }
    idx->ndim = ndim;
    idx->shape[0] = n;
    idx->stride[0] = 1;
    for (u32 i = 1; i < ndim; i++) {
        idx->shape[i] = ref_shape[i];
        idx->stride[i] = 0;
    }
    return idx;
}

bool DataLoader::next(Tensor<f32> &X_batch, Tensor<f32> &y_batch) {
    if (cursor >= n_samples)
        return false;
    u32 end = std::min(cursor + batch_size, n_samples);
    u32 n = end - cursor;
    const u32 *row_ids = indices.data() + cursor;
    TensorU32 X_idx =
        gather_row_idx(row_ids, n, X->ndim, X->shape, X->on_gpu());
    TensorU32 y_idx =
        gather_row_idx(row_ids, n, y->ndim, y->shape, y->on_gpu());
    X_batch = tensor_gather(X, X_idx, 0);
    y_batch = tensor_gather(y, y_idx, 0);
    cursor = end;
    return true;
}

// ── Optimizer
// ─────────────────────────────────────────────────────────────────

Optimizer::Optimizer(std::vector<Var> params, f32 lr)
    : lr(lr), params(std::move(params)) {
    for (auto &p : this->params)
        p->grad = tensor_zeros_like(p->data);
}

// ── sgd
// ───────────────────────────────────────────────────────────────────────

sgd::sgd(std::vector<Var> params, f32 lr, f32 lambda, f32 mu)
    : Optimizer(std::move(params), lr), lambda(lambda), mu(mu) {}

sgd::sgd(Layer &model, f32 lr, f32 lambda, f32 mu)
    : sgd(model.parameters(), lr, lambda, mu) {}

void sgd::step() {
    for (auto &p : params) {
        if (!p->grad.defined())
            continue;

        if (lambda > 0.0f) {
            Tensor<f32> reg = tensor_mul(p->data, lambda);
            tensor_add(p->grad, p->grad, reg);
        }

        if (mu > 0.0f) {
            Tensor<f32> &v = velocity[p.impl_.get()];
            if (!v.defined()) {
                v = tensor_create_like(p->data);
                tensor_clear(v);
            }
            // v = mu*v + grad
            tensor_mul(v, v, mu);
            tensor_add(v, v, p->grad);
            // param -= lr * v
            Tensor<f32> delta = tensor_mul(v, lr);
            tensor_sub(p->data, p->data, delta);
        } else {
            Tensor<f32> delta = tensor_mul(p->grad, lr);
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

// ── AdamW
// ─────────────────────────────────────────────────────────────────────

AdamW::AdamW(Layer &model, f32 lr, f32 beta1, f32 beta2, f32 eps, f32 lambda)
    : AdamW(model.parameters(), lr, beta1, beta2, eps, lambda) {}

AdamW::AdamW(std::vector<Var> params, f32 lr, f32 beta1, f32 beta2, f32 eps,
             f32 lambda)
    : Optimizer(std::move(params), lr), beta1(beta1), beta2(beta2),
      eps(eps), lambda(lambda), t(0) {
    for (auto &p : this->params) {
        Tensor<f32> mt = tensor_create_like(p->data);
        tensor_clear(mt);
        Tensor<f32> vt = tensor_create_like(p->data);
        tensor_clear(vt);
        m[p.impl_.get()] = mt;
        v[p.impl_.get()] = vt;
    }
}

void AdamW::step() {
    t++;
    f32 bc1 = 1.0f - powf(beta1, (f32)t);
    f32 bc2 = 1.0f - powf(beta2, (f32)t);
    f32 step_size = lr * sqrtf(bc2) / bc1;

    for (auto &p : params) {
        if (!p->grad.defined())
            continue;

        Tensor<f32> &mt = m[p.impl_.get()];
        Tensor<f32> &vt = v[p.impl_.get()];

        // mt = beta1*mt + (1-beta1)*grad
        tensor_mul(mt, mt, beta1);
        Tensor<f32> g1 = tensor_mul(p->grad, 1.0f - beta1);
        tensor_add(mt, mt, g1);

        // vt = beta2*vt + (1-beta2)*grad^2
        tensor_mul(vt, vt, beta2);
        Tensor<f32> g2 = tensor_mul(p->grad, p->grad);
        Tensor<f32> g2s = tensor_mul(g2, 1.0f - beta2);
        tensor_add(vt, vt, g2s);

        // param -= step_size * mt / (sqrt(vt) + eps)
        Tensor<f32> vt_sqrt = tensor_sqrt(vt);
        Tensor<f32> denom = tensor_add(vt_sqrt, eps);
        Tensor<f32> delta = tensor_div(mt, denom);
        Tensor<f32> scaled = tensor_mul(delta, step_size);
        tensor_sub(p->data, p->data, scaled);

        // decoupled weight decay: param -= lr * lambda * param
        if (lambda > 0.0f) {
            Tensor<f32> wd = tensor_mul(p->data, lr * lambda);
            tensor_sub(p->data, p->data, wd);
        }
    }
}

void AdamW::zero_grad() {
    for (auto &p : params) {
        if (p->grad.defined())
            tensor_clear(p->grad);
    }
}

// ── MultiStepLR
// ───────────────────────────────────────────────────────────────

MultiStepLR::MultiStepLR(Optimizer &optimizer, std::vector<int> milestones,
                         f32 gamma)
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
            printf("MultiStepLR: epoch %d — lr %.2e → %.2e\n", epoch + 1,
                   old_lr, new_lr);
            break;
        }
    }
}

// ── ReduceLROnPlateau
// ─────────────────────────────────────────────────────────

ReduceLROnPlateau::ReduceLROnPlateau(Optimizer &optimizer, f32 factor,
                                     int patience, f32 min_lr, f32 min_delta)
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
        printf("ReduceLROnPlateau: epoch %d — lr %.2e → %.2e\n", epoch + 1,
               old_lr, new_lr);
    }
}

// ── EarlyStopping
// ─────────────────────────────────────────────────────────────

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
        printf("EarlyStopping: no improvement for %d epochs, stopping at epoch "
               "%d\n",
               patience, epoch + 1);
        return true;
    }
    return false;
}
