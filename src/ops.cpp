#include "../include/ops.hpp"
#include <cstring>

// Sums grad over every dimension where target was broadcast (size == 1).
static Tensor reduce_grad(const Tensor &grad, const Tensor &target,
                          CudaMemArena *arena) {
    Tensor cur = tensor_view(grad);

    u32 target_expanded[MAX_NDIM];
    expanded_shape(target.impl(), grad->ndim, target_expanded);

    for (u32 i = 0; i < grad->ndim; i++) {
        if (target_expanded[i] == 1 && cur->shape[i] > 1)
            cur = tensor_sum(cur, i, true, arena);
    }

    tensor_reshape(cur, target->shape, target->ndim);
    return cur;
}

// Reduces [N,C,H,W] → [1,C,1,1] by summing over dims 0, 2, 3.
static Tensor reduce_nhw(const Tensor &t, CudaMemArena *arena) {
    return tensor_sum(tensor_sum(tensor_sum(t, 3, true, arena), 2, true, arena),
                      0, true, arena);
}

// ── mat_mul ──────────────────────────────────────────────────────────────────

Var mat_mul(Var a, Var b, CudaMemArena *arena) {
    Var out(tensor_mat_mul(a->data, b->data, arena));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor saved_a, saved_b;
        void backward(Tensor grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor bt = tensor_view(saved_b);
                tensor_transpose(bt, 0, 1);
                Tensor dA = tensor_mat_mul(grad, bt, arena);
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
                tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor at = tensor_view(saved_a);
                tensor_transpose(at, 0, 1);
                Tensor dB = tensor_mat_mul(at, grad, arena);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                tensor_add(inputs[1]->grad, inputs[1]->grad, dB);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {a, b};
    fn->saved_a = a->data;
    fn->saved_b = b->data;
    out->grad_fn = fn;
    return out;
}

// ── add ──────────────────────────────────────────────────────────────────────

Var add(Var a, Var b, CudaMemArena *arena) {
    Var out(tensor_add(a->data, b->data, arena));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        void backward(Tensor grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor dA = reduce_grad(grad, inputs[0]->data, arena);
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
                tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor dB = reduce_grad(grad, inputs[1]->data, arena);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                tensor_add(inputs[1]->grad, inputs[1]->grad, dB);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {a, b};
    out->grad_fn = fn;
    return out;
}

// ── relu ─────────────────────────────────────────────────────────────────────

Var relu(Var a, CudaMemArena *arena) {
    Var out(tensor_relu(a->data, arena));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        void backward(Tensor grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
            Tensor dA = tensor_relu_backward(grad, inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── conv2d ───────────────────────────────────────────────────────────────────

Var conv2d(Var input, Var weight, Unfold2dParams params, CudaMemArena *arena) {
    const Tensor &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C_out = weight->data->shape[1];
    u32 L = params.L_h * params.L_w;

    // [N, C, H, W] → [N*L, C*kH*kW]
    Tensor col = tensor_unfold2d(inp, params, arena);
    u32 col2[2] = {N * L, (u32)(col->numel() / (N * L))};
    tensor_reshape(col, col2, 2);

    // [N*L, C*kH*kW] @ [C*kH*kW, C_out] → [N*L, C_out]
    Tensor res = tensor_mat_mul(col, weight->data, arena);

    // [N*L, C_out] → [N, L_h, L_w, C_out] → [N, C_out, L_h, L_w]
    u32 res4[4] = {N, params.L_h, params.L_w, C_out};
    tensor_reshape(res, res4, 4);
    tensor_transpose(res, 1, 3);
    tensor_transpose(res, 2, 3);
    tensor_contiguous(res);

    Var out(res);

    if (!((input->flags | weight->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Unfold2dParams params;
        Tensor saved_col;
        u32 N, C_out, L;
        void backward(Tensor grad) override {
            // Reshape grad [N, C_out, L_h, L_w] → [N*L, C_out]
            Tensor g = tensor_view(grad);
            u32 s3[3] = {N, C_out, L};
            tensor_reshape(g, s3, 3);
            tensor_transpose(g, 1, 2);
            tensor_contiguous(g);
            u32 s2[2] = {N * L, C_out};
            tensor_reshape(g, s2, 2);

            // dInput = g @ weight^T → fold2d → [N, C, H, W]
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
                Tensor wt = tensor_view(inputs[1]->data);
                tensor_transpose(wt, 0, 1);
                Tensor col_grad = tensor_mat_mul(g, wt, arena);
                tensor_fold2d(inputs[0]->grad, col_grad, params);
            }

            // dWeight = col^T @ g → [C*kH*kW, C_out]
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                Tensor ct = tensor_view(saved_col);
                tensor_transpose(ct, 0, 1);
                Tensor dW = tensor_mat_mul(ct, g, arena);
                tensor_add(inputs[1]->grad, inputs[1]->grad, dW);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {input, weight};
    fn->params = params;
    fn->saved_col = col;
    fn->N = N;
    fn->C_out = C_out;
    fn->L = L;
    out->grad_fn = fn;
    return out;
}

// ── max_pool2d
// ────────────────────────────────────────────────────────────────

Var max_pool2d(Var input, Unfold2dParams params, CudaMemArena *arena) {
    const Tensor &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C = inp->shape[1];
    u32 L = params.L_h * params.L_w;
    u32 K = params.k_h * params.k_w;

    // [N, C, H, W] → [N, L, C*K]
    Tensor col = tensor_unfold2d(inp, params, arena);

    // [N, L, C*K] → [N, L, C, K]
    u32 s4[4] = {N, L, C, K};
    tensor_reshape(col, s4, 4);

    // argmax along K [N, L, C, 1] — saved for backward
    Tensor max_idx = tensor_argmax(col, 3, true, arena);

    // max along K → [N, L, C]
    Tensor pooled = tensor_max(col, 3, false, arena);

    // [N, L, C] → [N, L_h, L_w, C] → [N, C, L_h, L_w]
    u32 s_nlhwc[4] = {N, params.L_h, params.L_w, C};
    tensor_reshape(pooled, s_nlhwc, 4);
    tensor_transpose(pooled, 1, 3);
    tensor_transpose(pooled, 2, 3);
    tensor_contiguous(pooled);

    Var out(pooled);

    if (!(input->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Unfold2dParams params;
        Tensor saved_max_idx;
        u32 N, C, L, K;
        void backward(Tensor grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);

            // Reverse forward transpose: [N, C, L_h, L_w] → [N, L, C, 1]
            Tensor g = tensor_view(grad);
            tensor_transpose(g, 2, 3);
            tensor_transpose(g, 1, 3);
            tensor_contiguous(g);
            u32 s_nlc1[4] = {N, L, C, 1};
            tensor_reshape(g, s_nlc1, 4);

            // Route grads to max positions → [N, L, C, K]
            Tensor scattered = tensor_scatter_add(g, saved_max_idx, 3, K, arena);

            // [N, L, C, K] → [N, L, C*K]
            u32 s3[3] = {N, L, C * K};
            tensor_reshape(scattered, s3, 3);

            tensor_fold2d(inputs[0]->grad, scattered, params);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {input};
    fn->params = params;
    fn->saved_max_idx = max_idx;
    fn->N = N;
    fn->C = C;
    fn->L = L;
    fn->K = K;
    out->grad_fn = fn;
    return out;
}

// ── flatten ──────────────────────────────────────────────────────────────────

Var flatten(Var input, CudaMemArena *arena) {
    const Tensor &inp = input->data;
    u32 N = inp->shape[0];
    u32 flat = (u32)(inp->numel() / N);

    // Copy to contiguous buffer so reshape is always valid.
    Tensor out_data = tensor_create_like(inp, arena);
    tensor_copy(out_data, inp);
    u32 flat_shape[2] = {N, flat};
    tensor_reshape(out_data, flat_shape, 2);

    Var out(out_data);

    if (!(input->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        u32 saved_ndim;
        u32 saved_shape[MAX_NDIM];
        void backward(Tensor grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
            Tensor g = tensor_view(grad);
            tensor_reshape(g, saved_shape, saved_ndim);
            tensor_add(inputs[0]->grad, inputs[0]->grad, g);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {input};
    fn->saved_ndim = inp->ndim;
    memcpy(fn->saved_shape, inp->shape, inp->ndim * sizeof(u32));
    out->grad_fn = fn;
    return out;
}

// ── batch_norm ───────────────────────────────────────────────────────────────

Var batch_norm(Var input, Var gamma, Var beta, f32 eps, CudaMemArena *arena) {
    const Tensor &inp = input->data;
    bool on_gpu = inp->on_gpu();
    u32 C = inp->shape[1];
    u32 c4[4] = {1, C, 1, 1};

    // Per-channel mean/var: [1, C, 1, 1]
    Tensor mean = Tensor::make(4, c4, on_gpu, arena);
    Tensor var = Tensor::make(4, c4, on_gpu, arena);
    tensor_welford_mean_var(mean, var, inp, 1);

    // xhat = (inp - mean) / sqrt(var + eps)
    Tensor xhat = tensor_sub(inp, mean, arena);
    Tensor denom = tensor_add(var, eps, arena);
    tensor_sqrt(denom, denom);
    tensor_div(xhat, xhat, denom);

    // y = gamma * xhat + beta  (gamma/beta [C] → [1,C,1,1])
    Tensor gv = tensor_view(gamma->data);
    tensor_reshape(gv, c4, 4);
    Tensor bv = tensor_view(beta->data);
    tensor_reshape(bv, c4, 4);
    Tensor out_data = tensor_mul(xhat, gv, arena);
    tensor_add(out_data, out_data, bv);

    Var out(out_data);

    if (!((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        f32 eps;
        Tensor saved_mean, saved_var, saved_xhat;
        u32 C;
        void backward(Tensor grad) override {
            const Tensor &inp = inputs[0]->data;
            u32 N = inp->shape[0], H = inp->shape[2], W = inp->shape[3];
            f32 m = (f32)(N * H * W);
            u32 c4[4] = {1, C, 1, 1};
            u32 c_shape[1] = {C};

            Tensor std_dev = tensor_sqrt(tensor_add(saved_var, eps, arena), arena);

            // d_xhat = grad * gamma  [N,C,H,W]
            Tensor gv = tensor_view(inputs[1]->data);
            tensor_reshape(gv, c4, 4);
            Tensor d_xhat = tensor_mul(grad, gv, arena);

            // d_beta = sum(grad, N,H,W) → [C]
            if (inputs[2]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[2]->grad.defined())
                    inputs[2]->grad = tensor_create_like(inputs[2]->data, arena);
                Tensor db = reduce_nhw(grad, arena);
                tensor_reshape(db, c_shape, 1);
                tensor_add(inputs[2]->grad, inputs[2]->grad, db);
            }

            // d_gamma = sum(grad * xhat, N,H,W) → [C]
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                Tensor dg = reduce_nhw(tensor_mul(grad, saved_xhat, arena), arena);
                tensor_reshape(dg, c_shape, 1);
                tensor_add(inputs[1]->grad, inputs[1]->grad, dg);
            }

            // d_x = (1/std) * (d_xhat - (1/m)*sum(d_xhat)
            //                         - xhat*(1/m)*sum(d_xhat*xhat))
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);

                Tensor mean_dxhat = reduce_nhw(d_xhat, arena);
                tensor_div(mean_dxhat, mean_dxhat, m);

                Tensor mean_dxhat_x =
                    reduce_nhw(tensor_mul(d_xhat, saved_xhat, arena), arena);
                tensor_div(mean_dxhat_x, mean_dxhat_x, m);
                Tensor xhat_term = tensor_mul(saved_xhat, mean_dxhat_x, arena);

                Tensor dx = tensor_sub(d_xhat, mean_dxhat, arena);
                tensor_sub(dx, dx, xhat_term);
                tensor_div(dx, dx, std_dev);

                tensor_add(inputs[0]->grad, inputs[0]->grad, dx);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {input, gamma, beta};
    fn->eps = eps;
    fn->saved_mean = mean;
    fn->saved_var = var;
    fn->saved_xhat = xhat;
    fn->C = C;
    out->grad_fn = fn;
    return out;
}

// ── mse_loss ─────────────────────────────────────────────────────────────────

Var mse_loss(Var pred, Var target, CudaMemArena *arena) {
    const Tensor &p = pred->data;
    const Tensor &t = target->data;
    u64 N = p->numel();

    Tensor diff = tensor_sub(p, t, arena);
    tensor_mul(diff, diff, diff);
    Tensor out_data = tensor_sum(diff, arena);
    tensor_div(out_data, out_data, (f32)N);

    Var out(out_data);

    if (!((pred->flags | target->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        u64 N;
        void backward(Tensor grad) override {
            f32 scale = 2.0f / (f32)N;

            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
                Tensor d = tensor_sub(inputs[0]->data, inputs[1]->data, arena);
                tensor_mul(d, d, scale);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                Tensor d = tensor_sub(inputs[1]->data, inputs[0]->data, arena);
                tensor_mul(d, d, scale);
                tensor_mul(d, d, grad);
                tensor_add(inputs[1]->grad, inputs[1]->grad, d);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {pred, target};
    fn->N = N;
    out->grad_fn = fn;
    return out;
}

// ── cross_entropy_with_logits
// ─────────────────────────────────────────────────

Var cross_entropy_with_logits(Var logits, Var targets, CudaMemArena *arena) {
    const Tensor &log_t = logits->data;
    const Tensor &tar = targets->data;
    u64 N_batch = log_t->ndim >= 2 ? log_t->shape[0] : 1;

    Tensor softmax = tensor_softmax(log_t, arena);
    Tensor log_probs = tensor_log_softmax(log_t, arena);
    tensor_mul(log_probs, log_probs, tar);
    Tensor out_data = tensor_sum(log_probs, arena);
    tensor_div(out_data, out_data, -(f32)N_batch);

    Var out(out_data);

    if (!((logits->flags | targets->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor saved_softmax;
        u64 N_batch;
        void backward(Tensor grad) override {
            // d_logits = (softmax - targets) / N_batch * grad_scalar
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_create_like(inputs[0]->data, arena);
                Tensor d = tensor_sub(saved_softmax, inputs[1]->data, arena);
                tensor_div(d, d, (f32)N_batch);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            // d_targets = -log(softmax) / N_batch * grad_scalar
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_create_like(inputs[1]->data, arena);
                Tensor d = tensor_log(saved_softmax, arena);
                tensor_div(d, d, -(f32)N_batch);
                tensor_mul(d, d, grad);
                tensor_add(inputs[1]->grad, inputs[1]->grad, d);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {logits, targets};
    fn->saved_softmax = softmax;
    fn->N_batch = N_batch;
    out->grad_fn = fn;
    return out;
}
