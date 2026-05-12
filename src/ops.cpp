#include "../include/ops.hpp"
#include <cstring>

// Sums grad over every dimension where target was broadcast (size == 1).
static Tensor<f32> reduce_grad(const Tensor<f32> &grad,
                               const Tensor<f32> &target, CudaMemArena *arena) {
    Tensor<f32> cur = tensor_view(grad);

    u32 target_expanded[MAX_NDIM];
    expanded_shape(target.impl(), grad->ndim, target_expanded);

    for (u32 i = 0; i < grad->ndim; i++) {
        if (target_expanded[i] == 1 && cur->shape[i] > 1)
            cur = tensor_sum(cur, i, true, arena);
    }

    tensor_reshape(cur, target->shape, target->ndim, arena);
    return cur;
}

// Reduces [N,C,H,W] → [1,C,1,1] by summing over dims 0, 2, 3.
// Sum over all dims except dim 1 (the channel/feature dim), keeping dims.
static Tensor<f32> reduce_all_except_c(const Tensor<f32> &t,
                                       CudaMemArena *arena) {
    Tensor<f32> r = tensor_view(t);
    for (i32 d = (i32)r->ndim - 1; d >= 0; d--) {
        if ((u32)d == 1)
            continue;
        r = tensor_sum(r, (u32)d, true, arena);
    }
    return r;
}

// ── mat_mul ──────────────────────────────────────────────────────────────────

Var mat_mul(Var a, Var b, CudaMemArena *arena) {
    Var out(tensor_mat_mul(a->data, b->data, arena));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor<f32> saved_a, saved_b;
        void backward(Tensor<f32> grad) override {
            u32 nd = grad->ndim;
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> bt = tensor_view(saved_b);
                tensor_transpose(bt, nd - 2, nd - 1);
                Tensor<f32> dA = tensor_mat_mul(grad, bt, arena);
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> at = tensor_view(saved_a);
                tensor_transpose(at, nd - 2, nd - 1);
                Tensor<f32> dB = tensor_mat_mul(at, grad, arena);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
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

// ── reshape ───────────────────────────────────────────────────────────────────

Var reshape(Var a, const u32 *shape, u32 ndim, CudaMemArena *arena) {
    // tensor_reshape calls tensor_contiguous_impl internally, which writes back
    // to the tensor's own storage — unsafe on a shared view when non-contiguous.
    // Safe path: share storage when already contiguous (contiguous_impl is no-op).
    // Unsafe path: copy to fresh storage first so the source is never touched.
    Tensor<f32> out_data;
    if (tensor_is_contiguous(a->data)) {
        out_data = tensor_view(a->data);
    } else {
        out_data = Tensor<f32>::make(a->data->ndim, a->data->shape,
                                    a->data->on_gpu(), arena);
        tensor_copy(out_data, a->data);
    }
    tensor_reshape(out_data, shape, ndim, arena);

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        u32 orig_shape[MAX_NDIM];
        u32 orig_ndim;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            Tensor<f32> dA;
            if (tensor_is_contiguous(grad)) {
                dA = tensor_view(grad);
            } else {
                dA = Tensor<f32>::make(grad->ndim, grad->shape,
                                      grad->on_gpu(), arena);
                tensor_copy(dA, grad);
            }
            tensor_reshape(dA, orig_shape, orig_ndim, arena);
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->orig_ndim = a->data->ndim;
    memcpy(fn->orig_shape, a->data->shape, a->data->ndim * sizeof(u32));
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── transpose ─────────────────────────────────────────────────────────────────

Var transpose(Var a, u32 d0, u32 d1, CudaMemArena *arena) {
    // Pure metadata swap — never touches data.
    Tensor<f32> out_data = tensor_view(a->data);
    tensor_transpose(out_data, d0, d1);

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        u32 d0, d1;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            // Swap the same dims back — gives a non-contiguous view of grad
            // with the original strides. tensor_add handles non-contiguous via
            // its stride-based iterator path.
            Tensor<f32> dA = tensor_view(grad);
            tensor_transpose(dA, d0, d1);
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->d0 = d0;
    fn->d1 = d1;
    fn->inputs = {a};
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
        void backward(Tensor<f32> grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> dA = reduce_grad(grad, inputs[0]->data, arena);
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> dB = reduce_grad(grad, inputs[1]->data, arena);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
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
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            Tensor<f32> dA = tensor_relu_backward(grad, inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── gelu ─────────────────────────────────────────────────────────────────────

Var gelu(Var a, CudaMemArena *arena) {
    Var out(tensor_gelu(a->data, arena));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            Tensor<f32> dA = tensor_gelu_backward(grad, inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── mul (scalar) ─────────────────────────────────────────────────────────────

Var mul(Var a, f32 scalar, CudaMemArena *arena) {
    Var out(tensor_mul(a->data, scalar, arena));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        f32 scalar;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            Tensor<f32> dA = tensor_mul(grad, scalar, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->scalar = scalar;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── softmax ───────────────────────────────────────────────────────────────────

Var softmax(Var a, i32 dim, CudaMemArena *arena) {
    Tensor<f32> s = tensor_softmax(a->data, dim, arena);
    Var out(s);

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor<f32> saved_s;
        u32 axis;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            // dx = s * (dy - sum(s*dy, axis, keepdim=true))
            Tensor<f32> sdg = tensor_mul(saved_s, grad, arena);
            Tensor<f32> dot = tensor_sum(sdg, axis, true, arena);
            Tensor<f32> dA = tensor_sub(grad, dot, arena);
            tensor_mul(dA, dA, saved_s);
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->saved_s = s;
    fn->axis = (dim < 0) ? (u32)((i32)s->ndim + dim) : (u32)dim;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── dropout ──────────────────────────────────────────────────────────────────

Var dropout(Var a, f32 p, bool training, CudaMemArena *arena) {
    if (!training || p == 0.0f)
        return a;

    Tensor<f32> mask = tensor_create_like(a->data, arena);
    tensor_dropout_mask(mask, p);
    Var out(tensor_mul(a->data, mask, arena));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor<f32> mask;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            Tensor<f32> dA = tensor_mul(grad, mask, arena);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->arena = arena;
    fn->mask = mask;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── conv2d ───────────────────────────────────────────────────────────────────

Var conv2d(Var input, Var weight, Unfold2dParams params, CudaMemArena *arena) {
    const Tensor<f32> &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C_out = weight->data->shape[1];
    u32 L = params.L_h * params.L_w;

    // [N, C, H, W] → [N*L, C*kH*kW]
    Tensor<f32> col = tensor_unfold2d(inp, params, arena);
    u32 col2[2] = {N * L, (u32)(col->numel() / (N * L))};
    tensor_reshape(col, col2, 2, arena);

    // [N*L, C*kH*kW] @ [C*kH*kW, C_out] → [N*L, C_out]
    Tensor<f32> res = tensor_mat_mul(col, weight->data, arena);

    // [N*L, C_out] → [N, L_h, L_w, C_out] → [N, C_out, L_h, L_w]
    u32 res4[4] = {N, params.L_h, params.L_w, C_out};
    tensor_reshape(res, res4, 4, arena);
    tensor_transpose(res, 1, 3);
    tensor_transpose(res, 2, 3);
    tensor_contiguous(res, arena);

    Var out(res);

    if (!((input->flags | weight->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Unfold2dParams params;
        Tensor<f32> saved_col;
        u32 N, C_out, L;
        void backward(Tensor<f32> grad) override {
            // Reshape grad [N, C_out, L_h, L_w] → [N*L, C_out]
            Tensor<f32> g = tensor_view(grad);
            u32 s3[3] = {N, C_out, L};
            tensor_reshape(g, s3, 3, arena);
            tensor_transpose(g, 1, 2);
            tensor_contiguous(g, arena);
            u32 s2[2] = {N * L, C_out};
            tensor_reshape(g, s2, 2, arena);

            // dInput = g @ weight^T → fold2d → [N, C, H, W]
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                Tensor<f32> wt = tensor_view(inputs[1]->data);
                tensor_transpose(wt, 0, 1);
                Tensor<f32> col_grad = tensor_mat_mul(g, wt, arena);
                tensor_fold2d(inputs[0]->grad, col_grad, params);
            }

            // dWeight = col^T @ g → [C*kH*kW, C_out]
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
                Tensor<f32> ct = tensor_view(saved_col);
                tensor_transpose(ct, 0, 1);
                Tensor<f32> dW = tensor_mat_mul(ct, g, arena);
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
    const Tensor<f32> &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C = inp->shape[1];
    u32 L = params.L_h * params.L_w;
    u32 K = params.k_h * params.k_w;

    // [N, C, H, W] → [N, L, C*K]
    Tensor<f32> col = tensor_unfold2d(inp, params, arena);

    // [N, L, C*K] → [N, L, C, K]
    u32 s4[4] = {N, L, C, K};
    tensor_reshape(col, s4, 4, arena);

    // argmax along K [N, L, C, 1] — saved for backward
    TensorU32 max_idx = tensor_argmax(col, 3, true, arena);

    // max along K → [N, L, C]
    Tensor<f32> pooled = tensor_max(col, 3, false, arena);

    // [N, L, C] → [N, L_h, L_w, C] → [N, C, L_h, L_w]
    u32 s_nlhwc[4] = {N, params.L_h, params.L_w, C};
    tensor_reshape(pooled, s_nlhwc, 4, arena);
    tensor_transpose(pooled, 1, 3);
    tensor_transpose(pooled, 2, 3);
    tensor_contiguous(pooled, arena);

    Var out(pooled);

    if (!(input->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Unfold2dParams params;
        TensorU32 saved_max_idx;
        u32 N, C, L, K;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);

            // Reverse forward transpose: [N, C, L_h, L_w] → [N, L, C, 1]
            Tensor<f32> g = tensor_view(grad);
            tensor_transpose(g, 2, 3);
            tensor_transpose(g, 1, 3);
            tensor_contiguous(g, arena);
            u32 s_nlc1[4] = {N, L, C, 1};
            tensor_reshape(g, s_nlc1, 4, arena);

            // Route grads to max positions → [N, L, C, K]
            Tensor<f32> scattered =
                tensor_scatter_add(g, saved_max_idx, 3, K, arena);

            // [N, L, C, K] → [N, L, C*K]
            u32 s3[3] = {N, L, C * K};
            tensor_reshape(scattered, s3, 3, arena);

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

// ── batch_norm ───────────────────────────────────────────────────────────────

Var batch_norm(Var input, Var gamma, Var beta, Tensor<f32> running_mean,
               Tensor<f32> running_var, bool training, f32 momentum, f32 eps,
               CudaMemArena *arena) {
    const Tensor<f32> &inp = input->data;
    bool on_gpu = inp->on_gpu();
    u32 ndim = inp->ndim;
    u32 C = inp->shape[1];

    // Broadcast shape: [1, C, 1, ..., 1] matching inp->ndim
    u32 bcast_shape[MAX_NDIM];
    for (u32 d = 0; d < ndim; d++)
        bcast_shape[d] = (d == 1) ? C : 1;

    Tensor<f32> mean, var, xhat, out_data;

    if (training) {
        // Compute per-channel mean and raw M2 (sum of squared deviations)
        mean = Tensor<f32>::make(ndim, bcast_shape, on_gpu, arena);
        Tensor<f32> m2 = Tensor<f32>::make(ndim, bcast_shape, on_gpu, arena);
        tensor_welford_mean_var(mean, m2, inp, 1);

        f32 count = (f32)inp->numel() / (f32)C;

        // Biased var saved for backward; unbiased var for running stats EMA
        var = tensor_div(m2, count, arena);
        Tensor<f32> unbiased_var = tensor_div(m2, count - 1.0f, arena);

        // Update running stats: running = (1-momentum)*running + momentum*batch
        tensor_mul(running_mean, running_mean, 1.0f - momentum);
        tensor_add(running_mean, running_mean,
                   tensor_mul(mean, momentum, arena));
        tensor_mul(running_var, running_var, 1.0f - momentum);
        tensor_add(running_var, running_var,
                   tensor_mul(unbiased_var, momentum, arena));

        xhat = Tensor<f32>::make(ndim, inp->shape, on_gpu, arena);
        out_data = Tensor<f32>::make(ndim, inp->shape, on_gpu, arena);
        tensor_bn_fwd_normalize(out_data, xhat, inp, mean, m2, gamma->data,
                                beta->data, count, eps);
    } else {
        // Eval: normalize with running stats, no grad
        xhat = tensor_sub(inp, running_mean, arena);
        mean = running_mean;
        var = running_var;
        Tensor<f32> denom = tensor_add(var, eps, arena);
        tensor_sqrt(denom, denom);
        tensor_div(xhat, xhat, denom);
        Tensor<f32> gv = tensor_view(gamma->data);
        tensor_reshape(gv, bcast_shape, ndim, arena);
        Tensor<f32> bv = tensor_view(beta->data);
        tensor_reshape(bv, bcast_shape, ndim, arena);
        out_data = tensor_mul(xhat, gv, arena);
        tensor_add(out_data, out_data, bv);
    }

    Var out(out_data);

    if (!training ||
        !((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        f32 eps;
        Tensor<f32> saved_mean, saved_var, saved_xhat;
        u32 C;
        void backward(Tensor<f32> grad) override {
            const Tensor<f32> &inp = inputs[0]->data;
            f32 m = (f32)inp->numel() / (f32)C;

            auto ensure_grad = [&](Var &v) {
                if ((v->flags & FV_FLAG_REQUIERES_GRAD) && !v->grad.defined())
                    v->grad = tensor_zeros_like(v->data, arena);
            };
            ensure_grad(inputs[0]);
            ensure_grad(inputs[1]);
            ensure_grad(inputs[2]);

            Tensor<f32> dx_tmp, dgamma_tmp, dbeta_tmp;
            auto grad_or_tmp = [&](Var &v, Tensor<f32> &tmp) -> Tensor<f32> & {
                if (v->flags & FV_FLAG_REQUIERES_GRAD)
                    return v->grad;
                tmp = tensor_create_like(v->data, arena);
                return tmp;
            };
            Tensor<f32> &dx = grad_or_tmp(inputs[0], dx_tmp);
            Tensor<f32> &dgamma = grad_or_tmp(inputs[1], dgamma_tmp);
            Tensor<f32> &dbeta = grad_or_tmp(inputs[2], dbeta_tmp);

            tensor_bn_bwd(dx, dgamma, dbeta, grad, saved_xhat, inputs[1]->data,
                          saved_var, m, eps);
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

// ── embedding ────────────────────────────────────────────────────────────────

Var embedding(Var weight, TensorU32 indices, CudaMemArena *arena) {
    Var out(tensor_index_select(weight->data, indices, 0, arena));

    if (!(weight->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        TensorU32 indices;
        u32 vocab_size;
        CudaMemArena *arena;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
            // indices is [N] but scatter_add needs it to match grad's shape [N,
            // D, ...]. Unsqueeze trailing dims with stride=0 (broadcast, no
            // data copy).
            TensorU32 idx_view = tensor_view<u32>(indices);
            while (idx_view->ndim < grad->ndim) {
                tensor_unsqueeze(idx_view.impl(), idx_view->ndim);
                idx_view->shape[idx_view->ndim - 1] =
                    grad->shape[idx_view->ndim - 1];
            }
            tensor_scatter_add(inputs[0]->grad, grad, idx_view, 0);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->indices = indices;
    fn->vocab_size = weight->data->shape[0];
    fn->arena = arena;
    fn->inputs = {weight};
    out->grad_fn = fn;
    return out;
}

// ── layer_norm ───────────────────────────────────────────────────────────────

Var layer_norm(Var input, Var gamma, Var beta, f32 eps, CudaMemArena *arena) {
    const Tensor<f32> &inp = input->data;
    u32 last_dim = inp->ndim - 1;
    u32 D        = inp->shape[last_dim];
    bool on_gpu  = inp->on_gpu();

    // mean and m2 over last dim, shape [..., 1] (keep_dim)
    u32 stat_shape[MAX_NDIM];
    memcpy(stat_shape, inp->shape, inp->ndim * sizeof(u32));
    stat_shape[last_dim] = 1;
    Tensor<f32> mean = Tensor<f32>::make(inp->ndim, stat_shape, on_gpu, arena);
    Tensor<f32> m2   = Tensor<f32>::make(inp->ndim, stat_shape, on_gpu, arena);
    tensor_welford_mean_var(mean, m2, inp, last_dim);

    // var = m2 / D,  std = sqrt(var + eps)
    Tensor<f32> var = tensor_div(m2, (f32)D, arena);
    Tensor<f32> std = tensor_add(var, eps, arena);
    tensor_sqrt(std, std);

    // xhat = (x - mean) / std  — mean/std broadcast over last dim via stride=0
    Tensor<f32> xhat = tensor_sub(inp, mean, arena);
    tensor_div(xhat, xhat, std);

    // out = gamma * xhat + beta  — gamma/beta [D] broadcast to [..., D]
    Tensor<f32> gv = tensor_view(gamma->data);
    tensor_expand_shape(gv.impl(), inp->ndim);
    Tensor<f32> bv = tensor_view(beta->data);
    tensor_expand_shape(bv.impl(), inp->ndim);
    Tensor<f32> out_data = tensor_mul(xhat, gv, arena);
    tensor_add(out_data, out_data, bv);

    Var out(out_data);
    if (!((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        f32 eps;
        u32 D;
        Tensor<f32> saved_xhat, saved_std;

        void backward(Tensor<f32> grad) override {
            u32 last_dim = grad->ndim - 1;

            // grad_n = grad * gamma  [... , D]
            Tensor<f32> gv = tensor_view(inputs[1]->data);
            tensor_expand_shape(gv.impl(), grad->ndim);
            Tensor<f32> grad_n = tensor_mul(grad, gv, arena);

            // dx = (grad_n - mean(grad_n) - xhat * mean(grad_n * xhat)) / std
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> mean_gn = tensor_sum(grad_n, last_dim, true, arena);
                tensor_div(mean_gn, mean_gn, (f32)D);

                Tensor<f32> gn_x    = tensor_mul(grad_n, saved_xhat, arena);
                Tensor<f32> mean_gnx = tensor_sum(gn_x, last_dim, true, arena);
                tensor_div(mean_gnx, mean_gnx, (f32)D);

                Tensor<f32> dx = tensor_sub(grad_n, mean_gn, arena);
                Tensor<f32> correction = tensor_mul(saved_xhat, mean_gnx, arena);
                tensor_sub(dx, dx, correction);
                tensor_div(dx, dx, saved_std);

                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                tensor_add(inputs[0]->grad, inputs[0]->grad, dx);
            }

            // dgamma = sum(grad * xhat, all dims except last) → [D]
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> g = tensor_mul(grad, saved_xhat, arena);
                for (i32 d = (i32)grad->ndim - 2; d >= 0; d--)
                    g = tensor_sum(g, (u32)d, true, arena);
                tensor_reshape(g, inputs[1]->data->shape, inputs[1]->data->ndim, arena);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
                tensor_add(inputs[1]->grad, inputs[1]->grad, g);
            }

            // dbeta = sum(grad, all dims except last) → [D]
            if (inputs[2]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> g = tensor_view(grad);
                for (i32 d = (i32)grad->ndim - 2; d >= 0; d--)
                    g = tensor_sum(g, (u32)d, true, arena);
                tensor_reshape(g, inputs[2]->data->shape, inputs[2]->data->ndim, arena);
                if (!inputs[2]->grad.defined())
                    inputs[2]->grad = tensor_zeros_like(inputs[2]->data, arena);
                tensor_add(inputs[2]->grad, inputs[2]->grad, g);
            }
        }
    };
    auto fn      = std::make_shared<Fn>();
    fn->arena     = arena;
    fn->eps       = eps;
    fn->D         = D;
    fn->saved_xhat = xhat;
    fn->saved_std  = std;
    fn->inputs    = {input, gamma, beta};
    out->grad_fn  = fn;
    return out;
}

// ── mse_loss ─────────────────────────────────────────────────────────────────

Var mse_loss(Var pred, Var target, CudaMemArena *arena) {
    const Tensor<f32> &p = pred->data;
    const Tensor<f32> &t = target->data;
    u64 N = p->numel();

    Tensor<f32> diff = tensor_sub(p, t, arena);
    tensor_mul(diff, diff, diff);
    Tensor<f32> out_data = tensor_sum(diff, arena);
    tensor_div(out_data, out_data, (f32)N);

    Var out(out_data);

    if (!((pred->flags | target->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        u64 N;
        void backward(Tensor<f32> grad) override {
            f32 scale = 2.0f / (f32)N;

            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                Tensor<f32> d =
                    tensor_sub(inputs[0]->data, inputs[1]->data, arena);
                tensor_mul(d, d, scale);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
                Tensor<f32> d =
                    tensor_sub(inputs[1]->data, inputs[0]->data, arena);
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
    const Tensor<f32> &log_t = logits->data;
    const Tensor<f32> &tar = targets->data;
    u64 N_batch = log_t->ndim >= 2 ? log_t->shape[0] : 1;

    Tensor<f32> softmax = tensor_softmax(log_t, -1, arena);
    Tensor<f32> log_probs = tensor_log_softmax(log_t, -1, arena);
    tensor_mul(log_probs, log_probs, tar);
    Tensor<f32> out_data = tensor_sum(log_probs, arena);
    tensor_div(out_data, out_data, -(f32)N_batch);

    Var out(out_data);

    if (!((logits->flags | targets->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        CudaMemArena *arena;
        Tensor<f32> saved_softmax;
        u64 N_batch;
        void backward(Tensor<f32> grad) override {
            // d_logits = (softmax - targets) / N_batch * grad_scalar
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data, arena);
                Tensor<f32> d =
                    tensor_sub(saved_softmax, inputs[1]->data, arena);
                tensor_div(d, d, (f32)N_batch);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            // d_targets = -log(softmax) / N_batch * grad_scalar
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data, arena);
                Tensor<f32> d = tensor_log(saved_softmax, arena);
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
