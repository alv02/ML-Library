#include "../include/ops.hpp"
#include <cstring>

// ── mat_mul ──────────────────────────────────────────────────────────────────

Var mat_mul(Var a, Var b) {
    if (!a || !b) { printf("%s: null input\n", __func__); return Var{}; }
    if (!tensor_matmul_compat(a->data, b->data)) {
        printf("mat_mul: incompatible shapes\n");
        return Var{};
    }
    Var out(tensor_mat_mul(a->data, b->data));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> saved_a, saved_b;
        void backward(Tensor<f32> grad) override {
            Var a = inputs[0], b = inputs[1];
            if (a->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> bt = tensor_view(saved_b);
                tensor_transpose(bt, saved_b->ndim - 2, saved_b->ndim - 1);
                Tensor<f32> dA = tensor_mat_mul(grad, bt);
                if (!a->grad.defined())
                    a->grad = tensor_zeros_like(a->data);
                tensor_add(a->grad, a->grad, dA);
            }
            if (b->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> at = tensor_view(saved_a);
                tensor_transpose(at, saved_a->ndim - 2, saved_a->ndim - 1);
                Tensor<f32> dB = tensor_mat_mul(at, grad);
                if (!b->grad.defined())
                    b->grad = tensor_zeros_like(b->data);
                tensor_add(b->grad, b->grad, dB);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {a, b};
    fn->saved_a = a->data;
    fn->saved_b = b->data;
    out->grad_fn = fn;
    return out;
}

// ── reshape ──────────────────────────────────────────────────────────────────

Var reshape(Var a, const u32 *shape, u32 ndim) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Tensor<f32> out_data;
    if (tensor_is_contiguous(a->data)) {
        out_data = tensor_view(a->data);
    } else {
        out_data = Tensor<f32>::make(a->data->ndim, a->data->shape, a->data->on_gpu());
        tensor_copy(out_data, a->data);
    }
    tensor_reshape(out_data, shape, ndim);

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;
    struct Fn : Function {
        u32 orig_shape[MAX_NDIM];
        u32 orig_ndim;
        void backward(Tensor<f32> grad) override {
            Var a = inputs[0];
            if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            Tensor<f32> dA;
            if (tensor_is_contiguous(grad)) {
                dA = tensor_view(grad);
            } else {
                dA = Tensor<f32>::make(grad->ndim, grad->shape, grad->on_gpu());
                tensor_copy(dA, grad);
            }
            tensor_reshape(dA, orig_shape, orig_ndim);
            if (!a->grad.defined())
                a->grad = tensor_zeros_like(inputs[0]->data);
            tensor_add(a->grad, a->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->orig_ndim = a->data->ndim;
    memcpy(fn->orig_shape, a->data->shape, a->data->ndim * sizeof(u32));
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── transpose ────────────────────────────────────────────────────────────────

Var transpose(Var a, u32 d0, u32 d1) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Tensor<f32> out_data = tensor_view(a->data);
    tensor_transpose(out_data, d0, d1);

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;
    struct Fn : Function {
        u32 d0, d1;
        void backward(Tensor<f32> grad) override {
            Var a = inputs[0];
            if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            Tensor<f32> dA = tensor_view(grad);
            tensor_transpose(dA, d0, d1);
            if (!a->grad.defined())
                a->grad = tensor_zeros_like(a->data);
            tensor_add(a->grad, a->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->d0 = d0;
    fn->d1 = d1;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── unsqueeze ────────────────────────────────────────────────────────────────

Var unsqueeze(Var a, u32 dim) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Tensor<f32> out_data = tensor_view(a->data);
    tensor_unsqueeze(out_data.impl(), dim);

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;
    struct Fn : Function {
        u32 dim;
        void backward(Tensor<f32> grad) override {
            Var a = inputs[0];
            if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            Tensor<f32> dA = tensor_view(grad);
            tensor_squeeze(dA.impl(), dim);
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->dim = dim;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── broadcast_to ─────────────────────────────────────────────────────────────

Var broadcast_to(Var a, const u32 *target_shape, u32 target_ndim) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    if (a->data->ndim > target_ndim) {
        printf("broadcast_to: target_ndim < input ndim\n");
        return Var{};
    }
    Tensor<f32> out_data = tensor_view(a->data);
    if (!tensor_broadcast_to(out_data.impl(), target_shape, target_ndim)) {
        tensor_print(a->data.impl());
        printf("broadcast_to: incompatible shapes\n");
        return Var{};
    }

    Var out(out_data);
    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;
    struct Fn : Function {
        u32 orig_ndim;
        u32 orig_shape[MAX_NDIM];
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            u32 ndim_diff = grad->ndim - orig_ndim;
            Tensor<f32> dA = tensor_view(grad);
            for (u32 i = 0; i < grad->ndim; i++) {
                bool is_new = i < ndim_diff;
                bool is_bcast = !is_new && orig_shape[i - ndim_diff] == 1;
                if (is_new || is_bcast)
                    dA = tensor_sum(dA, i, true);
            }
            tensor_reshape(dA, orig_shape, orig_ndim);
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->orig_ndim = a->data->ndim;
    memcpy(fn->orig_shape, a->data->shape, a->data->ndim * sizeof(u32));
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── gather (differentiable) ──────────────────────────────────────────────────

Var gather(Var src, TensorU32 indices, u32 dim) {
    if (!src) { printf("%s: null input\n", __func__); return Var{}; }
    Var out(::tensor_gather(src->data, indices, dim));

    if (!(src->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        TensorU32 indices;
        u32 dim;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            tensor_scatter_add(inputs[0]->grad, grad, indices, dim);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->indices = indices;
    fn->dim = dim;
    fn->inputs = {src};
    out->grad_fn = fn;
    return out;
}

// ── add ──────────────────────────────────────────────────────────────────────

Var add(Var a, Var b) {
    if (!a || !b) { printf("%s: null input\n", __func__); return Var{}; }
    if (!tensor_shape_eq(a->data, b->data)) {
        printf("add: shape mismatch\n");
        return Var{};
    }
    Var out(tensor_add(a->data, b->data));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        void backward(Tensor<f32> grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                tensor_add(inputs[0]->grad, inputs[0]->grad, grad);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                tensor_add(inputs[1]->grad, inputs[1]->grad, grad);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {a, b};
    out->grad_fn = fn;
    return out;
}

// ── relu ─────────────────────────────────────────────────────────────────────

Var relu(Var a) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Var out(tensor_relu(a->data));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            Tensor<f32> dA = tensor_relu_backward(grad, inputs[0]->data);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── gelu ─────────────────────────────────────────────────────────────────────

Var gelu(Var a) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Var out(tensor_gelu(a->data));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            Tensor<f32> dA = tensor_gelu_backward(grad, inputs[0]->data);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── mul (scalar) ─────────────────────────────────────────────────────────────

Var mul(Var a, f32 scalar) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Var out(tensor_mul(a->data, scalar));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        f32 scalar;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            Tensor<f32> dA = tensor_mul(grad, scalar);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->scalar = scalar;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── mul (elementwise Var×Var) ────────────────────────────────────────────────

Var mul(Var a, Var b) {
    if (!a || !b) { printf("%s: null input\n", __func__); return Var{}; }
    if (!tensor_shape_eq(a->data, b->data)) {
        printf("mul: shape mismatch\n");
        return Var{};
    }
    Var out(tensor_mul(a->data, b->data));

    if (!((a->flags | b->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> saved_a, saved_b;
        void backward(Tensor<f32> grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                tensor_add(inputs[0]->grad, inputs[0]->grad, grad);
            }
            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                tensor_add(inputs[1]->grad, inputs[1]->grad, grad);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {a, b};
    fn->saved_a = a->data;
    fn->saved_b = b->data;
    out->grad_fn = fn;
    return out;
}

// ── softmax ───────────────────────────────────────────────────────────────────

Var softmax(Var a, i32 dim) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    Tensor<f32> s = tensor_softmax(a->data, dim);
    Var out(s);

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> saved_s;
        u32 axis;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            tensor_softmax_bwd(inputs[0]->grad, saved_s, grad, (i32)axis);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->saved_s = s;
    fn->axis = (dim < 0) ? (u32)((i32)s->ndim + dim) : (u32)dim;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── dropout ──────────────────────────────────────────────────────────────────

Var dropout(Var a, f32 p, bool training) {
    if (!a) { printf("%s: null input\n", __func__); return Var{}; }
    if (!training || p == 0.0f)
        return a;

    Tensor<f32> mask = tensor_create_like(a->data);
    tensor_dropout_mask(mask, p);
    Var out(tensor_mul(a->data, mask));

    if (!(a->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> mask;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
            Tensor<f32> dA = tensor_mul(grad, mask);
            tensor_add(inputs[0]->grad, inputs[0]->grad, dA);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->mask = mask;
    fn->inputs = {a};
    out->grad_fn = fn;
    return out;
}

// ── conv2d ───────────────────────────────────────────────────────────────────

Var conv2d(Var input, Var weight, Unfold2dParams params) {
    if (!input || !weight) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C_out = weight->data->shape[1];
    u32 L = params.L_h * params.L_w;

    Tensor<f32> col = tensor_unfold2d(inp, params);
    u32 col2[2] = {N * L, (u32)(col->numel() / (N * L))};
    tensor_reshape(col, col2, 2);

    Tensor<f32> res = tensor_mat_mul(col, weight->data);

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
        Unfold2dParams params;
        Tensor<f32> saved_col;
        u32 N, C_out, L;
        void backward(Tensor<f32> grad) override {
            Tensor<f32> g = tensor_view(grad);
            u32 s3[3] = {N, C_out, L};
            tensor_reshape(g, s3, 3);
            tensor_transpose(g, 1, 2);
            tensor_contiguous(g);
            u32 s2[2] = {N * L, C_out};
            tensor_reshape(g, s2, 2);

            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                Tensor<f32> wt = tensor_view(inputs[1]->data);
                tensor_transpose(wt, 0, 1);
                Tensor<f32> col_grad = tensor_mat_mul(g, wt);
                tensor_fold2d(inputs[0]->grad, col_grad, params);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                Tensor<f32> ct = tensor_view(saved_col);
                tensor_transpose(ct, 0, 1);
                Tensor<f32> dW = tensor_mat_mul(ct, g);
                tensor_add(inputs[1]->grad, inputs[1]->grad, dW);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {input, weight};
    fn->params = params;
    fn->saved_col = col;
    fn->N = N;
    fn->C_out = C_out;
    fn->L = L;
    out->grad_fn = fn;
    return out;
}

// ── max_pool2d ────────────────────────────────────────────────────────────────

Var max_pool2d(Var input, Unfold2dParams params) {
    if (!input) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &inp = input->data;
    params.compute_output_size(inp->shape[2], inp->shape[3]);
    u32 N = inp->shape[0];
    u32 C = inp->shape[1];
    u32 L = params.L_h * params.L_w;
    u32 K = params.k_h * params.k_w;

    Tensor<f32> col = tensor_unfold2d(inp, params);

    u32 s4[4] = {N, L, C, K};
    tensor_reshape(col, s4, 4);

    TensorU32 max_idx = tensor_argmax(col, 3, true);
    Tensor<f32> pooled = tensor_max(col, 3, false);

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
        Unfold2dParams params;
        TensorU32 saved_max_idx;
        u32 N, C, L, K;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);

            Tensor<f32> g = tensor_view(grad);
            tensor_transpose(g, 2, 3);
            tensor_transpose(g, 1, 3);
            tensor_contiguous(g);
            u32 s_nlc1[4] = {N, L, C, 1};
            tensor_reshape(g, s_nlc1, 4);

            u32 s_nlck[4] = {N, L, C, K};
            Tensor<f32> scattered =
                tensor_zeros<f32>(4, s_nlck, g->on_gpu());
            tensor_scatter_add(scattered, g, saved_max_idx, 3);

            u32 s3[3] = {N, L, C * K};
            tensor_reshape(scattered, s3, 3);

            tensor_fold2d(inputs[0]->grad, scattered, params);
        }
    };
    auto fn = std::make_shared<Fn>();
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
               Tensor<f32> running_var, bool training, f32 momentum, f32 eps) {
    if (!input || !gamma || !beta) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &inp = input->data;
    bool on_gpu = inp->on_gpu();
    u32 ndim = inp->ndim;
    u32 C = inp->shape[1];

    u32 bcast_shape[MAX_NDIM];
    for (u32 d = 0; d < ndim; d++)
        bcast_shape[d] = (d == 1) ? C : 1;

    Tensor<f32> mean, var, xhat, out_data;

    if (training) {
        mean = Tensor<f32>::make(ndim, bcast_shape, on_gpu);
        Tensor<f32> m2 = Tensor<f32>::make(ndim, bcast_shape, on_gpu);
        tensor_welford_mean_M2_skip(mean, m2, inp, 1);

        u64 count = inp->numel() / (u64)C;

        var = tensor_div(m2, (f32)count);
        Tensor<f32> unbiased_var = tensor_div(m2, count - 1.0f);

        tensor_mul(running_mean, running_mean, 1.0f - momentum);
        tensor_add(running_mean, running_mean, tensor_mul(mean, momentum));
        tensor_mul(running_var, running_var, 1.0f - momentum);
        tensor_add(running_var, running_var, tensor_mul(unbiased_var, momentum));

        xhat = Tensor<f32>::make(ndim, inp->shape, on_gpu);
        out_data = Tensor<f32>::make(ndim, inp->shape, on_gpu);
        tensor_bn_fwd_normalize(out_data, xhat, inp, mean, m2, gamma->data,
                                beta->data, count, eps);
    } else {
        xhat = tensor_sub(inp, running_mean);
        mean = running_mean;
        var = running_var;
        Tensor<f32> denom = tensor_add(var, eps);
        tensor_sqrt(denom, denom);
        tensor_div(xhat, xhat, denom);
        Tensor<f32> gv = tensor_view(gamma->data);
        tensor_reshape(gv, bcast_shape, ndim);
        Tensor<f32> bv = tensor_view(beta->data);
        tensor_reshape(bv, bcast_shape, ndim);
        out_data = tensor_mul(xhat, gv);
        tensor_add(out_data, out_data, bv);
    }

    Var out(out_data);

    if (!training ||
        !((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        f32 eps;
        Tensor<f32> saved_mean, saved_var, saved_xhat;
        u32 C;
        void backward(Tensor<f32> grad) override {
            const Tensor<f32> &inp = inputs[0]->data;
            u64 m = inp->numel() / (u64)C;

            auto ensure_grad = [&](Var &v) {
                if ((v->flags & FV_FLAG_REQUIERES_GRAD) && !v->grad.defined())
                    v->grad = tensor_zeros_like(v->data);
            };
            ensure_grad(inputs[0]);
            ensure_grad(inputs[1]);
            ensure_grad(inputs[2]);

            Tensor<f32> dx_tmp, dgamma_tmp, dbeta_tmp;
            auto grad_or_tmp = [&](Var &v, Tensor<f32> &tmp) -> Tensor<f32> & {
                if (v->flags & FV_FLAG_REQUIERES_GRAD)
                    return v->grad;
                tmp = tensor_create_like(v->data);
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
    fn->inputs = {input, gamma, beta};
    fn->eps = eps;
    fn->saved_mean = mean;
    fn->saved_var = var;
    fn->saved_xhat = xhat;
    fn->C = C;
    out->grad_fn = fn;
    return out;
}

// ── layer_norm ───────────────────────────────────────────────────────────────

Var layer_norm(Var input, Var gamma, Var beta, f32 eps) {
    if (!input || !gamma || !beta) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &inp = input->data;
    u32 last_dim = inp->ndim - 1;
    u32 D = inp->shape[last_dim];
    bool on_gpu = inp->on_gpu();

    u32 stat_shape[MAX_NDIM];
    memcpy(stat_shape, inp->shape, inp->ndim * sizeof(u32));
    stat_shape[last_dim] = 1;

    Tensor<f32> xhat     = tensor_create_like(inp);
    Tensor<f32> inv_std  = Tensor<f32>::make(inp->ndim, stat_shape, on_gpu);
    Tensor<f32> out_data = tensor_create_like(inp);
    tensor_ln_fwd(out_data, xhat, inv_std, inp, gamma->data, beta->data, eps);

    Var out(out_data);
    if (!((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        u32 D;
        Tensor<f32> saved_xhat, saved_inv_std;

        void backward(Tensor<f32> grad) override {
            u32 last_dim = grad->ndim - 1;

            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                tensor_ln_bwd(inputs[0]->grad, grad, saved_xhat, saved_inv_std,
                              inputs[1]->data);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> g = tensor_mul(grad, saved_xhat);
                for (i32 d = (i32)grad->ndim - 2; d >= 0; d--)
                    g = tensor_sum(g, (u32)d, true);
                tensor_reshape(g, inputs[1]->data->shape, inputs[1]->data->ndim);
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                tensor_add(inputs[1]->grad, inputs[1]->grad, g);
            }

            if (inputs[2]->flags & FV_FLAG_REQUIERES_GRAD) {
                Tensor<f32> g = tensor_view(grad);
                for (i32 d = (i32)grad->ndim - 2; d >= 0; d--)
                    g = tensor_sum(g, (u32)d, true);
                tensor_reshape(g, inputs[2]->data->shape, inputs[2]->data->ndim);
                if (!inputs[2]->grad.defined())
                    inputs[2]->grad = tensor_zeros_like(inputs[2]->data);
                tensor_add(inputs[2]->grad, inputs[2]->grad, g);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->D = D;
    fn->saved_xhat = xhat;
    fn->saved_inv_std = inv_std;
    fn->inputs = {input, gamma, beta};
    out->grad_fn = fn;
    return out;
}

// ── mse_loss ─────────────────────────────────────────────────────────────────

Var mse_loss(Var pred, Var target) {
    if (!pred || !target) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &p = pred->data;
    const Tensor<f32> &t = target->data;
    u64 N = p->numel();

    Tensor<f32> diff = tensor_sub(p, t);
    tensor_mul(diff, diff, diff);
    Tensor<f32> out_data = tensor_sum(diff);
    tensor_div(out_data, out_data, (f32)N);

    Var out(out_data);

    if (!((pred->flags | target->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        u64 N;
        void backward(Tensor<f32> grad) override {
            f32 scale = 2.0f / (f32)N;

            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                Tensor<f32> d = tensor_sub(inputs[0]->data, inputs[1]->data);
                tensor_mul(d, d, scale);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                Tensor<f32> d = tensor_sub(inputs[1]->data, inputs[0]->data);
                tensor_mul(d, d, scale);
                tensor_mul(d, d, grad);
                tensor_add(inputs[1]->grad, inputs[1]->grad, d);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {pred, target};
    fn->N = N;
    out->grad_fn = fn;
    return out;
}

// ── cross_entropy_with_logits ─────────────────────────────────────────────────

Var cross_entropy_with_logits(Var logits, Var targets) {
    if (!logits || !targets) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &log_t = logits->data;
    const Tensor<f32> &tar = targets->data;
    u64 N_batch = log_t->ndim >= 2 ? log_t->shape[0] : 1;

    Tensor<f32> softmax = tensor_softmax(log_t, -1);
    Tensor<f32> log_probs = tensor_log_softmax(log_t, -1);
    tensor_mul(log_probs, log_probs, tar);
    Tensor<f32> out_data = tensor_sum(log_probs);
    tensor_div(out_data, out_data, -(f32)N_batch);

    Var out(out_data);

    if (!((logits->flags | targets->flags) & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> saved_softmax;
        u64 N_batch;
        void backward(Tensor<f32> grad) override {
            if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[0]->grad.defined())
                    inputs[0]->grad = tensor_zeros_like(inputs[0]->data);
                Tensor<f32> d = tensor_sub(saved_softmax, inputs[1]->data);
                tensor_div(d, d, (f32)N_batch);
                tensor_mul(d, d, grad);
                tensor_add(inputs[0]->grad, inputs[0]->grad, d);
            }

            if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
                if (!inputs[1]->grad.defined())
                    inputs[1]->grad = tensor_zeros_like(inputs[1]->data);
                Tensor<f32> d = tensor_log(saved_softmax);
                tensor_div(d, d, -(f32)N_batch);
                tensor_mul(d, d, grad);
                tensor_add(inputs[1]->grad, inputs[1]->grad, d);
            }
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {logits, targets};
    fn->saved_softmax = softmax;
    fn->N_batch = N_batch;
    out->grad_fn = fn;
    return out;
}

// ── cross_entropy_with_logits (integer targets) ───────────────────────────────

Var cross_entropy_with_logits(Var logits, TensorU32 targets) {
    if (!logits || !targets) { printf("%s: null input\n", __func__); return Var{}; }
    const Tensor<f32> &log_t = logits->data;
    u64 N_batch = log_t->ndim >= 2 ? log_t->shape[0] : 1;

    Tensor<f32> softmax_t = tensor_softmax(log_t, -1);
    Tensor<f32> log_sm = tensor_log_softmax(log_t, -1);

    // unsqueeze targets [N] → [N, 1] as a view (no allocation)
    TensorU32 idx = tensor_view(targets);
    tensor_unsqueeze(idx.impl(), 1);

    // gathered[i, 0] = log_sm[i, targets[i]]
    Tensor<f32> gathered = tensor_gather(log_sm, idx, 1);

    Tensor<f32> out_data = tensor_sum(gathered);
    tensor_div(out_data, out_data, -(f32)N_batch);

    Var out(out_data);

    if (!(logits->flags & FV_FLAG_REQUIERES_GRAD))
        return out;
    out->flags |= FV_FLAG_REQUIERES_GRAD;

    struct Fn : Function {
        Tensor<f32> saved_softmax;
        TensorU32 targets;
        u64 N_batch;
        void backward(Tensor<f32> grad) override {
            if (!(inputs[0]->flags & FV_FLAG_REQUIERES_GRAD))
                return;
            if (!inputs[0]->grad.defined())
                inputs[0]->grad = tensor_zeros_like(inputs[0]->data);

            // d = softmax / N_batch, then subtract 1/N_batch at target positions
            Tensor<f32> d = tensor_mul(saved_softmax, 1.0f / (f32)N_batch);

            TensorU32 idx = tensor_view(targets);
            tensor_unsqueeze(idx.impl(), 1);
            u32 shape_n1[] = {(u32)N_batch, 1};
            Tensor<f32> correction =
                tensor_zeros<f32>(2, shape_n1, saved_softmax->on_gpu());
            tensor_fill(correction, -1.0f / (f32)N_batch);
            tensor_scatter_add(d, correction, idx, 1);

            // scale by upstream gradient
            tensor_mul(d, d, grad);
            tensor_add(inputs[0]->grad, inputs[0]->grad, d);
        }
    };
    auto fn = std::make_shared<Fn>();
    fn->inputs = {logits};
    fn->saved_softmax = softmax_t;
    fn->targets = targets;
    fn->N_batch = N_batch;
    out->grad_fn = fn;
    return out;
}
