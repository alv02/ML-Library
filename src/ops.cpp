#include "../include/ops.hpp"

// Reduces grad (broadcast output shape) to match target's shape by summing
// over every dimension that was broadcast. Returns a new owned tensor.
Tensor *reduce_grad(const Tensor *grad, const Tensor *target) {
    Tensor *cur = tensor_view(grad);
    Tensor *next = nullptr;
    u32 current_shape[MAX_NDIM];
    memcpy(current_shape, grad->shape, sizeof(u32) * grad->ndim);

    u32 target_expanded_shape[MAX_NDIM];
    expanded_shape(target, grad->ndim, target_expanded_shape);

    for (u32 i = 0; i < grad->ndim; i++) {
        if (target_expanded_shape[i] == 1 && cur->shape[i] > 1) {
            current_shape[i] = 1;
            next = new Tensor(cur->ndim, current_shape, cur->on_gpu);
            tensor_sum(next, cur, i, true, false);
            delete cur;
            cur = next;
        }
    }

    tensor_reshape(cur, target->shape, target->ndim);
    return cur;
}

// ── MatMulOp ─────────────────────────────────────────────────────────────────

function_var *MatMulOp::make_output() {
    const Tensor *a = inputs[0]->val;
    const Tensor *b = inputs[1]->val;
    u32 shape[2] = {a->shape[ROW_DIM(a)], b->shape[COL_DIM(b)]};

    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    function_var *out =
        new function_var(new Tensor(2, shape, inputs[0]->val->on_gpu), flags);
    out->grad_fn = this;
    return out;
}

void MatMulOp::forward(function_var *out) {
    u32 shape[2] = {inputs[0]->val->shape[ROW_DIM(inputs[0]->val)],
                    inputs[1]->val->shape[COL_DIM(inputs[1]->val)]};
    tensor_realloc(out->val, shape, 2);
    tensor_mat_mul(out->val, inputs[0]->val, inputs[1]->val, true);
}

void MatMulOp::backward(Tensor *grad_output) {
    function_var *A = inputs[0];
    function_var *B = inputs[1];

    // dA = grad @ B.T
    if (A->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!A->grad || !tensor_shape_eq(A->grad, A->val)) {
            delete A->grad;
            A->grad = tensor_create_like(A->val);
        }
        Tensor *Bt = tensor_view(B->val);
        tensor_transpose(Bt, ROW_DIM(Bt), COL_DIM(Bt));
        tensor_mat_mul(A->grad, grad_output, Bt, false);
        delete Bt;
    }

    // dB = A.T @ grad
    if (B->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!B->grad || !tensor_shape_eq(B->grad, B->val)) {
            delete B->grad;
            B->grad = tensor_create_like(B->val);
        }
        Tensor *At = tensor_view(A->val);
        tensor_transpose(At, ROW_DIM(At), COL_DIM(At));
        tensor_mat_mul(B->grad, At, grad_output, false);
        delete At;
    }
}

// ── AddOp ────────────────────────────────────────────────────────────────────

function_var *AddOp::make_output() {
    const Tensor *a = inputs[0]->val;
    const Tensor *b = inputs[1]->val;
    u32 out_shape[MAX_NDIM];
    u32 out_dim = broadcast_shape(a, b, out_shape);
    Tensor *out_tensor = new Tensor(out_dim, out_shape, inputs[0]->val->on_gpu);

    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    function_var *out = new function_var(out_tensor, flags);
    out->grad_fn = this;
    return out;
}

void AddOp::forward(function_var *out) {
    u32 shape[MAX_NDIM];
    u32 ndim = broadcast_shape(inputs[0]->val, inputs[1]->val, shape);
    tensor_realloc(out->val, shape, ndim);
    tensor_add(out->val, inputs[0]->val, inputs[1]->val);
}

void AddOp::backward(Tensor *grad_output) {
    function_var *A = inputs[0];
    function_var *B = inputs[1];

    if (A->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!A->grad || !tensor_shape_eq(A->grad, A->val)) {
            delete A->grad;
            A->grad = tensor_create_like(A->val);
        }
        Tensor *dA = reduce_grad(grad_output, A->val);
        tensor_add(A->grad, A->grad, dA);
        delete dA;
    }

    if (B->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!B->grad || !tensor_shape_eq(B->grad, B->val)) {
            delete B->grad;
            B->grad = tensor_create_like(B->val);
        }
        Tensor *dB = reduce_grad(grad_output, B->val);
        tensor_add(B->grad, B->grad, dB);
        delete dB;
    }
}

// ── ReluOp ───────────────────────────────────────────────────────────────────

function_var *ReluOp::make_output() {
    const function_var *A = inputs[0];
    u32 flags = FV_FLAG_NONE;
    if (A->flags & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    function_var *out = new function_var(tensor_create_like(A->val), flags);
    out->grad_fn = this;
    return out;
}

void ReluOp::forward(function_var *out) {
    tensor_realloc(out->val, inputs[0]->val->shape, inputs[0]->val->ndim);
    tensor_relu(out->val, inputs[0]->val);
}

void ReluOp::backward(Tensor *grad_output) {
    function_var *A = inputs[0];
    if (A->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!A->grad || !tensor_shape_eq(A->grad, A->val)) {
            delete A->grad;
            A->grad = tensor_create_like(A->val);
        }
        tensor_relu_backward(A->grad, grad_output, A->val);
    }
}

// ── Conv2dOp ─────────────────────────────────────────────────────────────────

function_var *Conv2dOp::make_output() {
    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    const function_var *input = inputs[0];
    const function_var *kernels = inputs[1];
    params.compute_output_size(input->n_rows(), input->n_cols());
    const u32 output_shape[4] = {input->val->shape[0], kernels->val->shape[1],
                                 params.L_h, params.L_w};

    function_var *out = new function_var(
        new Tensor(4, output_shape, input->val->on_gpu), flags);

    out->grad_fn = this;
    return out;
}
void Conv2dOp::forward(function_var *out) {
    const Tensor *input = inputs[0]->val;
    const Tensor *kernels = inputs[1]->val;

    u32 N = input->shape[0];
    u32 C = input->shape[1];
    u32 C_out = kernels->shape[1];
    u32 L = params.L_h * params.L_w;

    // unfold: [N, C, H, W] → [N*L, C*kH*kW]
    Tensor *col = tensor_unfold2d(input, params);
    u32 col_shape[2] = {N * L, C * params.k_h * params.k_w};
    tensor_reshape(col, col_shape, 2);
    delete saved_col;
    saved_col = col;

    // W stored pre-transposed as [C*kH*kW, C_out] — matches nn_model Wt
    // convention [N*L, C*kH*kW] @ [C*kH*kW, C_out] → [N*L, C_out]
    Tensor *res = tensor_mat_mul(col, kernels);

    // [N*L, C_out] → [N, L_h, L_w, C_out] → [N, C_out, L_h, L_w]
    u32 res_shape[4] = {N, params.L_h, params.L_w, C_out};
    tensor_reshape(res, res_shape, 4);
    tensor_transpose(res, 1, 3); // [N, C_out, L_w, L_h]
    tensor_transpose(res, 2, 3); // [N, C_out, L_h, L_w]

    std::swap(out->val, res);
    delete res;
}

void Conv2dOp::backward(Tensor *grad_output) {
    function_var *input = inputs[0];
    function_var *kernels = inputs[1];

    u32 N = input->val->shape[0];
    u32 C_out = kernels->val->shape[1];
    u32 L = params.L_h * params.L_w;

    // Reshapes grad_output [N, C_out, L_h, L_w] → [N*L, C_out]
    auto flat_grad = [&]() -> Tensor * {
        const u32 shape3[3] = {N, C_out, L};
        Tensor *g = tensor_view(grad_output);
        tensor_reshape(g, shape3, 3);
        tensor_transpose(g, 1, 2);
        const u32 shape2[2] = {N * L, C_out};
        tensor_reshape(g, shape2, 2);
        return g;
    };

    // dInput = flat_grad @ kernels^T  →  fold2d  →  [N, C, H, W]
    if (input->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!input->grad || !tensor_shape_eq(input->grad, input->val)) {
            delete input->grad;
            input->grad = tensor_create_like(input->val);
        }
        Tensor *g = flat_grad();
        Tensor *kernels_t = tensor_view(kernels->val);
        tensor_transpose(kernels_t, 0, 1);
        Tensor *col = tensor_mat_mul(g, kernels_t);
        tensor_fold2d(input->grad, col, params);
        delete g;
        delete kernels_t;
        delete col;
    }

    // dW = col^T @ flat_grad  →  [C*kH*kW, C_out]
    if (kernels->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!kernels->grad || !tensor_shape_eq(kernels->grad, kernels->val)) {
            delete kernels->grad;
            kernels->grad = tensor_create_like(kernels->val);
        }
        Tensor *g = flat_grad();
        Tensor *col_t = tensor_view(saved_col);
        tensor_transpose(col_t, 0, 1);
        tensor_mat_mul(kernels->grad, col_t, g, false);
        delete g;
        delete col_t;
    }
}

// ── MaxPool2dOp ────────────────────────────────────────────────────────────

function_var *MaxPool2dOp::make_output() {
    const function_var *input = inputs[0];
    u32 flags = FV_FLAG_NONE;
    if ((input->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    params.compute_output_size(input->n_rows(), input->n_cols());
    u32 N = input->val->shape[0];
    u32 C = input->val->shape[1];

    u32 shape[4] = {N, C, params.L_h, params.L_w};

    function_var *out =
        new function_var(new Tensor(4, shape, input->val->on_gpu), flags);

    out->grad_fn = this;

    return out;
}

void MaxPool2dOp::forward(function_var *out) {
    const Tensor *input = inputs[0]->val;
    u32 N = input->shape[0];
    u32 C = input->shape[1];
    params.compute_output_size(input->shape[2], input->shape[3]);
    u32 L = params.L_h * params.L_w;
    u32 K = params.k_h * params.k_w;

    // [N, C, H, W] → [N, L, C*K]
    Tensor *col = tensor_unfold2d(input, params);

    // [N, L, C*K] → [N, L, C, K]
    u32 shape4[4] = {N, L, C, K};
    tensor_reshape(col, shape4, 4);

    // save argmax indices [N, L, C, 1] for backward
    delete saved_max_idx;
    saved_max_idx = tensor_argmax(col, 3, true);

    // max along K → [N, L, C]
    Tensor *pooled = tensor_max(col, 3, false);
    delete col;

    // [N, L, C] → [N, L_h, L_w, C] → [N, C, L_h, L_w]
    u32 shape_nlhwc[4] = {N, params.L_h, params.L_w, C};
    tensor_reshape(pooled, shape_nlhwc, 4);
    tensor_transpose(pooled, 1, 3); // [N, C, L_w, L_h]
    tensor_transpose(pooled, 2, 3); // [N, C, L_h, L_w]

    std::swap(out->val, pooled);
    delete pooled;
}

void MaxPool2dOp::backward(Tensor *grad_output) {
    function_var *input = inputs[0];
    if (!(input->flags & FV_FLAG_REQUIERES_GRAD))
        return;

    if (!input->grad || !tensor_shape_eq(input->grad, input->val)) {
        delete input->grad;
        input->grad = tensor_create_like(input->val);
    }

    u32 N = input->val->shape[0];
    u32 C = input->val->shape[1];
    u32 K = params.k_h * params.k_w;
    u32 L = params.L_h * params.L_w;

    // Reverse forward transpose: [N, C, L_h, L_w] → [N, L, C, 1]
    Tensor *g = tensor_view(grad_output);
    tensor_transpose(g, 2, 3); // [N, C, L_w, L_h]
    tensor_transpose(g, 1, 3); // [N, L_h, L_w, C]
    u32 shape_nlc1[4] = {N, L, C, 1};
    tensor_reshape(g, shape_nlc1, 4);

    // route grads to max positions → [N, L, C, K]
    Tensor *scattered = tensor_scatter_add(g, saved_max_idx, 3, K);
    delete g;

    // [N, L, C, K] → [N, L, C*K]
    u32 shape_nlck[3] = {N, L, C * K};
    tensor_reshape(scattered, shape_nlck, 3);

    // fold2d accumulates into input->grad [N, C, H, W]
    tensor_fold2d(input->grad, scattered, params);
    delete scattered;
}

// ── FlattenOp ────────────────────────────────────────────────────────────────

function_var *FlattenOp::make_output() {
    const Tensor *inp = inputs[0]->val;
    u32 N = inp->shape[0];
    u32 flat = (u32)(inp->size / N);
    u32 out_shape[2] = {N, flat};
    u32 flags = (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD)
                    ? FV_FLAG_REQUIERES_GRAD
                    : FV_FLAG_NONE;
    function_var *out =
        new function_var(new Tensor(2, out_shape, inp->on_gpu), flags);
    out->grad_fn = this;
    return out;
}

void FlattenOp::forward(function_var *out) {
    const Tensor *inp = inputs[0]->val;
    saved_ndim = inp->ndim;
    // TODO: Use memcpy or some helper function
    for (u32 i = 0; i < inp->ndim; i++)
        saved_shape[i] = inp->shape[i];
    u32 N = inp->shape[0];
    u32 flat = (u32)(inp->size / N);
    u32 flat_shape[2] = {N, flat};
    tensor_realloc(out->val, flat_shape, 2);
    tensor_copy(out->val, inp);
}

void FlattenOp::backward(Tensor *grad_output) {
    function_var *inp = inputs[0];
    if (!(inp->flags & FV_FLAG_REQUIERES_GRAD))
        return;
    if (!inp->grad || !tensor_shape_eq(inp->grad, inp->val)) {
        delete inp->grad;
        inp->grad = tensor_create_like(inp->val);
    }
    Tensor *grad_reshaped = tensor_view(grad_output);
    tensor_reshape(grad_reshaped, saved_shape, saved_ndim);
    tensor_add(inp->grad, inp->grad, grad_reshaped);
    delete grad_reshaped;
}

// ── BatchNormOp ────────────────────────────────────────────────────────────

function_var *BatchNormOp::make_output() {
    const function_var *input = inputs[0];
    const function_var *gamma = inputs[1];
    const function_var *beta  = inputs[2];
    u32 flags =
        ((input->flags | gamma->flags | beta->flags) & FV_FLAG_REQUIERES_GRAD)
            ? FV_FLAG_REQUIERES_GRAD
            : FV_FLAG_NONE;

    function_var *out = new function_var(tensor_create_like(input->val), flags);
    out->grad_fn = this;
    return out;
}

void BatchNormOp::forward(function_var *out) {
    const Tensor *inp   = inputs[0]->val;
    const Tensor *gamma = inputs[1]->val;
    const Tensor *beta  = inputs[2]->val;
    bool on_gpu = inp->on_gpu;
    u32 C      = inp->shape[1];
    u32 c4[4]  = { 1, C, 1, 1 };

    Tensor *mean = new Tensor(4, c4, on_gpu);
    Tensor *var  = new Tensor(4, c4, on_gpu);
    Tensor *xhat = tensor_create_like(inp);
    delete saved_mean; saved_mean = mean;
    delete saved_var;  saved_var  = var;
    delete saved_xhat; saved_xhat = xhat;

    tensor_welford_mean_var(saved_mean, saved_var, inp, 1);

    // xhat = (inp - mean) / sqrt(var + eps)
    tensor_sub(saved_xhat, inp, saved_mean);
    Tensor *denom = tensor_add(saved_var, eps);
    tensor_sqrt(denom, denom);
    tensor_div(saved_xhat, saved_xhat, denom);
    delete denom;

    // y = gamma * xhat + beta  (gamma/beta [C] → view as [1,C,1,1])
    Tensor *g = tensor_view(gamma);
    Tensor *b = tensor_view(beta);
    tensor_reshape(g, c4, 4);
    tensor_reshape(b, c4, 4);

    tensor_realloc(out->val, inp->shape, inp->ndim);
    tensor_mul(out->val, saved_xhat, g);
    tensor_add(out->val, out->val, b);
    delete g;
    delete b;
}

void BatchNormOp::backward(Tensor *grad_output) {
    function_var *fv_inp   = inputs[0];
    function_var *fv_gamma = inputs[1];
    function_var *fv_beta  = inputs[2];

    const Tensor *inp   = fv_inp->val;
    const Tensor *gamma = fv_gamma->val;

    u32 N = inp->shape[0], C = inp->shape[1],
        H = inp->shape[2], W = inp->shape[3];
    f32 m      = (f32)(N * H * W);
    bool on_gpu = inp->on_gpu;
    u32 c4[4]      = { 1, C, 1, 1 };
    u32 c_shape[1] = { C };

    // std = sqrt(var + eps)  [1,C,1,1]
    Tensor *std_dev = tensor_add(saved_var, eps);
    tensor_sqrt(std_dev, std_dev);

    // reduces [N,C,H,W] → [1,C,1,1] by summing over N, H, W
    auto reduce_nhw = [&](const Tensor *t) -> Tensor * {
        u32 s1[4] = {N, C, H, 1}, s2[4] = {N, C, 1, 1}, s3[4] = {1, C, 1, 1};
        Tensor *a = new Tensor(4, s1, on_gpu);
        Tensor *b = new Tensor(4, s2, on_gpu);
        Tensor *c = new Tensor(4, s3, on_gpu);
        tensor_sum(a, t, 3, true);
        tensor_sum(b, a, 2, true);
        tensor_sum(c, b, 0, true);
        delete a;
        delete b;
        return c;
    };

    // d_xhat = grad_output * gamma  [N,C,H,W]
    Tensor *g_view = tensor_view(gamma);
    tensor_reshape(g_view, c4, 4);
    Tensor *d_xhat = tensor_mul(grad_output, g_view);
    delete g_view;

    // d_beta = sum(grad_output, N,H,W)  → [C]
    if (fv_beta->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!fv_beta->grad || !tensor_shape_eq(fv_beta->grad, fv_beta->val)) {
            delete fv_beta->grad;
            fv_beta->grad = tensor_create_like(fv_beta->val);
        }
        Tensor *db = reduce_nhw(grad_output);
        tensor_reshape(db, c_shape, 1);
        tensor_add(fv_beta->grad, fv_beta->grad, db);
        delete db;
    }

    // d_gamma = sum(grad_output * xhat, N,H,W)  → [C]
    if (fv_gamma->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!fv_gamma->grad || !tensor_shape_eq(fv_gamma->grad, fv_gamma->val)) {
            delete fv_gamma->grad;
            fv_gamma->grad = tensor_create_like(fv_gamma->val);
        }
        Tensor *tmp = tensor_mul(grad_output, saved_xhat);
        Tensor *dg  = reduce_nhw(tmp);
        delete tmp;
        tensor_reshape(dg, c_shape, 1);
        tensor_add(fv_gamma->grad, fv_gamma->grad, dg);
        delete dg;
    }

    // d_x = (1/std) * (d_xhat - (1/m)*sum(d_xhat) - xhat*(1/m)*sum(d_xhat*xhat))
    if (fv_inp->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!fv_inp->grad || !tensor_shape_eq(fv_inp->grad, inp)) {
            delete fv_inp->grad;
            fv_inp->grad = tensor_create_like(inp);
        }

        // (1/m) * sum(d_xhat)  [1,C,1,1]
        Tensor *mean_dxhat = reduce_nhw(d_xhat);
        tensor_div(mean_dxhat, mean_dxhat, m);

        // xhat * (1/m) * sum(d_xhat * xhat)  [N,C,H,W]
        Tensor *tmp          = tensor_mul(d_xhat, saved_xhat);
        Tensor *mean_dxhat_x = reduce_nhw(tmp);
        delete tmp;
        tensor_div(mean_dxhat_x, mean_dxhat_x, m);
        Tensor *xhat_term = tensor_mul(saved_xhat, mean_dxhat_x);
        delete mean_dxhat_x;

        // dx = (d_xhat - mean_dxhat - xhat_term) / std
        Tensor *dx = tensor_sub(d_xhat, mean_dxhat);
        delete mean_dxhat;
        tensor_sub(dx, dx, xhat_term);
        delete xhat_term;
        tensor_div(dx, dx, std_dev);

        tensor_add(fv_inp->grad, fv_inp->grad, dx);
        delete dx;
    }

    delete d_xhat;
    delete std_dev;
}

// ── MeanSquareErrorOp ────────────────────────────────────────────────────────

function_var *MeanSquareErrorOp::make_output() {
    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    u32 shape[1] = {1};
    function_var *out =
        new function_var(new Tensor(1, shape, inputs[0]->val->on_gpu), flags);
    out->grad_fn = this;
    return out;
}

void MeanSquareErrorOp::forward(function_var *out) {
    Tensor *diff = tensor_create_like(inputs[0]->val);
    tensor_sub(diff, inputs[0]->val, inputs[1]->val);
    tensor_mul(diff, diff, diff);
    N = diff->size;
    tensor_sum(out->val, diff);
    tensor_div(out->val, out->val, (f32)N);
    delete diff;
}

void MeanSquareErrorOp::backward(Tensor *grad_output) {
    // scale = grad_output * (2 / N)
    Tensor *scale = tensor_mul(grad_output, 2.0f / (f32)N);
    function_var *A = inputs[0];
    function_var *B = inputs[1];

    // dA = scale * (A - B)
    if (A->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!A->grad || !tensor_shape_eq(A->grad, A->val)) {
            delete A->grad;
            A->grad = tensor_create_like(A->val);
        }
        Tensor *diff = tensor_create_like(A->val);
        tensor_sub(diff, A->val, B->val);
        tensor_mul(diff, diff, scale);
        tensor_add(A->grad, A->grad, diff);
        delete diff;
    }

    // dB = scale * (B - A)
    if (B->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!B->grad || !tensor_shape_eq(B->grad, B->val)) {
            delete B->grad;
            B->grad = tensor_create_like(B->val);
        }
        Tensor *diff = tensor_create_like(B->val);
        tensor_sub(diff, B->val, A->val);
        tensor_mul(diff, diff, scale);
        tensor_add(B->grad, B->grad, diff);
        delete diff;
    }

    delete scale;
}

// ── CrossEntropyWithLogitsOp ─────────────────────────────────────────────────

function_var *CrossEntropyWithLogitsOp::make_output() {
    const Tensor *logits = inputs[0]->val;

    N_batch = logits->ndim >= 2 ? logits->shape[0] : 1;
    saved_softmax = tensor_create_like(logits);

    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD)
        flags |= FV_FLAG_REQUIERES_GRAD;

    u32 shape[1] = {1};
    function_var *out =
        new function_var(new Tensor(1, shape, logits->on_gpu), flags);
    out->grad_fn = this;
    return out;
}

void CrossEntropyWithLogitsOp::forward(function_var *out) {
    const Tensor *logits = inputs[0]->val;
    const Tensor *targets = inputs[1]->val;

    N_batch = logits->ndim >= 2 ? logits->shape[0] : 1;
    tensor_realloc(saved_softmax, logits->shape, logits->ndim);

    // softmax saved for backward; log_softmax used for loss (avoids log(~0) NaN)
    tensor_softmax(saved_softmax, logits);

    Tensor *log_probs = tensor_log_softmax(logits);
    tensor_mul(log_probs, log_probs, targets);
    tensor_sum(out->val, log_probs);
    tensor_div(out->val, out->val, -(f32)N_batch);
    delete log_probs;
}

void CrossEntropyWithLogitsOp::backward(Tensor *grad_output) {
    function_var *logits = inputs[0];
    function_var *targets = inputs[1];

    // d_logits = (softmax - targets) / N_batch * grad_output
    if (logits->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!logits->grad || !tensor_shape_eq(logits->grad, logits->val)) {
            delete logits->grad;
            logits->grad = tensor_create_like(logits->val);
        }

        Tensor *d = tensor_create_like(logits->val);
        tensor_sub(d, saved_softmax, targets->val);
        tensor_div(d, d, (f32)N_batch);
        tensor_mul(d, d, grad_output);
        tensor_add(logits->grad, logits->grad, d);
        delete d;
    }

    // d_targets = -log(softmax) / N_batch * grad_output
    if (targets->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!targets->grad || !tensor_shape_eq(targets->grad, targets->val)) {
            delete targets->grad;
            targets->grad = tensor_create_like(targets->val);
        }

        Tensor *d = tensor_create_like(targets->val);
        tensor_log(d, saved_softmax);
        tensor_div(d, d, -(f32)N_batch);
        tensor_mul(d, d, grad_output);
        tensor_add(targets->grad, targets->grad, d);
        delete d;
    }
}
