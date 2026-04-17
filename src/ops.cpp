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
    Tensor *inp = inputs[0]->val;
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

    // softmax(logits) — numerically stable, saved for backward
    tensor_softmax(saved_softmax, logits);

    // loss = -sum(targets * log(softmax)) / N_batch
    Tensor *log_probs = tensor_create_like(saved_softmax);
    tensor_log(log_probs, saved_softmax);
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
