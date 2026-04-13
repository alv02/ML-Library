#include "../include/ops.hpp"

// Reduces grad (broadcast output shape) to match target's shape by summing
// over every dimension that was broadcast. Returns a new owned tensor.
Tensor *reduce_grad(const Tensor *grad, const Tensor *target) {
    Tensor *cur = tensor_view(grad);
    Tensor *next = nullptr;
    u32 current_shape[MAX_NDIM];
    memcpy(current_shape, grad->shape, sizeof(u32) * grad->ndim);

    u32 target_expanded_shape[MAX_NDIM];
    expanded_shape(target, grad->shape, grad->ndim, target_expanded_shape);

    for (u32 i = 0; i < grad->ndim; i++) {
        // Reduce
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
    tensor_mat_mul(out->val, inputs[0]->val, inputs[1]->val, true);
}

void MatMulOp::backward(Tensor *grad_output) {
    // dA = grad @ B.T
    function_var *A = inputs[0];
    function_var *B = inputs[1];
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!A->grad)
            A->grad = tensor_create_like(A->val);

        Tensor *Bt = tensor_view(inputs[1]->val);
        tensor_transpose(Bt, ROW_DIM(Bt), COL_DIM(Bt));
        tensor_mat_mul(A->grad, grad_output, Bt, false);
        delete Bt;
    }

    // dB = A.T @ grad
    if (B->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!B->grad)
            B->grad = tensor_create_like(B->val);

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
    tensor_add(out->val, inputs[0]->val, inputs[1]->val);
}

void AddOp::backward(Tensor *grad_output) {
    function_var *A = inputs[0];
    function_var *B = inputs[1];

    // dA = grad
    if (A->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!A->grad)
            A->grad = tensor_create_like(A->val);
        Tensor *dA = reduce_grad(grad_output, A->val);
        tensor_add(A->grad, A->grad, dA);
        delete dA;
    }

    // dB = grad
    if (B->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!B->grad)
            B->grad = tensor_create_like(B->val);
        Tensor *dB = reduce_grad(grad_output, B->val);
        tensor_add(B->grad, B->grad, dB);
        delete dB;
    }
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
    // scale = grad_output * (2 / N) — stays on the same device as grad_output
    Tensor *scale = tensor_mul(grad_output, 2.0f / (f32)N);

    // dA = scale * (A - B)
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!inputs[0]->grad)
            inputs[0]->grad = tensor_create_like(inputs[0]->val);
        Tensor *diff = tensor_create_like(inputs[0]->val);
        tensor_sub(diff, inputs[0]->val, inputs[1]->val);
        tensor_mul(diff, diff, scale);
        tensor_add(inputs[0]->grad, inputs[0]->grad, diff);
        delete diff;
    }

    // dB = scale * (B - A)
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        // Lazy creation of grad
        if (!inputs[1]->grad)
            inputs[1]->grad = tensor_create_like(inputs[1]->val);
        Tensor *diff = tensor_create_like(inputs[1]->val);
        tensor_sub(diff, inputs[1]->val, inputs[0]->val);
        tensor_mul(diff, diff, scale);
        tensor_add(inputs[1]->grad, inputs[1]->grad, diff);
        delete diff;
    }

    delete scale;
}
