#include "../include/ops.hpp"

function_var *MatMulOp::forward() {
    Tensor *out = tensor_mat_mul(inputs[0]->val, inputs[1]->val);

    // If either of the inputs requires grad, then the output needs to require
    // grad
    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD) {
        flags |= FV_FLAG_REQUIERES_GRAD;
    }

    function_var *fv = fv_create(out, flags);

    fv->grad_fn = this;

    return fv;
}

void MatMulOp::backward(Tensor *grad_output) {
    // dA = grad @ B.T  — view of B, transpose it, matmul, done
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        // If the grad is not allocated yet, allocate it
        if (!inputs[0]->grad) {
            inputs[0]->grad = tensor_create_like(inputs[0]->val);
        }

        Tensor *Bt = tensor_view(inputs[1]->val);

        tensor_transpose(Bt, 0, 1);
        // Dont clear out so it accumulates gradients if there are multiple
        // paths to the same variable
        tensor_mat_mul(inputs[0]->grad, grad_output, Bt, false);

        tensor_free(Bt);
    }

    // dB = A.T @ grad
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        // If the grad is not allocated yet, allocate it
        if (!inputs[1]->grad) {
            inputs[1]->grad = tensor_create_like(inputs[1]->val);
        }

        Tensor *At = tensor_view(inputs[0]->val);

        tensor_transpose(At, 0, 1);
        // Dont clear out so it accumulates gradients if there are multiple
        // paths to the same variable
        tensor_mat_mul(inputs[1]->grad, At, grad_output, false);

        tensor_free(At);
    }
}

function_var *AddOp::forward() {
    Tensor *out = tensor_add(inputs[0]->val, inputs[1]->val);

    u32 flags = FV_FLAG_NONE;
    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD) {
        flags |= FV_FLAG_REQUIERES_GRAD;
    }

    function_var *fv = fv_create(out, flags);

    fv->grad_fn = this;

    return fv;
}

void AddOp::backward(Tensor *grad_output) {
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!inputs[0]->grad)
            inputs[0]->grad = tensor_create_like(inputs[0]->val);
        tensor_add(inputs[0]->grad, inputs[0]->grad, grad_output);
    }
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!inputs[1]->grad)
            inputs[1]->grad = tensor_create_like(inputs[1]->val);

        // b is (1,1) but grad_output is (N,1) — must sum over batch
        if (inputs[1]->val->size != grad_output->size) {
            f32 sum = tensor_sum(grad_output);
            inputs[1]->grad->data[0] += sum;
        } else {
            tensor_add(inputs[1]->grad, inputs[1]->grad, grad_output);
        }
    }
}
function_var *MeanSquareErrorOp::forward() {
    Tensor *diff = tensor_create_like(inputs[0]->val);
    u32 shape[1] = {1};
    Tensor *out = tensor_create(1, shape, false);
    // out = (a-b)^2
    tensor_sub(diff, inputs[0]->val, inputs[1]->val);
    tensor_mul(diff, diff, diff);

    // out = sum(out) / N
    f32 sum = tensor_sum(diff);
    u32 N = inputs[0]->val->shape[ROW_DIM(inputs[0]->val)];
    out->data[0] = sum / (f32)N;

    function_var *fv = fv_create(out, FV_FLAG_NONE);

    if ((inputs[0]->flags | inputs[1]->flags) & FV_FLAG_REQUIERES_GRAD) {
        fv->flags |= FV_FLAG_REQUIERES_GRAD;
    }

    fv->grad_fn = this;

    tensor_free(diff);

    return fv;
}

void MeanSquareErrorOp::backward(Tensor *grad_output) {
    u32 N = inputs[0]->val->shape[ROW_DIM(inputs[0]->val)];
    f32 scale = grad_output->data[0];
    Tensor *tmp2 = tensor_create_like(inputs[0]->val);
    tensor_sub(tmp2, inputs[0]->val, inputs[1]->val);
    printf("residual[0]: %f, pred[0]: %f, y[0]: %f\n", tmp2->data[0],
           inputs[0]->val->data[0], inputs[1]->val->data[0]);
    tensor_free(tmp2);

    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!inputs[0]->grad)
            inputs[0]->grad = tensor_create_like(inputs[0]->val);
        Tensor *tmp = tensor_create_like(inputs[0]->val);
        tensor_sub(tmp, inputs[0]->val, inputs[1]->val);
        tensor_scale(tmp, tmp, scale * 2.0f / (f32)N);
        tensor_add(inputs[0]->grad, inputs[0]->grad, tmp); // accumulate
        tensor_free(tmp);
    }
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        if (!inputs[1]->grad)
            inputs[1]->grad = tensor_create_like(inputs[1]->val);
        Tensor *tmp = tensor_create_like(inputs[1]->val);
        tensor_sub(tmp, inputs[1]->val, inputs[0]->val);
        tensor_scale(tmp, tmp, scale * 2.0f / (f32)N);
        tensor_add(inputs[1]->grad, inputs[1]->grad, tmp); // accumulate
        tensor_free(tmp);
    }
}
