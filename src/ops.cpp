#include "../include/ops.hpp"

void MatMulOp::forward(mem_arena *arena) {
    tensor_mat_mul(output->val, inputs[0]->val, inputs[1]->val);
}

void MatMulOp::backward(mem_arena *arena) {
    // dA = grad @ B.T  — view of B, transpose it, matmul, done
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        Tensor *Bt = tensor_view(arena, inputs[1]->val);
        tensor_transpose(Bt, 0, 1);
        tensor_mat_mul(inputs[0]->grad, output->grad, Bt);
    }

    // dB = A.T @ grad
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        Tensor *At = tensor_view(arena, inputs[0]->val);
        tensor_transpose(At, 0, 1);
        tensor_mat_mul(inputs[1]->grad, At, output->grad);
    }
}

void MeanSquareErrorOp::forward(mem_arena *arena) {
    Tensor *e = tensor_create_like(arena, inputs[0]->val);
    tensor_sub(e, inputs[0]->val, inputs[1]->val);

    Tensor *e_2 = tensor_create_like(arena, e);
    tensor_mul(e_2, e, e);

    f32 sum = tensor_sum(e_2);
    u32 N = e->shape[ROW_DIM(e)];

    output->val->data[0] = sum / N;
}

void MeanSquareErrorOp::backward(mem_arena *arena) {
    u32 N = inputs[0]->val->shape[ROW_DIM(inputs[0]->val)];

    // dA = a-b
    if (inputs[0]->flags & FV_FLAG_REQUIERES_GRAD) {
        tensor_sub(inputs[0]->grad, inputs[0]->val, inputs[1]->val);
        tensor_scale(inputs[0]->grad, 2.0f / (f32)N);
    }

    // dB = -(a-b) = b-a
    if (inputs[1]->flags & FV_FLAG_REQUIERES_GRAD) {
        tensor_sub(inputs[1]->grad, inputs[1]->val, inputs[0]->val);
        tensor_scale(inputs[1]->grad, 2.0f / (f32)N);
    }
}
