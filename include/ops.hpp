#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

struct MatMulOp : function {
    MatMulOp(function_var *a, function_var *b) {
        n_inputs = 2;
        inputs[0] = a;
        inputs[1] = b;
    }

    function_var *forward() override;
    void backward(Tensor *grad_output) override;
};

MatMulOp *matmul_op_create(function_var *a, function_var *b);

struct AddOp : function {
    AddOp(function_var *a, function_var *b) {
        n_inputs = 2;
        inputs[0] = a;
        inputs[1] = b;
    }

    function_var *forward() override;
    void backward(Tensor *grad_output) override;
};

struct MeanSquareErrorOp : function {
    MeanSquareErrorOp(function_var *pred, function_var *target) {
        n_inputs = 2;
        inputs[0] = pred;
        inputs[1] = target;
    }
    function_var *forward() override;
    void backward(Tensor *grad_output) override;
};
#endif
