#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

struct MatMulOp : function {
    MatMulOp(function_var *a, function_var *b) {
        n_inputs = 2;
        inputs[0] = a;
        inputs[1] = b;
    }

    function_var *make_output() override;
    void forward(function_var *out) override;
    void backward(Tensor *grad_output) override;
};

struct AddOp : function {
    AddOp(function_var *a, function_var *b) {
        n_inputs = 2;
        inputs[0] = a;
        inputs[1] = b;
    }

    function_var *make_output() override;
    void forward(function_var *out) override;
    void backward(Tensor *grad_output) override;
};

struct MeanSquareErrorOp : function {
    u64 N; // number of elements in the broadcasted diff — set during forward()

    MeanSquareErrorOp(function_var *pred, function_var *target) {
        n_inputs = 2;
        inputs[0] = pred;
        inputs[1] = target;
        N = 0;
    }
    function_var *make_output() override;
    void forward(function_var *out) override;
    void backward(Tensor *grad_output) override;
};
#endif
