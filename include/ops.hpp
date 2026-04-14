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

struct ReluOp : function {
    ReluOp(function_var *a) {
        n_inputs = 1;
        inputs[0] = a;
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

// Fused softmax + cross-entropy loss.
// inputs[0] = logits  [N, C]
// inputs[1] = targets [N, C]  (one-hot)
// output = scalar loss mean over the batch
struct CrossEntropyWithLogitsOp : function {
    Tensor *saved_softmax; // softmax(logits), allocated once, reused each step
    u64 N_batch;           // batch size — set in make_output

    CrossEntropyWithLogitsOp(function_var *logits, function_var *targets) {
        n_inputs = 2;
        inputs[0] = logits;
        inputs[1] = targets;
        saved_softmax = nullptr;
        N_batch = 0;
    }

    ~CrossEntropyWithLogitsOp() { delete saved_softmax; }

    function_var *make_output() override;
    void forward(function_var *out) override;
    void backward(Tensor *grad_output) override;
};

#endif
