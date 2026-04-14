#ifndef MODELS_HPP
#define MODELS_HPP
#include "autograd.hpp"
#include "ops.hpp"

struct linear_model {
    // Parameters
    function_var *W;
    function_var *b;

    // Inputs and targets
    function_var *X;
    function_var *y;

    // Intermediate variables — allocated once, reused every forward
    function_var *fv_xw;
    function_var *fv_pred;
    function_var *fv_loss;

    // Ops — created once, owned by model
    MatMulOp *op_matmul;
    AddOp *op_add;
    MeanSquareErrorOp *op_mse;

    // Graph — built once in constructor, owned by model
    Graph *graph;

    linear_model(Tensor *val_X, Tensor *val_y);
    ~linear_model();
    void forward();
};

struct nn_model {
    // Parameters
    std::vector<function_var *> W;
    std::vector<function_var *> b;

    // Inputs and targets
    function_var *X;
    function_var *y;

    // Intermediate variables — allocated once, reused every forward
    std::vector<function_var *> z;  // pre-activation: X @ W + b
    std::vector<function_var *> a;  // post-activation: relu(z)
    function_var *fv_loss;

    // Ops
    std::vector<MatMulOp *> op_matmul;
    std::vector<AddOp *> op_add;
    std::vector<ReluOp *> op_relu;
    CrossEntropyWithLogitsOp *op_loss;

    // Graph
    Graph *graph;

    nn_model(Tensor *val_X, Tensor *val_y, const std::vector<u32> &layer_sizes);
    ~nn_model();
    void forward();
};

#endif
