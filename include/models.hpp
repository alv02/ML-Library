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

#endif
