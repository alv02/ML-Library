#ifndef MODELS_HPP
#define MODELS_HPP
#include "autograd.hpp"

struct linear_model {
    // FV owned by the model, parameters
    function_var *W;
    function_var *b;

    // FV owned by the model, inputs and targets
    function_var *X;
    function_var *y;

    // FV owned by the model, intermediate variables of the model
    function_var *fv_xw;
    function_var *fv_pred;
    function_var *fv_loss;

    linear_model(Tensor *val_X, Tensor *val_y);
    ~linear_model();
    function_var *forward();
};

#endif
