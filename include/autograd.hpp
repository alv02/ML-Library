#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"
#include <vector>
#define MAX_INPUTS 2

enum function_var_flags {
    FV_FLAG_NONE = 0,
    FV_FLAG_REQUIERES_GRAD = (1 << 0),
    FV_FLAG_PARAMETER = (1 << 1),

};

struct function;

struct function_var {
    u32 flags;
    Tensor *val;
    Tensor *grad;

    function *grad_fn;
};

function_var *fv_create(Tensor *val, u32 flags);
void fv_free(function_var *fv);
void fv_print(const function_var *fv);

struct function {
    u32 n_inputs;
    function_var *inputs[MAX_INPUTS];

    virtual function_var *forward() = 0;
    virtual void backward(Tensor *grad_output) = 0;
};

void function_free(function *fn);

struct Graph {
    std::vector<function_var *> nodes;
};

Graph *graph_create(function_var *output);
void graph_free(Graph *graph);
void graph_backward(Graph *graph);
#endif
