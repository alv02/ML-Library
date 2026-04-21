#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"
#include <vector>
#define MAX_INPUTS 3

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

    function_var(Tensor *val, u32 flags);
    ~function_var();
    u32 n_rows() const;
    u32 n_cols() const;
};

void fv_print(const function_var *fv);

struct function {
    u32 n_inputs;
    function_var *inputs[MAX_INPUTS];

    // Allocate output fv with correct shape and flags, wire grad_fn. Called
    // once.
    virtual function_var *make_output() = 0;
    // Compute values into an already-allocated fv. Called every step.
    virtual void forward(function_var *out) = 0;
    virtual void backward(Tensor *grad_output) = 0;
};

struct Graph {
    std::vector<function_var *> nodes;
};

Graph *graph_create(function_var *output);
void graph_free(Graph *graph);
void graph_backward(Graph *graph);
#endif
