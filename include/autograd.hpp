#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"

#define MAX_INPUTS 2

enum function_var_flags {
    FV_FLAG_NONE = 0,
    FV_FLAG_REQUIERES_GRAD = (1 << 0),

};

struct function_var {
    u32 flags;
    Tensor *val;
    Tensor *grad;
};

struct function {
    u32 n_inputs;
    function_var *inputs[MAX_INPUTS];

    function_var *output;

    virtual void forward(mem_arena *arena);
    virtual void backward(mem_arena *arena);
};

#endif
