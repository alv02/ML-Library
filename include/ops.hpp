#ifndef OPS_H
#define OPS_H

#include "autograd.hpp"

struct MatMulOp : function {
    void forward(mem_arena *arena) override;
    void backward(mem_arena *arena) override;
};

struct MeanSquareErrorOp : function {
    void forward(mem_arena *arena) override;
    void backward(mem_arena *arena) override;
};
#endif
