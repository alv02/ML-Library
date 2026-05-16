#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.hpp"
#include <memory>
#include <vector>

enum var_flags {
    FV_FLAG_NONE = 0,
    FV_FLAG_REQUIERES_GRAD = (1 << 0),
    FV_FLAG_PARAMETER = (1 << 1),
};

struct Function;

struct VarImpl {
    u32 flags;
    Tensor<f32> data;
    Tensor<f32> grad;
    std::shared_ptr<Function> grad_fn;

    VarImpl(Tensor<f32> data, u32 flags)
        : data(data), flags(flags), grad_fn(nullptr) {}
    ~VarImpl() = default;
};

struct Var {
    std::shared_ptr<VarImpl> impl_;

    Var() = default;

    Var(Tensor<f32> data, u32 flags = FV_FLAG_NONE)
        : impl_(std::make_shared<VarImpl>(data, flags)) {}

    bool defined() const { return impl_ != nullptr; }
    explicit operator bool() { return defined(); }
    VarImpl *operator->() const { return impl_.get(); }
};

struct Function {
    std::vector<Var> inputs;
    virtual void backward(Tensor<f32> grad_output) = 0;
    virtual ~Function() = default;
};

void backward(Var loss);

#endif
