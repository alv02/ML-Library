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

template <typename T = f32> struct VarImpl_ {
    u32 flags;
    Tensor<T> data;
    Tensor<T> grad;
    std::shared_ptr<Function> grad_fn;

    VarImpl_(Tensor<T> data, u32 flags)
        : data(data), flags(flags), grad_fn(nullptr) {}
    ~VarImpl_() = default;
};

template <typename T = f32> struct Var_ {
    std::shared_ptr<VarImpl_<T>> impl_;

    Var_() = default;

    Var_(Tensor<T> data, u32 flags = FV_FLAG_NONE)
        : impl_(std::make_shared<VarImpl_<T>>(data, flags)) {}

    bool defined() const { return impl_ != nullptr; }
    explicit operator bool() { return defined(); }
    VarImpl_<T> *operator->() const { return impl_.get(); }
};

// ── Public aliases ────────────────────────────────────────────────────────────

using Var     = Var_<f32>;
using VarImpl = VarImpl_<f32>;
using VarU32  = Var_<u32>;
using VarI32  = Var_<i32>;

// ── Function ──────────────────────────────────────────────────────────────────

struct Function {
    std::vector<Var> inputs;
    virtual void backward(Tensor<f32> grad_output) = 0;
    virtual ~Function() = default;
};

void backward(Var loss, CudaMemArena *arena = nullptr);

#endif
