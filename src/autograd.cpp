#include "../include/autograd.hpp"
#include <algorithm>
#include <functional>
#include <unordered_set>

VarImpl::~VarImpl() = default;

Var::Var(Tensor data, u32 flags) {

    impl_ = std::make_shared<VarImpl>(data, flags);

    impl_->grad = Tensor();
    impl_->grad_fn = nullptr;
}

void backward(Var loss, CudaMemArena *arena) {
    std::vector<Var> order;
    std::unordered_set<VarImpl *> visited;

    std::function<void(Var)> visit = [&](Var v) {
        if (!v.defined() || visited.count(v.impl_.get()))
            return;
        visited.insert(v.impl_.get());
        if (v->grad_fn)
            for (auto &inp : v->grad_fn->inputs)
                visit(inp);
        order.push_back(v);
    };
    visit(loss);
    std::reverse(order.begin(), order.end());

    u32 one_shape[] = {1};
    loss->grad = Tensor::make(1, one_shape, loss->data->on_gpu(), arena);
    tensor_fill(loss->grad, 1.0f);

    for (auto &v : order)
        if (v->grad_fn && v->grad.defined())
            v->grad_fn->backward(v->grad);
}
