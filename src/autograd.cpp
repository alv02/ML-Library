#include "../include/autograd.hpp"
#include <algorithm>
#include <functional>
#include <unordered_set>

void backward(Var loss) {
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
    loss->grad = Tensor<f32>::make(1, one_shape, loss->data->on_gpu());
    tensor_fill(loss->grad, 1.0f);

    for (auto &v : order) {
        if (v->grad_fn && v->grad.defined())
            v->grad_fn->backward(v->grad);
        // Free saved tensors immediately (saved_softmax, saved_xhat, etc.)
        v->grad_fn = nullptr;
        if (!(v->flags & FV_FLAG_PARAMETER)) {
            // Propagated already — release the gradient tensor
            v->grad = Tensor<f32>();
            // Release the VarImpl itself. By reverse-topological order, all
            // downstream grad_fns have been nulled above, so their inputs[]
            // refs to this node are already gone. The only remaining holder
            // is order[i], which we drop now — activation data is freed as
            // its Storage refcount hits zero, exactly like PyTorch does.
            v.impl_ = nullptr;
        }
    }
}
