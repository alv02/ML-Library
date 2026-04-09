#include "../include/optimizers.hpp"

gd_optimizer::gd_optimizer(f32 learning_rate)
    : lr(learning_rate), graph(nullptr) {}

gd_optimizer::~gd_optimizer() {}

void gd_optimizer::zero_grad() {
    if (!graph) {
        printf("No graph set for optimizer\n");
        return;
    }
    for (function_var *fv : graph->nodes) {
        if ((fv->flags & FV_FLAG_REQUIERES_GRAD) && fv->grad) {
            tensor_fill(fv->grad, 0.0f);
        }
    }
}

void gd_optimizer::step() {
    if (!graph) {
        printf("No graph set for optimizer\n");
        return;
    }
    for (function_var *fv : graph->nodes) {
        if ((fv->flags & FV_FLAG_PARAMETER) && fv->grad) {
            // var = var - lr * dL/dvar
            tensor_scale(fv->grad, fv->grad, lr);
            tensor_sub(fv->val, fv->val, fv->grad);
        }
    }
}

void gd_optimizer::set_graph(Graph *graph) { this->graph = graph; }
