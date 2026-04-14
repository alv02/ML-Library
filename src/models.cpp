#include "../include/models.hpp"

// ── linear_model ─────────────────────────────────────────────────────────────

linear_model::linear_model(Tensor *val_X, Tensor *val_y) {
    u32 n_features = val_X->shape[COL_DIM(val_X)];

    u32 w_shape[2] = {n_features, 1};
    W = new function_var(new Tensor(2, w_shape, val_X->on_gpu),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    u32 b_shape[1] = {1};
    b = new function_var(new Tensor(1, b_shape, val_X->on_gpu),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    X = new function_var(val_X, FV_FLAG_NONE);
    y = new function_var(val_y, FV_FLAG_NONE);

    op_matmul = new MatMulOp(X, W);
    fv_xw = op_matmul->make_output();

    op_add = new AddOp(fv_xw, b);
    fv_pred = op_add->make_output();

    op_mse = new MeanSquareErrorOp(fv_pred, y);
    fv_loss = op_mse->make_output();

    graph = graph_create(fv_loss);
}

linear_model::~linear_model() {
    graph_free(graph);
    delete op_matmul;
    delete op_add;
    delete op_mse;
    delete W;
    delete b;
    delete X;
    delete y;
    delete fv_xw;
    delete fv_pred;
    delete fv_loss;
}

void linear_model::forward() {
    op_matmul->forward(fv_xw);
    op_add->forward(fv_pred);
    op_mse->forward(fv_loss);
}

// ── nn_model ─────────────────────────────────────────────────────────────────

nn_model::nn_model(Tensor *val_X, Tensor *val_y,
                   const std::vector<u32> &layer_sizes) {}

nn_model::~nn_model() {
    graph_free(graph);
    for (auto *op : op_matmul)
        delete op;
    for (auto *op : op_add)
        delete op;
    for (auto *op : op_relu)
        delete op;
    delete op_loss;
    for (auto *fv : W)
        delete fv;
    for (auto *fv : b)
        delete fv;
    for (auto *fv : z)
        delete fv;
    for (auto *fv : a)
        delete fv;
    delete fv_loss;
    delete X;
    delete y;
}

void nn_model::forward() {}
