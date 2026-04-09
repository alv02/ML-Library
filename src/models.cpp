#include "../include/models.hpp"
#include "../include/ops.hpp"

linear_model::linear_model(Tensor *val_X, Tensor *val_y) {
    u32 n_features = val_X->shape[COL_DIM(val_X)];

    u32 w_shape[2] = {n_features, 1};
    W = new function_var(new Tensor(2, w_shape, false),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    u32 b_shape[1] = {1};
    b = new function_var(new Tensor(1, b_shape, false),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    X = new function_var(val_X, FV_FLAG_NONE);
    y = new function_var(val_y, FV_FLAG_NONE);

    // Each op infers its output shape and flags from its inputs
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
