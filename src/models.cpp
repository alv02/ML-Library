#include "../include/models.hpp"
#include "../include/ops.hpp"

linear_model::linear_model(Tensor *val_X, Tensor *val_y) {
    u32 n_features = val_X->shape[COL_DIM(val_X)];

    u32 w_shape[2] = {n_features, 1};
    Tensor *val_W = tensor_create(2, w_shape, false);
    W = fv_create(val_W, FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    // b is shape (1,1), not (n_features, 1)
    u32 b_shape[2] = {1, 1};
    Tensor *val_b = tensor_create(2, b_shape, false);
    b = fv_create(val_b, FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    X = fv_create(val_X, FV_FLAG_NONE);
    y = fv_create(val_y, FV_FLAG_NONE);
    fv_xw = nullptr;
    fv_pred = nullptr;
    fv_loss = nullptr;
}
linear_model::~linear_model() {
    fv_free(W);
    fv_free(b);
    fv_free(X);
    fv_free(y);
    fv_free(fv_xw);
    fv_free(fv_pred);
    fv_free(fv_loss);
}

function_var *linear_model::forward() {
    if (fv_loss)
        fv_free(fv_loss);
    if (fv_pred)
        fv_free(fv_pred);
    if (fv_xw)
        fv_free(fv_xw);

    // X @ W  →  (N, 1)
    fv_xw = (new MatMulOp(X, W))->forward();

    // (X @ W) + b  →  (N, 1),  b is (1, 1) and broadcasts
    fv_pred = (new AddOp(fv_xw, b))->forward();

    fv_loss = (new MeanSquareErrorOp(fv_pred, y))->forward();
    return fv_loss;
}
