#include "../include/models.hpp"

// ── linear_model ─────────────────────────────────────────────────────────────

linear_model::linear_model(const Tensor *val_X, const Tensor *val_y) {
    u32 n_features = val_X->shape[COL_DIM(val_X)];

    u32 w_shape[2] = {n_features, 1};
    W = new function_var(new Tensor(2, w_shape, val_X->on_gpu),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    u32 b_shape[1] = {1};
    b = new function_var(new Tensor(1, b_shape, val_X->on_gpu),
                         FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER);

    X = new function_var(tensor_create_like(val_X), FV_FLAG_NONE);
    tensor_copy(X->val, val_X);
    y = new function_var(tensor_create_like(val_y), FV_FLAG_NONE);
    tensor_copy(y->val, val_y);

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

void linear_model::forward(const Tensor *val_X, const Tensor *val_y, b32 copy) {
    if (copy) {
        tensor_copy(X->val, val_X);
        tensor_copy(y->val, val_y);
    }
    op_matmul->forward(fv_xw);
    op_add->forward(fv_pred);
    op_mse->forward(fv_loss);
}

// ── nn_model ─────────────────────────────────────────────────────────────────

nn_model::nn_model(Tensor *val_X, Tensor *val_y,
                   const std::vector<u32> &layer_sizes) {
    u32 in_features = val_X->shape[COL_DIM(val_X)];

    X = new function_var(tensor_create_like(val_X), FV_FLAG_NONE);
    tensor_copy(X->val, val_X);
    y = new function_var(tensor_create_like(val_y), FV_FLAG_NONE);
    tensor_copy(y->val, val_y);

    b32 on_gpu = X->val->on_gpu;
    function_var *cur = X;
    for (u32 l = 0; l < (u32)layer_sizes.size(); l++) {
        u32 out_features = layer_sizes[l];
        u32 w_shape[2] = {in_features, out_features};
        u32 b_shape[2] = {1, out_features};

        W.push_back(
            new function_var(new Tensor(2, w_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        b.push_back(
            new function_var(new Tensor(2, b_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        MatMulOp *mm = new MatMulOp(cur, W[l]);
        op_matmul.push_back(mm);
        z.push_back(op_matmul[l]->make_output());
        AddOp *add = new AddOp(z[l], b[l]);
        op_add.push_back(add);

        bool is_last = (l == layer_sizes.size() - 1);
        if (!is_last) {
            zb.push_back(op_add[l]->make_output());

            ReluOp *relu = new ReluOp(zb[l]);
            op_relu.push_back(relu);
            a.push_back(op_relu[l]->make_output());

        } else {
            a.push_back(op_add[l]->make_output());
        }
        cur = a[l];
        in_features = out_features;
    }
    op_loss = new CrossEntropyWithLogitsOp(a[layer_sizes.size() - 1], y);
    fv_loss = op_loss->make_output();
    graph = graph_create(fv_loss);

    // He initialization for Ws
    for (auto *_w : W) {
        tensor_he_init(_w->val);
    }
}

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
    for (auto *fv : zb)
        delete fv;
    for (auto *fv : a)
        delete fv;
    delete fv_loss;
    delete X;
    delete y;
}

void nn_model::forward() {
    u32 n_layers = (u32)op_matmul.size();
    for (u32 l = 0; l < n_layers; l++) {
        op_matmul[l]->forward(z[l]);
        bool is_last = (l == n_layers - 1);
        if (!is_last) {
            op_add[l]->forward(zb[l]);
            op_relu[l]->forward(a[l]);
        } else {
            op_add[l]->forward(a[l]); // logits — no relu on last layer
        }
    }
    op_loss->forward(fv_loss);
}
