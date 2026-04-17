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

void linear_model::forward(const Tensor *val_X, const Tensor *val_y) {
    if (val_X) {
        tensor_realloc(X->val, val_X->shape, val_X->ndim);
        tensor_copy(X->val, val_X);
    }
    if (val_y) {
        tensor_realloc(y->val, val_y->shape, val_y->ndim);
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
        u32 w_shape[2] = {out_features, in_features};
        u32 b_shape[2] = {1, out_features};

        Tensor *w = new Tensor(2, w_shape, on_gpu);
        tensor_transpose(w, COL_DIM(w), ROW_DIM(w));
        Wt.push_back(
            new function_var(w, FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        b.push_back(
            new function_var(new Tensor(2, b_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        MatMulOp *mm = new MatMulOp(cur, Wt[l]);
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
    for (auto *_w : Wt) {
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
    for (auto *fv : Wt)
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

// ── cnn_model
// ─────────────────────────────────────────────────────────────────

cnn_model::cnn_model(Tensor *val_X, Tensor *val_y,
                     const std::vector<conv_layer_params> &conv_layers,
                     const std::vector<u32> &dense_sizes) {
    b32 on_gpu = val_X->on_gpu;

    X = new function_var(tensor_create_like(val_X), FV_FLAG_NONE);
    tensor_copy(X->val, val_X);
    y = new function_var(tensor_create_like(val_y), FV_FLAG_NONE);
    tensor_copy(y->val, val_y);

    u32 C_in = val_X->shape[1];
    function_var *cur = X;

    for (u32 l = 0; l < (u32)conv_layers.size(); l++) {
        const auto &spec = conv_layers[l];
        u32 kH = spec.params.k_h;
        u32 kW = spec.params.k_w;
        u32 C_out = spec.C_out;

        u32 kernel_shape[2] = {C_in * kH * kW, C_out};
        kernels.push_back(
            new function_var(new Tensor(2, kernel_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        u32 b_shape[4] = {1, C_out, 1, 1};
        conv_b.push_back(
            new function_var(new Tensor(4, b_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        Conv2dOp *conv = new Conv2dOp(cur, kernels[l], spec.params);
        op_conv.push_back(conv);
        conv_out.push_back(conv->make_output());

        AddOp *add_b = new AddOp(conv_out[l], conv_b[l]);
        op_conv_add.push_back(add_b);
        conv_biased.push_back(add_b->make_output());

        ReluOp *relu = new ReluOp(conv_biased[l]);
        op_conv_relu.push_back(relu);
        conv_relu.push_back(relu->make_output());

        cur = conv_relu[l];
        C_in = C_out;
    }

    op_flatten = new FlattenOp(cur);
    fv_flat = op_flatten->make_output();
    cur = fv_flat;

    u32 in_features = fv_flat->val->shape[1];
    for (u32 l = 0; l < (u32)dense_sizes.size(); l++) {
        u32 out_features = dense_sizes[l];
        u32 w_shape[2] = {out_features, in_features};
        u32 b_shape[2] = {1, out_features};

        Tensor *w = new Tensor(2, w_shape, on_gpu);
        tensor_transpose(w, COL_DIM(w), ROW_DIM(w));
        dense_Wt.push_back(
            new function_var(w, FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));
        dense_b.push_back(
            new function_var(new Tensor(2, b_shape, on_gpu),
                             FV_FLAG_REQUIERES_GRAD | FV_FLAG_PARAMETER));

        MatMulOp *mm = new MatMulOp(cur, dense_Wt[l]);
        op_matmul.push_back(mm);
        z.push_back(mm->make_output());

        AddOp *add = new AddOp(z[l], dense_b[l]);
        op_add.push_back(add);

        bool is_last = (l == dense_sizes.size() - 1);
        if (!is_last) {
            zb.push_back(add->make_output());
            ReluOp *relu = new ReluOp(zb[l]);
            op_relu.push_back(relu);
            a.push_back(relu->make_output());
        } else {
            a.push_back(add->make_output());
        }
        cur = a[l];
        in_features = out_features;
    }

    op_loss = new CrossEntropyWithLogitsOp(a[dense_sizes.size() - 1], y);
    fv_loss = op_loss->make_output();
    graph = graph_create(fv_loss);

    for (auto *w : kernels)
        tensor_he_init(w->val);
    for (auto *w : dense_Wt)
        tensor_he_init(w->val);
}

cnn_model::~cnn_model() {
    graph_free(graph);
    for (auto *op : op_conv)
        delete op;
    for (auto *op : op_conv_add)
        delete op;
    for (auto *op : op_conv_relu)
        delete op;
    delete op_flatten;
    for (auto *op : op_matmul)
        delete op;
    for (auto *op : op_add)
        delete op;
    for (auto *op : op_relu)
        delete op;
    delete op_loss;
    for (auto *fv : kernels)
        delete fv;
    for (auto *fv : conv_b)
        delete fv;
    for (auto *fv : dense_Wt)
        delete fv;
    for (auto *fv : dense_b)
        delete fv;
    for (auto *fv : conv_out)
        delete fv;
    for (auto *fv : conv_biased)
        delete fv;
    for (auto *fv : conv_relu)
        delete fv;
    delete fv_flat;
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

void cnn_model::forward(const Tensor *val_X, const Tensor *val_y) {
    if (val_X) {
        tensor_realloc(X->val, val_X->shape, val_X->ndim);
        tensor_copy(X->val, val_X);
    }
    if (val_y) {
        tensor_realloc(y->val, val_y->shape, val_y->ndim);
        tensor_copy(y->val, val_y);
    }
    u32 n_conv = (u32)op_conv.size();
    for (u32 l = 0; l < n_conv; l++) {
        op_conv[l]->forward(conv_out[l]);
        op_conv_add[l]->forward(conv_biased[l]);
        op_conv_relu[l]->forward(conv_relu[l]);
    }
    op_flatten->forward(fv_flat);
    u32 n_dense = (u32)op_matmul.size();
    for (u32 l = 0; l < n_dense; l++) {
        op_matmul[l]->forward(z[l]);
        bool is_last = (l == n_dense - 1);
        if (!is_last) {
            op_add[l]->forward(zb[l]);
            op_relu[l]->forward(a[l]);
        } else {
            op_add[l]->forward(a[l]);
        }
    }
    op_loss->forward(fv_loss);
}

void nn_model::forward(const Tensor *val_X, const Tensor *val_y) {
    if (val_X) {
        tensor_realloc(X->val, val_X->shape, val_X->ndim);
        tensor_copy(X->val, val_X);
    }
    if (val_y) {
        tensor_realloc(y->val, val_y->shape, val_y->ndim);
        tensor_copy(y->val, val_y);
    }
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
