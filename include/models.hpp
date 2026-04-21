#ifndef MODELS_HPP
#define MODELS_HPP
#include "autograd.hpp"
#include "ops.hpp"

struct linear_model {
    // Parameters
    function_var *W;
    function_var *b;

    // Inputs and targets
    function_var *X;
    function_var *y;

    // Intermediate variables — allocated once, reused every forward
    function_var *fv_xw;
    function_var *fv_pred;
    function_var *fv_loss;

    // Ops — created once, owned by model
    MatMulOp *op_matmul;
    AddOp *op_add;
    MeanSquareErrorOp *op_mse;

    // Graph — built once in constructor, owned by model
    Graph *graph;

    linear_model(const Tensor *val_X, const Tensor *val_y);
    ~linear_model();
    void forward(const Tensor *val_X = nullptr, const Tensor *val_y = nullptr);
};

struct nn_model {
    // Parameters
    std::vector<function_var *> Wt;
    std::vector<function_var *> b;

    // Inputs and targets
    function_var *X;
    function_var *y;

    // Hidden layers variables
    std::vector<function_var *> z;  // pre-activation: X @ W
    std::vector<function_var *> zb; // pre-activation: z + b
    std::vector<function_var *> a;  // post-activation: relu(zb)

    function_var *fv_loss;

    // Hidden layers ops
    std::vector<MatMulOp *> op_matmul;
    std::vector<AddOp *> op_add;
    std::vector<ReluOp *> op_relu;

    CrossEntropyWithLogitsOp *op_loss;

    // Graph
    Graph *graph;

    nn_model(Tensor *val_X, Tensor *val_y, const std::vector<u32> &layer_sizes);
    ~nn_model();
    void forward(const Tensor *val_X = nullptr, const Tensor *val_y = nullptr);
};

struct conv_layer_params {
    u32 C_out;
    Unfold2dParams params;
    bool pool = false;
    Unfold2dParams pool_params = {};
    bool bn = false;
};

struct cnn_model {
    // Conv stage parameters
    std::vector<function_var *> kernels; // [C_in*kH*kW, C_out]
    std::vector<function_var *> conv_b;  // [1, C_out, 1, 1]

    // Dense stage parameters
    std::vector<function_var *> dense_Wt;
    std::vector<function_var *> dense_b;

    // Inputs / targets
    function_var *X;
    function_var *y;

    // Conv stage intermediates
    std::vector<function_var *> conv_out;    // after Conv2dOp
    std::vector<function_var *> conv_biased; // after bias add
    std::vector<function_var *> conv_relu;   // after relu

    // BatchNorm parameters (nullptr for layers without BN)
    std::vector<function_var *> bn_gamma; // [C_out]
    std::vector<function_var *> bn_beta;  // [C_out]

    // Conv stage intermediates (BN)
    std::vector<function_var *> bn_out; // nullptr for layers without BN

    // Flatten
    FlattenOp *op_flatten;
    function_var *fv_flat;

    // Dense stage intermediates
    std::vector<function_var *> z;
    std::vector<function_var *> zb;
    std::vector<function_var *> a;

    // Conv ops
    std::vector<Conv2dOp *> op_conv;
    std::vector<AddOp *> op_conv_add;
    std::vector<BatchNormOp *> op_bn; // nullptr for layers without BN
    std::vector<ReluOp *> op_conv_relu;
    std::vector<MaxPool2dOp *> op_pool;
    std::vector<function_var *> pool_out; // nullptr for layers without pooling

    // Dense ops
    std::vector<MatMulOp *> op_matmul;
    std::vector<AddOp *> op_add;
    std::vector<ReluOp *> op_relu;

    CrossEntropyWithLogitsOp *op_loss;
    function_var *fv_loss;
    Graph *graph;

    cnn_model(Tensor *val_X, Tensor *val_y,
              const std::vector<conv_layer_params> &conv_layers,
              const std::vector<u32> &dense_sizes);
    ~cnn_model();
    void forward(const Tensor *val_X = nullptr, const Tensor *val_y = nullptr);
};

#endif
