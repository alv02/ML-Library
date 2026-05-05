#ifndef MODELS_HPP
#define MODELS_HPP
#include "autograd.hpp"
#include "ops.hpp"
#include <vector>

struct linear_model {
    Var W, b;

    linear_model(u32 n_features, bool on_gpu,
                 CudaMemArena *perm_arena = nullptr);
    std::vector<Var> parameters() const { return {W, b}; }
    Var predict(Var X, CudaMemArena *arena = nullptr);
    Var forward(Var X, Var y, CudaMemArena *arena = nullptr);
};

struct nn_model {
    std::vector<Var> Wt, b;

    nn_model(u32 in_features, const std::vector<u32> &layer_sizes, bool on_gpu,
             CudaMemArena *perm_arena = nullptr);
    std::vector<Var> parameters() const;
    Var predict(Var X, CudaMemArena *arena = nullptr);
    Var forward(Var X, Var y, CudaMemArena *arena = nullptr);
};

struct conv_layer_params {
    u32 C_out;
    Unfold2dParams params;
    bool pool = false;
    Unfold2dParams pool_params = {};
    bool bn = false;
};

struct cnn_model {
    std::vector<conv_layer_params> conv_specs;

    // Conv stage parameters
    std::vector<Var> kernels, conv_b;
    std::vector<Var> bn_gamma, bn_beta;         // one entry per BN layer
    std::vector<Tensor> bn_running_mean, bn_running_var; // [1,C,1,1], perm_arena

    // Dense stage parameters
    std::vector<Var> dense_Wt, dense_b;

    bool training = true;
    void set_training(bool t) { training = t; }

    // C_in, H, W: input spatial shape — needed to compute flat_features at init
    cnn_model(u32 C_in, u32 H, u32 W, bool on_gpu,
              const std::vector<conv_layer_params> &conv_layers,
              const std::vector<u32> &dense_sizes,
              CudaMemArena *perm_arena = nullptr);

    std::vector<Var> parameters() const;
    Var predict(Var X, CudaMemArena *arena = nullptr);
    Var forward(Var X, Var y, CudaMemArena *arena = nullptr);
};

#endif
