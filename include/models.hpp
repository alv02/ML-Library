#ifndef MODELS_HPP
#define MODELS_HPP

#include "layers.hpp"
#include <vector>

// ── GPTModel ──────────────────────────────────────────────────────────────────
// vocab_size  : token vocabulary size
// d_model     : embedding / residual stream dimension
// n_heads     : attention heads (must divide d_model)
// n_layers    : number of TransformerBlocks
// max_seq_len : maximum sequence length (for causal mask and pos embeddings)
// dropout_p   : applied in embeddings, attention, and residual connections

struct GPTModel {
    InputEmbedding embedding;
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_f;
    Linear lm_head;
    bool training = true;

    GPTModel(u32 vocab_size, u32 d_model, u32 n_heads, u32 n_layers,
             u32 max_seq_len, f32 dropout_p, bool on_gpu);

    Var forward(TensorU32 tokens);
    std::vector<Var> parameters();
    void train(bool mode = true);
    void eval() { train(false); }
};

struct conv_layer_params {
    u32 C_out;
    Unfold2dParams params;
    bool pool = false;
    Unfold2dParams pool_params = {};
    bool bn = false;
};

// Linear → (ReLU → Linear) × (n-1). Last layer has no activation.
Sequential make_mlp(u32 in_features, const std::vector<u32> &sizes, bool on_gpu);

// (Conv2d → [BatchNorm2d] → ReLU → [MaxPool2d]) × N → Reshape
// → (Linear → ReLU) × (n-1) → Linear
Sequential make_cnn(u32 C_in, u32 H, u32 W, bool on_gpu,
                    const std::vector<conv_layer_params> &conv_layers,
                    const std::vector<u32> &dense_sizes);

// ResNet for CIFAR-10 (input [N, 3, 32, 32]).
// stage_blocks: number of ResBlocks per stage, e.g. {2,2,2,2} for ResNet-18.
// Channels per stage are fixed at {64, 128, 256, 512}.
// Architecture: Conv→BN→ReLU → 4 stages of ResBlocks → GlobalAvgPool → Linear
Sequential make_resnet(u32 num_classes, bool on_gpu,
                       const std::vector<u32> &stage_blocks = {2, 2, 2, 2});

#endif
