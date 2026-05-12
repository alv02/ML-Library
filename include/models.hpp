#ifndef MODELS_HPP
#define MODELS_HPP

#include "layers.hpp"
#include <vector>

struct conv_layer_params {
    u32 C_out;
    Unfold2dParams params;
    bool pool = false;
    Unfold2dParams pool_params = {};
    bool bn = false;
};

// Linear → (ReLU → Linear) × (n-1). Last layer has no activation.
Sequential make_mlp(u32 in_features, const std::vector<u32> &sizes, bool on_gpu,
                    CudaMemArena *perm_arena = nullptr);

// (Conv2d → [BatchNorm2d] → ReLU → [MaxPool2d]) × N → Reshape
// → (Linear → ReLU) × (n-1) → Linear
Sequential make_cnn(u32 C_in, u32 H, u32 W, bool on_gpu,
                    const std::vector<conv_layer_params> &conv_layers,
                    const std::vector<u32> &dense_sizes,
                    CudaMemArena *perm_arena = nullptr);

// ResNet for CIFAR-10 (input [N, 3, 32, 32]).
// stage_blocks: number of ResBlocks per stage, e.g. {2,2,2,2} for ResNet-18.
// Channels per stage are fixed at {64, 128, 256, 512}.
// Architecture: Conv→BN→ReLU → 4 stages of ResBlocks → GlobalAvgPool → Linear
Sequential make_resnet(u32 num_classes, bool on_gpu,
                       const std::vector<u32> &stage_blocks = {2, 2, 2, 2},
                       CudaMemArena *perm_arena = nullptr);

#endif
