#include "../include/metrics.hpp"

f32 accuracy(const Tensor<f32> &logits, const Tensor<f32> &targets) {
    TensorU32 pred_cpu = tensor_to_cpu(tensor_argmax(logits,   1, false));
    TensorU32 tgt_cpu  = tensor_to_cpu(tensor_argmax(targets,  1, false));

    u32 n = pred_cpu->shape[0];
    u32 correct = 0;
    for (u32 i = 0; i < n; i++)
        correct += (pred_cpu->data()[i] == tgt_cpu->data()[i]);

    return (f32)correct / (f32)n;
}
