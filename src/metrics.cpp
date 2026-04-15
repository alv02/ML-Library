#include "../include/metrics.hpp"

f32 accuracy(const Tensor *logits, const Tensor *targets) {
    // argmax along class dim → [N] index tensors (on same device as inputs)
    Tensor *pred = tensor_argmax(logits,  1, false);
    Tensor *tgt  = tensor_argmax(targets, 1, false);

    // copy to CPU for the scalar comparison (only N floats, cheap)
    Tensor *pred_cpu = tensor_to_cpu(pred);
    Tensor *tgt_cpu  = tensor_to_cpu(tgt);
    delete pred;
    delete tgt;

    u32 n = pred_cpu->shape[0];
    u32 correct = 0;
    for (u32 i = 0; i < n; i++)
        correct += ((u32)pred_cpu->data[i] == (u32)tgt_cpu->data[i]);

    delete pred_cpu;
    delete tgt_cpu;

    return (f32)correct / (f32)n;
}
