#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include "tensor.hpp"

// Print one example: ANSI-colored image + per-class softmax bar + true label.
// All tensors are single-sample [1, ...] — any device.
// img_h / img_w default to 28 for MNIST.
void visualize_example(const Tensor *image, const Tensor *logits,
                       const Tensor *target,
                       u32 img_h = 28, u32 img_w = 28);

// Find and print up to n_examples correctly classified samples.
void visualize_correct(const Tensor *images, const Tensor *logits,
                       const Tensor *targets, u32 n_examples,
                       u32 img_h = 28, u32 img_w = 28);

// Find and print up to n_examples misclassified samples.
void visualize_wrong(const Tensor *images, const Tensor *logits,
                     const Tensor *targets, u32 n_examples,
                     u32 img_h = 28, u32 img_w = 28);

#endif // VISUALIZE_HPP
