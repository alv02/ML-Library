#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include "tensor.hpp"

// Print one example: ANSI-colored image + per-class softmax bar + true label.
// image must be [1, C, H, W] — any device.
void visualize_example(const Tensor *image, const Tensor *logits,
                       const Tensor *target);

// Find and print up to n_examples correctly classified samples.
// images must be [N, C, H, W].
void visualize_correct(const Tensor *images, const Tensor *logits,
                       const Tensor *targets, u32 n_examples);

// Find and print up to n_examples misclassified samples.
// images must be [N, C, H, W].
void visualize_wrong(const Tensor *images, const Tensor *logits,
                     const Tensor *targets, u32 n_examples);

#endif // VISUALIZE_HPP
