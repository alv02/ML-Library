#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include "tensor.hpp"

// Print one example: ANSI-colored image + per-class softmax bar + true label.
// image must be [1, C, H, W] — any device.
void visualize_example(const Tensor<f32> &image, const Tensor<f32> &logits,
                       const Tensor<f32> &target);

// Find and print up to n_examples correctly classified samples.
// images must be [N, C, H, W].
void visualize_correct(const Tensor<f32> &images, const Tensor<f32> &logits,
                       const Tensor<f32> &targets, u32 n_examples);

// Find and print up to n_examples misclassified samples.
// images must be [N, C, H, W].
void visualize_wrong(const Tensor<f32> &images, const Tensor<f32> &logits,
                     const Tensor<f32> &targets, u32 n_examples);

#endif
