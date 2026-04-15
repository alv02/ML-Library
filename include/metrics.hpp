#ifndef METRICS_HPP
#define METRICS_HPP

#include "tensor.hpp"

// logits  : [N, C] — raw scores or probabilities
// targets : [N, C] — one-hot encoded ground truth
// Returns fraction of samples where argmax(logits) == argmax(targets).
f32 accuracy(const Tensor *logits, const Tensor *targets);

#endif
