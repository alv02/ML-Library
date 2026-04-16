#include "../include/tensor_iterator.hpp"

// Copies shape and strides; initializes counter to all zeros and remaining to
// the total number of elements (product of shape). Strides may be 0 to indicate
// a broadcast dimension.
tensorIterator::tensorIterator(u32 ndim, const u32 *shape, const u64 *stride) {
    this->ndim = ndim;
    this->remaining = 1;

    for (u32 i = 0; i < ndim; i++) {
        this->shape[i] = shape[i];
        this->stride[i] = stride[i];
        this->counter[i] = 0;
        this->remaining *= shape[i];
    }
}
tensorIterator::~tensorIterator() {}

// Computes the flat memory offset for the current counter, decrements remaining,
// then advances the counter like an odometer (last dim increments first; when a
// dim overflows shape[i] it resets to 0 and the carry propagates to the left).
// Dims with stride=0 always contribute 0 to the offset regardless of counter,
// which implements broadcasting — the same element is returned for all indices.
u64 tensorIterator::next() {
    u64 offset = 0;
    remaining--;
    for (u32 i = 0; i < ndim; i++) {
        offset += counter[i] * stride[i];
    }
    for (u32 i = ndim; i-- > 0;) {
        counter[i] += 1;
        if (counter[i] < shape[i]) {
            break;
        }
        // Reset
        counter[i] = 0;
    }

    return offset;
}

b32 tensorIterator::has_next() { return remaining > 0; }
