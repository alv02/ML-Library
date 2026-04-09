#include "../include/tensor_iterator.hpp"

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
