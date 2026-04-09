#ifndef TENSOR_ITERATOR_HPP
#define TENSOR_ITERATOR_HPP
#include "tensor.hpp"
struct tensorIterator {
    u32 ndim;
    u64 remaining;
    u32 counter[MAX_NDIM];
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];

    tensorIterator(u32 ndim, const u32 *shape, const u64 *stride);
    ~tensorIterator();
    u64 next();
    b32 has_next();
};

#endif // !TENSOR_ITERATOR_HPP
