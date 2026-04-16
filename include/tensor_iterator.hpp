#ifndef TENSOR_ITERATOR_HPP
#define TENSOR_ITERATOR_HPP
#include "tensor.hpp"

// Multi-dimensional iterator that walks a tensor in row-major order and returns
// the flat memory offset for each element. Works with arbitrary strides, so it
// handles non-contiguous tensors (transposed, sliced) and broadcasting.
//
// Broadcasting via stride=0: if a dimension's stride is 0 the iterator adds 0
// to the offset every time that axis advances, so the same memory address is
// returned for all indices in that dimension — no data copy needed.
//
// Usage:
//   tensorIterator it(t->ndim, t->shape, t->stride);
//   while (it.has_next())
//       process(t->data[it.next()]);
struct tensorIterator {
    u32 ndim;              // number of dimensions
    u64 remaining;         // elements left to yield (decremented by next())
    u32 counter[MAX_NDIM]; // current multi-dim index (odometer digits)
    u32 shape[MAX_NDIM];   // size of each dimension
    u64 stride[MAX_NDIM];  // stride per dimension (0 = broadcast that dim)

    // Initializes the iterator; remaining = product of all shape values, counter = all zeros.
    tensorIterator(u32 ndim, const u32 *shape, const u64 *stride);
    ~tensorIterator();
    // Returns the flat offset of the current element, then advances the counter.
    // Counter increments like an odometer: last dim first, carry to previous dim
    // when it reaches shape[i], reset to 0 and continue.
    u64 next();
    // Returns true while there are elements left to yield.
    b32 has_next();
};

#endif // !TENSOR_ITERATOR_HPP
