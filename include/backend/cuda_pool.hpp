#ifndef CUDA_POOL_HPP
#define CUDA_POOL_HPP

#include "../base.hpp"
#include <map>

// GPU caching allocator — cudaFree is never called on individual tensors.
// Freed blocks go into a free list keyed by size and are reused on the next
// allocation of the same size. Call empty_cache() to actually release memory.
//
// ML training is highly repetitive in tensor shapes, so the exact-size free
// list achieves near-100% hit rate after the first step.
struct CudaCachingAllocator {
    std::multimap<u64, void *> free_blocks; // size → ptr
    u64 allocated = 0;                      // bytes currently held by live tensors
    u64 cached = 0;                         // bytes sitting in the free list

    void *alloc(u64 nbytes);
    void free(void *ptr, u64 nbytes);

    // Actually release all cached blocks back to the CUDA driver.
    void empty_cache();

    void print_stats() const;
};

extern CudaCachingAllocator g_cuda_pool;

#endif
