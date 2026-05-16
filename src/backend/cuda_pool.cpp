#include "../../include/backend/cuda_pool.hpp"
#include <cuda_runtime.h>
#include <cstdio>

CudaCachingAllocator g_cuda_pool;

static u64 round_size(u64 n) {
    return (n + 255u) & ~255ULL; // 256-byte alignment
}

void *CudaCachingAllocator::alloc(u64 nbytes) {
    if (nbytes == 0)
        return nullptr;
    nbytes = round_size(nbytes);

    auto it = free_blocks.find(nbytes);
    if (it != free_blocks.end()) {
        void *ptr = it->second;
        free_blocks.erase(it);
        cached -= nbytes;
        allocated += nbytes;
        return ptr;
    }

    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) {
        empty_cache();
        err = cudaMalloc(&ptr, nbytes);
        if (err != cudaSuccess) {
            printf("CudaCachingAllocator: cudaMalloc(%llu MiB) failed: %s\n",
                   (unsigned long long)(nbytes >> 20),
                   cudaGetErrorString(err));
            return nullptr;
        }
    }
    allocated += nbytes;
    return ptr;
}

void CudaCachingAllocator::free(void *ptr, u64 nbytes) {
    if (!ptr)
        return;
    nbytes = round_size(nbytes);
    free_blocks.emplace(nbytes, ptr);
    allocated -= nbytes;
    cached += nbytes;
}

void CudaCachingAllocator::empty_cache() {
    for (auto &[size, ptr] : free_blocks)
        cudaFree(ptr);
    free_blocks.clear();
    cached = 0;
}

void CudaCachingAllocator::print_stats() const {
    printf("CudaCachingAllocator: live=%.1f MiB  cached=%.1f MiB  blocks=%zu\n",
           allocated / (1024.0 * 1024.0),
           cached / (1024.0 * 1024.0),
           free_blocks.size());
}
