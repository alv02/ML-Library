#ifndef CUDA_ARENA_HPP
#define CUDA_ARENA_HPP

#include "../base.hpp"

#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define ALIGN 256 // Align to 256 bytes
struct CudaMemArena {
    u8 *base;
    u64 capacity;
    u64 pos;

    CudaMemArena(u64 capacity);
    ~CudaMemArena();
};

void *cuda_arena_push(CudaMemArena *arena, u64 size);
void cuda_arena_pop(CudaMemArena *arena, u64 size);
void cuda_arena_pop_to(CudaMemArena *arena, u64 pos);
void cuda_arena_clear(CudaMemArena *arena);

#endif
