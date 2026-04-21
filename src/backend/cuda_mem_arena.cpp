#include "../../include/backend/cuda_mem_arena.hpp"
#include <cuda_runtime.h>

CudaMemArena::CudaMemArena(u64 capacity) {
    this->capacity = capacity;
    this->pos = 0;
    cudaMalloc(&this->base, capacity);
}

CudaMemArena::~CudaMemArena() { cudaFree(this->base); }

void *cuda_arena_push(CudaMemArena *arena, u64 size) {
    u64 start = ALIGN_UP_POW2(arena->pos, ALIGN);
    u64 end = start + size;
    if (end > arena->capacity) {
        printf("No memory in arena\n");
        return nullptr;
    }
    arena->pos = end;
    return arena->base + start;
}

void cuda_arena_pop(CudaMemArena *arena, u64 size) {
    size = std::min(size, arena->pos);
    arena->pos -= size;
}
void cuda_arena_pop_to(CudaMemArena *arena, u64 pos) {
    u64 size = pos >= arena->pos ? 0 : arena->pos - pos;
    cuda_arena_pop(arena, size);
}
void cuda_arena_clear(CudaMemArena *arena) {
    cudaStreamSynchronize(0);
    cuda_arena_pop_to(arena, 0);
}
