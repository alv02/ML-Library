#include "include/tensor.hpp"
#include "util/arena.h"

int main() {
    mem_arena *perm_arena = arena_create(GiB(1), MiB(1));

    Tensor *a = tensor_load(perm_arena, "data/a.npy", false);
    Tensor *b = tensor_load(perm_arena, "data/b.npy", false);

    tensor_transpose(a, 0, 1);
    tensor_transpose(b, 0, 1);

    u32 shape[2] = {a->shape[ROW_DIM(a)], b->shape[COL_DIM(b)]};

    Tensor *c = tensor_create(perm_arena, a->ndim, shape, false);

    tensor_print(c);
    tensor_mat_mul(c, a, b);
    tensor_print(c);

    arena_destroy(perm_arena);
}
