#include "../include/augment.hpp"
#include <cstdlib>

Tensor RandomHorizontalFlip::apply(const Tensor &batch) {
    const TensorImpl &src = *batch.impl_;
    u32 N = src.shape[0], C = src.shape[1], H = src.shape[2], W = src.shape[3];

    Tensor out = Tensor::make(4, src.shape, false);
    TensorImpl &dst = *out.impl_;

    for (u32 n = 0; n < N; n++) {
        bool flip = ((float)rand() / RAND_MAX) < p;
        for (u32 c = 0; c < C; c++)
            for (u32 h = 0; h < H; h++)
                for (u32 w = 0; w < W; w++)
                    dst(n, c, h, w) = src(n, c, h, flip ? W - 1 - w : w);
    }
    return out;
}

Tensor RandomCrop::apply(const Tensor &batch) {
    const TensorImpl &src = *batch.impl_;
    u32 N = src.shape[0], C = src.shape[1], H = src.shape[2], W = src.shape[3];

    u32 out_shape[4] = {N, C, size, size};
    Tensor out = Tensor::make(4, out_shape, false);
    TensorImpl &dst = *out.impl_;

    u32 max_top  = H + 2 * padding - size;
    u32 max_left = W + 2 * padding - size;

    for (u32 n = 0; n < N; n++) {
        u32 top  = rand() % (max_top + 1);
        u32 left = rand() % (max_left + 1);

        for (u32 c = 0; c < C; c++) {
            for (u32 h = 0; h < size; h++) {
                for (u32 w = 0; w < size; w++) {
                    i32 h_src = (i32)(h + top)  - (i32)padding;
                    i32 w_src = (i32)(w + left) - (i32)padding;

                    float val = 0.0f;
                    if (h_src >= 0 && (u32)h_src < H && w_src >= 0 && (u32)w_src < W)
                        val = src(n, c, (u32)h_src, (u32)w_src);
                    dst(n, c, h, w) = val;
                }
            }
        }
    }
    return out;
}

bool Augmenter::next(Tensor &X_batch, Tensor &y_batch, CudaMemArena *arena) {
    if (!loader.next(X_batch, y_batch, nullptr))
        return false;

    for (auto &t : transforms)
        X_batch = t->apply(X_batch);

    if (arena) {
        X_batch = tensor_to_gpu(X_batch, arena);
        y_batch = tensor_to_gpu(y_batch, arena);
    }
    return true;
}
