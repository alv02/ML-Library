#ifndef AUGMENT_HPP
#define AUGMENT_HPP

#include "optimizers.hpp"
#include "tensor.hpp"
#include <memory>
#include <vector>

// Transforms operate on CPU float32 tensors of shape [N, C, H, W].
// Each sample in the batch gets an independent random transform.

struct Transform {
    virtual Tensor apply(const Tensor &batch) = 0;
    virtual ~Transform() = default;
};

// Flip each image horizontally with probability p.
struct RandomHorizontalFlip : Transform {
    float p;
    RandomHorizontalFlip(float p = 0.5f) : p(p) {}
    Tensor apply(const Tensor &batch) override;
};

// Zero-pad by `padding` pixels on each side, then take a random size×size crop.
// Crop origin is chosen independently per sample.
struct RandomCrop : Transform {
    u32 size, padding;
    RandomCrop(u32 size, u32 padding) : size(size), padding(padding) {}
    Tensor apply(const Tensor &batch) override;
};

// Wraps a CPU DataLoader: applies transforms to each batch, then uploads to GPU.
struct Augmenter {
    DataLoader &loader;
    std::vector<std::unique_ptr<Transform>> transforms;

    Augmenter(DataLoader &loader) : loader(loader) {}

    template <typename T, typename... Args>
    Augmenter &add(Args &&...args) {
        transforms.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return *this;
    }

    void shuffle() { loader.shuffle(); }

    // Gets next CPU batch, runs transforms, uploads X and y to GPU via arena.
    bool next(Tensor &X_batch, Tensor &y_batch, CudaMemArena *arena = nullptr);
};

#endif
