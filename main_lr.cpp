#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    CudaMemArena perm_arena(MiB(16));
    CudaMemArena batch_arena(MiB(256));

    Tensor val_X = tensor_load("data/X.npy", true);
    Tensor val_y = tensor_load("data/y.npy", true);

    linear_model model(val_X->shape[1], val_X->on_gpu(), &perm_arena);
    sgd optim(model.parameters(), 0.1f, 0.0f, 0.0f, &perm_arena);

    DataLoader loader(val_X, val_y, val_X->shape[0]);

    for (int epoch = 0; epoch < 1000; epoch++) {
        loader.shuffle();
        Tensor Xb, yb;
        while (true) {
            cuda_arena_clear(&batch_arena);
            if (!loader.next(Xb, yb, &batch_arena))
                break;
            Var loss = model.forward(Var(Xb), Var(yb), &batch_arena);
            backward(loss, &batch_arena);
            optim.step(&batch_arena);
            optim.zero_grad();
        }
    }
}
