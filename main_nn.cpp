#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    CudaMemArena perm_arena(MiB(64));
    CudaMemArena batch_arena(GiB(1));

    Tensor val_X = tensor_load("./data/cifar_X_train.npy", true);
    Tensor val_y = tensor_load("./data/cifar_y_train.npy", true);
    Tensor test_val_X = tensor_load("./data/cifar_X_test.npy", true);
    Tensor test_val_y = tensor_load("./data/cifar_y_test.npy", true);

    u32 flat_dim = val_X->shape[1] * val_X->shape[2] * val_X->shape[3];
    u32 flat_train[2] = {val_X->shape[0], flat_dim};
    u32 flat_test[2] = {test_val_X->shape[0], flat_dim};

    u32 orig_train_shape[4] = {val_X->shape[0], val_X->shape[1],
                               val_X->shape[2], val_X->shape[3]};
    u32 orig_test_shape[4] = {test_val_X->shape[0], test_val_X->shape[1],
                              test_val_X->shape[2], test_val_X->shape[3]};

    tensor_reshape(val_X, flat_train, 2);
    tensor_reshape(test_val_X, flat_test, 2);
    tensor_print(val_X.impl());

    // 784 → 1024 → 512 → 256 → 10
    nn_model model(flat_dim, {1024, 512, 256, 10}, true, &perm_arena);
    sgd optim(model.parameters(), 0.01f, 1e-4f, 0.9f, &perm_arena);
    DataLoader loader(val_X, val_y, 128);

    for (int epoch = 0; epoch < 50; epoch++) {
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

    DataLoader test_loader(test_val_X, test_val_y, 256);
    f32 total_acc = 0.0f, total_loss = 0.0f;
    u32 n_batches = 0;
    Tensor Xb_test, yb_test;
    while (true) {
        cuda_arena_clear(&batch_arena);
        if (!test_loader.next(Xb_test, yb_test, &batch_arena))
            break;
        Var logits = model.predict(Var(Xb_test), &batch_arena);
        Var loss = model.forward(Var(Xb_test), Var(yb_test), &batch_arena);
        Tensor lc = tensor_to_cpu(loss->data);
        total_loss += lc->data()[0];
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    printf("\nTest loss:     %.4f\n", total_loss / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    tensor_reshape(val_X, orig_train_shape, 4);
    tensor_reshape(test_val_X, orig_test_shape, 4);

    return 0;
}
