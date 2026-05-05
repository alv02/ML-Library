#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

// VGG-B with BatchNorm on CIFAR-10. Input [N,3,32,32].
//   Block 1: Conv(3→64)+BN+ReLU,   Conv(64→64)+BN+ReLU,   MaxPool →
//   [N,64,16,16] Block 2: Conv(64→128)+BN+ReLU, Conv(128→128)+BN+ReLU, MaxPool
//   → [N,128,8,8] Block 3: Conv(128→256)+BN+ReLU,Conv(256→256)+BN+ReLU, MaxPool
//   → [N,256,4,4] Flatten → [N,4096]   Dense: 4096 → 512 → 10
// Expected test accuracy: ~85-88% with 100 epochs.

int main() {
    CudaMemArena perm_arena(MiB(512));
    CudaMemArena batch_arena(GiB(6));

    Tensor val_X = tensor_load("data/X_train.npy", true);
    Tensor val_y = tensor_load("data/y_train.npy", true);
    Tensor test_val_X = tensor_load("data/X_test.npy", true);
    Tensor test_val_y = tensor_load("data/y_test.npy", true);

    tensor_print(val_X.impl());

    cnn_model model(
        3, 32, 32, true,
        {
            // Block 1
            {64, Unfold2dParams(3, 1, 1), false, {}, true},
            {64, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
            // Block 2
            {128, Unfold2dParams(3, 1, 1), false, {}, true},
            {128, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
            // Block 3
            {256, Unfold2dParams(3, 1, 1), false, {}, true},
            {256, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
        },
        {512, 10}, &perm_arena);

    sgd optim(model.parameters(), 0.01f, 5e-4f, 0.9f, &perm_arena);

    const int epochs = 100;
    const int batch_size = 64;
    DataLoader loader(val_X, val_y, batch_size);

    u32 scalar_shape[1] = {1};
    Tensor loss_accum = Tensor::make(1, scalar_shape, true, &perm_arena);

    const f32 min_delta = 1e-4f;
    const int patience = 10;
    f32 best_loss = 1e9f;
    int no_improve = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // MultiStepLR: divide by 10 at epoch 30 and 60
        if (epoch == 29) optim.set_lr(0.001f);
        if (epoch == 59) optim.set_lr(0.0001f);

        tensor_fill(loss_accum, 0.0f);
        loader.shuffle();
        Tensor Xb, yb;
        int batch = 0;
        while (true) {
            cuda_arena_clear(&batch_arena);
            if (!loader.next(Xb, yb, &batch_arena))
                break;
            Var loss = model.forward(Var(Xb), Var(yb), &batch_arena);
            tensor_add(loss_accum, loss_accum, loss->data);
            batch++;
            backward(loss, &batch_arena);
            optim.step(&batch_arena);
            optim.zero_grad();
        }
        Tensor lc = tensor_to_cpu(loss_accum);
        f32 avg_loss = lc->data()[0] / batch;
        printf("Epoch %d/%d done — avg loss %.4f\n", epoch + 1, epochs, avg_loss);

        // Early stopping
        if (best_loss - avg_loss > min_delta) {
            best_loss = avg_loss;
            no_improve = 0;
        } else {
            no_improve++;
        }
        if (no_improve >= patience) {
            printf("Early stopping at epoch %d\n", epoch + 1);
            break;
        }
    }

    model.set_training(false);
    DataLoader test_loader(test_val_X, test_val_y, 256);
    tensor_fill(loss_accum, 0.0f);
    f32 total_acc = 0.0f;
    u32 n_batches = 0;

    Tensor Xb_test, yb_test;
    while (true) {
        cuda_arena_clear(&batch_arena);
        if (!test_loader.next(Xb_test, yb_test, &batch_arena))
            break;
        Var loss = model.forward(Var(Xb_test), Var(yb_test), &batch_arena);
        Var logits = model.predict(Var(Xb_test), &batch_arena);
        tensor_add(loss_accum, loss_accum, loss->data);
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    Tensor test_lc = tensor_to_cpu(loss_accum);
    printf("\nTest loss:     %.4f\n", test_lc->data()[0] / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    {
        cuda_arena_clear(&batch_arena);
        DataLoader vis_loader(test_val_X, test_val_y, 128);
        Tensor vis_X, vis_y;
        vis_loader.next(vis_X, vis_y, &batch_arena);
        Var vis_logits = model.predict(Var(vis_X), &batch_arena);
        printf("\n--- Wrong predictions ---\n");
        visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
        printf("\n--- Correct predictions ---\n");
        visualize_correct(vis_X, vis_logits->data, vis_y, 3);
    }

    return 0;
}
