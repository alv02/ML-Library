#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/ops.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

// VGG-13-style with BatchNorm on CIFAR-10. Input [N,3,32,32].
//   Block 1: Conv(3→64)+BN+ReLU,   Conv(64→64)+BN+ReLU,   MaxPool → [N,64,16,16]
//   Block 2: Conv(64→128)+BN+ReLU,  Conv(128→128)+BN+ReLU, MaxPool → [N,128,8,8]
//   Block 3: Conv(128→256)+BN+ReLU, Conv(256→256)+BN+ReLU, MaxPool → [N,256,4,4]
//   Block 4: Conv(256→512)+BN+ReLU, Conv(512→512)+BN+ReLU, MaxPool → [N,512,2,2]
//   Flatten → [N,2048]  Dense: 2048 → 512 → 10

int main() {
    CudaMemArena perm_arena(MiB(512));
    CudaMemArena batch_arena(GiB(6));

    Tensor<f32> val_X = tensor_load("data/X_train.npy", true);
    Tensor<f32> val_y = tensor_load("data/y_train.npy", true);
    Tensor<f32> test_val_X = tensor_load("data/X_test.npy", true);
    Tensor<f32> test_val_y = tensor_load("data/y_test.npy", true);

    tensor_print(val_X.impl());

    Sequential model = make_cnn(
        3, 32, 32, true,
        {
            // Block 1
            {64,  Unfold2dParams(3, 1, 1), false, {}, true},
            {64,  Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
            // Block 2
            {128, Unfold2dParams(3, 1, 1), false, {}, true},
            {128, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
            // Block 3
            {256, Unfold2dParams(3, 1, 1), false, {}, true},
            {256, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
            // Block 4
            {512, Unfold2dParams(3, 1, 1), false, {}, true},
            {512, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2), true},
        },
        {512, 10}, &perm_arena);

    sgd optim(model, 0.05f, 5e-4f, 0.9f, &perm_arena);
    ReduceLROnPlateau scheduler(optim, 0.1f, 5);
    EarlyStopping early_stop(10);

    const int epochs = 100;
    const int batch_size = 64;
    DataLoader loader(val_X, val_y, batch_size);

    u32 scalar_shape[1] = {1};
    Tensor<f32> loss_accum = Tensor<f32>::make(1, scalar_shape, true, &perm_arena);

    for (int epoch = 0; epoch < epochs; epoch++) {
        tensor_fill(loss_accum, 0.0f);
        loader.shuffle();
        Tensor<f32> Xb, yb;
        int batch = 0;
        while (true) {
            cuda_arena_clear(&batch_arena);
            if (!loader.next(Xb, yb, &batch_arena))
                break;
            Var logits = model(Var(Xb), &batch_arena);
            Var loss = cross_entropy_with_logits(logits, Var(yb), &batch_arena);
            tensor_add(loss_accum, loss_accum, loss->data);
            batch++;
            backward(loss, &batch_arena);
            optim.step(&batch_arena);
            optim.zero_grad();
        }
        Tensor<f32> lc = tensor_to_cpu(loss_accum);
        f32 avg_loss = lc->data()[0] / batch;
        printf("Epoch %d/%d done — avg loss %.4f\n", epoch + 1, epochs, avg_loss);

        scheduler.step(avg_loss, epoch);
        if (early_stop.step(avg_loss, epoch))
            break;
    }

    model.eval();
    DataLoader test_loader(test_val_X, test_val_y, 256);
    tensor_fill(loss_accum, 0.0f);
    f32 total_acc = 0.0f;
    u32 n_batches = 0;

    Tensor<f32> Xb_test, yb_test;
    while (true) {
        cuda_arena_clear(&batch_arena);
        if (!test_loader.next(Xb_test, yb_test, &batch_arena))
            break;
        Var logits = model(Var(Xb_test), &batch_arena);
        Var loss = cross_entropy_with_logits(logits, Var(yb_test), &batch_arena);
        tensor_add(loss_accum, loss_accum, loss->data);
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    Tensor<f32> test_lc = tensor_to_cpu(loss_accum);
    printf("\nTest loss:     %.4f\n", test_lc->data()[0] / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    {
        cuda_arena_clear(&batch_arena);
        DataLoader vis_loader(test_val_X, test_val_y, 128);
        Tensor<f32> vis_X, vis_y;
        vis_loader.next(vis_X, vis_y, &batch_arena);
        Var vis_logits = model(Var(vis_X), &batch_arena);
        printf("\n--- Wrong predictions ---\n");
        visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
        printf("\n--- Correct predictions ---\n");
        visualize_correct(vis_X, vis_logits->data, vis_y, 3);
    }

    return 0;
}
