#include "include/augment.hpp"
#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/ops.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

// ResNet-18 on CIFAR-10. Input [N, 3, 32, 32].
// Architecture: Conv→BN→ReLU → 4×{2 ResBlocks} → GlobalAvgPool → Linear(512,10)
// Each ResBlock: Conv→BN→ReLU→Conv→BN + skip (identity or 1×1 projection)

int main() {
    Tensor<f32> train_X =
        tensor_load("data/X_train.npy", false); // CPU — augmented per batch
    Tensor<f32> train_y = tensor_load("data/y_train.npy", false);
    Tensor<f32> test_X =
        tensor_load("data/X_test.npy", true); // GPU — no augmentation
    Tensor<f32> test_y = tensor_load("data/y_test.npy", true);

    tensor_print(train_X.impl());

    // ResNet-18: 4 stages × 2 blocks each
    Sequential model = make_resnet(10, true, {2, 2, 2, 2});

    sgd optim(model, 0.1f, 1e-4f, 0.9f);
    MultiStepLR scheduler(optim, {50, 75}, 0.1f);
    EarlyStopping early_stop(15);

    const int epochs = 1;
    const int batch_size = 128;
    DataLoader loader(train_X, train_y, batch_size);
    Augmenter aug(loader);
    aug.add<RandomCrop>(32, 4);
    aug.add<RandomHorizontalFlip>(0.5f);

    u32 scalar_shape[1] = {1};
    Tensor<f32> loss_accum = Tensor<f32>::make(1, scalar_shape, true);

    for (int epoch = 0; epoch < epochs; epoch++) {
        tensor_fill(loss_accum, 0.0f);
        aug.shuffle();
        Tensor<f32> Xb, yb;
        int batch = 0;
        while (true) {
            if (!aug.next(Xb, yb))
                break;
            Var logits = model(Var(Xb));
            Var loss = cross_entropy_with_logits(logits, Var(yb));
            tensor_add(loss_accum, loss_accum, loss->data);
            batch++;
            backward(loss);
            optim.step();
            optim.zero_grad();
        }
        Tensor<f32> lc = tensor_to_cpu(loss_accum);
        f32 avg_loss = lc->data()[0] / batch;
        printf("Epoch %d/%d — avg loss %.4f\n", epoch + 1, epochs, avg_loss);

        scheduler.step(epoch);
        if (early_stop.step(avg_loss, epoch))
            break;
    }

    model.eval();
    DataLoader test_loader(test_X, test_y, 256);
    tensor_fill(loss_accum, 0.0f);
    f32 total_acc = 0.0f;
    u32 n_batches = 0;

    Tensor<f32> Xb_test, yb_test;
    while (true) {
        if (!test_loader.next(Xb_test, yb_test))
            break;
        Var logits = model(Var(Xb_test));
        Var loss = cross_entropy_with_logits(logits, Var(yb_test));
        tensor_add(loss_accum, loss_accum, loss->data);
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    Tensor<f32> test_lc = tensor_to_cpu(loss_accum);
    printf("\nTest loss:     %.4f\n", test_lc->data()[0] / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    {
        DataLoader vis_loader(test_X, test_y, 128);
        Tensor<f32> vis_X, vis_y;
        vis_loader.next(vis_X, vis_y);
        Var vis_logits = model(Var(vis_X));
        printf("\n--- Wrong predictions ---\n");
        visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
        printf("\n--- Correct predictions ---\n");
        visualize_correct(vis_X, vis_logits->data, vis_y, 3);
    }

    return 0;
}
