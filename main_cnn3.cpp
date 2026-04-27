#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

// VGG-B with BatchNorm on CIFAR-10. Input [N,3,32,32].
//   Block 1: Conv(3→64)+BN+ReLU,   Conv(64→64)+BN+ReLU,   MaxPool → [N,64,16,16]
//   Block 2: Conv(64→128)+BN+ReLU, Conv(128→128)+BN+ReLU, MaxPool → [N,128,8,8]
//   Block 3: Conv(128→256)+BN+ReLU,Conv(256→256)+BN+ReLU, MaxPool → [N,256,4,4]
//   Flatten → [N,4096]   Dense: 4096 → 512 → 10
// Expected test accuracy: ~85-88% with 100 epochs.

int main() {
    Tensor val_X      = tensor_load("data/X_train.npy", true);
    Tensor val_y      = tensor_load("data/y_train.npy", true);
    Tensor test_val_X = tensor_load("data/X_test.npy",  true);
    Tensor test_val_y = tensor_load("data/y_test.npy",  true);

    tensor_print(val_X.impl());

    cnn_model model(3, 32, 32, true,
                    {
                        // Block 1
                        {64,  Unfold2dParams(3, 1, 1), false, {},                  true},
                        {64,  Unfold2dParams(3, 1, 1), true,  Unfold2dParams(2,2), true},
                        // Block 2
                        {128, Unfold2dParams(3, 1, 1), false, {},                  true},
                        {128, Unfold2dParams(3, 1, 1), true,  Unfold2dParams(2,2), true},
                        // Block 3
                        {256, Unfold2dParams(3, 1, 1), false, {},                  true},
                        {256, Unfold2dParams(3, 1, 1), true,  Unfold2dParams(2,2), true},
                    },
                    {512, 10});

    sgd optim(model.parameters(), 0.01f, 5e-4f, 0.9f);

    const int epochs     = 1;
    const int batch_size = 128;
    DataLoader loader(val_X, val_y, batch_size);

    Var last_loss;
    for (int epoch = 0; epoch < epochs; epoch++) {
        loader.shuffle();
        Tensor Xb, yb;
        while (loader.next(Xb, yb)) {
            last_loss = model.forward(Var(Xb), Var(yb));
            backward(last_loss);
            optim.step();
            optim.zero_grad();
        }
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            Tensor loss_cpu = tensor_to_cpu(last_loss->data);
            printf("Epoch %3d  loss: %.4f\n", epoch, loss_cpu->data()[0]);
        }
    }

    DataLoader test_loader(test_val_X, test_val_y, 256);
    f32 total_loss = 0.0f, total_acc = 0.0f;
    u32 n_batches = 0;
    Tensor vis_X, vis_y;
    Var vis_logits;

    Tensor Xb_test, yb_test;
    while (test_loader.next(Xb_test, yb_test)) {
        Var loss   = model.forward(Var(Xb_test), Var(yb_test));
        Var logits = model.predict(Var(Xb_test));
        Tensor lc  = tensor_to_cpu(loss->data);
        total_loss += lc->data()[0];
        total_acc  += accuracy(logits->data, yb_test);
        n_batches++;
        vis_X      = Xb_test;
        vis_y      = yb_test;
        vis_logits = logits;
    }
    printf("\nTest loss:     %.4f\n", total_loss / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    printf("\n--- Wrong predictions ---\n");
    visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
    printf("\n--- Correct predictions ---\n");
    visualize_correct(vis_X, vis_logits->data, vis_y, 3);

    return 0;
}
