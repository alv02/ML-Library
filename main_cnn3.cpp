#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

// VGG-B with BatchNorm on CIFAR-10.
// Architecture (input [N,3,32,32]):
//   Block 1: Conv(3→64,k=3,p=1)+BN+ReLU, Conv(64→64,k=3,p=1)+BN+ReLU,
//   MaxPool(2,2)  → [N,64,16,16] Block 2: Conv(64→128,k=3,p=1)+BN+ReLU,
//   Conv(128→128,k=3,p=1)+BN+ReLU, MaxPool(2,2) → [N,128,8,8] Block 3:
//   Conv(128→256,k=3,p=1)+BN+ReLU, Conv(256→256,k=3,p=1)+BN+ReLU, MaxPool(2,2)
//   → [N,256,4,4] Flatten → [N,4096] Dense: 4096 → 512 → 10
// Expected test accuracy: ~85-88% with 100 epochs.

int main() {
    Tensor *val_X = tensor_load("data/X_train.npy", true);
    Tensor *val_y = tensor_load("data/y_train.npy", true);
    Tensor *test_val_X = tensor_load("data/X_test.npy", true);
    Tensor *test_val_y = tensor_load("data/y_test.npy", true);

    tensor_print(val_X);

    cnn_model model(
        val_X, val_y,
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
        {512, 10});

    sgd optim(0.01f, 5e-4f, 0.9f);
    optim.set_graph(model.graph);

    const int epochs = 1;
    const int batch_size = 128;
    DataLoader loader(val_X, val_y, batch_size);

    for (int epoch = 0; epoch < epochs; epoch++) {
        loader.shuffle();
        Tensor *Xb, *yb;
        while (loader.next(Xb, yb)) {
            model.forward(Xb, yb);
            graph_backward(model.graph);
            optim.step();
            optim.zero_grad();
            delete Xb;
            delete yb;
        }

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
            printf("Epoch %3d  loss: %.4f\n", epoch, loss_cpu->data[0]);
            delete loss_cpu;
        }
    }

    DataLoader test_loader(test_val_X, test_val_y, 256);
    f32 total_loss = 0.0f, total_acc = 0.0f;
    u32 n_batches = 0;
    Tensor *Xb_test = nullptr, *yb_test = nullptr;
    Tensor *vis_X = nullptr, *vis_y = nullptr;
    while (test_loader.next(Xb_test, yb_test)) {
        model.forward(Xb_test, yb_test);
        Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
        total_loss += loss_cpu->data[0];
        delete loss_cpu;
        total_acc += accuracy(model.a.back()->val, yb_test);
        n_batches++;
        delete vis_X;
        delete vis_y;
        vis_X = Xb_test;
        vis_y = yb_test;
    }
    printf("\nTest loss:     %.4f\n", total_loss / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    printf("\n--- Wrong predictions ---\n");
    visualize_wrong(vis_X, model.a.back()->val, vis_y, 5);
    printf("\n--- Correct predictions ---\n");
    visualize_correct(vis_X, model.a.back()->val, vis_y, 3);

    delete vis_X;
    delete vis_y;
    delete val_X;
    delete val_y;
    delete test_val_X;
    delete test_val_y;
    return 0;
}
