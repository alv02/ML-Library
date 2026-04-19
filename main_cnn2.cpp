#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    Tensor *val_X      = tensor_load("data/X_train.npy", true);
    Tensor *val_y      = tensor_load("data/y_train.npy", true);
    Tensor *test_val_X = tensor_load("data/X_test.npy",  true);
    Tensor *test_val_y = tensor_load("data/y_test.npy",  true);

    // [N, 784] → [N, 1, 28, 28]
    u32 train_shape[4] = {val_X->shape[0], 1, 28, 28};
    u32 test_shape[4]  = {test_val_X->shape[0], 1, 28, 28};
    tensor_reshape(val_X, train_shape, 4);
    tensor_reshape(test_val_X, test_shape, 4);

    tensor_print(val_X);

    // Conv(32,3,1,1) + ReLU + MaxPool(2,2) → [N,32,14,14]
    // Conv(64,3,1,1) + ReLU + MaxPool(2,2) → [N,64,7,7]
    // Flatten                               → [N,3136]
    // Dense: 3136 → 256 → 128 → 10
    cnn_model model(
        val_X, val_y,
        {
            {32, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2)},
            {64, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2)},
        },
        {256, 128, 10});

    sgd optim(0.01f, 1e-4f);
    optim.set_graph(model.graph);

    DataLoader loader(val_X, val_y, 64);

    for (int epoch = 0; epoch < 30; epoch++) {
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

        if (epoch % 2 == 0) {
            Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
            printf("Epoch %2d  loss: %.4f\n", epoch, loss_cpu->data[0]);
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
