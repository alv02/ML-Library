#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    Tensor *val_X = tensor_load("data/X_train.npy", true);
    Tensor *val_y = tensor_load("data/y_train.npy", true);
    Tensor *test_val_X = tensor_load("data/X_test.npy", true);
    Tensor *test_val_y = tensor_load("data/y_test.npy", true);

    // Reshape flat [N, 784] → [N, 1, 28, 28] for conv layers
    u32 train_shape[4] = {val_X->shape[0], 1, 28, 28};
    u32 test_shape[4] = {test_val_X->shape[0], 1, 28, 28};
    tensor_reshape(val_X, train_shape, 4);
    tensor_reshape(test_val_X, test_shape, 4);

    tensor_print(val_X);

    // Conv1: [N,1,28,28] → [N,8,28,28]   (k=3, pad=1, stride=1)
    // Conv2: [N,8,28,28] → [N,16,13,13]  (k=3, stride=2)
    // Flatten → [N, 2704]
    // Dense: 128 → 10
    cnn_model model(val_X, val_y,
                    {{8, Conv2dParams(3, 1, 1)}, {16, Conv2dParams(3, 2, 0)}},
                    {128, 10});

    sgd optim(0.05f, 1e-4f);
    optim.set_graph(model.graph);

    DataLoader loader(val_X, val_y, 64);

    for (int epoch = 0; epoch < 20; epoch++) {
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
            printf("Epoch %d loss: %f\n", epoch, loss_cpu->data[0]);
            delete loss_cpu;
        }
    }

    // Batched test evaluation — avoids large intermediate tensors (col buffers
    // for 10k samples at once would be ~750 MB in the second conv layer).
    DataLoader test_loader(test_val_X, test_val_y, 512);
    f32 total_loss = 0.0f, total_acc = 0.0f;
    u32 n_test_batches = 0;
    Tensor *Xb_test = nullptr, *yb_test = nullptr;
    Tensor *vis_X = nullptr, *vis_y = nullptr;
    while (test_loader.next(Xb_test, yb_test)) {
        model.forward(Xb_test, yb_test);
        Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
        total_loss += loss_cpu->data[0];
        delete loss_cpu;
        total_acc += accuracy(model.a.back()->val, yb_test);
        n_test_batches++;
        delete vis_X;
        delete vis_y;
        vis_X = Xb_test; // keep last batch alive for visualization
        vis_y = yb_test;
    }
    printf("Test loss: %f\n", total_loss / n_test_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_test_batches * 100.0f);

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
