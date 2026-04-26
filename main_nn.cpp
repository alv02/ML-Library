#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    Tensor *val_X = tensor_load("./data/mnist_X_train.npy", true);
    Tensor *val_y = tensor_load("./data/mnist_y_train.npy", true);
    Tensor *test_val_X = tensor_load("./data/mnist_X_test.npy", true);
    Tensor *test_val_y = tensor_load("./data/mnist_y_test.npy", true);

    u32 flat_dim = val_X->shape[1] * val_X->shape[2] * val_X->shape[3];

    u32 flat_train[2] = {val_X->shape[0], flat_dim};
    u32 flat_test[2] = {test_val_X->shape[0], flat_dim};

    u32 orig_train_shape[4] = {val_X->shape[0], val_X->shape[1],
                               val_X->shape[2], val_X->shape[3]};

    u32 orig_test_shape[4] = {test_val_X->shape[0], test_val_X->shape[1],
                              test_val_X->shape[2], test_val_X->shape[3]};
    tensor_reshape(val_X, flat_train, 2);
    tensor_reshape(test_val_X, flat_test, 2);
    tensor_print(val_X);

    // 3072 → 1024 → 512 → 256 → 10
    nn_model model(val_X, val_y, {1024, 512, 256, 10});

    sgd optim(0.01f, 1e-4f, 0.9f);
    optim.set_graph(model.graph);

    DataLoader loader(val_X, val_y, 128);

    for (int epoch = 0; epoch < 50; epoch++) {
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

        if (epoch % 5 == 0) {
            Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
            printf("Epoch %2d  loss: %.4f\n", epoch, loss_cpu->data[0]);
            delete loss_cpu;
        }
    }

    model.forward(test_val_X, test_val_y);
    Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
    printf("\nTest loss:     %.4f\n", loss_cpu->data[0]);
    delete loss_cpu;
    printf("Test accuracy: %.2f%%\n",
           accuracy(model.a.back()->val, test_val_y) * 100.0f);

    tensor_reshape(val_X, orig_train_shape, 4);
    tensor_reshape(test_val_X, orig_test_shape, 4);

    printf("\n--- Wrong predictions ---\n");
    visualize_wrong(test_val_X, model.a.back()->val, test_val_y, 5);

    printf("\n--- Correct predictions ---\n");
    visualize_correct(test_val_X, model.a.back()->val, test_val_y, 3);

    delete val_X;
    delete val_y;
    delete test_val_X;
    delete test_val_y;
    return 0;
}
