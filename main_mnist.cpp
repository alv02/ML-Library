#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    Tensor *val_X = tensor_load("data/X_train.npy", true);
    Tensor *val_y = tensor_load("data/y_train.npy", true);
    Tensor *test_val_X = tensor_load("./data/X_test.npy", true);
    Tensor *test_val_y = tensor_load("./data/y_test.npy", true);
    tensor_print(val_X);

    nn_model model(val_X, val_y, {256, 128, 10});

    sgd optim(0.1f);
    optim.set_graph(model.graph);

    DataLoader loader(val_X, val_y, 256);

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

    model.forward(test_val_X, test_val_y);
    Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
    printf("Test loss: %f\n", loss_cpu->data[0]);
    delete loss_cpu;
    printf("Test accuracy: %.2f%%\n", accuracy(model.a.back()->val, test_val_y) * 100.0f);

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
