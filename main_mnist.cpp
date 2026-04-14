#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {

    Tensor *val_X = tensor_load("data/X_train.npy", true);
    Tensor *val_y = tensor_load("data/y_train.npy", true);
    tensor_print(val_X);

    nn_model model(val_X, val_y, {256, 128, 10});

    gd_optimizer optim(0.1f);
    optim.set_graph(model.graph);

    for (int epoch = 0; epoch < 1000; epoch++) {
        model.forward();
        graph_backward(model.graph);
        optim.step();
        optim.zero_grad();

        if (epoch % 2 == 0) {
            Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
            printf("Epoch %d loss: %f\n", epoch, loss_cpu->data[0]);
            delete loss_cpu;
        }
    }

    delete val_X;
    delete val_y;
    return 0;
}
