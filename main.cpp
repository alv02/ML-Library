#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor *val_X = tensor_load("data/X.npy", true);
    Tensor *val_y = tensor_load("data/y.npy", true);

    linear_model model = linear_model(val_X, val_y);
    gd_optimizer optim = gd_optimizer(0.1f);
    optim.set_graph(model.graph);

    for (int epoch = 0; epoch < 1000; epoch++) {
        model.forward(val_X, val_y);
        graph_backward(model.graph);
        optim.step();
        if (epoch % 100 == 0) {
            Tensor *loss_cpu = tensor_to_cpu(model.fv_loss->val);
            printf("Epoch %d loss: %f\n", epoch, loss_cpu->data[0]);
            delete loss_cpu;
        }
        optim.zero_grad();
    }
}
