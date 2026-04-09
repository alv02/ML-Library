#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor *val_X = tensor_load("data/X.npy", false);
    Tensor *val_y = tensor_load("data/y.npy", false);

    linear_model model = linear_model(val_X, val_y);
    gd_optimizer optim = gd_optimizer(0.1f);
    optim.set_graph(model.graph);

    for (int epoch = 0; epoch < 1000; epoch++) {
        model.forward();
        graph_backward(model.graph);
        optim.step();
        if (epoch % 100 == 0) {
            printf("Epoch %d loss: %f\n", epoch, model.fv_loss->val->data[0]);
        }
        optim.zero_grad();
    }
}
