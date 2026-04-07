#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor *val_X = tensor_load("data/X.npy", false);
    Tensor *val_y = tensor_load("data/y.npy", false);

    gd_optimizer optim = gd_optimizer(0.1f);
    linear_model model = linear_model(val_X, val_y);
    for (int epoch = 0; epoch < 1000; epoch++) {

        model.forward();
        Graph *graph = graph_create(model.fv_loss);
        optim.set_graph(graph);

        graph_backward(graph);
        optim.step();

        optim.zero_grad();

        if (epoch % 100 == 0) {
            printf("Epoch %d loss: %f\n", epoch, model.fv_loss->val->data[0]);
        }
    }
}
