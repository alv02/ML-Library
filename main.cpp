#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor *val_X = tensor_load("data/X.npy", false);
    Tensor *val_y = tensor_load("data/y.npy", false);
    printf("X[0]: ");
    for (int i = 0; i < 8; i++)
        printf("%f ", val_X->data[i]);
    printf("\ny[0]: %f\n", val_y->data[0]);

    gd_optimizer optim = gd_optimizer(0.1f);
    linear_model model = linear_model(val_X, val_y);
    for (int epoch = 0; epoch < 1000; epoch++) {

        model.forward();
        Graph *graph = graph_create(model.fv_loss);
        optim.set_graph(graph);

        graph_backward(graph);
        optim.step();
        if (epoch % 100 == 0) {
            printf("Epoch %d loss: %f\n", epoch, model.fv_loss->val->data[0]);
            printf("W: ");
            for (u32 i = 0; i < 8; i++)
                printf("%f ", model.W->val->data[i]);
            printf("\nb: %f\n", model.b->val->data[0]);
            printf("W grads: ");
            for (int i = 0; i < 8; i++)
                printf("%f ", model.W->grad->data[i]);
            printf("\n");
        }
        optim.zero_grad();
    }
}
