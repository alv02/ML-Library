#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor val_X = tensor_load("data/X.npy", true);
    Tensor val_y = tensor_load("data/y.npy", true);

    linear_model model(val_X->shape[1], val_X->on_gpu());
    sgd optim(model.parameters(), 0.1f);

    DataLoader loader(val_X, val_y, val_X->shape[0]);

    Var last_loss;
    for (int epoch = 0; epoch < 1000; epoch++) {
        loader.shuffle();
        Tensor Xb, yb;
        while (loader.next(Xb, yb)) {
            last_loss = model.forward(Var(Xb), Var(yb));
            backward(last_loss);
            optim.step();
            optim.zero_grad();
        }
        if (epoch % 100 == 0) {
            Tensor loss_cpu = tensor_to_cpu(last_loss->data);
            printf("Epoch %d loss: %f\n", epoch, loss_cpu->data()[0]);
        }
    }
}
