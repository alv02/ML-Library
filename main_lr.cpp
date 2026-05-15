#include "include/layers.hpp"
#include "include/ops.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"

int main() {
    Tensor<f32> val_X = tensor_load("data/X.npy", true);
    Tensor<f32> val_y = tensor_load("data/y.npy", true);

    Linear model(val_X->shape[1], 1, val_X->on_gpu());
    sgd optim(model, 0.1f);

    DataLoader loader(val_X, val_y, val_X->shape[0]);

    for (int epoch = 0; epoch < 1000; epoch++) {
        loader.shuffle();
        Tensor<f32> Xb, yb;
        while (true) {
            if (!loader.next(Xb, yb))
                break;
            Var pred = model(Var(Xb));
            Var loss = mse_loss(pred, Var(yb));
            backward(loss);
            optim.step();
            optim.zero_grad();
        }
    }
}
