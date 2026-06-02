#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/ops.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

i32 main() {
    Tensor<f32> val_X = tensor_load("./data/cifar_X_train.npy", true);
    Tensor<f32> val_y = tensor_load("./data/cifar_y_train.npy", true);
    Tensor<f32> test_val_X = tensor_load("./data/cifar_X_test.npy", true);
    Tensor<f32> test_val_y = tensor_load("./data/cifar_y_test.npy", true);

    u32 flat_dim = val_X->shape[1] * val_X->shape[2] * val_X->shape[3];
    u32 flat_train[2] = {val_X->shape[0], flat_dim};
    u32 flat_test[2] = {test_val_X->shape[0], flat_dim};

    u32 orig_train_shape[4] = {val_X->shape[0], val_X->shape[1],
                               val_X->shape[2], val_X->shape[3]};
    u32 orig_test_shape[4] = {test_val_X->shape[0], test_val_X->shape[1],
                              test_val_X->shape[2], test_val_X->shape[3]};

    tensor_reshape(val_X, flat_train, 2);
    tensor_reshape(test_val_X, flat_test, 2);
    tensor_print(val_X.impl());

    // 784 → 1024 → 512 → 256 → 10
    Sequential model = make_mlp(flat_dim, {1024, 512, 256, 10}, true);
    sgd optim(model, 0.01f, 1e-4f, 0.9f);
    DataLoader loader(val_X, val_y, 128);

    for (i32 epoch = 0; epoch < 50; epoch++) {
        loader.shuffle();
        Tensor<f32> Xb, yb;
        while (true) {
            if (!loader.next(Xb, yb))
                break;
            Var logits = model(Var(Xb));
            Var loss = cross_entropy_with_logits(logits, Var(yb));
            backward(loss);
            optim.step();
            optim.zero_grad();
        }
    }

    model.eval();
    DataLoader test_loader(test_val_X, test_val_y, 256);
    f32 total_acc = 0.0f, total_loss = 0.0f;
    u32 n_batches = 0;
    Tensor<f32> Xb_test, yb_test;
    while (true) {
        if (!test_loader.next(Xb_test, yb_test))
            break;
        Var logits = model(Var(Xb_test));
        Var loss = cross_entropy_with_logits(logits, Var(yb_test));
        Tensor<f32> lc = tensor_to_cpu(loss->data);
        total_loss += lc->data()[0];
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    printf("\nTest loss:     %.4f\n", total_loss / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    tensor_reshape(val_X, orig_train_shape, 4);
    tensor_reshape(test_val_X, orig_test_shape, 4);

    return 0;
}
