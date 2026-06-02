#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/ops.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

i32 main() {
    Tensor<f32> val_X      = tensor_load("data/X_train.npy", true);
    Tensor<f32> val_y      = tensor_load("data/y_train.npy", true);
    Tensor<f32> test_val_X = tensor_load("data/X_test.npy",  true);
    Tensor<f32> test_val_y = tensor_load("data/y_test.npy",  true);

    tensor_print(val_X.impl());

    // Conv(3→32,  k=3,s=1,p=1) → [N,32,32,32]
    // Conv(32→64, k=3,s=2,p=1) → [N,64,16,16]
    // Conv(64→128,k=3,s=2,p=1) → [N,128,8,8]
    // Conv(128→256,k=3,s=2,p=1)→ [N,256,4,4]
    // Flatten → [N,4096]
    // Dense: 4096 → 512 → 10
    Sequential model = make_cnn(3, 32, 32, true,
                    {
                        {32,  Unfold2dParams(3, 1, 1)},
                        {64,  Unfold2dParams(3, 2, 1)},
                        {128, Unfold2dParams(3, 2, 1)},
                        {256, Unfold2dParams(3, 2, 1)},
                    },
                    {512, 10});

    sgd optim(model, 0.01f, 1e-4f, 0.9f);
    DataLoader loader(val_X, val_y, 64);

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
    f32 total_loss = 0.0f, total_acc = 0.0f;
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

    {
        DataLoader vis_loader(test_val_X, test_val_y, 256);
        Tensor<f32> vis_X, vis_y;
        vis_loader.next(vis_X, vis_y);
        Var vis_logits = model(Var(vis_X));
        printf("\n--- Wrong predictions ---\n");
        visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
        printf("\n--- Correct predictions ---\n");
        visualize_correct(vis_X, vis_logits->data, vis_y, 3);
    }

    return 0;
}
