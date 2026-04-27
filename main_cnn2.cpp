#include "include/metrics.hpp"
#include "include/models.hpp"
#include "include/optimizers.hpp"
#include "include/tensor.hpp"
#include "include/visualize.hpp"
#include <cstdio>

int main() {
    CudaMemArena perm_arena(MiB(256));
    CudaMemArena batch_arena(GiB(2));

    Tensor val_X = tensor_load("data/X_train.npy", true);
    Tensor val_y = tensor_load("data/y_train.npy", true);
    Tensor test_val_X = tensor_load("data/X_test.npy", true);
    Tensor test_val_y = tensor_load("data/y_test.npy", true);

    tensor_print(val_X.impl());

    // Conv(3→32, k=3,p=1) + MaxPool(2,2)  → [N,32,16,16]
    // Conv(32→64,k=3,p=1) + MaxPool(2,2)  → [N,64,8,8]
    // Conv(64→128,k=3,p=1)+ MaxPool(2,2)  → [N,128,4,4]
    // Flatten                              → [N,2048]
    // Dense: 2048 → 512 → 256 → 10
    cnn_model model(
        3, 32, 32, true,
        {
            {32, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2)},
            {64, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2)},
            {128, Unfold2dParams(3, 1, 1), true, Unfold2dParams(2, 2)},
        },
        {512, 256, 10}, &perm_arena);

    sgd optim(model.parameters(), 0.005f, 5e-4f, 0.9f, &perm_arena);
    DataLoader loader(val_X, val_y, 64);

    for (int epoch = 0; epoch < 60; epoch++) {
        loader.shuffle();
        Tensor Xb, yb;
        while (true) {
            cuda_arena_clear(&batch_arena);
            if (!loader.next(Xb, yb, &batch_arena))
                break;
            Var loss = model.forward(Var(Xb), Var(yb), &batch_arena);
            backward(loss, &batch_arena);
            optim.step(&batch_arena);
            optim.zero_grad();
        }
    }

    DataLoader test_loader(test_val_X, test_val_y, 256);
    f32 total_loss = 0.0f, total_acc = 0.0f;
    u32 n_batches = 0;

    Tensor Xb_test, yb_test;
    while (true) {
        cuda_arena_clear(&batch_arena);
        if (!test_loader.next(Xb_test, yb_test, &batch_arena))
            break;
        Var loss = model.forward(Var(Xb_test), Var(yb_test), &batch_arena);
        Var logits = model.predict(Var(Xb_test), &batch_arena);
        Tensor lc = tensor_to_cpu(loss->data);
        total_loss += lc->data()[0];
        total_acc += accuracy(logits->data, yb_test);
        n_batches++;
    }
    printf("\nTest loss:     %.4f\n", total_loss / n_batches);
    printf("Test accuracy: %.2f%%\n", total_acc / n_batches * 100.0f);

    {
        cuda_arena_clear(&batch_arena);
        DataLoader vis_loader(test_val_X, test_val_y, 256);
        Tensor vis_X, vis_y;
        vis_loader.next(vis_X, vis_y, &batch_arena);
        Var vis_logits = model.predict(Var(vis_X), &batch_arena);
        printf("\n--- Wrong predictions ---\n");
        visualize_wrong(vis_X, vis_logits->data, vis_y, 5);
        printf("\n--- Correct predictions ---\n");
        visualize_correct(vis_X, vis_logits->data, vis_y, 3);
    }

    return 0;
}
