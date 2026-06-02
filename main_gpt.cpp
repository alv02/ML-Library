#include "include/models.hpp"
#include "include/optimizers.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ── Config
// ────────────────────────────────────────────────────────────────────

static const u32 VOCAB_SIZE = 65;
static const u32 D_MODEL = 128;
static const u32 N_HEADS = 4;
static const u32 N_LAYERS = 4;
static const u32 MAX_SEQ_LEN = 128;
static const u32 BATCH_SIZE = 16;
static const u32 MAX_STEPS = 300;
static const u32 LOG_EVERY = 100;
static const f32 LR = 3e-4f;
static const f32 WEIGHT_DECAY = 0.1f;

// ── Helpers
// ───────────────────────────────────────────────────────────────────

static void make_batch(const u32 *all_tokens, u32 n_tokens, TensorU32 &inp,
                       TensorU32 &tgt, bool on_gpu) {
    u32 inp_shape[2] = {BATCH_SIZE, MAX_SEQ_LEN};
    u32 tgt_shape[1] = {BATCH_SIZE * MAX_SEQ_LEN};
    TensorU32 inp_cpu = TensorU32::make(2, inp_shape, false);
    TensorU32 tgt_cpu = TensorU32::make(1, tgt_shape, false);
    u32 *ip = inp_cpu->data(), *tp = tgt_cpu->data();
    for (u32 b = 0; b < BATCH_SIZE; b++) {
        u32 start = (u32)(rand() % (n_tokens - MAX_SEQ_LEN - 1));
        memcpy(ip + b * MAX_SEQ_LEN, all_tokens + start,
               MAX_SEQ_LEN * sizeof(u32));
        memcpy(tp + b * MAX_SEQ_LEN, all_tokens + start + 1,
               MAX_SEQ_LEN * sizeof(u32));
    }
    inp = on_gpu ? tensor_to_gpu(inp_cpu) : inp_cpu;
    tgt = on_gpu ? tensor_to_gpu(tgt_cpu) : tgt_cpu;
}

// ── main
// ──────────────────────────────────────────────────────────────────────

i32 main(i32 argc, char **argv) {
    bool on_gpu = true;
    for (i32 i = 1; i < argc; i++)
        if (strcmp(argv[i], "--cpu") == 0)
            on_gpu = false;

    srand(42);

    // Load tokens
    TensorU32 tokens_all =
        tensor_load<u32>("data/shakespeare/tokens.npy", false);
    if (!tokens_all) {
        fprintf(stderr, "Could not load data/shakespeare/tokens.npy\n"
                        "Run: python3 data/prepare_shakespeare.py\n");
        return 1;
    }
    u32 n_tokens = tokens_all->numel();
    const u32 *tok_ptr = tokens_all->data();
    printf("Loaded %u tokens, vocab %u\n", n_tokens, VOCAB_SIZE);

    GPTModel gpt(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, 0.0f,
                 on_gpu);
    AdamW opt(gpt.parameters(), LR, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY);

    printf("Parameters: %zu\n", gpt.parameters().size());
    printf("Training on %s\n\n", on_gpu ? "CUDA" : "CPU");

    for (u32 step = 1; step <= MAX_STEPS; step++) {
        TensorU32 inp, targets;
        make_batch(tok_ptr, n_tokens, inp, targets, on_gpu);

        // Forward
        Var logits = gpt.forward(inp); // [B, T, vocab]

        // Flatten to [B*T, vocab]
        u32 BT = BATCH_SIZE * MAX_SEQ_LEN;
        u32 flat_shape[2] = {BT, VOCAB_SIZE};
        Var logits_flat = reshape(logits, flat_shape, 2);

        Var loss = cross_entropy_with_logits(logits_flat, targets);

        // Backward + step
        backward(loss);
        opt.step();
        opt.zero_grad();

        if (step % LOG_EVERY == 0) {
            if (on_gpu)
                cudaDeviceSynchronize();
            f32 loss_val = tensor_to_cpu(loss->data)->data()[0];
            printf("step %4u / %u  loss %.4f\n", step, MAX_STEPS, loss_val);
            fflush(stdout);
        }
    }

    printf("\nDone.\n");
    return 0;
}
