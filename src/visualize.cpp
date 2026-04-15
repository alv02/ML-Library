#include "../include/visualize.hpp"
#include <cstdio>
#include <vector>

static constexpr u32 BAR_W = 20;

// ---- internal helpers ----------------------------------------------------

static void draw_image(const f32 *data, u32 h, u32 w) {
    for (u32 y = 0; y < h; y++) {
        for (u32 x = 0; x < w; x++) {
            f32 val = data[x + y * w];
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            u32 col = 232 + (u32)(val * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\x1b[0m\n");
    }
    printf("\x1b[0m");
}

static void print_predictions(const f32 *prob_row, u32 pred, u32 truth,
                              u32 n_classes) {
    printf("--- Predictions ---\n");
    for (u32 c = 0; c < n_classes; c++) {
        float prob   = prob_row[c];
        u32   filled = (u32)(prob * BAR_W);

        printf("[%u] %5.1f%% [", c, prob * 100.0f);
        for (u32 i = 0; i < BAR_W; i++)
            printf("%c", i < filled ? '#' : ' ');
        printf("]");

        if (c == pred)  printf(" <- PREDICTED");
        if (c == truth) printf(" <- TRUE");
        printf("\n");
    }
}

// ---- public API ----------------------------------------------------------

void visualize_example(const Tensor *image, const Tensor *logits,
                       const Tensor *target, u32 img_h, u32 img_w) {
    Tensor *prob     = tensor_softmax(logits);
    Tensor *img_cpu  = tensor_to_cpu(image);
    Tensor *prob_cpu = tensor_to_cpu(prob);
    Tensor *tgt_cpu  = tensor_to_cpu(target);
    delete prob;

    u32 n_classes = prob_cpu->shape[1];
    u32 pred  = (u32)tensor_to_cpu(tensor_argmax(logits, 1))->data[0];
    u32 truth = (u32)tensor_to_cpu(tensor_argmax(target, 1))->data[0];

    draw_image(img_cpu->data, img_h, img_w);
    print_predictions(prob_cpu->data, pred, truth, n_classes);
    printf("Result: %s\n\n", pred == truth ? "CORRECT" : "WRONG");

    delete img_cpu; delete prob_cpu; delete tgt_cpu;
}

static std::vector<u32> collect_indices(const Tensor *images, const Tensor *logits,
                                        const Tensor *targets, u32 n_examples,
                                        b32 want_correct) {
    Tensor *pred_t  = tensor_argmax(logits,  1);
    Tensor *truth_t = tensor_argmax(targets, 1);
    Tensor *mask    = tensor_equal(pred_t, truth_t);
    delete pred_t; delete truth_t;

    Tensor *mask_cpu = tensor_to_cpu(mask);
    delete mask;

    std::vector<u32> indices;
    for (u32 i = 0; i < mask_cpu->shape[0] && indices.size() < n_examples; i++)
        if ((mask_cpu->data[i] != 0.0f) == (bool)want_correct)
            indices.push_back(i);

    delete mask_cpu;
    return indices;
}

void visualize_correct(const Tensor *images, const Tensor *logits,
                       const Tensor *targets, u32 n_examples,
                       u32 img_h, u32 img_w) {
    for (u32 idx : collect_indices(images, logits, targets, n_examples, true)) {
        Tensor *img = tensor_index_select(images,  &idx, 1, 0);
        Tensor *log = tensor_index_select(logits,  &idx, 1, 0);
        Tensor *tgt = tensor_index_select(targets, &idx, 1, 0);
        printf("===== Sample %u =====\n", idx);
        visualize_example(img, log, tgt, img_h, img_w);
        delete img; delete log; delete tgt;
    }
}

void visualize_wrong(const Tensor *images, const Tensor *logits,
                     const Tensor *targets, u32 n_examples,
                     u32 img_h, u32 img_w) {
    for (u32 idx : collect_indices(images, logits, targets, n_examples, false)) {
        Tensor *img = tensor_index_select(images,  &idx, 1, 0);
        Tensor *log = tensor_index_select(logits,  &idx, 1, 0);
        Tensor *tgt = tensor_index_select(targets, &idx, 1, 0);
        printf("===== Sample %u =====\n", idx);
        visualize_example(img, log, tgt, img_h, img_w);
        delete img; delete log; delete tgt;
    }
}
