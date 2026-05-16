#include "../include/visualize.hpp"
#include <cstdio>
#include <vector>

static constexpr u32 BAR_W = 20;

static void draw_image(const f32 *data, u32 h, u32 w, u32 channels) {
    for (u32 y = 0; y < h; y++) {
        for (u32 x = 0; x < w; x++) {
            if (channels == 1) {
                f32 val = fminf(fmaxf(data[y * w + x], 0.0f), 1.0f);
                u32 col = 232 + (u32)(val * 23);
                printf("\x1b[48;5;%dm  ", col);
            } else {
                u32 r = (u32)(fminf(fmaxf(data[0 * h * w + y * w + x], 0.0f),
                                    1.0f) *
                              255);
                u32 g = (u32)(fminf(fmaxf(data[1 * h * w + y * w + x], 0.0f),
                                    1.0f) *
                              255);
                u32 b = (u32)(fminf(fmaxf(data[2 * h * w + y * w + x], 0.0f),
                                    1.0f) *
                              255);
                printf("\x1b[48;2;%d;%d;%dm  ", r, g, b);
            }
        }
        printf("\x1b[0m\n");
    }
    printf("\x1b[0m");
}

static void print_predictions(const f32 *prob_row, u32 pred, u32 truth,
                              u32 n_classes) {
    printf("--- Predictions ---\n");
    for (u32 c = 0; c < n_classes; c++) {
        float prob = prob_row[c];
        u32 filled = (u32)(prob * BAR_W);
        printf("[%u] %5.1f%% [", c, prob * 100.0f);
        for (u32 i = 0; i < BAR_W; i++)
            printf("%c", i < filled ? '#' : ' ');
        printf("]");
        if (c == pred)
            printf(" <- PREDICTED");
        if (c == truth)
            printf(" <- TRUE");
        printf("\n");
    }
}

void visualize_example(const Tensor<f32> &image, const Tensor<f32> &logits,
                       const Tensor<f32> &target) {
    Tensor<f32> prob = tensor_softmax(logits);
    Tensor<f32> img_cpu = tensor_to_cpu(image);
    Tensor<f32> prob_cpu = tensor_to_cpu(prob);

    u32 channels = img_cpu->shape[img_cpu->ndim - 3];
    u32 h = img_cpu->shape[img_cpu->ndim - 2];
    u32 w = img_cpu->shape[img_cpu->ndim - 1];
    u32 n_classes = prob_cpu->shape[1];

    TensorU32 pred_t = tensor_to_cpu(tensor_argmax(logits, 1));
    TensorU32 truth_t = tensor_to_cpu(tensor_argmax(target, 1));
    u32 pred = pred_t->data()[0];
    u32 truth = truth_t->data()[0];

    draw_image(img_cpu->data(), h, w, channels);
    print_predictions(prob_cpu->data(), pred, truth, n_classes);
    printf("Result: %s\n\n", pred == truth ? "CORRECT" : "WRONG");
}

static std::vector<u32> collect_indices(const Tensor<f32> &images,
                                        const Tensor<f32> &logits,
                                        const Tensor<f32> &targets,
                                        u32 n_examples, b32 want_correct) {
    TensorU32 pred_cpu = tensor_to_cpu(tensor_argmax(logits, 1, false));
    TensorU32 truth_cpu = tensor_to_cpu(tensor_argmax(targets, 1, false));
    u32 n = pred_cpu->shape[0];

    std::vector<u32> indices;
    for (u32 i = 0; i < n && indices.size() < n_examples; i++)
        if ((pred_cpu->data()[i] == truth_cpu->data()[i]) == (bool)want_correct)
            indices.push_back(i);
    return indices;
}

// Build a [1, d1, d2, ...] index tensor that selects row `idx` along dim 0,
// broadcasting the single value across all remaining dimensions.
static TensorU32 row_index(u32 idx, const Tensor<f32> &ref) {
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];
    shape[0] = 1;
    stride[0] = 1;
    for (u32 i = 1; i < ref->ndim; i++) {
        shape[i] = ref->shape[i];
        stride[i] = 0;
    }
    u32 one = 1;
    TensorU32 t = TensorU32::make(1, &one, ref->on_gpu());
    t->data()[0] = idx;
    t->ndim = ref->ndim;
    memcpy(t->shape, shape, ref->ndim * sizeof(u32));
    memcpy(t->stride, stride, ref->ndim * sizeof(u64));
    return t;
}

void visualize_correct(const Tensor<f32> &images, const Tensor<f32> &logits,
                       const Tensor<f32> &targets, u32 n_examples) {
    Tensor<f32> imgs = tensor_to_cpu(images);
    Tensor<f32> logs = tensor_to_cpu(logits);
    Tensor<f32> tgts = tensor_to_cpu(targets);
    for (u32 idx : collect_indices(imgs, logs, tgts, n_examples, true)) {
        Tensor<f32> img = tensor_gather(imgs, row_index(idx, imgs), 0);
        Tensor<f32> log = tensor_gather(logs, row_index(idx, logs), 0);
        Tensor<f32> tgt = tensor_gather(tgts, row_index(idx, tgts), 0);
        printf("===== Sample %u =====\n", idx);
        visualize_example(img, log, tgt);
    }
}

void visualize_wrong(const Tensor<f32> &images, const Tensor<f32> &logits,
                     const Tensor<f32> &targets, u32 n_examples) {
    Tensor<f32> imgs = tensor_to_cpu(images);
    Tensor<f32> logs = tensor_to_cpu(logits);
    Tensor<f32> tgts = tensor_to_cpu(targets);
    for (u32 idx : collect_indices(imgs, logs, tgts, n_examples, false)) {
        Tensor<f32> img = tensor_gather(imgs, row_index(idx, imgs), 0);
        Tensor<f32> log = tensor_gather(logs, row_index(idx, logs), 0);
        Tensor<f32> tgt = tensor_gather(tgts, row_index(idx, tgts), 0);
        printf("===== Sample %u =====\n", idx);
        visualize_example(img, log, tgt);
    }
}
