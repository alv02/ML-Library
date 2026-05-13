#include "../include/layers.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <unordered_set>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static b32 g_on_gpu = false;
static int passed = 0, failed = 0;

static void sync() {
    if (g_on_gpu) cudaDeviceSynchronize();
}

static void check(const char *name, bool ok) {
    printf("  [%s%s%s] %s\n", ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET, name);
    ok ? passed++ : failed++;
}

static void check_tensors(const char *name, const Tensor<f32> &got,
                           const Tensor<f32> &exp, f32 tol = 1e-5f) {
    if (!tensor_shape_eq(got, exp)) {
        printf("  [%sFAIL%s] %s — shape mismatch: got [", RED, RESET, name);
        for (u32 i = 0; i < got->ndim; i++)
            printf("%u%s", got->shape[i], i + 1 < got->ndim ? "," : "");
        printf("] expected [");
        for (u32 i = 0; i < exp->ndim; i++)
            printf("%u%s", exp->shape[i], i + 1 < exp->ndim ? "," : "");
        printf("]\n");
        failed++;
        return;
    }
    check(name, tensor_equals(got, exp, tol));
}

// Replicates PyTorch's out.backward(upstream_grad): full topological traversal
// starting from a non-scalar Var, using the given tensor as the initial gradient.
static void backward_with_grad(Var out, Tensor<f32> upstream) {
    std::vector<Var> order;
    std::unordered_set<VarImpl *> visited;

    std::function<void(Var)> visit = [&](Var v) {
        if (!v.defined() || visited.count(v.impl_.get()))
            return;
        visited.insert(v.impl_.get());
        if (v->grad_fn)
            for (auto &inp : v->grad_fn->inputs)
                visit(inp);
        order.push_back(v);
    };
    visit(out);
    std::reverse(order.begin(), order.end());

    out->grad = upstream;
    for (auto &v : order)
        if (v->grad_fn && v->grad.defined())
            v->grad_fn->backward(v->grad);
}

// ── EmbeddingLayer ────────────────────────────────────────────────────────────

static void test_embedding_layer() {
    printf("\n-- EmbeddingLayer --\n");

    Tensor<f32> weight   = tensor_load<f32>("../data/test/emb_layer/weight.npy",   g_on_gpu);
    TensorU32   tokens_t = tensor_load<u32>("../data/test/emb_layer/tokens.npy",   g_on_gpu);
    Tensor<f32> exp_out  = tensor_load<f32>("../data/test/emb_layer/out.npy",      false);
    Tensor<f32> exp_dw   = tensor_load<f32>("../data/test/emb_layer/d_weight.npy", false);

    if (!weight || !tokens_t || !exp_out || !exp_dw) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    EmbeddingLayer emb(10, 8, g_on_gpu);
    tensor_copy(emb.weight->data, weight);

    Var out = emb.forward(VarU32(tokens_t));
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_weight", tensor_to_cpu(emb.weight->grad), exp_dw);
}

// ── PositionalEmbeddingLayer ──────────────────────────────────────────────────

static void test_pos_embedding_layer() {
    printf("\n-- PositionalEmbeddingLayer --\n");

    Tensor<f32> weight  = tensor_load<f32>("../data/test/pos_emb_layer/weight.npy",   g_on_gpu);
    Tensor<f32> exp_out = tensor_load<f32>("../data/test/pos_emb_layer/out.npy",      false);
    Tensor<f32> exp_dw  = tensor_load<f32>("../data/test/pos_emb_layer/d_weight.npy", false);

    if (!weight || !exp_out || !exp_dw) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    PositionalEmbeddingLayer pos_emb(8, 6, g_on_gpu);
    tensor_copy(pos_emb.weight->data, weight);

    Var out = pos_emb.forward(4);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_weight", tensor_to_cpu(pos_emb.weight->grad), exp_dw);
}

// ── InputEmbedding ────────────────────────────────────────────────────────────

static void test_input_embedding() {
    printf("\n-- InputEmbedding --\n");

    Tensor<f32> tok_w    = tensor_load<f32>("../data/test/input_emb/tok_weight.npy",   g_on_gpu);
    Tensor<f32> pos_w    = tensor_load<f32>("../data/test/input_emb/pos_weight.npy",   g_on_gpu);
    TensorU32   tokens_t = tensor_load<u32>("../data/test/input_emb/tokens.npy",       g_on_gpu);
    Tensor<f32> exp_out  = tensor_load<f32>("../data/test/input_emb/out.npy",          false);
    Tensor<f32> exp_dtok = tensor_load<f32>("../data/test/input_emb/d_tok_weight.npy", false);
    Tensor<f32> exp_dpos = tensor_load<f32>("../data/test/input_emb/d_pos_weight.npy", false);

    if (!tok_w || !pos_w || !tokens_t || !exp_out || !exp_dtok || !exp_dpos) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    InputEmbedding emb(10, 8, 6, 0.0f, g_on_gpu);  // dropout=0 for determinism
    tensor_copy(emb.tok_emb.weight->data, tok_w);
    tensor_copy(emb.pos_emb.weight->data, pos_w);

    Var out = emb.forward(VarU32(tokens_t));
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_tok_weight", tensor_to_cpu(emb.tok_emb.weight->grad), exp_dtok);
    check_tensors("backward d_pos_weight", tensor_to_cpu(emb.pos_emb.weight->grad), exp_dpos);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) g_on_gpu = true;
        else if (strcmp(argv[i], "--cpu") == 0) g_on_gpu = false;
    }

    printf("\nBackend: %s\n", g_on_gpu ? "CUDA" : "CPU");

    test_embedding_layer();
    test_pos_embedding_layer();
    test_input_embedding();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
