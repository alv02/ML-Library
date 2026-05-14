#include "../include/models.hpp"
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

// ── LayerNorm ─────────────────────────────────────────────────────────────────

static void test_layer_norm() {
    printf("\n-- LayerNorm --\n");

    Tensor<f32> input   = tensor_load<f32>("../data/test/layer_norm/input.npy",   g_on_gpu);
    Tensor<f32> gamma_t = tensor_load<f32>("../data/test/layer_norm/gamma.npy",   g_on_gpu);
    Tensor<f32> beta_t  = tensor_load<f32>("../data/test/layer_norm/beta.npy",    g_on_gpu);
    Tensor<f32> exp_out = tensor_load<f32>("../data/test/layer_norm/out.npy",     false);
    Tensor<f32> exp_din = tensor_load<f32>("../data/test/layer_norm/d_input.npy", false);
    Tensor<f32> exp_dg  = tensor_load<f32>("../data/test/layer_norm/d_gamma.npy", false);
    Tensor<f32> exp_db  = tensor_load<f32>("../data/test/layer_norm/d_beta.npy",  false);

    if (!input || !gamma_t || !beta_t || !exp_out || !exp_din || !exp_dg || !exp_db) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    u32 D = gamma_t->shape[0];
    LayerNorm ln(D, g_on_gpu);
    tensor_copy(ln.gamma->data, gamma_t);
    tensor_copy(ln.beta->data,  beta_t);

    Var x   = Var(input, FV_FLAG_REQUIERES_GRAD);
    Var out = ln.forward(x);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_input", tensor_to_cpu(x->grad),          exp_din, 1e-4f);
    check_tensors("backward d_gamma", tensor_to_cpu(ln.gamma->grad),   exp_dg,  1e-4f);
    check_tensors("backward d_beta",  tensor_to_cpu(ln.beta->grad),    exp_db,  1e-4f);
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

static void test_mha() {
    printf("\n-- MultiHeadAttention --\n");

    Tensor<f32> x     = tensor_load<f32>("../data/test/mha/x.npy",    g_on_gpu);
    Tensor<f32> Wq    = tensor_load<f32>("../data/test/mha/Wq.npy",   g_on_gpu);
    Tensor<f32> Wk    = tensor_load<f32>("../data/test/mha/Wk.npy",   g_on_gpu);
    Tensor<f32> Wv    = tensor_load<f32>("../data/test/mha/Wv.npy",   g_on_gpu);
    Tensor<f32> Wo    = tensor_load<f32>("../data/test/mha/Wo.npy",   g_on_gpu);
    Tensor<f32> exp_out = tensor_load<f32>("../data/test/mha/out.npy", false);
    Tensor<f32> exp_dx  = tensor_load<f32>("../data/test/mha/d_x.npy", false);

    if (!x || !Wq || !Wk || !Wv || !Wo || !exp_out || !exp_dx) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    u32 B = x->shape[0], T = x->shape[1], D = x->shape[2];
    u32 H = 2;  // matches gen_layers.py

    MultiHeadAttention mha(D, H, T, 0.0f, g_on_gpu);
    tensor_copy(mha.W_q->data, Wq);
    tensor_copy(mha.W_k->data, Wk);
    tensor_copy(mha.W_v->data, Wv);
    tensor_copy(mha.W_o->data, Wo);

    Var xv = Var(x, FV_FLAG_REQUIERES_GRAD);
    Var out = mha.forward(xv);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out, 1e-4f);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_x", tensor_to_cpu(xv->grad), exp_dx, 1e-3f);
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

    Var out = emb.forward(tokens_t);
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

    Var out = emb.forward(tokens_t);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_tok_weight", tensor_to_cpu(emb.tok_emb.weight->grad), exp_dtok);
    check_tensors("backward d_pos_weight", tensor_to_cpu(emb.pos_emb.weight->grad), exp_dpos);
}

// ── TransformerBlock ─────────────────────────────────────────────────────────

static void test_transformer_block() {
    printf("\n-- TransformerBlock --\n");

    Tensor<f32> x      = tensor_load<f32>("../data/test/transformer_block/x.npy",      g_on_gpu);
    Tensor<f32> ln1_g  = tensor_load<f32>("../data/test/transformer_block/ln1_g.npy",  g_on_gpu);
    Tensor<f32> ln1_b  = tensor_load<f32>("../data/test/transformer_block/ln1_b.npy",  g_on_gpu);
    Tensor<f32> ln2_g  = tensor_load<f32>("../data/test/transformer_block/ln2_g.npy",  g_on_gpu);
    Tensor<f32> ln2_b  = tensor_load<f32>("../data/test/transformer_block/ln2_b.npy",  g_on_gpu);
    Tensor<f32> Wq     = tensor_load<f32>("../data/test/transformer_block/Wq.npy",     g_on_gpu);
    Tensor<f32> Wk     = tensor_load<f32>("../data/test/transformer_block/Wk.npy",     g_on_gpu);
    Tensor<f32> Wv     = tensor_load<f32>("../data/test/transformer_block/Wv.npy",     g_on_gpu);
    Tensor<f32> Wo     = tensor_load<f32>("../data/test/transformer_block/Wo.npy",     g_on_gpu);
    Tensor<f32> W_fc   = tensor_load<f32>("../data/test/transformer_block/W_fc.npy",   g_on_gpu);
    Tensor<f32> b_fc   = tensor_load<f32>("../data/test/transformer_block/b_fc.npy",   g_on_gpu);
    Tensor<f32> W_proj = tensor_load<f32>("../data/test/transformer_block/W_proj.npy", g_on_gpu);
    Tensor<f32> b_proj = tensor_load<f32>("../data/test/transformer_block/b_proj.npy", g_on_gpu);
    Tensor<f32> exp_out = tensor_load<f32>("../data/test/transformer_block/out.npy",   false);
    Tensor<f32> exp_dx  = tensor_load<f32>("../data/test/transformer_block/d_x.npy",   false);

    if (!x || !ln1_g || !ln1_b || !ln2_g || !ln2_b ||
        !Wq || !Wk || !Wv || !Wo ||
        !W_fc || !b_fc || !W_proj || !b_proj || !exp_out || !exp_dx) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    u32 B = x->shape[0], T = x->shape[1], D = x->shape[2];
    u32 H = 2;

    TransformerBlock tb(D, H, T, 0.0f, g_on_gpu);
    tensor_copy(tb.ln1.gamma->data, ln1_g);
    tensor_copy(tb.ln1.beta->data,  ln1_b);
    tensor_copy(tb.ln2.gamma->data, ln2_g);
    tensor_copy(tb.ln2.beta->data,  ln2_b);
    tensor_copy(tb.mha.W_q->data, Wq);
    tensor_copy(tb.mha.W_k->data, Wk);
    tensor_copy(tb.mha.W_v->data, Wv);
    tensor_copy(tb.mha.W_o->data, Wo);
    tensor_copy(tb.mlp_fc.W->data,   W_fc);
    tensor_copy(tb.mlp_fc.b->data,   b_fc);
    tensor_copy(tb.mlp_proj.W->data, W_proj);
    tensor_copy(tb.mlp_proj.b->data, b_proj);

    Var xv  = Var(x, FV_FLAG_REQUIERES_GRAD);
    Var out = tb.forward(xv);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out, 1e-4f);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_x", tensor_to_cpu(xv->grad), exp_dx, 1e-3f);
}

// ── GPTModel ─────────────────────────────────────────────────────────────────

static void test_gpt_model() {
    printf("\n-- GPTModel --\n");

    const u32 vocab = 16, D = 8, H = 2, n_layers = 2, max_seq_len = 8;
    const char *dir = "../data/test/gpt_model";

    TensorU32   tokens  = tensor_load<u32>("../data/test/gpt_model/tokens.npy", g_on_gpu);
    Tensor<f32> tok_w   = tensor_load<f32>("../data/test/gpt_model/tok_w.npy",  g_on_gpu);
    Tensor<f32> pos_w   = tensor_load<f32>("../data/test/gpt_model/pos_w.npy",  g_on_gpu);
    Tensor<f32> lnf_g   = tensor_load<f32>("../data/test/gpt_model/lnf_g.npy", g_on_gpu);
    Tensor<f32> lnf_b   = tensor_load<f32>("../data/test/gpt_model/lnf_b.npy", g_on_gpu);
    Tensor<f32> W_lmh   = tensor_load<f32>("../data/test/gpt_model/W_lmh.npy", g_on_gpu);
    Tensor<f32> b_lmh   = tensor_load<f32>("../data/test/gpt_model/b_lmh.npy", g_on_gpu);
    Tensor<f32> exp_out = tensor_load<f32>("../data/test/gpt_model/out.npy",    false);
    Tensor<f32> exp_dtok = tensor_load<f32>("../data/test/gpt_model/d_tok_w.npy", false);

    // Load per-block weights
    char path[256];
    Tensor<f32> blk_ln1_g[2], blk_ln1_b[2], blk_ln2_g[2], blk_ln2_b[2];
    Tensor<f32> blk_Wq[2], blk_Wk[2], blk_Wv[2], blk_Wo[2];
    Tensor<f32> blk_W_fc[2], blk_b_fc[2], blk_W_proj[2], blk_b_proj[2];
    for (u32 i = 0; i < n_layers; i++) {
#define LOAD_BLK(arr, name) \
    snprintf(path, sizeof(path), "%s/block%u_" name ".npy", dir, i); \
    arr[i] = tensor_load<f32>(path, g_on_gpu)
        LOAD_BLK(blk_ln1_g, "ln1_g"); LOAD_BLK(blk_ln1_b, "ln1_b");
        LOAD_BLK(blk_ln2_g, "ln2_g"); LOAD_BLK(blk_ln2_b, "ln2_b");
        LOAD_BLK(blk_Wq, "Wq"); LOAD_BLK(blk_Wk, "Wk");
        LOAD_BLK(blk_Wv, "Wv"); LOAD_BLK(blk_Wo, "Wo");
        LOAD_BLK(blk_W_fc, "W_fc"); LOAD_BLK(blk_b_fc, "b_fc");
        LOAD_BLK(blk_W_proj, "W_proj"); LOAD_BLK(blk_b_proj, "b_proj");
#undef LOAD_BLK
    }

    if (!tokens || !tok_w || !pos_w || !lnf_g || !lnf_b ||
        !W_lmh || !b_lmh || !exp_out || !exp_dtok) {
        printf("  [%sFAIL%s] could not load data files\n", RED, RESET);
        failed++;
        return;
    }

    GPTModel gpt(vocab, D, H, n_layers, max_seq_len, 0.0f, g_on_gpu);
    tensor_copy(gpt.embedding.tok_emb.weight->data, tok_w);
    tensor_copy(gpt.embedding.pos_emb.weight->data, pos_w);
    tensor_copy(gpt.ln_f.gamma->data, lnf_g);
    tensor_copy(gpt.ln_f.beta->data,  lnf_b);
    tensor_copy(gpt.lm_head.W->data, W_lmh);
    tensor_copy(gpt.lm_head.b->data, b_lmh);
    for (u32 i = 0; i < n_layers; i++) {
        auto &tb = gpt.blocks[i];
        tensor_copy(tb.ln1.gamma->data, blk_ln1_g[i]);
        tensor_copy(tb.ln1.beta->data,  blk_ln1_b[i]);
        tensor_copy(tb.ln2.gamma->data, blk_ln2_g[i]);
        tensor_copy(tb.ln2.beta->data,  blk_ln2_b[i]);
        tensor_copy(tb.mha.W_q->data, blk_Wq[i]);
        tensor_copy(tb.mha.W_k->data, blk_Wk[i]);
        tensor_copy(tb.mha.W_v->data, blk_Wv[i]);
        tensor_copy(tb.mha.W_o->data, blk_Wo[i]);
        tensor_copy(tb.mlp_fc.W->data,   blk_W_fc[i]);
        tensor_copy(tb.mlp_fc.b->data,   blk_b_fc[i]);
        tensor_copy(tb.mlp_proj.W->data, blk_W_proj[i]);
        tensor_copy(tb.mlp_proj.b->data, blk_b_proj[i]);
    }

    Var out = gpt.forward(tokens);
    sync();
    check_tensors("forward", tensor_to_cpu(out->data), exp_out, 1e-4f);

    Tensor<f32> ones = Tensor<f32>::make(out->data->ndim, out->data->shape, g_on_gpu);
    tensor_fill(ones, 1.0f);
    backward_with_grad(out, ones);
    sync();
    check_tensors("backward d_tok_w", tensor_to_cpu(gpt.embedding.tok_emb.weight->grad),
                  exp_dtok, 1e-3f);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) g_on_gpu = true;
        else if (strcmp(argv[i], "--cpu") == 0) g_on_gpu = false;
    }

    printf("\nBackend: %s\n", g_on_gpu ? "CUDA" : "CPU");

    test_layer_norm();
    test_mha();
    test_embedding_layer();
    test_pos_embedding_layer();
    test_input_embedding();
    test_transformer_block();
    test_gpt_model();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
