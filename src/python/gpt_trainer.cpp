#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "autograd.hpp"
#include "backend/cuda_mem_arena.hpp"
#include "models.hpp"
#include "ops.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"

namespace py = pybind11;

// ── Weight serialization ──────────────────────────────────────────────────────
// Format: [ndim u32][shape[0..ndim-1] u32][data f32 x numel]

static void param_save(const Tensor<f32> &t, const char *path) {
    Tensor<f32> cpu = tensor_to_cpu(t);
    FILE *f = fopen(path, "wb");
    if (!f)
        throw std::runtime_error(std::string("cannot open for write: ") + path);
    u32 ndim = cpu->ndim;
    fwrite(&ndim, sizeof(u32), 1, f);
    fwrite(cpu->shape, sizeof(u32), ndim, f);
    fwrite(cpu->data(), sizeof(f32), cpu->numel(), f);
    fclose(f);
}

static Tensor<f32> param_load(const char *path, bool on_gpu,
                               CudaMemArena *arena = nullptr) {
    FILE *f = fopen(path, "rb");
    if (!f)
        throw std::runtime_error(std::string("cannot open for read: ") + path);
    u32 ndim;
    fread(&ndim, sizeof(u32), 1, f);
    u32 shape[MAX_NDIM];
    fread(shape, sizeof(u32), ndim, f);
    Tensor<f32> cpu = Tensor<f32>::make(ndim, shape, false);
    fread(cpu->data(), sizeof(f32), cpu->numel(), f);
    fclose(f);
    if (on_gpu)
        return tensor_to_gpu(cpu, arena);
    return cpu;
}

// ── CPU softmax + multinomial sampling ───────────────────────────────────────

static u32 sample_token(const f32 *logits, u32 vocab, f32 temperature,
                        std::mt19937 &rng) {
    if (temperature <= 0.0f) {
        // Greedy
        u32 best = 0;
        for (u32 i = 1; i < vocab; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }
    // Softmax with temperature
    std::vector<f32> probs(vocab);
    f32 max_l = logits[0];
    for (u32 i = 1; i < vocab; i++)
        if (logits[i] > max_l) max_l = logits[i];
    f32 sum = 0.0f;
    for (u32 i = 0; i < vocab; i++) {
        probs[i] = expf((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    std::uniform_real_distribution<f32> dist(0.0f, sum);
    f32 r = dist(rng);
    f32 cum = 0.0f;
    for (u32 i = 0; i < vocab; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return vocab - 1;
}

// ── GPTTrainer ────────────────────────────────────────────────────────────────

struct GPTTrainer {
    CudaMemArena perm_arena;
    CudaMemArena batch_arena;
    GPTModel model;
    AdamW optim;
    u32 vocab_size_;
    u32 max_seq_len_;
    bool on_gpu_;
    std::mt19937 rng{42};

    GPTTrainer(u32 vocab_size, u32 d_model, u32 n_heads, u32 n_layers,
               u32 max_seq_len, f32 dropout_p, f32 lr, f32 weight_decay,
               bool on_gpu)
        : perm_arena(GiB(2), "PermArena"),
          batch_arena(GiB(4), "BatchArena"),
          model(vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout_p,
                on_gpu, &perm_arena),
          optim(model.parameters(), lr, 0.9f, 0.999f, 1e-8f, weight_decay,
                &perm_arena),
          vocab_size_(vocab_size), max_seq_len_(max_seq_len), on_gpu_(on_gpu) {}

    // ── Internal helpers ──────────────────────────────────────────────────────

    // Build inp [B,T] and tgt_buf [B*T] from a numpy [B, T+1] array.
    std::pair<TensorU32, std::vector<u32>>
    make_inp_tgt(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tokens_np) {
        auto buf = tokens_np.request();
        if (buf.ndim != 2)
            throw std::runtime_error("tokens must be 2-D [B, T+1]");
        u32 B = (u32)buf.shape[0], Tp1 = (u32)buf.shape[1];
        if (Tp1 < 2)
            throw std::runtime_error("sequence length must be >= 2");
        u32 T = Tp1 - 1;

        const uint32_t *src = static_cast<const uint32_t *>(buf.ptr);

        u32 seq_shape[2] = {B, T};
        TensorU32 inp_cpu = Tensor<u32>::make(2, seq_shape, false);
        std::vector<u32> tgt(B * T);

        for (u32 b = 0; b < B; b++) {
            memcpy(inp_cpu->data() + b * T, src + b * Tp1, T * sizeof(u32));
            memcpy(tgt.data() + b * T, src + b * Tp1 + 1, T * sizeof(u32));
        }

        TensorU32 inp = on_gpu_ ? tensor_to_gpu(inp_cpu, &batch_arena) : inp_cpu;
        return {inp, tgt};
    }

    // One-hot encode tgt [B*T] → Var [B*T, vocab].
    Var make_targets(const std::vector<u32> &tgt) {
        u32 N = (u32)tgt.size();
        u32 shape[2] = {N, vocab_size_};
        Tensor<f32> oh = tensor_zeros<f32>(2, shape, false, nullptr);
        f32 *p = oh->data();
        for (u32 i = 0; i < N; i++)
            p[i * vocab_size_ + tgt[i]] = 1.0f;
        if (on_gpu_)
            oh = tensor_to_gpu(oh, &batch_arena);
        return Var(oh, FV_FLAG_NONE);
    }

    // Forward → flat logits + loss.
    std::pair<Var, float> forward_loss(TensorU32 inp, const std::vector<u32> &tgt) {
        u32 B = inp->shape[0], T = inp->shape[1];
        Var logits = model.forward(inp, &batch_arena);    // [B, T, vocab]
        u32 flat_shape[2] = {B * T, vocab_size_};
        Var logits_flat = reshape(logits, flat_shape, 2, &batch_arena);
        Var targets = make_targets(tgt);
        Var loss = cross_entropy_with_logits(logits_flat, targets, &batch_arena);
        float loss_val = tensor_to_cpu(loss->data)->data()[0];
        return {loss, loss_val};
    }

    // ── Public API ────────────────────────────────────────────────────────────

    float train_step(py::array_t<uint32_t> tokens_np) {
        cuda_arena_clear(&batch_arena);
        auto [inp, tgt] = make_inp_tgt(tokens_np);
        auto [loss, loss_val] = forward_loss(inp, tgt);
        backward(loss, &batch_arena);
        optim.step(&batch_arena);
        optim.zero_grad();
        return loss_val;
    }

    float eval_loss(py::array_t<uint32_t> tokens_np) {
        cuda_arena_clear(&batch_arena);
        auto [inp, tgt] = make_inp_tgt(tokens_np);
        auto [loss, loss_val] = forward_loss(inp, tgt);
        return loss_val;
    }

    py::array_t<uint32_t> generate(py::array_t<uint32_t> context_np,
                                   int max_new_tokens,
                                   float temperature = 1.0f) {
        auto buf = context_np.request();
        if (buf.ndim != 2)
            throw std::runtime_error("context must be 2-D [B, T]");
        u32 B = (u32)buf.shape[0], T0 = (u32)buf.shape[1];

        // Working buffer: [B, current_len]
        std::vector<u32> ctx(B * (T0 + max_new_tokens));
        memcpy(ctx.data(), buf.ptr, B * T0 * sizeof(u32));

        u32 cur_len = T0;
        model.eval();

        for (int step = 0; step < max_new_tokens; step++) {
            cuda_arena_clear(&batch_arena);

            // Truncate to max_seq_len if needed
            u32 start = (cur_len > max_seq_len_) ? (cur_len - max_seq_len_) : 0;
            u32 T = cur_len - start;
            u32 inp_shape[2] = {B, T};
            TensorU32 inp_cpu = Tensor<u32>::make(2, inp_shape, false);
            for (u32 b = 0; b < B; b++)
                memcpy(inp_cpu->data() + b * T, ctx.data() + b * (T0 + max_new_tokens) + start,
                       T * sizeof(u32));

            TensorU32 inp = on_gpu_ ? tensor_to_gpu(inp_cpu, &batch_arena) : inp_cpu;
            Var logits = model.forward(inp, &batch_arena);  // [B, T, vocab]

            // Extract last-position logits [B, vocab] → CPU
            Tensor<f32> logits_cpu = tensor_to_cpu(logits->data);
            const f32 *lp = logits_cpu->data();
            u32 T_cur = logits_cpu->shape[1];

            for (u32 b = 0; b < B; b++) {
                const f32 *row = lp + b * T_cur * vocab_size_ + (T_cur - 1) * vocab_size_;
                u32 tok = sample_token(row, vocab_size_, temperature, rng);
                ctx[b * (T0 + max_new_tokens) + cur_len] = tok;
            }
            cur_len++;
        }

        model.train();

        // Return [B, cur_len]
        py::array_t<uint32_t> out({(py::ssize_t)B, (py::ssize_t)cur_len});
        auto out_buf = out.request();
        uint32_t *dst = static_cast<uint32_t *>(out_buf.ptr);
        for (u32 b = 0; b < B; b++)
            memcpy(dst + b * cur_len, ctx.data() + b * (T0 + max_new_tokens),
                   cur_len * sizeof(u32));
        return out;
    }

    void train_mode() { model.train(true); }
    void eval_mode()  { model.eval(); }

    void set_lr(float lr) { optim.lr = lr; }

    void save(const std::string &directory) {
        auto params = model.parameters();
        char path[512];
        for (u32 i = 0; i < (u32)params.size(); i++) {
            snprintf(path, sizeof(path), "%s/param_%u.bin", directory.c_str(), i);
            param_save(params[i]->data, path);
        }
        printf("Saved %zu parameters to %s\n", params.size(), directory.c_str());
    }

    void load(const std::string &directory) {
        auto params = model.parameters();
        char path[512];
        for (u32 i = 0; i < (u32)params.size(); i++) {
            snprintf(path, sizeof(path), "%s/param_%u.bin", directory.c_str(), i);
            Tensor<f32> loaded = param_load(path, on_gpu_, &perm_arena);
            tensor_copy(params[i]->data, loaded);
        }
        printf("Loaded %zu parameters from %s\n", params.size(), directory.c_str());
    }
};

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(gpt_lib, m) {
    m.doc() = "GPT trainer/inference bindings";

    py::class_<GPTTrainer>(m, "GPTTrainer")
        .def(py::init<u32, u32, u32, u32, u32, f32, f32, f32, bool>(),
             py::arg("vocab_size"), py::arg("d_model"), py::arg("n_heads"),
             py::arg("n_layers"), py::arg("max_seq_len"),
             py::arg("dropout_p") = 0.1f,
             py::arg("lr") = 3e-4f,
             py::arg("weight_decay") = 0.1f,
             py::arg("on_gpu") = true)
        .def("train_step", &GPTTrainer::train_step,
             "Forward + backward + AdamW step. tokens: uint32[B, T+1]. Returns loss.")
        .def("eval_loss", &GPTTrainer::eval_loss,
             "Forward + loss only (no backward). Returns loss.")
        .def("generate", &GPTTrainer::generate,
             py::arg("context"), py::arg("max_new_tokens"),
             py::arg("temperature") = 1.0f,
             "Autoregressive generation. context: uint32[B,T]. Returns uint32[B, T+N].")
        .def("train_mode", &GPTTrainer::train_mode)
        .def("eval_mode",  &GPTTrainer::eval_mode)
        .def("set_lr",     &GPTTrainer::set_lr)
        .def("save",       &GPTTrainer::save,   "Save weights to directory.")
        .def("load",       &GPTTrainer::load,   "Load weights from directory.");
}
