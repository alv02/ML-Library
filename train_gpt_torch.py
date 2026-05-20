"""
PyTorch GPT baseline — same architecture as the custom C++ library.
Use this to validate loss curves and compare GPU memory usage.

Architecture matches exactly:
  - Pre-LayerNorm (ln before attention and FFN)
  - 4×d_model FFN with GELU
  - Causal self-attention
  - Token + learned positional embeddings
  - AdamW + cosine LR with linear warmup

Usage:
    python3 train_gpt_torch.py [options]

Requires:
    pip install torch tiktoken numpy
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-bin",    default="data/fineweb/train.bin")
parser.add_argument("--save-dir",    default="checkpoints/fineweb_torch")
parser.add_argument("--cpu",         action="store_true")
parser.add_argument("--steps",       type=int,   default=100000)
# model
parser.add_argument("--d-model",     type=int,   default=512)
parser.add_argument("--n-heads",     type=int,   default=8)
parser.add_argument("--n-layers",    type=int,   default=6)
parser.add_argument("--max-seq-len", type=int,   default=1024)
parser.add_argument("--dropout",     type=float, default=0.1)
# optimiser
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--min-lr",       type=float, default=3e-5)
parser.add_argument("--warmup-steps", type=int,   default=2000)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--batch-size",   type=int,   default=16)
args = parser.parse_args()

device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout_p):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model,     bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)

        # causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                                      # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)                         # [B, T, C] each

        # reshape to [B, H, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale        # [B, H, T, T]
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout_p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout_p)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout_p):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout_p)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout_p)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                               # [B, T, vocab]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ── data ──────────────────────────────────────────────────────────────────────

tokens = np.memmap(args.data_bin, dtype=np.uint16, mode="r")
print(f"Dataset: {len(tokens):,} tokens  [{args.data_bin}]")

D_MODEL     = args.d_model
N_HEADS     = args.n_heads
N_LAYERS    = args.n_layers
MAX_SEQ_LEN = args.max_seq_len
BATCH_SIZE  = args.batch_size
VOCAB_SIZE  = 50257  # GPT-2 BPE

def random_batch():
    starts = np.random.randint(0, len(tokens) - MAX_SEQ_LEN - 1, size=BATCH_SIZE)
    x = np.stack([tokens[s : s + MAX_SEQ_LEN]     for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + MAX_SEQ_LEN + 1] for s in starts]).astype(np.int64)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y

# ── model + optimizer ─────────────────────────────────────────────────────────

model = GPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, args.dropout).to(device)
print(f"GPT: d={D_MODEL} h={N_HEADS} layers={N_LAYERS} seq={MAX_SEQ_LEN} "
      f"vocab={VOCAB_SIZE} params={model.num_params()/1e6:.1f}M")

# separate weight-decay and no-decay params (no decay on biases/LN)
decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optim = torch.optim.AdamW(
    [{"params": decay_params,    "weight_decay": args.weight_decay},
     {"params": no_decay_params, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
)

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step):
    if step <= args.warmup_steps:
        return args.lr * step / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

print(f"LR schedule: warmup={args.warmup_steps} steps, max={args.lr:.2e}, min={args.min_lr:.2e}")

# ── memory baseline ───────────────────────────────────────────────────────────

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU memory after model init: "
          f"{torch.cuda.memory_allocated()/1024**2:.0f} MiB allocated, "
          f"{torch.cuda.memory_reserved()/1024**2:.0f} MiB reserved")

# ── training ──────────────────────────────────────────────────────────────────

model.train()
os.makedirs(args.save_dir, exist_ok=True)
t0 = time.time()

for step in range(1, args.steps + 1):
    lr = get_lr(step)
    for g in optim.param_groups:
        g["lr"] = lr

    x, y = random_batch()
    _, loss = model(x, y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    optim.zero_grad(set_to_none=True)

    if step % 50 == 0:
        elapsed = time.time() - t0
        mem_str = ""
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**2
            peak  = torch.cuda.max_memory_allocated() / 1024**2
            mem_str = f"  mem={alloc:.0f}/{peak:.0f} MiB"
        print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
              f"lr {lr:.2e}  {elapsed:.1f}s{mem_str}")
        t0 = time.time()

# ── save ──────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))
print(f"\nSaved to {args.save_dir}/model.pt")
if device == "cuda":
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
