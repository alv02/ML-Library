"""
GPT training + generation — PyTorch baseline.
Architecture mirrors the C++ GPTModel exactly (pre-norm, no QKV bias, GELU tanh approx).
Usage:
    python3 train_gpt_pytorch.py [--cpu] [--steps N] [--generate] [--tokenizer {char,bpe}]

Requires:
    pip install torch tokenizers    # for --tokenizer char (default)
    pip install tiktoken            # for --tokenizer bpe
"""

import sys, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",       action="store_true",  help="Use CPU instead of GPU")
parser.add_argument("--steps",     type=int, default=500, help="Training steps")
parser.add_argument("--generate",  action="store_true",  help="Skip training, load & generate")
parser.add_argument("--text",      type=str, default="data/input.txt")
parser.add_argument("--save-dir",  type=str, default="checkpoints/gpt_pytorch")
parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                    help="char = character-level (HF tokenizers), bpe = tiktoken GPT-2")
args = parser.parse_args()

device = "cpu" if args.cpu else "cuda"

# ── tokenizer ─────────────────────────────────────────────────────────────────

tok_path = os.path.join(args.save_dir, "tokenizer.json")

if args.tokenizer == "bpe":
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = _enc.n_vocab
    def encode(text): return _enc.encode(text)
    def decode(ids):  return _enc.decode(ids)
    print(f"BPE tokenizer (tiktoken GPT-2), vocab={VOCAB_SIZE}")
else:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers import Regex

    if args.generate:
        tok = Tokenizer.from_file(tok_path)
    else:
        with open(args.text, encoding="utf-8") as f:
            raw = f.read()
        chars = sorted(set(raw))
        vocab = {"[UNK]": 0}
        vocab.update({c: i + 1 for i, c in enumerate(chars)})
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")

    VOCAB_SIZE = tok.get_vocab_size()
    def encode(text): return tok.encode(text).ids
    def decode(ids):  return "".join(tok.id_to_token(i) or "" for i in ids)
    print(f"Char-level tokenizer (HF), vocab={VOCAB_SIZE}")

# ── model ─────────────────────────────────────────────────────────────────────

D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 4
MAX_SEQ_LEN = 256
DROPOUT_P   = 0.1
LR          = 3e-4
WEIGHT_DECAY= 0.1
BATCH_SIZE  = 8


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_q(x).reshape(B, T, H, dh).transpose(1, 2)
        K = self.W_k(x).reshape(B, T, H, dh).transpose(1, 2)
        V = self.W_v(x).reshape(B, T, H, dh).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / dh ** 0.5
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.W_o((attn @ V).transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout_p):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2  = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout_p):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(max_seq_len, d_model)
        self.drop     = nn.Dropout(dropout_p)
        self.blocks   = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout_p)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))  # [B, T, vocab]

    @torch.no_grad()
    def generate(self, ctx, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            crop = ctx[:, -self.max_seq_len:]
            logits = self(crop)[:, -1, :]
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx, next_tok], dim=1)
        return ctx


model = GPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, DROPOUT_P).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"GPT: d={D_MODEL} h={N_HEADS} layers={N_LAYERS} seq={MAX_SEQ_LEN} "
      f"params={n_params:,} device={device.upper()}")

# ── load checkpoint if requested ──────────────────────────────────────────────

ckpt_path = os.path.join(args.save_dir, "model.pt")

if args.generate:
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

# ── training ──────────────────────────────────────────────────────────────────

if not args.generate:
    if args.tokenizer == "bpe":
        with open(args.text, encoding="utf-8") as f:
            raw = f.read()
    tokens = np.array(encode(raw), dtype=np.int64)
    print(f"Dataset: {len(tokens):,} tokens")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                               eps=1e-8, weight_decay=WEIGHT_DECAY)

    def random_batch():
        starts = np.random.randint(0, len(tokens) - MAX_SEQ_LEN - 1, size=BATCH_SIZE)
        batch = np.stack([tokens[s:s + MAX_SEQ_LEN + 1] for s in starts])
        t = torch.from_numpy(batch).to(device)
        return t[:, :-1], t[:, 1:]

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        inp, tgt = random_batch()
        logits = model(inp)                              # [B, T, vocab]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:4d}/{args.steps}  loss {loss.item():.4f}  {elapsed:.1f}s")
            t0 = time.time()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    if args.tokenizer == "char":
        tok.save(tok_path)
    print(f"Saved to {args.save_dir}")

# ── generation ────────────────────────────────────────────────────────────────

print("\n── Generation ───────────────────────────────")
model.eval()

prompts = ["Hello, world", "The quick brown fox"]
for prompt in prompts:
    ctx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8)
    print(f"\nPrompt: {prompt!r}")
    print(decode(out[0].tolist()))
    print()
