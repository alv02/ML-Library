"""
GPT training script.
Usage:
    python3 train_gpt.py [options]

Requires:
    pip install tokenizers   # for --tokenizer char
    pip install tiktoken     # for --tokenizer bpe
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# data / run
parser.add_argument("--cpu",       action="store_true")
parser.add_argument("--steps",     type=int,   default=500)
parser.add_argument("--text",      type=str,   default="data/input.txt")
parser.add_argument("--data-bin",  type=str,   default=None,
                    help="Pre-tokenized binary file (uint16). Overrides --text.")
parser.add_argument("--save-dir",  type=str,   default="checkpoints/gpt")
parser.add_argument("--generate",  action="store_true",
                    help="Skip training, load checkpoint and generate")
parser.add_argument("--tokenizer", choices=["char", "bpe", "pretrained"], default="char")
parser.add_argument("--tokenizer-path", type=str, default=None,
                    help="Path to a pre-built HF tokenizer JSON (required with --tokenizer pretrained)")
# model
parser.add_argument("--d-model",     type=int,   default=256)
parser.add_argument("--n-heads",     type=int,   default=4)
parser.add_argument("--n-layers",    type=int,   default=4)
parser.add_argument("--max-seq-len", type=int,   default=256)
parser.add_argument("--dropout",     type=float, default=0.1)
# optimiser
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--batch-size",   type=int,   default=8)
args = parser.parse_args()

# ── import ────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import gpt_lib

# ── tokenizer ─────────────────────────────────────────────────────────────────

tok_path = os.path.join(args.save_dir, "tokenizer.json")

if args.tokenizer == "bpe":
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = _enc.n_vocab  # 50257

    def encode(text):
        return _enc.encode(text)

    def decode(ids):
        return _enc.decode(ids)

    print(f"BPE tokenizer (tiktoken GPT-2), vocab={VOCAB_SIZE}")

elif args.tokenizer == "pretrained":
    from tokenizers import Tokenizer

    if args.generate:
        tok = Tokenizer.from_file(tok_path)
    else:
        if not args.tokenizer_path:
            parser.error("--tokenizer-path is required with --tokenizer pretrained")
        tok = Tokenizer.from_file(args.tokenizer_path)
        with open(args.text, encoding="utf-8") as f:
            raw = f.read()

    VOCAB_SIZE = tok.get_vocab_size()

    def encode(text):
        return tok.encode(text).ids

    def decode(ids):
        return tok.decode(ids)

    print(f"Pretrained HF tokenizer, vocab={VOCAB_SIZE}")

else:  # char
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split

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

    def encode(text):
        return tok.encode(text).ids

    def decode(ids):
        return "".join(tok.id_to_token(i) or "" for i in ids)

    print(f"Char-level tokenizer (HF), vocab={VOCAB_SIZE}")

# ── model ─────────────────────────────────────────────────────────────────────

D_MODEL     = args.d_model
N_HEADS     = args.n_heads
N_LAYERS    = args.n_layers
MAX_SEQ_LEN = args.max_seq_len
DROPOUT_P   = args.dropout
LR          = args.lr
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE  = args.batch_size
on_gpu      = not args.cpu

# If loading a checkpoint, read config from it so we don't need to re-specify
cfg_path = os.path.join(args.save_dir, "config.json")
if args.generate and os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    D_MODEL     = cfg["d_model"]
    N_HEADS     = cfg["n_heads"]
    N_LAYERS    = cfg["n_layers"]
    MAX_SEQ_LEN = cfg["max_seq_len"]
    print(f"Loaded config from {cfg_path}")

model = gpt_lib.GPTTrainer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    dropout_p=DROPOUT_P,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    on_gpu=on_gpu,
)
print(
    f"GPT: d={D_MODEL} h={N_HEADS} layers={N_LAYERS} seq={MAX_SEQ_LEN} "
    f"vocab={VOCAB_SIZE} device={'GPU' if on_gpu else 'CPU'}"
)

# ── load checkpoint ───────────────────────────────────────────────────────────

if args.generate:
    model.load(args.save_dir)
    model.eval_mode()

# ── training ──────────────────────────────────────────────────────────────────

if not args.generate:
    if args.data_bin:
        # Memory-mapped binary — no RAM cost, works for multi-billion token datasets
        tokens = np.memmap(args.data_bin, dtype=np.uint16, mode="r")
        print(f"Dataset (binary): {len(tokens):,} tokens  [{args.data_bin}]")
    else:
        if args.tokenizer == "bpe":
            with open(args.text, encoding="utf-8") as f:
                raw = f.read()
        tokens = np.array(encode(raw), dtype=np.uint32)
        print(f"Dataset: {len(tokens):,} tokens")

    def random_batch():
        starts = np.random.randint(0, len(tokens) - MAX_SEQ_LEN - 1, size=BATCH_SIZE)
        # Cast to uint32 — memmap gives uint16, model expects uint32
        return np.stack([tokens[s : s + MAX_SEQ_LEN + 1] for s in starts]).astype(np.uint32)

    model.train_mode()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = random_batch()
        loss = model.train_step(batch)
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:4d}/{args.steps}  loss {loss:.4f}  {elapsed:.1f}s")
            t0 = time.time()

    os.makedirs(args.save_dir, exist_ok=True)
    model.save(args.save_dir)
    if args.tokenizer == "char":
        tok.save(tok_path)
    elif args.tokenizer == "pretrained":
        import shutil
        shutil.copy2(args.tokenizer_path, tok_path)

    # Save model config so gpt.py can load it without extra flags
    with open(cfg_path, "w") as f:
        json.dump({
            "d_model":     D_MODEL,
            "n_heads":     N_HEADS,
            "n_layers":    N_LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
            "vocab_size":  VOCAB_SIZE,
            "tokenizer":   args.tokenizer,
        }, f, indent=2)
    print(f"Config saved to {cfg_path}")

# ── generation ────────────────────────────────────────────────────────────────

print("\n── Generation ───────────────────────────────")
model.eval_mode()

prompts = ["HAMLET:", "To be or not to be"]
for prompt in prompts:
    ctx = np.array([encode(prompt)], dtype=np.uint32)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8)
    model.eval_mode()
    print(f"\nPrompt: {prompt!r}")
    print(decode(out[0].tolist()))
    print()
