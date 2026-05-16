"""
Interactive GPT inference.
Usage: python3 gpt.py [--checkpoint DIR] [--cpu] [--temperature T] [--max-tokens N]

Model config and tokenizer type are loaded automatically from config.json in the
checkpoint directory (written by train_gpt.py).
"""

import argparse
import json
import os
import sys

import numpy as np

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default="checkpoints/gpt_shakespeare",
)
parser.add_argument("--cpu", action="store_true")
parser.add_argument(
    "--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy)"
)
parser.add_argument(
    "--max-tokens", type=int, default=300, help="Max new tokens per prompt"
)
args = parser.parse_args()

# ── import ────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import gpt_lib

# ── load config ───────────────────────────────────────────────────────────────

cfg_path = os.path.join(args.checkpoint, "config.json")
if not os.path.exists(cfg_path):
    print(f"Error: no config.json found in {args.checkpoint!r}")
    print("Train the model first with train_gpt.py")
    sys.exit(1)

with open(cfg_path) as f:
    cfg = json.load(f)

D_MODEL = cfg["d_model"]
N_HEADS = cfg["n_heads"]
N_LAYERS = cfg["n_layers"]
MAX_SEQ_LEN = cfg["max_seq_len"]
VOCAB_SIZE = cfg["vocab_size"]
TOKENIZER = cfg.get("tokenizer", "char")

# ── tokenizer ─────────────────────────────────────────────────────────────────

tok_path = os.path.join(args.checkpoint, "tokenizer.json")

if TOKENIZER == "bpe":
    import tiktoken

    _enc = tiktoken.get_encoding("gpt2")

    def encode(text):
        return _enc.encode(text)

    def decode_token(tok_id):
        return _enc.decode([tok_id])

elif TOKENIZER == "pretrained":
    from tokenizers import Tokenizer

    if not os.path.exists(tok_path):
        print(f"Error: tokenizer not found at {tok_path!r}")
        sys.exit(1)
    _tok = Tokenizer.from_file(tok_path)

    def encode(text):
        return _tok.encode(text).ids

    def decode_token(tok_id):
        return _tok.decode([tok_id])

else:  # char
    from tokenizers import Tokenizer

    if not os.path.exists(tok_path):
        print(f"Error: tokenizer not found at {tok_path!r}")
        sys.exit(1)
    _tok = Tokenizer.from_file(tok_path)

    def encode(text):
        return _tok.encode(text).ids

    def decode_token(tok_id):
        return _tok.id_to_token(tok_id) or ""


# ── model ─────────────────────────────────────────────────────────────────────

on_gpu = not args.cpu

model = gpt_lib.GPTTrainer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    dropout_p=0.0,
    lr=1e-3,
    weight_decay=0.0,
    on_gpu=on_gpu,
)
model.load(args.checkpoint)
model.eval_mode()

print(f"\nCheckpoint : {args.checkpoint}")
print(
    f"Model      : d={D_MODEL} h={N_HEADS} layers={N_LAYERS} seq={MAX_SEQ_LEN} vocab={VOCAB_SIZE}"
)
print(f"Device     : {'GPU' if on_gpu else 'CPU'} | tokenizer={TOKENIZER}")
print(f"Temperature: {args.temperature}  |  Max tokens: {args.max_tokens}")
print("\nType a prompt and press Enter.")
print("  Ctrl+C  — interrupt current generation")
print("  Ctrl+D or 'exit' — quit\n")

# ── generation ────────────────────────────────────────────────────────────────


def generate(prompt: str):
    ids = encode(prompt)
    if not ids:
        return
    ctx = np.array([ids], dtype=np.uint32)
    print(prompt, end="", flush=True)
    try:
        for _ in range(args.max_tokens):
            if ctx.shape[1] > MAX_SEQ_LEN:
                ctx = ctx[:, -MAX_SEQ_LEN:]
            model.eval_mode()
            out = model.generate(ctx, max_new_tokens=1, temperature=args.temperature)
            new_tok = int(out[0, -1])
            print(decode_token(new_tok), end="", flush=True)
            ctx = out
    except KeyboardInterrupt:
        pass
    print()


# ── REPL ──────────────────────────────────────────────────────────────────────

while True:
    try:
        prompt = input("> ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    prompt = prompt.strip()
    if not prompt:
        continue
    if prompt.lower() in ("exit", "quit", "q"):
        print("Bye!")
        break

    generate(prompt)
    print()
