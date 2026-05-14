"""
GPT training + generation demo.
Usage:
    python3 train_gpt.py [--cpu] [--steps N] [--generate] [--tokenizer {char,bpe}]

Requires:
    pip install tokenizers          # for --tokenizer char (default)
    pip install tiktoken            # for --tokenizer bpe
    A text file at data/input.txt  (e.g. Shakespeare, any plain text)
"""

import sys, os, time, argparse
import numpy as np

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--cpu",       action="store_true",  help="Use CPU instead of GPU")
parser.add_argument("--steps",     type=int, default=500, help="Training steps")
parser.add_argument("--generate",  action="store_true",  help="Skip training, load & generate")
parser.add_argument("--text",      type=str, default="data/input.txt")
parser.add_argument("--save-dir",  type=str, default="checkpoints/gpt")
parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                    help="char = character-level (HF tokenizers), bpe = tiktoken GPT-2")
args = parser.parse_args()

# ── import gpt_lib ────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import gpt_lib

# ── tokenizer ─────────────────────────────────────────────────────────────────

tok_path = os.path.join(args.save_dir, "tokenizer.json")

if args.tokenizer == "bpe":
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = _enc.n_vocab   # 50257
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

# ── model config ──────────────────────────────────────────────────────────────

D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 4
MAX_SEQ_LEN = 256
DROPOUT_P   = 0.1
LR          = 3e-4
WEIGHT_DECAY= 0.1
BATCH_SIZE  = 8

on_gpu = not args.cpu

model = gpt_lib.GPTTrainer(
    vocab_size   = VOCAB_SIZE,
    d_model      = D_MODEL,
    n_heads      = N_HEADS,
    n_layers     = N_LAYERS,
    max_seq_len  = MAX_SEQ_LEN,
    dropout_p    = DROPOUT_P,
    lr           = LR,
    weight_decay = WEIGHT_DECAY,
    on_gpu       = on_gpu,
)
print(f"GPT: d={D_MODEL} h={N_HEADS} layers={N_LAYERS} seq={MAX_SEQ_LEN} "
      f"device={'GPU' if on_gpu else 'CPU'}")

# ── load checkpoint if requested ──────────────────────────────────────────────

if args.generate:
    model.load(args.save_dir)
    model.eval_mode()

# ── training ──────────────────────────────────────────────────────────────────

if not args.generate:
    if args.tokenizer == "bpe":
        with open(args.text, encoding="utf-8") as f:
            raw = f.read()
    tokens = np.array(encode(raw), dtype=np.uint32)
    print(f"Dataset: {len(tokens):,} tokens")

    def random_batch():
        starts = np.random.randint(0, len(tokens) - MAX_SEQ_LEN - 1, size=BATCH_SIZE)
        return np.stack([tokens[s:s + MAX_SEQ_LEN + 1] for s in starts])

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

# ── generation ────────────────────────────────────────────────────────────────

print("\n── Generation ───────────────────────────────")
model.eval_mode()

prompts = ["Hello, world", "The quick brown fox"]
for prompt in prompts:
    ctx = np.array([encode(prompt)], dtype=np.uint32)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8)
    print(f"\nPrompt: {prompt!r}")
    print(decode(out[0].tolist()))
    print()
