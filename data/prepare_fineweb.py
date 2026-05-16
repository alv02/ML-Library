"""
Download and tokenize FineWeb-Edu for GPT training.

Usage:
    python3 data/prepare_fineweb.py [--max-tokens 1_000_000_000] [--out-dir data/fineweb]

Output:
    data/fineweb/train.bin   — uint16 token ids, flat array
    data/fineweb/val.bin     — uint16 token ids, flat array

Requires:
    pip install datasets tiktoken tqdm
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir",    default="data/fineweb")
parser.add_argument("--max-tokens", type=int, default=1_000_000_000,
                    help="Stop after this many tokens (default 1B — ~4 GB on disk)")
parser.add_argument("--val-docs",   type=int, default=5_000,
                    help="Number of documents to put in val split")
parser.add_argument("--dataset",    default="HuggingFaceFW/fineweb-edu",
                    help="HuggingFace dataset name")
parser.add_argument("--name",       default="sample-10BT",
                    help="Dataset config name")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── tokenizer ─────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode_single_token("<|endoftext|>")  # 50256 — separates documents

def tokenize(doc: dict) -> np.ndarray:
    ids = enc.encode_ordinary(doc["text"])
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)

# ── stream + write ────────────────────────────────────────────────────────────

print(f"Streaming {args.dataset} ({args.name})")
print(f"Target: {args.max_tokens:,} tokens  |  val: {args.val_docs:,} docs")

ds = load_dataset(args.dataset, name=args.name, split="train", streaming=True)

train_path = os.path.join(args.out_dir, "train.bin")
val_path   = os.path.join(args.out_dir, "val.bin")

train_file = open(train_path, "wb")
val_file   = open(val_path,   "wb")

total_tokens = 0
val_tokens   = 0
doc_count    = 0

pbar = tqdm(unit="tok", unit_scale=True, total=args.max_tokens)

for doc in ds:
    arr = tokenize(doc)
    if doc_count < args.val_docs:
        arr.tofile(val_file)
        val_tokens += len(arr)
    else:
        arr.tofile(train_file)
        total_tokens += len(arr)
        pbar.update(len(arr))
        if total_tokens >= args.max_tokens:
            break
    doc_count += 1

pbar.close()
train_file.close()
val_file.close()

train_bytes = os.path.getsize(train_path)
val_bytes   = os.path.getsize(val_path)

print(f"\nDone.")
print(f"  train: {total_tokens:>12,} tokens  ({train_bytes / 1e9:.2f} GB)  → {train_path}")
print(f"  val:   {val_tokens:>12,} tokens  ({val_bytes  / 1e6:.1f} MB)  → {val_path}")
print(f"\nTrain with:")
print(f"  python3 train_gpt.py --data-bin {train_path} --tokenizer bpe \\")
print(f"      --d-model 512 --n-heads 8 --n-layers 6 --max-seq-len 1024 \\")
print(f"      --batch-size 16 --steps 20000 --save-dir checkpoints/fineweb")
