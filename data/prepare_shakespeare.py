"""Download tiny-shakespeare and save char-level tokens as uint32 .npy."""
import os
import numpy as np
import urllib.request

URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shakespeare")

os.makedirs(OUT, exist_ok=True)

txt_path = os.path.join(OUT, "input.txt")
if not os.path.exists(txt_path):
    print("Downloading tiny-shakespeare...")
    urllib.request.urlretrieve(URL, txt_path)

with open(txt_path, "r") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}

tokens = np.array([stoi[c] for c in text], dtype=np.uint32)
np.save(os.path.join(OUT, "tokens.npy"), tokens)

with open(os.path.join(OUT, "itos.txt"), "w") as f:
    for c in chars:
        f.write(repr(c) + "\n")

print(f"vocab_size = {vocab_size}")
print(f"tokens     = {len(tokens):,}")
print(f"saved to   {OUT}/")
