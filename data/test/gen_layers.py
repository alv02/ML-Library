"""Generate PyTorch reference data for test_layer.cpp."""
import os
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))


def save_f32(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, t.detach().numpy().astype(np.float32))


def save_u32(t, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, t.numpy().astype(np.uint32))


# ── EmbeddingLayer ────────────────────────────────────────────────────────────
V, D, B, T = 10, 8, 2, 4
weight = torch.randn(V, D)
tokens = torch.tensor([[1, 3, 5, 2], [0, 4, 2, 7]], dtype=torch.long)

emb = nn.Embedding(V, D)
emb.weight = nn.Parameter(weight.clone())
out = emb(tokens)
out.backward(torch.ones_like(out))

d = f"{BASE}/emb_layer"
save_f32(weight,         f"{d}/weight.npy")
save_u32(tokens,         f"{d}/tokens.npy")
save_f32(out,            f"{d}/out.npy")
save_f32(emb.weight.grad,f"{d}/d_weight.npy")
print(f"emb_layer: weight{list(weight.shape)} tokens{list(tokens.shape)} out{list(out.shape)}")

# ── PositionalEmbeddingLayer ──────────────────────────────────────────────────
MAX_T, D, T = 8, 6, 4
pos_weight = torch.randn(MAX_T, D)

pos_emb = nn.Embedding(MAX_T, D)
pos_emb.weight = nn.Parameter(pos_weight.clone())
out_pos = pos_emb(torch.arange(T, dtype=torch.long))
out_pos.backward(torch.ones_like(out_pos))

d = f"{BASE}/pos_emb_layer"
save_f32(pos_weight,          f"{d}/weight.npy")
save_f32(out_pos,             f"{d}/out.npy")
save_f32(pos_emb.weight.grad, f"{d}/d_weight.npy")
print(f"pos_emb_layer: weight{list(pos_weight.shape)} out{list(out_pos.shape)}")

# ── InputEmbedding (dropout_p=0 for determinism) ──────────────────────────────
V, MAX_T, D, B, T = 10, 8, 6, 2, 4
tok_w = torch.randn(V, D)
pos_w = torch.randn(MAX_T, D)
tokens2 = torch.tensor([[1, 3, 5, 2], [0, 4, 2, 7]], dtype=torch.long)

tok_emb = nn.Embedding(V, D)
tok_emb.weight = nn.Parameter(tok_w.clone())
pos_emb2 = nn.Embedding(MAX_T, D)
pos_emb2.weight = nn.Parameter(pos_w.clone())

out2 = tok_emb(tokens2) + pos_emb2(torch.arange(T, dtype=torch.long))
out2.backward(torch.ones_like(out2))

d = f"{BASE}/input_emb"
save_f32(tok_w,              f"{d}/tok_weight.npy")
save_f32(pos_w,              f"{d}/pos_weight.npy")
save_u32(tokens2,            f"{d}/tokens.npy")
save_f32(out2,               f"{d}/out.npy")
save_f32(tok_emb.weight.grad,f"{d}/d_tok_weight.npy")
save_f32(pos_emb2.weight.grad,f"{d}/d_pos_weight.npy")
print(f"input_emb: out{list(out2.shape)}")

print("\nDone.")
