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


# ── LayerNorm ─────────────────────────────────────────────────────────────────
B, T, D = 2, 4, 8
ln_input = torch.randn(B, T, D, requires_grad=True)
ln_gamma = torch.randn(D)
ln_beta  = torch.randn(D)

ln = nn.LayerNorm(D, elementwise_affine=True)
ln.weight = nn.Parameter(ln_gamma.clone())
ln.bias   = nn.Parameter(ln_beta.clone())

ln_out = ln(ln_input)
ln_out.backward(torch.ones_like(ln_out))

d = f"{BASE}/layer_norm"
save_f32(ln_input.detach(), f"{d}/input.npy")
save_f32(ln_gamma,          f"{d}/gamma.npy")
save_f32(ln_beta,           f"{d}/beta.npy")
save_f32(ln_out,            f"{d}/out.npy")
save_f32(ln_input.grad,     f"{d}/d_input.npy")
save_f32(ln.weight.grad,    f"{d}/d_gamma.npy")
save_f32(ln.bias.grad,      f"{d}/d_beta.npy")
print(f"layer_norm: input{list(ln_input.shape)} out{list(ln_out.shape)}")

# ── MultiHeadAttention ───────────────────────────────────────────────────────
import math
B, T, D, H = 2, 4, 8, 2
d_head = D // H

x_mha = torch.randn(B, T, D, requires_grad=True)
# Linear stores W as [out, in] and computes x @ W^T + b  (b shape [1, D])
Wq = torch.randn(D, D); bq = torch.randn(1, D)
Wk = torch.randn(D, D); bk = torch.randn(1, D)
Wv = torch.randn(D, D); bv = torch.randn(1, D)
Wo = torch.randn(D, D); bo = torch.randn(1, D)

Q = x_mha @ Wq.T + bq                            # [B, T, D]
K = x_mha @ Wk.T + bk
V = x_mha @ Wv.T + bv

Q = Q.reshape(B, T, H, d_head).transpose(1, 2)   # [B, H, T, d_head]
K = K.reshape(B, T, H, d_head).transpose(1, 2)
V = V.reshape(B, T, H, d_head).transpose(1, 2)

scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)  # [B, H, T, T]

mask = torch.zeros(T, T)
mask[torch.triu(torch.ones(T, T), diagonal=1).bool()] = -1e9
scores = scores + mask

attn = torch.softmax(scores, dim=-1)
out_mha = attn @ V                                     # [B, H, T, d_head]
out_mha = out_mha.transpose(1, 2).reshape(B, T, D)    # [B, T, D]
out_mha = out_mha @ Wo.T + bo

out_mha.backward(torch.ones_like(out_mha))

d = f"{BASE}/mha"
save_f32(x_mha.detach(), f"{d}/x.npy")
save_f32(Wq,             f"{d}/Wq.npy")
save_f32(Wk,             f"{d}/Wk.npy")
save_f32(Wv,             f"{d}/Wv.npy")
save_f32(Wo,             f"{d}/Wo.npy")
save_f32(bq,             f"{d}/bq.npy")
save_f32(bk,             f"{d}/bk.npy")
save_f32(bv,             f"{d}/bv.npy")
save_f32(bo,             f"{d}/bo.npy")
save_f32(out_mha,        f"{d}/out.npy")
save_f32(x_mha.grad,     f"{d}/d_x.npy")
print(f"mha: x{list(x_mha.shape)} out{list(out_mha.shape)}")

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

# ── TransformerBlock ─────────────────────────────────────────────────────────
import torch.nn.functional as F

B, T, D, H = 2, 4, 8, 2
d_head = D // H
mlp_hidden = 4 * D

x_tb = torch.randn(B, T, D, requires_grad=True)
ln1_g = torch.randn(D); ln1_b = torch.randn(D)
ln2_g = torch.randn(D); ln2_b = torch.randn(D)
Wq_tb = torch.randn(D, D); bq_tb = torch.randn(1, D)
Wk_tb = torch.randn(D, D); bk_tb = torch.randn(1, D)
Wv_tb = torch.randn(D, D); bv_tb = torch.randn(1, D)
Wo_tb = torch.randn(D, D); bo_tb = torch.randn(1, D)
W_fc   = torch.randn(mlp_hidden, D); b_fc   = torch.randn(1, mlp_hidden)
W_proj = torch.randn(D, mlp_hidden); b_proj = torch.randn(1, D)

# MHA sublayer (pre-norm, no dropout)
h = F.layer_norm(x_tb, (D,), ln1_g, ln1_b)
Q = (h @ Wq_tb.T + bq_tb).reshape(B, T, H, d_head).transpose(1, 2)
K = (h @ Wk_tb.T + bk_tb).reshape(B, T, H, d_head).transpose(1, 2)
V = (h @ Wv_tb.T + bv_tb).reshape(B, T, H, d_head).transpose(1, 2)
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
mask_tb = torch.zeros(T, T)
mask_tb[torch.triu(torch.ones(T, T), diagonal=1).bool()] = -1e9
attn_tb = torch.softmax(scores + mask_tb, dim=-1)
mha_out = (attn_tb @ V).transpose(1, 2).reshape(B, T, D) @ Wo_tb.T + bo_tb
h1 = x_tb + mha_out

# MLP sublayer (pre-norm, no dropout)
h2 = F.layer_norm(h1, (D,), ln2_g, ln2_b)
mlp_out = F.gelu(h2 @ W_fc.T + b_fc, approximate='tanh') @ W_proj.T + b_proj
out_tb = h1 + mlp_out

out_tb.backward(torch.ones_like(out_tb))

d = f"{BASE}/transformer_block"
save_f32(x_tb.detach(), f"{d}/x.npy")
save_f32(ln1_g,          f"{d}/ln1_g.npy")
save_f32(ln1_b,          f"{d}/ln1_b.npy")
save_f32(ln2_g,          f"{d}/ln2_g.npy")
save_f32(ln2_b,          f"{d}/ln2_b.npy")
save_f32(Wq_tb,          f"{d}/Wq.npy")
save_f32(bq_tb,          f"{d}/bq.npy")
save_f32(Wk_tb,          f"{d}/Wk.npy")
save_f32(bk_tb,          f"{d}/bk.npy")
save_f32(Wv_tb,          f"{d}/Wv.npy")
save_f32(bv_tb,          f"{d}/bv.npy")
save_f32(Wo_tb,          f"{d}/Wo.npy")
save_f32(bo_tb,          f"{d}/bo.npy")
save_f32(W_fc,           f"{d}/W_fc.npy")
save_f32(b_fc,           f"{d}/b_fc.npy")
save_f32(W_proj,         f"{d}/W_proj.npy")
save_f32(b_proj,         f"{d}/b_proj.npy")
save_f32(out_tb,         f"{d}/out.npy")
save_f32(x_tb.grad,      f"{d}/d_x.npy")
print(f"transformer_block: x{list(x_tb.shape)} out{list(out_tb.shape)}")

# ── GPTModel ──────────────────────────────────────────────────────────────────
vocab, D, H, n_layers = 16, 8, 2, 2
max_seq_len, B, T = 8, 2, 4
d_head = D // H
mlp_hidden = 4 * D

tokens_gpt = torch.tensor([[1, 3, 5, 2], [0, 4, 2, 7]], dtype=torch.long)
tok_w_gpt = torch.randn(vocab, D)
pos_w_gpt = torch.randn(max_seq_len, D)

blk = []
for _ in range(n_layers):
    p = {
        'ln1_g': torch.randn(D), 'ln1_b': torch.randn(D),
        'ln2_g': torch.randn(D), 'ln2_b': torch.randn(D),
        'Wq': torch.randn(D, D), 'bq': torch.randn(1, D),
        'Wk': torch.randn(D, D), 'bk': torch.randn(1, D),
        'Wv': torch.randn(D, D), 'bv': torch.randn(1, D),
        'Wo': torch.randn(D, D), 'bo': torch.randn(1, D),
        'W_fc':   torch.randn(mlp_hidden, D), 'b_fc':   torch.randn(1, mlp_hidden),
        'W_proj': torch.randn(D, mlp_hidden), 'b_proj': torch.randn(1, D),
    }
    blk.append(p)

lnf_g = torch.randn(D); lnf_b = torch.randn(D)
W_lmh = torch.randn(vocab, D); b_lmh = torch.randn(1, vocab)

# Forward
tok_emb = nn.Embedding(vocab, D)
tok_emb.weight = nn.Parameter(tok_w_gpt.clone())
pos_emb = nn.Embedding(max_seq_len, D)
pos_emb.weight = nn.Parameter(pos_w_gpt.clone())

x_gpt = tok_emb(tokens_gpt) + pos_emb(torch.arange(T))  # [B, T, D]

for p in blk:
    mask_gpt = torch.zeros(T, T)
    mask_gpt[torch.triu(torch.ones(T, T), diagonal=1).bool()] = -1e9
    h = F.layer_norm(x_gpt, (D,), p['ln1_g'], p['ln1_b'])
    Q = (h @ p['Wq'].T + p['bq']).reshape(B, T, H, d_head).transpose(1, 2)
    K = (h @ p['Wk'].T + p['bk']).reshape(B, T, H, d_head).transpose(1, 2)
    V = (h @ p['Wv'].T + p['bv']).reshape(B, T, H, d_head).transpose(1, 2)
    attn = torch.softmax(Q @ K.transpose(-2,-1) / math.sqrt(d_head) + mask_gpt, dim=-1)
    mha_out = (attn @ V).transpose(1,2).reshape(B, T, D) @ p['Wo'].T + p['bo']
    h1 = x_gpt + mha_out
    h2 = F.layer_norm(h1, (D,), p['ln2_g'], p['ln2_b'])
    x_gpt = h1 + F.gelu(h2 @ p['W_fc'].T + p['b_fc'], approximate='tanh') @ p['W_proj'].T + p['b_proj']

x_gpt = F.layer_norm(x_gpt, (D,), lnf_g, lnf_b)
logits = x_gpt @ W_lmh.T + b_lmh  # [B, T, vocab]
logits.backward(torch.ones_like(logits))

d = f"{BASE}/gpt_model"
save_u32(tokens_gpt,         f"{d}/tokens.npy")
save_f32(tok_w_gpt,          f"{d}/tok_w.npy")
save_f32(pos_w_gpt,          f"{d}/pos_w.npy")
for i, p in enumerate(blk):
    for k, v in p.items():
        save_f32(v, f"{d}/block{i}_{k}.npy")
save_f32(lnf_g,              f"{d}/lnf_g.npy")
save_f32(lnf_b,              f"{d}/lnf_b.npy")
save_f32(W_lmh,              f"{d}/W_lmh.npy")
save_f32(b_lmh,              f"{d}/b_lmh.npy")
save_f32(logits,             f"{d}/out.npy")
save_f32(tok_emb.weight.grad,f"{d}/d_tok_w.npy")
print(f"gpt_model: tokens{list(tokens_gpt.shape)} logits{list(logits.shape)}")

print("\nDone.")
