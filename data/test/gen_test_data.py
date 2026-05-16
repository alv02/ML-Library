import os

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(42)


def save(dir, a, b):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/b.npy", b.astype(np.float32))
    np.save(f"{dir}/out.npy", (a + b).astype(np.float32))


# Same shape
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 4).astype(np.float32)
save("add_same", a, b)

# Broadcast row: (3,4) + (1,4)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(1, 4).astype(np.float32)
save("add_bcast_row", a, b)

# Broadcast col: (3,4) + (3,1)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(3, 1).astype(np.float32)
save("add_bcast_col", a, b)

# Broadcast scalar: (3,4) + (1,1)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(1, 1).astype(np.float32)
save("add_bcast_scalar", a, b)

# Different ndim: (3,4) + (4,)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4).astype(np.float32)
save("add_diff_ndim", a, b)


def save_matmul(dir, a, b):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/b.npy", b.astype(np.float32))
    np.save(f"{dir}/out.npy", (a @ b).astype(np.float32))


# Square: (4,4) @ (4,4)
a = np.random.randn(4, 4).astype(np.float32)
b = np.random.randn(4, 4).astype(np.float32)
save_matmul("matmul_square", a, b)

# Non-square: (3,4) @ (4,5) → (3,5)
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)
save_matmul("matmul_rect", a, b)

# Transposed A: (4,3)^T @ (4,5) → (3,5)  — tests non-contiguous strides
a = np.random.randn(4, 3).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)
os.makedirs("matmul_transA", exist_ok=True)
np.save(
    "matmul_transA/a.npy", a.astype(np.float32)
)  # saved as (4,3), will be transposed in C++
np.save("matmul_transA/b.npy", b.astype(np.float32))
np.save("matmul_transA/out.npy", (a.T @ b).astype(np.float32))


def save_sum(dir, a):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/out.npy", a.sum().astype(np.float32))


# Small: fits in a single block (< 256 elements)
a = np.random.randn(3, 4).astype(np.float32)
save_sum("sum_small", a)

# Medium: needs 2 passes (> 256, < 256*256 elements)
a = np.random.randn(128, 32).astype(np.float32)  # 4096 elements → 16 blocks → 1 block
save_sum("sum_medium", a)

# Large: needs 3 passes (> 256*256 elements)
a = np.random.randn(512, 512).astype(
    np.float32
)  # 262144 → 1024 blocks → 4 blocks → 1 block
save_sum("sum_large", a)

# Single element edge case
a = np.random.randn(1, 1).astype(np.float32)
save_sum("sum_single", a)


def save_sum_dim(dir, a, dim, keep_dim):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/out.npy", a.sum(axis=dim, keepdims=keep_dim).astype(np.float32))


# 2D: sum along dim=0 (rows), keep_dim=True  → (1,4)
a = np.random.randn(3, 4).astype(np.float32)
save_sum_dim("sum_dim0_2d_keep", a, dim=0, keep_dim=True)
save_sum_dim("sum_dim0_2d_nokeep", a, dim=0, keep_dim=False)

# 2D: sum along dim=1 (cols), keep_dim=True  → (3,1)
save_sum_dim("sum_dim1_2d_keep", a, dim=1, keep_dim=True)
save_sum_dim("sum_dim1_2d_nokeep", a, dim=1, keep_dim=False)

# 3D: sum along each dim
b = np.random.randn(2, 3, 4).astype(np.float32)
save_sum_dim("sum_dim0_3d_keep", b, dim=0, keep_dim=True)
save_sum_dim("sum_dim1_3d_keep", b, dim=1, keep_dim=True)
save_sum_dim("sum_dim2_3d_keep", b, dim=2, keep_dim=True)
save_sum_dim("sum_dim1_3d_nokeep", b, dim=1, keep_dim=False)

# Larger tensor
c = np.random.randn(64, 128).astype(np.float32)
save_sum_dim("sum_dim0_large_keep", c, dim=0, keep_dim=True)
save_sum_dim("sum_dim1_large_keep", c, dim=1, keep_dim=True)


# ── tensor_unfold2d ───────────────────────────────────────────────────────────


def save_unfold2d(dir_name, inp, kernel_hw, stride_hw=(1, 1), pad_hw=(0, 0)):
    os.makedirs(dir_name, exist_ok=True)
    # F.unfold returns [N, C*kH*kW, L]; transpose to [N, L, C*kH*kW]
    out = (
        F.unfold(
            torch.from_numpy(inp),
            kernel_size=kernel_hw,
            stride=stride_hw,
            padding=pad_hw,
        )
        .permute(0, 2, 1)
        .contiguous()
        .numpy()
    )
    np.save(f"{dir_name}/a.npy", inp.astype(np.float32))
    np.save(f"{dir_name}/out.npy", out.astype(np.float32))


# 1×1×4×4 — simplest case, k=3×3, stride=1, pad=0
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_unfold2d("unfold2d_1c_4x4_k3s1", a, (3, 3))

# 1×2×3×3 — multi-channel, k=2×2, stride=1, pad=0
a = np.concatenate(
    [
        np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        np.arange(10, 19, dtype=np.float32).reshape(1, 1, 3, 3),
    ],
    axis=1,
)
save_unfold2d("unfold2d_2c_3x3_k2s1", a, (2, 2))

# 2×1×4×4 — batch, k=3×3, stride=1, pad=0
a = np.stack(
    [
        np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        np.arange(100, 116, dtype=np.float32).reshape(1, 4, 4),
    ]
).reshape(2, 1, 4, 4)
save_unfold2d("unfold2d_batch_4x4_k3s1", a, (3, 3))

# stride tests (pad=0)
# 1×1×6×6 — k=3×3, stride=2 → L=(6-3)/2+1=2, output [1,2,2,1,3,3]
a = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
save_unfold2d("unfold2d_1c_6x6_k3s2", a, (3, 3), stride_hw=(2, 2))

# 1×1×5×5 — k=2×2, stride=2 → L=(5-2)/2+1=2, output [1,2,2,1,2,2]
a = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
save_unfold2d("unfold2d_1c_5x5_k2s2", a, (2, 2), stride_hw=(2, 2))

# padding tests (stride=1)
# 1×1×3×3 — k=3×3, pad=1 → L=(3+2-3)/1+1=3, output [1,3,3,1,3,3]
a = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
save_unfold2d("unfold2d_1c_3x3_k3p1", a, (3, 3), pad_hw=(1, 1))

# 1×1×4×4 — k=3×3, pad=1 → L=(4+2-3)/1+1=4, output [1,4,4,1,3,3]
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_unfold2d("unfold2d_1c_4x4_k3p1", a, (3, 3), pad_hw=(1, 1))

# stride + padding combined
# 1×1×4×4 — k=3×3, stride=2, pad=1 → L=(4+2-3)/2+1=2, output [1,2,2,1,3,3]
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_unfold2d("unfold2d_1c_4x4_k3s2p1", a, (3, 3), stride_hw=(2, 2), pad_hw=(1, 1))

# ── tensor_unfold2d non-contiguous input ─────────────────────────────────────
# Input saved as [C, N, H, W] (N and C swapped). In C++ the test loads it and
# calls tensor_transpose(0,1) to get a non-contiguous [N, C, H, W] view, then
# runs unfold — expected output is identical to the contiguous version.


def save_unfold2d_noncontig(dir_name, inp, kernel_hw, stride_hw=(1, 1), pad_hw=(0, 0)):
    os.makedirs(dir_name, exist_ok=True)
    out = (
        F.unfold(
            torch.from_numpy(inp),
            kernel_size=kernel_hw,
            stride=stride_hw,
            padding=pad_hw,
        )
        .permute(0, 2, 1)
        .contiguous()
        .numpy()
    )
    # save a transposed as [C, N, H, W] so C++ can make it non-contiguous
    np.save(f"{dir_name}/a.npy", inp.transpose(1, 0, 2, 3).astype(np.float32))
    np.save(f"{dir_name}/out.npy", out.astype(np.float32))


# 1×2×3×3 — multi-channel (C=2, N=1): transpose gives [2,1,3,3]
a = np.concatenate(
    [
        np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        np.arange(10, 19, dtype=np.float32).reshape(1, 1, 3, 3),
    ],
    axis=1,
)
save_unfold2d_noncontig("unfold2d_noncontig_2c_3x3_k2s1", a, (2, 2))

# 2×1×4×4 — batch (N=2, C=1): transpose gives [1,2,4,4]
a = np.stack(
    [
        np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        np.arange(100, 116, dtype=np.float32).reshape(1, 4, 4),
    ]
).reshape(2, 1, 4, 4)
save_unfold2d_noncontig("unfold2d_noncontig_batch_4x4_k3s1", a, (3, 3))

# 2×3×4×4 — batch + multichannel (N=2, C=3): transpose gives [3,2,4,4]
a = np.arange(96, dtype=np.float32).reshape(2, 3, 4, 4)
save_unfold2d_noncontig("unfold2d_noncontig_2n3c_4x4_k3s1", a, (3, 3))

# padding + non-contiguous
a = np.arange(96, dtype=np.float32).reshape(2, 3, 4, 4)
save_unfold2d_noncontig("unfold2d_noncontig_2n3c_4x4_k3p1", a, (3, 3), pad_hw=(1, 1))

# ── tensor_fold2d ─────────────────────────────────────────────────────────────
# Saves col [N*L, C*kH*kW] (a.npy) and expected folded output [N,C,H,W] (out.npy).
# col is produced by unfolding inp so that fold(unfold(inp)) == expected, which
# PyTorch computes via F.fold.


def save_fold2d(dir_name, inp, kernel_hw, stride_hw=(1, 1), pad_hw=(0, 0)):
    os.makedirs(dir_name, exist_ok=True)
    N, C, H, W = inp.shape
    t = torch.from_numpy(inp)
    # F.unfold → [N, C*kH*kW, L]; transpose to [N, L, C*kH*kW] then flatten N*L
    col_nckl = F.unfold(t, kernel_size=kernel_hw, stride=stride_hw, padding=pad_hw)
    L = col_nckl.shape[2]
    col_flat = (
        col_nckl.permute(0, 2, 1).contiguous().reshape(N * L, -1)
    )  # [N*L, C*kH*kW]
    # F.fold expects [N, C*kH*kW, L]
    out = F.fold(
        col_nckl,
        output_size=(H, W),
        kernel_size=kernel_hw,
        stride=stride_hw,
        padding=pad_hw,
    )
    np.save(f"{dir_name}/a.npy", col_flat.numpy().astype(np.float32))
    np.save(f"{dir_name}/out.npy", out.numpy().astype(np.float32))


# 1×1×4×4  k=3 s=1 p=0
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_fold2d("fold2d_1c_4x4_k3s1", a, (3, 3))

# 1×2×3×3  k=2 s=1 p=0  — multi-channel
a = np.concatenate(
    [
        np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3),
        np.arange(10, 19, dtype=np.float32).reshape(1, 1, 3, 3),
    ],
    axis=1,
)
save_fold2d("fold2d_2c_3x3_k2s1", a, (2, 2))

# 2×1×4×4  k=3 s=1 p=0  — batch
a = np.stack(
    [
        np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        np.arange(100, 116, dtype=np.float32).reshape(1, 4, 4),
    ]
).reshape(2, 1, 4, 4)
save_fold2d("fold2d_batch_4x4_k3s1", a, (3, 3))

# 1×1×6×6  k=3 s=2 p=0  — stride, no overlap so fold == inverse of unfold
a = np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6)
save_fold2d("fold2d_1c_6x6_k3s2", a, (3, 3), stride_hw=(2, 2))

# 1×1×4×4  k=3 s=1 p=1  — padding
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_fold2d("fold2d_1c_4x4_k3p1", a, (3, 3), pad_hw=(1, 1))

# 1×1×4×4  k=3 s=2 p=1  — stride + padding
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_fold2d("fold2d_1c_4x4_k3s2p1", a, (3, 3), stride_hw=(2, 2), pad_hw=(1, 1))

# 2×3×4×4  k=3 s=1 p=0  — batch + multi-channel (overlapping windows, stress test)
a = np.arange(96, dtype=np.float32).reshape(2, 3, 4, 4)
save_fold2d("fold2d_2n3c_4x4_k3s1", a, (3, 3))


# ── MaxPool2d forward (full pipeline check) ───────────────────────────────────
# We test the exact sequence MaxPool2dOp::forward performs:
#   unfold2d → reshape[N,L,C,K] → max(dim=3) → reshape[N,Lh,Lw,C] →
#   transpose(1,3) → transpose(2,3) → result [N,C,Lh,Lw]
# We save input (a.npy) and expected output (out.npy).


def save_maxpool2d(dir_name, inp, k, stride, pad):
    os.makedirs(dir_name, exist_ok=True)
    t = torch.from_numpy(inp)
    out = torch.nn.functional.max_pool2d(t, kernel_size=k, stride=stride, padding=pad)
    np.save(f"{dir_name}/a.npy", inp.astype(np.float32))
    np.save(f"{dir_name}/out.npy", out.numpy().astype(np.float32))


# 1×1×4×4  pool k=2 s=2 p=0  → [1,1,2,2]
a = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
save_maxpool2d("maxpool_1c_4x4_k2s2", a, k=2, stride=2, pad=0)

# 1×2×4×4  pool k=2 s=2 p=0  → [1,2,2,2]
a = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
save_maxpool2d("maxpool_2c_4x4_k2s2", a, k=2, stride=2, pad=0)

# 2×3×6×6  pool k=2 s=2 p=0  → [2,3,3,3]
a = np.random.randn(2, 3, 6, 6).astype(np.float32)
save_maxpool2d("maxpool_2n3c_6x6_k2s2", a, k=2, stride=2, pad=0)

# 2×4×6×6  pool k=3 s=1 p=0  → [2,4,4,4]
a = np.random.randn(2, 4, 6, 6).astype(np.float32)
save_maxpool2d("maxpool_2n4c_6x6_k3s1", a, k=3, stride=1, pad=0)


# ── tensor_welford_mean_var ───────────────────────────────────────────────────
# The implementation returns raw M2 (sum of squared deviations from the mean),
# not variance. Callers divide by count themselves. Compute in float64 for a
# high-precision reference.
def save_welford(dir_name, inp, dim):
    os.makedirs(dir_name, exist_ok=True)
    a = inp.astype(np.float64)
    all_dims = tuple(i for i in range(a.ndim) if i != dim)
    mean = a.mean(axis=all_dims)
    shape = [1] * a.ndim
    shape[dim] = a.shape[dim]
    m2 = np.sum((a - mean.reshape(shape)) ** 2, axis=all_dims)
    np.save(f"{dir_name}/a.npy", inp.astype(np.float32))
    np.save(f"{dir_name}/mean.npy", mean.astype(np.float32))
    np.save(f"{dir_name}/var.npy", m2.astype(np.float32))


a = np.random.randn(4, 3).astype(np.float32)
save_welford("welford_2d_dim0", a, dim=0)

a = np.random.randn(2, 4, 3, 3).astype(np.float32)
save_welford("welford_4d_2n4c_3x3", a, dim=1)

a = np.random.randn(4, 8, 6, 6).astype(np.float32)
save_welford("welford_4d_4n8c_6x6", a, dim=1)

a = np.random.randn(8, 16, 8, 8).astype(np.float32)
save_welford("welford_4d_8n16c_8x8", a, dim=1)

a = np.random.randn(16, 32, 16, 16).astype(np.float32)
save_welford("welford_4d_16n32c_16x16", a, dim=1)

# ── tensor_mat_mul batched ───────────────────────────────────────────────────
# trans_a/trans_b indicate the expected computation; the raw tensors are always
# saved in their pre-transpose shape so the C++ test can transpose with strides
# and verify the non-contiguous path through cuBLAS.


def save_matmul_batched(dir, a, b, trans_a=False, trans_b=False):
    os.makedirs(dir, exist_ok=True)
    at = np.swapaxes(a, -1, -2) if trans_a else a
    bt = np.swapaxes(b, -1, -2) if trans_b else b
    np.save(f"{dir}/a.npy", a.astype(np.float32))  # original shape, C++ transposes
    np.save(f"{dir}/b.npy", b.astype(np.float32))
    np.save(f"{dir}/out.npy", (at @ bt).astype(np.float32))


# 3D basic: [2,3,4] @ [2,4,5] → [2,3,5]
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 5).astype(np.float32)
save_matmul_batched("matmul_batched_3d", a, b)

# 3D trans_b: [2,3,4] @ [2,5,4]^T → [2,3,5]  (attention forward: Q @ K^T)
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 5, 4).astype(np.float32)  # stored [2,5,4], transposed in C++
save_matmul_batched("matmul_batched_3d_transB", a, b, trans_b=True)

# 3D trans_a: [2,4,3]^T @ [2,4,5] → [2,3,5]  (attention backward: dV, dK)
a = np.random.randn(2, 4, 3).astype(np.float32)  # stored [2,4,3], transposed in C++
b = np.random.randn(2, 4, 5).astype(np.float32)
save_matmul_batched("matmul_batched_3d_transA", a, b, trans_a=True)

# 4D basic: [2,3,4,5] @ [2,3,5,6] → [2,3,4,6]  (multi-head attention values)
a = np.random.randn(2, 3, 4, 5).astype(np.float32)
b = np.random.randn(2, 3, 5, 6).astype(np.float32)
save_matmul_batched("matmul_batched_4d", a, b)

# 4D trans_b: [2,3,4,5] @ [2,3,6,5]^T → [2,3,4,6]  (multi-head Q @ K^T)
a = np.random.randn(2, 3, 4, 5).astype(np.float32)
b = np.random.randn(2, 3, 6, 5).astype(
    np.float32
)  # stored [2,3,6,5], transposed in C++
save_matmul_batched("matmul_batched_4d_transB", a, b, trans_b=True)

# broadcast: lower-ndim b is broadcast across all batch dims of a
# 3D × 2D: [2,3,4] @ [4,5] → [2,3,5]  (Linear applied to batched input)
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(4, 5).astype(np.float32)
save_matmul_batched("matmul_bcast_3d2d", a, b)

# 4D × 2D: [2,3,4,5] @ [5,6] → [2,3,4,6]  (Linear inside MHA)
a = np.random.randn(2, 3, 4, 5).astype(np.float32)
b = np.random.randn(5, 6).astype(np.float32)
save_matmul_batched("matmul_bcast_4d2d", a, b)

# 4D × 2D trans_b: [2,3,4,5] @ [6,5]^T → [2,3,4,6]  (Linear with W^T broadcast)
a = np.random.randn(2, 3, 4, 5).astype(np.float32)
b = np.random.randn(6, 5).astype(np.float32)  # stored [6,5], transposed in C++
save_matmul_batched("matmul_bcast_4d2d_transB", a, b, trans_b=True)


# ── gather ────────────────────────────────────────────────────────────────────
# np.take(src, indices, axis=dim) matches our gather semantics:
#   out[..outer.., *indices_coords, ..inner..] = src[..outer.., indices[*], ..inner..]


def save_gather(dir_name, src, indices, dim):
    os.makedirs(dir_name, exist_ok=True)
    out = np.take(src, indices, axis=dim)
    np.save(f"{dir_name}/a.npy", src.astype(np.float32))
    np.save(f"{dir_name}/idx.npy", indices.astype(np.uint32))
    np.save(f"{dir_name}/out.npy", out.astype(np.float32))


# 2D src, 1D indices, dim=0: [6,4] → take 5 rows → [5,4]
src = np.random.randn(6, 4).astype(np.float32)
indices = np.array([2, 0, 4, 1, 5], dtype=np.uint32)
save_gather("gather_2d_1didx_dim0", src, indices, 0)

# 2D src, 2D indices, dim=0: [8,4] + [2,5] idx → [2,5,4]  (batch embedding lookup)
src = np.random.randn(8, 4).astype(np.float32)
indices = np.array([[0, 3, 7, 1, 2], [4, 6, 0, 5, 3]], dtype=np.uint32)
save_gather("gather_2d_2didx_dim0", src, indices, 0)

# 3D src, 1D indices, dim=1: [3,6,4] → select 4 channels → [3,4,4]
src = np.random.randn(3, 6, 4).astype(np.float32)
indices = np.array([1, 4, 0, 3], dtype=np.uint32)
save_gather("gather_3d_1didx_dim1", src, indices, 1)


def save_scatter_add(dir_name, src, indices, dim, out_shape):
    os.makedirs(dir_name, exist_ok=True)
    out = np.zeros(out_shape, dtype=np.float64)
    inner_dims = len(out_shape) - dim - 1
    idx_ndim = src.ndim - inner_dims
    src_flat = src.reshape(-1, *src.shape[idx_ndim:])  # [prod_idx_dims, *inner]
    idx_flat = indices.ravel()
    np.add.at(out, idx_flat, src_flat)
    np.save(f"{dir_name}/a.npy", src.astype(np.float32))
    np.save(f"{dir_name}/idx.npy", indices.astype(np.uint32))
    np.save(f"{dir_name}/out.npy", out.astype(np.float32))


# In-place dim=0: src [2,5,4], idx [2,5], out [8,4]  (embedding backward)
src = np.random.randn(2, 5, 4).astype(np.float32)
indices = np.random.randint(0, 8, size=(2, 5)).astype(np.uint32)
save_scatter_add("scatter_add_inplace_dim0", src, indices, 0, (8, 4))

print("Test data generated.")
