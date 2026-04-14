import numpy as np
import os

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
np.save("matmul_transA/a.npy", a.astype(np.float32))        # saved as (4,3), will be transposed in C++
np.save("matmul_transA/b.npy", b.astype(np.float32))
np.save("matmul_transA/out.npy", (a.T @ b).astype(np.float32))

def save_sum(dir, a):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/out.npy", np.array([a.sum()], dtype=np.float32))

# Small: fits in a single block (< 256 elements)
a = np.random.randn(3, 4).astype(np.float32)
save_sum("sum_small", a)

# Medium: needs 2 passes (> 256, < 256*256 elements)
a = np.random.randn(128, 32).astype(np.float32)   # 4096 elements → 16 blocks → 1 block
save_sum("sum_medium", a)

# Large: needs 3 passes (> 256*256 elements)
a = np.random.randn(512, 512).astype(np.float32)  # 262144 → 1024 blocks → 4 blocks → 1 block
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
save_sum_dim("sum_dim0_2d_keep",    a, dim=0, keep_dim=True)
save_sum_dim("sum_dim0_2d_nokeep",  a, dim=0, keep_dim=False)

# 2D: sum along dim=1 (cols), keep_dim=True  → (3,1)
save_sum_dim("sum_dim1_2d_keep",    a, dim=1, keep_dim=True)
save_sum_dim("sum_dim1_2d_nokeep",  a, dim=1, keep_dim=False)

# 3D: sum along each dim
b = np.random.randn(2, 3, 4).astype(np.float32)
save_sum_dim("sum_dim0_3d_keep",    b, dim=0, keep_dim=True)
save_sum_dim("sum_dim1_3d_keep",    b, dim=1, keep_dim=True)
save_sum_dim("sum_dim2_3d_keep",    b, dim=2, keep_dim=True)
save_sum_dim("sum_dim1_3d_nokeep",  b, dim=1, keep_dim=False)

# Larger tensor
c = np.random.randn(64, 128).astype(np.float32)
save_sum_dim("sum_dim0_large_keep", c, dim=0, keep_dim=True)
save_sum_dim("sum_dim1_large_keep", c, dim=1, keep_dim=True)

def save_index_select(dir, a, indices, axis):
    os.makedirs(dir, exist_ok=True)
    np.save(f"{dir}/a.npy", a.astype(np.float32))
    np.save(f"{dir}/out.npy", np.take(a, indices, axis=axis).astype(np.float32))

# Source: (3,4)
a = np.arange(12, dtype=np.float32).reshape(3, 4)

# dim=0: select rows [0,2] → (2,4)
save_index_select("idx_select_dim0_basic",   a, [0, 2],    axis=0)
# dim=0: permute rows [2,0,1] → (3,4)
save_index_select("idx_select_dim0_permute", a, [2, 0, 1], axis=0)
# dim=0: repeated index [1,1] → (2,4)
save_index_select("idx_select_dim0_repeat",  a, [1, 1],    axis=0)
# dim=1: select cols [1,3] → (3,2)
save_index_select("idx_select_dim1_basic",   a, [1, 3],    axis=1)
# dim=1: select cols [0,2,3] → (3,3)
save_index_select("idx_select_dim1_multi",   a, [0, 2, 3], axis=1)

# Larger source: (8,16) random
b = np.random.randn(8, 16).astype(np.float32)
save_index_select("idx_select_dim0_large",   b, [3, 7, 0, 5], axis=0)
save_index_select("idx_select_dim1_large",   b, [0, 4, 8, 12, 15], axis=1)

print("Test data generated.")
