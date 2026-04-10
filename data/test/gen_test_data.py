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

print("Test data generated.")
