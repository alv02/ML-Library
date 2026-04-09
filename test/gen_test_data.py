#!/usr/bin/env python3
"""Generate .npy test data for broadcasting tests.
Run from the test/ directory: python3 gen_test_data.py
Files are written to ../data/ without .npy extension so tensor_load() finds them.
"""
import numpy as np

def save(name, arr):
    """Save as raw npy bytes without .npy extension (matches tensor_load paths)."""
    path = f"../data/{name}"
    with open(path, "wb") as f:
        np.save(f, arr)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")

print("=== Generating broadcast test data ===")

# --- same shape ----------------------------------------------------------
# (5,3) + (5,3) → (5,3)
a = np.ones((5, 3), dtype=np.float32)
b = np.ones((5, 3), dtype=np.float32) * 2.0
save("bcast_same_a", a)
save("bcast_same_b", b)
save("bcast_same_expected", a + b)   # all 3.0

# --- broadcast dim 0 -----------------------------------------------------
# (5,3) + (1,3) → (5,3)
a = np.arange(15, dtype=np.float32).reshape(5, 3)
b = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)  # shape (1,3)
save("bcast_row_a", a)
save("bcast_row_b", b)
save("bcast_row_expected", a + b)

# --- broadcast dim 1 (column vector) -------------------------------------
# (5,3) + (5,1) → (5,3)
a = np.arange(15, dtype=np.float32).reshape(5, 3)
b = np.arange(5, dtype=np.float32).reshape(5, 1) * 100.0
save("bcast_col_a", a)
save("bcast_col_b", b)
save("bcast_col_expected", a + b)

# --- scalar broadcast -----------------------------------------------------
# (5,3) + (1,1) → (5,3)
a = np.arange(15, dtype=np.float32).reshape(5, 3)
b = np.array([[5.0]], dtype=np.float32)              # shape (1,1)
save("bcast_scalar_a", a)
save("bcast_scalar_b", b)
save("bcast_scalar_expected", a + b)

print("Done.")
