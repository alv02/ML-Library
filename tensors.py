import numpy as np

# Matrix A (2x3)
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# Matrix B (3x2)
b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)


print("A shape:", a.shape)

print("B shape:", b.shape)

np.save("a.npy", a)
np.save("b.npy", b)

a = np.transpose(a)
b = np.transpose(b)
c = a @ b
print(c)
