import numpy as np
from tensorflow.keras.datasets import cifar10, mnist

# ---------------- CIFAR-10 ----------------
print("Downloading CIFAR-10...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Convert to (N, C, H, W)
X_train = np.transpose(X_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

# Ensure contiguous
X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot encode
y_train_oh = np.zeros((len(y_train), 10), dtype=np.float32)
y_train_oh[np.arange(len(y_train)), y_train] = 1.0

y_test_oh = np.zeros((len(y_test), 10), dtype=np.float32)
y_test_oh[np.arange(len(y_test)), y_test] = 1.0

# Save CIFAR
np.save("data/cifar_X_train.npy", X_train)
np.save("data/cifar_y_train.npy", y_train_oh)
np.save("data/cifar_X_test.npy", X_test)
np.save("data/cifar_y_test.npy", y_test_oh)

print(f"CIFAR saved: {X_train.shape}, {y_train_oh.shape}")


# ---------------- MNIST ----------------
print("Downloading MNIST...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.astype(np.float32) / 255.0  # (N, 28, 28)
X_test = X_test.astype(np.float32) / 255.0

# Add channel dimension (grayscale → 1 channel)
X_train = X_train[:, np.newaxis, :, :]  # (N, 1, 28, 28)
X_test = X_test[:, np.newaxis, :, :]

# Ensure contiguous
X_train = np.ascontiguousarray(X_train)
X_test = np.ascontiguousarray(X_test)

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# One-hot encode
y_train_oh = np.zeros((len(y_train), 10), dtype=np.float32)
y_train_oh[np.arange(len(y_train)), y_train] = 1.0

y_test_oh = np.zeros((len(y_test), 10), dtype=np.float32)
y_test_oh[np.arange(len(y_test)), y_test] = 1.0

# Save MNIST with different names
np.save("data/mnist_X_train.npy", X_train)
np.save("data/mnist_y_train.npy", y_train_oh)
np.save("data/mnist_X_test.npy", X_test)
np.save("data/mnist_y_test.npy", y_test_oh)

print(f"MNIST saved: {X_train.shape}, {y_train_oh.shape}")
