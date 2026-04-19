import numpy as np
from tensorflow.keras.datasets import cifar10

print("Downloading CIFAR-10...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize
X_train = X_train.astype(np.float32) / 255.0  # (N, 32, 32, 3)
X_test = X_test.astype(np.float32) / 255.0

# Convert to (N, C, H, W)
X_train = np.transpose(X_train, (0, 3, 1, 2))  # -> (N, 3, 32, 32)
X_test = np.transpose(X_test, (0, 3, 1, 2))

# Ensure contiguous memory layout
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

# Save (already contiguous, but we can enforce again for safety)
np.save("data/X_train.npy", np.ascontiguousarray(X_train))
np.save("data/y_train.npy", np.ascontiguousarray(y_train_oh))
np.save("data/X_test.npy", np.ascontiguousarray(X_test))
np.save("data/y_test.npy", np.ascontiguousarray(y_test_oh))

print(f"Saved X_train {X_train.shape}, y_train {y_train_oh.shape}")
print(f"Saved X_test  {X_test.shape}, y_test  {y_test_oh.shape}")
