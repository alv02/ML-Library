import numpy as np
from sklearn.datasets import fetch_openml

print("Downloading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

X = mnist.data.astype(np.float32) / 255.0       # [70000, 784], normalized to [0, 1]
y_int = mnist.target.astype(np.int32)            # [70000]

# One-hot encode labels → [70000, 10]
y = np.zeros((len(y_int), 10), dtype=np.float32)
y[np.arange(len(y_int)), y_int] = 1.0

# Small subset for testing: 100 train, 100 test
X_train, X_test = X[:100], X[60000:60100]
y_train, y_test = y[:100], y[60000:60100]

np.save("data/X_train.npy", np.ascontiguousarray(X_train))
np.save("data/y_train.npy", np.ascontiguousarray(y_train))
np.save("data/X_test.npy",  np.ascontiguousarray(X_test))
np.save("data/y_test.npy",  np.ascontiguousarray(y_test))

print(f"Saved X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Saved X_test  {X_test.shape},  y_test  {y_test.shape}")
