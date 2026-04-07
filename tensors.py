import numpy as np
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()

X = data.data.astype(np.float32)
y = data.target.reshape(-1, 1).astype(np.float32)

# Optional: normalize (VERY recommended for training stability)
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std() + 1e-8
y = (y - y_mean) / y_std

# Save
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Saved X:", X.shape)
print("Saved y:", y.shape)
