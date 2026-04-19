import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Data ──────────────────────────────────────────────────────────────────────

X_train = torch.tensor(np.load("data/X_train.npy"), dtype=torch.float32)
y_train = torch.tensor(np.load("data/y_train.npy"), dtype=torch.float32)
X_test  = torch.tensor(np.load("data/X_test.npy"),  dtype=torch.float32)
y_test  = torch.tensor(np.load("data/y_test.npy"),  dtype=torch.float32)

# [N, 784] → [N, 1, 28, 28]
X_train = X_train.reshape(-1, 1, 28, 28)
X_test  = X_test.reshape(-1, 1, 28, 28)

# y is one-hot [N, 10] — convert to class indices for CrossEntropyLoss
y_train_idx = y_train.argmax(dim=1)
y_test_idx  = y_test.argmax(dim=1)

print(f"X_train: {list(X_train.shape)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train_idx = X_train.to(device), y_train_idx.to(device)
X_test,  y_test_idx  = X_test.to(device),  y_test_idx.to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train_idx),
                          batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test, y_test_idx),
                          batch_size=256, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
# Mirrors cnn_model exactly:
#   Conv(1→32, k=3, pad=1) + ReLU + MaxPool(2,2)  → [N, 32, 14, 14]
#   Conv(32→64, k=3, pad=1) + ReLU + MaxPool(2,2) → [N, 64, 7, 7]
#   Flatten                                         → [N, 3136]
#   Linear(3136→256) + ReLU
#   Linear(256→128)  + ReLU
#   Linear(128→10)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(3136, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 10)

        # He (Kaiming normal) init to match the C++ library
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = CNN().to(device)

# ── Training ──────────────────────────────────────────────────────────────────
# SGD with lr=0.01 and L2 weight_decay=1e-4, no momentum — matches sgd in C++

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(30):
    model.train()
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        print(f"Epoch {epoch:2d}  loss: {loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────

model.eval()
total_loss, correct, n_batches = 0.0, 0, 0
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb)
        total_loss += criterion(logits, yb).item()
        correct    += (logits.argmax(1) == yb).sum().item()
        n_batches  += 1

print(f"\nTest loss:     {total_loss / n_batches:.4f}")
print(f"Test accuracy: {correct / len(y_test_idx) * 100:.2f}%")
