import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, criterion, X_test, y_test_idx):
    model.eval()
    total_loss, correct = 0.0, 0
    loader = DataLoader(TensorDataset(X_test, y_test_idx),
                        batch_size=256, shuffle=False)
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb)
            total_loss += criterion(logits, yb).item()
            correct    += (logits.argmax(1) == yb).sum().item()
    n = len(loader)
    print(f"\nTest loss:     {total_loss / n:.4f}")
    print(f"Test accuracy: {correct / len(y_test_idx) * 100:.2f}%\n")

def train(model, criterion, optimizer, loader, epochs, log_every=5):
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % log_every == 0:
            print(f"Epoch {epoch:2d}  loss: {loss.item():.4f}")

def train_with_scheduler(model, criterion, optimizer, scheduler, loader,
                         epochs, early_stop_patience, log_every=5):
    best_loss = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if epoch % log_every == 0:
            print(f"Epoch {epoch+1:3d}  loss: {avg_loss:.4f}  lr: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(avg_loss)
        if best_loss - avg_loss > 1e-4:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop_patience:
            print(f"EarlyStopping at epoch {epoch+1}")
            break

# ── Model definitions ─────────────────────────────────────────────────────────

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 1024), nn.ReLU(),
            nn.Linear(1024, 512),  nn.ReLU(),
            nn.Linear(512,  256),  nn.ReLU(),
            nn.Linear(256,  10),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,  128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, 256),         nn.ReLU(),
            nn.Linear(256, 10),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_bn(in_c, out_c):
    return [nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU()]


# ── ResNet-18 (matches make_resnet in models.cpp) ─────────────────────────────
# Stem: Conv(3→64,k=3,s=1,p=1)+BN+ReLU  (no initial MaxPool — CIFAR-10 32×32)
# 4 stages × {2,2,2,2} ResBlocks, channels {64,128,256,512}
# Stage 0: stride=1. Stages 1-3: first block stride=2, rest stride=1.
# Head: MaxPool(4,4) → Flatten → Linear(512,10)
# Projection shortcut (1×1 Conv+BN) when channels or stride change.

class ResBlockPT(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),
            nn.Conv2d(C_out, C_out, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(C_out),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(C_out),
        ) if (C_in != C_out or stride != 1) else None

    def forward(self, x):
        skip = self.proj(x) if self.proj else x
        return torch.relu(self.residual(x) + skip)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        channels = [64, 128, 256, 512]
        stage_blocks = [2, 2, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        stages = []
        C_in = 64
        for s, n_blocks in enumerate(stage_blocks):
            C_out = channels[s]
            first_stride = 1 if s == 0 else 2
            for b in range(n_blocks):
                stages.append(ResBlockPT(C_in, C_out, first_stride if b == 0 else 1))
                C_in = C_out
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))

class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn(3,   64),  *conv_bn(64,  64),  nn.MaxPool2d(2, 2),
            *conv_bn(64,  128), *conv_bn(128, 128), nn.MaxPool2d(2, 2),
            *conv_bn(128, 256), *conv_bn(256, 256), nn.MaxPool2d(2, 2),
            *conv_bn(256, 512), *conv_bn(512, 512), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Augmented dataset ─────────────────────────────────────────────────────────

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X          # CPU tensors [N, C, H, W]
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train = torch.tensor(np.load("data/X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load("data/y_train.npy"), dtype=torch.float32)
    X_test  = torch.tensor(np.load("data/X_test.npy"),  dtype=torch.float32)
    y_test  = torch.tensor(np.load("data/y_test.npy"),  dtype=torch.float32)

    y_train_idx = y_train.argmax(dim=1)
    y_test_idx  = y_test.argmax(dim=1)

    print(f"X_train: {list(X_train.shape)}")

    # CPU copies kept for augmented dataset (transforms run on CPU)
    X_train_cpu     = X_train.clone()
    y_train_idx_cpu = y_train_idx.clone()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train_idx = X_train.to(device), y_train_idx.to(device)
    X_test,  y_test_idx  = X_test.to(device),  y_test_idx.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── main_nn — dense only ──────────────────────────────────────────────────
    # 3072 → 1024 → 512 → 256 → 10
    # SGD lr=0.01, weight_decay=1e-4, momentum=0.9, batch=128, epochs=50

    # print("=" * 50)
    # print("main_nn  (dense only)")
    # print("=" * 50)
    # nn_model  = NN().to(device)
    # nn_optim  = torch.optim.SGD(nn_model.parameters(), lr=0.01,
    #                             weight_decay=1e-4, momentum=0.9)
    # nn_loader = DataLoader(TensorDataset(X_train, y_train_idx),
    #                        batch_size=128, shuffle=True)
    # train(nn_model, criterion, nn_optim, nn_loader, epochs=50)
    # evaluate(nn_model, criterion, X_test, y_test_idx)

    # ── main_cnn — strided conv, no pool ──────────────────────────────────────
    # Conv(3→32,k=3,s=1,p=1) → Conv(32→64,k=3,s=2,p=1)
    # → Conv(64→128,k=3,s=2,p=1) → Conv(128→256,k=3,s=2,p=1)
    # → Flatten → 4096 → 512 → 10
    # SGD lr=0.01, weight_decay=1e-4, momentum=0.9, batch=64, epochs=50

    # print("=" * 50)
    # print("main_cnn  (strided conv, no pool)")
    # print("=" * 50)
    # cnn_model  = CNN().to(device)
    # cnn_optim  = torch.optim.SGD(cnn_model.parameters(), lr=0.01,
    #                              weight_decay=1e-4, momentum=0.9)
    # cnn_loader = DataLoader(TensorDataset(X_train, y_train_idx),
    #                         batch_size=64, shuffle=True)
    # train(cnn_model, criterion, cnn_optim, cnn_loader, epochs=50)
    # evaluate(cnn_model, criterion, X_test, y_test_idx)

    # ── main_cnn2 — conv + MaxPool ────────────────────────────────────────────
    # Conv(3→32,k=3,p=1) + MaxPool(2,2)  → [N,32,16,16]
    # Conv(32→64,k=3,p=1) + MaxPool(2,2) → [N,64,8,8]
    # Conv(64→128,k=3,p=1)+ MaxPool(2,2) → [N,128,4,4]
    # → Flatten → 2048 → 512 → 256 → 10
    # SGD lr=0.005, weight_decay=5e-4, momentum=0.9, batch=64, epochs=60

    # print("=" * 50)
    # print("main_cnn2  (conv + MaxPool)")
    # print("=" * 50)
    # cnn2_model  = CNN2().to(device)
    # cnn2_optim  = torch.optim.SGD(cnn2_model.parameters(), lr=0.005,
    #                               weight_decay=5e-4, momentum=0.9)
    # cnn2_loader = DataLoader(TensorDataset(X_train, y_train_idx),
    #                          batch_size=64, shuffle=True)
    # train(cnn2_model, criterion, cnn2_optim, cnn2_loader, epochs=60)
    # evaluate(cnn2_model, criterion, X_test, y_test_idx)

    # ── main_cnn3 — VGG-13-style + BatchNorm ─────────────────────────────────
    # Block 1: Conv(3→64)+BN+ReLU,   Conv(64→64)+BN+ReLU,   MaxPool → [N,64,16,16]
    # Block 2: Conv(64→128)+BN+ReLU,  Conv(128→128)+BN+ReLU,  MaxPool → [N,128,8,8]
    # Block 3: Conv(128→256)+BN+ReLU, Conv(256→256)+BN+ReLU,  MaxPool → [N,256,4,4]
    # Block 4: Conv(256→512)+BN+ReLU, Conv(512→512)+BN+ReLU,  MaxPool → [N,512,2,2]
    # → Flatten → 2048 → 512 → 10
    # SGD lr=0.05, weight_decay=5e-4, momentum=0.9, batch=64, epochs=100
    # ReduceLROnPlateau factor=0.1, patience=5 — EarlyStopping patience=10

    # print("=" * 50)
    # print("main_cnn3  (VGG-13-style + BatchNorm)")
    # print("=" * 50)
    # cnn3_model     = CNN3().to(device)
    # cnn3_optim     = torch.optim.SGD(cnn3_model.parameters(), lr=0.05,
    #                                  weight_decay=5e-4, momentum=0.9)
    # cnn3_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                      cnn3_optim, factor=0.1, patience=5)
    # cnn3_loader    = DataLoader(TensorDataset(X_train, y_train_idx),
    #                             batch_size=64, shuffle=True)
    # train_with_scheduler(cnn3_model, criterion, cnn3_optim, cnn3_scheduler,
    #                      cnn3_loader, epochs=100, early_stop_patience=10)
    # evaluate(cnn3_model, criterion, X_test, y_test_idx)

    # ── main_resnet — ResNet-18 ───────────────────────────────────────────────
    # Stem: Conv(3→64,k=3,s=1)+BN+ReLU → 4 stages × 2 ResBlocks
    # → MaxPool(4,4) → Flatten → Linear(512,10)
    # SGD lr=0.1, weight_decay=1e-4, momentum=0.9, batch=128, epochs=100
    # MultiStepLR milestones=[50,75] gamma=0.1 — EarlyStopping patience=15

    print("=" * 50)
    print("main_resnet  (ResNet-18, RandomCrop+HFlip augmentation)")
    print("=" * 50)
    resnet_model   = ResNet18(num_classes=10).to(device)
    resnet_optim   = torch.optim.SGD(resnet_model.parameters(), lr=0.1,
                                     weight_decay=1e-4, momentum=0.9)
    resnet_sched   = torch.optim.lr_scheduler.MultiStepLR(
                         resnet_optim, milestones=[50, 75], gamma=0.1)

    aug_transform  = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(0.5),
    ])
    resnet_dataset = AugmentedDataset(X_train_cpu, y_train_idx_cpu, aug_transform)
    resnet_loader  = DataLoader(resnet_dataset, batch_size=128, shuffle=True)

    best_loss, no_improve = float("inf"), 0
    for epoch in range(100):
        resnet_model.train()
        epoch_loss = 0.0
        for Xb, yb in resnet_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            resnet_optim.zero_grad()
            loss = criterion(resnet_model(Xb), yb)
            loss.backward()
            resnet_optim.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(resnet_loader)
        resnet_sched.step()
        print(f"Epoch {epoch+1:3d}  loss: {avg_loss:.4f}  lr: {resnet_optim.param_groups[0]['lr']:.2e}")
        if best_loss - avg_loss > 1e-4:
            best_loss, no_improve = avg_loss, 0
        else:
            no_improve += 1
        if no_improve >= 15:
            print(f"EarlyStopping at epoch {epoch+1}")
            break
    evaluate(resnet_model, criterion, X_test, y_test_idx)
