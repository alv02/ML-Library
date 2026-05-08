"""
Profile one training step of CNN3 to show where PyTorch spends its time.
Run with:  python3 profile_baseline.py
"""
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ── Model (same as pytorch_baseline.py CNN3) ──────────────────────────────────

def conv_bn(in_c, out_c):
    return [nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU()]

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

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Setup ─────────────────────────────────────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CNN3().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05,
                             weight_decay=5e-4, momentum=0.9)

X = torch.randn(64, 3, 32, 32, device=device)   # one batch
y = torch.randint(0, 10, (64,), device=device)

# Warmup (so cuDNN can pick algorithms before we start timing)
for _ in range(3):
    optimizer.zero_grad()
    criterion(model(X), y).backward()
    optimizer.step()

torch.cuda.synchronize()

# ── Profile ───────────────────────────────────────────────────────────────────

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    for _ in range(5):
        with record_function("forward"):
            loss = criterion(model(X), y)
        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()

torch.cuda.synchronize()

# ── Report ────────────────────────────────────────────────────────────────────

print("\n── Top 20 ops by CUDA time ──────────────────────────────────────────")
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20,
    max_name_column_width=40,
))

print("\n── Top 20 ops by CPU time ───────────────────────────────────────────")
print(prof.key_averages().table(
    sort_by="cpu_time_total", row_limit=20,
    max_name_column_width=40,
))

# ── Per-category summary ──────────────────────────────────────────────────────

totals = {}
for e in prof.key_averages():
    cat = (
        "conv_forward"  if "conv2d" in e.key and "backward" not in e.key else
        "conv_backward" if "convolution_backward" in e.key else
        "batch_norm"    if "batch_norm" in e.key else
        "linear/gemm"   if e.key in ("aten::mm", "aten::addmm", "aten::linear") else
        "other"
    )
    totals[cat] = totals.get(cat, 0) + e.cuda_time_total

total_cuda = sum(totals.values()) or 1
print("\n── CUDA time by category (5 steps) ─────────────────────────────────")
for cat, us in sorted(totals.items(), key=lambda x: -x[1]):
    print(f"  {cat:<20} {us/1000:8.2f} ms  ({us/total_cuda*100:5.1f}%)")
