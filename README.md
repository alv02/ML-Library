# ML Library — C++/CUDA Deep Learning Framework

A deep learning framework built from scratch in C++/CUDA, featuring automatic differentiation, hand-written CUDA kernels, and multiple neural network architectures — from CNNs trained on CIFAR-10 to a GPT language model trained on text.

Built as a learning project to understand how ML frameworks work at the systems level — tensors, autograd graphs, GPU kernels, and training loops, all without PyTorch or any ML library underneath.

---

## Features

- **Tensor module** with two backends (CPU and CUDA): arbitrary strides, zero-copy broadcasting, CPU↔GPU transfers, `.npy` file I/O
- **CUDA kernels written by hand**: shared-memory tiled matmul, parallel tree reductions, unfold2d/fold2d (im2col), scatter-add, atomic operations, batched matmul (strided and pointer-array variants)
- **Autograd engine**: computational graph (DAG), 15+ differentiable ops — MatMul, Conv2d, MaxPool2d, BatchNorm (1d/2d), ReLU, GELU, Softmax, LayerNorm, Dropout, Gather, Scatter, Reshape, Transpose, Broadcast, CrossEntropy
- **Layer abstraction**: `Layer` base class with `forward()`, `parameters()`, `train()`/`eval()`; `Sequential` container; `ResBlock` with identity/projection shortcuts; `MultiHeadAttention`, `TransformerBlock`, `InputEmbedding`, `GPTModel`
- **Optimizers**: SGD (momentum + L2), AdamW; MultiStepLR and ReduceLROnPlateau schedulers; EarlyStopping
- **Models**: linear regression, fully-connected network, CNN with strided convolutions, CNN with max pooling, VGG-style CNN with BatchNorm, ResNet-18, GPT
- **Python bindings** via pybind11: `gpt_lib.GPTTrainer` exposes training, evaluation, autoregressive generation, and checkpoint save/load to Python/NumPy
- **Data augmentation**: CPU preprocessing pipeline (`Augmenter`) with `RandomCrop` and `RandomHorizontalFlip`, applied per batch before GPU upload
- **Utilities**: DataLoader with Fisher-Yates shuffle, accuracy metric, terminal visualization of predictions (ANSI color)

---

## Requirements

| Dependency      | Version                         |
| --------------- | ------------------------------- |
| CMake           | ≥ 3.18                          |
| CUDA Toolkit    | ≥ 11.0                          |
| C++ compiler    | C++17 (GCC / Clang)             |
| Python          | 3.x                             |
| pybind11        | auto-fetched by CMake           |
| tokenizers      | `pip install tokenizers` (char-level GPT) |
| tiktoken        | `pip install tiktoken` (BPE GPT, optional) |
| tensorflow/keras | dataset download only          |
| PyTorch         | baseline comparison only        |

---

## Quickstart

### Image classification (CIFAR-10)

```bash
# 1. Prepare dataset
python prepare_dataset.py

# 2. Build
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# 3. Run
./build/main_nn      # Dense network
./build/main_cnn     # CNN with strided convolutions
./build/main_cnn2    # CNN with max pooling
./build/main_cnn3    # VGG-B with BatchNorm
./build/main_resnet  # ResNet-18
```

### GPT language model

```bash
# Build Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target gpt_lib

# Train on any plain-text file (char-level tokenizer by default)
python3 train_gpt.py --steps 5000 --text data/shakespeare/input.txt

# Generate from a saved checkpoint (no text file needed)
python3 train_gpt.py --generate

# Use BPE tokenizer instead (requires pip install tiktoken)
python3 train_gpt.py --tokenizer bpe --text data/shakespeare/input.txt

# PyTorch baseline with identical architecture
python3 train_gpt_pytorch.py --steps 5000 --text data/shakespeare/input.txt
```

---

## Models

### `main_nn` — Fully-Connected Network (CIFAR-10)

```
3072 → Linear → ReLU → Linear(1024) → ReLU → Linear(512) → ReLU → Linear(256) → ReLU → Linear(10) → CrossEntropyLoss
```

| Hyperparameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Epochs | 50 |

---

### `main_cnn` — CNN with Strided Convolutions (CIFAR-10)

```
[N,3,32,32] → Conv(32,s=1) → Conv(64,s=2) → Conv(128,s=2) → Conv(256,s=2) → Flatten → Linear(512) → Linear(10)
```

| Hyperparameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Batch size | 64 |
| Epochs | 50 |

---

### `main_cnn2` — CNN with Max Pooling (CIFAR-10)

```
[N,3,32,32] → (Conv→ReLU→MaxPool)×3 → Flatten → Linear(512) → Linear(256) → Linear(10)
```

| Hyperparameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.005 |
| Batch size | 64 |
| Epochs | 60 |

---

### `main_cnn3` — VGG-13/BN-style CNN (CIFAR-10)

Four blocks of two `Conv→BN→ReLU` layers each, followed by MaxPool. Dense head: `2048→512→10`.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.05 (ReduceLROnPlateau ×0.1, p=5) |
| Batch size | 64 |
| Epochs | 100 (EarlyStopping patience=10) |

---

### `main_resnet` — ResNet-18 (CIFAR-10)

CIFAR-10-adapted ResNet-18: stem without MaxPool, global avg-pool replaced by MaxPool(4,4), with data augmentation.

```
Stem: Conv(3→64, k=3) → BN → ReLU
Stage 0: 2× ResBlock(64,  stride=1)
Stage 1: 2× ResBlock(128, stride=2)
Stage 2: 2× ResBlock(256, stride=2)
Stage 3: 2× ResBlock(512, stride=2)
MaxPool(4,4) → Flatten → Linear(512→10)
```

| Hyperparameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 0.1 → MultiStepLR ×0.1 at epochs 50, 75 |
| Batch size | 128 |
| Epochs | 100 (EarlyStopping patience=15) |
| Augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip(0.5) |

---

### GPT — Character-level Language Model

Standard decoder-only GPT trained on plain text. Pre-norm architecture with causal self-attention and GELU MLP.

```
tokens [B,T] → InputEmbedding (tok + pos) → Dropout
             → N × TransformerBlock:
                 LayerNorm → MultiHeadAttention (causal) → residual
                 LayerNorm → MLP (Linear→GELU→Linear)   → residual
             → LayerNorm → LM head [B,T,vocab]
             → CrossEntropyLoss (next-token prediction)
```

Default config (`train_gpt.py`):

| Hyperparameter | Value |
| --- | --- |
| d_model | 256 |
| n_heads | 4 |
| n_layers | 4 |
| max_seq_len | 256 |
| Dropout | 0.1 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Batch size | 8 |
| Weight init | Normal(0, 0.02) — GPT-2 standard |

Tokenizer: character-level via HuggingFace `tokenizers` (default, ~65 tokens for Shakespeare) or GPT-2 BPE via `tiktoken` (50257 tokens). The tokenizer is saved alongside model weights so generation requires no text file.

---

## Results on CIFAR-10

| Model | Executable | Test Accuracy |
| --- | --- | --- |
| Dense FC (3072→1024→512→256→10) | `main_nn` | 53.64% |
| CNN strided conv | `main_cnn` | 70.36% |
| CNN + MaxPool | `main_cnn2` | 76.53% |
| VGG-13/BN-style | `main_cnn3` | 89.35% |
| ResNet-18 + RandomCrop + HFlip | `main_resnet` | 92.80% |

> Results may vary slightly between runs due to random weight initialization and batch shuffling.

`pytorch_baseline.py` reimplements all CIFAR-10 models in PyTorch with identical architectures and hyperparameters to verify the custom framework reaches comparable accuracy.

---

## Project Structure

```
.
├── include/
│   ├── tensor.hpp           # Tensor struct, file I/O, device transfers
│   ├── autograd.hpp         # Graph, Var, differentiable function base
│   ├── ops.hpp              # Op declarations (forward + backward)
│   ├── layers.hpp           # Layer, Sequential, ResBlock, Conv2d, BatchNorm,
│   │                        # MultiHeadAttention, TransformerBlock, GPTModel, …
│   ├── models.hpp           # Model factory functions
│   ├── optimizers.hpp       # DataLoader, SGD, AdamW, schedulers, EarlyStopping
│   ├── augment.hpp          # Augmenter, RandomCrop, RandomHorizontalFlip
│   ├── metrics.hpp          # Accuracy
│   ├── visualize.hpp        # Terminal ANSI rendering
│   ├── tensor_iterator.hpp  # Strided multi-dim iterator
│   └── backend/
│       ├── tensor_cpu.hpp
│       └── tensor_cuda.hpp
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp
│   ├── ops.cpp
│   ├── layers.cpp           # Layer forward/backward implementations
│   ├── models.cpp           # GPTModel, CNN, ResNet factory implementations
│   ├── optimizers.cpp
│   ├── augment.cpp
│   ├── metrics.cpp
│   ├── visualize.cpp
│   ├── tensor_iterator.cpp
│   ├── python/
│   │   └── gpt_trainer.cpp  # pybind11 GPTTrainer bindings
│   └── backend/
│       ├── tensor_cpu.cpp   # CPU kernel implementations
│       └── tensor_cuda.cu   # CUDA kernel implementations
├── test/
│   ├── test_tensor.cpp      # CPU/GPU tensor op validation
│   ├── test_layer.cpp       # Layer forward/backward vs PyTorch references
│   └── test_reduce.cpp      # Reduction op tests
├── data/
│   └── test/
│       └── gen_layers.py    # Generate PyTorch reference data for test_layer
├── main_lr.cpp              # Linear regression
├── main_nn.cpp              # Dense network on CIFAR-10
├── main_cnn.cpp             # Strided CNN on CIFAR-10
├── main_cnn2.cpp            # MaxPool CNN on CIFAR-10
├── main_cnn3.cpp            # VGG-B with BatchNorm on CIFAR-10
├── main_resnet.cpp          # ResNet-18 on CIFAR-10
├── main_gpt.cpp             # GPT char-level (C++ only, no Python)
├── train_gpt.py             # GPT training + generation via Python bindings
├── train_gpt_pytorch.py     # GPT PyTorch baseline (same architecture)
├── prepare_dataset.py       # Download and preprocess CIFAR-10 → .npy
├── pytorch_baseline.py      # PyTorch reference for CIFAR-10 models
└── CMakeLists.txt
```

---

## Implementation Notes

**Convolutions** are implemented via unfold-matmul-fold (im2col): the input is unfolded into a matrix, multiplied by the weight matrix, then folded back. The backward pass uses scatter-add to handle overlapping windows.

**MaxPool backward** saves the argmax index for each pooling window during the forward pass, then routes gradients back to those positions using scatter-add.

**Broadcasting** works by setting stride=0 on dimensions of size 1. Kernels compute the physical memory offset from the logical index using these strides, so no data is copied. Layers call `broadcast_to` explicitly before ops — broadcasting is never implicit in the op layer.

**BatchNorm** normalizes each channel over all other dimensions during training, then applies learnable per-channel γ and β. Works for any input rank — `[N,C]` (dense) or `[N,C,H,W]` (conv). Forward uses biased variance; running statistics use unbiased variance via EMA.

**Batched matmul** for 4D tensors (e.g. `[B,H,T,d]` after attention transpose) handles non-contiguous strides by decomposing the batch index into per-dimension coordinates rather than using a single batch stride. Falls back to `cublasSgemmBatched` on CUDA when batch dimensions are non-contiguous.

**Transformer** uses pre-norm (LayerNorm before each sub-layer), causal attention mask applied as an additive `-1e9` bias, GELU with tanh approximation for the MLP, and GPT-2 weight initialization (`Normal(0, 0.02)`) for all linear and embedding weights.

**Numerical stability**: log-softmax uses the log-sum-exp trick; cross-entropy is fused with softmax to avoid `log(0)`.
