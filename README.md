# ML Library — C++/CUDA Deep Learning Framework

A deep learning framework built from scratch in C++/CUDA, featuring automatic differentiation, hand-written CUDA kernels, and multiple neural network architectures trained on CIFAR-10.

Built as a learning project to understand how ML frameworks work at the systems level — tensors, autograd graphs, GPU kernels, and training loops, all without PyTorch or any ML library underneath.

---

## Features

- **Tensor module** with two backends (CPU and CUDA): arbitrary strides, zero-copy broadcasting, CPU↔GPU transfers, `.npy` file I/O
- **CUDA kernels written by hand**: shared-memory tiled matmul, parallel tree reductions, unfold2d/fold2d (im2col), scatter-add, atomic operations
- **Autograd engine**: computational graph (DAG), 8+ differentiable ops — MatMul, Conv2d, MaxPool2d, ReLU, softmax, cross-entropy
- **SGD optimizer** with momentum and L2 regularization
- **Models**: linear regression, fully-connected network, CNN with strided convolutions, CNN with max pooling
- **Utilities**: DataLoader with Fisher-Yates shuffle, accuracy metric, terminal visualization of predictions (ANSI color)

---

## Requirements

| Dependency         | Version                          |
| ------------------ | -------------------------------- |
| CMake              | ≥ 3.18                           |
| CUDA Toolkit       | ≥ 11.0                           |
| C++ compiler       | C++17 (GCC / Clang)              |
| Python             | 3.x (dataset preparation only)   |
| tensorflow / keras | dataset download only            |
| PyTorch            | PyTorch baseline comparison only |

---

## Quickstart

### 1. Prepare the dataset

Download and preprocess CIFAR-10 into `.npy` files:

```bash
python prepare_dataset.py
```

This downloads CIFAR-10 via Keras, normalizes pixel values to `[0, 1]`, converts to `(N, C, H, W)` format, one-hot encodes labels, and saves four files:

```
data/X_train.npy   (50000, 3, 32, 32)  float32
data/y_train.npy   (50000, 10)          float32  one-hot
data/X_test.npy    (10000, 3, 32, 32)  float32
data/y_test.npy    (10000, 10)          float32  one-hot
```

### 2. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

Binaries are placed in `build/`. All executables must be run from the **project root** (not from `build/`) so they can find the `data/` directory.

### 3. Run a model

```bash
# From the project root:
./build/main_nn      # Dense network on CIFAR-10
./build/main_cnn     # CNN with strided convolutions
./build/main_cnn2    # CNN with max pooling
```

Each executable prints per-epoch loss and final test loss + accuracy. The CNN executables also render a terminal visualization of correct and wrong predictions using ANSI color codes.

---

## Models

### `main_nn` — Fully-Connected Network (CIFAR-10)

Input flattened from `[N, 3, 32, 32]` to `[N, 3072]`.

```
3072 → Linear → ReLU
     → Linear → ReLU   (1024)
     → Linear → ReLU   (512)
     → Linear → ReLU   (256)
     → Linear           (10)
     → CrossEntropyLoss
```

| Hyperparameter | Value          |
| -------------- | -------------- |
| Optimizer      | SGD            |
| Learning rate  | 0.01           |
| Momentum       | 0.9            |
| Weight decay   | 1e-4           |
| Batch size     | 128            |
| Epochs         | 50             |
| Weight init    | Kaiming normal |

---

### `main_cnn` — CNN with Strided Convolutions (CIFAR-10)

Spatial downsampling via strided convolutions instead of pooling.

```
Input: [N, 3, 32, 32]

Conv(3→32,  k=3, s=1, p=1) → ReLU    [N, 32,  32, 32]
Conv(32→64, k=3, s=2, p=1) → ReLU    [N, 64,  16, 16]
Conv(64→128,k=3, s=2, p=1) → ReLU    [N, 128,  8,  8]
Conv(128→256,k=3,s=2, p=1) → ReLU    [N, 256,  4,  4]
Flatten                               [N, 4096]
Linear(4096→512) → ReLU
Linear(512→10)
CrossEntropyLoss
```

| Hyperparameter | Value          |
| -------------- | -------------- |
| Optimizer      | SGD            |
| Learning rate  | 0.01           |
| Momentum       | 0.9            |
| Weight decay   | 1e-4           |
| Batch size     | 64             |
| Epochs         | 50             |
| Weight init    | Kaiming normal |

---

### `main_cnn2` — CNN with Max Pooling (CIFAR-10)

Spatial downsampling via MaxPool2d(2, 2) after each conv block.

```
Input: [N, 3, 32, 32]

Conv(3→32,  k=3, p=1) → ReLU → MaxPool(2,2)   [N, 32,  16, 16]
Conv(32→64, k=3, p=1) → ReLU → MaxPool(2,2)   [N, 64,   8,  8]
Conv(64→128,k=3, p=1) → ReLU → MaxPool(2,2)   [N, 128,  4,  4]
Flatten                                         [N, 2048]
Linear(2048→512) → ReLU
Linear(512→256)  → ReLU
Linear(256→10)
CrossEntropyLoss
```

| Hyperparameter | Value          |
| -------------- | -------------- |
| Optimizer      | SGD            |
| Learning rate  | 0.005          |
| Momentum       | 0.9            |
| Weight decay   | 5e-4           |
| Batch size     | 64             |
| Epochs         | 60             |
| Weight init    | Kaiming normal |

---

## Results on CIFAR-10

| Model                           | Executable  | Test Accuracy |
| ------------------------------- | ----------- | ------------- |
| Dense FC (3072→1024→512→256→10) | `main_nn`   | 53.64%        |
| CNN strided conv                | `main_cnn`  | 70.36%        |
| CNN + MaxPool                   | `main_cnn2` | 76.53%        |

> Results may vary slightly between runs due to random weight initialization and batch shuffling.

### PyTorch baseline

`pytorch_baseline.py` reimplements all three models in PyTorch with identical architectures and hyperparameters, to verify that the custom framework reaches comparable accuracy:

```bash
python pytorch_baseline.py
```

---

## Project Structure

```
.
├── include/
│   ├── tensor.hpp          # Tensor struct, file I/O, device transfers
│   ├── autograd.hpp        # Graph, function_var, differentiable function base
│   ├── ops.hpp             # Operation declarations (forward + backward)
│   ├── models.hpp          # linear_model, nn_model, cnn_model
│   ├── optimizers.hpp      # DataLoader, SGD
│   ├── metrics.hpp         # Accuracy
│   ├── visualize.hpp       # Terminal ANSI rendering
│   ├── tensor_iterator.hpp # Strided multi-dim iterator
│   └── backend/
│       ├── tensor_cpu.hpp
│       └── tensor_cuda.hpp
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp
│   ├── ops.cpp
│   ├── models.cpp
│   ├── optimizers.cpp
│   ├── metrics.cpp
│   ├── visualize.cpp
│   ├── tensor_iterator.cpp
│   └── backend/
│       ├── tensor_cpu.cpp   # CPU kernel implementations
│       └── tensor_cuda.cu   # CUDA kernel implementations
├── test/
│   ├── test_tensor.cpp      # CPU/GPU op validation against NumPy references
│   └── broadcast.cpp        # Broadcasting correctness tests
├── main_lr.cpp              # Linear regression example
├── main_nn.cpp              # Dense network on CIFAR-10
├── main_cnn.cpp             # Strided CNN on CIFAR-10
├── main_cnn2.cpp            # MaxPool CNN on CIFAR-10
├── prepare_dataset.py       # Download and preprocess CIFAR-10 → .npy
├── pytorch_baseline.py      # PyTorch reference implementation
└── CMakeLists.txt
```

---

## Implementation Notes

**Convolutions** are implemented via unfold-matmul-fold (im2col): the input is unfolded into a matrix, multiplied by the weight matrix, then folded back. This converts spatial convolution into a single batched matrix multiply. The backward pass uses the same approach with scatter-add to handle overlapping windows.

**MaxPool backward** saves the argmax index for each pooling window during the forward pass, then routes gradients back to those positions using scatter-add.

**Broadcasting** works by setting stride=0 on dimensions of size 1. Both CPU and CUDA kernels compute the physical memory offset from the logical index using these strides, so no data is copied.

**Numerical stability**: log-softmax uses the log-sum-exp trick; cross-entropy is fused with softmax to avoid `log(0)`.
