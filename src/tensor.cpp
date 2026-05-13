#include "../include/tensor.hpp"
#include "../include/backend/tensor_cpu.hpp"
#include "../include/backend/tensor_cuda.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// ---- File I/O ------------------------------------------------------------

template <typename T>
Tensor<T> tensor_load(const char *filename, b32 on_gpu, CudaMemArena *arena) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return Tensor<T>{};
    }

    u8 magic[6];
    u8 version[2];
    fread(magic, sizeof(char), 6, file);
    fread(version, sizeof(char), 2, file);

    if (version[0] != 1) {
        printf("Unsopported NPY version\n");
        fclose(file);
        return Tensor<T>{};
    }

    u16 header_len;
    fread(&header_len, sizeof(u16), 1, file);

    char header[header_len + 1];
    fread(header, sizeof(char), header_len, file);
    header[header_len] = '\0';

    char *fortran_ptr = strstr(header, "fortran_order");
    if (fortran_ptr) {
        char *true_ptr = strstr(fortran_ptr, "True");
        char *false_ptr = strstr(fortran_ptr, "False");
        if (true_ptr && (!false_ptr || true_ptr < false_ptr)) {
            printf("WARNING: %s has fortran_order=True, loading will be "
                   "incorrect\n",
                   filename);
        }
    }

    char *shape_ptr = strstr(header, "shape");
    char *end_ptr;
    shape_ptr = strchr(shape_ptr, '(');

    u32 ndim = 0;
    while (*shape_ptr != ')') {
        if (strtoul(shape_ptr, &end_ptr, 10)) {
            ndim++;
            shape_ptr = end_ptr;
        } else {
            shape_ptr++;
        }
    }

    u32 shape[ndim];
    shape_ptr = strstr(header, "shape");
    shape_ptr = strchr(shape_ptr, '(') + 1;
    for (u32 i = 0; i < ndim; i++) {
        shape[i] = (u32)strtoul(shape_ptr, &end_ptr, 10);
        shape_ptr = end_ptr;
        while (*shape_ptr == ',')
            shape_ptr++;
    }

    Tensor<T> tensor = Tensor<T>::make(ndim, shape, false);
    fread(tensor->data(), sizeof(T), tensor->numel(), file);
    fclose(file);

    if (!on_gpu)
        return tensor;

    return tensor_to_gpu(tensor, arena);
}

template Tensor<f32> tensor_load(const char *, b32, CudaMemArena *);
template Tensor<u32> tensor_load(const char *, b32, CudaMemArena *);

// ---- Copy / contiguous ---------------------------------------------------

template <typename T> b32 tensor_copy(Tensor<T> &dst, const Tensor<T> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_copy: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_copy(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_copy(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_copy: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
static void tensor_contiguous_impl(TensorImpl<T> &t,
                                   CudaMemArena *arena = nullptr) {
    if (tensor_is_contiguous(t))
        return;
    if (t.on_gpu()) {
        tensor_cuda_contiguous(t, arena);
    } else {
        tensor_cpu_contigous(t);
    }
}

template <typename T>
void tensor_contiguous(Tensor<T> &t, CudaMemArena *arena) {
    tensor_contiguous_impl(t.impl(), arena);
}

// ---- Metadata / shape helpers (device-independent) -----------------------

u64 tensor_compute_strides(u64 *stride, const u32 *shape, u32 ndim) {
    u64 size = 1;
    for (u32 i = ndim; i-- > 0;) {
        stride[i] = size;
        size *= shape[i];
    }
    return size;
}

template <typename T>
b32 tensor_reshape(TensorImpl<T> &tensor, const u32 *shape, u32 ndim,
                   CudaMemArena *arena) {
    u64 new_size = 1;
    for (u32 i = 0; i < ndim; i++)
        new_size *= shape[i];
    if (tensor.numel() != new_size)
        return false;

    tensor_contiguous_impl(tensor, arena);

    for (u32 i = 0; i < ndim; i++)
        tensor.shape[i] = shape[i];
    tensor.ndim = ndim;
    tensor_compute_strides(tensor.stride, tensor.shape, ndim);
    return true;
}

template <typename T>
b32 tensor_reshape(Tensor<T> &t, const u32 *shape, u32 ndim,
                   CudaMemArena *arena) {
    return tensor_reshape(t.impl(), shape, ndim, arena);
}

template <typename T>
b32 tensor_flatten(Tensor<T> &t, u32 start_dim, u32 end_dim,
                   CudaMemArena *arena) {

    if (start_dim >= t->ndim || end_dim >= t->ndim || start_dim > end_dim)
        return false;

    u32 new_shape[MAX_NDIM];
    u32 new_ndim = 0;

    // dims before flattened region
    for (u32 i = 0; i < start_dim; i++)
        new_shape[new_ndim++] = t->shape[i];

    // flattened dimension
    u32 flat = 1;
    for (u32 i = start_dim; i <= end_dim; i++)
        flat *= t->shape[i];

    new_shape[new_ndim++] = flat;

    // dims after flattened region
    for (u32 i = end_dim + 1; i < t->ndim; i++)
        new_shape[new_ndim++] = t->shape[i];

    return tensor_reshape(t.impl(), new_shape, new_ndim, arena);
}

template <typename T> void tensor_print(const TensorImpl<T> &tensor) {
    printf("Tensor(shape=[");
    for (u32 i = 0; i < tensor.ndim; i++) {
        printf("%d", tensor.shape[i]);
        if (i < tensor.ndim - 1)
            printf(", ");
    }
    printf("], stride=[");
    for (u32 i = 0; i < tensor.ndim; i++) {
        printf("%ld", tensor.stride[i]);
        if (i < tensor.ndim - 1)
            printf(", ");
    }
    printf("])\n");
}

// ---- fill / arange -------------------------------------------------------

template <typename T> void tensor_fill(Tensor<T> &t, T value) {
    if (t->on_gpu())
        tensor_cuda_fill(t.impl(), value);
    else
        tensor_cpu_fill(t.impl(), value);
}

template <typename T> void tensor_arange(Tensor<T> &t) {
    if (t->on_gpu())
        tensor_cuda_arange(t.impl());
    else
        tensor_cpu_arange(t.impl());
}

// ---- Dispatch helpers ----------------------------------------------------

template <typename T>
static b32 check_reduction_shape(const TensorImpl<T> &out,
                                 const TensorImpl<T> &src, u32 dim,
                                 const char *op) {
    if (out.ndim != src.ndim) {
        printf("%s: out ndim (%u) != src ndim (%u)\n", op, out.ndim, src.ndim);
        return false;
    }
    for (u32 i = 0; i < src.ndim; i++) {
        u32 expected = (i == dim) ? 1 : src.shape[i];
        if (out.shape[i] != expected) {
            printf("%s: out->shape[%u]=%u, expected %u\n", op, i, out.shape[i],
                   expected);
            return false;
        }
    }
    return true;
}

// Variant for argmax: out is u32, src is T
template <typename T>
static b32 check_argmax_shape(const TensorImpl<u32> &out,
                              const TensorImpl<T> &src, u32 dim,
                              const char *op) {
    if (out.ndim != src.ndim) {
        printf("%s: out ndim (%u) != src ndim (%u)\n", op, out.ndim, src.ndim);
        return false;
    }
    for (u32 i = 0; i < src.ndim; i++) {
        u32 expected = (i == dim) ? 1 : src.shape[i];
        if (out.shape[i] != expected) {
            printf("%s: out->shape[%u]=%u, expected %u\n", op, i, out.shape[i],
                   expected);
            return false;
        }
    }
    return true;
}

template <typename T>
static b32 check_broadcast(const TensorImpl<T> &out, const TensorImpl<T> &a,
                           const TensorImpl<T> &b, const char *op) {
    u32 bcast[MAX_NDIM];
    u32 bndim = broadcast_shape(a, b, bcast);
    if (bndim == 0) {
        printf("%s: shapes not broadcastable\n", op);
        return false;
    }
    if (bndim != out.ndim) {
        printf("%s: out ndim mismatch\n", op);
        return false;
    }
    for (u32 i = 0; i < bndim; i++) {
        if (bcast[i] != out.shape[i]) {
            printf("%s: out shape mismatch at dim %u\n", op, i);
            return false;
        }
    }
    return true;
}

// ---- activations (relu, exp) — f32 only ---------------------------------

b32 tensor_relu(Tensor<f32> &dst, const Tensor<f32> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_relu: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_relu(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_relu(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_relu: tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_relu(const Tensor<f32> &src, CudaMemArena *arena) {
    Tensor<f32> dst = tensor_create_like(src, arena);
    if (!tensor_relu(dst, src))
        return Tensor<f32>{};
    return dst;
}

b32 tensor_gelu(Tensor<f32> &dst, const Tensor<f32> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_relu: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_gelu(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_gelu(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_relu: tensors must be on the same device\n");
        return false;
    }
}
Tensor<f32> tensor_gelu(const Tensor<f32> &src, CudaMemArena *arena) {
    Tensor<f32> dst = tensor_create_like(src, arena);
    if (!tensor_gelu(dst, src))
        return Tensor<f32>{};
    return dst;
}

b32 tensor_gelu_backward(Tensor<f32> &out, const Tensor<f32> &grad,
                         const Tensor<f32> &input) {
    if (!tensor_shape_eq(out.impl(), grad.impl())) {
        printf("tensor_gelu_backward: shape mismatch\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | input->on_gpu()) {
    case 0b00:
        tensor_cpu_gelu_backward(out.impl(), grad.impl(), input.impl());
        return true;
    case 0b11:
        tensor_cuda_gelu_backward(out.impl(), grad.impl(), input.impl());
        return true;
    default:
        printf("tensor_gelu_backward: tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_gelu_backward(const Tensor<f32> &grad,
                                 const Tensor<f32> &input,
                                 CudaMemArena *arena) {
    Tensor<f32> out = tensor_create_like(grad, arena);
    if (!tensor_gelu_backward(out, grad, input))
        return Tensor<f32>{};
    return out;
}

void tensor_dropout_mask(Tensor<f32> &mask, f32 p) {
    if (mask->on_gpu())
        tensor_cuda_dropout_mask(mask.impl(), p);
    else
        tensor_cpu_dropout_mask(mask.impl(), p);
}

b32 tensor_exp(Tensor<f32> &dst, const Tensor<f32> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_exp: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_exp(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_exp(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_exp: tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_exp(const Tensor<f32> &src, CudaMemArena *arena) {
    Tensor<f32> dst = tensor_create_like(src, arena);
    if (!tensor_exp(dst, src))
        return Tensor<f32>{};
    return dst;
}

b32 tensor_log(Tensor<f32> &dst, const Tensor<f32> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_log: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_log(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_log(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_log: tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_log(const Tensor<f32> &src, CudaMemArena *arena) {
    Tensor<f32> dst = tensor_create_like(src, arena);
    if (!tensor_log(dst, src))
        return Tensor<f32>{};
    return dst;
}

b32 tensor_sqrt(Tensor<f32> &dst, const Tensor<f32> &src) {
    if (!tensor_shape_eq(dst.impl(), src.impl())) {
        printf("tensor_sqrt: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_sqrt(dst.impl(), src.impl());
        return true;
    case 0b11:
        tensor_cuda_sqrt(dst.impl(), src.impl());
        return true;
    default:
        printf("tensor_sqrt: tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_sqrt(const Tensor<f32> &src, CudaMemArena *arena) {
    Tensor<f32> dst = tensor_create_like(src, arena);
    if (!tensor_sqrt(dst, src))
        return Tensor<f32>{};
    return dst;
}

// ---- add -----------------------------------------------------------------

template <typename T>
b32 tensor_add(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b) {
    if (!check_broadcast(out.impl(), a.impl(), b.impl(), "tensor_add"))
        return false;
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_add(out.impl(), a.impl(), b.impl());
        return true;
    case 0b111:
        tensor_cuda_add(out.impl(), a.impl(), b.impl());
        return true;
    default:
        printf("tensor_add: all tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_add(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_add: shapes not broadcastable\n");
        return Tensor<T>{};
    }
    Tensor<T> out = Tensor<T>::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_add(out, a, b))
        return Tensor<T>{};
    return out;
}

// ---- sub -----------------------------------------------------------------

template <typename T>
b32 tensor_sub(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b) {
    if (!check_broadcast(out.impl(), a.impl(), b.impl(), "tensor_sub"))
        return false;
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_sub(out.impl(), a.impl(), b.impl());
        return true;
    case 0b111:
        tensor_cuda_sub(out.impl(), a.impl(), b.impl());
        return true;
    default:
        printf("tensor_sub: all tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_sub(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_sub: shapes not broadcastable\n");
        return Tensor<T>{};
    }
    Tensor<T> out = Tensor<T>::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_sub(out, a, b))
        return Tensor<T>{};
    return out;
}

// ---- mul (elementwise) ---------------------------------------------------

template <typename T>
b32 tensor_mul(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b) {
    if (!check_broadcast(out.impl(), a.impl(), b.impl(), "tensor_mul"))
        return false;
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_mul(out.impl(), a.impl(), b.impl());
        return true;
    case 0b111:
        tensor_cuda_mul(out.impl(), a.impl(), b.impl());
        return true;
    default:
        printf("tensor_mul: all tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_mul(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_mul: shapes not broadcastable\n");
        return Tensor<T>{};
    }
    Tensor<T> out = Tensor<T>::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_mul(out, a, b))
        return Tensor<T>{};
    return out;
}

// ---- div (elementwise) ---------------------------------------------------

template <typename T>
b32 tensor_div(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b) {
    if (!check_broadcast(out.impl(), a.impl(), b.impl(), "tensor_div"))
        return false;
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_div(out.impl(), a.impl(), b.impl());
        return true;
    case 0b111:
        tensor_cuda_div(out.impl(), a.impl(), b.impl());
        return true;
    default:
        printf("tensor_div: all tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_div(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_div: shapes not broadcastable\n");
        return Tensor<T>{};
    }
    Tensor<T> out = Tensor<T>::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_div(out, a, b))
        return Tensor<T>{};
    return out;
}

// ---- equal (elementwise) -------------------------------------------------

template <typename T>
b32 tensor_equal(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b) {
    if (!check_broadcast(out.impl(), a.impl(), b.impl(), "tensor_equal"))
        return false;
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_equal(out.impl(), a.impl(), b.impl());
        return true;
    case 0b111:
        tensor_cuda_equal(out.impl(), a.impl(), b.impl());
        return true;
    default:
        printf("tensor_equal: all tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_equal(const Tensor<T> &a, const Tensor<T> &b,
                       CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_equal: shapes not broadcastable\n");
        return Tensor<T>{};
    }
    Tensor<T> out = Tensor<T>::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_equal(out, a, b))
        return Tensor<T>{};
    return out;
}

// ---- relu_grad (elementwise) — f32 only ---------------------------------

b32 tensor_relu_backward(Tensor<f32> &out, const Tensor<f32> &grad,
                         const Tensor<f32> &in) {
    if (!tensor_shape_eq(out.impl(), in.impl()) ||
        !tensor_shape_eq(grad.impl(), in.impl())) {
        printf("tensor_relu_backward: shape mismatch\n");
        return false;
    }
    switch ((out->on_gpu() << 2) | (grad->on_gpu() << 1) | in->on_gpu()) {
    case 0b000:
        tensor_cpu_relu_backward(out.impl(), grad.impl(), in.impl());
        return true;
    case 0b111:
        tensor_cuda_relu_backward(out.impl(), grad.impl(), in.impl());
        return true;
    default:
        printf(
            "tensor_relu_backward: all tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_relu_backward(const Tensor<f32> &grad, const Tensor<f32> &in,
                                 CudaMemArena *arena) {
    Tensor<f32> out = tensor_create_like(in, arena);
    if (!tensor_relu_backward(out, grad, in))
        return Tensor<f32>{};
    return out;
}

// ---- add (scalar) --------------------------------------------------------

template <typename T>
b32 tensor_add(Tensor<T> &out, const Tensor<T> &a, T scalar) {
    if (!tensor_shape_eq(out.impl(), a.impl())) {
        printf("tensor_add: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | a->on_gpu()) {
    case 0b00:
        tensor_cpu_add(out.impl(), a.impl(), scalar);
        return true;
    case 0b11:
        tensor_cuda_add(out.impl(), a.impl(), scalar);
        return true;
    default:
        printf("tensor_add: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_add(const Tensor<T> &a, T scalar, CudaMemArena *arena) {
    Tensor<T> out = tensor_create_like(a, arena);
    if (!tensor_add(out, a, scalar))
        return Tensor<T>{};
    return out;
}

// ---- sub (scalar) --------------------------------------------------------

template <typename T>
b32 tensor_sub(Tensor<T> &out, const Tensor<T> &a, T scalar) {
    if (!tensor_shape_eq(out.impl(), a.impl())) {
        printf("tensor_sub: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | a->on_gpu()) {
    case 0b00:
        tensor_cpu_sub(out.impl(), a.impl(), scalar);
        return true;
    case 0b11:
        tensor_cuda_sub(out.impl(), a.impl(), scalar);
        return true;
    default:
        printf("tensor_sub: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_sub(const Tensor<T> &a, T scalar, CudaMemArena *arena) {
    Tensor<T> out = tensor_create_like(a, arena);
    if (!tensor_sub(out, a, scalar))
        return Tensor<T>{};
    return out;
}

// ---- mul (scalar) --------------------------------------------------------

template <typename T>
b32 tensor_mul(Tensor<T> &out, const Tensor<T> &a, T scalar) {
    if (!tensor_shape_eq(out.impl(), a.impl())) {
        printf("tensor_mul: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | a->on_gpu()) {
    case 0b00:
        tensor_cpu_mul(out.impl(), a.impl(), scalar);
        return true;
    case 0b11:
        tensor_cuda_mul(out.impl(), a.impl(), scalar);
        return true;
    default:
        printf("tensor_mul: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_mul(const Tensor<T> &a, T scalar, CudaMemArena *arena) {
    Tensor<T> out = tensor_create_like(a, arena);
    if (!tensor_mul(out, a, scalar))
        return Tensor<T>{};
    return out;
}

// ---- div (scalar) --------------------------------------------------------

template <typename T>
b32 tensor_div(Tensor<T> &out, const Tensor<T> &a, T scalar) {
    if (!tensor_shape_eq(out.impl(), a.impl())) {
        printf("tensor_div: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | a->on_gpu()) {
    case 0b00:
        tensor_cpu_div(out.impl(), a.impl(), scalar);
        return true;
    case 0b11:
        tensor_cuda_div(out.impl(), a.impl(), scalar);
        return true;
    default:
        printf("tensor_div: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_div(const Tensor<T> &a, T scalar, CudaMemArena *arena) {
    Tensor<T> out = tensor_create_like(a, arena);
    if (!tensor_div(out, a, scalar))
        return Tensor<T>{};
    return out;
}

// ---- mat_mul — f32 only --------------------------------------------------

b32 tensor_mat_mul(Tensor<f32> &out, const Tensor<f32> &a, const Tensor<f32> &b,
                   b32 clear_out) {
    u32 nd = a->ndim;
    if (nd < 2 || nd != b->ndim || nd != out->ndim) {
        printf("tensor_mat_mul: ndim mismatch\n");
        return false;
    }
    if (a->shape[nd - 1] != b->shape[nd - 2] ||
        a->shape[nd - 2] != out->shape[nd - 2] ||
        b->shape[nd - 1] != out->shape[nd - 1]) {
        printf("tensor_mat_mul: shape mismatch\n");
        return false;
    }
    bool batched = (nd > 2);
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        if (batched)
            tensor_cpu_mat_mul_batched(out.impl(), a.impl(), b.impl(),
                                       clear_out);
        else
            tensor_cpu_mat_mul(out.impl(), a.impl(), b.impl(), clear_out);
        return true;
    case 0b111:
        if (batched)
            tensor_cuda_mat_mul_batched(out.impl(), a.impl(), b.impl(),
                                        clear_out);
        else
            tensor_cuda_mat_mul_cublas(out.impl(), a.impl(), b.impl(),
                                       clear_out);
        return true;
    default:
        printf("tensor_mat_mul: all tensors must be on the same device\n");
        return false;
    }
}

Tensor<f32> tensor_mat_mul(const Tensor<f32> &a, const Tensor<f32> &b,
                           CudaMemArena *arena) {
    u32 nd = a->ndim;
    u32 out_shape[MAX_NDIM];
    // Copy leading (batch) dims + M from A, then N from B.
    memcpy(out_shape, a->shape, (nd - 1) * sizeof(u32));
    out_shape[nd - 1] = b->shape[nd - 1];
    Tensor<f32> out = tensor_zeros<f32>(nd, out_shape, a->on_gpu(), arena);
    if (!tensor_mat_mul(out, a, b, false))
        return Tensor<f32>{};
    return out;
}

// ---- sum -----------------------------------------------------------------

template <typename T>
b32 tensor_sum(Tensor<T> &out, const Tensor<T> &t, b32 clear_out) {
    if (out->numel() != 1) {
        printf("tensor_sum: out must be a scalar tensor (size=1)\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | t->on_gpu()) {
    case 0b00:
        tensor_cpu_sum(out.impl(), t.impl(), clear_out);
        return true;
    case 0b11:
        tensor_cuda_sum(out.impl(), t.impl());
        return true;
    default:
        printf("tensor_sum: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
b32 tensor_sum(Tensor<T> &out, const Tensor<T> &t, u32 dim, b32 keep_dim,
               b32 clear_out) {
    if (dim >= t->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return false;
    }
    if (!check_reduction_shape(out.impl(), t.impl(), dim, "tensor_sum"))
        return false;
    switch ((out->on_gpu() << 1) | t->on_gpu()) {
    case 0b00:
        tensor_cpu_sum(out.impl(), t.impl(), dim, clear_out);
        break;
    case 0b11:
        tensor_cuda_sum(out.impl(), t.impl(), dim);
        break;
    default:
        printf("tensor_sum: tensors must be on the same device\n");
        return false;
    }
    if (!keep_dim) {
        for (u32 i = dim; i < out->ndim - 1; i++) {
            out->shape[i] = out->shape[i + 1];
            out->stride[i] = out->stride[i + 1];
        }
        out->ndim--;
    }
    return true;
}

template <typename T>
Tensor<T> tensor_sum(const Tensor<T> &t, CudaMemArena *arena) {
    u32 shape[1] = {1};
    Tensor<T> out = Tensor<T>::make(1, shape, t->on_gpu(), arena);
    if (!tensor_sum(out, t))
        return Tensor<T>{};
    return out;
}

template <typename T>
Tensor<T> tensor_sum(const Tensor<T> &t, u32 dim, b32 keep_dim,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor<T> out = Tensor<T>::make(t->ndim, out_shape, t->on_gpu(), arena);
    if (!tensor_sum(out, t, dim, keep_dim))
        return Tensor<T>{};
    return out;
}

// ---- max -----------------------------------------------------------------

template <typename T>
static b32 tensor_max_global(Tensor<T> &out, const Tensor<T> &t) {
    if (out->numel() != 1) {
        printf("tensor_max: out must be a scalar tensor (size=1)\n");
        return false;
    }
    switch ((out->on_gpu() << 1) | t->on_gpu()) {
    case 0b00:
        tensor_cpu_max(out.impl(), t.impl());
        return true;
    case 0b11:
        tensor_cuda_max(out.impl(), t.impl());
        return true;
    default:
        printf("tensor_max: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
b32 tensor_max(Tensor<T> &out, const Tensor<T> &t, u32 dim, b32 keep_dim) {
    if (dim >= t->ndim) {
        printf("tensor_max: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return false;
    }
    if (!check_reduction_shape(out.impl(), t.impl(), dim, "tensor_max"))
        return false;
    switch ((out->on_gpu() << 1) | t->on_gpu()) {
    case 0b00:
        tensor_cpu_max(out.impl(), t.impl(), dim);
        break;
    case 0b11:
        tensor_cuda_max(out.impl(), t.impl(), dim);
        break;
    default:
        printf("tensor_max: tensors must be on the same device\n");
        return false;
    }
    if (!keep_dim) {
        for (u32 i = dim; i < out->ndim - 1; i++) {
            out->shape[i] = out->shape[i + 1];
            out->stride[i] = out->stride[i + 1];
        }
        out->ndim--;
    }
    return true;
}

template <typename T>
Tensor<T> tensor_max(const Tensor<T> &t, u32 dim, b32 keep_dim,
                     CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor<T> out = Tensor<T>::make(t->ndim, out_shape, t->on_gpu(), arena);
    tensor_max(out, t, dim, keep_dim);
    return out;
}

// ---- argmax --------------------------------------------------------------

template <typename T>
b32 tensor_argmax(TensorU32 &out, const Tensor<T> &t, u32 dim, b32 keep_dim) {
    if (dim >= t->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return false;
    }
    if (!check_argmax_shape(out.impl(), t.impl(), dim, "tensor_argmax"))
        return false;
    switch ((out->on_gpu() << 1) | t->on_gpu()) {
    case 0b00:
        tensor_cpu_argmax(out.impl(), t.impl(), dim);
        break;
    case 0b11:
        tensor_cuda_argmax(out.impl(), t.impl(), dim);
        break;
    default:
        printf("tensor_argmax: tensors must be on the same device\n");
        return false;
    }
    if (!keep_dim) {
        for (u32 i = dim; i < out->ndim - 1; i++) {
            out->shape[i] = out->shape[i + 1];
            out->stride[i] = out->stride[i + 1];
        }
        out->ndim--;
    }
    return true;
}

template <typename T>
TensorU32 tensor_argmax(const Tensor<T> &t, u32 dim, b32 keep_dim,
                        CudaMemArena *arena) {
    if (dim >= t->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return TensorU32{};
    }
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    TensorU32 out = TensorU32::make(t->ndim, out_shape, t->on_gpu(), arena);
    if (!tensor_argmax(out, t, dim, keep_dim))
        return TensorU32{};
    return out;
}

// ---- welford mean+m2 — f32 only -----------------------------------------

b32 tensor_welford_mean_var(Tensor<f32> &mean, Tensor<f32> &m2,
                            const Tensor<f32> &src, u32 dim) {
    if (dim >= src->ndim) {
        printf("tensor_welford_mean_var: dim %u out of range (ndim=%u)\n", dim,
               src->ndim);
        return false;
    }
    if (mean->numel() != src->shape[dim] || m2->numel() != src->shape[dim]) {
        printf("tensor_welford_mean_var: mean and m2 must have size=%u "
               "(shape[dim])\n",
               src->shape[dim]);
        return false;
    }
    switch ((mean->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_welford_mean_var(mean.impl(), m2.impl(), src.impl(), dim);
        return true;
    case 0b11:
        tensor_cuda_welford_mean_var(mean.impl(), m2.impl(), src.impl(), dim);
        return true;
    default:
        printf("tensor_welford_mean_var: tensors must be on the same device\n");
        return false;
    }
}

// ---- fused batch norm — f32 only ----------------------------------------

void tensor_bn_fwd_normalize(Tensor<f32> &out, Tensor<f32> &xhat,
                             const Tensor<f32> &inp, const Tensor<f32> &mean,
                             const Tensor<f32> &m2, const Tensor<f32> &gamma,
                             const Tensor<f32> &beta, f32 count, f32 eps) {
    if (inp->on_gpu())
        tensor_cuda_bn_fwd_normalize(out.impl(), xhat.impl(), inp.impl(),
                                     mean.impl(), m2.impl(), gamma.impl(),
                                     beta.impl(), count, eps);
    else
        tensor_cpu_bn_fwd_normalize(out.impl(), xhat.impl(), inp.impl(),
                                    mean.impl(), m2.impl(), gamma.impl(),
                                    beta.impl(), count, eps);
}

void tensor_bn_bwd(Tensor<f32> &dx, Tensor<f32> &d_gamma, Tensor<f32> &d_beta,
                   const Tensor<f32> &grad, const Tensor<f32> &xhat,
                   const Tensor<f32> &gamma, const Tensor<f32> &var, f32 m,
                   f32 eps) {
    if (grad->on_gpu())
        tensor_cuda_bn_bwd(dx.impl(), d_gamma.impl(), d_beta.impl(),
                           grad.impl(), xhat.impl(), gamma.impl(), var.impl(),
                           m, eps);
    else
        tensor_cpu_bn_bwd(dx.impl(), d_gamma.impl(), d_beta.impl(), grad.impl(),
                          xhat.impl(), gamma.impl(), var.impl(), m, eps);
}

// ---- softmax — f32 only -------------------------------------------------

b32 tensor_softmax(Tensor<f32> &out, const Tensor<f32> &in, i32 dim,
                   CudaMemArena *arena) {
    if (!tensor_shape_eq(out.impl(), in.impl())) {
        printf("tensor_softmax: shape mismatch\n");
        return false;
    }
    if (out->on_gpu() != in->on_gpu()) {
        printf("tensor_softmax: tensors must be on the same device\n");
        return false;
    }

    u32 axis = (dim < 0) ? (u32)((i32)in->ndim + dim) : (u32)dim;
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, in->shape, in->ndim * sizeof(u32));
    row_shape[axis] = 1;

    Tensor<f32> row_max =
        Tensor<f32>::make(in->ndim, row_shape, in->on_gpu(), arena);
    Tensor<f32> row_sum =
        Tensor<f32>::make(in->ndim, row_shape, in->on_gpu(), arena);

    tensor_max(row_max, in, axis, true);
    tensor_sub(out, in, row_max);
    tensor_exp(out, out);
    tensor_sum(row_sum, out, axis, true);
    tensor_div(out, out, row_sum);

    return true;
}

Tensor<f32> tensor_softmax(const Tensor<f32> &in, i32 dim,
                           CudaMemArena *arena) {
    Tensor<f32> out = tensor_create_like(in, arena);
    if (!tensor_softmax(out, in, dim, arena))
        return Tensor<f32>{};
    return out;
}

// ---- log_softmax — f32 only ---------------------------------------------

b32 tensor_log_softmax(Tensor<f32> &out, const Tensor<f32> &in, i32 dim,
                       CudaMemArena *arena) {
    if (!tensor_shape_eq(out.impl(), in.impl())) {
        printf("tensor_log_softmax: shape mismatch\n");
        return false;
    }
    if (out->on_gpu() != in->on_gpu()) {
        printf("tensor_log_softmax: tensors must be on the same device\n");
        return false;
    }

    u32 axis = (dim < 0) ? (u32)((i32)in->ndim + dim) : (u32)dim;
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, in->shape, in->ndim * sizeof(u32));
    row_shape[axis] = 1;

    Tensor<f32> row_max =
        Tensor<f32>::make(in->ndim, row_shape, in->on_gpu(), arena);
    Tensor<f32> row_lse =
        Tensor<f32>::make(in->ndim, row_shape, in->on_gpu(), arena);

    tensor_max(row_max, in, axis, true);
    tensor_sub(out, in, row_max);
    tensor_exp(out, out);
    tensor_sum(row_lse, out, axis, true);
    tensor_log(row_lse, row_lse);
    tensor_sub(out, in, row_max);
    tensor_sub(out, out, row_lse);

    return true;
}

Tensor<f32> tensor_log_softmax(const Tensor<f32> &in, i32 dim,
                               CudaMemArena *arena) {
    Tensor<f32> out = tensor_create_like(in, arena);
    if (!tensor_log_softmax(out, in, dim, arena))
        return Tensor<f32>{};
    return out;
}

// ---- scattering ----------------------------------------------------------

template <typename T>
b32 tensor_scatter_add(Tensor<T> &out, const Tensor<T> &src,
                       const TensorU32 &indices, u32 dim) {

    switch (out->on_gpu() << 2 | src->on_gpu() << 1 | indices->on_gpu()) {
    case 0b000:
        tensor_cpu_scatter_add(out.impl(), src.impl(), indices.impl(), dim);
        return true;
    case 0b111:
        tensor_cuda_scatter_add(out.impl(), src.impl(), indices.impl(), dim);
        return true;
    default:
        printf("tensor_scatter_add: tensors must be on the same device\n");
        return false;
    }
}

// ---- indexing ------------------------------------------------------------

template <typename T>
b32 gather(Tensor<T> &dst, const Tensor<T> &src, const TensorU32 &indices,
           u32 dim) {
    if (dim >= src->ndim) {
        printf("gather: dim out of range\n");
        return false;
    }

    if (src->ndim != indices->ndim || src->ndim != dst->ndim) {
        printf("gather: ndim mismatch\n");
        return false;
    }

    for (u32 i = 0; i < src->ndim; i++) {

        // dst and indices MUST always match
        if (dst->shape[i] != indices->shape[i]) {
            printf("gather: dst and indices shape mismatch\n");
            return false;
        }

        // src must match dst except at gather dim
        if (i != dim) {
            if (src->shape[i] != dst->shape[i]) {
                printf("gather: src/dst shape mismatch\n");
                return false;
            }
        }
    }

    switch (dst->on_gpu() << 2 | src->on_gpu() << 1 | indices->on_gpu()) {
    case 0b000:
        tensor_cpu_gather(dst.impl(), src.impl(), indices.impl(), dim);
        return true;
    case 0b111:
        tensor_cuda_gather(dst.impl(), src.impl(), indices.impl(), dim);
        return true;
    default:
        printf("gather: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> gather(const Tensor<T> &src, const TensorU32 &indices, u32 dim,
                 CudaMemArena *arena) {

    Tensor<T> dst =
        Tensor<T>::make(indices->ndim, indices->shape, src->on_gpu(), arena);

    if (!gather(dst, src, indices, dim))
        return Tensor<T>{};
    return dst;
}

// ---- initializing — f32 only --------------------------------------------

void tensor_he_init(Tensor<f32> &t) {
    if (t->on_gpu()) {
        tensor_cuda_he_init(t.impl());
    } else {
        tensor_cpu_he_init(t.impl());
    }
}

// ---- Conv2dParams constructor -------------------------------------------

Unfold2dParams::Unfold2dParams(u32 k, u32 stride, u32 pad, u32 dil,
                               f32 pad_constant)
    : k_h(k), k_w(k), stride_h(stride), stride_w(stride), pad_h(pad),
      pad_w(pad), pad_constant(pad_constant) {}

// ---- spatial / patch operations ------------------------------------------

template <typename T>
b32 tensor_unfold2d(Tensor<T> &out, const Tensor<T> &input,
                    Unfold2dParams params) {
    if (input->ndim != 4) {
        printf("tensor_unfold2d: input must be 4-dimensional [N,C,H,W], got "
               "%u dims\n",
               input->ndim);
        return false;
    }

    u32 N = input->shape[0];
    u32 C = input->shape[1];
    u32 H = input->shape[2];
    u32 W = input->shape[3];
    params.compute_output_size(H, W);
    u32 expected_size =
        N * params.L_h * params.L_w * C * params.k_h * params.k_w;

    if (out->numel() != expected_size) {
        printf("tensor_unfold2d: out has wrong size (got %llu, expected %u)\n",
               (unsigned long long)out->numel(), expected_size);
        return false;
    }

    switch (out->on_gpu() << 1 | input->on_gpu()) {
    case 0b00:
        tensor_cpu_unfold2d(out.impl(), input.impl(), params);
        return true;
    case 0b11:
        tensor_cuda_unfold2d(out.impl(), input.impl(), params);
        return true;
    default:
        printf("tensor_unfold2d: tensors must be on the same device\n");
        return false;
    }
}

template <typename T>
Tensor<T> tensor_unfold2d(const Tensor<T> &input, Unfold2dParams params,
                          CudaMemArena *arena) {
    if (input->ndim != 4) {
        printf("tensor_unfold2d: input must be 4-dimensional [N,C,H,W], got "
               "%u dims\n",
               input->ndim);
        return Tensor<T>{};
    }
    u32 N = input->shape[0], C = input->shape[1];
    u32 H = input->shape[2], W = input->shape[3];
    params.compute_output_size(H, W);

    u32 shape[MAX_NDIM] = {N, params.L_h * params.L_w,
                           C * params.k_h * params.k_w};
    Tensor<T> out = Tensor<T>::make(3, shape, input->on_gpu(), arena);
    if (!tensor_unfold2d(out, input, params))
        return Tensor<T>{};
    return out;
}

template <typename T>
b32 tensor_fold2d(Tensor<T> &dst, const Tensor<T> &col, Unfold2dParams params) {
    if (dst->ndim != 4) {
        printf("tensor_fold2d: dst must be 4-dimensional [N,C,H,W]\n");
        return false;
    }
    u32 N = dst->shape[0];
    u32 C = dst->shape[1];
    u32 H = dst->shape[2];
    u32 W = dst->shape[3];
    params.compute_output_size(H, W);
    u32 expected_size =
        N * params.L_h * params.L_w * C * params.k_h * params.k_w;
    if (col->numel() != expected_size) {
        printf("tensor_fold2d: col has wrong size (got %llu, expected %u)\n",
               (unsigned long long)col->numel(), expected_size);
        return false;
    }
    switch (dst->on_gpu() << 1 | col->on_gpu()) {
    case 0b00:
        tensor_cpu_fold2d(dst.impl(), col.impl(), params);
        return true;
    case 0b11:
        tensor_cuda_fold2d(dst.impl(), col.impl(), params);
        return true;
    default:
        printf("tensor_fold2d: tensors must be on the same device\n");
        return false;
    }
}

// ---- comparison — f32 only ----------------------------------------------

b32 tensor_equals(const Tensor<f32> &a, const Tensor<f32> &b, f32 tol) {
    if (!tensor_shape_eq(a.impl(), b.impl()))
        return false;
    switch (a->on_gpu() << 1 | b->on_gpu()) {
    case 0b00:
        return tensor_cpu_equals(a.impl(), b.impl(), tol);
    case 0b11:
        return tensor_cuda_equals(a.impl(), b.impl(), tol);
    default:
        printf("tensor_equals: tensors must be on the same device\n");
        return false;
    }
}

// ---- Explicit instantiations --------------------------------------------

#define INST(T)                                                                \
    template b32 tensor_copy(Tensor<T> &, const Tensor<T> &);                  \
    template void tensor_contiguous(Tensor<T> &, CudaMemArena *);              \
    template b32 tensor_reshape(TensorImpl<T> &, const u32 *, u32,             \
                                CudaMemArena *);                               \
    template b32 tensor_reshape(Tensor<T> &, const u32 *, u32,                 \
                                CudaMemArena *);                               \
    template b32 tensor_flatten(Tensor<T> &t, u32 start_dim, u32 end_dim,      \
                                CudaMemArena *);                               \
    template void tensor_print(const TensorImpl<T> &);                         \
    template void tensor_fill(Tensor<T> &, T);                                 \
    template void tensor_arange(Tensor<T> &);                                  \
    template b32 tensor_add(Tensor<T> &, const Tensor<T> &,                    \
                            const Tensor<T> &);                                \
    template Tensor<T> tensor_add(const Tensor<T> &, const Tensor<T> &,        \
                                  CudaMemArena *);                             \
    template b32 tensor_sub(Tensor<T> &, const Tensor<T> &,                    \
                            const Tensor<T> &);                                \
    template Tensor<T> tensor_sub(const Tensor<T> &, const Tensor<T> &,        \
                                  CudaMemArena *);                             \
    template b32 tensor_mul(Tensor<T> &, const Tensor<T> &,                    \
                            const Tensor<T> &);                                \
    template Tensor<T> tensor_mul(const Tensor<T> &, const Tensor<T> &,        \
                                  CudaMemArena *);                             \
    template b32 tensor_div(Tensor<T> &, const Tensor<T> &,                    \
                            const Tensor<T> &);                                \
    template Tensor<T> tensor_div(const Tensor<T> &, const Tensor<T> &,        \
                                  CudaMemArena *);                             \
    template b32 tensor_equal(Tensor<T> &, const Tensor<T> &,                  \
                              const Tensor<T> &);                              \
    template Tensor<T> tensor_equal(const Tensor<T> &, const Tensor<T> &,      \
                                    CudaMemArena *);                           \
    template b32 tensor_add(Tensor<T> &, const Tensor<T> &, T);                \
    template Tensor<T> tensor_add(const Tensor<T> &, T, CudaMemArena *);       \
    template b32 tensor_sub(Tensor<T> &, const Tensor<T> &, T);                \
    template Tensor<T> tensor_sub(const Tensor<T> &, T, CudaMemArena *);       \
    template b32 tensor_mul(Tensor<T> &, const Tensor<T> &, T);                \
    template Tensor<T> tensor_mul(const Tensor<T> &, T, CudaMemArena *);       \
    template b32 tensor_div(Tensor<T> &, const Tensor<T> &, T);                \
    template Tensor<T> tensor_div(const Tensor<T> &, T, CudaMemArena *);       \
    template b32 tensor_sum(Tensor<T> &, const Tensor<T> &, b32);              \
    template b32 tensor_sum(Tensor<T> &, const Tensor<T> &, u32, b32, b32);    \
    template Tensor<T> tensor_sum(const Tensor<T> &, CudaMemArena *);          \
    template Tensor<T> tensor_sum(const Tensor<T> &, u32, b32,                 \
                                  CudaMemArena *);                             \
    template b32 tensor_max(Tensor<T> &, const Tensor<T> &, u32, b32);         \
    template Tensor<T> tensor_max(const Tensor<T> &, u32, b32,                 \
                                  CudaMemArena *);                             \
    template b32 tensor_argmax(TensorU32 &, const Tensor<T> &, u32, b32);      \
    template TensorU32 tensor_argmax(const Tensor<T> &, u32, b32,              \
                                     CudaMemArena *);                          \
    template b32 tensor_scatter_add(Tensor<T> &, const Tensor<T> &,            \
                                    const TensorU32 &, u32);                   \
    template b32 gather(Tensor<T> &, const Tensor<T> &, const TensorU32 &,     \
                        u32);                                                  \
    template Tensor<T> gather(const Tensor<T> &, const TensorU32 &, u32,       \
                              CudaMemArena *);                                 \
    template b32 tensor_unfold2d(Tensor<T> &, const Tensor<T> &,               \
                                 Unfold2dParams);                              \
    template Tensor<T> tensor_unfold2d(const Tensor<T> &, Unfold2dParams,      \
                                       CudaMemArena *);                        \
    template b32 tensor_fold2d(Tensor<T> &, const Tensor<T> &, Unfold2dParams);

INST(f32)
INST(u32)
