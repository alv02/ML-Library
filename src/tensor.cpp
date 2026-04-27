#include "../include/tensor.hpp"
#include "../include/backend/tensor_cpu.hpp"
#include "../include/backend/tensor_cuda.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// ---- Storage constructor / destructor ------------------------------------

Storage::Storage(u64 nbytes, b32 on_gpu, CudaMemArena *arena)
    : nbytes(nbytes), on_gpu(on_gpu), arena(arena) {
    if (on_gpu) {
        if (arena) {
            data = (f32 *)cuda_arena_push(arena, nbytes);
        } else {
            cudaMallocAsync(&data, nbytes, 0);
            if (nbytes > 0)
                cudaMemsetAsync(data, 0, nbytes, 0);
        }
    } else {
        data = (f32 *)malloc(nbytes);
        if (data && nbytes > 0)
            memset(data, 0, nbytes);
    }
}

Storage::~Storage() {
    if (on_gpu) {
        if (!arena)
            cudaFreeAsync(data, 0);
        // arena manages its own memory lifetime
    } else {
        free(data);
    }
}

// ---- TensorImpl constructors ---------------------------------------------

TensorImpl::TensorImpl(u32 ndim, const u32 *shape, b32 on_gpu,
                       CudaMemArena *arena) {
    this->ndim = ndim;
    memcpy(this->shape, shape, ndim * sizeof(u32));
    u64 n = tensor_compute_strides(this->stride, this->shape, ndim);
    storage = std::make_shared<Storage>(n * sizeof(f32), on_gpu, arena);
}

TensorImpl::TensorImpl(u32 ndim, const u32 *shape, const u64 *stride,
                       b32 on_gpu, CudaMemArena *arena) {
    this->ndim = ndim;
    memcpy(this->shape, shape, ndim * sizeof(u32));
    memcpy(this->stride, stride, ndim * sizeof(u64));
    u64 n = 1;
    for (u32 i = 0; i < ndim; i++)
        n *= shape[i];
    storage = std::make_shared<Storage>(n * sizeof(f32), on_gpu, arena);
}

// ---- File I/O ------------------------------------------------------------

Tensor tensor_load(const char *filename, b32 on_gpu, CudaMemArena *arena) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return Tensor{};
    }

    u8 magic[6];
    u8 version[2];
    fread(magic, sizeof(char), 6, file);
    fread(version, sizeof(char), 2, file);

    if (version[0] != 1) {
        printf("Unsopported NPY version\n");
        fclose(file);
        return Tensor{};
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

    Tensor tensor = Tensor::make(ndim, shape, false);
    fread(tensor->data(), sizeof(f32), tensor->numel(), file);
    fclose(file);

    if (!on_gpu)
        return tensor;

    return tensor_to_gpu(tensor, arena);
}

b32 tensor_copy(Tensor &dst, const Tensor &src) {
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

Tensor tensor_to_gpu(const Tensor &t, CudaMemArena *arena) {
    Tensor dst = Tensor::make(t->ndim, t->shape, t->stride, true, arena);
    cudaMemcpyKind kind =
        t->on_gpu() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(dst->data(), t->data(), t->numel() * sizeof(f32), kind);
    return dst;
}

Tensor tensor_to_cpu(const Tensor &t, CudaMemArena *arena) {
    Tensor dst = Tensor::make(t->ndim, t->shape, t->stride, false, arena);
    cudaMemcpyKind kind =
        t->on_gpu() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    cudaMemcpy(dst->data(), t->data(), t->numel() * sizeof(f32), kind);
    return dst;
}

static void tensor_contiguous_impl(TensorImpl &t) {
    if (tensor_is_contiguous(t))
        return;
    if (t.on_gpu()) {
        tensor_cuda_contiguous(t);
    } else {
        tensor_cpu_contigous(t);
    }
}

void tensor_contiguous(Tensor &t) { tensor_contiguous_impl(t.impl()); }

// Creates a Tensor that shares src's storage but has independent metadata,
// so reshape/transpose on the view won't affect the original.
// NOTE: Tensor::make allocates a new storage internally, which is then
// immediately replaced with src's storage. The new storage is freed on
// replacement. This is wasteful but correct; views are not on hot paths.
Tensor tensor_view(const Tensor &src, CudaMemArena *arena) {
    Tensor v =
        Tensor::make(src->ndim, src->shape, src->stride, src->on_gpu(), arena);
    v->storage = src->storage;
    v->storage_offset = src->storage_offset;
    return v;
}

Tensor tensor_create_like(const Tensor &src, CudaMemArena *arena) {
    return Tensor::make(src->ndim, src->shape, src->on_gpu(), arena);
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

b32 tensor_is_contiguous(const TensorImpl &t) {
    u64 expected = 1;
    for (u32 i = t.ndim; i-- > 0;) {
        if (t.stride[i] != expected)
            return false;
        expected *= t.shape[i];
    }
    return true;
}

b32 tensor_is_contiguous(const Tensor &t) {
    return tensor_is_contiguous(t.impl());
}

b32 tensor_shape_eq(const TensorImpl &a, const TensorImpl &b) {
    return a.ndim == b.ndim &&
           memcmp(a.shape, b.shape, a.ndim * sizeof(u32)) == 0;
}

b32 tensor_shape_eq(const Tensor &a, const Tensor &b) {
    return tensor_shape_eq(a.impl(), b.impl());
}

b32 tensor_transpose(TensorImpl &tensor, u32 dim0, u32 dim1) {
    if (dim0 == dim1 || dim0 >= tensor.ndim || dim1 >= tensor.ndim)
        return false;
    swap(tensor.shape[dim0], tensor.shape[dim1]);
    swap(tensor.stride[dim0], tensor.stride[dim1]);
    return true;
}

b32 tensor_transpose(Tensor &t, u32 dim0, u32 dim1) {
    return tensor_transpose(t.impl(), dim0, dim1);
}

b32 tensor_reshape(TensorImpl &tensor, const u32 *shape, u32 ndim) {
    u64 new_size = 1;
    for (u32 i = 0; i < ndim; i++)
        new_size *= shape[i];
    if (tensor.numel() != new_size)
        return false;

    tensor_contiguous_impl(tensor);

    for (u32 i = 0; i < ndim; i++)
        tensor.shape[i] = shape[i];
    tensor.ndim = ndim;
    tensor_compute_strides(tensor.stride, tensor.shape, ndim);
    return true;
}

b32 tensor_reshape(Tensor &t, const u32 *shape, u32 ndim) {
    return tensor_reshape(t.impl(), shape, ndim);
}

void tensor_print(const TensorImpl &tensor) {
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

u32 broadcast_shape(const TensorImpl &a, const TensorImpl &b, u32 *out_shape) {
    u32 expanded_ndim = max(a.ndim, b.ndim);
    for (u32 i = 0; i < expanded_ndim; i++) {
        i32 ai = (i32)i - (i32)(expanded_ndim - a.ndim);
        i32 bi = (i32)i - (i32)(expanded_ndim - b.ndim);
        u32 da = (ai >= 0) ? a.shape[ai] : 1;
        u32 db = (bi >= 0) ? b.shape[bi] : 1;
        if (da != db && min(da, db) != 1)
            return 0;
        out_shape[i] = max(da, db);
    }
    return expanded_ndim;
}

void expanded_shape(const TensorImpl &t, u32 expanded_ndim,
                    u32 *t_expanded_shape) {
    u32 n_prepend = expanded_ndim - t.ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        t_expanded_shape[i] = 1;
    for (; i < expanded_ndim; i++)
        t_expanded_shape[i] = t.shape[i - n_prepend];
}

b32 tensor_expand_shape(TensorImpl &t, u32 expanded_ndim) {
    if (t.ndim > expanded_ndim)
        return false;
    if (t.ndim == expanded_ndim)
        return true;
    u32 new_shape[MAX_NDIM];
    u64 new_stride[MAX_NDIM];
    expanded_shape(t, expanded_ndim, new_shape);
    expanded_stride(t, expanded_ndim, new_stride);
    for (u32 i = 0; i < expanded_ndim; i++) {
        t.shape[i] = new_shape[i];
        t.stride[i] = new_stride[i];
    }
    t.ndim = expanded_ndim;
    return true;
}

void expanded_stride(const TensorImpl &t, u32 expanded_ndim,
                     u64 *t_expanded_stride) {
    u32 n_prepend = expanded_ndim - t.ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        t_expanded_stride[i] = 0;
    for (; i < expanded_ndim; i++) {
        u32 ti = i - n_prepend;
        t_expanded_stride[i] = (t.shape[ti] != 1) ? t.stride[ti] : 0;
    }
}

// ---- Dispatch helpers ----------------------------------------------------

static b32 check_reduction_shape(const TensorImpl &out, const TensorImpl &src,
                                 u32 dim, const char *op) {
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

static b32 check_scatter_shape(const TensorImpl &out, const TensorImpl &src,
                               const TensorImpl &indices, u32 dim,
                               const char *op) {
    if (out.ndim != src.ndim || src.ndim != indices.ndim) {
        printf("%s: ndim mismatch\n", op);
        return false;
    }
    if (src.shape[dim] != 1 || indices.shape[dim] != 1) {
        printf("%s: src/indices shape[%u] must be 1\n", op, dim);
        return false;
    }
    for (u32 i = 0; i < src.ndim; i++) {
        if (i == dim)
            continue;
        if (out.shape[i] != src.shape[i] || src.shape[i] != indices.shape[i]) {
            printf("%s: shape mismatch at dim %u\n", op, i);
            return false;
        }
    }
    return true;
}

static b32 check_broadcast(const TensorImpl &out, const TensorImpl &a,
                           const TensorImpl &b, const char *op) {
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

// ---- fill / clear --------------------------------------------------------

void tensor_fill(Tensor &t, f32 value) {
    if (t->on_gpu())
        tensor_cuda_fill(t.impl(), value);
    else
        tensor_cpu_fill(t.impl(), value);
}

void tensor_clear(Tensor &t) {
    if (t->on_gpu())
        tensor_cuda_clear(t.impl());
    else
        tensor_cpu_clear(t.impl());
}

// ---- activations (relu, exp) ---------------------------------------------

b32 tensor_relu(Tensor &dst, const Tensor &src) {
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

Tensor tensor_relu(const Tensor &src, CudaMemArena *arena) {
    Tensor dst = tensor_create_like(src, arena);
    if (!tensor_relu(dst, src))
        return Tensor{};
    return dst;
}

b32 tensor_exp(Tensor &dst, const Tensor &src) {
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

Tensor tensor_exp(const Tensor &src, CudaMemArena *arena) {
    Tensor dst = tensor_create_like(src, arena);
    if (!tensor_exp(dst, src))
        return Tensor{};
    return dst;
}

b32 tensor_log(Tensor &dst, const Tensor &src) {
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

Tensor tensor_log(const Tensor &src, CudaMemArena *arena) {
    Tensor dst = tensor_create_like(src, arena);
    if (!tensor_log(dst, src))
        return Tensor{};
    return dst;
}

b32 tensor_sqrt(Tensor &dst, const Tensor &src) {
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

Tensor tensor_sqrt(const Tensor &src, CudaMemArena *arena) {
    Tensor dst = tensor_create_like(src, arena);
    if (!tensor_sqrt(dst, src))
        return Tensor{};
    return dst;
}

// ---- add -----------------------------------------------------------------

b32 tensor_add(Tensor &out, const Tensor &a, const Tensor &b) {
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

Tensor tensor_add(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_add: shapes not broadcastable\n");
        return Tensor{};
    }
    Tensor out = Tensor::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_add(out, a, b))
        return Tensor{};
    return out;
}

// ---- sub -----------------------------------------------------------------

b32 tensor_sub(Tensor &out, const Tensor &a, const Tensor &b) {
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

Tensor tensor_sub(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_sub: shapes not broadcastable\n");
        return Tensor{};
    }
    Tensor out = Tensor::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_sub(out, a, b))
        return Tensor{};
    return out;
}

// ---- mul (elementwise) ---------------------------------------------------

b32 tensor_mul(Tensor &out, const Tensor &a, const Tensor &b) {
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

Tensor tensor_mul(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_mul: shapes not broadcastable\n");
        return Tensor{};
    }
    Tensor out = Tensor::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_mul(out, a, b))
        return Tensor{};
    return out;
}

// ---- div (elementwise) ---------------------------------------------------

b32 tensor_div(Tensor &out, const Tensor &a, const Tensor &b) {
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

Tensor tensor_div(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_div: shapes not broadcastable\n");
        return Tensor{};
    }
    Tensor out = Tensor::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_div(out, a, b))
        return Tensor{};
    return out;
}

// ---- equal (elementwise) -------------------------------------------------

b32 tensor_equal(Tensor &out, const Tensor &a, const Tensor &b) {
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

Tensor tensor_equal(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a.impl(), b.impl(), out_shape);
    if (out_ndim == 0) {
        printf("tensor_equal: shapes not broadcastable\n");
        return Tensor{};
    }
    Tensor out = Tensor::make(out_ndim, out_shape, a->on_gpu(), arena);
    if (!tensor_equal(out, a, b))
        return Tensor{};
    return out;
}

// ---- relu_grad (elementwise) ---------------------------------------------

b32 tensor_relu_backward(Tensor &out, const Tensor &grad, const Tensor &in) {
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

Tensor tensor_relu_backward(const Tensor &grad, const Tensor &in, CudaMemArena *arena) {
    Tensor out = tensor_create_like(in, arena);
    if (!tensor_relu_backward(out, grad, in))
        return Tensor{};
    return out;
}

// ---- add (scalar) --------------------------------------------------------

b32 tensor_add(Tensor &out, const Tensor &a, f32 scalar) {
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

Tensor tensor_add(const Tensor &a, f32 scalar, CudaMemArena *arena) {
    Tensor out = tensor_create_like(a, arena);
    if (!tensor_add(out, a, scalar))
        return Tensor{};
    return out;
}

// ---- sub (scalar) --------------------------------------------------------

b32 tensor_sub(Tensor &out, const Tensor &a, f32 scalar) {
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

Tensor tensor_sub(const Tensor &a, f32 scalar, CudaMemArena *arena) {
    Tensor out = tensor_create_like(a, arena);
    if (!tensor_sub(out, a, scalar))
        return Tensor{};
    return out;
}

// ---- mul (scalar) --------------------------------------------------------

b32 tensor_mul(Tensor &out, const Tensor &a, f32 scalar) {
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

Tensor tensor_mul(const Tensor &a, f32 scalar, CudaMemArena *arena) {
    Tensor out = tensor_create_like(a, arena);
    if (!tensor_mul(out, a, scalar))
        return Tensor{};
    return out;
}

// ---- div (scalar) --------------------------------------------------------

b32 tensor_div(Tensor &out, const Tensor &a, f32 scalar) {
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

Tensor tensor_div(const Tensor &a, f32 scalar, CudaMemArena *arena) {
    Tensor out = tensor_create_like(a, arena);
    if (!tensor_div(out, a, scalar))
        return Tensor{};
    return out;
}

// ---- mat_mul -------------------------------------------------------------

b32 tensor_mat_mul(Tensor &out, const Tensor &a, const Tensor &b,
                   b32 clear_out) {
    if (a->ndim != 2 || b->ndim != 2) {
        printf("tensor_mat_mul: only 2-D tensors supported\n");
        return false;
    }
    if (a->shape[COL_DIM(a.impl())] != b->shape[ROW_DIM(b.impl())] ||
        a->shape[ROW_DIM(a.impl())] != out->shape[ROW_DIM(out.impl())] ||
        b->shape[COL_DIM(b.impl())] != out->shape[COL_DIM(out.impl())]) {
        printf("tensor_mat_mul: shape mismatch\n");
        return false;
    }
    switch ((out->on_gpu() << 2) | (a->on_gpu() << 1) | b->on_gpu()) {
    case 0b000:
        tensor_cpu_mat_mul(out.impl(), a.impl(), b.impl(), clear_out);
        return true;
    case 0b111:
        tensor_cuda_mat_mul(out.impl(), a.impl(), b.impl(), clear_out);
        return true;
    default:
        printf("tensor_mat_mul: all tensors must be on the same device\n");
        return false;
    }
}

Tensor tensor_mat_mul(const Tensor &a, const Tensor &b, CudaMemArena *arena) {
    u32 shape[2] = {a->shape[ROW_DIM(a.impl())], b->shape[COL_DIM(b.impl())]};
    Tensor out = Tensor::make(2, shape, a->on_gpu(), arena);
    if (!tensor_mat_mul(out, a, b, false)) {
        printf("Shape of A: [%d, %d]\n", a->shape[ROW_DIM(a.impl())],
               a->shape[COL_DIM(a.impl())]);
        printf("Shape of B: [%d, %d]\n", b->shape[ROW_DIM(b.impl())],
               b->shape[COL_DIM(b.impl())]);
        return Tensor{};
    }
    return out;
}

// ---- sum -----------------------------------------------------------------

b32 tensor_sum(Tensor &out, const Tensor &t, b32 clear_out) {
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

b32 tensor_sum(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim,
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

Tensor tensor_sum(const Tensor &t, CudaMemArena *arena) {
    u32 shape[1] = {1};
    Tensor out = Tensor::make(1, shape, t->on_gpu(), arena);
    if (!tensor_sum(out, t))
        return Tensor{};
    return out;
}

Tensor tensor_sum(const Tensor &t, u32 dim, b32 keep_dim, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor out = Tensor::make(t->ndim, out_shape, t->on_gpu(), arena);
    if (!tensor_sum(out, t, dim, keep_dim))
        return Tensor{};
    return out;
}

// ---- max -----------------------------------------------------------------

b32 tensor_max(Tensor &out, const Tensor &t) {
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

Tensor tensor_max(const Tensor &t) {
    u32 shape[1] = {1};
    Tensor out = Tensor::make(1, shape, t->on_gpu());
    if (!tensor_max(out, t))
        return Tensor{};
    return out;
}

b32 tensor_max(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim) {
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

Tensor tensor_max(const Tensor &t, u32 dim, b32 keep_dim, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor out = Tensor::make(t->ndim, out_shape, t->on_gpu(), arena);
    tensor_max(out, t, dim, keep_dim);
    return out;
}

// ---- argmax --------------------------------------------------------------

b32 tensor_argmax(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim) {
    if (dim >= t->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return false;
    }
    if (!check_reduction_shape(out.impl(), t.impl(), dim, "tensor_argmax"))
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

Tensor tensor_argmax(const Tensor &t, u32 dim, b32 keep_dim, CudaMemArena *arena) {
    if (dim >= t->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim, t->ndim);
        return Tensor{};
    }
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, t->shape, t->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor out = Tensor::make(t->ndim, out_shape, t->on_gpu(), arena);
    if (!tensor_argmax(out, t, dim, keep_dim))
        return Tensor{};
    return out;
}

// ---- welford mean+var ----------------------------------------------------

b32 tensor_welford_mean_var(Tensor &mean, Tensor &var, const Tensor &src,
                            u32 dim) {
    if (dim >= src->ndim) {
        printf("tensor_welford_mean_var: dim %u out of range (ndim=%u)\n", dim,
               src->ndim);
        return false;
    }
    if (mean->numel() != src->shape[dim] || var->numel() != src->shape[dim]) {
        printf("tensor_welford_mean_var: mean and var must have size=%u "
               "(shape[dim])\n",
               src->shape[dim]);
        return false;
    }
    switch ((mean->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_welford_mean_var(mean.impl(), var.impl(), src.impl(), dim);
        return true;
    case 0b11:
        tensor_cuda_welford_mean_var(mean.impl(), var.impl(), src.impl(), dim);
        return true;
    default:
        printf("tensor_welford_mean_var: tensors must be on the same device\n");
        return false;
    }
}

// ---- softmax -------------------------------------------------------------

b32 tensor_softmax(Tensor &out, const Tensor &in, CudaMemArena *arena) {
    if (!tensor_shape_eq(out.impl(), in.impl())) {
        printf("tensor_softmax: shape mismatch\n");
        return false;
    }
    if (out->on_gpu() != in->on_gpu()) {
        printf("tensor_softmax: tensors must be on the same device\n");
        return false;
    }

    u32 col_dim = COL_DIM(in.impl());
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, in->shape, in->ndim * sizeof(u32));
    row_shape[col_dim] = 1;

    Tensor row_max = Tensor::make(in->ndim, row_shape, in->on_gpu(), arena);
    Tensor row_sum = Tensor::make(in->ndim, row_shape, in->on_gpu(), arena);

    tensor_max(row_max, in, col_dim, true);
    tensor_sub(out, in, row_max);
    tensor_exp(out, out);
    tensor_sum(row_sum, out, col_dim, true);
    tensor_div(out, out, row_sum);

    return true;
}

Tensor tensor_softmax(const Tensor &in, CudaMemArena *arena) {
    Tensor out = tensor_create_like(in, arena);
    if (!tensor_softmax(out, in, arena))
        return Tensor{};
    return out;
}

// ---- log_softmax ---------------------------------------------------------

b32 tensor_log_softmax(Tensor &out, const Tensor &in, CudaMemArena *arena) {
    if (!tensor_shape_eq(out.impl(), in.impl())) {
        printf("tensor_log_softmax: shape mismatch\n");
        return false;
    }
    if (out->on_gpu() != in->on_gpu()) {
        printf("tensor_log_softmax: tensors must be on the same device\n");
        return false;
    }

    u32 col_dim = COL_DIM(in.impl());
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, in->shape, in->ndim * sizeof(u32));
    row_shape[col_dim] = 1;

    Tensor row_max = Tensor::make(in->ndim, row_shape, in->on_gpu(), arena);
    Tensor row_lse = Tensor::make(in->ndim, row_shape, in->on_gpu(), arena);

    tensor_max(row_max, in, col_dim, true);
    tensor_sub(out, in, row_max);
    tensor_exp(out, out);
    tensor_sum(row_lse, out, col_dim, true);
    tensor_log(row_lse, row_lse);
    tensor_sub(out, in, row_max);
    tensor_sub(out, out, row_lse);

    return true;
}

Tensor tensor_log_softmax(const Tensor &in, CudaMemArena *arena) {
    Tensor out = tensor_create_like(in, arena);
    if (!tensor_log_softmax(out, in, arena))
        return Tensor{};
    return out;
}

// ---- scattering ----------------------------------------------------------

b32 tensor_scatter_add(Tensor &out, const Tensor &src, const Tensor &indices,
                       u32 dim) {
    if (!check_scatter_shape(out.impl(), src.impl(), indices.impl(), dim,
                             "tensor_scatter_add"))
        return false;
    switch ((out->on_gpu() << 2) | (src->on_gpu() << 1) | indices->on_gpu()) {
    case 0b000:
        tensor_cpu_scatter_add(out.impl(), src.impl(), indices.impl(), dim);
        return true;
    case 0b111:
        tensor_cuda_scatter_add(out.impl(), src.impl(), indices.impl(), dim);
        return true;
    default:
        printf("tensor_scatter_add: all tensors must be on the same device\n");
        return false;
    }
}

Tensor tensor_scatter_add(const Tensor &src, const Tensor &indices, u32 dim,
                          u32 dim_size, CudaMemArena *arena) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, src->shape, src->ndim * sizeof(u32));
    out_shape[dim] = dim_size;
    Tensor out = Tensor::make(src->ndim, out_shape, src->on_gpu(), arena);
    if (!tensor_scatter_add(out, src, indices, dim))
        return Tensor{};
    return out;
}

// ---- initializing --------------------------------------------------------

void tensor_he_init(Tensor &t) {
    if (t->on_gpu()) {
        tensor_cuda_he_init(t.impl());
    } else {
        tensor_cpu_he_init(t.impl());
    }
}

// ---- indexing ------------------------------------------------------------

static b32 check_index_select(const TensorImpl &dst, const TensorImpl &src,
                              u32 n_indices, u32 dim) {
    if (dim >= src.ndim) {
        printf("tensor_index_select: dim %u out of range (ndim=%u)\n", dim,
               src.ndim);
        return false;
    }
    if (dst.ndim != src.ndim) {
        printf("tensor_index_select: dst and src must have the same ndim\n");
        return false;
    }
    if (dst.shape[dim] != n_indices) {
        printf("tensor_index_select: dst->shape[%u]=%u does not match "
               "n_indices=%u\n",
               dim, dst.shape[dim], n_indices);
        return false;
    }
    for (u32 i = 0; i < src.ndim; i++) {
        if (i != dim && dst.shape[i] != src.shape[i]) {
            printf("tensor_index_select: shape mismatch at dim %u "
                   "(dst=%u, src=%u)\n",
                   i, dst.shape[i], src.shape[i]);
            return false;
        }
    }
    return true;
}

b32 tensor_index_select(Tensor &dst, const Tensor &src, const u32 *indices,
                        u32 n_indices, u32 dim) {
    if (!check_index_select(dst.impl(), src.impl(), n_indices, dim))
        return false;
    switch ((dst->on_gpu() << 1) | src->on_gpu()) {
    case 0b00:
        tensor_cpu_index_select(dst.impl(), src.impl(), indices, n_indices,
                                dim);
        return true;
    case 0b11:
        tensor_cuda_index_select(dst.impl(), src.impl(), indices, n_indices,
                                 dim);
        return true;
    default:
        printf("tensor_index_select: tensors must be on the same device\n");
        return false;
    }
}

Tensor tensor_index_select(const Tensor &src, const u32 *indices, u32 n_indices,
                           u32 dim, CudaMemArena *arena) {
    if (dim >= src->ndim) {
        printf("tensor_index_select: dim %u out of range (ndim=%u)\n", dim,
               src->ndim);
        return Tensor{};
    }
    u32 shape[MAX_NDIM];
    for (u32 i = 0; i < src->ndim; i++)
        shape[i] = (i == dim) ? n_indices : src->shape[i];
    Tensor dst = Tensor::make(src->ndim, shape, src->on_gpu(), arena);
    if (!tensor_index_select(dst, src, indices, n_indices, dim))
        return Tensor{};
    return dst;
}

// ---- Conv2dParams constructor -------------------------------------------

Unfold2dParams::Unfold2dParams(u32 k, u32 stride, u32 pad, u32 dil,
                               f32 pad_constant)
    : k_h(k), k_w(k), stride_h(stride), stride_w(stride), pad_h(pad),
      pad_w(pad), pad_constant(pad_constant) {}

// ---- spatial / patch operations ------------------------------------------

b32 tensor_unfold2d(Tensor &out, const Tensor &input, Unfold2dParams params) {
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

Tensor tensor_unfold2d(const Tensor &input, Unfold2dParams params,
                       CudaMemArena *arena) {
    if (input->ndim != 4) {
        printf("tensor_unfold2d: input must be 4-dimensional [N,C,H,W], got "
               "%u dims\n",
               input->ndim);
        return Tensor{};
    }
    u32 N = input->shape[0], C = input->shape[1];
    u32 H = input->shape[2], W = input->shape[3];
    params.compute_output_size(H, W);

    u32 shape[MAX_NDIM] = {N, params.L_h * params.L_w,
                           C * params.k_h * params.k_w};
    Tensor out = Tensor::make(3, shape, input->on_gpu(), arena);
    if (!tensor_unfold2d(out, input, params))
        return Tensor{};
    return out;
}

b32 tensor_fold2d(Tensor &dst, const Tensor &col, Unfold2dParams params) {
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

// ---- comparison ----------------------------------------------------------

b32 tensor_equals(const Tensor &a, const Tensor &b, f32 tol) {
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
