#include "../include/tensor.hpp"
#include "../include/backend/tensor_cpu.hpp"
#include "../include/backend/tensor_cuda.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>
using namespace std;

// ---- Tensor constructors / destructor ------------------------------------

Tensor::Tensor(u32 ndim, const u32 *shape, b32 on_gpu) {
    this->ndim = ndim;
    this->on_gpu = on_gpu;
    this->size = 1;
    this->owns_data = true;

    for (u32 i = 0; i < ndim; i++)
        this->shape[i] = shape[i];

    for (u32 i = ndim; i-- > 0;) {
        this->stride[i] = this->size;
        this->size *= this->shape[i];
    }

    if (on_gpu) {
        tensor_cuda_alloc(this);
    } else {
        this->data = (f32 *)malloc(this->size * sizeof(f32));
        memset(this->data, 0, this->size * sizeof(f32));
    }
}

Tensor::Tensor(const Tensor *src) {
    data = src->data;
    size = src->size;
    ndim = src->ndim;
    on_gpu = src->on_gpu;
    owns_data = false;

    memcpy(shape, src->shape, MAX_NDIM * sizeof(u32));
    memcpy(stride, src->stride, MAX_NDIM * sizeof(u64));
}

Tensor::~Tensor() {
    if (owns_data) {
        if (on_gpu) {
            tensor_cuda_free(this);
        } else {
            free(data);
        }
    }
}

// ---- File I/O ------------------------------------------------------------

Tensor *tensor_load(const char *filename, b32 on_gpu) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return nullptr;
    }

    u8 magic[6];
    u8 version[2];
    fread(magic, sizeof(char), 6, file);
    fread(version, sizeof(char), 2, file);

    if (version[0] != 1) {
        printf("Unsopported NPY version\n");
        fclose(file);
        return nullptr;
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

    Tensor *tensor = new Tensor(ndim, shape, false);
    fread(tensor->data, sizeof(f32), tensor->size, file);
    fclose(file);

    if (!on_gpu)
        return tensor;

    Tensor *gpu = tensor_cuda_to_gpu(tensor);
    delete tensor;
    return gpu;
}

Tensor *tensor_to_gpu(const Tensor *t) {
    if (t->on_gpu)
        return tensor_cuda_copy(t);
    return tensor_cuda_to_gpu(t);
}

Tensor *tensor_to_cpu(const Tensor *t) {
    if (!t->on_gpu) {
        Tensor *copy = new Tensor(t->ndim, t->shape, false);
        memcpy(copy->stride, t->stride, t->ndim * sizeof(u64));
        memcpy(copy->data, t->data, t->size * sizeof(f32));
        return copy;
    }
    return tensor_cuda_to_cpu(t);
}

// ---- Metadata / shape helpers (device-independent) -----------------------

b32 tensor_is_contiguous(const Tensor *t) {
    u64 expected = 1;
    for (u32 i = t->ndim; i-- > 0;) {
        if (t->stride[i] != expected)
            return false;
        expected *= t->shape[i];
    }
    return true;
}

b32 tensor_shape_eq(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim)
        return false;
    for (u32 i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            return false;
    return true;
}

b32 tensor_transpose(Tensor *tensor, u32 dim0, u32 dim1) {
    if (dim0 == dim1 || dim0 >= tensor->ndim || dim1 >= tensor->ndim)
        return false;

    swap(tensor->shape[dim0], tensor->shape[dim1]);
    swap(tensor->stride[dim0], tensor->stride[dim1]);
    return true;
}

b32 tensor_reshape(Tensor *tensor, const u32 *shape, u32 ndim) {
    u64 new_size = 1;
    for (u32 i = 0; i < ndim; i++)
        new_size *= shape[i];
    if (tensor->size != new_size)
        return false;

    for (u32 i = 0; i < ndim; i++)
        tensor->shape[i] = shape[i];

    new_size = 1;
    for (u32 i = ndim; i-- > 0;) {
        tensor->stride[i] = new_size;
        new_size *= shape[i];
    }
    tensor->ndim = ndim;
    return true;
}

Tensor *tensor_view(const Tensor *src) { return new Tensor(src); }

Tensor *tensor_create_like(const Tensor *src) {
    return new Tensor(src->ndim, src->shape, src->on_gpu);
}

void tensor_print(const Tensor *tensor) {
    printf("Tensor(shape=[");
    for (u32 i = 0; i < tensor->ndim; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->ndim - 1)
            printf(", ");
    }
    printf("], stride=[");
    for (u32 i = 0; i < tensor->ndim; i++) {
        printf("%ld", tensor->stride[i]);
        if (i < tensor->ndim - 1)
            printf(", ");
    }
    printf("])\n");
}

u32 broadcast_shape(const Tensor *a, const Tensor *b, u32 *out_shape) {
    u32 expanded_ndim = max(a->ndim, b->ndim);
    for (u32 i = 0; i < expanded_ndim; i++) {
        i32 ai = (i32)i - (i32)(expanded_ndim - a->ndim);
        i32 bi = (i32)i - (i32)(expanded_ndim - b->ndim);
        u32 da = (ai >= 0) ? a->shape[ai] : 1;
        u32 db = (bi >= 0) ? b->shape[bi] : 1;
        if (da != db && min(da, db) != 1)
            return 0;
        out_shape[i] = max(da, db);
    }
    return expanded_ndim;
}

void expanded_shape(const Tensor *t, const u32 *expanded_shape,
                    u32 expanded_ndim, u32 *t_expanded_shape) {
    u32 n_prepend = expanded_ndim - t->ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        t_expanded_shape[i] = 1;
    for (; i < expanded_ndim; i++)
        t_expanded_shape[i] = t->shape[i - n_prepend];
}

void expanded_stride(const Tensor *t, const u32 *expanded_shape,
                     u32 expanded_ndim, u64 *t_expanded_stride) {
    u32 n_prepend = expanded_ndim - t->ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        t_expanded_stride[i] = 0;
    for (; i < expanded_ndim; i++) {
        u32 ti = i - n_prepend;
        t_expanded_stride[i] = (t->shape[ti] != 1) ? t->stride[ti] : 0;
    }
}

// ---- Dispatch helpers ----------------------------------------------------

// Check that `out` is the correct keep_dim=true reduction output of `src`
// along `dim`: same ndim, same shape everywhere except out->shape[dim] == 1.
static b32 check_reduction_shape(const Tensor *out, const Tensor *src, u32 dim,
                                 const char *op) {
    if (out->ndim != src->ndim) {
        printf("%s: out ndim (%u) != src ndim (%u)\n", op, out->ndim,
               src->ndim);
        return false;
    }
    for (u32 i = 0; i < src->ndim; i++) {
        u32 expected = (i == dim) ? 1 : src->shape[i];
        if (out->shape[i] != expected) {
            printf("%s: out->shape[%u]=%u, expected %u\n", op, i, out->shape[i],
                   expected);
            return false;
        }
    }
    return true;
}

// Check that `out` has the broadcast shape of a and b.
static b32 check_broadcast(const Tensor *out, const Tensor *a, const Tensor *b,
                           const char *op) {
    u32 bcast[MAX_NDIM];
    u32 bndim = broadcast_shape(a, b, bcast);
    if (bndim == 0) {
        printf("%s: shapes not broadcastable\n", op);
        return false;
    }
    if (bndim != out->ndim) {
        printf("%s: out ndim mismatch\n", op);
        return false;
    }
    for (u32 i = 0; i < bndim; i++) {
        if (bcast[i] != out->shape[i]) {
            printf("%s: out shape mismatch at dim %u\n", op, i);
            return false;
        }
    }
    return true;
}

// ---- realloc -------------------------------------------------------------

void tensor_realloc(Tensor *t, const u32 *new_shape, u32 new_ndim) {
    if (t->ndim == new_ndim) {
        bool same = true;
        for (u32 i = 0; i < new_ndim; i++)
            if (t->shape[i] != new_shape[i]) {
                same = false;
                break;
            }
        if (same)
            return;
    }

    if (t->owns_data) {
        if (t->on_gpu)
            tensor_cuda_free(t);
        else
            free(t->data);
    }

    t->ndim = new_ndim;
    t->size = 1;
    for (u32 i = 0; i < new_ndim; i++)
        t->shape[i] = new_shape[i];
    for (u32 i = new_ndim; i-- > 0;) {
        t->stride[i] = t->size;
        t->size *= t->shape[i];
    }

    t->owns_data = true;
    if (t->on_gpu) {
        tensor_cuda_alloc(t);
    } else {
        t->data = (f32 *)malloc(t->size * sizeof(f32));
        memset(t->data, 0, t->size * sizeof(f32));
    }
}

// ---- copy ----------------------------------------------------------------

void tensor_copy(Tensor *dst, const Tensor *src) {
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        tensor_cpu_copy(dst, src);
        break;
    default:
        tensor_cuda_copy(dst, src);
        break;
    }
}

// ---- fill / clear --------------------------------------------------------

void tensor_fill(Tensor *tensor, f32 value) {
    if (tensor->on_gpu)
        tensor_cuda_fill(tensor, value);
    else
        tensor_cpu_fill(tensor, value);
}

void tensor_clear(Tensor *tensor) {
    if (tensor->on_gpu)
        tensor_cuda_clear(tensor);
    else
        tensor_cpu_clear(tensor);
}

// ---- activations (relu, exp) ---------------------------------------------

b32 tensor_relu(Tensor *dst, const Tensor *src) {
    if (!tensor_shape_eq(dst, src)) {
        printf("tensor_relu: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        tensor_cpu_relu(dst, src);
        return true;
    case 0b11:
        tensor_cuda_relu(dst, src);
        return true;
    default:
        printf("tensor_relu: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_relu(const Tensor *src) {
    Tensor *dst = tensor_create_like(src);
    if (!tensor_relu(dst, src)) {
        delete dst;
        return nullptr;
    }
    return dst;
}

b32 tensor_exp(Tensor *dst, const Tensor *src) {
    if (!tensor_shape_eq(dst, src)) {
        printf("tensor_exp: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        tensor_cpu_exp(dst, src);
        return true;
    case 0b11:
        tensor_cuda_exp(dst, src);
        return true;
    default:
        printf("tensor_exp: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_exp(const Tensor *src) {
    Tensor *dst = tensor_create_like(src);
    if (!tensor_exp(dst, src)) {
        delete dst;
        return nullptr;
    }
    return dst;
}

b32 tensor_log(Tensor *dst, const Tensor *src) {
    if (!tensor_shape_eq(dst, src)) {
        printf("tensor_log: shape mismatch\n");
        return false;
    }
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        tensor_cpu_log(dst, src);
        return true;
    case 0b11:
        tensor_cuda_log(dst, src);
        return true;
    default:
        printf("tensor_log: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_log(const Tensor *src) {
    Tensor *dst = tensor_create_like(src);
    if (!tensor_log(dst, src)) {
        delete dst;
        return nullptr;
    }
    return dst;
}

// ---- add -----------------------------------------------------------------

b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!check_broadcast(out, a, b, "tensor_add"))
        return false;
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_add(out, a, b);
        return true;
    case 0b111:
        tensor_cuda_add(out, a, b);
        return true;
    default:
        printf("tensor_add: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("tensor_add: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_add(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- sub -----------------------------------------------------------------

b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!check_broadcast(out, a, b, "tensor_sub"))
        return false;
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_sub(out, a, b);
        return true;
    case 0b111:
        tensor_cuda_sub(out, a, b);
        return true;
    default:
        printf("tensor_sub: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_sub(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("tensor_sub: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_sub(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- mul (elementwise) ---------------------------------------------------

b32 tensor_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!check_broadcast(out, a, b, "tensor_mul"))
        return false;
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_mul(out, a, b);
        return true;
    case 0b111:
        tensor_cuda_mul(out, a, b);
        return true;
    default:
        printf("tensor_mul: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_mul(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("tensor_mul: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_mul(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- div (elementwise) ---------------------------------------------------

b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!check_broadcast(out, a, b, "tensor_div"))
        return false;
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_div(out, a, b);
        return true;
    case 0b111:
        tensor_cuda_div(out, a, b);
        return true;
    default:
        printf("tensor_div: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_div(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("tensor_div: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_div(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}
// ---- equals(elementwise) ------------------------------------------------

b32 tensor_equal(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!check_broadcast(out, a, b, "tensor_equal"))
        return false;
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_equal(out, a, b);
        return true;
    case 0b111:
        tensor_cuda_equal(out, a, b);
        return true;
    default:
        printf("tensor_div: all tensors must be on the same device\n");
        return false;
    }
}
Tensor *tensor_equal(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("tensor_equal: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_equal(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- relu_grad (elementwise) ---------------------------------------------
b32 tensor_relu_backward(Tensor *out, const Tensor *grad, const Tensor *in) {
    if (!tensor_shape_eq(out, in) || !tensor_shape_eq(grad, in)) {
        printf("tensor_relu_backward: shape mismatch\n");
        return false;
    }
    switch ((out->on_gpu << 2) | (grad->on_gpu << 1) | in->on_gpu) {
    case 0b000:
        tensor_cpu_relu_backward(out, grad, in);
        return true;
    case 0b111:
        tensor_cuda_relu_backward(out, grad, in);
        return true;
    default:
        printf(
            "tensor_relu_backward: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_relu_backward(const Tensor *grad, const Tensor *in) {
    Tensor *out = tensor_create_like(in);
    if (!tensor_relu_backward(out, grad, in)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- softmax -------------------------------------------------------------
// Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

b32 tensor_softmax(Tensor *out, const Tensor *in) {
    if (!tensor_shape_eq(out, in)) {
        printf("tensor_softmax: shape mismatch\n");
        return false;
    }
    if (out->on_gpu != in->on_gpu) {
        printf("tensor_softmax: tensors must be on the same device\n");
        return false;
    }

    // Per-row max and sum — shape [N, 1] for a [N, C] input
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, in->shape, in->ndim * sizeof(u32));
    u32 col_dim = COL_DIM(in);
    row_shape[col_dim] = 1;

    Tensor *row_max = new Tensor(in->ndim, row_shape, in->on_gpu);
    Tensor *row_sum = new Tensor(in->ndim, row_shape, in->on_gpu);

    tensor_max(row_max, in, col_dim, true); // per-row max  [N,1]
    tensor_sub(out, in, row_max);           // broadcast subtract
    tensor_exp(out, out);
    tensor_sum(row_sum, out, col_dim, true); // per-row sum  [N,1]
    tensor_div(out, out, row_sum);           // broadcast divide

    delete row_max;
    delete row_sum;
    return true;
}

Tensor *tensor_softmax(const Tensor *in) {
    Tensor *out = tensor_create_like(in);
    if (!tensor_softmax(out, in)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- add (scalar) --------------------------------------------------------

b32 tensor_add(Tensor *out, const Tensor *a, f32 scalar) {
    if (!tensor_shape_eq(out, a)) {
        printf("tensor_add: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu << 1) | a->on_gpu) {
    case 0b00:
        tensor_cpu_add(out, a, scalar);
        return true;
    case 0b11:
        tensor_cuda_add(out, a, scalar);
        return true;
    default:
        printf("tensor_add: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_add(const Tensor *a, f32 scalar) {
    Tensor *out = tensor_create_like(a);
    if (!tensor_add(out, a, scalar)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- sub (scalar) --------------------------------------------------------

b32 tensor_sub(Tensor *out, const Tensor *a, f32 scalar) {
    if (!tensor_shape_eq(out, a)) {
        printf("tensor_sub: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu << 1) | a->on_gpu) {
    case 0b00:
        tensor_cpu_sub(out, a, scalar);
        return true;
    case 0b11:
        tensor_cuda_sub(out, a, scalar);
        return true;
    default:
        printf("tensor_sub: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_sub(const Tensor *a, f32 scalar) {
    Tensor *out = tensor_create_like(a);
    if (!tensor_sub(out, a, scalar)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- mul (scalar) --------------------------------------------------------

b32 tensor_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    if (!tensor_shape_eq(out, tensor)) {
        printf("tensor_mul: out and tensor must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_mul(out, tensor, scalar);
        return true;
    case 0b11:
        tensor_cuda_mul(out, tensor, scalar);
        return true;
    default:
        printf("tensor_mul: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_mul(const Tensor *tensor, f32 scalar) {
    Tensor *out = tensor_create_like(tensor);
    if (!tensor_mul(out, tensor, scalar)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- div (scalar) --------------------------------------------------------

b32 tensor_div(Tensor *out, const Tensor *a, f32 scalar) {
    if (!tensor_shape_eq(out, a)) {
        printf("tensor_div: out and a must have the same shape\n");
        return false;
    }
    switch ((out->on_gpu << 1) | a->on_gpu) {
    case 0b00:
        tensor_cpu_div(out, a, scalar);
        return true;
    case 0b11:
        tensor_cuda_div(out, a, scalar);
        return true;
    default:
        printf("tensor_div: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_div(const Tensor *a, f32 scalar) {
    Tensor *out = tensor_create_like(a);
    if (!tensor_div(out, a, scalar)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- mat_mul -------------------------------------------------------------

b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                   b32 clear_out) {
    if (a->ndim != 2 || b->ndim != 2) {
        printf("tensor_mat_mul: only 2-D tensors supported\n");
        return false;
    }
    if (a->shape[COL_DIM(a)] != b->shape[ROW_DIM(b)] ||
        a->shape[ROW_DIM(a)] != out->shape[ROW_DIM(out)] ||
        b->shape[COL_DIM(b)] != out->shape[COL_DIM(out)]) {
        printf("tensor_mat_mul: shape mismatch\n");
        return false;
    }
    switch ((out->on_gpu << 2) | (a->on_gpu << 1) | b->on_gpu) {
    case 0b000:
        tensor_cpu_mat_mul(out, a, b, clear_out);
        return true;
    case 0b111:
        tensor_cuda_mat_mul(out, a, b, clear_out);
        return true;
    default:
        printf("tensor_mat_mul: all tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_mat_mul(const Tensor *a, const Tensor *b) {
    u32 shape[2] = {a->shape[ROW_DIM(a)], b->shape[COL_DIM(b)]};
    Tensor *out = new Tensor(2, shape, a->on_gpu);
    if (!tensor_mat_mul(out, a, b, false)) {
        printf("Shape of A: [%d, %d]\n", a->shape[ROW_DIM(a)],
               a->shape[COL_DIM(a)]);
        printf("Shape of B: [%d, %d]\n", b->shape[ROW_DIM(b)],
               b->shape[COL_DIM(b)]);
        delete out;
        return nullptr;
    }
    return out;
}

// ---- sum -----------------------------------------------------------------

b32 tensor_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
    if (out->size != 1) {
        printf("tensor_sum: out must be a scalar tensor (size=1)\n");
        return false;
    }
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_sum(out, tensor, clear_out);
        return true;
    case 0b11:
        tensor_cuda_sum(out, tensor);
        return true;
    default:
        printf("tensor_sum: tensors must be on the same device\n");
        return false;
    }
}

b32 tensor_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
               b32 clear_out) {
    if (dim >= tensor->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return false;
    }
    if (!check_reduction_shape(out, tensor, dim, "tensor_sum"))
        return false;
    // backends always receive out with keep_dim shape (shape[dim]=1, same ndim)
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_sum(out, tensor, dim, clear_out);
        break;
    case 0b11:
        tensor_cuda_sum(out, tensor, dim);
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

Tensor *tensor_sum(const Tensor *tensor) {
    u32 shape[1] = {1};
    Tensor *out = new Tensor(1, shape, tensor->on_gpu);
    if (!tensor_sum(out, tensor)) {
        delete out;
        return nullptr;
    }
    return out;
}

Tensor *tensor_sum(const Tensor *tensor, u32 dim, b32 keep_dim) {
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, tensor->shape, tensor->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor *out = new Tensor(tensor->ndim, out_shape, tensor->on_gpu);
    if (!tensor_sum(out, tensor, dim, keep_dim)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- max -----------------------------------------------------------------

b32 tensor_max(Tensor *out, const Tensor *tensor) {
    if (out->size != 1) {
        printf("tensor_max: out must be a scalar tensor (size=1)\n");
        return false;
    }
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_max(out, tensor);
        return true;
    case 0b11:
        tensor_cuda_max(out, tensor);
        return true;
    default:
        printf("tensor_max: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_max(const Tensor *tensor) {
    u32 shape[1] = {1};
    Tensor *out = new Tensor(1, shape, tensor->on_gpu);
    if (!tensor_max(out, tensor)) {
        delete out;
        return nullptr;
    }
    return out;
}

b32 tensor_max(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim) {
    if (dim >= tensor->ndim) {
        printf("tensor_max: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return false;
    }
    if (!check_reduction_shape(out, tensor, dim, "tensor_max"))
        return false;
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_max(out, tensor, dim);
        break;
    case 0b11:
        tensor_cuda_max(out, tensor, dim);
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

Tensor *tensor_max(const Tensor *tensor, u32 dim, b32 keep_dim) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = tensor->ndim;
    memcpy(out_shape, tensor->shape, out_ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor *out = new Tensor(out_ndim, out_shape, tensor->on_gpu);
    tensor_max(out, tensor, dim, keep_dim);
    return out;
}

// ---- argmax --------------------------------------------------------------

b32 tensor_argmax(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim) {
    if (dim >= tensor->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return false;
    }
    if (!check_reduction_shape(out, tensor, dim, "tensor_argmax"))
        return false;
    switch ((out->on_gpu << 1) | tensor->on_gpu) {
    case 0b00:
        tensor_cpu_argmax(out, tensor, dim);
        break;
    case 0b11:
        tensor_cuda_argmax(out, tensor, dim);
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

Tensor *tensor_argmax(const Tensor *tensor, u32 dim, b32 keep_dim) {
    if (dim >= tensor->ndim) {
        printf("tensor_argmax: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return nullptr;
    }
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, tensor->shape, tensor->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor *out = new Tensor(tensor->ndim, out_shape, tensor->on_gpu);
    if (!tensor_argmax(out, tensor, dim, keep_dim)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- intializing ---------------------------------------------------------
void tensor_he_init(Tensor *tensor) {
    if (tensor->on_gpu) {
        tensor_cuda_he_init(tensor);
    } else {
        tensor_cpu_he_init(tensor);
    }
}

// ---- indexing ------------------------------------------------------------

static b32 check_index_select(const Tensor *dst, const Tensor *src,
                              u32 n_indices, u32 dim) {
    if (dim >= src->ndim) {
        printf("tensor_index_select: dim %u out of range (ndim=%u)\n", dim,
               src->ndim);
        return false;
    }
    if (dst->ndim != src->ndim) {
        printf("tensor_index_select: dst and src must have the same ndim\n");
        return false;
    }
    if (dst->shape[dim] != n_indices) {
        printf("tensor_index_select: dst->shape[%u]=%u does not match "
               "n_indices=%u\n",
               dim, dst->shape[dim], n_indices);
        return false;
    }
    for (u32 i = 0; i < src->ndim; i++) {
        if (i != dim && dst->shape[i] != src->shape[i]) {
            printf("tensor_index_select: shape mismatch at dim %u "
                   "(dst=%u, src=%u)\n",
                   i, dst->shape[i], src->shape[i]);
            return false;
        }
    }
    return true;
}

b32 tensor_index_select(Tensor *dst, const Tensor *src, const u32 *indices,
                        u32 n_indices, u32 dim) {
    if (!check_index_select(dst, src, n_indices, dim))
        return false;
    switch ((dst->on_gpu << 1) | src->on_gpu) {
    case 0b00:
        tensor_cpu_index_select(dst, src, indices, n_indices, dim);
        return true;
    case 0b11:
        tensor_cuda_index_select(dst, src, indices, n_indices, dim);
        return true;
    default:
        printf("tensor_index_select: tensors must be on the same device\n");
        return false;
    }
}

Tensor *tensor_index_select(const Tensor *src, const u32 *indices,
                            u32 n_indices, u32 dim) {
    if (dim >= src->ndim) {
        printf("tensor_index_select: dim %u out of range (ndim=%u)\n", dim,
               src->ndim);
        return nullptr;
    }
    u32 shape[MAX_NDIM];
    for (u32 i = 0; i < src->ndim; i++)
        shape[i] = (i == dim) ? n_indices : src->shape[i];
    Tensor *dst = new Tensor(src->ndim, shape, src->on_gpu);
    if (!tensor_index_select(dst, src, indices, n_indices, dim)) {
        delete dst;
        return nullptr;
    }
    return dst;
}
