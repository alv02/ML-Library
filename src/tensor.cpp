#include "../include/tensor.hpp"
#include "../include/backend/tensor_cpu.hpp"
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
        // cudaMalloc(&this->data, this->size * sizeof(f32));
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
            // cudaFree(data);
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

    Tensor *tensor = new Tensor(ndim, shape, on_gpu);
    fread(tensor->data, sizeof(f32), tensor->size, file);
    fclose(file);

    return tensor;
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

// ---- Dispatched operations -----------------------------------------------
// Each 3-param (in-place) function dispatches to the right backend.
// Allocating 2-param wrappers handle shape logic then call the dispatcher.

void tensor_fill(Tensor *tensor, f32 value) {
    if (tensor->on_gpu) {
        // tensor_cuda_fill(tensor, value);
    } else {
        tensor_cpu_fill(tensor, value);
    }
}

void tensor_clear(Tensor *tensor) {
    if (tensor->on_gpu) {
        // tensor_cuda_clear(tensor);
    } else {
        tensor_cpu_clear(tensor);
    }
}

// ---- add -----------------------------------------------------------------

b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_add
    return tensor_cpu_add(out, a, b);
}

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("Failed to add tensors: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_add(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

Tensor *tensor_add(const Tensor *a, f32 scalar) {
    Tensor *out = tensor_create_like(a);
    if (a->on_gpu) {
        // tensor_cuda_add_scalar(out, a, scalar);
    } else {
        tensor_cpu_add_scalar(out, a, scalar);
    }
    return out;
}

// ---- sub -----------------------------------------------------------------

b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_sub
    return tensor_cpu_sub(out, a, b);
}

Tensor *tensor_sub(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("Failed to sub tensors: shapes not broadcastable\n");
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
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_mul
    return tensor_cpu_mul(out, a, b);
}

Tensor *tensor_mul(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("Failed to mul tensors: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_mul(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- mul (scalar) --------------------------------------------------------

void tensor_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    if (tensor->on_gpu) {
        // tensor_cuda_mul_scalar(out, tensor, scalar);
    } else {
        tensor_cpu_mul_scalar(out, tensor, scalar);
    }
}

Tensor *tensor_mul(const Tensor *tensor, f32 scalar) {
    Tensor *out = tensor_create_like(tensor);
    tensor_mul(out, tensor, scalar);
    return out;
}

// ---- div (elementwise) ---------------------------------------------------

b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b) {
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_div
    return tensor_cpu_div(out, a, b);
}

Tensor *tensor_div(const Tensor *a, const Tensor *b) {
    u32 out_shape[MAX_NDIM];
    u32 out_ndim = broadcast_shape(a, b, out_shape);
    if (out_ndim == 0) {
        printf("Failed to div tensors: shapes not broadcastable\n");
        return nullptr;
    }
    Tensor *out = new Tensor(out_ndim, out_shape, a->on_gpu);
    if (!tensor_div(out, a, b)) {
        delete out;
        return nullptr;
    }
    return out;
}

// ---- div (scalar) --------------------------------------------------------

b32 tensor_div(Tensor *out, const Tensor *a, f32 scalar) {
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_div_scalar
    return tensor_cpu_div_scalar(out, a, scalar);
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
    if (a->on_gpu)
        return false; // TODO: tensor_cuda_mat_mul
    return tensor_cpu_mat_mul(out, a, b, clear_out);
}

Tensor *tensor_mat_mul(const Tensor *a, const Tensor *b) {
    u32 shape[2] = {a->shape[ROW_DIM(a)], b->shape[COL_DIM(b)]};
    Tensor *out = new Tensor(2, shape, a->on_gpu);
    if (!tensor_mat_mul(out, a, b, false)) {
        printf("Matrix multiplication failed due to shape mismatch\n");
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
    if (tensor->on_gpu)
        return false; // TODO: tensor_cuda_sum
    return tensor_cpu_sum(out, tensor, clear_out);
}

b32 tensor_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
               b32 clear_out) {
    if (tensor->on_gpu)
        return false; // TODO: tensor_cuda_sum_dim
    return tensor_cpu_sum_dim(out, tensor, dim, keep_dim, clear_out);
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
    if (dim >= tensor->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return nullptr;
    }
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
