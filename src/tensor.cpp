#include "../include/tensor.hpp"
#include "../include/tensor_iterator.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>
using namespace std;
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
    // TODO: Handle header other than shape
    char *shape_ptr = strstr(header, "shape");
    char *end_ptr;
    shape_ptr = strchr(shape_ptr, '(');

    u32 ndim = 0;

    // Calculate ndims, not dimension of shape 0 allowed, TODO: maybe better way
    // of doing this
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
        while (*shape_ptr == ',') {
            shape_ptr++;
        }
    }

    Tensor *tensor = new Tensor(ndim, shape, on_gpu);

    fread(tensor->data, sizeof(f32), tensor->size, file);

    fclose(file);

    return tensor;
}

void tensor_fill(Tensor *tensor, f32 value) {
    for (u64 i = 0; i < tensor->size; i++) {
        tensor->data[i] = value;
    }
}

b32 shape_eq(u32 ndim_a, const u32 *shape_a, u32 ndim_b, const u32 *shape_b) {
    if (ndim_a != ndim_b) {
        return false;
    }

    for (u32 i = 0; i < ndim_a; i++) {
        if (shape_a[i] != shape_b[i]) {
            return false;
        }
    }
    return true;
}
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
    return shape_eq(a->ndim, a->shape, b->ndim, b->shape);
}

b32 tensor_transpose(Tensor *tensor, u32 dim0, u32 dim1) {
    if (dim0 == dim1) {
        return false;
    }

    if (dim0 >= tensor->ndim || dim1 >= tensor->ndim) {
        return false;
    }

    u32 temp_shape = tensor->shape[dim1];
    tensor->shape[dim1] = tensor->shape[dim0];
    tensor->shape[dim0] = temp_shape;

    u64 temp_stride = tensor->stride[dim1];
    tensor->stride[dim1] = tensor->stride[dim0];
    tensor->stride[dim0] = temp_stride;

    return true;
}
void tensor_clear(Tensor *tensor) {
    memset(tensor->data, 0, sizeof(f32) * tensor->size);
}

u32 broadcast_shape(const Tensor *a, const Tensor *b, u32 *out_shape) {
    u32 expanded_ndim = max(a->ndim, b->ndim);

    for (u32 i = 0; i < expanded_ndim; i++) {
        // align from the right (numpy semantics)
        i32 ai = (i32)i - (i32)(expanded_ndim - a->ndim);
        i32 bi = (i32)i - (i32)(expanded_ndim - b->ndim);
        u32 da = (ai >= 0) ? a->shape[ai] : 1;
        u32 db = (bi >= 0) ? b->shape[bi] : 1;
        if (da != db) {
            if (min(da, db) != 1)
                return 0;
        }
        out_shape[i] = max(da, db);
    }
    return expanded_ndim;
}

void expanded_shape(const Tensor *t, const u32 *expanded_shape,
                    u32 expanded_ndim, u32 *t_expanded_shape) {

    u32 n_prepend = expanded_ndim - t->ndim;
    u32 i = 0;

    // prepend
    for (; i < n_prepend; i++)
        t_expanded_shape[i] = 1;

    for (; i < expanded_ndim; i++) {
        u32 ti = i - n_prepend;
        t_expanded_shape[i] = t->shape[ti];
    }
}

void expanded_stride(const Tensor *t, const u32 *expanded_shape,
                     u32 expanded_ndim, u64 *t_expanded_stride) {
    u32 n_prepend = expanded_ndim - t->ndim;
    u32 i = 0;

    // prepend
    for (; i < n_prepend; i++)
        t_expanded_stride[i] = 0;

    for (; i < expanded_ndim; i++) {
        u32 ti = i - n_prepend;
        t_expanded_stride[i] = (t->shape[ti] != 1) ? t->stride[ti] : 0;
    }
}

template <typename Fn>
b32 elementwise_binary(Tensor *out, const Tensor *a, const Tensor *b, Fn fn) {
    u32 out_shape[MAX_NDIM];
    u32 out_dim = broadcast_shape(a, b, out_shape);
    if (out_dim == 0) {
        printf("Tensors are not broadcastable. \n");
        return false;
    }

    if (!shape_eq(out_dim, out_shape, out->ndim, out->shape)) {
        printf("Tensor out is not in broadcastable shape. \n");
        printf("Out: ");
        tensor_print(out);
        printf("a: ");
        tensor_print(a);
        printf("b: ");
        tensor_print(b);
        return false;
    }

    // Use flat index for speed up if not need to broadcast
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(a) &&
        tensor_is_contiguous(b)) {
        for (u64 i = 0; i < out->size; i++) {
            out->data[i] = fn(a->data[i], b->data[i]);
        }
        return true;
    }

    u64 a_strides[MAX_NDIM];
    u64 b_strides[MAX_NDIM];
    expanded_stride(a, out_shape, out->ndim, a_strides);
    expanded_stride(b, out_shape, out->ndim, b_strides);

    tensorIterator a_iter(out->ndim, out_shape, a_strides);
    tensorIterator b_iter(out->ndim, out_shape, b_strides);

    for (u64 i = 0; i < out->size; i++)
        out->data[i] = fn(a->data[a_iter.next()], b->data[b_iter.next()]);

    return true;
}
b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x + y; });
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
    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + scalar;
    }
    return out;
}

b32 tensor_div(Tensor *out, const Tensor *a, f32 scalar) {
    if (!tensor_shape_eq(out, a)) {
        printf("Different shape in div");
        return false;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] / scalar;
    }
    return true;
}
Tensor *tensor_div(const Tensor *a, f32 scalar) {
    Tensor *out = tensor_create_like(a);
    tensor_div(out, a, scalar);
    return out;
}

b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x - y; });
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

b32 tensor_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x * y; });
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

b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x / y; });
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

inline u32 mat_rows(const Tensor *t) { return t->shape[ROW_DIM(t)]; }
inline u32 mat_cols(const Tensor *t) { return t->shape[COL_DIM(t)]; }

void _tensor_mat_mul_nn(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    for (u32 i = 0; i < M; i++) {
        for (u32 k = 0; k < N; k++) {
            for (u32 j = 0; j < P; j++) {
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
            }
        }
    }
}

void _tensor_mat_mul_nt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    for (u32 i = 0; i < M; i++) {
        for (u32 j = 0; j < P; j++) {
            for (u32 k = 0; k < N; k++) {
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
            }
        }
    }
}

void _tensor_mat_mul_tn(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);
    for (u32 k = 0; k < N; k++) {
        for (u32 i = 0; i < M; i++) {
            for (u32 j = 0; j < P; j++) {
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
            }
        }
    }
}

void _tensor_mat_mul_tt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    for (u32 j = 0; j < P; j++) {
        for (u32 k = 0; k < N; k++) {
            for (u32 i = 0; i < M; i++) {
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
            }
        }
    }
}

void _tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b) {

    u64 a_r = a->stride[ROW_DIM(a)];
    u64 a_c = a->stride[COL_DIM(a)];
    u64 b_r = b->stride[ROW_DIM(b)];
    u64 b_c = b->stride[COL_DIM(b)];

    b32 transpose_a = a_r < a_c;
    b32 transpose_b = b_r < b_c;

    u8 transpose = (transpose_a << 1) | (transpose_b);
    switch (transpose) {
    case 0b00:
        _tensor_mat_mul_nn(out, a, b);
        break;
    case 0b01:
        _tensor_mat_mul_nt(out, a, b);
        break;
    case 0b10:
        _tensor_mat_mul_tn(out, a, b);
        break;
    case 0b11:
        _tensor_mat_mul_tt(out, a, b);
        break;
    }
}

// m*n @ n*p = m*p
b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                   b32 clear_out) {
    // TODO: Right now only handle 2 dims, if ndim > 2 should batch
    if (a->ndim != 2 || b->ndim != 2) {
        printf("Not supported mat_mul");
        return false;
    }

    u32 a_rows = mat_rows(a);
    u32 a_cols = mat_cols(a);
    u32 b_rows = mat_rows(b);
    u32 b_cols = mat_cols(b);
    u32 out_rows = mat_rows(out);
    u32 out_cols = mat_cols(out);
    ;

    if ((a_cols != b_rows) || (a_rows != out_rows) || (b_cols != out_cols)) {
        return false;
    }

    if (clear_out) {
        tensor_clear(out);
    }
    _tensor_mat_mul(out, a, b);
    return true;
}

Tensor *tensor_mat_mul(const Tensor *a, const Tensor *b) {
    // Create the tensor and call the above function
    u32 M = mat_rows(a);
    u32 P = mat_cols(b);
    u32 shape[2] = {M, P};
    Tensor *out = new Tensor(2, shape, a->on_gpu);
    if (!tensor_mat_mul(out, a, b, false)) {
        printf("Matrix multiplication failed due to shape mismatch\n");
        printf("Shape of A: [%d, %d]\n", mat_rows(a), mat_cols(a));
        printf("Shape of B: [%d, %d]\n", mat_rows(b), mat_cols(b));
        printf("Shape of Output: [%d, %d]\n", mat_rows(out), mat_cols(out));

        delete out;
        return nullptr;
    }
    return out;
}

void tensor_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    for (u64 i = 0; i < tensor->size; i++) {
        out->data[i] = tensor->data[i] * scalar;
    }
}

Tensor *tensor_mul(const Tensor *tensor, f32 scalar) {
    Tensor *out = tensor_create_like(tensor);

    tensor_mul(out, tensor, scalar);

    return out;
}

b32 tensor_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
    if (out->size != 1) {
        printf("tensor_sum: out must be a scalar tensor (size=1)\n");
        return false;
    }
    if (clear_out)
        tensor_clear(out);
    tensorIterator it(tensor->ndim, tensor->shape, tensor->stride);
    while (it.has_next())
        out->data[0] += tensor->data[it.next()];
    return true;
}

b32 tensor_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
               b32 clear_out) {
    if (dim >= tensor->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return false;
    }
    if (clear_out)
        tensor_clear(out);
    // Output strides with 0 for the summed dim: every element along that
    // dim maps to the same output slot (same idea as broadcast_stride).
    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out->stride, tensor->ndim * sizeof(u64));
    out_strides[dim] = 0;
    out->shape[dim] = 1;

    tensorIterator in_it(tensor->ndim, tensor->shape, tensor->stride);
    tensorIterator out_it(tensor->ndim, tensor->shape, out_strides);
    while (in_it.has_next())
        out->data[out_it.next()] += tensor->data[in_it.next()];

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

Tensor *tensor_sum(const Tensor *tensor, u32 dim) {
    if (dim >= tensor->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return nullptr;
    }
    u32 out_shape[MAX_NDIM];
    memcpy(out_shape, tensor->shape, tensor->ndim * sizeof(u32));
    out_shape[dim] = 1;
    Tensor *out = new Tensor(tensor->ndim, out_shape, tensor->on_gpu);
    if (!tensor_sum(out, tensor, dim)) {
        delete out;
        return nullptr;
    }
    return out;
}

Tensor *tensor_view(const Tensor *src) { return new Tensor(src); }

Tensor *tensor_create_like(const Tensor *src) {
    return new Tensor(src->ndim, src->shape, src->on_gpu);
}

b32 tensor_reshape(Tensor *tensor, const u32 *shape, u32 ndim) {
    u64 shape_size = 1;
    for (u32 i = 0; i < ndim; i++) {
        shape_size *= shape[i];
    }
    if (tensor->size != shape_size) {
        return false;
    }

    for (u32 i = 0; i < ndim; i++)
        tensor->shape[i] = shape[i];

    shape_size = 1;
    for (u32 i = ndim; i-- > 0;) {
        tensor->stride[i] = shape_size;
        shape_size *= shape[i];
    }
    tensor->ndim = ndim;

    return true;
}

void tensor_print(const Tensor *tensor) {
    // print shape
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
