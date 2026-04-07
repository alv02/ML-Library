#include "../include/tensor.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>

using namespace std;
Tensor *tensor_create(u32 ndim, u32 *shape, b32 on_gpu) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->ndim = ndim;
    tensor->on_gpu = on_gpu;
    tensor->size = 1;
    tensor->owns_data = true;

    tensor->shape = (u32 *)malloc(ndim * sizeof(u32));
    for (u32 i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->stride = (u64 *)malloc(ndim * sizeof(u64));
    for (u32 i = ndim; i-- > 0;) {
        tensor->stride[i] = tensor->size;
        tensor->size *= tensor->shape[i];
    }

    if (on_gpu) {
        // cudaMalloc(&tensor->data, tensor->size * sizeof(f32));
    } else {
        tensor->data = (f32 *)malloc(tensor->size * sizeof(f32));
        memset(tensor->data, 0, tensor->size * sizeof(f32));
    }

    return tensor;
}

void tensor_free(Tensor *tensor) {
    free(tensor->shape);
    free(tensor->stride);
    if (tensor->owns_data) {
        if (tensor->on_gpu) {
            // cudaFree(tensor->data);

        } else {
            free(tensor->data);
        }
    }
    free(tensor);
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

    Tensor *tensor = tensor_create(ndim, shape, on_gpu);

    fread(tensor->data, sizeof(f32), tensor->size, file);

    fclose(file);

    return tensor;
}

void tensor_fill(Tensor *tensor, f32 value) {
    for (u64 i = 0; i < tensor->size; i++) {
        tensor->data[i] = value;
    }
}

b32 tensor_shape_eq(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim)
        return false;
    for (u32 i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return false;
    }
    return true;
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

// TODO: Maybe try to do broadcasting, look how pytorch do its without affecting
// performace
b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    // if (!tensor_shape_eq(a, b)) {
    //     return false;
    // }
    // if (!tensor_shape_eq(a, out)) {
    //     return false;
    // }

    // TODO: Hardcoded
    if (a->size == 1) {
        for (u64 i = 0; i < out->size; i++) {
            out->data[i] = a->data[0] + b->data[i];
        }
        return true;
    }

    if (b->size == 1) {
        for (u64 i = 0; i < out->size; i++) {
            out->data[i] = a->data[i] + b->data[0];
        }
        return true;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    Tensor *out = tensor_create_like(a);
    if (!tensor_add(out, a, b)) {
        printf("Failed to add tensors\n");
        tensor_free(out);
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

b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!tensor_shape_eq(a, b)) {
        return false;
    }
    if (!tensor_shape_eq(a, out)) {
        return false;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

b32 tensor_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!tensor_shape_eq(a, b)) {
        return false;
    }
    if (!tensor_shape_eq(a, out)) {
        return false;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return true;
}
b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!tensor_shape_eq(a, b)) {
        return false;
    }
    if (!tensor_shape_eq(a, out)) {
        return false;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }

    return true;
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

    if ((a_cols != b_rows) | (a_rows != out_rows) | (b_cols != out_cols)) {
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
    Tensor *out = tensor_create(2, shape, a->on_gpu);
    if (!tensor_mat_mul(out, a, b, false)) {
        printf("Matrix multiplication failed due to shape mismatch\n");
        printf("Shape of A: [%d, %d]\n", mat_rows(a), mat_cols(a));
        printf("Shape of B: [%d, %d]\n", mat_rows(b), mat_cols(b));
        printf("Shape of Output: [%d, %d]\n", mat_rows(out), mat_cols(out));
    }
    return out;
}

void tensor_scale(Tensor *out, const Tensor *tensor, f32 scale) {
    for (u64 i = 0; i < tensor->size; i++) {
        out->data[i] = tensor->data[i] * scale;
    }
}
f32 tensor_sum(const Tensor *tensor) {
    f32 sum = 0;
    for (u64 i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }

    return sum;
}

Tensor *tensor_view(const Tensor *src) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = src->data;
    tensor->size = src->size;
    tensor->ndim = src->ndim;
    tensor->on_gpu = src->on_gpu;
    tensor->owns_data = false;

    tensor->shape = (u32 *)malloc(tensor->ndim * sizeof(u32));

    tensor->stride = (u64 *)malloc(tensor->ndim * sizeof(u64));
    memcpy(tensor->shape, src->shape, src->ndim * sizeof(u32));
    memcpy(tensor->stride, src->stride, src->ndim * sizeof(u64));

    return tensor;
}
Tensor *tensor_create_like(const Tensor *src) {
    Tensor *t = tensor_create(src->ndim, src->shape, src->on_gpu);
    return t;
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
