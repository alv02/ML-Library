#include "../include/tensor.hpp"
#include "string.h"
#include <bits/stdc++.h>
#include <cstdlib>

using namespace std;
Tensor *tensor_create(mem_arena *arena, u32 ndim, u32 *shape, b32 on_gpu) {
    Tensor *tensor = PUSH_STRUCT(arena, Tensor);
    tensor->ndim = ndim;
    tensor->on_gpu = on_gpu;
    tensor->size = 1;
    tensor->shape = PUSH_ARRAY(arena, u32, ndim);
    for (u32 i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->stride = PUSH_ARRAY(arena, u64, ndim);
    // Compute size and stride
    for (u32 i = ndim; i-- > 0;) {
        tensor->stride[i] = tensor->size;
        tensor->size *= tensor->shape[i];
    }

    tensor->data = PUSH_ARRAY(arena, f32, tensor->size);

    return tensor;
}
Tensor *tensor_load(mem_arena *arena, const char *filename, b32 on_gpu) {
    FILE *file = fopen(filename, "rb");

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

    Tensor *tensor = tensor_create(arena, ndim, shape, on_gpu);

    fread(tensor->data, sizeof(f32), tensor->size, file);

    fclose(file);

    return tensor;
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

b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!tensor_shape_eq(a, b)) {
        return false;
    }
    if (!tensor_shape_eq(a, out)) {
        return false;
    }

    for (u64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
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

    u64 a_r = a->stride[ROW_DIM(a)];
    u64 a_c = a->stride[COL_DIM(a)];
    u64 b_r = b->stride[ROW_DIM(b)];
    u64 b_c = b->stride[COL_DIM(b)];
    u64 o_r = out->stride[ROW_DIM(out)];
    u64 o_c = out->stride[COL_DIM(out)];

    for (u32 i = 0; i < M; i++) {
        for (u32 k = 0; k < N; k++) {
            for (u32 j = 0; j < P; j++) {
                out->data[i * o_r + j * o_c] +=
                    a->data[i * a_r + k * a_c] * b->data[k * b_r + j * b_c];
            }
        }
    }
}

void _tensor_mat_mul_nt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    u64 a_r = a->stride[ROW_DIM(a)];
    u64 a_c = a->stride[COL_DIM(a)];
    u64 b_r = b->stride[ROW_DIM(b)];
    u64 b_c = b->stride[COL_DIM(b)];
    u64 o_r = out->stride[ROW_DIM(out)];
    u64 o_c = out->stride[COL_DIM(out)];

    for (u32 i = 0; i < M; i++) {
        for (u32 j = 0; j < P; j++) {
            for (u32 k = 0; k < N; k++) {
                out->data[i * o_r + j * o_c] +=
                    a->data[i * a_r + k * a_c] * b->data[k * b_r + j * b_c];
            }
        }
    }
}

void _tensor_mat_mul_tn(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    u64 a_r = a->stride[ROW_DIM(a)];
    u64 a_c = a->stride[COL_DIM(a)];
    u64 b_r = b->stride[ROW_DIM(b)];
    u64 b_c = b->stride[COL_DIM(b)];
    u64 o_r = out->stride[ROW_DIM(out)];
    u64 o_c = out->stride[COL_DIM(out)];

    for (u32 k = 0; k < N; k++) {
        for (u32 i = 0; i < M; i++) {
            for (u32 j = 0; j < P; j++) {
                out->data[i * o_r + j * o_c] +=
                    a->data[i * a_r + k * a_c] * b->data[k * b_r + j * b_c];
            }
        }
    }
}

void _tensor_mat_mul_tt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a);
    u32 N = mat_cols(a);
    u32 P = mat_cols(b);

    u64 a_r = a->stride[ROW_DIM(a)];
    u64 a_c = a->stride[COL_DIM(a)];
    u64 b_r = b->stride[ROW_DIM(b)];
    u64 b_c = b->stride[COL_DIM(b)];
    u64 o_r = out->stride[ROW_DIM(out)];
    u64 o_c = out->stride[COL_DIM(out)];

    for (u32 j = 0; j < P; j++) {
        for (u32 k = 0; k < N; k++) {
            for (u32 i = 0; i < M; i++) {
                out->data[i * o_r + j * o_c] +=
                    a->data[i * a_r + k * a_c] * b->data[k * b_r + j * b_c];
            }
        }
    }
}

// TODO: Maximize performance loops so it uses sequential data, see the strides
// and decide how to loop with that
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
b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b) {
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

    // The out tensor must be 0, TODO: Maybe just always assume its 0 i dont
    // know if this affects performace significally
    tensor_clear(out);
    _tensor_mat_mul(out, a, b);
    return true;
}

void tensor_scale(Tensor *tensor, f32 scale) {
    for (u64 i = 0; i < tensor->size; i++) {
        tensor->data[i] *= scale;
    }
}
f32 tensor_sum(Tensor *tensor) {
    f32 sum = 0;
    for (u64 i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }

    return sum;
}

b32 tensor_sum(Tensor *out, const Tensor *a, u32 dim) {
    if (dim >= a->ndim) {
        return false;
    }
    if (out->ndim != a->ndim) {
        return false;
    }
    for (u32 i = 0; i < a->ndim; i++) {
        u32 expected = (i == dim) ? 1 : a->shape[i];
        if (out->shape[i] != expected) {
            return false;
        }
    }
    return true;
}
Tensor *tensor_view(mem_arena *arena, Tensor *src) {
    Tensor *tensor = PUSH_STRUCT(arena, Tensor);
    tensor->data = src->data;
    tensor->size = src->size;
    tensor->ndim = src->ndim;
    tensor->on_gpu = src->on_gpu;

    tensor->shape = PUSH_ARRAY(arena, u32, tensor->ndim);
    tensor->stride = PUSH_ARRAY(arena, u64, tensor->ndim);
    memcpy(tensor->shape, src->shape, src->ndim * sizeof(u32));
    memcpy(tensor->stride, src->stride, src->ndim * sizeof(u64));

    return tensor;
}
Tensor *tensor_create_like(mem_arena *arena, const Tensor *src) {
    Tensor *t = tensor_create(arena, src->ndim, src->shape, src->on_gpu);
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
    printf("], data=\n");

    // print data — only makes sense for 1D/2D really
    for (u64 i = 0; i < tensor->size; i++) {
        printf("%.4f ", tensor->data[i]);
    }
    printf(")\n");
}
