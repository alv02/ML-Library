#include "../../include/backend/tensor_cpu.hpp"
#include "../../include/tensor_iterator.hpp"
#include <cstdio>
#include <cstring>
using namespace std;

// Internal helper — same as tensor_shape_eq but works on raw arrays
static b32 shape_eq(u32 ndim_a, const u32 *shape_a, u32 ndim_b,
                    const u32 *shape_b) {
    if (ndim_a != ndim_b)
        return false;
    for (u32 i = 0; i < ndim_a; i++)
        if (shape_a[i] != shape_b[i])
            return false;
    return true;
}

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(Tensor *tensor, f32 value) {
    for (u64 i = 0; i < tensor->size; i++)
        tensor->data[i] = value;
}

void tensor_cpu_clear(Tensor *tensor) {
    memset(tensor->data, 0, sizeof(f32) * tensor->size);
}

// ---- elementwise binary (add / sub / mul / div) --------------------------

template <typename Fn>
static b32 elementwise_binary(Tensor *out, const Tensor *a, const Tensor *b,
                              Fn fn) {
    u32 out_shape[MAX_NDIM];
    u32 out_dim = broadcast_shape(a, b, out_shape);
    if (out_dim == 0) {
        printf("Tensors are not broadcastable.\n");
        return false;
    }
    if (!shape_eq(out_dim, out_shape, out->ndim, out->shape)) {
        printf("Tensor out is not in broadcastable shape.\n");
        printf("Out: ");
        tensor_print(out);
        printf("a: ");
        tensor_print(a);
        printf("b: ");
        tensor_print(b);
        return false;
    }

    // Fast path: no broadcast needed, contiguous memory
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(a) &&
        tensor_is_contiguous(b)) {
        for (u64 i = 0; i < out->size; i++)
            out->data[i] = fn(a->data[i], b->data[i]);
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

b32 tensor_cpu_add(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x + y; });
}

b32 tensor_cpu_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x - y; });
}

b32 tensor_cpu_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x * y; });
}

b32 tensor_cpu_div(Tensor *out, const Tensor *a, const Tensor *b) {
    return elementwise_binary(out, a, b, [](f32 x, f32 y) { return x / y; });
}

// ---- scalar operations ---------------------------------------------------

void tensor_cpu_mul_scalar(Tensor *out, const Tensor *tensor, f32 scalar) {
    for (u64 i = 0; i < tensor->size; i++)
        out->data[i] = tensor->data[i] * scalar;
}

b32 tensor_cpu_div_scalar(Tensor *out, const Tensor *a, f32 scalar) {
    if (!tensor_shape_eq(out, a)) {
        printf("tensor_div_scalar: shape mismatch\n");
        return false;
    }
    for (u64 i = 0; i < out->size; i++)
        out->data[i] = a->data[i] / scalar;
    return true;
}

void tensor_cpu_add_scalar(Tensor *out, const Tensor *a, f32 scalar) {
    for (u64 i = 0; i < out->size; i++)
        out->data[i] = a->data[i] + scalar;
}

// ---- matrix multiply -----------------------------------------------------

static inline u32 mat_rows(const Tensor *t) { return t->shape[ROW_DIM(t)]; }
static inline u32 mat_cols(const Tensor *t) { return t->shape[COL_DIM(t)]; }

static void _mm_nn(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 k = 0; k < N; k++)
            for (u32 j = 0; j < P; j++)
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
}

static void _mm_nt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < P; j++)
            for (u32 k = 0; k < N; k++)
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
}

static void _mm_tn(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 k = 0; k < N; k++)
        for (u32 i = 0; i < M; i++)
            for (u32 j = 0; j < P; j++)
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
}

static void _mm_tt(Tensor *out, const Tensor *a, const Tensor *b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 j = 0; j < P; j++)
        for (u32 k = 0; k < N; k++)
            for (u32 i = 0; i < M; i++)
                (*out)(i, j) += (*a)(i, k) * (*b)(k, j);
}

static void _mat_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    b32 ta = a->stride[ROW_DIM(a)] < a->stride[COL_DIM(a)];
    b32 tb = b->stride[ROW_DIM(b)] < b->stride[COL_DIM(b)];
    switch ((ta << 1) | tb) {
    case 0b00:
        _mm_nn(out, a, b);
        break;
    case 0b01:
        _mm_nt(out, a, b);
        break;
    case 0b10:
        _mm_tn(out, a, b);
        break;
    case 0b11:
        _mm_tt(out, a, b);
        break;
    }
}

b32 tensor_cpu_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                       b32 clear_out) {
    if (a->ndim != 2 || b->ndim != 2) {
        printf("tensor_mat_mul: only 2-D tensors supported\n");
        return false;
    }
    if ((mat_cols(a) != mat_rows(b)) || (mat_rows(a) != mat_rows(out)) ||
        (mat_cols(b) != mat_cols(out))) {
        return false;
    }
    if (clear_out)
        tensor_cpu_clear(out);
    _mat_mul(out, a, b);
    return true;
}

// ---- reduction (sum) -----------------------------------------------------

b32 tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
    if (out->size != 1) {
        printf("tensor_sum: out must be a scalar tensor (size=1)\n");
        return false;
    }
    if (clear_out)
        tensor_cpu_clear(out);
    tensorIterator it(tensor->ndim, tensor->shape, tensor->stride);
    while (it.has_next())
        out->data[0] += tensor->data[it.next()];
    return true;
}

b32 tensor_cpu_sum_dim(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
                       b32 clear_out) {
    if (dim >= tensor->ndim) {
        printf("tensor_sum: dim %u out of range (ndim=%u)\n", dim,
               tensor->ndim);
        return false;
    }
    if (clear_out)
        tensor_cpu_clear(out);

    // Stride trick: map every element along `dim` to the same output slot
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
