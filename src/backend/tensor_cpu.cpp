#include "../../include/backend/tensor_cpu.hpp"
#include "../../include/tensor_iterator.hpp"
#include <cstdio>
#include <cstring>

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(Tensor *tensor, f32 value) {
    for (u64 i = 0; i < tensor->size; i++)
        tensor->data[i] = value;
}

void tensor_cpu_clear(Tensor *tensor) {
    memset(tensor->data, 0, sizeof(f32) * tensor->size);
}

// ---- elementwise binary (add / sub / mul / div) --------------------------
// Validation is done by the dispatcher in tensor.cpp before calling here.

template <typename Fn>
static void elementwise_binary(Tensor *out, const Tensor *a, const Tensor *b,
                               Fn fn) {
    // Fast path: same shape, contiguous memory — no broadcast needed
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(a) &&
        tensor_is_contiguous(b)) {
        for (u64 i = 0; i < out->size; i++)
            out->data[i] = fn(a->data[i], b->data[i]);
        return;
    }

    // Broadcast path: use out->shape/ndim (already validated by dispatcher)
    u64 a_strides[MAX_NDIM];
    u64 b_strides[MAX_NDIM];
    expanded_stride(a, out->shape, out->ndim, a_strides);
    expanded_stride(b, out->shape, out->ndim, b_strides);

    tensorIterator a_iter(out->ndim, out->shape, a_strides);
    tensorIterator b_iter(out->ndim, out->shape, b_strides);

    for (u64 i = 0; i < out->size; i++)
        out->data[i] = fn(a->data[a_iter.next()], b->data[b_iter.next()]);

    return;
}

void tensor_cpu_add(Tensor *out, const Tensor *a, const Tensor *b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x + y; });
}

void tensor_cpu_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x - y; });
}

void tensor_cpu_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x * y; });
}

void tensor_cpu_div(Tensor *out, const Tensor *a, const Tensor *b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x / y; });
}

// ---- scalar operations ---------------------------------------------------

void tensor_cpu_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    for (u64 i = 0; i < tensor->size; i++)
        out->data[i] = tensor->data[i] * scalar;
}

void tensor_cpu_div(Tensor *out, const Tensor *a, f32 scalar) {
    for (u64 i = 0; i < out->size; i++)
        out->data[i] = a->data[i] / scalar;
}

void tensor_cpu_add(Tensor *out, const Tensor *a, f32 scalar) {
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

void tensor_cpu_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                        b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);
    _mat_mul(out, a, b);
}

// ---- reduction (sum) -----------------------------------------------------

void tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);
    for (u64 i = 0; i < tensor->size; i++)
        out->data[0] += tensor->data[i];
}

void tensor_cpu_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim,
                    b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);

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
}
