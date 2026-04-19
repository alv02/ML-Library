#include "../../include/backend/tensor_cpu.hpp"
#include "../../include/tensor_iterator.hpp"
#include <cstdio>
#include <cstring>
#include <random>

// ---- copy ----------------------------------------------------------------

void tensor_cpu_copy(Tensor *dst, const Tensor *src) {
    if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        memcpy(dst->data, src->data, src->size * sizeof(f32));
        return;
    }
    tensorIterator dst_iter(dst->ndim, dst->shape, dst->stride);
    tensorIterator src_iter(src->ndim, src->shape, src->stride);
    for (u64 i = 0; i < dst->size; i++)
        dst->data[dst_iter.next()] = src->data[src_iter.next()];
}

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(Tensor *tensor, f32 value) {
    for (u64 i = 0; i < tensor->size; i++)
        tensor->data[i] = value;
}

void tensor_cpu_clear(Tensor *tensor) {
    memset(tensor->data, 0, sizeof(f32) * tensor->size);
}

// ---- activations (relu, exp) ---------------------------------------------

// Applies fn element-wise: out[i] = fn(a[i]).
// Fast path when both tensors are contiguous (direct flat index). Falls back to
// two tensorIterators when either tensor is non-contiguous so strides are
// respected for both reads and writes.
template <typename Fn>
static void elementwise_unary(Tensor *out, const Tensor *a, Fn fn) {
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        for (u64 i = 0; i < out->size; i++)
            out->data[i] = fn(a->data[i]);
        return;
    }
    tensorIterator out_iter(out->ndim, out->shape, out->stride);
    tensorIterator a_iter(a->ndim, a->shape, a->stride);
    for (u64 i = 0; i < out->size; i++)
        out->data[out_iter.next()] = fn(a->data[a_iter.next()]);
}

void tensor_cpu_relu(Tensor *dst, const Tensor *src) {
    elementwise_unary(dst, src, [](f32 x) { return x > 0.0f ? x : 0.0f; });
}

void tensor_cpu_exp(Tensor *dst, const Tensor *src) {
    elementwise_unary(dst, src, [](f32 x) { return std::exp(x); });
}

void tensor_cpu_log(Tensor *dst, const Tensor *src) {
    elementwise_unary(dst, src, [](f32 x) { return std::log(x); });
}

// ---- elementwise binary (add / sub / mul / div) --------------------------
// Validation is done by the dispatcher in tensor.cpp before calling here.

// Applies fn element-wise with broadcasting: out[i] = fn(a[...], b[...]).
// Fast path when all three tensors have the same shape and are contiguous.
// General path: three iterators walk the output shape using each tensor's own
// strides (0 on broadcast dims for inputs), so reads from a/b and writes to out
// all respect non-contiguous layouts (e.g. transposed weight matrices).
template <typename Fn>
static void elementwise_binary(Tensor *out, const Tensor *a, const Tensor *b,
                               Fn fn) {
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(out) &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        for (u64 i = 0; i < out->size; i++)
            out->data[i] = fn(a->data[i], b->data[i]);
        return;
    }
    u64 a_strides[MAX_NDIM];
    u64 b_strides[MAX_NDIM];
    expanded_stride(a, out->ndim, a_strides);
    expanded_stride(b, out->ndim, b_strides);

    tensorIterator out_iter(out->ndim, out->shape, out->stride);
    tensorIterator a_iter(out->ndim, out->shape, a_strides);
    tensorIterator b_iter(out->ndim, out->shape, b_strides);

    for (u64 i = 0; i < out->size; i++)
        out->data[out_iter.next()] =
            fn(a->data[a_iter.next()], b->data[b_iter.next()]);
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

void tensor_cpu_equal(Tensor *out, const Tensor *a, const Tensor *b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x == y; });
}

void tensor_cpu_relu_backward(Tensor *out, const Tensor *grad,
                              const Tensor *in) {
    elementwise_binary(out, grad, in,
                       [](f32 g, f32 x) { return x > 0.0f ? g : 0.0f; });
}

// ---- scalar operations ---------------------------------------------------

template <typename Fn>
static void elementwise_scalar(Tensor *out, const Tensor *a, f32 scalar,
                               Fn fn) {
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        for (u64 i = 0; i < out->size; i++)
            out->data[i] = fn(a->data[i], scalar);
        return;
    }
    tensorIterator out_iter(out->ndim, out->shape, out->stride);
    tensorIterator a_iter(a->ndim, a->shape, a->stride);
    for (u64 i = 0; i < out->size; i++)
        out->data[out_iter.next()] = fn(a->data[a_iter.next()], scalar);
}

void tensor_cpu_add(Tensor *out, const Tensor *a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x + s; });
}

void tensor_cpu_sub(Tensor *out, const Tensor *a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x - s; });
}

void tensor_cpu_mul(Tensor *out, const Tensor *tensor, f32 scalar) {
    elementwise_scalar(out, tensor, scalar, [](f32 x, f32 s) { return x * s; });
}

void tensor_cpu_div(Tensor *out, const Tensor *a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x / s; });
}

// ---- matrix multiply -----------------------------------------------------

static inline u32 mat_rows(const Tensor *t) { return t->shape[ROW_DIM(t)]; }
static inline u32 mat_cols(const Tensor *t) { return t->shape[COL_DIM(t)]; }

// The four loop orderings below all compute the same matmul out = a @ b but
// iterate in different orders to maximise cache locality depending on whether
// each matrix is row-major or column-major in memory.
//
// In a row-major matrix stride[row] > stride[col], so walking along columns is
// sequential in memory (cache-friendly). In a column-major (transposed) matrix
// stride[row] < stride[col], so walking along rows is sequential instead.
// The outermost loop should iterate over the dimension that is NOT sequential
// in memory so the inner loops can stream through cache lines.
//
//   nn: a row-major, b row-major — ikj order (streams b's rows in the k loop)
//   nt: a row-major, b col-major — ijk order (streams b's columns in the k
//   loop) tn: a col-major, b row-major — kij order (streams a's columns in the
//   i loop) tt: a col-major, b col-major — jki order (streams both col-major
//   matrices)

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

// Selects the loop ordering based on each matrix's memory layout.
// stride[ROW] < stride[COL] means the matrix is column-major (transposed):
// stepping along rows is sequential, stepping along cols is strided.
// The two booleans are packed into a 2-bit key: ta=bit1, tb=bit0.
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

// ---- reduction (sum, max, argmax) ----------------------------------------

void tensor_cpu_sum(Tensor *out, const Tensor *tensor, b32 clear_out) {
    // Kahan compensated summation
    f32 sum = clear_out ? 0.0f : out->data[0];
    f32 c = 0.0f;
    for (u64 i = 0; i < tensor->size; i++) {
        f32 y = tensor->data[i] - c;
        f32 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out->data[0] = sum;
}

// Stride=0 trick: copy out's strides then zero the target dim. The out iterator
// now maps every input position along that dim to the same output slot, so
// all values along the dim accumulate into a single output element.
// Example: tensor=[3,4], dim=1 → out_strides[1]=0 collapses all 4 columns into
// the same slot for each row, summing them.
void tensor_cpu_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out->stride, out->ndim * sizeof(u64));
    out_strides[dim] = 0;

    tensorIterator in_it(tensor->ndim, tensor->shape, tensor->stride);
    tensorIterator out_it(tensor->ndim, tensor->shape, out_strides);
    while (in_it.has_next())
        out->data[out_it.next()] += tensor->data[in_it.next()];
}

void tensor_cpu_max(Tensor *out, const Tensor *tensor) {
    f32 max_val = -__FLT_MAX__;
    for (u64 i = 0; i < tensor->size; i++)
        max_val = std::max(max_val, tensor->data[i]);
    out->data[0] = max_val;
}

void tensor_cpu_max(Tensor *out, const Tensor *tensor, u32 dim) {
    tensor_cpu_fill(out, -__FLT_MAX__);

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out->stride, out->ndim * sizeof(u64));
    out_strides[dim] = 0;

    tensorIterator in_it(tensor->ndim, tensor->shape, tensor->stride);
    tensorIterator out_it(tensor->ndim, tensor->shape, out_strides);
    while (in_it.has_next()) {
        u64 out_idx = out_it.next();
        f32 val = tensor->data[in_it.next()];
        if (val > out->data[out_idx])
            out->data[out_idx] = val;
    }
}

// Uses three parallel iterators running over the full input shape:
//   in_it  — walks input with real strides (reads values).
//   out_it — same shape but out_strides[dim]=0, collapses dim so all positions
//            along it map to the same output slot (same stride=0 trick as sum).
//   dim_it — all strides 0 except dim_strides[dim]=1, so its offset counts
//   0,1,2,...
//            as the iterator advances along that axis and 0 everywhere else.
//            This gives the current index within the reduction dimension.
// Whenever a new max is found the corresponding dim_it offset is stored as the
// index.
void tensor_cpu_argmax(Tensor *out, const Tensor *tensor, u32 dim) {
    Tensor *max_vals = new Tensor(out->ndim, out->shape, false);
    tensor_cpu_fill(max_vals, -__FLT_MAX__);
    tensor_cpu_fill(out, 0.0f);

    // out_strides: stride=0 on dim → collapses it, maps each input to its slot
    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out->stride, out->ndim * sizeof(u64));
    out_strides[dim] = 0;

    // dim_strides: stride=1 on dim, 0 elsewhere → yields index along dim
    u64 dim_strides[MAX_NDIM] = {};
    dim_strides[dim] = 1;

    tensorIterator in_it(tensor->ndim, tensor->shape, tensor->stride);
    tensorIterator out_it(tensor->ndim, tensor->shape, out_strides);
    tensorIterator dim_it(tensor->ndim, tensor->shape, dim_strides);

    while (in_it.has_next()) {
        u64 out_idx = out_it.next();
        u64 dim_idx = dim_it.next();
        f32 val = tensor->data[in_it.next()];
        if (val > max_vals->data[out_idx]) {
            max_vals->data[out_idx] = val;
            out->data[out_idx] = (f32)dim_idx;
        }
    }

    delete max_vals;
}

// ---- scattering ----------------------------------------------------------
void tensor_cpu_scatter_add(Tensor *out, const Tensor *src,
                            const Tensor *indices, u32 dim) {
    // out_strides: collapse dim to 0, gives the "base" flat offset ignoring dim
    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out->stride, out->ndim * sizeof(u64));
    out_strides[dim] = 0;
    u64 out_stride_dim = out->stride[dim];

    tensorIterator src_it(src->ndim, src->shape, src->stride);
    tensorIterator idx_it(indices->ndim, indices->shape, indices->stride);
    tensorIterator out_it(src->ndim, src->shape,
                          out_strides); // walks src shape

    while (src_it.has_next()) {
        u64 src_off = src_it.next();
        u64 idx_off = idx_it.next();
        u64 out_base = out_it.next();
        u32 k = (u32)indices->data[idx_off]; // index along dim
        out->data[out_base + k * out_stride_dim] += src->data[src_off];
    }
}

// ---- initializing --------------------------------------------------------

void tensor_cpu_he_init(Tensor *tensor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    u32 in_features = tensor->shape[ROW_DIM(tensor)];
    float stddev = std::sqrt(2.0f / in_features);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (u64 i = 0; i < tensor->size; i++)
        tensor->data[i] = dist(gen);
}

// ---- indexing ------------------------------------------------------------

// Decomposes the tensor into outer slices, index slices, and inner slices:
//   inner_size = src->stride[dim]  → number of elements in one slice along dim
//   outer_size = total / (shape[dim] * inner_size) → number of outer batches
// For each outer batch and each requested index, copies one inner_size block
// from the source to the destination via memcpy.
// Example: src=[5,3], indices=[2,0], dim=0 → inner_size=3, outer_size=1
//   copies row 2 (src+6) then row 0 (src+0) into dst.
void tensor_cpu_index_select(Tensor *dst, const Tensor *src, const u32 *indices,
                             u32 n_indices, u32 dim) {
    u64 inner_size = src->stride[dim];
    u64 outer_size = src->size / (src->shape[dim] * inner_size);

    for (u64 o = 0; o < outer_size; o++) {
        for (u32 n = 0; n < n_indices; n++) {
            f32 *dst_ptr = dst->data + (o * n_indices + n) * inner_size;
            const f32 *src_ptr =
                src->data + (o * src->shape[dim] + indices[n]) * inner_size;
            memcpy(dst_ptr, src_ptr, inner_size * sizeof(f32));
        }
    }
}

// ---- comparison ----------------------------------------------------------

b32 tensor_cpu_equals(const Tensor *a, const Tensor *b, f32 tol) {
    tensorIterator a_iter(a->ndim, a->shape, a->stride);
    tensorIterator b_iter(b->ndim, b->shape, b->stride);
    for (u64 i = 0; i < a->size; i++)
        if (fabsf(a->data[a_iter.next()] - b->data[b_iter.next()]) > tol)
            return false;
    return true;
}

// ---- spatial / patch operations ------------------------------------------

void tensor_cpu_unfold2d(Tensor *dst, const Tensor *src,
                         Unfold2dParams params) {
    u32 N = src->shape[0];
    u32 C = src->shape[1];
    u32 H = src->shape[2];
    u32 W = src->shape[3];
    u32 kH = params.k_h;
    u32 kW = params.k_w;
    params.compute_output_size(H, W);
    u32 L = params.L_h * params.L_w;

    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w, C, kH, kW};
    tensor_reshape(dst, shape6, 6);

    for (u32 n = 0; n < N; n++)
        for (u32 lh = 0; lh < params.L_h; lh++)
            for (u32 lw = 0; lw < params.L_w; lw++)
                for (u32 c = 0; c < C; c++)
                    for (u32 kh = 0; kh < kH; kh++)
                        for (u32 kw = 0; kw < kW; kw++) {
                            i32 h = (i32)(lh * params.stride_h + kh) -
                                    (i32)params.pad_h;
                            i32 w = (i32)(lw * params.stride_w + kw) -
                                    (i32)params.pad_w;
                            u64 dst_off = (u64)n * dst->stride[0] +
                                          (u64)lh * dst->stride[1] +
                                          (u64)lw * dst->stride[2] +
                                          (u64)c * dst->stride[3] +
                                          (u64)kh * dst->stride[4] +
                                          (u64)kw * dst->stride[5];
                            if (h < 0 || (u32)h >= H || w < 0 || (u32)w >= W) {
                                dst->data[dst_off] = params.pad_constant;
                            } else {
                                u64 src_off = (u64)n * src->stride[0] +
                                              (u64)c * src->stride[1] +
                                              (u64)h * src->stride[2] +
                                              (u64)w * src->stride[3];
                                dst->data[dst_off] = src->data[src_off];
                            }
                        }

    u32 shape3[MAX_NDIM] = {N, L, C * kH * kW};
    tensor_reshape(dst, shape3, 3);
}

void tensor_cpu_fold2d(Tensor *dst, const Tensor *col, Unfold2dParams params) {
    u32 N = dst->shape[0];
    u32 C = dst->shape[1];
    u32 H = dst->shape[2];
    u32 W = dst->shape[3];
    u32 kH = params.k_h;
    u32 kW = params.k_w;
    params.compute_output_size(H, W);

    // reshape col to 6D so strides match the unfold layout: [N, L_h, L_w, C,
    // kH, kW]
    Tensor *col6 = tensor_view(col);
    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w, C, kH, kW};
    tensor_reshape(col6, shape6, 6);

    for (u32 n = 0; n < N; n++)
        for (u32 lh = 0; lh < params.L_h; lh++)
            for (u32 lw = 0; lw < params.L_w; lw++)
                for (u32 c = 0; c < C; c++)
                    for (u32 kh = 0; kh < kH; kh++)
                        for (u32 kw = 0; kw < kW; kw++) {
                            i32 h = (i32)(lh * params.stride_h + kh) -
                                    (i32)params.pad_h;
                            i32 w = (i32)(lw * params.stride_w + kw) -
                                    (i32)params.pad_w;
                            if (h < 0 || (u32)h >= H || w < 0 || (u32)w >= W)
                                continue;
                            u64 col_off = (u64)n * col6->stride[0] +
                                          (u64)lh * col6->stride[1] +
                                          (u64)lw * col6->stride[2] +
                                          (u64)c * col6->stride[3] +
                                          (u64)kh * col6->stride[4] +
                                          (u64)kw * col6->stride[5];
                            u64 dst_off = (u64)n * dst->stride[0] +
                                          (u64)c * dst->stride[1] +
                                          (u64)h * dst->stride[2] +
                                          (u64)w * dst->stride[3];
                            dst->data[dst_off] += col6->data[col_off];
                        }

    delete col6;
}
