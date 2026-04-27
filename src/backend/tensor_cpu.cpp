#include "../../include/backend/tensor_cpu.hpp"
#include "../../include/tensor_iterator.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---- copy ----------------------------------------------------------------

void tensor_cpu_copy(TensorImpl &dst, const TensorImpl &src) {
    if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        memcpy(dst.data(), src.data(), src.numel() * sizeof(f32));
        return;
    }
    tensorIterator dst_it(dst.ndim, dst.shape, dst.stride);
    tensorIterator src_it(src.ndim, src.shape, src.stride);
    while (src_it.has_next())
        dst.data()[dst_it.next()] = src.data()[src_it.next()];
}

void tensor_cpu_contigous(TensorImpl &t) {
    Tensor temp_t = Tensor::make(t.ndim, t.shape, t.on_gpu());
    TensorImpl &temp = temp_t.impl();

    tensorIterator src_iter(t.ndim, t.shape, t.stride);
    for (u64 i = 0; i < temp.numel(); i++)
        temp.data()[i] = t.data()[src_iter.next()];

    memcpy(t.data(), temp.data(), t.numel() * sizeof(f32));
    tensor_compute_strides(t.stride, t.shape, t.ndim);
}

// ---- fill / clear --------------------------------------------------------

void tensor_cpu_fill(TensorImpl &tensor, f32 value) {
    for (u64 i = 0; i < tensor.numel(); i++)
        tensor.data()[i] = value;
}

void tensor_cpu_clear(TensorImpl &tensor) {
    memset(tensor.data(), 0, sizeof(f32) * tensor.numel());
}

// ---- activations (relu, exp) ---------------------------------------------

template <typename Fn>
static void elementwise_unary(TensorImpl &out, const TensorImpl &a, Fn fn) {
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        for (u64 i = 0; i < out.numel(); i++)
            out.data()[i] = fn(a.data()[i]);
        return;
    }
    tensorIterator out_iter(out.ndim, out.shape, out.stride);
    tensorIterator a_iter(a.ndim, a.shape, a.stride);
    for (u64 i = 0; i < out.numel(); i++)
        out.data()[out_iter.next()] = fn(a.data()[a_iter.next()]);
}

void tensor_cpu_relu(TensorImpl &dst, const TensorImpl &src) {
    elementwise_unary(dst, src, [](f32 x) { return x > 0.0f ? x : 0.0f; });
}

void tensor_cpu_exp(TensorImpl &dst, const TensorImpl &src) {
    elementwise_unary(dst, src, [](f32 x) { return std::exp(x); });
}

void tensor_cpu_log(TensorImpl &dst, const TensorImpl &src) {
    elementwise_unary(dst, src, [](f32 x) { return std::log(x); });
}

void tensor_cpu_sqrt(TensorImpl &dst, const TensorImpl &src) {
    elementwise_unary(dst, src, [](f32 x) { return std::sqrt(x); });
}

// ---- elementwise binary (add / sub / mul / div) --------------------------

template <typename Fn>
static void elementwise_binary(TensorImpl &out, const TensorImpl &a,
                               const TensorImpl &b, Fn fn) {
    if (tensor_shape_eq(a, b) && tensor_is_contiguous(out) &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b)) {
        for (u64 i = 0; i < out.numel(); i++)
            out.data()[i] = fn(a.data()[i], b.data()[i]);
        return;
    }
    u64 a_strides[MAX_NDIM];
    u64 b_strides[MAX_NDIM];
    expanded_stride(a, out.ndim, a_strides);
    expanded_stride(b, out.ndim, b_strides);

    tensorIterator out_iter(out.ndim, out.shape, out.stride);
    tensorIterator a_iter(out.ndim, out.shape, a_strides);
    tensorIterator b_iter(out.ndim, out.shape, b_strides);

    for (u64 i = 0; i < out.numel(); i++)
        out.data()[out_iter.next()] =
            fn(a.data()[a_iter.next()], b.data()[b_iter.next()]);
}

void tensor_cpu_add(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x + y; });
}

void tensor_cpu_sub(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x - y; });
}

void tensor_cpu_mul(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x * y; });
}

void tensor_cpu_div(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x / y; });
}

void tensor_cpu_equal(TensorImpl &out, const TensorImpl &a,
                      const TensorImpl &b) {
    elementwise_binary(out, a, b, [](f32 x, f32 y) { return x == y; });
}

void tensor_cpu_relu_backward(TensorImpl &out, const TensorImpl &grad,
                              const TensorImpl &in) {
    elementwise_binary(out, grad, in,
                       [](f32 g, f32 x) { return x > 0.0f ? g : 0.0f; });
}

// ---- scalar operations ---------------------------------------------------

template <typename Fn>
static void elementwise_scalar(TensorImpl &out, const TensorImpl &a, f32 scalar,
                               Fn fn) {
    if (tensor_is_contiguous(out) && tensor_is_contiguous(a)) {
        for (u64 i = 0; i < out.numel(); i++)
            out.data()[i] = fn(a.data()[i], scalar);
        return;
    }
    tensorIterator out_iter(out.ndim, out.shape, out.stride);
    tensorIterator a_iter(a.ndim, a.shape, a.stride);
    for (u64 i = 0; i < out.numel(); i++)
        out.data()[out_iter.next()] = fn(a.data()[a_iter.next()], scalar);
}

void tensor_cpu_add(TensorImpl &out, const TensorImpl &a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x + s; });
}

void tensor_cpu_sub(TensorImpl &out, const TensorImpl &a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x - s; });
}

void tensor_cpu_mul(TensorImpl &out, const TensorImpl &tensor, f32 scalar) {
    elementwise_scalar(out, tensor, scalar, [](f32 x, f32 s) { return x * s; });
}

void tensor_cpu_div(TensorImpl &out, const TensorImpl &a, f32 scalar) {
    elementwise_scalar(out, a, scalar, [](f32 x, f32 s) { return x / s; });
}

// ---- matrix multiply -----------------------------------------------------

static inline u32 mat_rows(const TensorImpl &t) { return t.shape[ROW_DIM(t)]; }
static inline u32 mat_cols(const TensorImpl &t) { return t.shape[COL_DIM(t)]; }

static void _mm_nn(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 k = 0; k < N; k++)
            for (u32 j = 0; j < P; j++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_nt(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < P; j++)
            for (u32 k = 0; k < N; k++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_tn(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 k = 0; k < N; k++)
        for (u32 i = 0; i < M; i++)
            for (u32 j = 0; j < P; j++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_tt(TensorImpl &out, const TensorImpl &a, const TensorImpl &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 j = 0; j < P; j++)
        for (u32 k = 0; k < N; k++)
            for (u32 i = 0; i < M; i++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mat_mul(TensorImpl &out, const TensorImpl &a,
                     const TensorImpl &b) {
    b32 ta = a.stride[ROW_DIM(a)] < a.stride[COL_DIM(a)];
    b32 tb = b.stride[ROW_DIM(b)] < b.stride[COL_DIM(b)];
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

void tensor_cpu_mat_mul(TensorImpl &out, const TensorImpl &a,
                        const TensorImpl &b, b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);
    _mat_mul(out, a, b);
}

// ---- reduction (sum, max, argmax) ----------------------------------------

void tensor_cpu_sum(TensorImpl &out, const TensorImpl &tensor, b32 clear_out) {
    f32 sum = clear_out ? 0.0f : out.data()[0];
    f32 c = 0.0f;
    for (u64 i = 0; i < tensor.numel(); i++) {
        f32 y = tensor.data()[i] - c;
        f32 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out.data()[0] = sum;
}

void tensor_cpu_sum(TensorImpl &out, const TensorImpl &tensor, u32 dim,
                    b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out.stride, out.ndim * sizeof(u64));
    out_strides[dim] = 0;

    tensorIterator in_it(tensor.ndim, tensor.shape, tensor.stride);
    tensorIterator out_it(tensor.ndim, tensor.shape, out_strides);
    while (in_it.has_next())
        out.data()[out_it.next()] += tensor.data()[in_it.next()];
}

void tensor_cpu_max(TensorImpl &out, const TensorImpl &tensor) {
    f32 max_val = -__FLT_MAX__;
    for (u64 i = 0; i < tensor.numel(); i++)
        max_val = std::max(max_val, tensor.data()[i]);
    out.data()[0] = max_val;
}

void tensor_cpu_max(TensorImpl &out, const TensorImpl &tensor, u32 dim) {
    tensor_cpu_fill(out, -__FLT_MAX__);

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out.stride, out.ndim * sizeof(u64));
    out_strides[dim] = 0;

    tensorIterator in_it(tensor.ndim, tensor.shape, tensor.stride);
    tensorIterator out_it(tensor.ndim, tensor.shape, out_strides);
    while (in_it.has_next()) {
        u64 out_idx = out_it.next();
        f32 val = tensor.data()[in_it.next()];
        if (val > out.data()[out_idx])
            out.data()[out_idx] = val;
    }
}

void tensor_cpu_argmax(TensorImpl &out, const TensorImpl &tensor, u32 dim) {
    Tensor max_vals_t = Tensor::make(out.ndim, out.shape, false);
    TensorImpl &max_vals = max_vals_t.impl();
    tensor_cpu_fill(max_vals, -__FLT_MAX__);
    tensor_cpu_fill(out, 0.0f);

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out.stride, out.ndim * sizeof(u64));
    out_strides[dim] = 0;

    u64 dim_strides[MAX_NDIM] = {};
    dim_strides[dim] = 1;

    tensorIterator in_it(tensor.ndim, tensor.shape, tensor.stride);
    tensorIterator out_it(tensor.ndim, tensor.shape, out_strides);
    tensorIterator dim_it(tensor.ndim, tensor.shape, dim_strides);

    while (in_it.has_next()) {
        u64 out_idx = out_it.next();
        u64 dim_idx = dim_it.next();
        f32 val = tensor.data()[in_it.next()];
        if (val > max_vals.data()[out_idx]) {
            max_vals.data()[out_idx] = val;
            out.data()[out_idx] = (f32)dim_idx;
        }
    }
}

// ---- welford mean+var ----------------------------------------------------

void tensor_cpu_welford_mean_var(TensorImpl &mean, TensorImpl &var,
                                 const TensorImpl &src, u32 dim) {
    u32 C = src.shape[dim];

    std::vector<f32> mu(C, 0.0f), M2(C, 0.0f);
    std::vector<u32> n(C, 0);

    u64 ch_strides[MAX_NDIM] = {};
    ch_strides[dim] = 1;

    tensorIterator in_it(src.ndim, src.shape, src.stride);
    tensorIterator ch_it(src.ndim, src.shape, ch_strides);

    while (in_it.has_next()) {
        u32 c = (u32)ch_it.next();
        f32 x = src.data()[in_it.next()];
        n[c]++;
        f32 delta = x - mu[c];
        mu[c] += delta / (f32)n[c];
        M2[c] += delta * (x - mu[c]);
    }

    for (u32 c = 0; c < C; c++) {
        mean.data()[c] = mu[c];
        var.data()[c] = M2[c] / (f32)(n[c] - 1);
    }
}

// ---- scattering ----------------------------------------------------------

void tensor_cpu_scatter_add(TensorImpl &out, const TensorImpl &src,
                            const TensorImpl &indices, u32 dim) {
    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out.stride, out.ndim * sizeof(u64));
    out_strides[dim] = 0;
    u64 out_stride_dim = out.stride[dim];

    tensorIterator src_it(src.ndim, src.shape, src.stride);
    tensorIterator idx_it(indices.ndim, indices.shape, indices.stride);
    tensorIterator out_it(src.ndim, src.shape, out_strides);

    while (src_it.has_next()) {
        u64 src_off = src_it.next();
        u64 idx_off = idx_it.next();
        u64 out_base = out_it.next();
        u32 k = (u32)indices.data()[idx_off];
        out.data()[out_base + k * out_stride_dim] += src.data()[src_off];
    }
}

// ---- initializing --------------------------------------------------------

void tensor_cpu_he_init(TensorImpl &tensor) {
    std::random_device rd;
    std::mt19937 gen(rd());
    u32 in_features = tensor.shape[ROW_DIM(tensor)];
    float stddev = std::sqrt(2.0f / in_features);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (u64 i = 0; i < tensor.numel(); i++)
        tensor.data()[i] = dist(gen);
}

// ---- indexing ------------------------------------------------------------

void tensor_cpu_index_select(TensorImpl &dst, const TensorImpl &src,
                             const u32 *indices, u32 n_indices, u32 dim) {
    u64 inner_size = src.stride[dim];
    u64 outer_size = src.numel() / (src.shape[dim] * inner_size);

    for (u64 o = 0; o < outer_size; o++) {
        for (u32 n = 0; n < n_indices; n++) {
            f32 *dst_ptr = dst.data() + (o * n_indices + n) * inner_size;
            const f32 *src_ptr =
                src.data() + (o * src.shape[dim] + indices[n]) * inner_size;
            memcpy(dst_ptr, src_ptr, inner_size * sizeof(f32));
        }
    }
}

// ---- comparison ----------------------------------------------------------

b32 tensor_cpu_equals(const TensorImpl &a, const TensorImpl &b, f32 tol) {
    tensorIterator a_iter(a.ndim, a.shape, a.stride);
    tensorIterator b_iter(b.ndim, b.shape, b.stride);
    for (u64 i = 0; i < a.numel(); i++)
        if (fabsf(a.data()[a_iter.next()] - b.data()[b_iter.next()]) > tol)
            return false;
    return true;
}

// ---- spatial / patch operations ------------------------------------------

void tensor_cpu_unfold2d(TensorImpl &dst, const TensorImpl &src,
                         Unfold2dParams params) {
    u32 N = src.shape[0];
    u32 C = src.shape[1];
    u32 H = src.shape[2];
    u32 W = src.shape[3];
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
                            u64 dst_off = (u64)n * dst.stride[0] +
                                          (u64)lh * dst.stride[1] +
                                          (u64)lw * dst.stride[2] +
                                          (u64)c * dst.stride[3] +
                                          (u64)kh * dst.stride[4] +
                                          (u64)kw * dst.stride[5];
                            if (h < 0 || (u32)h >= H || w < 0 || (u32)w >= W) {
                                dst.data()[dst_off] = params.pad_constant;
                            } else {
                                u64 src_off = (u64)n * src.stride[0] +
                                              (u64)c * src.stride[1] +
                                              (u64)h * src.stride[2] +
                                              (u64)w * src.stride[3];
                                dst.data()[dst_off] = src.data()[src_off];
                            }
                        }

    u32 shape3[MAX_NDIM] = {N, L, C * kH * kW};
    tensor_reshape(dst, shape3, 3);
}

void tensor_cpu_fold2d(TensorImpl &dst, const TensorImpl &col,
                       Unfold2dParams params) {
    u32 N = dst.shape[0];
    u32 C = dst.shape[1];
    u32 H = dst.shape[2];
    u32 W = dst.shape[3];
    u32 kH = params.k_h;
    u32 kW = params.k_w;
    params.compute_output_size(H, W);

    // Compute 6D contiguous strides for col without creating a view tensor
    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w, C, kH, kW};
    u64 stride6[MAX_NDIM];
    tensor_compute_strides(stride6, shape6, 6);

    const f32 *col_data = col.data();

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
                            u64 col_off =
                                (u64)n * stride6[0] + (u64)lh * stride6[1] +
                                (u64)lw * stride6[2] + (u64)c * stride6[3] +
                                (u64)kh * stride6[4] + (u64)kw * stride6[5];
                            u64 dst_off = (u64)n * dst.stride[0] +
                                          (u64)c * dst.stride[1] +
                                          (u64)h * dst.stride[2] +
                                          (u64)w * dst.stride[3];
                            dst.data()[dst_off] += col_data[col_off];
                        }
}
