#include "../../include/backend/tensor_cpu.hpp"
#include "../../include/tensor_iterator.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// ---- copy ----------------------------------------------------------------

template <typename T>
void tensor_cpu_copy(TensorImpl<T> &dst, const TensorImpl<T> &src) {
    if (tensor_is_contiguous(dst) && tensor_is_contiguous(src)) {
        memcpy(dst.data(), src.data(), src.numel() * sizeof(T));
        return;
    }
    tensorIterator dst_it(dst.ndim, dst.shape, dst.stride);
    tensorIterator src_it(src.ndim, src.shape, src.stride);
    while (src_it.has_next())
        dst.data()[dst_it.next()] = src.data()[src_it.next()];
}

template <typename T> void tensor_cpu_contigous(TensorImpl<T> &t) {
    Tensor<T> temp_t = Tensor<T>::make(t.ndim, t.shape, t.on_gpu());
    TensorImpl<T> &temp = temp_t.impl();

    tensorIterator src_iter(t.ndim, t.shape, t.stride);
    for (u64 i = 0; i < temp.numel(); i++)
        temp.data()[i] = t.data()[src_iter.next()];

    memcpy(t.data(), temp.data(), t.numel() * sizeof(T));
    tensor_compute_strides(t.stride, t.shape, t.ndim);
}

// ---- fill / clear --------------------------------------------------------

template <typename T> void tensor_cpu_fill(TensorImpl<T> &tensor, T value) {
    for (u64 i = 0; i < tensor.numel(); i++)
        tensor.data()[i] = value;
}

template <typename T> void tensor_cpu_clear(TensorImpl<T> &tensor) {
    memset(tensor.data(), 0, sizeof(T) * tensor.numel());
}

template <typename T> void tensor_cpu_arange(TensorImpl<T> &out) {
    for (u64 i = 0; i < out.numel(); i++)
        out.data()[i] = (T)i;
}

// ---- activations (relu, exp) — f32 only ---------------------------------

template <typename Fn>
static void elementwise_unary_f32(TensorImpl<f32> &out,
                                  const TensorImpl<f32> &a, Fn fn) {
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

void tensor_cpu_relu(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    elementwise_unary_f32(dst, src, [](f32 x) { return x > 0.0f ? x : 0.0f; });
}

void tensor_cpu_gelu(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    constexpr f32 c = 0.7978845608f;
    constexpr f32 k = 0.044715f;
    elementwise_unary_f32(dst, src, [c, k](f32 x) {
        return 0.5f * x * (1.0f + std::tanh(c * (x + k * x * x * x)));
    });
}

void tensor_cpu_exp(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    elementwise_unary_f32(dst, src, [](f32 x) { return std::exp(x); });
}

void tensor_cpu_log(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    elementwise_unary_f32(dst, src, [](f32 x) { return std::log(x); });
}

void tensor_cpu_sqrt(TensorImpl<f32> &dst, const TensorImpl<f32> &src) {
    elementwise_unary_f32(dst, src, [](f32 x) { return std::sqrt(x); });
}

void tensor_cpu_dropout_mask(TensorImpl<f32> &mask, f32 p) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
    f32 scale = 1.0f / (1.0f - p);
    for (u64 i = 0; i < mask.numel(); i++)
        mask.data()[i] = (dist(rng) >= p) ? scale : 0.0f;
}

// ---- elementwise binary (add / sub / mul / div) --------------------------

template <typename T, typename Fn>
static void elementwise_binary(TensorImpl<T> &out, const TensorImpl<T> &a,
                               const TensorImpl<T> &b, Fn fn) {
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

template <typename T>
void tensor_cpu_add(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b) {
    elementwise_binary(out, a, b, [](T x, T y) { return x + y; });
}

template <typename T>
void tensor_cpu_sub(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b) {
    elementwise_binary(out, a, b, [](T x, T y) { return x - y; });
}

template <typename T>
void tensor_cpu_mul(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b) {
    elementwise_binary(out, a, b, [](T x, T y) { return x * y; });
}

template <typename T>
void tensor_cpu_div(TensorImpl<T> &out, const TensorImpl<T> &a,
                    const TensorImpl<T> &b) {
    elementwise_binary(out, a, b, [](T x, T y) { return x / y; });
}

template <typename T>
void tensor_cpu_equal(TensorImpl<T> &out, const TensorImpl<T> &a,
                      const TensorImpl<T> &b) {
    elementwise_binary(out, a, b,
                       [](T x, T y) -> T { return x == y ? T(1) : T(0); });
}

void tensor_cpu_relu_backward(TensorImpl<f32> &out, const TensorImpl<f32> &grad,
                              const TensorImpl<f32> &in) {
    elementwise_binary(out, grad, in,
                       [](f32 g, f32 x) { return x > 0.0f ? g : 0.0f; });
}

void tensor_cpu_gelu_backward(TensorImpl<f32> &out, const TensorImpl<f32> &grad,
                              const TensorImpl<f32> &input) {
    constexpr f32 c = 0.7978845608f;
    constexpr f32 k = 0.044715f;
    elementwise_binary(out, grad, input, [c, k](f32 g, f32 x) {
        f32 t = std::tanh(c * (x + k * x * x * x));
        f32 dtdx = (1.0f - t * t) * c * (1.0f + 3.0f * k * x * x);
        return g * 0.5f * (1.0f + t + x * dtdx);
    });
}

// ---- scalar operations ---------------------------------------------------

template <typename T, typename Fn>
static void elementwise_scalar(TensorImpl<T> &out, const TensorImpl<T> &a,
                               T scalar, Fn fn) {
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

template <typename T>
void tensor_cpu_add(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar) {
    elementwise_scalar(out, a, scalar, [](T x, T s) { return x + s; });
}

template <typename T>
void tensor_cpu_sub(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar) {
    elementwise_scalar(out, a, scalar, [](T x, T s) { return x - s; });
}

template <typename T>
void tensor_cpu_mul(TensorImpl<T> &out, const TensorImpl<T> &tensor, T scalar) {
    elementwise_scalar(out, tensor, scalar, [](T x, T s) { return x * s; });
}

template <typename T>
void tensor_cpu_div(TensorImpl<T> &out, const TensorImpl<T> &a, T scalar) {
    elementwise_scalar(out, a, scalar, [](T x, T s) { return x / s; });
}

// ---- matrix multiply — f32 only -----------------------------------------

static inline u32 mat_rows(const TensorImpl<f32> &t) {
    return t.shape[ROW_DIM(t)];
}
static inline u32 mat_cols(const TensorImpl<f32> &t) {
    return t.shape[COL_DIM(t)];
}

static void _mm_nn(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                   const TensorImpl<f32> &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 k = 0; k < N; k++)
            for (u32 j = 0; j < P; j++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_nt(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                   const TensorImpl<f32> &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 i = 0; i < M; i++)
        for (u32 j = 0; j < P; j++)
            for (u32 k = 0; k < N; k++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_tn(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                   const TensorImpl<f32> &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 k = 0; k < N; k++)
        for (u32 i = 0; i < M; i++)
            for (u32 j = 0; j < P; j++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mm_tt(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                   const TensorImpl<f32> &b) {
    u32 M = mat_rows(a), N = mat_cols(a), P = mat_cols(b);
    for (u32 j = 0; j < P; j++)
        for (u32 k = 0; k < N; k++)
            for (u32 i = 0; i < M; i++)
                out(i, j) += a(i, k) * b(k, j);
}

static void _mat_mul(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                     const TensorImpl<f32> &b) {
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

void tensor_cpu_mat_mul(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                        const TensorImpl<f32> &b, b32 clear_out) {
    if (clear_out)
        tensor_cpu_clear(out);
    _mat_mul(out, a, b);
}

// TODO: Use the nn, nt, ... functions
void tensor_cpu_mat_mul_batched(TensorImpl<f32> &out, const TensorImpl<f32> &a,
                                const TensorImpl<f32> &b, b32 clear_out) {
    u32 nd = a.ndim;
    u32 M = a.shape[nd - 2], K = a.shape[nd - 1], N = b.shape[nd - 1];
    u32 batch_ndim = nd - 2;
    u64 batch_total = 1;
    for (u32 i = 0; i < batch_ndim; i++)
        batch_total *= a.shape[i];

    u64 sr_a = a.stride[nd - 2], sc_a = a.stride[nd - 1];
    u64 sr_b = b.stride[nd - 2], sc_b = b.stride[nd - 1];
    u64 sr_c = out.stride[nd - 2], sc_c = out.stride[nd - 1];

    if (clear_out)
        tensor_cpu_clear(out);
    for (u64 bat = 0; bat < batch_total; bat++) {
        // Compute per-input offsets using multi-dim batch indices so
        // non-contiguous layouts (e.g. after transpose) and stride-0 broadcast
        // dims both work.
        u64 off_a = 0, off_b = 0, off_c = 0;
        u64 idx = bat;
        for (i32 d = (i32)batch_ndim - 1; d >= 0; d--) {
            u64 coord = idx % a.shape[d];
            idx /= a.shape[d];
            off_a += coord * a.stride[d];
            off_b += coord * b.stride[d];
            off_c += coord * out.stride[d];
        }
        const f32 *ap = a.data() + off_a;
        const f32 *bp = b.data() + off_b;
        f32 *cp = out.data() + off_c;
        for (u32 i = 0; i < M; i++)
            for (u32 j = 0; j < N; j++) {
                f32 s = 0.0f;
                for (u32 k = 0; k < K; k++)
                    s += ap[i * sr_a + k * sc_a] * bp[k * sr_b + j * sc_b];
                cp[i * sr_c + j * sc_c] += s;
            }
    }
}

// ---- reduction (sum, max, argmax) ----------------------------------------

template <typename T>
static void cpu_sum_dim(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                        u32 dim, b32 clear_out) {
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

template <typename T>
void tensor_cpu_sum_global(TensorImpl<T> &out, const TensorImpl<T> &tensor) {
    f64 sum = 0.0, c = 0.0;
    tensorIterator it(tensor.ndim, tensor.shape, tensor.stride);
    while (it.has_next()) {
        f64 y = (f64)tensor.data()[it.next()] - c;
        f64 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    out.data()[0] = (T)sum;
}

template <typename T>
void tensor_cpu_sum_dim(TensorImpl<T> &out, const TensorImpl<T> &tensor,
                        u32 dim) {
    if (out.numel() == 1) {
        tensor_cpu_sum_global(out, tensor);
        return;
    }
    cpu_sum_dim(out, tensor, dim, true);
}

void tensor_cpu_sum_skip(TensorImpl<f32> &out, const TensorImpl<f32> &src,
                         u32 dim) {
    tensor_cpu_clear(out);
    u64 out_strides[MAX_NDIM] = {};
    out_strides[dim] = 1;
    tensorIterator in_it(src.ndim, src.shape, src.stride);
    tensorIterator out_it(src.ndim, src.shape, out_strides);
    while (in_it.has_next())
        out.data()[out_it.next()] += src.data()[in_it.next()];
}

template <typename T>
void tensor_cpu_max(TensorImpl<T> &out, const TensorImpl<T> &tensor) {
    T max_val = std::numeric_limits<T>::lowest();
    for (u64 i = 0; i < tensor.numel(); i++)
        if (tensor.data()[i] > max_val)
            max_val = tensor.data()[i];
    out.data()[0] = max_val;
}

template <typename T>
void tensor_cpu_max(TensorImpl<T> &out, const TensorImpl<T> &tensor, u32 dim) {
    tensor_cpu_fill(out, std::numeric_limits<T>::lowest());

    u64 out_strides[MAX_NDIM];
    memcpy(out_strides, out.stride, out.ndim * sizeof(u64));
    out_strides[dim] = 0;

    tensorIterator in_it(tensor.ndim, tensor.shape, tensor.stride);
    tensorIterator out_it(tensor.ndim, tensor.shape, out_strides);
    while (in_it.has_next()) {
        u64 out_idx = out_it.next();
        T val = tensor.data()[in_it.next()];
        if (val > out.data()[out_idx])
            out.data()[out_idx] = val;
    }
}

template <typename T>
void tensor_cpu_argmax(TensorImpl<u32> &out, const TensorImpl<T> &tensor,
                       u32 dim) {
    Tensor<T> max_vals_t = Tensor<T>::make(out.ndim, out.shape, false);
    TensorImpl<T> &max_vals = max_vals_t.impl();
    tensor_cpu_fill(max_vals, std::numeric_limits<T>::lowest());
    memset(out.data(), 0, out.numel() * sizeof(u32));

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
        T val = tensor.data()[in_it.next()];
        if (val > max_vals.data()[out_idx]) {
            max_vals.data()[out_idx] = val;
            out.data()[out_idx] = (u32)dim_idx;
        }
    }
}

// ---- welford mean+M2 — f32 only ----------------------------------------

void tensor_cpu_welford_global(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                               const TensorImpl<f32> &src) {
    f64 mu = 0.0, m2 = 0.0;
    u64 n = 0;
    for (u64 i = 0; i < src.numel(); i++) {
        f64 x = (f64)src.data()[i];
        n++;
        f64 delta = x - mu;
        mu += delta / (f64)n;
        m2 += delta * (x - mu);
    }
    mean.data()[0] = (f32)mu;
    M2.data()[0] = (f32)m2;
}

void tensor_cpu_welford_dim(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                            const TensorImpl<f32> &src, u32 dim) {
    if (mean.numel() == 1) {
        tensor_cpu_welford_global(mean, M2, src);
        return;
    }
    u64 dim_stride = src.stride[dim];
    u64 dim_size = src.shape[dim];
    u64 n_out = mean.numel();

    // Use mean.shape (shape[dim]=1) with src strides (stride[dim]=0) so the
    // iterator visits exactly n_out unique base offsets in src — one per
    // output.
    u64 base_strides[MAX_NDIM];
    memcpy(base_strides, src.stride, src.ndim * sizeof(u64));
    base_strides[dim] = 0;
    tensorIterator base_it(mean.ndim, mean.shape, base_strides);

    std::vector<f64> mu(n_out, 0.0), m2(n_out, 0.0);

    for (u64 o = 0; o < n_out; o++) {
        u64 base = base_it.next();
        u32 n = 0;
        for (u64 i = 0; i < dim_size; i++) {
            f64 x = (f64)src.data()[base + i * dim_stride];
            n++;
            f64 delta = x - mu[o];
            mu[o] += delta / (f64)n;
            m2[o] += delta * (x - mu[o]);
        }
    }
    for (u64 o = 0; o < n_out; o++) {
        mean.data()[o] = (f32)mu[o];
        M2.data()[o] = (f32)m2[o];
    }
}

void tensor_cpu_welford_skip(TensorImpl<f32> &mean, TensorImpl<f32> &M2,
                             const TensorImpl<f32> &src, u32 dim) {
    u64 n_kept = src.shape[dim];
    std::vector<f64> mu(n_kept, 0.0), m2(n_kept, 0.0);
    std::vector<u64> cnt(n_kept, 0);

    u64 dim_strides[MAX_NDIM] = {};
    dim_strides[dim] = 1;
    tensorIterator in_it(src.ndim, src.shape, src.stride);
    tensorIterator idx_it(src.ndim, src.shape, dim_strides);

    while (in_it.has_next()) {
        u64 k = idx_it.next();
        f64 x = (f64)src.data()[in_it.next()];
        cnt[k]++;
        f64 delta = x - mu[k];
        mu[k] += delta / (f64)cnt[k];
        m2[k] += delta * (x - mu[k]);
    }
    for (u64 k = 0; k < n_kept; k++) {
        mean.data()[k] = (f32)mu[k];
        M2.data()[k] = (f32)m2[k];
    }
}

// ---- fused softmax backward — f32 only ----------------------------------

void tensor_cpu_softmax_bwd(TensorImpl<f32> &dx, const TensorImpl<f32> &s,
                            const TensorImpl<f32> &grad, i32 dim) {
    u32 axis = (dim < 0) ? (u32)((i32)s.ndim + dim) : (u32)dim;
    u64 D = s.shape[axis];
    u64 stride = s.stride[axis];
    u64 base_strides[MAX_NDIM];
    memcpy(base_strides, s.stride, s.ndim * sizeof(u64));
    base_strides[axis] = 0;
    u32 row_shape[MAX_NDIM];
    memcpy(row_shape, s.shape, s.ndim * sizeof(u32));
    row_shape[axis] = 1;
    u64 n_rows = s.numel() / D;
    tensorIterator it(s.ndim, row_shape, base_strides);
    for (u64 r = 0; r < n_rows; r++) {
        u64 base = it.next();
        f64 dot = 0.0;
        for (u64 i = 0; i < D; i++)
            dot += (f64)s.data()[base + i * stride] *
                   (f64)grad.data()[base + i * stride];
        for (u64 i = 0; i < D; i++) {
            u64 off = base + i * stride;
            dx.data()[off] += s.data()[off] * (grad.data()[off] - (f32)dot);
        }
    }
}

// ---- fused layer norm — f32 only ----------------------------------------

void tensor_cpu_ln_fwd(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                       TensorImpl<f32> &inv_std, const TensorImpl<f32> &inp,
                       const TensorImpl<f32> &gamma,
                       const TensorImpl<f32> &beta, f32 eps) {
    u32 last = inp.ndim - 1;
    u64 D = inp.shape[last];
    u64 stride = inp.stride[last];
    u64 base_strides[MAX_NDIM];
    memcpy(base_strides, inp.stride, inp.ndim * sizeof(u64));
    base_strides[last] = 0;
    // use inv_std.shape (last dim = 1) so iterator visits exactly n_rows row
    // bases
    u64 n_rows = inp.numel() / D;
    tensorIterator it(inv_std.ndim, inv_std.shape, base_strides);
    for (u64 r = 0; r < n_rows; r++) {
        u64 base = it.next();
        f64 mu = 0.0, m2 = 0.0;
        u64 n = 0;
        for (u64 i = 0; i < D; i++) {
            f64 x = inp.data()[base + i * stride];
            n++;
            f64 delta = x - mu;
            mu += delta / (f64)n;
            m2 += delta * (x - mu);
        }
        f32 inv_s = (f32)(1.0 / sqrt(m2 / (f64)D + (f64)eps));
        inv_std.data()[r] = inv_s;
        for (u64 i = 0; i < D; i++) {
            u64 off = base + i * stride;
            f32 xh = ((f32)inp.data()[off] - (f32)mu) * inv_s;
            xhat.data()[off] = xh;
            out.data()[off] = gamma.data()[i] * xh + beta.data()[i];
        }
    }
}

void tensor_cpu_ln_bwd(TensorImpl<f32> &dx, const TensorImpl<f32> &grad,
                       const TensorImpl<f32> &xhat,
                       const TensorImpl<f32> &inv_std,
                       const TensorImpl<f32> &gamma) {
    u32 last = grad.ndim - 1;
    u64 D = grad.shape[last];
    u64 stride = grad.stride[last];
    u64 base_strides[MAX_NDIM];
    memcpy(base_strides, grad.stride, grad.ndim * sizeof(u64));
    base_strides[last] = 0;
    // use inv_std.shape (last dim = 1) so iterator visits exactly n_rows row
    // bases
    u64 n_rows = grad.numel() / D;
    tensorIterator it(inv_std.ndim, inv_std.shape, base_strides);
    for (u64 r = 0; r < n_rows; r++) {
        u64 base = it.next();
        f32 isd = inv_std.data()[r];
        f64 sum_gn = 0.0, sum_gnx = 0.0;
        for (u64 i = 0; i < D; i++) {
            u64 off = base + i * stride;
            f64 gn = (f64)grad.data()[off] * (f64)gamma.data()[i];
            f64 xh = xhat.data()[off];
            sum_gn += gn;
            sum_gnx += gn * xh;
        }
        f32 mean_gn = (f32)(sum_gn / (f64)D);
        f32 mean_gnx = (f32)(sum_gnx / (f64)D);
        for (u64 i = 0; i < D; i++) {
            u64 off = base + i * stride;
            f32 gn = grad.data()[off] * gamma.data()[i];
            f32 xh = xhat.data()[off];
            dx.data()[off] += (gn - mean_gn - xh * mean_gnx) * isd;
        }
    }
}

// ---- fused batch norm — f32 only ----------------------------------------

void tensor_cpu_bn_fwd_normalize(TensorImpl<f32> &out, TensorImpl<f32> &xhat,
                                 const TensorImpl<f32> &inp,
                                 const TensorImpl<f32> &mean,
                                 const TensorImpl<f32> &m2,
                                 const TensorImpl<f32> &gamma,
                                 const TensorImpl<f32> &beta, u64 count,
                                 f32 eps) {
    u64 ch_strides[MAX_NDIM] = {};
    ch_strides[1] = 1;

    tensorIterator inp_it(inp.ndim, inp.shape, inp.stride);
    tensorIterator out_it(out.ndim, out.shape, out.stride);
    tensorIterator xhat_it(xhat.ndim, xhat.shape, xhat.stride);
    tensorIterator ch_it(inp.ndim, inp.shape, ch_strides);

    while (inp_it.has_next()) {
        u64 inp_off = inp_it.next();
        u64 out_off = out_it.next();
        u64 xhat_off = xhat_it.next();
        u32 c = (u32)ch_it.next();

        f32 std_c = sqrtf(m2.data()[c] / count + eps);
        f32 x_hat = (inp.data()[inp_off] - mean.data()[c]) / std_c;
        xhat.data()[xhat_off] = x_hat;
        out.data()[out_off] = gamma.data()[c] * x_hat + beta.data()[c];
    }
}

void tensor_cpu_bn_bwd(TensorImpl<f32> &dx, TensorImpl<f32> &d_gamma,
                       TensorImpl<f32> &d_beta, const TensorImpl<f32> &grad,
                       const TensorImpl<f32> &xhat,
                       const TensorImpl<f32> &gamma, const TensorImpl<f32> &var,
                       f32 m, f32 eps) {
    u32 C = grad.shape[1];
    std::vector<f32> sum_grad(C, 0.0f), sum_grad_xhat(C, 0.0f);

    u64 ch_strides[MAX_NDIM] = {};
    ch_strides[1] = 1;

    {
        tensorIterator grad_it(grad.ndim, grad.shape, grad.stride);
        tensorIterator xhat_it(xhat.ndim, xhat.shape, xhat.stride);
        tensorIterator ch_it(grad.ndim, grad.shape, ch_strides);
        while (grad_it.has_next()) {
            f32 g = grad.data()[grad_it.next()];
            f32 x = xhat.data()[xhat_it.next()];
            u32 c = (u32)ch_it.next();
            sum_grad[c] += g;
            sum_grad_xhat[c] += g * x;
        }
    }

    for (u32 c = 0; c < C; c++) {
        d_gamma.data()[c] += sum_grad_xhat[c];
        d_beta.data()[c] += sum_grad[c];
    }

    {
        tensorIterator grad_it(grad.ndim, grad.shape, grad.stride);
        tensorIterator xhat_it(xhat.ndim, xhat.shape, xhat.stride);
        tensorIterator dx_it(dx.ndim, dx.shape, dx.stride);
        tensorIterator ch_it(grad.ndim, grad.shape, ch_strides);
        while (grad_it.has_next()) {
            f32 g = grad.data()[grad_it.next()];
            f32 x = xhat.data()[xhat_it.next()];
            u64 dx_off = dx_it.next();
            u32 c = (u32)ch_it.next();
            f32 std_c = sqrtf(var.data()[c] + eps);
            dx.data()[dx_off] +=
                gamma.data()[c] / std_c *
                (g - sum_grad[c] / m - x * sum_grad_xhat[c] / m);
        }
    }
}

// ---- scattering ----------------------------------------------------------

template <typename T>
void tensor_cpu_scatter_add(TensorImpl<T> &out, const TensorImpl<T> &src,
                            const TensorImpl<u32> &indices, u32 dim) {
    u64 out_base_strides[MAX_NDIM];
    memcpy(out_base_strides, out.stride, out.ndim * sizeof(u64));
    out_base_strides[dim] = 0;

    tensorIterator src_it(src.ndim, src.shape, src.stride);
    tensorIterator idx_it(indices.ndim, indices.shape, indices.stride);
    tensorIterator out_it(src.ndim, src.shape, out_base_strides);

    for (u64 i = 0; i < src.numel(); i++) {
        u64 src_off = src_it.next();
        u64 idx_off = idx_it.next();
        u64 out_off =
            out_it.next() + (u64)indices.data()[idx_off] * out.stride[dim];
        out.data()[out_off] += src.data()[src_off];
    }
}

// ---- initializing — f32 only --------------------------------------------

void tensor_cpu_he_init(TensorImpl<f32> &tensor, f32 std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    u32 in_features = tensor.shape[COL_DIM(tensor)];
    f32 stddev = (std > 0.0f) ? std : std::sqrt(2.0f / in_features);
    std::normal_distribution<f32> dist(0.0f, stddev);
    for (u64 i = 0; i < tensor.numel(); i++)
        tensor.data()[i] = dist(gen);
}

void tensor_cpu_causal_mask(TensorImpl<f32> &t) {
    u32 rows = t.shape[t.ndim - 2];
    u32 cols = t.shape[t.ndim - 1];
    u64 n_last = (u64)rows * cols;
    tensorIterator it(t.ndim, t.shape, t.stride);
    for (u64 flat = 0; it.has_next(); flat++) {
        u32 i = (flat % n_last) / cols;
        u32 j = flat % cols;
        t.data()[it.next()] = (j <= i) ? 0.0f : -1e9f;
    }
}

// ---- indexing ------------------------------------------------------------

template <typename T>
void tensor_cpu_gather(TensorImpl<T> &dst, const TensorImpl<T> &src,
                       const TensorImpl<u32> &indices, u32 dim) {
    u64 src_base_strides[MAX_NDIM];
    memcpy(src_base_strides, src.stride, src.ndim * sizeof(u64));
    src_base_strides[dim] = 0;

    tensorIterator dst_it(dst.ndim, dst.shape, dst.stride);
    tensorIterator idx_it(indices.ndim, indices.shape, indices.stride);
    tensorIterator src_it(src.ndim, src.shape, src_base_strides);

    for (u64 i = 0; i < dst.numel(); i++) {
        u64 dst_off = dst_it.next();
        u64 idx_off = idx_it.next();
        u64 src_off =
            src_it.next() + (u64)indices.data()[idx_off] * src.stride[dim];
        dst.data()[dst_off] = src.data()[src_off];
    }
}

// ---- comparison — f32 only ----------------------------------------------

b32 tensor_cpu_equals(const TensorImpl<f32> &a, const TensorImpl<f32> &b,
                      f32 tol) {
    tensorIterator a_iter(a.ndim, a.shape, a.stride);
    tensorIterator b_iter(b.ndim, b.shape, b.stride);
    for (u64 i = 0; i < a.numel(); i++)
        if (fabsf(a.data()[a_iter.next()] - b.data()[b_iter.next()]) > tol)
            return false;
    return true;
}

// ---- spatial / patch operations ------------------------------------------

template <typename T>
void tensor_cpu_unfold2d(TensorImpl<T> &dst, const TensorImpl<T> &src,
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
                                dst.data()[dst_off] = T(params.pad_constant);
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

template <typename T>
void tensor_cpu_fold2d(TensorImpl<T> &dst, const TensorImpl<T> &col,
                       Unfold2dParams params) {
    u32 N = dst.shape[0];
    u32 C = dst.shape[1];
    u32 H = dst.shape[2];
    u32 W = dst.shape[3];
    u32 kH = params.k_h;
    u32 kW = params.k_w;
    params.compute_output_size(H, W);

    u32 shape6[MAX_NDIM] = {N, params.L_h, params.L_w, C, kH, kW};
    u64 stride6[MAX_NDIM];
    tensor_compute_strides(stride6, shape6, 6);

    const T *col_data = col.data();

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

// ---- Explicit instantiations --------------------------------------------

#define INST(T)                                                                \
    template void tensor_cpu_copy(TensorImpl<T> &, const TensorImpl<T> &);     \
    template void tensor_cpu_contigous(TensorImpl<T> &);                       \
    template void tensor_cpu_fill(TensorImpl<T> &, T);                         \
    template void tensor_cpu_clear(TensorImpl<T> &);                           \
    template void tensor_cpu_arange(TensorImpl<T> &);                          \
    template void tensor_cpu_add(TensorImpl<T> &, const TensorImpl<T> &,       \
                                 const TensorImpl<T> &);                       \
    template void tensor_cpu_sub(TensorImpl<T> &, const TensorImpl<T> &,       \
                                 const TensorImpl<T> &);                       \
    template void tensor_cpu_mul(TensorImpl<T> &, const TensorImpl<T> &,       \
                                 const TensorImpl<T> &);                       \
    template void tensor_cpu_div(TensorImpl<T> &, const TensorImpl<T> &,       \
                                 const TensorImpl<T> &);                       \
    template void tensor_cpu_equal(TensorImpl<T> &, const TensorImpl<T> &,     \
                                   const TensorImpl<T> &);                     \
    template void tensor_cpu_add(TensorImpl<T> &, const TensorImpl<T> &, T);   \
    template void tensor_cpu_sub(TensorImpl<T> &, const TensorImpl<T> &, T);   \
    template void tensor_cpu_mul(TensorImpl<T> &, const TensorImpl<T> &, T);   \
    template void tensor_cpu_div(TensorImpl<T> &, const TensorImpl<T> &, T);   \
    template void tensor_cpu_sum_global(TensorImpl<T> &,                       \
                                        const TensorImpl<T> &);                \
    template void tensor_cpu_sum_dim(TensorImpl<T> &, const TensorImpl<T> &,   \
                                     u32);                                     \
    template void tensor_cpu_max(TensorImpl<T> &, const TensorImpl<T> &);      \
    template void tensor_cpu_max(TensorImpl<T> &, const TensorImpl<T> &, u32); \
    template void tensor_cpu_argmax(TensorImpl<u32> &, const TensorImpl<T> &,  \
                                    u32);                                      \
    template void tensor_cpu_scatter_add(                                      \
        TensorImpl<T> &, const TensorImpl<T> &, const TensorImpl<u32> &, u32); \
    template void tensor_cpu_gather(TensorImpl<T> &, const TensorImpl<T> &,    \
                                    const TensorImpl<u32> &, u32);             \
    template void tensor_cpu_unfold2d(TensorImpl<T> &, const TensorImpl<T> &,  \
                                      Unfold2dParams);                         \
    template void tensor_cpu_fold2d(TensorImpl<T> &, const TensorImpl<T> &,    \
                                    Unfold2dParams);

INST(f32)
INST(u32)
