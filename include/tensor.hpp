#ifndef TENSOR_HPP
#define TENSOR_HPP

#define MAX_NDIM 8
#include "backend/cuda_mem_arena.hpp"
#include "base.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>

#define ROW_DIM(t) ((t).ndim - 2)
#define COL_DIM(t) ((t).ndim - 1)

// Forward-declared so TensorImpl's inline constructor can call it.
// Full declaration appears again in the metadata section below.
u64 tensor_compute_strides(u64 *stride, const u32 *shape, u32 ndim);

// ---- Storage<T>: owns a typed memory buffer (CPU or GPU) --------------------

template <typename T = f32> struct Storage {
    T *data;
    u64 nbytes;
    b32 on_gpu;
    CudaMemArena *arena;

    Storage(u64 nbytes, b32 on_gpu, CudaMemArena *arena = nullptr)
        : nbytes(nbytes), on_gpu(on_gpu), arena(arena) {
        if (on_gpu) {
            if (arena)
                data = (T *)cuda_arena_push(arena, nbytes);
            else
                cudaMallocAsync(&data, nbytes, 0);
        } else {
            data = (T *)malloc(nbytes);
        }
    }

    ~Storage() {
        if (on_gpu) {
            if (!arena)
                cudaFreeAsync(data, 0);
        } else {
            free(data);
        }
    }

    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
};

// ---- TensorImpl<T>: typed tensor metadata + storage reference ---------------

template <typename T = f32> struct TensorImpl {
    std::shared_ptr<Storage<T>> storage;
    u64 storage_offset = 0;
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];
    u32 ndim;

    TensorImpl(u32 ndim, const u32 *shape, b32 on_gpu,
               CudaMemArena *arena = nullptr) {
        this->ndim = ndim;
        memcpy(this->shape, shape, ndim * sizeof(u32));
        u64 n = tensor_compute_strides(this->stride, this->shape, ndim);
        storage = std::make_shared<Storage<T>>(n * sizeof(T), on_gpu, arena);
    }

    TensorImpl(u32 ndim, const u32 *shape, const u64 *stride, b32 on_gpu,
               CudaMemArena *arena = nullptr) {
        this->ndim = ndim;
        memcpy(this->shape, shape, ndim * sizeof(u32));
        memcpy(this->stride, stride, ndim * sizeof(u64));
        u64 n = 1;
        for (u32 i = 0; i < ndim; i++)
            n *= shape[i];
        storage = std::make_shared<Storage<T>>(n * sizeof(T), on_gpu, arena);
    }

    ~TensorImpl() = default;
    TensorImpl(const TensorImpl &) = delete;
    TensorImpl &operator=(const TensorImpl &) = delete;

    T *data() const { return storage->data + storage_offset; }
    b32 on_gpu() const { return storage->on_gpu; }
    u64 numel() const {
        u64 n = 1;
        for (u32 i = 0; i < ndim; i++)
            n *= shape[i];
        return n;
    }

    template <typename... Dims> T &operator()(Dims... dims) {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data()[offset];
    }

    template <typename... Dims> const T &operator()(Dims... dims) const {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data()[offset];
    }
};

// ---- Tensor<T>: ref-counted handle around TensorImpl<T> ---------------------

template <typename T = f32> struct Tensor {
    std::shared_ptr<TensorImpl<T>> impl_;

    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl<T>> i) : impl_(std::move(i)) {}

    static Tensor<T> make(u32 ndim, const u32 *shape, b32 on_gpu,
                          CudaMemArena *arena = nullptr) {
        return Tensor<T>(
            std::make_shared<TensorImpl<T>>(ndim, shape, on_gpu, arena));
    }

    static Tensor<T> make(u32 ndim, const u32 *shape, const u64 *stride,
                          b32 on_gpu, CudaMemArena *arena = nullptr) {
        return Tensor<T>(std::make_shared<TensorImpl<T>>(ndim, shape, stride,
                                                         on_gpu, arena));
    }

    bool defined() const { return impl_ != nullptr; }
    explicit operator bool() const { return defined(); }

    TensorImpl<T> *operator->() const { return impl_.get(); }
    TensorImpl<T> &operator*() const { return *impl_; }
    TensorImpl<T> &impl() const { return *impl_; }
};

// ---- Type aliases -----------------------------------------------------------

using TensorU32 = Tensor<u32>;
using TensorI32 = Tensor<i32>;
using TensorF64 = Tensor<f64>;

// ---- Unfold2dParams ---------------------------------------------------------

struct Unfold2dParams {
    u32 k_h, k_w;
    u32 stride_h, stride_w;
    u32 pad_h, pad_w;
    f32 pad_constant;
    u32 L_h = 0, L_w = 0;

    Unfold2dParams() {}
    Unfold2dParams(u32 k, u32 stride = 1, u32 pad = 0, u32 dil = 1,
                   f32 pad_constant = 0.0f);

    void compute_output_size(u32 H, u32 W) {
        L_h = (H + 2 * pad_h - k_h) / stride_h + 1;
        L_w = (W + 2 * pad_w - k_w) / stride_w + 1;
    }
};

// ---- file I/O / device transfers --------------------------------------------

template <typename T = f32>
Tensor<T> tensor_load(const char *filename, b32 on_gpu,
                      CudaMemArena *arena = nullptr);

template <typename T> b32 tensor_copy(Tensor<T> &dst, const Tensor<T> &src);

template <typename T>
void tensor_contiguous(Tensor<T> &t, CudaMemArena *arena = nullptr);

// ---- metadata (device-independent) -----------------------------------------

u64 tensor_compute_strides(u64 *stride, const u32 *shape, u32 ndim);

template <typename T>
b32 tensor_reshape(TensorImpl<T> &t, const u32 *shape, u32 ndim,
                   CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_reshape(Tensor<T> &t, const u32 *shape, u32 ndim,
                   CudaMemArena *arena = nullptr);
template <typename T>

b32 tensor_flatten(Tensor<T> &t, u32 start_dim, u32 end_dim,
                   CudaMemArena *arena = nullptr);

template <typename T> void tensor_print(const TensorImpl<T> &t);

// ---- fill / arange ----------------------------------------------------------

template <typename T> void tensor_fill(Tensor<T> &t, T value);
// Fills t in-place with 0, 1, …, numel-1 cast to T.
template <typename T> void tensor_arange(Tensor<T> &t);

// ---- f32 activations --------------------------------------------------------

b32 tensor_relu(Tensor<f32> &dst, const Tensor<f32> &src);
Tensor<f32> tensor_relu(const Tensor<f32> &src, CudaMemArena *arena = nullptr);
b32 tensor_gelu(Tensor<f32> &dst, const Tensor<f32> &src);
Tensor<f32> tensor_gelu(const Tensor<f32> &src, CudaMemArena *arena = nullptr);
b32 tensor_gelu_backward(Tensor<f32> &out, const Tensor<f32> &grad,
                         const Tensor<f32> &input);
Tensor<f32> tensor_gelu_backward(const Tensor<f32> &grad,
                                 const Tensor<f32> &input,
                                 CudaMemArena *arena = nullptr);
b32 tensor_exp(Tensor<f32> &dst, const Tensor<f32> &src);
Tensor<f32> tensor_exp(const Tensor<f32> &src, CudaMemArena *arena = nullptr);
b32 tensor_log(Tensor<f32> &dst, const Tensor<f32> &src);
Tensor<f32> tensor_log(const Tensor<f32> &src, CudaMemArena *arena = nullptr);
b32 tensor_sqrt(Tensor<f32> &dst, const Tensor<f32> &src);
Tensor<f32> tensor_sqrt(const Tensor<f32> &src, CudaMemArena *arena = nullptr);

// ---- elementwise binary -----------------------------------------------------

template <typename T>
b32 tensor_add(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b);
template <typename T>
Tensor<T> tensor_add(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_sub(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b);
template <typename T>
Tensor<T> tensor_sub(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_mul(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b);
template <typename T>
Tensor<T> tensor_mul(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_div(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b);
template <typename T>
Tensor<T> tensor_div(const Tensor<T> &a, const Tensor<T> &b,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_equal(Tensor<T> &out, const Tensor<T> &a, const Tensor<T> &b);
template <typename T>
Tensor<T> tensor_equal(const Tensor<T> &a, const Tensor<T> &b,
                       CudaMemArena *arena = nullptr);

b32 tensor_relu_backward(Tensor<f32> &out, const Tensor<f32> &grad,
                         const Tensor<f32> &in);
Tensor<f32> tensor_relu_backward(const Tensor<f32> &grad, const Tensor<f32> &in,
                                 CudaMemArena *arena = nullptr);

// Fills mask in-place: each element is 0 (dropped, prob p) or 1/(1-p) (kept).
void tensor_dropout_mask(Tensor<f32> &mask, f32 p);

b32 tensor_softmax(Tensor<f32> &out, const Tensor<f32> &in, i32 dim = -1,
                   CudaMemArena *arena = nullptr);
Tensor<f32> tensor_softmax(const Tensor<f32> &in, i32 dim = -1,
                           CudaMemArena *arena = nullptr);
b32 tensor_log_softmax(Tensor<f32> &out, const Tensor<f32> &in, i32 dim = -1,
                       CudaMemArena *arena = nullptr);
Tensor<f32> tensor_log_softmax(const Tensor<f32> &in, i32 dim = -1,
                               CudaMemArena *arena = nullptr);

// ---- scalar ops -------------------------------------------------------------

template <typename T>
b32 tensor_add(Tensor<T> &out, const Tensor<T> &a, T scalar);
template <typename T>
Tensor<T> tensor_add(const Tensor<T> &a, T scalar,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_sub(Tensor<T> &out, const Tensor<T> &a, T scalar);
template <typename T>
Tensor<T> tensor_sub(const Tensor<T> &a, T scalar,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_mul(Tensor<T> &out, const Tensor<T> &a, T scalar);
template <typename T>
Tensor<T> tensor_mul(const Tensor<T> &a, T scalar,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_div(Tensor<T> &out, const Tensor<T> &a, T scalar);
template <typename T>
Tensor<T> tensor_div(const Tensor<T> &a, T scalar,
                     CudaMemArena *arena = nullptr);

// ---- f32 matmul -------------------------------------------------------------

b32 tensor_mat_mul(Tensor<f32> &out, const Tensor<f32> &a, const Tensor<f32> &b,
                   b32 clear_out = true);
Tensor<f32> tensor_mat_mul(const Tensor<f32> &a, const Tensor<f32> &b,
                           CudaMemArena *arena = nullptr);

// ---- reductions -------------------------------------------------------------

// Multi-axis: the core form, backends implement only this.
template <typename T>
b32 tensor_sum(Tensor<T> &out, const Tensor<T> &t, const u32 *axes, u32 n_axes,
               b32 keep_dim = true, b32 clear_out = true);
template <typename T>
Tensor<T> tensor_sum(const Tensor<T> &t, const u32 *axes, u32 n_axes,
                     b32 keep_dim = true, CudaMemArena *arena = nullptr);

// Single-dim: delegates to multi-axis.
template <typename T>
b32 tensor_sum(Tensor<T> &out, const Tensor<T> &t, u32 dim, b32 keep_dim = true,
               b32 clear_out = true);
template <typename T>
Tensor<T> tensor_sum(const Tensor<T> &t, u32 dim, b32 keep_dim = true,
                     CudaMemArena *arena = nullptr);

// Global: delegates to multi-axis with all axes.
template <typename T>
b32 tensor_sum(Tensor<T> &out, const Tensor<T> &t, b32 clear_out = true);
template <typename T>
Tensor<T> tensor_sum(const Tensor<T> &t, CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_max(Tensor<T> &out, const Tensor<T> &t, u32 dim,
               b32 keep_dim = true);
template <typename T>
Tensor<T> tensor_max(const Tensor<T> &t, u32 dim, b32 keep_dim = true,
                     CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_argmax(TensorU32 &out, const Tensor<T> &t, u32 dim,
                  b32 keep_dim = true);
template <typename T>
TensorU32 tensor_argmax(const Tensor<T> &t, u32 dim, b32 keep_dim = true,
                        CudaMemArena *arena = nullptr);

b32 tensor_welford_mean_var(Tensor<f32> &mean, Tensor<f32> &var,
                            const Tensor<f32> &src, const u32 *axes,
                            u32 n_axes);
// Reduce all dims except skip_dim (e.g. skip_dim=1 for batch norm over N,H,W).
b32 tensor_welford_mean_var(Tensor<f32> &mean, Tensor<f32> &var,
                            const Tensor<f32> &src, u32 skip_dim);

// ---- scatter ----------------------------------------------------------------

// dst[outer(p), index[idx(p)], inner(p)] += src[p]
//   index.shape == src.shape[:src.ndim - (dst.ndim - dim - 1)]
template <typename T>
b32 tensor_scatter_add(Tensor<T> &out, const Tensor<T> &src,
                       const TensorU32 &indices, u32 dim);

template <typename T>
Tensor<T> scatter_add(const Tensor<T> &src, const TensorU32 &indices, u32 dim,
                      CudaMemArena *arena = nullptr);

// ---- f32 init ---------------------------------------------------------------

void tensor_he_init(Tensor<f32> &t);
// Fills t with 0 on/below last-two-dim diagonal, -1e9 above. t must be [..., n,
// n].
void tensor_causal_mask(Tensor<f32> &t);

// ---- indexing ---------------------------------------------------------------

template <typename T>
b32 tensor_gather(Tensor<T> &dst, const Tensor<T> &src,
                  const TensorU32 &indices, u32 dim);
template <typename T>
Tensor<T> tensor_gather(const Tensor<T> &src, const TensorU32 &indices, u32 dim,
                        CudaMemArena *arena = nullptr);
// ---- spatial ----------------------------------------------------------------

template <typename T>
b32 tensor_unfold2d(Tensor<T> &out, const Tensor<T> &input,
                    Unfold2dParams params);
template <typename T>
Tensor<T> tensor_unfold2d(const Tensor<T> &input, Unfold2dParams params,
                          CudaMemArena *arena = nullptr);
template <typename T>
b32 tensor_fold2d(Tensor<T> &dst, const Tensor<T> &col, Unfold2dParams params);

// ---- f32 comparison ---------------------------------------------------------

b32 tensor_equals(const Tensor<f32> &a, const Tensor<f32> &b, f32 tol = 1e-5f);

// ============================================================================
// Generic template utilities — work for any element type T.
// ============================================================================

// ---- shape / stride (T-independent) -----------------------------------------

template <typename T> inline b32 tensor_is_contiguous(const TensorImpl<T> &t) {
    u64 expected = 1;
    for (u32 i = t.ndim; i-- > 0;) {
        if (t.stride[i] != expected)
            return false;
        expected *= t.shape[i];
    }
    return true;
}

template <typename T> inline b32 tensor_is_contiguous(const Tensor<T> &t) {
    return tensor_is_contiguous(t.impl());
}

template <typename T, typename U = T>
inline b32 tensor_shape_eq(const TensorImpl<T> &a, const TensorImpl<U> &b) {
    return a.ndim == b.ndim &&
           memcmp(a.shape, b.shape, a.ndim * sizeof(u32)) == 0;
}

template <typename T, typename U = T>
inline b32 tensor_shape_eq(const Tensor<T> &a, const Tensor<U> &b) {
    return tensor_shape_eq(a.impl(), b.impl());
}

template <typename T> inline void tensor_unsqueeze(TensorImpl<T> &t, u32 dim) {
    for (u32 i = t.ndim; i > dim; i--) {
        t.shape[i] = t.shape[i - 1];
        t.stride[i] = t.stride[i - 1];
    }
    t.shape[dim] = 1;
    t.stride[dim] = 0;
    t.ndim++;
}

template <typename T>
inline b32 tensor_transpose(TensorImpl<T> &t, u32 dim0, u32 dim1) {
    if (dim0 == dim1 || dim0 >= t.ndim || dim1 >= t.ndim)
        return false;
    u32 ts = t.shape[dim0];
    t.shape[dim0] = t.shape[dim1];
    t.shape[dim1] = ts;
    u64 tr = t.stride[dim0];
    t.stride[dim0] = t.stride[dim1];
    t.stride[dim1] = tr;
    return true;
}

template <typename T>
inline b32 tensor_transpose(Tensor<T> &t, u32 dim0, u32 dim1) {
    return tensor_transpose(t.impl(), dim0, dim1);
}

template <typename T>
inline void expanded_shape(const TensorImpl<T> &t, u32 expanded_ndim,
                           u32 *out) {
    u32 n_prepend = expanded_ndim - t.ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        out[i] = 1;
    for (; i < expanded_ndim; i++)
        out[i] = t.shape[i - n_prepend];
}

template <typename T>
inline void expanded_stride(const TensorImpl<T> &t, u32 expanded_ndim,
                            u64 *out) {
    u32 n_prepend = expanded_ndim - t.ndim;
    u32 i = 0;
    for (; i < n_prepend; i++)
        out[i] = 0;
    for (; i < expanded_ndim; i++) {
        u32 ti = i - n_prepend;
        out[i] = (t.shape[ti] != 1) ? t.stride[ti] : 0;
    }
}

// Left-pad t with size-1/stride-0 dims so t.ndim == expanded_ndim.
// Existing size-1 dims also get stride=0. Shapes stay minimal (all new dims
// are size 1); use tensor_broadcast_to if you know the target shape.
template <typename T>
inline b32 tensor_expand_shape(TensorImpl<T> &t, u32 expanded_ndim) {
    if (t.ndim > expanded_ndim)
        return false;
    if (t.ndim == expanded_ndim)
        return true;
    u32 new_shape[MAX_NDIM];
    u64 new_stride[MAX_NDIM];
    expanded_shape(t, expanded_ndim, new_shape);
    expanded_stride(t, expanded_ndim, new_stride);
    for (u32 i = 0; i < expanded_ndim; i++) {
        t.shape[i] = new_shape[i];
        t.stride[i] = new_stride[i];
    }
    t.ndim = expanded_ndim;
    return true;
}

// General broadcast-to: left-pad to target_ndim, then expand every
// broadcast dim (stride==0) to the matching size in target_shape.
// Returns false if a non-broadcast dim doesn't match target_shape.
// Handles numpy-style right-alignment: to broadcast a middle dim
// (e.g. [C] → [N,C,H,W]) unsqueeze to [1,C,1,1] first.
template <typename T>
inline b32 tensor_broadcast_to(TensorImpl<T> &t, const u32 *target_shape,
                               u32 target_ndim) {
    if (!tensor_expand_shape(t, target_ndim))
        return false;
    for (u32 i = 0; i < target_ndim; i++) {
        if (t.stride[i] == 0) {
            t.shape[i] = target_shape[i];
        } else if (t.shape[i] != target_shape[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, typename U = T>
inline u32 broadcast_shape(const TensorImpl<T> &a, const TensorImpl<U> &b,
                           u32 *out_shape) {
    u32 expanded_ndim = a.ndim > b.ndim ? a.ndim : b.ndim;
    for (u32 i = 0; i < expanded_ndim; i++) {
        i32 ai = (i32)i - (i32)(expanded_ndim - a.ndim);
        i32 bi = (i32)i - (i32)(expanded_ndim - b.ndim);
        u32 da = (ai >= 0) ? a.shape[ai] : 1;
        u32 db = (bi >= 0) ? b.shape[bi] : 1;
        u32 smaller = da < db ? da : db;
        if (da != db && smaller != 1)
            return 0;
        out_shape[i] = da > db ? da : db;
    }
    return expanded_ndim;
}

// ---- generic memory utilities -----------------------------------------------

template <typename T> inline void tensor_clear(Tensor<T> &t) {
    if (t->on_gpu())
        cudaMemset(t->data(), 0, t->numel() * sizeof(T));
    else
        memset(t->data(), 0, t->numel() * sizeof(T));
}

template <typename T>
inline Tensor<T> tensor_create_like(const Tensor<T> &src,
                                    CudaMemArena *arena = nullptr) {
    return Tensor<T>::make(src->ndim, src->shape, src->on_gpu(), arena);
}

template <typename T>
inline Tensor<T> tensor_zeros(u32 ndim, const u32 *shape, b32 on_gpu,
                              CudaMemArena *arena = nullptr) {
    Tensor<T> t = Tensor<T>::make(ndim, shape, on_gpu, arena);
    tensor_clear(t);
    return t;
}

template <typename T>
inline Tensor<T> tensor_zeros_like(const Tensor<T> &src,
                                   CudaMemArena *arena = nullptr) {
    Tensor<T> t = tensor_create_like(src, arena);
    tensor_clear(t);
    return t;
}

template <typename T>
inline Tensor<T> tensor_view(const Tensor<T> &src,
                             CudaMemArena *arena = nullptr) {
    Tensor<T> v = Tensor<T>::make(src->ndim, src->shape, src->stride,
                                  src->on_gpu(), arena);
    v->storage = src->storage;
    v->storage_offset = src->storage_offset;
    return v;
}

template <typename T>
inline Tensor<T> tensor_to_gpu(const Tensor<T> &t,
                               CudaMemArena *arena = nullptr) {
    Tensor<T> dst = Tensor<T>::make(t->ndim, t->shape, t->stride, true, arena);
    cudaMemcpyKind kind =
        t->on_gpu() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(dst->data(), t->data(), t->numel() * sizeof(T), kind);
    return dst;
}

template <typename T>
inline Tensor<T> tensor_to_cpu(const Tensor<T> &t,
                               CudaMemArena *arena = nullptr) {
    Tensor<T> dst = Tensor<T>::make(t->ndim, t->shape, t->stride, false, arena);
    cudaMemcpyKind kind =
        t->on_gpu() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    cudaMemcpy(dst->data(), t->data(), t->numel() * sizeof(T), kind);
    return dst;
}

#endif
