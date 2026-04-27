#ifndef TENSOR_HPP
#define TENSOR_HPP

#define MAX_NDIM 8
#include "backend/cuda_mem_arena.hpp"
#include "base.hpp"
#include <memory>

#define ROW_DIM(t) ((t).ndim - 2)
#define COL_DIM(t) ((t).ndim - 1)

struct Storage {
    f32 *data;
    u64 nbytes;
    b32 on_gpu;
    // TODO: Change to allocator in the future
    CudaMemArena *arena;

    Storage(u64 nbytes, b32 on_gpu, CudaMemArena *arena = nullptr);
    ~Storage();

    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
};

struct TensorImpl {
    std::shared_ptr<Storage> storage;
    u64 storage_offset = 0;
    u32 shape[MAX_NDIM];
    u64 stride[MAX_NDIM];
    u32 ndim;

    // Allocates a new storage and tensor metadata
    TensorImpl(u32 ndim, const u32 *shape, b32 on_gpu,
               CudaMemArena *arena = nullptr);
    TensorImpl(u32 ndim, const u32 *shape, const u64 *stride, b32 on_gpu,
               CudaMemArena *arena = nullptr);

    // Default destructor — shared_ptr handles storage cleanup automatically
    ~TensorImpl() = default;

    TensorImpl(const TensorImpl &) = delete;
    TensorImpl &operator=(const TensorImpl &) = delete;

    // Forwarding accessors
    f32 *data() const { return storage->data + storage_offset; }
    b32 on_gpu() const { return storage->on_gpu; }
    u64 numel() const {
        u64 n = 1;
        for (u32 i = 0; i < ndim; i++)
            n *= shape[i];
        return n;
    }

    template <typename... Dims> f32 &operator()(Dims... dims) {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data()[offset];
    }

    template <typename... Dims> const f32 &operator()(Dims... dims) const {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data()[offset];
    }
};

// ---- Tensor: refcounted handle -------------------------------------------
struct Tensor {
    std::shared_ptr<TensorImpl> impl_;

    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> i) : impl_(std::move(i)) {}

    static Tensor make(u32 ndim, const u32 *shape, b32 on_gpu,
                       CudaMemArena *arena = nullptr) {
        return Tensor(std::make_shared<TensorImpl>(ndim, shape, on_gpu, arena));
    };
    static Tensor make(u32 ndim, const u32 *shape, const u64 *stride,
                       b32 on_gpu, CudaMemArena *arena = nullptr) {
        return Tensor(
            std::make_shared<TensorImpl>(ndim, shape, stride, on_gpu));
    };

    bool defined() const { return impl_ != nullptr; }
    explicit operator bool() const { return defined(); }

    TensorImpl *operator->() const { return impl_.get(); }
    TensorImpl &operator*() const { return *impl_; }

    TensorImpl &impl() const { return *impl_; }
};

struct Unfold2dParams {
    u32 k_h, k_w;
    u32 stride_h, stride_w;
    u32 pad_h, pad_w;
    f32 pad_constant;
    u32 L_h = 0, L_w = 0; // output spatial dims, set by compute_output_size()

    Unfold2dParams() {}
    // Square kernel, same value for both dims
    Unfold2dParams(u32 k, u32 stride = 1, u32 pad = 0, u32 dil = 1,
                   f32 pad_constant = 0.0f);

    void compute_output_size(u32 H, u32 W) {
        L_h = (H + 2 * pad_h - k_h) / stride_h + 1;
        L_w = (W + 2 * pad_w - k_w) / stride_w + 1;
    }
};

// ---- file I/O / device transfers -----------------------------------------
Tensor tensor_load(const char *filename, b32 on_gpu);
b32 tensor_copy(Tensor &dst, const Tensor &src);
Tensor tensor_to_gpu(const Tensor &t);
Tensor tensor_to_cpu(const Tensor &t);

void tensor_contiguous(Tensor &t);
Tensor tensor_view(const Tensor &src);
Tensor tensor_create_like(const Tensor &src);

// ---- metadata (these stay on TensorImpl* — they're cheap, internal) ------
// Rationale: shape helpers don't allocate, don't return tensors, and are
// called from inside other tensor ops. Keeping them on TensorImpl* avoids
// pointless refcount bumps in hot paths.
u64 tensor_compute_strides(u64 *stride, const u32 *shape, u32 ndim);
b32 tensor_expand_shape(TensorImpl &t, u32 expanded_ndim);
b32 tensor_is_contiguous(const TensorImpl &t);
b32 tensor_shape_eq(const TensorImpl &a, const TensorImpl &b);
b32 tensor_transpose(TensorImpl &t, u32 dim0, u32 dim1);
b32 tensor_reshape(TensorImpl &t, const u32 *shape, u32 ndim);
u32 broadcast_shape(const TensorImpl &a, const TensorImpl &b, u32 *out_shape);
void expanded_shape(const TensorImpl &t, u32 expanded_ndim, u32 *out);
void expanded_stride(const TensorImpl &t, u32 expanded_ndim, u64 *out);
void tensor_print(const TensorImpl &t);

// You might want public Tensor& overloads for the few that users call:
b32 tensor_transpose(Tensor &t, u32 dim0, u32 dim1);
b32 tensor_reshape(Tensor &t, const u32 *shape, u32 ndim);
b32 tensor_shape_eq(const Tensor &a, const Tensor &b);
b32 tensor_is_contiguous(const Tensor &t);

// ---- fill / clear --------------------------------------------------------
void tensor_fill(Tensor &t, f32 value);
void tensor_clear(Tensor &t);

// ---- activations ---------------------------------------------------------
b32 tensor_relu(Tensor &dst, const Tensor &src);
Tensor tensor_relu(const Tensor &src);
b32 tensor_exp(Tensor &dst, const Tensor &src);
Tensor tensor_exp(const Tensor &src);
b32 tensor_log(Tensor &dst, const Tensor &src);
Tensor tensor_log(const Tensor &src);
b32 tensor_sqrt(Tensor &dst, const Tensor &src);
Tensor tensor_sqrt(const Tensor &src);

// ---- elementwise binary --------------------------------------------------
b32 tensor_add(Tensor &out, const Tensor &a, const Tensor &b);
Tensor tensor_add(const Tensor &a, const Tensor &b);
b32 tensor_sub(Tensor &out, const Tensor &a, const Tensor &b);
Tensor tensor_sub(const Tensor &a, const Tensor &b);
b32 tensor_mul(Tensor &out, const Tensor &a, const Tensor &b);
Tensor tensor_mul(const Tensor &a, const Tensor &b);
b32 tensor_div(Tensor &out, const Tensor &a, const Tensor &b);
Tensor tensor_div(const Tensor &a, const Tensor &b);
b32 tensor_equal(Tensor &out, const Tensor &a, const Tensor &b);
Tensor tensor_equal(const Tensor &a, const Tensor &b);

b32 tensor_relu_backward(Tensor &out, const Tensor &grad, const Tensor &in);
Tensor tensor_relu_backward(const Tensor &grad, const Tensor &in);

b32 tensor_softmax(Tensor &out, const Tensor &in);
Tensor tensor_softmax(const Tensor &in);
b32 tensor_log_softmax(Tensor &out, const Tensor &in);
Tensor tensor_log_softmax(const Tensor &in);

// ---- scalar ops ----------------------------------------------------------
b32 tensor_add(Tensor &out, const Tensor &a, f32 scalar);
Tensor tensor_add(const Tensor &a, f32 scalar);
b32 tensor_sub(Tensor &out, const Tensor &a, f32 scalar);
Tensor tensor_sub(const Tensor &a, f32 scalar);
b32 tensor_mul(Tensor &out, const Tensor &a, f32 scalar);
Tensor tensor_mul(const Tensor &a, f32 scalar);
b32 tensor_div(Tensor &out, const Tensor &a, f32 scalar);
Tensor tensor_div(const Tensor &a, f32 scalar);

// ---- matmul --------------------------------------------------------------
b32 tensor_mat_mul(Tensor &out, const Tensor &a, const Tensor &b,
                   b32 clear_out = true);
Tensor tensor_mat_mul(const Tensor &a, const Tensor &b);

// ---- reductions ----------------------------------------------------------
b32 tensor_sum(Tensor &out, const Tensor &t, b32 clear_out = true);
b32 tensor_sum(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim = true,
               b32 clear_out = true);
Tensor tensor_sum(const Tensor &t);
Tensor tensor_sum(const Tensor &t, u32 dim, b32 keep_dim = true);
b32 tensor_max(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim = true);
Tensor tensor_max(const Tensor &t, u32 dim, b32 keep_dim = true);
b32 tensor_argmax(Tensor &out, const Tensor &t, u32 dim, b32 keep_dim = true);
Tensor tensor_argmax(const Tensor &t, u32 dim, b32 keep_dim = true);

b32 tensor_welford_mean_var(Tensor &mean, Tensor &var, const Tensor &src,
                            u32 dim);

// ---- scatter -------------------------------------------------------------
b32 tensor_scatter_add(Tensor &out, const Tensor &src, const Tensor &indices,
                       u32 dim);
Tensor tensor_scatter_add(const Tensor &src, const Tensor &indices, u32 dim,
                          u32 dim_size);

// ---- init ----------------------------------------------------------------
void tensor_he_init(Tensor &t);

// ---- indexing ------------------------------------------------------------
b32 tensor_index_select(Tensor &dst, const Tensor &src, const u32 *indices,
                        u32 n_indices, u32 dim);
Tensor tensor_index_select(const Tensor &src, const u32 *indices, u32 n_indices,
                           u32 dim);

// ---- spatial -------------------------------------------------------------
b32 tensor_unfold2d(Tensor &out, const Tensor &input, Unfold2dParams params);
Tensor tensor_unfold2d(const Tensor &input, Unfold2dParams params);
b32 tensor_fold2d(Tensor &dst, const Tensor &col, Unfold2dParams params);

// ---- comparison ----------------------------------------------------------
b32 tensor_equals(const Tensor &a, const Tensor &b, f32 tol = 1e-5f);

#endif
