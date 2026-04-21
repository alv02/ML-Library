#ifndef TENSOR_HPP
#define TENSOR_HPP

#define MAX_NDIM 8
#include "base.hpp"

#define ROW_DIM(t) ((t)->ndim - 2)
#define COL_DIM(t) ((t)->ndim - 1)

struct Tensor {
    u32 shape[MAX_NDIM];  // size of each dimension
    u64 stride[MAX_NDIM]; // step in elements to advance one index in each dim
                          // (row-major)
    f32 *data;            // flat data buffer (CPU or GPU pointer)
    u32 ndim;             // number of active dimensions
    u64 size;             // total number of elements (product of shape)
    b32 on_gpu;           // true if data lives on the GPU
    b32 owns_data; // if false, data is borrowed (view) — destructor won't free
                   // it

    // Allocates a zero-filled buffer of the given shape on CPU or GPU.
    Tensor(u32 ndim, const u32 *shape, b32 on_gpu);
    // Creates a view: shares the same data pointer, owns_data=false.
    Tensor(const Tensor *src);
    ~Tensor();

    // Element access via multi-dimensional index, e.g. t(n, c, h, w).
    // Uses strides so it works correctly on transposed/viewed tensors.
    template <typename... Dims> f32 &operator()(Dims... dims) {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data[offset];
    }
    template <typename... Dims> const f32 &operator()(Dims... dims) const {
        u64 offset = 0;
        u32 indices[] = {(u32)dims...};
        for (u32 i = 0; i < sizeof...(dims); i++)
            offset += indices[i] * stride[i];
        return data[offset];
    }
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

// Loads a .npy file (version 1, float32, C-order) into a new Tensor.
// If on_gpu=true the data is transferred to the GPU after loading.
Tensor *tensor_load(const char *filename, b32 on_gpu);
// Returns a new GPU copy of t (no-op copy if already on GPU).
Tensor *tensor_to_gpu(const Tensor *t);
// Returns a new CPU copy of t (no-op copy if already on CPU).
Tensor *tensor_to_cpu(const Tensor *t);

// ---- metadata / shape helpers (device-independent) -----------------------

// Computes row-major strides from shape and writes them into stride.
// stride[ndim-1]=1, stride[i] = shape[i+1]*stride[i+1].
// Returns the total element count (product of all shape values).
// Use this whenever shape changes so strides stay consistent.
u64 tensor_compute_strides(u64 *stride, const u32 *shape, u32 ndim);
// Expands t in-place to expanded_ndim by prepending 1s to shape and
// recomputing strides. Returns false if t->ndim > expanded_ndim.
b32 tensor_expand_shape(Tensor *t, u32 expanded_ndim);
// Returns true if strides are packed row-major (no gaps, no transpositions).
b32 tensor_is_contiguous(const Tensor *t);
// Returns true if both tensors have the same ndim and identical shape.
b32 tensor_shape_eq(const Tensor *a, const Tensor *b);
// Swaps dim0 and dim1 in-place (swaps shape and stride entries). Returns false
// if dims are out of range.
b32 tensor_transpose(Tensor *tensor, u32 dim0, u32 dim1);
// Reinterprets the tensor with a new shape in-place. Total size must be
// unchanged. Returns false on mismatch.
b32 tensor_reshape(Tensor *tensor, const u32 *shape, u32 ndim);
// Computes the broadcast output shape of a and b (NumPy rules).
// Writes result into out_shape and returns the output ndim (0 on incompatible
// shapes). Example: a=[3,1,4], b=[2,4] → out_shape=[3,2,4], returns 3.
u32 broadcast_shape(const Tensor *a, const Tensor *b, u32 *out_shape);
// Writes the shape of t left-padded with 1s to reach expanded_ndim into
// t_expanded_shape. Example: t.shape=[4,5], expanded_ndim=4 →
// t_expanded_shape=[1,1,4,5].
void expanded_shape(const Tensor *t, u32 expanded_ndim, u32 *t_expanded_shape);
// Writes the strides of t left-padded with 0s to reach expanded_ndim into
// t_expanded_stride. Prepended dims and dims where shape==1 get stride 0 so
// indexing repeats the same element (broadcast). Example: t.shape=[1,5],
// t.stride=[5,1], expanded_ndim=3 → [0, 0, 1].
void expanded_stride(const Tensor *t, u32 expanded_ndim,
                     u64 *t_expanded_stride);
// Creates a view of src: same data pointer, same shape/strides,
// owns_data=false.
Tensor *tensor_view(const Tensor *src);
// Allocates a new zero-filled tensor with the same shape and device as src.
Tensor *tensor_create_like(const Tensor *src);
// Prints shape and strides to stdout.
void tensor_print(const Tensor *tensor);

// ---- realloc -------------------------------------------------------------

// Reallocates t's data buffer if the new shape has a different total size or
// ndim. No-op (keeps existing buffer) if the shape already matches.
void tensor_realloc(Tensor *t, const u32 *new_shape, u32 new_ndim);

// ---- copy ----------------------------------------------------------------

// Copies all elements from src into dst. Shapes must match. Handles CPU↔GPU
// transfers.
void tensor_copy(Tensor *dst, const Tensor *src);
// If t is non-contiguous, copies its data into a new contiguous buffer and
// frees the old one (if owned). No-op if already contiguous.
void tensor_contiguous(Tensor *t);

// ---- fill / clear --------------------------------------------------------

// Sets every element to value.
void tensor_fill(Tensor *tensor, f32 value);
// Sets every element to 0. Equivalent to tensor_fill(t, 0).
void tensor_clear(Tensor *tensor);

// ---- activations (relu, exp, log, softmax) --------------------------------
// All elementwise. Two overloads per op:
//   (dst, src) — writes into pre-allocated dst, returns false on shape
//   mismatch. (src)      — allocates and returns a new tensor.

b32 tensor_relu(Tensor *dst, const Tensor *src);
Tensor *tensor_relu(const Tensor *src);
b32 tensor_exp(Tensor *dst, const Tensor *src);
Tensor *tensor_exp(const Tensor *src);
b32 tensor_log(Tensor *dst, const Tensor *src);
Tensor *tensor_log(const Tensor *src);
b32 tensor_sqrt(Tensor *dst, const Tensor *src);
Tensor *tensor_sqrt(const Tensor *src);

// ---- elementwise binary (add / sub / mul / div) --------------------------
// All ops support broadcasting (NumPy rules). Same two-overload pattern as
// activations.

b32 tensor_add(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_add(const Tensor *a, const Tensor *b);
b32 tensor_sub(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_sub(const Tensor *a, const Tensor *b);
b32 tensor_mul(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_mul(const Tensor *a, const Tensor *b);
b32 tensor_div(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_div(const Tensor *a, const Tensor *b);
b32 tensor_equal(Tensor *out, const Tensor *a, const Tensor *b);
Tensor *tensor_equal(const Tensor *a, const Tensor *b);
// Elementwise: out[i] = grad[i] if in[i] > 0 else 0. Backward pass for ReLU.
b32 tensor_relu_backward(Tensor *out, const Tensor *grad, const Tensor *in);
Tensor *tensor_relu_backward(const Tensor *grad, const Tensor *in);
// Row-wise softmax: out[i] = exp(in[i]) / sum(exp(in[row])) for each row.
b32 tensor_softmax(Tensor *out, const Tensor *in);
Tensor *tensor_softmax(const Tensor *in);
// Row-wise log-softmax via log-sum-exp — numerically stable, no log(0) NaN.
b32 tensor_log_softmax(Tensor *out, const Tensor *in);
Tensor *tensor_log_softmax(const Tensor *in);

// ---- scalar operations ---------------------------------------------------
// Same as the binary ops above but broadcasts a scalar across the whole tensor.

b32 tensor_add(Tensor *out, const Tensor *a, f32 scalar);
Tensor *tensor_add(const Tensor *a, f32 scalar);
b32 tensor_sub(Tensor *out, const Tensor *a, f32 scalar);
Tensor *tensor_sub(const Tensor *a, f32 scalar);
b32 tensor_mul(Tensor *out, const Tensor *tensor, f32 scalar);
Tensor *tensor_mul(const Tensor *tensor, f32 scalar);
b32 tensor_div(Tensor *out, const Tensor *a, f32 scalar);
Tensor *tensor_div(const Tensor *a, f32 scalar);

// ---- matrix multiply -----------------------------------------------------
// 2D matmul: out = a @ b. a is [M, K], b is [K, N], out is [M, N].
// clear_out=true zeros out before accumulating (set false to accumulate into
// existing out).

b32 tensor_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                   b32 clear_out = true);
Tensor *tensor_mat_mul(const Tensor *a, const Tensor *b);

// ---- reduction (sum, max, argmax) ----------------------------------------
// No-dim variants reduce the entire tensor to a scalar.
// Dim variants reduce along the specified axis; keep_dim=true keeps that axis
// as size 1.

b32 tensor_sum(Tensor *out, const Tensor *tensor, b32 clear_out = true);
b32 tensor_sum(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim = true,
               b32 clear_out = true);
Tensor *tensor_sum(const Tensor *tensor);
Tensor *tensor_sum(const Tensor *tensor, u32 dim, b32 keep_dim = true);
b32 tensor_max(Tensor *out, const Tensor *tensor);
Tensor *tensor_max(const Tensor *tensor);
b32 tensor_max(Tensor *out, const Tensor *tensor, u32 dim, b32 keep_dim = true);
Tensor *tensor_max(const Tensor *tensor, u32 dim, b32 keep_dim = true);
// Returns indices (as f32) of the max value along dim.
b32 tensor_argmax(Tensor *out, const Tensor *tensor, u32 dim,
                  b32 keep_dim = true);
Tensor *tensor_argmax(const Tensor *tensor, u32 dim, b32 keep_dim = true);

b32 tensor_welford_mean_var(Tensor *mean, Tensor *var, const Tensor *src,
                            u32 dim);

// ---- scattering ----------------------------------------------------------

b32 tensor_scatter_add(Tensor *out, const Tensor *src, const Tensor *indices,
                       u32 dim);
Tensor *tensor_scatter_add(const Tensor *src, const Tensor *indices, u32 dim,
                           u32 dim_size);

// ---- initializing --------------------------------------------------------
// He (Kaiming) normal init: fills with values ~ N(0, sqrt(2 / fan_in)).
// fan_in is the product of all dims except the last (output features).
void tensor_he_init(Tensor *tensor);

// ---- indexing ------------------------------------------------------------
// Gathers n_indices slices from src along dim using the indices array.
// Equivalent to: dst = src[indices, :] for dim=0.
// Example: src=[5,3], indices=[2,0,4], dim=0 → dst=[3,3] (rows 2, 0, 4).

b32 tensor_index_select(Tensor *dst, const Tensor *src, const u32 *indices,
                        u32 n_indices, u32 dim);
Tensor *tensor_index_select(const Tensor *src, const u32 *indices,
                            u32 n_indices, u32 dim);
// ---- spatial / patch operations ------------------------------------------

b32 tensor_unfold2d(Tensor *out, const Tensor *input, Unfold2dParams params);
Tensor *tensor_unfold2d(const Tensor *input, Unfold2dParams params);

// Inverse of unfold2d: scatter-adds col [N*L, C*kH*kW] back into dst [N,C,H,W].
// dst must be zeroed before calling. params must match the forward unfold call.
b32 tensor_fold2d(Tensor *dst, const Tensor *col, Unfold2dParams params);

// ---- comparison ----------------------------------------------------------

// Returns true if a and b have the same shape and every element pair differs
// by at most tol. Both tensors must be on CPU.
b32 tensor_equals(const Tensor *a, const Tensor *b, f32 tol = 1e-5f);
#endif
