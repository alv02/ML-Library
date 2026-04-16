#ifndef TENSOR_CUDA_HPP
#define TENSOR_CUDA_HPP

#include "../tensor.hpp"

#ifdef __CUDACC__
#define ML_DEVICE __device__
#else
#define ML_DEVICE
#endif

// GPU-side mirror of a Tensor's shape and stride metadata.
// Copied to the device so kernels can do multi-dimensional indexing without
// accessing the host-side Tensor struct.
struct TensorMeta {
    u64 size;              // total number of elements
    u32 ndim;              // number of dimensions
    u32 shape[MAX_NDIM];   // size of each dimension
    u64 stride[MAX_NDIM];  // stride per dimension (0 = broadcast that dim)

    // Copies shape and strides directly from t.
    TensorMeta(const Tensor *t);
    // Broadcast-expanded constructor: left-pads shape with 1s and strides with 0s
    // to reach bcast_ndim. Dims where t->shape==1 also get stride=0 so kernels
    // using offset_from() automatically repeat the same element there.
    TensorMeta(const Tensor *t, const u32 *bcast_shape, u32 bcast_ndim);

    ML_DEVICE u32 rows() const { return shape[ndim - 2]; }
    ML_DEVICE u32 cols() const { return shape[ndim - 1]; }
    // Returns the flat offset for the element at (row, col) using the last two dims.
    ML_DEVICE u64 at(u64 row, u64 col) const {
        return row * stride[ndim - 2] + col * stride[ndim - 1];
    }
    // Maps a flat output index to a flat source offset, handling broadcasting.
    // Uses *this* tensor's strides as divisors to decompose flat_idx into
    // per-dimension indices, then re-indexes with src.stride.
    // Because src may have stride=0 on broadcast dims, those dimensions always
    // resolve to offset 0 regardless of the index — same element is reused.
    // Example: out is [3,2,4], src is [1,2,4] (broadcast on dim 0).
    //   flat_idx=9 → out divides to coords (1,0,1) → src offset = 0+0+1 = 1.
    ML_DEVICE u64 offset_from(u64 flat_idx, const TensorMeta &src) const {
        u64 remaining = flat_idx;
        u64 offset = 0;
        for (u32 i = 0; i < ndim; i++) {
            u64 idx_i = remaining / stride[i];
            remaining -= idx_i * stride[i];
            offset += idx_i * src.stride[i];
        }
        return offset;
    }
};

// ---- memory management (alloc / free / transfers) ------------------------

void tensor_cuda_alloc(Tensor *tensor);
void tensor_cuda_free(Tensor *tensor);
Tensor *tensor_cuda_to_gpu(const Tensor *t_cpu);
Tensor *tensor_cuda_to_cpu(const Tensor *t_gpu);
Tensor *tensor_cuda_copy(const Tensor *t_gpu);

// ---- copy (into existing tensor) ----------------------------------------

void tensor_cuda_copy(Tensor *dst, const Tensor *src);

// ---- fill / clear --------------------------------------------------------

void tensor_cuda_fill(Tensor *tensor, f32 value);
void tensor_cuda_clear(Tensor *tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cuda_relu(Tensor *dst, const Tensor *src);
void tensor_cuda_exp(Tensor *dst, const Tensor *src);
void tensor_cuda_log(Tensor *dst, const Tensor *src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cuda_add(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_sub(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_mul(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_div(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_equal(Tensor *out, const Tensor *a, const Tensor *b);
void tensor_cuda_relu_backward(Tensor *out, const Tensor *grad,
                               const Tensor *in);

// ---- scalar operations ---------------------------------------------------

void tensor_cuda_add(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_sub(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_mul(Tensor *out, const Tensor *tensor, f32 scalar);
void tensor_cuda_div(Tensor *out, const Tensor *tensor, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cuda_mat_mul(Tensor *out, const Tensor *a, const Tensor *b,
                         b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

void tensor_cuda_sum(Tensor *out, const Tensor *tensor);
void tensor_cuda_sum(Tensor *out, const Tensor *tensor, u32 dim);
void tensor_cuda_max(Tensor *out, const Tensor *tensor);
void tensor_cuda_max(Tensor *out, const Tensor *tensor, u32 dim);
void tensor_cuda_argmax(Tensor *out, const Tensor *tensor, u32 dim);

// ---- initializing --------------------------------------------------------

void tensor_cuda_he_init(Tensor *tensor);

// ---- indexing ------------------------------------------------------------

void tensor_cuda_index_select(Tensor *dst, const Tensor *src,
                              const u32 *indices, u32 n_indices, u32 dim);

#endif // TENSOR_CUDA_HPP
