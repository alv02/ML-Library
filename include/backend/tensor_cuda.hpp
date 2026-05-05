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
    u64 size;             // total number of elements
    u32 ndim;             // number of dimensions
    u32 shape[MAX_NDIM];  // size of each dimension
    u64 stride[MAX_NDIM]; // stride per dimension (0 = broadcast that dim)

    TensorMeta() = default;
    // Copies shape and strides directly from t.
    TensorMeta(const TensorImpl &t);
    // Broadcast-expanded constructor: left-pads shape with 1s and strides with
    // 0s to reach bcast_ndim. Dims where t->shape==1 also get stride=0 so
    // kernels using offset_from() automatically repeat the same element there.
    TensorMeta(const TensorImpl &t, const u32 *bcast_shape, u32 bcast_ndim);

    ML_DEVICE u32 rows() const { return shape[ndim - 2]; }
    ML_DEVICE u32 cols() const { return shape[ndim - 1]; }
    // Returns the flat offset for the element at (row, col) using the last two
    // dims.
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

// ---- memory management transfers) ----------------------------------------
void tensor_cuda_copy(TensorImpl &dst, const TensorImpl &src);
void tensor_cuda_contiguous(TensorImpl &t, CudaMemArena *arena = nullptr);
// ---- fill / clear --------------------------------------------------------

void tensor_cuda_fill(TensorImpl &tensor, f32 value);
void tensor_cuda_clear(TensorImpl &tensor);

// ---- activations (relu, exp) ---------------------------------------------

void tensor_cuda_relu(TensorImpl &dst, const TensorImpl &src);
void tensor_cuda_exp(TensorImpl &dst, const TensorImpl &src);
void tensor_cuda_log(TensorImpl &dst, const TensorImpl &src);
void tensor_cuda_sqrt(TensorImpl &dst, const TensorImpl &src);

// ---- elementwise binary (add / sub / mul / div) --------------------------

void tensor_cuda_add(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cuda_sub(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cuda_mul(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cuda_div(TensorImpl &out, const TensorImpl &a, const TensorImpl &b);
void tensor_cuda_equal(TensorImpl &out, const TensorImpl &a,
                       const TensorImpl &b);
void tensor_cuda_relu_backward(TensorImpl &out, const TensorImpl &grad,
                               const TensorImpl &in);

// ---- scalar operations ---------------------------------------------------

void tensor_cuda_add(TensorImpl &out, const TensorImpl &tensor, f32 scalar);
void tensor_cuda_sub(TensorImpl &out, const TensorImpl &tensor, f32 scalar);
void tensor_cuda_mul(TensorImpl &out, const TensorImpl &tensor, f32 scalar);
void tensor_cuda_div(TensorImpl &out, const TensorImpl &tensor, f32 scalar);

// ---- matrix multiply -----------------------------------------------------

void tensor_cuda_mat_mul(TensorImpl &out, const TensorImpl &a,
                         const TensorImpl &b, b32 clear_out);

// ---- reduction (sum, max, argmax) ----------------------------------------

void tensor_cuda_sum(TensorImpl &out, const TensorImpl &tensor);
void tensor_cuda_sum(TensorImpl &out, const TensorImpl &tensor, u32 dim);
void tensor_cuda_welford_mean_var(TensorImpl &mean, TensorImpl &var,
                                  const TensorImpl &src, u32 dim);
void tensor_cuda_max(TensorImpl &out, const TensorImpl &tensor);
void tensor_cuda_max(TensorImpl &out, const TensorImpl &tensor, u32 dim);
void tensor_cuda_argmax(TensorImpl &out, const TensorImpl &tensor, u32 dim);

// ---- scattering ----------------------------------------------------------
void tensor_cuda_scatter_add(TensorImpl &out, const TensorImpl &src,
                             const TensorImpl &indices, u32 dim);
// ---- initializing --------------------------------------------------------

void tensor_cuda_he_init(TensorImpl &tensor);

// ---- indexing ------------------------------------------------------------

void tensor_cuda_index_select(TensorImpl &dst, const TensorImpl &src,
                              const u32 *indices, u32 n_indices, u32 dim);

void tensor_cuda_unfold2d(TensorImpl &dst, const TensorImpl &src,
                          Unfold2dParams params);
void tensor_cuda_fold2d(TensorImpl &dst, const TensorImpl &col,
                        Unfold2dParams params);

// ---- comparison ----------------------------------------------------------

b32 tensor_cuda_equals(const TensorImpl &a, const TensorImpl &b, f32 tol);

#endif // TENSOR_CUDA_HPP
