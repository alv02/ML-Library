#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include "../tensor.hpp"

// Works as both a __device__ __forceinline__ (nvcc) and a plain inline (g++).
#ifdef __CUDACC__
#define RED_DEVICE __device__ __forceinline__
#else
#define RED_DEVICE inline
#endif

// ── ReductionPlan ─────────────────────────────────────────────────────────────
//
// Axes are partitioned into K (kept → output elements) and R (reduced).
// Both sub-spaces carry their own shape, their strides into the original input
// tensor, and precomputed row-major strides for flat→coordinate decode.
//
// Build one with make_reduction_plan(); use flat_to_offset() inside kernels.

struct ReductionPlan {
    u32 k_ndim, r_ndim;

    u32 k_shape[MAX_NDIM];    // size of each kept axis
    u64 k_stride[MAX_NDIM];   // input tensor strides for kept axes
    u64 k_cstride[MAX_NDIM];  // row-major strides over k_shape (flat→coord)

    u32 r_shape[MAX_NDIM];    // size of each reduced axis
    u64 r_stride[MAX_NDIM];   // input tensor strides for reduced axes
    u64 r_cstride[MAX_NDIM];  // row-major strides over r_shape (flat→coord)

    u64 k_size;   // product(k_shape) = number of output elements
    u64 r_size;   // product(r_shape) = elements to reduce per output
};

// Build a ReductionPlan from a tensor and an explicit list of axes to reduce.
// Kept axes appear in their original order in k_*, reduced axes in r_*.
template <typename T>
static ReductionPlan make_reduction_plan(const TensorImpl<T> &in,
                                          const u32 *axes, u32 n_axes) {
    ReductionPlan p = {};
    bool is_reduced[MAX_NDIM] = {};
    for (u32 i = 0; i < n_axes; i++)
        is_reduced[axes[i]] = true;

    u32 ki = 0, ri = 0;
    for (u32 d = 0; d < in.ndim; d++) {
        if (is_reduced[d]) {
            p.r_shape[ri]  = in.shape[d];
            p.r_stride[ri] = in.stride[d];
            ri++;
        } else {
            p.k_shape[ki]  = in.shape[d];
            p.k_stride[ki] = in.stride[d];
            ki++;
        }
    }
    p.k_ndim = ki;
    p.r_ndim = ri;

    tensor_compute_strides(p.k_cstride, p.k_shape, p.k_ndim);
    tensor_compute_strides(p.r_cstride, p.r_shape, p.r_ndim);

    p.k_size = 1;
    for (u32 i = 0; i < p.k_ndim; i++) p.k_size *= p.k_shape[i];
    p.r_size = 1;
    for (u32 i = 0; i < p.r_ndim; i++) p.r_size *= p.r_shape[i];

    return p;
}

// Decode a flat index inside one sub-space (K or R) into a byte-offset into
// the original input tensor.
//   flat      — index in [0, subspace_size)
//   shape     — axis sizes for this sub-space (k_shape or r_shape)
//   cstride   — precomputed row-major strides for shape[] (k_cstride/r_cstride)
//   in_stride — actual input strides for these axes (k_stride or r_stride)
//   ndim      — number of axes in this sub-space
RED_DEVICE u64 flat_to_offset(u64 flat, const u32 *shape, const u64 *cstride,
                               const u64 *in_stride, u32 ndim) {
    u64 remaining = flat;
    u64 offset = 0;
    for (u32 i = 0; i < ndim; i++) {
        u64 coord = remaining / cstride[i];
        remaining -= coord * cstride[i];
        offset += coord * in_stride[i];
    }
    return offset;
}

#endif // REDUCTION_HPP
