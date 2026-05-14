#include "../include/tensor.hpp"
#include <cstdio>
#include <cmath>

static bool approx(float a, float b, float tol = 1e-4f) {
    return fabsf(a - b) <= tol;
}

static int failures = 0;

static void check(bool ok, const char *msg) {
    if (!ok) {
        printf("FAIL: %s\n", msg);
        failures++;
    } else {
        printf("PASS: %s\n", msg);
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

static Tensor<f32> cpu_arange(u32 ndim, const u32 *shape) {
    Tensor<f32> t = Tensor<f32>::make(ndim, shape, false);
    for (u64 i = 0; i < t->numel(); i++)
        t->data()[i] = (f32)i;
    return t;
}

// ─── tests ────────────────────────────────────────────────────────────────────

// (3,4) reduce axes {0} → shape (4,): column sums
static void test_2d_reduce_axis0() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // arange: 0..11
    // col j sum = j + (j+4) + (j+8) = 3j+12
    u32 axes[] = {0};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 1u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = true;
    for (u32 j = 0; j < 4; j++)
        ok &= approx(cpu->data()[j], 3.f*j + 12.f);
    check(ok, "2D reduce axis=0, keep_dim=false → shape (4,)");
}

// (3,4) reduce axes {1} → shape (3,): row sums
static void test_2d_reduce_axis1() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // row i: 4i, 4i+1, 4i+2, 4i+3 → sum = 16i+6
    u32 axes[] = {1};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 1u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = true;
    for (u32 i = 0; i < 3; i++)
        ok &= approx(cpu->data()[i], 16.f*i + 6.f);
    check(ok, "2D reduce axis=1, keep_dim=false → shape (3,)");
}

// (3,4) reduce all axes → scalar
static void test_2d_reduce_all() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // sum 0..11 = 66
    u32 axes[] = {0, 1};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 2u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);
    check(approx(cpu->data()[0], 66.f), "2D reduce all axes → scalar 66");
}

// (2,3,4) reduce axes {0,2} → shape (3,)
static void test_3d_reduce_outer_inner() {
    u32 shape[] = {2, 3, 4};
    Tensor<f32> t = cpu_arange(3, shape);
    // out[c] = sum over b in [0,2), w in [0,4) of t[b,c,w]
    //        = sum_b sum_w (b*12 + c*4 + w)
    //        = sum_b (2*b*12 + c*4*4 + 6) [sum_w w = 6, sum_w 1 = 4]
    // Wait let me just compute it directly.
    // t[b,c,w] = b*12 + c*4 + w (row-major)
    // out[c] = sum_{b=0}^{1} sum_{w=0}^{3} (b*12 + c*4 + w)
    //        = sum_b [ 4*(b*12 + c*4) + (0+1+2+3) ]
    //        = sum_b [ 4*b*12 + 4*c*4 + 6 ]
    //        = sum_b [ 48b + 16c + 6 ]
    //        = (48*0+48*1) + 2*(16c+6)
    //        = 48 + 32c + 12
    //        = 60 + 32c
    u32 axes[] = {0, 2};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 2u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 60.f + 32.f*c);
    check(ok, "3D reduce axes={0,2} → shape (3,)");
}

// (2,3,4,5) reduce axes {0,2,3} → shape (3,) -- the BN-style case
static void test_4d_bnstyle() {
    u32 shape[] = {2, 3, 4, 5};
    Tensor<f32> t = cpu_arange(4, shape);
    // t[n,c,h,w] = n*60 + c*20 + h*5 + w
    // out[c] = sum_{n,h,w} t[n,c,h,w]
    //        = N*H*W*c*20 + (sum_n n)*H*W*60 + N*(sum_h h)*W*5 + N*H*(sum_w w)
    // N=2,H=4,W=5: NHW=40
    // sum_n n = 0+1 = 1 (for N=2)
    // sum_h h = 0+1+2+3 = 6 (for H=4)
    // sum_w w = 0+1+2+3+4 = 10 (for W=5)
    // out[c] = 40*c*20 + 1*20*60 + 2*6*5*5 + 2*4*10
    //        = 800c + 1200 + 300 + 80
    //        = 800c + 1580
    u32 axes[] = {0, 2, 3};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 3u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 800.f*c + 1580.f);
    check(ok, "4D reduce axes={0,2,3} → shape (3,) (BN-style)");
}

// keep_dim=true: shape stays same with 1s at reduced positions
static void test_keep_dim() {
    u32 shape[] = {2, 3, 4};
    Tensor<f32> t = cpu_arange(3, shape);
    u32 axes[] = {0, 2};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 2u, /*keep_dim*/true);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 3 &&
               cpu->shape[0] == 1 &&
               cpu->shape[1] == 3 &&
               cpu->shape[2] == 1);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 60.f + 32.f*c);
    check(ok, "3D reduce axes={0,2} keep_dim=true → shape (1,3,1)");
}

// Single-axis reduction (exercises existing path via new API)
static void test_single_axis_via_new_api() {
    u32 shape[] = {4, 6};
    Tensor<f32> t = cpu_arange(2, shape);
    // sum along axis 1: row sums: 15, 51, 87, 123 (rows of 6 elements)
    u32 axes[] = {1};
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), axes, 1u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 4);
    float expected[] = {15.f, 51.f, 87.f, 123.f};
    for (u32 i = 0; i < 4; i++)
        ok &= approx(cpu->data()[i], expected[i]);
    check(ok, "single-axis via new API: (4,6) reduce axis=1 → (4,)");
}

// Large reduction: (1, 1, 1024, 1024) reduce {2,3} → (1,1)
// Verifies hierarchical two-phase chunking.
static void test_large_r_size() {
    u32 shape[] = {1, 1, 1024, 1024};
    u64 n = 1024ull * 1024;
    Tensor<f32> cpu_t = Tensor<f32>::make(4, shape, false);
    // Fill with 1.0 so sum = n
    for (u64 i = 0; i < n; i++)
        cpu_t->data()[i] = 1.0f;
    Tensor<f32> gpu_t = tensor_to_gpu(cpu_t);

    u32 axes[] = {2, 3};
    Tensor<f32> out = tensor_sum(gpu_t, axes, 2u, /*keep_dim*/false);
    Tensor<f32> cpu_out = tensor_to_cpu(out);

    check(approx(cpu_out->data()[0], (f32)n, 1.0f),
          "large R: (1,1,1024,1024) reduce {2,3} → sum=1048576");
}

int main() {
    test_2d_reduce_axis0();
    test_2d_reduce_axis1();
    test_2d_reduce_all();
    test_3d_reduce_outer_inner();
    test_4d_bnstyle();
    test_keep_dim();
    test_single_axis_via_new_api();
    test_large_r_size();

    printf("\n%s — %d failure(s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
