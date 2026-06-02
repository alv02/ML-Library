#include "../include/tensor.hpp"
#include <cstdio>
#include <cmath>

static bool approx(f32 a, f32 b, f32 tol = 1e-4f) {
    return fabsf(a - b) <= tol;
}

static i32 failures = 0;

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

// (3,4) reduce axis 0 → shape (4,): column sums
static void test_2d_reduce_axis0() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // col j sum = j + (j+4) + (j+8) = 3j+12
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), 0u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 4);
    for (u32 j = 0; j < 4; j++)
        ok &= approx(cpu->data()[j], 3.f*j + 12.f);
    check(ok, "2D reduce axis=0, keep_dim=false → shape (4,)");
}

// (3,4) reduce axis 1 → shape (3,): row sums
static void test_2d_reduce_axis1() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // row i: 4i, 4i+1, 4i+2, 4i+3 → sum = 16i+6
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), 1u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 i = 0; i < 3; i++)
        ok &= approx(cpu->data()[i], 16.f*i + 6.f);
    check(ok, "2D reduce axis=1, keep_dim=false → shape (3,)");
}

// (3,4) reduce all → scalar
static void test_2d_reduce_all() {
    u32 shape[] = {3, 4};
    Tensor<f32> t = cpu_arange(2, shape);
    // sum 0..11 = 66
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t));
    Tensor<f32> cpu = tensor_to_cpu(out);
    check(approx(cpu->data()[0], 66.f), "2D reduce all axes → scalar 66");
}

// (2,3,4) reduce axes {0,2} → shape (3,): use skip dim=1
static void test_3d_reduce_outer_inner() {
    u32 shape[] = {2, 3, 4};
    Tensor<f32> t = cpu_arange(3, shape);
    // out[c] = sum_{b,w} t[b,c,w] = 60 + 32c
    Tensor<f32> out = tensor_sum_skip(tensor_to_gpu(t), 1u);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 60.f + 32.f*c);
    check(ok, "3D reduce axes={0,2} via skip dim=1 → shape (3,)");
}

// (2,3,4,5) reduce axes {0,2,3} → shape (3,): BN-style, use skip dim=1
static void test_4d_bnstyle() {
    u32 shape[] = {2, 3, 4, 5};
    Tensor<f32> t = cpu_arange(4, shape);
    // out[c] = 800c + 1580
    Tensor<f32> out = tensor_sum_skip(tensor_to_gpu(t), 1u);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 800.f*c + 1580.f);
    check(ok, "4D reduce axes={0,2,3} via skip dim=1 → shape (3,) (BN-style)");
}

// keep_dim=true: (2,3,4) reduce axis=0, shape stays (1,3,4)
static void test_keep_dim() {
    u32 shape[] = {2, 3, 4};
    Tensor<f32> t = cpu_arange(3, shape);
    // reduce axis 0: out[0,c,w] = t[0,c,w] + t[1,c,w]
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), 0u, /*keep_dim*/true);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 3 && cpu->shape[0] == 1 && cpu->shape[1] == 3 && cpu->shape[2] == 4);
    // t[b,c,w] = b*12 + c*4 + w; out[0,c,w] = c*4+w + (12+c*4+w) = 12 + 2*(c*4+w)
    for (u32 c = 0; c < 3 && ok; c++)
        for (u32 w = 0; w < 4 && ok; w++)
            ok &= approx(cpu->data()[c*4+w], 12.f + 2.f*(c*4+w));
    check(ok, "3D reduce axis=0 keep_dim=true → shape (1,3,4)");
}

// (4,6) reduce axis 1 → (4,)
static void test_single_dim() {
    u32 shape[] = {4, 6};
    Tensor<f32> t = cpu_arange(2, shape);
    // row sums: 15, 51, 87, 123
    Tensor<f32> out = tensor_sum(tensor_to_gpu(t), 1u, /*keep_dim*/false);
    Tensor<f32> cpu = tensor_to_cpu(out);

    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 4);
    f32 expected[] = {15.f, 51.f, 87.f, 123.f};
    for (u32 i = 0; i < 4; i++)
        ok &= approx(cpu->data()[i], expected[i]);
    check(ok, "single-dim: (4,6) reduce axis=1 → (4,)");
}

// Large: (1,1,1024,1024) reduce all → sum=1048576
static void test_large_global() {
    u32 shape[] = {1, 1, 1024, 1024};
    u64 n = 1024ull * 1024;
    Tensor<f32> cpu_t = Tensor<f32>::make(4, shape, false);
    for (u64 i = 0; i < n; i++)
        cpu_t->data()[i] = 1.0f;
    Tensor<f32> gpu_t = tensor_to_gpu(cpu_t);

    Tensor<f32> out = tensor_sum(gpu_t);
    Tensor<f32> cpu_out = tensor_to_cpu(out);

    check(approx(cpu_out->data()[0], (f32)n, 1.0f),
          "large global: (1,1,1024,1024) → sum=1048576");
}

// skip: (2,3,4) skip dim=1 matches sequential per-channel sum
static void test_skip_correctness() {
    u32 shape[] = {2, 3, 4};
    Tensor<f32> t = cpu_arange(3, shape);
    Tensor<f32> out = tensor_sum_skip(tensor_to_gpu(t), 1u);
    Tensor<f32> cpu = tensor_to_cpu(out);

    // same values as test_3d_reduce_outer_inner
    bool ok = (cpu->ndim == 1 && cpu->shape[0] == 3);
    for (u32 c = 0; c < 3; c++)
        ok &= approx(cpu->data()[c], 60.f + 32.f*c);
    check(ok, "skip dim=1: (2,3,4) → shape (3,), values 60+32c");
}

i32 main() {
    test_2d_reduce_axis0();
    test_2d_reduce_axis1();
    test_2d_reduce_all();
    test_3d_reduce_outer_inner();
    test_4d_bnstyle();
    test_keep_dim();
    test_single_dim();
    test_large_global();
    test_skip_correctness();

    printf("\n%s — %d failure(s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
