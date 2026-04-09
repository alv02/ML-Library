#include "../include/tensor.hpp"
#include <cstdio>

// ── helpers ──────────────────────────────────────────────────────────────────

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int passed = 0, failed = 0;

static void check(const char *name, bool ok) {
    printf("  [%s%s%s] %s\n", ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET,
           name);
    ok ? passed++ : failed++;
}

static bool shapes_eq(const u32 *got, const u32 *expected, u32 ndim) {
    for (u32 i = 0; i < ndim; i++)
        if (got[i] != expected[i])
            return false;
    return true;
}

static Tensor *make(u32 ndim, u32 *shape) {
    return new Tensor(ndim, shape, false);
}

// ── broadcast_shape tests
// ─────────────────────────────────────────────────────

static void test_broadcast_shape() {
    printf("\n-- broadcast_shape --\n");
    u32 out[4];
    u32 ndim;

    // identical shapes
    {
        u32 sa[2] = {5, 3}, sb[2] = {5, 3};
        Tensor *a = make(2, sa), *b = make(2, sb);
        ndim = broadcast_shape(a, b, out);
        u32 ex[2] = {5, 3};
        check("(5,3)+(5,3) → (5,3)", ndim == 2 && shapes_eq(out, ex, 2));
        delete a;
        delete b;
    }

    // broadcast along dim 0
    {
        u32 sa[2] = {5, 3}, sb[2] = {1, 3};
        Tensor *a = make(2, sa), *b = make(2, sb);
        ndim = broadcast_shape(a, b, out);
        u32 ex[2] = {5, 3};
        check("(5,3)+(1,3) → (5,3)", ndim == 2 && shapes_eq(out, ex, 2));
        delete a;
        delete b;
    }

    // broadcast along dim 1
    {
        u32 sa[2] = {5, 3}, sb[2] = {5, 1};
        Tensor *a = make(2, sa), *b = make(2, sb);
        ndim = broadcast_shape(a, b, out);
        u32 ex[2] = {5, 3};
        check("(5,3)+(5,1) → (5,3)", ndim == 2 && shapes_eq(out, ex, 2));
        delete a;
        delete b;
    }

    // scalar (1,1)
    {
        u32 sa[2] = {5, 3}, sb[2] = {1, 1};
        Tensor *a = make(2, sa), *b = make(2, sb);
        ndim = broadcast_shape(a, b, out);
        u32 ex[2] = {5, 3};
        check("(5,3)+(1,1) → (5,3)", ndim == 2 && shapes_eq(out, ex, 2));
        delete a;
        delete b;
    }

    // incompatible shapes → should return 0
    {
        u32 sa[2] = {5, 3}, sb[2] = {4, 3};
        Tensor *a = make(2, sa), *b = make(2, sb);
        ndim = broadcast_shape(a, b, out);
        check("(5,3)+(4,3) → error (0)", ndim == 0);
        delete a;
        delete b;
    }
}

// ── broadcast_strides tests
// ───────────────────────────────────────────────────

static bool strides_eq(const u64 *got, const u64 *expected, u32 ndim) {
    for (u32 i = 0; i < ndim; i++)
        if ((u64)got[i] != expected[i])
            return false;
    return true;
}

static void test_broadcast_strides() {
    printf("\n-- broadcast_strides --\n");
    u64 bstrides[MAX_NDIM];

    // no broadcast — (3,4) into (3,4): strides pass through unchanged
    {
        u32 s[2] = {3, 4};
        Tensor *t = make(2, s);
        // t->stride = [4, 1]
        u32 out_shape[2] = {3, 4};
        expanded_stride(t, out_shape, 2, bstrides);
        u64 expected[2] = {t->stride[0], t->stride[1]}; // [4, 1]
        check("(3,4)→(3,4) strides unchanged",
              strides_eq(bstrides, expected, 2));
        delete t;
    }

    // broadcast dim 0 — (1,4) into (3,4): stride[0] becomes 0
    {
        u32 s[2] = {1, 4};
        Tensor *t = make(2, s);
        // t->stride = [4, 1]
        u32 out_shape[2] = {3, 4};
        expanded_stride(t, out_shape, 2, bstrides);
        u64 expected[2] = {0, t->stride[1]}; // [0, 1]
        check("(1,4)→(3,4) dim0 stride=0", strides_eq(bstrides, expected, 2));
        delete t;
    }

    // broadcast dim 1 — (3,1) into (3,4): stride[1] becomes 0
    {
        u32 s[2] = {3, 1};
        Tensor *t = make(2, s);
        // t->stride = [1, 1]
        u32 out_shape[2] = {3, 4};
        expanded_stride(t, out_shape, 2, bstrides);
        u64 expected[2] = {t->stride[0], 0}; // [1, 0]
        check("(3,1)→(3,4) dim1 stride=0", strides_eq(bstrides, expected, 2));
        delete t;
    }

    // broadcast all — (1,1) into (3,4): all strides 0
    {
        u32 s[2] = {1, 1};
        Tensor *t = make(2, s);
        u32 out_shape[2] = {3, 4};
        expanded_stride(t, out_shape, 2, bstrides);
        u64 expected[2] = {0, 0};
        check("(1,1)→(3,4) all strides=0", strides_eq(bstrides, expected, 2));
        delete t;
    }

    // different ndim — (4,) into (3,4): right-aligned, prepended dim gets 0
    {
        u32 s[1] = {4};
        Tensor *t = make(1, s);
        // t->stride = [1]
        u32 out_shape[2] = {3, 4};
        expanded_stride(t, out_shape, 2, bstrides);
        // (4,) right-aligns to dim 1 of (3,4): bstrides = [0, t->stride[0]]
        u64 expected[2] = {0, t->stride[0]}; // [0, 1]
        check("(4,)→(3,4) prepended dim gets 0",
              strides_eq(bstrides, expected, 2));
        delete t;
    }
}

// ── main
// ──────────────────────────────────────────────────────────────────────

int main() {
    test_broadcast_shape();
    test_broadcast_strides();
    //   test_tensor_add_bcast();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
