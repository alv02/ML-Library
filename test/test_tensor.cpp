#include "../include/tensor.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static b32 g_on_gpu = false;
static int passed = 0, failed = 0;

static void sync() {
    if (g_on_gpu)
        cudaDeviceSynchronize();
}

static void check(const char *name, bool ok) {
    printf("  [%s%s%s] %s\n", ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET, name);
    ok ? passed++ : failed++;
}

static bool tensors_close(const Tensor *got, const Tensor *expected, f32 tol = 1e-5f) {
    if (got->size != expected->size) return false;
    for (u64 i = 0; i < got->size; i++) {
        if (fabsf(got->data[i] - expected->data[i]) > tol) {
            printf("    mismatch at [%llu]: got %.6f  expected %.6f\n",
                   (unsigned long long)i, got->data[i], expected->data[i]);
            return false;
        }
    }
    return true;
}

static void test_add(const char *name, const char *dir) {
    char pa[256], pb[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pb,   sizeof(pb),   "../data/test/%s/b.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = tensor_load(pa,   g_on_gpu);
    Tensor *b   = tensor_load(pb,   g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete b; delete exp;
        return;
    }

    Tensor *out = tensor_add(a, b);
    sync();

    Tensor *out_cpu = tensor_to_cpu(out);
    check(name, tensors_close(out_cpu, exp));

    delete a; delete b; delete exp; delete out; delete out_cpu;
}

static void test_mat_mul(const char *name, const char *dir, bool trans_a = false) {
    char pa[256], pb[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pb,   sizeof(pb),   "../data/test/%s/b.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = tensor_load(pa,   g_on_gpu);
    Tensor *b   = tensor_load(pb,   g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete b; delete exp;
        return;
    }

    if (trans_a)
        tensor_transpose(a, 0, 1);

    Tensor *out = tensor_mat_mul(a, b);
    sync();

    Tensor *out_cpu = tensor_to_cpu(out);
    check(name, tensors_close(out_cpu, exp, 1e-4f));

    delete a; delete b; delete exp; delete out; delete out_cpu;
}

static void test_sum(const char *name, const char *dir) {
    char pa[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = tensor_load(pa,   g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete exp;
        return;
    }

    Tensor *out = tensor_sum(a);
    sync();

    Tensor *out_cpu = tensor_to_cpu(out);
    check(name, tensors_close(out_cpu, exp, 1e-3f));

    delete a; delete exp; delete out; delete out_cpu;
}

static void test_sum_dim(const char *name, const char *dir, u32 dim, b32 keep_dim) {
    char pa[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = tensor_load(pa,   g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete exp;
        return;
    }

    Tensor *out = tensor_sum(a, dim, keep_dim);
    sync();

    Tensor *out_cpu = tensor_to_cpu(out);

    bool data_ok  = tensors_close(out_cpu, exp, 1e-4f);
    bool ndim_ok  = (out_cpu->ndim == exp->ndim);
    bool shape_ok = true;
    for (u32 i = 0; i < exp->ndim && shape_ok; i++)
        shape_ok = (out_cpu->shape[i] == exp->shape[i]);

    if (!ndim_ok)
        printf("    ndim mismatch: got %u expected %u\n", out_cpu->ndim, exp->ndim);
    if (!shape_ok) {
        printf("    shape mismatch: got [");
        for (u32 i = 0; i < out_cpu->ndim; i++)
            printf("%u%s", out_cpu->shape[i], i+1 < out_cpu->ndim ? "," : "");
        printf("] expected [");
        for (u32 i = 0; i < exp->ndim; i++)
            printf("%u%s", exp->shape[i], i+1 < exp->ndim ? "," : "");
        printf("]\n");
    }

    check(name, data_ok && ndim_ok && shape_ok);

    delete a; delete exp; delete out; delete out_cpu;
}

static void test_index_select(const char *name, const char *dir,
                              const u32 *indices, u32 n_indices, u32 dim) {
    char pa[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = tensor_load(pa,   g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete exp;
        return;
    }

    Tensor *out = tensor_index_select(a, indices, n_indices, dim);
    if (!out) {
        printf("  [%sFAIL%s] %s — tensor_index_select returned nullptr\n", RED, RESET, name);
        failed++;
        delete a; delete exp;
        return;
    }
    sync();

    Tensor *out_cpu = tensor_to_cpu(out);

    bool shape_ok = (out_cpu->ndim == exp->ndim);
    for (u32 i = 0; i < exp->ndim && shape_ok; i++)
        shape_ok = (out_cpu->shape[i] == exp->shape[i]);

    if (!shape_ok) {
        printf("    shape mismatch: got [");
        for (u32 i = 0; i < out_cpu->ndim; i++)
            printf("%u%s", out_cpu->shape[i], i+1 < out_cpu->ndim ? "," : "");
        printf("] expected [");
        for (u32 i = 0; i < exp->ndim; i++)
            printf("%u%s", exp->shape[i], i+1 < exp->ndim ? "," : "");
        printf("]\n");
    }

    check(name, shape_ok && tensors_close(out_cpu, exp));

    delete a; delete exp; delete out; delete out_cpu;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0) g_on_gpu = true;
        else if (strcmp(argv[i], "--cpu") == 0) g_on_gpu = false;
    }

    const char *backend = g_on_gpu ? "CUDA" : "CPU";
    printf("\nBackend: %s\n", backend);

    printf("\n-- tensor_add --\n");
    test_add("same shape   (3,4)+(3,4)",      "add_same");
    test_add("bcast row    (3,4)+(1,4)",       "add_bcast_row");
    test_add("bcast col    (3,4)+(3,1)",       "add_bcast_col");
    test_add("bcast scalar (3,4)+(1,1)",       "add_bcast_scalar");
    test_add("diff ndim    (3,4)+(4,)",        "add_diff_ndim");

    printf("\n-- tensor_mat_mul --\n");
    test_mat_mul("square        (4,4)@(4,4)",    "matmul_square");
    test_mat_mul("rect          (3,4)@(4,5)",    "matmul_rect");
    test_mat_mul("transposed A  (4,3)^T@(4,5)",  "matmul_transA", true);

    printf("\n-- tensor_sum --\n");
    test_sum("single element  (1,1)",    "sum_single");
    test_sum("small           (3,4)",    "sum_small");
    test_sum("medium          (128,32)", "sum_medium");
    test_sum("large           (512,512)","sum_large");

    printf("\n-- tensor_sum_dim --\n");
    test_sum_dim("2D dim=0 keep    (3,4)→(1,4)",     "sum_dim0_2d_keep",    0, true);
    test_sum_dim("2D dim=0 no-keep (3,4)→(4,)",      "sum_dim0_2d_nokeep",  0, false);
    test_sum_dim("2D dim=1 keep    (3,4)→(3,1)",     "sum_dim1_2d_keep",    1, true);
    test_sum_dim("2D dim=1 no-keep (3,4)→(3,)",      "sum_dim1_2d_nokeep",  1, false);
    test_sum_dim("3D dim=0 keep    (2,3,4)→(1,3,4)", "sum_dim0_3d_keep",    0, true);
    test_sum_dim("3D dim=1 keep    (2,3,4)→(2,1,4)", "sum_dim1_3d_keep",    1, true);
    test_sum_dim("3D dim=2 keep    (2,3,4)→(2,3,1)", "sum_dim2_3d_keep",    2, true);
    test_sum_dim("3D dim=1 no-keep (2,3,4)→(2,4)",   "sum_dim1_3d_nokeep",  1, false);
    test_sum_dim("large dim=0 keep (64,128)→(1,128)", "sum_dim0_large_keep", 0, true);
    test_sum_dim("large dim=1 keep (64,128)→(64,1)",  "sum_dim1_large_keep", 1, true);

    printf("\n-- tensor_index_select --\n");
    { const u32 idx[] = {0, 2};
      test_index_select("dim=0 rows [0,2]     (3,4)→(2,4)",  "idx_select_dim0_basic",   idx, 2, 0); }
    { const u32 idx[] = {2, 0, 1};
      test_index_select("dim=0 permute [2,0,1](3,4)→(3,4)",  "idx_select_dim0_permute", idx, 3, 0); }
    { const u32 idx[] = {1, 1};
      test_index_select("dim=0 repeat  [1,1]  (3,4)→(2,4)",  "idx_select_dim0_repeat",  idx, 2, 0); }
    { const u32 idx[] = {1, 3};
      test_index_select("dim=1 cols [1,3]     (3,4)→(3,2)",  "idx_select_dim1_basic",   idx, 2, 1); }
    { const u32 idx[] = {0, 2, 3};
      test_index_select("dim=1 cols [0,2,3]   (3,4)→(3,3)",  "idx_select_dim1_multi",   idx, 3, 1); }
    { const u32 idx[] = {3, 7, 0, 5};
      test_index_select("dim=0 large [3,7,0,5](8,16)→(4,16)","idx_select_dim0_large",   idx, 4, 0); }
    { const u32 idx[] = {0, 4, 8, 12, 15};
      test_index_select("dim=1 large 5 cols   (8,16)→(8,5)", "idx_select_dim1_large",   idx, 5, 1); }

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
