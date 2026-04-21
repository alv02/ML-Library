#include "../include/tensor.hpp"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static b32 g_on_gpu = false;
static int passed = 0, failed = 0;

static void sync() {
    if (g_on_gpu)
        cudaDeviceSynchronize();
}

static void check(const char *name, bool ok) {
    printf("  [%s%s%s] %s\n", ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET,
           name);
    ok ? passed++ : failed++;
}

static void check_tensors(const char *name, const Tensor *got,
                          const Tensor *exp, f32 tol = 1e-5f) {
    if (!tensor_shape_eq(got, exp)) {
        printf("  [%sFAIL%s] %s — shape mismatch: got [", RED, RESET, name);
        for (u32 i = 0; i < got->ndim; i++)
            printf("%u%s", got->shape[i], i + 1 < got->ndim ? "," : "");
        printf("] expected [");
        for (u32 i = 0; i < exp->ndim; i++)
            printf("%u%s", exp->shape[i], i + 1 < exp->ndim ? "," : "");
        printf("]\n");
        failed++;
        return;
    }
    check(name, tensor_equals(got, exp, tol));
}

static void test_add(const char *name, const char *dir) {
    char pa[256], pb[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pb, sizeof(pb), "../data/test/%s/b.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *b = tensor_load(pb, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete b;
        delete exp;
        return;
    }

    Tensor *out = tensor_add(a, b);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp);

    delete a;
    delete b;
    delete exp;
    delete out;
    delete out_cpu;
}

static void test_mat_mul(const char *name, const char *dir,
                         bool trans_a = false) {
    char pa[256], pb[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pb, sizeof(pb), "../data/test/%s/b.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *b = tensor_load(pb, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete b;
        delete exp;
        return;
    }

    if (trans_a)
        tensor_transpose(a, 0, 1);

    Tensor *out = tensor_mat_mul(a, b);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp, 1e-4f);

    delete a;
    delete b;
    delete exp;
    delete out;
    delete out_cpu;
}

static void test_sum(const char *name, const char *dir) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete exp;
        return;
    }

    Tensor *out = tensor_sum(a);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp, 1e-3f);

    delete a;
    delete exp;
    delete out;
    delete out_cpu;
}

static void test_sum_dim(const char *name, const char *dir, u32 dim,
                         b32 keep_dim) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete exp;
        return;
    }

    Tensor *out = tensor_sum(a, dim, keep_dim);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp, 1e-4f);

    delete a;
    delete exp;
    delete out;
    delete out_cpu;
}

static void test_index_select(const char *name, const char *dir,
                              const u32 *indices, u32 n_indices, u32 dim) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete exp;
        return;
    }

    Tensor *out = tensor_index_select(a, indices, n_indices, dim);
    if (!out) {
        printf("  [%sFAIL%s] %s — returned nullptr\n", RED, RESET, name);
        failed++;
        delete a;
        delete exp;
        return;
    }
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp);

    delete a;
    delete exp;
    delete out;
    delete out_cpu;
}

// Input saved as [C, N, H, W]; transpose(0,1) gives non-contiguous [N, C, H, W]
static void test_unfold2d_noncontig(const char *name, const char *dir,
                                    Unfold2dParams params) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete exp;
        return;
    }

    tensor_transpose(a, 0, 1); // [C,N,H,W] → [N,C,H,W] non-contiguous

    u32 flat[1] = {(u32)exp->size};
    Tensor *out = new Tensor(1, flat, g_on_gpu);
    tensor_unfold2d(out, a, params);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp);

    delete a;
    delete exp;
    delete out;
    delete out_cpu;
}

// Replicates MaxPool2dOp::forward exactly — good canary for the full pipeline
static void test_maxpool2d(const char *name, const char *dir,
                           Unfold2dParams params) {
    char pa[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *input = tensor_load(pa,   g_on_gpu);
    Tensor *exp   = tensor_load(pout, false);

    if (!input || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete input; delete exp;
        return;
    }

    u32 N = input->shape[0];
    u32 C = input->shape[1];
    params.compute_output_size(input->shape[2], input->shape[3]);
    u32 L = params.L_h * params.L_w;
    u32 K = params.k_h * params.k_w;

    // Replicate MaxPool2dOp::forward
    Tensor *col = tensor_unfold2d(input, params);
    u32 shape4[4] = {N, L, C, K};
    tensor_reshape(col, shape4, 4);

    Tensor *pooled = tensor_max(col, 3, false);
    delete col;

    u32 shape_nlhwc[4] = {N, params.L_h, params.L_w, C};
    tensor_reshape(pooled, shape_nlhwc, 4);
    tensor_transpose(pooled, 1, 3);
    tensor_transpose(pooled, 2, 3);

    sync();
    Tensor *out_cpu = tensor_to_cpu(pooled);
    check_tensors(name, out_cpu, exp, 1e-5f);

    delete input; delete exp; delete pooled; delete out_cpu;
}

static void test_welford_mean_var(const char *name, const char *dir, u32 dim) {
    char pa[256], pmean[256], pvar[256];
    snprintf(pa,    sizeof(pa),    "../data/test/%s/a.npy",    dir);
    snprintf(pmean, sizeof(pmean), "../data/test/%s/mean.npy", dir);
    snprintf(pvar,  sizeof(pvar),  "../data/test/%s/var.npy",  dir);

    Tensor *a        = tensor_load(pa,    g_on_gpu);
    Tensor *exp_mean = tensor_load(pmean, false);
    Tensor *exp_var  = tensor_load(pvar,  false);

    if (!a || !exp_mean || !exp_var) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete exp_mean; delete exp_var;
        return;
    }

    u32 mean_shape[1] = { a->shape[dim] };
    Tensor *got_mean = new Tensor(1, mean_shape, g_on_gpu);
    Tensor *got_var  = new Tensor(1, mean_shape, g_on_gpu);

    tensor_welford_mean_var(got_mean, got_var, a, dim);
    sync();

    Tensor *mean_cpu = tensor_to_cpu(got_mean);
    Tensor *var_cpu  = tensor_to_cpu(got_var);

    char mean_label[300], var_label[300];
    snprintf(mean_label, sizeof(mean_label), "%s [mean]", name);
    snprintf(var_label,  sizeof(var_label),  "%s [var]",  name);
    check_tensors(mean_label, mean_cpu, exp_mean, 1e-4f);
    check_tensors(var_label,  var_cpu,  exp_var,  1e-4f);

    delete a; delete exp_mean; delete exp_var;
    delete got_mean; delete got_var;
    delete mean_cpu; delete var_cpu;
}

static void test_scatter_add(const char *name, const char *dir, u32 dim,
                             u32 K) {
    char pa[256], pb[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pb,   sizeof(pb),   "../data/test/%s/b.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *src     = tensor_load(pa,   g_on_gpu);
    Tensor *indices = tensor_load(pb,   g_on_gpu);
    Tensor *exp     = tensor_load(pout, false);

    if (!src || !indices || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete src; delete indices; delete exp;
        return;
    }

    Tensor *out = tensor_scatter_add(src, indices, dim, K);
    if (!out) {
        printf("  [%sFAIL%s] %s — returned nullptr\n", RED, RESET, name);
        failed++;
        delete src; delete indices; delete exp;
        return;
    }
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp, 1e-5f);

    delete src; delete indices; delete exp; delete out; delete out_cpu;
}

static void test_fold2d(const char *name, const char *dir,
                        Unfold2dParams params, u32 N, u32 C, u32 H, u32 W) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *col = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!col || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete col;
        delete exp;
        return;
    }

    u32 dst_shape[4] = {N, C, H, W};
    Tensor *out = new Tensor(4, dst_shape, g_on_gpu);
    tensor_clear(out);
    tensor_fold2d(out, col, params);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp);

    delete col;
    delete exp;
    delete out;
    delete out_cpu;
}

static void test_unfold2d(const char *name, const char *dir,
                          Unfold2dParams params) {
    char pa[256], pout[256];
    snprintf(pa, sizeof(pa), "../data/test/%s/a.npy", dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a = tensor_load(pa, g_on_gpu);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET,
               name);
        failed++;
        delete a;
        delete exp;
        return;
    }

    u32 flat[1] = {(u32)exp->size};
    Tensor *out = new Tensor(1, flat, g_on_gpu);
    tensor_unfold2d(out, a, params);
    sync();
    Tensor *out_cpu = tensor_to_cpu(out);
    check_tensors(name, out_cpu, exp);

    delete a;
    delete exp;
    delete out;
    delete out_cpu;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0)
            g_on_gpu = true;
        else if (strcmp(argv[i], "--cpu") == 0)
            g_on_gpu = false;
    }

    printf("\nBackend: %s\n", g_on_gpu ? "CUDA" : "CPU");

    printf("\n-- tensor_add --\n");
    test_add("same shape   (3,4)+(3,4)", "add_same");
    test_add("bcast row    (3,4)+(1,4)", "add_bcast_row");
    test_add("bcast col    (3,4)+(3,1)", "add_bcast_col");
    test_add("bcast scalar (3,4)+(1,1)", "add_bcast_scalar");
    test_add("diff ndim    (3,4)+(4,)", "add_diff_ndim");

    printf("\n-- tensor_mat_mul --\n");
    test_mat_mul("square        (4,4)@(4,4)", "matmul_square");
    test_mat_mul("rect          (3,4)@(4,5)", "matmul_rect");
    test_mat_mul("transposed A  (4,3)^T@(4,5)", "matmul_transA", true);

    printf("\n-- tensor_sum --\n");
    test_sum("single element  (1,1)", "sum_single");
    test_sum("small           (3,4)", "sum_small");
    test_sum("medium          (128,32)", "sum_medium");
    test_sum("large           (512,512)", "sum_large");

    printf("\n-- tensor_sum_dim --\n");
    test_sum_dim("2D dim=0 keep    (3,4)→(1,4)", "sum_dim0_2d_keep", 0, true);
    test_sum_dim("2D dim=0 no-keep (3,4)→(4,)", "sum_dim0_2d_nokeep", 0, false);
    test_sum_dim("2D dim=1 keep    (3,4)→(3,1)", "sum_dim1_2d_keep", 1, true);
    test_sum_dim("2D dim=1 no-keep (3,4)→(3,)", "sum_dim1_2d_nokeep", 1, false);
    test_sum_dim("3D dim=0 keep    (2,3,4)→(1,3,4)", "sum_dim0_3d_keep", 0,
                 true);
    test_sum_dim("3D dim=1 keep    (2,3,4)→(2,1,4)", "sum_dim1_3d_keep", 1,
                 true);
    test_sum_dim("3D dim=2 keep    (2,3,4)→(2,3,1)", "sum_dim2_3d_keep", 2,
                 true);
    test_sum_dim("3D dim=1 no-keep (2,3,4)→(2,4)", "sum_dim1_3d_nokeep", 1,
                 false);
    test_sum_dim("large dim=0 keep (64,128)→(1,128)", "sum_dim0_large_keep", 0,
                 true);
    test_sum_dim("large dim=1 keep (64,128)→(64,1)", "sum_dim1_large_keep", 1,
                 true);

    printf("\n-- tensor_index_select --\n");
    {
        const u32 idx[] = {0, 2};
        test_index_select("dim=0 rows [0,2]     (3,4)→(2,4)",
                          "idx_select_dim0_basic", idx, 2, 0);
    }
    {
        const u32 idx[] = {2, 0, 1};
        test_index_select("dim=0 permute [2,0,1](3,4)→(3,4)",
                          "idx_select_dim0_permute", idx, 3, 0);
    }
    {
        const u32 idx[] = {1, 1};
        test_index_select("dim=0 repeat  [1,1]  (3,4)→(2,4)",
                          "idx_select_dim0_repeat", idx, 2, 0);
    }
    {
        const u32 idx[] = {1, 3};
        test_index_select("dim=1 cols [1,3]     (3,4)→(3,2)",
                          "idx_select_dim1_basic", idx, 2, 1);
    }
    {
        const u32 idx[] = {0, 2, 3};
        test_index_select("dim=1 cols [0,2,3]   (3,4)→(3,3)",
                          "idx_select_dim1_multi", idx, 3, 1);
    }
    {
        const u32 idx[] = {3, 7, 0, 5};
        test_index_select("dim=0 large [3,7,0,5](8,16)→(4,16)",
                          "idx_select_dim0_large", idx, 4, 0);
    }
    {
        const u32 idx[] = {0, 4, 8, 12, 15};
        test_index_select("dim=1 large 5 cols   (8,16)→(8,5)",
                          "idx_select_dim1_large", idx, 5, 1);
    }

    printf("\n-- tensor_unfold2d --\n");
    test_unfold2d("1x1x4x4  k=3 s=1 p=0", "unfold2d_1c_4x4_k3s1",
                  Unfold2dParams(3));
    test_unfold2d("1x2x3x3  k=2 s=1 p=0", "unfold2d_2c_3x3_k2s1",
                  Unfold2dParams(2));
    test_unfold2d("2x1x4x4  k=3 s=1 p=0", "unfold2d_batch_4x4_k3s1",
                  Unfold2dParams(3));
    test_unfold2d("1x1x6x6  k=3 s=2 p=0", "unfold2d_1c_6x6_k3s2",
                  Unfold2dParams(3, 2, 0));
    test_unfold2d("1x1x5x5  k=2 s=2 p=0", "unfold2d_1c_5x5_k2s2",
                  Unfold2dParams(2, 2, 0));
    test_unfold2d("1x1x3x3  k=3 s=1 p=1", "unfold2d_1c_3x3_k3p1",
                  Unfold2dParams(3, 1, 1));
    test_unfold2d("1x1x4x4  k=3 s=1 p=1", "unfold2d_1c_4x4_k3p1",
                  Unfold2dParams(3, 1, 1));
    test_unfold2d("1x1x4x4  k=3 s=2 p=1", "unfold2d_1c_4x4_k3s2p1",
                  Unfold2dParams(3, 2, 1));

    printf("\n-- tensor_unfold2d (non-contiguous input) --\n");
    test_unfold2d_noncontig("1x2x3x3  k=2 s=1 p=0",
                            "unfold2d_noncontig_2c_3x3_k2s1",
                            Unfold2dParams(2));
    test_unfold2d_noncontig("2x1x4x4  k=3 s=1 p=0",
                            "unfold2d_noncontig_batch_4x4_k3s1",
                            Unfold2dParams(3));
    test_unfold2d_noncontig("2x3x4x4  k=3 s=1 p=0",
                            "unfold2d_noncontig_2n3c_4x4_k3s1",
                            Unfold2dParams(3));
    test_unfold2d_noncontig("2x3x4x4  k=3 s=1 p=1",
                            "unfold2d_noncontig_2n3c_4x4_k3p1",
                            Unfold2dParams(3, 1, 1));

    printf("\n-- tensor_fold2d --\n");
    test_fold2d("1x1x4x4  k=3 s=1 p=0", "fold2d_1c_4x4_k3s1", Unfold2dParams(3),
                1, 1, 4, 4);
    test_fold2d("1x2x3x3  k=2 s=1 p=0", "fold2d_2c_3x3_k2s1", Unfold2dParams(2),
                1, 2, 3, 3);
    test_fold2d("2x1x4x4  k=3 s=1 p=0", "fold2d_batch_4x4_k3s1",
                Unfold2dParams(3), 2, 1, 4, 4);
    test_fold2d("1x1x6x6  k=3 s=2 p=0", "fold2d_1c_6x6_k3s2",
                Unfold2dParams(3, 2, 0), 1, 1, 6, 6);
    test_fold2d("1x1x4x4  k=3 s=1 p=1", "fold2d_1c_4x4_k3p1",
                Unfold2dParams(3, 1, 1), 1, 1, 4, 4);
    test_fold2d("1x1x4x4  k=3 s=2 p=1", "fold2d_1c_4x4_k3s2p1",
                Unfold2dParams(3, 2, 1), 1, 1, 4, 4);
    test_fold2d("2x3x4x4  k=3 s=1 p=0", "fold2d_2n3c_4x4_k3s1",
                Unfold2dParams(3), 2, 3, 4, 4);

    printf("\n-- maxpool2d forward --\n");
    test_maxpool2d("1×1×4×4 k=2 s=2 p=0 → [1,1,2,2]",   "maxpool_1c_4x4_k2s2",    Unfold2dParams(2, 2, 0));
    test_maxpool2d("1×2×4×4 k=2 s=2 p=0 → [1,2,2,2]",   "maxpool_2c_4x4_k2s2",    Unfold2dParams(2, 2, 0));
    test_maxpool2d("2×3×6×6 k=2 s=2 p=0 → [2,3,3,3]",   "maxpool_2n3c_6x6_k2s2",  Unfold2dParams(2, 2, 0));
    test_maxpool2d("2×4×6×6 k=3 s=1 p=0 → [2,4,4,4]",   "maxpool_2n4c_6x6_k3s1",  Unfold2dParams(3, 1, 0));

    printf("\n-- tensor_welford_mean_var --\n");
    test_welford_mean_var("2D   dim=0 (4,3)→(3)",              "welford_2d_dim0",          0);
    test_welford_mean_var("4D   dim=1 (2,4,3,3)→(4)",          "welford_4d_2n4c_3x3",      1);
    test_welford_mean_var("4D   dim=1 (4,8,6,6)→(8)",          "welford_4d_4n8c_6x6",      1);
    test_welford_mean_var("4D   dim=1 (8,16,8,8)→(16)",        "welford_4d_8n16c_8x8",     1);
    test_welford_mean_var("4D   dim=1 (16,32,16,16)→(32)",     "welford_4d_16n32c_16x16",  1);

    printf("\n-- tensor_scatter_add --\n");
    test_scatter_add("2D  dim=1 k=3  [4,1]→[4,3]",         "scatter_add_2d_dim1_k3",  1, 3);
    test_scatter_add("3D  dim=2 k=4  [2,3,1]→[2,3,4]",     "scatter_add_3d_dim2_k4",  2, 4);
    test_scatter_add("4D  dim=3 k=9  [2,4,3,1]→[2,4,3,9]", "scatter_add_4d_dim3_k9",  3, 9);
    test_scatter_add("4D  dim=1 k=5  [2,1,3,4]→[2,5,3,4]", "scatter_add_4d_dim1_k5",  1, 5);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
