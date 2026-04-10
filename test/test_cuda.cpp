#include "../include/tensor.hpp"
#include "../include/backend/tensor_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int passed = 0, failed = 0;

static void check(const char *name, bool ok) {
    printf("  [%s%s%s] %s\n", ok ? GREEN : RED, ok ? "PASS" : "FAIL", RESET, name);
    ok ? passed++ : failed++;
}

static Tensor *load_gpu(const char *path) {
    Tensor *cpu = tensor_load(path, false);
    if (!cpu) return nullptr;
    Tensor *gpu = tensor_to_gpu(cpu);
    delete cpu;
    return gpu;
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

static void test_mat_mul(const char *name, const char *dir, bool trans_a = false) {
    char pa[256], pb[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pb,   sizeof(pb),   "../data/test/%s/b.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a_cpu = tensor_load(pa, false);
    Tensor *b     = load_gpu(pb);
    Tensor *exp   = tensor_load(pout, false);

    if (!a_cpu || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a_cpu; delete b; delete exp;
        return;
    }

    if (trans_a)
        tensor_transpose(a_cpu, 0, 1);

    Tensor *a = tensor_to_gpu(a_cpu);
    delete a_cpu;

    Tensor *out = tensor_mat_mul(a, b);
    cudaDeviceSynchronize();

    Tensor *out_cpu = tensor_to_cpu(out);
    check(name, tensors_close(out_cpu, exp, 1e-4f));

    delete a; delete b; delete exp; delete out; delete out_cpu;
}

static void test_add(const char *name, const char *dir) {
    char pa[256], pb[256], pout[256];
    snprintf(pa,   sizeof(pa),   "../data/test/%s/a.npy",   dir);
    snprintf(pb,   sizeof(pb),   "../data/test/%s/b.npy",   dir);
    snprintf(pout, sizeof(pout), "../data/test/%s/out.npy", dir);

    Tensor *a   = load_gpu(pa);
    Tensor *b   = load_gpu(pb);
    Tensor *exp = tensor_load(pout, false);

    if (!a || !b || !exp) {
        printf("  [%sFAIL%s] %s — could not load data files\n", RED, RESET, name);
        failed++;
        delete a; delete b; delete exp;
        return;
    }

    Tensor *out = tensor_add(a, b);
    cudaDeviceSynchronize();

    Tensor *out_cpu = tensor_to_cpu(out);
    check(name, tensors_close(out_cpu, exp));

    delete a; delete b; delete exp; delete out; delete out_cpu;
}

int main() {
    printf("\n-- CUDA tensor_add --\n");
    test_add("same shape   (3,4)+(3,4)",      "add_same");
    test_add("bcast row    (3,4)+(1,4)",       "add_bcast_row");
    test_add("bcast col    (3,4)+(3,1)",       "add_bcast_col");
    test_add("bcast scalar (3,4)+(1,1)",       "add_bcast_scalar");
    test_add("diff ndim    (3,4)+(4,)",        "add_diff_ndim");

    printf("\n-- CUDA tensor_mat_mul --\n");
    test_mat_mul("square        (4,4)@(4,4)",    "matmul_square");
    test_mat_mul("rect          (3,4)@(4,5)",    "matmul_rect");
    test_mat_mul("transposed A  (4,3)^T@(4,5)",  "matmul_transA", true);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
