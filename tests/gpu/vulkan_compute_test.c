/*
 * vulkan_compute_test.c — Vulkan compute kernels against a CPU reference.
 *
 * For every Vulkan device with shaderFloat64 (all ICDs the loader finds, or
 * the one VK_DRIVER_FILES selects) this test opens the device through
 * gpu_vulkan.c and checks:
 *   - every admitted elementwise op is bit-identical to the host expression
 *     on fresh random data, edge values and sizes that straddle workgroups;
 *   - every admitted reduction is bit-identical to the host mirror of the
 *     fixed chunk tree (ESHKOL_VK_REDUCE_CHUNK), and within the standard
 *     bound (n-1)*u*sum|x| of the plain sequential sum;
 *   - matmul f64/f32 are bit-identical to the sequential non-fused sum;
 *   - wrapped buffers write results back to the caller's memory, allocated
 *     buffers expose mapped memory, sync round-trips, in-place works;
 *   - unimplemented ops (f64 transcendentals) and non-admitted ops return
 *     ESHKOL_VK_UNSUPPORTED so the dispatch takes the CPU path.
 * Exit 0 = pass, 1 = fail, 77 = no Vulkan loader or no f64 compute device.
 *
 * Build with -ffp-contract=off: the references must not be fused.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "gpu_vulkan.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                   \
    do {                                                   \
        g_checks++;                                        \
        if (!(cond)) {                                     \
            g_failures++;                                  \
            fprintf(stderr, "  FAIL: " __VA_ARGS__);       \
            fprintf(stderr, "\n");                         \
        }                                                  \
    } while (0)

static uint64_t g_seed = 0x0123456789abcdefull;
static uint64_t rng(void) {
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return g_seed;
}
static double rnd_normal(void) {  /* finite, varied magnitude */
    double m = (double)(rng() >> 11) / 9007199254740992.0 - 0.5;
    return ldexp(m, (int)(rng() % 40) - 20);
}
static double rnd_any(void) {     /* includes subnormals, inf, NaN, signed zero */
    uint64_t r = rng();
    if (r % 5 == 0) {
        uint64_t bits = rng();
        double d;
        memcpy(&d, &bits, sizeof(d));
        return d;
    }
    if (r % 97 == 1) return -0.0;
    return rnd_normal();
}

static int same_bits(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    return memcmp(&a, &b, sizeof(double)) == 0;
}

static double ref_elementwise(double x, double y, int has_b, EshkolElementwiseOp op) {
    volatile double r;
    switch (op) {
        case ESHKOL_ELEMWISE_ADD: r = x + (has_b ? y : 0.0); break;
        case ESHKOL_ELEMWISE_SUB: r = x - (has_b ? y : 0.0); break;
        case ESHKOL_ELEMWISE_MUL: r = x * (has_b ? y : 1.0); break;
        case ESHKOL_ELEMWISE_DIV: r = x / (has_b ? y : 1.0); break;
        case ESHKOL_ELEMWISE_NEG: r = -x; break;
        case ESHKOL_ELEMWISE_ABS: r = x < 0 ? -x : x; break;
        case ESHKOL_ELEMWISE_RELU: r = x > 0 ? x : 0; break;
        case ESHKOL_ELEMWISE_SQRT: r = sqrt(x); break;
        case ESHKOL_ELEMWISE_RECIPROCAL: r = 1.0 / x; break;
        default: r = NAN; break;
    }
    return r;
}

static const EshkolElementwiseOp k_ops[] = {
    ESHKOL_ELEMWISE_ADD, ESHKOL_ELEMWISE_SUB, ESHKOL_ELEMWISE_MUL, ESHKOL_ELEMWISE_DIV,
    ESHKOL_ELEMWISE_NEG, ESHKOL_ELEMWISE_ABS, ESHKOL_ELEMWISE_RELU, ESHKOL_ELEMWISE_SQRT,
    ESHKOL_ELEMWISE_RECIPROCAL
};
static const char* const k_op_names[] = {
    "add", "sub", "mul", "div", "neg", "abs", "exp", "log", "sin", "cos", "tanh",
    "relu", "sigmoid", "sqrt", "reciprocal"
};
static const char* const k_reduce_names[] = { "sum", "prod", "min", "max", "mean" };

static void test_elementwise(void) {
    static const uint64_t sizes[] = { 1, 255, 256, 257, 4099, 300007 };
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        uint64_t n = sizes[si];
        double* a = (double*)malloc(n * sizeof(double));
        double* b = (double*)malloc(n * sizeof(double));
        double* o = (double*)malloc(n * sizeof(double));
        for (uint64_t i = 0; i < n; i++) { a[i] = rnd_any(); b[i] = rnd_any(); }
        EshkolGPUBuffer ba, bb, bo;
        CHECK(eshkol_vk_wrap_host(a, n * 8, &ba) == ESHKOL_VK_OK, "wrap a n=%llu", (unsigned long long)n);
        CHECK(eshkol_vk_wrap_host(b, n * 8, &bb) == ESHKOL_VK_OK, "wrap b");
        CHECK(eshkol_vk_wrap_host(o, n * 8, &bo) == ESHKOL_VK_OK, "wrap o");
        CHECK(ba.host_ptr == a && ba.backend == ESHKOL_GPU_VULKAN, "wrap keeps host_ptr");
        for (size_t k = 0; k < sizeof(k_ops) / sizeof(k_ops[0]); k++) {
            EshkolElementwiseOp op = k_ops[k];
            int binary = op <= ESHKOL_ELEMWISE_DIV;
            for (int with_b = binary; with_b >= 0; with_b--) {
                memset(o, 0xA5, n * sizeof(double));
                int rc = eshkol_vk_elementwise_f64(&ba, with_b ? &bb : NULL, &bo, n, op);
                if (!eshkol_vk_elementwise_admitted(op)) {
                    CHECK(rc == ESHKOL_VK_UNSUPPORTED, "non-admitted %s must defer to CPU", k_op_names[op]);
                    continue;
                }
                CHECK(rc == ESHKOL_VK_OK, "%s rc=%d n=%llu", k_op_names[op], rc, (unsigned long long)n);
                uint64_t bad = 0, first = 0;
                for (uint64_t i = 0; i < n; i++)
                    if (!same_bits(o[i], ref_elementwise(a[i], b[i], with_b, op))) {
                        if (!bad) first = i;
                        bad++;
                    }
                CHECK(bad == 0, "%s%s n=%llu: %llu mismatches (first i=%llu a=%a b=%a got=%a want=%a)",
                      k_op_names[op], with_b ? "" : " (no b)", (unsigned long long)n,
                      (unsigned long long)bad, (unsigned long long)first, a[first], b[first], o[first],
                      ref_elementwise(a[first], b[first], with_b, op));
            }
        }
        /* In place: out aliases a. */
        if (eshkol_vk_elementwise_admitted(ESHKOL_ELEMWISE_MUL)) {
            double* copy = (double*)malloc(n * sizeof(double));
            memcpy(copy, a, n * sizeof(double));
            CHECK(eshkol_vk_elementwise_f64(&ba, &bb, &ba, n, ESHKOL_ELEMWISE_MUL) == ESHKOL_VK_OK,
                  "in-place mul");
            uint64_t bad = 0;
            for (uint64_t i = 0; i < n; i++)
                if (!same_bits(a[i], ref_elementwise(copy[i], b[i], 1, ESHKOL_ELEMWISE_MUL))) bad++;
            CHECK(bad == 0, "in-place mul: %llu mismatches", (unsigned long long)bad);
            free(copy);
        }
        /* Transcendentals have no f64 form in GLSL.std.450: always the CPU path. */
        CHECK(eshkol_vk_elementwise_f64(&ba, NULL, &bo, n, ESHKOL_ELEMWISE_EXP) == ESHKOL_VK_UNSUPPORTED,
              "exp must defer to CPU");
        CHECK(eshkol_vk_elementwise_f64(&ba, NULL, &bo, n, ESHKOL_ELEMWISE_TANH) == ESHKOL_VK_UNSUPPORTED,
              "tanh must defer to CPU");
        eshkol_vk_free(&ba);
        eshkol_vk_free(&bb);
        eshkol_vk_free(&bo);
        free(a); free(b); free(o);
    }
}

static double identity_of(int op) {
    switch (op) {
        case ESHKOL_REDUCE_PROD: return 1.0;
        case ESHKOL_REDUCE_MIN: return INFINITY;
        case ESHKOL_REDUCE_MAX: return -INFINITY;
        default: return 0.0;
    }
}

/* Host mirror of the device's fixed chunk tree. */
static double tree_reduce(const double* x, uint64_t n, int op) {
    double* t = (double*)malloc(n * sizeof(double));
    memcpy(t, x, n * sizeof(double));
    uint64_t count = n;
    for (;;) {
        uint64_t parts = (count + ESHKOL_VK_REDUCE_CHUNK - 1) / ESHKOL_VK_REDUCE_CHUNK;
        for (uint64_t p = 0; p < parts; p++) {
            volatile double r = identity_of(op);
            uint64_t end = (p + 1) * ESHKOL_VK_REDUCE_CHUNK;
            if (end > count) end = count;
            for (uint64_t i = p * ESHKOL_VK_REDUCE_CHUNK; i < end; i++) {
                double v = t[i];
                if (op == ESHKOL_REDUCE_SUM || op == ESHKOL_REDUCE_MEAN) r = r + v;
                else if (op == ESHKOL_REDUCE_PROD) r = r * v;
                else if (op == ESHKOL_REDUCE_MIN) r = (v < r) ? v : r;
                else r = (v > r) ? v : r;
            }
            t[p] = r;
        }
        count = parts;
        if (parts == 1) break;
    }
    double res = t[0];
    free(t);
    if (op == ESHKOL_REDUCE_MEAN) res = res / (double)n;
    return res;
}

static void test_reduce(void) {
    static const uint64_t sizes[] = { 1, 256, 257, 65536, 65537, 1000003 };
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        uint64_t n = sizes[si];
        double* x = (double*)malloc(n * sizeof(double));
        double out = 0.0;
        for (uint64_t i = 0; i < n; i++) x[i] = rnd_normal();
        /* prod of many values under/overflows; keep it informative */
        EshkolGPUBuffer bx, bo;
        CHECK(eshkol_vk_wrap_host(x, n * 8, &bx) == ESHKOL_VK_OK, "wrap reduce input");
        CHECK(eshkol_vk_wrap_host(&out, 8, &bo) == ESHKOL_VK_OK, "wrap reduce output");
        for (int op = ESHKOL_REDUCE_SUM; op <= ESHKOL_REDUCE_MEAN; op++) {
            int rc = eshkol_vk_reduce_f64(&bx, &bo, n, (EshkolReduceOp)op);
            if (!eshkol_vk_reduce_admitted((EshkolReduceOp)op)) {
                CHECK(rc == ESHKOL_VK_UNSUPPORTED, "non-admitted reduce %s must defer", k_reduce_names[op]);
                continue;
            }
            CHECK(rc == ESHKOL_VK_OK, "reduce %s rc=%d", k_reduce_names[op], rc);
            double want = tree_reduce(x, n, op);
            CHECK(same_bits(out, want), "reduce %s n=%llu got %a want %a", k_reduce_names[op],
                  (unsigned long long)n, out, want);
            if (op == ESHKOL_REDUCE_SUM) {
                /* Plain sequential sum within (n-1) u sum|x| of the tree. */
                volatile double seq = 0.0;
                double abs_sum = 0.0;
                for (uint64_t i = 0; i < n; i++) { seq = seq + x[i]; abs_sum += fabs(x[i]); }
                double bound = 2.0 * (double)(n > 1 ? n - 1 : 1) * (DBL_EPSILON / 2.0) * abs_sum;
                CHECK(fabs(out - seq) <= bound, "sum n=%llu off sequential by %g > bound %g",
                      (unsigned long long)n, fabs(out - seq), bound);
            }
            if (op == ESHKOL_REDUCE_MIN || op == ESHKOL_REDUCE_MAX) {
                double seq = identity_of(op);
                for (uint64_t i = 0; i < n; i++)
                    seq = (op == ESHKOL_REDUCE_MIN) ? ((x[i] < seq) ? x[i] : seq)
                                                    : ((x[i] > seq) ? x[i] : seq);
                CHECK(same_bits(out, seq), "%s equals the sequential scan", k_reduce_names[op]);
            }
        }
        eshkol_vk_free(&bx);
        eshkol_vk_free(&bo);
        free(x);
    }
}

static void test_matmul(void) {
    static const uint64_t shapes[][3] = {
        { 1, 1, 1 }, { 16, 16, 16 }, { 33, 7, 65 }, { 128, 257, 64 }, { 5, 1000, 3 }
    };
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        uint64_t M = shapes[s][0], K = shapes[s][1], N = shapes[s][2];
        double* A = (double*)malloc(M * K * 8);
        double* B = (double*)malloc(K * N * 8);
        double* C = (double*)malloc(M * N * 8);
        for (uint64_t i = 0; i < M * K; i++) A[i] = rnd_normal();
        for (uint64_t i = 0; i < K * N; i++) B[i] = rnd_normal();
        EshkolGPUBuffer ba, bb, bc;
        eshkol_vk_wrap_host(A, M * K * 8, &ba);
        eshkol_vk_wrap_host(B, K * N * 8, &bb);
        eshkol_vk_wrap_host(C, M * N * 8, &bc);
        int rc = eshkol_vk_matmul_f64(&ba, &bb, &bc, M, K, N);
        if (!eshkol_vk_matmul_admitted(0)) {
            CHECK(rc == ESHKOL_VK_UNSUPPORTED, "non-admitted matmul f64 must defer");
        } else {
            CHECK(rc == ESHKOL_VK_OK, "matmul f64 %llux%llux%llu rc=%d", (unsigned long long)M,
                  (unsigned long long)K, (unsigned long long)N, rc);
            uint64_t bad = 0;
            for (uint64_t i = 0; i < M; i++)
                for (uint64_t j = 0; j < N; j++) {
                    volatile double acc = 0.0;
                    for (uint64_t k = 0; k < K; k++) {
                        volatile double p = A[i * K + k] * B[k * N + j];
                        acc = acc + p;
                    }
                    if (!same_bits(C[i * N + j], acc)) bad++;
                }
            CHECK(bad == 0, "matmul f64 %llux%llux%llu: %llu mismatches", (unsigned long long)M,
                  (unsigned long long)K, (unsigned long long)N, (unsigned long long)bad);
        }
        eshkol_vk_free(&ba); eshkol_vk_free(&bb); eshkol_vk_free(&bc);

        /* f32 on allocated (mapped) buffers. */
        EshkolGPUBuffer fa, fb, fc;
        CHECK(eshkol_vk_alloc(M * K * 4, &fa) == ESHKOL_VK_OK && fa.host_ptr != NULL, "alloc f32 A");
        CHECK(eshkol_vk_alloc(K * N * 4, &fb) == ESHKOL_VK_OK, "alloc f32 B");
        CHECK(eshkol_vk_alloc(M * N * 4, &fc) == ESHKOL_VK_OK, "alloc f32 C");
        float* af = (float*)fa.host_ptr;
        float* bf = (float*)fb.host_ptr;
        float* cf = (float*)fc.host_ptr;
        for (uint64_t i = 0; i < M * K; i++) af[i] = (float)rnd_normal();
        for (uint64_t i = 0; i < K * N; i++) bf[i] = (float)rnd_normal();
        rc = eshkol_vk_matmul_f32(&fa, &fb, &fc, M, K, N);
        if (!eshkol_vk_matmul_admitted(1)) {
            CHECK(rc == ESHKOL_VK_UNSUPPORTED, "non-admitted matmul f32 must defer");
        } else {
            CHECK(rc == ESHKOL_VK_OK, "matmul f32 rc=%d", rc);
            uint64_t bad = 0;
            for (uint64_t i = 0; i < M; i++)
                for (uint64_t j = 0; j < N; j++) {
                    volatile float acc = 0.0f;
                    for (uint64_t k = 0; k < K; k++) {
                        volatile float p = af[i * K + k] * bf[k * N + j];
                        acc = acc + p;
                    }
                    float want = acc, got = cf[i * N + j];
                    if (memcmp(&want, &got, sizeof(float)) != 0) bad++;
                }
            CHECK(bad == 0, "matmul f32 %llux%llux%llu: %llu mismatches", (unsigned long long)M,
                  (unsigned long long)K, (unsigned long long)N, (unsigned long long)bad);
        }
        eshkol_vk_free(&fa); eshkol_vk_free(&fb); eshkol_vk_free(&fc);
        free(A); free(B); free(C);
    }
}

static void test_sync(void) {
    double host[1000];
    EshkolGPUBuffer w;
    for (int i = 0; i < 1000; i++) host[i] = i * 0.5;
    CHECK(eshkol_vk_wrap_host(host, sizeof(host), &w) == ESHKOL_VK_OK, "wrap for sync");
    CHECK(eshkol_vk_sync(&w, ESHKOL_SYNC_HOST_TO_DEVICE) == ESHKOL_VK_OK, "sync h2d");
    for (int i = 0; i < 1000; i++) host[i] = -1.0;
    CHECK(eshkol_vk_sync(&w, ESHKOL_SYNC_DEVICE_TO_HOST) == ESHKOL_VK_OK, "sync d2h");
    int ok = 1;
    for (int i = 0; i < 1000; i++) ok &= host[i] == i * 0.5;
    CHECK(ok, "sync round-trip restores the device copy");
    eshkol_vk_free(&w);
    CHECK(w.backend_data == NULL && w.host_ptr == NULL, "free clears the descriptor");
}

int main(void) {
    int count = eshkol_vk_device_count();
    if (count <= 0) {
        printf("SKIP: no Vulkan device (%s)\n", eshkol_vk_status());
        return 77;
    }
    int tested = 0;
    for (int d = 0; d < count; d++) {
        char name[256];
        int type = -1, f64 = 0;
        eshkol_vk_device_describe(d, name, sizeof(name), &type, &f64);
        if (!f64) {
            printf("device %d: %s (type %d) has no shaderFloat64, not tested\n", d, name, type);
            continue;
        }
        if (!eshkol_vk_init_device(d)) {
            printf("device %d: %s not usable: %s\n", d, name, eshkol_vk_status());
            continue;
        }
        int before_fail = g_failures, before_checks = g_checks;
        printf("device %d: %s (type %d)\n  %s\n", d, eshkol_vk_device_name(), eshkol_vk_device_type(),
               eshkol_vk_status());
        printf("  admitted elementwise:");
        for (size_t k = 0; k < sizeof(k_ops) / sizeof(k_ops[0]); k++)
            if (eshkol_vk_elementwise_admitted(k_ops[k])) printf(" %s", k_op_names[k_ops[k]]);
        printf("\n  admitted reduce:");
        for (int op = 0; op <= ESHKOL_REDUCE_MEAN; op++)
            if (eshkol_vk_reduce_admitted((EshkolReduceOp)op)) printf(" %s", k_reduce_names[op]);
        printf("\n  admitted matmul: f64=%d f32=%d\n", eshkol_vk_matmul_admitted(0),
               eshkol_vk_matmul_admitted(1));
        test_elementwise();
        test_reduce();
        test_matmul();
        test_sync();
        printf("  %d/%d checks passed\n", (g_checks - before_checks) - (g_failures - before_fail),
               g_checks - before_checks);
        tested++;
    }
    eshkol_vk_shutdown();
    if (!tested) {
        printf("SKIP: no usable Vulkan f64 compute device\n");
        return 77;
    }
    if (g_failures) {
        printf("FAIL: %d of %d checks failed on %d device(s)\n", g_failures, g_checks, tested);
        return 1;
    }
    printf("PASS: %d checks on %d device(s)\n", g_checks, tested);
    return 0;
}
