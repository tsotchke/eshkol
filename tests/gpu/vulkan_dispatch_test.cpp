/*
 * vulkan_dispatch_test.cpp — the ordinary GPU API (eshkol_gpu_*) with the
 * Vulkan compute backend selected by ESHKOL_GPU_BACKEND=vulkan.
 *
 * Links the production GPU backend objects through eshkol-runtime, so it
 * exercises whichever host file the build chose (gpu_memory_cuda.cpp or
 * gpu_memory_stub.cpp) together with gpu_vulkan.c:
 *   - selection: init, backend query/name/availability, f64 support;
 *   - device kernels through wrapped and allocated buffers (bit-exact against
 *     the sequential non-fused reference where the kernel is admitted);
 *   - CPU continuation for operations Vulkan does not run (exp, transpose,
 *     softmax) on Vulkan buffers;
 *   - eshkol_matmul_dispatch above the threshold; shutdown returns to CPU.
 * Exit 0 = pass, 1 = fail, 77 = no usable Vulkan f64 compute device.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include <eshkol/backend/gpu/gpu_memory.h>
#include "gpu_vulkan.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        g_checks++;                                       \
        if (!(cond)) {                                    \
            g_fail++;                                     \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__);   \
            std::fprintf(stderr, "\n");                   \
        }                                                 \
    } while (0)

static uint64_t g_s = 0x9e3779b97f4a7c15ull;
static double rnd() {
    g_s ^= g_s << 13;
    g_s ^= g_s >> 7;
    g_s ^= g_s << 17;
    return (double)(g_s >> 11) / 9007199254740992.0 * 4.0 - 2.0;
}

static bool same_bits(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

static std::vector<double> ref_matmul(const std::vector<double>& A, const std::vector<double>& B,
                                      uint64_t M, uint64_t K, uint64_t N) {
    std::vector<double> C(M * N);
    for (uint64_t i = 0; i < M; i++)
        for (uint64_t j = 0; j < N; j++) {
            volatile double acc = 0.0;
            for (uint64_t k = 0; k < K; k++) {
                volatile double p = A[i * K + k] * B[k * N + j];
                acc = acc + p;
            }
            C[i * N + j] = acc;
        }
    return C;
}

static void check_matmul(const char* label, const std::vector<double>& got,
                         const std::vector<double>& want, bool exact) {
    uint64_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        bool ok = exact ? same_bits(got[i], want[i])
                        : std::fabs(got[i] - want[i]) <= 1e-12 * (1.0 + std::fabs(want[i]));
        if (!ok) bad++;
    }
    CHECK(bad == 0, "%s: %llu of %zu elements differ (%s comparison)", label,
          (unsigned long long)bad, got.size(), exact ? "bitwise" : "1e-12 relative");
}

int main() {
#if defined(_WIN32)
    _putenv_s("ESHKOL_GPU_BACKEND", "vulkan");
#else
    setenv("ESHKOL_GPU_BACKEND", "vulkan", 0);
#endif
    if (!eshkol_gpu_init()) {
        std::printf("SKIP: no usable Vulkan f64 compute device (%s)\n", eshkol_vk_status());
        return 77;
    }
    CHECK(eshkol_gpu_get_backend() == ESHKOL_GPU_VULKAN, "backend is %d, expected Vulkan",
          (int)eshkol_gpu_get_backend());
    CHECK(eshkol_gpu_backend_available(ESHKOL_GPU_VULKAN) == 1, "Vulkan reported available");
    CHECK(eshkol_gpu_backend_available(ESHKOL_GPU_CUDA) == 0, "CUDA not active");
    CHECK(eshkol_gpu_supports_f64() == 1 && eshkol_gpu_has_fp64() == 1, "f64 support reported");
    CHECK(std::strstr(eshkol_gpu_backend_name(ESHKOL_GPU_VULKAN), "Vulkan") != nullptr,
          "backend name mentions Vulkan");
    std::printf("device: %s\n%s\n", eshkol_vk_device_name(), eshkol_vk_status());

    const uint64_t M = 64, K = 48, N = 40;
    std::vector<double> A(M * K), B(K * N), C(M * N, 0.0);
    for (auto& v : A) v = rnd();
    for (auto& v : B) v = rnd();
    const std::vector<double> Cref = ref_matmul(A, B, M, K, N);
    const bool mm_exact = eshkol_vk_matmul_admitted(0) != 0;

    /* matmul through wrapped host memory */
    {
        EshkolGPUBuffer ba, bb, bc;
        CHECK(eshkol_gpu_wrap_host(A.data(), A.size() * 8, &ba) == 0, "wrap A");
        CHECK(eshkol_gpu_wrap_host(B.data(), B.size() * 8, &bb) == 0, "wrap B");
        CHECK(eshkol_gpu_wrap_host(C.data(), C.size() * 8, &bc) == 0, "wrap C");
        CHECK(ba.backend == ESHKOL_GPU_VULKAN && ba.host_ptr == A.data(), "wrapped buffer keeps the caller pointer");
        CHECK(eshkol_gpu_matmul_f64(&ba, &bb, &bc, M, K, N) == 0, "matmul_f64");
        check_matmul("eshkol_gpu_matmul_f64", C, Cref, mm_exact);
        eshkol_gpu_free(&ba);
        eshkol_gpu_free(&bb);
        eshkol_gpu_free(&bc);
        CHECK(ba.backend_data == nullptr, "free clears the descriptor");
    }

    /* elementwise: device ops bit-exact, transcendentals continue on the CPU */
    {
        const uint64_t n = 5000;
        std::vector<double> a(n), b(n), o(n);
        for (uint64_t i = 0; i < n; i++) { a[i] = rnd(); b[i] = rnd() + 3.0; }
        EshkolGPUBuffer ba, bb, bo;
        eshkol_gpu_wrap_host(a.data(), n * 8, &ba);
        eshkol_gpu_wrap_host(b.data(), n * 8, &bb);
        eshkol_gpu_wrap_host(o.data(), n * 8, &bo);
        struct { EshkolElementwiseOp op; const char* name; } cases[] = {
            { ESHKOL_ELEMWISE_ADD, "add" }, { ESHKOL_ELEMWISE_DIV, "div" },
            { ESHKOL_ELEMWISE_SQRT, "sqrt" }, { ESHKOL_ELEMWISE_EXP, "exp" },
            { ESHKOL_ELEMWISE_TANH, "tanh" }, { ESHKOL_ELEMWISE_RELU, "relu" },
        };
        for (auto& c : cases) {
            CHECK(eshkol_gpu_elementwise_f64(&ba, &bb, &bo, n, c.op) == 0, "elementwise %s", c.name);
            uint64_t bad = 0;
            for (uint64_t i = 0; i < n; i++) {
                volatile double want;
                switch (c.op) {
                    case ESHKOL_ELEMWISE_ADD: want = a[i] + b[i]; break;
                    case ESHKOL_ELEMWISE_DIV: want = a[i] / b[i]; break;
                    case ESHKOL_ELEMWISE_SQRT: want = std::sqrt(a[i]); break;
                    case ESHKOL_ELEMWISE_EXP: want = std::exp(a[i]); break;
                    case ESHKOL_ELEMWISE_TANH: want = std::tanh(a[i]); break;
                    default: want = a[i] > 0 ? a[i] : 0; break;
                }
                if (!same_bits(o[i], want)) bad++;
            }
            CHECK(bad == 0, "elementwise %s: %llu mismatches", c.name, (unsigned long long)bad);
        }
        CHECK(eshkol_vk_elementwise_f64(&ba, nullptr, &bo, n, ESHKOL_ELEMWISE_EXP) == ESHKOL_VK_UNSUPPORTED,
              "exp is not a device kernel (CPU continuation)");

        /* reductions */
        double r = 0.0;
        EshkolGPUBuffer br;
        eshkol_gpu_wrap_host(&r, 8, &br);
        CHECK(eshkol_gpu_reduce_f64(&ba, &br, n, ESHKOL_REDUCE_MAX) == 0, "reduce max");
        double mx = -INFINITY;
        for (uint64_t i = 0; i < n; i++) mx = a[i] > mx ? a[i] : mx;
        CHECK(same_bits(r, mx), "reduce max %a vs %a", r, mx);
        CHECK(eshkol_gpu_reduce_f64(&ba, &br, n, ESHKOL_REDUCE_SUM) == 0, "reduce sum");
        double seq = 0.0, abs_sum = 0.0;
        for (uint64_t i = 0; i < n; i++) { seq += a[i]; abs_sum += std::fabs(a[i]); }
        CHECK(std::fabs(r - seq) <= 2.0 * (double)(n - 1) * 1.1102230246251565e-16 * abs_sum,
              "reduce sum %.17g vs sequential %.17g", r, seq);
        CHECK(eshkol_gpu_reduce_f64(&ba, &br, n, ESHKOL_REDUCE_MEAN) == 0, "reduce mean");
        CHECK(std::fabs(r - seq / (double)n) <= 1e-13, "reduce mean %.17g", r);

        /* transpose / softmax are CPU on Vulkan buffers */
        std::vector<double> t(n);
        EshkolGPUBuffer bt;
        eshkol_gpu_wrap_host(t.data(), n * 8, &bt);
        CHECK(eshkol_gpu_transpose_f64(&ba, &bt, 50, 100) == 0, "transpose");
        bool tok = true;
        for (uint64_t i = 0; i < 50 && tok; i++)
            for (uint64_t j = 0; j < 100 && tok; j++) tok = t[j * 50 + i] == a[i * 100 + j];
        CHECK(tok, "transpose result");
        CHECK(eshkol_gpu_softmax_f64(&ba, &bt, 50, 100) == 0, "softmax");
        double row = 0.0;
        for (uint64_t j = 0; j < 100; j++) row += t[j];
        CHECK(std::fabs(row - 1.0) < 1e-12, "softmax row sums to 1 (%.17g)", row);
        eshkol_gpu_free(&bt);
        eshkol_gpu_free(&br);
        eshkol_gpu_free(&ba);
        eshkol_gpu_free(&bb);
        eshkol_gpu_free(&bo);
    }

    /* allocated (mapped) buffers + sync */
    {
        const uint64_t n = 3000;
        EshkolGPUBuffer x, y, z;
        CHECK(eshkol_gpu_alloc(n * 8, ESHKOL_MEM_UNIFIED, &x) == 0, "alloc x");
        CHECK(eshkol_gpu_alloc(n * 8, ESHKOL_MEM_DEVICE, &y) == 0, "alloc y");
        CHECK(eshkol_gpu_alloc_aligned(n * 8, 256, ESHKOL_MEM_HOST, &z) == 0, "alloc_aligned z");
        double* xp = (double*)x.host_ptr;
        double* yp = (double*)y.host_ptr;
        double* zp = (double*)z.host_ptr;
        CHECK(xp && yp && zp, "allocated buffers are host-mapped");
        for (uint64_t i = 0; i < n; i++) { xp[i] = rnd(); yp[i] = rnd(); }
        CHECK(eshkol_gpu_sync(&x, ESHKOL_SYNC_HOST_TO_DEVICE) == 0, "sync h2d");
        CHECK(eshkol_gpu_elementwise_f64(&x, &y, &z, n, ESHKOL_ELEMWISE_MUL) == 0, "mul on allocated buffers");
        CHECK(eshkol_gpu_sync(&z, ESHKOL_SYNC_DEVICE_TO_HOST) == 0, "sync d2h");
        uint64_t bad = 0;
        for (uint64_t i = 0; i < n; i++) {
            volatile double want = xp[i] * yp[i];
            if (!same_bits(zp[i], want)) bad++;
        }
        CHECK(bad == 0, "mul on allocated buffers: %llu mismatches", (unsigned long long)bad);
        eshkol_gpu_wait(&z);
        eshkol_gpu_free(&x);
        eshkol_gpu_free(&y);
        eshkol_gpu_free(&z);
    }

    /* the generated-code entry point above the dispatch threshold */
    {
        const size_t saved = eshkol_gpu_get_threshold();
        eshkol_gpu_set_threshold(1);
        CHECK(eshkol_gpu_should_use(M * N) == 1, "should_use above threshold");
        std::vector<double> D(M * N, -7.0);
        eshkol_matmul_dispatch(A.data(), B.data(), D.data(), M, K, N, 0);
        check_matmul("eshkol_matmul_dispatch", D, Cref, mm_exact);
        eshkol_gpu_set_threshold(saved);
    }

    eshkol_gpu_shutdown();
    CHECK(eshkol_gpu_get_backend() == ESHKOL_GPU_NONE, "shutdown returns to CPU");

    if (g_fail) {
        std::printf("FAIL: %d of %d checks\n", g_fail, g_checks);
        return 1;
    }
    std::printf("PASS: %d checks through eshkol_gpu_* on Vulkan\n", g_checks);
    return 0;
}
