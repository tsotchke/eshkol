/*
 * Cross-Platform BLAS Backend Implementation for Eshkol
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/blas_backend.h"
#include "eshkol/backend/cpu_features.h"
#include "eshkol/logger.h"
#include <cstdlib>
#include <cstring>

// GPU backend for hardware acceleration
#ifdef ESHKOL_GPU_ENABLED
#include "eshkol/backend/gpu/gpu_memory.h"
#endif

// SIMD intrinsics headers
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define ESHKOL_HAS_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX__)
#include <immintrin.h>
#define ESHKOL_HAS_AVX 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define ESHKOL_HAS_SSE2 1
#endif
#endif

// ===== Adaptive Dispatch System =====
//
// Instead of hard thresholds, we use a cost model that estimates
// execution time for each backend and picks the fastest.
//
// Cost model parameters are calibrated at runtime based on actual
// measurements, adapting to the specific hardware.

namespace {

// Adaptive cost model parameters (calibrated at runtime)
struct CostModelParams {
    // BLAS parameters
    double blas_overhead_ns = 5000;      // Fixed dispatch overhead (5us)
    double blas_peak_gflops = 150;       // Peak throughput
    double blas_efficiency_scale = 10000; // Elements needed for full efficiency

    // GPU softfloat parameters
    double gpu_overhead_ns = 50000;      // Metal command buffer overhead (50us)
    double gpu_peak_gflops = 120;        // Peak throughput (measured ~137 at 2048x2048)
    double gpu_efficiency_scale = 500000; // Needs more parallelism to saturate

    // SIMD parameters
    double simd_overhead_ns = 100;       // Minimal overhead
    double simd_peak_gflops = 25;        // Peak throughput
    double simd_efficiency_scale = 1000; // Saturates quickly

    // Calibration state
    bool calibrated = false;
    int sample_count = 0;
};

static CostModelParams g_cost_model;

// Estimate time for each backend
struct DispatchCost {
    double blas_ns;
    double gpu_ns;
    double simd_ns;
    double scalar_ns;
};

inline DispatchCost estimate_matmul_cost(uint64_t M, uint64_t K, uint64_t N) {
    double flops = 2.0 * M * K * N;
    double elements = static_cast<double>(M * N);

    auto& p = g_cost_model;

    // BLAS: good for medium-large matrices, has fixed overhead
    double blas_efficiency = std::min(1.0, elements / p.blas_efficiency_scale);
    double blas_compute = flops / (p.blas_peak_gflops * blas_efficiency * 1e9) * 1e9;
    double blas_ns = p.blas_overhead_ns + blas_compute;

    // GPU: high overhead but massive parallelism for large matrices
    double gpu_efficiency = std::min(1.0, elements / p.gpu_efficiency_scale);
    double gpu_compute = flops / (p.gpu_peak_gflops * gpu_efficiency * 1e9) * 1e9;
    double gpu_ns = p.gpu_overhead_ns + gpu_compute;

    // SIMD: low overhead, good for small-medium matrices
    double simd_efficiency = std::min(1.0, elements / p.simd_efficiency_scale);
    double simd_compute = flops / (p.simd_peak_gflops * simd_efficiency * 1e9) * 1e9;
    double simd_ns = p.simd_overhead_ns + simd_compute;

    // Scalar: baseline (very slow for large matrices)
    double scalar_gflops = 0.5;  // ~500 MFLOPS
    double scalar_ns = flops / (scalar_gflops * 1e9) * 1e9;

    return {blas_ns, gpu_ns, simd_ns, scalar_ns};
}

enum class DispatchBackend {
    SCALAR,
    SIMD,
    BLAS,
    GPU
};

inline DispatchBackend select_best_backend(uint64_t M, uint64_t K, uint64_t N,
                                            bool gpu_available, bool blas_available,
                                            bool simd_available) {
    auto cost = estimate_matmul_cost(M, K, N);

    double best_time = cost.scalar_ns;
    DispatchBackend best = DispatchBackend::SCALAR;

    if (simd_available && cost.simd_ns < best_time) {
        best_time = cost.simd_ns;
        best = DispatchBackend::SIMD;
    }

    if (blas_available && cost.blas_ns < best_time) {
        best_time = cost.blas_ns;
        best = DispatchBackend::BLAS;
    }

    if (gpu_available && cost.gpu_ns < best_time) {
        best_time = cost.gpu_ns;
        best = DispatchBackend::GPU;
    }

    return best;
}

// Environment initialization
struct CostModelInitializer {
    CostModelInitializer() {
        if (const char* env = std::getenv("ESHKOL_BLAS_PEAK_GFLOPS")) {
            g_cost_model.blas_peak_gflops = std::atof(env);
        }
        if (const char* env = std::getenv("ESHKOL_GPU_PEAK_GFLOPS")) {
            g_cost_model.gpu_peak_gflops = std::atof(env);
        }
    }
};
static CostModelInitializer cost_model_init;

} // anonymous namespace

#ifdef ESHKOL_BLAS_ENABLED

namespace eshkol {
namespace blas {

// BLAS threshold: use cBLAS for matrices with >= this many elements
// Lowered from 4096 to 64 because our naive SIMD is not competitive with cBLAS.
// cBLAS overhead (~5μs) is negligible compared to its 100x performance advantage.
size_t g_blas_threshold = 64;

// Initialize from environment on startup
namespace {
    struct ThresholdInitializer {
        ThresholdInitializer() {
            if (const char* env = std::getenv("ESHKOL_BLAS_THRESHOLD")) {
                g_blas_threshold = static_cast<size_t>(std::atol(env));
            }
        }
    };
    static ThresholdInitializer init;
}

void blas_set_threshold(size_t threshold) {
    g_blas_threshold = threshold;
}

size_t blas_get_threshold() {
    return g_blas_threshold;
}

// ===== Backend Info =====

bool isAvailable() {
    return true;
}

const char* getBackendName() {
#ifdef ESHKOL_BLAS_ACCELERATE
    return "Apple Accelerate";
#elif defined(ESHKOL_BLAS_OPENBLAS)
    return "OpenBLAS";
#else
    return "Generic CBLAS";
#endif
}

// ===== Matrix Operations =====

void dgemm(char transA, char transB,
           int M, int N, int K,
           double alpha,
           const double* A, int lda,
           const double* B, int ldb,
           double beta,
           double* C, int ldc) {
    // Convert to CBLAS enum types
    CBLAS_TRANSPOSE ta = (transA == 'T' || transA == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (transB == 'T' || transB == 't') ? CblasTrans : CblasNoTrans;

    // CBLAS uses row-major by default with CblasRowMajor
    // Our tensors are stored in row-major order
    cblas_dgemm(CblasRowMajor, ta, tb,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}

void matmul(const double* A, const double* B, double* C,
            size_t M, size_t K, size_t N) {
    // Simple C = A * B (no transpose, alpha=1, beta=0)
    // A is M x K, B is K x N, C is M x N
    // All row-major
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0,  // alpha
                A, static_cast<int>(K),   // lda = K for row-major A[M][K]
                B, static_cast<int>(N),   // ldb = N for row-major B[K][N]
                0.0,  // beta
                C, static_cast<int>(N));  // ldc = N for row-major C[M][N]
}

// ===== Vector Operations =====

double ddot(int n, const double* x, int incx, const double* y, int incy) {
    return cblas_ddot(n, x, incx, y, incy);
}

void dscal(int n, double alpha, double* x, int incx) {
    cblas_dscal(n, alpha, x, incx);
}

void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}

} // namespace blas
} // namespace eshkol

#endif // ESHKOL_BLAS_ENABLED

// ===== SIMD Matmul Implementations =====

// Threshold for SIMD path (elements, not ops): 64 = 8x8 matrix
static constexpr size_t SIMD_THRESHOLD = 64;

// Scalar matmul - baseline implementation
static void matmul_scalar(const double* A, const double* B, double* C,
                          uint64_t M, uint64_t K, uint64_t N) {
    // C[i][j] = sum_k(A[i][k] * B[k][j])
    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (uint64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#if defined(ESHKOL_HAS_NEON)
// ===== Optimized ARM NEON SIMD matmul =====
// Uses cache blocking + 4x4 register micro-kernel for ~10-20x speedup over naive

// Block sizes tuned for Apple M-series L1 cache (64KB data, 128KB unified)
// Each block should fit in L1: 64*64*8 bytes = 32KB per matrix block
static constexpr uint64_t BLOCK_M = 64;
static constexpr uint64_t BLOCK_N = 64;
static constexpr uint64_t BLOCK_K = 64;

// 4x4 micro-kernel: computes a 4x4 block of C using register accumulation
// Processes 4 rows of A and 4 columns of B (as 2x float64x2_t = 4 doubles)
static inline void microkernel_4x4(
    const double* A, const double* B, double* C,
    uint64_t K, uint64_t lda, uint64_t ldb, uint64_t ldc) {

    // 4x4 = 16 accumulators, but NEON has 32 registers, so we use 4x2 = 8 float64x2_t
    // Each float64x2_t holds 2 doubles, so 8 registers = 16 doubles = 4x4 block
    float64x2_t c00 = vdupq_n_f64(0.0), c01 = vdupq_n_f64(0.0);  // C[0][0:3]
    float64x2_t c10 = vdupq_n_f64(0.0), c11 = vdupq_n_f64(0.0);  // C[1][0:3]
    float64x2_t c20 = vdupq_n_f64(0.0), c21 = vdupq_n_f64(0.0);  // C[2][0:3]
    float64x2_t c30 = vdupq_n_f64(0.0), c31 = vdupq_n_f64(0.0);  // C[3][0:3]

    for (uint64_t k = 0; k < K; ++k) {
        // Load 4 elements from column k of A (one per row)
        float64x2_t a0 = vdupq_n_f64(A[0 * lda + k]);
        float64x2_t a1 = vdupq_n_f64(A[1 * lda + k]);
        float64x2_t a2 = vdupq_n_f64(A[2 * lda + k]);
        float64x2_t a3 = vdupq_n_f64(A[3 * lda + k]);

        // Load row k of B (4 elements as 2x float64x2_t)
        float64x2_t b0 = vld1q_f64(&B[k * ldb + 0]);
        float64x2_t b1 = vld1q_f64(&B[k * ldb + 2]);

        // Accumulate: C[i][j] += A[i][k] * B[k][j]
        c00 = vfmaq_f64(c00, a0, b0); c01 = vfmaq_f64(c01, a0, b1);
        c10 = vfmaq_f64(c10, a1, b0); c11 = vfmaq_f64(c11, a1, b1);
        c20 = vfmaq_f64(c20, a2, b0); c21 = vfmaq_f64(c21, a2, b1);
        c30 = vfmaq_f64(c30, a3, b0); c31 = vfmaq_f64(c31, a3, b1);
    }

    // Store accumulated results back to C
    vst1q_f64(&C[0 * ldc + 0], vaddq_f64(vld1q_f64(&C[0 * ldc + 0]), c00));
    vst1q_f64(&C[0 * ldc + 2], vaddq_f64(vld1q_f64(&C[0 * ldc + 2]), c01));
    vst1q_f64(&C[1 * ldc + 0], vaddq_f64(vld1q_f64(&C[1 * ldc + 0]), c10));
    vst1q_f64(&C[1 * ldc + 2], vaddq_f64(vld1q_f64(&C[1 * ldc + 2]), c11));
    vst1q_f64(&C[2 * ldc + 0], vaddq_f64(vld1q_f64(&C[2 * ldc + 0]), c20));
    vst1q_f64(&C[2 * ldc + 2], vaddq_f64(vld1q_f64(&C[2 * ldc + 2]), c21));
    vst1q_f64(&C[3 * ldc + 0], vaddq_f64(vld1q_f64(&C[3 * ldc + 0]), c30));
    vst1q_f64(&C[3 * ldc + 2], vaddq_f64(vld1q_f64(&C[3 * ldc + 2]), c31));
}

// Scalar fallback for edge tiles
static inline void scalar_block(
    const double* A, const double* B, double* C,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t lda, uint64_t ldb, uint64_t ldc) {
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (uint64_t kk = 0; kk < k; ++kk) {
                sum += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void matmul_simd_neon(const double* A, const double* B, double* C,
                              uint64_t M, uint64_t K, uint64_t N) {
    // Zero the result matrix
    std::memset(C, 0, M * N * sizeof(double));

    // Cache-blocked matrix multiplication
    // Outer loops iterate over blocks, inner loops use micro-kernels
    for (uint64_t i0 = 0; i0 < M; i0 += BLOCK_M) {
        uint64_t m_block = std::min(BLOCK_M, M - i0);

        for (uint64_t j0 = 0; j0 < N; j0 += BLOCK_N) {
            uint64_t n_block = std::min(BLOCK_N, N - j0);

            for (uint64_t k0 = 0; k0 < K; k0 += BLOCK_K) {
                uint64_t k_block = std::min(BLOCK_K, K - k0);

                // Process 4x4 tiles within this block using micro-kernel
                uint64_t i = 0;
                for (; i + 4 <= m_block; i += 4) {
                    uint64_t j = 0;
                    for (; j + 4 <= n_block; j += 4) {
                        microkernel_4x4(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            k_block, K, N, N);
                    }
                    // Handle remaining columns with scalar
                    if (j < n_block) {
                        scalar_block(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            4, n_block - j, k_block, K, N, N);
                    }
                }
                // Handle remaining rows with scalar
                if (i < m_block) {
                    scalar_block(
                        &A[(i0 + i) * K + k0],
                        &B[k0 * N + j0],
                        &C[(i0 + i) * N + j0],
                        m_block - i, n_block, k_block, K, N, N);
                }
            }
        }
    }
}
#endif

#if defined(ESHKOL_HAS_AVX)
// ===== Optimized x86 AVX SIMD matmul =====
// Uses cache blocking + 4x8 register micro-kernel

// Block sizes for x86 L1 cache (~32KB)
static constexpr uint64_t AVX_BLOCK_M = 64;
static constexpr uint64_t AVX_BLOCK_N = 64;
static constexpr uint64_t AVX_BLOCK_K = 64;

// 4x8 micro-kernel: computes a 4x8 block of C using register accumulation
// Processes 4 rows of A and 8 columns of B (as 2x __m256d = 8 doubles)
static inline void avx_microkernel_4x8(
    const double* A, const double* B, double* C,
    uint64_t K, uint64_t lda, uint64_t ldb, uint64_t ldc) {

    // 4 rows x 8 cols = 32 doubles = 8x __m256d accumulators
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();  // C[0][0:7]
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();  // C[1][0:7]
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();  // C[2][0:7]
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();  // C[3][0:7]

    for (uint64_t k = 0; k < K; ++k) {
        // Load 4 elements from column k of A (one per row)
        __m256d a0 = _mm256_set1_pd(A[0 * lda + k]);
        __m256d a1 = _mm256_set1_pd(A[1 * lda + k]);
        __m256d a2 = _mm256_set1_pd(A[2 * lda + k]);
        __m256d a3 = _mm256_set1_pd(A[3 * lda + k]);

        // Load row k of B (8 elements as 2x __m256d)
        __m256d b0 = _mm256_loadu_pd(&B[k * ldb + 0]);
        __m256d b1 = _mm256_loadu_pd(&B[k * ldb + 4]);

        // Accumulate: C[i][j] += A[i][k] * B[k][j]
#if defined(__FMA__)
        c00 = _mm256_fmadd_pd(a0, b0, c00); c01 = _mm256_fmadd_pd(a0, b1, c01);
        c10 = _mm256_fmadd_pd(a1, b0, c10); c11 = _mm256_fmadd_pd(a1, b1, c11);
        c20 = _mm256_fmadd_pd(a2, b0, c20); c21 = _mm256_fmadd_pd(a2, b1, c21);
        c30 = _mm256_fmadd_pd(a3, b0, c30); c31 = _mm256_fmadd_pd(a3, b1, c31);
#else
        c00 = _mm256_add_pd(c00, _mm256_mul_pd(a0, b0));
        c01 = _mm256_add_pd(c01, _mm256_mul_pd(a0, b1));
        c10 = _mm256_add_pd(c10, _mm256_mul_pd(a1, b0));
        c11 = _mm256_add_pd(c11, _mm256_mul_pd(a1, b1));
        c20 = _mm256_add_pd(c20, _mm256_mul_pd(a2, b0));
        c21 = _mm256_add_pd(c21, _mm256_mul_pd(a2, b1));
        c30 = _mm256_add_pd(c30, _mm256_mul_pd(a3, b0));
        c31 = _mm256_add_pd(c31, _mm256_mul_pd(a3, b1));
#endif
    }

    // Store accumulated results back to C
    _mm256_storeu_pd(&C[0 * ldc + 0], _mm256_add_pd(_mm256_loadu_pd(&C[0 * ldc + 0]), c00));
    _mm256_storeu_pd(&C[0 * ldc + 4], _mm256_add_pd(_mm256_loadu_pd(&C[0 * ldc + 4]), c01));
    _mm256_storeu_pd(&C[1 * ldc + 0], _mm256_add_pd(_mm256_loadu_pd(&C[1 * ldc + 0]), c10));
    _mm256_storeu_pd(&C[1 * ldc + 4], _mm256_add_pd(_mm256_loadu_pd(&C[1 * ldc + 4]), c11));
    _mm256_storeu_pd(&C[2 * ldc + 0], _mm256_add_pd(_mm256_loadu_pd(&C[2 * ldc + 0]), c20));
    _mm256_storeu_pd(&C[2 * ldc + 4], _mm256_add_pd(_mm256_loadu_pd(&C[2 * ldc + 4]), c21));
    _mm256_storeu_pd(&C[3 * ldc + 0], _mm256_add_pd(_mm256_loadu_pd(&C[3 * ldc + 0]), c30));
    _mm256_storeu_pd(&C[3 * ldc + 4], _mm256_add_pd(_mm256_loadu_pd(&C[3 * ldc + 4]), c31));
}

// Scalar fallback for edge tiles (shared with other implementations)
static inline void avx_scalar_block(
    const double* A, const double* B, double* C,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t lda, uint64_t ldb, uint64_t ldc) {
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (uint64_t kk = 0; kk < k; ++kk) {
                sum += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void matmul_simd_avx(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    std::memset(C, 0, M * N * sizeof(double));

    for (uint64_t i0 = 0; i0 < M; i0 += AVX_BLOCK_M) {
        uint64_t m_block = std::min(AVX_BLOCK_M, M - i0);

        for (uint64_t j0 = 0; j0 < N; j0 += AVX_BLOCK_N) {
            uint64_t n_block = std::min(AVX_BLOCK_N, N - j0);

            for (uint64_t k0 = 0; k0 < K; k0 += AVX_BLOCK_K) {
                uint64_t k_block = std::min(AVX_BLOCK_K, K - k0);

                // Process 4x8 tiles using micro-kernel
                uint64_t i = 0;
                for (; i + 4 <= m_block; i += 4) {
                    uint64_t j = 0;
                    for (; j + 8 <= n_block; j += 8) {
                        avx_microkernel_4x8(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            k_block, K, N, N);
                    }
                    if (j < n_block) {
                        avx_scalar_block(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            4, n_block - j, k_block, K, N, N);
                    }
                }
                if (i < m_block) {
                    avx_scalar_block(
                        &A[(i0 + i) * K + k0],
                        &B[k0 * N + j0],
                        &C[(i0 + i) * N + j0],
                        m_block - i, n_block, k_block, K, N, N);
                }
            }
        }
    }
}

#elif defined(ESHKOL_HAS_SSE2)
// ===== Optimized x86 SSE2 SIMD matmul =====
// Uses cache blocking + 4x4 register micro-kernel

static constexpr uint64_t SSE_BLOCK_M = 64;
static constexpr uint64_t SSE_BLOCK_N = 64;
static constexpr uint64_t SSE_BLOCK_K = 64;

// 4x4 micro-kernel for SSE2 (2 doubles per register)
static inline void sse_microkernel_4x4(
    const double* A, const double* B, double* C,
    uint64_t K, uint64_t lda, uint64_t ldb, uint64_t ldc) {

    __m128d c00 = _mm_setzero_pd(), c01 = _mm_setzero_pd();
    __m128d c10 = _mm_setzero_pd(), c11 = _mm_setzero_pd();
    __m128d c20 = _mm_setzero_pd(), c21 = _mm_setzero_pd();
    __m128d c30 = _mm_setzero_pd(), c31 = _mm_setzero_pd();

    for (uint64_t k = 0; k < K; ++k) {
        __m128d a0 = _mm_set1_pd(A[0 * lda + k]);
        __m128d a1 = _mm_set1_pd(A[1 * lda + k]);
        __m128d a2 = _mm_set1_pd(A[2 * lda + k]);
        __m128d a3 = _mm_set1_pd(A[3 * lda + k]);

        __m128d b0 = _mm_loadu_pd(&B[k * ldb + 0]);
        __m128d b1 = _mm_loadu_pd(&B[k * ldb + 2]);

        c00 = _mm_add_pd(c00, _mm_mul_pd(a0, b0));
        c01 = _mm_add_pd(c01, _mm_mul_pd(a0, b1));
        c10 = _mm_add_pd(c10, _mm_mul_pd(a1, b0));
        c11 = _mm_add_pd(c11, _mm_mul_pd(a1, b1));
        c20 = _mm_add_pd(c20, _mm_mul_pd(a2, b0));
        c21 = _mm_add_pd(c21, _mm_mul_pd(a2, b1));
        c30 = _mm_add_pd(c30, _mm_mul_pd(a3, b0));
        c31 = _mm_add_pd(c31, _mm_mul_pd(a3, b1));
    }

    _mm_storeu_pd(&C[0 * ldc + 0], _mm_add_pd(_mm_loadu_pd(&C[0 * ldc + 0]), c00));
    _mm_storeu_pd(&C[0 * ldc + 2], _mm_add_pd(_mm_loadu_pd(&C[0 * ldc + 2]), c01));
    _mm_storeu_pd(&C[1 * ldc + 0], _mm_add_pd(_mm_loadu_pd(&C[1 * ldc + 0]), c10));
    _mm_storeu_pd(&C[1 * ldc + 2], _mm_add_pd(_mm_loadu_pd(&C[1 * ldc + 2]), c11));
    _mm_storeu_pd(&C[2 * ldc + 0], _mm_add_pd(_mm_loadu_pd(&C[2 * ldc + 0]), c20));
    _mm_storeu_pd(&C[2 * ldc + 2], _mm_add_pd(_mm_loadu_pd(&C[2 * ldc + 2]), c21));
    _mm_storeu_pd(&C[3 * ldc + 0], _mm_add_pd(_mm_loadu_pd(&C[3 * ldc + 0]), c30));
    _mm_storeu_pd(&C[3 * ldc + 2], _mm_add_pd(_mm_loadu_pd(&C[3 * ldc + 2]), c31));
}

static inline void sse_scalar_block(
    const double* A, const double* B, double* C,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t lda, uint64_t ldb, uint64_t ldc) {
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (uint64_t kk = 0; kk < k; ++kk) {
                sum += A[i * lda + kk] * B[kk * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

static void matmul_simd_sse2(const double* A, const double* B, double* C,
                              uint64_t M, uint64_t K, uint64_t N) {
    std::memset(C, 0, M * N * sizeof(double));

    for (uint64_t i0 = 0; i0 < M; i0 += SSE_BLOCK_M) {
        uint64_t m_block = std::min(SSE_BLOCK_M, M - i0);

        for (uint64_t j0 = 0; j0 < N; j0 += SSE_BLOCK_N) {
            uint64_t n_block = std::min(SSE_BLOCK_N, N - j0);

            for (uint64_t k0 = 0; k0 < K; k0 += SSE_BLOCK_K) {
                uint64_t k_block = std::min(SSE_BLOCK_K, K - k0);

                uint64_t i = 0;
                for (; i + 4 <= m_block; i += 4) {
                    uint64_t j = 0;
                    for (; j + 4 <= n_block; j += 4) {
                        sse_microkernel_4x4(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            k_block, K, N, N);
                    }
                    if (j < n_block) {
                        sse_scalar_block(
                            &A[(i0 + i) * K + k0],
                            &B[k0 * N + (j0 + j)],
                            &C[(i0 + i) * N + (j0 + j)],
                            4, n_block - j, k_block, K, N, N);
                    }
                }
                if (i < m_block) {
                    sse_scalar_block(
                        &A[(i0 + i) * K + k0],
                        &B[k0 * N + j0],
                        &C[(i0 + i) * N + j0],
                        m_block - i, n_block, k_block, K, N, N);
                }
            }
        }
    }
}
#endif

// Dispatch to best available SIMD implementation
static void matmul_simd(const double* A, const double* B, double* C,
                        uint64_t M, uint64_t K, uint64_t N) {
#if defined(ESHKOL_HAS_NEON)
    matmul_simd_neon(A, B, C, M, K, N);
#elif defined(ESHKOL_HAS_AVX)
    matmul_simd_avx(A, B, C, M, K, N);
#elif defined(ESHKOL_HAS_SSE2)
    matmul_simd_sse2(A, B, C, M, K, N);
#else
    // No SIMD available, use scalar
    matmul_scalar(A, B, C, M, K, N);
#endif
}

// Check if SIMD is available at runtime
static bool simd_available() {
    auto& caps = eshkol::CPUCapabilities::instance();
    return caps.getVectorWidth() > 1;
}

// ===== C-compatible Runtime Functions =====

extern "C" {

void eshkol_matmul_f64(const double* A, const double* B, double* C,
                        uint64_t M, uint64_t K, uint64_t N) {
    const uint64_t output_elements = M * N;

    // ===== Fast path dispatch (zero cost model overhead) =====
    // Thresholds tuned empirically on Apple M-series chips.

    // Tiny matrices (<= 16 elements, up to 4x4): scalar is fastest
    // SIMD/BLAS overhead exceeds compute time for these
    if (output_elements <= 16) {
        matmul_scalar(A, B, C, M, K, N);
        return;
    }

    // Small to large matrices (17 - 16M elements): cBLAS
    // cBLAS achieves 600-900 GFLOPS up to 4000x4000
    // GPU needs 16M+ elements to amortize shader compilation + transfer overhead
#ifdef ESHKOL_BLAS_ENABLED
    if (output_elements < 16000000) {
        eshkol::blas::matmul(A, B, C, M, K, N);
        return;
    }
#else
    // Fallback to SIMD if BLAS not available
    if (output_elements < 16000000) {
        matmul_simd(A, B, C, M, K, N);
        return;
    }
#endif

    // ===== Very large matrices (>= 16M elements): GPU vs BLAS =====
    // Lazy-init GPU only when we have matrices large enough to benefit
    static bool gpu_initialized = false;
    static bool gpu_available = false;

#ifdef ESHKOL_GPU_ENABLED
    if (!gpu_initialized) {
        eshkol_gpu_init();
        gpu_available = (eshkol_gpu_get_backend() != ESHKOL_GPU_NONE);
        gpu_initialized = true;
    }
#endif

    // For large matrices, use cost model to decide GPU vs BLAS
    bool simd_avail = simd_available();
    bool blas_available = true;
    DispatchBackend backend = select_best_backend(M, K, N, gpu_available, blas_available, simd_avail);

    // Execute on selected backend
    switch (backend) {
        case DispatchBackend::GPU: {
#ifdef ESHKOL_GPU_ENABLED
            EshkolGPUBuffer buf_a, buf_b, buf_c;
            size_t a_size = M * K * sizeof(double);
            size_t b_size = K * N * sizeof(double);
            size_t c_size = M * N * sizeof(double);

            if (eshkol_gpu_wrap_host((void*)A, a_size, &buf_a) == 0 &&
                eshkol_gpu_wrap_host((void*)B, b_size, &buf_b) == 0 &&
                eshkol_gpu_wrap_host((void*)C, c_size, &buf_c) == 0) {

                if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                    eshkol_gpu_free(&buf_a);
                    eshkol_gpu_free(&buf_b);
                    eshkol_gpu_free(&buf_c);
                    return;
                }
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
            }
            // GPU failed, fall through to BLAS
#endif
        }
        [[fallthrough]];

        case DispatchBackend::BLAS: {
#ifdef ESHKOL_BLAS_ENABLED
            eshkol::blas::matmul(A, B, C, M, K, N);
            return;
#endif
        }
        [[fallthrough]];

        case DispatchBackend::SIMD: {
            if (simd_avail) {
                matmul_simd(A, B, C, M, K, N);
                return;
            }
        }
        [[fallthrough]];

        case DispatchBackend::SCALAR:
        default:
            matmul_scalar(A, B, C, M, K, N);
            break;
    }
}

int eshkol_blas_available(void) {
#ifdef ESHKOL_BLAS_ENABLED
    return 1;
#else
    return 0;
#endif
}

uint64_t eshkol_blas_get_threshold(void) {
#ifdef ESHKOL_BLAS_ENABLED
    return static_cast<uint64_t>(eshkol::blas::blas_get_threshold());
#else
    return 0;
#endif
}

void eshkol_blas_set_threshold(uint64_t threshold) {
#ifdef ESHKOL_BLAS_ENABLED
    eshkol::blas::blas_set_threshold(static_cast<size_t>(threshold));
#else
    (void)threshold;
#endif
}

} // extern "C"
