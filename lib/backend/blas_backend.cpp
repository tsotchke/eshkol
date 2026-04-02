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
    // BLAS parameters (calibrated: Apple Accelerate AMX peaks at ~1100 GFLOPS)
    double blas_overhead_ns = 5000;      // Fixed dispatch overhead (5us)
    double blas_peak_gflops = 1100;      // Peak throughput (measured on M-series)
    double blas_efficiency_scale = 10000; // Elements needed for full efficiency

    // GPU softfloat parameters (sf64 emulation via Metal compute shaders)
    // Current sf64 throughput is ~100-200 GFLOPS — well below native cBLAS.
    // GPU will only be selected when it genuinely outperforms cBLAS.
    double gpu_overhead_ns = 200000;     // Metal command buffer + data transfer (200us)
    double gpu_peak_gflops = 200;        // Measured sf64 throughput
    double gpu_efficiency_scale = 100000000; // Needs 100M+ elements to saturate GPU

    // SIMD parameters
    double simd_overhead_ns = 100;       // Minimal overhead
    double simd_peak_gflops = 25;        // Peak throughput
    double simd_efficiency_scale = 1000; // Saturates quickly

    // Calibration state
    bool calibrated = false;
    int sample_count = 0;
};

static CostModelParams g_cost_model;

// GPU matmul threshold: override via ESHKOL_GPU_MATMUL_THRESHOLD env var
// Default 1B elements — GPU only for super-massive matrices
static uint64_t g_gpu_matmul_threshold = 1000000000ULL;

// GPU precision tier: "exact" (sf64), "high" (df64), "fast" (f32)
// Override via ESHKOL_GPU_PRECISION env var
static int g_gpu_precision_tier = 0;  // 0=exact, 1=high, 2=fast

struct GPUEnvInitializer {
    GPUEnvInitializer() {
        if (const char* env = std::getenv("ESHKOL_GPU_MATMUL_THRESHOLD")) {
            g_gpu_matmul_threshold = static_cast<uint64_t>(std::atoll(env));
        }
        if (const char* env = std::getenv("ESHKOL_GPU_PRECISION")) {
            if (std::strcmp(env, "high") == 0) g_gpu_precision_tier = 1;
            else if (std::strcmp(env, "fast") == 0) g_gpu_precision_tier = 2;
            else g_gpu_precision_tier = 0;  // "exact" or default
        }
    }
};
static GPUEnvInitializer g_gpu_env_init;

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

// Matmul backward: given C = A @ B, accumulate gradients dA and dB
// dA += grad_C @ B^T : (M×N) @ (N×K) = (M×K)
// dB += A^T @ grad_C : (K×M) @ (M×N) = (K×N)
void matmul_backward(const double* grad_c, const double* a, const double* b,
                     double* grad_a, double* grad_b,
                     size_t M, size_t K, size_t N) {
    // dA += grad_C @ B^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(M), static_cast<int>(K), static_cast<int>(N),
                1.0,
                grad_c, static_cast<int>(N),
                b, static_cast<int>(N),
                1.0,   // beta=1 to accumulate
                grad_a, static_cast<int>(K));

    // dB += A^T @ grad_C
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(K), static_cast<int>(N), static_cast<int>(M),
                1.0,
                a, static_cast<int>(K),
                grad_c, static_cast<int>(N),
                1.0,   // beta=1 to accumulate
                grad_b, static_cast<int>(N));
}

// Matrix-vector multiply: y = alpha * A * x + beta * y
void dgemv(char trans, int M, int N,
           double alpha, const double* A, int lda,
           const double* x, int incx,
           double beta, double* y, int incy) {
    CBLAS_TRANSPOSE t = (trans == 'T' || trans == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasRowMajor, t, M, N, alpha, A, lda, x, incx, beta, y, incy);
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

// ===== Batched Matrix Operations =====

void batched_dgemm(char transA, char transB,
                   int M, int N, int K,
                   double alpha,
                   const double* A, int lda,
                   const double* B, int ldb,
                   double beta,
                   double* C, int ldc,
                   int batch_count) {
    CBLAS_TRANSPOSE ta = (transA == 'T' || transA == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (transB == 'T' || transB == 't') ? CblasTrans : CblasNoTrans;

    size_t a_stride = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t b_stride = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t c_stride = static_cast<size_t>(M) * static_cast<size_t>(N);

    for (int b = 0; b < batch_count; b++) {
        cblas_dgemm(CblasRowMajor, ta, tb,
                    M, N, K,
                    alpha,
                    A + b * a_stride, lda,
                    B + b * b_stride, ldb,
                    beta,
                    C + b * c_stride, ldc);
    }
}

void batched_matmul(const double* A, const double* B, double* C,
                    size_t M, size_t K, size_t N, size_t batch_count) {
    size_t a_stride = M * K;
    size_t b_stride = K * N;
    size_t c_stride = M * N;

    for (size_t b = 0; b < batch_count; b++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0,
                    A + b * a_stride, static_cast<int>(K),
                    B + b * b_stride, static_cast<int>(N),
                    0.0,
                    C + b * c_stride, static_cast<int>(N));
    }
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
    eshkol_debug("matmul_f64: M=%llu K=%llu N=%llu", M, K, N);
    const uint64_t output_elements = M * N;

    // ===== Fast path dispatch (zero cost model overhead) =====
    // Thresholds tuned empirically on Apple M-series chips.

    // Tiny matrices (<= 16 elements, up to 4x4): scalar is fastest
    // SIMD/BLAS overhead exceeds compute time for these
    if (output_elements <= 16) {
        matmul_scalar(A, B, C, M, K, N);
        return;
    }

    // Small to super-massive matrices (17 - 1B elements): cBLAS
    // cBLAS (Apple Accelerate AMX) achieves 1100+ GFLOPS even at 225M elements.
    // GPU (sf64 softfloat) only used when cBLAS is computationally infeasible
    // — matrices ~31600×31600 and larger (1B+ output elements).
#ifdef ESHKOL_BLAS_ENABLED
    if (output_elements < g_gpu_matmul_threshold) {
        eshkol::blas::matmul(A, B, C, M, K, N);
        return;
    }
#else
    // Fallback to SIMD if BLAS not available
    if (output_elements < g_gpu_matmul_threshold) {
        matmul_simd(A, B, C, M, K, N);
        return;
    }
#endif

    // ===== Super massive matrices (>= 1B elements): GPU vs BLAS =====
    // Lazy-init GPU only when cBLAS is computationally infeasible
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

/**
 * eshkol_matmul_backward_f64 — GPU/BLAS/SIMD-dispatched backward matmul.
 *
 * Forward: C = A @ B, where A is (M,K), B is (K,N), C is (M,N)
 * Backward: grad_A = grad_out @ B^T, grad_B = A^T @ grad_out
 *
 * Uses the same dispatch hierarchy as eshkol_matmul_f64:
 *   GPU → BLAS → SIMD → scalar
 */
/* Helper: explicit out-of-place transpose for GPU/SIMD paths that can't use CblasTrans */
static void transpose_f64(const double* src, double* dst, uint64_t rows, uint64_t cols) {
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            dst[j * rows + i] = src[i * cols + j];
}

void eshkol_matmul_backward_f64(
    const double* grad_out,
    const double* saved_A, const double* saved_B,
    double* grad_A, double* grad_B,
    uint64_t M, uint64_t K, uint64_t N)
{
    eshkol_debug("matmul_backward_f64: M=%llu K=%llu N=%llu", M, K, N);

    /* Backward matmul = 2 matmuls:
     *   grad_A = grad_out @ B^T: (M,N) @ (N,K) -> (M,K)
     *   grad_B = A^T @ grad_out: (K,M) @ (M,N) -> (K,N)
     *
     * Dispatch: GPU → BLAS → SIMD → scalar (mirrors forward eshkol_matmul_f64) */

    const uint64_t output_a = M * K;
    const uint64_t output_b = K * N;

    // ===== Tiny matrices: scalar =====
    if (output_a <= 16 && output_b <= 16) {
        memset(grad_A, 0, (size_t)(M * K) * sizeof(double));
        for (uint64_t i = 0; i < M; i++)
            for (uint64_t j = 0; j < K; j++)
                for (uint64_t n = 0; n < N; n++)
                    grad_A[i * K + j] += grad_out[i * N + n] * saved_B[j * N + n];
        memset(grad_B, 0, (size_t)(K * N) * sizeof(double));
        for (uint64_t j = 0; j < K; j++)
            for (uint64_t n = 0; n < N; n++)
                for (uint64_t i = 0; i < M; i++)
                    grad_B[j * N + n] += saved_A[i * K + j] * grad_out[i * N + n];
        return;
    }

    // ===== GPU path for super-massive matrices =====
#ifdef ESHKOL_GPU_ENABLED
    {
        uint64_t max_output = output_a > output_b ? output_a : output_b;
        if (max_output >= g_gpu_matmul_threshold) {
            static bool gpu_initialized = false;
            static bool gpu_available = false;
            if (!gpu_initialized) {
                eshkol_gpu_init();
                gpu_available = (eshkol_gpu_get_backend() != ESHKOL_GPU_NONE);
                gpu_initialized = true;
            }
            if (gpu_available) {
                /* GPU matmul doesn't support CblasTrans — need explicit transpose.
                 * Transpose B → B^T, then: grad_A = grad_out @ B^T via eshkol_gpu_matmul_f64
                 * Transpose A → A^T, then: grad_B = A^T @ grad_out via eshkol_gpu_matmul_f64 */
                double* B_T = (double*)malloc((size_t)(K * N) * sizeof(double));
                double* A_T = (double*)malloc((size_t)(M * K) * sizeof(double));
                if (B_T && A_T) {
                    transpose_f64(saved_B, B_T, K, N);  // B(K,N) → B^T(N,K)
                    transpose_f64(saved_A, A_T, M, K);  // A(M,K) → A^T(K,M)

                    /* grad_A = grad_out @ B^T: (M,N) @ (N,K) -> (M,K) */
                    EshkolGPUBuffer buf_go, buf_bt, buf_ga;
                    if (eshkol_gpu_wrap_host((void*)grad_out, M*N*sizeof(double), &buf_go) == 0 &&
                        eshkol_gpu_wrap_host(B_T, N*K*sizeof(double), &buf_bt) == 0 &&
                        eshkol_gpu_wrap_host(grad_A, M*K*sizeof(double), &buf_ga) == 0) {
                        if (eshkol_gpu_matmul_f64(&buf_go, &buf_bt, &buf_ga, M, N, K) == 0) {
                            eshkol_gpu_free(&buf_go); eshkol_gpu_free(&buf_bt); eshkol_gpu_free(&buf_ga);

                            /* grad_B = A^T @ grad_out: (K,M) @ (M,N) -> (K,N) */
                            EshkolGPUBuffer buf_at, buf_go2, buf_gb;
                            if (eshkol_gpu_wrap_host(A_T, K*M*sizeof(double), &buf_at) == 0 &&
                                eshkol_gpu_wrap_host((void*)grad_out, M*N*sizeof(double), &buf_go2) == 0 &&
                                eshkol_gpu_wrap_host(grad_B, K*N*sizeof(double), &buf_gb) == 0) {
                                if (eshkol_gpu_matmul_f64(&buf_at, &buf_go2, &buf_gb, K, M, N) == 0) {
                                    eshkol_gpu_free(&buf_at); eshkol_gpu_free(&buf_go2); eshkol_gpu_free(&buf_gb);
                                    free(B_T); free(A_T);
                                    return;  // GPU success for both gradients
                                }
                                eshkol_gpu_free(&buf_at); eshkol_gpu_free(&buf_go2); eshkol_gpu_free(&buf_gb);
                            }
                        } else {
                            eshkol_gpu_free(&buf_go); eshkol_gpu_free(&buf_bt); eshkol_gpu_free(&buf_ga);
                        }
                    }
                }
                free(B_T); free(A_T);
                // GPU failed, fall through to BLAS
            }
        }
    }
#endif

    // ===== BLAS path: uses CblasTrans flags, no explicit transpose =====
#ifdef ESHKOL_BLAS_ENABLED
    memset(grad_A, 0, (size_t)(M * K) * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(M), static_cast<int>(K), static_cast<int>(N),
                1.0, grad_out, static_cast<int>(N),
                saved_B, static_cast<int>(N),
                0.0, grad_A, static_cast<int>(K));

    memset(grad_B, 0, (size_t)(K * N) * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(K), static_cast<int>(N), static_cast<int>(M),
                1.0, saved_A, static_cast<int>(K),
                grad_out, static_cast<int>(N),
                0.0, grad_B, static_cast<int>(N));
    return;
#endif

    // ===== Scalar fallback =====
    memset(grad_A, 0, (size_t)(M * K) * sizeof(double));
    for (uint64_t i = 0; i < M; i++)
        for (uint64_t j = 0; j < K; j++)
            for (uint64_t n = 0; n < N; n++)
                grad_A[i * K + j] += grad_out[i * N + n] * saved_B[j * N + n];
    memset(grad_B, 0, (size_t)(K * N) * sizeof(double));
    for (uint64_t j = 0; j < K; j++)
        for (uint64_t n = 0; n < N; n++)
            for (uint64_t i = 0; i < M; i++)
                grad_B[j * N + n] += saved_A[i * K + j] * grad_out[i * N + n];
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

void eshkol_batched_matmul_f64(const double* A, const double* B, double* C,
                                uint64_t M, uint64_t K, uint64_t N,
                                uint64_t batch_count) {
    size_t a_stride = M * K;
    size_t b_stride = K * N;
    size_t c_stride = M * N;

    for (uint64_t b = 0; b < batch_count; b++) {
#ifdef ESHKOL_BLAS_ENABLED
        eshkol::blas::matmul(A + b * a_stride, B + b * b_stride, C + b * c_stride,
                             M, K, N);
#else
        matmul_simd(A + b * a_stride, B + b * b_stride, C + b * c_stride,
                    M, K, N);
#endif
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * XLA Runtime Stubs — CPU fallbacks when XLA/StableHLO is not linked.
 *
 * The tensor codegen emits calls to these functions unconditionally.
 * When XLA is enabled, xla_runtime.cpp provides the real implementations
 * with GPU dispatch. When XLA is not enabled, these stubs provide
 * scalar CPU implementations so programs still work.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifndef ESHKOL_XLA_ENABLED

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cfloat>

/* Tensor struct — must match xla_runtime.cpp definition */
typedef struct {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;       /* doubles stored as int64 bit patterns */
    uint64_t  total_elements;
} xla_stub_tensor_t;

extern "C" xla_stub_tensor_t* arena_allocate_tensor_full(
    void* arena, uint64_t num_dims, uint64_t total_elements);

/* Helper: write double to tensor elements (stored as int64 bit pattern) */
static void tensor_set(xla_stub_tensor_t* t, int64_t idx, double val) {
    union { double d; int64_t i; } u; u.d = val;
    t->elements[idx] = u.i;
}
static double tensor_get(const int64_t* elements, int64_t idx) {
    union { int64_t i; double d; } u; u.i = elements[idx];
    return u.d;
}

void* eshkol_xla_reduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t op_code)
{
    if (total_elements <= 0 || !data) return nullptr;

    /* Full reduction (axis == -1): return scalar as 1-element tensor */
    if (axis == -1) {
        double result;
        /* Op codes from tensor_codegen.cpp: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4 */
        switch (op_code) {
            case 0: /* SUM */  result = 0; for (int64_t i = 0; i < total_elements; i++) result += data[i]; break;
            case 1: /* MEAN */ result = 0; for (int64_t i = 0; i < total_elements; i++) result += data[i]; result /= total_elements; break;
            case 2: /* MAX */  result = data[0]; for (int64_t i = 1; i < total_elements; i++) if (data[i] > result) result = data[i]; break;
            case 3: /* MIN */  result = data[0]; for (int64_t i = 1; i < total_elements; i++) if (data[i] < result) result = data[i]; break;
            case 4: /* PROD */ result = 1; for (int64_t i = 0; i < total_elements; i++) result *= data[i]; break;
            default: result = 0; break;
        }
        xla_stub_tensor_t* t = arena_allocate_tensor_full(arena, 1, 1);
        if (!t) return nullptr;
        t->dimensions[0] = 1;
        tensor_set(t, 0, result);
        return t;
    }

    /* Axis-specific reduction */
    if (axis < 0 || axis >= rank) return nullptr;
    int64_t outer = 1, mid = (int64_t)shape[axis], inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= (int64_t)shape[i];
    for (int64_t i = axis + 1; i < rank; i++) inner *= (int64_t)shape[i];
    int64_t out_total = outer * inner;

    xla_stub_tensor_t* t = arena_allocate_tensor_full(arena, rank - 1, (uint64_t)out_total);
    if (!t) return nullptr;
    int d = 0;
    for (int64_t i = 0; i < rank; i++) if (i != axis) t->dimensions[d++] = shape[i];

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            /* Op codes: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4 */
            double acc;
            switch (op_code) {
                case 0: case 1: acc = 0; break;  /* SUM, MEAN: init 0 */
                case 4: acc = 1; break;           /* PROD: init 1 */
                case 2: case 3: acc = data[o * mid * inner + i]; break; /* MAX, MIN: init first */
                default: acc = 0; break;
            }
            for (int64_t m = 0; m < mid; m++) {
                double v = data[o * mid * inner + m * inner + i];
                switch (op_code) {
                    case 0: case 1: acc += v; break;  /* SUM, MEAN */
                    case 4: acc *= v; break;           /* PROD */
                    case 2: if (v > acc) acc = v; break; /* MAX */
                    case 3: if (v < acc) acc = v; break; /* MIN */
                    default: break;
                }
            }
            if (op_code == 1) acc /= mid;  /* MEAN */
            tensor_set(t, o * inner + i, acc);
        }
    }
    return t;
}

void* eshkol_xla_softmax(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis)
{
    if (total_elements <= 0 || !data || rank <= 0) return nullptr;
    if (axis < 0) axis = rank - 1;

    xla_stub_tensor_t* t = arena_allocate_tensor_full(arena, (uint64_t)rank, (uint64_t)total_elements);
    if (!t) return nullptr;
    for (int64_t i = 0; i < rank; i++) t->dimensions[i] = shape[i];

    int64_t outer = 1, mid = (int64_t)shape[axis], inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= (int64_t)shape[i];
    for (int64_t i = axis + 1; i < rank; i++) inner *= (int64_t)shape[i];

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            double mx = -DBL_MAX;
            for (int64_t m = 0; m < mid; m++) {
                double v = data[o * mid * inner + m * inner + i];
                if (v > mx) mx = v;
            }
            double sum = 0;
            for (int64_t m = 0; m < mid; m++) {
                double v = exp(data[o * mid * inner + m * inner + i] - mx);
                tensor_set(t, o * mid * inner + m * inner + i, v);
                sum += v;
            }
            for (int64_t m = 0; m < mid; m++) {
                double v = tensor_get(t->elements, o * mid * inner + m * inner + i);
                tensor_set(t, o * mid * inner + m * inner + i, v / sum);
            }
        }
    }
    return t;
}

void* eshkol_xla_normalize(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    const double* gamma,
    const double* beta,
    double epsilon)
{
    if (total_elements <= 0 || !data || rank <= 0) return nullptr;
    if (axis < 0) axis = rank - 1;
    int64_t feature_size = (int64_t)shape[axis];
    int64_t num_samples = total_elements / feature_size;

    xla_stub_tensor_t* t = arena_allocate_tensor_full(arena, (uint64_t)rank, (uint64_t)total_elements);
    if (!t) return nullptr;
    for (int64_t i = 0; i < rank; i++) t->dimensions[i] = shape[i];

    for (int64_t s = 0; s < num_samples; s++) {
        const double* row = data + s * feature_size;
        double mean = 0;
        for (int64_t f = 0; f < feature_size; f++) mean += row[f];
        mean /= feature_size;
        double var = 0;
        for (int64_t f = 0; f < feature_size; f++) { double d = row[f] - mean; var += d * d; }
        var /= feature_size;
        double inv_std = 1.0 / sqrt(var + epsilon);
        for (int64_t f = 0; f < feature_size; f++) {
            double norm = (row[f] - mean) * inv_std;
            double val = gamma ? gamma[f] * norm + (beta ? beta[f] : 0) : norm;
            tensor_set(t, s * feature_size + f, val);
        }
    }
    return t;
}

void* eshkol_xla_argreduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t is_max)
{
    if (total_elements <= 0 || !data || rank <= 0) return nullptr;
    if (axis < 0) axis = rank - 1;

    int64_t outer = 1, mid = (int64_t)shape[axis], inner = 1;
    for (int64_t i = 0; i < axis; i++) outer *= (int64_t)shape[i];
    for (int64_t i = axis + 1; i < rank; i++) inner *= (int64_t)shape[i];
    int64_t out_total = outer * inner;

    xla_stub_tensor_t* t = arena_allocate_tensor_full(arena, (uint64_t)(rank - 1), (uint64_t)out_total);
    if (!t) return nullptr;
    int d = 0;
    for (int64_t i = 0; i < rank; i++) if (i != axis) t->dimensions[d++] = shape[i];

    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < inner; i++) {
            double best = data[o * mid * inner + i];
            int64_t best_idx = 0;
            for (int64_t m = 1; m < mid; m++) {
                double v = data[o * mid * inner + m * inner + i];
                if (is_max ? (v > best) : (v < best)) { best = v; best_idx = m; }
            }
            tensor_set(t, o * inner + i, (double)best_idx);
        }
    }
    return t;
}

#endif /* !ESHKOL_XLA_ENABLED */

} // extern "C"
