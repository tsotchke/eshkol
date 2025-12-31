/*
 * Cross-Platform BLAS Backend Implementation for Eshkol
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/blas_backend.h"
#include "eshkol/backend/cpu_features.h"
#include <cstdlib>
#include <cstring>

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

#ifdef ESHKOL_BLAS_ENABLED

namespace eshkol {
namespace blas {

// ===== Global Threshold =====

// Default: 4096 elements (64x64 matrix)
// Below this, SIMD is often faster due to BLAS call overhead
size_t g_blas_threshold = 4096;

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
// ARM NEON SIMD matmul - processes 2 doubles at a time
static void matmul_simd_neon(const double* A, const double* B, double* C,
                              uint64_t M, uint64_t K, uint64_t N) {
    // Zero the result matrix
    std::memset(C, 0, M * N * sizeof(double));

    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t k = 0; k < K; ++k) {
            // Broadcast A[i][k] to all lanes
            float64x2_t a_val = vdupq_n_f64(A[i * K + k]);

            // Process 2 elements at a time
            uint64_t j = 0;
            for (; j + 2 <= N; j += 2) {
                // Load B[k][j:j+2] and C[i][j:j+2]
                float64x2_t b_vec = vld1q_f64(&B[k * N + j]);
                float64x2_t c_vec = vld1q_f64(&C[i * N + j]);

                // C += A[i][k] * B[k][j:j+2]
                c_vec = vfmaq_f64(c_vec, a_val, b_vec);

                // Store result
                vst1q_f64(&C[i * N + j], c_vec);
            }

            // Scalar tail
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
#endif

#if defined(ESHKOL_HAS_AVX)
// x86 AVX SIMD matmul - processes 4 doubles at a time
static void matmul_simd_avx(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    // Zero the result matrix
    std::memset(C, 0, M * N * sizeof(double));

    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t k = 0; k < K; ++k) {
            // Broadcast A[i][k] to all 4 lanes
            __m256d a_val = _mm256_set1_pd(A[i * K + k]);

            // Process 4 elements at a time
            uint64_t j = 0;
            for (; j + 4 <= N; j += 4) {
                // Load B[k][j:j+4] and C[i][j:j+4]
                __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);

                // C += A[i][k] * B[k][j:j+4] using FMA if available
#if defined(__FMA__)
                c_vec = _mm256_fmadd_pd(a_val, b_vec, c_vec);
#else
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_val, b_vec));
#endif

                // Store result
                _mm256_storeu_pd(&C[i * N + j], c_vec);
            }

            // Scalar tail
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
#elif defined(ESHKOL_HAS_SSE2)
// x86 SSE2 SIMD matmul - processes 2 doubles at a time
static void matmul_simd_sse2(const double* A, const double* B, double* C,
                              uint64_t M, uint64_t K, uint64_t N) {
    // Zero the result matrix
    std::memset(C, 0, M * N * sizeof(double));

    for (uint64_t i = 0; i < M; ++i) {
        for (uint64_t k = 0; k < K; ++k) {
            // Broadcast A[i][k] to both lanes
            __m128d a_val = _mm_set1_pd(A[i * K + k]);

            // Process 2 elements at a time
            uint64_t j = 0;
            for (; j + 2 <= N; j += 2) {
                // Load B[k][j:j+2] and C[i][j:j+2]
                __m128d b_vec = _mm_loadu_pd(&B[k * N + j]);
                __m128d c_vec = _mm_loadu_pd(&C[i * N + j]);

                // C += A[i][k] * B[k][j:j+2]
                c_vec = _mm_add_pd(c_vec, _mm_mul_pd(a_val, b_vec));

                // Store result
                _mm_storeu_pd(&C[i * N + j], c_vec);
            }

            // Scalar tail
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
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
    size_t total_elements = M * N;  // Result matrix size
    size_t total_ops = M * K * N;   // Computational complexity

#ifdef ESHKOL_BLAS_ENABLED
    // Use BLAS for large matrices (threshold based on ops)
    if (eshkol::blas::shouldUseBLAS(total_ops)) {
        eshkol::blas::matmul(A, B, C, M, K, N);
        return;
    }
#endif

    // Use SIMD for medium matrices if available
    if (total_elements >= SIMD_THRESHOLD && simd_available()) {
        matmul_simd(A, B, C, M, K, N);
        return;
    }

    // Scalar fallback for small matrices or no SIMD
    matmul_scalar(A, B, C, M, K, N);
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
