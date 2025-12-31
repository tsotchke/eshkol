/*
 * Cross-Platform BLAS Backend Implementation for Eshkol
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/blas_backend.h"
#include <cstdlib>
#include <cstring>

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

// ===== C-compatible Runtime Functions =====

extern "C" {

void eshkol_matmul_f64(const double* A, const double* B, double* C,
                        uint64_t M, uint64_t K, uint64_t N) {
    size_t total_ops = M * K * N;

#ifdef ESHKOL_BLAS_ENABLED
    // Use BLAS for large matrices
    if (eshkol::blas::shouldUseBLAS(total_ops)) {
        eshkol::blas::matmul(A, B, C, M, K, N);
        return;
    }
#endif

    // Scalar fallback for small matrices or when BLAS unavailable
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
