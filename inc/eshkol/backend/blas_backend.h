/*
 * Cross-Platform BLAS Backend for Eshkol
 *
 * Provides unified access to optimized BLAS operations:
 * - Apple Accelerate on macOS (NEON, AMX optimized)
 * - OpenBLAS on Linux/other platforms
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BLAS_BACKEND_H
#define ESHKOL_BLAS_BACKEND_H

#include <cstddef>
#include <cstdint>

#ifdef ESHKOL_BLAS_ENABLED

#ifdef ESHKOL_BLAS_ACCELERATE
// Apple Accelerate framework
// Use legacy CBLAS API for compatibility (symbols without $NEWLAPACK suffix)
// The deprecation warnings are cosmetic - the API is still fully supported
#include <Accelerate/Accelerate.h>
#elif defined(ESHKOL_BLAS_OPENBLAS)
// OpenBLAS
#include <cblas.h>
#else
#error "ESHKOL_BLAS_ENABLED is set but no BLAS backend defined. Define ESHKOL_BLAS_ACCELERATE (macOS) or ESHKOL_BLAS_OPENBLAS (Linux)."
#endif

namespace eshkol {
namespace blas {

/**
 * Runtime-configurable threshold for BLAS vs SIMD dispatch.
 * Operations below this element count use SIMD, above use BLAS.
 * Default: 4096 (64x64 matrix)
 *
 * Can be overridden via ESHKOL_BLAS_THRESHOLD environment variable.
 */
extern size_t g_blas_threshold;

/**
 * Set BLAS threshold at runtime.
 * @param threshold Minimum elements to use BLAS
 */
void blas_set_threshold(size_t threshold);

/**
 * Get current BLAS threshold.
 * @return Current threshold
 */
size_t blas_get_threshold();

/**
 * Check if BLAS should be used for an operation.
 * @param num_elements Total elements in operation
 * @return true if BLAS should be used
 */
inline bool shouldUseBLAS(size_t num_elements) {
    return num_elements >= g_blas_threshold;
}

/**
 * Check if BLAS backend is available.
 * @return true if BLAS is compiled in and functional
 */
bool isAvailable();

/**
 * Get BLAS backend description.
 * @return Human-readable backend name
 */
const char* getBackendName();

// ===== Matrix Operations =====

/**
 * General matrix multiply: C = alpha * A * B + beta * C
 *
 * Computes C = alpha * op(A) * op(B) + beta * C where op(X) is X or X^T.
 *
 * @param transA 'N' for A, 'T' for A^T
 * @param transB 'N' for B, 'T' for B^T
 * @param M Rows of op(A) and C
 * @param N Columns of op(B) and C
 * @param K Columns of op(A) / Rows of op(B)
 * @param alpha Scalar multiplier for A*B
 * @param A Matrix A (M x K if transA='N', K x M if transA='T')
 * @param lda Leading dimension of A
 * @param B Matrix B (K x N if transB='N', N x K if transB='T')
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Result matrix (M x N)
 * @param ldc Leading dimension of C
 */
void dgemm(char transA, char transB,
           int M, int N, int K,
           double alpha,
           const double* A, int lda,
           const double* B, int ldb,
           double beta,
           double* C, int ldc);

/**
 * Simple matrix multiply: C = A * B
 *
 * Convenience wrapper for common case (no transpose, alpha=1, beta=0).
 *
 * @param A Matrix A (M x K), row-major
 * @param B Matrix B (K x N), row-major
 * @param C Result matrix (M x N), row-major
 * @param M Rows of A and C
 * @param K Columns of A / Rows of B
 * @param N Columns of B and C
 */
void matmul(const double* A, const double* B, double* C,
            size_t M, size_t K, size_t N);

/**
 * Matmul backward: given C = A @ B, accumulate gradients
 * dA += grad_C @ B^T, dB += A^T @ grad_C
 */
void matmul_backward(const double* grad_c, const double* a, const double* b,
                     double* grad_a, double* grad_b,
                     size_t M, size_t K, size_t N);

/**
 * Matrix-vector multiply: y = alpha * A * x + beta * y
 */
void dgemv(char trans, int M, int N,
           double alpha, const double* A, int lda,
           const double* x, int incx,
           double beta, double* y, int incy);

// ===== Vector Operations =====

/**
 * Vector dot product: result = x · y
 * @param n Vector length
 * @param x First vector
 * @param incx Stride for x
 * @param y Second vector
 * @param incy Stride for y
 * @return Dot product
 */
double ddot(int n, const double* x, int incx, const double* y, int incy);

/**
 * Vector scale: x = alpha * x
 * @param n Vector length
 * @param alpha Scale factor
 * @param x Vector to scale (in-place)
 * @param incx Stride for x
 */
void dscal(int n, double alpha, double* x, int incx);

/**
 * Vector add: y = alpha * x + y
 * @param n Vector length
 * @param alpha Scale factor for x
 * @param x First vector
 * @param incx Stride for x
 * @param y Second vector (in-place result)
 * @param incy Stride for y
 */
void daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);

} // namespace blas
} // namespace eshkol

#else // !ESHKOL_BLAS_ENABLED

// Stub declarations when BLAS is not available
namespace eshkol {
namespace blas {

inline bool isAvailable() { return false; }
inline const char* getBackendName() { return "none"; }
inline bool shouldUseBLAS(size_t) { return false; }

} // namespace blas
} // namespace eshkol

#endif // ESHKOL_BLAS_ENABLED

// ===== C-compatible Runtime Functions =====
// These can be called from JIT-compiled and AOT-compiled code

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Runtime matrix multiplication with automatic BLAS dispatch.
 * Called from generated LLVM IR for tensor-dot operations.
 * Uses BLAS for large matrices, scalar fallback for small ones.
 *
 * @param A Matrix A data (row-major, M x K)
 * @param B Matrix B data (row-major, K x N)
 * @param C Output matrix data (row-major, M x N)
 * @param M Rows of A and C
 * @param K Columns of A / Rows of B
 * @param N Columns of B and C
 */
void eshkol_matmul_f64(const double* A, const double* B, double* C,
                        uint64_t M, uint64_t K, uint64_t N);

/**
 * Check if BLAS is available at runtime.
 * @return 1 if BLAS available, 0 otherwise
 */
int eshkol_blas_available(void);

/**
 * Get BLAS threshold.
 * @return Current threshold for BLAS dispatch
 */
uint64_t eshkol_blas_get_threshold(void);

/**
 * Set BLAS threshold.
 * @param threshold New threshold
 */
void eshkol_blas_set_threshold(uint64_t threshold);

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_BLAS_BACKEND_H
