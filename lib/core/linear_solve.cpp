/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Dense linear solver Ax = b with a full-f64 accuracy guarantee.
 *
 * On Apple (Accelerate) the O(n^3) factorization is done in fp32 and the
 * solution is polished back to full f64 by mixed-precision iterative
 * refinement (Langou/Baboulin/Dongarra): the residual r = b - A*x is
 * recomputed in f64 (an O(n^2) GEMV), a fp32 back-solve corrects x, and the
 * loop repeats until the relative backward residual is certified below
 * ACCEPT_TOL. Because the fp32 factorization runs ~2-3x faster than an f64
 * one on the AMX units while the refinement tail is asymptotically free, a
 * well-conditioned solve reaches the *same* f64 residual as a plain LAPACK
 * dgesv at a fraction of the cost.
 *
 * The speedup is opportunistic; correctness is not. When refinement cannot
 * certify a full-f64 result within the iteration cap (an ill-conditioned
 * system, where kappa * eps_fp32 >~ 1), the solver silently falls back to a
 * plain-f64 LAPACK dgesv and returns that answer. Either way the caller gets
 * a genuine full-precision f64 solution.
 *
 * Non-Apple builds have no guaranteed fast fp32 LAPACK path, so they use a
 * self-contained f64 LU with partial pivoting directly: correct everywhere,
 * with the IR speedup Apple-first for now.
 *
 * Called from LLVM-generated native code and from the bytecode VM through the
 * extern "C" name below. Inputs are raw row-major double buffers (the native
 * tensor stores f64 bit patterns in an int64 array; the VM stores doubles —
 * both alias a `const double*` here).
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/linear_solve.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

// Refinement acceptance: certify the IR solution only once its relative
// backward residual ||b - Ax|| / ||b|| is at or below this. Well-conditioned
// systems reach ~1e-15; anything that cannot get here falls back to dgesv.
static const double ESHKOL_LINSOLVE_ACCEPT_TOL = 1e-13;
static const int    ESHKOL_LINSOLVE_MAX_ITERS  = 30;

#if defined(ESHKOL_BLAS_ACCELERATE)

// Fortran LAPACK symbols exported by Accelerate (column-major, LP64 => the
// LAPACK integer is a 32-bit int). Declared locally so this translation unit
// does not depend on the exact shape of Accelerate's (legacy vs new) LAPACK
// headers; the linker resolves them against the Accelerate framework.
extern "C" {
void sgetrf_(const int* m, const int* n, float* a, const int* lda,
             int* ipiv, int* info);
void sgetrs_(const char* trans, const int* n, const int* nrhs,
             const float* a, const int* lda, const int* ipiv,
             float* b, const int* ldb, int* info);
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
            int* ipiv, double* b, const int* ldb, int* info);

// CBLAS entry points (enums are int-sized, so plain int is ABI-compatible).
double cblas_dnrm2(int n, const double* x, int incx);
void cblas_dgemv(int order, int trans, int m, int n, double alpha,
                 const double* a, int lda, const double* x, int incx,
                 double beta, double* y, int incy);
}

// CBLAS constants (avoid pulling the header for two enum values).
#define ESHKOL_CBLAS_ROW_MAJOR 101
#define ESHKOL_CBLAS_NO_TRANS  111

// Plain-f64 fallback: solve A x = b via LAPACK dgesv on a column-major copy.
// Returns ESHKOL_LINSOLVE_OK or ESHKOL_LINSOLVE_SINGULAR.
static int64_t linsolve_dgesv_fallback(const double* A, const double* b,
                                       double* x, int n) {
    double* Ac = (double*)std::malloc((size_t)n * (size_t)n * sizeof(double));
    double* xb = (double*)std::malloc((size_t)n * sizeof(double));
    int*    ip = (int*)std::malloc((size_t)n * sizeof(int));
    if (!Ac || !xb || !ip) {
        std::free(Ac); std::free(xb); std::free(ip);
        return ESHKOL_LINSOLVE_SINGULAR;  // treat OOM as unsolvable here
    }
    // Row-major A -> column-major Ac (transpose of the buffer layout).
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Ac[i + (size_t)j * n] = A[(size_t)i * n + j];
    std::memcpy(xb, b, (size_t)n * sizeof(double));

    int nn = n, nrhs = 1, info = 0;
    dgesv_(&nn, &nrhs, Ac, &nn, ip, xb, &nn, &info);

    int64_t status = (info == 0) ? ESHKOL_LINSOLVE_OK : ESHKOL_LINSOLVE_SINGULAR;
    if (status == ESHKOL_LINSOLVE_OK)
        std::memcpy(x, xb, (size_t)n * sizeof(double));
    std::free(Ac); std::free(xb); std::free(ip);
    return status;
}

// Mixed-precision iterative refinement, Apple/Accelerate path.
static int64_t linsolve_ir_accelerate(const double* A, const double* b,
                                      double* x, int n, uint32_t options) {
    if ((options & ESHKOL_LINSOLVE_FORCE_DGESV) != 0) {
        return linsolve_dgesv_fallback(A, b, x, n);
    }

    if (n == 0) return ESHKOL_LINSOLVE_OK;
    
    double bnorm = cblas_dnrm2(n, b, 1);
    double denom = (bnorm > 0.0) ? bnorm : 1.0;

    float* As = (float*)std::malloc((size_t)n * (size_t)n * sizeof(float));
    float* xf = (float*)std::malloc((size_t)n * sizeof(float));
    double* r = (double*)std::malloc((size_t)n * sizeof(double));
    int*   ip = (int*)std::malloc((size_t)n * sizeof(int));
    if (!As || !xf || !r || !ip) {
        std::free(As); std::free(xf); std::free(r); std::free(ip);
        return linsolve_dgesv_fallback(A, b, x, n);
    }

    // Row-major f64 A -> column-major fp32 As (so LAPACK factors A, not A^T).
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            As[i + (size_t)j * n] = (float)A[(size_t)i * n + j];

    int nn = n, nrhs = 1, info = 0;
    sgetrf_(&nn, &nn, As, &nn, ip, &info);
    if (info != 0) {
        // fp32 factorization hit a zero pivot; let dgesv decide singularity.
        std::free(As); std::free(xf); std::free(r); std::free(ip);
        return linsolve_dgesv_fallback(A, b, x, n);
    }

    // Initial fp32 solve: x0 = A^{-1} b.
    for (int i = 0; i < n; i++) xf[i] = (float)b[i];
    sgetrs_("N", &nn, &nrhs, As, &nn, ip, xf, &nn, &info);
    for (int i = 0; i < n; i++) x[i] = (double)xf[i];

    double prev = 1e300;
    for (int it = 0; it < ESHKOL_LINSOLVE_MAX_ITERS; it++) {
        // r = b - A*x in full f64 (O(n^2)); A is row-major.
        std::memcpy(r, b, (size_t)n * sizeof(double));
        cblas_dgemv(ESHKOL_CBLAS_ROW_MAJOR, ESHKOL_CBLAS_NO_TRANS, n, n,
                    -1.0, A, n, x, 1, 1.0, r, 1);
        double relres = cblas_dnrm2(n, r, 1) / denom;

        if (relres <= ESHKOL_LINSOLVE_ACCEPT_TOL) {
            std::free(As); std::free(xf); std::free(r); std::free(ip);
            return ESHKOL_LINSOLVE_OK;  // certified full-f64 IR solution
        }
        // Not improving by at least 10% => stalled/diverging; stop refining
        // and hand off to the guaranteed f64 path.
        if (it > 0 && relres > 0.9 * prev) break;
        prev = relres;

        // fp32 correction d = A^{-1} r; x += d.
        for (int i = 0; i < n; i++) xf[i] = (float)r[i];
        sgetrs_("N", &nn, &nrhs, As, &nn, ip, xf, &nn, &info);
        for (int i = 0; i < n; i++) x[i] += (double)xf[i];
    }

    std::free(As); std::free(xf); std::free(r); std::free(ip);
    // IR did not certify full f64 within the cap -> plain-f64 LAPACK solve.
    return linsolve_dgesv_fallback(A, b, x, n);
}

#else  // !ESHKOL_BLAS_ACCELERATE

// Self-contained f64 LU with partial pivoting (row-major, in place on a copy).
// Correct on every platform; used where no fast fp32 LAPACK path is present.
static int64_t linsolve_lu_f64(const double* A, const double* b,
                               double* x, int n) {
    if (n == 0) return ESHKOL_LINSOLVE_OK;
    double* M = (double*)std::malloc((size_t)n * (size_t)n * sizeof(double));
    int*    piv = (int*)std::malloc((size_t)n * sizeof(int));
    if (!M || !piv) { std::free(M); std::free(piv); return ESHKOL_LINSOLVE_SINGULAR; }
    std::memcpy(M, A, (size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) { piv[i] = i; x[i] = b[i]; }

    for (int k = 0; k < n; k++) {
        double maxv = 0.0; int maxr = k;
        for (int i = k; i < n; i++) {
            double v = std::fabs(M[(size_t)i * n + k]);
            if (v > maxv) { maxv = v; maxr = i; }
        }
        if (maxv == 0.0) { std::free(M); std::free(piv); return ESHKOL_LINSOLVE_SINGULAR; }
        if (maxr != k) {
            for (int j = 0; j < n; j++) {
                double t = M[(size_t)k * n + j];
                M[(size_t)k * n + j] = M[(size_t)maxr * n + j];
                M[(size_t)maxr * n + j] = t;
            }
            double tb = x[k]; x[k] = x[maxr]; x[maxr] = tb;
        }
        double pivot = M[(size_t)k * n + k];
        for (int i = k + 1; i < n; i++) {
            double f = M[(size_t)i * n + k] / pivot;
            M[(size_t)i * n + k] = f;
            for (int j = k + 1; j < n; j++)
                M[(size_t)i * n + j] -= f * M[(size_t)k * n + j];
            x[i] -= f * x[k];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = x[i];
        for (int j = i + 1; j < n; j++) s -= M[(size_t)i * n + j] * x[j];
        x[i] = s / M[(size_t)i * n + i];
    }
    std::free(M); std::free(piv);
    return ESHKOL_LINSOLVE_OK;
}

#endif  // ESHKOL_BLAS_ACCELERATE

/**
 * @brief Solve the dense linear system A x = b with a full-f64 guarantee.
 *
 * Validates shapes, then solves. On Apple this uses mixed-precision iterative
 * refinement with a plain-f64 dgesv fallback; elsewhere a direct f64 LU. The
 * returned solution is always full-precision f64 when the status is OK.
 *
 * @param a_ndim Rank of A (must be 2).
 * @param a_dims Shape of A (row-major); a_dims[0] x a_dims[1].
 * @param b_ndim Rank of b.
 * @param b_dims Shape of b (its element product must equal N).
 * @param A      Row-major N*N f64 coefficient buffer (not modified).
 * @param b      Length-N f64 right-hand side (not modified).
 * @param x      Length-N f64 output solution (caller-allocated).
 * @return       ESHKOL_LINSOLVE_OK, or a nonzero catchable error code.
 */
extern "C" int64_t eshkol_linear_solve_with_options(
    int64_t a_ndim, const int64_t* a_dims,
    int64_t b_ndim, const int64_t* b_dims,
    const double* A, const double* b, double* x, uint32_t options) {

    if (a_ndim != 2 || !a_dims || a_dims[0] != a_dims[1] || a_dims[0] < 0)
        return ESHKOL_LINSOLVE_NOT_SQUARE;
    int64_t n = a_dims[0];

    int64_t b_total = 1;
    if (b_ndim <= 0 || !b_dims) {
        b_total = 0;
    } else {
        for (int64_t d = 0; d < b_ndim; d++) b_total *= b_dims[d];
    }
    if (b_total != n) return ESHKOL_LINSOLVE_DIM_MISMATCH;

    if (n == 0) return ESHKOL_LINSOLVE_OK;

#if defined(ESHKOL_BLAS_ACCELERATE)
    return linsolve_ir_accelerate(A, b, x, (int)n, options);
#else
    (void)options;
    return linsolve_lu_f64(A, b, x, (int)n);
#endif
}
