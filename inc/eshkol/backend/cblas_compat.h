/*
 * Internal CBLAS compatibility contract.
 *
 * Accelerate and OpenBLAS provide the standard declarations directly.  Native
 * ClangCL/MSVC builds use Eshkol's pinned Eigen implementation, avoiding any
 * dependency on a MinGW CRT/OpenMP archive while retaining vectorized kernels.
 */
#ifndef ESHKOL_BACKEND_CBLAS_COMPAT_H
#define ESHKOL_BACKEND_CBLAS_COMPAT_H

#if defined(ESHKOL_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(ESHKOL_BLAS_OPENBLAS)
#include <cblas.h>
#elif defined(ESHKOL_BLAS_EIGEN)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;

typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
} CBLAS_TRANSPOSE;

void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
                 CBLAS_TRANSPOSE trans_b, int m, int n, int k,
                 double alpha, const double* a, int lda,
                 const double* b, int ldb, double beta,
                 double* c, int ldc);
void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                 int m, int n, double alpha, const double* a, int lda,
                 const double* x, int incx, double beta,
                 double* y, int incy);
double cblas_ddot(int n, const double* x, int incx,
                  const double* y, int incy);
void cblas_dscal(int n, double alpha, double* x, int incx);
void cblas_daxpy(int n, double alpha, const double* x, int incx,
                 double* y, int incy);

#ifdef __cplusplus
}
#endif

#else
#error "ESHKOL_BLAS_ENABLED requires Accelerate, OpenBLAS, or Eigen kernels"
#endif

#endif
