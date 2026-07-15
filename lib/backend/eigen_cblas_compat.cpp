/*
 * Native Windows CBLAS subset backed by pinned Eigen kernels.
 *
 * Eshkol's tensor runtime uses only real-valued GEMM/GEMV and BLAS level-1
 * primitives.  Keeping that contract here lets the higher layers share their
 * tested CBLAS path across platforms while ClangCL/MSVC stays entirely within
 * the native Windows ABI.
 */
#include <eshkol/backend/cblas_compat.h>

#if defined(ESHKOL_BLAS_EIGEN)

#include <Eigen/Core>

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void invalid_argument(const char* operation, const char* reason) {
    std::fprintf(stderr, "Eshkol Eigen BLAS %s: invalid argument (%s)\n",
                 operation, reason);
    std::abort();
}

bool is_transposed(CBLAS_TRANSPOSE trans) {
    if (trans == CblasNoTrans) return false;
    if (trans == CblasTrans || trans == CblasConjTrans) return true;
    invalid_argument("dispatch", "unsupported transpose mode");
}

template <int StorageOrder>
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

template <int StorageOrder>
using ConstMatrixMap = Eigen::Map<const Matrix<StorageOrder>, Eigen::Unaligned,
                                  Eigen::OuterStride<Eigen::Dynamic>>;

template <int StorageOrder>
using MatrixMap = Eigen::Map<Matrix<StorageOrder>, Eigen::Unaligned,
                             Eigen::OuterStride<Eigen::Dynamic>>;

template <int StorageOrder>
void dgemm_impl(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
                int m, int n, int k, double alpha,
                const double* a, int lda, const double* b, int ldb,
                double beta, double* c, int ldc) {
    const bool ta = is_transposed(trans_a);
    const bool tb = is_transposed(trans_b);
    const int a_rows = ta ? k : m;
    const int a_cols = ta ? m : k;
    const int b_rows = tb ? n : k;
    const int b_cols = tb ? k : n;

    ConstMatrixMap<StorageOrder> a_map(
        a, a_rows, a_cols, Eigen::OuterStride<Eigen::Dynamic>(lda));
    ConstMatrixMap<StorageOrder> b_map(
        b, b_rows, b_cols, Eigen::OuterStride<Eigen::Dynamic>(ldb));
    MatrixMap<StorageOrder> c_map(
        c, m, n, Eigen::OuterStride<Eigen::Dynamic>(ldc));

    if (beta == 0.0) {
        c_map.setZero();
    } else if (beta != 1.0) {
        c_map *= beta;
    }
    if (alpha == 0.0 || k == 0) return;

    if (!ta && !tb) {
        c_map.noalias() += alpha * (a_map * b_map);
    } else if (ta && !tb) {
        c_map.noalias() += alpha * (a_map.transpose() * b_map);
    } else if (!ta && tb) {
        c_map.noalias() += alpha * (a_map * b_map.transpose());
    } else {
        c_map.noalias() += alpha * (a_map.transpose() * b_map.transpose());
    }
}

template <int StorageOrder>
void dgemv_impl(CBLAS_TRANSPOSE trans, int m, int n, double alpha,
                const double* a, int lda, const double* x, int incx,
                double beta, double* y, int incy) {
    const bool transpose = is_transposed(trans);
    ConstMatrixMap<StorageOrder> a_map(
        a, m, n, Eigen::OuterStride<Eigen::Dynamic>(lda));
    const int x_size = transpose ? m : n;
    const int y_size = transpose ? n : m;
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned,
                                      Eigen::InnerStride<Eigen::Dynamic>>;
    using VectorMap = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned,
                                 Eigen::InnerStride<Eigen::Dynamic>>;
    ConstVectorMap x_map(x, x_size, Eigen::InnerStride<Eigen::Dynamic>(incx));
    VectorMap y_map(y, y_size, Eigen::InnerStride<Eigen::Dynamic>(incy));

    if (beta == 0.0) {
        y_map.setZero();
    } else if (beta != 1.0) {
        y_map *= beta;
    }
    if (alpha == 0.0 || m == 0 || n == 0) return;
    if (transpose) {
        y_map.noalias() += alpha * (a_map.transpose() * x_map);
    } else {
        y_map.noalias() += alpha * (a_map * x_map);
    }
}

void validate_common(const char* operation, int m, int n,
                     const double* a, const double* b, double* c) {
    if (m < 0 || n < 0) invalid_argument(operation, "negative dimension");
    if ((m > 0 && n > 0) && (!a || !b || !c)) {
        invalid_argument(operation, "null data pointer");
    }
}

}  // namespace

extern "C" void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
                             CBLAS_TRANSPOSE trans_b, int m, int n, int k,
                             double alpha, const double* a, int lda,
                             const double* b, int ldb, double beta,
                             double* c, int ldc) {
    validate_common("dgemm", m, n, a, b, c);
    if (k < 0) invalid_argument("dgemm", "negative reduction dimension");
    if (m == 0 || n == 0) return;
    if (lda <= 0 || ldb <= 0 || ldc <= 0) {
        invalid_argument("dgemm", "non-positive leading dimension");
    }
    if (order == CblasRowMajor) {
        dgemm_impl<Eigen::RowMajor>(trans_a, trans_b, m, n, k, alpha,
                                    a, lda, b, ldb, beta, c, ldc);
    } else if (order == CblasColMajor) {
        dgemm_impl<Eigen::ColMajor>(trans_a, trans_b, m, n, k, alpha,
                                    a, lda, b, ldb, beta, c, ldc);
    } else {
        invalid_argument("dgemm", "unsupported storage order");
    }
}

extern "C" void cblas_dgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
                             int m, int n, double alpha, const double* a,
                             int lda, const double* x, int incx, double beta,
                             double* y, int incy) {
    validate_common("dgemv", m, n, a, x, y);
    if (m == 0 || n == 0) return;
    if (lda <= 0 || incx == 0 || incy == 0) {
        invalid_argument("dgemv", "invalid stride");
    }
    if (order == CblasRowMajor) {
        dgemv_impl<Eigen::RowMajor>(trans, m, n, alpha, a, lda,
                                    x, incx, beta, y, incy);
    } else if (order == CblasColMajor) {
        dgemv_impl<Eigen::ColMajor>(trans, m, n, alpha, a, lda,
                                    x, incx, beta, y, incy);
    } else {
        invalid_argument("dgemv", "unsupported storage order");
    }
}

extern "C" double cblas_ddot(int n, const double* x, int incx,
                              const double* y, int incy) {
    if (n <= 0) return 0.0;
    if (!x || !y || incx == 0 || incy == 0) {
        invalid_argument("ddot", "invalid vector or stride");
    }
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned,
                                      Eigen::InnerStride<Eigen::Dynamic>>;
    ConstVectorMap x_map(x, n, Eigen::InnerStride<Eigen::Dynamic>(incx));
    ConstVectorMap y_map(y, n, Eigen::InnerStride<Eigen::Dynamic>(incy));
    return x_map.dot(y_map);
}

extern "C" void cblas_dscal(int n, double alpha, double* x, int incx) {
    if (n <= 0) return;
    if (!x || incx == 0) invalid_argument("dscal", "invalid vector or stride");
    using VectorMap = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned,
                                 Eigen::InnerStride<Eigen::Dynamic>>;
    VectorMap x_map(x, n, Eigen::InnerStride<Eigen::Dynamic>(incx));
    x_map *= alpha;
}

extern "C" void cblas_daxpy(int n, double alpha, const double* x, int incx,
                             double* y, int incy) {
    if (n <= 0) return;
    if (!x || !y || incx == 0 || incy == 0) {
        invalid_argument("daxpy", "invalid vector or stride");
    }
    using ConstVectorMap = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned,
                                      Eigen::InnerStride<Eigen::Dynamic>>;
    using VectorMap = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned,
                                 Eigen::InnerStride<Eigen::Dynamic>>;
    ConstVectorMap x_map(x, n, Eigen::InnerStride<Eigen::Dynamic>(incx));
    VectorMap y_map(y, n, Eigen::InnerStride<Eigen::Dynamic>(incy));
    y_map += alpha * x_map;
}

#endif
