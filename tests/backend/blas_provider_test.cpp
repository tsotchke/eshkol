#include <eshkol/backend/blas_backend.h>
#include <eshkol/build_config.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

bool close(double actual, double expected, double tolerance = 1e-12) {
    return std::abs(actual - expected) <= tolerance;
}

bool expect_vector(const char* name, const double* actual,
                   const double* expected, int count, int stride = 1) {
    for (int i = 0; i < count; ++i) {
        if (!close(actual[i * stride], expected[i])) {
            std::fprintf(stderr, "%s[%d]: got %.17g, expected %.17g\n",
                         name, i, actual[i * stride], expected[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace eshkol::blas;
    bool ok = isAvailable() && std::strlen(getBackendName()) > 0;

#if defined(ESHKOL_BLAS_OPENBLAS) && !defined(__APPLE__) && !defined(_WIN32)
    // CMake target links and raw AOT links are distinct contracts.  The
    // provider test must fail if a Linux BLAS build cannot replay a concrete
    // BLAS closure after libeshkol-runtime.a in generated executables.
    if (std::strlen(ESHKOL_HOST_BLAS_LINK_ARGS) == 0) {
        std::fprintf(stderr,
                     "BLAS provider is enabled but the AOT link closure is empty\n");
        ok = false;
    }
#endif

    const double a[] = {1, 2, 3, 4, 5, 6};
    const double b[] = {7, 8, 9, 10, 11, 12};
    const double expected_product[] = {58, 64, 139, 154};
    double c[] = {1, 1, 1, 1};
    dgemm('N', 'N', 2, 2, 3, 1.0, a, 3, b, 2, 2.0, c, 2);
    const double expected_accumulated[] = {60, 66, 141, 156};
    ok = expect_vector("dgemm beta", c, expected_accumulated, 4) && ok;

    const double a_transposed_storage[] = {1, 4, 2, 5, 3, 6};
    double c_ta[] = {0, 0, 0, 0};
    dgemm('T', 'N', 2, 2, 3, 1.0, a_transposed_storage, 2,
          b, 2, 0.0, c_ta, 2);
    ok = expect_vector("dgemm trans-a", c_ta, expected_product, 4) && ok;

    const double b_transposed_storage[] = {7, 9, 11, 8, 10, 12};
    double c_tb[] = {0, 0, 0, 0};
    dgemm('N', 'T', 2, 2, 3, 1.0, a, 3,
          b_transposed_storage, 3, 0.0, c_tb, 2);
    ok = expect_vector("dgemm trans-b", c_tb, expected_product, 4) && ok;

    const double x[] = {2, 3, 4};
    double y[] = {0, 0};
    const double expected_y[] = {20, 47};
    dgemv('N', 2, 3, 1.0, a, 3, x, 1, 0.0, y, 1);
    ok = expect_vector("dgemv", y, expected_y, 2) && ok;

    const double tx[] = {2, 3};
    double ty[] = {0, 0, 0};
    const double expected_ty[] = {14, 19, 24};
    dgemv('T', 2, 3, 1.0, a, 3, tx, 1, 0.0, ty, 1);
    ok = expect_vector("dgemv transpose", ty, expected_ty, 3) && ok;

    double sx[] = {1, -99, 2, -99, 3};
    double sy[] = {4, -99, 5, -99, 6};
    ok = close(ddot(3, sx, 2, sy, 2), 32.0) && ok;
    dscal(3, 2.0, sx, 2);
    const double expected_sx[] = {2, 4, 6};
    ok = expect_vector("dscal stride", sx, expected_sx, 3, 2) && ok;
    daxpy(3, 0.5, sx, 2, sy, 2);
    const double expected_sy[] = {5, 7, 9};
    ok = expect_vector("daxpy stride", sy, expected_sy, 3, 2) && ok;

    if (!ok) return 1;
    std::printf("BLAS provider (%s): PASS\n", getBackendName());
    return 0;
}
