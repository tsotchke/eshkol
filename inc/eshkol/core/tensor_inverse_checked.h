#ifndef ESHKOL_CORE_TENSOR_INVERSE_CHECKED_H
#define ESHKOL_CORE_TENSOR_INVERSE_CHECKED_H

/** @file tensor_inverse_checked.h
 * @brief Portable checked row-major f64 matrix inversion kernel shared by
 *        native code generation and the VM (including its WASM build).
 *
 * The kernel uses partial pivoting and treats only an exact zero pivot as
 * singular. It reports non-finite inputs/results and size errors; it does not
 * certify an error bound or numerical accuracy for ill-conditioned matrices.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    ESHKOL_TENSOR_INVERSE_OK = 0,
    ESHKOL_TENSOR_INVERSE_INVALID = 1,
    ESHKOL_TENSOR_INVERSE_SINGULAR = 2,
    ESHKOL_TENSOR_INVERSE_NONFINITE = 3,
    ESHKOL_TENSOR_INVERSE_SIZE = 4
};

/* Portable row-major f64 inverse. `work` and `inverse` each hold n*n doubles.
 * Exact zero pivots alone define singularity, preserving scale invariance. */
static inline double eshkol_inverse_load(const void* p, size_t i) {
    double v;
    memcpy(&v, (const unsigned char*)p + i * sizeof(double), sizeof(v));
    return v;
}
static inline void eshkol_inverse_store(void* p, size_t i, double v) {
    memcpy((unsigned char*)p + i * sizeof(double), &v, sizeof(v));
}

static inline int eshkol_tensor_inverse_checked(const void* input,
        double* work, void* inverse, int64_t n) {
    if (!input || !work || !inverse || n <= 0) return ESHKOL_TENSOR_INVERSE_INVALID;
    if ((uint64_t)n > SIZE_MAX / (uint64_t)n / sizeof(double))
        return ESHKOL_TENSOR_INVERSE_SIZE;
    size_t nn = (size_t)n * (size_t)n;
    for (size_t i = 0; i < nn; ++i) {
        double v = eshkol_inverse_load(input, i);
        if (!isfinite(v)) return ESHKOL_TENSOR_INVERSE_NONFINITE;
        work[i] = v; eshkol_inverse_store(inverse, i, 0.0);
    }
    for (int64_t i = 0; i < n; ++i) eshkol_inverse_store(inverse, (size_t)i*n + (size_t)i, 1.0);
    for (int64_t k = 0; k < n; ++k) {
        int64_t pr = k;
        double largest = fabs(work[(size_t)k*n + (size_t)k]);
        for (int64_t r = k + 1; r < n; ++r) {
            double v = fabs(work[(size_t)r*n + (size_t)k]);
            if (v > largest) { largest = v; pr = r; }
        }
        if (largest == 0.0) return ESHKOL_TENSOR_INVERSE_SINGULAR;
        if (pr != k) for (int64_t c = 0; c < n; ++c) {
            size_t x = (size_t)k*n + (size_t)c, y = (size_t)pr*n + (size_t)c;
            double t = work[x]; work[x] = work[y]; work[y] = t;
            t = eshkol_inverse_load(inverse, x);
            eshkol_inverse_store(inverse, x, eshkol_inverse_load(inverse, y));
            eshkol_inverse_store(inverse, y, t);
        }
        double pivot = work[(size_t)k*n + (size_t)k];
        for (int64_t c = 0; c < n; ++c) {
            size_t x = (size_t)k*n + (size_t)c;
            work[x] /= pivot;
            double iv = eshkol_inverse_load(inverse, x) / pivot;
            eshkol_inverse_store(inverse, x, iv);
            if (!isfinite(work[x]) || !isfinite(iv)) return ESHKOL_TENSOR_INVERSE_NONFINITE;
        }
        for (int64_t r = 0; r < n; ++r) if (r != k) {
            double f = work[(size_t)r*n + (size_t)k];
            for (int64_t c = 0; c < n; ++c) {
                size_t x = (size_t)r*n + (size_t)c, y = (size_t)k*n + (size_t)c;
                work[x] -= f * work[y];
                double iv = eshkol_inverse_load(inverse, x) - f * eshkol_inverse_load(inverse, y);
                eshkol_inverse_store(inverse, x, iv);
                if (!isfinite(work[x]) || !isfinite(iv)) return ESHKOL_TENSOR_INVERSE_NONFINITE;
            }
        }
    }
    return ESHKOL_TENSOR_INVERSE_OK;
}

#endif
