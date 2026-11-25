/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Portable checked f64 solve for the bytecode VM. Kept in C so the standalone
 * VM and WASM source lists need no hosted C++ solver or BLAS dependency.
 */
#include <eshkol/core/linear_solve_portable.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int eshkol_linear_solve_portable(const double* matrix, const double* rhs,
                                 double* solution, int64_t n) {
    if (!matrix || !rhs || !solution || n <= 0)
        return ESHKOL_LINEAR_SOLVE_PORTABLE_INVALID_ARGUMENT;

    const uint64_t un = (uint64_t)n;
    if (un > SIZE_MAX / un || un * un > SIZE_MAX / sizeof(double))
        return ESHKOL_LINEAR_SOLVE_PORTABLE_SIZE_OVERFLOW;
    const size_t count = (size_t)(un * un);
    const size_t matrix_bytes = count * sizeof(double);
    if (un > SIZE_MAX / sizeof(double))
        return ESHKOL_LINEAR_SOLVE_PORTABLE_SIZE_OVERFLOW;
    const size_t rhs_bytes = (size_t)un * sizeof(double);
    if (matrix_bytes > SIZE_MAX - rhs_bytes)
        return ESHKOL_LINEAR_SOLVE_PORTABLE_SIZE_OVERFLOW;

    const double* a_end = matrix + count;
    for (const double* p = matrix; p != a_end; ++p)
        if (!isfinite(*p)) return ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE;
    for (int64_t i = 0; i < n; ++i)
        if (!isfinite(rhs[i])) return ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE;

    double* storage = (double*)malloc(matrix_bytes + rhs_bytes);
    if (!storage) return ESHKOL_LINEAR_SOLVE_PORTABLE_OUT_OF_MEMORY;
    double* lu = storage;
    double* y = (double*)((unsigned char*)storage + matrix_bytes);
    memcpy(lu, matrix, matrix_bytes);
    memcpy(y, rhs, rhs_bytes);

    int status = ESHKOL_LINEAR_SOLVE_PORTABLE_OK;
    for (int64_t k = 0; k < n; ++k) {
        int64_t pivot_row = k;
        double pivot_abs = fabs(lu[(size_t)k * (size_t)n + (size_t)k]);
        for (int64_t i = k + 1; i < n; ++i) {
            const double candidate = fabs(lu[(size_t)i * (size_t)n + (size_t)k]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = i;
            }
        }
        if (pivot_abs == 0.0) {
            status = ESHKOL_LINEAR_SOLVE_PORTABLE_SINGULAR;
            break;
        }
        if (pivot_row != k) {
            for (int64_t j = 0; j < n; ++j) {
                const size_t ki = (size_t)k * (size_t)n + (size_t)j;
                const size_t pi = (size_t)pivot_row * (size_t)n + (size_t)j;
                const double tmp = lu[ki]; lu[ki] = lu[pi]; lu[pi] = tmp;
            }
            const double tmp = y[k]; y[k] = y[pivot_row]; y[pivot_row] = tmp;
        }

        const double pivot = lu[(size_t)k * (size_t)n + (size_t)k];
        for (int64_t i = k + 1; i < n; ++i) {
            const size_t ik = (size_t)i * (size_t)n + (size_t)k;
            const double factor = lu[ik] / pivot;
            if (!isfinite(factor)) { status = ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE; break; }
            lu[ik] = factor;
            for (int64_t j = k + 1; j < n; ++j) {
                const size_t ij = (size_t)i * (size_t)n + (size_t)j;
                const size_t kj = (size_t)k * (size_t)n + (size_t)j;
                lu[ij] -= factor * lu[kj];
                if (!isfinite(lu[ij])) { status = ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE; break; }
            }
            if (status != ESHKOL_LINEAR_SOLVE_PORTABLE_OK) break;
            y[i] -= factor * y[k];
            if (!isfinite(y[i])) { status = ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE; break; }
        }
        if (status != ESHKOL_LINEAR_SOLVE_PORTABLE_OK) break;
    }

    if (status == ESHKOL_LINEAR_SOLVE_PORTABLE_OK) {
        for (int64_t i = n; i-- > 0;) {
            double sum = y[i];
            for (int64_t j = i + 1; j < n; ++j)
                sum -= lu[(size_t)i * (size_t)n + (size_t)j] * y[j];
            y[i] = sum / lu[(size_t)i * (size_t)n + (size_t)i];
            if (!isfinite(y[i])) {
                status = ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE;
                break;
            }
        }
    }
    if (status == ESHKOL_LINEAR_SOLVE_PORTABLE_OK)
        memcpy(solution, y, rhs_bytes);
    free(storage);
    return status;
}
