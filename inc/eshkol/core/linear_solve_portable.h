/**
 * @file linear_solve_portable.h
 * @brief Portable checked f64 linear solve used by the bytecode VM.
 */
#ifndef ESHKOL_CORE_LINEAR_SOLVE_PORTABLE_H
#define ESHKOL_CORE_LINEAR_SOLVE_PORTABLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ESHKOL_LINEAR_SOLVE_PORTABLE_OK = 0,
    ESHKOL_LINEAR_SOLVE_PORTABLE_INVALID_ARGUMENT = 1,
    ESHKOL_LINEAR_SOLVE_PORTABLE_SIZE_OVERFLOW = 2,
    ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE = 3,
    ESHKOL_LINEAR_SOLVE_PORTABLE_SINGULAR = 4,
    ESHKOL_LINEAR_SOLVE_PORTABLE_OUT_OF_MEMORY = 5
};

/**
 * @brief Solve one row-major square system with partial pivoting.
 *
 * Inputs are copied before elimination and `x` is written only on success.
 * Pivots use exact-zero singularity detection, so nonzero tiny pivots are not
 * rejected by a scale-dependent tolerance.
 *
 * @param matrix Row-major n-by-n finite coefficient matrix.
 * @param rhs Length-n finite right-hand side.
 * @param solution Caller-owned length-n output buffer.
 * @param n Positive system dimension.
 * @return One of the ESHKOL_LINEAR_SOLVE_PORTABLE_* status values.
 */
int eshkol_linear_solve_portable(const double* matrix, const double* rhs,
                                 double* solution, int64_t n);

#ifdef __cplusplus
}
#endif

#endif
