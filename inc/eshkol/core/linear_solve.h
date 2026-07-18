/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_CORE_LINEAR_SOLVE_H
#define ESHKOL_CORE_LINEAR_SOLVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// =====================================================================
// Solver status codes shared by core and hosted wrappers.
// =====================================================================

#define ESHKOL_LINSOLVE_OK            0
#define ESHKOL_LINSOLVE_SINGULAR      1
#define ESHKOL_LINSOLVE_NOT_SQUARE    2
#define ESHKOL_LINSOLVE_DIM_MISMATCH  3

// =====================================================================
// Solver options (bitmask) for deterministic core execution.
// =====================================================================

// If set, skip the mixed-precision IR path and force a direct LAPACK dgesv
// solve where that path exists (Apple/Accelerate).
#define ESHKOL_LINSOLVE_FORCE_DGESV ((uint32_t)1u)

// =====================================================================
// Core / hosted entry points.
// =====================================================================

/**
 * @brief Solve A x = b in the numerical core.
 *
 * The core entry is deterministic and does not inspect host process
 * environment or call hosted-only runtime services. Callers must pass all
 * behavior controls explicitly in `options`.
 *
 * @param a_ndim Rank of A (must be 2).
 * @param a_dims Shape of A (row-major); a_dims[0] x a_dims[1].
 * @param b_ndim Rank of b.
 * @param b_dims Shape of b (product must equal N).
 * @param A      Row-major N*N f64 coefficient matrix.
 * @param b      Length-N f64 right-hand side.
 * @param x      Length-N f64 output solution (caller-allocated).
 * @param options Solver controls bitmask, e.g. ESHKOL_LINSOLVE_FORCE_DGESV.
 * @return       ESHKOL_LINSOLVE_OK or a nonzero catchable error code.
 */
int64_t eshkol_linear_solve_with_options(
    int64_t a_ndim, const int64_t* a_dims,
    int64_t b_ndim, const int64_t* b_dims,
    const double* A, const double* b, double* x,
    uint32_t options);

/**
 * @brief Hosted ABI entry point used by VM/JIT/AOT.
 *
 * Preserves the existing public ABI by deriving `options` from
 * `ESHKOL_LINSOLVE_FORCE_DGESV` in the host process environment.
 */
int64_t eshkol_linear_solve(
    int64_t a_ndim, const int64_t* a_dims,
    int64_t b_ndim, const int64_t* b_dims,
    const double* A, const double* b, double* x);

/**
 * @brief Raise a catchable exception for a nonzero solver status.
 */
void eshkol_linear_solve_raise(int64_t status);

/**
 * @brief Query solver options for the hosted process environment.
 */
uint32_t eshkol_linear_solve_query_options(void);

#ifdef __cplusplus
}
#endif

#endif  // ESHKOL_CORE_LINEAR_SOLVE_H
