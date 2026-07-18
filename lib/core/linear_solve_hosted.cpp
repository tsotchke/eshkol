/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted wrapper for dense linear solve ABI and error handling.
 */

#include <eshkol/core/linear_solve.h>
#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>

#include <cstdlib>
#include <cstdint>

extern "C" {

// eshkol_runtime_fatal is defined in runtime_errors_hosted.cpp and constructs
// the catchable runtime exception on fatal paths.
void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/**
 * @brief Inspect the hosted environment for the linear-solve control flag.
 *
 * Existing semantics are preserved:
 * - unset: false
 * - empty string: false
 * - leading '0': false
 * - any other non-empty string: true
 */
uint32_t eshkol_linear_solve_query_options(void) {
    const char* force = std::getenv("ESHKOL_LINSOLVE_FORCE_DGESV");
    if (!force || force[0] == '\0' || force[0] == '0') return 0;
    return ESHKOL_LINSOLVE_FORCE_DGESV;
}

/**
 * @brief Hosted ABI entry point that retains the historical symbol shape.
 */
int64_t eshkol_linear_solve(
    int64_t a_ndim, const int64_t* a_dims,
    int64_t b_ndim, const int64_t* b_dims,
    const double* A, const double* b, double* x) {
    return eshkol_linear_solve_with_options(
        a_ndim, a_dims, b_ndim, b_dims, A, b, x,
        eshkol_linear_solve_query_options());
}

/**
 * @brief Raise the catchable exception for a nonzero eshkol_linear_solve()
 *        status.
 *
 * The VM path for native execution handles this by catching the runtime
 * exception. This helper keeps the core independent of hosted runtime internals.
 */
void eshkol_linear_solve_raise(int64_t status) {
    const char* msg;
    switch (status) {
        case ESHKOL_LINSOLVE_SINGULAR:
            msg = "linear-solve: matrix is singular"; break;
        case ESHKOL_LINSOLVE_NOT_SQUARE:
            msg = "linear-solve: A must be a square 2-D matrix"; break;
        case ESHKOL_LINSOLVE_DIM_MISMATCH:
            msg = "linear-solve: dimension mismatch (b length must equal N)"; break;
        default:
            msg = "linear-solve: error"; break;
    }
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s", msg);
}

}  // extern "C"
