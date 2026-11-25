/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#include <eshkol/core/linear_solve_portable.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(const char* message) {
    fprintf(stderr, "linear_solve_portable_test: %s\n", message);
    return 1;
}

int main(void) {
    const double A[9] = {
        0.0, 2.0, 1.0,
        1.0, -2.0, -3.0,
        4.0, -7.0, 1.0
    };
    const double b[3] = {3.0, 0.0, -11.0};
    const double expected[3] = {1.0, 2.0, -1.0};
    double solution[3] = {91.0, 92.0, 93.0};
    double a_copy[9], b_copy[3];
    memcpy(a_copy, A, sizeof A);
    memcpy(b_copy, b, sizeof b);
    if (eshkol_linear_solve_portable(A, b, solution, 3) != ESHKOL_LINEAR_SOLVE_PORTABLE_OK)
        return fail("pivoted nonsymmetric solve failed");
    for (int i = 0; i < 3; ++i)
        if (fabs(solution[i] - expected[i]) > 1e-12)
            return fail("pivoted solve returned the wrong solution");
    if (memcmp(A, a_copy, sizeof A) || memcmp(b, b_copy, sizeof b))
        return fail("solver modified an input");

    const double tiny_A[4] = {0.0, 2e-200, 1e-200, 0.0};
    const double tiny_b[2] = {-8e-200, 3e-200};
    double tiny_x[2] = {0.0, 0.0};
    if (eshkol_linear_solve_portable(tiny_A, tiny_b, tiny_x, 2) != ESHKOL_LINEAR_SOLVE_PORTABLE_OK ||
        fabs(tiny_x[0] - 3.0) > 1e-12 || fabs(tiny_x[1] + 4.0) > 1e-12)
        return fail("tiny nonzero pivots were rejected or solved incorrectly");

    const double singular_A[4] = {1.0, 2.0, 2.0, 4.0};
    const double singular_b[2] = {3.0, 6.0};
    double untouched[2] = {17.0, 19.0};
    if (eshkol_linear_solve_portable(singular_A, singular_b, untouched, 2) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_SINGULAR)
        return fail("singular matrix did not report singular status");
    if (untouched[0] != 17.0 || untouched[1] != 19.0)
        return fail("failed solve modified the output buffer");

    const double nan_A[1] = {NAN}, one[1] = {1.0};
    double out[1] = {0.0};
    if (eshkol_linear_solve_portable(nan_A, one, out, 1) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE)
        return fail("nonfinite input was not refused");
    const double overflow_A[1] = {1e-309};
    if (eshkol_linear_solve_portable(overflow_A, one, out, 1) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_NONFINITE)
        return fail("nonfinite computed solution was not refused");
    if (eshkol_linear_solve_portable(one, one, out, 0) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_INVALID_ARGUMENT)
        return fail("zero dimension was accepted");
    if (eshkol_linear_solve_portable(one, one, out, INT64_MAX) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_SIZE_OVERFLOW)
        return fail("oversized dimensions did not report overflow");
    if (eshkol_linear_solve_portable(NULL, one, out, 1) !=
        ESHKOL_LINEAR_SOLVE_PORTABLE_INVALID_ARGUMENT)
        return fail("null input was accepted");

    puts("PASS: portable linear solve helper");
    return 0;
}
