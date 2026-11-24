/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Operand dynamic-range guard for reduced-precision f64 matrix multiplication
 * routes (f32, df64 double-float, fp24, and reduced-precision GPU paths).
 *
 * Narrowing finite IEEE-754 binary64 inputs outside the safe single-precision
 * product and accumulation window silently underflows small values (such as
 * 2^-180) to 0.0 or overflows large values (such as 2^180) to +/-inf, yielding
 * a normalized error of 1.0 or NaN against any finite accuracy budget.
 *
 * Any call whose operands contain a non-finite entry or a non-zero magnitude
 * outside the safe window declines the reduced-precision path and falls back
 * to exact IEEE-754 binary64 execution.
 */

#ifndef ESHKOL_BACKEND_GPU_MATMUL_RANGE_GUARD_H
#define ESHKOL_BACKEND_GPU_MATMUL_RANGE_GUARD_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return 1 iff every entry in A (M x K) and B (K x N) is finite and
 *        every non-zero entry lies inside the safe single-precision product
 *        and K-term accumulation window [min_nonzero_abs, max_abs].
 *
 * Exact zeros (+/-0.0) are always admitted.
 * For plain f32 / fp24 routes pass df64_pair = 0 (min_nonzero_abs = 2^-60,
 * guaranteeing |a*b| >= 2^-120 > FLT_MIN = 2^-126 with zero subnormal flush).
 * For double-float (hi + lo) routes pass df64_pair = 1 (min_nonzero_abs = 2^-48,
 * keeping low-limb products above FLT_MIN as well).
 */
static inline int eshkol_f64_matmul_operands_safe_for_f32(
    const double* A,
    const double* B,
    uint64_t M,
    uint64_t K,
    uint64_t N,
    int df64_pair) {
    if (!A || !B) return 0;
    if (M == 0 || K == 0 || N == 0) return 1;

    /* Guard against uint64_t element-count overflow. */
    if (M > UINT64_MAX / K || K > UINT64_MAX / N) return 0;
    const uint64_t na = M * K;
    const uint64_t nb = K * N;

    /* 2^-60 ~= 8.6736e-19 for f32; 2^-48 ~= 3.5527e-15 for df64 hi+lo pairs. */
    const double min_nonzero = df64_pair ? ldexp(1.0, -48) : ldexp(1.0, -60);
    /* Keep sum_{k=0}^{K-1} |A_ik * B_kj| <= FLT_MAX / 2 with 2x headroom. */
    const double max_abs = sqrt(((double)FLT_MAX * 0.5) / (double)K);
    if (!(max_abs > min_nonzero)) return 0;

    for (uint64_t i = 0; i < na; ++i) {
        const double v = A[i];
        if (!isfinite(v)) return 0;
        if (v != 0.0) {
            const double av = fabs(v);
            if (av < min_nonzero || av > max_abs) return 0;
        }
    }
    for (uint64_t i = 0; i < nb; ++i) {
        const double v = B[i];
        if (!isfinite(v)) return 0;
        if (v != 0.0) {
            const double av = fabs(v);
            if (av < min_nonzero || av > max_abs) return 0;
        }
    }
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_GPU_MATMUL_RANGE_GUARD_H */
