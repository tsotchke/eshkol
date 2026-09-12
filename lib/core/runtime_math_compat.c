/**
 * @file runtime_math_compat.c
 * @brief Math library functions the target C runtime does not provide.
 *
 * SPDX-License-Identifier: MIT
 *
 * Scheme's `round` is banker's rounding: a tie goes to the even neighbour.
 * The backend emits `llvm.roundeven.f64` for it. Whether that intrinsic
 * becomes an instruction or a call to the C library's `roundeven` is a
 * target decision: on AArch64 it is a single instruction (`frintn`), and on
 * x86-64 it is one only when SSE4.1's `roundsd` is in the baseline. Neither
 * the Windows Universal CRT nor Apple's libm ships `roundeven` -- it is
 * C23 -- so on x86-64 targets every ahead-of-time link failed with an
 * undefined symbol, while the same source linked on arm64 because the
 * intrinsic never became a call there. That is why this surfaced only in
 * the release workflow's x86-64 asset jobs.
 *
 * Raising the x86-64 baseline to SSE4.1 would also inline it, but that is a
 * change to the CPUs a shipped binary runs on, which is not a decision to
 * make from a link error. Supplying the function is semantically neutral.
 *
 * Compiled only where the platform does not already provide the symbol;
 * CMake decides with check_symbol_exists, so a C runtime that gains it
 * later silently stops using this definition instead of colliding with it.
 */

#include <math.h>

#if !defined(ESHKOL_HAVE_ROUNDEVEN)

double eshkol_compat_roundeven(double x);
float eshkol_compat_roundevenf(float x);

/**
 * @brief Round to the nearest integral value, ties to even.
 *
 * Written without `nearbyint`, which follows the current rounding mode;
 * this must be ties-to-even whatever that mode is. NaN and the infinities
 * pass through and the sign of a zero result is preserved, as C23 requires.
 *
 * @param x Value to round.
 * @return The nearest integral value, ties resolved to the even neighbour.
 */
double eshkol_compat_roundeven(double x) {
    if (isnan(x) || isinf(x) || x == 0.0) {
        return x;
    }
    double truncated = trunc(x);
    double fraction = fabs(x - truncated);
    double away = truncated + copysign(1.0, x);
    if (fraction > 0.5) {
        return away;
    }
    if (fraction < 0.5) {
        return copysign(truncated, x);
    }
    /* Exactly halfway: take whichever neighbour is even. fmod is exact on
     * integral values, so this comparison does not itself round. */
    return (fmod(truncated, 2.0) == 0.0) ? truncated : away;
}

/** @brief Single-precision @ref eshkol_compat_roundeven. */
float eshkol_compat_roundevenf(float x) {
    return (float)eshkol_compat_roundeven((double)x);
}

/* The names LLVM's lowering emits. Defined as aliases so the compat
 * implementation is testable under its own name. */
double roundeven(double x) { return eshkol_compat_roundeven(x); }
float roundevenf(float x) { return eshkol_compat_roundevenf(x); }

#endif /* !ESHKOL_HAVE_ROUNDEVEN */
