/**
 * @file complex_math.h
 * @brief Pure complex transcendental core, shared by the native runtime and
 *        the bytecode VM.
 *
 * This header is the SINGLE source of truth for every complex-domain
 * transcendental Eshkol implements. Both engines include it:
 *
 *   * the native runtime (`lib/core/complex_math.cpp`) wraps each function in
 *     an `extern "C"` symbol (`eshkol_complex_sqrt`, ...) that the LLVM
 *     back end emits calls to;
 *   * the bytecode VM (`lib/backend/vm_complex.c`) calls the inline functions
 *     directly.
 *
 * Because both engines evaluate the same expressions in the same order, the
 * two back ends agree bit-for-bit — engine parity is a property of the code
 * layout, not of a hand-maintained pair of implementations that can drift.
 * This mirrors the arrangement already used for i128 (`<eshkol/core/i128.h>`).
 *
 * Everything is written in the C/C++ common subset over plain `double` pairs:
 * no `_Complex`, no `std::complex`, so a C translation unit and a C++
 * translation unit both compile it without a language-specific shim.
 *
 * Branch cuts follow C99 Annex G / R7RS 6.2.6 (the principal branch):
 *   * `sqrt`  — cut along the negative real axis, continuous from above;
 *   * `log`   — cut along the negative real axis, imaginary part in (-pi, pi];
 *   * `asin`/`acos`  — cuts outside [-1, 1] on the real axis;
 *   * `atan`  — cuts outside [-i, i] on the imaginary axis.
 * Signed zero is respected where it selects the branch, so
 * `sqrt(-1 + 0i) = 0 + 1i` and `sqrt(-1 - 0i) = 0 - 1i`.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_CORE_COMPLEX_MATH_H
#define ESHKOL_CORE_COMPLEX_MATH_H

#include <math.h>

/* MSVC/clang-cl only expose the POSIX math constants (M_PI, M_LN2, ...)
 * when _USE_MATH_DEFINES is defined before <math.h> is first included, and
 * that define is out of this header's control (it comes from whichever
 * translation unit includes us first). Guard each constant this header
 * actually uses so both engines (native runtime + bytecode VM) build on
 * Windows without relying on caller-side defines. Mirrors the M_PI/M_E
 * guard already used for the same reason in lib/quantum/quantum_rng.c.
 */
#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458176568
#endif

#ifndef M_LN10
#define M_LN10 2.302585092994045684017991454684364208
#endif

#ifndef M_PI_2
#define M_PI_2 1.570796326794896619231321691639751442
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A complex value as a bare pair of IEEE-754 doubles.
 *
 * Layout-compatible with `eshkol_complex_number_t` (native runtime) and
 * `VmComplex` (VM): both are `{ double real; double imag; }` with no padding,
 * so a pointer to either can be reinterpreted as this type.
 */
typedef struct eshkol_cpx {
    double re;
    double im;
} eshkol_cpx;

/** @brief Build a complex value from its two components. */
static inline eshkol_cpx eshkol_cpx_make(double re, double im) {
    eshkol_cpx z;
    z.re = re;
    z.im = im;
    return z;
}

/* ─────────────────────────────── arithmetic ─────────────────────────────── */

/** @brief Sum of two complex values. */
static inline eshkol_cpx eshkol_cpx_add(eshkol_cpx a, eshkol_cpx b) {
    return eshkol_cpx_make(a.re + b.re, a.im + b.im);
}

/** @brief Difference of two complex values. */
static inline eshkol_cpx eshkol_cpx_sub(eshkol_cpx a, eshkol_cpx b) {
    return eshkol_cpx_make(a.re - b.re, a.im - b.im);
}

/** @brief Product of two complex values: (ac - bd) + (ad + bc)i. */
static inline eshkol_cpx eshkol_cpx_mul(eshkol_cpx a, eshkol_cpx b) {
    return eshkol_cpx_make(a.re * b.re - a.im * b.im,
                           a.re * b.im + a.im * b.re);
}

/**
 * @brief Quotient of two complex values via Smith's formula.
 *
 * Smith's algorithm scales by the larger denominator component first, so the
 * intermediate `c*c + d*d` that would overflow for operands near DBL_MAX is
 * never formed. This matches the division already documented for `/` in
 * COMPLETE_LANGUAGE_SPECIFICATION.md 15.2.2.
 */
static inline eshkol_cpx eshkol_cpx_div(eshkol_cpx a, eshkol_cpx b) {
    double r, den;
    if (fabs(b.im) >= fabs(b.re)) {
        if (b.im == 0.0) {
            /* Both components zero: division by exact complex zero. Let IEEE
             * produce the infinity/NaN pattern instead of inventing one. */
            return eshkol_cpx_make(a.re / b.re, a.im / b.re);
        }
        r = b.re / b.im;
        den = b.re * r + b.im;
        return eshkol_cpx_make((a.re * r + a.im) / den,
                               (a.im * r - a.re) / den);
    }
    r = b.im / b.re;
    den = b.im * r + b.re;
    return eshkol_cpx_make((a.re + a.im * r) / den,
                           (a.im - a.re * r) / den);
}

/** @brief Scale a complex value by a real factor. */
static inline eshkol_cpx eshkol_cpx_scale(eshkol_cpx z, double s) {
    return eshkol_cpx_make(z.re * s, z.im * s);
}

/** @brief Multiply by i: i(a+bi) = -b + ai. */
static inline eshkol_cpx eshkol_cpx_mul_i(eshkol_cpx z) {
    return eshkol_cpx_make(-z.im, z.re);
}

/** @brief Divide by i, i.e. multiply by -i: -i(a+bi) = b - ai. */
static inline eshkol_cpx eshkol_cpx_div_i(eshkol_cpx z) {
    return eshkol_cpx_make(z.im, -z.re);
}

/** @brief Overflow-safe magnitude |a+bi|, computed with `hypot`. */
static inline double eshkol_cpx_abs(eshkol_cpx z) {
    return hypot(z.re, z.im);
}

/** @brief Principal argument arg(a+bi) = atan2(b, a), in (-pi, pi]. */
static inline double eshkol_cpx_arg(eshkol_cpx z) {
    return atan2(z.im, z.re);
}

/* ────────────────────────── roots, exp and log ─────────────────────────── */

/**
 * @brief Principal square root.
 *
 * Uses the half-angle identity in its numerically stable algebraic form
 * (C99 Annex G): one `hypot`, one `sqrt` and one division, with the branch
 * selected by `copysign` on the imaginary part. Computing it in polar form
 * instead (`sqrt(r) * (cos(t/2) + i sin(t/2))`) loses the exact zero: for
 * `-1 + 0i` the polar route returns `6.1e-17 + 1i` where the exact principal
 * value — and what ESHKOL_LANGUAGE_GUIDE.md documents — is `0 + 1i`.
 */
static inline eshkol_cpx eshkol_cpx_sqrt(eshkol_cpx z) {
    double t;
    if (z.re == 0.0 && z.im == 0.0) {
        /* sqrt(+-0 +- 0i) = +0 with the imaginary sign preserved. */
        return eshkol_cpx_make(0.0, z.im);
    }
    t = sqrt((fabs(z.re) + hypot(z.re, z.im)) * 0.5);
    if (z.re >= 0.0) {
        return eshkol_cpx_make(t, z.im / (2.0 * t));
    }
    return eshkol_cpx_make(fabs(z.im) / (2.0 * t), copysign(t, z.im));
}

/** @brief Complex exponential: exp(a+bi) = exp(a)(cos b + i sin b). */
static inline eshkol_cpx eshkol_cpx_exp(eshkol_cpx z) {
    double ea = exp(z.re);
    return eshkol_cpx_make(ea * cos(z.im), ea * sin(z.im));
}

/** @brief Principal natural logarithm: log|z| + i arg(z). */
static inline eshkol_cpx eshkol_cpx_log(eshkol_cpx z) {
    return eshkol_cpx_make(log(hypot(z.re, z.im)), atan2(z.im, z.re));
}

/** @brief Base-2 logarithm: log(z) / log(2). */
static inline eshkol_cpx eshkol_cpx_log2(eshkol_cpx z) {
    return eshkol_cpx_scale(eshkol_cpx_log(z), 1.0 / M_LN2);
}

/** @brief Base-10 logarithm: log(z) / log(10). */
static inline eshkol_cpx eshkol_cpx_log10(eshkol_cpx z) {
    return eshkol_cpx_scale(eshkol_cpx_log(z), 1.0 / M_LN10);
}

/** @brief Base-2 exponential: exp(z * log 2). */
static inline eshkol_cpx eshkol_cpx_exp2(eshkol_cpx z) {
    return eshkol_cpx_exp(eshkol_cpx_scale(z, M_LN2));
}

/** @brief Principal power: a^b = exp(b log a), with 0^0 = 1 per R7RS. */
static inline eshkol_cpx eshkol_cpx_pow(eshkol_cpx a, eshkol_cpx b) {
    if (a.re == 0.0 && a.im == 0.0) {
        if (b.re == 0.0 && b.im == 0.0) return eshkol_cpx_make(1.0, 0.0);
        return eshkol_cpx_make(0.0, 0.0);
    }
    return eshkol_cpx_exp(eshkol_cpx_mul(b, eshkol_cpx_log(a)));
}

/* ──────────────────────────── circular trig ────────────────────────────── */

/** @brief sin(a+bi) = sin a cosh b + i cos a sinh b. */
static inline eshkol_cpx eshkol_cpx_sin(eshkol_cpx z) {
    return eshkol_cpx_make(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im));
}

/** @brief cos(a+bi) = cos a cosh b - i sin a sinh b. */
static inline eshkol_cpx eshkol_cpx_cos(eshkol_cpx z) {
    return eshkol_cpx_make(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im));
}

/**
 * @brief tan(a+bi) = (sin 2a + i sinh 2b) / (cos 2a + cosh 2b).
 *
 * The double-angle form is used rather than sin(z)/cos(z): its denominator is
 * a sum of a bounded and a positive term, so it never cancels, and it stays
 * finite for large |b| where sin(z) and cos(z) both overflow.
 */
static inline eshkol_cpx eshkol_cpx_tan(eshkol_cpx z) {
    double den;
    if (fabs(z.im) > 350.0) {
        /* cosh(2b) overflows; tan converges to +-i to well beyond 1 ulp. */
        return eshkol_cpx_make(0.0, z.im > 0.0 ? 1.0 : -1.0);
    }
    den = cos(2.0 * z.re) + cosh(2.0 * z.im);
    return eshkol_cpx_make(sin(2.0 * z.re) / den, sinh(2.0 * z.im) / den);
}

/* ─────────────────────────── hyperbolic trig ───────────────────────────── */

/** @brief sinh(a+bi) = sinh a cos b + i cosh a sin b. */
static inline eshkol_cpx eshkol_cpx_sinh(eshkol_cpx z) {
    return eshkol_cpx_make(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im));
}

/** @brief cosh(a+bi) = cosh a cos b + i sinh a sin b. */
static inline eshkol_cpx eshkol_cpx_cosh(eshkol_cpx z) {
    return eshkol_cpx_make(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im));
}

/**
 * @brief tanh(a+bi) = (sinh 2a + i sin 2b) / (cosh 2a + cos 2b).
 *
 * Double-angle form, stable for the same reason as ::eshkol_cpx_tan.
 */
static inline eshkol_cpx eshkol_cpx_tanh(eshkol_cpx z) {
    double den;
    if (fabs(z.re) > 350.0) {
        return eshkol_cpx_make(z.re > 0.0 ? 1.0 : -1.0, 0.0);
    }
    den = cosh(2.0 * z.re) + cos(2.0 * z.im);
    return eshkol_cpx_make(sinh(2.0 * z.re) / den, sin(2.0 * z.im) / den);
}

/* ────────────────────────── inverse functions ──────────────────────────── */

/** @brief asin(z) = -i log(iz + sqrt(1 - z^2)). */
static inline eshkol_cpx eshkol_cpx_asin(eshkol_cpx z) {
    eshkol_cpx one = eshkol_cpx_make(1.0, 0.0);
    eshkol_cpx root = eshkol_cpx_sqrt(eshkol_cpx_sub(one, eshkol_cpx_mul(z, z)));
    eshkol_cpx inner = eshkol_cpx_add(eshkol_cpx_mul_i(z), root);
    return eshkol_cpx_div_i(eshkol_cpx_log(inner));
}

/** @brief acos(z) = pi/2 - asin(z). */
static inline eshkol_cpx eshkol_cpx_acos(eshkol_cpx z) {
    eshkol_cpx s = eshkol_cpx_asin(z);
    return eshkol_cpx_make(M_PI_2 - s.re, -s.im);
}

/** @brief atan(z) = (i/2)(log(1 - iz) - log(1 + iz)). */
static inline eshkol_cpx eshkol_cpx_atan(eshkol_cpx z) {
    eshkol_cpx one = eshkol_cpx_make(1.0, 0.0);
    eshkol_cpx iz = eshkol_cpx_mul_i(z);
    eshkol_cpx lo = eshkol_cpx_log(eshkol_cpx_sub(one, iz));
    eshkol_cpx hi = eshkol_cpx_log(eshkol_cpx_add(one, iz));
    return eshkol_cpx_scale(eshkol_cpx_mul_i(eshkol_cpx_sub(lo, hi)), 0.5);
}

/** @brief asinh(z) = log(z + sqrt(z^2 + 1)). */
static inline eshkol_cpx eshkol_cpx_asinh(eshkol_cpx z) {
    eshkol_cpx one = eshkol_cpx_make(1.0, 0.0);
    eshkol_cpx root = eshkol_cpx_sqrt(eshkol_cpx_add(eshkol_cpx_mul(z, z), one));
    return eshkol_cpx_log(eshkol_cpx_add(z, root));
}

/**
 * @brief acosh(z) = log(z + sqrt(z+1) sqrt(z-1)).
 *
 * The product of two separate square roots (rather than `sqrt(z^2 - 1)`) is
 * what places the branch cut on (-inf, 1] as C99 Annex G requires.
 */
static inline eshkol_cpx eshkol_cpx_acosh(eshkol_cpx z) {
    eshkol_cpx one = eshkol_cpx_make(1.0, 0.0);
    eshkol_cpx rp = eshkol_cpx_sqrt(eshkol_cpx_add(z, one));
    eshkol_cpx rm = eshkol_cpx_sqrt(eshkol_cpx_sub(z, one));
    return eshkol_cpx_log(eshkol_cpx_add(z, eshkol_cpx_mul(rp, rm)));
}

/** @brief atanh(z) = (log(1 + z) - log(1 - z)) / 2. */
static inline eshkol_cpx eshkol_cpx_atanh(eshkol_cpx z) {
    eshkol_cpx one = eshkol_cpx_make(1.0, 0.0);
    eshkol_cpx hi = eshkol_cpx_log(eshkol_cpx_add(one, z));
    eshkol_cpx lo = eshkol_cpx_log(eshkol_cpx_sub(one, z));
    return eshkol_cpx_scale(eshkol_cpx_sub(hi, lo), 0.5);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_CORE_COMPLEX_MATH_H */
