/**
 * @file riemannian_core.h
 * @brief Closed-form constant-curvature geometry (Poincare ball, Euclidean
 *        space, round sphere) in f64, shared by the VM's geometric opcodes.
 *
 * WHY THIS FILE EXISTS. `lib/backend/vm_geometric.c` used to implement the
 * curved operations as their FLAT counterparts: `hyperbolic-exp-map` was vector
 * addition, `hyperbolic-log-map` subtraction, `geodesic-distance` and
 * `poincare-distance` the L2 distance, `mobius-add` addition, `mobius-scalar-mul`
 * a scale, `parallel-transport` and `riemannian-grad` the identity. Each of them
 * accepted a curvature argument and discarded it. That body was the one every
 * shipped VM build compiled — every CI lane, every release binary, the WASM
 * playground. The result was Euclidean answers returned under Riemannian names,
 * with nothing in the output showing the argument had been dropped: the same
 * plausible-wrong-number class `frechet_mean_core.h` documents for the Euclidean
 * weighted average that used to stand in for the Frechet mean. That second
 * legacy qLLM dispatch body has since been deleted
 * outright — it did not compile against the current libsemiclassical_qllm ABI
 * and was fp32 throughout — so the forms below are the ONE implementation of
 * this geometry on the VM engine, not the default of two.
 *
 * WHY A HEADER AND NOT A LIBRARY TU. Identical reason to
 * `inc/eshkol/backend/frechet_mean_core.h`: `lib/backend/vm_geometric.c` is a
 * unity-build include consumed by `lib/backend/eshkol_vm.c`, which is also built
 * as a single translation unit on its own (the `eshkol-vm-standalone-test`
 * target), so a call to an external symbol would not link there. Static
 * functions in a header give every caller ONE source of truth with no link edge.
 *
 * ═══ THE MODEL, IN ONE PLACE (ADR-0018) ════════════════════════════════════
 *
 * Every entry point below takes @c K, the SIGNED SECTIONAL CURVATURE, and
 * evaluates ONE chart for every sign: the kappa-stereographic chart in the
 * canonical-flat gauge alpha = 1 (docs/design/adr/0018-signed-curvature-
 * stereographic-geometry.md). The chart and its normalisation are fixed by one
 * constant and two accessors, and nothing else in this file or its callers
 * open-codes them:
 *
 *   ESHKOL_RM_ALPHA         the conformal factor at the origin (alpha = 1)
 *   eshkol_rm_ball_param(c) the chart parameter B = c alpha^2 / 4, c = -K
 *   eshkol_rm_lambda(x,K,n) the conformal factor at a point
 *
 * The metric is
 *
 *       g_K(x) = lambda_x^2 <.,.>,   lambda_x = alpha / (1 - B |x|^2),
 *       B = -K alpha^2 / 4  (so rho = K/4 = -B at alpha = 1),
 *
 * and the gyrovector operations are Mobius addition (+)_B with the SAME signed
 * B. Consequences, all of them by construction rather than by special case:
 *
 *   K < 0   B > 0: the open ball of radius 1/sqrt(B) = 2/sqrt(-K).
 *   K = 0   B = 0: canonical Euclidean R^n, lambda = 1, (+)_0 = vector sum.
 *   K > 0   B < 0: the round sphere of radius 1/sqrt(K) minus one point (the
 *           POLE, at infinity in the chart). Every finite point is a point of
 *           the manifold; the antipode of x is -x/(|B| |x|^2), its CUT LOCUS.
 *
 * The family is real-analytic in K through K = 0 at fixed chart coordinates.
 * The generalized trigonometric ratios (eshkol_rm_psi, eshkol_rm_tan_ratio,
 * eshkol_rm_artan_ratio) are functions of one dimensionless signed argument
 * and switch to a degree-10 series for |argument| <= ESHKOL_RM_Q_SERIES, so the
 * K -> 0 corner and the K = 0 point are one expression. Positive cut loci and
 * the negative ball boundary are explicit domain errors, never clamps.
 *
 * Legacy coordinates (LEGACY_BALL_V1: the alpha = 2 ball of radius 1/sqrt(-K),
 * K = 0 with lambda = 1, K > 0 as an ambient sphere in R^{n+1}) are read only
 * through the migration helpers at the end of this file: x_v2 = 2 x_v1 for ball
 * points and tangent vectors, the stereographic projection for ambient sphere
 * points. Serialized geometry carries ESHKOL_GEOMETRY_CONVENTION_* and is
 * refused when the id is missing or does not match.
 *
 * The ambient round sphere survives as an EMBEDDED MODEL, not as a second
 * chart: the `eshkol_rm_sphere_*` functions take points of R^{n+1} on the
 * sphere of radius 1/sqrt(K) and back the ops whose contract is that embedding
 * (`great-circle-distance`, `slerp`, `spherical-exp/log/project`, and the
 * space-form SPHERICAL factor). eshkol_rm_sphere_to_chart and
 * eshkol_rm_chart_to_sphere are the isometry between the two.
 *
 * exp, log, parallel transport, projection and Mobius addition depend on the
 * chart parameter B only: a constant conformal rescale of a metric leaves the
 * Levi-Civita connection unchanged. Distances, norms, the gradient conversion
 * and the curvature derivatives carry alpha.
 *
 * RELATION TO THE AD BRIDGE. `lib/bridge/qllm_bridge.cpp` includes this header
 * and calls these same f64 primitives for its hyperbolic distance, Poincare
 * exponential/logarithmic maps, and geodesic-attention distance scores. The
 * bridge validates that its public curvature is negative, converts it to the
 * common ball parameter c = -K, and retains the common forward weights for its
 * reverse rule. There is no second bridge-side forward formula to drift from
 * this model: the VM and bridge distance/map forwards are one implementation.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_RIEMANNIAN_CORE_H
#define ESHKOL_BACKEND_RIEMANNIAN_CORE_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>
#include <stdlib.h>
#ifdef __cplusplus
#include <cmath>
#include <vector>
#endif

/* The compensated kernels below are error-free transformations only when the
 * compiler keeps their rounded products and sums as separate operations. Keep
 * that policy local to this shared implementation; callers may use a faster
 * contraction policy elsewhere in the translation unit. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#pragma clang fp contract(off)
#endif

/* The conformal factor at the origin of the chart: the gauge alpha of
 * ADR-0018. alpha = 1 is the canonical-flat gauge: K = 0 is the Euclidean
 * metric, Riemannian gradients at K = 0 are unscaled, and the negative ball has
 * radius 2/sqrt(-K). */
#define ESHKOL_RM_ALPHA 1.0

/* Serialized-geometry convention ids. Every checkpoint, curvature schedule,
 * product factor or attention temperature that leaves the process carries one;
 * eshkol_rm_check_geometry_header refuses anything else. */
#define ESHKOL_GEOMETRY_CONVENTION_UNVERSIONED 0u
#define ESHKOL_GEOMETRY_CONVENTION_LEGACY_BALL_V1 1u
#define ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1 2u

/* Relative tolerance on |x| = R for a point of the EMBEDDED sphere model
 * (eshkol_rm_sphere_*). A point off the sphere has no geodesic relation to
 * another point on it, so those ops refuse rather than project silently;
 * `spherical-project` is the op that moves a point onto the sphere. */
#define ESHKOL_RM_SPHERE_TOL 1e-9

/* Relative tolerance on <x,v> = 0 for a vector claimed to be tangent to the
 * embedded sphere at x. RELATIVE TO |v| AND R, both: the test is
 * |<x,v>| <= tol R |v|, which is scale-free in v. */
#define ESHKOL_RM_TANGENT_TOL 1e-9

/* The one series switch of ADR-0018. Every generalized ratio is a function of
 * a dimensionless signed argument q (q = K r^2/4 up to sign conventions noted
 * at each function); for |q| <= 2^-8 the degree-10 series is evaluated, else
 * the closed form. The first omitted term is below 2^-88 relative, so the
 * switch is invisible at f64. The series also avoids forming B*r^2 when that
 * product is below the f64 subnormal floor. */
#define ESHKOL_RM_Q_SERIES 0x1p-8

static void eshkol_rm_two_sum(double a, double b, double* sum, double* error);
static void eshkol_rm_two_product(double a, double b, double* product,
                                  double* error);
static void eshkol_rm_dd_add(double a_hi, double a_lo,
                             double b_hi, double b_lo,
                             double* out_hi, double* out_lo);

static double eshkol_rm_dot(const double* a, const double* b, int n) {
    double sum_hi = 0.0, sum_lo = 0.0;
    for (int i = 0; i < n; i++) {
        double term, term_error, next_hi, next_lo;
        eshkol_rm_two_product(a[i], b[i], &term, &term_error);
        eshkol_rm_dd_add(sum_hi, sum_lo, term, term_error,
                         &next_hi, &next_lo);
        sum_hi = next_hi;
        sum_lo = next_lo;
    }
    return sum_hi + sum_lo;
}

/* Error-free sum of two finite doubles. */
static void eshkol_rm_two_sum(double a, double b, double* sum, double* error) {
    double s = a + b;
    double bb = s - a;
    *sum = s;
    *error = (a - (s - bb)) + (b - bb);
}

/* Error-free product of two finite doubles. The rounded product is
 * materialised before the explicit FMA computes its residual. */
static void eshkol_rm_two_product(double a, double b, double* product,
                                  double* error) {
    double p = a * b;
#ifdef __cplusplus
    double e = std::fma(a, b, -p);
#else
    double e = fma(a, b, -p);
#endif
    *product = p;
    *error = e;
}

static double eshkol_rm_fma(double a, double b, double c) {
#ifdef __cplusplus
    return std::fma(a, b, c);
#else
    return fma(a, b, c);
#endif
}

static void eshkol_rm_dd_add(double a_hi, double a_lo,
                             double b_hi, double b_lo,
                             double* out_hi, double* out_lo) {
    double high, high_error;
    double low, low_error;
    double middle, middle_error;
    double rounded, rounded_error;
    eshkol_rm_two_sum(a_hi, b_hi, &high, &high_error);
    eshkol_rm_two_sum(a_lo, b_lo, &low, &low_error);
    eshkol_rm_two_sum(high_error, low, &middle, &middle_error);
    eshkol_rm_two_sum(high, middle, &rounded, &rounded_error);
    double tail, tail_error;
    eshkol_rm_two_sum(rounded_error, middle_error, &tail, &tail_error);
    double final_low, final_low_error;
    eshkol_rm_two_sum(tail, low_error, &final_low, &final_low_error);
    eshkol_rm_two_sum(rounded, final_low, out_hi, out_lo);
    *out_lo += final_low_error;
}

/* Multiply two double-double expansions and retain the rounded high and low
 * words. Every product entering the expansion has an explicit residual. */
static void eshkol_rm_dd_mul(double a_hi, double a_lo,
                             double b_hi, double b_lo,
                             double* out_hi, double* out_lo) {
    double p, p_error;
    double cross_a, cross_a_error;
    double cross_b, cross_b_error;
    double cross, cross_error;
    double tail, tail_error;
    double hi, lo;
    eshkol_rm_two_product(a_hi, b_hi, &p, &p_error);
    eshkol_rm_two_product(a_hi, b_lo, &cross_a, &cross_a_error);
    eshkol_rm_two_product(a_lo, b_hi, &cross_b, &cross_b_error);
    eshkol_rm_two_sum(cross_a, cross_b, &cross, &cross_error);
    eshkol_rm_dd_add(p, p_error, cross, cross_error, &hi, &lo);
    eshkol_rm_two_product(a_lo, b_lo, &tail, &tail_error);
    eshkol_rm_dd_add(hi, lo, tail, tail_error, out_hi, out_lo);
}

static double eshkol_rm_norm(const double* a, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double ai = fabs(a[i]);
        if (!isfinite(ai)) return ai;
        if (ai > max_abs) max_abs = ai;
    }
    if (max_abs == 0.0) return 0.0;
    double scaled_norm2_hi = 0.0, scaled_norm2_lo = 0.0;
    for (int i = 0; i < n; i++) {
        double z = a[i] / max_abs;
        double term, term_error, next_hi, next_lo;
        eshkol_rm_two_product(z, z, &term, &term_error);
        eshkol_rm_dd_add(scaled_norm2_hi, scaled_norm2_lo,
                         term, term_error, &next_hi, &next_lo);
        scaled_norm2_hi = next_hi;
        scaled_norm2_lo = next_lo;
    }
    double scaled_norm = sqrt(scaled_norm2_hi + scaled_norm2_lo);
    return max_abs * scaled_norm;
}

/* The scaled squared norm times a nonnegative factor. Forming |a|^2 first
 * overflows for otherwise valid finite coordinates; multiplying the factor by
 * max_abs before multiplying by max_abs again preserves finite results such as
 * B * (1e308)^2 when B is small. */
static double eshkol_rm_scaled_product4(double a, double b, double c, double d);

/* Square root of every finite nonnegative f64 value with explicit exponent
 * halving. frexp/ldexp keeps the operation well-defined at both the subnormal
 * and DBL_MAX ends of the range. */
static double eshkol_rm_sqrt_nonnegative(double x) {
    if (!(x > 0.0) || !isfinite(x)) return sqrt(x);
    int exponent = 0;
    double mantissa = frexp(x, &exponent);
    if (exponent % 2 != 0) {
        mantissa *= 2.0;
        --exponent;
    }
    return ldexp(sqrt(mantissa), exponent / 2);
}

static double eshkol_rm_scaled_norm2_times(const double* a, double factor, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double ai = fabs(a[i]);
        if (!isfinite(ai)) return ai;
        if (ai > max_abs) max_abs = ai;
    }
    if (max_abs == 0.0 || factor == 0.0) return 0.0;
    double scaled_norm2_hi = 0.0, scaled_norm2_lo = 0.0;
    for (int i = 0; i < n; i++) {
        double z = a[i] / max_abs;
        double term, term_error, next_hi, next_lo;
        eshkol_rm_two_product(z, z, &term, &term_error);
        eshkol_rm_dd_add(scaled_norm2_hi, scaled_norm2_lo,
                         term, term_error, &next_hi, &next_lo);
        scaled_norm2_hi = next_hi;
        scaled_norm2_lo = next_lo;
    }
    return eshkol_rm_scaled_product4(factor, max_abs, max_abs,
                                     scaled_norm2_hi + scaled_norm2_lo);
}

/* Multiply four finite factors in one exponent-scaled operation.  In
 * particular, do not form factor*max_abs first: that product can underflow
 * even when factor*max_abs*max_abs*sum is representable. */
static double eshkol_rm_scaled_product4(double a, double b, double c, double d) {
    if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) return 0.0;
    int ea, eb, ec, ed;
    double ma = frexp(a, &ea), mb = frexp(b, &eb);
    double mc = frexp(c, &ec), md = frexp(d, &ed);
    double hi, lo;
    eshkol_rm_dd_mul(ma, 0.0, mb, 0.0, &hi, &lo);
    eshkol_rm_dd_mul(hi, lo, mc, 0.0, &hi, &lo);
    eshkol_rm_dd_mul(hi, lo, md, 0.0, &hi, &lo);
    return scalbn(hi + lo, ea + eb + ec + ed);
}

/* Multiply a short list of finite factors without exposing an overflowing or
 * underflowing intermediate.  Curvature derivatives use this for powers of
 * the pair distance whose final value, unlike an intermediate square, is
 * representable. */
static double eshkol_rm_scaled_product_n(const double* factors, int count) {
    double mantissa = 1.0;
    int exponent = 0;
    int sign = 1;
    for (int i = 0; i < count; ++i) {
        const double factor = factors[i];
        if (factor == 0.0) return 0.0;
        if (!isfinite(factor)) return NAN;
        if (factor < 0.0) sign = -sign;
        int factor_exponent = 0;
        mantissa *= frexp(fabs(factor), &factor_exponent);
        exponent += factor_exponent;
    }
    return scalbn(sign * mantissa, exponent);
}

/* The same product with one f64 residual retained. The residual is important
 * when the product is close to one and the desired result is 1-product: the
 * returned high word alone would expose the subtraction to the f64 ulp at 1. */
static void eshkol_rm_scaled_product4_dd(double a, double b, double c, double d,
                                         double* hi_out, double* lo_out) {
    int ea, eb, ec, ed;
    double ma = frexp(a, &ea), mb = frexp(b, &eb);
    double mc = frexp(c, &ec), md = frexp(d, &ed);
    double hi, lo;
    eshkol_rm_dd_mul(ma, 0.0, mb, 0.0, &hi, &lo);
    eshkol_rm_dd_mul(hi, lo, mc, 0.0, &hi, &lo);
    eshkol_rm_dd_mul(hi, lo, md, 0.0, &hi, &lo);
    const int exponent = ea + eb + ec + ed;
    *hi_out = scalbn(hi, exponent);
    *lo_out = scalbn(lo, exponent);
}

/* A curvature-weighted square with the curvature exponent carried separately
 * through the product.  In particular, never form factor*a first: at the
 * subnormal curvature floor that intermediate has too few significand bits
 * for the derivative even when factor*a*a is representable. */
static double eshkol_rm_scaled_square(double factor, double a) {
    return eshkol_rm_scaled_product4(factor, a, a, 1.0);
}

/* The same dimensionless difference norm, but with the multiplier retained
 * inside the exponent-scaled product.  In particular, do not form a[i]-b[i]
 * before scaling: opposite finite endpoints can have a difference above
 * DBL_MAX even when factor*|a-b|^2 is representable. */
static double eshkol_rm_difference_norm2_times2(const double* a,
                                                const double* b, double factor,
                                                double factor2, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        const double aa = fabs(a[i]), bb = fabs(b[i]);
        if (!isfinite(aa) || !isfinite(bb)) return NAN;
        if (aa > max_abs) max_abs = aa;
        if (bb > max_abs) max_abs = bb;
    }
    if (max_abs == 0.0 || factor == 0.0) return 0.0;
    int exponent = 0;
    (void)frexp(max_abs, &exponent);
    const double scale = scalbn(0.5, exponent);
    double sum_hi = 0.0, sum_lo = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u = a[i] / scale - b[i] / scale;
        double term, term_error;
        eshkol_rm_two_product(u, u, &term, &term_error);
        double next_hi, next_lo;
        eshkol_rm_dd_add(sum_hi, sum_lo, term, term_error,
                         &next_hi, &next_lo);
        sum_hi = next_hi;
        sum_lo = next_lo;
    }
    return eshkol_rm_scaled_product4(factor, scale, scale,
                                      factor2 * (sum_hi + sum_lo));
}

static double eshkol_rm_difference_norm2_times(const double* a,
                                               const double* b,
                                               double factor, int n) {
    return eshkol_rm_difference_norm2_times2(a, b, factor, 1.0, n);
}

/* A scaled dot product for formulas whose true value is multiplied by factor.
 * This keeps factor*<a,b> representable when <a,b> itself would overflow. */
static double eshkol_rm_scaled_dot_factor(const double* a, const double* b,
                                          double factor, int n) {
    double ma = 0.0, mb = 0.0;
    for (int i = 0; i < n; i++) {
        double aa = fabs(a[i]), bb = fabs(b[i]);
        if (!isfinite(aa) || !isfinite(bb)) return NAN;
        if (aa > ma) ma = aa;
        if (bb > mb) mb = bb;
    }
    if (ma == 0.0 || mb == 0.0 || factor == 0.0) return 0.0;
    double sum_hi = 0.0, sum_lo = 0.0;
    for (int i = 0; i < n; i++) {
        double term, term_error, next_hi, next_lo;
        eshkol_rm_two_product(a[i] / ma, b[i] / mb, &term, &term_error);
        eshkol_rm_dd_add(sum_hi, sum_lo, term, term_error,
                         &next_hi, &next_lo);
        sum_hi = next_hi;
        sum_lo = next_lo;
    }
    return eshkol_rm_scaled_product4(factor, ma, mb, sum_hi + sum_lo);
}

/* Exact equality is the only coincidence predicate. In particular, a nonzero
 * separation must not be flattened just because its norm is small in absolute
 * units: the metric and point scale determine its gradient. */
static int eshkol_rm_points_equal(const double* x, const double* y, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] != y[i]) return 0;
    return 1;
}

/* Negative collinearity after normalization is the common, scale-aware
 * antipode predicate used by spherical distance and spherical log. */
/* A finite binary64 value is an exact signed dyadic rational.  Store its
 * absolute value as an odd integer times a power of two.  Removing powers of
 * two from the integer is the same lossless renormalization that frexp/ldexp
 * would provide, but it remains exact even when the requested scale is
 * outside binary64's exponent range. */
typedef struct {
    uint64_t mantissa;
    int exponent;
} eshkol_rm_dyadic_t;

static int eshkol_rm_dyadic_from_double(double value,
                                        eshkol_rm_dyadic_t* out) {
    uint64_t bits = 0;
    uint64_t fraction;
    uint64_t biased_exponent;
    if (!out) return 0;
    memcpy(&bits, &value, sizeof bits);
    fraction = bits & UINT64_C(0x000fffffffffffff);
    biased_exponent = (bits >> 52) & UINT64_C(0x7ff);
    if (biased_exponent == UINT64_C(0)) {
        if (fraction == 0) return 0;
        out->mantissa = fraction;
        out->exponent = -1074;
    } else if (biased_exponent != UINT64_C(0x7ff)) {
        out->mantissa = UINT64_C(0x0010000000000000) | fraction;
        out->exponent = (int)biased_exponent - 1023 - 52;
    } else {
        return 0;
    }
    while ((out->mantissa & UINT64_C(1)) == 0) {
        out->mantissa >>= 1;
        ++out->exponent;
    }
    return 1;
}

static int eshkol_rm_all_finite(const double* a, int n) {
    if (!a) return 0;
    for (int i = 0; i < n; i++)
        if (!isfinite(a[i])) return 0;
    return 1;
}

static uint64_t eshkol_rm_gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

/* Compare a/b with c/d exactly without forming either floating-point product.
 * The dyadic exponents must agree, and gcd cancellation reduces the odd
 * mantissas before comparing them.  This is exact rational arithmetic using
 * only integer division, so subnormal residuals cannot underflow to zero. */
static int eshkol_rm_cross_ratio_equal(double a, double b, double c, double d) {
    eshkol_rm_dyadic_t da, db, dc, dd;
    uint64_t left_a, left_b, right_a, right_b, g;
    if (!eshkol_rm_dyadic_from_double(a, &da) ||
        !eshkol_rm_dyadic_from_double(b, &db) ||
        !eshkol_rm_dyadic_from_double(c, &dc) ||
        !eshkol_rm_dyadic_from_double(d, &dd)) return 0;
    if (da.exponent + db.exponent != dc.exponent + dd.exponent)
        return 0;

    left_a = da.mantissa;
    left_b = db.mantissa;
    right_a = dc.mantissa;
    right_b = dd.mantissa;
    g = eshkol_rm_gcd_u64(left_a, right_a);
    left_a /= g;
    right_a /= g;
    g = eshkol_rm_gcd_u64(left_a, right_b);
    left_a /= g;
    right_b /= g;
    g = eshkol_rm_gcd_u64(left_b, right_a);
    left_b /= g;
    right_a /= g;
    g = eshkol_rm_gcd_u64(left_b, right_b);
    left_b /= g;
    right_b /= g;
    return left_a == 1 && left_b == 1 && right_a == 1 && right_b == 1;
}

static int eshkol_rm_sphere_antipodal(const double* x, const double* y, int n) {
    int pivot = -1;
    for (int i = 0; i < n; i++) {
        int x_zero = (x[i] == 0.0);
        int y_zero = (y[i] == 0.0);
        if (x_zero != y_zero) return 0;
        if (!x_zero) {
            if ((x[i] < 0.0) == (y[i] < 0.0)) return 0;
            if (pivot < 0) pivot = i;
        }
    }
    if (pivot < 0) return 0;

    for (int i = 0; i < n; i++) {
        if (x[i] != 0.0 &&
            !eshkol_rm_cross_ratio_equal(x[i], y[pivot],
                                         x[pivot], y[i])) return 0;
    }
    return 1;
}

/* Cancellation-free spherical geometry. The input points are first
 * canonicalized to the requested radius. If u is non-NULL it receives the
 * tangent at x. */
static const char* eshkol_rm_sphere_log_map_stable(const double* x,
                                                   const double* y,
                                                   double R, int n,
                                                   double* out) {
    double xn = eshkol_rm_norm(x, n);
    double yn = eshkol_rm_norm(y, n);
    if (!(xn > 0.0) || !(yn > 0.0)) return "a sphere point has zero norm";

    double cosine = 0.0;
    double sine = 0.0;
    for (int i = 0; i < n; i++) {
        double ai = x[i] / xn;
        double bi = y[i] / yn;
        cosine += ai * bi;
        for (int j = i + 1; j < n; j++) {
            double aj = x[j] / xn;
            double bj = y[j] / yn;
            sine = hypot(sine, ai * bj - aj * bi);
        }
        out[i] = 0.0;
    }
    if (!(sine > 0.0)) return "the points are coincident: sphere log is zero";

    /* out is T_k = sum wedge_ij * d(wedge_ij)/d(a_k). */
    for (int i = 0; i < n; i++) {
        double ai = x[i] / xn;
        double bi = y[i] / yn;
        for (int j = i + 1; j < n; j++) {
            double aj = x[j] / xn;
            double bj = y[j] / yn;
            double wedge = ai * bj - aj * bi;
            out[i] += wedge * bj;
            out[j] -= wedge * bi;
        }
    }

    double theta = atan2(sine, cosine);
    double denominator = sine * sine + cosine * cosine;
    for (int i = 0; i < n; i++) {
        double bi = y[i] / yn;
        double dtheta = (cosine * (out[i] / sine) - sine * bi) /
                        denominator;
        out[i] = -R * theta * dtheta;
    }
    return NULL;
}

static double eshkol_rm_sphere_angle(const double* x, const double* y,
                                     double R, int n, double* u) {
    double xn = eshkol_rm_norm(x, n);
    double yn = eshkol_rm_norm(y, n);
    double chord = 0.0;
    for (int i = 0; i < n; i++) {
        double delta = y[i] / yn - x[i] / xn;
        chord = hypot(chord, delta);
    }
    double half_chord_sq = 0.5 * chord * chord;
    double cosine = 1.0 - half_chord_sq;
    double tangent_cosine = 0.0;
    for (int i = 0; i < n; i++)
        tangent_cosine += (x[i] / xn) * (y[i] / yn);
    double sine = 0.0;
    for (int i = 0; i < n; i++) {
        double xi = x[i] / xn;
        double yi = y[i] / yn;
        for (int j = i + 1; j < n; j++) {
            double xj = x[j] / xn;
            double yj = y[j] / yn;
            sine = hypot(sine, xi * yj - xj * yi);
        }
        if (u) u[i] = R * (yi - tangent_cosine * xi);
    }
    return atan2(sine, cosine);
}

/**
 * @brief The great-circle angle between the DIRECTIONS of two nonzero vectors.
 *
 * `great-circle-distance` is documented (docs/reference/stdlib/geometry.md,
 * id 819) as the scale-invariant angle between two arbitrary vectors, not the
 * distance between two points already on the sphere: each input is normalized
 * with the hypot norm first, so a pair of subnormals and a pair spanning
 * 1e300/1e-300 give the same answer as their unit representatives. Computing
 * it as `acos(<x,y>/(|x||y|))` instead — the raw dot product divided by the
 * product of the norms — underflows both the dot product and |x||y| to zero for
 * subnormal inputs and overflows |x||y| for large ones, so it reported an
 * angle of acos(0/0) or acos(0) rather than the angle that is there.
 *
 * Contract, in the order the checks apply:
 *   - a non-finite coordinate propagates as NaN under IEEE rules rather than
 *     raising or collapsing to zero;
 *   - a zero-norm endpoint has no direction and the documented answer is 0.0;
 *   - an EXACTLY negatively-collinear pair is the cut locus, where the
 *     shortest geodesic is not unique, and raises. The predicate is the exact
 *     dyadic-rational one (eshkol_rm_sphere_antipodal), so a genuine
 *     near-antipode whose products underflow is still evaluated;
 *   - otherwise the cancellation-free atan2 form (eshkol_rm_sphere_angle).
 *
 * @return NULL on success (angle written to @p out), else a reason.
 */
static const char* eshkol_rm_sphere_direction_angle(const double* x,
                                                    const double* y,
                                                    int n, double* out) {
    if (!x || !y || !out || n <= 0) return "great-circle distance needs two "
                                           "vectors of the same nonzero size";
    if (!eshkol_rm_all_finite(x, n) || !eshkol_rm_all_finite(y, n)) {
        /* IEEE propagation: an unordered coordinate makes the angle unordered. */
        *out = (double)NAN;
        return NULL;
    }
    {
        const double xn = eshkol_rm_norm(x, n);
        const double yn = eshkol_rm_norm(y, n);
        if (!(xn > 0.0) || !(yn > 0.0)) { *out = 0.0; return NULL; }
    }
    if (eshkol_rm_sphere_antipodal(x, y, n))
        return "the two points are antipodal: the great-circle geodesic is "
               "not unique there";
    *out = eshkol_rm_sphere_angle(x, y, 1.0, n, NULL);
    return isfinite(*out) ? NULL
                          : "the great-circle distance is not finite";
}

/**
 * @brief <a,b> in double-double: returns the rounded sum and sets *@p lo to the
 *        residual, so that hi + lo is the dot product to about 32 digits.
 *
 * WHY THE BALL CHART NEEDS THIS. Every quantity in Poincare-ball geometry is
 * built from 1 - B|x|^2 and 1 + B<x,y>, and those are exactly the quantities the
 * chart destroys near the boundary: a point at Euclidean distance eps from the
 * boundary has 1 - B|x|^2 ~ 2 eps, and computing it as one minus a rounded sum
 * of squares knows it only to an ABSOLUTE 1e-16 -- eight significant digits at
 * eps = 1e-9, and none at all at eps = 1e-16. Near-boundary points are not an
 * edge case here; they are the entire reason to use the ball, because that is
 * where hyperbolic embeddings put their leaves. Accumulating the sum exactly and
 * closing with a fused multiply-add gives 1 - B|x|^2 to full RELATIVE precision
 * instead, so the formulas downstream inherit a well-conditioned input.
 */
static double eshkol_rm_dot_dd(const double* a, const double* b, int n,
                               double* lo) {
    double hi = 0.0, low = 0.0;
    for (int i = 0; i < n; i++) {
        double p, e, next_hi, next_lo;
        eshkol_rm_two_product(a[i], b[i], &p, &e);
        eshkol_rm_dd_add(hi, low, p, e, &next_hi, &next_lo);
        hi = next_hi;
        low = next_lo;
    }
    *lo = low;
    return hi;
}

/** @brief 1 - B<a,b>, to full relative precision. The fma performs the
 *         subtraction with a single rounding of the exact product, so a result
 *         of size 1e-12 is accurate to 1e-12 * eps rather than to eps. */
static double eshkol_rm_one_minus_dot(const double* a, const double* b, double B,
                                      int n) {
    double lo = 0.0;
    double hi = eshkol_rm_dot_dd(a, b, n, &lo);
    double result = eshkol_rm_fma(-B, lo, eshkol_rm_fma(-B, hi, 1.0));
    return isfinite(result) ? result : 1.0 - eshkol_rm_scaled_dot_factor(a, b, B, n);
}

/** @brief 1 + B<a,b>, to full relative precision. */
static double eshkol_rm_one_plus_dot(const double* a, const double* b, double B,
                                     int n) {
    double lo = 0.0;
    double hi = eshkol_rm_dot_dd(a, b, n, &lo);
    double result = eshkol_rm_fma(B, lo, eshkol_rm_fma(B, hi, 1.0));
    return isfinite(result) ? result : 1.0 + eshkol_rm_scaled_dot_factor(a, b, B, n);
}

/** @brief 1 - (f1 f2)|x|^2 to full relative precision, with the factor f1 f2
 *         carried through the exponent-separated product (never formed as one
 *         rounded double). */
static double eshkol_rm_one_minus_factor_norm2(const double* x, double f1,
                                               double f2, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        double ai = fabs(x[i]);
        if (!isfinite(ai)) return ai;
        if (ai > max_abs) max_abs = ai;
    }
    if (max_abs == 0.0 || f1 == 0.0 || f2 == 0.0) return 1.0;
    int max_exp = 0;
    (void)frexp(max_abs, &max_exp);
    /* scalbn(0.5, max_exp) is finite at both f64 exponent ends. */
    const double scale = scalbn(0.5, max_exp);

    /* sigma_hi + sigma_lo is the normalized sum. Product FMA residuals and
     * TwoSum addition residuals stay in that double-double expansion and are
     * applied once, after multiplication by B*s^2. */
    double sigma_hi = 0.0, sigma_lo = 0.0;
    for (int i = 0; i < n; ++i) {
        const double u = x[i] / scale;
        double term, term_error;
        eshkol_rm_two_product(u, u, &term, &term_error);
        double next_hi = 0.0, next_lo = 0.0;
        eshkol_rm_dd_add(sigma_hi, sigma_lo, term, term_error,
                         &next_hi, &next_lo);
        sigma_hi = next_hi;
        sigma_lo = next_lo;
    }

    double bs2_hi = 0.0, bs2_lo = 0.0;
    eshkol_rm_scaled_product4_dd(f1, scale, scale, f2, &bs2_hi, &bs2_lo);
    /* d_hi is fma(-B*s^2, sigma, 1); the remaining quantity is the single
     * carried error from the normalized sum and the scaled B*s^2 product. */
    const double d_hi = eshkol_rm_fma(-bs2_hi, sigma_hi, 1.0);
    double rounded_product = 0.0, rounded_product_error = 0.0;
    eshkol_rm_two_product(bs2_hi, sigma_hi, &rounded_product,
                          &rounded_product_error);
    double rounded_subtraction = 0.0, subtraction_error = 0.0;
    eshkol_rm_two_sum(1.0, -rounded_product, &rounded_subtraction,
                      &subtraction_error);
    const double high_correction = (rounded_subtraction - d_hi)
        + subtraction_error - rounded_product_error;
    const double residual_product =
        eshkol_rm_scaled_product4(bs2_lo, sigma_hi, 1.0, 1.0)
        + eshkol_rm_scaled_product4(bs2_hi, sigma_lo, 1.0, 1.0)
        + eshkol_rm_scaled_product4(bs2_lo, sigma_lo, 1.0, 1.0);
    const double result = d_hi + high_correction - residual_product;
    if (isfinite(result)) return result;
    return 1.0 - eshkol_rm_scaled_product4(
        f2, 1.0, 1.0, eshkol_rm_scaled_norm2_times(x, f1, n));
}

/** @brief 1 - B|x|^2, the chart's conformal denominator, to full relative
 *         precision. */
static double eshkol_rm_one_minus_bnorm2(const double* x, double B, int n) {
    return eshkol_rm_one_minus_factor_norm2(x, B, 1.0, n);
}

/** @brief The conformal denominator 1 + K alpha^2 |x|^2 / 4 (= 1 - B|x|^2)
 *         from K directly, so it is exact in its parameter for EVERY finite K:
 *         B = -K alpha^2/4 itself loses its low bits below |K| = 2^-1020. */
static double eshkol_rm_chart_den(const double* x, double K, int n) {
    return eshkol_rm_one_minus_factor_norm2(
        x, -K * ESHKOL_RM_ALPHA * ESHKOL_RM_ALPHA, 0.25, n);
}


static const char* eshkol_rm_check_output(const double* out, int n,
                                          const char* reason) {
    for (int i = 0; i < n; i++)
        if (!isfinite(out[i])) return reason;
    return NULL;
}

/**
 * @brief p*a + q*b, componentwise, with the two products formed exactly and
 *        summed with their residuals.
 *
 * The Mobius numerator is p x + q y with p and q BOTH of size 1 - B|x|^2, and
 * for two nearly-antipodal near-boundary points the two terms cancel down to
 * the square of that. Rounding each product to a double first throws the answer
 * away even when p and q are themselves exact: at B = 1,
 * x = 0.999999999, y = -0.999999998 the terms are 2e-9 and the numerator is
 * 3e-18, so a 1e-25 rounding in each product is a 1e-7 relative error in the
 * result. Two-product plus two-sum keeps it.
 */
static void eshkol_rm_axpby_exact(double p, const double* a, double q,
                                  const double* b, int n, double* out) {
    for (int i = 0; i < n; i++) {
        double p1, e1, p2, e2, hi, lo;
        eshkol_rm_two_product(p, a[i], &p1, &e1);
        eshkol_rm_two_product(q, b[i], &p2, &e2);
        eshkol_rm_dd_add(p1, e1, p2, e2, &hi, &lo);
        out[i] = hi + lo;
    }
}

/** @brief The chart parameter B for the curvature K = -c: B = c alpha^2/4.
 *
 * Signed: B > 0 for K < 0 (the ball |x|^2 < 1/B), B = 0 for K = 0, B < 0 for
 * K > 0 (the stereographic sphere chart). Mobius addition is (+)_B for every
 * sign. THE ONE PLACE the gauge enters the chart parameter. */
static double eshkol_rm_ball_param(double c) {
    return eshkol_rm_scaled_product4(c, ESHKOL_RM_ALPHA, ESHKOL_RM_ALPHA, 0.25);
}

/** @brief The conformal factor lambda_x = alpha/(1 - B|x|^2) of the metric of
 *         curvature @p K at @p x, for every sign of K. THE ONE PLACE this is
 *         computed. At K = 0 it is exactly alpha. Infinite or zero when the
 *         denominator is not representable (the pole of a positive chart). */
static double eshkol_rm_lambda(const double* x, double K, int n) {
    if (K == 0.0) return ESHKOL_RM_ALPHA;
    return ESHKOL_RM_ALPHA / eshkol_rm_chart_den(x, K, n);
}

/* ═══ Exact chart parameter at every binade of K ═════════════════════════
 *
 * B = -K alpha^2/4 is an exact double except in the bottom binades of K
 * (|K| below about 2^-1020, where the quotient by 4 drops low bits and
 * K = DBL_TRUE_MIN gives B = 0). There the ops use the scale covariance of
 * the chart,
 *
 *     g_K(x) = s^2 g_{K s^2}(x / s),   s = 2,
 *
 * whose parameter -4K alpha^2/4 = -K alpha^2 is exact. Distances and
 * coordinates scale by s, tangent vectors at the same point likewise, and
 * the conformal denominator is unchanged. The halved coordinates are exact
 * whenever the largest coordinate is at least 2^-1000; below that the whole
 * curvature term B|x|^2 is beneath f64 resolution and the direct evaluation
 * is already exact. */
static int eshkol_rm_needs_rescale(double K, const double* x, const double* y,
                                   int n) {
    if (K == 0.0 || !isfinite(K)) return 0;
    const double B = eshkol_rm_ball_param(-K);
    if (B * 4.0 == -K * ESHKOL_RM_ALPHA * ESHKOL_RM_ALPHA) return 0;
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(x[i]) > m) m = fabs(x[i]);
        if (y && fabs(y[i]) > m) m = fabs(y[i]);
    }
    return m >= 0x1p-1000;
}

/* Halved copies of up to three n-vectors in one allocation (the caller frees). */
static double* eshkol_rm_halved(const double* a, const double* b,
                                const double* c, int n) {
    double* h = (double*)malloc((size_t)(3 * (n > 0 ? n : 1)) * sizeof(double));
    if (!h) return NULL;
    for (int i = 0; i < n; i++) {
        h[i] = 0.5 * a[i];
        h[n + i] = b ? 0.5 * b[i] : 0.0;
        h[2 * n + i] = c ? 0.5 * c[i] : 0.0;
    }
    return h;
}

/** @brief The Riemannian norm of tangent vector @p v at @p x. */
static double eshkol_rm_metric_norm(const double* v, const double* x, double K,
                                    int n) {
    double lam = eshkol_rm_lambda(x, K, n);
    double vn = eshkol_rm_norm(v, n);
    return lam * vn;
}

/* ═══ Generalized trigonometric ratios ════════════════════════════════════
 *
 * Each ratio is an analytic function of ONE dimensionless signed argument and
 * is evaluated by the same rule: the degree-10 Taylor series for
 * |argument| <= ESHKOL_RM_Q_SERIES, the closed form otherwise. A positive
 * argument is the hyperbolic branch, a negative one the spherical branch, and
 * zero is the flat value 1; there is no third code path.
 *
 *   psi(w)        = asinh(sqrt w)/sqrt w,  = asin(sqrt -w)/sqrt -w for w < 0
 *   tan_ratio(s)  = tanh(sqrt s)/sqrt s,   = tan(sqrt -s)/sqrt -s  for s < 0
 *   artan_ratio(s)= artanh(sqrt s)/sqrt s, = atan(sqrt -s)/sqrt -s for s < 0
 */

/** @brief psi(w) = asinh(sqrt w)/sqrt w for w >= 0 and asin(sqrt(-w))/sqrt(-w)
 *         for -1 <= w < 0, with psi'(w) and psi''(w).
 *
 * The geodesic distance is alpha sqrt(E/P) psi(B E/P) on the chart (see
 * eshkol_rm_distance). Both branches share one Taylor series,
 * psi(w) = sum_n a_n w^n, a_n = (-1)^n (2n)!/(4^n (n!)^2 (2n+1)), so K -> 0 from
 * either side and K = 0 itself are one expression. Outside the series window the
 * derivatives use psi' = (g - psi)/(2w), psi'' = (g' - 3 psi')/(2w) with
 * g = (1+w)^(-1/2), which hold on both branches.
 *
 * @param d1 psi'(w), may be NULL.
 * @param d2 psi''(w), may be NULL.
 * @return NaN for w < -1 (beyond the antipode; no chart pair produces it).
 */
static double eshkol_rm_psi(double w, double* d1, double* d2) {
    static const double a[11] = {
        1.0, -1.0 / 6.0, 3.0 / 40.0, -5.0 / 112.0,
        35.0 / 1152.0, -63.0 / 2816.0, 231.0 / 13312.0,
        -143.0 / 10240.0, 6435.0 / 557056.0, -12155.0 / 1245184.0,
        46189.0 / 5505024.0
    };
    if (!(w >= -1.0)) {
        if (d1) *d1 = (double)NAN;
        if (d2) *d2 = (double)NAN;
        return (double)NAN;
    }
    if (fabs(w) <= ESHKOL_RM_Q_SERIES) {
        double psi = a[10];
        double dp  = 10.0 * a[10];
        double dpp = 90.0 * a[10];
        for (int k = 9; k >= 0; k--) psi = psi * w + a[k];
        for (int k = 9; k >= 1; k--) dp  = dp  * w + (double)k * a[k];
        for (int k = 9; k >= 2; k--) dpp = dpp * w + (double)(k * (k - 1)) * a[k];
        if (d1) *d1 = dp;
        if (d2) *d2 = dpp;
        return psi;
    }
    double psi;
    if (w > 0.0) {
        const double u = sqrt(w);
        psi = asinh(u) / u;
    } else {
        const double u = sqrt(-w);
        /* atan2 form: asin(u) = atan2(u, sqrt(1-u^2)), well conditioned up to
         * the antipode u = 1, where it is pi/2. */
        psi = atan2(u, sqrt(1.0 + w)) / u;
    }
    if ((d1 || d2) && fabs(w) <= 0.5) {
        /* psi' and psi'' from the convergent series (64 terms reach f64 at
         * |w| = 1/2): the closed forms below cancel O(log2(1/|w|)) bits each,
         * which is what the curvature jets would inherit. The value branch is
         * unchanged. a_{n+1} = -a_n (2n+1)^2/((2n+2)(2n+3)). */
        double an = 1.0, p1 = 0.0, p2 = 0.0, wk1 = 1.0, wk2 = 1.0;
        for (int k = 1; k <= 64; k++) {
            an *= -((2.0 * k - 1.0) * (2.0 * k - 1.0)) / ((2.0 * k) * (2.0 * k + 1.0));
            p1 += (double)k * an * wk1;
            if (k >= 2) {
                p2 += (double)k * (double)(k - 1) * an * wk2;
                wk2 *= w;
            }
            wk1 *= w;
        }
        if (d1) *d1 = p1;
        if (d2) *d2 = p2;
    } else if (d1 || d2) {
        const double g = 1.0 / sqrt(1.0 + w);
        const double p1 = (g - psi) / (2.0 * w);
        if (d1) *d1 = p1;
        if (d2) {
            const double g1 = -0.5 * g * g * g;
            *d2 = (g1 - 3.0 * p1) / (2.0 * w);
        }
    }
    return psi;
}

/** @brief tan_ratio(s) = tanh(sqrt s)/sqrt s (s >= 0), tan(sqrt -s)/sqrt -s
 *         (s < 0), analytic at 0 with value 1, and its derivative.
 *
 * The exponential map's step factor: exp_x(v) = x (+)_B [(lambda_x/alpha)
 * tan_ratio(s) v] with s = B (lambda_x |v| / alpha)^2. On the positive branch
 * the ratio has a pole at sqrt(-s) = pi/2: the geodesic reaches the pole of the
 * chart there, and the caller reports it.
 *
 * @param d1 tan_ratio'(s), may be NULL. Off the series window it is
 *           (1 - s T^2 - T)/(2s) on both branches.
 */
static double eshkol_rm_tan_ratio(double s, double* d1) {
    /* tan(z)/z = sum tau_k z^(2k); with s = -z^2 on the spherical branch and
     * s = z^2 (sign flipped termwise) on the hyperbolic one, both are
     * sum tau_k (-s)^k. */
    static const double tau[11] = {
        1.0, 1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0,
        1382.0 / 155925.0, 21844.0 / 6081075.0, 929569.0 / 638512875.0,
        6404582.0 / 10854718875.0, 443861162.0 / 1856156927625.0,
        18888466084.0 / 194896477400625.0
    };
    if (fabs(s) <= ESHKOL_RM_Q_SERIES) {
        const double m = -s;
        double t = tau[10];
        double dt = 10.0 * tau[10];
        for (int k = 9; k >= 0; k--) t = t * m + tau[k];
        for (int k = 9; k >= 1; k--) dt = dt * m + (double)k * tau[k];
        if (d1) *d1 = -dt;
        return t;
    }
    double T;
    if (s > 0.0) {
        const double z = sqrt(s);
        T = tanh(z) / z;
    } else {
        const double z = sqrt(-s);
        T = tan(z) / z;
    }
    if (d1) *d1 = (1.0 - s * T * T - T) / (2.0 * s);
    return T;
}

/** @brief artan_ratio(s) = artanh(sqrt s)/sqrt s (0 <= s < 1),
 *         atan(sqrt -s)/sqrt -s (s < 0); analytic at 0 with value 1.
 *         NaN for s >= 1 (on or beyond the ball boundary). */
static double eshkol_rm_artan_ratio(double s) {
    if (fabs(s) <= ESHKOL_RM_Q_SERIES) {
        double r = 1.0 / 21.0;
        for (int k = 9; k >= 0; k--) r = r * s + 1.0 / (double)(2 * k + 1);
        return r;
    }
    if (s > 0.0) {
        if (!(s < 1.0)) return (double)NAN;
        const double u = sqrt(s);
        return atanh(u) / u;
    }
    const double u = sqrt(-s);
    return atan(u) / u;
}

/**
 * @brief The Mobius denominator 1 + 2B<x,y> + B^2 |x|^2 |y|^2, evaluated as
 *
 *     (1 + B<x,y>)^2 + B^2 (|x|^2|y|^2 - <x,y>^2)
 *
 * which is the same number and is accurate where the direct sum is not.
 *
 * WHY. For two interior points the denominator is bounded below by
 * (1 - sqrt(B)|x| sqrt(B)|y|)^2 > 0, so it never vanishes -- but it can be
 * ARBITRARILY SMALL, and the direct sum computes it as a difference of terms of
 * size 1. At B = 1, x = 0.999999999, y = -0.999999998 the true denominator is
 * about 9e-18 and every digit of it is lost; the old code then floored the
 * result at 1e-15 without scaling the numerator, turning an exact quotient of
 * 1/3 into 0.003. The grouping above is exact instead: 1 + B<x,y> is a
 * subtraction of two nearby numbers, hence exact in f64, and the Gram term is
 * non-negative and computed without cancellation.
 */
static double eshkol_rm_mobius_den(const double* x, const double* y, double B,
                                   int n) {
    double q  = eshkol_rm_one_plus_dot(x, y, B, n);
    /* B^2 Gram: sign-free, so |B| carries the scale for both chart signs. */
    double root_B = eshkol_rm_sqrt_nonnegative(fabs(B));
    double gram = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double t = (root_B * x[i]) * (root_B * y[j]) -
                       (root_B * x[j]) * (root_B * y[i]);
            gram += t * t;
        }
    return q * q + gram;
}

/** @brief The Mobius denominator of the pair (-x, y), i.e.
 *         (1 - B<x,y>)^2 + B^2(|x|^2|y|^2 - <x,y>^2), without materialising -x.
 *         Negating x flips the sign of <x,y> and leaves the Gram term alone. */
static double eshkol_rm_mobius_den_negx(const double* x, const double* y,
                                        double B, int n) {
    double q  = eshkol_rm_one_minus_dot(x, y, B, n);
    /* B^2 Gram: sign-free, so |B| carries the scale for both chart signs. */
    double root_B = eshkol_rm_sqrt_nonnegative(fabs(B));
    double gram = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double t = (root_B * x[i]) * (root_B * y[j]) -
                       (root_B * x[j]) * (root_B * y[i]);
            gram += t * t;
        }
    return q * q + gram;
}

/**
 * @brief Mobius addition on the ball of parameter @p B > 0:
 *
 *   x (+)_B y = ((1 + 2B<x,y> + B|y|^2) x + (1 - B|x|^2) y)
 *               / (1 + 2B<x,y> + B^2 |x|^2 |y|^2)
 *
 * At B = 0 this is x + y, which is why the Euclidean branch of every caller is
 * the same code with B = 0 rather than a separate special case.
 *
 * THE DENOMINATOR IS NOT FLOORED. For two points strictly inside the ball,
 * writing p = sqrt(B)|x| < 1 and q = sqrt(B)|y| < 1, the denominator is
 * 1 + 2pq cos(theta) + p^2 q^2 >= (1 - pq)^2 > 0, so it cannot vanish and there
 * is nothing to guard. It used to be floored at 1e-15 without scaling the
 * numerator, which for x = 0.999999999, y = -0.999999998 at B = 1 turned an
 * exact quotient of 1/3 into 0.003: a valid interior sum, off by two orders of
 * magnitude, with no diagnostic. Callers that could hand this arbitrary vectors
 * -- parallel transport was the only one -- no longer do; transport uses the
 * linear gyration form below.
 */
static void eshkol_rm_mobius_add(const double* x, const double* y, double B,
                                 int n, double* out) {
    double den   = eshkol_rm_mobius_den(x, y, B, n);
    /* num_y is the chart's conformal denominator at x, computed to full
     * relative precision; num_x then follows from the identity
     * num_x = den + B|y|^2 num_y, which is exact algebra and keeps num_x from
     * being formed by its own cancelling sum 1 + 2B<x,y> + B|y|^2. */
    double num_y = eshkol_rm_one_minus_bnorm2(x, B, n);
    double By2 = eshkol_rm_scaled_norm2_times(y, B, n);
    double num_x = den + By2 * num_y;
    eshkol_rm_axpby_exact(num_x, x, num_y, y, n, out);
    for (int i = 0; i < n; i++) out[i] /= den;
}

/**
 * @brief Mobius addition on the chart of curvature @p K (every sign), with
 *        the chart's domain checks: x (+)_B y, B = -K alpha^2/4. A sum that is
 *        the pole of the positive chart, or that leaves the ball, is refused.
 */
static const char* eshkol_rm_check_point(const double* x, double K, int n);
static const char* eshkol_rm_require_interior(const double* out, double K, int n);
static const char* eshkol_rm_check_output(const double* out, int n,
                                          const char* reason);
static const char* eshkol_rm_chart_mobius_add(const double* x, const double* y,
                                              double K, int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (!why) why = eshkol_rm_check_point(y, K, n);
    if (why) return why;
    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_chart_mobius_add(h, h + n, 4.0 * K, n, out);
        free(h);
        if (!why) for (int i = 0; i < n; i++) out[i] *= 2.0;
        return why;
    }
    eshkol_rm_mobius_add(x, y, eshkol_rm_ball_param(-K), n, out);
    why = eshkol_rm_check_output(out, n, "the Mobius sum is the pole of the chart");
    if (!why) why = eshkol_rm_require_interior(out, K, n);
    return why;
}

/**
 * @brief The gyration gyr[a,b]w on the ball of parameter @p B, in CLOSED LINEAR
 *        FORM:
 *
 *   D = 1 + 2B<a,b> + B^2 |a|^2 |b|^2
 *   A = -B^2 <a,w> |b|^2 + B <b,w> + 2 B^2 <a,b><b,w>
 *   C = -B^2 <b,w> |a|^2 - B <a,w>
 *   gyr[a,b]w = w + 2 (A a + C b) / D
 *
 * WHY NOT THE MOBIUS COMPOSITION. gyr[a,b]w is also
 * -(a (+) b) (+) (a (+) (b (+) w)), and that is how transport used to compute
 * it -- by feeding the TANGENT VECTOR w into Mobius POINT addition. A tangent
 * vector is not constrained to the ball, so an intermediate denominator can
 * vanish even though the gyration is a globally defined linear isometry.
 * Counterexample at B = 1, x = y = (0.5), v = (2): every gyration in one
 * dimension is the identity, so transport from a point to itself must return
 * (2); the intermediate (-0.5) (+) 2 has denominator zero, the old floor turned
 * its exact 0/0 cancellation into zero, and the routine returned (0.5).
 *
 * The form above is linear in w and its only denominator D is the Mobius
 * denominator of the two POINTS a and b, which are validated ball points, so it
 * is bounded below by (1 - sqrt(B)|a| sqrt(B)|b|)^2 > 0.
 */
static void eshkol_rm_gyration(const double* a, const double* b, const double* w,
                               double B, int n, double* out) {
    double ab = eshkol_rm_scaled_dot_factor(a, b, B, n);
    double a2 = eshkol_rm_scaled_norm2_times(a, B, n);
    double b2 = eshkol_rm_scaled_norm2_times(b, B, n);
    double aw = eshkol_rm_scaled_dot_factor(a, w, B, n);
    double bw = eshkol_rm_scaled_dot_factor(b, w, B, n);
    double D  = eshkol_rm_mobius_den(a, b, B, n);
    double A  = -aw * b2 + bw + 2.0 * ab * bw;
    double C  = -bw * a2 - aw;
    for (int i = 0; i < n; i++)
        out[i] = w[i] + 2.0 * (A * a[i] + C * b[i]) / D;
}

/**
 * @brief Check that @p x is a legal point of the chart of curvature @p K.
 *
 * K < 0: strictly inside the ball of radius 2/sqrt(-K) (alpha = 1).
 * K = 0: any finite point.
 * K > 0: any finite point whose conformal denominator 1 + K|x|^2/4 is
 *        representable; a point where it overflows is numerically the POLE of
 *        the chart, which is not a point of the chart.
 * @return NULL when it is, else a reason naming what is wrong.
 */
static const char* eshkol_rm_check_point(const double* x, double K, int n) {
    if (!isfinite(K)) return "curvature is not finite";
    for (int i = 0; i < n; i++)
        if (!isfinite(x[i])) return "a coordinate is not finite";
    if (K == 0.0) return NULL;
    double den = eshkol_rm_chart_den(x, K, n);
    if (K < 0.0) {
        if (!(den > 0.0))
            return "the point must lie strictly inside the ball of radius "
                   "2/sqrt(-K)";
    } else if (!isfinite(den)) {
        return "the point is at the pole of the positive-curvature chart "
               "(1 + K|x|^2/4 is not representable)";
    }
    return NULL;
}

/**
 * @brief Check that a computed chart point is a point of the chart.
 *
 * tanh rounds to exactly 1.0 for arguments beyond about 19, so the exponential
 * map and Mobius scalar multiplication can land on the ball boundary, which is
 * not a point of the open ball; on the positive chart a geodesic that reaches
 * the pole produces a non-representable conformal denominator. Refusing names
 * the representability limit.
 *
 * @return NULL when the point is in the chart, else a reason.
 */
static const char* eshkol_rm_require_interior(const double* out, double K, int n) {
    if (K == 0.0) return NULL;
    double den = eshkol_rm_chart_den(out, K, n);
    if (K < 0.0 && !(den > 0.0))
        return "the result is on or outside the ball boundary: no strictly "
               "interior double-precision point exists at this magnitude";
    if (K > 0.0 && !isfinite(den))
        return "the result is at the pole of the positive-curvature chart";
    return NULL;
}

/**
 * @brief Check a tangent vector of the chart: every finite vector is tangent.
 * @return NULL when every component is finite, else a reason.
 */
static const char* eshkol_rm_check_tangent(const double* x, const double* v,
                                           double K, int n) {
    (void)x;
    (void)K;
    for (int i = 0; i < n; i++)
        if (!isfinite(v[i])) return "a tangent-vector component is not finite";
    return NULL;
}

/**
 * @brief Check that @p x lies ON the embedded sphere of radius 1/sqrt(K) in
 *        ambient coordinates (the model behind the `eshkol_rm_sphere_*` ops).
 * @return NULL when it does, else a reason.
 */
static const char* eshkol_rm_sphere_check_point(const double* x, double K, int n) {
    if (!isfinite(K)) return "curvature is not finite";
    if (!(K > 0.0)) return "the embedded sphere requires positive curvature";
    for (int i = 0; i < n; i++)
        if (!isfinite(x[i])) return "a coordinate is not finite";
    double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    if (!isfinite(R)) return "the sphere radius is not finite";
    double xn = eshkol_rm_norm(x, n);
    if (!(fabs(xn - R) <= ESHKOL_RM_SPHERE_TOL * R))
        return "the point must lie ON the sphere of radius 1/sqrt(K) "
               "(use spherical-project to move it there)";
    return NULL;
}

/**
 * @brief Check that @p v is tangent to the embedded sphere of curvature @p K
 *        at @p x: <x,v> must vanish, tested relative to |x| and |v|.
 * @return NULL when tangent, else a reason.
 */
static const char* eshkol_rm_sphere_check_tangent(const double* x,
                                                  const double* v,
                                                  double K, int n) {
    (void)K;
    double vn = eshkol_rm_norm(v, n);
    if (vn == 0.0) return NULL;
    double xm = 0.0, vm = 0.0, x2 = 0.0, v2 = 0.0, dot = 0.0;
    for (int i = 0; i < n; i++) {
        double ax = fabs(x[i]), av = fabs(v[i]);
        if (!isfinite(ax) || !isfinite(av))
            return "a tangent-vector component is not finite";
        if (ax > xm) xm = ax;
        if (av > vm) vm = av;
    }
    if (xm == 0.0 || vm == 0.0) return NULL;
    for (int i = 0; i < n; i++) {
        double nx = x[i] / xm, nv = v[i] / vm;
        dot += nx * nv;
        x2 += nx * nx;
        v2 += nv * nv;
    }
    /* Compare normalized quantities. The unscaled dot can underflow at the
     * same exponent boundary where a wholly radial vector must be rejected. */
    if (!(fabs(dot) <= ESHKOL_RM_TANGENT_TOL * sqrt(x2) * sqrt(v2)))
        return "the vector is not tangent to the sphere at this point (<x,v> "
               "must vanish)";
    return NULL;
}

/* Multiply a finite value by B^(-p/2), p in {1,3,5}, without first forming
 * B^(p/2) or its reciprocal.  The mantissa/exponent pair is essential when
 * B is a large negative-curvature magnitude: B^(-5/2) is subnormal or
 * unrepresentable as a standalone double even when its product with the
 * dimensionless curvature derivative is finite. */
static double eshkol_rm_inverse_power_mul(double B, int p, double value) {
    if (value == 0.0) return 0.0;
    if (!(B > 0.0) || !isfinite(B) || !isfinite(value)) return NAN;
    int exponent = 0, value_exponent = 0;
    double mantissa = frexp(B, &exponent);
    if (exponent & 1) {
        mantissa *= 2.0;
        --exponent;
    }
    const double power_mantissa = pow(mantissa, -0.5 * (double)p);
    const double value_mantissa = frexp(value, &value_exponent);
    return scalbn(power_mantissa * value_mantissa,
                  -(p * exponent) / 2 + value_exponent);
}

/**
 * @brief Geodesic distance between @p x and @p y on the chart of curvature
 *        @p K, into *@p out.
 *
 * With E = |x-y|^2, P = (1 - B|x|^2)(1 - B|y|^2) and the signed dimensionless
 * w = B E/P (w = -q, q = K E/(4P)):
 *
 *     d = alpha sqrt(E/P) psi(w)
 *
 * for every sign: asinh on the ball, asin on the sphere chart, the series for
 * |w| <= ESHKOL_RM_Q_SERIES, and exactly alpha |x-y| at K = 0. Outside the
 * series window the positive chart is evaluated as
 *
 *     d = alpha atan2(sqrt(|B| E), sqrt(Dn)) / sqrt(|B|),
 *     Dn = (1 - B<x,y>)^2 + B^2 (|x|^2|y|^2 - <x,y>^2) = P (1 + w),
 *
 * which is well conditioned up to the antipode, where Dn = 0 and the distance
 * is pi/sqrt(K). The asinh form replaces arccosh(1 + 2B E/P) on the ball, which
 * loses every digit of a small separation.
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_distance(const double* x, const double* y, double K,
                                      int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;

    if (eshkol_rm_points_equal(x, y, n)) {
        *out = 0.0;
        return NULL;
    }

    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_distance(h, h + n, 4.0 * K, n, out);
        free(h);
        if (!why) *out *= 2.0;
        return why;
    }
    double delta = 0.0;
    for (int i = 0; i < n; i++) delta = hypot(delta, x[i] - y[i]);
    if (K == 0.0) {
        *out = ESHKOL_RM_ALPHA * delta;
        return isfinite(*out) ? NULL : "the geodesic distance is not finite";
    }
    const double B = eshkol_rm_ball_param(-K);
    const double pa = eshkol_rm_one_minus_bnorm2(x, B, n);
    const double pb = eshkol_rm_one_minus_bnorm2(y, B, n);
    const double scaled_delta = (delta / sqrt(pa)) / sqrt(pb);
    const double w = eshkol_rm_scaled_square(B, scaled_delta);
    if (K < 0.0 || fabs(w) <= ESHKOL_RM_Q_SERIES) {
        double ps = eshkol_rm_psi(w, NULL, NULL);
        *out = ESHKOL_RM_ALPHA * scaled_delta * ps;
    } else {
        const double abs_B = fabs(B);
        const double bE = eshkol_rm_difference_norm2_times(x, y, abs_B, n);
        const double dn = eshkol_rm_mobius_den_negx(x, y, B, n);
        if (!(dn > 0.0))
            return "the two points are antipodal (the cut locus of the "
                   "positive-curvature chart): distance is not differentiable";
        const double h = atan2(sqrt(bE), sqrt(dn));
        *out = ESHKOL_RM_ALPHA * eshkol_rm_inverse_power_mul(abs_B, 1, h);
    }
    if (!isfinite(*out)) return "the geodesic distance is not finite";
    return NULL;
}

/**
 * @brief Squared geodesic distance d(x,y)^2 on the chart of curvature @p K.
 *
 * The value only. Its gradient is never formed from d (see
 * eshkol_rm_sq_distance_grad): d^2 is smooth at coincidence while d is not.
 */
static const char* eshkol_rm_sq_distance(const double* x, const double* y,
                                         double K, int n, double* out) {
    double d = 0.0;
    const char* why = eshkol_rm_distance(x, y, K, n, &d);
    if (why) return why;
    *out = d * d;
    return isfinite(*out) ? NULL : "the squared geodesic distance is not finite";
}

/**
 * @brief Exponential map exp_x(v) on the chart of curvature @p K.
 *
 *     exp_x(v) = x (+)_B [ (lambda_x/alpha) tan_ratio(s) v ],
 *     s = B (lambda_x |v| / alpha)^2,
 *
 * for every sign: tanh on the ball, tan on the sphere chart, the series near
 * s = 0, and x + v at K = 0. On the positive chart a geodesic that reaches the
 * pole (sqrt(-s) = pi/2) is refused; past it, the chart formula continues onto
 * the far hemisphere.
 *
 * @param scratch n doubles.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_exp_map(const double* x, const double* v, double K,
                                     int n, double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_tangent(x, v, K, n);
    if (why) return why;

    double vn = eshkol_rm_norm(v, n);
    if (vn == 0.0) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = x[i] + v[i];
        return eshkol_rm_check_output(out, n, "the exponential-map result is not finite");
    }
    if (eshkol_rm_needs_rescale(K, x, v, n)) {
        double* h = eshkol_rm_halved(x, v, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_exp_map(h, h + n, 4.0 * K, n, out, scratch);
        free(h);
        if (!why) for (int i = 0; i < n; i++) out[i] *= 2.0;
        return why;
    }
    double B   = eshkol_rm_ball_param(-K);
    double lam = eshkol_rm_lambda(x, K, n);
    double s   = eshkol_rm_scaled_square(B, lam * vn / ESHKOL_RM_ALPHA);
    double f   = (lam / ESHKOL_RM_ALPHA) * eshkol_rm_tan_ratio(s, NULL);
    if (!isfinite(f))
        return "the geodesic reaches the pole of the positive-curvature chart";
    for (int i = 0; i < n; i++) scratch[i] = f * v[i];
    eshkol_rm_mobius_add(x, scratch, B, n, out);
    why = eshkol_rm_check_output(out, n, "the exponential-map result is not finite");
    if (why) return why;
    return eshkol_rm_require_interior(out, K, n);
}

/**
 * @brief The unit direction @p u of log_x(y) on the chart of curvature
 *        @p K != 0, and the distance *@p d_out.
 *
 * The direction is the Mobius numerator of (-x) (+)_B y written in the
 * separation delta = y - x:
 *
 *     V = (1 - B|x|^2) delta - (B |delta|^2) x,
 *
 * which equals (1 - B|x|^2) y - (Dn + B|y|^2 (1 - B|x|^2)) x term for term
 * (their x-coefficients differ by exactly B|y - x|^2) but has no O(1)
 * cancellation as y -> x: a one-ulp separation keeps its direction. The
 * coefficient pair is normalised before use so V never overflows, and the
 * products are summed with their rounding residuals. Coincidence and the
 * positive-chart antipode are refused here (callers handle coincidence).
 *
 * @param scratch n doubles.
 */
static const char* eshkol_rm_log_direction(const double* x, const double* y,
                                           double K, int n, double* u,
                                           double* scratch, double* d_out) {
    const double B = eshkol_rm_ball_param(-K);
    if (B < 0.0 && !(eshkol_rm_mobius_den_negx(x, y, B, n) > 0.0))
        return "the two points are antipodal (the cut locus of the "
               "positive-curvature chart): log is not single-valued there";
    const char* why = eshkol_rm_distance(x, y, K, n, d_out);
    if (why) return why;
    double p = eshkol_rm_one_minus_bnorm2(x, B, n);
    double q = -eshkol_rm_difference_norm2_times(y, x, B, n);
    if (!isfinite(p) || !isfinite(q)) {
        /* Both coefficients scaled by 2^-128 in the exponent-separated
         * products (only the direction of V matters). When either overflowed
         * B|x|^2 or B|y-x|^2 exceeds 2^1024, so the 1 in 1 - B|x|^2 is far
         * below one ulp of the scaled value. */
        p = (eshkol_rm_one_minus_factor_norm2(x, B, 0x1p-128, n) - 1.0) + 0x1p-128;
        q = -eshkol_rm_difference_norm2_times2(y, x, B, 0x1p-128, n);
    }
    const double m = fmax(fabs(p), fabs(q));
    if (!(m > 0.0) || !isfinite(m))
        return "the logarithmic-map direction is not finite";
    p /= m;
    q /= m;
    double* V = scratch;
    for (int i = 0; i < n; i++) {
        V[i] = y[i] - x[i];
        if (!isfinite(V[i])) return "the point separation is not finite";
    }
    eshkol_rm_axpby_exact(p, V, q, x, n, V);
    const double vn = eshkol_rm_norm(V, n);
    if (!(vn > 0.0) || !isfinite(vn))
        return "the logarithmic-map direction is not defined for this pair";
    for (int i = 0; i < n; i++) u[i] = V[i] / vn;
    return NULL;
}

/**
 * @brief Logarithmic map log_x(y) on the chart of curvature @p K, the inverse
 *        of eshkol_rm_exp_map().
 *
 * The DIRECTION comes from the Mobius numerator of (-x) (+)_B y and the
 * MAGNITUDE from the geodesic distance, |log_x(y)| = d(x,y)/lambda_x:
 *
 *   - the direction is taken from the numerator vector alone, never divided by
 *     the Mobius denominator, which is the quantity that goes to zero for two
 *     nearly-antipodal points;
 *   - the magnitude is d/lambda_x from eshkol_rm_distance's stable form, never
 *     (2/(sqrt(B) lambda_x)) artanh(sqrt(B)|u|), which loses the answer when
 *     sqrt(B)|u| rounds to 1 for a valid interior pair.
 *
 * On the positive chart the antipode (Mobius denominator zero) is the cut
 * locus and is refused.
 *
 * @param scratch n doubles.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_log_map(const double* x, const double* y, double K,
                                     int n, double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;

    if (eshkol_rm_points_equal(x, y, n)) {
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return NULL;
    }

    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = y[i] - x[i];
        return eshkol_rm_check_output(out, n, "the logarithmic-map result is not finite");
    }
    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_log_map(h, h + n, 4.0 * K, n, out, scratch);
        free(h);
        if (!why) for (int i = 0; i < n; i++) out[i] *= 2.0;
        return why;
    }
    double d = 0.0, scale = 0.0;
    why = eshkol_rm_log_direction(x, y, K, n, out, scratch, &d);
    if (why) return why;
    /* |log_x(y)| = d / lambda_x = d (1 - B|x|^2) / alpha */
    scale = (d * eshkol_rm_chart_den(x, K, n)) / ESHKOL_RM_ALPHA;
    for (int i = 0; i < n; i++) out[i] *= scale;
    return eshkol_rm_check_output(out, n, "the logarithmic-map result is not "
                                  "representable in f64");
}

/**
 * @brief Coordinate gradients of the SQUARED distance:
 *        @p gx = d(d^2)/dx = -2 lambda_x^2 log_x(y), @p gy likewise.
 *
 * The Riemannian gradient of d^2 is -2 log_x(y); the coordinate gradient is
 * that vector pushed through the metric, lambda_x^2. It is evaluated in the
 * log-map form and never through d: at x == y bitwise the log map is exactly
 * zero, so the gradient is exactly zero with no epsilon guard. Either output
 * may be NULL.
 *
 * @param scratch 2n doubles.
 */
static const char* eshkol_rm_sq_distance_grad(const double* x, const double* y,
                                              double K, int n, double* gx,
                                              double* gy, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (!why) why = eshkol_rm_check_point(y, K, n);
    if (why) return why;
    if (eshkol_rm_points_equal(x, y, n)) {
        /* log_x(x) = 0: the gradient is exactly zero, no epsilon involved */
        for (int i = 0; i < n; i++) {
            if (gx) gx[i] = 0.0;
            if (gy) gy[i] = 0.0;
        }
        return NULL;
    }
    if (K == 0.0) {
        for (int i = 0; i < n; i++) {
            const double t = 2.0 * (x[i] - y[i]);
            if (gx) gx[i] = t;
            if (gy) gy[i] = -t;
        }
        if (gx && (why = eshkol_rm_check_output(gx, n, "the squared-distance gradient is not finite"))) return why;
        if (gy && (why = eshkol_rm_check_output(gy, n, "the squared-distance gradient is not finite"))) return why;
        return NULL;
    }
    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        /* d^2(K; x, y) = 4 d^2(4K; x/2, y/2): the gradient scales by 2. */
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_sq_distance_grad(h, h + n, 4.0 * K, n, gx, gy, scratch);
        free(h);
        if (why) return why;
        for (int i = 0; i < n; i++) {
            if (gx) gx[i] *= 2.0;
            if (gy) gy[i] *= 2.0;
        }
        return NULL;
    }
    double* u = scratch + n;
    for (int side = 0; side < 2; side++) {
        double* g = side ? gy : gx;
        if (!g) continue;
        const double* a = side ? y : x;
        const double* b = side ? x : y;
        double d = 0.0;
        why = eshkol_rm_log_direction(a, b, K, n, u, scratch, &d);
        if (why) return why;
        /* -2 lambda_a^2 log_a(b) = -2 lambda_a d u: lambda d is formed before
         * the unit vector, never the possibly unrepresentable |log|. */
        const double f = -2.0 * (ESHKOL_RM_ALPHA / eshkol_rm_chart_den(a, K, n)) * d;
        for (int i = 0; i < n; i++) g[i] = f * u[i];
        why = eshkol_rm_check_output(g, n, "the squared-distance gradient is not finite");
        if (why) return why;
    }
    return NULL;
}

/**
 * @brief Parallel transport of tangent vector @p v from @p x to @p y along the
 *        connecting geodesic of the chart of curvature @p K.
 *
 *     P_{x->y}(v) = (lambda_x / lambda_y) gyr[y, -x] v,
 *
 * with the gyration in its LINEAR closed form (eshkol_rm_gyration), whose only
 * denominator is the Mobius denominator of the two points. On the positive
 * chart that denominator vanishes at the antipode, where the geodesic and hence
 * the transport is not unique, and the op refuses.
 *
 * @param scratch n doubles.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_transport(const double* x, const double* y,
                                       const double* v, double K, int n,
                                       double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;
    for (int i = 0; i < n; i++)
        if (!(v[i] == v[i])) return "a tangent-vector component is NaN";
    why = eshkol_rm_check_tangent(x, v, K, n);
    if (why) return why;

    if (K == 0.0) {
        memcpy(out, v, (size_t)n * sizeof(double));
        return NULL;
    }
    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        /* transport is linear in v: only the points are rescaled */
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_transport(h, h + n, v, 4.0 * K, n, out, scratch);
        free(h);
        return why;
    }
    double B = eshkol_rm_ball_param(-K);
    if (!(eshkol_rm_mobius_den_negx(x, y, B, n) > 0.0))
        return "the endpoints are antipodal (the cut locus of the "
               "positive-curvature chart): parallel transport between them is "
               "not unique";
    double* negx = scratch;
    for (int i = 0; i < n; i++) negx[i] = -x[i];
    eshkol_rm_gyration(y, negx, v, B, n, out);
    double ratio = eshkol_rm_lambda(x, K, n) / eshkol_rm_lambda(y, K, n);
    for (int i = 0; i < n; i++) out[i] *= ratio;
    return eshkol_rm_check_output(out, n, "the parallel-transport result is not finite");
}

/**
 * @brief Mobius scalar multiplication r (x)_B x = tan_B(r artan_B(|x|)) x/|x|
 *        on the chart of curvature @p K, every sign:
 *
 *     a   = r |x| artan_ratio(B|x|^2)           (r artan_B |x|, length units)
 *     out = r artan_ratio(B|x|^2) tan_ratio(B a^2) x
 *
 * r x at K = 0. Refuses when the result is not a point of the chart (the ball
 * boundary for K < 0, the pole for K > 0).
 */
static const char* eshkol_rm_mobius_scalar(double r, const double* x, double K,
                                           int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    if (!isfinite(r)) return "the Mobius scalar is not finite";
    double xn = eshkol_rm_norm(x, n);
    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = r * x[i];
        return eshkol_rm_check_output(out, n, "the Mobius scalar result is not finite");
    }
    if (xn == 0.0) {
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return NULL;
    }
    if (eshkol_rm_needs_rescale(K, x, NULL, n)) {
        double* h = eshkol_rm_halved(x, NULL, NULL, n);
        if (!h) return "out of memory";
        why = eshkol_rm_mobius_scalar(r, h, 4.0 * K, n, out);
        free(h);
        if (!why) for (int i = 0; i < n; i++) out[i] *= 2.0;
        return why;
    }
    double B = eshkol_rm_ball_param(-K);
    double A = eshkol_rm_artan_ratio(eshkol_rm_scaled_norm2_times(x, B, n));
    double a = r * xn * A;
    double coef = r * A * eshkol_rm_tan_ratio(eshkol_rm_scaled_square(B, a), NULL);
    for (int i = 0; i < n; i++) out[i] = coef * x[i];
    why = eshkol_rm_check_output(out, n, "the Mobius scalar result is not finite");
    if (why) return why;
    return eshkol_rm_require_interior(out, K, n);
}

/**
 * @brief Project @p x onto the chart of curvature @p K.
 *
 * K < 0: rescale onto the open ball when @p x is on or outside it, leaving
 * interior points untouched. K >= 0: every finite point is a chart point, so a
 * copy (a positive-chart point at the pole is refused).
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_project(const double* x, double K, int n, double* out) {
    if (!isfinite(K)) return "curvature is not finite";
    for (int i = 0; i < n; i++)
        if (!isfinite(x[i])) return "a coordinate is not finite";
    if (K >= 0.0) {
        const char* why = eshkol_rm_check_point(x, K, n);
        if (why) return why;
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    double xn = eshkol_rm_norm(x, n);
    /* radius 2/(alpha sqrt(-K)), formed from K so it is exact-parameter at
     * every binade */
    double R = 2.0 / (ESHKOL_RM_ALPHA * eshkol_rm_sqrt_nonnegative(-K));
    /* Leave a margin so the projected point is STRICTLY inside: a point exactly
     * on the boundary makes lambda infinite and every log map degenerate to
     * zero, which is the failure mode frechet_mean_core.h refuses to let look
     * like convergence. */
    double limit = R * (1.0 - 1e-12);
    if (xn <= limit) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    double s = limit / xn;
    for (int i = 0; i < n; i++) out[i] = s * x[i];
    return NULL;
}

/**
 * @brief Convert a Euclidean (coordinate) gradient @p g at @p x into the
 *        Riemannian gradient g / lambda_x^2, every sign of K. At K = 0 it is
 *        g / alpha^2 = g.
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_egrad_to_rgrad(const double* g, const double* x,
                                            double K, int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    double lam = eshkol_rm_lambda(x, K, n);
    if (!isfinite(lam) || !(lam > 0.0)) return "the conformal factor is not finite";
    double inv_lam = 1.0 / lam;
    double s = inv_lam * inv_lam;
    for (int i = 0; i < n; i++) out[i] = s * g[i];
    return eshkol_rm_check_output(out, n, "the Riemannian gradient is not finite");
}

/* ═══ The embedded sphere model (ambient coordinates) ═════════════════════
 *
 * Points of R^{n+1} on the sphere of radius R = 1/sqrt(K). These back the ops
 * whose documented contract is the embedding (`great-circle-distance`,
 * `slerp`, `spherical-exp/log/project`, the space-form SPHERICAL factor); every
 * curvature-parameterized op uses the chart above. */

/** @brief Geodesic distance R theta on the embedded sphere, theta from the
 *         cancellation-free atan2 form; the antipode is refused. */
static const char* eshkol_rm_sphere_distance(const double* x, const double* y,
                                             double K, int n, double* out) {
    const char* why = eshkol_rm_sphere_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_sphere_check_point(y, K, n);
    if (why) return why;
    if (eshkol_rm_points_equal(x, y, n)) { *out = 0.0; return NULL; }
    double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    if (eshkol_rm_sphere_antipodal(x, y, n))
        return "the two points are antipodal: distance is not differentiable";
    *out = R * eshkol_rm_sphere_angle(x, y, R, n, NULL);
    return isfinite(*out) ? NULL : "the geodesic distance is not finite";
}

/** @brief exp_x(v) = cos(|v|/R) x + R sin(|v|/R) v/|v| on the embedded sphere,
 *         with the tangency of v checked rather than assumed. */
static const char* eshkol_rm_sphere_exp_map(const double* x, const double* v,
                                            double K, int n, double* out) {
    const char* why = eshkol_rm_sphere_check_point(x, K, n);
    if (why) return why;
    for (int i = 0; i < n; i++)
        if (!isfinite(v[i])) return "a tangent-vector component is not finite";
    why = eshkol_rm_sphere_check_tangent(x, v, K, n);
    if (why) return why;
    double vn = eshkol_rm_norm(v, n);
    if (vn == 0.0) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    double R  = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    double th = vn / R;
    double ca = cos(th), sa = sin(th);
    for (int i = 0; i < n; i++) out[i] = ca * x[i] + R * sa * v[i] / vn;
    return eshkol_rm_check_output(out, n, "the exponential-map result is not finite");
}

/** @brief log_x(y) on the embedded sphere; the antipode is refused. */
static const char* eshkol_rm_sphere_log_map(const double* x, const double* y,
                                            double K, int n, double* out) {
    const char* why = eshkol_rm_sphere_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_sphere_check_point(y, K, n);
    if (why) return why;
    if (eshkol_rm_points_equal(x, y, n)) {
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return NULL;
    }
    double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    if (!isfinite(R)) return "the sphere radius is not finite";
    if (eshkol_rm_sphere_antipodal(x, y, n))
        return "the two points are antipodal: log is not single-valued there";
    why = eshkol_rm_sphere_log_map_stable(x, y, R, n, out);
    if (why) return why;
    return eshkol_rm_check_output(out, n, "the logarithmic-map result is not finite");
}

/** @brief Rescale @p x onto the embedded sphere of radius 1/sqrt(K); the
 *         origin has no projection and is refused. */
static const char* eshkol_rm_sphere_project(const double* x, double K, int n,
                                            double* out) {
    if (!isfinite(K) || !(K > 0.0)) return "the embedded sphere requires positive curvature";
    double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) return "a coordinate is not finite";
        if (fabs(x[i]) > max_abs) max_abs = fabs(x[i]);
    }
    if (max_abs == 0.0)
        return "the origin has no projection onto the sphere";
    double scaled_norm2_hi = 0.0, scaled_norm2_lo = 0.0;
    for (int i = 0; i < n; i++) {
        double z = x[i] / max_abs;
        double term, term_error, next_hi, next_lo;
        eshkol_rm_two_product(z, z, &term, &term_error);
        eshkol_rm_dd_add(scaled_norm2_hi, scaled_norm2_lo,
                         term, term_error, &next_hi, &next_lo);
        scaled_norm2_hi = next_hi;
        scaled_norm2_lo = next_lo;
    }
    double scaled_norm = sqrt(scaled_norm2_hi + scaled_norm2_lo);
    if (!(scaled_norm > 0.0) || !isfinite(scaled_norm))
        return "the vector could not be normalised to the sphere";
    double radius_scale = R / scaled_norm;
    for (int i = 0; i < n; i++) out[i] = radius_scale * (x[i] / max_abs);
    for (int i = 0; i < n; i++)
        if (!isfinite(out[i])) return "the projected point is not finite";
    return NULL;
}

/**
 * @brief The isometry from the embedded sphere of radius R = 1/sqrt(K) in
 *        R^{n+1} (@p p, n+1 coordinates, last one the polar axis) to the
 *        alpha = 1 chart (@p x, n coordinates):
 *
 *     x = 2 R p' / (R + p_n),
 *
 * the stereographic projection from the pole p = -R e_n. Where p_n < 0 the
 * denominator is formed as |p'|^2/(R - p_n) (the same number on the sphere)
 * so the far hemisphere keeps its relative precision. The pole is refused.
 */
static const char* eshkol_rm_sphere_to_chart(const double* p, double K, int n,
                                             double* x) {
    const char* why = eshkol_rm_sphere_check_point(p, K, n + 1);
    if (why) return why;
    const double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    const double pn = p[n];
    if (pn >= 0.0) {
        const double den = R + pn;
        for (int i = 0; i < n; i++) x[i] = 2.0 * R * (p[i] / den);
    } else {
        const double pp = eshkol_rm_norm(p, n);
        if (!(pp > 0.0))
            return "the point is the pole of the chart: it has no chart "
                   "coordinates";
        const double far = R - pn;
        for (int i = 0; i < n; i++)
            x[i] = 2.0 * R * ((p[i] / pp) * (far / pp));
    }
    return eshkol_rm_check_output(x, n, "the chart coordinates are not finite");
}

/**
 * @brief The inverse isometry, chart -> embedded sphere:
 *
 *     p' = lambda_x x,  p_n = R (2 lambda_x - 1) = R (1 - K|x|^2/4) lambda_x,
 *     lambda_x = 1/(1 + K|x|^2/4).
 */
static const char* eshkol_rm_chart_to_sphere(const double* x, double K, int n,
                                             double* p) {
    if (!(K > 0.0)) return "the embedded sphere requires positive curvature";
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    const double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    const double B = eshkol_rm_ball_param(-K);
    const double den = eshkol_rm_one_minus_bnorm2(x, B, n);  /* 1 + K|x|^2/4 */
    const double q = eshkol_rm_scaled_norm2_times(x, -B, n); /* K|x|^2/4 */
    for (int i = 0; i < n; i++) p[i] = x[i] / den;
    p[n] = R * ((1.0 - q) / den);
    return eshkol_rm_check_output(p, n + 1, "the sphere coordinates are not finite");
}

/* Evaluate the chart distance and its first two B-derivatives from the
 * degree-10 series in q = K r^2 / 4 = -B z (either sign of K, and K = 0), where
 *
 *   z = |x-y|^2 / P,  P = (1-B|x|^2)(1-B|y|^2),
 *   d = LAMBDA0 sqrt(z) H(-q),
 *   H(w) = asinh(sqrt(w))/sqrt(w).
 *
 * The expressions below are the termwise derivatives of
 *   LAMBDA0 sqrt(D) sum_n a_n B^n D^n P^(-n-1/2),
 * with D = |x-y|^2.  The n=0 and n=1 pieces of d'' are divided by B^2
 * algebraically before evaluation.  This is the limit-preserving part: at
 * B = DBL_TRUE_MIN, B*D can round to zero while D, d' and d'' remain
 * representable. */
static const char* eshkol_rm_distance_dK_small_q(
    const double* x, const double* y, double B, double dBdK, double pa,
    double pb, double w, int n, double* d_out, double* d1_out,
    double* d2_out) {
    static const double a[11] = {
        1.0, -1.0 / 6.0, 3.0 / 40.0, -5.0 / 112.0,
        35.0 / 1152.0, -63.0 / 2816.0, 231.0 / 13312.0,
        -143.0 / 10240.0, 6435.0 / 557056.0, -12155.0 / 1245184.0,
        46189.0 / 5505024.0
    };
    /* sqrt(z) = |x-y| / sqrt(P), from the hypot separation so subnormal
     * separations keep their bits; z itself enters only as root_z^2 inside
     * the exponent-separated products. */
    double delta = 0.0;
    for (int i = 0; i < n; i++) delta = hypot(delta, x[i] - y[i]);
    const double U = eshkol_rm_scaled_norm2_times(x, 1.0, n);
    const double V = eshkol_rm_scaled_norm2_times(y, 1.0, n);
    if (!(U >= 0.0) || !(V >= 0.0) || !(pa > 0.0) || !(pb > 0.0) ||
        !isfinite(delta) || !isfinite(U) || !isfinite(V) ||
        !isfinite(pa) || !isfinite(pb))
        return "the small-curvature series inputs are not finite";

    const double root_z = (delta / sqrt(pa)) / sqrt(pb);
    const double ru = U / pa;
    const double rv = V / pb;
    const double r = ru + rv;
    const double s2 = ru * ru + rv * rv;
    /* R = B r and B^2 s2 = (B ru)^2 + (B rv)^2: dimensionless, formed without
     * B^2 (which underflows) or s2 (which can overflow) on its own. */
    const double bu = B * ru, bv = B * rv;
    const double R = bu + bv;
    const double T = R + bu * bu + bv * bv;
    if (!isfinite(root_z) || !isfinite(r) || !isfinite(R) || !isfinite(T))
        return "the small-curvature series is not finite";

    double H = a[10];
    for (int k = 9; k >= 0; --k) H = H * w + a[k];
    double first = 0.0;
    double power = 1.0;
    for (int k = 1; k <= 10; ++k) {
        const double alpha = (double)k + 0.5;
        first += a[k] * power * ((double)k + alpha * R);
        power *= w;
    }

    /* dB/dK enters every factor list, so d' and d'' are never formed as an
     * intermediate B-derivative that could overflow before the 1/4 and 1/16
     * are applied. */
    const double d_factors[] = {ESHKOL_RM_ALPHA, root_z, H};
    const double d = eshkol_rm_scaled_product_n(d_factors, 3);
    const double d1_zero_factors[] = {
        ESHKOL_RM_ALPHA, root_z, 0.5 * r, dBdK
    };
    const double d1_series_factors[] = {
        ESHKOL_RM_ALPHA, root_z, root_z, root_z, first, dBdK
    };
    const double d1 = eshkol_rm_scaled_product_n(d1_zero_factors, 4) +
                      eshkol_rm_scaled_product_n(d1_series_factors, 6);

    /* The n=0 term is (T-R)/B^2 + R^2/B^2, with the quotients expanded
     * directly as s2 and r^2.  The n=1 term is treated the same way. */
    const double d2_zero_factors[] = {
        ESHKOL_RM_ALPHA, root_z, 0.5 * s2 + 0.25 * r * r, dBdK, dBdK
    };
    const double alpha1 = 1.5;
    const double one_bracket = alpha1 * (2.0 * r + (bu * ru + bv * rv)) +
                               alpha1 * alpha1 * R * r;
    const double d2_one_factors[] = {
        ESHKOL_RM_ALPHA, root_z, root_z, root_z, a[1] * one_bracket, dBdK, dBdK
    };
    double d2 = eshkol_rm_scaled_product_n(d2_zero_factors, 5) +
                eshkol_rm_scaled_product_n(d2_one_factors, 7);

    power = 1.0;
    for (int k = 2; k <= 10; ++k) {
        const double alpha = (double)k + 0.5;
        const double log_derivative = (double)k + alpha * R;
        const double second_factor = log_derivative * log_derivative -
                                     log_derivative + alpha * T;
        const double d2_factors[] = {
            ESHKOL_RM_ALPHA, root_z, root_z, root_z, root_z, root_z, a[k],
            power, second_factor, dBdK, dBdK
        };
        d2 += eshkol_rm_scaled_product_n(d2_factors, 11);
        power *= w;
    }

    *d1_out = d1;
    *d2_out = d2;
    if (d_out) *d_out = d;
    if (!isfinite(d) || !isfinite(*d1_out) || !isfinite(*d2_out))
        return "the curvature derivative is not finite";
    return NULL;
}

/**
 * @brief Geodesic distance between @p x and @p y AND its first two derivatives
 *        with respect to the sectional curvature K, at fixed chart points:
 *        the curvature JET (value, d/dK, d^2/dK^2) of ADR-0018.
 *
 * Exact closed-form derivatives, not difference quotients. One formula for
 * every sign: with B = -K alpha^2/4 (dB/dK = -alpha^2/4), A = B|x|^2,
 * C = B|y|^2, w = B|x-y|^2 / ((1-A)(1-C)) and D = B d/dB,
 *
 *   L  = D log w = 1 + A/(1-A) + C/(1-C)
 *   h  = sqrt|w| psi(w)          (asinh on the ball, asin on the sphere chart)
 *   d  = alpha |B|^(-1/2) h
 *   Dh = q L,  D^2 h = q (DL + L^2/(2(1+w))),  q = sqrt|w| / (2 sqrt(1+w)),
 *
 * evaluated in the cancellation-free t1/t2 form below. For |w| <=
 * ESHKOL_RM_Q_SERIES the degree-10 series is differentiated termwise instead
 * (eshkol_rm_distance_dK_small_q), which is what carries the jet THROUGH
 * K = 0: at B = 0 the series is evaluated at w = 0 and the left and right jets
 * are the same expression. Coincident points give an identically zero d, so
 * every derivative is exactly zero.
 *
 * A non-finite requested output is refused (for example d/dK at the subnormal
 * negative-curvature floor with radius-scaled points, whose analytic value is
 * beyond DBL_MAX); there is no blanket binade refusal.
 *
 * @param d_out   the distance itself (may be NULL).
 * @param d1_out  d(distance)/dK.
 * @param d2_out  d^2(distance)/dK^2.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_distance_dK(const double* x, const double* y,
                                         double K, int n, double* d_out,
                                         double* d1_out, double* d2_out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;

    if (eshkol_rm_points_equal(x, y, n)) {
        if (d_out)  *d_out  = 0.0;
        *d1_out = 0.0;
        *d2_out = 0.0;
        return NULL;
    }
    if (eshkol_rm_needs_rescale(K, x, y, n)) {
        /* d(K; x, y) = 2 d(4K; x/2, y/2): d' scales by 8, d'' by 32. */
        double* h = eshkol_rm_halved(x, y, NULL, n);
        if (!h) return "out of memory";
        double d = 0.0, d1 = 0.0, d2 = 0.0;
        why = eshkol_rm_distance_dK(h, h + n, 4.0 * K, n, &d, &d1, &d2);
        free(h);
        if (why) return why;
        if (d_out) *d_out = 2.0 * d;
        *d1_out = 8.0 * d1;
        *d2_out = 32.0 * d2;
        if (!isfinite(*d1_out) || !isfinite(*d2_out))
            return "the curvature derivative is not finite";
        return NULL;
    }
    const double B = eshkol_rm_ball_param(-K);
    /* Dimensionless quantities. A = B|x|^2, C = B|y|^2 and G = B|x-y|^2 remain
     * representable across all curvature binades in which the points are chart
     * points; the dimensional squares are never formed. */
    const double A = eshkol_rm_scaled_norm2_times(x, B, n);
    const double C = eshkol_rm_scaled_norm2_times(y, B, n);
    const double gamma = eshkol_rm_difference_norm2_times(x, y, B, n);
    const double dBdK = -(ESHKOL_RM_ALPHA * ESHKOL_RM_ALPHA) / 4.0;
    const double pa  = eshkol_rm_one_minus_bnorm2(x, B, n);
    const double pb  = eshkol_rm_one_minus_bnorm2(y, B, n);
    const double w = (gamma / pa) / pb;
    /* On the positive chart w in [-1, 0]; within an ulp of the antipode it
     * may round below -1, and 1 + w is then taken from the Mobius form. */
    if (!isfinite(w) || (B > 0.0 && w < 0.0))
        return "the dimensionless curvature argument is not finite";

    /* Do not require w to be nonzero: at the subnormal curvature floor and at
     * K = 0 itself B*|x-y|^2 is zero while the derivatives are ordinary f64
     * values. */
    if (fabs(w) <= ESHKOL_RM_Q_SERIES)
        return eshkol_rm_distance_dK_small_q(
            x, y, B, dBdK, pa, pb, w, n, d_out, d1_out, d2_out);

    /* 1 + w = Dn/P with Dn the Mobius denominator of (-x, y): on the positive
     * chart it is the quantity that vanishes at the antipode, so it is taken
     * from the cancellation-free Mobius form rather than from 1 + w. */
    double onepw = 1.0 + w;
    if (B < 0.0) {
        const double dn = eshkol_rm_mobius_den_negx(x, y, B, n);
        if (!(dn > 0.0))
            return "the two points are antipodal (the cut locus of the "
                   "positive-curvature chart): no curvature derivative";
        onepw = (dn / pa) / pb;
    }
    const double ra = A / pa;
    const double rb = C / pb;
    const double R = ra + rb;
    const double T = R + ra * ra + rb * rb;
    const double root_w = sqrt(fabs(w));
    const double h = (w > 0.0) ? asinh(root_w) : atan2(root_w, sqrt(onepw));
    double t1, t2;
    const double t0 = h;
    if (fabs(w) <= 0.5) {
        /* Factor the cancellations in t1 and t2 through psi'(w) and psi''(w)
         * (series-evaluated here). The residual R/T terms are the dependence
         * of P on B. */
        const double q = 0.5 * root_w / sqrt(onepw);
        double psi1 = 0.0, psi2 = 0.0;
        (void)eshkol_rm_psi(w, &psi1, &psi2);
        t1 = root_w * w * psi1 + q * R;
        /* T + (2R + R^2)/(2(1+w)) - 2R, regrouped so the O(R) terms do not
         * cancel: (ra^2 + rb^2) + R^2/(2(1+w)) - R w/(1+w). */
        const double bracket = (ra * ra + rb * rb) + (R * R) / (2.0 * onepw) -
                               R * (w / onepw);
        t2 = root_w * w * w * psi2 + q * bracket;
    } else {
        /* sinh^2 h = w (ball) and sin^2 h = -w (sphere chart) give one form:
         * with tau = sqrt|w| / sqrt(1+w) (tanh h, resp. tan h) and
         * 1 + w = cosh^2 h, resp. cos^2 h,
         *   Dh = tau L / 2,  D^2 h = tau L^2 / (4(1+w)) + tau T / 2.
         * Near the positive-chart antipode tau ~ 1/eps while L ~ eps, so
         * this is the form in which no O(1/eps^3) terms are formed only to
         * cancel. */
        const double L = 1.0 + R;
        const double tau = root_w / sqrt(onepw);
        const double Dh = 0.5 * tau * L;
        const double D2h = 0.25 * tau * L * (L / onepw) + 0.5 * tau * T;
        t1 = Dh - 0.5 * h;
        t2 = D2h - 2.0 * Dh + 0.75 * h;
    }
    const double abs_B = fabs(B);
    const double sign_B = (B > 0.0) ? 1.0 : -1.0;
    /* dB/dK is folded in before the |B|^(-p/2) scaling so a representable
     * d' or d'' is never preceded by an overflowing B-derivative. */
    const double d = ESHKOL_RM_ALPHA *
        eshkol_rm_inverse_power_mul(abs_B, 1, t0);
    const double d1 = sign_B * ESHKOL_RM_ALPHA *
        eshkol_rm_inverse_power_mul(abs_B, 3, t1 * dBdK);
    const double d2 = ESHKOL_RM_ALPHA *
        eshkol_rm_inverse_power_mul(abs_B, 5, (t2 * dBdK) * dBdK);
    if (d_out) *d_out = d;
    *d1_out = d1;
    *d2_out = d2;
    if (!isfinite(d) || !isfinite(*d1_out) || !isfinite(*d2_out))
        return "the curvature derivative is not finite";
    return NULL;
}

/**
 * @brief The curvature jet of the SQUARED distance, from the distance jet:
 *        (d^2)' = 2 d d',  (d^2)'' = 2 (d'^2 + d d'').
 */
static const char* eshkol_rm_sq_distance_dK(const double* x, const double* y,
                                            double K, int n, double* s_out,
                                            double* s1_out, double* s2_out) {
    double d = 0.0, d1 = 0.0, d2 = 0.0;
    const char* why = eshkol_rm_distance_dK(x, y, K, n, &d, &d1, &d2);
    if (why) return why;
    if (s_out) *s_out = d * d;
    *s1_out = 2.0 * d * d1;
    *s2_out = 2.0 * (d1 * d1 + d * d2);
    if (!isfinite(d * d) || !isfinite(*s1_out) || !isfinite(*s2_out))
        return "the squared-distance curvature derivative is not finite";
    return NULL;
}

/* ═══ Squared-distance geodesic attention (ADR-0018 decision 5) ═══════════
 *
 * Logits are -d^2/(2t) with a finite temperature t in squared-distance units;
 * there is no curvature-dependent scale anywhere. The default temperature is
 * sqrt(head_dim): at K = 0 the logit is -|q-k|^2/(2 sqrt(dim)), which differs
 * from scaled dot-product attention q.k/sqrt(dim) only by terms constant in the
 * softmax row and the key-norm bias.
 *
 * eshkol_rm_attention_row is the ONE stabilized softmax every engine runs: the
 * VM ops, the AD bridge forward, and the native module. The shift is the
 * row's minimum squared distance, so no absolute logit is materialised; a gap
 * whose scaled logit underflows has the exact weight zero. */

/** @brief The default attention temperature sqrt(head_dim). */
static double eshkol_rm_default_temperature(int head_dim) {
    return sqrt((double)head_dim);
}

/**
 * @brief One attention row: weights_j = softmax_j(-d^2(q, k_j)/(2t)) over the
 *        first @p limit keys (the causal mask is the caller's limit).
 *
 * @param keys        first key row; key j starts at keys + j*key_stride
 * @param sqd         limit doubles: receives d^2(q, k_j)
 * @param weights     limit doubles: receives the normalised weights
 * @return NULL on success, else a reason (a refused pair names the domain).
 */
static const char* eshkol_rm_attention_row(const double* q, const double* keys,
                                           long key_stride, int limit, int dim,
                                           double K, double temperature,
                                           double* sqd, double* weights) {
    if (!(temperature > 0.0) || !isfinite(temperature))
        return "the attention temperature must be finite and positive";
    if (limit <= 0) return "an attention row needs at least one key";
    double min_sq = HUGE_VAL;
    for (int j = 0; j < limit; j++) {
        const char* why = eshkol_rm_sq_distance(q, keys + (long)j * key_stride,
                                                K, dim, &sqd[j]);
        if (why) return why;
        if (sqd[j] < min_sq) min_sq = sqd[j];
    }
    const double inv_two_t = 0.5 / temperature;
    double sum = 0.0;
    for (int j = 0; j < limit; j++) {
        const double gap = sqd[j] - min_sq;
        if (!isfinite(gap) || gap < 0.0)
            return "the attention row could not form a finite squared-distance gap";
        weights[j] = (gap > DBL_MAX / inv_two_t) ? 0.0 : exp(-gap * inv_two_t);
        sum += weights[j];
    }
    if (!(sum > 0.0) || !isfinite(sum))
        return "the attention softmax normalisation is not finite";
    for (int j = 0; j < limit; j++) weights[j] /= sum;
    return NULL;
}

/* ═══ Convention ids and legacy migration (ADR-0018 Migration) ════════════ */

/** @brief The header every serialized geometry record carries. */
typedef struct {
    uint32_t convention;   /* ESHKOL_GEOMETRY_CONVENTION_* */
    uint32_t reserved;     /* must be zero */
    double   alpha;        /* the gauge; ESHKOL_RM_ALPHA for v2 */
    double   curvature;    /* signed sectional curvature K */
} eshkol_geometry_header_t;

/** @brief Fill @p h for the current convention at curvature @p K. */
static void eshkol_rm_geometry_header(double K, eshkol_geometry_header_t* h) {
    h->convention = ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1;
    h->reserved = 0u;
    h->alpha = ESHKOL_RM_ALPHA;
    h->curvature = K;
}

/**
 * @brief Accept a serialized header only when it names the current convention
 *        and gauge. An unversioned or legacy record is refused, never guessed:
 *        legacy data goes through the explicit migration helpers below.
 */
static const char* eshkol_rm_check_geometry_header(const eshkol_geometry_header_t* h) {
    if (!h) return "the geometry record has no header";
    if (h->convention == ESHKOL_GEOMETRY_CONVENTION_UNVERSIONED)
        return "the geometry record carries no convention id; unversioned "
               "coordinates are never reinterpreted";
    if (h->convention == ESHKOL_GEOMETRY_CONVENTION_LEGACY_BALL_V1)
        return "the geometry record uses the legacy ball convention; migrate it "
               "(x_v2 = 2 x_v1) before loading";
    if (h->convention != ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1)
        return "the geometry record names an unknown convention id";
    if (h->reserved != 0u) return "the geometry record header is malformed";
    if (!(h->alpha == ESHKOL_RM_ALPHA))
        return "the geometry record gauge alpha does not match this build";
    if (!isfinite(h->curvature)) return "the geometry record curvature is not finite";
    return NULL;
}

/**
 * @brief Migrate a legacy ball point (alpha = 2, radius 1/sqrt(-K)) to the
 *        current chart: x_v2 = 2 x_v1. K = 0 legacy coordinates were already
 *        canonical Euclidean and are copied. Positive legacy data was ambient
 *        (use eshkol_rm_sphere_to_chart).
 */
static const char* eshkol_rm_migrate_legacy_point(const double* x_v1, double K,
                                                  int n, double* x_v2) {
    if (!isfinite(K)) return "curvature is not finite";
    if (K > 0.0)
        return "legacy positive-curvature points are ambient sphere points; "
               "migrate them with the stereographic projection";
    for (int i = 0; i < n; i++)
        if (!isfinite(x_v1[i])) return "a coordinate is not finite";
    if (K < 0.0) {
        /* legacy ball: |x|^2 < 1/c with c = -K */
        if (!(eshkol_rm_one_minus_bnorm2(x_v1, -K, n) > 0.0))
            return "the legacy point is not strictly inside the legacy ball";
        for (int i = 0; i < n; i++) x_v2[i] = 2.0 * x_v1[i];
        return eshkol_rm_check_point(x_v2, K, n);
    }
    memcpy(x_v2, x_v1, (size_t)n * sizeof(double));
    return NULL;
}

/** @brief Migrate a legacy tangent vector (or first-moment optimizer state)
 *         by the differential of x_v2 = 2 x_v1. Squared Riemannian norms
 *         (second-moment state) are invariant and are not touched. */
static const char* eshkol_rm_migrate_legacy_tangent(const double* v_v1, double K,
                                                    int n, double* v_v2) {
    if (!isfinite(K)) return "curvature is not finite";
    if (K > 0.0)
        return "legacy positive-curvature tangents are ambient; push them "
               "through the differential of the stereographic projection";
    const double scale = (K < 0.0) ? 2.0 : 1.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(v_v1[i])) return "a tangent-vector component is not finite";
        v_v2[i] = scale * v_v1[i];
    }
    return NULL;
}

/**
 * @brief Migrate a legacy attention score scale s (logit = -s d) to a
 *        squared-distance temperature t (logit = -d^2/(2t)).
 *
 * No temperature reproduces a linear-distance score exactly. The migration
 * matches the two logits to first order at the caller's reference distance
 * d_ref: softmax rows are invariant to a constant offset, and the slopes
 * agree when t = d_ref / s. Callers record d_ref (for example the median
 * attention distance of the checkpoint) next to the migrated temperature.
 */
static const char* eshkol_rm_migrate_legacy_temperature(double legacy_scale,
                                                        double reference_distance,
                                                        double* temperature) {
    if (!(legacy_scale > 0.0) || !isfinite(legacy_scale))
        return "the legacy attention scale must be finite and positive";
    if (!(reference_distance > 0.0) || !isfinite(reference_distance))
        return "the reference distance must be finite and positive";
    *temperature = reference_distance / legacy_scale;
    return (isfinite(*temperature) && *temperature > 0.0)
        ? NULL : "the migrated temperature is not finite";
}

#ifdef __cplusplus

/*
 * Directional derivatives of the shared forward formulas.
 *
 * These are deliberately kept beside the C forward implementation.  A
 * reverse rule must not rebuild a second set of squared norms or Mobius
 * denominators: those copies lose the max-abs scaling which makes the forward
 * valid at both exponent ends.  The value members below are obtained from the
 * shared helpers, while the tangent members differentiate the same algebra
 * with those scale factors held fixed at the primal point.  Holding a valid
 * normalization scale fixed is an identity-preserving way to differentiate
 * the scaled expression, and avoids forming an otherwise overflowing square.
 */
struct eshkol_rm_directional {
    double value;
    double tangent;
};

static eshkol_rm_directional eshkol_rm_dadd(eshkol_rm_directional a,
                                             eshkol_rm_directional b) {
    return {a.value + b.value, a.tangent + b.tangent};
}

static eshkol_rm_directional eshkol_rm_dsub(eshkol_rm_directional a,
                                             eshkol_rm_directional b) {
    return {a.value - b.value, a.tangent - b.tangent};
}

static eshkol_rm_directional eshkol_rm_dneg(eshkol_rm_directional a) {
    return {-a.value, -a.tangent};
}

static eshkol_rm_directional eshkol_rm_dmul(eshkol_rm_directional a,
                                             eshkol_rm_directional b) {
    return {a.value * b.value,
            a.tangent * b.value + a.value * b.tangent};
}

static eshkol_rm_directional eshkol_rm_ddiv(eshkol_rm_directional a,
                                             eshkol_rm_directional b) {
    const double value = a.value / b.value;
    return {value, a.tangent / b.value - value * (b.tangent / b.value)};
}

static eshkol_rm_directional eshkol_rm_dsqrt(eshkol_rm_directional a) {
    if (a.value == 0.0) return {0.0, 0.0};
    double r = sqrt(a.value);
    return {r, a.tangent / (2.0 * r)};
}

/* Directional derivative of eshkol_rm_tan_ratio in its own argument s:
 * the executed branch (series or closed form) and its derivative. */
static eshkol_rm_directional eshkol_rm_dtan_ratio(eshkol_rm_directional s) {
    double d1 = 0.0;
    const double value = eshkol_rm_tan_ratio(s.value, &d1);
    return {value, d1 * s.tangent};
}

static eshkol_rm_directional eshkol_rm_dnorm(const double* a,
                                              const double* da, int n) {
    const double value = eshkol_rm_norm(a, n);
    if (value == 0.0) return {0.0, 0.0};
    return {value, eshkol_rm_scaled_dot_factor(a, da, 1.0, n) / value};
}

static eshkol_rm_directional eshkol_rm_dscaled_norm2_times(
    const double* a, const double* da, double factor, int n) {
    return {eshkol_rm_scaled_norm2_times(a, factor, n),
            2.0 * eshkol_rm_scaled_dot_factor(a, da, factor, n)};
}

static eshkol_rm_directional eshkol_rm_done_minus_bnorm2(
    const double* x, const double* dx, double B, int n) {
    return {eshkol_rm_one_minus_bnorm2(x, B, n),
            -2.0 * eshkol_rm_scaled_dot_factor(x, dx, B, n)};
}

static eshkol_rm_directional eshkol_rm_done_plus_dot(
    const double* a, const double* b, const double* da, const double* db,
    double B, int n) {
    return {eshkol_rm_one_plus_dot(a, b, B, n),
            eshkol_rm_scaled_dot_factor(da, b, B, n) +
            eshkol_rm_scaled_dot_factor(a, db, B, n)};
}

static eshkol_rm_directional eshkol_rm_done_minus_dot(
    const double* a, const double* b, const double* da, const double* db,
    double B, int n) {
    return {eshkol_rm_one_minus_dot(a, b, B, n),
            -eshkol_rm_scaled_dot_factor(da, b, B, n) -
             eshkol_rm_scaled_dot_factor(a, db, B, n)};
}

static eshkol_rm_directional eshkol_rm_dmobius_den(
    const double* a, const double* b, const double* da, const double* db,
    double B, int n, bool negate_a) {
    eshkol_rm_directional q = negate_a
        ? eshkol_rm_done_minus_dot(a, b, da, db, B, n)
        : eshkol_rm_done_plus_dot(a, b, da, db, B, n);
    eshkol_rm_directional result = eshkol_rm_dmul(q, q);
    const double root_B = eshkol_rm_sqrt_nonnegative(B);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            eshkol_rm_directional ai = {root_B * a[i], root_B * da[i]};
            eshkol_rm_directional aj = {root_B * a[j], root_B * da[j]};
            eshkol_rm_directional bi = {root_B * b[i], root_B * db[i]};
            eshkol_rm_directional bj = {root_B * b[j], root_B * db[j]};
            eshkol_rm_directional t = eshkol_rm_dsub(
                eshkol_rm_dmul(ai, bj), eshkol_rm_dmul(aj, bi));
            result = eshkol_rm_dadd(result, eshkol_rm_dmul(t, t));
        }
    }
    result.value = negate_a
        ? eshkol_rm_mobius_den_negx(a, b, B, n)
        : eshkol_rm_mobius_den(a, b, B, n);
    return result;
}

static void eshkol_rm_daxpby_exact(
    eshkol_rm_directional p, const double* a, const double* da,
    eshkol_rm_directional q, const double* b, const double* db, int n,
    eshkol_rm_directional* out) {
    std::vector<double> values((size_t)n);
    eshkol_rm_axpby_exact(p.value, a, q.value, b, n, values.data());
    for (int i = 0; i < n; ++i) {
        const eshkol_rm_directional pa = {a[i], da[i]};
        const eshkol_rm_directional pb = {b[i], db[i]};
        out[i] = {values[(size_t)i],
                  eshkol_rm_dmul(p, pa).tangent +
                  eshkol_rm_dmul(q, pb).tangent};
    }
}

static void eshkol_rm_dmobius_add(
    const double* a, const double* b, const double* da, const double* db,
    double B, int n, eshkol_rm_directional* out) {
    eshkol_rm_directional den = eshkol_rm_dmobius_den(
        a, b, da, db, B, n, false);
    eshkol_rm_directional num_y = eshkol_rm_done_minus_bnorm2(a, da, B, n);
    eshkol_rm_directional by2 = eshkol_rm_dscaled_norm2_times(b, db, B, n);
    eshkol_rm_directional num_x = eshkol_rm_dadd(
        den, eshkol_rm_dmul(by2, num_y));
    eshkol_rm_daxpby_exact(num_x, a, da, num_y, b, db, n, out);
    std::vector<double> values((size_t)n);
    eshkol_rm_mobius_add(a, b, B, n, values.data());
    for (int i = 0; i < n; ++i) {
        const double numerator = out[i].value;
        out[i].value = values[(size_t)i];
        out[i].tangent = (out[i].tangent * den.value -
                          numerator * den.tangent) /
                         (den.value * den.value);
    }
}

/* Directional derivative of the exact shared distance. */
static bool eshkol_rm_distance_directional(
    const double* x, const double* y, const double* dx, const double* dy,
    double K, int n, double* out, double* dout) {
    if (eshkol_rm_check_point(x, K, n) || eshkol_rm_check_point(y, K, n))
        return false;
    if (eshkol_rm_points_equal(x, y, n)) return false;
    if (K == 0.0) {
        std::vector<double> delta((size_t)n), ddelta((size_t)n);
        for (int i = 0; i < n; ++i) {
            delta[(size_t)i] = x[i] - y[i];
            ddelta[(size_t)i] = dx[i] - dy[i];
        }
        eshkol_rm_directional d = eshkol_rm_dnorm(
            delta.data(), ddelta.data(), n);
        if (out) *out = ESHKOL_RM_ALPHA * d.value;
        if (dout) *dout = ESHKOL_RM_ALPHA * d.tangent;
        return isfinite(d.value) && isfinite(d.tangent);
    }
    if (K >= 0.0) return false;

    const double B = eshkol_rm_ball_param(-K);
    std::vector<double> delta((size_t)n), ddelta((size_t)n);
    for (int i = 0; i < n; ++i) {
        delta[(size_t)i] = x[i] - y[i];
        ddelta[(size_t)i] = dx[i] - dy[i];
    }
    eshkol_rm_directional delta_norm = eshkol_rm_dnorm(
        delta.data(), ddelta.data(), n);
    eshkol_rm_directional px = eshkol_rm_done_minus_bnorm2(x, dx, B, n);
    eshkol_rm_directional py = eshkol_rm_done_minus_bnorm2(y, dy, B, n);
    /* The forward's order: (|x-y| / sqrt(pa)) / sqrt(pb). */
    eshkol_rm_directional scaled_delta = eshkol_rm_ddiv(
        eshkol_rm_ddiv(delta_norm, eshkol_rm_dsqrt(px)), eshkol_rm_dsqrt(py));
    /* Keep the forward's multiplication order.  The squared scaled distance
     * can overflow before its B factor is applied even when B*distance^2 is
     * finite. */
    eshkol_rm_directional w = {
        eshkol_rm_scaled_square(B, scaled_delta.value),
        eshkol_rm_scaled_product4(B, 2.0, scaled_delta.value,
                                  scaled_delta.tangent)};
    double psi1 = 0.0, psi2 = 0.0;
    const double psi_value = eshkol_rm_psi(w.value, &psi1, &psi2);
    eshkol_rm_directional distance = {
        ESHKOL_RM_ALPHA * scaled_delta.value * psi_value,
        ESHKOL_RM_ALPHA *
            (scaled_delta.tangent * psi_value +
             scaled_delta.value * psi1 * w.tangent)};
    double shared = 0.0;
    if (eshkol_rm_distance(x, y, K, n, &shared)) return false;
    if (out) *out = shared;
    if (dout) *dout = distance.tangent;
    return isfinite(shared) && isfinite(distance.tangent);
}

/* Directional derivative of exp_x(v), using the exact shared Mobius forward. */
static bool eshkol_rm_exp_directional(
    const double* x, const double* v, const double* dx, const double* dv,
    double K, int n, double* out, double* dout) {
    std::vector<double> shared((size_t)n), scratch((size_t)n);
    if (eshkol_rm_exp_map(x, v, K, n, shared.data(), scratch.data()))
        return false;
    if (K >= 0.0) {
        if (K > 0.0) return false;
        for (int i = 0; i < n; ++i) {
            if (out) out[i] = shared[(size_t)i];
            if (dout) dout[i] = dx[i] + dv[i];
        }
        return true;
    }
    const double B = eshkol_rm_ball_param(-K);
    eshkol_rm_directional vn = eshkol_rm_dnorm(v, dv, n);
    eshkol_rm_directional nb = eshkol_rm_done_minus_bnorm2(x, dx, B, n);
    eshkol_rm_directional lambda = eshkol_rm_ddiv(
        {ESHKOL_RM_ALPHA, 0.0}, nb);
    /* s = B (lambda |v| / alpha)^2 in the forward's multiplication order. */
    eshkol_rm_directional r = eshkol_rm_ddiv(
        eshkol_rm_dmul(lambda, vn), {ESHKOL_RM_ALPHA, 0.0});
    eshkol_rm_directional sarg = {
        eshkol_rm_scaled_square(B, r.value),
        eshkol_rm_scaled_product4(B, 2.0, r.value, r.tangent)};
    eshkol_rm_directional f = eshkol_rm_dmul(
        eshkol_rm_ddiv(lambda, {ESHKOL_RM_ALPHA, 0.0}),
        eshkol_rm_dtan_ratio(sarg));
    std::vector<double> s((size_t)n), ds((size_t)n);
    for (int i = 0; i < n; ++i) {
        s[(size_t)i] = f.value * v[i];
        ds[(size_t)i] = f.tangent * v[i] + f.value * dv[i];
    }
    std::vector<eshkol_rm_directional> mapped((size_t)n);
    eshkol_rm_dmobius_add(x, s.data(), dx, ds.data(), B, n, mapped.data());
    for (int i = 0; i < n; ++i) {
        if (out) out[i] = shared[(size_t)i];
        if (dout) dout[i] = mapped[(size_t)i].tangent;
    }
    return true;
}

/* Directional derivative of log_x(y), including the exact coincidence limit. */
static bool eshkol_rm_log_directional(
    const double* x, const double* y, const double* dx, const double* dy,
    double K, int n, double* out, double* dout) {
    std::vector<double> shared((size_t)n), scratch((size_t)n);
    if (eshkol_rm_log_map(x, y, K, n, shared.data(), scratch.data()))
        return false;
    if (eshkol_rm_points_equal(x, y, n)) {
        for (int i = 0; i < n; ++i) {
            if (out) out[i] = shared[(size_t)i];
            if (dout) dout[i] = dy[i] - dx[i];
        }
        return true;
    }
    if (K >= 0.0) {
        if (K > 0.0) return false;
        for (int i = 0; i < n; ++i) {
            if (out) out[i] = shared[(size_t)i];
            if (dout) dout[i] = dy[i] - dx[i];
        }
        return true;
    }
    const double B = eshkol_rm_ball_param(-K);
    eshkol_rm_directional den = eshkol_rm_dmobius_den(
        x, y, dx, dy, B, n, true);
    eshkol_rm_directional nb = eshkol_rm_done_minus_bnorm2(x, dx, B, n);
    eshkol_rm_directional y2 = eshkol_rm_dscaled_norm2_times(y, dy, B, n);
    eshkol_rm_directional na = eshkol_rm_dadd(
        den, eshkol_rm_dmul(y2, nb));
    std::vector<eshkol_rm_directional> V((size_t)n);
    eshkol_rm_daxpby_exact(nb, y, dy, eshkol_rm_dneg(na), x, dx, n,
                           V.data());
    std::vector<double> Vv((size_t)n), dV((size_t)n);
    for (int i = 0; i < n; ++i) {
        Vv[(size_t)i] = V[(size_t)i].value;
        dV[(size_t)i] = V[(size_t)i].tangent;
    }
    eshkol_rm_directional Vn = eshkol_rm_dnorm(Vv.data(), dV.data(), n);
    double distance_value = 0.0, distance_tangent = 0.0;
    if (!eshkol_rm_distance_directional(x, y, dx, dy, K, n,
                                        &distance_value, &distance_tangent))
        return false;
    eshkol_rm_directional coefficient = eshkol_rm_ddiv(
        eshkol_rm_ddiv({distance_value, distance_tangent},
                       eshkol_rm_ddiv({ESHKOL_RM_ALPHA, 0.0}, nb)), Vn);
    for (int i = 0; i < n; ++i) {
        eshkol_rm_directional result = eshkol_rm_dmul(
            coefficient, V[(size_t)i]);
        if (out) out[i] = shared[(size_t)i];
        if (dout) dout[i] = result.tangent;
    }
    return true;
}

#endif /* __cplusplus */

#if defined(__clang__)
#pragma STDC FP_CONTRACT ON
#pragma clang fp contract(on)
#endif

#endif /* ESHKOL_BACKEND_RIEMANNIAN_CORE_H */
