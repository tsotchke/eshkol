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
 * ═══ THE MODEL, IN ONE PLACE ═══════════════════════════════════════════════
 *
 * Every entry point below takes @c K, the SECTIONAL CURVATURE, and dispatches on
 * its sign. The chart and its normalisation are fixed by TWO constants and two
 * accessors, and nothing else in this file or in vm_geometric.c open-codes them:
 *
 *   ESHKOL_RM_LAMBDA0       the conformal factor at the origin of the ball chart
 *   ESHKOL_RM_FLAT_LAMBDA   the conformal factor the K = 0 branch uses
 *   eshkol_rm_ball_param(c) the chart's ball parameter B, from c = -K
 *   eshkol_rm_lambda(x,c,n) the conformal factor at a point
 *
 *   K < 0   Poincare ball. With c = -K and B = eshkol_rm_ball_param(c), the
 *           metric is g_x = lambda_x^2 <.,.> with
 *
 *               lambda_x = LAMBDA0 / (1 - B |x|^2),   B = c LAMBDA0^2 / 4,
 *
 *           so the ball has Euclidean radius 1/sqrt(B) = 2/(LAMBDA0 sqrt(c)) and
 *           sectional curvature exactly -c. At the shipped LAMBDA0 = 2 this is
 *           B = c, radius 1/sqrt(c) and lambda_x = 2/(1-c|x|^2): the convention
 *           of Ganea et al., Nickel-Kiela and geoopt, which is what every
 *           in-tree call site already passes -- `(make-hyperbolic-manifold 2
 *           -1.0)`, `(poincare-distance x y -1.0)` -- and what
 *           `eshkol_frechet_mean_compute` takes.
 *   K = 0   Flat R^n with the metric ESHKOL_RM_FLAT_LAMBDA^2 <.,.>.
 *   K > 0   Round sphere of radius R = 1/sqrt(K), points required to lie ON it.
 *
 * THE FAMILY IS DISCONTINUOUS AT K = 0 AS SHIPPED, AND THAT IS A KNOWN OPEN
 * QUESTION, NOT AN OVERSIGHT. The c -> 0 limit of the ball branch is flat space
 * with the metric LAMBDA0^2 <.,.>, because lambda_0 = LAMBDA0 for every c. The
 * K = 0 branch instead uses FLAT_LAMBDA = 1, the CANONICAL Euclidean metric,
 * which is what "K = 0 is Euclidean" means to a caller and what every existing
 * flat-reduction test asserts. Those two cannot both hold: with LAMBDA0 = 2 and
 * FLAT_LAMBDA = 1 the geodesic distance jumps by a factor of 2 as K crosses 0,
 * and the Riemannian gradient by a factor of 4.
 *
 * Which of the two to keep is a CONVENTION RULING, not a bug fix, because it
 * changes published numbers and the AD bridge's contract along with them.
 * Setting FLAT_LAMBDA to LAMBDA0 is the entire change on this side: the family
 * becomes real-analytic in K on K <= 0, `geodesic-distance` at K = 0 returns
 * 2|x-y| and `riemannian-grad` returns g/4, and eshkol_rm_distance_dK stops
 * refusing at K = 0. Setting LAMBDA0 to 1 is the other resolution -- a ball of
 * radius 2/sqrt(c) whose flat limit is canonical -- and is also a one-line
 * change here, though it moves every published hyperbolic constant.
 *
 * exp, log, parallel transport, projection and Mobius addition are NOT affected
 * by either constant beyond the ball parameter: a CONSTANT conformal rescale of
 * a metric leaves the Levi-Civita connection unchanged, so those five maps are
 * the same maps under any LAMBDA0. Only distances, norms and the
 * gradient conversion carry the factor.
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
#include <string.h>
#include <float.h>
#ifdef __cplusplus
#include <vector>
#endif

/* The conformal factor at the origin of the ball chart. See the model block
 * above: this fixes the ball's radius (2/(LAMBDA0 sqrt c)) and the metric's
 * normalisation together, because both are determined by requiring the
 * sectional curvature to be exactly -c. */
#define ESHKOL_RM_LAMBDA0 2.0

/* The conformal factor the K = 0 branch uses. Equal to ESHKOL_RM_LAMBDA0 makes
 * the K <= 0 family continuous (and real-analytic) in K; equal to 1 makes K = 0
 * the canonical Euclidean metric. It cannot be both. UNDER RULING -- see the
 * model block above and docs/reference/stdlib/geometry.md. */
#define ESHKOL_RM_FLAT_LAMBDA 1.0

/* Relative tolerance on |x| = R for a point claimed to lie on the sphere of
 * radius R. A point off the sphere has no geodesic relation to another point on
 * it, so the ops refuse rather than project silently; `manifold-project` is the
 * op that moves a point onto the manifold. */
#define ESHKOL_RM_SPHERE_TOL 1e-9

/* Relative tolerance on <x,v> = 0 for a vector claimed to be tangent to the
 * sphere at x. RELATIVE TO |v| AND R, both: the test is |<x,v>| <= tol R |v|,
 * which is scale-free in v. It used to be |<x,v>|/R <= tol (|v| + R), whose
 * additive R made it absolute, so a WHOLLY RADIAL v of norm 1e-10 passed at
 * R = 1 -- 100% radial and accepted, because the threshold did not shrink with
 * the vector. */
#define ESHKOL_RM_TANGENT_TOL 1e-9

/* Below this |w| the series for asinh(sqrt w)/sqrt w and its two derivatives is
 * used instead of the closed forms, which lose digits to cancellation there.
 * Seven terms at w = 1e-2 truncate at ~1e-16 relative. */
#define ESHKOL_RM_PSI_SMALL 1e-2

/* Below this |z| the series for tanh(z)/z is used. */
#define ESHKOL_RM_TAU_SMALL 1e-4

static double eshkol_rm_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double eshkol_rm_norm(const double* a, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double ai = fabs(a[i]);
        if (!isfinite(ai)) return ai;
        if (ai > max_abs) max_abs = ai;
    }
    if (max_abs == 0.0) return 0.0;
    double scaled_norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        double z = a[i] / max_abs;
        scaled_norm2 += z * z;
    }
    double scaled_norm = sqrt(scaled_norm2);
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
    double scaled_norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        double z = a[i] / max_abs;
        scaled_norm2 += z * z;
    }
    return eshkol_rm_scaled_product4(factor, max_abs, max_abs,
                                     scaled_norm2);
}

/* Multiply four finite factors in one exponent-scaled operation.  In
 * particular, do not form factor*max_abs first: that product can underflow
 * even when factor*max_abs*max_abs*sum is representable. */
static double eshkol_rm_scaled_product4(double a, double b, double c, double d) {
    if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) return 0.0;
    int ea, eb, ec, ed;
    double ma = frexp(a, &ea), mb = frexp(b, &eb);
    double mc = frexp(c, &ec), md = frexp(d, &ed);
    return scalbn(((ma * mb) * mc) * md, ea + eb + ec + ed);
}

/* The same product with one f64 residual retained. The residual is important
 * when the product is close to one and the desired result is 1-product: the
 * returned high word alone would expose the subtraction to the f64 ulp at 1. */
static void eshkol_rm_scaled_product4_dd(double a, double b, double c, double d,
                                         double* hi_out, double* lo_out) {
    int ea, eb, ec, ed;
    double ma = frexp(a, &ea), mb = frexp(b, &eb);
    double mc = frexp(c, &ec), md = frexp(d, &ed);
    double p1 = ma * mb;
    double e1 = fma(ma, mb, -p1);
    double p2 = p1 * mc;
    double e2 = fma(p1, mc, -p2) + e1 * mc;
    double p3 = p2 * md;
    double e3 = fma(p2, md, -p3) + e2 * md;
    const int exponent = ea + eb + ec + ed;
    *hi_out = scalbn(p3, exponent);
    *lo_out = scalbn(e3, exponent);
}

/* A curvature-weighted square with the curvature exponent carried separately
 * through the product.  In particular, never form factor*a first: at the
 * subnormal curvature floor that intermediate has too few significand bits
 * for the derivative even when factor*a*a is representable. */
static double eshkol_rm_scaled_square(double factor, double a) {
    return eshkol_rm_scaled_product4(factor, a, a, 1.0);
}

static double eshkol_rm_difference_norm2(const double* a, const double* b, int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (!isfinite(d)) return d;
        if (d > max_abs) max_abs = d;
    }
    if (max_abs == 0.0) return 0.0;
    double scaled = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (a[i] - b[i]) / max_abs;
        scaled += d * d;
    }
    return eshkol_rm_scaled_product4(1.0, max_abs, max_abs, scaled);
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
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (a[i] / ma) * (b[i] / mb);
    return eshkol_rm_scaled_product4(factor, ma, mb, sum);
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
static int eshkol_rm_sphere_antipodal(const double* x, const double* y, int n) {
    double xn = eshkol_rm_norm(x, n);
    double yn = eshkol_rm_norm(y, n);
    if (!(xn > 0.0) || !(yn > 0.0)) return 0;
    for (int i = 0; i < n; i++)
        if (y[i] / yn != -(x[i] / xn)) return 0;
    return 1;
}

/* Cancellation-free spherical geometry. The input points are first
 * canonicalized to the requested radius. If u is non-NULL it receives the
 * tangent at x. */
static double eshkol_rm_sphere_angle(const double* x, const double* y,
                                     double R, int n, double* u) {
    double xn = eshkol_rm_norm(x, n);
    double yn = eshkol_rm_norm(y, n);
    double chord = 0.0;
    for (int i = 0; i < n; i++) {
        double delta = R * (y[i] / yn) - R * (x[i] / xn);
        chord = hypot(chord, delta);
    }
    double chord_over_R = chord / R;
    double half_chord_sq = 0.5 * chord_over_R * chord_over_R;
    double un = 0.0;
    for (int i = 0; i < n; i++) {
        double bar_x = R * (x[i] / xn);
        double ux = R * (y[i] / yn) - bar_x + half_chord_sq * bar_x;
        if (u) u[i] = ux;
        un = hypot(un, ux);
    }
    return atan2(un / R, 1.0 - half_chord_sq);
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
    double hi = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double p = a[i] * b[i];
        double e = fma(a[i], b[i], -p);      /* exact residual of the product */
        double s = hi + p;                   /* two-sum                       */
        double bb = s - hi;
        c += (hi - (s - bb)) + (p - bb) + e;
        hi = s;
    }
    *lo = c;
    return hi;
}

/** @brief 1 - B<a,b>, to full relative precision. The fma performs the
 *         subtraction with a single rounding of the exact product, so a result
 *         of size 1e-12 is accurate to 1e-12 * eps rather than to eps. */
static double eshkol_rm_one_minus_dot(const double* a, const double* b, double B,
                                      int n) {
    double lo = 0.0;
    double hi = eshkol_rm_dot_dd(a, b, n, &lo);
    double result = fma(-B, lo, fma(-B, hi, 1.0));
    return isfinite(result) ? result : 1.0 - eshkol_rm_scaled_dot_factor(a, b, B, n);
}

/** @brief 1 + B<a,b>, to full relative precision. */
static double eshkol_rm_one_plus_dot(const double* a, const double* b, double B,
                                     int n) {
    double lo = 0.0;
    double hi = eshkol_rm_dot_dd(a, b, n, &lo);
    double result = fma(B, lo, fma(B, hi, 1.0));
    return isfinite(result) ? result : 1.0 + eshkol_rm_scaled_dot_factor(a, b, B, n);
}

/** @brief 1 - B|x|^2, the ball chart's conformal denominator, to full relative
 *         precision. */
static double eshkol_rm_one_minus_bnorm2(const double* x, double B, int n) {
    /* Track d = 1 - B|x|^2 as a compensated scaled sum. The direct squared
     * norm can overflow even when the scaled radius is finite, while
     * subtracting a rounded product from one loses all relative digits near
     * the boundary. */
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        double ai = fabs(x[i]);
        if (!isfinite(ai)) return ai;
        if (ai > max_abs) max_abs = ai;
    }
    if (max_abs == 0.0 || B == 0.0) return 1.0;
    double scaled_norm2 = 0.0, compensation = 0.0;
    for (int i = 0; i < n; ++i) {
        double z = x[i] / max_abs;
        double term = z * z;
        double y = term - compensation;
        double sum = scaled_norm2 + y;
        compensation = (sum - scaled_norm2) - y;
        scaled_norm2 = sum;
    }
    double product_hi = 0.0, product_lo = 0.0;
    eshkol_rm_scaled_product4_dd(B, max_abs, max_abs, scaled_norm2,
                                 &product_hi, &product_lo);
    /* The compensated part of the normalized sum is a fifth factor. At the
     * boundary the three-factor product is O(1), so this correction retains
     * the low word without creating an overflowing intermediate. */
    product_lo += eshkol_rm_scaled_product4(B, max_abs, max_abs,
                                            -compensation);
    double rounded = 1.0 - product_hi;
    /* TwoSum(1, -product_hi): spelling the residual as (-product_hi) -
     * (rounded - 1) avoids recomputing 1-rounded, which would repeat the
     * cancellation this branch is preserving. */
    double rounded_minus_one = rounded - 1.0;
    double split = -product_hi - rounded_minus_one;
    double residual = split - product_lo;
    double result = rounded + residual;
    if (isfinite(result)) return result;
    return 1.0 - eshkol_rm_scaled_norm2_times(x, B, n);
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
        double p1 = p * a[i], e1 = fma(p, a[i], -p1);
        double p2 = q * b[i], e2 = fma(q, b[i], -p2);
        double s  = p1 + p2;
        double bb = s - p1;
        double err = (p1 - (s - bb)) + (p2 - bb) + e1 + e2;
        out[i] = s + err;
    }
}

/** @brief The ball parameter B of the chart of curvature -c: the number for
 *         which the ball is |x|^2 < 1/B and Mobius addition is (+)_B. */
static double eshkol_rm_ball_param(double c) {
    return eshkol_rm_scaled_product4(c, ESHKOL_RM_LAMBDA0,
                                     ESHKOL_RM_LAMBDA0, 0.25);
}

/** @brief The conformal factor lambda_x of the metric of curvature @p K at
 *         @p x: g_x = lambda_x^2 <.,.>. THE ONE PLACE this is computed. */
static double eshkol_rm_lambda(const double* x, double K, int n) {
    if (K >= 0.0) return (K == 0.0) ? ESHKOL_RM_FLAT_LAMBDA : 1.0;
    double B = eshkol_rm_ball_param(-K);
    return ESHKOL_RM_LAMBDA0 / eshkol_rm_one_minus_bnorm2(x, B, n);
}

/** @brief The Riemannian norm of tangent vector @p v at @p x. */
static double eshkol_rm_metric_norm(const double* v, const double* x, double K,
                                    int n) {
    double lam = eshkol_rm_lambda(x, K, n);
    double vn = eshkol_rm_norm(v, n);
    return lam * vn;
}

/** @brief tanh(z)/z, analytic at 0 with value 1. The series is used near zero
 *         because the quotient is 0/0 there, which is exactly the c -> 0 corner
 *         of the exponential map. */
static double eshkol_rm_tanh_over(double z) {
    double w = z * z;
    if (w < ESHKOL_RM_TAU_SMALL * ESHKOL_RM_TAU_SMALL)
        return 1.0 - w / 3.0 + 2.0 * w * w / 15.0;
    return tanh(z) / z;
}

/** @brief psi(w) = asinh(sqrt w)/sqrt w, analytic at 0 with value 1, together
 *         with psi'(w) and psi''(w).
 *
 * The hyperbolic distance is 2 sqrt(R) psi(c R) with R = |x-y|^2/P (see
 * eshkol_rm_distance), which is the STABLE form: the arccosh(1 + 2cR) it
 * replaces loses every digit of a small separation to the 1 + eps cancellation
 * (at c = 1, x = 0, y = 1e-9 it returns exactly 0 where the distance is 2e-9),
 * and psi is what makes the K -> 0 corner and the near-coincident corner one
 * expression instead of two special cases.
 *
 * @param d1 psi'(w), may be NULL.
 * @param d2 psi''(w), may be NULL.
 */
static double eshkol_rm_psi(double w, double* d1, double* d2) {
    /* asinh(u) = sum_n (-1)^n (2n)! / (4^n (n!)^2 (2n+1)) u^(2n+1), so
     * psi(w) = sum_n a_n w^n with the same coefficients. */
    static const double a[7] = {
        1.0, -1.0 / 6.0, 3.0 / 40.0, -5.0 / 112.0,
        35.0 / 1152.0, -0.022372159090909091, 0.017352764423076923
    };
    if (w < 0.0) w = 0.0;
    if (w < ESHKOL_RM_PSI_SMALL) {
        /* Horner on the truncated series, and on its two term-by-term
         * derivatives. Exact at w = 0, where the closed forms are 0/0. */
        double psi = a[6];
        double dp  = 6.0 * a[6];
        double dpp = 30.0 * a[6];
        for (int k = 5; k >= 0; k--) psi = psi * w + a[k];
        for (int k = 5; k >= 1; k--) dp  = dp  * w + (double)k * a[k];
        for (int k = 5; k >= 2; k--) dpp = dpp * w + (double)(k * (k - 1)) * a[k];
        if (d1) *d1 = dp;
        if (d2) *d2 = dpp;
        return psi;
    }
    {
        double u = sqrt(w);
        double s = sqrt(1.0 + w);
        double A = asinh(u);
        double N  = u / s - A;                 /* = -u^3/3 + 3u^5/10 - ...   */
        double Np = 1.0 / (s * s * s) - 1.0 / s;
        if (d1) *d1 = N / (2.0 * u * u * u);
        if (d2) *d2 = (Np * u - 3.0 * N) / (4.0 * u * u * u * u * u);
        return A / u;
    }
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
    double root_B = eshkol_rm_sqrt_nonnegative(B);
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
    double root_B = eshkol_rm_sqrt_nonnegative(B);
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
 * @brief Check that @p x is a legal point of the manifold of curvature @p K.
 * @return NULL when it is, else a reason naming what is wrong.
 */
static const char* eshkol_rm_check_point(const double* x, double K, int n) {
    if (!isfinite(K)) return "curvature is not finite";
    for (int i = 0; i < n; i++)
        if (!isfinite(x[i])) return "a coordinate is not finite";
    if (K < 0.0) {
        double B = eshkol_rm_ball_param(-K);
        if (!(eshkol_rm_one_minus_bnorm2(x, B, n) > 0.0))
            return "the point must lie strictly inside the Poincare ball";
    } else if (K > 0.0) {
        double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        if (!isfinite(R)) return "the sphere radius is not finite";
        double xn = eshkol_rm_norm(x, n);
        if (!(fabs(xn - R) <= ESHKOL_RM_SPHERE_TOL * R))
            return "the point must lie ON the sphere of radius 1/sqrt(K) "
                   "(use manifold-project to move it there)";
    }
    return NULL;
}

/**
 * @brief Check that a computed ball point is STRICTLY interior.
 *
 * tanh rounds to exactly 1.0 for arguments beyond about 19, so both the
 * exponential map and Mobius scalar multiplication can land on the boundary --
 * `exp_0(20)` at K = -1 returned exactly 1.0, and `mobius-scalar-mul(100, 0.5,
 * -1)` likewise -- which is not a point of the open ball and is the one place
 * these ops could return a value outside their own codomain. Refusing names the
 * representability limit; returning the boundary point would hand the caller a
 * point at which lambda is infinite and every subsequent log is zero.
 *
 * @return NULL when strictly interior, else a reason.
 */
static const char* eshkol_rm_require_interior(const double* out, double K, int n) {
    if (K >= 0.0) return NULL;
    double B = eshkol_rm_ball_param(-K);
    if (!(eshkol_rm_one_minus_bnorm2(out, B, n) > 0.0))
        return "the result is on or outside the ball boundary: no strictly "
               "interior double-precision point exists at this magnitude";
    return NULL;
}

/**
 * @brief Check that @p v is tangent to the sphere of curvature @p K at @p x.
 *        A no-op for K <= 0, where every vector is tangent.
 * @return NULL when tangent (or K <= 0), else a reason.
 */
static const char* eshkol_rm_check_tangent(const double* x, const double* v,
                                           double K, int n) {
    if (K <= 0.0) return NULL;
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

/* The differentiable spherical distance domain shared by attention forward
 * and reverse.  All calculations use unit vectors, so neither R^2 nor a
 * squared residual is formed; this remains valid when R = 1/sqrt(K) is large
 * but finite. */
static const char* eshkol_rm_sphere_distance_domain(const double* x,
                                                    const double* y,
                                                    double K, int n) {
    if (!(K > 0.0) || !isfinite(K))
        return "spherical distance requires finite positive curvature";
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;
    if (eshkol_rm_points_equal(x, y, n))
        return "the two points coincide: spherical distance has no derivative";
    if (eshkol_rm_sphere_antipodal(x, y, n))
        return "the two points are antipodal: distance is not differentiable";
    const double xn = eshkol_rm_norm(x, n);
    const double yn = eshkol_rm_norm(y, n);
    if (!(xn > 0.0) || !(yn > 0.0))
        return "a spherical distance endpoint is the zero vector";
    double cs = 0.0;
    for (int i = 0; i < n; ++i)
        cs += (x[i] / xn) * (y[i] / yn);
    double sn = 0.0;
    for (int i = 0; i < n; ++i)
        sn = hypot(sn, y[i] / yn - cs * (x[i] / xn));
    if (!(sn > 0.0) || !isfinite(sn) || !isfinite(cs))
        return "the spherical distance derivative is not finite";
    const double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
    const double theta = atan2(sn, cs);
    if (!isfinite(R) || !isfinite(theta) || !isfinite(R * theta))
        return "the spherical distance is not finite";
    return NULL;
}

/* Gradient of the same spherical distance.  Normalising first is algebraically
 * identical to (cs*x-y)/(R*sin(theta)), while avoiding R^2 and R-scaled
 * residual squares at the true-minimum positive curvature. */
static int eshkol_rm_sphere_distance_gradient(const double* x,
                                              const double* y, double K, int n,
                                              double* gx, double* gy) {
    if (eshkol_rm_sphere_distance_domain(x, y, K, n)) return false;
    const double xn = eshkol_rm_norm(x, n);
    const double yn = eshkol_rm_norm(y, n);
    double cs = 0.0;
    for (int i = 0; i < n; ++i)
        cs += (x[i] / xn) * (y[i] / yn);
    double sn = 0.0;
    for (int i = 0; i < n; ++i)
        sn = hypot(sn, y[i] / yn - cs * (x[i] / xn));
    if (!(sn > 0.0) || !isfinite(sn)) return false;
    for (int i = 0; i < n; ++i) {
        const double ux = x[i] / xn;
        const double uy = y[i] / yn;
        gx[i] = (cs * ux - uy) / sn;
        gy[i] = (cs * uy - ux) / sn;
    }
    return eshkol_rm_check_output(gx, n, "the spherical distance gradient is not finite") == NULL &&
           eshkol_rm_check_output(gy, n, "the spherical distance gradient is not finite") == NULL;
}

/**
 * @brief Geodesic distance between @p x and @p y on the manifold of curvature
 *        @p K, into *@p out.
 *
 * Hyperbolic: 2 sqrt(R) psi(B R) scaled by LAMBDA0/2, with R = |x-y|^2/P and
 * P = (1 - B|x|^2)(1 - B|y|^2). Equivalently
 * (LAMBDA0/sqrt B) asinh(sqrt(B) |x-y| / sqrt P), which is the same number as
 * the arccosh(1 + 2 B |x-y|^2 / P)/sqrt(B) it replaces and is STABLE where that
 * form is not: arccosh(1 + eps) throws away every digit of a small separation,
 * returning exactly 0 for two points 2e-9 apart at B = 1.
 *
 * Spherical: R theta with theta = atan2(|y - cos(theta) x| / R, cos(theta)),
 * which is stable near both 0 and pi where acos(<x,y>/R^2) is not.
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

    if (K == 0.0) {
        double delta = 0.0;
        for (int i = 0; i < n; i++) delta = hypot(delta, x[i] - y[i]);
        *out = ESHKOL_RM_FLAT_LAMBDA * delta;
        return isfinite(*out) ? NULL : "the geodesic distance is not finite";
    }
    if (K < 0.0) {
        double B  = eshkol_rm_ball_param(-K);
        double delta = 0.0;
        for (int i = 0; i < n; i++) delta = hypot(delta, x[i] - y[i]);
        double P  = eshkol_rm_one_minus_bnorm2(x, B, n) *
                    eshkol_rm_one_minus_bnorm2(y, B, n);
        double scaled_delta = delta / sqrt(P);
        double ps = eshkol_rm_psi(eshkol_rm_scaled_square(B, scaled_delta), NULL, NULL);
        *out = ESHKOL_RM_LAMBDA0 * scaled_delta * ps;
        if (!isfinite(*out)) return "the geodesic distance is not finite";
        return NULL;
    }
    {
        double R  = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        if (eshkol_rm_sphere_antipodal(x, y, n))
            return "the two points are antipodal: distance is not differentiable";
        *out = R * eshkol_rm_sphere_angle(x, y, R, n, NULL);
        if (!isfinite(*out)) return "the geodesic distance is not finite";
        return NULL;
    }
}

/**
 * @brief Exponential map exp_x(v) on the manifold of curvature @p K.
 *
 * Hyperbolic: x (+)_B [ (lambda_x/LAMBDA0) tanh(z)/z * v ] with
 * z = sqrt(B) lambda_x |v| / LAMBDA0. Written through tanh(z)/z rather than
 * tanh(...)/(sqrt(B)|v|) so that the B -> 0 corner is the same expression
 * (tanh(z)/z -> 1) instead of a 0/0 special case.
 *
 * Spherical: cos(|v|/R) x + R sin(|v|/R) v/|v|, with the tangency of v checked
 * rather than assumed -- a v with a radial component would leave the sphere.
 *
 * @param scratch n doubles.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_exp_map(const double* x, const double* v, double K,
                                     int n, double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    for (int i = 0; i < n; i++)
        if (!isfinite(v[i])) return "a tangent-vector component is not finite";
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
    if (K < 0.0) {
        double B   = eshkol_rm_ball_param(-K);
        double lam = eshkol_rm_lambda(x, K, n);
        double z   = eshkol_rm_sqrt_nonnegative(B) * lam * vn / ESHKOL_RM_LAMBDA0;
        double f   = (lam / ESHKOL_RM_LAMBDA0) * eshkol_rm_tanh_over(z);
        for (int i = 0; i < n; i++) scratch[i] = f * v[i];
        eshkol_rm_mobius_add(x, scratch, B, n, out);
        why = eshkol_rm_check_output(out, n, "the exponential-map result is not finite");
        if (why) return why;
        return eshkol_rm_require_interior(out, K, n);
    }
    {
        double R  = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        double th = vn / R;
        double ca = cos(th), sa = sin(th);
        for (int i = 0; i < n; i++) out[i] = ca * x[i] + R * sa * v[i] / vn;
        return eshkol_rm_check_output(out, n, "the exponential-map result is not finite");
    }
}

/**
 * @brief Logarithmic map log_x(y) on the manifold of curvature @p K, the inverse
 *        of eshkol_rm_exp_map().
 *
 * Hyperbolic: the DIRECTION comes from the Mobius numerator of (-x) (+)_B y and
 * the MAGNITUDE from the geodesic distance, |log_x(y)| = d(x,y)/lambda_x. Both
 * halves matter:
 *
 *   - the direction is taken from the numerator vector alone, never divided by
 *     the Mobius denominator, which is the quantity that goes to zero for two
 *     nearly-antipodal interior points;
 *   - the magnitude is d/lambda_x, computed by eshkol_rm_distance's stable
 *     asinh form, rather than (2/(sqrt(B) lambda_x)) artanh(sqrt(B)|u|).
 *
 * This op used to REFUSE whenever sqrt(B)|u| rounded to 1, on the grounds that
 * no finite log existed. That was wrong: EVERY pair of strictly interior ball
 * points has a unique finite log, and the rounding was a numerical failure of
 * the artanh route, not a statement about the manifold. For
 * x = 0.999999999, y = -0.999999999 at B = 1 the true distance is
 * 4 artanh(x) ~ 42.83; the old code refused and the AD bridge, on the same
 * input, clamped to 1 - 1e-12 and fabricated 28.32.
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
    if (K < 0.0) {
        double B  = eshkol_rm_ball_param(-K);
        double By2 = eshkol_rm_scaled_norm2_times(y, B, n);
        /* Numerator of (-x) (+)_B y; its denominator is positive and only
         * scales the vector, so the direction is this alone. Both coefficients
         * come from the well-conditioned pair (den, 1 - B|x|^2) rather than
         * from their own cancelling sums -- see eshkol_rm_dot_dd. */
        double* V = scratch;
        double nb = eshkol_rm_one_minus_bnorm2(x, B, n);
        double dn = eshkol_rm_mobius_den_negx(x, y, B, n);
        double na = dn + By2 * nb;
        eshkol_rm_axpby_exact(nb, y, -na, x, n, V);
        double Vn = eshkol_rm_norm(V, n);
        double d = 0.0;
        why = eshkol_rm_distance(x, y, K, n, &d);
        if (why) return why;
        double coef = (d / eshkol_rm_lambda(x, K, n)) / Vn;
        for (int i = 0; i < n; i++) out[i] = coef * V[i];
        return eshkol_rm_check_output(out, n, "the logarithmic-map result is not finite");
    }
    {
        /* Sphere: log_x(y) = theta R * u/|u| with u = y - (<x,y>/R^2) x. */
        double R  = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        if (!isfinite(R)) return "the sphere radius is not finite";
        double* u = scratch;
        if (eshkol_rm_sphere_antipodal(x, y, n))
            return "the two points are antipodal: log is not single-valued there";
        double th = eshkol_rm_sphere_angle(x, y, R, n, u);
        double un = eshkol_rm_norm(u, n);
        if (un == 0.0) {
            for (int i = 0; i < n; i++) out[i] = 0.0;
            return NULL;
        }
        double coef = th * R / un;
        for (int i = 0; i < n; i++) out[i] = coef * u[i];
        return eshkol_rm_check_output(out, n, "the logarithmic-map result is not finite");
    }
}

/**
 * @brief Parallel transport of tangent vector @p v from @p x to @p y along the
 *        connecting geodesic.
 *
 * Hyperbolic: P_{x->y}(v) = (lambda_x / lambda_y) gyr[y, -x] v, with the
 * gyration in its LINEAR closed form (eshkol_rm_gyration) rather than expanded
 * through Mobius point addition -- see that function for the counterexample the
 * expansion got wrong.
 *
 * Spherical: P_{x->y}(v) = v - (<y,v>/(R^2 + <x,y>)) (x + y), the closed form,
 * which is exactly tangent at y by construction. Refuses a non-tangent v (the
 * old code returned a radial v unchanged, which is not even tangent at the
 * destination) and refuses antipodal endpoints, where the geodesic and hence
 * the transport is not unique.
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
    if (K < 0.0) {
        double B = eshkol_rm_ball_param(-K);
        double* negx = scratch;
        for (int i = 0; i < n; i++) negx[i] = -x[i];
        eshkol_rm_gyration(y, negx, v, B, n, out);
        double ratio = eshkol_rm_lambda(x, K, n) / eshkol_rm_lambda(y, K, n);
        for (int i = 0; i < n; i++) out[i] *= ratio;
        return eshkol_rm_check_output(out, n, "the parallel-transport result is not finite");
    }
    {
        double R2 = 1.0 / K;
        double xy = eshkol_rm_dot(x, y, n);
        double den = R2 + xy;
        if (!(fabs(den) > ESHKOL_RM_SPHERE_TOL * R2))
            return "the endpoints are antipodal: parallel transport between "
                   "them is not unique";
        double f = eshkol_rm_dot(y, v, n) / den;
        for (int i = 0; i < n; i++) out[i] = v[i] - f * (x[i] + y[i]);
        return eshkol_rm_check_output(out, n, "the parallel-transport result is not finite");
    }
}

/**
 * @brief Mobius scalar multiplication r (x)_B x
 *      = (1/sqrt(B)) tanh(r artanh(sqrt(B)|x|)) x/|x|.
 *
 * @return NULL, or a reason when @p x is not strictly inside the ball, or when
 *         the result would land on it (see eshkol_rm_require_interior).
 */
static const char* eshkol_rm_mobius_scalar(double r, const double* x, double K,
                                           int n, double* out) {
    if (K > 0.0)
        return "Mobius scalar multiplication is a gyrovector operation of the "
               "Poincare ball and is not defined on the sphere";
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    double xn = eshkol_rm_norm(x, n);
    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = r * x[i];
        return eshkol_rm_check_output(out, n, "the Mobius scalar result is not finite");
    }
    if (xn == 0.0) {
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return NULL;
    }
    double B = eshkol_rm_ball_param(-K);
    double s = eshkol_rm_sqrt_nonnegative(B);
    double t = s * xn;
    double coef = tanh(r * atanh(t)) / (s * xn);
    for (int i = 0; i < n; i++) out[i] = coef * x[i];
    why = eshkol_rm_check_output(out, n, "the Mobius scalar result is not finite");
    if (why) return why;
    return eshkol_rm_require_interior(out, K, n);
}

/**
 * @brief Project @p x onto the manifold of curvature @p K.
 *
 * Hyperbolic: rescale onto the open ball when @p x is on or outside it, leaving
 * interior points untouched. Spherical: rescale to radius 1/sqrt(K). Euclidean:
 * a copy.
 *
 * @return NULL on success, else a reason (only when the input cannot be scaled,
 *         i.e. it is the origin on a sphere).
 */
static const char* eshkol_rm_project(const double* x, double K, int n, double* out) {
    if (!isfinite(K)) return "curvature is not finite";
    for (int i = 0; i < n; i++)
        if (!isfinite(x[i])) return "a coordinate is not finite";
    if (K == 0.0) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    if (K < 0.0) {
        double xn = eshkol_rm_norm(x, n);
        double R = 1.0 / eshkol_rm_sqrt_nonnegative(eshkol_rm_ball_param(-K));
        /* Leave a margin so the projected point is STRICTLY inside: a point
         * exactly on the boundary makes lambda infinite and every log map
         * degenerate to zero, which is the failure mode frechet_mean_core.h
         * refuses to let look like convergence. */
        double limit = R * (1.0 - 1e-12);
        if (xn <= limit) {
            memcpy(out, x, (size_t)n * sizeof(double));
            return NULL;
        }
        double s = limit / xn;
        for (int i = 0; i < n; i++) out[i] = s * x[i];
        return NULL;
    }
    {
        double R = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        double max_abs = 0.0;
        for (int i = 0; i < n; i++) {
            if (!isfinite(x[i])) return "a coordinate is not finite";
            if (fabs(x[i]) > max_abs) max_abs = fabs(x[i]);
        }
        if (max_abs == 0.0)
            return "the origin has no projection onto the sphere";
        double scaled_norm2 = 0.0;
        for (int i = 0; i < n; i++) {
            double z = x[i] / max_abs;
            scaled_norm2 += z * z;
        }
        double scaled_norm = sqrt(scaled_norm2);
        if (!(scaled_norm > 0.0) || !isfinite(scaled_norm))
            return "the vector could not be normalised to the sphere";
        double radius_scale = R / scaled_norm;
        for (int i = 0; i < n; i++) out[i] = radius_scale * (x[i] / max_abs);
        for (int i = 0; i < n; i++)
            if (!isfinite(out[i])) return "the projected point is not finite";
        return NULL;
    }
}

/**
 * @brief Convert a Euclidean gradient @p g at @p x into the Riemannian gradient.
 *
 * Conformal branches (K <= 0): g / lambda_x^2, with lambda from the ONE
 * accessor -- so the K = 0 value follows ESHKOL_RM_FLAT_LAMBDA and moves with
 * the convention ruling rather than being open-coded here. Spherical: the
 * ambient gradient projected onto the tangent space at @p x.
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_egrad_to_rgrad(const double* g, const double* x,
                                            double K, int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    if (K <= 0.0) {
        double lam = eshkol_rm_lambda(x, K, n);
        if (!isfinite(lam)) return "the conformal factor is not finite";
        double inv_lam = 1.0 / lam;
        double s = inv_lam * inv_lam;
        for (int i = 0; i < n; i++) out[i] = s * g[i];
        return eshkol_rm_check_output(out, n, "the Riemannian gradient is not finite");
    }
    {
        double gx = eshkol_rm_scaled_dot_factor(g, x, K, n);
        for (int i = 0; i < n; i++) out[i] = g[i] - gx * x[i];
        return eshkol_rm_check_output(out, n, "the Riemannian gradient is not finite");
    }
}

/**
 * @brief Geodesic distance between @p x and @p y AND its first two derivatives
 *        with respect to the sectional curvature K, at fixed points.
 *
 * WHY THIS EXISTS. `curvature-gradient` (852) returned the plain SUM of a
 * tensor's elements, `curvature-hessian` (855) returned the constant 0.0, and
 * `adaptive-curvature-step` (856) moved K by a fixed 0.01 times that sum. None
 * of the three differentiated anything: the first two are not derivatives of
 * any objective, and a Hessian that is identically zero is the assertion that
 * every objective is affine in K, made without looking at one. This function is
 * the measurement they now report -- exact closed-form d/dK and d^2/dK^2, not a
 * difference quotient, so there is no step size to choose and no truncation
 * error to bound.
 *
 * HYPERBOLIC BRANCH (K < 0, c = -K, B = ball_param(c)). With a = |x|^2,
 * b = |y|^2, E = |x-y|^2,
 *
 *   P(B) = (1 - B a)(1 - B b),  Rr = E/P,  w = B Rr,
 *   d    = LAMBDA0 sqrt(Rr) psi(w)
 *
 * and the derivatives are the exact chain rule on that composition. Written
 * through psi rather than arccosh because the two agree exactly and only this
 * one stays accurate as B -> 0 and as the points approach each other.
 * Coincident points are handled separately: E = 0 makes d identically zero in
 * B, so every K-derivative is exactly zero.
 *
 * SPHERICAL BRANCH (K > 0). A point of the sphere of radius 1/sqrt(K) is NOT a
 * point of the sphere of a different radius, so "hold the points fixed and vary
 * K" is not a curve in any single manifold. The family this branch differen-
 * tiates instead holds the pair at FIXED ANGULAR POSITION and lets the radius
 * follow K: with theta fixed, d = theta K^(-1/2), so d' = -theta K^(-3/2)/2 and
 * d'' = 3 theta K^(-5/2)/4.
 *
 * The spherical formulas are published only when both derivatives are
 * representable in f64. At the positive subnormal floor, the fixed-angle
 * analytic derivative can be outside f64 even while d itself is finite; that
 * case is an explicit refusal, never a successful infinity. For theta = 0 the
 * continuous value is d' = d'' = 0, including at that floor.
 *
 * K = 0 IS REFUSED WHILE ESHKOL_RM_FLAT_LAMBDA != ESHKOL_RM_LAMBDA0. The ball
 * branch tends to LAMBDA0 |x-y| as K -> 0-, the flat branch returns
 * FLAT_LAMBDA |x-y|, and the spherical branch diverges as K -> 0+. With the two
 * constants unequal the family is discontinuous there, so no number is the
 * derivative and returning one would be the plausible-wrong-number case this
 * surface exists to exclude. If the convention ruling equalises them the ball
 * branch below is already valid AT B = 0 -- psi is analytic there -- and this
 * refusal becomes removable.
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

    if (K == 0.0 && ESHKOL_RM_FLAT_LAMBDA != ESHKOL_RM_LAMBDA0)
        return "the curvature derivative is not defined at K = 0: the ball "
               "branch tends to ESHKOL_RM_LAMBDA0 |x-y| as K -> 0- and the "
               "K = 0 branch returns ESHKOL_RM_FLAT_LAMBDA |x-y|, so with those "
               "two constants unequal the metric family is discontinuous there";

    double E = eshkol_rm_difference_norm2(x, y, n);

    if (K <= 0.0) {
        if (E <= 0.0) {
            /* Coincident points: d is identically zero in B, so both
             * derivatives are exactly 0 -- not a limit through a 0/0 form. */
            if (d_out)  *d_out  = 0.0;
            *d1_out = 0.0;
            *d2_out = 0.0;
            return NULL;
        }
        double a = eshkol_rm_scaled_norm2_times(x, 1.0, n);
        double b = eshkol_rm_scaled_norm2_times(y, 1.0, n);
        double B = eshkol_rm_ball_param(-K);
        /* dB/dc = LAMBDA0^2/4 and c = -K, so dB/dK = -LAMBDA0^2/4. */
        double dBdK = -(ESHKOL_RM_LAMBDA0 * ESHKOL_RM_LAMBDA0) / 4.0;

        double pa  = eshkol_rm_one_minus_bnorm2(x, B, n);
        double pb  = eshkol_rm_one_minus_bnorm2(y, B, n);
        double P   = pa * pb;
        double Pp  = -a * pb - b * pa;                /* dP/dB   */
        double Ppp = 2.0 * a * b;                     /* d2P/dB2 */
        double Rr  = E / P;
        double Rp  = -E * Pp / (P * P);
        double Rpp = -E * (Ppp * P - 2.0 * Pp * Pp) / (P * P * P);
        double w   = B * Rr;
        double wp  = Rr + B * Rp;
        double wpp = 2.0 * Rp + B * Rpp;

        double psi1 = 0.0, psi2 = 0.0;
        double psi = eshkol_rm_psi(w, &psi1, &psi2);

        double rs  = sqrt(Rr);                        /* Rr^(1/2)  */
        double ri  = 1.0 / rs;                        /* Rr^(-1/2) */
        double ri3 = ri / Rr;                         /* Rr^(-3/2) */

        double d  = ESHKOL_RM_LAMBDA0 * rs * psi;
        /* d/dB of LAMBDA0 Rr^(1/2) psi(w), then dB/dK by the chain rule. */
        double dB1 = ESHKOL_RM_LAMBDA0 *
                     (0.5 * ri * Rp * psi + rs * psi1 * wp);
        double dB2 = ESHKOL_RM_LAMBDA0 *
                     (-0.25 * ri3 * Rp * Rp * psi + 0.5 * ri * Rpp * psi
                      + ri * Rp * psi1 * wp
                      + rs * (psi2 * wp * wp + psi1 * wpp));
        if (d_out) *d_out = d;
        *d1_out = dB1 * dBdK;
        *d2_out = dB2 * dBdK * dBdK;
        if (!isfinite(d) || !isfinite(*d1_out) || !isfinite(*d2_out))
            return "the curvature derivative is not finite";
        return NULL;
    }

    {
        double R  = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        double cs = eshkol_rm_dot(x, y, n) / (R * R);
        double sn = 0.0;
        for (int i = 0; i < n; i++) { double t = y[i] - cs * x[i]; sn += t * t; }
        double th = atan2(sqrt(sn) / R, cs);
        double rk = 1.0 / eshkol_rm_sqrt_nonnegative(K);
        if (d_out) *d_out = th * rk;
        if (th == 0.0) {
            *d1_out = 0.0;
            *d2_out = 0.0;
            return NULL;
        }
        if (K <= DBL_TRUE_MIN)
            return "the spherical curvature derivative is outside f64 range "
                   "at the positive subnormal curvature floor";
        /* The fixed-angle spherical family has d = theta K^(-1/2), so its
         * analytic derivatives are -theta K^(-3/2)/2 and
         * 3 theta K^(-5/2)/4.  At the subnormal end of f64 those values can
         * exceed the result type even though d itself is finite.  Do not
         * publish an infinity as a successful derivative: the caller must
         * see the representability boundary and choose a higher-precision or
         * reparameterised calculation. */
        *d1_out = -0.5 * th * rk / K;
        *d2_out =  0.75 * th * rk / (K * K);
        if (!isfinite(*d1_out) || !isfinite(*d2_out))
            return "the spherical curvature derivative is outside f64 range "
                   "at this positive K (analytic fixed-angle limit is "
                   "unrepresentable)";
        return NULL;
    }
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

static eshkol_rm_directional eshkol_rm_dtanh_over(eshkol_rm_directional z) {
    const double small = ESHKOL_RM_TAU_SMALL;
    const double w = z.value * z.value;
    if (w < small * small) {
        const double value = 1.0 - w / 3.0 + 2.0 * w * w / 15.0;
        const double derivative = (-2.0 * z.value / 3.0 +
                                   8.0 * z.value * w / 15.0) * z.tangent;
        return {value, derivative};
    }
    const double t = tanh(z.value);
    const double value = t / z.value;
    const double derivative = ((1.0 - t * t) * z.value - t) /
                              (z.value * z.value) * z.tangent;
    return {value, derivative};
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
        if (out) *out = ESHKOL_RM_FLAT_LAMBDA * d.value;
        if (dout) *dout = ESHKOL_RM_FLAT_LAMBDA * d.tangent;
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
    eshkol_rm_directional P = eshkol_rm_dmul(px, py);
    eshkol_rm_directional scaled_delta = eshkol_rm_ddiv(
        delta_norm, eshkol_rm_dsqrt(P));
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
        ESHKOL_RM_LAMBDA0 * scaled_delta.value * psi_value,
        ESHKOL_RM_LAMBDA0 *
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
        {ESHKOL_RM_LAMBDA0, 0.0}, nb);
    eshkol_rm_directional z = eshkol_rm_dmul(
        {eshkol_rm_sqrt_nonnegative(B), 0.0}, eshkol_rm_ddiv(
            eshkol_rm_dmul(lambda, vn), {ESHKOL_RM_LAMBDA0, 0.0}));
    eshkol_rm_directional f = eshkol_rm_dmul(
        eshkol_rm_ddiv(lambda, {ESHKOL_RM_LAMBDA0, 0.0}),
        eshkol_rm_dtanh_over(z));
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
                       eshkol_rm_ddiv({ESHKOL_RM_LAMBDA0, 0.0}, nb)), Vn);
    for (int i = 0; i < n; ++i) {
        eshkol_rm_directional result = eshkol_rm_dmul(
            coefficient, V[(size_t)i]);
        if (out) out[i] = shared[(size_t)i];
        if (dout) dout[i] = result.tangent;
    }
    return true;
}

#endif /* __cplusplus */

#endif /* ESHKOL_BACKEND_RIEMANNIAN_CORE_H */
