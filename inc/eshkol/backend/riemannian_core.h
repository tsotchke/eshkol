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
 * body has since been deleted outright (it did not compile against the current
 * libsemiclassical_qllm ABI and was fp32 throughout), so the forms below are
 * the ONE implementation of this geometry on the VM engine, not the default of
 * two.
 *
 * WHY A HEADER AND NOT A LIBRARY TU. Identical reason to
 * `inc/eshkol/backend/frechet_mean_core.h`: `lib/backend/vm_geometric.c` is a
 * unity-build include consumed by `lib/backend/eshkol_vm.c`, which is also built
 * as a single translation unit on its own (the `eshkol-vm-standalone-test`
 * target), so a call to an external symbol would not link there. Static
 * functions in a header give every caller ONE source of truth with no link edge.
 *
 * CURVATURE CONVENTION. Every entry point below takes @c K, the SECTIONAL
 * CURVATURE, and dispatches on its sign:
 *
 *   K < 0   Poincare ball of curvature K, i.e. ball parameter c = -K and
 *           Euclidean radius 1/sqrt(c). This is the convention every in-tree
 *           call site already uses -- `(make-hyperbolic-manifold 2 -1.0)`,
 *           `(poincare-distance x y -1.0)` -- and the one
 *           `eshkol_frechet_mean_compute` takes, so `frechet-mean` and the ops
 *           around it now read their curvature argument the same way.
 *   K = 0   Euclidean space. Every operation reduces to its flat form EXACTLY
 *           (exp is addition, log is subtraction, distance is L2, transport is
 *           the identity), which is not an approximation but the c -> 0 limit of
 *           the formulas below. The old flat bodies were therefore right for
 *           this one value of the argument and wrong for every other.
 *   K > 0   Round sphere of radius R = 1/sqrt(K), points required to lie ON it.
 *
 * RELATION TO THE AD BRIDGE. `lib/bridge/qllm_bridge.cpp` (`ad_hyperbolic_distance`,
 * `ad_poincare_exp_map`, `ad_poincare_log_map`, `ad_geodesic_attention`) is the
 * reference semantics for the hyperbolic case, and the formulas here are the
 * same ones: Mobius addition with the same denominator floor, exp with
 * coefficient tanh(sqrt(c) lambda_x |v| / 2)/(sqrt(c)|v|), log with
 * (2/(sqrt(c) lambda_x)) artanh(sqrt(c)|u|)/|u|, distance as
 * arccosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2)))/sqrt(c). The bridge takes its
 * argument as `c = (curvature == 0) ? 1 : |curvature|` because its callers pass a
 * ball parameter; passing THIS file's K = -c to those entry points yields
 * bit-comparable results on the hyperbolic branch, which is what the VM's ops
 * are checked against.
 *
 * ONE DELIBERATE DIVERGENCE FROM THE BRIDGE. `ad_poincare_log_map` clamps
 * sqrt(c)|u| to 1 - 1e-12 when it reaches 1. That substitutes a fabricated log
 * magnitude (~13.8) for one that does not exist, which is exactly what
 * `eshkol_frechet_log_map` refuses to do and says why. This file refuses too:
 * `eshkol_rm_log_map` returns a reason and the VM raises a catchable condition.
 * The bridge's clamp is a separate defect on a separate surface and is left for
 * its own change rather than altered from here.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_RIEMANNIAN_CORE_H
#define ESHKOL_BACKEND_RIEMANNIAN_CORE_H

#include <math.h>
#include <string.h>

/* Below this Euclidean norm a tangent vector is treated as zero: exp returns its
 * base point and log returns the zero vector. Matches the bridge's 1e-15. */
#define ESHKOL_RM_ZERO_NORM 1e-15

/* Floor on the MAGNITUDE of the Mobius denominator. Not a perturbation of an
 * input and not a difference quotient: the Mobius quotient is undefined where
 * the denominator vanishes, and this keeps the value finite with the sign the
 * expression had. Same constant and same convention as `kMobiusDenFloor` in
 * lib/bridge/qllm_bridge.cpp and the Poincare-ball divisor clamps in
 * lib/backend/autodiff_codegen.cpp. */
#define ESHKOL_RM_MOBIUS_DEN_FLOOR 1e-15

/* Relative tolerance on |x| = R for a point claimed to lie on the sphere of
 * radius R. A point off the sphere has no geodesic relation to another point on
 * it, so the ops refuse rather than project silently; `manifold-project` is the
 * op that moves a point onto the manifold. */
#define ESHKOL_RM_SPHERE_TOL 1e-9

static double eshkol_rm_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double eshkol_rm_norm(const double* a, int n) {
    return sqrt(eshkol_rm_dot(a, a, n));
}

/**
 * @brief Mobius addition on the Poincare ball of ball parameter @p c > 0:
 *
 *   x (+)_c y = ((1 + 2c<x,y> + c|y|^2) x + (1 - c|x|^2) y)
 *               / (1 + 2c<x,y> + c^2 |x|^2 |y|^2)
 *
 * At c = 0 this is x + y, which is why the Euclidean branch of every caller is
 * the same code with c = 0 rather than a separate special case.
 */
static void eshkol_rm_mobius_add(const double* x, const double* y, double c,
                                 int n, double* out) {
    double xy = eshkol_rm_dot(x, y, n);
    double x2 = eshkol_rm_dot(x, x, n);
    double y2 = eshkol_rm_dot(y, y, n);
    double num_x = 1.0 + 2.0 * c * xy + c * y2;
    double num_y = 1.0 - c * x2;
    double den_raw = 1.0 + 2.0 * c * xy + c * c * x2 * y2;
    double den = den_raw;
    if (fabs(den_raw) < ESHKOL_RM_MOBIUS_DEN_FLOOR)
        den = (den_raw < 0.0) ? -ESHKOL_RM_MOBIUS_DEN_FLOOR : ESHKOL_RM_MOBIUS_DEN_FLOOR;
    for (int i = 0; i < n; i++) out[i] = (num_x * x[i] + num_y * y[i]) / den;
}

/**
 * @brief Mobius scalar multiplication r (x)_c x
 *      = (1/sqrt(c)) tanh(r artanh(sqrt(c)|x|)) x/|x|.
 *
 * @return NULL, or a reason when @p x is not strictly inside the ball (artanh
 *         has no value at or beyond the boundary).
 */
static const char* eshkol_rm_mobius_scalar(double r, const double* x, double c,
                                           int n, double* out) {
    double xn = eshkol_rm_norm(x, n);
    if (c == 0.0) {
        for (int i = 0; i < n; i++) out[i] = r * x[i];
        return NULL;
    }
    if (xn < ESHKOL_RM_ZERO_NORM) {
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return NULL;
    }
    double s = sqrt(c);
    double t = s * xn;
    if (!(t < 1.0))
        return "the point is on or outside the Poincare ball boundary, where "
               "artanh has no value";
    double coef = tanh(r * atanh(t)) / (s * xn);
    for (int i = 0; i < n; i++) out[i] = coef * x[i];
    return NULL;
}

/**
 * @brief Check that @p x is a legal point of the manifold of curvature @p K.
 * @return NULL when it is, else a reason naming what is wrong.
 */
static const char* eshkol_rm_check_point(const double* x, double K, int n) {
    if (!(K == K)) return "curvature is NaN";
    for (int i = 0; i < n; i++)
        if (!(x[i] == x[i])) return "a coordinate is NaN";
    if (K < 0.0) {
        double c = -K;
        if (!(c * eshkol_rm_dot(x, x, n) < 1.0))
            return "the point must lie strictly inside the Poincare ball of "
                   "radius 1/sqrt(-K)";
    } else if (K > 0.0) {
        double R = 1.0 / sqrt(K);
        double xn = eshkol_rm_norm(x, n);
        if (!(fabs(xn - R) <= ESHKOL_RM_SPHERE_TOL * R))
            return "the point must lie ON the sphere of radius 1/sqrt(K) "
                   "(use manifold-project to move it there)";
    }
    return NULL;
}

/**
 * @brief Geodesic distance between @p x and @p y on the manifold of curvature
 *        @p K, into *@p out.
 *
 * Hyperbolic: arccosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2))) / sqrt(c), the
 * form `ad_hyperbolic_distance` computes. Spherical: R arccos(<x,y>/R^2).
 * Euclidean: |x-y|.
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_distance(const double* x, const double* y, double K,
                                      int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;

    if (K == 0.0) {
        double s = 0.0;
        for (int i = 0; i < n; i++) { double d = x[i] - y[i]; s += d * d; }
        *out = sqrt(s);
        return NULL;
    }
    if (K < 0.0) {
        double c = -K;
        double diff2 = 0.0;
        for (int i = 0; i < n; i++) { double d = x[i] - y[i]; diff2 += d * d; }
        double dx = 1.0 - c * eshkol_rm_dot(x, x, n);
        double dy = 1.0 - c * eshkol_rm_dot(y, y, n);
        double arg = 1.0 + 2.0 * c * diff2 / (dx * dy);
        if (arg < 1.0) arg = 1.0;   /* only reachable by rounding below 1 */
        *out = acosh(arg) / sqrt(c);
        return NULL;
    }
    {
        double R = 1.0 / sqrt(K);
        double cs = eshkol_rm_dot(x, y, n) / (R * R);
        if (cs > 1.0) cs = 1.0;
        if (cs < -1.0) cs = -1.0;
        *out = R * acos(cs);
        return NULL;
    }
}

/**
 * @brief Exponential map exp_x(v) on the manifold of curvature @p K.
 *
 * @param scratch n doubles.
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_exp_map(const double* x, const double* v, double K,
                                     int n, double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    for (int i = 0; i < n; i++)
        if (!(v[i] == v[i])) return "a tangent-vector component is NaN";

    double vn = eshkol_rm_norm(v, n);
    if (vn < ESHKOL_RM_ZERO_NORM) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = x[i] + v[i];
        return NULL;
    }
    if (K < 0.0) {
        double c = -K;
        double s = sqrt(c);
        double lam = 2.0 / (1.0 - c * eshkol_rm_dot(x, x, n));
        double coef = tanh(s * lam * vn / 2.0) / (s * vn);
        for (int i = 0; i < n; i++) scratch[i] = coef * v[i];
        eshkol_rm_mobius_add(x, scratch, c, n, out);
        return NULL;
    }
    {
        /* Sphere of radius R: exp_x(v) = cos(|v|/R) x + R sin(|v|/R) v/|v|.
         * The tangency of v is not assumed -- a v with a radial component would
         * leave the sphere, so it is rejected rather than silently projected. */
        double R = 1.0 / sqrt(K);
        double radial = eshkol_rm_dot(x, v, n) / R;
        if (!(fabs(radial) <= ESHKOL_RM_SPHERE_TOL * (vn + R)))
            return "the tangent vector is not tangent to the sphere (<x,v> must "
                   "vanish)";
        double th = vn / R;
        double ca = cos(th), sa = sin(th);
        for (int i = 0; i < n; i++) out[i] = ca * x[i] + R * sa * v[i] / vn;
        return NULL;
    }
}

/**
 * @brief Logarithmic map log_x(y) on the manifold of curvature @p K, the inverse
 *        of eshkol_rm_exp_map().
 *
 * @param scratch 2n doubles.
 * @return NULL on success, else a reason. The hyperbolic branch REFUSES when
 *         sqrt(c)|(-x) (+)_c y| reaches 1: no finite log exists there, and
 *         clamping the argument would return a fabricated magnitude the caller
 *         cannot distinguish from a real one. Reachable from two points each
 *         strictly inside the ball -- |u| is formed by cancellation, so points
 *         about 19 units of hyperbolic distance apart drive it to 1 in f64.
 */
static const char* eshkol_rm_log_map(const double* x, const double* y, double K,
                                     int n, double* out, double* scratch) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    why = eshkol_rm_check_point(y, K, n);
    if (why) return why;

    if (K == 0.0) {
        for (int i = 0; i < n; i++) out[i] = y[i] - x[i];
        return NULL;
    }
    if (K < 0.0) {
        double c = -K;
        double s = sqrt(c);
        double* neg = scratch;
        double* u   = scratch + n;
        for (int i = 0; i < n; i++) neg[i] = -x[i];
        eshkol_rm_mobius_add(neg, y, c, n, u);
        double un = eshkol_rm_norm(u, n);
        if (un < ESHKOL_RM_ZERO_NORM) {
            for (int i = 0; i < n; i++) out[i] = 0.0;
            return NULL;
        }
        double t = s * un;
        if (!(t < 1.0))
            return "log has no finite value here: the two points are too far "
                   "apart in hyperbolic distance for the ambient ball "
                   "coordinates to separate them (refusing rather than "
                   "substituting a fabricated log magnitude)";
        double lam = 2.0 / (1.0 - c * eshkol_rm_dot(x, x, n));
        double coef = (2.0 / (s * lam)) * atanh(t) / un;
        for (int i = 0; i < n; i++) out[i] = coef * u[i];
        return NULL;
    }
    {
        /* Sphere: log_x(y) = theta R * u/|u| with u = y - (<x,y>/R^2) x. */
        double R = 1.0 / sqrt(K);
        double cs = eshkol_rm_dot(x, y, n) / (R * R);
        if (cs > 1.0) cs = 1.0;
        if (cs < -1.0) cs = -1.0;
        double* u = scratch;
        for (int i = 0; i < n; i++) u[i] = y[i] - cs * x[i];
        double un = eshkol_rm_norm(u, n);
        double th = acos(cs);
        if (un < ESHKOL_RM_ZERO_NORM) {
            if (th > 1.0) return "the two points are antipodal: log is not "
                                 "single-valued there";
            for (int i = 0; i < n; i++) out[i] = 0.0;
            return NULL;
        }
        double coef = th * R / un;
        for (int i = 0; i < n; i++) out[i] = coef * u[i];
        return NULL;
    }
}

/**
 * @brief Parallel transport of tangent vector @p v from @p x to @p y along the
 *        connecting geodesic.
 *
 * Hyperbolic: P_{x->y}(v) = (lambda_x / lambda_y) gyr[y, -x] v, with the
 * gyration expanded through Mobius addition as
 * gyr[u,w]z = (-(u (+) w)) (+) (u (+) (w (+) z)). Spherical: rotation of the
 * component of v along the geodesic direction by the arc angle. Euclidean: the
 * identity -- which is what this op used to return for EVERY curvature.
 *
 * @param scratch 5n doubles.
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

    if (K == 0.0) {
        memcpy(out, v, (size_t)n * sizeof(double));
        return NULL;
    }
    if (K < 0.0) {
        double c  = -K;
        double* negx = scratch;
        double* t1   = scratch + n;
        double* t2   = scratch + 2 * n;
        double* t3   = scratch + 3 * n;
        double* negt1 = scratch + 4 * n;
        for (int i = 0; i < n; i++) negx[i] = -x[i];
        eshkol_rm_mobius_add(y, negx, c, n, t1);       /* y (+) (-x)          */
        eshkol_rm_mobius_add(negx, v, c, n, t2);       /* (-x) (+) v          */
        eshkol_rm_mobius_add(y, t2, c, n, t3);         /* y (+) ((-x) (+) v)  */
        for (int i = 0; i < n; i++) negt1[i] = -t1[i];
        eshkol_rm_mobius_add(negt1, t3, c, n, out);    /* gyr[y,-x] v         */
        double lam_x = 2.0 / (1.0 - c * eshkol_rm_dot(x, x, n));
        double lam_y = 2.0 / (1.0 - c * eshkol_rm_dot(y, y, n));
        double ratio = lam_x / lam_y;
        for (int i = 0; i < n; i++) out[i] *= ratio;
        return NULL;
    }
    {
        double R = 1.0 / sqrt(K);
        double* lg = scratch;                 /* log_x(y), needs 2n scratch    */
        const char* e = eshkol_rm_log_map(x, y, K, n, lg, scratch + n);
        if (e) return e;
        double d = eshkol_rm_norm(lg, n);
        if (d < ESHKOL_RM_ZERO_NORM) {
            memcpy(out, v, (size_t)n * sizeof(double));
            return NULL;
        }
        double th = d / R;
        double* u = scratch + 3 * n;
        for (int i = 0; i < n; i++) u[i] = lg[i] / d;   /* unit initial direction */
        double vu = eshkol_rm_dot(v, u, n);
        double ca = cos(th), sa = sin(th);
        for (int i = 0; i < n; i++)
            out[i] = v[i] + vu * ((ca - 1.0) * u[i] - sa * x[i] / R);
        return NULL;
    }
}

/**
 * @brief Project @p x onto the manifold of curvature @p K.
 *
 * Hyperbolic: rescale onto the open ball of radius 1/sqrt(c) when @p x is on or
 * outside it, leaving interior points untouched. Spherical: rescale to radius
 * 1/sqrt(K). Euclidean: a copy.
 *
 * @return NULL on success, else a reason (only when the input cannot be scaled,
 *         i.e. it is the origin on a sphere).
 */
static const char* eshkol_rm_project(const double* x, double K, int n, double* out) {
    for (int i = 0; i < n; i++)
        if (!(x[i] == x[i])) return "a coordinate is NaN";
    if (K == 0.0) {
        memcpy(out, x, (size_t)n * sizeof(double));
        return NULL;
    }
    double xn = eshkol_rm_norm(x, n);
    if (K < 0.0) {
        double R = 1.0 / sqrt(-K);
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
        double R = 1.0 / sqrt(K);
        if (xn < ESHKOL_RM_ZERO_NORM)
            return "the origin has no projection onto the sphere";
        double s = R / xn;
        for (int i = 0; i < n; i++) out[i] = s * x[i];
        return NULL;
    }
}

/**
 * @brief Convert a Euclidean gradient @p g at @p x into the Riemannian gradient.
 *
 * Hyperbolic: the ball metric is conformal, g_x = lambda_x^2 <.,.>, so the
 * Riemannian gradient is g / lambda_x^2 = ((1 - c|x|^2)^2 / 4) g. Spherical: the
 * ambient gradient projected onto the tangent space at @p x. Euclidean: a copy
 * -- which is what this op used to return for every curvature.
 *
 * @return NULL on success, else a reason.
 */
static const char* eshkol_rm_egrad_to_rgrad(const double* g, const double* x,
                                            double K, int n, double* out) {
    const char* why = eshkol_rm_check_point(x, K, n);
    if (why) return why;
    if (K == 0.0) {
        memcpy(out, g, (size_t)n * sizeof(double));
        return NULL;
    }
    if (K < 0.0) {
        double c = -K;
        double f = 1.0 - c * eshkol_rm_dot(x, x, n);
        double s = (f * f) / 4.0;
        for (int i = 0; i < n; i++) out[i] = s * g[i];
        return NULL;
    }
    {
        double R = 1.0 / sqrt(K);
        double gx = eshkol_rm_dot(g, x, n) / (R * R);
        for (int i = 0; i < n; i++) out[i] = g[i] - gx * x[i];
        return NULL;
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
 * the measurement they now report -- the exact closed-form d/dK and d^2/dK^2 of
 * the geodesic distance, not a difference quotient, so there is no step size to
 * choose and no truncation error to bound.
 *
 * HYPERBOLIC BRANCH (K < 0, c = -K). With a = |x|^2, b = |y|^2, D = |x-y|^2,
 *
 *   P(c) = (1 - c a)(1 - c b),  Q(c) = c / P(c),  A(c) = 1 + 2 D Q(c),
 *   d(c) = arccosh(A) / sqrt(c),
 *
 * and the derivatives are the exact chain rule on that composition (W = arccosh
 * A below). Coincident points are handled separately: D = 0 makes d identically
 * zero in c, so every K-derivative is exactly zero, and taking the limit through
 * the formula would divide by sqrt(A^2 - 1) = 0.
 *
 * SPHERICAL BRANCH (K > 0). A point of the sphere of radius 1/sqrt(K) is NOT a
 * point of the sphere of a different radius, so "hold the points fixed and vary
 * K" is not a curve in any single manifold. The family this branch differen-
 * tiates instead holds the pair at FIXED ANGULAR POSITION and lets the radius
 * follow K, which is the only reparametrisation under which the objective is
 * defined in a neighbourhood of K: with theta = arccos(<x,y>/R^2) fixed,
 * d = theta K^(-1/2), so d' = -theta K^(-3/2)/2 and d'' = 3 theta K^(-5/2)/4.
 *
 * K = 0 IS REFUSED, and the refusal is not squeamishness. The two curved
 * branches do not agree with the flat one in the limit: the ball model of
 * curvature -c has conformal factor lambda_x = 2/(1 - c|x|^2), so its distance
 * tends to 2|x-y| as c -> 0, twice the Euclidean value the K = 0 branch of
 * eshkol_rm_distance returns, while the spherical branch diverges. The family
 * is genuinely discontinuous at K = 0, so no number is the derivative there and
 * returning one would be the plausible-wrong-number case this surface exists to
 * exclude.
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

    if (K == 0.0)
        return "the curvature derivative is not defined at K = 0: the ball "
               "branch tends to 2|x-y| as K -> 0- (its conformal factor is "
               "lambda = 2 at the origin) and the spherical branch diverges as "
               "K -> 0+, so the metric family is discontinuous there";

    double D = 0.0;
    for (int i = 0; i < n; i++) { double t = x[i] - y[i]; D += t * t; }

    if (K < 0.0) {
        double c = -K;
        if (D <= 0.0) {
            /* Coincident points: d(c) = 0 identically, so both derivatives are
             * exactly 0 -- not a limit taken through a 0/0 form. */
            if (d_out)  *d_out  = 0.0;
            *d1_out = 0.0;
            *d2_out = 0.0;
            return NULL;
        }
        double a = eshkol_rm_dot(x, x, n);
        double b = eshkol_rm_dot(y, y, n);
        double P   = (1.0 - c * a) * (1.0 - c * b);
        double Pp  = -(a + b) + 2.0 * c * a * b;
        double N   = P - c * Pp;
        double Np  = -2.0 * a * b * c;                 /* = -c * P'' */
        double Q   = c / P;
        double Qp  = N / (P * P);
        double Qpp = (Np * P - 2.0 * N * Pp) / (P * P * P);
        double A   = 1.0 + 2.0 * D * Q;
        double Ap  = 2.0 * D * Qp;
        double App = 2.0 * D * Qpp;
        double s1  = sqrt(A * A - 1.0);
        if (!(s1 > 0.0))
            return "the two points are numerically coincident in the ball "
                   "metric, where the curvature derivative degenerates";
        double W   = acosh(A);
        double W1  = Ap / s1;
        double W2  = App / s1 - Ap * Ap * A / (s1 * s1 * s1);
        double rc  = 1.0 / sqrt(c);
        /* d/dc, then dK = -dc: the first derivative changes sign, the second
         * does not. */
        double dc1 = W1 * rc - 0.5 * W * rc / c;
        double dc2 = W2 * rc - W1 * rc / c + 0.75 * W * rc / (c * c);
        if (d_out) *d_out = W * rc;
        *d1_out = -dc1;
        *d2_out = dc2;
        return NULL;
    }

    {
        double R  = 1.0 / sqrt(K);
        double cs = eshkol_rm_dot(x, y, n) / (R * R);
        if (cs > 1.0) cs = 1.0;
        if (cs < -1.0) cs = -1.0;
        double th = acos(cs);
        double rk = 1.0 / sqrt(K);
        if (d_out) *d_out = th * rk;
        *d1_out = -0.5 * th * rk / K;
        *d2_out =  0.75 * th * rk / (K * K);
        return NULL;
    }
}

#endif /* ESHKOL_BACKEND_RIEMANNIAN_CORE_H */
