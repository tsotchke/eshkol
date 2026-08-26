/**
 * @file frechet_mean_core.h
 * @brief The weighted Fréchet (Karcher) mean forward pass on the Poincaré ball,
 *        shared verbatim by every path that computes it.
 *
 * WHY THIS IS A HEADER AND NOT A LIBRARY TU. The two callers cannot link a
 * common object file. `lib/backend/vm_geometric.c` is a unity-build include
 * consumed by `lib/backend/eshkol_vm.c`, which is also built as a single
 * translation unit on its own (the `eshkol-vm-standalone-test` target), so a
 * call from it to an external symbol would not link there. Static functions in
 * a header give both callers ONE source of truth without a link edge.
 *
 * WHY ONE SOURCE OF TRUTH IS LOAD-BEARING HERE, not merely tidy. The derivative
 * of this op (`tensor_frechet_mean_backward`, lib/bridge/tensor_backward.cpp)
 * is taken by implicit differentiation of the stationarity condition
 *
 *     F(mu) = sum_i w_i log_mu(x_i) = 0,
 *
 * which is only valid AT the fixed point. The backward therefore RECOMPUTES the
 * relative residual and refuses if it exceeds its tolerance. The forward gate
 * below and the backward gate are the same formula at the same default bar
 * (1e-9, in Riemannian units) precisely so that a forward which returns
 * successfully produces a mean the derivative will accept. A second, drifting
 * copy of this iteration would produce means that pass one gate and fail the
 * other — or worse, means that pass both while differing, which is the
 * plausible-wrong-number case the whole design exists to exclude.
 *
 * Callers: `lib/backend/vm_geometric.c` (VM opcode 817, `frechet-mean`) and
 * `lib/bridge/qllm_bridge.cpp` (`ad_frechet_mean`, which records the
 * AD_NODE_FRECHET_MEAN tape node the backward differentiates).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_FRECHET_MEAN_CORE_H
#define ESHKOL_BACKEND_FRECHET_MEAN_CORE_H

#include <math.h>

/*******************************************************************************
 * Weighted Fréchet (Karcher) mean on the Poincaré ball
 *
 * frechet-mean(points, weights, curvature) is the Riemannian center of mass:
 *
 *     mu* = argmin_mu sum_i w_i d(mu, x_i)^2,
 *
 * equivalently the solution of the stationarity condition
 *
 *     sum_i w_i log_mu(x_i) = 0.                                          (*)
 *
 * WHAT THIS REPLACES. This op used to return the EUCLIDEAN weighted average
 * sum_i w_i x_i / sum_i w_i, discarding the curvature argument entirely. On the
 * Poincaré ball that is not the Fréchet mean and not an approximation of it —
 * the geodesics are circular arcs orthogonal to the boundary, so the Riemannian
 * center of mass moves away from the chord midpoint, by more and more as the
 * points approach the boundary. It agreed with the real answer only at the
 * origin or at zero curvature. A result that carries a curvature parameter it
 * ignores is the plausible-wrong-number case: nothing in the output shows the
 * argument was dropped.
 *
 * WHY f64 AND NOT THE fp32 PATH. The derivative of this op (see
 * tensor_frechet_mean_backward in lib/bridge/tensor_backward.cpp) is taken by
 * implicit differentiation of (*), which is only valid AT the fixed point, and
 * is therefore gated on the stationarity residual. An fp32 mean carries
 * |mu - mu*| ~ 1e-7, so its residual sits around 1e-7 relative and can never
 * satisfy a 1e-9 gate: an fp32 forward makes the exact derivative unavailable by
 * construction. Both the portable and the linked-library builds therefore use
 * this f64 iteration, which converges to ~1e-16 relative residual on ordinary
 * inputs and leaves the gate satisfiable.
 *
 * Curvature convention: the argument is the sectional curvature K <= 0, matching
 * every in-tree call site (make-hyperbolic-manifold 2 -1.0, poincare-distance
 * ... -1.0). The ball has radius 1/sqrt(-K); K = 0 is the Euclidean case, where
 * the weighted average IS the Fréchet mean and is used exactly.
 ******************************************************************************/

/* Iteration budget and relative residual bar. The bar matches the default
 * tolerance of the backward's gate (kFrechetResidualTol), so a forward that
 * returns successfully produces a mean the derivative will accept. */
#define ESHKOL_FRECHET_MAX_ITERS 256
#define ESHKOL_FRECHET_RESID_TOL 1e-9
/* Consecutive iterations allowed to make no real progress before the run is
 * declared stagnant. See the stagnation comment in the iteration below: a
 * plateau is the residual's own evaluation noise floor, and continuing past it
 * only draws again against the tolerance. */
#define ESHKOL_FRECHET_MAX_STALL 4

static double eshkol_frechet_dot(const double* a, const double* b, int n) {
    double t = 0.0;
    for (int i = 0; i < n; i++) t += a[i] * b[i];
    return t;
}

/** @brief Möbius addition out = a (+)_c x on the ball of curvature -c. */
static void eshkol_frechet_mobius_add(const double* a, const double* x, double c,
                                  int n, double* out) {
    double ax = eshkol_frechet_dot(a, x, n);
    double aa = eshkol_frechet_dot(a, a, n);
    double xx = eshkol_frechet_dot(x, x, n);
    double A1 = 1.0 + 2.0 * c * ax + c * xx;
    double B1 = 1.0 - c * aa;
    double D  = 1.0 + 2.0 * c * ax + c * c * aa * xx;
    if (D == 0.0) D = 1e-300;
    for (int i = 0; i < n; i++) out[i] = (A1 * a[i] + B1 * x[i]) / D;
}

/**
 * @brief log_mu(x) into @p out; scratch must hold n doubles.
 *
 * @return 0 if u = (-mu) (+)_c x lands on or outside the ball boundary in
 *         floating point, in which case artanh has no value and NO finite log
 *         exists to return. This is reachable from operands that are each
 *         strictly inside the ball: |u| is formed by cancellation, so a mu and
 *         an x separated by more than about 19 units of hyperbolic distance
 *         drive |u| to 1 in f64 even though both are interior points. Clamping
 *         sr to 1 - 1e-15 here (what this used to do) substitutes a fabricated
 *         log magnitude of ~17.6 for one that does not exist, and the iteration
 *         then converges on it and reports success — a plausible wrong mean.
 *         The caller turns this into a catchable refusal instead. The backward
 *         rule already refuses on exactly this condition; the two must agree,
 *         or the forward hands out means whose derivative is undefined.
 */
static int eshkol_frechet_log_map(const double* mu, const double* x, double c,
                              int n, double* out, double* scratch) {
    for (int i = 0; i < n; i++) scratch[i] = -mu[i];
    eshkol_frechet_mobius_add(scratch, x, c, n, out);       /* out = u */
    double s = sqrt(c);
    double r = sqrt(eshkol_frechet_dot(out, out, n));
    if (r <= 0.0) {                                     /* x == mu */
        for (int i = 0; i < n; i++) out[i] = 0.0;
        return 1;
    }
    double sr = s * r;
    if (!(sr < 1.0)) return 0;                          /* no finite log exists */
    double k = (1.0 - c * eshkol_frechet_dot(mu, mu, n)) / s;
    double f = k * atanh(sr) / r;
    for (int i = 0; i < n; i++) out[i] *= f;
    return 1;
}

/** @brief exp_mu(v) into @p out; scratch must hold n doubles. */
static void eshkol_frechet_exp_map(const double* mu, const double* v, double c,
                               int n, double* out, double* scratch) {
    double s = sqrt(c);
    double nv = sqrt(eshkol_frechet_dot(v, v, n));
    if (nv <= 0.0) {
        for (int i = 0; i < n; i++) out[i] = mu[i];
        return;
    }
    /* sqrt(c) * lambda_mu / 2 = sqrt(c)/(1 - c|mu|^2) */
    double denom = 1.0 - c * eshkol_frechet_dot(mu, mu, n);
    if (denom <= 0.0) denom = 1e-300;
    double t = tanh(s * nv / denom) / (s * nv);
    for (int i = 0; i < n; i++) scratch[i] = t * v[i];
    eshkol_frechet_mobius_add(mu, scratch, c, n, out);
}

/**
 * @brief Compute the weighted Fréchet mean, or report why it cannot be computed.
 *
 * @param pts      n_points x dim, row-major
 * @param wts      n_points weights (NULL means uniform)
 * @param n_w      number of entries actually available in @p wts
 * @param K        sectional curvature, must be <= 0
 * @param mu       out: dim doubles
 * @param scratch  4 * dim doubles of scratch
 * @param resid_out out: the achieved relative stationarity residual
 * @return NULL on success, else a static human-readable reason. The reason is
 *         returned rather than raised here so the caller owns the VM error
 *         path; @p detail receives the numbers for the message.
 */
static const char* eshkol_frechet_mean_compute(const double* pts, const double* wts,
                                           int64_t n_w, int n_points, int dim,
                                           double K, double* mu, double* scratch,
                                           double* resid_out) {
    *resid_out = 0.0;
    if (n_points <= 0 || dim <= 0) return "needs at least one point of positive dimension";
    if (!(K <= 0.0) || !(K == K))
        return "curvature must be <= 0 (frechet-mean is the hyperbolic/Euclidean "
               "Riemannian center of mass; the ball has radius 1/sqrt(-K))";

    double wsum = 0.0;
    for (int i = 0; i < n_points; i++) {
        double w = (wts && i < n_w) ? wts[i] : 1.0;
        if (!(w == w)) return "a weight is NaN";
        if (w < 0.0) return "weights must be non-negative";
        wsum += w;
    }
    if (!(wsum > 0.0)) return "total weight must be positive (the mean is undefined otherwise)";

    for (int64_t t = 0; t < (int64_t)n_points * dim; t++)
        if (!(pts[t] == pts[t])) return "a point coordinate is NaN";

    /* ---- Euclidean case: the weighted average is exactly the mean ------ */
    if (K == 0.0) {
        for (int k = 0; k < dim; k++) mu[k] = 0.0;
        for (int i = 0; i < n_points; i++) {
            double w = (wts && i < n_w) ? wts[i] : 1.0;
            for (int k = 0; k < dim; k++) mu[k] += w * pts[(int64_t)i * dim + k];
        }
        for (int k = 0; k < dim; k++) mu[k] /= wsum;
        return NULL;
    }

    /* ---- Hyperbolic case ---------------------------------------------- */
    const double c = -K;
    const double radius = 1.0 / sqrt(c);
    for (int i = 0; i < n_points; i++) {
        const double* xi = pts + (int64_t)i * dim;
        if (!(sqrt(eshkol_frechet_dot(xi, xi, dim)) < radius))
            return "every point must lie strictly inside the Poincare ball of "
                   "radius 1/sqrt(-K)";
    }

    double* step    = scratch;
    double* lg      = scratch + dim;
    double* tmp     = scratch + 2 * dim;
    double* best_mu = scratch + 3 * dim;

    /* Seed from the weighted Euclidean average — inside the ball because the
     * ball is convex in the ambient coordinates. */
    for (int k = 0; k < dim; k++) mu[k] = 0.0;
    for (int i = 0; i < n_points; i++) {
        double w = (wts && i < n_w) ? wts[i] : 1.0;
        for (int k = 0; k < dim; k++) mu[k] += w * pts[(int64_t)i * dim + k] / wsum;
    }

    double resid_rel    = 0.0;
    double best_rel     = HUGE_VAL; /* smallest residual seen; best_mu is its iterate */
    double progress_rel = HUGE_VAL; /* reference for the stagnation test */
    int    stall        = 0;        /* consecutive iterations with no real progress */
    int    confirm      = 0;        /* consecutive sub-tolerance iterates */
    int    stagnated    = 0;        /* ended on the residual's noise plateau */
    for (int k = 0; k < dim; k++) best_mu[k] = mu[k];
    for (int it = 0; it < ESHKOL_FRECHET_MAX_ITERS; it++) {
        /* The iterate must stay strictly inside the ball. It should, because the
         * exp map is a Mobius addition and the ball is invariant under it, but a
         * step taken from within one ulp of the boundary can round onto it. That
         * case must not be allowed to look like convergence: at |mu| = 1/sqrt(c)
         * the factor k = (1 - c|mu|^2)/sqrt(c) is zero, so every log_mu(x_i)
         * evaluates to the zero vector and the residual is exactly zero. */
        double mu_sq = eshkol_frechet_dot(mu, mu, dim);
        if (!(c * mu_sq < 1.0))
            return "the iterate reached the Poincare ball boundary, where the log "
                   "map degenerates to zero and a zero residual does not mean "
                   "stationarity";

        for (int k = 0; k < dim; k++) step[k] = 0.0;
        double max_log = 0.0;
        for (int i = 0; i < n_points; i++) {
            double w = (wts && i < n_w) ? wts[i] : 1.0;
            if (!eshkol_frechet_log_map(mu, pts + (int64_t)i * dim, c, dim, lg, tmp))
                return "a log_mu(x_i) has no finite value in f64 — the iterate and "
                       "that point are too far apart in hyperbolic distance for the "
                       "ambient ball coordinates to separate them (refusing rather "
                       "than substituting a fabricated log magnitude)";
            for (int k = 0; k < dim; k++) {
                step[k] += w * lg[k];
                double a = fabs(lg[k]);
                if (a > max_log) max_log = a;
            }
        }
        double resid = sqrt(eshkol_frechet_dot(step, step, dim));
        /* The residual is measured in RIEMANNIAN units, not in the ambient ball
         * coordinates the logs are represented in. The tangent space at mu carries
         * the conformal metric lambda_mu^2 * <.,.> with lambda_mu = 2/(1 - c|mu|^2),
         * so the invariant length of a tangent vector v is lambda_mu * |v|_2.
         *
         * Scaling matters because of the absolute floor. The bar is relative WITH a
         * floor — 1 + max|log|, not max|log| — because when the points coincide with
         * the iterate every log_mu(x_i) is zero to rounding (~1e-19), and a purely
         * relative bar would divide a numerically exact residual by that noise and
         * reject the most exact case there is. But a floor is only meaningful in
         * units where 1 means something, and it does not in the ambient
         * coordinates: as mu approaches the boundary lambda_mu diverges, every
         * ambient |log| collapses toward zero, the floor swamps the relative term,
         * and the bar degenerates to |resid|_ambient <= tol * wsum — which a mean
         * that is wrong by O(1) unit of hyperbolic distance passes easily. That was
         * measured, not feared: with the ambient scale the iteration returned a
         * "converged" mean wrong by 8.8e-8 (points 1e-9 inside the boundary) and by
         * 7.6e-6 (points one ulp inside) and reported success. In Riemannian units
         * 1 means one unit of hyperbolic distance, so the floor keeps protecting the
         * coincident-point case while the relative term stays live near the
         * boundary. The backward rule in lib/bridge/tensor_backward.cpp uses the
         * identical scale — the forward's gate is what makes the backward's gate
         * satisfiable, so they must not drift apart. */
        double lambda = 2.0 / (1.0 - c * mu_sq);
        resid_rel = (lambda * resid) / (wsum * (1.0 + lambda * max_log));
        *resid_out = resid_rel;

        /* ---- Acceptance needs TWO consecutive sub-tolerance iterates ------
         * One sample is not evidence. Near the boundary the residual has an
         * evaluation noise floor of its own — the ambient logs are formed by
         * cancellation, so |F| cannot be resolved below roughly lambda_mu times
         * the rounding error of the log terms. Once the iterate is inside that
         * floor, successive iterations resample rounding noise, and if the loop
         * is allowed to keep drawing, one draw eventually cancels below the bar
         * by luck and the gate passes a mean that is not stationary. Measured on
         * two points 1e-9 inside the boundary: the residual oscillated between
         * 2.4e-7 and 4.9e-6 for 131 iterations and then produced a single
         * 9.86e-10 draw, which the one-sample test accepted; the accepted mean
         * was wrong by 3.0e-8, two orders worse than the bar it had just passed,
         * and an independent evaluation of the same formula at the same point
         * gave 1.2e-6.
         *
         * A genuine fixed point survives the retest: the iteration is locally
         * contracting there, so the next iterate is at least as good. Noise is
         * not reproducible, so demanding two consecutive passes at two distinct
         * iterates squares the probability of a lucky draw. Combined with the
         * stagnation break below it removes the lottery rather than shortening
         * it. The machine-exact case is accepted by the moved == 0 path instead,
         * which is a stronger witness than any residual sample: the iteration
         * literally cannot move.
         *
         * Remember the best ITERATE, not merely the best number: the residual is
         * not monotone in the last few digits, so the second of the two
         * confirming iterates can be marginally worse than the first. Returning
         * the better one keeps the exactly-representable fixtures exact (the
         * weights-3:1 diameter case lands on 0.5 to the last bit) and hands the
         * backward the smallest residual available, which is precisely the
         * quantity the backward's own gate is measured against. */
        if (resid_rel < best_rel) {
            best_rel = resid_rel;
            for (int k = 0; k < dim; k++) best_mu[k] = mu[k];
        }

        /* ---- Acceptance needs THREE consecutive sub-tolerance iterates ------
         * One sample is not evidence, and neither, on every platform, are two.
         * Near the boundary the residual has an evaluation noise floor of its
         * own — the ambient logs are formed by cancellation, so |F| cannot be
         * resolved below roughly lambda_mu times the rounding error of the log
         * terms. Once the iterate is inside that floor, successive iterations
         * resample rounding noise, and if the loop is allowed to keep drawing,
         * a draw eventually cancels below the bar by luck and the gate passes a
         * mean that is not stationary. Measured on two points 1e-9 inside the
         * boundary: the residual oscillated between 2.4e-7 and 4.9e-6 for 131
         * iterations and then produced a single 9.86e-10 draw, which a
         * one-sample test accepted; the accepted mean was wrong by 3.0e-8, two
         * orders worse than the bar it had just passed.
         *
         * Two consecutive draws closes that case but is not architecture-
         * independent: at x0 = 1 - 1e-7 (two points, weights 2:1) one platform's
         * f64 rounding trajectory through this same plateau produces exactly two
         * consecutive sub-tolerance noise draws where another platform's does
         * not, because the dot-product and atanh/tanh evaluations that feed the
         * residual are not required by IEEE 754 to round identically across
         * targets (FMA contraction availability, in particular, differs between
         * baseline aarch64 and baseline x86-64). A criterion whose accept/refuse
         * verdict depends on which CPU it runs on is wrong regardless of which
         * side happens to be "right" on any one machine, so the fix is not to
         * chase a platform-specific threshold — it is to demand a streak long
         * enough that two independent platforms' noise floors are both
         * exceedingly unlikely to satisfy it by chance, which squares again with
         * a third independent draw.
         *
         * A genuine fixed point survives the retest: the iteration is locally
         * contracting there, so the next iterate is at least as good, and a
         * well-inside-the-ball mean (case 6a below) reaches a residual many
         * orders under tolerance long before the loop reaches the stall cap, so
         * three consecutive sub-tolerance samples cost it nothing. The
         * machine-exact case is accepted by the moved == 0 path instead, which
         * is a stronger witness than any residual sample: the iteration
         * literally cannot move. Combined with the stagnation break below,
         * three consecutive draws removes the lottery rather than shortening
         * it. */
        if (resid_rel <= ESHKOL_FRECHET_RESID_TOL) {
            if (++confirm >= 3) break;      /* accepted below, at best_mu */
        } else {
            confirm = 0;
        }

        /* ---- Stagnation is the noise floor, not slow convergence ----------
         * The Karcher iteration is a contraction on a Hadamard manifold, and the
         * diameter representable in f64 ambient ball coordinates is bounded (a
         * point more than about 19 units of hyperbolic distance from the iterate
         * drives |u| to 1 and is refused above), so a genuinely converging run
         * improves its residual by a factor bounded well away from 1 on every
         * iteration. A residual that fails to improve by even 0.1% over four
         * consecutive iterations has therefore reached the floor of its own
         * evaluation, and further iterations buy nothing but fresh draws against
         * the tolerance. Stop, and refuse: `stagnated` records that the run ended
         * on the plateau rather than at a fixed point, so the acceptance test
         * below will not honour a lucky sample taken on the way there. */
        if (resid_rel < progress_rel * (1.0 - 1e-3)) {
            progress_rel = resid_rel;
            stall = 0;
        } else if (++stall >= ESHKOL_FRECHET_MAX_STALL) {
            stagnated = 1;
            break;
        }

        for (int k = 0; k < dim; k++) step[k] /= wsum;
        eshkol_frechet_exp_map(mu, step, c, dim, tmp, lg);
        double moved = 0.0;
        for (int k = 0; k < dim; k++) {
            double dk = fabs(tmp[k] - mu[k]);
            if (dk > moved) moved = dk;
            mu[k] = tmp[k];
        }
        /* A step that does not move the iterate at all cannot improve the
         * residual either; stop rather than spin out the budget. This is not a
         * failure: an iterate the map cannot leave at f64 resolution is a fixed
         * point to the precision available, and it is a reproducible witness
         * rather than a sample, so it needs no confirming second draw. */
        if (moved == 0.0) break;
    }

    /* ---- Acceptance -------------------------------------------------------
     * Report the best iterate found, and only if the run did not end on the
     * residual's noise plateau. A stagnated run may well have drawn a single
     * sub-tolerance residual on its way across the plateau; honouring that is
     * exactly the lottery the confirmation rule exists to close, so the
     * stagnation flag vetoes it regardless of best_rel. */
    for (int k = 0; k < dim; k++) mu[k] = best_mu[k];
    *resid_out = best_rel;
    if (!stagnated && best_rel <= ESHKOL_FRECHET_RESID_TOL) return NULL;
    return stagnated
        ? "the Karcher iteration stagnated above the stationarity tolerance — the "
          "residual stopped improving, which means it reached the precision floor "
          "of its own evaluation in f64 ambient ball coordinates (these points are "
          "too close to the boundary for their Frechet mean to be resolved)"
        : "the Karcher iteration did not converge";
}

#endif /* ESHKOL_BACKEND_FRECHET_MEAN_CORE_H */
