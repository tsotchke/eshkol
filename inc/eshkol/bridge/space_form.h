/**
 * @file space_form.h
 * @brief Exact squared geodesic distance on the space forms, and on products
 *        of them, as a differentiable AD primitive.
 *
 * WHY THIS PRIMITIVE EXISTS, AND WHY IT IS NOT `d` SQUARED
 *
 * On a two-point homogeneous space — `R^n`, `S^n`, `H^n`, and Riemannian
 * products of them factorwise — any pairwise score invariant under the
 * isometry group is a function of the geodesic distance alone, and any such
 * function that is twice differentiable ACROSS THE DIAGONAL must factor
 * through `d^2` on these spaces.
 * So `d^2` is not one admissible scoring primitive among several: it is the
 * one every admissible score is built from.
 *
 * The distance `d` itself has a cone point at `x == y`. Approaching along the
 * geodesic `exp_y(s u)` gives `grad d = u` for every `s > 0`, so the limit
 * depends on the direction of approach and no derivative exists — the
 * Riemannian norm stays exactly 1 all the way down while the directions stay
 * `pi` apart. `ad_hyperbolic_distance` refuses there, and that refusal is
 * correct for `d`.
 *
 * It is WRONG for `d^2`, and this file exists so that the two are not
 * conflated. `grad_x d^2(x,y) = -2 log_x(y)`, the Riemannian logarithm, which
 * is smooth on the whole injectivity ball and vanishes exactly at coincidence.
 * The rules below therefore evaluate the log-map form DIRECTLY. They never
 * form `d` and square it, and they never differentiate `sqrt(d^2)`: doing
 * either reintroduces a `0/0` at the diagonal that the mathematics does not
 * have.
 *
 * NUMERICAL CONTRACT AT THE DIAGONAL
 *
 * Every formula here is written in the separation `delta = y - x` so that
 * coincidence is reached by `delta` becoming zero rather than by two `O(1)`
 * quantities cancelling:
 *
 *   hyperbolic  (Poincare ball, signed sectional curvature K < 0)
 *       c = -K, B = c * ESHKOL_RM_LAMBDA0^2 / 4, and
 *       P = (1-B|x|^2)(1-B|y|^2), E = |delta|^2
 *       d^2 = ESHKOL_RM_LAMBDA0^2 * (E/P) * psi(B*E/P)^2,
 *       psi(z) = asinh(sqrt(z))/sqrt(z), psi(0) = 1.
 *       The shared Riemannian core evaluates this form and its log map with
 *       full-relative-precision boundary denominators; it never rejects a
 *       valid pair because an intermediate artanh argument rounded to 1.
 *
 *   spherical   (signed sectional curvature K > 0, radius R = 1/sqrt(K))
 *       points must already lie on the sphere within the shared core's
 *       tolerance; invalid points are refused rather than projected.
 *       1 - cos(theta) = D2/(2R^2) exactly, u_x = delta + (D2/(2R^2))*x
 *       theta = atan2(|u_x|/R, cos theta),  d^2 = R^2 theta^2
 *       grad_x d^2 = -2 log_x(y), with the exact antipode refused.
 *
 *   euclidean
 *       d^2 = |delta|^2, grad_x = 2*(x - y). Exact to floating point, and
 *       the flat factor of any product.
 *
 * In all three, `x == y` bitwise makes `delta` exactly zero, which makes every
 * gradient exactly `+/-0.0` with no epsilon guard doing the work. That is the
 * property the composite scoring function needs: `softmax(-d^2/2t)` over a
 * self-attention row contains an exact `q == k` entry in every row.
 *
 * PRODUCTS
 *
 * `d^2_M = sum_f w_f d^2_f` over index-contiguous factors, which is the
 * squared distance of the genuine Riemannian product metric with factor `f`'s
 * metric rescaled by `w_f`. The backward is the concatenation of `w_f` times
 * each factor's own gradient: no cross terms exist, because the factors share
 * no coordinates.
 *
 * WHAT IS STILL REFUSED
 *
 * Coincidence is fine. The CUT LOCUS is not: on `S^n` the antipode is at
 * distance `pi R`, `log_x` is undefined there (every direction is a minimising
 * geodesic), and `d^2` genuinely has no derivative. Every representable
 * `theta < pi` is answered; only the actual/representationally singular
 * antipode is refused. `H^n` and `R^n` have infinite injectivity radius and
 * refuse only for arguments outside the model (a point on or beyond the ball
 * boundary).
 *
 */

#ifndef ESHKOL_BRIDGE_SPACE_FORM_H
#define ESHKOL_BRIDGE_SPACE_FORM_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque handle to an Eshkol automatic-differentiation tape. */
typedef struct ad_tape ad_tape_t;

/** @brief Opaque handle to a node on an Eshkol AD tape (a recorded value). */
typedef struct ad_node ad_node_t;

/**
 * @brief Which of the three space forms a product factor is.
 *
 * These are the complete list of simply-connected constant-curvature
 * manifolds, and — together with their products — the complete list of
 * two-point homogeneous spaces for which the forced-form argument above
 * applies.
 */
typedef enum {
    /** Flat `R^n`. `d^2 = |x-y|^2`. Curvature ignored. */
    ESHKOL_SPACE_FORM_EUCLIDEAN = 0,
    /** `H^n` in the Poincare ball model of signed sectional curvature `K < 0`.
     *  Points live in the open ball of radius `1/sqrt(-K)` at the shipped
     *  chart normalisation. */
    ESHKOL_SPACE_FORM_HYPERBOLIC = 1,
    /** `S^n` of signed sectional curvature `K > 0`, as the sphere of radius
     *  `1/sqrt(K)` in ambient `R^{n+1}`. `dim` counts AMBIENT coordinates,
     *  so the round 2-sphere is `dim = 3`. Arguments must already be on it. */
    ESHKOL_SPACE_FORM_SPHERICAL = 2
} eshkol_space_form_t;

/**
 * @brief One factor of a product manifold.
 *
 * Factors partition the coordinate vector in order: factor `0` owns
 * coordinates `[0, dim_0)`, factor `1` owns `[dim_0, dim_0 + dim_1)`, and so
 * on. The sum of `dim` over all factors must equal the element count of both
 * point tensors.
 */
typedef struct {
    /** One of `eshkol_space_form_t`. */
    int32_t form;
    /** Reserved; must be zero. Keeps the struct's layout explicit across the
     *  language boundary rather than dependent on enum-size choices. */
    int32_t reserved;
    /** Number of AMBIENT coordinates this factor owns. */
    int64_t dim;
    /** Signed sectional curvature `K`: exactly `0` for Euclidean, negative for
     *  hyperbolic, and positive for spherical. A form/sign mismatch and every
     *  non-finite value are rejected. */
    double curvature;
    /** Finite `w_f >= 0` in `d^2_M = sum_f w_f d^2_f`. Zero is a genuine
     *  zero-weight factor and contributes neither value nor gradient. */
    double weight;
} eshkol_manifold_factor_t;

/**
 * @brief Squared geodesic distance on a single space form.
 *
 * `x` and `y` are tensors of identical element count; the result is a scalar
 * (shape `[1]`) node of type `AD_NODE_SQUARED_DISTANCE`.
 *
 * DIFFERENTIABLE AT `x == y`, unlike `ad_hyperbolic_distance`. The gradient
 * there is exactly zero, which is the correct value and not a stand-in for a
 * missing one: `grad_x d^2 = -2 log_x(y)` and `log_x(x) = 0`.
 *
 * @param tape       tape to record on, or NULL to compute the forward only
 * @param x          first point
 * @param y          second point
 * @param form       one of `eshkol_space_form_t`
 * @param curvature  signed sectional curvature `K`; its sign must agree with
 *                   `form` (Euclidean 0, hyperbolic negative, spherical positive)
 * @return the node, or NULL if the arguments are rejected (mismatched shapes,
 *         a point outside the Poincare ball, a spherical argument not on the
 *         required sphere, an antipodal pair, non-finite input, or a
 *         form/curvature mismatch). Rejection is reported through `eshkol_error`.
 */
ad_node_t* ad_squared_distance(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* y,
    int form,
    double curvature
);

/**
 * @brief Squared geodesic distance on a product of space forms.
 *
 * `d^2_M(x,y) = sum_f w_f * d^2_f(x_f, y_f)` over index-contiguous factors.
 * Composition is a plain sum because the product metric is block diagonal, so
 * the backward carries no cross terms and each factor's rule is the
 * single-factor rule above scaled by `w_f`.
 *
 * @param tape          tape to record on, or NULL for forward only
 * @param x             first point, ambient coordinates of all factors
 * @param y             second point, same layout
 * @param factors       factor descriptors, in coordinate order
 * @param num_factors   number of descriptors; must be >= 1
 * @return the node, or NULL if the arguments are rejected
 */
ad_node_t* ad_product_squared_distance(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* y,
    const eshkol_manifold_factor_t* factors,
    size_t num_factors
);

/**
 * @brief Riemannian logarithm on a single space form, into `out`.
 *
 * Exposed because it IS the gradient rule: `grad_x d^2 = -2 log_x(y)` in the
 * Riemannian metric, and the coordinate gradient the tape needs is that vector
 * pushed through the metric tensor. It is here for callers that need the log
 * map in its own right — the Frechet-mean stationarity condition and the
 * space-form Hessian are both written in terms of it — rather than to save
 * anyone a derivation.
 *
 * A gradcheck should NOT use it as its reference. It shares the file, and very
 * nearly the algebra, with the backward it would be checking; the test in
 * tests/bridge/squared_distance_gradcheck_test.cpp writes its own log map from
 * the Mobius/Ganea form for that reason, and checks THIS one against that.
 *
 * @return false if the pair is outside the model or beyond the injectivity
 *         radius, in which case `out` is untouched.
 */
bool eshkol_space_form_log_map(
    int form,
    double curvature,
    const double* x,
    const double* y,
    size_t n,
    double* out
);

/**
 * @brief Geodesic distance on a single space form.
 *
 * Provided for tests and for callers that need the metric itself. NOT used by
 * the squared-distance forward, which never forms `d`: `d^2` is computed in
 * the log-map form directly so that no `sqrt` sits between the value and its
 * derivative.
 *
 * @return a negative value if the pair is outside the model.
 */
double eshkol_space_form_distance(
    int form,
    double curvature,
    const double* x,
    const double* y,
    size_t n
);

/**
 * @brief Fraction of `pi` beyond which the spherical rule refuses.
 *
 * The cut locus of `S^n` is the antipode, at `theta = pi`. `log_x` has no
 * value there and `d^2` no derivative. The implementation refuses only the
 * actual/representationally singular antipode; every representable `theta <
 * pi` is answered.
 */
#define ESHKOL_SPHERE_INJECTIVITY_FRACTION 1.0

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BRIDGE_SPACE_FORM_H */
