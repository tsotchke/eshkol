/**
 * @file space_form_ad.cpp
 * @brief Exact squared geodesic distance on the space forms and their
 *        products, as a differentiable AD primitive.
 *
 * The mathematics, the numerical contract at the diagonal, and the reason this
 * node must NOT inherit `ad_hyperbolic_distance`'s coincidence refusal are all
 * stated in inc/eshkol/bridge/space_form.h. This file is the implementation of
 * exactly what that header specifies.
 *
 * TWO THINGS THIS FILE DELIBERATELY DOES NOT DO
 *
 * 1. It never computes `d` and squares it, and it never differentiates
 *    `sqrt(d^2)`. `d^2` is assembled from the log-map form directly. The
 *    difference matters only at one place and it is the place that matters:
 *    `d(d^2)/dx = 2*d*grad d` has a `0/0` at `x == y` that the mathematics
 *    does not have, and any implementation routed through `sqrt` inherits it.
 *    `eshkol_space_form_distance` exists for callers who want the metric, and
 *    nothing in the forward or backward calls it.
 *
 * 2. It contains no difference quotient. Every derivative here is a closed
 *    form. The file is listed in `.icc/ad-carrier-manifest.yaml`'s
 *    `fd_scan.paths` so that this stays true rather than being asserted.
 *
 */

#include <cmath>
#include <cstddef>
#include <cstring>

#include "eshkol/eshkol.h"
#include "eshkol/logger.h"
#include "eshkol/bridge/space_form.h"

/*******************************************************************************
 * Runtime allocation surface
 *
 * Declared locally, the same convention lib/bridge/qllm_bridge.cpp and
 * lib/bridge/tensor_backward.cpp use, so the bridge does not pull in the full
 * arena/codegen headers.
 ******************************************************************************/

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
    void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
}

namespace {

/** @brief Zero-initialised double buffer from the global arena. */
double* alloc_doubles(size_t n) {
    if (n == 0) return nullptr;
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/** @brief Product of a shape's dimensions. */
size_t elem_count(const int64_t* shape, size_t ndim) {
    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) return 0;
        n *= (size_t)shape[i];
    }
    return n;
}

/** @brief Validate that a node carries a usable tensor payload. */
bool has_tensor(const ad_node_t* n) {
    return n && n->tensor_value && n->ndim > 0 && n->shape;
}

/**
 * @brief Give a node a persistent, zero-filled gradient buffer.
 *
 * Same reason as lib/bridge/qllm_bridge.cpp's ensure_grad(): the backward
 * dispatcher runs each rule inside an arena scope that is rewound on return,
 * so a gradient buffer allocated lazily *inside* a rule is reclaimed the
 * instant that rule finishes and the next node upstream reads freed memory.
 * Allocating at forward time, outside any backward scope, is what makes a
 * recorded chain actually differentiable.
 */
void ensure_grad(ad_node_t* n) {
    if (!n || n->tensor_gradient || !n->shape || n->ndim == 0) return;
    size_t count = elem_count(n->shape, n->ndim);
    if (count == 0) return;
    n->tensor_gradient = arena_allocate_zeroed(get_global_arena(),
                                               count * sizeof(double));
}

/** @brief `artanh(t)/t`, the smooth even function behind the ball formulas.
 *
 *  `A(0) = 1` exactly. For `t` small enough that `artanh(t)` returns `t`
 *  unchanged the quotient is 1 to the last bit, so no series expansion is
 *  needed — only the `t == 0` case, where the quotient is `0/0`. */
double artanh_over_t(double t) {
    if (t == 0.0) return 1.0;
    return std::atanh(t) / t;
}

/** @brief `theta/sin(theta)` expressed through the chord, for the sphere.
 *
 *  Given `sn = sin(theta)` and `theta` itself, this is a plain quotient
 *  everywhere except `sn == 0`, which is coincidence and where the limit is 1.
 */
double theta_over_sin(double theta, double sn) {
    if (sn == 0.0) return 1.0;
    return theta / sn;
}

/*******************************************************************************
 * Single-factor kernels
 *
 * Each returns `false` and leaves its outputs untouched when the pair is
 * outside the model or beyond the injectivity radius. `grad_x` / `grad_y` are
 * the COORDINATE gradients — the metric tensor has already been applied, which
 * is what the reverse tape accumulates. The Riemannian gradient the theory
 * states, `-2 log_x(y)`, is `g^{-1}` times what comes out here, and
 * `sf_log_map` below returns exactly that log so the identity can be checked
 * against an independent evaluation rather than against a rearrangement of
 * these same lines.
 ******************************************************************************/

struct SfPair {
    double d2;          /* squared geodesic distance of this factor */
    bool   ok;          /* false: outside the model / past the cut locus */
};

/** @brief Flat `R^n`. `d^2 = |x-y|^2`, `grad_x = 2(x-y)`, exact to fp. */
SfPair euclidean_sq(const double* x, const double* y, size_t n,
                    double* grad_x, double* grad_y) {
    double d2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double delta = x[i] - y[i];
        d2 += delta * delta;
        if (grad_x) grad_x[i] =  2.0 * delta;
        if (grad_y) grad_y[i] = -2.0 * delta;
    }
    return SfPair{ d2, true };
}

/**
 * @brief `H^n` in the Poincare ball of curvature `-c`.
 *
 * Written entirely in `delta = y - x` so coincidence is reached by `delta`
 * going to zero rather than by two `O(1)` quantities cancelling. The two
 * identities that make this possible:
 *
 *     den   = (1-c|x|^2)(1-c|y|^2) + c|delta|^2      [ = the Mobius denominator
 *                                                       1 - 2c<x,y> + c^2|x|^2|y|^2,
 *                                                       but with every term
 *                                                       positive ]
 *     u     = (-x) (+)_c y = (alpha*delta - c*D2*x) / den
 *     |u|^2 = D2 / den                                [ exact ]
 *
 * The second is the standard expansion of Mobius addition with the `x` terms
 * collected; the third follows from `d = (2/sqrt c) artanh(sqrt c |u|)` and
 * `d = (1/sqrt c) arcosh(1 + 2c D2/(alpha beta))` being the same number.
 *
 * Then `d^2 = 4 A^2 D2 / den` with `A = artanh(t)/t`, `t = sqrt(c D2/den)`,
 * `log_x(y) = alpha * A * u`, and, since the ball metric is conformal with
 * `lambda_x = 2/alpha`,
 *
 *     grad_x d^2 = lambda_x^2 * (-2 log_x(y)) = -(8A/(alpha*den)) * num_x
 *
 * where `num_x = alpha*delta - c*D2*x`. Nothing in that chain divides by
 * anything that vanishes at `x == y`: `den >= alpha*beta > 0` strictly inside
 * the ball, and `A(0) = 1`.
 */
SfPair hyperbolic_sq(const double* x, const double* y, size_t n, double c,
                     double* grad_x, double* grad_y) {
    double x2 = 0.0, y2 = 0.0, D2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x2 += x[i] * x[i];
        y2 += y[i] * y[i];
        double delta = y[i] - x[i];
        D2 += delta * delta;
    }
    double alpha = 1.0 - c * x2;
    double beta  = 1.0 - c * y2;
    if (!(alpha > 0.0) || !(beta > 0.0)) return SfPair{ 0.0, false };

    double den = alpha * beta + c * D2;
    /* den > 0 whenever alpha and beta are, so this is a structural check that
     * a NaN upstream has not made it here, not a clamp on a real zero. */
    if (!(den > 0.0)) return SfPair{ 0.0, false };

    double t = std::sqrt(c * D2 / den);
    if (!(t < 1.0)) return SfPair{ 0.0, false };   /* |u| at or past the boundary */
    double A = artanh_over_t(t);

    double d2 = 4.0 * A * A * D2 / den;

    if (grad_x || grad_y) {
        double kx = -8.0 * A / (alpha * den);
        double ky = -8.0 * A / (beta  * den);
        for (size_t i = 0; i < n; ++i) {
            double delta = y[i] - x[i];
            if (grad_x) grad_x[i] = kx * (alpha * delta - c * D2 * x[i]);
            /* By the symmetry d^2(x,y) = d^2(y,x): swap the roles, which flips
             * delta's sign and exchanges alpha for beta. */
            if (grad_y) grad_y[i] = ky * (-beta * delta - c * D2 * y[i]);
        }
    }
    return SfPair{ d2, true };
}

/**
 * @brief `S^n` of curvature `+c`, radius `R = 1/sqrt(c)`, ambient `R^{n+1}`.
 *
 * Arguments are normalised onto the sphere first. That is not a convenience:
 * it makes the forward 0-homogeneous in each argument, so its ambient gradient
 * has no radial component and is exactly the tangent vector `-2 log_x(y)`
 * rather than that plus a normal term the caller would have to project away.
 *
 * Cancellation-free again, this time through `1 - cos(theta) = D2/(2R^2)`:
 *
 *     u_x   = y - cos(theta) x = delta + (D2/(2R^2)) x       [ delta = y - x ]
 *     |u_x| = R sin(theta)
 *     theta = atan2(|u_x|/R, cos theta)                       [ well conditioned
 *                                                               at BOTH ends,
 *                                                               unlike arccos ]
 *     d^2   = R^2 theta^2
 *     grad_x d^2 = -(2R/|x|_raw) * log_x(y)
 *                = -(2R^2/|x|_raw) * (theta/|u_x|) * u_x
 *
 * `theta/|u_x| = (1/R) * theta/sin(theta)` tends to `1/R`, so the gradient
 * tends to `-(2R/|x|) u_x`, which goes to zero with `u_x`. Smooth.
 *
 * REFUSES near the antipode. That is the cut locus, `log_x` has no value there
 * because every direction is a minimising geodesic, and `d^2` genuinely has no
 * derivative — a different refusal from the coincidence one, and a correct one.
 */
SfPair spherical_sq(const double* x, const double* y, size_t n, double c,
                    double* grad_x, double* grad_y) {
    double R = 1.0 / std::sqrt(c);

    double nx2 = 0.0, ny2 = 0.0;
    for (size_t i = 0; i < n; ++i) { nx2 += x[i] * x[i]; ny2 += y[i] * y[i]; }
    if (!(nx2 > 0.0) || !(ny2 > 0.0)) return SfPair{ 0.0, false };
    double nx = std::sqrt(nx2), ny = std::sqrt(ny2);

    /* Projected points, on the sphere of radius R. */
    double sx = R / nx, sy = R / ny;

    double D2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double delta = y[i] * sy - x[i] * sx;
        D2 += delta * delta;
    }
    double half = D2 / (2.0 * R * R);      /* = 1 - cos(theta), exactly */
    double ca = 1.0 - half;

    /* |u_x| = |u_y| = R sin(theta): compute it from the cancellation-free u. */
    double un2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double delta = y[i] * sy - x[i] * sx;
        double ui = delta + half * (x[i] * sx);
        un2 += ui * ui;
    }
    double un = std::sqrt(un2);
    double theta = std::atan2(un / R, ca);

    if (theta >= ESHKOL_SPHERE_INJECTIVITY_FRACTION * M_PI) {
        return SfPair{ 0.0, false };       /* at or past the cut locus */
    }

    double d2 = R * R * theta * theta;

    if (grad_x || grad_y) {
        /* theta/|u_x| = (1/R) * theta/sin(theta), finite and equal to 1/R at
         * coincidence. Taking the limit here rather than dividing by |u_x| is
         * what keeps the diagonal a value rather than a NaN. */
        double sn = un / R;
        double ratio = theta_over_sin(theta, sn) / R;   /* = theta/|u_x| */
        double kx = -(2.0 * R * R / nx) * ratio;
        double ky = -(2.0 * R * R / ny) * ratio;
        for (size_t i = 0; i < n; ++i) {
            double delta = y[i] * sy - x[i] * sx;
            if (grad_x) grad_x[i] = kx * ( delta + half * (x[i] * sx));
            if (grad_y) grad_y[i] = ky * (-delta + half * (y[i] * sy));
        }
    }
    return SfPair{ d2, true };
}

/** @brief Route one factor to its kernel. */
SfPair factor_sq(int form, double c, const double* x, const double* y, size_t n,
                 double* grad_x, double* grad_y) {
    switch (form) {
        case ESHKOL_SPACE_FORM_EUCLIDEAN:  return euclidean_sq(x, y, n, grad_x, grad_y);
        case ESHKOL_SPACE_FORM_HYPERBOLIC: return hyperbolic_sq(x, y, n, c, grad_x, grad_y);
        case ESHKOL_SPACE_FORM_SPHERICAL:  return spherical_sq(x, y, n, c, grad_x, grad_y);
        default: return SfPair{ 0.0, false };
    }
}

/** @brief Normalise a descriptor's curvature and weight to their defaults. */
double eff_curvature(int form, double c) {
    if (form == ESHKOL_SPACE_FORM_EUCLIDEAN) return 0.0;
    double a = std::fabs(c);
    return (a == 0.0) ? 1.0 : a;
}
double eff_weight(double w) { return (w == 0.0) ? 1.0 : w; }

/**
 * @brief Validate a factor list against a point's element count.
 * @return total ambient dimension, or 0 on rejection (already reported).
 */
size_t validate_factors(const eshkol_manifold_factor_t* f, size_t k, size_t n) {
    if (!f || k == 0) {
        eshkol_error("space form: squared distance needs at least one manifold factor");
        return 0;
    }
    size_t total = 0;
    for (size_t i = 0; i < k; ++i) {
        if (f[i].dim <= 0) {
            eshkol_error("space form: factor %zu has non-positive dimension %lld",
                         i, (long long)f[i].dim);
            return 0;
        }
        if (f[i].form != ESHKOL_SPACE_FORM_EUCLIDEAN &&
            f[i].form != ESHKOL_SPACE_FORM_HYPERBOLIC &&
            f[i].form != ESHKOL_SPACE_FORM_SPHERICAL) {
            eshkol_error("space form: factor %zu names unknown space form %d",
                         i, (int)f[i].form);
            return 0;
        }
        if (f[i].weight < 0.0) {
            eshkol_error("space form: factor %zu has negative weight %g; "
                         "d^2_M = sum_f w_f d^2_f is a metric only for w_f >= 0",
                         i, f[i].weight);
            return 0;
        }
        total += (size_t)f[i].dim;
    }
    if (total != n) {
        eshkol_error("space form: factor dimensions sum to %zu but the points "
                     "carry %zu coordinates", total, n);
        return 0;
    }
    return total;
}

/**
 * @brief The whole forward, shared by the producer and the backward.
 *
 * The backward re-runs it because the gradient of a product is factorwise and
 * costs the same arithmetic as the value: there is nothing worth saving on the
 * node beyond the factor list itself, and re-deriving from the recorded inputs
 * removes any chance of the saved intermediates and the inputs disagreeing.
 *
 * @return false if any factor rejected the pair; `*out_d2` and the gradient
 *         buffers are then meaningless.
 */
bool product_forward(const eshkol_manifold_factor_t* factors, size_t k,
                     const double* X, const double* Y, size_t n,
                     double* out_d2, double* grad_x, double* grad_y,
                     size_t* out_bad_factor) {
    double acc = 0.0;
    size_t off = 0;
    for (size_t i = 0; i < k; ++i) {
        size_t dim = (size_t)factors[i].dim;
        double c = eff_curvature(factors[i].form, factors[i].curvature);
        double w = eff_weight(factors[i].weight);
        SfPair r = factor_sq(factors[i].form, c, X + off, Y + off, dim,
                             grad_x ? grad_x + off : nullptr,
                             grad_y ? grad_y + off : nullptr);
        if (!r.ok) {
            if (out_bad_factor) *out_bad_factor = i;
            return false;
        }
        acc += w * r.d2;
        /* The product metric is block diagonal: factor f's coordinates appear
         * in no other factor's distance, so w_f scales that block of the
         * gradient and there are no cross terms to add. */
        if (w != 1.0) {
            for (size_t j = 0; j < dim; ++j) {
                if (grad_x) grad_x[off + j] *= w;
                if (grad_y) grad_y[off + j] *= w;
            }
        }
        off += dim;
    }
    if (out_d2) *out_d2 = acc;
    (void)n;
    return true;
}

/** @brief Human-readable form name for diagnostics. */
const char* form_name(int form) {
    switch (form) {
        case ESHKOL_SPACE_FORM_EUCLIDEAN:  return "euclidean";
        case ESHKOL_SPACE_FORM_HYPERBOLIC: return "hyperbolic";
        case ESHKOL_SPACE_FORM_SPHERICAL:  return "spherical";
        default: return "unknown";
    }
}

} /* namespace */

/*******************************************************************************
 * Public kernels (no tape)
 ******************************************************************************/

extern "C" bool eshkol_space_form_log_map(int form, double curvature,
                                          const double* x, const double* y,
                                          size_t n, double* out) {
    if (!x || !y || !out || n == 0) return false;
    double c = eff_curvature(form, curvature);

    switch (form) {
        case ESHKOL_SPACE_FORM_EUCLIDEAN:
            for (size_t i = 0; i < n; ++i) out[i] = y[i] - x[i];
            return true;

        case ESHKOL_SPACE_FORM_HYPERBOLIC: {
            /* log_x(y) = alpha * A * u, the same alpha/A/u hyperbolic_sq uses.
             * Kept as its own evaluation so a caller checking
             * grad_x d^2 == -2 lambda_x^2 log_x(y) is comparing two
             * expressions, not one expression against itself. */
            double x2 = 0.0, y2 = 0.0, D2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                x2 += x[i] * x[i];
                y2 += y[i] * y[i];
                double d = y[i] - x[i];
                D2 += d * d;
            }
            double alpha = 1.0 - c * x2, beta = 1.0 - c * y2;
            if (!(alpha > 0.0) || !(beta > 0.0)) return false;
            double den = alpha * beta + c * D2;
            if (!(den > 0.0)) return false;
            double t = std::sqrt(c * D2 / den);
            if (!(t < 1.0)) return false;
            double A = artanh_over_t(t);
            for (size_t i = 0; i < n; ++i)
                out[i] = alpha * A * (alpha * (y[i] - x[i]) - c * D2 * x[i]) / den;
            return true;
        }

        case ESHKOL_SPACE_FORM_SPHERICAL: {
            double R = 1.0 / std::sqrt(c);
            double nx2 = 0.0, ny2 = 0.0;
            for (size_t i = 0; i < n; ++i) { nx2 += x[i]*x[i]; ny2 += y[i]*y[i]; }
            if (!(nx2 > 0.0) || !(ny2 > 0.0)) return false;
            double sx = R / std::sqrt(nx2), sy = R / std::sqrt(ny2);
            double D2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double d = y[i]*sy - x[i]*sx;
                D2 += d * d;
            }
            double half = D2 / (2.0 * R * R);
            double ca = 1.0 - half;
            double un2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double ui = (y[i]*sy - x[i]*sx) + half * (x[i]*sx);
                un2 += ui * ui;
            }
            double un = std::sqrt(un2);
            double theta = std::atan2(un / R, ca);
            if (theta >= ESHKOL_SPHERE_INJECTIVITY_FRACTION * M_PI) return false;
            double ratio = theta_over_sin(theta, un / R) / R;   /* theta/|u| */
            for (size_t i = 0; i < n; ++i)
                out[i] = (R * R) * ratio * ((y[i]*sy - x[i]*sx) + half * (x[i]*sx));
            return true;
        }

        default:
            return false;
    }
}

extern "C" double eshkol_space_form_distance(int form, double curvature,
                                             const double* x, const double* y,
                                             size_t n) {
    if (!x || !y || n == 0) return -1.0;
    double c = eff_curvature(form, curvature);
    switch (form) {
        case ESHKOL_SPACE_FORM_EUCLIDEAN: {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) { double d = x[i]-y[i]; s += d*d; }
            return std::sqrt(s);
        }
        case ESHKOL_SPACE_FORM_HYPERBOLIC: {
            double x2 = 0.0, y2 = 0.0, D2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                x2 += x[i]*x[i]; y2 += y[i]*y[i];
                double d = y[i]-x[i]; D2 += d*d;
            }
            double alpha = 1.0 - c*x2, beta = 1.0 - c*y2;
            if (!(alpha > 0.0) || !(beta > 0.0)) return -1.0;
            double den = alpha*beta + c*D2;
            double t = std::sqrt(c * D2 / den);
            if (!(t < 1.0)) return -1.0;
            return (2.0 / std::sqrt(c)) * std::atanh(t);
        }
        case ESHKOL_SPACE_FORM_SPHERICAL: {
            double R = 1.0 / std::sqrt(c);
            double nx2 = 0.0, ny2 = 0.0;
            for (size_t i = 0; i < n; ++i) { nx2 += x[i]*x[i]; ny2 += y[i]*y[i]; }
            if (!(nx2 > 0.0) || !(ny2 > 0.0)) return -1.0;
            double sx = R/std::sqrt(nx2), sy = R/std::sqrt(ny2);
            double D2 = 0.0;
            for (size_t i = 0; i < n; ++i) { double d = y[i]*sy - x[i]*sx; D2 += d*d; }
            double half = D2 / (2.0*R*R);
            double un2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double ui = (y[i]*sy - x[i]*sx) + half*(x[i]*sx);
                un2 += ui*ui;
            }
            return R * std::atan2(std::sqrt(un2)/R, 1.0 - half);
        }
        default:
            return -1.0;
    }
}

/*******************************************************************************
 * Producers
 ******************************************************************************/

namespace {

/**
 * @brief Allocate the scalar node and attach the factor list.
 *
 * The factor list does not fit in `ad_node_t::params` (a product of three
 * factors is 96 bytes against the union's 48), so it is retained through
 * `saved_tensors[0]` with `num_saved = 1` — the same slot
 * `ad_tensor_attention` uses to retain its softmax weights. `params.axis`
 * carries the factor count so the backward can read the array without a
 * second allocation to describe it.
 */
ad_node_t* make_sqdist_node(ad_tape_t* tape, ad_node_t* x, ad_node_t* y,
                            double d2,
                            const eshkol_manifold_factor_t* factors, size_t k) {
    double* out = alloc_doubles(1);
    if (!out) return nullptr;
    out[0] = d2;

    int64_t* shape = (int64_t*)arena_allocate_zeroed(get_global_arena(),
                                                     sizeof(int64_t));
    if (!shape) return nullptr;
    shape[0] = 1;

    eshkol_manifold_factor_t* saved_factors =
        (eshkol_manifold_factor_t*)arena_allocate_zeroed(
            get_global_arena(), k * sizeof(eshkol_manifold_factor_t));
    void** saved = (void**)arena_allocate_zeroed(get_global_arena(), sizeof(void*));
    if (!saved_factors || !saved) {
        eshkol_error("space form: could not retain the factor list; refusing to "
                     "record a node whose backward would not know its manifold");
        return nullptr;
    }
    std::memcpy(saved_factors, factors, k * sizeof(eshkol_manifold_factor_t));
    saved[0] = (void*)saved_factors;

    ad_node_t* node = arena_allocate_ad_node(get_global_arena());
    if (!node) {
        eshkol_error("space form: failed to allocate AD node for squared distance");
        return nullptr;
    }
    node->type = AD_NODE_SQUARED_DISTANCE;
    node->tensor_value = (void*)out;
    node->value = d2;                 /* scalar mirror, as make_node does for [1] */
    node->shape = shape;
    node->ndim = 1;
    node->input1 = x;
    node->input2 = y;
    node->saved_tensors = saved;
    node->num_saved = 1;
    node->params.axis = (int64_t)k;

    if (tape) {
        node->id = tape->num_nodes;
        arena_tape_add_node(tape, node);
        ensure_grad(node);
        ensure_grad(x);
        ensure_grad(y);
    }
    return node;
}

} /* namespace */

extern "C" ad_node_t* ad_product_squared_distance(
        ad_tape_t* tape, ad_node_t* x, ad_node_t* y,
        const eshkol_manifold_factor_t* factors, size_t num_factors) {
    if (!has_tensor(x) || !has_tensor(y)) {
        eshkol_error("space form: ad_product_squared_distance needs two points");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    if (n == 0 || n != elem_count(y->shape, y->ndim)) {
        eshkol_error("space form: ad_product_squared_distance point dimensions differ");
        return nullptr;
    }
    if (validate_factors(factors, num_factors, n) == 0) return nullptr;

    const double* X = (const double*)x->tensor_value;
    const double* Y = (const double*)y->tensor_value;

    double d2 = 0.0;
    size_t bad = 0;
    if (!product_forward(factors, num_factors, X, Y, n, &d2,
                         nullptr, nullptr, &bad)) {
        eshkol_error("space form: factor %zu (%s, curvature %g) rejected the pair "
                     "— a point outside the model, or a separation at or beyond "
                     "the injectivity radius where no logarithm exists",
                     bad, form_name(factors[bad].form),
                     eff_curvature(factors[bad].form, factors[bad].curvature));
        return nullptr;
    }
    return make_sqdist_node(tape, x, y, d2, factors, num_factors);
}

extern "C" ad_node_t* ad_squared_distance(ad_tape_t* tape, ad_node_t* x,
                                          ad_node_t* y, int form,
                                          double curvature) {
    if (!has_tensor(x)) {
        eshkol_error("space form: ad_squared_distance needs two points");
        return nullptr;
    }
    eshkol_manifold_factor_t f;
    f.form = form;
    f.reserved = 0;
    f.dim = (int64_t)elem_count(x->shape, x->ndim);
    f.curvature = curvature;
    f.weight = 1.0;
    return ad_product_squared_distance(tape, x, y, &f, 1);
}

/*******************************************************************************
 * Backward
 *
 * `grad_x d^2 = -2 log_x(y)` in the Riemannian metric, evaluated in the
 * coordinate form each kernel returns. The output is a scalar, so the upstream
 * gradient is one number and the rule is a scaling of the two coordinate
 * gradients — no Jacobian assembly, and nothing that behaves differently on
 * the diagonal from anywhere else.
 ******************************************************************************/

extern "C" void tensor_squared_distance_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* x_node = node->input1;
    ad_node_t* y_node = node->input2;
    if (!x_node || !y_node) return;
    if (!node->saved_tensors || node->num_saved < 1 || !node->saved_tensors[0]) {
        eshkol_fatal("space form: squared-distance node carries no factor list; "
                     "its backward cannot know which manifold it is on. This "
                     "means the node was built by something other than "
                     "ad_squared_distance / ad_product_squared_distance.");
    }

    const eshkol_manifold_factor_t* factors =
        (const eshkol_manifold_factor_t*)node->saved_tensors[0];
    size_t k = (size_t)node->params.axis;
    size_t n = elem_count(x_node->shape, x_node->ndim);
    if (k == 0 || n == 0) return;

    const double* X = (const double*)x_node->tensor_value;
    const double* Y = (const double*)y_node->tensor_value;
    if (!X || !Y) return;

    /* Upstream is dL/d(d^2), a single number: the node's value is shape [1]. */
    const double upstream = ((const double*)node->tensor_gradient)[0];

    double* gx = alloc_doubles(n);
    double* gy = alloc_doubles(n);
    if (!gx || !gy) {
        eshkol_fatal("space form: could not allocate the squared-distance "
                     "gradient buffers; refusing to return a partial gradient");
    }

    size_t bad = 0;
    if (!product_forward(factors, k, X, Y, n, nullptr, gx, gy, &bad)) {
        /* The forward accepted this pair when the node was recorded, so
         * reaching here means the recorded inputs have since moved out of the
         * model. Refusing is the only honest answer: there is no derivative to
         * report, and a zero would be indistinguishable from coincidence —
         * which is precisely the case this whole node exists to differentiate. */
        eshkol_fatal("space form: squared-distance backward: factor %zu (%s) no "
                     "longer admits the recorded pair; no derivative exists to "
                     "return and a zero would be read as coincidence", bad,
                     form_name(factors[bad].form));
    }

    if (x_node->tensor_gradient) {
        double* dx = (double*)x_node->tensor_gradient;
        for (size_t i = 0; i < n; ++i) dx[i] += upstream * gx[i];
    }
    if (y_node->tensor_gradient) {
        double* dy = (double*)y_node->tensor_gradient;
        for (size_t i = 0; i < n; ++i) dy[i] += upstream * gy[i];
    }
}
