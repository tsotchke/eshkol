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
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/logger.h"
#include "eshkol/bridge/space_form.h"
#include "eshkol/backend/riemannian_core.h"

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

/*******************************************************************************
 * Single-factor kernels
 *
 * Each returns `false` and leaves its outputs untouched when the pair is
 * outside the model or beyond the injectivity radius. `grad_x` / `grad_y` are
 * the COORDINATE gradients — the metric tensor has already been applied, which
 * is what the reverse tape accumulates. The Riemannian gradient the theory
 * states, `-2 log_x(y)` (with the sphere's canonicalization Jacobian applied to
 * raw accepted inputs), is `g^{-1}` times what comes out here, and
 * `sf_log_map` below returns exactly that log so the identity can be checked
 * against an independent evaluation rather than against a rearrangement of
 * these same lines.
 ******************************************************************************/

struct SfPair {
    double d2;          /* squared geodesic distance of this factor */
    bool   ok;          /* false: outside the model / past the cut locus */
};

/** @brief Validate the redundant public form/K pair. */
bool valid_form_curvature(int form, double K) {
    if (!std::isfinite(K)) return false;
    switch (form) {
        case ESHKOL_SPACE_FORM_EUCLIDEAN:  return K == 0.0;
        case ESHKOL_SPACE_FORM_HYPERBOLIC: return K < 0.0;
        case ESHKOL_SPACE_FORM_SPHERICAL:  return K > 0.0;
        default: return false;
    }
}

bool finite_point(const double* p, size_t n) {
    if (!p) return false;
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

const char* form_name(int form);

/** @brief Direct stable squared distance using the shared Riemannian core. */
bool space_form_sq_value(int form, double K, const double* x, const double* y,
                         size_t n, double distance_scale, double* out) {
    if (!valid_form_curvature(form, K) || !finite_point(x, n) ||
        !finite_point(y, n) || eshkol_rm_check_point(x, K, (int)n) != nullptr ||
        eshkol_rm_check_point(y, K, (int)n) != nullptr) return false;
    if (eshkol_rm_points_equal(x, y, (int)n)) {
        *out = 0.0;
        return true;
    }
    if (K == 0.0) {
        double unweighted = 0.0;
        bool direct_finite = true;
        for (size_t i = 0; i < n; ++i) {
            double d = x[i] - y[i];
            unweighted += d * d;
            if (!std::isfinite(unweighted)) {
                direct_finite = false;
                break;
            }
        }
        if (direct_finite && std::isfinite(distance_scale * distance_scale *
                                            unweighted)) {
            *out = ESHKOL_RM_FLAT_LAMBDA * ESHKOL_RM_FLAT_LAMBDA *
                   (distance_scale * distance_scale * unweighted);
            return std::isfinite(*out);
        }
        double e = 0.0;
        for (size_t i = 0; i < n; ++i) {
            /* Scale before subtracting opposite-signed huge coordinates: the
             * weighted distance can be finite even when |x-y| is not. */
            double d;
            if ((x[i] < 0.0 && y[i] > 0.0) ||
                (x[i] > 0.0 && y[i] < 0.0))
                d = distance_scale * x[i] - distance_scale * y[i];
            else
                d = distance_scale * (x[i] - y[i]);
            e += d * d;
        }
        *out = ESHKOL_RM_FLAT_LAMBDA * ESHKOL_RM_FLAT_LAMBDA * e;
        return std::isfinite(*out);
    }
    if (K < 0.0) {
        double B = eshkol_rm_ball_param(-K);
        double a = eshkol_rm_one_minus_bnorm2(x, B, (int)n);
        double b = eshkol_rm_one_minus_bnorm2(y, B, (int)n);
        double delta = 0.0;
        for (size_t i = 0; i < n; ++i)
            delta = std::hypot(delta, x[i] - y[i]);
        double scaled_delta = delta / std::sqrt(a * b);
        double scaled_argument = std::sqrt(B) * scaled_delta;
        double psi = eshkol_rm_psi(scaled_argument * scaled_argument,
                                   nullptr, nullptr);
        double d = ESHKOL_RM_LAMBDA0 *
                   (distance_scale * scaled_delta) * psi;
        *out = d * d;
        return std::isfinite(*out);
    }
    double R = 1.0 / std::sqrt(K);
    if (eshkol_rm_sphere_antipodal(x, y, (int)n)) return false;
    double theta = eshkol_rm_sphere_angle(x, y, R, (int)n, nullptr);
    /* Form the finite physical distance before squaring.  R*R can overflow
     * even when (R*theta)^2 is finite for a small angular separation. */
    double d = R * (distance_scale * theta);
    *out = d * d;
    return std::isfinite(*out);
}

/** @brief Shared-core squared distance and coordinate gradients. */
SfPair factor_sq(int form, double K, const double* x, const double* y, size_t n,
                 double weight, double* grad_x, double* grad_y) {
    double d2 = 0.0;
    double distance_scale = std::sqrt(weight);
    if (!space_form_sq_value(form, K, x, y, n, distance_scale, &d2)) {
        return SfPair{0.0, false};
    }
    if (grad_x || grad_y) {
        if (K == 0.0) {
            for (size_t i = 0; i < n; ++i) {
                if (weight == 1.0) {
                    double d = x[i] - y[i];
                    if (grad_x) grad_x[i] = 2.0 * d;
                    if (grad_y) grad_y[i] = -2.0 * d;
                } else {
                    double raw_d = x[i] - y[i];
                    double direct_g = 2.0 * weight * raw_d;
                    double wx = distance_scale * x[i];
                    double wy = distance_scale * y[i];
                    double d = wx - wy;
                    double g = std::isfinite(direct_g) ? direct_g
                                                       : 2.0 * distance_scale * d;
                    if (grad_x) grad_x[i] = g;
                    if (grad_y) grad_y[i] = -g;
                }
            }
        } else {
            std::vector<double> log_x(n), log_y(n), scratch(n);
            if (eshkol_rm_log_map(x, y, K, (int)n, log_x.data(), scratch.data()) ||
                eshkol_rm_log_map(y, x, K, (int)n, log_y.data(), scratch.data())) {
                return SfPair{0.0, false};
            }
            double sx = (K < 0.0) ? eshkol_rm_lambda(x, K, (int)n) *
                                    eshkol_rm_lambda(x, K, (int)n)
                                  : (1.0 / std::sqrt(K)) /
                                    eshkol_rm_norm(x, (int)n);
            double sy = (K < 0.0) ? eshkol_rm_lambda(y, K, (int)n) *
                                    eshkol_rm_lambda(y, K, (int)n)
                                  : (1.0 / std::sqrt(K)) /
                                    eshkol_rm_norm(y, (int)n);
            for (size_t i = 0; i < n; ++i) {
                if (grad_x) grad_x[i] = -2.0 * weight * sx * log_x[i];
                if (grad_y) grad_y[i] = -2.0 * weight * sy * log_y[i];
            }
        }
    }
    return SfPair{d2, true};
}

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
        if (f[i].reserved != 0) {
            eshkol_error("space form: factor %zu has non-zero reserved field", i);
            return 0;
        }
        if (!valid_form_curvature(f[i].form, f[i].curvature)) {
            eshkol_error("space form: factor %zu has curvature %g inconsistent "
                         "with form %s (Euclidean requires K=0, hyperbolic K<0, "
                         "spherical K>0)", i, f[i].curvature,
                         form_name(f[i].form));
            return 0;
        }
        if (!std::isfinite(f[i].weight) || f[i].weight < 0.0) {
            eshkol_error("space form: factor %zu has invalid weight %g; "
                         "weights must be finite and non-negative", i, f[i].weight);
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
        double K = factors[i].curvature;
        double w = factors[i].weight;
        if (w == 0.0) {
            /* A zero-weight factor still has to name a valid finite pair on
             * its manifold, but it must not enter distance or gradient
             * arithmetic: a valid finite pair can have an overflowing
             * unweighted squared distance. */
            if (!finite_point(X + off, dim) || !finite_point(Y + off, dim) ||
                eshkol_rm_check_point(X + off, K, (int)dim) != nullptr ||
                eshkol_rm_check_point(Y + off, K, (int)dim) != nullptr) {
                if (out_bad_factor) *out_bad_factor = i;
                return false;
            }
            off += dim;
            continue;
        }
        SfPair r = factor_sq(factors[i].form, K, X + off, Y + off, dim, w,
                             grad_x ? grad_x + off : nullptr,
                             grad_y ? grad_y + off : nullptr);
        if (!r.ok) {
            if (out_bad_factor) *out_bad_factor = i;
            return false;
        }
        /* factor_sq has already formed sqrt(w)*d and returned w*d^2, so an
         * overflowing unweighted factor cannot poison a finite weighted sum. */
        if (!std::isfinite(r.d2) || !std::isfinite(acc + r.d2)) {
            if (out_bad_factor) *out_bad_factor = i;
            return false;
        }
        acc += r.d2;
        /* The product metric is block diagonal: factor f's coordinates appear
         * in no other factor's distance, so w_f scales that block of the
         * gradient and there are no cross terms to add. */
        if (grad_x || grad_y) {
            for (size_t j = 0; j < dim; ++j) {
                if ((grad_x && !std::isfinite(grad_x[off + j])) ||
                    (grad_y && !std::isfinite(grad_y[off + j]))) {
                    if (out_bad_factor) *out_bad_factor = i;
                    return false;
                }
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
    if (!out || n == 0 || !valid_form_curvature(form, curvature) ||
        !finite_point(x, n) || !finite_point(y, n) ||
        eshkol_rm_check_point(x, curvature, (int)n) != nullptr ||
        eshkol_rm_check_point(y, curvature, (int)n) != nullptr) return false;
    std::vector<double> scratch(n);
    return eshkol_rm_log_map(x, y, curvature, (int)n, out, scratch.data()) == nullptr;
}

extern "C" double eshkol_space_form_distance(int form, double curvature,
                                             const double* x, const double* y,
                                             size_t n) {
    if (n == 0 || !valid_form_curvature(form, curvature) ||
        !finite_point(x, n) || !finite_point(y, n) ||
        eshkol_rm_check_point(x, curvature, (int)n) != nullptr ||
        eshkol_rm_check_point(y, curvature, (int)n) != nullptr) return -1.0;
    double out = -1.0;
    if (eshkol_rm_distance(x, y, curvature, (int)n, &out) != nullptr)
        return -1.0;
    return out;
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
                     factors[bad].curvature);
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
