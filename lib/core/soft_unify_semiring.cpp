/*
 * Certified Differentiable Unification, 12-Semiring Provenance & Pearl Level-3 SCM
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/soft_unify_semiring.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

inline double dn(double x) {
    return std::isfinite(x) ? std::nextafter(x, -std::numeric_limits<double>::infinity()) : x;
}

inline double up(double x) {
    return std::isfinite(x) ? std::nextafter(x, std::numeric_limits<double>::infinity()) : x;
}

inline double log_add_exp(double a, double b) {
    if (std::isinf(a) && a < 0) return b;
    if (std::isinf(b) && b < 0) return a;
    double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

inline uint64_t fnv1a_mix(uint64_t h, const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        h ^= static_cast<uint64_t>(bytes[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t double_bits(double x) {
    if (x == 0.0) x = 0.0; /* canonicalize -0.0 */
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

/* Belnap-4 truth-order join (OR) and meet (AND) tables.
 * Lattice ordering: F < {N, B} < T, with N join B = T, N meet B = F. */
constexpr eshkol_belnap_val_t kBelnapOr[4][4] = {
    /*            N                      F                      T                    B */
    /* N */ { ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_TRUE, ESHKOL_BELNAP_TRUE },
    /* F */ { ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_TRUE, ESHKOL_BELNAP_BOTH },
    /* T */ { ESHKOL_BELNAP_TRUE,    ESHKOL_BELNAP_TRUE,    ESHKOL_BELNAP_TRUE, ESHKOL_BELNAP_TRUE },
    /* B */ { ESHKOL_BELNAP_TRUE,    ESHKOL_BELNAP_BOTH,    ESHKOL_BELNAP_TRUE, ESHKOL_BELNAP_BOTH }
};

constexpr eshkol_belnap_val_t kBelnapAnd[4][4] = {
    /*            N                      F                      T                      B */
    /* N */ { ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_FALSE },
    /* F */ { ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_FALSE },
    /* T */ { ESHKOL_BELNAP_NEITHER, ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_TRUE,    ESHKOL_BELNAP_BOTH },
    /* B */ { ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_FALSE,   ESHKOL_BELNAP_BOTH,    ESHKOL_BELNAP_BOTH }
};

} /* anonymous namespace */

/* ============================================================================
 * 1. Product-Manifold Soft Unification (R^{d_E} x B_c^{d_H})
 * ============================================================================ */

extern "C" eshkol_soft_unify_config_t eshkol_soft_unify_default_config(void) {
    eshkol_soft_unify_config_t cfg;
    cfg.curvature = 1.0;
    cfg.weight_eucl = 1.0;
    cfg.weight_hyp = 1.0;
    cfg.temperature = 1.0;
    return cfg;
}

extern "C" bool eshkol_soft_unify_eval(const double* e1, const double* e2, uint32_t dim_e,
                                       const double* h1, const double* h2, uint32_t dim_h,
                                       const eshkol_soft_unify_config_t* cfg,
                                       eshkol_soft_unify_result_t* out_result,
                                       double* grad_e1, double* grad_e2,
                                       double* grad_h1_eucl, double* grad_h2_eucl,
                                       double* grad_h1_riem, double* grad_h2_riem) {
    if (!out_result) return false;
    std::memset(out_result, 0, sizeof(*out_result));

    eshkol_soft_unify_config_t c = cfg ? *cfg : eshkol_soft_unify_default_config();
    if (!std::isfinite(c.curvature) || c.curvature <= 0.0 ||
        !std::isfinite(c.weight_eucl) || c.weight_eucl < 0.0 ||
        !std::isfinite(c.weight_hyp) || c.weight_hyp < 0.0 ||
        !std::isfinite(c.temperature) || c.temperature <= 0.0) {
        return false;
    }
    if (dim_e > 0 && (!e1 || !e2)) return false;
    if (dim_h > 0 && (!h1 || !h2)) return false;

    double eucl_sq = 0.0;
    for (uint32_t i = 0; i < dim_e; i++) {
        if (!std::isfinite(e1[i]) || !std::isfinite(e2[i])) return false;
        double diff = e1[i] - e2[i];
        eucl_sq += diff * diff;
    }

    double h1_sq = 0.0;
    double h2_sq = 0.0;
    double h_diff_sq = 0.0;
    for (uint32_t i = 0; i < dim_h; i++) {
        if (!std::isfinite(h1[i]) || !std::isfinite(h2[i])) return false;
        h1_sq += h1[i] * h1[i];
        h2_sq += h2[i] * h2[i];
        double diff = h1[i] - h2[i];
        h_diff_sq += diff * diff;
    }

    double alpha = 1.0 - c.curvature * h1_sq;
    double beta = 1.0 - c.curvature * h2_sq;
    if (dim_h > 0 && (alpha <= 1e-12 || beta <= 1e-12)) {
        /* Strictly inside open Poincare ball */
        return false;
    }

    double hyp_dist = 0.0;
    double gamma = 1.0;
    double s_factor = 1.0; /* arcosh(gamma) / sqrt(gamma^2 - 1), smoothly 1 at gamma = 1 */
    if (dim_h > 0) {
        double eps = (2.0 * c.curvature * h_diff_sq) / (alpha * beta);
        gamma = 1.0 + eps;
        if (eps > 1e-11) {
            double ac = std::acosh(gamma);
            double denom = std::sqrt(eps * (eps + 2.0));
            hyp_dist = ac / std::sqrt(c.curvature);
            s_factor = ac / denom;
        } else {
            /* Exact Taylor expansion around eps = 0:
             * arcosh(1 + eps) = sqrt(2*eps) * (1 - eps/12 + 3*eps^2/160 - ...)
             * sqrt(eps*(eps+2)) = sqrt(2*eps) * (1 + eps/4 - eps^2/32 + ...)
             * ratio = 1 - eps/3 + 2*eps^2/15 - ... */
            hyp_dist = std::sqrt(2.0 * eps) * (1.0 - eps / 12.0) / std::sqrt(c.curvature);
            s_factor = 1.0 - eps / 3.0 + (2.0 * eps * eps) / 15.0;
        }
    }

    double total_dist_sq = c.weight_eucl * eucl_sq + c.weight_hyp * hyp_dist * hyp_dist;
    double score = std::exp(-total_dist_sq / c.temperature);

    out_result->score = score;
    out_result->eucl_dist_sq = eucl_sq;
    out_result->hyp_dist = hyp_dist;
    out_result->fd_evals = 0;
    out_result->valid = std::isfinite(score);

    const double coeff_e = -(2.0 * c.weight_eucl * score) / c.temperature;
    for (uint32_t i = 0; i < dim_e; i++) {
        double g = coeff_e * (e1[i] - e2[i]);
        if (grad_e1) grad_e1[i] = g;
        if (grad_e2) grad_e2[i] = -g;
    }

    if (dim_h > 0) {
        const double pref = -(c.weight_hyp * score / c.temperature) * (8.0 * s_factor / (alpha * beta));
        const double inv_metric_1 = (alpha * alpha) * 0.25;
        const double inv_metric_2 = (beta * beta) * 0.25;
        const double c_r2_over_a = (c.curvature * h_diff_sq) / alpha;
        const double c_r2_over_b = (c.curvature * h_diff_sq) / beta;

        for (uint32_t i = 0; i < dim_h; i++) {
            double d = h1[i] - h2[i];
            double gh1 = pref * (d + c_r2_over_a * h1[i]);
            double gh2 = pref * (-d + c_r2_over_b * h2[i]);
            if (grad_h1_eucl) grad_h1_eucl[i] = gh1;
            if (grad_h2_eucl) grad_h2_eucl[i] = gh2;
            if (grad_h1_riem) grad_h1_riem[i] = inv_metric_1 * gh1;
            if (grad_h2_riem) grad_h2_riem[i] = inv_metric_2 * gh2;
        }
    }

    return out_result->valid;
}

/* ============================================================================
 * 2. 12-Semiring Provenance Engine
 * ============================================================================ */

extern "C" eshkol_semiring_val_t eshkol_sr_zero(eshkol_semiring_kind_t kind) {
    eshkol_semiring_val_t v{kind, 0.0, 0.0, 0u};
    switch (kind) {
        case ESHKOL_SR_BOOLEAN:
        case ESHKOL_SR_VITERBI:
        case ESHKOL_SR_SUM_PRODUCT:
        case ESHKOL_SR_LUKASIEWICZ:
        case ESHKOL_SR_GODEL:
        case ESHKOL_SR_PRODUCT_FUZZY:
            v.a = 0.0;
            break;
        case ESHKOL_SR_TROPICAL:
            v.a = std::numeric_limits<double>::infinity();
            break;
        case ESHKOL_SR_MAX_PLUS:
        case ESHKOL_SR_LOG_SPACE:
            v.a = -std::numeric_limits<double>::infinity();
            break;
        case ESHKOL_SR_BELNAP4:
            v.disc = ESHKOL_BELNAP_FALSE;
            break;
        case ESHKOL_SR_NARS:
            v.a = 0.5;
            v.b = 0.0;
            break;
        case ESHKOL_SR_CERTIFIED_INTERVAL:
            v.a = 0.0;
            v.b = 0.0;
            break;
    }
    return v;
}

extern "C" eshkol_semiring_val_t eshkol_sr_one(eshkol_semiring_kind_t kind) {
    eshkol_semiring_val_t v{kind, 1.0, 0.0, 0u};
    switch (kind) {
        case ESHKOL_SR_BOOLEAN:
        case ESHKOL_SR_VITERBI:
        case ESHKOL_SR_SUM_PRODUCT:
        case ESHKOL_SR_LUKASIEWICZ:
        case ESHKOL_SR_GODEL:
        case ESHKOL_SR_PRODUCT_FUZZY:
            v.a = 1.0;
            break;
        case ESHKOL_SR_TROPICAL:
        case ESHKOL_SR_MAX_PLUS:
        case ESHKOL_SR_LOG_SPACE:
            v.a = 0.0;
            break;
        case ESHKOL_SR_BELNAP4:
            v.a = 0.0;
            v.disc = ESHKOL_BELNAP_TRUE;
            break;
        case ESHKOL_SR_NARS:
            v.a = 1.0;
            v.b = std::nextafter(1.0, 0.0);
            break;
        case ESHKOL_SR_CERTIFIED_INTERVAL:
            v.a = 1.0;
            v.b = 1.0;
            break;
    }
    return v;
}

extern "C" eshkol_semiring_val_t eshkol_sr_scalar(eshkol_semiring_kind_t kind, double val) {
    eshkol_semiring_val_t v{kind, val, 0.0, 0u};
    if (kind == ESHKOL_SR_BOOLEAN) {
        v.a = (val != 0.0) ? 1.0 : 0.0;
    } else if (kind == ESHKOL_SR_CERTIFIED_INTERVAL) {
        v.a = val;
        v.b = val;
    }
    return v;
}

extern "C" eshkol_semiring_val_t eshkol_sr_nars(double frequency, double confidence) {
    return {ESHKOL_SR_NARS, frequency, confidence, 0u};
}

extern "C" eshkol_semiring_val_t eshkol_sr_interval(double lo, double hi) {
    return {ESHKOL_SR_CERTIFIED_INTERVAL, lo, hi, 0u};
}

extern "C" eshkol_semiring_val_t eshkol_sr_belnap(eshkol_belnap_val_t b) {
    return {ESHKOL_SR_BELNAP4, 0.0, 0.0, static_cast<uint32_t>(b)};
}

extern "C" bool eshkol_sr_add(eshkol_semiring_val_t x, eshkol_semiring_val_t y,
                              eshkol_semiring_val_t* out) {
    if (!out || x.kind != y.kind) return false;
    eshkol_semiring_val_t r{x.kind, 0.0, 0.0, 0u};
    switch (x.kind) {
        case ESHKOL_SR_BOOLEAN:
            r.a = (x.a != 0.0 || y.a != 0.0) ? 1.0 : 0.0;
            break;
        case ESHKOL_SR_VITERBI:
        case ESHKOL_SR_LUKASIEWICZ:
        case ESHKOL_SR_GODEL:
        case ESHKOL_SR_MAX_PLUS:
            if (std::isnan(x.a) || std::isnan(y.a)) return false;
            r.a = std::max(x.a, y.a);
            break;
        case ESHKOL_SR_SUM_PRODUCT:
            if (std::isnan(x.a) || std::isnan(y.a) || x.a < 0.0 || y.a < 0.0) return false;
            r.a = x.a + y.a;
            break;
        case ESHKOL_SR_TROPICAL:
            if (std::isnan(x.a) || std::isnan(y.a)) return false;
            r.a = std::min(x.a, y.a);
            break;
        case ESHKOL_SR_LOG_SPACE:
            if (std::isnan(x.a) || std::isnan(y.a)) return false;
            r.a = log_add_exp(x.a, y.a);
            break;
        case ESHKOL_SR_PRODUCT_FUZZY:
            if (std::isnan(x.a) || std::isnan(y.a) ||
                x.a < 0.0 || x.a > 1.0 || y.a < 0.0 || y.a > 1.0) return false;
            r.a = std::min(1.0, std::max(0.0, x.a + y.a - x.a * y.a));
            break;
        case ESHKOL_SR_BELNAP4:
            if (x.disc > 3 || y.disc > 3) return false;
            r.disc = kBelnapOr[x.disc][y.disc];
            break;
        case ESHKOL_SR_NARS: {
            eshkol_nars_truth_t nx{x.a, x.b};
            eshkol_nars_truth_t ny{y.a, y.b};
            eshkol_nars_truth_t nr{};
            if (!eshkol_nars_revision(nx, ny, ESHKOL_NARS_DEFAULT_HORIZON, &nr)) return false;
            r.a = nr.frequency;
            r.b = nr.confidence;
            break;
        }
        case ESHKOL_SR_CERTIFIED_INTERVAL:
            if (!std::isfinite(x.a) || !std::isfinite(x.b) ||
                !std::isfinite(y.a) || !std::isfinite(y.b) ||
                x.a < 0.0 || y.a < 0.0 || x.a > x.b || y.a > y.b) return false;
            r.a = std::max(0.0, dn(x.a + y.a));
            r.b = up(x.b + y.b);
            break;
    }
    *out = r;
    return true;
}

extern "C" bool eshkol_sr_mul(eshkol_semiring_val_t x, eshkol_semiring_val_t y,
                              eshkol_semiring_val_t* out) {
    if (!out || x.kind != y.kind) return false;
    eshkol_semiring_val_t r{x.kind, 0.0, 0.0, 0u};
    switch (x.kind) {
        case ESHKOL_SR_BOOLEAN:
            r.a = (x.a != 0.0 && y.a != 0.0) ? 1.0 : 0.0;
            break;
        case ESHKOL_SR_VITERBI:
        case ESHKOL_SR_PRODUCT_FUZZY:
            if (std::isnan(x.a) || std::isnan(y.a) ||
                x.a < 0.0 || x.a > 1.0 || y.a < 0.0 || y.a > 1.0) return false;
            r.a = x.a * y.a;
            break;
        case ESHKOL_SR_SUM_PRODUCT:
            if (std::isnan(x.a) || std::isnan(y.a) || x.a < 0.0 || y.a < 0.0) return false;
            r.a = x.a * y.a;
            break;
        case ESHKOL_SR_TROPICAL:
            if (std::isnan(x.a) || std::isnan(y.a)) return false;
            if (std::isinf(x.a) || std::isinf(y.a)) r.a = std::numeric_limits<double>::infinity();
            else r.a = x.a + y.a;
            break;
        case ESHKOL_SR_MAX_PLUS:
        case ESHKOL_SR_LOG_SPACE:
            if (std::isnan(x.a) || std::isnan(y.a)) return false;
            if ((std::isinf(x.a) && x.a < 0) || (std::isinf(y.a) && y.a < 0)) {
                r.a = -std::numeric_limits<double>::infinity();
            } else {
                r.a = x.a + y.a;
            }
            break;
        case ESHKOL_SR_LUKASIEWICZ:
            if (std::isnan(x.a) || std::isnan(y.a) ||
                x.a < 0.0 || x.a > 1.0 || y.a < 0.0 || y.a > 1.0) return false;
            r.a = std::max(0.0, x.a + y.a - 1.0);
            break;
        case ESHKOL_SR_GODEL:
            if (std::isnan(x.a) || std::isnan(y.a) ||
                x.a < 0.0 || x.a > 1.0 || y.a < 0.0 || y.a > 1.0) return false;
            r.a = std::min(x.a, y.a);
            break;
        case ESHKOL_SR_BELNAP4:
            if (x.disc > 3 || y.disc > 3) return false;
            r.disc = kBelnapAnd[x.disc][y.disc];
            break;
        case ESHKOL_SR_NARS: {
            eshkol_nars_truth_t nx{x.a, x.b};
            eshkol_nars_truth_t ny{y.a, y.b};
            eshkol_nars_truth_t nr{};
            if (!eshkol_nars_intersection(nx, ny, &nr)) return false;
            r.a = nr.frequency;
            r.b = nr.confidence;
            break;
        }
        case ESHKOL_SR_CERTIFIED_INTERVAL:
            if (!std::isfinite(x.a) || !std::isfinite(x.b) ||
                !std::isfinite(y.a) || !std::isfinite(y.b) ||
                x.a < 0.0 || y.a < 0.0 || x.a > x.b || y.a > y.b) return false;
            r.a = std::max(0.0, dn(x.a * y.a));
            r.b = up(x.b * y.b);
            break;
    }
    *out = r;
    return true;
}

extern "C" bool eshkol_sr_equal(eshkol_semiring_val_t x, eshkol_semiring_val_t y, double tol) {
    if (x.kind != y.kind) return false;
    if (x.kind == ESHKOL_SR_BELNAP4) return x.disc == y.disc;
    auto eq = [&](double p, double q) -> bool {
        if (std::isinf(p) && std::isinf(q)) return (p > 0) == (q > 0);
        return std::fabs(p - q) <= tol;
    };
    if (x.kind == ESHKOL_SR_NARS || x.kind == ESHKOL_SR_CERTIFIED_INTERVAL) {
        return eq(x.a, y.a) && eq(x.b, y.b);
    }
    return eq(x.a, y.a);
}

/* ============================================================================
 * Differentiable diff-top-k (Smooth Entropic Top-k + Exact Implicit Pullback)
 * ============================================================================ */

static inline double sigmoid_stable(double z) {
    if (z >= 0.0) {
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    }
    double ez = std::exp(z);
    return ez / (1.0 + ez);
}

extern "C" bool eshkol_diff_topk_forward(const double* scores,
                                         uint32_t n,
                                         double k,
                                         double temperature,
                                         double* out_probs,
                                         double* out_threshold) {
    if (!scores || !out_probs || n == 0) return false;
    if (!std::isfinite(k) || k <= 0.0 || k >= static_cast<double>(n)) return false;
    if (!std::isfinite(temperature) || temperature <= 0.0) return false;

    double s_min = scores[0];
    double s_max = scores[0];
    for (uint32_t i = 0; i < n; i++) {
        if (!std::isfinite(scores[i])) return false;
        s_min = std::min(s_min, scores[i]);
        s_max = std::max(s_max, scores[i]);
    }

    auto sum_probs = [&](double lam) -> double {
        double acc = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            acc += sigmoid_stable((scores[i] - lam) / temperature);
        }
        return acc;
    };

    /* Bracket root of f(lam) = sum_i sigmoid((s_i - lam)/tau) - k = 0 (strictly decreasing) */
    double lo = s_min - 40.0 * temperature;
    double hi = s_max + 40.0 * temperature;
    while (sum_probs(lo) < k) lo -= 40.0 * temperature;
    while (sum_probs(hi) > k) hi += 40.0 * temperature;

    /* Safeguarded Newton-Bisection to machine precision */
    double lam = 0.5 * (lo + hi);
    for (int it = 0; it < 100; it++) {
        double f = 0.0;
        double df = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            double p = sigmoid_stable((scores[i] - lam) / temperature);
            f += p;
            df -= p * (1.0 - p) / temperature;
        }
        f -= k;
        if (std::fabs(f) < 1e-14) break;
        if (f > 0.0) lo = lam;
        else hi = lam;
        double cand = (df < -1e-30) ? (lam - f / df) : (0.5 * (lo + hi));
        if (cand <= lo || cand >= hi) cand = 0.5 * (lo + hi);
        lam = cand;
    }

    for (uint32_t i = 0; i < n; i++) {
        out_probs[i] = sigmoid_stable((scores[i] - lam) / temperature);
    }
    if (out_threshold) *out_threshold = lam;
    return true;
}

extern "C" bool eshkol_diff_topk_backward(const double* probs,
                                          const double* dL_dprobs,
                                          uint32_t n,
                                          double temperature,
                                          double* out_dL_dscores,
                                          uint64_t* out_fd_evals) {
    if (!probs || !dL_dprobs || !out_dL_dscores || n == 0) return false;
    if (!std::isfinite(temperature) || temperature <= 0.0) return false;
    if (out_fd_evals) *out_fd_evals = 0;

    double w_sum = 0.0;
    double wg_sum = 0.0;
    std::vector<double> w(n);
    for (uint32_t i = 0; i < n; i++) {
        if (!std::isfinite(probs[i]) || !std::isfinite(dL_dprobs[i])) return false;
        w[i] = probs[i] * (1.0 - probs[i]) / temperature;
        w_sum += w[i];
        wg_sum += w[i] * dL_dprobs[i];
    }

    double shift = (w_sum > 1e-300) ? (wg_sum / w_sum) : 0.0;
    for (uint32_t i = 0; i < n; i++) {
        out_dL_dscores[i] = w[i] * (dL_dprobs[i] - shift);
    }
    return true;
}

/* ============================================================================
 * 3. Replayable Merkle-Hashed PROOF_DAG
 * ============================================================================ */

struct ProofDagNode {
    eshkol_proof_node_kind_t kind;
    std::string label;
    std::vector<uint32_t> parents;
    eshkol_semiring_val_t value;
    uint64_t hash;
};

struct eshkol_proof_dag {
    std::vector<ProofDagNode> nodes;

    static uint64_t compute_node_hash(const ProofDagNode& n,
                                      const std::vector<uint64_t>& parent_hashes) {
        uint64_t h = 14695981039346656037ULL;
        uint32_t k = static_cast<uint32_t>(n.kind);
        uint32_t sr = static_cast<uint32_t>(n.value.kind);
        uint64_t ab = double_bits(n.value.a);
        uint64_t bb = double_bits(n.value.b);
        uint32_t np = static_cast<uint32_t>(parent_hashes.size());
        h = fnv1a_mix(h, &k, sizeof(k));
        h = fnv1a_mix(h, n.label.data(), n.label.size());
        h = fnv1a_mix(h, &sr, sizeof(sr));
        h = fnv1a_mix(h, &ab, sizeof(ab));
        h = fnv1a_mix(h, &bb, sizeof(bb));
        h = fnv1a_mix(h, &n.value.disc, sizeof(n.value.disc));
        h = fnv1a_mix(h, &np, sizeof(np));
        for (uint64_t ph : parent_hashes) {
            h = fnv1a_mix(h, &ph, sizeof(ph));
        }
        return h;
    }
};

extern "C" eshkol_proof_dag_t* eshkol_proof_dag_create(void) {
    return new (std::nothrow) eshkol_proof_dag();
}

extern "C" void eshkol_proof_dag_destroy(eshkol_proof_dag_t* dag) {
    delete dag;
}

extern "C" uint32_t eshkol_proof_dag_add_node(eshkol_proof_dag_t* dag,
                                              eshkol_proof_node_kind_t kind,
                                              const char* label,
                                              const uint32_t* parents,
                                              uint32_t num_parents,
                                              eshkol_semiring_val_t value) {
    if (!dag || (num_parents > 0 && !parents)) return UINT32_MAX;
    std::vector<uint64_t> ph;
    ph.reserve(num_parents);
    for (uint32_t i = 0; i < num_parents; i++) {
        if (parents[i] >= dag->nodes.size()) return UINT32_MAX;
        ph.push_back(dag->nodes[parents[i]].hash);
    }

    ProofDagNode node;
    node.kind = kind;
    node.label = label ? label : "";
    if (num_parents > 0) node.parents.assign(parents, parents + num_parents);
    node.value = value;
    node.hash = eshkol_proof_dag::compute_node_hash(node, ph);

    uint32_t id = static_cast<uint32_t>(dag->nodes.size());
    dag->nodes.push_back(std::move(node));
    return id;
}

extern "C" uint32_t eshkol_proof_dag_num_nodes(const eshkol_proof_dag_t* dag) {
    return dag ? static_cast<uint32_t>(dag->nodes.size()) : 0u;
}

extern "C" uint64_t eshkol_proof_dag_node_hash(const eshkol_proof_dag_t* dag,
                                               uint32_t node_id) {
    if (!dag || node_id >= dag->nodes.size()) return 0ULL;
    return dag->nodes[node_id].hash;
}

extern "C" bool eshkol_proof_dag_get_value(const eshkol_proof_dag_t* dag,
                                           uint32_t node_id,
                                           eshkol_semiring_val_t* out_val) {
    if (!dag || !out_val || node_id >= dag->nodes.size()) return false;
    *out_val = dag->nodes[node_id].value;
    return true;
}

extern "C" bool eshkol_proof_dag_corrupt_value_for_test(eshkol_proof_dag_t* dag,
                                                        uint32_t node_id,
                                                        double new_a) {
    if (!dag || node_id >= dag->nodes.size()) return false;
    dag->nodes[node_id].value.a = new_a;
    return true;
}

extern "C" bool eshkol_proof_dag_verify_replay(const eshkol_proof_dag_t* dag) {
    if (!dag) return false;
    std::vector<uint64_t> replayed_hashes(dag->nodes.size(), 0ULL);

    for (uint32_t i = 0; i < dag->nodes.size(); i++) {
        const ProofDagNode& n = dag->nodes[i];
        std::vector<uint64_t> ph;
        ph.reserve(n.parents.size());
        for (uint32_t p : n.parents) {
            if (p >= i) return false; /* Must be strictly DAG-ordered */
            ph.push_back(replayed_hashes[p]);
        }

        /* Re-verify semiring derivation semantics for ADD/MUL nodes */
        if (n.kind == ESHKOL_PROOF_SEMIRING_ADD || n.kind == ESHKOL_PROOF_SEMIRING_MUL) {
            if (n.parents.size() != 2) return false;
            eshkol_semiring_val_t expected{};
            bool ok = (n.kind == ESHKOL_PROOF_SEMIRING_ADD)
                ? eshkol_sr_add(dag->nodes[n.parents[0]].value,
                                dag->nodes[n.parents[1]].value, &expected)
                : eshkol_sr_mul(dag->nodes[n.parents[0]].value,
                                dag->nodes[n.parents[1]].value, &expected);
            if (!ok || !eshkol_sr_equal(expected, n.value, 1e-12)) return false;
        }

        uint64_t expected_hash = eshkol_proof_dag::compute_node_hash(n, ph);
        if (expected_hash != n.hash) return false;
        replayed_hashes[i] = expected_hash;
    }
    return true;
}

/* ============================================================================
 * 4. Pearl Level-3 Structural Causal Model (SCM) & Counterfactual Engine
 * ============================================================================ */

struct ExogVar {
    std::string name;
    std::vector<double> probs;
};

struct EndogVar {
    std::string name;
    uint32_t domain_size = 2;
    std::vector<uint32_t> parents;
    std::vector<uint32_t> exog;
    eshkol_scm_fn_t fn = nullptr;
    void* ctx = nullptr;
};

struct eshkol_scm {
    std::vector<ExogVar> exog_vars;
    std::vector<EndogVar> endog_vars;

    /* Evaluate all endogenous variables for one exogenous assignment, optionally with do(). */
    bool eval_world(const std::vector<uint32_t>& u_assign,
                    const std::vector<int32_t>* do_override,
                    std::vector<uint32_t>& out_v) const {
        out_v.assign(endog_vars.size(), 0u);
        std::vector<uint32_t> pvals;
        std::vector<uint32_t> uvals;
        for (uint32_t i = 0; i < endog_vars.size(); i++) {
            if (do_override && (*do_override)[i] >= 0) {
                out_v[i] = static_cast<uint32_t>((*do_override)[i]);
                continue;
            }
            const EndogVar& ev = endog_vars[i];
            pvals.resize(ev.parents.size());
            for (size_t k = 0; k < ev.parents.size(); k++) {
                pvals[k] = out_v[ev.parents[k]];
            }
            uvals.resize(ev.exog.size());
            for (size_t k = 0; k < ev.exog.size(); k++) {
                uvals[k] = u_assign[ev.exog[k]];
            }
            uint32_t val = ev.fn(pvals.empty() ? nullptr : pvals.data(),
                                 static_cast<uint32_t>(pvals.size()),
                                 uvals.empty() ? nullptr : uvals.data(),
                                 static_cast<uint32_t>(uvals.size()),
                                 ev.ctx);
            if (val >= ev.domain_size) return false;
            out_v[i] = val;
        }
        return true;
    }

    uint64_t total_exog_worlds() const {
        uint64_t total = 1;
        for (const ExogVar& u : exog_vars) {
            total *= static_cast<uint64_t>(u.probs.size());
            if (total > (1ULL << 22)) return 0;
        }
        return total;
    }

    void decode_exog(uint64_t idx, std::vector<uint32_t>& u_assign, double& out_prob) const {
        u_assign.resize(exog_vars.size());
        out_prob = 1.0;
        uint64_t rem = idx;
        for (size_t i = exog_vars.size(); i-- > 0; ) {
            uint64_t d = exog_vars[i].probs.size();
            uint32_t s = static_cast<uint32_t>(rem % d);
            rem /= d;
            u_assign[i] = s;
            out_prob *= exog_vars[i].probs[s];
        }
    }

    bool matches_constraints(const std::vector<uint32_t>& v,
                             const uint32_t* vars,
                             const uint32_t* vals,
                             uint32_t n) const {
        for (uint32_t i = 0; i < n; i++) {
            if (v[vars[i]] != vals[i]) return false;
        }
        return true;
    }
};

extern "C" eshkol_scm_t* eshkol_scm_create(void) {
    return new (std::nothrow) eshkol_scm();
}

extern "C" void eshkol_scm_destroy(eshkol_scm_t* scm) {
    delete scm;
}

extern "C" uint32_t eshkol_scm_add_exogenous(eshkol_scm_t* scm,
                                             const char* name,
                                             const double* prior_probs,
                                             uint32_t domain_size) {
    if (!scm || !prior_probs || domain_size == 0) return UINT32_MAX;
    double sum = 0.0;
    for (uint32_t i = 0; i < domain_size; i++) {
        if (!std::isfinite(prior_probs[i]) || prior_probs[i] < 0.0) return UINT32_MAX;
        sum += prior_probs[i];
    }
    if (std::fabs(sum - 1.0) > 1e-6) return UINT32_MAX;

    ExogVar u;
    u.name = name ? name : "";
    u.probs.assign(prior_probs, prior_probs + domain_size);
    for (double& p : u.probs) p /= sum;
    uint32_t id = static_cast<uint32_t>(scm->exog_vars.size());
    scm->exog_vars.push_back(std::move(u));
    return id;
}

extern "C" uint32_t eshkol_scm_add_endogenous(eshkol_scm_t* scm,
                                              const char* name,
                                              uint32_t domain_size,
                                              const uint32_t* parent_endog,
                                              uint32_t num_parents,
                                              const uint32_t* exog_vars,
                                              uint32_t num_exog,
                                              eshkol_scm_fn_t fn,
                                              void* user_ctx) {
    if (!scm || domain_size == 0 || !fn) return UINT32_MAX;
    if ((num_parents > 0 && !parent_endog) || (num_exog > 0 && !exog_vars)) return UINT32_MAX;

    for (uint32_t i = 0; i < num_parents; i++) {
        if (parent_endog[i] >= scm->endog_vars.size()) return UINT32_MAX;
    }
    for (uint32_t i = 0; i < num_exog; i++) {
        if (exog_vars[i] >= scm->exog_vars.size()) return UINT32_MAX;
    }

    EndogVar ev;
    ev.name = name ? name : "";
    ev.domain_size = domain_size;
    if (num_parents > 0) ev.parents.assign(parent_endog, parent_endog + num_parents);
    if (num_exog > 0) ev.exog.assign(exog_vars, exog_vars + num_exog);
    ev.fn = fn;
    ev.ctx = user_ctx;

    uint32_t id = static_cast<uint32_t>(scm->endog_vars.size());
    scm->endog_vars.push_back(std::move(ev));
    return id;
}

extern "C" bool eshkol_scm_observational_query(const eshkol_scm_t* scm,
                                               uint32_t target_var,
                                               uint32_t target_val,
                                               const uint32_t* cond_vars,
                                               const uint32_t* cond_vals,
                                               uint32_t num_cond,
                                               double* out_prob) {
    if (!scm || !out_prob || target_var >= scm->endog_vars.size()) return false;
    if (target_val >= scm->endog_vars[target_var].domain_size) return false;
    if (num_cond > 0 && (!cond_vars || !cond_vals)) return false;
    for (uint32_t i = 0; i < num_cond; i++) {
        if (cond_vars[i] >= scm->endog_vars.size() ||
            cond_vals[i] >= scm->endog_vars[cond_vars[i]].domain_size) {
            return false;
        }
    }

    uint64_t worlds = scm->total_exog_worlds();
    if (worlds == 0) return false;

    double numer = 0.0;
    double denom = 0.0;
    std::vector<uint32_t> u_assign;
    std::vector<uint32_t> v_assign;
    for (uint64_t w = 0; w < worlds; w++) {
        double pw = 0.0;
        scm->decode_exog(w, u_assign, pw);
        if (pw == 0.0) continue;
        if (!scm->eval_world(u_assign, nullptr, v_assign)) return false;
        if (scm->matches_constraints(v_assign, cond_vars, cond_vals, num_cond)) {
            denom += pw;
            if (v_assign[target_var] == target_val) numer += pw;
        }
    }
    if (denom <= 0.0) return false;
    *out_prob = numer / denom;
    return true;
}

extern "C" bool eshkol_scm_do_query(const eshkol_scm_t* scm,
                                    uint32_t target_var,
                                    uint32_t target_val,
                                    const uint32_t* do_vars,
                                    const uint32_t* do_vals,
                                    uint32_t num_do,
                                    double* out_prob) {
    if (!scm || !out_prob || target_var >= scm->endog_vars.size()) return false;
    if (target_val >= scm->endog_vars[target_var].domain_size) return false;
    if (num_do > 0 && (!do_vars || !do_vals)) return false;

    std::vector<int32_t> do_override(scm->endog_vars.size(), -1);
    for (uint32_t i = 0; i < num_do; i++) {
        if (do_vars[i] >= scm->endog_vars.size() ||
            do_vals[i] >= scm->endog_vars[do_vars[i]].domain_size) {
            return false;
        }
        do_override[do_vars[i]] = static_cast<int32_t>(do_vals[i]);
    }

    uint64_t worlds = scm->total_exog_worlds();
    if (worlds == 0) return false;

    double prob = 0.0;
    std::vector<uint32_t> u_assign;
    std::vector<uint32_t> v_assign;
    for (uint64_t w = 0; w < worlds; w++) {
        double pw = 0.0;
        scm->decode_exog(w, u_assign, pw);
        if (pw == 0.0) continue;
        if (!scm->eval_world(u_assign, &do_override, v_assign)) return false;
        if (v_assign[target_var] == target_val) prob += pw;
    }
    *out_prob = prob;
    return true;
}

extern "C" bool eshkol_scm_counterfactual(const eshkol_scm_t* scm,
                                          uint32_t target_var,
                                          uint32_t target_val,
                                          const uint32_t* do_vars,
                                          const uint32_t* do_vals,
                                          uint32_t num_do,
                                          const uint32_t* obs_vars,
                                          const uint32_t* obs_vals,
                                          uint32_t num_obs,
                                          double* out_prob) {
    if (!scm || !out_prob || target_var >= scm->endog_vars.size()) return false;
    if (target_val >= scm->endog_vars[target_var].domain_size) return false;
    if ((num_do > 0 && (!do_vars || !do_vals)) || (num_obs > 0 && (!obs_vars || !obs_vals))) {
        return false;
    }

    std::vector<int32_t> do_override(scm->endog_vars.size(), -1);
    for (uint32_t i = 0; i < num_do; i++) {
        if (do_vars[i] >= scm->endog_vars.size() ||
            do_vals[i] >= scm->endog_vars[do_vars[i]].domain_size) {
            return false;
        }
        do_override[do_vars[i]] = static_cast<int32_t>(do_vals[i]);
    }
    for (uint32_t i = 0; i < num_obs; i++) {
        if (obs_vars[i] >= scm->endog_vars.size() ||
            obs_vals[i] >= scm->endog_vars[obs_vars[i]].domain_size) {
            return false;
        }
    }

    uint64_t worlds = scm->total_exog_worlds();
    if (worlds == 0) return false;

    /* Step 1 (Abduction) + Step 2 (Action) + Step 3 (Prediction) */
    double obs_mass = 0.0;
    double cf_mass = 0.0;
    std::vector<uint32_t> u_assign;
    std::vector<uint32_t> v_factual;
    std::vector<uint32_t> v_counterfactual;

    for (uint64_t w = 0; w < worlds; w++) {
        double pw = 0.0;
        scm->decode_exog(w, u_assign, pw);
        if (pw == 0.0) continue;

        /* Abduction: evaluate unmodified factual world */
        if (!scm->eval_world(u_assign, nullptr, v_factual)) return false;
        if (!scm->matches_constraints(v_factual, obs_vars, obs_vals, num_obs)) continue;
        obs_mass += pw;

        /* Action + Prediction: evaluate mutilated counterfactual world under same U */
        if (!scm->eval_world(u_assign, &do_override, v_counterfactual)) return false;
        if (v_counterfactual[target_var] == target_val) cf_mass += pw;
    }

    if (obs_mass <= 0.0) return false;
    *out_prob = cf_mass / obs_mass;
    return true;
}
