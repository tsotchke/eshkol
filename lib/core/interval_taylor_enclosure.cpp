/*
 * Graded Interval-Taylor Certified Neural World Model Enclosures (Order-2)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/interval_taylor_enclosure.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/* Exact IEEE-754 outward-rounded interval arithmetic via nextafter. */
struct Iv {
    double lo = 0.0;
    double hi = 0.0;

    static double dn(double x) { return std::isfinite(x) ? std::nextafter(x, -kInf) : x; }
    static double up(double x) { return std::isfinite(x) ? std::nextafter(x, +kInf) : x; }
    static double dn2(double x) { return dn(dn(x)); }
    static double up2(double x) { return up(up(x)); }

    static Iv point(double x) { return {x, x}; }
    static Iv sym(double r) {
        double rr = std::fabs(r);
        return {-rr, rr};
    }

    Iv operator+(const Iv& o) const { return {dn(lo + o.lo), up(hi + o.hi)}; }
    Iv operator-(const Iv& o) const { return {dn(lo - o.hi), up(hi - o.lo)}; }
    Iv operator*(const Iv& o) const {
        double p1 = lo * o.lo, p2 = lo * o.hi, p3 = hi * o.lo, p4 = hi * o.hi;
        return {dn(std::min({p1, p2, p3, p4})), up(std::max({p1, p2, p3, p4}))};
    }
    Iv scale(double s) const {
        if (s >= 0.0) return {dn(lo * s), up(hi * s)};
        return {dn(hi * s), up(lo * s)};
    }
    double max_abs() const { return std::max(std::fabs(lo), std::fabs(hi)); }
};

inline double sigmoid_fn(double x) {
    if (x >= 0.0) {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

inline double softplus_fn(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline double eval_act_point(eshkol_tm_activation_t act, double x) {
    switch (act) {
        case ESHKOL_TM_ACT_IDENTITY:       return x;
        case ESHKOL_TM_ACT_TANH:           return std::tanh(x);
        case ESHKOL_TM_ACT_SIGMOID:        return sigmoid_fn(x);
        case ESHKOL_TM_ACT_SOFTPLUS:       return softplus_fn(x);
        case ESHKOL_TM_ACT_SIN:            return std::sin(x);
        case ESHKOL_TM_ACT_SQUARE:         return x * x;
        case ESHKOL_TM_ACT_HEAVISIDE_STEP: return (x >= 0.0) ? 1.0 : 0.0;
    }
    return x;
}

/* Exact 0th, 1st, 2nd derivatives at center c and rigorous global upper bound M3 on |phi'''|. */
bool eval_act_taylor_coeffs(eshkol_tm_activation_t act,
                            double c,
                            double& d0, double& d1, double& d2,
                            double& m3_bound) {
    switch (act) {
        case ESHKOL_TM_ACT_IDENTITY:
            d0 = c; d1 = 1.0; d2 = 0.0; m3_bound = 0.0;
            return true;
        case ESHKOL_TM_ACT_TANH: {
            double t = std::tanh(c);
            double s2 = 1.0 - t * t;
            d0 = t;
            d1 = s2;
            d2 = -2.0 * t * s2;
            m3_bound = 2.0; /* Exact global sup |tanh'''(x)| = 2 at x = 0 */
            return true;
        }
        case ESHKOL_TM_ACT_SIGMOID: {
            double s = sigmoid_fn(c);
            double sp = s * (1.0 - s);
            d0 = s;
            d1 = sp;
            d2 = sp * (1.0 - 2.0 * s);
            m3_bound = 0.125; /* Exact global sup |sigma'''(x)| = 1/8 */
            return true;
        }
        case ESHKOL_TM_ACT_SOFTPLUS: {
            double s = sigmoid_fn(c);
            double sp = s * (1.0 - s);
            d0 = softplus_fn(c);
            d1 = s;
            d2 = sp;
            m3_bound = 0.0963; /* Exact global sup |softplus'''(x)| = 1/(6*sqrt(3)) < 0.09623 */
            return true;
        }
        case ESHKOL_TM_ACT_SIN:
            d0 = std::sin(c);
            d1 = std::cos(c);
            d2 = -std::sin(c);
            m3_bound = 1.0; /* Exact global sup |sin'''(x)| = 1 */
            return true;
        case ESHKOL_TM_ACT_SQUARE:
            d0 = c * c;
            d1 = 2.0 * c;
            d2 = 2.0;
            m3_bound = 0.0;
            return true;
        case ESHKOL_TM_ACT_HEAVISIDE_STEP:
            return false; /* Discontinuous / non-Lipschitz -> fail closed! */
    }
    return false;
}

/* Outward-rounded univariate interval image for naive IBP baseline. */
bool eval_act_naive_iv(eshkol_tm_activation_t act, const Iv& in, Iv& out) {
    switch (act) {
        case ESHKOL_TM_ACT_IDENTITY:
            out = in;
            return true;
        case ESHKOL_TM_ACT_TANH:
            out = {Iv::dn2(std::tanh(in.lo)), Iv::up2(std::tanh(in.hi))};
            return true;
        case ESHKOL_TM_ACT_SIGMOID:
            out = {Iv::dn2(sigmoid_fn(in.lo)), Iv::up2(sigmoid_fn(in.hi))};
            return true;
        case ESHKOL_TM_ACT_SOFTPLUS:
            out = {Iv::dn2(softplus_fn(in.lo)), Iv::up2(softplus_fn(in.hi))};
            return true;
        case ESHKOL_TM_ACT_SQUARE: {
            double a = in.lo * in.lo, b = in.hi * in.hi;
            double lo = (in.lo <= 0.0 && in.hi >= 0.0) ? 0.0 : Iv::dn(std::min(a, b));
            double hi = Iv::up(std::max(a, b));
            out = {lo, hi};
            return true;
        }
        case ESHKOL_TM_ACT_SIN: {
            /* Sound conservative enclosure for sin over [lo, hi] */
            if (in.hi - in.lo >= 6.283185307179586) {
                out = {-1.0, 1.0};
            } else {
                double slo = std::sin(in.lo), shi = std::sin(in.hi);
                double lo = Iv::dn2(std::min(slo, shi));
                double hi = Iv::up2(std::max(slo, shi));
                /* Check for interior extrema pi/2 + k*pi */
                double k_start = std::floor((in.lo - 1.5707963267948966) / 3.1415926535897932);
                for (int step = 0; step < 5; step++) {
                    double cand = 1.5707963267948966 + (k_start + step) * 3.1415926535897932;
                    if (cand >= in.lo && cand <= in.hi) {
                        double v = std::sin(cand);
                        lo = std::min(lo, Iv::dn2(v));
                        hi = std::max(hi, Iv::up2(v));
                    }
                }
                out = {std::max(-1.0, lo), std::min(1.0, hi)};
            }
            return true;
        }
        case ESHKOL_TM_ACT_HEAVISIDE_STEP:
            return false;
    }
    return false;
}

/* Order-2 Multivariate Taylor Model over z in [-1, 1]^d:
 *   T(z) = a0 + sum_i g[i] z_i + 0.5 * sum_{i,j} H[i,j] z_i z_j + rem */
struct TaylorModel2 {
    uint32_t d = 0;
    double a0 = 0.0;
    std::vector<double> g;
    std::vector<double> H; /* Symmetric d x d */
    Iv rem{0.0, 0.0};

    explicit TaylorModel2(uint32_t dim = 0)
        : d(dim), a0(0.0), g(dim, 0.0), H(static_cast<size_t>(dim) * dim, 0.0), rem{0.0, 0.0} {}

    /* Rigorous outward-rounded bound of linear part L(z) = sum_i g_i z_i. */
    Iv bound_linear() const {
        double r = 0.0;
        for (uint32_t i = 0; i < d; i++) {
            r = Iv::up(r + std::fabs(g[i]));
        }
        return {-r, r};
    }

    /* Rigorous outward-rounded bound of quadratic part Q(z) = 0.5 * z^T H z. */
    Iv bound_quadratic() const {
        Iv acc = Iv::point(0.0);
        for (uint32_t i = 0; i < d; i++) {
            double diag = H[i * d + i];
            /* 0.5 * diag * z_i^2 with z_i^2 in [0, 1] */
            Iv term = (diag >= 0.0) ? Iv{0.0, Iv::up(0.5 * diag)}
                                    : Iv{Iv::dn(0.5 * diag), 0.0};
            acc = acc + term;
            for (uint32_t j = i + 1; j < d; j++) {
                /* 0.5 * (H_ij + H_ji) * z_i * z_j = H_ij * [-1, 1] */
                double off = std::fabs(0.5 * (H[i * d + j] + H[j * d + i]));
                acc = acc + Iv{-Iv::up(off), Iv::up(off)};
            }
        }
        return acc;
    }

    /* Rigorous outward-rounded bound of deviation delta(z) = T(z) - a0. */
    Iv bound_deviation() const {
        return bound_linear() + bound_quadratic() + rem;
    }

    /* Rigorous outward-rounded bound of total value T(z) over [-1, 1]^d. */
    Iv bound_total() const {
        return Iv::point(a0) + bound_deviation();
    }
};

/* Apply affine combination y = sum_k w_k * T_k + bias with outward-rounded fp error tracking. */
TaylorModel2 tm_affine_combine(uint32_t d,
                               const std::vector<TaylorModel2>& inputs,
                               const double* weights,
                               double bias,
                               const TaylorModel2* residual_opt) {
    TaylorModel2 out(d);
    Iv a0_iv = Iv::point(bias);
    Iv rem_acc = Iv::point(0.0);

    std::vector<Iv> g_iv(d, Iv::point(0.0));
    std::vector<Iv> h_iv(static_cast<size_t>(d) * d, Iv::point(0.0));

    for (size_t k = 0; k < inputs.size(); k++) {
        double w = weights[k];
        if (w == 0.0) continue;
        const TaylorModel2& tk = inputs[k];
        a0_iv = a0_iv + Iv::point(tk.a0).scale(w);
        rem_acc = rem_acc + tk.rem.scale(w);
        for (uint32_t i = 0; i < d; i++) {
            g_iv[i] = g_iv[i] + Iv::point(tk.g[i]).scale(w);
        }
        for (size_t ij = 0; ij < h_iv.size(); ij++) {
            h_iv[ij] = h_iv[ij] + Iv::point(tk.H[ij]).scale(w);
        }
    }

    if (residual_opt) {
        a0_iv = a0_iv + Iv::point(residual_opt->a0);
        rem_acc = rem_acc + residual_opt->rem;
        for (uint32_t i = 0; i < d; i++) {
            g_iv[i] = g_iv[i] + Iv::point(residual_opt->g[i]);
        }
        for (size_t ij = 0; ij < h_iv.size(); ij++) {
            h_iv[ij] = h_iv[ij] + Iv::point(residual_opt->H[ij]);
        }
    }

    /* Extract midpoints into polynomial coefficients and sweep exact rounding widths into rem */
    out.a0 = 0.5 * (a0_iv.lo + a0_iv.hi);
    rem_acc = rem_acc + (a0_iv - Iv::point(out.a0));

    for (uint32_t i = 0; i < d; i++) {
        out.g[i] = 0.5 * (g_iv[i].lo + g_iv[i].hi);
        double err = std::max(out.g[i] - g_iv[i].lo, g_iv[i].hi - out.g[i]);
        rem_acc = rem_acc + Iv::sym(Iv::up(err));
    }
    for (uint32_t i = 0; i < d; i++) {
        for (uint32_t j = 0; j < d; j++) {
            size_t ij = static_cast<size_t>(i) * d + j;
            out.H[ij] = 0.5 * (h_iv[ij].lo + h_iv[ij].hi);
            double err = std::max(out.H[ij] - h_iv[ij].lo, h_iv[ij].hi - out.H[ij]);
            rem_acc = rem_acc + Iv::sym(Iv::up(0.5 * err));
        }
    }
    out.rem = rem_acc;
    return out;
}

/* Apply smooth C^infty univariate activation phi(T) with rigorous order-2 Taylor remainder. */
bool tm_apply_activation(eshkol_tm_activation_t act,
                         const TaylorModel2& in,
                         TaylorModel2& out) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, m3 = 0.0;
    if (!eval_act_taylor_coeffs(act, in.a0, d0, d1, d2, m3)) {
        return false; /* Non-Lipschitz / discontinuous -> fail closed */
    }

    const uint32_t d = in.d;
    out = TaylorModel2(d);
    out.a0 = d0;

    /* Account for libm rounding in d0, d1, d2 conservatively */
    Iv rem = Iv::sym(Iv::up2(std::fabs(d0) * 1e-15 + 1e-15));
    Iv iv_L = in.bound_linear();
    Iv iv_Q = in.bound_quadratic();
    Iv iv_QR = iv_Q + in.rem;
    Iv iv_delta = iv_L + iv_QR;

    /* Linear part: g_out = d1 * g_in */
    for (uint32_t i = 0; i < d; i++) {
        Iv gi = Iv::point(in.g[i]).scale(d1);
        out.g[i] = 0.5 * (gi.lo + gi.hi);
        double err = std::max(out.g[i] - gi.lo, gi.hi - out.g[i]);
        rem = rem + Iv::sym(Iv::up(err));
    }

    /* Quadratic part: H_out = d1 * H_in + d2 * (g_in * g_in^T) */
    for (uint32_t i = 0; i < d; i++) {
        for (uint32_t j = 0; j < d; j++) {
            size_t ij = static_cast<size_t>(i) * d + j;
            Iv hij = Iv::point(in.H[ij]).scale(d1) +
                     (Iv::point(in.g[i]) * Iv::point(in.g[j])).scale(d2);
            out.H[ij] = 0.5 * (hij.lo + hij.hi);
            double err = std::max(out.H[ij] - hij.lo, hij.hi - out.H[ij]);
            rem = rem + Iv::sym(Iv::up(0.5 * err));
        }
    }

    /* Remainder contributions:
     *   1. d1 * in.rem
     *   2. 0.5 * d2 * (delta^2 - L^2) = 0.5 * d2 * (Q + rem) * (2*L + Q + rem)
     *   3. Lagrange 3rd-order remainder: (1/6) * [-M3, M3] * delta^3 */
    rem = rem + in.rem.scale(d1);
    Iv quad_excess = (iv_QR * (iv_L.scale(2.0) + iv_QR)).scale(0.5 * d2);
    rem = rem + quad_excess;

    if (m3 > 0.0) {
        double max_d = iv_delta.max_abs();
        double d3_up = Iv::up(Iv::up(max_d * max_d) * max_d);
        double r3_up = Iv::up(Iv::up(m3 * d3_up) / 6.0);
        rem = rem + Iv{-r3_up, r3_up};
    }

    out.rem = rem;
    return true;
}

struct WorldLayer {
    uint32_t in_dim = 0;
    uint32_t out_dim = 0;
    std::vector<double> W;
    std::vector<double> b;
    eshkol_tm_activation_t act = ESHKOL_TM_ACT_IDENTITY;
    bool add_residual = false;
};

} /* anonymous namespace */

struct eshkol_tm_world_model {
    uint32_t input_dim = 0;
    uint32_t current_dim = 0;
    std::vector<WorldLayer> layers;

    explicit eshkol_tm_world_model(uint32_t in_d) : input_dim(in_d), current_dim(in_d) {}
};

extern "C" eshkol_tm_world_model_t* eshkol_tm_model_create(uint32_t input_dim) {
    if (input_dim == 0) return nullptr;
    return new (std::nothrow) eshkol_tm_world_model(input_dim);
}

extern "C" void eshkol_tm_model_destroy(eshkol_tm_world_model_t* model) {
    delete model;
}

extern "C" bool eshkol_tm_model_add_layer(eshkol_tm_world_model_t* model,
                                          uint32_t out_dim,
                                          const double* weights,
                                          const double* bias,
                                          eshkol_tm_activation_t activation,
                                          bool add_residual) {
    if (!model || out_dim == 0 || !weights || !bias) return false;
    if (add_residual && out_dim != model->current_dim) return false;
    size_t wsize = static_cast<size_t>(out_dim) * model->current_dim;
    for (size_t i = 0; i < wsize; i++) if (!std::isfinite(weights[i])) return false;
    for (uint32_t i = 0; i < out_dim; i++) if (!std::isfinite(bias[i])) return false;

    WorldLayer ly;
    ly.in_dim = model->current_dim;
    ly.out_dim = out_dim;
    ly.W.assign(weights, weights + wsize);
    ly.b.assign(bias, bias + out_dim);
    ly.act = activation;
    ly.add_residual = add_residual;
    model->layers.push_back(std::move(ly));
    model->current_dim = out_dim;
    return true;
}

extern "C" uint32_t eshkol_tm_model_input_dim(const eshkol_tm_world_model_t* model) {
    return model ? model->input_dim : 0u;
}

extern "C" uint32_t eshkol_tm_model_output_dim(const eshkol_tm_world_model_t* model) {
    return model ? model->current_dim : 0u;
}

extern "C" bool eshkol_tm_model_eval_point(const eshkol_tm_world_model_t* model,
                                           const double* input,
                                           double* out_output) {
    if (!model || !input || !out_output) return false;
    std::vector<double> cur(input, input + model->input_dim);
    std::vector<double> nxt;
    for (const WorldLayer& ly : model->layers) {
        nxt.assign(ly.out_dim, 0.0);
        for (uint32_t i = 0; i < ly.out_dim; i++) {
            double acc = ly.b[i];
            const double* wrow = ly.W.data() + static_cast<size_t>(i) * ly.in_dim;
            for (uint32_t j = 0; j < ly.in_dim; j++) acc += wrow[j] * cur[j];
            double y = eval_act_point(ly.act, acc);
            if (ly.add_residual) y += cur[i];
            nxt[i] = y;
        }
        cur.swap(nxt);
    }
    std::memcpy(out_output, cur.data(), cur.size() * sizeof(double));
    return true;
}

extern "C" eshkol_tm_status_t eshkol_tm_model_certify(const eshkol_tm_world_model_t* model,
                                                      const eshkol_tm_interval_t* input_box,
                                                      eshkol_tm_interval_t* out_bounds) {
    if (!model || !input_box || !out_bounds) return ESHKOL_TM_ERR_INVALID_ARG;
    const uint32_t d = model->input_dim;
    const uint32_t out_d = model->current_dim;

    auto set_uncertified = [&]() {
        for (uint32_t i = 0; i < out_d; i++) {
            out_bounds[i].lo = -kInf;
            out_bounds[i].hi = +kInf;
        }
    };

    for (uint32_t i = 0; i < d; i++) {
        if (!std::isfinite(input_box[i].lo) || !std::isfinite(input_box[i].hi) ||
            input_box[i].lo > input_box[i].hi) {
            set_uncertified();
            return ESHKOL_TM_ERR_INVALID_ARG;
        }
    }

    /* Initialize input Taylor models T_i(z) = c_i + r_i * z_i + fp_rem */
    std::vector<TaylorModel2> cur(d, TaylorModel2(d));
    for (uint32_t i = 0; i < d; i++) {
        double c = 0.5 * (input_box[i].lo + input_box[i].hi);
        double r = Iv::up(0.5 * (input_box[i].hi - input_box[i].lo));
        cur[i].a0 = c;
        cur[i].g[i] = r;
        double err_lo = Iv::dn(c - r) - input_box[i].lo;
        double err_hi = input_box[i].hi - Iv::up(c + r);
        double pad = std::max(0.0, std::max(err_lo, err_hi));
        cur[i].rem = {-pad, pad};
    }

    for (const WorldLayer& ly : model->layers) {
        std::vector<TaylorModel2> nxt(ly.out_dim, TaylorModel2(d));
        for (uint32_t i = 0; i < ly.out_dim; i++) {
            const double* wrow = ly.W.data() + static_cast<size_t>(i) * ly.in_dim;
            TaylorModel2 pre = tm_affine_combine(d, cur, wrow, ly.b[i], nullptr);
            TaylorModel2 post(d);
            if (!tm_apply_activation(ly.act, pre, post)) {
                set_uncertified();
                return ESHKOL_TM_ERR_NON_LIPSCHITZ;
            }
            if (ly.add_residual) {
                /* Exact Taylor-2 addition preserves linear skip correlation! */
                double one = 1.0;
                std::vector<TaylorModel2> single = {std::move(post)};
                nxt[i] = tm_affine_combine(d, single, &one, 0.0, &cur[i]);
            } else {
                nxt[i] = std::move(post);
            }
        }
        cur.swap(nxt);
    }

    for (uint32_t i = 0; i < out_d; i++) {
        Iv b = cur[i].bound_total();
        out_bounds[i].lo = b.lo;
        out_bounds[i].hi = b.hi;
    }
    return ESHKOL_TM_OK;
}

extern "C" eshkol_tm_status_t eshkol_tm_model_naive_interval(const eshkol_tm_world_model_t* model,
                                                             const eshkol_tm_interval_t* input_box,
                                                             eshkol_tm_interval_t* out_bounds) {
    if (!model || !input_box || !out_bounds) return ESHKOL_TM_ERR_INVALID_ARG;
    const uint32_t d = model->input_dim;
    const uint32_t out_d = model->current_dim;

    auto set_uncertified = [&]() {
        for (uint32_t i = 0; i < out_d; i++) {
            out_bounds[i].lo = -kInf;
            out_bounds[i].hi = +kInf;
        }
    };

    std::vector<Iv> cur(d);
    for (uint32_t i = 0; i < d; i++) cur[i] = {input_box[i].lo, input_box[i].hi};

    for (const WorldLayer& ly : model->layers) {
        std::vector<Iv> nxt(ly.out_dim);
        for (uint32_t i = 0; i < ly.out_dim; i++) {
            Iv acc = Iv::point(ly.b[i]);
            const double* wrow = ly.W.data() + static_cast<size_t>(i) * ly.in_dim;
            for (uint32_t j = 0; j < ly.in_dim; j++) {
                acc = acc + cur[j].scale(wrow[j]);
            }
            Iv post{};
            if (!eval_act_naive_iv(ly.act, acc, post)) {
                set_uncertified();
                return ESHKOL_TM_ERR_NON_LIPSCHITZ;
            }
            nxt[i] = ly.add_residual ? (post + cur[i]) : post;
        }
        cur.swap(nxt);
    }

    for (uint32_t i = 0; i < out_d; i++) {
        out_bounds[i].lo = cur[i].lo;
        out_bounds[i].hi = cur[i].hi;
    }
    return ESHKOL_TM_OK;
}
