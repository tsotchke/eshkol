/* Gradcheck for the shared scaled geometry forward and its bridge adjoints. */

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/tensor_backward.h"
#include "eshkol/backend/riemannian_core.h"

extern "C" {
typedef struct arena arena_t;
arena_t* get_global_arena(void);
void* arena_allocate_zeroed(arena_t*, size_t);
ad_tape_t* arena_allocate_tape(arena_t*, size_t);
ad_node_t* arena_allocate_ad_node(arena_t*);
}

struct Dual {
    long double value;
    long double tangent;
};

static uint64_t random_state = UINT64_C(0x9e3779b97f4a7c15);
static double random_unit() {
    random_state = random_state * UINT64_C(6364136223846793005) + 1;
    return (double)((random_state >> 11) & UINT64_C(0x1fffffffffffff)) /
           9007199254740992.0;
}

static Dual add(Dual a, Dual b) { return {a.value + b.value, a.tangent + b.tangent}; }
static Dual sub(Dual a, Dual b) { return {a.value - b.value, a.tangent - b.tangent}; }
static Dual neg(Dual a) { return {-a.value, -a.tangent}; }
static Dual mul(Dual a, Dual b) {
    return {a.value * b.value, a.tangent * b.value + a.value * b.tangent};
}
static Dual divd(Dual a, Dual b) {
    return {a.value / b.value,
            (a.tangent * b.value - a.value * b.tangent) /
                (b.value * b.value)};
}
static Dual mulc(Dual a, long double b) { return {a.value * b, a.tangent * b}; }
static Dual sqrtd(Dual a) {
    const long double r = sqrtl(a.value);
    return {r, a.tangent / (2.0L * r)};
}
static Dual tanhd(Dual a) {
    const long double t = tanhl(a.value);
    return {t, a.tangent * (1.0L - t * t)};
}
static Dual asinhd(Dual a) {
    return {asinhl(a.value), a.tangent / sqrtl(1.0L + a.value * a.value)};
}

static Dual normd(const std::vector<Dual>& a) {
    Dual sum = {0.0L, 0.0L};
    for (const Dual& x : a) sum = add(sum, mul(x, x));
    return sqrtd(sum);
}
static Dual dotd(const std::vector<Dual>& a, const std::vector<Dual>& b) {
    Dual sum = {0.0L, 0.0L};
    for (size_t i = 0; i < a.size(); ++i) sum = add(sum, mul(a[i], b[i]));
    return sum;
}

static Dual distance_dual(const std::vector<Dual>& x,
                          const std::vector<Dual>& y, long double c) {
    std::vector<Dual> delta(x.size());
    for (size_t i = 0; i < x.size(); ++i) delta[i] = sub(x[i], y[i]);
    const Dual dn = normd(delta);
    if (c == 0.0L) return dn;
    const Dual one = {1.0L, 0.0L};
    const Dual nx = sub(one, mulc(dotd(x, x), c));
    const Dual ny = sub(one, mulc(dotd(y, y), c));
    const Dual den = sqrtd(mul(nx, ny));
    const Dual z = mulc(divd(dn, den), sqrtl(c));
    return mulc(asinhd(z), 2.0L / sqrtl(c));
}

static std::vector<Dual> exp_dual(const std::vector<Dual>& x,
                                  const std::vector<Dual>& v, long double c) {
    std::vector<Dual> out(x.size());
    if (c == 0.0L) {
        for (size_t i = 0; i < x.size(); ++i) out[i] = add(x[i], v[i]);
        return out;
    }
    const Dual one = {1.0L, 0.0L};
    const Dual nb = sub(one, mulc(dotd(x, x), c));
    const Dual lambda = divd({2.0L, 0.0L}, nb);
    const Dual vn = normd(v);
    const Dual z = mulc(mul(lambda, vn), sqrtl(c) / 2.0L);
    Dual tau;
    if (z.value == 0.0L) tau = {1.0L, 0.0L};
    else tau = divd(tanhd(z), z);
    const Dual f = mulc(mul(lambda, tau), 0.5L);
    std::vector<Dual> s(x.size());
    for (size_t i = 0; i < x.size(); ++i) s[i] = mul(f, v[i]);
    const Dual xs = dotd(x, s);
    const Dual ss = dotd(s, s);
    const Dual xx = dotd(x, x);
    const Dual A = add(add(one, mulc(xs, 2.0L * c)), mulc(ss, c));
    const Dual B = sub(one, mulc(xx, c));
    const Dual D = add(add(one, mulc(xs, 2.0L * c)), mulc(mul(xx, ss), c * c));
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = divd(add(mul(A, x[i]), mul(B, s[i])), D);
    return out;
}

static std::vector<Dual> log_dual(const std::vector<Dual>& x,
                                  const std::vector<Dual>& y, long double c) {
    std::vector<Dual> out(x.size());
    if (c == 0.0L) {
        for (size_t i = 0; i < x.size(); ++i) out[i] = sub(y[i], x[i]);
        return out;
    }
    const Dual one = {1.0L, 0.0L};
    const Dual nb = sub(one, mulc(dotd(x, x), c));
    const Dual q = add(one, mulc(dotd(x, y), -c));
    const Dual yy = dotd(y, y);
    const Dual xx = dotd(x, x);
    const Dual xy = dotd(x, y);
    const Dual gram = sub(mul(xx, yy), mul(xy, xy));
    const Dual den = add(mul(q, q), mulc(gram, c * c));
    const Dual na = add(den, mul(mulc(yy, c), nb));
    std::vector<Dual> u(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        u[i] = sub(mul(nb, y[i]), mul(na, x[i]));
    const Dual un = normd(u);
    const Dual distance = distance_dual(x, y, c);
    const Dual lambda = divd({2.0L, 0.0L}, nb);
    const Dual coefficient = divd(divd(distance, lambda), un);
    for (size_t i = 0; i < x.size(); ++i) out[i] = mul(coefficient, u[i]);
    return out;
}

static ad_node_t* variable(const std::vector<double>& values) {
    const int64_t shape[1] = {(int64_t)values.size()};
    ad_node_t* node = arena_allocate_ad_node(get_global_arena());
    double* data = (double*)arena_allocate_zeroed(get_global_arena(),
                                                   values.size() * sizeof(double));
    int64_t* dims = (int64_t*)arena_allocate_zeroed(get_global_arena(), sizeof shape);
    std::memcpy(data, values.data(), values.size() * sizeof(double));
    std::memcpy(dims, shape, sizeof shape);
    node->type = AD_NODE_VARIABLE;
    node->tensor_value = data;
    node->shape = dims;
    node->ndim = 1;
    return node;
}

static void sweep(ad_tape_t* tape) {
    for (size_t i = tape->num_nodes; i-- > 0;)
        eshkol_tensor_backward_dispatch(tape->nodes[i]);
}

static double relative_error(long double got, long double want) {
    const long double scale = 1.0L + fabsl(want);
    return (double)(fabsl(got - want) / scale);
}

static bool check_case(const std::vector<double>& x, const std::vector<double>& y,
                       const std::vector<double>& v, double curvature) {
    const size_t n = x.size();
    const long double c = -(long double)curvature;
    std::vector<double> upstream(n);
    for (size_t i = 0; i < n; ++i) upstream[i] = ((int)(i % 5) - 2) * 0.37;
    bool ok = true;
    const bool report_boundary = curvature == -1e-320 && n == 3 &&
                                  x[0] == 0.25;
    double max_dual_rel = 0.0;
    const double dual_tolerance = std::abs(curvature) < 1e-100 ? 2e-12 : 2e-9;

    ad_tape_t* td = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xd = variable(x), *yd = variable(y);
    ad_node_t* od = ad_hyperbolic_distance(td, xd, yd, curvature);
    if (!od) {
        std::printf("distance forward refused n=%zu curvature=%.17g\n", n, curvature);
        return false;
    }
    ((double*)od->tensor_gradient)[0] = 1.0;
    sweep(td);
    for (size_t j = 0; j < n; ++j) {
        std::vector<Dual> xd1(n), yd1(n);
        for (size_t i = 0; i < n; ++i)
            xd1[i] = {(long double)x[i], i == j ? 1.0L : 0.0L};
        for (size_t i = 0; i < n; ++i) yd1[i] = {(long double)y[i], 0.0L};
        const long double want_x = distance_dual(xd1, yd1, c).tangent;
        yd1[j] = {(long double)y[j], 1.0L};
        for (size_t i = 0; i < n; ++i) xd1[i] = {(long double)x[i], 0.0L};
        const long double want_y = distance_dual(xd1, yd1, c).tangent;
        const double err_x = relative_error(((double*)xd->tensor_gradient)[j], want_x);
        const double err_y = relative_error(((double*)yd->tensor_gradient)[j], want_y);
        max_dual_rel = std::max(max_dual_rel, std::max(err_x, err_y));
        ok = ok && err_x < dual_tolerance && err_y < dual_tolerance;
    }
    if (!ok) {
        std::printf("scaled_or_distance_failed n=%zu\n", n);
        return false;
    }

    ad_tape_t* te = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xe = variable(x), *ve = variable(v);
    ad_node_t* oe = ad_poincare_exp_map(te, xe, ve, curvature);
    if (!oe) {
        std::printf("exp forward refused n=%zu curvature=%.17g\n", n, curvature);
        return false;
    }
    std::memcpy(oe->tensor_gradient, upstream.data(), n * sizeof(double));
    sweep(te);
    for (size_t j = 0; j < n; ++j) {
        std::vector<Dual> xd1(n), vd1(n);
        for (size_t i = 0; i < n; ++i) {
            xd1[i] = {(long double)x[i], i == j ? 1.0L : 0.0L};
            vd1[i] = {(long double)v[i], 0.0L};
        }
        std::vector<Dual> out = exp_dual(xd1, vd1, c);
        long double want_x = 0.0L;
        for (size_t i = 0; i < n; ++i) want_x += (long double)upstream[i] * out[i].tangent;
        for (size_t i = 0; i < n; ++i) {
            xd1[i] = {(long double)x[i], 0.0L};
            vd1[i] = {(long double)v[i], i == j ? 1.0L : 0.0L};
        }
        out = exp_dual(xd1, vd1, c);
        long double want_v = 0.0L;
        for (size_t i = 0; i < n; ++i) want_v += (long double)upstream[i] * out[i].tangent;
        const double err_x = relative_error(((double*)xe->tensor_gradient)[j], want_x);
        const double err_v = relative_error(((double*)ve->tensor_gradient)[j], want_v);
        max_dual_rel = std::max(max_dual_rel, std::max(err_x, err_v));
        ok = ok && err_x < dual_tolerance && err_v < dual_tolerance;
    }
    if (!ok) {
        std::printf("scaled_exp_failed n=%zu\n", n);
        return false;
    }

    ad_tape_t* tl = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xl = variable(x), *yl = variable(y);
    ad_node_t* ol = ad_poincare_log_map(tl, xl, yl, curvature);
    if (!ol) {
        std::printf("log forward refused n=%zu curvature=%.17g\n", n, curvature);
        return false;
    }
    std::memcpy(ol->tensor_gradient, upstream.data(), n * sizeof(double));
    sweep(tl);
    for (size_t j = 0; j < n; ++j) {
        std::vector<Dual> xd1(n), yd1(n);
        for (size_t i = 0; i < n; ++i) {
            xd1[i] = {(long double)x[i], i == j ? 1.0L : 0.0L};
            yd1[i] = {(long double)y[i], 0.0L};
        }
        std::vector<Dual> out = log_dual(xd1, yd1, c);
        long double want_x = 0.0L;
        for (size_t i = 0; i < n; ++i) want_x += (long double)upstream[i] * out[i].tangent;
        for (size_t i = 0; i < n; ++i) {
            xd1[i] = {(long double)x[i], 0.0L};
            yd1[i] = {(long double)y[i], i == j ? 1.0L : 0.0L};
        }
        out = log_dual(xd1, yd1, c);
        long double want_y = 0.0L;
        for (size_t i = 0; i < n; ++i) want_y += (long double)upstream[i] * out[i].tangent;
        const double err_x = relative_error(((double*)xl->tensor_gradient)[j], want_x);
        const double err_y = relative_error(((double*)yl->tensor_gradient)[j], want_y);
        max_dual_rel = std::max(max_dual_rel, std::max(err_x, err_y));
        ok = ok && err_x < dual_tolerance && err_y < dual_tolerance;
    }
    if (!ok) return false;
    if (ok && n == 3 && curvature == -0.7) {
        const double h = 1e-6;
        auto distance_value = [&](const std::vector<double>& a,
                                  const std::vector<double>& b) {
            ad_node_t* an = variable(a), *bn = variable(b);
            ad_node_t* result = ad_hyperbolic_distance(nullptr, an, bn, curvature);
            return result ? ((const double*)result->tensor_value)[0] : NAN;
        };
        auto map_loss = [&](const std::vector<double>& a,
                            const std::vector<double>& b, bool logarithm) {
            ad_node_t* an = variable(a), *bn = variable(b);
            ad_node_t* result = logarithm
                ? ad_poincare_log_map(nullptr, an, bn, curvature)
                : ad_poincare_exp_map(nullptr, an, bn, curvature);
            if (!result) return (double)NAN;
            const double* values = (const double*)result->tensor_value;
            double loss = 0.0;
            for (size_t i = 0; i < n; ++i) loss += upstream[i] * values[i];
            return loss;
        };
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> plus = x, minus = x;
            plus[j] += h; minus[j] -= h;
            const double fd_d = (distance_value(plus, y) - distance_value(minus, y)) / (2.0 * h);
            const double fd_e = (map_loss(plus, v, false) - map_loss(minus, v, false)) / (2.0 * h);
            const double fd_l = (map_loss(plus, y, true) - map_loss(minus, y, true)) / (2.0 * h);
            ok = ok && std::fabs(fd_d - ((double*)xd->tensor_gradient)[j]) < 2e-6;
            ok = ok && std::fabs(fd_e - ((double*)xe->tensor_gradient)[j]) < 2e-6;
            ok = ok && std::fabs(fd_l - ((double*)xl->tensor_gradient)[j]) < 2e-6;
        }
    }
    if (report_boundary)
        std::printf("subnormal_boundary: max_binary128_dual_relative_error=%.17g\n",
                    max_dual_rel);
    return ok;
}

static bool check_subnormal_finite_difference() {
    const int64_t shape[1] = {3};
    const double curvature = -1e-320;
    const std::vector<double> x = {0.25, -0.125, 0.0625};
    const std::vector<double> y = {-0.125, 0.0625, -0.03125};
    const std::vector<double> upstream = {-0.74, -0.37, 0.0};
    ad_tape_t* td = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xd = variable(x), *yd = variable(y);
    ad_node_t* od = ad_hyperbolic_distance(td, xd, yd, curvature);
    if (!od) return false;
    ((double*)od->tensor_gradient)[0] = 1.0;
    sweep(td);
    ad_tape_t* tl = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xl = variable(x), *yl = variable(y);
    ad_node_t* ol = ad_poincare_log_map(tl, xl, yl, curvature);
    if (!ol) return false;
    std::memcpy(ol->tensor_gradient, upstream.data(), upstream.size() * sizeof(double));
    sweep(tl);
    auto distance_value = [&](const std::vector<double>& a,
                              const std::vector<double>& b) {
        ad_node_t* an = variable(a), *bn = variable(b);
        ad_node_t* r = ad_hyperbolic_distance(nullptr, an, bn, curvature);
        return r ? ((const double*)r->tensor_value)[0] : NAN;
    };
    auto log_loss = [&](const std::vector<double>& a,
                        const std::vector<double>& b) {
        ad_node_t* an = variable(a), *bn = variable(b);
        ad_node_t* r = ad_poincare_log_map(nullptr, an, bn, curvature);
        if (!r) return (double)NAN;
        const double* values = (const double*)r->tensor_value;
        double loss = 0.0;
        for (size_t i = 0; i < upstream.size(); ++i) loss += upstream[i] * values[i];
        return loss;
    };
    const double h = 1e-7;
    double max_distance_err = 0.0, max_log_err = 0.0;
    for (size_t j = 0; j < x.size(); ++j) {
        std::vector<double> plus = x, minus = x;
        plus[j] += h; minus[j] -= h;
        const double fd_d = (distance_value(plus, y) - distance_value(minus, y)) / (2.0 * h);
        const double fd_l = (log_loss(plus, y) - log_loss(minus, y)) / (2.0 * h);
        max_distance_err = std::max(max_distance_err,
            std::fabs(fd_d - ((double*)xd->tensor_gradient)[j]) / (1.0 + std::fabs(fd_d)));
        max_log_err = std::max(max_log_err,
            std::fabs(fd_l - ((double*)xl->tensor_gradient)[j]) / (1.0 + std::fabs(fd_l)));
    }
    for (size_t j = 0; j < y.size(); ++j) {
        std::vector<double> plus = y, minus = y;
        plus[j] += h; minus[j] -= h;
        const double fd_d = (distance_value(x, plus) - distance_value(x, minus)) / (2.0 * h);
        const double fd_l = (log_loss(x, plus) - log_loss(x, minus)) / (2.0 * h);
        max_distance_err = std::max(max_distance_err,
            std::fabs(fd_d - ((double*)yd->tensor_gradient)[j]) / (1.0 + std::fabs(fd_d)));
        max_log_err = std::max(max_log_err,
            std::fabs(fd_l - ((double*)yl->tensor_gradient)[j]) / (1.0 + std::fabs(fd_l)));
    }
    std::printf("subnormal_boundary: max_distance_fd_relative_error=%.17g "
                "max_log_fd_relative_error=%.17g\n", max_distance_err, max_log_err);
    return max_distance_err < 1e-8 && max_log_err < 1e-8;
}

static bool check_log_coincidence() {
    const std::vector<double> x = {0.0, 0.0};
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = variable(x), *yn = variable(x);
    ad_node_t* out = ad_poincare_log_map(tape, xn, yn, -1.0);
    if (!out) return false;
    ((double*)out->tensor_gradient)[0] = 0.7;
    ((double*)out->tensor_gradient)[1] = -0.4;
    sweep(tape);
    const double* dx = (const double*)xn->tensor_gradient;
    const double* dy = (const double*)yn->tensor_gradient;
    return dx && dy && dx[0] == -0.7 && dx[1] == 0.4 &&
           dy[0] == 0.7 && dy[1] == -0.4;
}

static bool check_tiny_attention() {
    const int64_t shape[3] = {1, 2, 1};
    const std::vector<double> q = {0.0, 3e-300};
    const std::vector<double> k = {1e-300, 5e-300};
    const std::vector<double> v = {1.0, 2.0};
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    auto make = [&](const std::vector<double>& values) {
        ad_node_t* node = arena_allocate_ad_node(get_global_arena());
        double* data = (double*)arena_allocate_zeroed(get_global_arena(), values.size() * sizeof(double));
        int64_t* dims = (int64_t*)arena_allocate_zeroed(get_global_arena(), sizeof shape);
        std::memcpy(data, values.data(), values.size() * sizeof(double));
        std::memcpy(dims, shape, sizeof shape);
        node->type = AD_NODE_VARIABLE; node->tensor_value = data;
        node->shape = dims; node->ndim = 3; return node;
    };
    ad_node_t* qn = make(q), *kn = make(k), *vn = make(v);
    ad_node_t* out = ad_geodesic_attention(tape, qn, kn, vn, 1, 0.0, false);
    if (!out) return false;
    ((double*)out->tensor_gradient)[0] = 1.0;
    ((double*)out->tensor_gradient)[1] = -0.5;
    sweep(tape);
    if (!qn->tensor_gradient || !kn->tensor_gradient || !vn->tensor_gradient) return false;
    for (int i = 0; i < 2; ++i)
        if (!std::isfinite(((double*)qn->tensor_gradient)[i]) ||
            !std::isfinite(((double*)kn->tensor_gradient)[i]) ||
            !std::isfinite(((double*)vn->tensor_gradient)[i])) return false;
    return true;
}

static ad_node_t* tensor3(const std::vector<double>& values,
                           const int64_t shape[3]) {
    ad_node_t* node = arena_allocate_ad_node(get_global_arena());
    size_t count = (size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2];
    double* data = (double*)arena_allocate_zeroed(get_global_arena(), count * sizeof(double));
    int64_t* dims = (int64_t*)arena_allocate_zeroed(get_global_arena(), 3 * sizeof(int64_t));
    std::memcpy(data, values.data(), count * sizeof(double));
    std::memcpy(dims, shape, 3 * sizeof(int64_t));
    node->type = AD_NODE_VARIABLE; node->tensor_value = data;
    node->shape = dims; node->ndim = 3;
    return node;
}

static bool check_attention_score_overflow() {
    const int64_t shape[3] = {1, 2, 1};
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* q = tensor3({0.0, 1e149}, shape);
    ad_node_t* k = tensor3({2e149, 3e149}, shape);
    ad_node_t* v = tensor3({1.0, 2.0}, shape);
    ad_node_t* out = ad_geodesic_attention(tape, q, k, v, 1, -1e-320, false);
    if (!out) return false;
    double* upstream = (double*)out->tensor_gradient;
    upstream[0] = 0.75;
    upstream[1] = -0.25;
    sweep(tape);
    const double* values = (const double*)out->tensor_value;
    const double* dq = (const double*)q->tensor_gradient;
    const double* dk = (const double*)k->tensor_gradient;
    const double* dv = (const double*)v->tensor_gradient;
    if (!dq || !dk || !dv) return false;
    double max_gradient_abs = 0.0;
    for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(dq[i]) ||
            !std::isfinite(dk[i]) || !std::isfinite(dv[i])) return false;
        max_gradient_abs = std::max(max_gradient_abs,
            std::max(std::fabs(dq[i]), std::max(std::fabs(dk[i]), std::fabs(dv[i]))));
    }
    std::printf("huge_attention: output0=%.17g output1=%.17g max_gradient_abs=%.17g\n",
                values[0], values[1], max_gradient_abs);
    return true;
}

static bool check_true_min_spherical_attention() {
    const int64_t shape[3] = {1, 2, 2};
    const double K = DBL_TRUE_MIN;
    const double R = 1.0 / std::sqrt(K);
    const double h = R / std::sqrt(2.0);
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* q = tensor3({R, 0.0, 0.0, R}, shape);
    ad_node_t* k = tensor3({h, h, -h, h}, shape);
    ad_node_t* v = tensor3({1.0, 2.0, 3.0, 4.0}, shape);
    ad_node_t* out = ad_geodesic_attention(tape, q, k, v, 1, K, false);
    if (!out) return false;
    const double* values = (const double*)out->tensor_value;
    if (!std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        !std::isfinite(values[2]) || !std::isfinite(values[3])) return false;
    double* upstream = (double*)out->tensor_gradient;
    for (int i = 0; i < 4; ++i) upstream[i] = (i & 1) ? -0.5 : 0.75;
    sweep(tape);
    const double* dq = (const double*)q->tensor_gradient;
    const double* dk = (const double*)k->tensor_gradient;
    const double* dv = (const double*)v->tensor_gradient;
    if (!dq || !dk || !dv) return false;
    double max_gradient_abs = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(dq[i]) || !std::isfinite(dk[i]) ||
            !std::isfinite(dv[i])) return false;
        max_gradient_abs = std::max(max_gradient_abs,
            std::max(std::fabs(dq[i]), std::max(std::fabs(dk[i]), std::fabs(dv[i]))));
    }
    std::printf("true_min_spherical_attention: output=(%.17g,%.17g,%.17g,%.17g) "
                "max_gradient_abs=%.17g\n", values[0], values[1], values[2], values[3],
                max_gradient_abs);
    return true;
}

static bool check_ball_parameter_extremes() {
    const double curvatures[] = {DBL_MAX, DBL_MAX / 2.0, DBL_MAX / 4.0};
    for (double c : curvatures) {
        if (!std::isfinite(eshkol_rm_ball_param(c)) ||
            eshkol_rm_ball_param(c) != c ||
            !std::isfinite(eshkol_rm_ball_param(-c)) ||
            eshkol_rm_ball_param(-c) != -c)
            return false;
    }
    return true;
}

static bool check_scaled_boundary_denominator() {
    const double x[2] = {0.0, 9.99999999999e149};
    const double B = 1e-300;
    const double got = eshkol_rm_one_minus_bnorm2(x, B, 2);
    double product_hi = 0.0, product_lo = 0.0;
    eshkol_rm_scaled_product4_dd(B, x[1], x[1], 1.0,
                                 &product_hi, &product_lo);
    /* Binary128 reference for these exact f64 inputs. x86 long double has
     * only 64 significand bits and rounds this near-one product too early. */
    const long double want = 1.9999100196141823908743937057113335866e-12L;
    const long double rel = std::fabs((long double)got - want) /
                            std::fabs(want);
    std::printf("scaled_boundary: denominator=%.17g reference=%.21Lg rel=%.3Le "
                "product=(%.17g,%.17g)\n", got, want, rel,
                product_hi, product_lo);
    return std::isfinite(got) && rel < 1e-12L;
}

static bool check_boundary_dimension_sweep() {
    double worst_rel = 0.0;
    int worst_n = 0;
    for (int n = 1; n <= 64; ++n) {
        std::vector<double> x((size_t)n);
        const double radius = 1.0 / std::sqrt((double)n);
        double interior_radius = radius;
        for (int ulp = 0; ulp < 4; ++ulp)
            interior_radius = std::nextafter(interior_radius, 0.0);
        for (int i = 0; i < n; ++i) x[(size_t)i] = interior_radius;
        const long double reference = [&] {
            long double sum = 0.0L;
            for (double value : x) sum += (long double)value * value;
            return 1.0L - sum;
        }();
        const double got = eshkol_rm_one_minus_bnorm2(x.data(), 1.0, n);
        if (!(reference > 0.0L) || !(got > 0.0)) {
            std::printf("boundary_dimensions_invalid n=%d reference=%.21Lg got=%.17g\n",
                        n, reference, got);
            return false;
        }
        const double relative = (double)(std::fabs((long double)got - reference) /
                                         reference);
        if (relative > worst_rel) {
            worst_rel = relative;
            worst_n = n;
        }
        /* The portable test binary has only the platform long double as a
         * reference (binary128 validation for the witness is reported by the
         * focused exact harness).  Keep this sweep as a sign/regression guard;
         * the binary128 witness below carries the tight numerical threshold. */
        if (relative >= 1e-3) {
            std::printf("boundary_dimensions_inaccurate n=%d reference=%.21Lg "
                        "got=%.17g relative=%.17g\n",
                        n, reference, got, relative);
            return false;
        }
    }
    /* This is the original two-coordinate near-boundary witness, retained in
     * addition to the n=1..64 family so the multidimensional residual path
     * cannot regress to the fix7 per-component scaling error. */
    const double witness[2] = {0.70183641524540785, 0.71233815441507509};
    const long double reference =
        8.28137702690322433096417081472246456e-17L;
    const double got = eshkol_rm_one_minus_bnorm2(witness, 1.0, 2);
    const double relative = (double)(std::fabs((long double)got - reference) /
                                     std::fabs(reference));
    std::printf("boundary_dimensions n=1..64 worst_rel=%.17g worst_n=%d "
                "witness_ref=%.21Lg witness_got=%.17g witness_rel=%.3Le\n",
                worst_rel, worst_n, reference, got, (long double)relative);
    return got > 0.0 && relative < 1e-12;
}

static bool check_curvature_hessian_extreme() {
    const double curvature = -std::ldexp(1.0, 404);
    const double radius = 1.0 / std::sqrt(-curvature);
    const double x[1] = {0.0};
    const double y[1] = {0.25 * radius};
    double distance = 0.0, gradient = 0.0, hessian = 0.0;
    const char* why = eshkol_rm_distance_dK(
        x, y, curvature, 1, &distance, &gradient, &hessian);
    const double expected_gradient = -4.2376387815863510661e-185;
    const double expected_hessian = 8.1751304839522634993e-308;
    const double gradient_rel = std::fabs(gradient - expected_gradient) /
                                std::fabs(expected_gradient);
    const double hessian_rel = std::fabs(hessian - expected_hessian) /
                               std::fabs(expected_hessian);
    std::printf("curvature_extreme K=-2^404 why=%s d1=%.17g d2=%.17g "
                "relative=(%.3e,%.3e)\n", why ? why : "PASS", gradient,
                hessian, gradient_rel, hessian_rel);
    return !why && std::isfinite(gradient) && std::isfinite(hessian) &&
           gradient_rel < 1e-12 && hessian_rel < 1e-12;
}

int main() {
    int passed = 0, failed = 0;
    for (int n = 1; n <= 64; ++n) {
        std::vector<double> x((size_t)n), y((size_t)n), v((size_t)n);
        for (int i = 0; i < n; ++i) {
            x[(size_t)i] = (2.0 * random_unit() - 1.0) * 0.05 / n;
            y[(size_t)i] = (2.0 * random_unit() - 1.0) * 0.04 / n;
            v[(size_t)i] = (2.0 * random_unit() - 1.0) * 0.02 / n;
        }
        if (check_case(x, y, v, -0.7)) ++passed; else ++failed;
    }
    for (int n = 1; n <= 64; ++n) {
        std::vector<double> x((size_t)n, 0.0), y((size_t)n), v((size_t)n);
        for (int i = 0; i < n; ++i) {
            y[(size_t)i] = (2.0 * random_unit() - 1.0) * 5e159 / n;
            v[(size_t)i] = (2.0 * random_unit() - 1.0) * 1e160 / n;
        }
        v[0] = 1e160 / n;
        if (check_case(x, y, v, -1e-320)) ++passed; else ++failed;
    }
    const bool coincidence = check_log_coincidence();
    if (coincidence) ++passed; else ++failed;
    const bool attention = check_tiny_attention();
    if (attention) ++passed; else ++failed;
    const bool score_overflow = check_attention_score_overflow();
    if (score_overflow) ++passed; else ++failed;
    const bool subnormal = check_case(
        {0.25, -0.125, 0.0625}, {-0.125, 0.0625, -0.03125},
        {0.02, -0.03, 0.01}, -1e-320);
    if (subnormal) ++passed; else ++failed;
    const bool subnormal_fd = check_subnormal_finite_difference();
    if (subnormal_fd) ++passed; else ++failed;
    const bool true_min_sphere = check_true_min_spherical_attention();
    if (true_min_sphere) ++passed; else ++failed;
    const bool ball_parameter = check_ball_parameter_extremes();
    if (ball_parameter) ++passed; else ++failed;
    const bool scaled_boundary = check_scaled_boundary_denominator();
    if (scaled_boundary) ++passed; else ++failed;
    const bool boundary_dimensions = check_boundary_dimension_sweep();
    if (boundary_dimensions) ++passed; else ++failed;
    const bool curvature_extreme = check_curvature_hessian_extreme();
    if (curvature_extreme) ++passed; else ++failed;
    std::printf("Results: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
