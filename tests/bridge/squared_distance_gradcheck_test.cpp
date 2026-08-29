/**
 * @file squared_distance_gradcheck_test.cpp
 * @brief Gradient check for AD_NODE_SQUARED_DISTANCE — d^2 on the space forms
 *        and on products of them.
 *
 * WHAT IS BEING DEFENDED
 *
 * `grad_x d^2(x,y) = -2 log_x(y)`, everywhere inside the injectivity ball,
 * INCLUDING at `x == y` where it is exactly zero. That last clause is the
 * whole reason this node exists separately from `ad_hyperbolic_distance`,
 * whose backward correctly refuses at coincidence because `d` has a cone point
 * there. `d^2` does not, and a node that inherited the refusal would be wrong
 * — self-attention scores every query against itself, so the diagonal is the
 * ordinary case rather than an edge one.
 *
 * ORDER OF EVIDENCE. Finite differences are the LAST check in this file, not
 * the first. Four exact references are asserted before any difference quotient
 * is taken:
 *
 *   1. THE FLAT CASE, BIT FOR BIT. On `R^n`, `d^2 = |x-y|^2` and
 *      `grad_x = 2(x-y)`. Both are exact in floating point, so the comparison
 *      is `==`, not a tolerance.
 *
 *   2. COINCIDENCE, BIT FOR BIT. `x == y` bitwise on every form and on the
 *      product: the value is exactly `0.0` and every gradient component is
 *      exactly `0.0`. No tolerance, no epsilon guard doing the work — the
 *      separation `delta = y - x` is exactly zero and carries the zero
 *      through.
 *
 *   3. THE LOG-MAP IDENTITY. `grad_x d^2` against `-2 log_x(y)` pushed through
 *      the metric, with `log_x` evaluated by THIS FILE's own transcription of
 *      the Mobius/Ganea form — a different expression from the one the
 *      implementation uses, so an agreement is two derivations meeting rather
 *      than one line read twice. Plus the Gauss lemma, `|grad d^2|_g = 2 d`,
 *      which brings the distance in as an independent third quantity.
 *
 *   4. THE GOLDEN VECTORS. tests/qllm_oracle/golden/squared_distance.json,
 *      produced by tests/qllm_oracle/squared_distance.esk, which computes
 *      `d^2` the OTHER way — `arcosh`/`arccos` closed form, squared, then
 *      differentiated by Eshkol's generic reverse-mode AD. Two routes, two
 *      languages, no shared code. Cases are cited by id below.
 *
 * The same golden file also records what the arcosh route does AT the
 * diagonal: value `0.0`, gradient non-finite, because `arcosh'(1)` is infinite
 * and `2 d d'` is `0 * inf`. This file asserts that the shipped node returns
 * exact zero at the same point. The pair is the design argument in executable
 * form: `d^2` must be evaluated in the log-map form, never as `sqrt(d^2)`
 * squared or differentiated.
 *
 * EVERYTHING RUNS THROUGH THE REAL PRODUCER in lib/bridge/space_form_ad.cpp
 * AND THE REAL DISPATCH. Nodes are
 * recorded by `ad_squared_distance` / `ad_product_squared_distance` onto a real
 * tape and swept by `eshkol_tensor_backward_dispatch`, so the test covers the
 * two registration sites (the arm in lib/backend/tensor_backward.cpp and the
 * row in lib/bridge/tensor_backward.cpp's table) as well as the rule. A
 * hand-built fixture would agree with the backward by construction and could
 * not see a producer that fills the contract wrongly.
 *
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/space_form.h"
#include "eshkol/backend/riemannian_core.h"

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
    void eshkol_tensor_backward_dispatch(void* ad_node_ptr);
}

namespace {

int g_passed = 0;
int g_failed = 0;

void report(const char* name, bool ok, const char* detail = nullptr) {
    std::printf("  %-56s %s", name, ok ? "PASS" : "FAIL");
    if (detail) std::printf("   [%s]", detail);
    std::printf("\n");
    if (ok) ++g_passed; else ++g_failed;
}

/* ===== Tape plumbing ===================================================== */

ad_node_t* var_node(const double* data, size_t n) {
    ad_node_t* v = arena_allocate_ad_node(get_global_arena());
    double* buf = (double*)arena_allocate_zeroed(get_global_arena(),
                                                 n * sizeof(double));
    int64_t* sh = (int64_t*)arena_allocate_zeroed(get_global_arena(),
                                                  sizeof(int64_t));
    std::memcpy(buf, data, n * sizeof(double));
    sh[0] = (int64_t)n;
    v->type = AD_NODE_VARIABLE;
    v->tensor_value = buf;
    v->shape = sh;
    v->ndim = 1;
    return v;
}

/** @brief Result of one producer + reverse sweep. */
struct Run {
    double d2 = 0.0;
    std::vector<double> gx, gy;
    bool ok = false;
};

/** @brief Record d^2 for a product spec, seed dL/d(d^2) = 1, sweep. */
Run run_product(const double* X, const double* Y, size_t n,
                const eshkol_manifold_factor_t* f, size_t k,
                double upstream = 1.0) {
    Run r;
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = var_node(X, n);
    ad_node_t* yn = var_node(Y, n);
    ad_node_t* out = ad_product_squared_distance(tape, xn, yn, f, k);
    if (!out) return r;
    r.d2 = ((const double*)out->tensor_value)[0];
    ((double*)out->tensor_gradient)[0] = upstream;
    for (size_t i = tape->num_nodes; i-- > 0;)
        eshkol_tensor_backward_dispatch(tape->nodes[i]);
    r.gx.assign((const double*)xn->tensor_gradient,
                (const double*)xn->tensor_gradient + n);
    r.gy.assign((const double*)yn->tensor_gradient,
                (const double*)yn->tensor_gradient + n);
    r.ok = true;
    return r;
}

Run run_single(const double* X, const double* Y, size_t n, int form, double c,
               double upstream = 1.0) {
    eshkol_manifold_factor_t f;
    f.form = form;
    f.reserved = 0;
    f.dim = (int64_t)n;
    f.curvature = c;
    f.weight = 1.0;
    return run_product(X, Y, n, &f, 1, upstream);
}

/** @brief Forward-only value, for finite differences. */
double value_only(const double* X, const double* Y, size_t n,
                  const eshkol_manifold_factor_t* f, size_t k) {
    ad_node_t* xn = var_node(X, n);
    ad_node_t* yn = var_node(Y, n);
    ad_node_t* out = ad_product_squared_distance(nullptr, xn, yn, f, k);
    if (!out) return NAN;
    return ((const double*)out->tensor_value)[0];
}

/* Independent high-precision oracle for the sensitive cases below.  The
 * production path is binary64; this reference evaluates the defining
 * formulas in the platform's extended long-double arithmetic and performs
 * the finite difference there, without calling value_only or any shared
 * geometry helper. */
long double hp_dot(const long double* a, const long double* b, size_t n) {
    long double s = 0.0L;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

long double hp_norm(const long double* a, size_t n) {
    long double s = 0.0L;
    for (size_t i = 0; i < n; ++i) s = std::hypotl(s, a[i]);
    return s;
}

void hp_project_sphere(long double* p, size_t n, long double K) {
    long double radius = 1.0L / std::sqrt(K);
    long double scale = radius / hp_norm(p, n);
    for (size_t i = 0; i < n; ++i) p[i] *= scale;
}

long double hp_single_value(const long double* X, const long double* Y,
                            size_t n, int form, long double K) {
    if (form == ESHKOL_SPACE_FORM_EUCLIDEAN) {
        long double e = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            long double d = X[i] - Y[i];
            e += d * d;
        }
        return e;
    }
    if (form == ESHKOL_SPACE_FORM_HYPERBOLIC) {
        long double c = -K;
        long double a = 1.0L - c * hp_dot(X, X, n);
        long double b = 1.0L - c * hp_dot(Y, Y, n);
        long double e = 0.0L;
        for (size_t i = 0; i < n; ++i) {
            long double d = X[i] - Y[i];
            e += d * d;
        }
        long double q = std::sqrt(c * e / (a * b));
        long double h = std::asinh(q);
        return 4.0L * h * h / c;
    }
    long double radius = 1.0L / std::sqrt(K);
    long double xn = hp_norm(X, n), yn = hp_norm(Y, n);
    long double cs = 0.0L;
    for (size_t i = 0; i < n; ++i)
        cs += (X[i] / xn) * (Y[i] / yn);
    if (cs > 1.0L) cs = 1.0L;
    if (cs < -1.0L) cs = -1.0L;
    long double theta = std::acos(cs);
    return radius * radius * theta * theta;
}

long double hp_fd_component(const double* X, const double* Y, size_t n,
                            int form, double K, size_t component,
                            bool differentiate_x, long double h) {
    std::vector<long double> xp(n), xm(n), yp(n), ym(n);
    for (size_t i = 0; i < n; ++i) {
        xp[i] = xm[i] = static_cast<long double>(X[i]);
        yp[i] = ym[i] = static_cast<long double>(Y[i]);
    }
    if (differentiate_x) {
        xp[component] += h;
        xm[component] -= h;
        if (form == ESHKOL_SPACE_FORM_SPHERICAL) {
            hp_project_sphere(xp.data(), n, static_cast<long double>(K));
            hp_project_sphere(xm.data(), n, static_cast<long double>(K));
        }
    } else {
        yp[component] += h;
        ym[component] -= h;
        if (form == ESHKOL_SPACE_FORM_SPHERICAL) {
            hp_project_sphere(yp.data(), n, static_cast<long double>(K));
            hp_project_sphere(ym.data(), n, static_cast<long double>(K));
        }
    }
    const long double vp = differentiate_x
        ? hp_single_value(xp.data(), yp.data(), n, form, K)
        : hp_single_value(xp.data(), yp.data(), n, form, K);
    const long double vm = differentiate_x
        ? hp_single_value(xm.data(), ym.data(), n, form, K)
        : hp_single_value(xm.data(), ym.data(), n, form, K);
    return (vp - vm) / (2.0L * h);
}

double hp_fd_max_rel(const double* X, const double* Y, size_t n, int form,
                     double K, const std::vector<double>& analytic,
                     bool differentiate_x, long double h) {
    long double worst = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        long double reference = hp_fd_component(X, Y, n, form, K, i,
                                                differentiate_x, h);
        long double error = std::fabs(reference - analytic[i]) /
                            (1.0L + std::fabs(reference));
        worst = std::max(worst, error);
    }
    return static_cast<double>(worst);
}

/* ===== This file's own geometry, written from the definitions ============ */

double dotv(const double* a, const double* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
double normv(const double* a, size_t n) { return std::sqrt(dotv(a, a, n)); }

/** @brief Mobius addition a (+)_c b on the ball of curvature -c, written in
 *  the standard expanded form rather than in the delta-collected one the
 *  implementation uses. */
void mobius_add_ref(const double* a, const double* b, double c, size_t n,
                    double* out) {
    double ab = dotv(a, b, n), aa = dotv(a, a, n), bb = dotv(b, b, n);
    double na = 1.0 + 2.0 * c * ab + c * bb;
    double nb = 1.0 - c * aa;
    double den = 1.0 + 2.0 * c * ab + c * c * aa * bb;
    for (size_t i = 0; i < n; ++i) out[i] = (na * a[i] + nb * b[i]) / den;
}

/** @brief log_x(y) on the ball, Ganea form:
 *  ((1-c|x|^2)/sqrt(c)) * artanh(sqrt(c)|u|) u/|u|, u = (-x) (+)_c y. */
void ball_log_ref(const double* x, const double* y, double c, size_t n,
                  double* out) {
    std::vector<double> nx(n), u(n);
    for (size_t i = 0; i < n; ++i) nx[i] = -x[i];
    mobius_add_ref(nx.data(), y, c, n, u.data());
    double un = normv(u.data(), n);
    if (un == 0.0) { for (size_t i = 0; i < n; ++i) out[i] = 0.0; return; }
    double sc = std::sqrt(c);
    double lam = 2.0 / (1.0 - c * dotv(x, x, n));
    double coef = (2.0 / (sc * lam)) * std::atanh(sc * un) / un;
    for (size_t i = 0; i < n; ++i) out[i] = coef * u[i];
}

/** @brief d on the ball, arcosh form — a third expression again. */
double ball_dist_ref(const double* x, const double* y, double c, size_t n) {
    double dd = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = x[i]-y[i]; dd += d*d; }
    double a = 1.0 - c * dotv(x, x, n);
    double b = 1.0 - c * dotv(y, y, n);
    return std::acosh(1.0 + 2.0 * c * dd / (a * b)) / std::sqrt(c);
}

/* Independent closed-form evaluation for the audit's rounded-t=1 case. This
 * uses the audit's binary64 asinh route, not the shared core. */
double ball_sq_audit_reference(const double* x, const double* y,
                               double c, size_t n) {
    double x2 = 0.0, y2 = 0.0, e = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double xi = x[i], yi = y[i], d = xi - yi;
        x2 += xi * xi;
        y2 += yi * yi;
        e += d * d;
    }
    double q = std::sqrt(c * e / ((1.0 - c * x2) * (1.0 - c * y2)));
    double h = std::asinh(q);
    return 4.0 * h * h / c;
}

/** @brief log_x(y) on the sphere of radius R = 1/sqrt(c), arccos form. */
void sphere_log_ref(const double* x, const double* y, double c, size_t n,
                    double* out) {
    double R = 1.0 / std::sqrt(c);
    double sx = R / normv(x, n), sy = R / normv(y, n);
    std::vector<double> xp(n), yp(n), u(n);
    for (size_t i = 0; i < n; ++i) { xp[i] = x[i]*sx; yp[i] = y[i]*sy; }
    double ca = dotv(xp.data(), yp.data(), n) / (R * R);
    if (ca >  1.0) ca =  1.0;
    if (ca < -1.0) ca = -1.0;
    for (size_t i = 0; i < n; ++i) u[i] = yp[i] - ca * xp[i];
    double un = normv(u.data(), n);
    double th = std::acos(ca);
    if (un == 0.0) { for (size_t i = 0; i < n; ++i) out[i] = 0.0; return; }
    for (size_t i = 0; i < n; ++i) out[i] = (R * th) * u[i] / un;
}

double maxabs_diff(const std::vector<double>& a, const double* b, size_t n) {
    double w = 0.0;
    for (size_t i = 0; i < n; ++i) w = std::max(w, std::fabs(a[i] - b[i]));
    return w;
}
double max_rel(const std::vector<double>& a, const double* b, size_t n) {
    double w = 0.0;
    for (size_t i = 0; i < n; ++i)
        w = std::max(w, std::fabs(a[i] - b[i]) / (1.0 + std::fabs(b[i])));
    return w;
}
bool all_exactly(const std::vector<double>& a, double v) {
    for (double e : a) if (e != v) return false;
    return true;
}

/* ===== Fixtures, matching tests/qllm_oracle/squared_distance.esk ========= */

const double kEucX[3] = {  0.5, -1.25,  2.0   };
const double kEucY[3] = { -0.75, 0.5,   0.125 };

const double kBallX[3] = {  0.3, -0.2, 0.1 };
const double kBallY[3] = { -0.15, 0.25, 0.4 };

/* Already on the unit sphere, as the .esk seeds are after vwith-norm. */
const double kSphX[3] = { 0.6, -0.8, 0.0 };
const double kSphY[3] = { 0.20628424925175867, 0.309426373877638,
                          0.9282791216329139 };

/* H^2(c=1,w=1) x S^2(ambient 3, c=1, w=2) x R^2(w=0.5) — 2+3+2 = 7. */
const eshkol_manifold_factor_t kProdF[3] = {
    { ESHKOL_SPACE_FORM_HYPERBOLIC, 0, 2, -1.0, 1.0 },
    { ESHKOL_SPACE_FORM_SPHERICAL,  0, 3, 1.0, 2.0 },
    { ESHKOL_SPACE_FORM_EUCLIDEAN,  0, 2, 0.0, 0.5 },
};
const double kProdX[7] = {  0.25, -0.1,  0.6, -0.8, 0.0,  1.5,  0.25 };
const double kProdY[7] = { -0.3,   0.2,  0.20628424925175867,
                           0.309426373877638, 0.9282791216329139,
                           -0.5,  1.75 };

/* ===== Golden vectors ====================================================
 * tests/qllm_oracle/golden/squared_distance.json, 17 significant digits, so
 * these round-trip exactly. Each block names the case id it was taken from. */

/* case squared_distance.euclidean.d3 */
const double kGoldEucD2 = 8.140625;
const double kGoldEucG[3] = { 2.5, -3.5, 3.75 };

/* case squared_distance.ball.d3.c1 */
const double kGoldBallD2 = 2.488583550322671;
const double kGoldBallG[3] = { 5.220650016932333, -4.738068922930185,
                              -2.0326900626151105 };

/* case squared_distance.sphere.d3.c1 */
const double kGoldSphD2 = 2.872635450625902;
const double kGoldSphG[3] = { -0.9583573646009185, -0.7187680234506886,
                              -3.171035397576568 };

/* case squared_distance.product.h2s2r2 */
const double kGoldProdD2 = 10.563201437978517;
const double kGoldProdG[7] = { 4.974407747741088, -2.596587685832792,
                              -1.916714729201836, -1.4375360469013774,
                              -6.342070795153134,  2.0, -1.5 };

/* ===== Finite differences (last line of defence) ========================= */

double fd_max_rel(const double* X, const double* Y, size_t n,
                  const eshkol_manifold_factor_t* f, size_t k,
                  const std::vector<double>& analytic, double h) {
    std::vector<double> xp(X, X + n);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double save = xp[i];
        xp[i] = save + h; double vp = value_only(xp.data(), Y, n, f, k);
        xp[i] = save - h; double vm = value_only(xp.data(), Y, n, f, k);
        xp[i] = save;
        double num = (vp - vm) / (2.0 * h);
        worst = std::max(worst,
                         std::fabs(num - analytic[i]) / (1.0 + std::fabs(num)));
    }
    return worst;
}

void project_sphere_factors(std::vector<double>& p,
                            const eshkol_manifold_factor_t* f, size_t k) {
    size_t off = 0;
    for (size_t i = 0; i < k; ++i) {
        size_t dim = (size_t)f[i].dim;
        if (f[i].form == ESHKOL_SPACE_FORM_SPHERICAL) {
            double norm = normv(p.data() + off, dim);
            double radius = 1.0 / std::sqrt(f[i].curvature);
            double scale = radius / norm;
            for (size_t j = 0; j < dim; ++j) p[off + j] *= scale;
        }
        off += dim;
    }
}

double fd_max_rel_y(const double* X, const double* Y, size_t n,
                    const eshkol_manifold_factor_t* f, size_t k,
                    const std::vector<double>& analytic, double h) {
    std::vector<double> yp(Y, Y + n);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double save = Y[i];
        yp.assign(Y, Y + n);
        yp[i] = save + h; project_sphere_factors(yp, f, k);
        double vp = value_only(X, yp.data(), n, f, k);
        yp.assign(Y, Y + n); yp[i] = save - h;
        project_sphere_factors(yp, f, k);
        double vm = value_only(X, yp.data(), n, f, k);
        yp[i] = save;
        double num = (vp - vm) / (2.0 * h);
        worst = std::max(worst,
                         std::fabs(num - analytic[i]) / (1.0 + std::fabs(num)));
    }
    return worst;
}

double fd_max_rel_projected_x(const double* X, const double* Y, size_t n,
                              const eshkol_manifold_factor_t* f, size_t k,
                              const std::vector<double>& analytic, double h) {
    std::vector<double> xp(X, X + n);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double save = X[i];
        xp.assign(X, X + n);
        xp[i] = save + h; project_sphere_factors(xp, f, k);
        double vp = value_only(xp.data(), Y, n, f, k);
        xp.assign(X, X + n); xp[i] = save - h;
        project_sphere_factors(xp, f, k);
        double vm = value_only(xp.data(), Y, n, f, k);
        xp[i] = save;
        double num = (vp - vm) / (2.0 * h);
        worst = std::max(worst,
                         std::fabs(num - analytic[i]) / (1.0 + std::fabs(num)));
    }
    return worst;
}

} /* namespace */

int main() {
    std::printf("=== squared-distance backward gradcheck ===\n");

    /* ---- 1. flat case, bit for bit -------------------------------------- */
    {
        Run r = run_single(kEucX, kEucY, 3, ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        double want_d2 = 0.0;
        std::vector<double> wx(3), wy(3);
        for (size_t i = 0; i < 3; ++i) {
            double d = kEucX[i] - kEucY[i];
            want_d2 += d * d;
            wx[i] =  2.0 * d;
            wy[i] = -2.0 * d;
        }
        bool ok = r.ok && r.d2 == want_d2;
        for (size_t i = 0; i < 3 && ok; ++i)
            ok = (r.gx[i] == wx[i]) && (r.gy[i] == wy[i]);
        report("flat: d2 == |x-y|^2 and grad == 2(x-y), EXACTLY", ok);
    }

    /* ---- audit P1: signed curvature is part of the manifold contract ----- */
    {
        const double zero[1] = { 0.0 };
        const double half[1] = { 0.5 };
        Run flat = run_single(zero, half, 1, ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        Run wrong_form = run_single(zero, half, 1,
                                    ESHKOL_SPACE_FORM_HYPERBOLIC, 0.0);
        bool ok = flat.ok && flat.d2 == 0.25 && flat.gx[0] == -1.0 &&
                  flat.gy[0] == 1.0 && !wrong_form.ok;
        report("audit: K=0 is Euclidean and hyperbolic K=0 is refused", ok);
    }
    {
        Run bad = run_single(kBallX, kBallY, 3,
                             ESHKOL_SPACE_FORM_HYPERBOLIC,
                             std::numeric_limits<double>::quiet_NaN());
        report("audit: non-finite curvature is refused", !bad.ok);
    }

    /* ---- audit P1: stable near-boundary Poincare distance ---------------- */
    {
        const double x[1] = { 0.999999999 };
        const double y[1] = { -0.999999999 };
        Run r = run_single(x, y, 1, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        const double audit_d2 = 1834.6509908199114;
        double reference = ball_sq_audit_reference(x, y, 1.0, 1);
        double rel = std::fabs(r.d2 - reference) / (1.0 + reference);
        double audit_rel = std::fabs(r.d2 - audit_d2) / (1.0 + audit_d2);
        bool finite = r.ok && std::isfinite(r.d2) && std::isfinite(r.gx[0]) &&
                      std::isfinite(r.gy[0]);
        char detail[128];
        std::snprintf(detail, sizeof detail, "d2 %.17g ref %.17g rel %.3e",
                      r.d2, reference, rel);
        report("audit: valid near-boundary H pair is finite and exact",
               finite && rel < 1e-9 && audit_rel < 1e-6, detail);
    }
    {
        const double x[3] = { 0.6, -0.4, 0.2 };
        const double y[3] = { -0.3, 0.5, 0.8 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -0.25);
        double want = ball_sq_audit_reference(x, y, 0.25, 3);
        double rel = std::fabs(r.d2 - want) / (1.0 + want);
        report("audit: non-unit negative curvature keeps its scale", r.ok && rel < 1e-13);
    }
    {
        const double x[3] = { 0.5 * kSphX[0], 0.5 * kSphX[1], 0.5 * kSphX[2] };
        const double y[3] = { 0.5 * kSphY[0], 0.5 * kSphY[1], 0.5 * kSphY[2] };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_SPHERICAL, 4.0);
        double theta = std::acos(dotv(x, y, 3) / 0.25);
        double want = 0.25 * theta * theta;
        double rel = std::fabs(r.d2 - want) / (1.0 + want);
        const double off_radius_x[2] = { 1.0 + 5e-10, 0.0 };
        const double off_radius_y[2] = { 0.0, 1.0 };
        Run canonical = run_single(off_radius_x, off_radius_y, 2,
                                   ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        const double pi = std::acos(-1.0);
        double canonical_gx = -pi / (1.0 + 5e-10);
        bool chain_ok = canonical.ok &&
                        std::fabs(canonical.d2 - 0.25 * pi * pi) < 1e-14 &&
                        std::fabs(canonical.gx[1] - canonical_gx) < 1e-14 &&
                        std::fabs(canonical.gy[0] + pi) < 1e-14;
        report("audit: non-unit positive curvature and canonicalization chain rule",
               r.ok && rel < 1e-13 && chain_ok);
    }

    /* The old shared 1e-15 direction cutoff erased this valid derivative.  The
     * finite-difference reference is evaluated independently in long double,
     * with an upstream chosen to keep the expected signal order one. */
    {
        const double x[1] = { 0.0 };
        const double y[1] = { 1e-16 };
        Run r = run_single(x, y, 1, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0, 1e16);
        std::vector<double> want_x(1), want_y(1);
        want_x[0] = static_cast<double>(hp_fd_component(
            x, y, 1, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0, 0, true, 1e-18L)) * 1e16;
        want_y[0] = static_cast<double>(hp_fd_component(
            x, y, 1, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0, 0, false, 1e-18L)) * 1e16;
        double ex = std::fabs(r.gx[0] - want_x[0]);
        double ey = std::fabs(r.gy[0] - want_y[0]);
        char detail[128];
        std::snprintf(detail, sizeof detail, "gx %.17g ref %.17g, gy %.17g ref %.17g",
                      r.gx[0], want_x[0], r.gy[0], want_y[0]);
        const double huge_x[1] = { 0.0 };
        const double huge_y[1] = { 5e-155 };
        Run huge = run_single(huge_x, huge_y, 1,
                              ESHKOL_SPACE_FORM_HYPERBOLIC, -1e308);
        /* This is the stable Poincare value 4*asinh(1/sqrt(3))^2 / 1e308;
         * in particular, no c*4 intermediate is allowed to overflow. */
        const double huge_expected = 1.206948960812582e-308;
        report("audit: tiny nonzero H separation keeps its finite gradient",
               r.ok && ex < 1e-12 && ey < 1e-12 && huge.ok &&
               std::isfinite(huge.d2) &&
               std::fabs(huge.d2 - huge_expected) / huge_expected < 1e-12,
               detail);
    }
    {
        const double x[3] = { 0.13, -0.21, 0.17 };
        const double y[3] = { -0.19, 0.07, 0.22 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -0.25);
        std::vector<double> gx = r.gx, gy = r.gy;
        double wx = hp_fd_max_rel(x, y, 3, ESHKOL_SPACE_FORM_HYPERBOLIC,
                                  -0.25, gx, true, 1e-7L);
        double wy = hp_fd_max_rel(x, y, 3, ESHKOL_SPACE_FORM_HYPERBOLIC,
                                  -0.25, gy, false, 1e-7L);
        char detail[96];
        std::snprintf(detail, sizeof detail, "x %.3e, y %.3e", wx, wy);
        report("audit: non-unit H curvature backward matches high-precision FD",
               r.ok && wx < 1e-10 && wy < 1e-10, detail);
    }
    {
        const double R = 5e-16;
        const double x[3] = { R, 0.0, 0.0 };
        const double y[3] = { 0.0, R, 0.0 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_SPHERICAL, 4e30);
        const double pi = std::acos(-1.0);
        const double expected_d2 = 0.25 * pi * pi * R * R;
        const double expected = -pi * R;
        char detail[128];
        std::snprintf(detail, sizeof detail, "d2 %.17g, gy %.17g expected %.17g",
                      r.d2, r.gx[1], expected);
        const double tiny_K = 9.99988867182683e-321;
        const double tiny_R = 1.0 / std::sqrt(tiny_K);
        const double tiny_x[2] = { tiny_R, 0.0 };
        const double tiny_y[2] = { tiny_R * std::cos(1e-10),
                                   tiny_R * std::sin(1e-10) };
        Run tiny = run_single(tiny_x, tiny_y, 2,
                              ESHKOL_SPACE_FORM_SPHERICAL, tiny_K);
        const double tiny_expected_d2 = 1.000011132941258e300;
        bool tiny_ok = tiny.ok && std::isfinite(tiny.d2) &&
                       std::fabs(tiny.d2 - tiny_expected_d2) /
                           tiny_expected_d2 < 1e-12;
        report("audit: spherical scaling stays finite before squaring",
               r.ok && std::fabs(r.d2 - expected_d2) < 1e-45 &&
               std::fabs(r.gx[0]) < 1e-30 &&
               std::fabs(r.gx[1] - expected) < 1e-30 && tiny_ok,
               detail);
    }

    /* ---- 2. coincidence is a value, not a refusal, and it is exactly 0 --- */
    {
        Run rh = run_single(kBallX, kBallX, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        bool ok = rh.ok && rh.d2 == 0.0
                  && all_exactly(rh.gx, 0.0) && all_exactly(rh.gy, 0.0);
        report("hyperbolic: x == y gives d2 == 0 and grad == 0 exactly", ok);
    }
    {
        Run rs = run_single(kSphX, kSphX, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        bool ok = rs.ok && rs.d2 == 0.0
                  && all_exactly(rs.gx, 0.0) && all_exactly(rs.gy, 0.0);
        report("spherical: x == y gives d2 == 0 and grad == 0 exactly", ok);
    }
    {
        Run re = run_single(kEucX, kEucX, 3, ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        bool ok = re.ok && re.d2 == 0.0
                  && all_exactly(re.gx, 0.0) && all_exactly(re.gy, 0.0);
        report("euclidean: x == y gives d2 == 0 and grad == 0 exactly", ok);
    }
    {
        Run rp = run_product(kProdX, kProdX, 7, kProdF, 3);
        bool ok = rp.ok && rp.d2 == 0.0
                  && all_exactly(rp.gx, 0.0) && all_exactly(rp.gy, 0.0);
        report("product: x == y gives d2 == 0 and grad == 0 exactly", ok);
    }

    /* ---- 3. the log-map identity, against an independent log ------------- */
    {
        Run r = run_single(kBallX, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        double lg[3];
        ball_log_ref(kBallX, kBallY, 1.0, 3, lg);
        double lam = 2.0 / (1.0 - dotv(kBallX, kBallX, 3));
        double want[3];
        /* Coordinate gradient = g * (Riemannian gradient) = lambda^2 * (-2 log). */
        for (int i = 0; i < 3; ++i) want[i] = -2.0 * lam * lam * lg[i];
        double worst = max_rel(r.gx, want, 3);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", worst);
        report("ball: grad_x == -2 lambda_x^2 log_x(y)", worst < 1e-14, detail);
    }
    {
        Run r = run_single(kBallX, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        /* Gauss lemma, with the distance as an independent third quantity.
         * The RIEMANNIAN norm of the gradient is |grad|_euclid / lambda_x,
         * because the Riemannian gradient is g^{-1} = lambda^{-2} times the
         * coordinate one. Getting lambda onto the other side is off by
         * lambda^2 and this check is exactly where that shows. */
        double lam = 2.0 / (1.0 - dotv(kBallX, kBallX, 3));
        double gn = normv(r.gx.data(), 3);
        double d = ball_dist_ref(kBallX, kBallY, 1.0, 3);
        double rel = std::fabs(gn / lam - 2.0 * d) / (1.0 + 2.0 * d);
        char detail[96];
        std::snprintf(detail, sizeof detail, "rel = %.3e", rel);
        report("ball: Gauss lemma |grad|/lambda_x == 2 d", rel < 1e-14, detail);
    }
    {
        Run r = run_single(kSphX, kSphY, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        double lg[3];
        sphere_log_ref(kSphX, kSphY, 1.0, 3, lg);
        double want[3];
        for (int i = 0; i < 3; ++i) want[i] = -2.0 * lg[i];
        double worst = max_rel(r.gx, want, 3);
        /* The ambient metric on the sphere is the identity, so no radial part
         * may appear: <grad, x> is zero for a map that is 0-homogeneous in x. */
        double radial = std::fabs(dotv(r.gx.data(), kSphX, 3));
        char detail[112];
        std::snprintf(detail, sizeof detail, "max rel = %.3e, radial = %.3e",
                      worst, radial);
        report("sphere: grad_x == -2 log_x(y), no radial component",
               worst < 1e-13 && radial < 1e-14, detail);
    }
    {
        /* Symmetry: d^2(x,y) == d^2(y,x) and the two gradients swap. Not a
         * restatement — the two arguments take different code paths (alpha vs
         * beta, x vs y in the numerator), so a sign or an index error in one
         * of them shows here and nowhere else. */
        Run a = run_single(kBallX, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        Run b = run_single(kBallY, kBallX, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        double dv = std::fabs(a.d2 - b.d2);
        double gv = std::max(maxabs_diff(a.gx, b.gy.data(), 3),
                             maxabs_diff(a.gy, b.gx.data(), 3));
        char detail[112];
        std::snprintf(detail, sizeof detail, "d2 %.3e, grad %.3e", dv, gv);
        report("ball: d2 and its gradients are symmetric in x, y",
               dv == 0.0 && gv == 0.0, detail);
    }

    /* ---- 3b. the exported helpers agree with the independent references -- */
    {
        /* eshkol_space_form_log_map and eshkol_space_form_distance are public
         * API. They are checked here against this file's own transcriptions
         * rather than being trusted, and they are deliberately NOT what the
         * gradient checks above compare against: they share a file, and nearly
         * the algebra, with the backward they would be grading. */
        double got[3], want[3];
        bool ok = eshkol_space_form_log_map(ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0,
                                            kBallX, kBallY, 3, got);
        ball_log_ref(kBallX, kBallY, 1.0, 3, want);
        double w1 = max_rel(std::vector<double>(got, got + 3), want, 3);

        ok = ok && eshkol_space_form_log_map(ESHKOL_SPACE_FORM_SPHERICAL, 1.0,
                                             kSphX, kSphY, 3, got);
        sphere_log_ref(kSphX, kSphY, 1.0, 3, want);
        double w2 = max_rel(std::vector<double>(got, got + 3), want, 3);

        char detail[112];
        std::snprintf(detail, sizeof detail, "ball %.3e, sphere %.3e", w1, w2);
        report("exported log map matches the independent transcriptions",
               ok && w1 < 1e-14 && w2 < 1e-13, detail);
    }
    {
        double db = eshkol_space_form_distance(ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0,
                                               kBallX, kBallY, 3);
        double wb = ball_dist_ref(kBallX, kBallY, 1.0, 3);
        double ds = eshkol_space_form_distance(ESHKOL_SPACE_FORM_SPHERICAL, 1.0,
                                               kSphX, kSphY, 3);
        double ws = std::acos(dotv(kSphX, kSphY, 3));
        double de = eshkol_space_form_distance(ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0,
                                               kEucX, kEucY, 3);
        double we = 0.0;
        for (int i = 0; i < 3; ++i) {
            double d = kEucX[i] - kEucY[i];
            we += d * d;
        }
        we = std::sqrt(we);
        double worst = std::max(std::max(std::fabs(db - wb) / wb,
                                         std::fabs(ds - ws) / ws),
                                std::fabs(de - we) / we);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", worst);
        report("exported distance matches arcosh / arccos / |x-y|",
               worst < 1e-14, detail);
    }
    {
        /* And the two agree with each other where the geometry says they must:
         * |log_x(y)|_g == d. On the ball that is lambda_x |log|_euclid. */
        double lg[3];
        eshkol_space_form_log_map(ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0,
                                  kBallX, kBallY, 3, lg);
        double lam = 2.0 / (1.0 - dotv(kBallX, kBallX, 3));
        double d = eshkol_space_form_distance(ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0,
                                              kBallX, kBallY, 3);
        double rel = std::fabs(lam * normv(lg, 3) - d) / (1.0 + d);
        char detail[96];
        std::snprintf(detail, sizeof detail, "rel = %.3e", rel);
        report("ball: |log_x(y)|_g == d", rel < 1e-15, detail);
    }

    /* ---- 4. product additivity, exactly --------------------------------- */
    {
        Run p = run_product(kProdX, kProdY, 7, kProdF, 3);
        Run h = run_single(kProdX + 0, kProdY + 0, 2,
                           ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        Run s = run_single(kProdX + 2, kProdY + 2, 3,
                           ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        Run e = run_single(kProdX + 5, kProdY + 5, 2,
                           ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        double sum = 1.0 * h.d2 + 2.0 * s.d2 + 0.5 * e.d2;
        bool ok = p.ok && std::fabs(p.d2 - sum) <= 4.0 * 2.22e-16 * std::fabs(sum);
        /* The gradient is the concatenation of w_f times each factor's own —
         * the product metric is block diagonal, so no cross term exists. */
        for (size_t i = 0; i < 2 && ok; ++i) ok = (p.gx[i]     == 1.0 * h.gx[i]);
        for (size_t i = 0; i < 3 && ok; ++i) ok = (p.gx[2 + i] == 2.0 * s.gx[i]);
        for (size_t i = 0; i < 2 && ok; ++i) ok = (p.gx[5 + i] == 0.5 * e.gx[i]);
        char detail[112];
        std::snprintf(detail, sizeof detail, "d2 %.17g vs %.17g", p.d2, sum);
        report("product: d2 and grad are the weighted factor sum", ok, detail);
    }

    /* ---- 5. golden vectors ---------------------------------------------- */
    {
        Run r = run_single(kEucX, kEucY, 3, ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        double w = max_rel(r.gx, kGoldEucG, 3);
        bool ok = r.ok && r.d2 == kGoldEucD2 && w == 0.0;
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", w);
        report("golden squared_distance.euclidean.d3", ok, detail);
    }
    {
        Run r = run_single(kBallX, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        double w = max_rel(r.gx, kGoldBallG, 3);
        double dv = std::fabs(r.d2 - kGoldBallD2) / kGoldBallD2;
        char detail[112];
        std::snprintf(detail, sizeof detail, "d2 rel %.3e, grad rel %.3e", dv, w);
        report("golden squared_distance.ball.d3.c1",
               r.ok && dv < 1e-15 && w < 1e-14, detail);
    }
    {
        Run r = run_single(kSphX, kSphY, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        double w = max_rel(r.gx, kGoldSphG, 3);
        double dv = std::fabs(r.d2 - kGoldSphD2) / kGoldSphD2;
        char detail[112];
        std::snprintf(detail, sizeof detail, "d2 rel %.3e, grad rel %.3e", dv, w);
        report("golden squared_distance.sphere.d3.c1",
               r.ok && dv < 1e-14 && w < 1e-13, detail);
    }
    {
        Run r = run_product(kProdX, kProdY, 7, kProdF, 3);
        double w = max_rel(r.gx, kGoldProdG, 7);
        double dv = std::fabs(r.d2 - kGoldProdD2) / kGoldProdD2;
        char detail[112];
        std::snprintf(detail, sizeof detail, "d2 rel %.3e, grad rel %.3e", dv, w);
        report("golden squared_distance.product.h2s2r2",
               r.ok && dv < 1e-14 && w < 1e-13, detail);
    }
    {
        /* The other half of case squared_distance.ball.coincident.d3.c1. The
         * .esk exporter differentiates the arcosh route at x == y and gets a
         * non-finite gradient with a finite value — recorded there as nulls
         * with all_finite: false. The shipped node reaches the same point
         * through the log map. This is the contrast the design rests on. */
        Run r = run_single(kBallX, kBallX, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        bool finite = r.ok;
        for (size_t i = 0; i < r.gx.size() && finite; ++i)
            finite = std::isfinite(r.gx[i]);
        report("golden ...ball.coincident: log-map route is finite and zero",
               finite && r.d2 == 0.0 && all_exactly(r.gx, 0.0));
    }

    /* ---- 6. smoothness through the diagonal ----------------------------- */
    {
        /* Walk in along a fixed direction and check the gradient magnitude
         * against the closed form 2*lambda_x*d, which is what the Gauss lemma
         * gives in coordinates. `d` here is the ball's own distance at the
         * constructed separation, so the check is the shape of the decay, not
         * a re-derivation of the point.
         *
         * The bound loosens with r on purpose: T10 of the study measures the
         * ball chart's own distance resolution as ~eps/r, so at r = 1e-12 no
         * implementation can do better than ~1e-4 relative. What must hold at
         * every radius is that the gradient stays FINITE, keeps shrinking, and
         * tracks 2*lambda*r to that floor — rather than converging to lambda,
         * which is what the gradient of `d` itself does. */
        const double dir[3] = { 0.8017837257372732,   /* unit direction */
                                0.5345224838248488,
                               -0.2672612419124244 };
        double lam = 2.0 / (1.0 - dotv(kBallX, kBallX, 3));
        bool ok = true;
        double prev = 1e300;
        std::printf("      r          |grad|            2*lambda*r        rel\n");
        for (int e = 2; e <= 12; ++e) {
            double rr = std::pow(10.0, -e);
            /* y = exp_x(rr * unit-Riemannian direction). In the conformal ball
             * metric a Riemannian unit vector is dir/lambda in coordinates. */
            double tang[3], scaled[3], y[3];
            for (int i = 0; i < 3; ++i) tang[i] = dir[i] / lam;
            double vn = normv(tang, 3);
            double coef = std::tanh(lam * vn * rr / 2.0) / vn;
            for (int i = 0; i < 3; ++i) scaled[i] = coef * tang[i];
            mobius_add_ref(kBallX, scaled, 1.0, 3, y);

            Run r = run_single(kBallX, y, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
            double gn = normv(r.gx.data(), 3);
            double want = 2.0 * lam * rr;
            double rel = std::fabs(gn - want) / want;
            double bar = (e <= 10) ? 1e-6 : 1e-3;
            bool step_ok = r.ok && std::isfinite(gn) && gn < prev && rel < bar;
            prev = gn;
            ok = ok && step_ok;
            std::printf("      1e-%-3d  %.12e  %.12e  %.2e%s\n",
                        e, gn, want, rel, step_ok ? "" : "   <-- FAIL");
        }
        report("diagonal: |grad| tracks 2*lambda*r from 1e-2 down to 1e-12", ok);
    }

    /* ---- 7. refusals ---------------------------------------------------- */
    {
        /* Outside the ball. Not a coincidence question: the point is off the
         * manifold and no distance exists, so the producer returns NULL. */
        const double out_of_ball[3] = { 1.5, 0.0, 0.0 };
        Run r = run_single(out_of_ball, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        report("refuses a point outside the Poincare ball", !r.ok);
    }
    {
        /* The antipode is the sphere's cut locus. Every direction there is a
         * minimising geodesic, log_x has no value, and d^2 has no derivative —
         * a refusal that is correct for the SAME reason the coincidence
         * refusal is wrong. */
        const double a[3] = {  0.6, -0.8, 0.0 };
        const double b[3] = { -0.6,  0.8, 0.0 };
        Run r = run_single(a, b, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        report("refuses an antipodal pair on the sphere", !r.ok);
    }
    {
        /* Just inside the cut locus still answers, so the refusal above is a
         * boundary and not a blanket. */
        const double a[3] = { 1.0, 0.0, 0.0 };
        const double b[3] = { -std::cos(4.5e-3), std::sin(4.5e-3), 0.0 };
        Run r = run_single(a, b, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        bool finite = r.ok;
        for (size_t i = 0; i < r.gx.size() && finite; ++i)
            finite = std::isfinite(r.gx[i]);
        report("still answers just inside the sphere's cut locus", finite);
    }
    {
        const double x[3] = { 2.0, 0.0, 0.0 };
        const double y[3] = { 0.0, 3.0, 0.0 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        report("audit: off-sphere inputs are refused, not projected", !r.ok);
    }
    {
        /* This accepted point is not unit length in binary64.  Coincidence is
         * nevertheless exact because the core checks equality before sphere
         * normalization/cancellation. */
        Run r = run_single(kSphY, kSphY, 3,
                           ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        report("audit: non-exact-radius spherical coincidence is exactly zero",
               r.ok && r.d2 == 0.0 && all_exactly(r.gx, 0.0) &&
               all_exactly(r.gy, 0.0));
    }
    {
        const double y[3] = { -kSphY[0], -kSphY[1], -kSphY[2] };
        const double exact_x[3] = { 0.9999999925494194,
                                    0.0001220703115905053, 0.0 };
        const double exact_y[3] = { -0.999999993480742,
                                    -0.00012207031170419214, -0.0 };
        Run r = run_single(kSphY, y, 3,
                           ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        Run exact = run_single(exact_x, exact_y, 3,
                               ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        report("audit: exact spherical antipodes are refused consistently",
               !r.ok && !exact.ok);
    }
    {
        /* The singularity predicate is also exercised directly because the
         * vectors in this regression are not required to have sphere radius
         * one. In particular, x=(.75,2m), y=(-.75,-3m) has a nonzero exact
         * cross comparison even though the residuals of the floating product
         * comparisons underflow. Sweep every subnormal coordinate
         * position for n=1..64 and all pairs km,(k+1)m through 1000m. */
        const double m = std::numeric_limits<double>::denorm_min();
        std::vector<double> x, y;
        bool ok = true;
        size_t checked = 0;
        for (int n = 1; n <= 64 && ok; ++n) {
            x.assign((size_t)n, 0.0);
            y.assign((size_t)n, 0.0);
            for (int pos = 0; pos < n && ok; ++pos) {
                const int anchor = (n == 1) ? -1 : (pos == 0 ? 1 : 0);
                if (anchor >= 0) {
                    x[(size_t)anchor] = 0.75;
                    y[(size_t)anchor] = -0.75;
                }
                for (int k = 1; k <= 1000 && ok; ++k) {
                    x[(size_t)pos] = (double)k * m;
                    y[(size_t)pos] = -(double)(k + 1) * m;
                    bool got = eshkol_rm_sphere_antipodal(x.data(), y.data(), n);
                    bool want = (n == 1);
                    ok = ok && (got == want);
                    ++checked;

                    y[(size_t)pos] = -(double)k * m;
                    got = eshkol_rm_sphere_antipodal(x.data(), y.data(), n);
                    ok = ok && got;
                    ++checked;
                }
                x[(size_t)pos] = 0.0;
                y[(size_t)pos] = 0.0;
                if (anchor >= 0) {
                    x[(size_t)anchor] = 0.0;
                    y[(size_t)anchor] = 0.0;
                }
            }
        }
        char detail[96];
        std::snprintf(detail, sizeof detail, "%zu exact-rational cases", checked);
        report("audit: subnormal antipode sweep is exact without products", ok,
               detail);
    }
    {
        const double x[3] = { 1.0, 0.0, 0.0 };
        const double eps = 1e-6;
        const double y[3] = { -std::cos(eps), std::sin(eps), 0.0 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        const double theta = std::acos(-1.0) - eps;
        double expected = theta * theta;
        double drel = std::fabs(r.d2 - expected) / (1.0 + expected);
        bool ok = r.ok && drel < 1e-14 &&
                  std::fabs(r.gx[0]) < 1e-12 &&
                  std::fabs(r.gx[1] + 2.0 * theta) < 1e-12;
        char detail[128];
        std::snprintf(detail, sizeof detail, "d2 %.17g rel %.3e gx1 %.17g",
                      r.d2, drel, r.gx[1]);
        report("audit: differentiable points arbitrarily near antipode answer", ok,
               detail);
    }
    {
        /* This is deliberately inside the former 8*(n+1)*epsilon cosine
         * exclusion band: the point is near, but not at, the cut locus. */
        const int n = 64;
        const double eps = 1e-7;
        std::vector<double> x(n, 0.0), y(n, 0.0);
        x[0] = 1.0;
        y[0] = -std::cos(eps);
        y[1] = std::sin(eps);
        Run r = run_single(x.data(), y.data(), n,
                           ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        const double theta = std::acos(-1.0) - eps;
        const double drel = std::fabs(r.d2 - theta * theta) /
                            (1.0 + theta * theta);
        bool ok = r.ok && drel < 1e-14 &&
                  std::fabs(r.gx[0]) < 1e-12 &&
                  std::fabs(r.gx[1] + 2.0 * theta) < 1e-12;
        char detail[128];
        std::snprintf(detail, sizeof(detail),
                      "d2 %.17g rel %.3e gx0 %.17g gx1 %.17g",
                      r.d2, drel, r.gx[0], r.gx[1]);
        report("audit: 64D near-antipode inside former tolerance band answers",
               ok, detail);
    }
    {
        /* Bit-exact denormal-pivot counterexample. The first coordinate is
         * denorm_min, so the old max-abs division selected a denormal pivot;
         * every pivot product then underflowed to zero and a non-antipodal
         * pair at theta ~= 3.0352963674903606 was refused. */
        const double m = std::numeric_limits<double>::denorm_min();
        const double x[3] = { m, 0.99498743710662, -0.1 };
        const double y[3] = { -m, -0.9787619731068428, 0.205 };
        Run r = run_single(x, y, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        const double expected_d2 = 9.213024038500178;
        bool ok = r.ok && std::isfinite(r.d2) &&
                  std::fabs(r.d2 - expected_d2) < 2e-14;
        char detail[128];
        std::snprintf(detail, sizeof(detail), "d2 %.17g expected %.17g",
                      r.d2, expected_d2);
        report("audit: denormal pivot does not refuse valid spherical pair",
               ok, detail);
    }
    {
        /* The raw y coordinates deliberately make cos(eps) round to 1 for
         * eps <= 1e-9. The reverse rule must still retain the radial-looking
         * gy[0] tangent component supplied by the normalized point. */
        const double x[3] = { 1.0, 0.0, 0.0 };
        const double epsilons[] = { 1e-6, 1e-7, 1e-8, 1e-9,
                                    1e-10, 1e-11, 1e-12 };
        bool ok = true;
        char detail[160] = "";
        for (double eps : epsilons) {
            const double y[3] = { -std::cos(eps), std::sin(eps), 0.0 };
            Run r = run_single(x, y, 3,
                               ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
            double theta = std::atan2(std::sin(eps), -std::cos(eps));
            double expected = -2.0 * theta * std::sin(eps) /
                              std::sqrt(y[0] * y[0] + y[1] * y[1]);
            double tolerance = std::fabs(expected) * 1e-11 + 1e-15;
            ok = ok && r.ok && std::isfinite(r.gy[0]) &&
                 std::fabs(r.gy[0] - expected) < tolerance;
            if (eps == 1e-9)
                std::snprintf(detail, sizeof(detail),
                              "eps %.0e gy0 %.17g expected %.17g",
                              eps, r.gy[0], expected);
        }
        report("audit: near-antipode y-gradient keeps gy0 through 1e-12",
               ok, detail);
    }
    {
        const double x[1] = { 0.0 };
        const double y[1] = { 3.0 };
        eshkol_manifold_factor_t f = {
            ESHKOL_SPACE_FORM_EUCLIDEAN, 0, 1, 0.0, 0.0 };
        Run r = run_product(x, y, 1, &f, 1);
        report("audit: zero factor weight contributes exact zero", r.ok &&
               r.d2 == 0.0 && all_exactly(r.gx, 0.0) && all_exactly(r.gy, 0.0));
    }
    {
        const double x[1] = { 1e308 };
        const double y[1] = { -1e308 };
        eshkol_manifold_factor_t f = {
            ESHKOL_SPACE_FORM_EUCLIDEAN, 0, 1, 0.0, 0.0 };
        Run zero = run_product(x, y, 1, &f, 1);
        f.weight = 1.0;
        Run positive = run_product(x, y, 1, &f, 1);
        const double wx[1] = { 1e155 };
        const double wy[1] = { -1e155 };
        eshkol_manifold_factor_t weighted = {
            ESHKOL_SPACE_FORM_EUCLIDEAN, 0, 1, 0.0, 1e-3 };
        Run finite_weight = run_product(wx, wy, 1, &weighted, 1);
        double weight_d2_rel = 1.0, weight_gx_rel = 1.0, weight_gy_rel = 1.0;
        if (finite_weight.ok) {
            weight_d2_rel = std::fabs(finite_weight.d2 - 4e307) / 4e307;
            weight_gx_rel = std::fabs(finite_weight.gx[0] - 4e152) / 4e152;
            weight_gy_rel = std::fabs(finite_weight.gy[0] + 4e152) / 4e152;
        }
        report("audit: zero weight skips overflowing distance arithmetic",
               zero.ok && zero.d2 == 0.0 && all_exactly(zero.gx, 0.0) &&
               all_exactly(zero.gy, 0.0) && !positive.ok && finite_weight.ok &&
               weight_d2_rel < 1e-14 && weight_gx_rel < 1e-14 &&
               weight_gy_rel < 1e-14);
    }
    {
        Run neg = run_single(kEucX, kEucY, 3,
                             ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0, -0.7);
        Run three = run_single(kEucX, kEucY, 3,
                               ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0, 3.0);
        bool ok = neg.ok && three.ok;
        for (size_t i = 0; i < 3 && ok; ++i)
            ok = neg.gx[i] == -0.7 * (2.0 * (kEucX[i] - kEucY[i])) &&
                 three.gx[i] == 3.0 * (2.0 * (kEucX[i] - kEucY[i]));
        report("audit: upstream cotangent scales both backward rules", ok);
    }
    {
        eshkol_manifold_factor_t bad[2] = {
            { ESHKOL_SPACE_FORM_HYPERBOLIC, 0, 2, -1.0, 1.0 },
            { ESHKOL_SPACE_FORM_EUCLIDEAN,  0, 2, 0.0, 1.0 },
        };
        Run r = run_product(kProdX, kProdY, 7, bad, 2);   /* 2+2 != 7 */
        report("refuses factor dimensions that do not cover the point", !r.ok);
    }
    {
        eshkol_manifold_factor_t bad[1] = { { 7, 0, 3, 1.0, 1.0 } };
        Run r = run_product(kBallX, kBallY, 3, bad, 1);
        report("refuses an unknown space form", !r.ok);
    }
    {
        eshkol_manifold_factor_t bad[1] = {
            { ESHKOL_SPACE_FORM_EUCLIDEAN, 0, 3, 0.0, -1.0 } };
        Run r = run_product(kEucX, kEucY, 3, bad, 1);
        report("refuses a negative factor weight", !r.ok);
    }

    /* ---- 8. finite differences, last ------------------------------------ */
    {
        eshkol_manifold_factor_t f = { ESHKOL_SPACE_FORM_HYPERBOLIC, 0, 3, -1.0, 1.0 };
        Run r = run_single(kBallX, kBallY, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        double w = fd_max_rel(kBallX, kBallY, 3, &f, 1, r.gx, 1e-6);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", w);
        report("FD: hyperbolic", w < 1e-8, detail);
    }
    {
        eshkol_manifold_factor_t f = { ESHKOL_SPACE_FORM_SPHERICAL, 0, 3, 1.0, 1.0 };
        Run r = run_single(kSphX, kSphY, 3, ESHKOL_SPACE_FORM_SPHERICAL, 1.0);
        double w = fd_max_rel_projected_x(kSphX, kSphY, 3, &f, 1, r.gx, 1e-6);
        double wy = fd_max_rel_y(kSphX, kSphY, 3, &f, 1, r.gy, 1e-6);
        char detail[96];
        std::snprintf(detail, sizeof detail, "x %.3e, y %.3e", w, wy);
        report("FD: spherical x and y on the sphere", w < 1e-8 && wy < 1e-8, detail);
    }
    {
        eshkol_manifold_factor_t f = { ESHKOL_SPACE_FORM_EUCLIDEAN, 0, 3, 0.0, 1.0 };
        Run r = run_single(kEucX, kEucY, 3, ESHKOL_SPACE_FORM_EUCLIDEAN, 0.0);
        double w = fd_max_rel(kEucX, kEucY, 3, &f, 1, r.gx, 1e-6);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", w);
        report("FD: euclidean", w < 1e-8, detail);
    }
    {
        Run r = run_product(kProdX, kProdY, 7, kProdF, 3);
        double w = fd_max_rel_projected_x(kProdX, kProdY, 7, kProdF, 3, r.gx, 1e-6);
        double wy = fd_max_rel_y(kProdX, kProdY, 7, kProdF, 3, r.gy, 1e-6);
        char detail[96];
        std::snprintf(detail, sizeof detail, "x %.3e, y %.3e", w, wy);
        report("FD: product H2 x S2 x R2, x and y", w < 1e-8 && wy < 1e-8, detail);
    }
    {
        /* FD AT COINCIDENCE. A central difference here steps to x +/- h, both
         * at distance ~h from x, so both values are ~h^2 and the quotient is
         * ~0 — which happens to be right. It is included because it is the one
         * place FD agrees with the exact answer for the wrong reason, and
         * saying so is cheaper than letting a reader assume it was the check
         * that carried the case. Checks 2 and 5 are what carry it. */
        eshkol_manifold_factor_t f = { ESHKOL_SPACE_FORM_HYPERBOLIC, 0, 3, -1.0, 1.0 };
        Run r = run_single(kBallX, kBallX, 3, ESHKOL_SPACE_FORM_HYPERBOLIC, -1.0);
        double w = fd_max_rel(kBallX, kBallX, 3, &f, 1, r.gx, 1e-6);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max rel = %.3e", w);
        report("FD: hyperbolic at coincidence", w < 1e-8, detail);
    }

    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
