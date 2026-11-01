/*
 * signed_curvature_reference_grid_test.cpp — the ADR-0018 reference-grid gate.
 *
 * The f64 shared core (inc/eshkol/backend/riemannian_core.h) is checked
 * against an INDEPENDENT reference written in binary128 (__float128, 113-bit
 * significand) across every finite binade of the signed curvature K:
 *
 *   K = +-2^e for e = -1074 .. 1023 (all subnormal binades, DBL_TRUE_MIN and
 *   2^1023), K = +-DBL_MAX, and K = 0 exactly,
 *
 * on a fixed set of point configurations per K: chart points near the origin,
 * radius-scaled points at fixed fractions of 2/sqrt|K| (the negative ball
 * boundary, the positive equator), near-boundary / far-hemisphere points,
 * one-ulp separations, exact coincidence, a near-antipodal pair, mixed
 * 1e300/1e-300 scales and orthogonal subnormals. For each case it compares
 * the distance, the squared distance, the curvature jet (d, dd/dK, d2d/dK2)
 * and the endpoint gradient of d^2, and the REFUSAL CLASS: a case the
 * reference marks outside the chart (or unrepresentable in f64) must be
 * refused, and every other case must be answered.
 *
 * The reference does not share algebra with the core. It works in the
 * Bachmann-Becigneul-Ganea coordinates u = x/2 with Mobius addition (+)_K and
 *
 *     d = (2/sqrt|K|) artan_K(sqrt|K| |(-u) (+)_K v|) = 2 r A(K r^2),
 *     A(s) = atan(sqrt s)/sqrt s  (s > 0),  artanh(sqrt -s)/sqrt -s  (s < 0),
 *
 * evaluated in binary128 with second-order Taylor jets for the K and
 * coordinate derivatives. The core evaluates the asinh/asin form in the
 * alpha = 1 chart with compensated f64 kernels.
 *
 * Where binary128 arithmetic is not available (Apple arm64, MSVC) the test
 * runs the same comparisons against the committed golden subset
 * (signed_curvature_reference_golden.h), which the binary128 build re-derives
 * and checks bit for bit, so the golden rows cannot drift from the reference.
 *
 * It also checks the identities of the ADR gate that need no reference:
 * left/right jets at K = 0, exp/log round trips, transport isometry, exact
 * zero squared-distance gradient at coincidence, the raw-distance derivative
 * refusing at coincidence, attention continuity through K = 0 and its flat
 * reduction, and the convention/migration contract.
 *
 * Output ends with "Results: N passed, 0 failed" and one
 * "GRID <mode> cases=... binades=..." summary line read by
 * scripts/run_signed_curvature_gate.sh.
 */

#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__SIZEOF_FLOAT128__) && __has_include(<quadmath.h>) && \
    !defined(ESHKOL_DISABLE_BINARY128)
#include <quadmath.h>
#define ESHKOL_HAS_BINARY128 1
#endif

#include "eshkol/backend/riemannian_core.h"
#include "signed_curvature_reference_golden.h"

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* what) {
    if (ok) { ++g_pass; return; }
    ++g_fail;
    std::printf("FAIL: %s\n", what);
}

static const int N = 3;

/* ── configurations (all f64 inputs; the reference reads the same bits) ── */

enum { CFG_FIXED, CFG_RADIUS, CFG_FAR, CFG_ULP, CFG_COINCIDENT, CFG_ANTIPODE,
       CFG_MIXED, CFG_SUBNORMAL, CFG_COUNT };
static const char* kCfgName[CFG_COUNT] = {
    "fixed", "radius", "far", "ulp", "coincident", "antipode", "mixed",
    "subnormal"};

/* Fill x, y for configuration @p cfg at curvature @p K. Returns false when
 * the configuration does not apply at this sign of K. */
static bool make_case(int cfg, double K, double* x, double* y) {
    const double rho = (K == 0.0) ? 1.0 : 2.0 / std::sqrt(std::fabs(K));
    switch (cfg) {
    case CFG_FIXED:
        x[0] = 0.3; x[1] = -0.2; x[2] = 0.1;
        y[0] = -0.25; y[1] = 0.15; y[2] = 0.2;
        return true;
    case CFG_RADIUS:
        x[0] = rho * 0.5; x[1] = rho * 0.3; x[2] = rho * -0.2;
        y[0] = rho * -0.6; y[1] = rho * 0.55; y[2] = rho * 0.4;
        return std::isfinite(rho);
    case CFG_FAR:
        if (K < 0.0) {
            x[0] = rho * 0.999; x[1] = 0.0; x[2] = 0.0;
            y[0] = rho * -0.5; y[1] = rho * 0.86; y[2] = 0.0;
        } else {
            x[0] = rho * 3.0; x[1] = rho * 0.1; x[2] = 0.0;
            y[0] = rho * -0.2; y[1] = rho * 5.0; y[2] = rho * 1.0;
        }
        return std::isfinite(rho);
    case CFG_ULP:
        x[0] = rho * 0.5; x[1] = rho * 0.3; x[2] = rho * -0.2;
        for (int i = 0; i < N; i++) y[i] = std::nextafter(x[i], HUGE_VAL);
        return std::isfinite(rho);
    case CFG_COINCIDENT:
        x[0] = rho * 0.5; x[1] = rho * 0.3; x[2] = rho * -0.2;
        for (int i = 0; i < N; i++) y[i] = x[i];
        return std::isfinite(rho);
    case CFG_ANTIPODE: {
        if (!(K > 0.0)) return false;
        /* The antipode of x is -x/(|B||x|^2), B = -K/4. Perturb by 2^-30. */
        x[0] = rho * 0.5; x[1] = rho * 0.2; x[2] = rho * 0.1;
        const double x2 = 0.25 + 0.04 + 0.01; /* |x|^2 / rho^2 */
        const double s = -(1.0 + 0x1p-30) / x2;  /* times rho in units */
        y[0] = rho * (s * 0.5); y[1] = rho * (s * 0.2); y[2] = rho * (s * 0.1 + 1e-6);
        return std::isfinite(rho);
    }
    case CFG_MIXED:
        if (K < 0.0) return false;
        x[0] = 1e300; x[1] = -1e-300; x[2] = 0.0;
        y[0] = 1e-300; y[1] = 1e300; y[2] = 0.0;
        return true;
    case CFG_SUBNORMAL:
        x[0] = 3.0 * DBL_TRUE_MIN; x[1] = 0.0; x[2] = 0.0;
        y[0] = 0.0; y[1] = 5.0 * DBL_TRUE_MIN; y[2] = 0.0;
        return true;
    }
    return false;
}

/* ── comparison ───────────────────────────────────────────────────────── */

/* Answered f64 vs reference rounded to f64: relative tolerance plus an
 * absolute floor in the subnormal range, where f64 has no relative
 * precision to offer. */
static bool close(double got, double ref, double rel) {
    if (got == ref) return true;
    if (!std::isfinite(got) || !std::isfinite(ref)) return false;
    const double err = std::fabs(got - ref);
    return err <= rel * std::fabs(ref) || err <= 0x1p-1060;
}

static double rel_err(double got, double ref) {
    if (got == ref) return 0.0;
    const double s = std::fabs(ref);
    if (s < 0x1p-1000) return 0.0; /* subnormal outputs are judged absolutely */
    return std::fabs(got - ref) / s;
}

struct Expect {
    int  domain_ok;     /* 1: chart points, finite pair; 0: must be refused */
    int  jet_ok;        /* 1: d, d', d'' representable; 0: refusal expected; 2: edge */
    int  grad_ok;       /* 1: gradient representable; 0: refusal; 2: edge */
    double d, d1, d2;   /* distance jet */
    double g[N];        /* d(d^2)/dx */
};

struct Stats { double d = 0, sq = 0, d1 = 0, d2 = 0, g = 0; long cases = 0, refused = 0; };

static const double TOL_D  = 0x1p-46;  /* distance, squared distance */
static const double TOL_J  = 0x1p-36;  /* curvature jets */
static const double TOL_G  = 0x1p-40;  /* endpoint gradient of d^2 */

static void compare_case(const char* tag, int cfg, double K, const double* x,
                         const double* y, const Expect& e, Stats& st) {
    char what[256];
    /* The near-antipodal pair sits 2^-30 from the cut locus, where the log
     * direction (hence the d^2 gradient) has condition number ~2^30: f64
     * rounding of the inputs alone moves it by ~2^-23. */
    const double tol_g = (cfg == CFG_ANTIPODE) ? 0x1p-18 : TOL_G;
    const double tol_j = (cfg == CFG_ANTIPODE) ? 0x1p-18 : TOL_J;
    double d = 0.0;
    const char* why = eshkol_rm_distance(x, y, K, N, &d);
    st.cases++;
    if (!e.domain_ok) {
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: outside the chart, "
                      "must be refused", tag, kCfgName[cfg], K);
        check(why != nullptr, what);
        st.refused++;
        return;
    }
    std::snprintf(what, sizeof what, "%s cfg=%s K=%a: distance answered (%s)",
                  tag, kCfgName[cfg], K, why ? why : "");
    check(why == nullptr, what);
    if (why) return;
    std::snprintf(what, sizeof what, "%s cfg=%s K=%a: d=%a ref=%a", tag,
                  kCfgName[cfg], K, d, e.d);
    check(close(d, e.d, TOL_D), what);
    if (rel_err(d, e.d) > st.d) st.d = rel_err(d, e.d);

    double sq = 0.0;
    why = eshkol_rm_sq_distance(x, y, K, N, &sq);
    const double sq_ref = e.d * e.d;
    if (std::isfinite(sq_ref) && sq_ref < DBL_MAX) {
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: d^2=%a ref=%a", tag,
                      kCfgName[cfg], K, sq, sq_ref);
        check(why == nullptr && close(sq, sq_ref, 2 * TOL_D), what);
        if (!why && rel_err(sq, sq_ref) > st.sq) st.sq = rel_err(sq, sq_ref);
    }

    double jd = 0.0, j1 = 0.0, j2 = 0.0;
    why = eshkol_rm_distance_dK(x, y, K, N, &jd, &j1, &j2);
    if (e.jet_ok == 1) {
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: jet answered (%s) "
                      "d1=%a ref=%a d2=%a ref=%a", tag, kCfgName[cfg], K,
                      why ? why : "", j1, e.d1, j2, e.d2);
        check(why == nullptr && close(jd, e.d, TOL_D) && close(j1, e.d1, tol_j) &&
              close(j2, e.d2, tol_j), what);
        if (!why && cfg != CFG_ANTIPODE) {
            if (rel_err(j1, e.d1) > st.d1) st.d1 = rel_err(j1, e.d1);
            if (rel_err(j2, e.d2) > st.d2) st.d2 = rel_err(j2, e.d2);
        }
    } else if (e.jet_ok == 0) {
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: unrepresentable jet "
                      "must be refused", tag, kCfgName[cfg], K);
        check(why != nullptr, what);
    }

    double g[N], scratch[2 * N];
    why = eshkol_rm_sq_distance_grad(x, y, K, N, g, nullptr, scratch);
    if (e.grad_ok == 1) {
        bool ok = (why == nullptr);
        for (int i = 0; ok && i < N; i++) ok = close(g[i], e.g[i], tol_g);
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: grad answered (%s) "
                      "g0=%a ref=%a", tag, kCfgName[cfg], K, why ? why : "",
                      g[0], e.g[0]);
        check(ok, what);
        if (!why && cfg != CFG_ANTIPODE)
            for (int i = 0; i < N; i++)
                if (rel_err(g[i], e.g[i]) > st.g) st.g = rel_err(g[i], e.g[i]);
    } else if (e.grad_ok == 0) {
        std::snprintf(what, sizeof what, "%s cfg=%s K=%a: unrepresentable grad "
                      "must be refused", tag, kCfgName[cfg], K);
        check(why != nullptr, what);
    }
}

/* ══ The binary128 reference ═══════════════════════════════════════════ */
#ifdef ESHKOL_HAS_BINARY128
typedef __float128 q;

/* f(t0 + t) = v + a t + b t^2 */
struct J { q v, a, b; };
static J jc(q v) { return {v, 0, 0}; }
static J operator+(J x, J y) { return {x.v + y.v, x.a + y.a, x.b + y.b}; }
static J operator-(J x, J y) { return {x.v - y.v, x.a - y.a, x.b - y.b}; }
static J operator-(J x) { return {-x.v, -x.a, -x.b}; }
static J operator*(J x, J y) {
    return {x.v * y.v, x.v * y.a + x.a * y.v, x.v * y.b + x.a * y.a + x.b * y.v};
}
static J apply(J x, q f, q f1, q f2) {
    return {f, f1 * x.a, f1 * x.b + f2 * x.a * x.a / 2};
}
static J recip(J x) {
    const q r = 1 / x.v;
    return apply(x, r, -r * r, 2 * r * r * r);
}
static J operator/(J x, J y) { return x * recip(y); }
static J jsqrt(J x) {
    const q r = sqrtq(x.v);
    return apply(x, r, 1 / (2 * r), -1 / (4 * r * x.v));
}

/* A(s) = atan(sqrt s)/sqrt s = sum (-1)^k s^k/(2k+1), with A' and A''. */
static void A_ref(q s, q* A, q* A1, q* A2) {
    if (fabsq(s) <= (q)0.5) {
        /* p0 = s^k, p1 = s^(k-1), p2 = s^(k-2) */
        q a = 0, a1 = 0, a2 = 0, p0 = 1, p1 = 0, p2 = 0;
        for (int k = 0; k < 400; k++) {
            const q c = (q)((k & 1) ? -1 : 1) / (q)(2 * k + 1);
            a += c * p0;
            a1 += c * (q)k * p1;
            a2 += c * (q)k * (q)(k - 1) * p2;
            p2 = p1; p1 = p0; p0 *= s;
            if (k > 8 && fabsq(p0) < (q)1e-45) break;
        }
        *A = a; *A1 = a1; *A2 = a2;
        return;
    }
    q a;
    if (s > 0) { const q u = sqrtq(s); a = atanq(u) / u; }
    else       { const q u = sqrtq(-s); a = atanhq(u) / u; }
    const q g = 1 / (1 + s), g1 = -g * g;
    const q a1 = (g - a) / (2 * s);
    *A = a; *A1 = a1; *A2 = (g1 - 3 * a1) / (2 * s);
}

/* d(K; x, y) with every argument a jet (one of them seeded). */
static bool d_ref(J K, const J* x, const J* y, J* out) {
    J u[N], v[N];
    for (int i = 0; i < N; i++) { u[i] = x[i] * jc((q)0.5); v[i] = y[i] * jc((q)0.5); }
    J uv = jc(0), uu = jc(0), vv = jc(0);
    for (int i = 0; i < N; i++) { uv = uv - u[i] * v[i]; uu = uu + u[i] * u[i]; vv = vv + v[i] * v[i]; }
    /* domain: 1 + K|u|^2 > 0 for both (u = x/2) */
    if (!((jc(1) + K * uu).v > 0) || !((jc(1) + K * vv).v > 0)) return false;
    /* (-u) (+)_K v, Bachmann et al.: a = -u, b = v, <a,b> = uv */
    const J c1 = jc(1) - jc(2) * K * uv - K * vv;
    const J c2 = jc(1) + K * uu;
    const J den = jc(1) - jc(2) * K * uv + K * K * uu * vv;
    if (den.v == 0) return false; /* the antipode */
    J m2 = jc(0);
    for (int i = 0; i < N; i++) {
        const J mi = (c1 * (-u[i]) + c2 * v[i]) / den;
        m2 = m2 + mi * mi;
    }
    if (m2.v == 0) { *out = jc(0); return true; }
    const J r = jsqrt(m2);
    const J s = K * r * r;
    q A, A1, A2;
    A_ref(s.v, &A, &A1, &A2);
    *out = jc(2) * r * apply(s, A, A1, A2);
    return true;
}

static bool f64_finite(q v) { return fabsq(v) <= (q)DBL_MAX; }

/* Reference expectation for one case. */
static Expect reference(double K, const double* x, const double* y) {
    Expect e{};
    for (int i = 0; i < N; i++)
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return e;
    /* f64 representability of the chart's conformal denominators. */
    for (const double* p : {x, y}) {
        q n2 = 0; for (int i = 0; i < N; i++) n2 += (q)p[i] * (q)p[i];
        const q den = 1 + (q)K * n2 / 4;
        if (!(den > 0) || !f64_finite(den)) return e;
    }
    J Kj = {(q)K, (q)1, (q)0}, xj[N], yj[N];
    for (int i = 0; i < N; i++) { xj[i] = jc((q)x[i]); yj[i] = jc((q)y[i]); }
    J d;
    if (!d_ref(Kj, xj, yj, &d)) return e;
    if (!f64_finite(d.v)) return e;
    e.domain_ok = 1;
    e.d = (double)d.v;
    const q d1 = d.a, d2 = 2 * d.b;
    const q big = (q)DBL_MAX;
    if (f64_finite(d1) && f64_finite(d2) &&
        fabsq(d1) < big / 4 && fabsq(d2) < big / 4) e.jet_ok = 1;
    else if (!f64_finite(d1) || !f64_finite(d2)) e.jet_ok = 0;
    else e.jet_ok = 2;
    e.d1 = (double)d1; e.d2 = (double)d2;
    e.grad_ok = 1;
    for (int i = 0; i < N; i++) {
        J xs[N]; for (int k = 0; k < N; k++) xs[k] = jc((q)x[k]);
        xs[i].a = (q)1;
        J di;
        if (!d_ref(jc((q)K), xs, yj, &di)) { e.grad_ok = 2; break; }
        const q gi = 2 * di.v * di.a;
        if (!f64_finite(gi)) { e.grad_ok = 0; break; }
        if (fabsq(gi) > big / 4) e.grad_ok = 2;
        e.g[i] = (double)gi;
    }
    /* Unrepresentable outputs are compared only by refusal class; store a
     * canonical 0 so the golden rows stay finite literals. */
    if (e.jet_ok != 1 && !(std::isfinite(e.d1) && std::isfinite(e.d2))) e.d1 = e.d2 = 0.0;
    if (e.grad_ok != 1)
        for (int i = 0; i < N; i++) if (!std::isfinite(e.g[i])) e.g[i] = 0.0;
    if (std::memcmp(x, y, sizeof(double) * N) == 0) {
        e.d = e.d1 = e.d2 = 0.0;
        for (int i = 0; i < N; i++) e.g[i] = 0.0;
        e.jet_ok = 1; e.grad_ok = 1;
    }
    return e;
}

/* Independent legacy (alpha = 2, ball 1/sqrt c) distance for the migration
 * fixture: acosh(1 + 2c|x-y|^2/((1-c|x|^2)(1-c|y|^2)))/sqrt c. */
static q legacy_distance(q c, const double* x, const double* y) {
    q e = 0, a = 0, b = 0;
    for (int i = 0; i < N; i++) {
        e += ((q)x[i] - y[i]) * ((q)x[i] - y[i]);
        a += (q)x[i] * x[i]; b += (q)y[i] * y[i];
    }
    return acoshq(1 + 2 * c * e / ((1 - c * a) * (1 - c * b))) / sqrtq(c);
}
#endif /* ESHKOL_HAS_BINARY128 */

/* ── the curvature grid ───────────────────────────────────────────────── */

static std::vector<double> curvature_grid() {
    std::vector<double> ks;
    ks.push_back(0.0);
    for (int e = -1074; e <= 1023; e++) {
        ks.push_back(std::ldexp(1.0, e));
        ks.push_back(-std::ldexp(1.0, e));
    }
    ks.push_back(DBL_MAX);
    ks.push_back(-DBL_MAX);
    return ks;
}

/* The golden subset: every 64th binade, the binade edges and the series
 * switch neighbourhood. */
static bool golden_k(double K) {
    if (K == 0.0 || std::fabs(K) == DBL_MAX) return true;
    int e = 0;
    (void)std::frexp(std::fabs(K), &e);
    e -= 1;
    return (e % 64 == 0) || e == -1074 || e == -1022 || e == 1023 ||
           (e >= -10 && e <= 2);
}

static int golden_cfg(int cfg) {
    return cfg == CFG_FIXED || cfg == CFG_RADIUS || cfg == CFG_FAR ||
           cfg == CFG_COINCIDENT;
}

/* ── identities that need no reference ───────────────────────────────── */

static void identity_checks() {
    char what[200];
    const double x[N] = {0.3, -0.2, 0.1}, y[N] = {-0.25, 0.15, 0.2};
    const double v[N] = {0.7, 0.2, -0.3};

    /* Left and right jets at K = 0 agree with the K = 0 jet. */
    double d0 = 0, a0 = 0, b0 = 0, dp = 1, ap = 1, bp = 1, dm = 1, am = 1, bm = 1;
    check(!eshkol_rm_distance_dK(x, y, 0.0, N, &d0, &a0, &b0) &&
          !eshkol_rm_distance_dK(x, y, DBL_TRUE_MIN, N, &dp, &ap, &bp) &&
          !eshkol_rm_distance_dK(x, y, -DBL_TRUE_MIN, N, &dm, &am, &bm),
          "jets answered at K = 0 and +-DBL_TRUE_MIN");
    check(d0 == dp && d0 == dm && a0 == ap && a0 == am && b0 == bp && b0 == bm,
          "left and right curvature jets agree with the K = 0 jet");
    double dflat = -1.0;
    check(!eshkol_rm_distance(x, y, 0.0, N, &dflat) &&
          dflat == std::hypot(std::hypot(std::hypot(0.0, x[0] - y[0]), x[1] - y[1]), x[2] - y[2]),
          "K = 0 distance is exactly the Euclidean |x - y| (alpha = 1)");

    /* exp/log round trip and transport isometry on both signs. */
    for (double K : {-4.0, -1.0, -0x1p-20, 0.0, 0x1p-20, 1.0, 4.0, 30.0}) {
        double e[N], l[N], s1[N], s2[N], t[N], s3[N];
        bool ok = !eshkol_rm_log_map(x, y, K, N, l, s1) &&
                  !eshkol_rm_exp_map(x, l, K, N, e, s2);
        for (int i = 0; ok && i < N; i++) ok = std::fabs(e[i] - y[i]) <= 1e-14;
        std::snprintf(what, sizeof what, "exp_x(log_x y) = y at K=%g", K);
        check(ok, what);
        ok = !eshkol_rm_transport(x, y, v, K, N, t, s3);
        const double nx = eshkol_rm_lambda(x, K, N) * eshkol_rm_norm(v, N);
        const double ny = eshkol_rm_lambda(y, K, N) * eshkol_rm_norm(t, N);
        std::snprintf(what, sizeof what, "transport is an isometry at K=%g", K);
        check(ok && std::fabs(nx - ny) <= 1e-14 * nx, what);
    }

    /* d^2 gradient is exactly zero at coincidence; raw d has no derivative. */
    for (double K : {-1.0, 0.0, 1.0}) {
        double g[N] = {1, 1, 1}, scratch[2 * N];
        const char* why = eshkol_rm_sq_distance_grad(x, x, K, N, g, nullptr, scratch);
        uint64_t bits = 0;
        for (int i = 0; i < N; i++) { uint64_t b; std::memcpy(&b, &g[i], 8); bits |= (b << 1); }
        std::snprintf(what, sizeof what, "d^2 gradient is exactly +-0 at x == y, K=%g", K);
        check(why == nullptr && bits == 0, what);
    }
    {
        double out = 0, dout = 0;
        const double dx[N] = {1, 0, 0}, dy[N] = {0, 0, 0};
        check(!eshkol_rm_distance_directional(x, x, dx, dy, -1.0, N, &out, &dout),
              "raw distance derivative refuses at coincidence");
    }

    /* Attention: weights continuous through K = 0, and the flat reduction. */
    {
        const double keys[3 * N] = {0.3, -0.2, 0.1, -0.25, 0.15, 0.2, 0.5, 0.4, -0.1};
        double sq[3], w0[3], wp[3], wm[3];
        const double t = eshkol_rm_default_temperature(N);
        bool ok = !eshkol_rm_attention_row(x, keys, N, 3, N, 0.0, t, sq, w0) &&
                  !eshkol_rm_attention_row(x, keys, N, 3, N, DBL_TRUE_MIN, t, sq, wp) &&
                  !eshkol_rm_attention_row(x, keys, N, 3, N, -DBL_TRUE_MIN, t, sq, wm);
        for (int j = 0; ok && j < 3; j++) ok = (w0[j] == wp[j] && w0[j] == wm[j]);
        check(ok, "attention weights continuous through K = 0");
        double ref[3], mx = -HUGE_VAL, sum = 0;
        for (int j = 0; j < 3; j++) {
            double e2 = 0;
            for (int i = 0; i < N; i++) e2 += (x[i] - keys[j * N + i]) * (x[i] - keys[j * N + i]);
            ref[j] = -e2 / (2 * t);
            if (ref[j] > mx) mx = ref[j];
        }
        for (int j = 0; j < 3; j++) { ref[j] = std::exp(ref[j] - mx); sum += ref[j]; }
        ok = true;
        for (int j = 0; j < 3; j++) ok = ok && std::fabs(w0[j] - ref[j] / sum) <= 1e-15;
        check(ok, "flat reduction: K = 0 weights are softmax(-|q-k|^2/(2 sqrt d))");
        /* finite logits across every binade of both signs */
        ok = true;
        for (int e = -1074; ok && e <= 1023; e += 1) {
            for (double sgn : {-1.0, 1.0}) {
                const double K = sgn * std::ldexp(1.0, e);
                /* fixed chart inputs for |K| <= 1 (inside every ball of that
                 * size), radius-scaled ones above, so d^2 is representable */
                const double rho = (std::fabs(K) <= 1.0) ? 1.0
                                                         : 2.0 / std::sqrt(std::fabs(K));
                double q[N], kk[2 * N];
                for (int i = 0; i < N; i++) { q[i] = rho * 0.2 * (i + 1); kk[i] = q[i]; kk[N + i] = rho * -0.1 * (i + 1); }
                double sq2[2], w[2];
                const char* why = eshkol_rm_attention_row(q, kk, N, 2, N, K, t, sq2, w);
                ok = (why == nullptr) && std::isfinite(w[0]) && std::isfinite(w[1]) &&
                     std::fabs(w[0] + w[1] - 1.0) <= 1e-15;
                if (!ok) std::printf("attention binade K=%a: %s\n", K, why ? why : "non-finite");
            }
        }
        check(ok, "attention weights finite and normalised across every K binade");
    }

    /* Convention and migration. */
    {
        eshkol_geometry_header_t h;
        eshkol_rm_geometry_header(-1.0, &h);
        check(eshkol_rm_check_geometry_header(&h) == nullptr, "current header accepted");
        h.convention = ESHKOL_GEOMETRY_CONVENTION_UNVERSIONED;
        check(eshkol_rm_check_geometry_header(&h) != nullptr, "unversioned record refused");
        h.convention = ESHKOL_GEOMETRY_CONVENTION_LEGACY_BALL_V1;
        check(eshkol_rm_check_geometry_header(&h) != nullptr, "legacy record refused");
        h.convention = 99u;
        check(eshkol_rm_check_geometry_header(&h) != nullptr, "unknown convention refused");
        eshkol_rm_geometry_header(-1.0, &h);
        h.alpha = 2.0;
        check(eshkol_rm_check_geometry_header(&h) != nullptr, "mismatched alpha refused");
    }
    for (double K : {-0.25, -1.0, -9.0}) {
        /* legacy ball points (radius 1/sqrt(-K)) */
        const double c = -K, r = 1.0 / std::sqrt(c);
        const double x1[N] = {0.4 * r, -0.3 * r, 0.1 * r}, y1[N] = {-0.5 * r, 0.2 * r, 0.6 * r};
        double x2[N], y2[N], d2;
        bool ok = !eshkol_rm_migrate_legacy_point(x1, K, N, x2) &&
                  !eshkol_rm_migrate_legacy_point(y1, K, N, y2) &&
                  !eshkol_rm_distance(x2, y2, K, N, &d2);
        for (int i = 0; ok && i < N; i++) ok = (x2[i] == 2.0 * x1[i]);
#ifdef ESHKOL_HAS_BINARY128
        const double dl = (double)legacy_distance(c, x1, y1);
#else
        /* independent f64 legacy closed form when binary128 is unavailable */
        double e = 0, a = 0, b = 0;
        for (int i = 0; i < N; i++) { e += (x1[i] - y1[i]) * (x1[i] - y1[i]); a += x1[i] * x1[i]; b += y1[i] * y1[i]; }
        const double dl = std::acosh(1 + 2 * c * e / ((1 - c * a) * (1 - c * b))) / std::sqrt(c);
#endif
        std::snprintf(what, sizeof what, "legacy point migration preserves distance at K=%g", K);
        check(ok && std::fabs(d2 - dl) <= 1e-13 * dl, what);
        /* tangent: legacy |v|_g = 2/(1-c|x|^2)|v|, migrated with x_v2 = 2x_v1 */
        const double v1[N] = {0.1, 0.2, -0.3};
        double v2[N];
        double a1 = 0; for (int i = 0; i < N; i++) a1 += x1[i] * x1[i];
        const double nl = 2.0 / (1.0 - c * a1) * eshkol_rm_norm(v1, N);
        ok = !eshkol_rm_migrate_legacy_tangent(v1, K, N, v2);
        const double nn = eshkol_rm_lambda(x2, K, N) * eshkol_rm_norm(v2, N);
        std::snprintf(what, sizeof what, "legacy tangent migration preserves the norm at K=%g", K);
        check(ok && std::fabs(nn - nl) <= 1e-14 * nl, what);
    }
    {
        double t = 0;
        check(!eshkol_rm_migrate_legacy_temperature(0.5, 2.0, &t) && t == 4.0,
              "temperature migration t = d_ref / s");
        check(eshkol_rm_migrate_legacy_temperature(0.0, 2.0, &t) != nullptr,
              "non-positive legacy scale refused");
        const double p[N + 1] = {0.6, 0.0, 0.0, 0.8};
        double xc[N], pb[N + 1];
        bool ok = !eshkol_rm_sphere_to_chart(p, 1.0, N, xc) &&
                  !eshkol_rm_chart_to_sphere(xc, 1.0, N, pb);
        for (int i = 0; ok && i <= N; i++) ok = std::fabs(pb[i] - p[i]) <= 1e-15;
        check(ok, "embedded sphere <-> chart round trip");
        double xs[N] = {0.3, -0.2, 0.1}, ys[N] = {-0.25, 0.15, 0.2}, ps[N + 1], qs[N + 1];
        double dc = 0, da = 0;
        ok = !eshkol_rm_chart_to_sphere(xs, 4.0, N, ps) &&
             !eshkol_rm_chart_to_sphere(ys, 4.0, N, qs) &&
             !eshkol_rm_distance(xs, ys, 4.0, N, &dc) &&
             !eshkol_rm_sphere_distance(ps, qs, 4.0, N + 1, &da);
        check(ok && std::fabs(dc - da) <= 1e-15 * da,
              "chart distance equals the embedded-sphere arc length");
    }
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* emit = nullptr;
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--emit-golden") == 0 && i + 1 < argc) emit = argv[++i];

    identity_checks();

    Stats st;
    long golden_rows = 0, binades = 0;
#ifdef ESHKOL_HAS_BINARY128
    long golden_matched = 0;
#endif
    const char* mode = "golden";

#ifdef ESHKOL_HAS_BINARY128
    mode = "binary128";
    FILE* gf = emit ? std::fopen(emit, "w") : nullptr;
    if (emit && !gf) { std::printf("FAIL: cannot write %s\n", emit); return 1; }
    if (gf) std::fprintf(gf, "/* Generated by signed_curvature_reference_grid_test --emit-golden.\n"
                             " * binary128 reference values rounded to f64. Do not edit. */\n"
                             "#define ESHKOL_SCURV_GOLDEN_ROWS \\\n");
    const std::vector<double> ks = curvature_grid();
    binades = (long)ks.size();
    for (double K : ks) {
        for (int cfg = 0; cfg < CFG_COUNT; cfg++) {
            double x[N], y[N];
            if (!make_case(cfg, K, x, y)) continue;
            const Expect e = reference(K, x, y);
            compare_case("grid", cfg, K, x, y, e, st);
            if (golden_k(K) && golden_cfg(cfg)) {
                golden_rows++;
                if (gf)
                    std::fprintf(gf, "  {%d, %a, %d, %d, %d, %a, %a, %a, {%a, %a, %a}}, \\\n",
                                 cfg, K, e.domain_ok, e.jet_ok, e.grad_ok, e.d, e.d1,
                                 e.d2, e.g[0], e.g[1], e.g[2]);
                /* the committed golden row must equal the live reference */
                for (const auto& r : kSignedCurvatureGolden) {
                    if (r.cfg == cfg && r.K == K) {
                        bool same = r.domain_ok == e.domain_ok && r.jet_ok == e.jet_ok &&
                                    r.grad_ok == e.grad_ok &&
                                    std::memcmp(&r.d, &e.d, 8) == 0 &&
                                    std::memcmp(&r.d1, &e.d1, 8) == 0 &&
                                    std::memcmp(&r.d2, &e.d2, 8) == 0 &&
                                    std::memcmp(r.g, e.g, sizeof e.g) == 0;
                        if (same) golden_matched++;
                        else std::printf("golden row differs: cfg=%s K=%a\n", kCfgName[cfg], K);
                        break;
                    }
                }
            }
        }
    }
    if (gf) { std::fprintf(gf, "\n"); std::fclose(gf); }
    if (!emit) {
        char what[160];
        std::snprintf(what, sizeof what, "committed golden subset equals the live "
                      "binary128 reference (%ld of %ld rows, %zu committed)",
                      golden_matched, golden_rows,
                      sizeof kSignedCurvatureGolden / sizeof kSignedCurvatureGolden[0]);
        check(golden_matched == golden_rows &&
              (size_t)golden_rows == sizeof kSignedCurvatureGolden /
                                         sizeof kSignedCurvatureGolden[0], what);
    }
#else
    (void)emit;
    for (const auto& r : kSignedCurvatureGolden) {
        double x[N], y[N];
        if (!make_case(r.cfg, r.K, x, y)) { check(false, "golden row names an inapplicable case"); continue; }
        Expect e{};
        e.domain_ok = r.domain_ok; e.jet_ok = r.jet_ok; e.grad_ok = r.grad_ok;
        e.d = r.d; e.d1 = r.d1; e.d2 = r.d2;
        for (int i = 0; i < N; i++) e.g[i] = r.g[i];
        compare_case("golden", r.cfg, r.K, x, y, e, st);
        golden_rows++;
    }
    binades = 0;
    for (const auto& r : kSignedCurvatureGolden) if (r.cfg == CFG_FIXED) binades++;
#endif

    std::printf("GRID %s cases=%ld refused=%ld curvatures=%ld golden_rows=%ld "
                "max_rel d=%.3g d2=%.3g dK=%.3g dKK=%.3g grad=%.3g\n",
                mode, st.cases, st.refused, binades, golden_rows,
                st.d, st.sq, st.d1, st.d2, st.g);
    std::printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
