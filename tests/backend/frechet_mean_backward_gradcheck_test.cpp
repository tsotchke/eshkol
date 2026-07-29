/**
 * @file frechet_mean_backward_gradcheck_test.cpp
 * @brief Gradient check for the Fréchet (Karcher) mean backward pass.
 *
 * The weighted Fréchet mean on the Poincaré ball is defined implicitly, as the
 * stationary point of the weighted variance:
 *
 *     F(mu; X, w) = sum_i w_i log_mu(x_i) = 0.
 *
 * `tensor_frechet_mean_backward` differentiates THAT, via the implicit function
 * theorem, rather than unrolling the fixed-point iteration that computes it. The
 * two are different functions: the unrolled derivative depends on the starting
 * point and the iteration count, neither of which is a property of the Fréchet
 * mean. So this test cannot check the rule against "the derivative of the
 * iteration as written" — it checks it against central finite differences of the
 * mean itself, taken with the iteration converged far tighter than the
 * difference step, which is the derivative of the mathematical object.
 *
 * The forward here is written independently from the definitions (Möbius
 * addition, exp/log at a basepoint, Karcher iteration) so that a sign error
 * cannot be shared with the implementation under test.
 *
 * The other half of the test is the RESIDUAL GATE. An implicit derivative
 * evaluated at a point that has not converged is not a large error or a NaN —
 * it is a smooth, plausible, wrong vector, which is the worst failure class
 * available, because nothing downstream can tell it from a gradient. The rule
 * must refuse there, and checks 5-8 assert that it does.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol.h>
#include <eshkol/backend/tensor_backward.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#define ESHKOL_HAVE_FORK_DEATH_TESTS 1
#endif

namespace {

int g_passed = 0;
int g_failed = 0;

void report(const char* name, bool ok, const char* detail = nullptr) {
    std::printf("  %-48s %s", name, ok ? "PASS" : "FAIL");
    if (detail) std::printf("   [%s]", detail);
    std::printf("\n");
    if (ok) ++g_passed; else ++g_failed;
}

/* ===== Independent Poincaré-ball forward ================================== */

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double t = 0.0;
    for (size_t i = 0; i < a.size(); i++) t += a[i] * b[i];
    return t;
}
double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

/** @brief Möbius addition a (+)_c x on the ball of curvature -c. */
std::vector<double> mobius_add(const std::vector<double>& a,
                               const std::vector<double>& x, double c) {
    double ax = dot(a, x), aa = dot(a, a), xx = dot(x, x);
    double A1 = 1.0 + 2.0 * c * ax + c * xx;
    double B1 = 1.0 - c * aa;
    double D  = 1.0 + 2.0 * c * ax + c * c * aa * xx;
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); i++) out[i] = (A1 * a[i] + B1 * x[i]) / D;
    return out;
}

/** @brief log_mu(x) = ((1-c|mu|^2)/sqrt(c)) * artanh(sqrt(c)|u|)/|u| * u,
 *         u = (-mu) (+)_c x. */
std::vector<double> log_map(const std::vector<double>& mu,
                            const std::vector<double>& x, double c) {
    std::vector<double> neg(mu.size());
    for (size_t i = 0; i < mu.size(); i++) neg[i] = -mu[i];
    std::vector<double> u = mobius_add(neg, x, c);
    double s = std::sqrt(c), r = norm(u);
    std::vector<double> out(mu.size(), 0.0);
    if (r <= 0.0) return out;
    double k = (1.0 - c * dot(mu, mu)) / s;
    double f = k * std::atanh(s * r) / r;
    for (size_t i = 0; i < mu.size(); i++) out[i] = f * u[i];
    return out;
}

/** @brief exp_mu(v) = mu (+)_c [ tanh(sqrt(c)|v|/(1-c|mu|^2)) * v/(sqrt(c)|v|) ]. */
std::vector<double> exp_map(const std::vector<double>& mu,
                            const std::vector<double>& v, double c) {
    double s = std::sqrt(c), nv = norm(v);
    if (nv <= 0.0) return mu;
    double lam_half = s / (1.0 - c * dot(mu, mu));   /* sqrt(c) * lambda_mu / 2 */
    double t = std::tanh(lam_half * nv) / (s * nv);
    std::vector<double> scaled(v.size());
    for (size_t i = 0; i < v.size(); i++) scaled[i] = t * v[i];
    return mobius_add(mu, scaled, c);
}

/** @brief Weighted Karcher iteration to the stationary point. Iterates
 *  mu <- exp_mu( sum_i w_i log_mu(x_i) / sum_i w_i ) until the stationarity
 *  residual stops improving. Converges far below any finite-difference step so
 *  that FD of this function measures the derivative of the mean itself. */
std::vector<double> frechet_mean(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& w, double c,
                                 double* out_resid = nullptr) {
    const size_t n = X.size(), d = X[0].size();
    double wsum = 0.0;
    for (double wi : w) wsum += wi;

    /* Start from the weighted Euclidean average, projected inside the ball. */
    std::vector<double> mu(d, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < d; k++) mu[k] += w[i] * X[i][k] / wsum;

    double resid = 0.0;
    for (int it = 0; it < 4000; it++) {
        std::vector<double> step(d, 0.0);
        for (size_t i = 0; i < n; i++) {
            std::vector<double> lg = log_map(mu, X[i], c);
            for (size_t k = 0; k < d; k++) step[k] += w[i] * lg[k];
        }
        resid = norm(step);
        for (size_t k = 0; k < d; k++) step[k] /= wsum;
        if (resid == 0.0) break;
        std::vector<double> next = exp_map(mu, step, c);
        double mv = 0.0;
        for (size_t k = 0; k < d; k++) mv = std::max(mv, std::fabs(next[k] - mu[k]));
        mu = next;
        if (mv == 0.0) break;   /* fixed point to the last bit */
    }
    if (out_resid) *out_resid = resid;
    return mu;
}

/* ===== Fixture ============================================================ */

constexpr double kK = -1.0;              /* sectional curvature; c = 1        */
constexpr double kC = 1.0;
constexpr int64_t kN = 4;
constexpr int64_t kD = 3;

/* Points strictly inside the unit ball, deliberately asymmetric so the mean is
 * not at the origin and no coordinate accidentally decouples. */
const double kPts[kN * kD] = {
     0.30, -0.10,  0.20,
    -0.45,  0.25, -0.05,
     0.10,  0.50,  0.15,
    -0.20, -0.35,  0.40,
};
const double kW[kN] = { 0.7, 1.3, 0.4, 2.1 };
/* Upstream cotangent dL/dmu — distinct per coordinate. */
const double kG[kD] = { 1.0, -2.5, 0.75 };

std::vector<std::vector<double>> points_from(const double* flat) {
    std::vector<std::vector<double>> X((size_t)kN, std::vector<double>((size_t)kD));
    for (int64_t i = 0; i < kN; i++)
        for (int64_t k = 0; k < kD; k++) X[(size_t)i][(size_t)k] = flat[i * kD + k];
    return X;
}

/** @brief L(X, w) = <g, mu*(X, w)>. */
double loss(const double* flat_pts, const double* w, double c) {
    std::vector<std::vector<double>> X = points_from(flat_pts);
    std::vector<double> wv(w, w + kN);
    std::vector<double> mu = frechet_mean(X, wv, c);
    double acc = 0.0;
    for (int64_t k = 0; k < kD; k++) acc += kG[k] * mu[(size_t)k];
    return acc;
}

/** @brief Nodes for one Fréchet-mean backward. */
struct Fixture {
    ad_node_t pts{};
    ad_node_t wts{};
    ad_node_t out{};
    int64_t   pts_shape[2] = { kN, kD };
    int64_t   w_shape[1]   = { kN };
    int64_t   out_shape[1] = { kD };
    std::vector<double> mu;
    std::vector<double> dmu;
    std::vector<double> pts_data;
    std::vector<double> w_data;
    double resid = 0.0;

    explicit Fixture(double curvature_K = kK) {
        double c = -curvature_K;
        pts_data.assign(kPts, kPts + (size_t)(kN * kD));
        w_data.assign(kW, kW + (size_t)kN);
        if (c > 0.0) {
            mu = frechet_mean(points_from(pts_data.data()), w_data, c, &resid);
        } else {
            /* Euclidean limit: the weighted average. */
            mu.assign((size_t)kD, 0.0);
            double wsum = 0.0;
            for (double wi : w_data) wsum += wi;
            for (int64_t i = 0; i < kN; i++)
                for (int64_t k = 0; k < kD; k++)
                    mu[(size_t)k] += w_data[(size_t)i] * pts_data[(size_t)(i * kD + k)] / wsum;
        }
        dmu.assign(kG, kG + (size_t)kD);

        pts.type         = AD_NODE_VARIABLE;
        pts.tensor_value = pts_data.data();
        pts.shape        = pts_shape;
        pts.ndim         = 2;

        wts.type         = AD_NODE_VARIABLE;
        wts.tensor_value = w_data.data();
        wts.shape        = w_shape;
        wts.ndim         = 1;

        out.type   = AD_NODE_FRECHET_MEAN;
        out.input1 = &pts;
        out.input2 = &wts;
        out.shape  = out_shape;
        out.ndim   = 1;
        out.tensor_value    = mu.data();
        out.tensor_gradient = dmu.data();

        int64_t* p = (int64_t*)&out.params;
        p[0] = kN;
        p[1] = kD;
        double Kd = curvature_K;
        std::memcpy(&p[2], &Kd, sizeof Kd);
        p[3] = 0;   /* default tolerance */
    }

    void dispatch() { eshkol_tensor_backward_dispatch(&out); }
    const double* grad_pts() const { return (const double*)pts.tensor_gradient; }
    const double* grad_w()   const { return (const double*)wts.tensor_gradient; }
};

double rel_err(double a, double b) {
    return std::fabs(a - b) / (1.0 + std::fabs(b));
}

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
bool refuses(void (*body)()) {
    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        FILE* devnull = std::freopen("/dev/null", "w", stderr);
        (void)devnull;
        body();
        std::_Exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    if (WIFSIGNALED(status)) return true;
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

/* A mu displaced off the fixed point. This is THE dangerous case: the implicit
 * formulas evaluate happily here and return a plausible wrong gradient. */
void body_not_converged() {
    Fixture f;
    f.mu[0] += 1e-3;
    f.dispatch();
}

/* Barely off — still must refuse, because "nearly stationary" is not stationary
 * and the error in the returned gradient is first order in the displacement. */
void body_barely_not_converged() {
    Fixture f;
    f.mu[0] += 1e-6;
    f.dispatch();
}

void body_point_outside_ball() {
    Fixture f;
    f.pts_data[0] = 1.5;      /* |x| > 1 with c = 1 */
    f.dispatch();
}

void body_mean_outside_ball() {
    Fixture f;
    f.mu[0] = 1.25;
    f.dispatch();
}

void body_missing_points() {
    Fixture f;
    f.out.input1 = nullptr;
    f.dispatch();
}

void body_nonpositive_weight_sum() {
    Fixture f;
    for (int64_t i = 0; i < kN; i++) f.w_data[(size_t)i] = 0.0;
    f.dispatch();
}
#endif  /* ESHKOL_HAVE_FORK_DEATH_TESTS */

}  // namespace

int main() {
    std::printf("=== Frechet mean backward: implicit differentiation ===\n");

    /* ---- 0. the forward reaches a genuine fixed point ------------------ */
    {
        Fixture f;
        char detail[128];
        std::snprintf(detail, sizeof detail,
                      "|sum w_i log_mu(x_i)| = %.3e, |mu| = %.6f",
                      f.resid, norm(f.mu));
        report("forward converges to a stationary point", f.resid < 1e-14, detail);
    }

    /* ---- 1/2. implicit gradient vs central finite differences ---------- */
    {
        Fixture f;
        f.dispatch();
        const double* dpts = f.grad_pts();
        const double* dw   = f.grad_w();

        if (!dpts || !dw) {
            report("d L / d points vs finite differences", false, "no gradient");
            report("d L / d weights vs finite differences", false, "no gradient");
        } else {
            /* The iteration is converged to the last bit, so the only error left
             * in the difference is the O(h^2) truncation of the central stencil.
             * h = 1e-5 puts that near 1e-10 while keeping cancellation small. */
            const double h = 1e-5;

            double worst_x = 0.0;
            std::vector<double> P(kPts, kPts + (size_t)(kN * kD));
            for (size_t t = 0; t < P.size(); t++) {
                double saved = P[t];
                P[t] = saved + h; double lp = loss(P.data(), kW, kC);
                P[t] = saved - h; double lm = loss(P.data(), kW, kC);
                P[t] = saved;
                double fd = (lp - lm) / (2.0 * h);
                worst_x = std::max(worst_x, rel_err(dpts[t], fd));
            }
            char d1[128];
            std::snprintf(d1, sizeof d1, "max rel err = %.3e (bar 1e-6)", worst_x);
            report("d L / d points vs finite differences", worst_x < 1e-6, d1);

            double worst_w = 0.0;
            std::vector<double> W(kW, kW + (size_t)kN);
            for (size_t t = 0; t < W.size(); t++) {
                double saved = W[t];
                W[t] = saved + h; double lp = loss(kPts, W.data(), kC);
                W[t] = saved - h; double lm = loss(kPts, W.data(), kC);
                W[t] = saved;
                double fd = (lp - lm) / (2.0 * h);
                worst_w = std::max(worst_w, rel_err(dw[t], fd));
            }
            char d2[128];
            std::snprintf(d2, sizeof d2, "max rel err = %.3e (bar 1e-6)", worst_w);
            report("d L / d weights vs finite differences", worst_w < 1e-6, d2);
        }
    }

    /* ---- 3. the implicit gradient is NOT the unrolled one -------------- */
    {
        /* Differentiating a truncated iteration gives a different answer. Run
         * the iteration for a fixed small number of steps and finite-difference
         * THAT; if the implicit rule happened to agree, the distinction this
         * whole rule rests on would be untestable. Showing the two differ is
         * what makes the choice of implicit differentiation a real decision
         * rather than a restatement. */
        auto truncated_loss = [](const double* flat, int iters) {
            std::vector<std::vector<double>> X = points_from(flat);
            std::vector<double> w(kW, kW + kN);
            double wsum = 0.0;
            for (double wi : w) wsum += wi;
            std::vector<double> mu((size_t)kD, 0.0);
            for (int64_t i = 0; i < kN; i++)
                for (int64_t k = 0; k < kD; k++)
                    mu[(size_t)k] += w[(size_t)i] * X[(size_t)i][(size_t)k] / wsum;
            for (int it = 0; it < iters; it++) {
                std::vector<double> step((size_t)kD, 0.0);
                for (int64_t i = 0; i < kN; i++) {
                    std::vector<double> lg = log_map(mu, X[(size_t)i], kC);
                    for (int64_t k = 0; k < kD; k++)
                        step[(size_t)k] += w[(size_t)i] * lg[(size_t)k];
                }
                for (int64_t k = 0; k < kD; k++) step[(size_t)k] /= wsum;
                mu = exp_map(mu, step, kC);
            }
            double acc = 0.0;
            for (int64_t k = 0; k < kD; k++) acc += kG[k] * mu[(size_t)k];
            return acc;
        };

        Fixture f;
        f.dispatch();
        const double* dpts = f.grad_pts();
        double biggest_gap = 0.0;
        if (dpts) {
            const double h = 1e-5;
            std::vector<double> P(kPts, kPts + (size_t)(kN * kD));
            for (size_t t = 0; t < P.size(); t++) {
                double saved = P[t];
                P[t] = saved + h; double lp = truncated_loss(P.data(), 1);
                P[t] = saved - h; double lm = truncated_loss(P.data(), 1);
                P[t] = saved;
                double fd1 = (lp - lm) / (2.0 * h);
                biggest_gap = std::max(biggest_gap, std::fabs(dpts[t] - fd1));
            }
        }
        char detail[128];
        std::snprintf(detail, sizeof detail,
                      "one-step-unrolled differs by up to %.3e", biggest_gap);
        report("implicit != unrolled (the decision is real)",
               dpts != nullptr && biggest_gap > 1e-4, detail);
    }

    /* ---- 4. Euclidean limit K = 0 against the closed form -------------- */
    {
        Fixture f(0.0);
        f.dispatch();
        const double* dpts = f.grad_pts();
        const double* dw   = f.grad_w();
        double wsum = 0.0;
        for (double wi : f.w_data) wsum += wi;

        bool ok = (dpts != nullptr && dw != nullptr);
        double worst = 0.0;
        if (ok) {
            for (int64_t i = 0; i < kN; i++) {
                for (int64_t k = 0; k < kD; k++) {
                    double want = kW[i] / wsum * kG[k];
                    worst = std::max(worst, std::fabs(dpts[i * kD + k] - want));
                }
                double want_w = 0.0;
                for (int64_t k = 0; k < kD; k++)
                    want_w += kG[k] * (kPts[i * kD + k] - f.mu[(size_t)k]);
                want_w /= wsum;
                worst = std::max(worst, std::fabs(dw[i] - want_w));
            }
            ok = worst < 1e-14;
        }
        char detail[128];
        std::snprintf(detail, sizeof detail, "max abs err = %.3e", worst);
        report("Euclidean limit (K=0) vs closed form", ok, detail);
    }

    /* ---- 5-10. the gates ---------------------------------------------- */
#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
    report("refuses a non-converged mean (1e-3 off)",  refuses(&body_not_converged));
    report("refuses a barely non-converged mean (1e-6)", refuses(&body_barely_not_converged));
    report("refuses a point outside the ball",         refuses(&body_point_outside_ball));
    report("refuses a mean outside the ball",          refuses(&body_mean_outside_ball));
    report("refuses missing retained points",          refuses(&body_missing_points));
    report("refuses a non-positive total weight",      refuses(&body_nonpositive_weight_sum));
#else
    std::printf("  (gate checks skipped: no fork on this platform)\n");
#endif

    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
