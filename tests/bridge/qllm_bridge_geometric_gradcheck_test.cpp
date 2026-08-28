/**
 * @file qllm_bridge_geometric_gradcheck_test.cpp
 * @brief Gradient checks for the four geometric bridge ops whose backwards were
 *        missing entirely — SW-65.
 *
 * ad_hyperbolic_distance, ad_poincare_exp_map, ad_poincare_log_map and
 * ad_geodesic_attention record TENSOR-valued AD nodes (types 33, 34, 35, 37).
 * None had a backward. They did not refuse and they did not warn: those type
 * numbers sit in the band eshkol_tensor_backward_dispatch treated as "scalar
 * ops differentiated by codegen", so they fell into its default: and the
 * reverse sweep propagated nothing. Every input gradient came back exactly 0.0,
 * which a caller cannot distinguish from a genuine zero.
 *
 * WHAT EACH GRADIENT IS CHECKED AGAINST. Finite differences are the last line
 * here, not the first. Three independent exact references come first:
 *
 *  1. THE COMMITTED GOLDEN VECTORS. tests/qllm_oracle/golden/ holds full
 *     Jacobians for exp_0 and log_0 computed by Eshkol's reverse-mode AD over
 *     an INDEPENDENTLY WRITTEN Eshkol transcription of the same formulas. Two
 *     cases are asserted here by their case id, so these C rules are tied to
 *     the same oracle qLLM's own kernels are tested against. This is the
 *     strongest check in the file: a shared-mistake failure would have to occur
 *     in two separate implementations, in two languages, written from different
 *     sources.
 *
 *  2. THE CONFORMAL-FACTOR IDENTITY. For the Poincare distance, the Euclidean
 *     gradient magnitudes are exactly the conformal factors at each argument:
 *     |grad_x d| = lambda_x = 2/(1-c|x|^2) and |grad_y d| = lambda_y. This
 *     holds at EVERY interior pair, not just at a fixture point, because the
 *     Riemannian gradient of a distance function is a unit vector. It is a
 *     property of the geometry, derived nowhere near the code under test.
 *
 *  3. THE INVERSE-JACOBIAN IDENTITY. log_x(exp_x(v)) = v identically, so
 *     J_log * J_exp = I. That couples the two rules to each other with no
 *     appeal to either derivation: a sign error in one would have to be
 *     mirrored exactly in the other to survive.
 *
 * Both refusals are checked in a forked child, because eshkol_fatal exits the
 * process. A refusal that did not actually terminate would otherwise read as a
 * pass.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdarg>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/tensor_backward.h"

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
}

#define TOLERANCE 1e-6
#define STEP      1e-6

static int g_passed = 0, g_failed = 0;
static void report(const char* name, bool ok, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::printf("%s: %s — ", ok ? "PASS" : "FAIL", name);
    std::vprintf(fmt, ap); std::printf("\n"); va_end(ap);
    if (ok) ++g_passed; else ++g_failed;
}

static double* ad_doubles(size_t n) {
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}
static ad_node_t* var_node(const double* data, const int64_t* shape, size_t ndim) {
    ad_node_t* n = arena_allocate_ad_node(get_global_arena());
    size_t c = 1; for (size_t i = 0; i < ndim; ++i) c *= (size_t)shape[i];
    double* b = ad_doubles(c); std::memcpy(b, data, c * sizeof(double));
    int64_t* s = (int64_t*)arena_allocate_zeroed(get_global_arena(), ndim * sizeof(int64_t));
    std::memcpy(s, shape, ndim * sizeof(int64_t));
    n->type = AD_NODE_VARIABLE; n->tensor_value = b; n->shape = s; n->ndim = ndim;
    return n;
}
static void sweep(ad_tape_t* t) {
    for (size_t i = t->num_nodes; i-- > 0;) eshkol_tensor_backward_dispatch(t->nodes[i]);
}
struct RelAcc { double dn = 0.0, nn = 0.0, an = 0.0; };
static void acc_add(RelAcc* a, double num, double ana) {
    double d = num - ana; a->dn += d*d; a->nn += num*num; a->an += ana*ana;
}
static double acc_val(const RelAcc* a) {
    double den = std::sqrt(a->nn) + std::sqrt(a->an);
    return den > 0.0 ? std::sqrt(a->dn)/den : 0.0;
}

/** @brief Run `fn` in a forked child; true iff the child exited non-zero
 *  (i.e. the rule actually refused rather than returning a number). */
static bool child_refuses(void (*fn)(void)) {
    std::fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) { fn(); _exit(0); }          /* reached only if no refusal */
    int st = 0; waitpid(pid, &st, 0);
    return !(WIFEXITED(st) && WEXITSTATUS(st) == 0);
}

/* =============================================== hyperbolic distance ===== */

/* |grad_x d| == lambda_x and |grad_y d| == lambda_y at every interior pair. */
static void check_distance_conformal_identity(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    struct { double x[3], y[3]; } cases[] = {
        {{ 0.30,-0.10, 0.20}, {-0.20, 0.25, 0.10}},
        {{ 0.00, 0.00, 0.00}, { 0.30,-0.40, 0.00}},
        {{ 0.55, 0.10,-0.20}, {-0.45,-0.30, 0.35}},
    };
    double worst = 0.0;
    for (auto& cs : cases) {
        ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
        ad_node_t* xn = var_node(cs.x, sh, 1);
        ad_node_t* yn = var_node(cs.y, sh, 1);
        ad_node_t* o = ad_hyperbolic_distance(t, xn, yn, -c);
        if (!o) { report("distance.conformal_identity", false, "forward refused"); return; }
        ((double*)o->tensor_gradient)[0] = 1.0;
        sweep(t);
        const double* gx = (const double*)xn->tensor_gradient;
        const double* gy = (const double*)yn->tensor_gradient;
        double nx = 0.0, ny = 0.0, xx = 0.0, yy = 0.0;
        for (int i = 0; i < 3; ++i) {
            nx += gx[i]*gx[i]; ny += gy[i]*gy[i];
            xx += cs.x[i]*cs.x[i]; yy += cs.y[i]*cs.y[i];
        }
        double lx = 2.0/(1.0 - c*xx), ly = 2.0/(1.0 - c*yy);
        worst = std::fmax(worst, std::fabs(std::sqrt(nx) - lx)/lx);
        worst = std::fmax(worst, std::fabs(std::sqrt(ny) - ly)/ly);
    }
    report("distance.conformal_factor_identity", worst < 1e-12,
           "max rel dev of |grad| from lambda over 3 pairs = %.3e", worst);
}

static void check_distance_fd(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3] = { 0.30,-0.10, 0.20 }, Y[3] = {-0.20, 0.25, 0.10 };
    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = var_node(X, sh, 1); ad_node_t* yn = var_node(Y, sh, 1);
    ad_node_t* o = ad_hyperbolic_distance(t, xn, yn, -c);
    ((double*)o->tensor_gradient)[0] = 1.0; sweep(t);
    double gx[3], gy[3];
    std::memcpy(gx, xn->tensor_gradient, sizeof gx);
    std::memcpy(gy, yn->tensor_gradient, sizeof gy);

    auto fwd = [&](const double* a, const double* b) {
        ad_node_t* an = var_node(a, sh, 1); ad_node_t* bn = var_node(b, sh, 1);
        ad_node_t* r = ad_hyperbolic_distance(nullptr, an, bn, -c);
        return ((const double*)r->tensor_value)[0];
    };
    RelAcc acc;
    for (int i = 0; i < 3; ++i) {
        double p[3], m[3];
        std::memcpy(p, X, sizeof p); std::memcpy(m, X, sizeof m);
        p[i] += STEP; m[i] -= STEP;
        acc_add(&acc, (fwd(p,Y)-fwd(m,Y))/(2*STEP), gx[i]);
        std::memcpy(p, Y, sizeof p); std::memcpy(m, Y, sizeof m);
        p[i] += STEP; m[i] -= STEP;
        acc_add(&acc, (fwd(X,p)-fwd(X,m))/(2*STEP), gy[i]);
    }
    double r = acc_val(&acc);
    report("distance.finite_difference", r < TOLERANCE,
           "aggregate L2 rel err %.3e over 6 partials", r);
}

static void distance_coincident_child(void) {
    const int64_t sh[1] = { 3 };
    double X[3] = { 0.2,-0.1, 0.3 };
    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* a = var_node(X, sh, 1), *b = var_node(X, sh, 1);
    ad_node_t* o = ad_hyperbolic_distance(t, a, b, -1.0);
    if (o && o->tensor_gradient) ((double*)o->tensor_gradient)[0] = 1.0;
    sweep(t);
}
static void check_distance_refuses_coincident(void) {
    report("distance.refuses_coincident_points", child_refuses(distance_coincident_child),
           "d(x,x) is a cone point; no derivative exists there");
}

/* ================================= exp / log maps vs golden vectors ====== */

/** @brief Full Jacobian of log_x(y) wrt y, by one reverse sweep per output. */
static void log_jacobian_wrt_y(const double* X, const double* Y, int n,
                               double c, double* J) {
    const int64_t sh[1] = { n };
    for (int k = 0; k < n; ++k) {
        ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
        ad_node_t* xn = var_node(X, sh, 1); ad_node_t* yn = var_node(Y, sh, 1);
        ad_node_t* o = ad_poincare_log_map(t, xn, yn, -c);
        double* g = (double*)o->tensor_gradient;
        for (int j = 0; j < n; ++j) g[j] = (j == k) ? 1.0 : 0.0;
        sweep(t);
        std::memcpy(&J[(size_t)k*n], yn->tensor_gradient, (size_t)n*sizeof(double));
    }
}
/** @brief Full Jacobian of exp_x(v) wrt v. */
static void exp_jacobian_wrt_v(const double* X, const double* V, int n,
                               double c, double* J) {
    const int64_t sh[1] = { n };
    for (int k = 0; k < n; ++k) {
        ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
        ad_node_t* xn = var_node(X, sh, 1); ad_node_t* vn = var_node(V, sh, 1);
        ad_node_t* o = ad_poincare_exp_map(t, xn, vn, -c);
        double* g = (double*)o->tensor_gradient;
        for (int j = 0; j < n; ++j) g[j] = (j == k) ? 1.0 : 0.0;
        sweep(t);
        std::memcpy(&J[(size_t)k*n], vn->tensor_gradient, (size_t)n*sizeof(double));
    }
}

static void check_log_golden(void) {
    /* tests/qllm_oracle/golden/poincare_log_map_origin.json,
     * case "poincare_log_map_origin.d2.c1.u0p5": y = (0.3, -0.4), c = 1,
     * d_out_d_y as committed. Base point is the origin, where the bridge's
     * log_x reduces to the log_0 the exporter transcribes. */
    const double O2[2] = { 0.0, 0.0 };
    const double Y[2]  = { 0.3, -0.4 };
    const double want[4] = {
         1.1831118647475904, -0.11266610143930732,
        -0.11266610143930728,  1.2488337572538528
    };
    double J[4];
    log_jacobian_wrt_y(O2, Y, 2, 1.0, J);
    double worst = 0.0;
    for (int i = 0; i < 4; ++i)
        worst = std::fmax(worst, std::fabs(J[i]-want[i])/std::fabs(want[i]));
    report("log_map.golden_vector_d2.c1.u0p5", worst < 1e-13,
           "max rel dev from the committed oracle Jacobian = %.3e", worst);
}

static void check_exp_golden(void) {
    /* tests/qllm_oracle/golden/poincare_exp_map_origin.json,
     * case "poincare_exp_map_origin.d2.c1.tv0p1": v = (0.06, -0.08), c = 1. */
    const double O2[2] = { 0.0, 0.0 };
    const double V[2]  = { 0.06, -0.08000000000000002 };
    const double want[4] = {
        0.9942990303047957, 0.003174554593016766,
        0.003174554593016787, 0.9924472067922024
    };
    double J[4];
    exp_jacobian_wrt_v(O2, V, 2, 1.0, J);
    double worst = 0.0;
    for (int i = 0; i < 4; ++i)
        worst = std::fmax(worst, std::fabs(J[i]-want[i])/std::fabs(want[i]));
    report("exp_map.golden_vector_d2.c1.tv0p1", worst < 1e-13,
           "max rel dev from the committed oracle Jacobian = %.3e", worst);
}

static void check_exp_log_inverse_jacobians(void) {
    /* log_x(exp_x(v)) = v, so J_log(y=exp_x(v)) * J_exp(v) = I. Couples the two
     * rules with no appeal to either derivation. Base point off the origin so
     * the Mobius terms are live. */
    const int64_t sh[1] = { 3 };
    const double X[3] = { 0.20,-0.15, 0.10 };
    const double V[3] = { 0.08, 0.05,-0.03 };
    const double c = 1.0;

    ad_node_t* xn = var_node(X, sh, 1); ad_node_t* vn = var_node(V, sh, 1);
    ad_node_t* e = ad_poincare_exp_map(nullptr, xn, vn, -c);
    if (!e) { report("exp_log.inverse_jacobians", false, "exp forward refused"); return; }
    double Yv[3]; std::memcpy(Yv, e->tensor_value, sizeof Yv);

    double Je[9], Jl[9];
    exp_jacobian_wrt_v(X, V, 3, c, Je);
    log_jacobian_wrt_y(X, Yv, 3, c, Jl);

    double worst = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < 3; ++m) acc += Jl[(size_t)i*3+m] * Je[(size_t)m*3+j];
            worst = std::fmax(worst, std::fabs(acc - (i==j ? 1.0 : 0.0)));
        }
    report("exp_log.inverse_jacobians", worst < 1e-9,
           "max |J_log * J_exp - I| = %.3e", worst);
}

static void check_exp_log_fd(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3] = { 0.20,-0.15, 0.10 }, V[3] = { 0.08, 0.05,-0.03 };
    double Y[3] = {-0.10, 0.30, 0.05 };

    /* exp: both operands. */
    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = var_node(X, sh, 1), *vn = var_node(V, sh, 1);
    ad_node_t* o = ad_poincare_exp_map(t, xn, vn, -c);
    double cot[3] = { 1.0, -0.5, 0.25 };
    std::memcpy(o->tensor_gradient, cot, sizeof cot); sweep(t);
    double gx[3], gv[3];
    std::memcpy(gx, xn->tensor_gradient, sizeof gx);
    std::memcpy(gv, vn->tensor_gradient, sizeof gv);
    auto expLoss = [&](const double* a, const double* b) {
        ad_node_t* an = var_node(a, sh, 1), *bn = var_node(b, sh, 1);
        ad_node_t* r = ad_poincare_exp_map(nullptr, an, bn, -c);
        const double* ov = (const double*)r->tensor_value;
        double s = 0.0; for (int i = 0; i < 3; ++i) s += cot[i]*ov[i]; return s;
    };
    RelAcc acc;
    for (int i = 0; i < 3; ++i) {
        double p[3], m[3];
        std::memcpy(p,X,sizeof p); std::memcpy(m,X,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (expLoss(p,V)-expLoss(m,V))/(2*STEP), gx[i]);
        std::memcpy(p,V,sizeof p); std::memcpy(m,V,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (expLoss(X,p)-expLoss(X,m))/(2*STEP), gv[i]);
    }
    double re = acc_val(&acc);
    report("exp_map.finite_difference", re < TOLERANCE,
           "aggregate L2 rel err %.3e over 6 partials", re);

    /* log: both operands. */
    ad_tape_t* t2 = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn2 = var_node(X, sh, 1), *yn2 = var_node(Y, sh, 1);
    ad_node_t* o2 = ad_poincare_log_map(t2, xn2, yn2, -c);
    std::memcpy(o2->tensor_gradient, cot, sizeof cot); sweep(t2);
    double lx[3], ly[3];
    std::memcpy(lx, xn2->tensor_gradient, sizeof lx);
    std::memcpy(ly, yn2->tensor_gradient, sizeof ly);
    auto logLoss = [&](const double* a, const double* b) {
        ad_node_t* an = var_node(a, sh, 1), *bn = var_node(b, sh, 1);
        ad_node_t* r = ad_poincare_log_map(nullptr, an, bn, -c);
        const double* ov = (const double*)r->tensor_value;
        double s = 0.0; for (int i = 0; i < 3; ++i) s += cot[i]*ov[i]; return s;
    };
    RelAcc acc2;
    for (int i = 0; i < 3; ++i) {
        double p[3], m[3];
        std::memcpy(p,X,sizeof p); std::memcpy(m,X,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc2, (logLoss(p,Y)-logLoss(m,Y))/(2*STEP), lx[i]);
        std::memcpy(p,Y,sizeof p); std::memcpy(m,Y,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc2, (logLoss(X,p)-logLoss(X,m))/(2*STEP), ly[i]);
    }
    double rl = acc_val(&acc2);
    report("log_map.finite_difference", rl < TOLERANCE,
           "aggregate L2 rel err %.3e over 6 partials", rl);
}

/* ================================================ geodesic attention ===== */

enum { GB = 1, GS = 3, GD = 4 };
static double gQ[GB*GS*GD], gK[GB*GS*GD], gV[GB*GS*GD], gCot[GB*GS*GD];

static void check_geodesic_fd(void) {
    const int64_t sh[3] = { GB, GS, GD };
    /* Distinct Q and K: coincident rows are a genuine non-differentiable point
     * and are covered by the refusal check below, not smuggled in here. */
    const double Q0[GB*GS*GD] = { 0.10, 0.20,-0.10, 0.05,
                                  -0.15, 0.08, 0.22,-0.04,
                                   0.03,-0.25, 0.11, 0.17 };
    const double K0[GB*GS*GD] = { 0.05,-0.20, 0.15, 0.10,
                                   0.18, 0.06,-0.12, 0.09,
                                  -0.07, 0.21, 0.04,-0.16 };
    const double V0[GB*GS*GD] = { 1.0, 2.0,-1.0, 0.5,
                                  0.3,-0.7, 1.4, 2.2,
                                 -0.9, 0.6, 0.1,-1.3 };
    const double C0[GB*GS*GD] = { 1.0,-0.5, 0.25, 0.75,
                                 -0.3, 0.9, 0.2,-1.1,
                                  0.4, 0.15,-0.6, 0.8 };
    std::memcpy(gQ,Q0,sizeof gQ); std::memcpy(gK,K0,sizeof gK);
    std::memcpy(gV,V0,sizeof gV); std::memcpy(gCot,C0,sizeof gCot);

    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* qn = var_node(gQ, sh, 3), *kn = var_node(gK, sh, 3), *vn = var_node(gV, sh, 3);
    ad_node_t* o = ad_geodesic_attention(t, qn, kn, vn, 2, -1.0, false);
    if (!o) { report("geodesic.finite_difference", false, "forward refused"); return; }
    std::memcpy(o->tensor_gradient, gCot, sizeof gCot);
    sweep(t);
    double dq[GB*GS*GD], dk[GB*GS*GD], dv[GB*GS*GD];
    std::memcpy(dq, qn->tensor_gradient, sizeof dq);
    std::memcpy(dk, kn->tensor_gradient, sizeof dk);
    std::memcpy(dv, vn->tensor_gradient, sizeof dv);

    auto loss = [&](const double* q, const double* k, const double* v) {
        ad_node_t* a = var_node(q, sh, 3), *b = var_node(k, sh, 3), *cc = var_node(v, sh, 3);
        ad_node_t* r = ad_geodesic_attention(nullptr, a, b, cc, 2, -1.0, false);
        const double* ov = (const double*)r->tensor_value;
        double s = 0.0; for (int i = 0; i < GB*GS*GD; ++i) s += gCot[i]*ov[i]; return s;
    };
    RelAcc acc;
    for (int i = 0; i < GB*GS*GD; ++i) {
        double p[GB*GS*GD], m[GB*GS*GD];
        std::memcpy(p,gQ,sizeof p); std::memcpy(m,gQ,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (loss(p,gK,gV)-loss(m,gK,gV))/(2*STEP), dq[i]);
        std::memcpy(p,gK,sizeof p); std::memcpy(m,gK,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (loss(gQ,p,gV)-loss(gQ,m,gV))/(2*STEP), dk[i]);
        std::memcpy(p,gV,sizeof p); std::memcpy(m,gV,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (loss(gQ,gK,p)-loss(gQ,gK,m))/(2*STEP), dv[i]);
    }
    double r = acc_val(&acc);
    report("geodesic.finite_difference", r < TOLERANCE,
           "aggregate L2 rel err %.3e over %d partials", r, 3*GB*GS*GD);
}

static void check_geodesic_causal_fd(void) {
    const int64_t sh[3] = { GB, GS, GD };
    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* qn = var_node(gQ, sh, 3), *kn = var_node(gK, sh, 3), *vn = var_node(gV, sh, 3);
    ad_node_t* o = ad_geodesic_attention(t, qn, kn, vn, 2, -1.0, true);
    if (!o) { report("geodesic.causal_finite_difference", false, "forward refused"); return; }
    std::memcpy(o->tensor_gradient, gCot, sizeof gCot);
    sweep(t);
    double dq[GB*GS*GD], dv[GB*GS*GD];
    std::memcpy(dq, qn->tensor_gradient, sizeof dq);
    std::memcpy(dv, vn->tensor_gradient, sizeof dv);
    auto loss = [&](const double* q, const double* v) {
        ad_node_t* a = var_node(q, sh, 3), *b = var_node(gK, sh, 3), *cc = var_node(v, sh, 3);
        ad_node_t* r = ad_geodesic_attention(nullptr, a, b, cc, 2, -1.0, true);
        const double* ov = (const double*)r->tensor_value;
        double s = 0.0; for (int i = 0; i < GB*GS*GD; ++i) s += gCot[i]*ov[i]; return s;
    };
    RelAcc acc;
    for (int i = 0; i < GB*GS*GD; ++i) {
        double p[GB*GS*GD], m[GB*GS*GD];
        std::memcpy(p,gQ,sizeof p); std::memcpy(m,gQ,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (loss(p,gV)-loss(m,gV))/(2*STEP), dq[i]);
        std::memcpy(p,gV,sizeof p); std::memcpy(m,gV,sizeof m); p[i]+=STEP; m[i]-=STEP;
        acc_add(&acc, (loss(gQ,p)-loss(gQ,m))/(2*STEP), dv[i]);
    }
    double r = acc_val(&acc);
    /* The last key column is masked out of every row, so dK there is exactly
     * zero and FD sees the same; it is excluded from the ratio by construction
     * (both legs zero contribute nothing to either norm). */
    report("geodesic.causal_finite_difference", r < TOLERANCE,
           "aggregate L2 rel err %.3e with causal masking", r);
}

static void geodesic_coincident_child(void) {
    const int64_t sh[3] = { GB, GS, GD };
    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* qn = var_node(gQ, sh, 3);
    ad_node_t* vn = var_node(gV, sh, 3);
    /* Q and K the same node: every diagonal (i, i) pair coincides exactly. */
    ad_node_t* o = ad_geodesic_attention(t, qn, qn, vn, 2, -1.0, false);
    if (o && o->tensor_gradient) std::memcpy(o->tensor_gradient, gCot, sizeof gCot);
    sweep(t);
}
static void check_geodesic_refuses_coincident(void) {
    report("geodesic.refuses_query_equals_key",
           child_refuses(geodesic_coincident_child),
           "distance scoring is non-differentiable when a query row equals a key row");
}

/* ============================ structural: the loud default (SW-65 root) === */

static void unregistered_tensor_node_child(void) {
    /* A tensor-carrying node with an input, of a type no rule handles. Before
     * the fix this returned silently, having propagated nothing. */
    const int64_t sh[1] = { 3 };
    double X[3] = { 0.1, 0.2, 0.3 };
    ad_node_t* in = var_node(X, sh, 1);
    ad_node_t* fake = arena_allocate_ad_node(get_global_arena());
    fake->type = AD_NODE_MOBIUS_MATMUL;      /* tensor-ish, deliberately unregistered */
    fake->tensor_value = ad_doubles(3);
    fake->tensor_gradient = ad_doubles(3);
    ((double*)fake->tensor_gradient)[0] = 1.0;
    fake->shape = in->shape; fake->ndim = 1;
    fake->input1 = in;
    eshkol_tensor_backward_dispatch(fake);
}
static void check_loud_default(void) {
    report("dispatch.unregistered_tensor_node_is_fatal",
           child_refuses(unregistered_tensor_node_child),
           "a tensor-carrying node with inputs and no rule must abort, not zero");
}

static void leaf_variable_child(void) {
    /* The companion: a LEAF carrying a tensor payload must still be a no-op.
     * The fatal keys on "payload AND has inputs" precisely so input variables,
     * which carry tensor_value, are not swept up by it. */
    const int64_t sh[1] = { 3 };
    double X[3] = { 0.1, 0.2, 0.3 };
    ad_node_t* leaf = var_node(X, sh, 1);
    leaf->tensor_gradient = ad_doubles(3);
    eshkol_tensor_backward_dispatch(leaf);
}
static void check_leaf_still_silent(void) {
    report("dispatch.leaf_variable_is_not_fatal",
           !child_refuses(leaf_variable_child),
           "input variables carry tensor_value and must remain a legitimate no-op");
}


/* ================================================ boundary behaviour ===== */
/* SW-76. ad_poincare_log_map used to clamp artanh's argument to 1 - 1e-12 and
 * return a finite tangent vector where the map has none; ad_poincare_exp_map
 * never checked its base point at all, so an out-of-ball x produced a NEGATIVE
 * conformal factor and ran the map backwards; and ad_geodesic_attention scored
 * an off-manifold head-slice as HUGE_VAL, silently dropping that key from the
 * softmax. All three returned a full, finite, plausible answer with no
 * diagnostic, and the reverse sweep then differentiated it.
 *
 * These checks pin the two halves of the fix. STRICTLY INSIDE still computes,
 * and computes exactly -- including a hair from the boundary, where the whole
 * point is that a real value exists and must not be traded for a fabricated
 * one. ON OR OUTSIDE refuses, by returning NULL after a diagnostic naming the
 * op and the measured sqrt(c)|.|; these ops signal with eshkol_error(), which
 * logs and returns, so a refusal is a NULL node, not a process exit (that is
 * why these are not fork()ed like the eshkol_fatal refusals above). */

/* Direction is deliberately not axis-aligned so every component is live in the
 * near-boundary gradient checks. */
static void unit_dir(double* d) { d[0] = 0.6; d[1] = -0.8; d[2] = 0.0; }
static void at_radius(double r, double* out) {
    double d[3]; unit_dir(d);
    for (int i = 0; i < 3; ++i) out[i] = r * d[i];
}

/** @brief A point at radius r on an AXIS, used wherever the test's meaning
 *  depends on |x| being exactly r.
 *
 *  0.6^2 + 0.8^2 happens to round to exactly 1.0 in f64, but that is a fact
 *  about one evaluation order: norm_sq() is a `+=` loop, and a compiler free to
 *  contract it into an FMA may land a half-ulp below 1. A test whose subject is
 *  "sqrt(c)|x| == 1 exactly refuses" must not be able to drift to |x| = 1 - eps
 *  and quietly start asserting the interior case instead. r^2 + 0 + 0 is exact
 *  under every evaluation order, FMA or not. */
static void at_radius_exact(double r, double* out) {
    out[0] = r; out[1] = 0.0; out[2] = 0.0;
}

/** @brief |grad_x d| == lambda_x still holds exactly at |x| = 1 - 1e-6, where
 *  lambda_x is about 1e6. The identity is a property of the geometry, so it is
 *  the strong reference here; the finite-difference check below is the weak
 *  one. */
static void check_distance_near_boundary_identity(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3], Y[3] = { -0.20, 0.25, 0.10 };
    at_radius(1.0 - 1e-6, X);

    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = var_node(X, sh, 1), *yn = var_node(Y, sh, 1);
    ad_node_t* o = ad_hyperbolic_distance(t, xn, yn, -c);
    if (!o) {
        report("boundary.distance_near_boundary_identity", false,
               "forward refused at |x| = 1 - 1e-6, which is INSIDE the ball");
        return;
    }
    ((double*)o->tensor_gradient)[0] = 1.0;
    sweep(t);
    const double* gx = (const double*)xn->tensor_gradient;
    double nx = 0.0, xx = 0.0;
    for (int i = 0; i < 3; ++i) { nx += gx[i]*gx[i]; xx += X[i]*X[i]; }
    double lx = 2.0/(1.0 - c*xx);
    double rel = std::fabs(std::sqrt(nx) - lx)/lx;
    report("boundary.distance_near_boundary_identity", rel < 1e-9,
           "|x| = 1-1e-6, lambda_x = %.6e, rel dev of |grad_x| = %.3e", lx, rel);
}

/** @brief The same point against central differences. The step is scaled to the
 *  distance to the boundary, h = 1e-4 * (1 - c|x|^2), rather than fixed. The
 *  file's usual STEP of 1e-6 is HALF the entire remaining gap to the boundary
 *  here (1 - c|x|^2 = 2e-6), so the shifted points sit in a completely
 *  different part of the geometry: measured, a 1e-6 step gives a finite
 *  difference 30% off the true derivative. Scaled steps, measured against the
 *  exact lambda_x: 1e-2*gap -> 7.2e-5, 1e-3*gap -> 7.4e-7, 1e-4*gap -> 7.9e-9.
 *  1e-4 is chosen for two orders of margin under the 1e-6 bar; going further
 *  would start trading truncation error for cancellation. */
static void check_distance_near_boundary_fd(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3], Y[3] = { -0.20, 0.25, 0.10 };
    at_radius(1.0 - 1e-6, X);
    double gap = 1.0 - c*(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
    double h = 1e-4 * gap;

    ad_tape_t* t = arena_allocate_tape(get_global_arena(), 8);
    ad_node_t* xn = var_node(X, sh, 1), *yn = var_node(Y, sh, 1);
    ad_node_t* o = ad_hyperbolic_distance(t, xn, yn, -c);
    if (!o) { report("boundary.distance_near_boundary_fd", false, "forward refused"); return; }
    ((double*)o->tensor_gradient)[0] = 1.0; sweep(t);
    double gx[3]; std::memcpy(gx, xn->tensor_gradient, sizeof gx);

    auto fwd = [&](const double* a) {
        ad_node_t* an = var_node(a, sh, 1), *bn = var_node(Y, sh, 1);
        ad_node_t* r = ad_hyperbolic_distance(nullptr, an, bn, -c);
        return r ? ((const double*)r->tensor_value)[0] : NAN;
    };
    RelAcc acc;
    for (int i = 0; i < 3; ++i) {
        double pl[3], mi[3];
        std::memcpy(pl, X, sizeof pl); std::memcpy(mi, X, sizeof mi);
        pl[i] += h; mi[i] -= h;
        acc_add(&acc, (fwd(pl)-fwd(mi))/(2*h), gx[i]);
    }
    double r = acc_val(&acc);
    report("boundary.distance_near_boundary_fd", r < 1e-6,
           "|x| = 1-1e-6, h = %.3e, aggregate L2 rel err %.3e over 3 partials", h, r);
}

/** @brief exp and log remain mutually inverse with the BASE POINT at
 *  |x| = 1 - 1e-6, where the conformal factor lambda_x is about 1e6 and the
 *  two rules' Jacobian entries span twelve orders of magnitude. Nothing is
 *  clamped, so J_log * J_exp must still be I. */
static void check_log_map_near_boundary(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3], V[3] = { 1.0e-7, -6.0e-8, 4.0e-8 };
    at_radius(1.0 - 1e-6, X);

    ad_node_t* xn = var_node(X, sh, 1), *vn = var_node(V, sh, 1);
    ad_node_t* e = ad_poincare_exp_map(nullptr, xn, vn, -c);
    if (!e) {
        report("boundary.exp_log_inverse_near_boundary", false,
               "exp forward refused at |x| = 1 - 1e-6, which is INSIDE the ball");
        return;
    }
    double Yv[3]; std::memcpy(Yv, e->tensor_value, sizeof Yv);

    double Je[9], Jl[9];
    exp_jacobian_wrt_v(X, V, 3, c, Je);
    log_jacobian_wrt_y(X, Yv, 3, c, Jl);
    double worst = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int m = 0; m < 3; ++m) acc += Jl[(size_t)i*3+m] * Je[(size_t)m*3+j];
            worst = std::fmax(worst, std::fabs(acc - (i==j ? 1.0 : 0.0)));
        }
    report("boundary.exp_log_inverse_near_boundary", worst < 1e-6,
           "|x| = 1-1e-6, max |J_log * J_exp - I| = %.3e", worst);
}

/** @brief The artanh argument itself driven to a hair below 1, with both points
 *  valid. sqrt(c)|u| approaches 1 when the two points are far apart in
 *  HYPERBOLIC distance, not when either is near the boundary; the cleanest way
 *  to arrange it is x at |x| = 1 - 1e-6 and y at the ORIGIN, which gives
 *  u = (-x) (+)_c 0 = -x and therefore t = sqrt(c)|x| = 1 - 1e-6 exactly. That
 *  is four orders of magnitude closer to 1 than the old clamp's 1 - 1e-12
 *  substitute would have to be to matter, and a real value exists.
 *
 *  Checked against |log_x(y)| == d(x,y): the length of the log map IS the
 *  geodesic distance, by definition of the exponential map. That is a fact
 *  about the manifold, not about either implementation, and it pins the
 *  artanh magnitude the clamp used to fabricate. */
static void check_log_map_artanh_near_one(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double X[3], O[3] = { 0.0, 0.0, 0.0 };
    at_radius(1.0 - 1e-6, X);

    ad_node_t* xn = var_node(X, sh, 1), *on = var_node(O, sh, 1);
    ad_node_t* l = ad_poincare_log_map(nullptr, xn, on, -c);
    if (!l) {
        report("boundary.log_map_artanh_near_one", false,
               "refused at t = sqrt(c)|u| = 1 - 1e-6, which is INSIDE the domain");
        return;
    }
    const double* L = (const double*)l->tensor_value;
    double lm = std::sqrt(L[0]*L[0] + L[1]*L[1] + L[2]*L[2]);

    ad_node_t* x2 = var_node(X, sh, 1), *o2 = var_node(O, sh, 1);
    ad_node_t* d = ad_hyperbolic_distance(nullptr, x2, o2, -c);
    if (!d) { report("boundary.log_map_artanh_near_one", false, "distance refused"); return; }
    double dv = ((const double*)d->tensor_value)[0];
    double rel = std::fabs(lm - dv)/dv;
    report("boundary.log_map_artanh_near_one", rel < 1e-9,
           "t = 1-1e-6: |log_x(0)| = %.12f, d(x,0) = %.12f, rel dev = %.3e "
           "(the clamp would have returned artanh(1-1e-12) ~ 14.2 here)",
           lm, dv, rel);
}

/** @brief Two points each STRICTLY inside the ball, far enough apart in
 *  hyperbolic distance that sqrt(c)|(-x) (+)_c y| rounds to exactly 1.0 in f64
 *  (~43 units apart here). This is the case the clamp existed for, and the one
 *  it fabricated: artanh(1 - 1e-12) is about 14.2, a specific finite magnitude
 *  no caller could tell from a real one. There is no finite log here, so the
 *  op must refuse -- note that BOTH inputs pass the in-ball test, so this
 *  refusal cannot come from the membership check. */
static void check_log_map_refuses_no_finite_log(void) {
    const int64_t sh[1] = { 3 };
    const double r = 1.0 - 1e-9;
    double X[3], Y[3];
    at_radius(r, X);
    at_radius(-r, Y);
    ad_node_t* xn = var_node(X, sh, 1), *yn = var_node(Y, sh, 1);
    ad_node_t* o = ad_poincare_log_map(nullptr, xn, yn, -1.0);
    report("boundary.log_map_refuses_when_no_finite_log", o == nullptr,
           "|x| = |y| = 1-1e-12 antipodal (both inside): sqrt(c)|u| reaches 1, "
           "op returned %s", o ? "a fabricated tangent vector" : "NULL");
}

/** @brief On the boundary and outside it, every geometric op refuses. |x| == 1
 *  exactly is the boundary itself: no tangent space, no finite log, no
 *  derivative. */
static void check_ops_refuse_on_and_outside_boundary(void) {
    const int64_t sh[1] = { 3 };
    const double c = 1.0;
    double Yin[3] = { -0.20, 0.25, 0.10 };
    double Vt[3]  = { 0.01, 0.02, -0.03 };

    struct { const char* label; double r; } radii[] = {
        { "on_boundary",  1.0 },   /* sqrt(c)|x| == 1 exactly */
        { "outside",      1.5 },
    };
    for (auto& rc : radii) {
        /* Exact-radius spelling: the whole point of the r = 1 row is that the
         * norm is 1 and not 1 - half an ulp. */
        double X[3]; at_radius_exact(rc.r, X);
        ad_node_t* xn = var_node(X, sh, 1);
        ad_node_t* yn = var_node(Yin, sh, 1);
        ad_node_t* vn = var_node(Vt, sh, 1);

        char nm[128];
        std::snprintf(nm, sizeof nm, "boundary.distance_refuses_%s", rc.label);
        ad_node_t* d = ad_hyperbolic_distance(nullptr, xn, yn, -c);
        report(nm, d == nullptr, "|x| = %.1f, got %s", rc.r, d ? "a value" : "NULL");

        std::snprintf(nm, sizeof nm, "boundary.exp_map_refuses_%s", rc.label);
        ad_node_t* e = ad_poincare_exp_map(nullptr, xn, vn, -c);
        report(nm, e == nullptr, "|x| = %.1f, got %s", rc.r, e ? "a value" : "NULL");

        std::snprintf(nm, sizeof nm, "boundary.log_map_refuses_%s", rc.label);
        ad_node_t* l = ad_poincare_log_map(nullptr, xn, yn, -c);
        report(nm, l == nullptr, "|x| = %.1f, got %s", rc.r, l ? "a value" : "NULL");

        /* Second argument off-manifold too, not just the first. */
        std::snprintf(nm, sizeof nm, "boundary.log_map_refuses_y_%s", rc.label);
        ad_node_t* yb = var_node(X, sh, 1);
        ad_node_t* xi = var_node(Yin, sh, 1);
        ad_node_t* l2 = ad_poincare_log_map(nullptr, xi, yb, -c);
        report(nm, l2 == nullptr, "|y| = %.1f, got %s", rc.r, l2 ? "a value" : "NULL");
    }
}

/** @brief A NaN coordinate must refuse rather than propagate onto the tape.
 *  This is what the strict `sn < 1.0` in poincare_in_ball() buys: the
 *  comparison is false for NaN, so NaN takes the refusal branch. Written as its
 *  own check because a `!(sn >= 1.0)` spelling would pass every other test in
 *  this file and silently admit NaN. */
static void check_ops_refuse_nan_coordinate(void) {
    const int64_t sh[1] = { 3 };
    double Xn[3] = { 0.2, NAN, 0.1 };
    double Y[3]  = { -0.20, 0.25, 0.10 };
    ad_node_t* xn = var_node(Xn, sh, 1), *yn = var_node(Y, sh, 1);
    report("boundary.distance_refuses_nan",
           ad_hyperbolic_distance(nullptr, xn, yn, -1.0) == nullptr,
           "NaN coordinate in x takes the refusal branch");
    report("boundary.log_map_refuses_nan",
           ad_poincare_log_map(nullptr, xn, yn, -1.0) == nullptr,
           "NaN coordinate in x takes the refusal branch");
    report("boundary.exp_map_refuses_nan",
           ad_poincare_exp_map(nullptr, xn, yn, -1.0) == nullptr,
           "NaN coordinate in x takes the refusal branch");
}

/** @brief Geodesic attention refuses an off-manifold head-slice instead of
 *  scoring it HUGE_VAL and dropping it from the softmax. The slice is placed at
 *  a non-zero head and a non-zero position so the pre-pass has to find it
 *  rather than trip over the first row. */
static void check_geodesic_refuses_off_manifold_slice(void) {
    const int64_t sh[3] = { 1, 2, 4 };   /* batch 1, seq 2, dim 4, 2 heads */
    const int heads = 2, head_dim = 2;
    double Q[8], K[8], V[8];
    for (int i = 0; i < 8; ++i) { Q[i] = 0.10; K[i] = -0.05; V[i] = 0.03 * i; }

    /* Sanity: the all-interior version must still compute. */
    {
        ad_node_t* q = var_node(Q, sh, 3), *k = var_node(K, sh, 3), *v = var_node(V, sh, 3);
        ad_node_t* o = ad_geodesic_attention(nullptr, q, k, v, heads, -1.0, false);
        report("boundary.geodesic_interior_still_computes", o != nullptr,
               "all Q/K head-slices strictly inside the ball");
    }

    /* Slice offset is (b*seq + pos)*dim + head*head_dim, the same arithmetic the
     * op uses. Position 1, head 1 => (0*2+1)*4 + 1*2 = 6. */
    const size_t k_bad = (size_t)(0*2 + 1)*4 + (size_t)1*head_dim;
    double Kbad[8]; std::memcpy(Kbad, K, sizeof Kbad);
    Kbad[k_bad] = 0.9; Kbad[k_bad + 1] = 0.9;   /* |slice| = 1.2728 > 1 */
    {
        ad_node_t* q = var_node(Q, sh, 3), *k = var_node(Kbad, sh, 3), *v = var_node(V, sh, 3);
        ad_node_t* o = ad_geodesic_attention(nullptr, q, k, v, heads, -1.0, false);
        report("boundary.geodesic_refuses_off_manifold_key", o == nullptr,
               "K slice at position 1, head 1 has |k| = 1.2728; got %s",
               o ? "an attention output with that key silently dropped" : "NULL");
    }
    /* And a query slice exactly ON the boundary: position 0, head 1 =>
     * (0*2+0)*4 + 1*2 = 2. |slice| == 1 exactly, in any evaluation order. */
    const size_t q_bad = (size_t)(0*2 + 0)*4 + (size_t)1*head_dim;
    double Qbad[8]; std::memcpy(Qbad, Q, sizeof Qbad);
    Qbad[q_bad] = 1.0; Qbad[q_bad + 1] = 0.0;
    {
        ad_node_t* q = var_node(Qbad, sh, 3), *k = var_node(K, sh, 3), *v = var_node(V, sh, 3);
        ad_node_t* o = ad_geodesic_attention(nullptr, q, k, v, heads, -1.0, false);
        report("boundary.geodesic_refuses_on_boundary_query", o == nullptr,
               "Q slice at position 0, head 1 has |q| = 1 exactly; got %s",
               o ? "an attention output" : "NULL");
    }
}

int main(void) {
    check_distance_conformal_identity();     /* 1 */
    check_distance_fd();                     /* 1 */
    check_distance_refuses_coincident();     /* 1 */
    check_log_golden();                      /* 1 */
    check_exp_golden();                      /* 1 */
    check_exp_log_inverse_jacobians();       /* 1 */
    check_exp_log_fd();                      /* 2 */
    check_geodesic_fd();                     /* 1 */
    check_geodesic_causal_fd();              /* 1 */
    check_geodesic_refuses_coincident();     /* 1 */
    check_loud_default();                    /* 1 */
    check_leaf_still_silent();               /* 1 */
    /* SW-76 boundary behaviour: inside computes exactly, on/outside refuses. */
    check_distance_near_boundary_identity(); /* 1 */
    check_distance_near_boundary_fd();       /* 1 */
    check_log_map_near_boundary();           /* 1 */
    check_log_map_artanh_near_one();         /* 1 */
    check_log_map_refuses_no_finite_log();   /* 1 */
    check_ops_refuse_on_and_outside_boundary(); /* 8 */
    check_ops_refuse_nan_coordinate();       /* 3 */
    check_geodesic_refuses_off_manifold_slice(); /* 3 */
    std::printf("Results: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
