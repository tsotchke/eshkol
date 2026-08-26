/**
 * @file qllm_bridge_producer_gradcheck_test.cpp
 * @brief Gradient checks for the two AD-node producers added for the
 *        external-tensor bridge: ad_tensor_embedding and ad_frechet_mean.
 *
 * WHY A SEPARATE FILE FROM qllm_bridge_gradcheck_test.cpp. That file's harness
 * is a single-scalar-loss pipeline (x/w -> op -> cross-entropy -> scalar).
 * Neither op here fits it: embedding takes an integer-valued index operand that
 * carries no gradient, and the Fréchet mean is a vector-valued map whose full
 * Jacobian needs one reverse sweep per output component.
 *
 * WHY THESE CHECKS RUN THROUGH THE PRODUCERS AND NOT HAND-BUILT NODES. Before
 * this change, tensor_embedding_backward and tensor_frechet_mean_backward were
 * gradchecked only against `ad_node_t`s assembled by hand in the test
 * (tests/backend/tensor_embedding_backward_gradcheck_test.cpp,
 * tests/backend/frechet_mean_backward_gradcheck_test.cpp). Those tests prove
 * the rules are right; they cannot prove any producer fills the contract the
 * rules read, because there was no producer. A hand-built fixture agrees with
 * the backward by construction — it is written from the same contract — so the
 * one defect class it structurally cannot catch is a producer that fills the
 * contract wrongly. Everything below records its node through the real
 * forward entry point and lets the real dispatch find it.
 *
 * WHAT EACH GRADIENT IS CHECKED AGAINST. Every case carries an EXACT analytic
 * reference in addition to (or instead of) finite differences, because this is
 * gradient code and FD alone cannot distinguish "right" from "close":
 *
 *   - embedding: the adjoint of a gather IS a scatter-add, so the analytic
 *     reference is exact and the comparison is bit-for-bit, not a tolerance.
 *   - Fréchet mean at K = 0: the mean is the weighted average, so both
 *     Jacobians are closed forms — d mu_k / d x_{i,l} = (w_i / W) delta_kl and
 *     d mu_k / d w_i = (x_{i,k} - mu_k) / W. Exact, and it exercises the
 *     Euclidean branch of the implicit rule with no iteration involved.
 *   - Fréchet mean at coincident points (K < 0): every x_i equal to x forces
 *     mu* = x. Near coincidence log_mu(x_i) is the identity to first order
 *     (at the origin log_0(y) = artanh(sqrt(c)|y|) y / (sqrt(c)|y|) = y +
 *     O(|y|^3), and the Fréchet mean is equivariant under the Möbius isometry
 *     that carries mu to the origin), so the stationarity condition linearises
 *     to sum_i w_i (x_i - mu) = 0 and d mu / d x_i = (w_i / W) I exactly. This
 *     is an analytic case IN the hyperbolic regime, where the implicit
 *     machinery, the Möbius Jacobians and the LU solve are all live.
 *   - Fréchet mean, general hyperbolic configuration: central finite
 *     differences of the same forward, aggregate L2 relative error. This is the
 *     only case with no closed form, and it is deliberately the last line of
 *     defence rather than the first.
 *
 * FD METHOD. Aggregate L2 relative error ||num - ana|| / (||num|| + ||ana||)
 * over all sampled coordinates, the same measure and the same 1e-6 bar as
 * tests/bridge/qllm_bridge_gradcheck_test.cpp. Per-element ratios are avoided
 * because they blow up wherever the analytic gradient is ~0 — and the embedding
 * gradient is exactly 0 on most of its rows by construction.
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

#define TOLERANCE 1e-6   /* same bar as the other bridge gradcheck */
#define STEP      1e-5   /* central-difference step (double regime) */

static int g_passed = 0, g_failed = 0;

static void report(const char* name, bool ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("%s: %s — ", ok ? "PASS" : "FAIL", name);
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    if (ok) ++g_passed; else ++g_failed;
}

/* ---------------------------------------------------------------- helpers */

static double* arena_doubles(size_t n) {
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/** @brief Build an input-variable node holding a copy of @p data. */
static ad_node_t* var_node(const double* data, const int64_t* shape, size_t ndim) {
    ad_node_t* n = arena_allocate_ad_node(get_global_arena());
    size_t count = 1;
    for (size_t i = 0; i < ndim; ++i) count *= (size_t)shape[i];
    double* buf = arena_doubles(count);
    std::memcpy(buf, data, count * sizeof(double));
    int64_t* sh = (int64_t*)arena_allocate_zeroed(get_global_arena(), ndim * sizeof(int64_t));
    std::memcpy(sh, shape, ndim * sizeof(int64_t));
    n->type = AD_NODE_VARIABLE;
    n->tensor_value = buf;
    n->shape = sh;
    n->ndim = ndim;
    return n;
}

/** @brief Walk the tape in reverse evaluation order, as the runtime does. */
static void sweep(ad_tape_t* tape) {
    for (size_t i = tape->num_nodes; i-- > 0;) {
        eshkol_tensor_backward_dispatch(tape->nodes[i]);
    }
}

struct RelAcc { double dn = 0.0, nn = 0.0, an = 0.0; };
static void relacc_add(RelAcc* a, double num, double ana) {
    double d = num - ana;
    a->dn += d * d; a->nn += num * num; a->an += ana * ana;
}
static double relacc_value(const RelAcc* a) {
    double den = std::sqrt(a->nn) + std::sqrt(a->an);
    return (den > 0.0) ? std::sqrt(a->dn) / den : 0.0;
}

static unsigned long g_rng = 0x9e3779b97f4a7c15UL;
static double urand(void) { /* deterministic uniform in [-1, 1]; no RNG seeding */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((double)(g_rng >> 11) / (double)(1UL << 53)) * 2.0 - 1.0;
}

/* ===================================================== EMBEDDING ========= */

/* Vocabulary deliberately larger than the number of lookups, and the index
 * list deliberately repeats one row and skips several: the two properties that
 * make a gather's adjoint different from a dense one are that unreferenced rows
 * get EXACTLY zero and a row looked up k times gets the SUM of k upstream rows.
 * A fixture where every row is hit once cannot see either. */
enum { VOCAB = 5, DMODEL = 3, NIDX = 4 };
static const double kIdx[NIDX] = { 3.0, 1.0, 3.0, 0.0 };  /* row 3 twice; rows 2, 4 never */
static double g_W[VOCAB * DMODEL];
static double g_cot[NIDX * DMODEL];       /* upstream dL/dy */

static void init_embedding_fixture(void) {
    for (int i = 0; i < VOCAB * DMODEL; ++i) g_W[i] = urand();
    for (int i = 0; i < NIDX * DMODEL; ++i) g_cot[i] = urand();
}

/** @brief Run the producer + backward once; returns dL/dW (VOCAB*DMODEL). */
static const double* embedding_grad(const double* W) {
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    int64_t wshape[2] = { VOCAB, DMODEL };
    int64_t ishape[1] = { NIDX };
    ad_node_t* wn = var_node(W, wshape, 2);
    ad_node_t* in = var_node(kIdx, ishape, 1);
    ad_node_t* out = ad_tensor_embedding(tape, wn, in);
    if (!out) return nullptr;
    std::memcpy(out->tensor_gradient, g_cot, sizeof g_cot);
    sweep(tape);
    return (const double*)wn->tensor_gradient;
}

/** @brief Forward-only value of L = <cot, y(W)>. */
static double embedding_loss(const double* W) {
    int64_t wshape[2] = { VOCAB, DMODEL };
    int64_t ishape[1] = { NIDX };
    ad_node_t* wn = var_node(W, wshape, 2);
    ad_node_t* in = var_node(kIdx, ishape, 1);
    ad_node_t* out = ad_tensor_embedding(nullptr, wn, in);   /* forward only */
    if (!out) return NAN;
    const double* y = (const double*)out->tensor_value;
    double acc = 0.0;
    for (int i = 0; i < NIDX * DMODEL; ++i) acc += g_cot[i] * y[i];
    return acc;
}

static void check_embedding_analytic(void) {
    const double* dW = embedding_grad(g_W);
    if (!dW) { report("embedding.analytic", false, "producer returned NULL"); return; }

    /* The exact adjoint, written out independently of the rule under test. */
    double ref[VOCAB * DMODEL];
    std::memset(ref, 0, sizeof ref);
    for (int i = 0; i < NIDX; ++i) {
        int row = (int)kIdx[i];
        for (int d = 0; d < DMODEL; ++d) ref[row * DMODEL + d] += g_cot[i * DMODEL + d];
    }

    /* Bit-for-bit: a scatter-add of the same doubles in the same order has no
     * rounding freedom left, so a tolerance here would only hide a real
     * difference. */
    int mismatches = 0;
    for (int i = 0; i < VOCAB * DMODEL; ++i) if (dW[i] != ref[i]) ++mismatches;
    report("embedding.analytic_scatter_add", mismatches == 0,
           "exact scatter-add over %d weights, %d mismatch(es)",
           VOCAB * DMODEL, mismatches);

    /* Rows never looked up must be EXACTLY zero, not merely small: the adjoint
     * of a gather is genuinely sparse. */
    bool sparse_ok = true;
    for (int d = 0; d < DMODEL; ++d) {
        if (dW[2 * DMODEL + d] != 0.0) sparse_ok = false;
        if (dW[4 * DMODEL + d] != 0.0) sparse_ok = false;
    }
    report("embedding.unreferenced_rows_exactly_zero", sparse_ok,
           "rows 2 and 4 are never looked up");

    /* Row 3 is looked up twice; it must carry the SUM, not the last write. */
    bool accum_ok = true;
    for (int d = 0; d < DMODEL; ++d) {
        double want = g_cot[0 * DMODEL + d] + g_cot[2 * DMODEL + d];
        if (dW[3 * DMODEL + d] != want) accum_ok = false;
    }
    report("embedding.repeated_index_accumulates", accum_ok,
           "row 3 is looked up at positions 0 and 2");
}

static void check_embedding_fd(void) {
    const double* dW = embedding_grad(g_W);
    if (!dW) { report("embedding.fd", false, "producer returned NULL"); return; }
    double ana[VOCAB * DMODEL];
    std::memcpy(ana, dW, sizeof ana);

    RelAcc acc;
    for (int i = 0; i < VOCAB * DMODEL; ++i) {
        double saved = g_W[i];
        double Wp[VOCAB * DMODEL], Wm[VOCAB * DMODEL];
        std::memcpy(Wp, g_W, sizeof Wp); std::memcpy(Wm, g_W, sizeof Wm);
        Wp[i] = saved + STEP; Wm[i] = saved - STEP;
        double num = (embedding_loss(Wp) - embedding_loss(Wm)) / (2.0 * STEP);
        relacc_add(&acc, num, ana[i]);
    }
    double rel = relacc_value(&acc);
    report("embedding.finite_difference", rel < TOLERANCE,
           "aggregate L2 rel err %.3e over %d weights (bar %.0e)",
           rel, VOCAB * DMODEL, TOLERANCE);
}

static void check_embedding_refuses_bad_index(void) {
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 4);
    int64_t wshape[2] = { VOCAB, DMODEL };
    int64_t ishape[1] = { NIDX };
    double bad[NIDX] = { 0.0, 1.5, 2.0, 3.0 };     /* 1.5 is not an index */
    ad_node_t* wn = var_node(g_W, wshape, 2);
    ad_node_t* in = var_node(bad, ishape, 1);
    ad_node_t* out = ad_tensor_embedding(tape, wn, in);
    report("embedding.refuses_fractional_index", out == nullptr,
           "a fractional index is refused rather than rounded into a wrong row");
}

/* ================================================== FRECHET MEAN ========= */

enum { NPTS = 4, FDIM = 3 };

/** @brief One reverse sweep per output component: the full Jacobian.
 *  @param dmu_dx  out, NPTS*FDIM*FDIM: [k][i][l] = d mu_k / d x_{i,l}
 *  @param dmu_dw  out, FDIM*NPTS: [k][i] = d mu_k / d w_i (NULL to skip)
 *  @param mu_out  out, FDIM primal mean (NULL to skip)
 *  @return false if the producer refused. */
static bool frechet_jacobian(const double* pts, const double* wts, double K,
                             double* dmu_dx, double* dmu_dw, double* mu_out) {
    for (int k = 0; k < FDIM; ++k) {
        ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
        int64_t pshape[2] = { NPTS, FDIM };
        int64_t wshape[1] = { NPTS };
        ad_node_t* pn = var_node(pts, pshape, 2);
        ad_node_t* wn = wts ? var_node(wts, wshape, 1) : nullptr;
        ad_node_t* out = ad_frechet_mean(tape, pn, wn, K, 0.0);
        if (!out) return false;
        if (mu_out && k == 0)
            std::memcpy(mu_out, out->tensor_value, FDIM * sizeof(double));
        /* Seed e_k: the reverse sweep then yields row k of the Jacobian. */
        double* g = (double*)out->tensor_gradient;
        for (int j = 0; j < FDIM; ++j) g[j] = (j == k) ? 1.0 : 0.0;
        sweep(tape);
        const double* dx = (const double*)pn->tensor_gradient;
        for (int i = 0; i < NPTS * FDIM; ++i)
            dmu_dx[(size_t)k * NPTS * FDIM + i] = dx[i];
        if (dmu_dw && wn) {
            const double* dw = (const double*)wn->tensor_gradient;
            for (int i = 0; i < NPTS; ++i) dmu_dw[(size_t)k * NPTS + i] = dw[i];
        }
    }
    return true;
}

/** @brief Forward-only mean, for the finite-difference leg. */
static bool frechet_forward(const double* pts, const double* wts, double K, double* mu) {
    int64_t pshape[2] = { NPTS, FDIM };
    int64_t wshape[1] = { NPTS };
    ad_node_t* pn = var_node(pts, pshape, 2);
    ad_node_t* wn = wts ? var_node(wts, wshape, 1) : nullptr;
    ad_node_t* out = ad_frechet_mean(nullptr, pn, wn, K, 0.0);
    if (!out) return false;
    std::memcpy(mu, out->tensor_value, FDIM * sizeof(double));
    return true;
}

static const double kWts[NPTS] = { 0.7, 1.3, 0.4, 2.1 };

/* Asymmetric, strictly inside the unit ball, no coordinate decoupled. */
static const double kPts[NPTS * FDIM] = {
     0.30, -0.10,  0.20,
    -0.45,  0.25, -0.05,
     0.10,  0.50,  0.15,
    -0.20, -0.35,  0.40,
};

static void check_frechet_euclidean_closed_form(void) {
    std::vector<double> J((size_t)FDIM * NPTS * FDIM, 0.0);
    std::vector<double> Jw((size_t)FDIM * NPTS, 0.0);
    double mu[FDIM];
    if (!frechet_jacobian(kPts, kWts, 0.0, J.data(), Jw.data(), mu)) {
        report("frechet.euclidean_closed_form", false, "producer returned NULL at K=0");
        return;
    }
    double wsum = 0.0;
    for (int i = 0; i < NPTS; ++i) wsum += kWts[i];

    /* d mu_k / d x_{i,l} = (w_i / W) delta_kl  and  d mu_k / d w_i = (x_ik - mu_k)/W */
    RelAcc ax, aw;
    for (int k = 0; k < FDIM; ++k) {
        for (int i = 0; i < NPTS; ++i) {
            for (int l = 0; l < FDIM; ++l) {
                double want = (k == l) ? (kWts[i] / wsum) : 0.0;
                relacc_add(&ax, J[((size_t)k * NPTS + i) * FDIM + l], want);
            }
            double wantw = (kPts[i * FDIM + k] - mu[k]) / wsum;
            relacc_add(&aw, Jw[(size_t)k * NPTS + i], wantw);
        }
    }
    double rx = relacc_value(&ax), rw = relacc_value(&aw);
    report("frechet.euclidean_closed_form", rx < 1e-14 && rw < 1e-14,
           "d mu/d x rel %.3e, d mu/d w rel %.3e vs the exact weighted average",
           rx, rw);
}

static void check_frechet_coincident_analytic(void) {
    /* Every point at the same interior location: mu* = x, and the stationarity
     * condition linearises to the weighted average, so the Jacobian is exactly
     * (w_i / W) I even though the curvature machinery is fully live. */
    const double x[FDIM] = { 0.20, -0.15, 0.10 };
    double pts[NPTS * FDIM];
    for (int i = 0; i < NPTS; ++i)
        for (int l = 0; l < FDIM; ++l) pts[i * FDIM + l] = x[l];

    std::vector<double> J((size_t)FDIM * NPTS * FDIM, 0.0);
    double mu[FDIM];
    if (!frechet_jacobian(pts, kWts, -1.0, J.data(), nullptr, mu)) {
        report("frechet.coincident_points_analytic", false, "producer returned NULL");
        return;
    }
    double wsum = 0.0;
    for (int i = 0; i < NPTS; ++i) wsum += kWts[i];

    double mu_err = 0.0;
    for (int l = 0; l < FDIM; ++l) mu_err = std::fmax(mu_err, std::fabs(mu[l] - x[l]));

    RelAcc acc;
    for (int k = 0; k < FDIM; ++k)
        for (int i = 0; i < NPTS; ++i)
            for (int l = 0; l < FDIM; ++l) {
                double want = (k == l) ? (kWts[i] / wsum) : 0.0;
                relacc_add(&acc, J[((size_t)k * NPTS + i) * FDIM + l], want);
            }
    double rel = relacc_value(&acc);
    report("frechet.coincident_points_analytic", rel < 1e-9 && mu_err < 1e-12,
           "K=-1, |mu*-x| %.3e, Jacobian rel %.3e vs the exact (w_i/W) I",
           mu_err, rel);
}

static void check_frechet_hyperbolic_fd(void) {
    std::vector<double> J((size_t)FDIM * NPTS * FDIM, 0.0);
    std::vector<double> Jw((size_t)FDIM * NPTS, 0.0);
    if (!frechet_jacobian(kPts, kWts, -1.0, J.data(), Jw.data(), nullptr)) {
        report("frechet.hyperbolic_finite_difference", false, "producer returned NULL");
        return;
    }

    RelAcc acc;
    /* d mu / d x */
    for (int i = 0; i < NPTS * FDIM; ++i) {
        double pp[NPTS * FDIM], pm[NPTS * FDIM], mp[FDIM], mm[FDIM];
        std::memcpy(pp, kPts, sizeof pp); std::memcpy(pm, kPts, sizeof pm);
        pp[i] += STEP; pm[i] -= STEP;
        if (!frechet_forward(pp, kWts, -1.0, mp) || !frechet_forward(pm, kWts, -1.0, mm)) {
            report("frechet.hyperbolic_finite_difference", false, "FD forward refused");
            return;
        }
        for (int k = 0; k < FDIM; ++k) {
            double num = (mp[k] - mm[k]) / (2.0 * STEP);
            relacc_add(&acc, num, J[((size_t)k * NPTS * FDIM) + i]);
        }
    }
    /* d mu / d w */
    for (int i = 0; i < NPTS; ++i) {
        double wp[NPTS], wm[NPTS], mp[FDIM], mm[FDIM];
        std::memcpy(wp, kWts, sizeof wp); std::memcpy(wm, kWts, sizeof wm);
        wp[i] += STEP; wm[i] -= STEP;
        if (!frechet_forward(kPts, wp, -1.0, mp) || !frechet_forward(kPts, wm, -1.0, mm)) {
            report("frechet.hyperbolic_finite_difference", false, "FD forward refused");
            return;
        }
        for (int k = 0; k < FDIM; ++k) {
            double num = (mp[k] - mm[k]) / (2.0 * STEP);
            relacc_add(&acc, num, Jw[(size_t)k * NPTS + i]);
        }
    }
    double rel = relacc_value(&acc);
    report("frechet.hyperbolic_finite_difference", rel < TOLERANCE,
           "aggregate L2 rel err %.3e over %d partials (bar %.0e)",
           rel, NPTS * FDIM * FDIM + NPTS * FDIM, TOLERANCE);
}

static void check_frechet_refuses_outside_ball(void) {
    /* A point on the boundary has no finite log map, so no mean and no
     * derivative exist. The producer must refuse rather than record a node the
     * backward would refuse later, after the caller has used the value. */
    double pts[NPTS * FDIM];
    std::memcpy(pts, kPts, sizeof pts);
    pts[0] = 1.0; pts[1] = 0.0; pts[2] = 0.0;    /* exactly on the unit sphere */
    int64_t pshape[2] = { NPTS, FDIM };
    int64_t wshape[1] = { NPTS };
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 4);
    ad_node_t* pn = var_node(pts, pshape, 2);
    ad_node_t* wn = var_node(kWts, wshape, 1);
    ad_node_t* out = ad_frechet_mean(tape, pn, wn, -1.0, 0.0);
    report("frechet.refuses_point_on_boundary", out == nullptr,
           "a point on the ball boundary is refused at forward time");
}

/* ======================================================== main =========== */

int main(void) {
    init_embedding_fixture();

    check_embedding_analytic();            /* 3 checks */
    check_embedding_fd();                  /* 1 */
    check_embedding_refuses_bad_index();   /* 1 */

    check_frechet_euclidean_closed_form(); /* 1 */
    check_frechet_coincident_analytic();   /* 1 */
    check_frechet_hyperbolic_fd();         /* 1 */
    check_frechet_refuses_outside_ball();  /* 1 */

    std::printf("Results: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
