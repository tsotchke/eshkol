/**
 * @file qllm_backward_gradcheck_test.c
 * @brief Numerical vs. analytical gradient check for the SDNC weight-matrix
 *        backward pass (`lib/backend/qllm_backward.c`).
 *
 * Companion test for the paper artefact *"The Self-Differentiating Neural
 * Computer: Computable Transformers via Analytical Weight Construction"*
 * (tsotchke, 2026). It certifies that the reverse-mode gradients emitted by
 * `qllm_backward_ffn_square` and `qllm_backward_ffn_gated` — the two FFN
 * layer types the computable transformer weight-implements — agree with a
 * central finite-difference reference to relative error < 1e-6.
 *
 * Why double. A float32 central-difference gradient check bottoms out near
 * 1e-3 relative error (the finite-difference rounding floor at single
 * precision), which is too loose to catch a subtle backward bug. The backward
 * math in qllm_backward.c is precision-generic (`qllm_real`); this test builds
 * that same source with `-DQLLM_REAL=double` so the finite-difference floor
 * drops to ~1e-11 and the analytical gradient is validated to ~1e-8, well
 * below the 1e-6 bar. The production/QLMW instantiation remains float.
 *
 * Metric. Aggregate L2 relative error over a sampled set of parameter and
 * input coordinates:  ||num - ana||_2 / (||num||_2 + ||ana||_2). This is the
 * standard robust gradient-check statistic; per-element ratios are avoided
 * because they blow up on coordinates whose analytical gradient is ~0.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "eshkol/backend/qllm_backward.h"

/* This test only makes sense in double precision (see file header). */
#if (QLLM_REAL - 0) == 0
/* QLLM_REAL is defined to a real type; nothing to assert at preprocess time. */
#endif

#define STEP        1e-3   /* central-difference step (double regime) */
#define TOLERANCE   1e-6   /* documented gradient-check bar */
#define N_W_SAMPLES 40     /* sampled weight coordinates per layer */
#define N_X_SAMPLES 20     /* sampled input coordinates per layer */

static unsigned long g_rng = 0x9e3779b97f4a7c15UL;
static double urand(void) { /* deterministic uniform in [-1, 1] */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((double)(g_rng >> 11) / (double)(1UL << 53)) * 2.0 - 1.0;
}
static int irand(int n) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (int)((g_rng >> 11) % (unsigned)n);
}

/* -------- forward passes (double), matching qllm_backward.c semantics ----- */

/* SQUARE FFN: h = x@Wu + bu ; a = h^2 ; fo = a@Wd + bd */
static void fwd_square(const qllm_real *wu, const qllm_real *wd,
                       const qllm_real *bu, const qllm_real *bd,
                       const qllm_real *x, qllm_real *h_pre, qllm_real *fo) {
    for (int j = 0; j < FFN_DIM; j++) {
        qllm_real u = bu[j];
        for (int i = 0; i < D; i++) u += x[i] * wu[i * FFN_DIM + j];
        h_pre[j] = u;
    }
    for (int i = 0; i < D; i++) {
        qllm_real s = bd[i];
        for (int j = 0; j < FFN_DIM; j++) s += h_pre[j] * h_pre[j] * wd[j * D + i];
        fo[i] = s;
    }
}

/* Gated FFN: gate = sigmoid(x@Wg+bg) ; up = x@Wu+bu ; h = gate*up ; fo = h@Wd+bd */
static void fwd_gated(const qllm_real *wu, const qllm_real *wd, const qllm_real *wg,
                      const qllm_real *bu, const qllm_real *bd, const qllm_real *bg,
                      const qllm_real *x, qllm_real *gate_post, qllm_real *up_post,
                      qllm_real *h, qllm_real *fo) {
    for (int j = 0; j < FFN_DIM; j++) {
        qllm_real g = bg[j], u = bu[j];
        for (int i = 0; i < D; i++) {
            g += x[i] * wg[i * FFN_DIM + j];
            u += x[i] * wu[i * FFN_DIM + j];
        }
        gate_post[j] = (qllm_real)(1.0 / (1.0 + exp(-(double)g)));
        up_post[j]   = u;
        h[j]         = gate_post[j] * u;
    }
    for (int i = 0; i < D; i++) {
        qllm_real s = bd[i];
        for (int j = 0; j < FFN_DIM; j++) s += h[j] * wd[j * D + i];
        fo[i] = s;
    }
}

/* Scalar loss L = 0.5 * sum((fo - tgt)^2); dL/dfo = fo - tgt. */
static double loss_from_output(const qllm_real *fo, const qllm_real *tgt,
                               qllm_real *dfo) {
    double L = 0.0;
    for (int i = 0; i < D; i++) {
        qllm_real d = fo[i] - tgt[i];
        if (dfo) dfo[i] = d;
        L += 0.5 * (double)d * (double)d;
    }
    return L;
}

/* Aggregate L2 relative error accumulator. */
typedef struct { double sd, sa, sn; } RelAcc;
static void relacc_add(RelAcc *a, double num, double ana) {
    double d = num - ana;
    a->sd += d * d; a->sa += ana * ana; a->sn += num * num;
}
static double relacc_value(const RelAcc *a) {
    return sqrt(a->sd) / (sqrt(a->sa) + sqrt(a->sn) + 1e-300);
}

/* ------------------------------ SQUARE FFN -------------------------------- */

static double check_square(void) {
    static qllm_real wu[D * FFN_DIM], wd[FFN_DIM * D], bu[FFN_DIM], bd[D];
    static qllm_real x[D], tgt[D], h_pre[FFN_DIM], fo[D], dfo[D], dx[D];
    static qllm_real dwu[D * FFN_DIM], dwd[FFN_DIM * D], dbu[FFN_DIM], dbd[D];

    for (int i = 0; i < D * FFN_DIM; i++) wu[i] = urand() * 0.05;
    for (int i = 0; i < FFN_DIM * D; i++) wd[i] = urand() * 0.05;
    for (int j = 0; j < FFN_DIM; j++) bu[j] = urand() * 0.01;
    for (int i = 0; i < D; i++) { bd[i] = urand() * 0.01; x[i] = urand() * 0.3; tgt[i] = urand(); }

    fwd_square(wu, wd, bu, bd, x, h_pre, fo);
    (void)loss_from_output(fo, tgt, dfo);
    qllm_backward_ffn_square(wu, wd, bu, bd, x, h_pre, dfo, dx, dwu, dbu, dwd, dbd);

    RelAcc acc = {0, 0, 0};
    for (int s = 0; s < N_W_SAMPLES; s++) {
        int idx = irand(D * FFN_DIM); qllm_real sv = wu[idx];
        wu[idx] = sv + (qllm_real)STEP; fwd_square(wu, wd, bu, bd, x, h_pre, fo);
        double Lp = loss_from_output(fo, tgt, NULL);
        wu[idx] = sv - (qllm_real)STEP; fwd_square(wu, wd, bu, bd, x, h_pre, fo);
        double Lm = loss_from_output(fo, tgt, NULL);
        wu[idx] = sv;
        relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), (double)dwu[idx]);
    }
    for (int s = 0; s < N_X_SAMPLES; s++) {
        int idx = irand(D); qllm_real sv = x[idx];
        x[idx] = sv + (qllm_real)STEP; fwd_square(wu, wd, bu, bd, x, h_pre, fo);
        double Lp = loss_from_output(fo, tgt, NULL);
        x[idx] = sv - (qllm_real)STEP; fwd_square(wu, wd, bu, bd, x, h_pre, fo);
        double Lm = loss_from_output(fo, tgt, NULL);
        x[idx] = sv;
        relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), (double)dx[idx]);
    }
    return relacc_value(&acc);
}

/* ------------------------------- gated FFN -------------------------------- */

static double check_gated(void) {
    static qllm_real wu[D * FFN_DIM], wd[FFN_DIM * D], wg[D * FFN_DIM];
    static qllm_real bu[FFN_DIM], bd[D], bg[FFN_DIM];
    static qllm_real x[D], tgt[D];
    static qllm_real gate_post[FFN_DIM], up_post[FFN_DIM], h[FFN_DIM], fo[D], dfo[D], dx[D];
    static qllm_real dwu[D * FFN_DIM], dwd[FFN_DIM * D], dwg[D * FFN_DIM];
    static qllm_real dbu[FFN_DIM], dbd[D], dbg[FFN_DIM];

    for (int i = 0; i < D * FFN_DIM; i++) { wu[i] = urand() * 0.05; wg[i] = urand() * 0.05; }
    for (int i = 0; i < FFN_DIM * D; i++) wd[i] = urand() * 0.05;
    for (int j = 0; j < FFN_DIM; j++) { bu[j] = urand() * 0.01; bg[j] = urand() * 0.01; }
    for (int i = 0; i < D; i++) { bd[i] = urand() * 0.01; x[i] = urand(); tgt[i] = urand(); }

    fwd_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h, fo);
    (void)loss_from_output(fo, tgt, dfo);
    qllm_backward_ffn_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h,
                            dfo, dx, dwu, dbu, dwd, dbd, dwg, dbg);

    RelAcc acc = {0, 0, 0};
    /* sample W_up, W_gate, W_down and the input */
    const qllm_real *wsrc[3] = { wu, wg, wd };
    qllm_real *wgrad[3] = { dwu, dwg, dwd };
    int wlen[3] = { D * FFN_DIM, D * FFN_DIM, FFN_DIM * D };
    for (int b = 0; b < 3; b++) {
        qllm_real *w = (qllm_real *)wsrc[b];
        for (int s = 0; s < N_W_SAMPLES; s++) {
            int idx = irand(wlen[b]); qllm_real sv = w[idx];
            w[idx] = sv + (qllm_real)STEP;
            fwd_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h, fo);
            double Lp = loss_from_output(fo, tgt, NULL);
            w[idx] = sv - (qllm_real)STEP;
            fwd_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h, fo);
            double Lm = loss_from_output(fo, tgt, NULL);
            w[idx] = sv;
            relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), (double)wgrad[b][idx]);
        }
    }
    for (int s = 0; s < N_X_SAMPLES; s++) {
        int idx = irand(D); qllm_real sv = x[idx];
        x[idx] = sv + (qllm_real)STEP;
        fwd_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h, fo);
        double Lp = loss_from_output(fo, tgt, NULL);
        x[idx] = sv - (qllm_real)STEP;
        fwd_gated(wu, wd, wg, bu, bd, bg, x, gate_post, up_post, h, fo);
        double Lm = loss_from_output(fo, tgt, NULL);
        x[idx] = sv;
        relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), (double)dx[idx]);
    }
    return relacc_value(&acc);
}

int main(void) {
    printf("=== SDNC qllm_backward gradient check (QLLM_REAL=%s) ===\n",
           sizeof(qllm_real) == sizeof(double) ? "double" : "float");

    double sq = check_square();
    double ga = check_gated();

    int ok_sq = (sq < TOLERANCE);
    int ok_ga = (ga < TOLERANCE);

    printf("  SQUARE FFN backward: L2 rel err = %.3e  (tol %.0e)  %s\n",
           sq, (double)TOLERANCE, ok_sq ? "PASS" : "FAIL");
    printf("  gated  FFN backward: L2 rel err = %.3e  (tol %.0e)  %s\n",
           ga, (double)TOLERANCE, ok_ga ? "PASS" : "FAIL");

    int passed = ok_sq + ok_ga, failed = (!ok_sq) + (!ok_ga);
    printf("=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
