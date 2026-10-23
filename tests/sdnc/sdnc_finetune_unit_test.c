/*
 * Unit checks for the exact-forward control fine-tuning facility
 * (lib/core/sdnc_core.h backward smoothing, lib/core/sdnc_finetune.c).
 *
 *   1. the shared logistic / softmax VJP rule matches central differences of
 *      sigma(beta x) and softmax(beta z) for several beta;
 *   2. sdnc_backward is bit-identical to sdnc_backward_smoothed at beta = 1;
 *   3. on saturated gates the exact gate derivative is identically zero and the
 *      smoothed one is not;
 *   4. tied gate groups follow identical gate columns and biases;
 *   5. SDTR transitions and QLMW weights round-trip, malformed files refuse;
 *   6. on a two-neuron synthetic interpreter whose gate is switched off, the
 *      optimizer restores every transition with one tied-gate move.
 *
 * Usage: sdnc_finetune_unit_test <scratch-dir>   (files are written there)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sdnc_finetune.h"

#define D SDNC_D
#define F SDNC_FFN_DIM

static int g_pass = 0, g_fail = 0;
static void check(const char* what, int ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) g_pass++; else g_fail++;
}

static double sig(double x) { return 1.0 / (1.0 + exp(-x)); }

static void rule_checks(void) {
    const float betas[] = { 1.0f, 0.5f, 1.0f / 30.0f };
    int ok = 1;
    for (int b = 0; b < 3; b++) {
        for (float x = -6.0f; x <= 6.0f; x += 0.75f) {
            float beta = betas[b];
            float s = (float)sig((double)beta * x);
            double fd = (sig((double)beta * (x + 1e-4)) - sig((double)beta * (x - 1e-4))) / 2e-4;
            double got = eshkol_logistic_vjp_sdnc(1.0f, s, beta);
            if (fabs(got - fd) > 1e-5 + 1e-3 * fabs(fd)) ok = 0;
        }
    }
    check("logistic VJP equals d/dx sigma(beta x) (beta 1, 1/2, 1/30)", ok);

    ok = 1;
    const float z[5] = { 0.3f, -1.2f, 2.0f, 0.0f, 0.7f };
    const float da[5] = { 0.5f, -0.25f, 1.0f, 0.1f, -0.6f };
    for (int b = 0; b < 3; b++) {
        float beta = betas[b];
        double e[5], sum = 0; float a[5]; float dz[5];
        for (int p = 0; p < 5; p++) { e[p] = exp((double)beta * z[p]); sum += e[p]; }
        for (int p = 0; p < 5; p++) a[p] = (float)(e[p] / sum);
        eshkol_softmax_vjp_sdnc(a, da, 5, beta, dz);
        for (int q = 0; q < 5; q++) {
            double f[2];
            for (int sgn = 0; sgn < 2; sgn++) {
                double h = sgn ? -1e-4 : 1e-4, ee[5], ss = 0, dot = 0;
                for (int p = 0; p < 5; p++) { ee[p] = exp((double)beta * (z[p] + (p == q ? h : 0))); ss += ee[p]; }
                for (int p = 0; p < 5; p++) dot += ee[p] / ss * da[p];
                f[sgn] = dot;
            }
            double fd = (f[0] - f[1]) / 2e-4;
            if (fabs(dz[q] - fd) > 1e-5 + 1e-3 * fabs(fd)) ok = 0;
        }
    }
    check("softmax VJP equals d/dz <softmax(beta z), da> (beta 1, 1/2, 1/30)", ok);
}

static unsigned long g_rng = 0x1234567UL;
static float rnd(void) {
    g_rng = g_rng * 6364136223846793005UL + 1442695040888963407UL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0xffffff - 0.5f;
}

static void backward_checks(void) {
    SdncWeights* w = (SdncWeights*)malloc(sizeof(SdncWeights));
    SdncGrads* a = (SdncGrads*)calloc(1, sizeof(SdncGrads));
    SdncGrads* b = (SdncGrads*)calloc(1, sizeof(SdncGrads));
    SdncFwdCache* c = (SdncFwdCache*)malloc(sizeof(SdncFwdCache));
    static float pe[4][D];
    float st[D], dn[D], da[D], db[D], nx1[D], nx2[D];
    sdnc_init_weights(w);
    for (int p = 0; p < 4; p++) for (int i = 0; i < D; i++) pe[p][i] = 0.2f * rnd();
    for (int i = 0; i < D; i++) { st[i] = 0.2f * rnd(); dn[i] = rnd(); }

    sdnc_backward(w, st, (const float (*)[D])pe, 4, dn, a, da, c);
    sdnc_backward_smoothed(w, st, (const float (*)[D])pe, 4, dn, sdnc_backward_smoothing_exact(), b, db, c);
    check("sdnc_backward == sdnc_backward_smoothed(beta = 1), bit for bit",
          memcmp(a, b, sizeof(SdncGrads)) == 0 && memcmp(da, db, sizeof(da)) == 0);

    sdnc_forward(w, st, (const float (*)[D])pe, 4, nx1);
    sdnc_forward_cached(w, st, (const float (*)[D])pe, 4, nx2, c);
    check("sdnc_forward_cached output == sdnc_forward, bit for bit", memcmp(nx1, nx2, sizeof(nx1)) == 0);

    /* Saturate every gate of the gated layer: exact derivative vanishes. */
    for (int i = 0; i < F; i++) w->ff_gate_b[1][i] += 1000.0f;
    sdnc_zero_grads(a); sdnc_zero_grads(b);
    sdnc_backward(w, st, (const float (*)[D])pe, 4, dn, a, NULL, c);
    SdncBackwardSmoothing sm = { 1e-3f, 1.0f };
    sdnc_backward_smoothed(w, st, (const float (*)[D])pe, 4, dn, sm, b, NULL, c);
    int exact_zero = 1, smooth_nonzero = 0;
    for (int i = 0; i < F; i++) {
        if (a->dff_gate_b[1][i] != 0.0f) exact_zero = 0;
        if (b->dff_gate_b[1][i] != 0.0f) smooth_nonzero = 1;
    }
    check("saturated gates: exact gate-bias derivative identically zero", exact_zero);
    check("saturated gates: smoothed gate-bias derivative non-zero", smooth_nonzero);
    /* Gate values and up projections stay exact: the value derivatives of the
     * smoothed layer and of every later layer are unchanged (earlier layers
     * see the smoothed input gradient). */
    int value_same = 1;
    for (int L = 1; L < SDNC_N_LAYERS; L++)
        value_same = value_same &&
            memcmp(a->dff_down[L], b->dff_down[L], sizeof(a->dff_down[L])) == 0 &&
            memcmp(a->dff_up[L], b->dff_up[L], sizeof(a->dff_up[L])) == 0 &&
            memcmp(a->dff_up_b[L], b->dff_up_b[L], sizeof(a->dff_up_b[L])) == 0;
    check("smoothing keeps the value derivatives of the smoothed and later layers", value_same);
    free(w); free(a); free(b); free(c);
}

/* A two-neuron interpreter: gate g = [x0 >= 1] (pre = 300 x0 - 150), neuron 0
 * adds g*x1 to output 2, neuron 1 (the same gate) adds g*x1 to output 3. */
static void synthetic(SdncWeights* w) {
    memset(w, 0, sizeof(*w));
    w->ff_type[1] = 2;
    for (int n = 0; n < 2; n++) {
        w->ff_gate[1][0 * F + n] = SDNC_CONSTRUCTION_UNIT;
        w->ff_gate_b[1][n] = -0.5f * SDNC_CONSTRUCTION_UNIT;
        w->ff_up[1][1 * F + n] = 1.0f;
        w->ff_down[1][n * D + 2 + n] = 1.0f;
    }
}

static void group_checks(void) {
    SdncWeights* w = (SdncWeights*)malloc(sizeof(SdncWeights));
    SdncGateGroups* g = (SdncGateGroups*)malloc(sizeof(SdncGateGroups));
    synthetic(w);
    w->ff_gate[1][0 * F + 2] = SDNC_CONSTRUCTION_UNIT;   /* same column, other bias */
    w->ff_gate_b[1][2] = 0.5f * SDNC_CONSTRUCTION_UNIT;
    w->ff_gate[1][7 * F + 3] = -0.0f;                     /* -0 gates like +0 */
    w->ff_gate[1][0 * F + 3] = SDNC_CONSTRUCTION_UNIT;
    w->ff_gate_b[1][3] = -0.5f * SDNC_CONSTRUCTION_UNIT;
    sdnc_gate_groups_build(w, g);
    check("tied gates: identical column+bias share a group (incl. -0 == +0)",
          g->group[1][0] == 0 && g->group[1][1] == 0 && g->group[1][3] == 0);
    check("tied gates: a different bias is a different gate", g->group[1][2] == 1 && g->n_groups == 2);
    check("tied gates: neurons without a gate and non-gated layers are -1",
          g->group[1][4] == -1 && g->group[0][0] == -1 && g->group[3][0] == -1);
    free(w); free(g);
}

static void record(const SdncWeights* w, SdncTransitionSet* ts) {
    static float pe[2][D];
    for (int x0 = 0; x0 <= 1; x0++)
        for (int x1 = 1; x1 <= 4; x1++) {
            float st[D], nx[D];
            memset(st, 0, sizeof(st));
            st[0] = (float)x0; st[1] = (float)x1; st[2] = 10.0f;
            sdnc_forward(w, st, (const float (*)[D])pe, 0, nx);
            sdnc_transitions_add(ts, st, &pe[0][0], 0, nx);
        }
}

static void io_checks(const char* dir) {
    char path[4096];
    SdncTransitionSet ts, back;
    sdnc_transitions_init(&ts);
    static float pe[3][D];
    float st[D], nx[D];
    for (int i = 0; i < D; i++) { st[i] = rnd(); nx[i] = rnd(); }
    for (int p = 0; p < 3; p++) for (int i = 0; i < D; i++) pe[p][i] = rnd();
    sdnc_transitions_add(&ts, st, &pe[0][0], 3, nx);
    sdnc_transitions_add(&ts, nx, &pe[0][0], 3, st);   /* same program */
    sdnc_transitions_add(&ts, st, &pe[0][0], 2, nx);   /* new program */
    check("transitions: consecutive runs of one program share embeddings",
          ts.n == 3 && ts.n_progs == 2 && ts.n_rows == 5);
    snprintf(path, sizeof(path), "%s/unit.sdtr", dir);
    int ok = sdnc_transitions_save(&ts, path) == 0 && sdnc_transitions_load(&back, path) == 0;
    ok = ok && back.n == 3 && back.n_progs == 2 && back.n_rows == 5 &&
         memcmp(back.in, ts.in, sizeof(float) * D * 3) == 0 &&
         memcmp(back.out, ts.out, sizeof(float) * D * 3) == 0 &&
         memcmp(back.pe_rows, ts.pe_rows, sizeof(float) * D * 5) == 0 &&
         back.prog[2] == 1 && back.prog_len[1] == 2;
    check("transitions: SDTR save/load round trip", ok);
    if (ok) sdnc_transitions_free(&back);
    FILE* f = fopen(path, "r+b");
    if (f) { uint32_t bad = 7; fseek(f, 8, SEEK_SET); fwrite(&bad, 4, 1, f); fclose(f); }
    check("transitions: a file for another width is refused", sdnc_transitions_load(&back, path) == -1);
    sdnc_transitions_free(&ts);

    SdncWeights* w = (SdncWeights*)malloc(sizeof(SdncWeights));
    SdncWeights* r = (SdncWeights*)malloc(sizeof(SdncWeights));
    sdnc_init_weights(w);
    snprintf(path, sizeof(path), "%s/unit.qlmw", dir);
    uint32_t h[7] = { 0x514C4D57u, 3u, (uint32_t)SDNC_D, (uint32_t)SDNC_N_LAYERS,
                      (uint32_t)SDNC_FFN_DIM, 16u, (uint32_t)SDNC_HD };
    f = fopen(path, "wb");
    if (f) { fwrite(h, sizeof(h), 1, f); fwrite(w, sizeof(*w), 1, f); fclose(f); }
    check("weights: QLMW v3 load reproduces the struct",
          sdnc_weights_load_qlmw(r, path) == 0 && memcmp(r, w, sizeof(*w)) == 0);
    h[4] = 1024u;
    f = fopen(path, "wb");
    if (f) { fwrite(h, sizeof(h), 1, f); fwrite(w, sizeof(*w), 1, f); fclose(f); }
    check("weights: another architecture is refused", sdnc_weights_load_qlmw(r, path) == -2);
    h[4] = (uint32_t)SDNC_FFN_DIM;
    f = fopen(path, "wb");
    if (f) { fwrite(h, sizeof(h), 1, f); fwrite(w, sizeof(*w) / 2, 1, f); fclose(f); }
    check("weights: a truncated file is refused", sdnc_weights_load_qlmw(r, path) != 0);
    free(w); free(r);
}

static void optimizer_checks(void) {
    SdncWeights* w0 = (SdncWeights*)malloc(sizeof(SdncWeights));
    SdncWeights* w = (SdncWeights*)malloc(sizeof(SdncWeights));
    SdncGrads* g = (SdncGrads*)calloc(1, sizeof(SdncGrads));
    synthetic(w0);
    SdncTransitionSet ts; sdnc_transitions_init(&ts);
    record(w0, &ts);
    memcpy(w, w0, sizeof(*w));
    for (int n = 0; n < 2; n++) w->ff_gate_b[1][n] -= SDNC_CONSTRUCTION_UNIT;   /* switch the gate off */
    int mm = 0;
    sdnc_transition_objective(w, &ts, &mm);
    check("synthetic: the damaged gate changes 4 of 8 transitions", mm == 4);
    sdnc_control_gradient(w, &ts, sdnc_backward_smoothing_exact(), g);
    check("synthetic: exact control gradient identically zero", sdnc_control_gradient_is_zero(g));
    SdncFinetuneOptions opt;
    sdnc_finetune_default_options(&opt);
    SdncFinetuneReport rep;
    int rc = sdnc_control_finetune(w, &ts, NULL, &opt, &rep);
    check("synthetic: one tied-gate move restores every transition",
          rc == 0 && rep.mismatched_before == 4 && rep.mismatched_after == 0 &&
          rep.gate_moves == 1 && rep.neuron_moves == 0 && rep.tied_gates == 1 &&
          rep.exact_control_gradient_zero == 1);
    check("synthetic: the repaired thresholds are the construction's",
          memcmp(w->ff_gate_b, w0->ff_gate_b, sizeof(w->ff_gate_b)) == 0);
    sdnc_transitions_free(&ts);
    free(w0); free(w); free(g);
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : ".";
    printf("=== SDNC control fine-tuning: unit checks ===\n");
    rule_checks();
    backward_checks();
    group_checks();
    io_checks(dir);
    optimizer_checks();
    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
