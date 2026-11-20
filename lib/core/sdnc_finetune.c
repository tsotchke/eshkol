/*
 * SDNC control fine-tuning with an exact forward pass. See sdnc_finetune.h and
 * docs/design/adr/0032-exact-forward-control-fine-tuning.md.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "sdnc_finetune.h"

#define FT_D   SDNC_D
#define FT_NL  SDNC_N_LAYERS
#define FT_F   SDNC_FFN_DIM
#define FT_NFWD (SDNC_N_LAYERS - 1)   /* forward layers 0..N_LAYERS-2 */

/* ===================================================================== */
/* Recorded transitions                                                   */
/* ===================================================================== */

void sdnc_transitions_init(SdncTransitionSet* ts) {
    memset(ts, 0, sizeof(*ts));
}

void sdnc_transitions_free(SdncTransitionSet* ts) {
    free(ts->in); free(ts->out); free(ts->prog);
    free(ts->prog_len); free(ts->prog_row); free(ts->pe_rows);
    memset(ts, 0, sizeof(*ts));
}

static int ft_grow(void** p, size_t elem, int* cap, int need) {
    if (need <= *cap) return 0;
    int nc = *cap ? *cap : 256;
    while (nc < need) nc *= 2;
    void* q = realloc(*p, (size_t)nc * elem);
    if (!q) return -1;
    *p = q; *cap = nc;
    return 0;
}

static const float* ft_prog_pe(const SdncTransitionSet* ts, int k) {
    return ts->pe_rows + ts->prog_row[k] * FT_D;
}

int sdnc_transitions_add(SdncTransitionSet* ts, const float* state,
                         const float* pe, int np, const float* next) {
    if (!ts || !state || !next || np < 0 || np > 256 || (np > 0 && !pe)) return -1;
    /* Consecutive transitions of one program run share its embeddings. */
    int k = ts->n_progs - 1;
    int same = k >= 0 && ts->prog_len[k] == np &&
               memcmp(ft_prog_pe(ts, k), pe, sizeof(float) * (size_t)np * FT_D) == 0;
    if (!same) {
        int cp = ts->cap_progs;
        if (ft_grow((void**)&ts->prog_len, sizeof(int), &cp, ts->n_progs + 1)) return -1;
        int cp2 = ts->cap_progs;
        if (ft_grow((void**)&ts->prog_row, sizeof(size_t), &cp2, ts->n_progs + 1)) return -1;
        ts->cap_progs = cp < cp2 ? cp : cp2;
        if (ts->n_rows + (size_t)np > ts->cap_rows) {
            size_t nc = ts->cap_rows ? ts->cap_rows : 4096;
            while (nc < ts->n_rows + (size_t)np) nc *= 2;
            float* q = (float*)realloc(ts->pe_rows, nc * FT_D * sizeof(float));
            if (!q) return -1;
            ts->pe_rows = q; ts->cap_rows = nc;
        }
        if (np > 0)
            memcpy(ts->pe_rows + ts->n_rows * FT_D, pe, sizeof(float) * (size_t)np * FT_D);
        ts->prog_len[ts->n_progs] = np;
        ts->prog_row[ts->n_progs] = ts->n_rows;
        ts->n_rows += (size_t)np;
        k = ts->n_progs++;
    }
    int c1 = ts->cap, c2 = ts->cap, c3 = ts->cap;
    if (ft_grow((void**)&ts->in, sizeof(float) * FT_D, &c1, ts->n + 1)) return -1;
    if (ft_grow((void**)&ts->out, sizeof(float) * FT_D, &c2, ts->n + 1)) return -1;
    if (ft_grow((void**)&ts->prog, sizeof(int), &c3, ts->n + 1)) return -1;
    ts->cap = c1 < c2 ? (c1 < c3 ? c1 : c3) : (c2 < c3 ? c2 : c3);
    memcpy(ts->in + (size_t)ts->n * FT_D, state, sizeof(float) * FT_D);
    memcpy(ts->out + (size_t)ts->n * FT_D, next, sizeof(float) * FT_D);
    ts->prog[ts->n] = k;
    ts->n++;
    return 0;
}

#define SDTR_MAGIC   0x52544453u  /* "SDTR" */
#define SDTR_VERSION 1u

int sdnc_transitions_save(const SdncTransitionSet* ts, const char* path) {
    if (!ts || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t hdr[5] = { SDTR_MAGIC, SDTR_VERSION, (uint32_t)FT_D,
                        (uint32_t)ts->n_progs, (uint32_t)ts->n };
    uint64_t rows = ts->n_rows;
    int ok = fwrite(hdr, sizeof(hdr), 1, f) == 1 && fwrite(&rows, sizeof(rows), 1, f) == 1;
    for (int k = 0; ok && k < ts->n_progs; k++) {
        int32_t len = ts->prog_len[k];
        uint64_t row = ts->prog_row[k];
        ok = fwrite(&len, sizeof(len), 1, f) == 1 && fwrite(&row, sizeof(row), 1, f) == 1;
    }
    if (ok && rows) ok = fwrite(ts->pe_rows, sizeof(float) * FT_D, rows, f) == rows;
    for (int t = 0; ok && t < ts->n; t++) {
        int32_t k = ts->prog[t];
        ok = fwrite(&k, sizeof(k), 1, f) == 1;
    }
    if (ok && ts->n) ok = fwrite(ts->in, sizeof(float) * FT_D, (size_t)ts->n, f) == (size_t)ts->n;
    if (ok && ts->n) ok = fwrite(ts->out, sizeof(float) * FT_D, (size_t)ts->n, f) == (size_t)ts->n;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

int sdnc_transitions_load(SdncTransitionSet* ts, const char* path) {
    if (!ts || !path) return -1;
    sdnc_transitions_init(ts);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t hdr[5]; uint64_t rows = 0;
    int ok = fread(hdr, sizeof(hdr), 1, f) == 1 && fread(&rows, sizeof(rows), 1, f) == 1 &&
             hdr[0] == SDTR_MAGIC && hdr[1] == SDTR_VERSION && hdr[2] == (uint32_t)FT_D &&
             hdr[3] <= (1u << 24) && hdr[4] <= (1u << 26) && rows <= ((uint64_t)1 << 32);
    int np = ok ? (int)hdr[3] : 0, n = ok ? (int)hdr[4] : 0;
    if (ok) {
        ts->prog_len = (int*)malloc(sizeof(int) * (size_t)(np ? np : 1));
        ts->prog_row = (size_t*)malloc(sizeof(size_t) * (size_t)(np ? np : 1));
        ts->pe_rows  = (float*)malloc(sizeof(float) * FT_D * (size_t)(rows ? rows : 1));
        ts->prog     = (int*)malloc(sizeof(int) * (size_t)(n ? n : 1));
        ts->in       = (float*)malloc(sizeof(float) * FT_D * (size_t)(n ? n : 1));
        ts->out      = (float*)malloc(sizeof(float) * FT_D * (size_t)(n ? n : 1));
        ok = ts->prog_len && ts->prog_row && ts->pe_rows && ts->prog && ts->in && ts->out;
    }
    for (int k = 0; ok && k < np; k++) {
        int32_t len; uint64_t row;
        ok = fread(&len, sizeof(len), 1, f) == 1 && fread(&row, sizeof(row), 1, f) == 1 &&
             len >= 0 && len <= 256 && row + (uint64_t)len <= rows;
        if (ok) { ts->prog_len[k] = len; ts->prog_row[k] = (size_t)row; }
    }
    if (ok && rows) ok = fread(ts->pe_rows, sizeof(float) * FT_D, (size_t)rows, f) == (size_t)rows;
    for (int t = 0; ok && t < n; t++) {
        int32_t k;
        ok = fread(&k, sizeof(k), 1, f) == 1 && k >= 0 && k < np;
        if (ok) ts->prog[t] = k;
    }
    if (ok && n) ok = fread(ts->in, sizeof(float) * FT_D, (size_t)n, f) == (size_t)n;
    if (ok && n) ok = fread(ts->out, sizeof(float) * FT_D, (size_t)n, f) == (size_t)n;
    if (ok) { char extra; ok = fread(&extra, 1, 1, f) == 0; }
    fclose(f);
    if (!ok) { sdnc_transitions_free(ts); return -1; }
    ts->n = ts->cap = n;
    ts->n_progs = ts->cap_progs = np;
    ts->n_rows = ts->cap_rows = (size_t)rows;
    return 0;
}

#define QLMW_MAGIC 0x514C4D57u  /* "QLMW" */

int sdnc_weights_load_qlmw(SdncWeights* w, const char* path) {
    if (!w || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[7];
    if (fread(h, sizeof(h), 1, f) != 1) { fclose(f); return -1; }
    if (h[0] != QLMW_MAGIC || h[1] != 3u || h[2] != (uint32_t)SDNC_D ||
        h[3] != (uint32_t)SDNC_N_LAYERS || h[4] != (uint32_t)SDNC_FFN_DIM ||
        h[6] != (uint32_t)SDNC_HD) { fclose(f); return -2; }
    SdncWeights* tmp = (SdncWeights*)malloc(sizeof(SdncWeights));
    if (!tmp) { fclose(f); return -1; }
    char extra;
    int ok = fread(tmp, sizeof(SdncWeights), 1, f) == 1 && fread(&extra, 1, 1, f) == 0;
    fclose(f);
    for (int L = 0; ok && L < SDNC_N_LAYERS; L++)
        ok = tmp->ff_type[L] >= 0 && tmp->ff_type[L] <= 2;
    if (ok) memcpy(w, tmp, sizeof(SdncWeights));
    free(tmp);
    return ok ? 0 : -2;
}

/* ===================================================================== */
/* Objective                                                              */
/* ===================================================================== */

static double ft_huber(const float* out, const float* target) {
    double L = 0.0;
    for (int i = 0; i < FT_D; i++) {
        double e = (double)out[i] - (double)target[i], a = fabs(e);
        L += a <= 1.0 ? 0.5 * e * e : a - 0.5;
    }
    return L;
}

static int ft_differs(const float* out, const float* target) {
    return memcmp(out, target, sizeof(float) * FT_D) != 0;
}

typedef const float (*ft_pe_t)[SDNC_D];

double sdnc_transition_objective(const SdncWeights* w,
                                 const SdncTransitionSet* ts, int* mismatched) {
    double L = 0.0; int mm = 0;
    for (int t = 0; t < ts->n; t++) {
        int k = ts->prog[t];
        float nx[FT_D];
        sdnc_forward(w, ts->in + (size_t)t * FT_D, (ft_pe_t)ft_prog_pe(ts, k),
                     ts->prog_len[k], nx);
        const float* tgt = ts->out + (size_t)t * FT_D;
        L += ft_huber(nx, tgt);
        mm += ft_differs(nx, tgt);
    }
    if (mismatched) *mismatched = mm;
    return L;
}

/* ===================================================================== */
/* Tied construction gates                                                */
/* ===================================================================== */

static uint64_t ft_hash_float(uint64_t h, float v) {
    if (v == 0.0f) v = 0.0f;              /* -0 and +0 gate identically */
    uint32_t b; memcpy(&b, &v, sizeof(b));
    for (int k = 0; k < 4; k++) { h ^= (b >> (8 * k)) & 0xffu; h *= 1099511628211ULL; }
    return h;
}

static int ft_gate_has_weight(const SdncWeights* w, int L, int i) {
    if (w->ff_gate_b[L][i] != 0.0f) return 1;
    for (int j = 0; j < FT_D; j++) if (w->ff_gate[L][j * FT_F + i] != 0.0f) return 1;
    return 0;
}

static int ft_same_gate(const SdncWeights* w, int L, int a, int b) {
    if (w->ff_gate_b[L][a] != w->ff_gate_b[L][b]) return 0;
    for (int j = 0; j < FT_D; j++)
        if (w->ff_gate[L][j * FT_F + a] != w->ff_gate[L][j * FT_F + b]) return 0;
    return 1;
}

void sdnc_gate_groups_build(const SdncWeights* w, SdncGateGroups* out) {
    enum { TABLE = 8192 };              /* > 2 * FFN_DIM, power of two */
    static const int EMPTY = -1;
    int table[TABLE];
    out->n_groups = 0;
    for (int L = 0; L < FT_NL; L++)
        for (int i = 0; i < FT_F; i++) out->group[L][i] = -1;
    for (int L = 0; L < FT_NL; L++) {
        if (w->ff_type[L] != 2) continue;
        for (int s = 0; s < TABLE; s++) table[s] = EMPTY;
        for (int i = 0; i < FT_F; i++) {
            if (!ft_gate_has_weight(w, L, i)) continue;
            uint64_t h = 1469598103934665603ULL;
            h = ft_hash_float(h, w->ff_gate_b[L][i]);
            for (int j = 0; j < FT_D; j++) h = ft_hash_float(h, w->ff_gate[L][j * FT_F + i]);
            int s = (int)(h & (TABLE - 1));
            for (;;) {
                int rep = table[s];
                if (rep == EMPTY) {
                    table[s] = i;
                    out->group[L][i] = out->n_groups++;
                    break;
                }
                if (ft_same_gate(w, L, rep, i)) { out->group[L][i] = out->group[L][rep]; break; }
                s = (s + 1) & (TABLE - 1);
            }
        }
    }
}

/* ===================================================================== */
/* Gradient                                                               */
/* ===================================================================== */

static void ft_huber_grad(const float* out, const float* target, float* dn) {
    for (int i = 0; i < FT_D; i++) {
        float e = out[i] - target[i];
        dn[i] = e > 1.0f ? 1.0f : (e < -1.0f ? -1.0f : e);
    }
}

/* Gradient over the transitions flagged in `mm` (all when mm is NULL), using
 * the cached current outputs when `cur` is non-NULL. */
static void ft_gradient(const SdncWeights* w, const SdncTransitionSet* ts,
                        SdncBackwardSmoothing sm, SdncGrads* g,
                        const unsigned char* mm, const float* cur,
                        SdncFwdCache* cache) {
    sdnc_zero_grads(g);
    for (int t = 0; t < ts->n; t++) {
        if (mm && !mm[t]) continue;
        int k = ts->prog[t];
        const float* in = ts->in + (size_t)t * FT_D;
        const float* tgt = ts->out + (size_t)t * FT_D;
        float nx[FT_D], dn[FT_D];
        if (cur) memcpy(nx, cur + (size_t)t * FT_D, sizeof(nx));
        else sdnc_forward(w, in, (ft_pe_t)ft_prog_pe(ts, k), ts->prog_len[k], nx);
        if (!ft_differs(nx, tgt)) continue;
        ft_huber_grad(nx, tgt, dn);
        sdnc_backward_smoothed(w, in, (ft_pe_t)ft_prog_pe(ts, k), ts->prog_len[k],
                               dn, sm, g, NULL, cache);
    }
}

void sdnc_control_gradient(const SdncWeights* w, const SdncTransitionSet* ts,
                           SdncBackwardSmoothing sm, SdncGrads* g) {
    SdncFwdCache* cache = (SdncFwdCache*)malloc(sizeof(SdncFwdCache));
    if (!cache) { sdnc_zero_grads(g); return; }
    ft_gradient(w, ts, sm, g, NULL, NULL, cache);
    free(cache);
}

static int ft_all_zero(const float* v, size_t n) {
    for (size_t i = 0; i < n; i++) if (v[i] != 0.0f) return 0;
    return 1;
}

int sdnc_control_gradient_is_zero(const SdncGrads* g) {
    return ft_all_zero(&g->dwq[0][0], sizeof(g->dwq) / sizeof(float)) &&
           ft_all_zero(&g->dwk[0][0], sizeof(g->dwk) / sizeof(float)) &&
           ft_all_zero(&g->dbq[0][0], sizeof(g->dbq) / sizeof(float)) &&
           ft_all_zero(&g->dff_gate[0][0], sizeof(g->dff_gate) / sizeof(float)) &&
           ft_all_zero(&g->dff_gate_b[0][0], sizeof(g->dff_gate_b) / sizeof(float));
}

/* ===================================================================== */
/* Exact incremental evaluation                                           */
/* ===================================================================== */

/* Per-transition state under the current weights: the input of every forward
 * FFN layer (x_post_attn), the output, its Huber term and mismatch flag. A
 * threshold move in layer L changes nothing before FFN L, so a candidate
 * re-runs only the tail from L, and only for transitions where some moved gate
 * is not saturated to the same value on both sides of the move. Everything
 * else is bit-identical by construction. */
typedef struct {
    const SdncTransitionSet* ts;
    float*         xpa;      /* n * FT_NFWD * D */
    float*         cur;      /* n * D */
    double*        huber;    /* n */
    unsigned char* mm;       /* n */
    int            mismatched;
    double         objective;
} FtEval;

static float* ft_xpa(const FtEval* ev, int t, int L) {
    return ev->xpa + ((size_t)t * FT_NFWD + (size_t)L) * FT_D;
}

static void ft_totals(FtEval* ev) {
    double L = 0.0; int mm = 0;
    for (int t = 0; t < ev->ts->n; t++) { L += ev->huber[t]; mm += ev->mm[t]; }
    ev->objective = L; ev->mismatched = mm;
}

static void ft_eval_full(FtEval* ev, const SdncWeights* w) {
    const SdncTransitionSet* ts = ev->ts;
    for (int t = 0; t < ts->n; t++) {
        int k = ts->prog[t];
        float x[FT_D];
        memcpy(x, ts->in + (size_t)t * FT_D, sizeof(x));
        sdnc_layer0_attention_residual(w, x, (ft_pe_t)ft_prog_pe(ts, k), ts->prog_len[k]);
        sdnc_forward_tail(w, 0, x, (float (*)[FT_D])ft_xpa(ev, t, 0));
        memcpy(ev->cur + (size_t)t * FT_D, x, sizeof(x));
        const float* tgt = ts->out + (size_t)t * FT_D;
        ev->huber[t] = ft_huber(x, tgt);
        ev->mm[t] = (unsigned char)ft_differs(x, tgt);
    }
    ft_totals(ev);
}

/* -1 / +1 when the gate argument is clamped by sdnc_sigmoidf with a margin
 * that absorbs any rounding difference in the pre-activation; 0 otherwise. */
static int ft_clamp_class(float pre, float scale_hint) {
    float margin = 1.0f + 1e-4f * scale_hint;
    if (pre > 20.0f + margin) return 1;
    if (pre < -20.0f - margin) return -1;
    return 0;
}

typedef struct { int L; int n; const int* idx; const float* old_b; const float* new_b; } FtMove;

static int ft_move_affects(const SdncWeights* w, const FtMove* mv, const float* x) {
    for (int m = 0; m < mv->n; m++) {
        int i = mv->idx[m];
        float acc = 0.0f, mag = 0.0f;
        for (int j = 0; j < FT_D; j++) {
            float term = x[j] * w->ff_gate[mv->L][j * FT_F + i];
            acc += term; mag += fabsf(term);
        }
        float hint = mag + fabsf(mv->old_b[m]) + fabsf(mv->new_b[m]);
        int a = ft_clamp_class(acc + mv->old_b[m], hint);
        int b = ft_clamp_class(acc + mv->new_b[m], hint);
        if (a == 0 || b == 0 || a != b) return 1;
    }
    return 0;
}

/* Evaluate the weights `w` (with the move already applied): transitions the
 * move cannot change keep their cached terms, the others re-run from layer L.
 * Sums in transition order, so the totals equal a full evaluation. */
static void ft_eval_move(const FtEval* ev, const SdncWeights* w, const FtMove* mv,
                         int* mismatched, double* objective) {
    const SdncTransitionSet* ts = ev->ts;
    double L = 0.0; int mm = 0;
    for (int t = 0; t < ts->n; t++) {
        const float* xL = ft_xpa(ev, t, mv->L);
        if (ft_move_affects(w, mv, xL)) {
            float x[FT_D];
            memcpy(x, xL, sizeof(x));
            sdnc_forward_tail(w, mv->L, x, NULL);
            const float* tgt = ts->out + (size_t)t * FT_D;
            L += ft_huber(x, tgt); mm += ft_differs(x, tgt);
        } else {
            L += ev->huber[t]; mm += ev->mm[t];
        }
    }
    *mismatched = mm; *objective = L;
}

/* Commit an accepted move: refresh every cached layer input after L for the
 * transitions it touched. */
static void ft_commit_move(FtEval* ev, const SdncWeights* w, const FtMove* mv) {
    const SdncTransitionSet* ts = ev->ts;
    for (int t = 0; t < ts->n; t++) {
        if (!ft_move_affects(w, mv, ft_xpa(ev, t, mv->L))) continue;
        float x[FT_D];
        memcpy(x, ft_xpa(ev, t, mv->L), sizeof(x));
        sdnc_forward_tail(w, mv->L, x, (float (*)[FT_D])ft_xpa(ev, t, 0));
        const float* tgt = ts->out + (size_t)t * FT_D;
        memcpy(ev->cur + (size_t)t * FT_D, x, sizeof(x));
        ev->huber[t] = ft_huber(x, tgt);
        ev->mm[t] = (unsigned char)ft_differs(x, tgt);
    }
    ft_totals(ev);
}

/* ===================================================================== */
/* Optimizer                                                              */
/* ===================================================================== */

void sdnc_finetune_default_options(SdncFinetuneOptions* opt) {
    memset(opt, 0, sizeof(*opt));
    opt->smoothing.gate_beta = 10.0f / SDNC_CONSTRUCTION_UNIT;
    opt->smoothing.attn_beta = 10.0f / SDNC_CONSTRUCTION_UNIT;
    opt->step = SDNC_CONSTRUCTION_UNIT;
    opt->gate_candidates = 16;
    opt->gate_moves_max = 64;
    opt->neuron_candidates = 64;
    opt->neuron_moves_max = 64;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
static void ft_log(const SdncFinetuneOptions* opt, const char* fmt, ...) {
    if (!opt->log) return;
    char line[256];
    va_list ap; va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    opt->log(opt->log_ud, line);
}

typedef struct { int id; double mag; } FtRank;

static int ft_rank_cmp(const void* a, const void* b) {
    const FtRank* x = (const FtRank*)a; const FtRank* y = (const FtRank*)b;
    if (x->mag > y->mag) return -1;
    if (x->mag < y->mag) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

typedef struct {
    int  n_groups;
    int* start;   /* n_groups + 1 offsets into members */
    int* layer;   /* layer of each group */
    int* member;  /* neuron indices, grouped */
} FtGroupIndex;

static int ft_group_index(const SdncGateGroups* gg, FtGroupIndex* gi) {
    gi->n_groups = gg->n_groups;
    gi->start = (int*)calloc((size_t)gg->n_groups + 1, sizeof(int));
    gi->layer = (int*)calloc((size_t)(gg->n_groups ? gg->n_groups : 1), sizeof(int));
    gi->member = (int*)malloc(sizeof(int) * FT_NL * FT_F);
    if (!gi->start || !gi->layer || !gi->member) return -1;
    for (int L = 0; L < FT_NL; L++)
        for (int i = 0; i < FT_F; i++)
            if (gg->group[L][i] >= 0) { gi->start[gg->group[L][i] + 1]++; gi->layer[gg->group[L][i]] = L; }
    for (int k = 0; k < gg->n_groups; k++) gi->start[k + 1] += gi->start[k];
    int* fill = (int*)calloc((size_t)(gg->n_groups ? gg->n_groups : 1), sizeof(int));
    if (!fill) return -1;
    for (int L = 0; L < FT_NL; L++)
        for (int i = 0; i < FT_F; i++) {
            int k = gg->group[L][i];
            if (k >= 0) gi->member[gi->start[k] + fill[k]++] = i;
        }
    free(fill);
    return 0;
}

static void ft_group_index_free(FtGroupIndex* gi) {
    free(gi->start); free(gi->layer); free(gi->member);
}

/* Try one threshold move; returns 1 when it beats (best_mm, best_obj). */
static int ft_try_move(FtEval* ev, SdncWeights* w, int L, const int* idx, int n,
                       float delta, float* old_b, float* new_b,
                       int* best_mm, double* best_obj) {
    for (int m = 0; m < n; m++) {
        old_b[m] = w->ff_gate_b[L][idx[m]];
        new_b[m] = old_b[m] + delta;
        w->ff_gate_b[L][idx[m]] = new_b[m];
    }
    FtMove mv = { L, n, idx, old_b, new_b };
    int mm; double obj;
    ft_eval_move(ev, w, &mv, &mm, &obj);
    for (int m = 0; m < n; m++) w->ff_gate_b[L][idx[m]] = old_b[m];
    if (mm < *best_mm || (mm == *best_mm && obj < *best_obj)) {
        *best_mm = mm; *best_obj = obj;
        return 1;
    }
    return 0;
}

static void ft_apply_move(FtEval* ev, SdncWeights* w, int L, const int* idx, int n,
                          float delta, float* old_b, float* new_b) {
    for (int m = 0; m < n; m++) {
        old_b[m] = w->ff_gate_b[L][idx[m]];
        new_b[m] = old_b[m] + delta;
        w->ff_gate_b[L][idx[m]] = new_b[m];
    }
    FtMove mv = { L, n, idx, old_b, new_b };
    ft_commit_move(ev, w, &mv);
}

int sdnc_control_finetune(SdncWeights* w, const SdncTransitionSet* ts,
                          const SdncGateGroups* groups,
                          const SdncFinetuneOptions* opt_in,
                          SdncFinetuneReport* rep) {
    SdncFinetuneOptions opt;
    if (opt_in) opt = *opt_in; else sdnc_finetune_default_options(&opt);
    SdncFinetuneReport r; memset(&r, 0, sizeof(r));
    r.transitions = ts->n;

    if (!groups) { if (rep) *rep = r; return -2; }
    int status = -1;
    FtGroupIndex gi; memset(&gi, 0, sizeof(gi));
    FtEval ev; memset(&ev, 0, sizeof(ev));
    SdncGrads* g = (SdncGrads*)malloc(sizeof(SdncGrads));
    SdncFwdCache* cache = (SdncFwdCache*)malloc(sizeof(SdncFwdCache));
    double* G = NULL; FtRank* rank = NULL;
    float* old_b = (float*)malloc(sizeof(float) * FT_F);
    float* new_b = (float*)malloc(sizeof(float) * FT_F);
    size_t n = (size_t)(ts->n ? ts->n : 1);
    ev.ts = ts;
    ev.xpa = (float*)malloc(sizeof(float) * FT_D * FT_NFWD * n);
    ev.cur = (float*)malloc(sizeof(float) * FT_D * n);
    ev.huber = (double*)malloc(sizeof(double) * n);
    ev.mm = (unsigned char*)malloc(n);
    if (!g || !cache || !old_b || !new_b || !ev.xpa || !ev.cur || !ev.huber || !ev.mm) goto done;

    if (ft_group_index(groups, &gi)) goto done;
    r.tied_gates = groups->n_groups;
    G = (double*)malloc(sizeof(double) * (size_t)(gi.n_groups ? gi.n_groups : 1));
    rank = (FtRank*)malloc(sizeof(FtRank) * (size_t)(gi.n_groups > FT_NL * FT_F ? gi.n_groups : FT_NL * FT_F));
    if (!G || !rank) goto done;

    ft_eval_full(&ev, w);
    r.mismatched_before = ev.mismatched;
    r.objective_before = ev.objective;
    ft_log(&opt, "start: %d of %d transitions differ from the record, huber %.6g, %d tied gates",
           ev.mismatched, ts->n, ev.objective, gi.n_groups);

    /* The exact derivative of every control weight, for the report. */
    ft_gradient(w, ts, sdnc_backward_smoothing_exact(), g, ev.mm, ev.cur, cache);
    r.exact_control_gradient_zero = sdnc_control_gradient_is_zero(g);
    ft_log(&opt, "exact control gradient: %s",
           r.exact_control_gradient_zero ? "identically zero" : "non-zero");

    /* Level 1: tied construction gates. */
    for (int it = 0; it < opt.gate_moves_max && ev.mismatched > 0; it++) {
        ft_gradient(w, ts, opt.smoothing, g, ev.mm, ev.cur, cache);
        int nr = 0;
        for (int k = 0; k < gi.n_groups; k++) {
            double s = 0.0;
            int L = gi.layer[k];
            for (int m = gi.start[k]; m < gi.start[k + 1]; m++) s += g->dff_gate_b[L][gi.member[m]];
            G[k] = s;
            if (s != 0.0) { rank[nr].id = k; rank[nr].mag = fabs(s); nr++; }
        }
        qsort(rank, (size_t)nr, sizeof(FtRank), ft_rank_cmp);
        int best = -1, best_mm = ev.mismatched; double best_obj = ev.objective;
        for (int a = 0; a < nr && a < opt.gate_candidates; a++) {
            int k = rank[a].id;
            float delta = G[k] > 0.0 ? -opt.step : opt.step;
            if (ft_try_move(&ev, w, gi.layer[k], gi.member + gi.start[k],
                            gi.start[k + 1] - gi.start[k], delta, old_b, new_b,
                            &best_mm, &best_obj)) best = k;
        }
        if (best < 0) {
            ft_log(&opt, "tied gates: none of the top %d ranked gates lowers the exact objective", opt.gate_candidates);
            break;
        }
        float delta = G[best] > 0.0 ? -opt.step : opt.step;
        ft_apply_move(&ev, w, gi.layer[best], gi.member + gi.start[best],
                      gi.start[best + 1] - gi.start[best], delta, old_b, new_b);
        r.gate_moves++;
        ft_log(&opt, "tied gate %d (layer %d, %d neurons) %+g: %d transitions differ, huber %.6g",
               best, gi.layer[best], gi.start[best + 1] - gi.start[best], (double)delta,
               ev.mismatched, ev.objective);
    }

    /* Level 2: single neurons of the tied gates. */
    for (int it = 0; it < opt.neuron_moves_max && ev.mismatched > 0; it++) {
        ft_gradient(w, ts, opt.smoothing, g, ev.mm, ev.cur, cache);
        int nr = 0;
        for (int L = 0; L < FT_NL; L++)
            for (int i = 0; i < FT_F; i++)
                if (groups->group[L][i] >= 0 && g->dff_gate_b[L][i] != 0.0f) {
                    rank[nr].id = L * FT_F + i; rank[nr].mag = fabs((double)g->dff_gate_b[L][i]); nr++;
                }
        qsort(rank, (size_t)nr, sizeof(FtRank), ft_rank_cmp);
        int best = -1, best_mm = ev.mismatched; double best_obj = ev.objective;
        for (int a = 0; a < nr && a < opt.neuron_candidates; a++) {
            int L = rank[a].id / FT_F, i = rank[a].id % FT_F;
            float delta = g->dff_gate_b[L][i] > 0.0f ? -opt.step : opt.step;
            if (ft_try_move(&ev, w, L, &i, 1, delta, old_b, new_b, &best_mm, &best_obj))
                best = rank[a].id;
        }
        if (best < 0) {
            ft_log(&opt, "neurons: none of the top %d ranked neurons lowers the exact objective", opt.neuron_candidates);
            break;
        }
        int L = best / FT_F, i = best % FT_F;
        float delta = g->dff_gate_b[L][i] > 0.0f ? -opt.step : opt.step;
        ft_apply_move(&ev, w, L, &i, 1, delta, old_b, new_b);
        r.neuron_moves++;
        ft_log(&opt, "neuron %d of layer %d %+g: %d transitions differ, huber %.6g",
               i, L, (double)delta, ev.mismatched, ev.objective);
    }

    /* The incremental bookkeeping is exact; confirm with a full evaluation. */
    {
        int mm = 0;
        double obj = sdnc_transition_objective(w, ts, &mm);
        r.mismatched_after = mm;
        r.objective_after = obj;
    }
    ft_log(&opt, "done: %d -> %d differing transitions in %d tied-gate and %d neuron moves",
           r.mismatched_before, r.mismatched_after, r.gate_moves, r.neuron_moves);
    status = 0;

done:
    free(g); free(cache); free(G); free(rank); free(old_b); free(new_b);
    ft_group_index_free(&gi);
    free(ev.xpa); free(ev.cur); free(ev.huber); free(ev.mm);
    if (rep) *rep = r;
    return status;
}
