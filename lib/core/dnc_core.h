/*
 * DNC Core — shared differentiable-external-memory math.
 *
 * Self-contained header exposing the addressing/read/write/backprop math of
 * lib/backend/diff_memory_prototype.c (the standalone NTM/DNC de-risking
 * artifact) so the Eshkol runtime (lib/core/dnc_api.c) can drive a
 * differentiable external memory — content addressing (cosine + softmax),
 * location addressing (indicator bump), NTM erase/add write, least-used
 * allocation, and hand-derived backprop through read + content addressing —
 * on real memory banks from .esk.
 *
 * The functions here are byte-for-byte mirrors of the static functions in
 * lib/backend/diff_memory_prototype.c, GENERALISED from its compile-time
 * N=64,W=8 to runtime row/col counts (so a .esk caller can size the bank).
 * At the prototype's N=64,W=8 they reduce to exactly the same arithmetic, in
 * the same order, so the .esk builtins reproduce the prototype's own main()
 * oracle bit-for-bit (the "C is source of truth" dylib rule).
 *
 * Keeping this in a standalone header (rather than un-static'ing symbols in a
 * separate link unit that has its own main()) means the diff_memory_prototype
 * executable and its three acceptance checks build entirely unchanged.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_DNC_CORE_H
#define ESHKOL_DNC_CORE_H

#include <stddef.h>
#include <math.h>

/* sigmoid (double) mirroring diff_memory_prototype.c's sigmoidd saturation. */
static inline double dnc_sigmoidd(double x) {
    if (x > 40.0) return 1.0;
    if (x < -40.0) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

/* Generalized indicator bump at integer address k, evaluated at x.
 * Identical in form to diff_memory_prototype.c's indicator(); beta plays SCALE. */
static inline double dnc_indicator(double x, double k, double beta) {
    return dnc_sigmoidd(beta * (x - k + 0.5)) - dnc_sigmoidd(beta * (x - k - 0.5));
}

/* Numerically stable softmax of logits[n] into out[n] (mirror softmax). */
static inline void dnc_softmax(const double *logits, int n, double *out) {
    double m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { out[i] = exp(logits[i] - m); s += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= s;
}

/* Location addressing: indicator bump over N rows, normalized (mirror loc_weights). */
static inline void dnc_loc_weights(double addr, double beta, int N, double *w) {
    for (int i = 0; i < N; i++) w[i] = dnc_indicator(addr, (double)i, beta);
    double s = 0.0;
    for (int i = 0; i < N; i++) s += w[i];
    if (s > 1e-300) for (int i = 0; i < N; i++) w[i] /= s;
}

/* cosine_sim over n dims (mirror cosine_sim). */
static inline double dnc_cosine_sim(const double *a, const double *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    double denom = sqrt(na) * sqrt(nb) + 1e-12;
    return dot / denom;
}

/* Content addressing: softmax(beta * cosine(key, row_i)) over N rows
 * (mirror content_weights). mem is row-major N x W. */
static inline void dnc_content_weights(const double *mem, int N, int W,
                                       const double *key, double beta, double *w) {
    /* logits buffer is caller-free: compute into w then softmax in place via tmp. */
    /* Use a local VLA-free scheme: store logits into w, then softmax(w,...). */
    for (int i = 0; i < N; i++) w[i] = beta * dnc_cosine_sim(key, &mem[(size_t)i*W], W);
    /* softmax in place: */
    double m = w[0];
    for (int i = 1; i < N; i++) if (w[i] > m) m = w[i];
    double s = 0.0;
    for (int i = 0; i < N; i++) { w[i] = exp(w[i] - m); s += w[i]; }
    for (int i = 0; i < N; i++) w[i] /= s;
}

/* read = sum_i w_i * mem[i] (mirror read_mem). out length W. */
static inline void dnc_read_mem(const double *mem, int N, int W,
                                const double *w, double *out) {
    for (int j = 0; j < W; j++) {
        double acc = 0.0;
        for (int i = 0; i < N; i++) acc += w[i] * mem[(size_t)i*W + j];
        out[j] = acc;
    }
}

/* NTM erase/add: mem[i] = mem[i]*(1 - w_i*erase) + w_i*add; usage[i]+=w_i
 * (mirror write_mem). */
static inline void dnc_write_mem(double *mem, double *usage, int N, int W,
                                 const double *w, const double *erase, const double *add) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < W; j++) {
            double *cell = &mem[(size_t)i*W + j];
            *cell = (*cell) * (1.0 - w[i]*erase[j]) + w[i]*add[j];
        }
        usage[i] += w[i];
    }
}

/* Allocation: softmax(-beta * usage) (mirror alloc_weights). */
static inline void dnc_alloc_weights(const double *usage, int N, double beta, double *w) {
    for (int i = 0; i < N; i++) w[i] = -beta * usage[i];
    dnc_softmax(w, N, w);
}

/* Backprop through L = 0.5 * sum_j (read_j - target_j)^2, read = sum_i w_i M_i,
 * w = softmax(beta * cosine(key, M_i)). Fills grad_key[W] = dL/dkey and the full
 * grad_mem[N*W] = dL/dM (row-major). Returns L. Mirror loss_and_grads, extended
 * from the prototype's single-row grad_row to the full memory gradient (each row
 * enters BOTH the read and its own similarity term — identical per-row formula).
 *
 * Scratch arrays (caller-allocated, never freed here): s[N], wv[N], rd[W],
 * dread[W], dw[N], ds[N]. */
static inline double dnc_loss_and_grads(const double *mem, int N, int W,
                                        const double *key, const double *target, double beta,
                                        double *grad_key, double *grad_mem,
                                        double *s, double *wv, double *rd,
                                        double *dread, double *dw, double *ds) {
    /* ---- forward ---- */
    for (int i = 0; i < N; i++) s[i] = dnc_cosine_sim(key, &mem[(size_t)i*W], W);
    for (int i = 0; i < N; i++) wv[i] = beta * s[i];
    dnc_softmax(wv, N, wv);            /* wv now = softmax(beta*s) */
    dnc_read_mem(mem, N, W, wv, rd);

    double L = 0.0;
    for (int j = 0; j < W; j++) {
        double e = rd[j] - target[j];
        L += 0.5 * e * e;
        dread[j] = e;
    }

    /* dL/dw_i = sum_j dread_j * M[i][j] */
    for (int i = 0; i < N; i++) {
        double acc = 0.0;
        for (int j = 0; j < W; j++) acc += dread[j] * mem[(size_t)i*W + j];
        dw[i] = acc;
    }

    /* Through softmax: dL/dlogit_k = w_k*(dw_k - sum_i w_i dw_i); dL/ds_k = beta*that */
    double wdotdw = 0.0;
    for (int i = 0; i < N; i++) wdotdw += wv[i] * dw[i];
    for (int k = 0; k < N; k++) ds[k] = beta * wv[k] * (dw[k] - wdotdw);

    /* ---- dL/dkey via cosine sims (mirror prototype) ---- */
    double na = 0.0;
    for (int t = 0; t < W; t++) na += key[t]*key[t];
    double sqrt_na = sqrt(na);
    for (int t = 0; t < W; t++) grad_key[t] = 0.0;
    for (int i = 0; i < N; i++) {
        double dot = 0.0, nbi = 0.0;
        const double *Mi = &mem[(size_t)i*W];
        for (int t = 0; t < W; t++) { dot += key[t]*Mi[t]; nbi += Mi[t]*Mi[t]; }
        double sqrt_nbi = sqrt(nbi);
        double denom = sqrt_na * sqrt_nbi + 1e-12;
        double inv_denom = 1.0 / denom;
        double dsqrt_na_term = (sqrt_na > 1e-300) ? (sqrt_nbi / sqrt_na) : 0.0;
        for (int t = 0; t < W; t++) {
            double ddot = Mi[t];
            double ddenom = dsqrt_na_term * key[t];
            double dsi_dkeyt = ddot * inv_denom - dot * ddenom * inv_denom * inv_denom;
            grad_key[t] += ds[i] * dsi_dkeyt;
        }
    }

    /* ---- dL/dM for every row (mirror prototype's single-row formula) ----
     * Row r enters BOTH the read (read = sum_i w_i M_i) and similarity s_r. */
    for (int r = 0; r < N; r++) {
        const double *Mr = &mem[(size_t)r*W];
        double *gr = &grad_mem[(size_t)r*W];
        double dot = 0.0, nbr = 0.0;
        for (int t = 0; t < W; t++) { dot += key[t]*Mr[t]; nbr += Mr[t]*Mr[t]; }
        double sqrt_nbr = sqrt(nbr);
        double denom = sqrt_na * sqrt_nbr + 1e-12;
        double inv_denom = 1.0 / denom;
        double dsqrt_nb_term = (sqrt_nbr > 1e-300) ? (sqrt_na / sqrt_nbr) : 0.0;
        for (int t = 0; t < W; t++) {
            double ddot = key[t];
            double ddenom = dsqrt_nb_term * Mr[t];
            double dsr_dMt = ddot * inv_denom - dot * ddenom * inv_denom * inv_denom;
            gr[t] = dread[t] * wv[r]      /* read path */
                  + ds[r] * dsr_dMt;       /* sim path  */
        }
    }

    return L;
}

#endif /* ESHKOL_DNC_CORE_H */
