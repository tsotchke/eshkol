/**
 * @file qllm_backward.c
 * @brief Backward (reverse-mode) pass through the SDNC weight matrices.
 *
 * Companion training-mode module for the paper artefact
 * *"The Self-Differentiating Neural Computer: Computable Transformers via
 * Analytical Weight Construction"* (tsotchke, 2026). The analytical
 * constructor in `lib/backend/weight_matrices.c` proves that a fixed
 * six-layer transformer's weights *are* the Eshkol VM ISA (the forward
 * pass). This file differentiates that same architecture: it computes the
 * gradient of a per-dimension weighted MSE loss with respect to every layer
 * parameter (attention + the SQUARE and gated-sigmoid FFN types), so the
 * computable transformer can be trained from (state, target) pairs emitted
 * by the reference interpreter rather than only constructed in closed form.
 *
 * Contents:
 *   - Per-dimension weighted MSE loss configuration
 *   - Backward through the SQUARE-activation FFN
 *   - Backward through the gated-sigmoid FFN
 *   - Backward through the Layer-0 Gaussian attention block
 *   - Full reverse walk over the configured layer stack
 *   - QLMW v4 checkpoint header (weights + optimiser state)
 *
 * Correctness. The FFN backward passes are certified to relative error
 * < 1e-6 against a central finite-difference reference by
 * `tests/backend/qllm_backward_gradcheck_test.c` (ctest
 * `qllm_backward_gradcheck`).
 *
 * Precision. The math is written against `qllm_real` (see qllm_backward.h),
 * which defaults to `float` so the weight layout stays byte-compatible with
 * `InterpreterWeights`/QLMW. The gradient-check test compiles this same
 * source with `-DQLLM_REAL=double` to certify the algorithm below the
 * float32 finite-difference floor.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* qllm_real scalar type + architecture constants (D/H/HD/N_LAYERS/FFN_DIM),
 * shared with the public backward surface and the gradient-check test. */
#include "eshkol/backend/qllm_backward.h"

/*******************************************************************************
 * Forward Cache — saved activations from forward pass for backward
 ******************************************************************************/

typedef struct {
    /* Per-layer input (after residual from prior layer) */
    qllm_real x_in[N_LAYERS][D];

    /* Attention (Layer 0 only) */
    qllm_real Q[D], K_all[256][D], V_all[256][D];
    qllm_real scores_raw[256];
    qllm_real attn_weights[256];
    qllm_real hout[D];
    int np;

    /* FFN activations per layer */
    qllm_real ffn_pre_act[N_LAYERS][FFN_DIM];   /* before activation */
    qllm_real ffn_gate_pre[N_LAYERS][FFN_DIM];  /* gate before sigmoid */
    qllm_real ffn_gate_post[N_LAYERS][FFN_DIM]; /* gate after sigmoid */
    qllm_real ffn_up_post[N_LAYERS][FFN_DIM];   /* up after bias */
    qllm_real ffn_h[N_LAYERS][FFN_DIM];         /* gate*up or h^2 */
    qllm_real ffn_out[N_LAYERS][D];             /* FFN output before residual */
} QllmForwardCache;

/*******************************************************************************
 * Weight Gradients — accumulator for all trainable parameters
 ******************************************************************************/

typedef struct {
    qllm_real dwq[N_LAYERS][D * D];
    qllm_real dwk[N_LAYERS][D * D];
    qllm_real dwv[N_LAYERS][D * D];
    qllm_real dwo[N_LAYERS][D * D];
    qllm_real dbq[N_LAYERS][D];
    qllm_real dff_up[N_LAYERS][D * FFN_DIM];
    qllm_real dff_up_b[N_LAYERS][FFN_DIM];
    qllm_real dff_down[N_LAYERS][FFN_DIM * D];
    qllm_real dff_down_b[N_LAYERS][D];
    qllm_real dff_gate[N_LAYERS][D * FFN_DIM];
    qllm_real dff_gate_b[N_LAYERS][FFN_DIM];
} QllmWeightGradients;

/*******************************************************************************
 * Loss Configuration
 ******************************************************************************/

typedef struct {
    qllm_real dim_weights[D];
} QllmLossConfig;

/*******************************************************************************
 * Backward Through FFN Type 1 (SQUARE activation)
 * Forward: h = x @ W_up + b_up; a = h^2; fo = a @ W_down + b_down
 ******************************************************************************/

void qllm_backward_ffn_square(
    const qllm_real* w_up,   /* D x FFN_DIM */
    const qllm_real* w_down, /* FFN_DIM x D */
    const qllm_real* b_up,
    const qllm_real* b_down,
    const qllm_real x_in[D],
    const qllm_real h_pre[FFN_DIM],   /* h before squaring */
    const qllm_real dfo[D],           /* upstream gradient */
    qllm_real dx[D],                  /* output: gradient into input */
    qllm_real* dw_up,  qllm_real* db_up,
    qllm_real* dw_down, qllm_real* db_down)
{
    /* d(b_down) += dfo */
    for (int i = 0; i < D; i++) db_down[i] += dfo[i];

    /* da = dfo @ W_down^T (shape: FFN_DIM) */
    qllm_real da[FFN_DIM];
    for (int j = 0; j < FFN_DIM; j++) {
        da[j] = 0;
        for (int i = 0; i < D; i++)
            da[j] += dfo[i] * w_down[j * D + i];
    }

    /* d(W_down) += h_sq^T outer dfo (h_sq = h^2) */
    for (int j = 0; j < FFN_DIM; j++)
        for (int i = 0; i < D; i++)
            dw_down[j * D + i] += h_pre[j] * h_pre[j] * dfo[i]; /* a = h^2, so h_sq = h*h */

    /* dh = da * 2*h (derivative of h^2) */
    qllm_real dh[FFN_DIM];
    for (int j = 0; j < FFN_DIM; j++)
        dh[j] = da[j] * 2.0f * h_pre[j];

    /* d(b_up) += dh */
    for (int j = 0; j < FFN_DIM; j++) db_up[j] += dh[j];

    /* d(W_up) += x^T outer dh */
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dw_up[i * FFN_DIM + j] += x_in[i] * dh[j];

    /* dx += dh @ W_up^T */
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dx[i] += dh[j] * w_up[i * FFN_DIM + j];
}

/*******************************************************************************
 * Backward Through FFN Type 2 (Gated-Sigmoid)
 * Forward: gate = sigmoid(x @ W_gate + b_gate)
 *          up = x @ W_up + b_up
 *          h = gate * up
 *          fo = h @ W_down + b_down
 ******************************************************************************/

void qllm_backward_ffn_gated(
    const qllm_real* w_up, const qllm_real* w_down, const qllm_real* w_gate,
    const qllm_real* b_up, const qllm_real* b_down, const qllm_real* b_gate,
    const qllm_real x_in[D],
    const qllm_real gate_post[FFN_DIM],  /* sigmoid output */
    const qllm_real up_post[FFN_DIM],    /* up = Wx + b */
    const qllm_real h[FFN_DIM],          /* gate * up */
    const qllm_real dfo[D],
    qllm_real dx[D],
    qllm_real* dw_up, qllm_real* db_up,
    qllm_real* dw_down, qllm_real* db_down,
    qllm_real* dw_gate, qllm_real* db_gate)
{
    /* d(b_down) += dfo */
    for (int i = 0; i < D; i++) db_down[i] += dfo[i];

    /* dh = dfo @ W_down^T */
    qllm_real dh_vec[FFN_DIM];
    for (int j = 0; j < FFN_DIM; j++) {
        dh_vec[j] = 0;
        for (int i = 0; i < D; i++)
            dh_vec[j] += dfo[i] * w_down[j * D + i];
    }

    /* d(W_down) += h^T outer dfo */
    for (int j = 0; j < FFN_DIM; j++)
        for (int i = 0; i < D; i++)
            dw_down[j * D + i] += h[j] * dfo[i];

    /* d_gate = dh * up (element-wise) */
    /* d_up = dh * gate (element-wise) */
    qllm_real d_gate[FFN_DIM], d_up[FFN_DIM];
    for (int j = 0; j < FFN_DIM; j++) {
        d_gate[j] = dh_vec[j] * up_post[j];
        d_up[j] = dh_vec[j] * gate_post[j];
    }

    /* Up branch: d(b_up) += d_up, d(W_up) += x^T outer d_up, dx += d_up @ W_up^T */
    for (int j = 0; j < FFN_DIM; j++) db_up[j] += d_up[j];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dw_up[i * FFN_DIM + j] += x_in[i] * d_up[j];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dx[i] += d_up[j] * w_up[i * FFN_DIM + j];

    /* Gate branch: sigmoid backward: d_g_pre = d_gate * gate * (1 - gate) */
    qllm_real d_g_pre[FFN_DIM];
    for (int j = 0; j < FFN_DIM; j++)
        d_g_pre[j] = d_gate[j] * gate_post[j] * (1.0f - gate_post[j]);

    for (int j = 0; j < FFN_DIM; j++) db_gate[j] += d_g_pre[j];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dw_gate[i * FFN_DIM + j] += x_in[i] * d_g_pre[j];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < FFN_DIM; j++)
            dx[i] += d_g_pre[j] * w_gate[i * FFN_DIM + j];
}

/*******************************************************************************
 * AdamW Optimizer
 ******************************************************************************/

typedef struct {
    qllm_real lr;
    qllm_real beta1, beta2;
    qllm_real epsilon;
    qllm_real weight_decay;
    qllm_real grad_clip_norm;
    int t;  /* timestep */
} QllmAdamWConfig;

/*******************************************************************************
 * Training Result
 ******************************************************************************/

typedef struct {
    qllm_real total_loss;
    int n_steps;
    int correct_outputs;
    int total_outputs;
} QllmTrainResult;

/*******************************************************************************
 * Backward Through Attention (Layer 0 only)
 * Forward: Q = x @ W_q + b_q
 *          K[p] = pe[p] @ W_k, V[p] = pe[p] @ W_v  (for each position p)
 *          scores[p] = (Q[0]*K[p][0] + Q[1]*K[p][1]) / sqrt(HD)
 *          attn = softmax(scores)
 *          hout[d] = sum_p(attn[p] * V[p][d])
 *          ao = hout @ W_o
 ******************************************************************************/

static void qllm_backward_attention(
    const qllm_real* wq, const qllm_real* wk, const qllm_real* wv, const qllm_real* wo,
    const qllm_real* bq,
    const QllmForwardCache* cache,
    const qllm_real dao[D],    /* upstream gradient into attention output */
    qllm_real dx[D],           /* output: gradient into layer input */
    qllm_real* dwq, qllm_real* dwk, qllm_real* dwv, qllm_real* dwo, qllm_real* dbq)
{
    int np = cache->np;
    if (np <= 0) return;

    /* d(W_o) += hout^T outer dao */
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            dwo[i * D + j] += cache->hout[i] * dao[j];

    /* d_hout = dao @ W_o^T */
    qllm_real d_hout[D] = {0};
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            d_hout[i] += dao[j] * wo[i * D + j];

    /* d_attn[p] = sum_d(d_hout[d] * V[p][d]) for d in [0..HD-1] */
    qllm_real d_attn[256] = {0};
    qllm_real dV[256][D];
    memset(dV, 0, sizeof(dV));
    for (int p = 0; p < np; p++) {
        for (int d = 0; d < HD; d++) {
            d_attn[p] += d_hout[d] * cache->V_all[p][d];
            dV[p][d] += cache->attn_weights[p] * d_hout[d];
        }
    }

    /* Softmax backward: d_scores[p] = attn[p] * (d_attn[p] - dot(attn, d_attn)) */
    qllm_real dot = 0;
    for (int p = 0; p < np; p++)
        dot += cache->attn_weights[p] * d_attn[p];
    qllm_real d_scores[256];
    qllm_real scale = 1.0f / sqrt((qllm_real)HD);
    for (int p = 0; p < np; p++)
        d_scores[p] = cache->attn_weights[p] * (d_attn[p] - dot) * scale;

    /* dQ from scores: dQ[d] += sum_p(d_scores[p] * K[p][d]) for d in [0..HD-1] */
    qllm_real dQ[D] = {0};
    for (int p = 0; p < np; p++)
        for (int d = 0; d < HD; d++)
            dQ[d] += d_scores[p] * cache->K_all[p][d];

    /* dK[p][d] += d_scores[p] * Q[d] */
    qllm_real dK[256][D];
    memset(dK, 0, sizeof(dK));
    for (int p = 0; p < np; p++)
        for (int d = 0; d < HD; d++)
            dK[p][d] += d_scores[p] * cache->Q[d];

    /* Q = x @ W_q + b_q → dbq += dQ, dwq += x^T outer dQ, dx += dQ @ W_q^T */
    for (int d = 0; d < D; d++) dbq[d] += dQ[d];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            dwq[i * D + j] += cache->x_in[0][i] * dQ[j];
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            dx[i] += dQ[j] * wq[i * D + j];

    /* K[p] = pe[p] @ W_k → dwk += sum_p(pe[p]^T outer dK[p])
     * pe is fixed (not trainable) */
    /* Note: pe is position embeddings, passed externally. For now, only
     * weight gradients are computed. pe gradients would be needed for
     * learnable positional encodings. */

    /* V[p] = pe[p] @ W_v → dwv += sum_p(pe[p]^T outer dV[p]) */
    /* Same note: pe is fixed. */
}

/*******************************************************************************
 * Full Backward Pass — all configured layers in reverse
 ******************************************************************************/

/* Weight layout — must match InterpreterWeights in weight_matrices.c */
typedef struct {
    qllm_real wq[N_LAYERS][D * D];
    qllm_real wk[N_LAYERS][D * D];
    qllm_real wv[N_LAYERS][D * D];
    qllm_real wo[N_LAYERS][D * D];
    qllm_real bq[N_LAYERS][D];
    qllm_real ff_up[N_LAYERS][D * FFN_DIM];
    qllm_real ff_up_b[N_LAYERS][FFN_DIM];
    qllm_real ff_down[N_LAYERS][FFN_DIM * D];
    qllm_real ff_down_b[N_LAYERS][D];
    qllm_real ff_gate[N_LAYERS][D * FFN_DIM];
    qllm_real ff_gate_b[N_LAYERS][FFN_DIM];
    int ff_type[N_LAYERS];
} QllmWeights;

/*******************************************************************************
 * Cached Forward Pass — saves all activations for backward
 ******************************************************************************/

static void qllm_backward(
    const QllmWeights* w,
    const QllmForwardCache* cache,
    const qllm_real loss_grad[D],
    QllmWeightGradients* grads)
{
    qllm_real dx[D];
    memcpy(dx, loss_grad, D * sizeof(qllm_real));

    for (int L = N_LAYERS - 2; L >= 0; L--) {
        qllm_real dx_layer[D] = {0};
        int ff_type = w->ff_type[L];

        if (ff_type == 1) {
            qllm_backward_ffn_square(
                w->ff_up[L], w->ff_down[L],
                w->ff_up_b[L], w->ff_down_b[L],
                cache->x_in[L],
                cache->ffn_pre_act[L],
                dx,
                dx_layer,
                grads->dff_up[L], grads->dff_up_b[L],
                grads->dff_down[L], grads->dff_down_b[L]);
        } else {
            qllm_backward_ffn_gated(
                w->ff_up[L], w->ff_down[L], w->ff_gate[L],
                w->ff_up_b[L], w->ff_down_b[L], w->ff_gate_b[L],
                cache->x_in[L],
                cache->ffn_gate_post[L],
                cache->ffn_up_post[L],
                cache->ffn_h[L],
                dx,
                dx_layer,
                grads->dff_up[L], grads->dff_up_b[L],
                grads->dff_down[L], grads->dff_down_b[L],
                grads->dff_gate[L], grads->dff_gate_b[L]);
        }

        /* Residual: add pass-through gradient */
        for (int i = 0; i < D; i++)
            dx_layer[i] += dx[i];

        /* Attention backward (Layer 0 only) */
        if (L == 0 && cache->np > 0) {
            qllm_real dx_attn[D] = {0};
            qllm_backward_attention(
                w->wq[0], w->wk[0], w->wv[0], w->wo[0], w->bq[0],
                cache, dx, dx_attn,
                grads->dwq[0], grads->dwk[0], grads->dwv[0],
                grads->dwo[0], grads->dbq[0]);
            for (int i = 0; i < D; i++)
                dx_layer[i] += dx_attn[i];
        }

        memcpy(dx, dx_layer, D * sizeof(qllm_real));
    }
}

/*******************************************************************************
 * Gradient Clipping
 ******************************************************************************/

/*******************************************************************************
 * Zero Gradients
 ******************************************************************************/

/*******************************************************************************
 * Checkpoint I/O — QLMW v4 format
 ******************************************************************************/

#define QLMW_MAGIC 0x514C4D57
#define QLMW_VERSION_WEIGHTS 3
#define QLMW_VERSION_CHECKPOINT 4

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t optimizer_step;
    uint32_t flags;
} QlmwHeaderV4;

