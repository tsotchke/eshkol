/**
 * @file qllm_backward.c
 * @brief Backward pass + optimizer for qLLM computational transformer.
 *
 * Implements:
 *   - Per-dimension weighted MSE loss
 *   - Backward pass through 5-layer transformer (attention + 2 FFN types)
 *   - AdamW optimizer with cosine LR schedule
 *   - Gradient clipping
 *   - QLMW v4 checkpoint format (weights + optimizer state)
 *
 * The forward pass is in qllm_interpreter.c. This file provides the
 * training loop that learns weights from (state, target) pairs generated
 * by the reference interpreter in weight_matrices.c.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Architecture constants (must match qllm_interpreter.c) */
#ifndef D
#define D 36
#endif
#ifndef H
#define H 16
#endif
#ifndef HD
#define HD 2
#endif
#ifndef N_LAYERS
#define N_LAYERS 5
#endif
#ifndef FFN_DIM
#define FFN_DIM 512
#endif

/*******************************************************************************
 * Forward Cache — saved activations from forward pass for backward
 ******************************************************************************/

typedef struct {
    /* Per-layer input (after residual from prior layer) */
    float x_in[N_LAYERS][D];

    /* Attention (Layer 0 only) */
    float Q[D], K_all[256][D], V_all[256][D];
    float scores_raw[256];
    float attn_weights[256];
    float hout[D];
    int np;

    /* FFN activations per layer */
    float ffn_pre_act[N_LAYERS][FFN_DIM];   /* before activation */
    float ffn_gate_pre[N_LAYERS][FFN_DIM];  /* gate before sigmoid */
    float ffn_gate_post[N_LAYERS][FFN_DIM]; /* gate after sigmoid */
    float ffn_up_post[N_LAYERS][FFN_DIM];   /* up after bias */
    float ffn_h[N_LAYERS][FFN_DIM];         /* gate*up or h^2 */
    float ffn_out[N_LAYERS][D];             /* FFN output before residual */
} QllmForwardCache;

/*******************************************************************************
 * Weight Gradients — accumulator for all trainable parameters
 ******************************************************************************/

typedef struct {
    float dwq[N_LAYERS][D * D];
    float dwk[N_LAYERS][D * D];
    float dwv[N_LAYERS][D * D];
    float dwo[N_LAYERS][D * D];
    float dbq[N_LAYERS][D];
    float dff_up[N_LAYERS][D * FFN_DIM];
    float dff_up_b[N_LAYERS][FFN_DIM];
    float dff_down[N_LAYERS][FFN_DIM * D];
    float dff_down_b[N_LAYERS][D];
    float dff_gate[N_LAYERS][D * FFN_DIM];
    float dff_gate_b[N_LAYERS][FFN_DIM];
} QllmWeightGradients;

/*******************************************************************************
 * Loss Configuration
 ******************************************************************************/

typedef struct {
    float dim_weights[D];
} QllmLossConfig;

static void qllm_loss_config_default(QllmLossConfig* cfg) {
    for (int i = 0; i < D; i++) cfg->dim_weights[i] = 1.0f;
    cfg->dim_weights[0] = 2.0f;   /* PC */
    cfg->dim_weights[1] = 2.0f;   /* TOS */
    cfg->dim_weights[2] = 2.0f;   /* SOS */
    cfg->dim_weights[6] = 2.0f;   /* OUTPUT */
    cfg->dim_weights[7] = 5.0f;   /* HALT */
    cfg->dim_weights[14] = 3.0f;  /* HAS_OUT */
    cfg->dim_weights[28] = 3.0f;  /* IS_CALL */
    cfg->dim_weights[29] = 3.0f;  /* IS_RET */
    cfg->dim_weights[30] = 3.0f;  /* IS_NATIVE */
}

static float qllm_compute_loss(const float pred[D], const float target[D],
                                 const QllmLossConfig* cfg) {
    float loss = 0, wsum = 0;
    for (int i = 0; i < D; i++) {
        float d = pred[i] - target[i];
        loss += cfg->dim_weights[i] * d * d;
        wsum += cfg->dim_weights[i];
    }
    return loss / wsum;
}

static void qllm_compute_loss_grad(const float pred[D], const float target[D],
                                     const QllmLossConfig* cfg, float grad[D]) {
    float wsum = 0;
    for (int i = 0; i < D; i++) wsum += cfg->dim_weights[i];
    for (int i = 0; i < D; i++)
        grad[i] = 2.0f * cfg->dim_weights[i] * (pred[i] - target[i]) / wsum;
}

/*******************************************************************************
 * Backward Through FFN Type 1 (SQUARE activation)
 * Forward: h = x @ W_up + b_up; a = h^2; fo = a @ W_down + b_down
 ******************************************************************************/

static void qllm_backward_ffn_square(
    const float* w_up,   /* D x FFN_DIM */
    const float* w_down, /* FFN_DIM x D */
    const float* b_up,
    const float* b_down,
    const float x_in[D],
    const float h_pre[FFN_DIM],   /* h before squaring */
    const float dfo[D],           /* upstream gradient */
    float dx[D],                  /* output: gradient into input */
    float* dw_up,  float* db_up,
    float* dw_down, float* db_down)
{
    /* d(b_down) += dfo */
    for (int i = 0; i < D; i++) db_down[i] += dfo[i];

    /* da = dfo @ W_down^T (shape: FFN_DIM) */
    float da[FFN_DIM];
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
    float dh[FFN_DIM];
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

static void qllm_backward_ffn_gated(
    const float* w_up, const float* w_down, const float* w_gate,
    const float* b_up, const float* b_down, const float* b_gate,
    const float x_in[D],
    const float gate_post[FFN_DIM],  /* sigmoid output */
    const float up_post[FFN_DIM],    /* up = Wx + b */
    const float h[FFN_DIM],          /* gate * up */
    const float dfo[D],
    float dx[D],
    float* dw_up, float* db_up,
    float* dw_down, float* db_down,
    float* dw_gate, float* db_gate)
{
    /* d(b_down) += dfo */
    for (int i = 0; i < D; i++) db_down[i] += dfo[i];

    /* dh = dfo @ W_down^T */
    float dh_vec[FFN_DIM];
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
    float d_gate[FFN_DIM], d_up[FFN_DIM];
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
    float d_g_pre[FFN_DIM];
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
    float lr;
    float beta1, beta2;
    float epsilon;
    float weight_decay;
    float grad_clip_norm;
    int t;  /* timestep */
} QllmAdamWConfig;

static void qllm_adamw_config_default(QllmAdamWConfig* cfg) {
    cfg->lr = 1e-4f;
    cfg->beta1 = 0.9f;
    cfg->beta2 = 0.999f;
    cfg->epsilon = 1e-8f;
    cfg->weight_decay = 0.01f;
    cfg->grad_clip_norm = 1.0f;
    cfg->t = 0;
}

/* Apply one AdamW update to a single parameter array */
static void qllm_adamw_update(float* param, float* grad,
                                float* m, float* v,
                                int n, QllmAdamWConfig* cfg) {
    float bc1 = 1.0f - powf(cfg->beta1, (float)cfg->t);
    float bc2 = 1.0f - powf(cfg->beta2, (float)cfg->t);

    for (int i = 0; i < n; i++) {
        m[i] = cfg->beta1 * m[i] + (1.0f - cfg->beta1) * grad[i];
        v[i] = cfg->beta2 * v[i] + (1.0f - cfg->beta2) * grad[i] * grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] = param[i] * (1.0f - cfg->lr * cfg->weight_decay)
                   - cfg->lr * m_hat / (sqrtf(v_hat) + cfg->epsilon);
    }
}

/* Cosine LR schedule with warmup */
static float qllm_lr_schedule(int step, int warmup, int total,
                                float base_lr, float min_lr) {
    if (step < warmup) return base_lr * (float)step / (float)warmup;
    float progress = (float)(step - warmup) / (float)(total - warmup);
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(3.14159265f * progress));
}

/*******************************************************************************
 * Training Result
 ******************************************************************************/

typedef struct {
    float total_loss;
    int n_steps;
    int correct_outputs;
    int total_outputs;
} QllmTrainResult;
