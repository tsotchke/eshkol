/**
 * @file qllm_backward.h
 * @brief Public surface for the SDNC weight-matrix backward pass.
 *
 * Companion training-mode module for the paper artefact
 * *"The Self-Differentiating Neural Computer: Computable Transformers via
 * Analytical Weight Construction"* (tsotchke, 2026). It provides the
 * reverse-mode (backward) pass through the six-layer computable-transformer
 * weight matrices whose forward semantics are the Eshkol VM ISA. Where the
 * analytical constructor in `lib/backend/weight_matrices.c` proves that
 * *fixed* weights execute the ISA, this module differentiates that same
 * architecture so the weights become trainable from (state, target) pairs.
 *
 * The FFN gradients are validated to relative error < 1e-6 against
 * a central finite-difference reference by
 * `tests/backend/qllm_backward_gradcheck_test.c`. The attention key and value
 * weight test compares against finite differences and the exact trainer VJP.
 *
 * Precision. The backward math is written against `qllm_real`, a compile-time
 * scalar type that defaults to `float` so the layout stays byte-compatible
 * with `InterpreterWeights` (QLMW). Building a translation unit with
 * `-DQLLM_REAL=double` instantiates the identical algorithm in double, which
 * is what the gradient-check test does to certify correctness below the
 * float32 finite-difference floor.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_BACKEND_QLLM_BACKWARD_H
#define ESHKOL_BACKEND_QLLM_BACKWARD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time scalar type. Default float => QLMW-compatible layout. */
#ifndef QLLM_REAL
#define QLLM_REAL float
#endif
typedef QLLM_REAL qllm_real;

/* Architecture constants — single source of truth shared by the backward
 * implementation and its consumers (must match weight_matrices.c). */
#ifndef D
#define D 256
#endif
#ifndef H
#define H 16
#endif
#ifndef HD
#define HD 2
#endif
#ifndef N_LAYERS
#define N_LAYERS 6
#endif
#ifndef FFN_DIM
#define FFN_DIM 2304
#endif

/* Accumulate the Layer 0 attention VJP from its forward intermediates.
 * Positions, keys and values have np rows. The weight layout is output-major. */
void qllm_backward_attention(
    int np, const qllm_real positions[][D], const qllm_real x[D],
    const qllm_real query[D], const qllm_real keys[][D], const qllm_real values[][D],
    const qllm_real probabilities[], const qllm_real head[D],
    const qllm_real wq[D * D], const qllm_real wo[D * D], const qllm_real upstream[D],
    qllm_real dx[D], qllm_real dwq[D * D], qllm_real dwk[D * D], qllm_real dwv[D * D],
    qllm_real dwo[D * D], qllm_real dbq[D]);

/*
 * Backward through FFN Type 1 (SQUARE activation).
 * Forward: h = x @ W_up + b_up ; a = h^2 ; fo = a @ W_down + b_down.
 * Accumulates parameter/input gradients (caller pre-zeroes the outputs).
 */
void qllm_backward_ffn_square(
    const qllm_real *w_up,
    const qllm_real *w_down,
    const qllm_real *b_up,
    const qllm_real *b_down,
    const qllm_real  x_in[D],
    const qllm_real  h_pre[FFN_DIM],
    const qllm_real  dfo[D],
    qllm_real        dx[D],
    qllm_real       *dw_up,  qllm_real *db_up,
    qllm_real       *dw_down, qllm_real *db_down);

/*
 * Backward through FFN Type 2 (gated-sigmoid).
 * Forward: gate = sigmoid(x @ W_gate + b_gate) ; up = x @ W_up + b_up ;
 *          h = gate * up ; fo = h @ W_down + b_down.
 * Accumulates parameter/input gradients (caller pre-zeroes the outputs).
 */
void qllm_backward_ffn_gated(
    const qllm_real *w_up, const qllm_real *w_down, const qllm_real *w_gate,
    const qllm_real *b_up, const qllm_real *b_down, const qllm_real *b_gate,
    const qllm_real  x_in[D],
    const qllm_real  gate_post[FFN_DIM],
    const qllm_real  up_post[FFN_DIM],
    const qllm_real  h[FFN_DIM],
    const qllm_real  dfo[D],
    qllm_real        dx[D],
    qllm_real       *dw_up, qllm_real *db_up,
    qllm_real       *dw_down, qllm_real *db_down,
    qllm_real       *dw_gate, qllm_real *db_gate);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_QLLM_BACKWARD_H */
