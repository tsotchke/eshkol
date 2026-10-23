/*
 * SDNC control fine-tuning with an exact forward pass.
 *
 * The constructed SDNC interpreter (lib/backend/weight_matrices.c) runs every
 * gate and every attention logit at a saturating scale, so its forward pass is
 * exact and the exact derivative of every control weight (gate thresholds,
 * gate columns, W_Q, W_K, b_Q) is identically zero. This facility fine-tunes
 * the control weights anyway, without ever softening the forward pass:
 *
 *   1. Teacher data are recorded transitions (state, program, next state) of
 *      the interpreter (SdncTransitionSet).
 *   2. The objective is a Huber loss (delta 1) over the transition outputs plus
 *      the exact count of transitions whose output is not byte-identical to the
 *      teacher (sdnc_transition_objective).
 *   3. Gate thresholds are parameterized per construction gate: neurons whose
 *      gate column and bias are identical in the construction are one tied
 *      gate (SdncGateGroups).
 *   4. The backward pass uses the logistic-perturbed gate
 *      (sdnc_backward_smoothed) to RANK threshold moves of one construction
 *      unit; the exact forward pass ACCEPTS a move only if it lowers the exact
 *      mismatch count (or keeps it and lowers the Huber loss). Tied gates
 *      first, then single neurons (sdnc_control_finetune).
 *
 * Design note: docs/design/adr/0032-exact-forward-control-fine-tuning.md.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_SDNC_FINETUNE_H
#define ESHKOL_SDNC_FINETUNE_H

#include <stddef.h>
#include "sdnc_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Recorded transitions ------------------------------------------------ */

/* One transition is (in, program, out): `out` is the forward output of the
 * teacher weights on `in` with the program's position embeddings. Programs are
 * stored once (consecutive transitions of one run share them). */
typedef struct {
    int     n, cap;          /* transitions */
    float*  in;              /* n * SDNC_D */
    float*  out;             /* n * SDNC_D */
    int*    prog;            /* n: program index */
    int     n_progs, cap_progs;
    int*    prog_len;        /* positions of each program (<= 256) */
    size_t* prog_row;        /* first row of each program in pe_rows */
    float*  pe_rows;         /* n_rows * SDNC_D position embeddings */
    size_t  n_rows, cap_rows;
} SdncTransitionSet;

void sdnc_transitions_init(SdncTransitionSet* ts);
void sdnc_transitions_free(SdncTransitionSet* ts);
/* Append one transition. `pe` holds np rows (np <= 256). Returns 0 or -1. */
int  sdnc_transitions_add(SdncTransitionSet* ts, const float* state,
                          const float* pe, int np, const float* next);
/* Binary transition file ("SDTR" v1). Return 0 on success, -1 on failure. */
int  sdnc_transitions_save(const SdncTransitionSet* ts, const char* path);
int  sdnc_transitions_load(SdncTransitionSet* ts, const char* path);

/* Weights in the QLMW v3 layout written by weight_matrices
 * (ESHKOL_WEIGHTS_OUT): a 28-byte header, then the raw weight struct.
 * Returns 0 on success, -1 on I/O failure, -2 on an architecture mismatch. */
int  sdnc_weights_load_qlmw(SdncWeights* w, const char* path);

/* ---- Objective ------------------------------------------------------------ */

/* Huber loss (delta 1) summed over every output element of every transition;
 * `*mismatched` (may be NULL) receives the number of transitions whose output
 * is not byte-identical to the recorded one. */
double sdnc_transition_objective(const SdncWeights* w,
                                 const SdncTransitionSet* ts, int* mismatched);

/* ---- Tied construction gates ---------------------------------------------- */

/* group[L][i] = tied-gate index of neuron i of gated layer L, or -1 when the
 * neuron has no gate (all-zero gate column and bias, or a non-gated layer). */
typedef struct {
    int n_groups;
    int group[SDNC_N_LAYERS][SDNC_FFN_DIM];
} SdncGateGroups;

/* Group neurons whose gate column and gate bias are equal. Build the groups
 * from the CONSTRUCTION, not from the weights being repaired: the thresholds
 * of neighbouring construction gates differ by whole construction units, so a
 * threshold moved by a unit can coincide with another gate's and would be tied
 * to it. */
void sdnc_gate_groups_build(const SdncWeights* w, SdncGateGroups* out);

/* ---- Optimizer ------------------------------------------------------------ */

typedef struct {
    SdncBackwardSmoothing smoothing; /* backward only; forward is exact */
    float step;                /* threshold move (construction unit) */
    int   gate_candidates;     /* top tied gates tried per move */
    int   gate_moves_max;
    int   neuron_candidates;   /* top neurons tried per move */
    int   neuron_moves_max;
    void (*log)(void* ud, const char* line);   /* optional progress lines */
    void* log_ud;
} SdncFinetuneOptions;

typedef struct {
    int    transitions;
    int    tied_gates;
    int    mismatched_before, mismatched_after;
    double objective_before, objective_after;
    int    gate_moves, neuron_moves;
    int    exact_control_gradient_zero; /* 1: the beta = 1 derivative of every
                                           control weight was exactly 0 */
} SdncFinetuneReport;

/* Defaults for the constructed interpreter: step = SDNC_CONSTRUCTION_UNIT,
 * backward beta = 10 / SDNC_CONSTRUCTION_UNIT for gates and attention,
 * 16 tied-gate candidates, 64 neuron candidates, at most 64 moves per level. */
void sdnc_finetune_default_options(SdncFinetuneOptions* opt);

/* Accumulate the gradient of the Huber objective over the mismatched
 * transitions with the given backward smoothing. */
void sdnc_control_gradient(const SdncWeights* w, const SdncTransitionSet* ts,
                           SdncBackwardSmoothing sm, SdncGrads* g);

/* 1 iff every control-weight entry of g (dwq, dwk, dbq, dff_gate, dff_gate_b)
 * is exactly zero. */
int  sdnc_control_gradient_is_zero(const SdncGrads* g);

/* Fine-tune the gate thresholds of `w` in place against `ts`, moving the tied
 * gates of `groups` (built from the construction) and then their single
 * neurons. `opt` may be NULL (defaults). Returns 0 on success, -1 on
 * allocation failure, -2 when `groups` is NULL. */
int  sdnc_control_finetune(SdncWeights* w, const SdncTransitionSet* ts,
                           const SdncGateGroups* groups,
                           const SdncFinetuneOptions* opt,
                           SdncFinetuneReport* rep);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_SDNC_FINETUNE_H */
