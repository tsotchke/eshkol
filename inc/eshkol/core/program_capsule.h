/*
 * ADR-0005 Differentiable Program-Capsule Foundations
 *
 * Cleanly decouples symbolic decision/arithmetic rules from compiled
 * differentiable capsule weights:
 *   - Strict mode rejects discontinuous hard-step rules (if g(x) > 0 then t else e)
 *     with ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE.
 *   - Smooth relaxation mode (temperature tau > 0) replaces hard gates with
 *     sigmoid(g(x) / tau) and computes exact analytical input and parameter
 *     gradients (zero finite-difference evaluations).
 *   - Compiled capsules deep-copy parameters so mutating capsule weights never
 *     alters the original symbolic rule until explicitly re-linked after
 *     verification.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_PROGRAM_CAPSULE_H
#define ESHKOL_CORE_PROGRAM_CAPSULE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_capsule_mode {
    ESHKOL_CAPSULE_MODE_STRICT = 0, /* Rejects hard-step discontinuities */
    ESHKOL_CAPSULE_MODE_SMOOTH = 1  /* Relaxes hard gates via sigmoid(g(x) / tau) */
} eshkol_capsule_mode_t;

typedef enum eshkol_capsule_status {
    ESHKOL_CAPSULE_OK                      = 0,
    ESHKOL_CAPSULE_ERR_INVALID_ARG         = 1,
    ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE  = 2,
    ESHKOL_CAPSULE_ERR_UNVERIFIED_RELINK   = 3
} eshkol_capsule_status_t;

typedef struct eshkol_sym_rule eshkol_sym_rule_t;
typedef struct eshkol_program_capsule eshkol_program_capsule_t;

/* Create/destroy an isolated symbolic rule accepting `input_dim` inputs. */
eshkol_sym_rule_t* eshkol_sym_rule_create(uint32_t input_dim);
void eshkol_sym_rule_destroy(eshkol_sym_rule_t* rule);

/*
 * Add a discontinuous decision branch:
 *   if (w_gate^T x + b_gate > 0) then (w_then^T x + b_then) else (w_else^T x + b_else)
 * Branch outputs sum across all branches in the rule.
 */
bool eshkol_sym_rule_add_step_branch(eshkol_sym_rule_t* rule,
                                     const double* w_gate, double b_gate,
                                     const double* w_then, double b_then,
                                     const double* w_else, double b_else);

/*
 * Add an inherently smooth continuous branch:
 *   alpha * tanh(w^T x + b) + c
 * Permitted in both STRICT and SMOOTH compilation modes.
 */
bool eshkol_sym_rule_add_smooth_branch(eshkol_sym_rule_t* rule,
                                       const double* w, double b,
                                       double alpha, double c);

/* Evaluate the exact logical symbolic rule (unaffected by compiled capsule mutations). */
double eshkol_sym_rule_eval(const eshkol_sym_rule_t* rule, const double* x);
uint32_t eshkol_sym_rule_input_dim(const eshkol_sym_rule_t* rule);

/*
 * Compile symbolic rule into a decoupled differentiable program capsule.
 * - STRICT mode fails with ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE if any step branch exists.
 * - SMOOTH mode requires tau > 0.0 and deep-copies all parameters into private capsule storage.
 */
eshkol_capsule_status_t eshkol_capsule_compile(
    const eshkol_sym_rule_t* rule,
    eshkol_capsule_mode_t mode,
    double tau,
    eshkol_program_capsule_t** out_capsule);

void eshkol_capsule_destroy(eshkol_program_capsule_t* capsule);

size_t eshkol_capsule_param_count(const eshkol_program_capsule_t* capsule);
bool eshkol_capsule_get_param(const eshkol_program_capsule_t* capsule,
                              size_t idx, double* out_val);
bool eshkol_capsule_set_param(eshkol_program_capsule_t* capsule,
                              size_t idx, double new_val);

/*
 * Evaluate compiled differentiable capsule value + exact analytical gradients
 * w.r.t. inputs (`out_input_grads`, optional) and parameters (`out_param_grads`, optional).
 * Always sets `*out_fd_evals = 0` (pure analytical differentiation).
 */
bool eshkol_capsule_eval_value_grad(const eshkol_program_capsule_t* capsule,
                                    const double* x,
                                    double* out_value,
                                    double* out_input_grads,
                                    double* out_param_grads,
                                    uint64_t* out_fd_evals);

/*
 * Explicitly re-link verified capsule weights back into the symbolic rule.
 * Refuses with ESHKOL_CAPSULE_ERR_UNVERIFIED_RELINK unless `verification_passed` is true.
 */
eshkol_capsule_status_t eshkol_capsule_relink_to_rule(
    const eshkol_program_capsule_t* capsule,
    eshkol_sym_rule_t* target_rule,
    bool verification_passed);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_PROGRAM_CAPSULE_H */
