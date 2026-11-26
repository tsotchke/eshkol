/*
 * Trajectory-Balance GFlowNet & Predictive Coding Network (PCN) Energy Core
 *
 * Implements:
 *   1. Trajectory-Balance (TB) GFlowNet over finite DAGs:
 *      - Exact analytical TB loss and gradients w.r.t. forward policy logits
 *        and log Z (zero finite-difference evaluations)
 *      - Exact topological terminal distribution and Total Variation distance
 *        to the target Boltzmann reward distribution R(x) / Z
 *      - Deterministic trajectory sampling and Adam optimizer step
 *   2. Predictive Coding Network (PCN) / Energy-Based Layer:
 *      - Multi-layer variational free-energy inference equilibrium with purely
 *        local symmetric nudged Hebbian weight updates (zero FD weight evals)
 *      - Exact analytical backpropagation reference comparator
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_GFLOWNET_EBM_H
#define ESHKOL_CORE_GFLOWNET_EBM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Trajectory-Balance (TB) GFlowNet Over Finite DAGs
 * ============================================================================ */

typedef struct eshkol_gfn_dag eshkol_gfn_dag_t;

eshkol_gfn_dag_t* eshkol_gfn_create(uint32_t num_states);
void eshkol_gfn_destroy(eshkol_gfn_dag_t* gfn);

/* Add directed forward edge u -> v with initial logit (must preserve DAG order u < v). */
uint32_t eshkol_gfn_add_edge(eshkol_gfn_dag_t* gfn,
                             uint32_t from_state,
                             uint32_t to_state,
                             double init_logit);

/* Mark state `s` as terminal with positive reward R(s) > 0. */
bool eshkol_gfn_set_terminal_reward(eshkol_gfn_dag_t* gfn,
                                    uint32_t state,
                                    double reward);

uint32_t eshkol_gfn_num_edges(const eshkol_gfn_dag_t* gfn);
double eshkol_gfn_get_log_z(const eshkol_gfn_dag_t* gfn);
void eshkol_gfn_set_log_z(eshkol_gfn_dag_t* gfn, double log_z);
bool eshkol_gfn_get_edge_logit(const eshkol_gfn_dag_t* gfn, uint32_t edge_id, double* out_logit);
bool eshkol_gfn_set_edge_logit(eshkol_gfn_dag_t* gfn, uint32_t edge_id, double logit);

/*
 * Evaluate exact Trajectory-Balance loss 0.5 * Delta(tau)^2 and exact analytical
 * gradients w.r.t. log_z and all edge logits for trajectory `edge_seq[0..len-1]`
 * starting at state 0 and ending at a terminal state (sets *out_fd_evals = 0).
 */
bool eshkol_gfn_tb_loss_and_grad(const eshkol_gfn_dag_t* gfn,
                                 const uint32_t* edge_seq,
                                 uint32_t seq_len,
                                 double* out_loss,
                                 double* out_grad_log_z,
                                 double* out_grad_logits,
                                 uint32_t logits_len,
                                 uint64_t* out_fd_evals);

/* Train GFlowNet for `num_steps` epochs using exact analytical TB gradients + Adam. */
bool eshkol_gfn_train(eshkol_gfn_dag_t* gfn,
                      uint32_t num_steps,
                      double lr_logits,
                      double lr_log_z,
                      uint64_t seed);

/* Compute exact induced terminal distribution p_theta(s) for all states [0..num_states-1]
 * and Total Variation distance TV(p_theta, R/Z) + exact log Z. */
bool eshkol_gfn_eval_distribution(const eshkol_gfn_dag_t* gfn,
                                  double* out_state_probs,
                                  uint32_t probs_len,
                                  double* out_tv_distance,
                                  double* out_exact_log_z);

/*
 * Per-state forward-vs-backward local TV calibration:
 * Computes local TV(P_F(.|s), Q_B(.|s)) for every non-terminal state s and
 * |F_F(s) - R(s)/Z| for every terminal state s, writing per-state values into
 * `out_state_tv[0..num_states-1]` (when non-NULL), plus max per-state local TV
 * and max terminal relative error |p(x) - R(x)/Z| / (R(x)/Z).
 */
bool eshkol_gfn_eval_per_state_tv(const eshkol_gfn_dag_t* gfn,
                                  double* out_state_tv,
                                  uint32_t tv_len,
                                  double* out_max_state_tv,
                                  double* out_max_terminal_rel_err);

/* ============================================================================
 * 2. Predictive Coding Network (PCN) / Energy-Based Layer
 * ============================================================================ */

typedef struct eshkol_pcn eshkol_pcn_t;

/* Create an L-layer PCN with layer widths `dims[0..num_layers-1]` (tanh activations). */
eshkol_pcn_t* eshkol_pcn_create(const uint32_t* layer_dims,
                                uint32_t num_layers,
                                uint64_t seed);

void eshkol_pcn_destroy(eshkol_pcn_t* pcn);

uint32_t eshkol_pcn_total_params(const eshkol_pcn_t* pcn);

/*
 * Compute weight & bias gradients via local Predictive Coding inference equilibrium
 * (symmetric weak nudging +-beta with local iterative relaxation + local Hebbian rule).
 * Zero finite differences over parameters (*out_fd_evals = 0).
 */
bool eshkol_pcn_local_hebbian_grad(const eshkol_pcn_t* pcn,
                                   const double* input,
                                   const double* target,
                                   double beta,
                                   uint32_t infer_steps,
                                   double infer_lr,
                                   double* out_grads,
                                   uint32_t grads_len,
                                   uint64_t* out_fd_evals);

/* Compute exact analytical backpropagation gradients of 0.5 * ||y_hat - target||^2. */
bool eshkol_pcn_exact_backprop_grad(const eshkol_pcn_t* pcn,
                                    const double* input,
                                    const double* target,
                                    double* out_grads,
                                    uint32_t grads_len,
                                    double* out_loss);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_GFLOWNET_EBM_H */
