/*
 * CSR Differentiable Forney Factor Graphs & Certified Free-Energy Enclosures
 *
 * Implements:
 *   - Compressed Sparse Row (CSR) discrete Forney factor graphs
 *   - Damped synchronous and residual-priority asynchronous log-space BP
 *   - Tree-Reweighted Belief Propagation (TRW-BP)
 *   - Exact Bethe Free Energy F_Bethe = sum_a (E[E_a] - H(b_a)) + sum_i (d_i - 1) H(b_i)
 *   - Exact analytic implicit gradients (Bethe envelope theorem + adjoint BP VJP)
 *     with zero finite-difference evaluations
 *   - Certified outward-rounded interval enclosures [F_L, F_U] bounding exact -log Z
 *     via TRW spanning-tree convex upper bounds and variational lower/upper duality
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_FORNEY_FG_H
#define ESHKOL_CORE_FORNEY_FG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_bp_schedule {
    ESHKOL_BP_SYNC           = 0,
    ESHKOL_BP_RESIDUAL_ASYNC = 1,
    ESHKOL_BP_TRW            = 2
} eshkol_bp_schedule_t;

typedef struct eshkol_bp_config {
    eshkol_bp_schedule_t schedule;
    uint32_t max_iterations;   /* Default 200 */
    double tolerance;          /* Default 1e-10 */
    double damping;            /* In (0, 1]: 1.0 = undamped, 0.5 = half damping */
} eshkol_bp_config_t;

typedef struct eshkol_bp_stats {
    bool converged;
    uint32_t iterations;
    uint64_t edge_updates;
    double max_residual;
    double bethe_free_energy;
    double trw_free_energy;
} eshkol_bp_stats_t;

/* Outward-rounded certified interval enclosure [lower, upper] for exact -log Z. */
typedef struct eshkol_fe_interval {
    double lower;              /* Rigorous lower bound F_L <= -log Z */
    double upper;              /* Rigorous upper bound F_U >= -log Z */
    double bethe_estimate;     /* Point Bethe free-energy estimate */
    uint32_t num_trees;        /* Number of spanning trees in TRW decomposition */
    bool valid;                /* True iff lower <= upper and both finite */
} eshkol_fe_interval_t;

typedef struct eshkol_forney_fg eshkol_forney_fg_t;

/* Return recommended default BP configuration. */
eshkol_bp_config_t eshkol_bp_default_config(void);

/* Create a CSR Forney factor graph with `num_vars` variables and state counts `var_dims`. */
eshkol_forney_fg_t* eshkol_forney_fg_create(uint32_t num_vars,
                                            const uint32_t* var_dims);

/* Destroy a Forney factor graph and release all internal buffers. */
void eshkol_forney_fg_destroy(eshkol_forney_fg_t* fg);

/*
 * Add a factor over `vars[0..arity-1]` with log-potentials `log_potentials[0..cpt_size-1]`.
 * Row-major indexing: index = sum_k state_k * stride_k (last var fastest).
 * Returns factor index, or UINT32_MAX on invalid input.
 */
uint32_t eshkol_forney_fg_add_factor(eshkol_forney_fg_t* fg,
                                     const uint32_t* vars,
                                     uint32_t arity,
                                     const double* log_potentials,
                                     uint32_t cpt_size);

/* Update log-potentials of an existing factor in place (invalidates converged messages). */
bool eshkol_forney_fg_set_factor_potentials(eshkol_forney_fg_t* fg,
                                            uint32_t factor_idx,
                                            const double* log_potentials,
                                            uint32_t cpt_size);

/* Set TRW appearance probability rho in (0, 1] for a specific factor. */
bool eshkol_forney_fg_set_factor_rho(eshkol_forney_fg_t* fg,
                                     uint32_t factor_idx,
                                     double rho);

/* Clamp (or unclamp when state < 0) a hard observation on variable `var_idx`. */
bool eshkol_forney_fg_observe(eshkol_forney_fg_t* fg,
                              uint32_t var_idx,
                              int32_t state);

/* Query graph topology sizes. */
uint32_t eshkol_forney_fg_num_vars(const eshkol_forney_fg_t* fg);
uint32_t eshkol_forney_fg_num_factors(const eshkol_forney_fg_t* fg);
uint32_t eshkol_forney_fg_total_params(const eshkol_forney_fg_t* fg);
uint32_t eshkol_forney_fg_factor_size(const eshkol_forney_fg_t* fg, uint32_t factor_idx);

/* Run Belief Propagation (sync, residual-priority async, or TRW-BP). */
bool eshkol_forney_fg_infer(eshkol_forney_fg_t* fg,
                            const eshkol_bp_config_t* cfg,
                            eshkol_bp_stats_t* out_stats);

/* Read converged marginal probabilities for variable `var_idx` (length >= var_dims[var_idx]). */
bool eshkol_forney_fg_var_marginal(const eshkol_forney_fg_t* fg,
                                   uint32_t var_idx,
                                   double* out_probs,
                                   uint32_t out_len);

/* Read converged joint marginal probabilities for factor `factor_idx`. */
bool eshkol_forney_fg_factor_marginal(const eshkol_forney_fg_t* fg,
                                      uint32_t factor_idx,
                                      double* out_probs,
                                      uint32_t out_len);

/* Compute Bethe Free Energy F_Bethe from current beliefs. */
double eshkol_forney_fg_bethe_free_energy(const eshkol_forney_fg_t* fg);

/* Compute exact -log Z by brute-force enumeration (for small oracle graphs <= 2^20 states). */
bool eshkol_forney_fg_exact_neg_log_z(const eshkol_forney_fg_t* fg,
                                      double* out_neg_log_z);

/*
 * Exact analytic gradient d(F_Bethe)/d(theta) via the Bethe envelope theorem
 * at converged BP beliefs (zero finite-difference evaluations).
 * Writes concatenated factor gradients into `out_grads[0..total_params-1]` and
 * sets `*out_fd_evals = 0` when non-NULL.
 */
bool eshkol_fg_free_energy_grad(const eshkol_forney_fg_t* fg,
                                double* out_grads,
                                uint32_t grads_len,
                                uint64_t* out_fd_evals);

/*
 * Exact implicit differentiation (adjoint BP VJP) for an arbitrary loss L(b_vars):
 * given upstream cotangents `dL_db_vars` (concatenated across all variables' states),
 * solves (I - J_BP^T) m_bar = (db/dm)^T b_bar to compute exact dL/d(theta) with
 * zero finite-difference evaluations.
 */
bool eshkol_fg_marginal_vjp(const eshkol_forney_fg_t* fg,
                            const double* dL_db_vars,
                            uint32_t var_states_len,
                            double* out_dL_dtheta,
                            uint32_t theta_len,
                            uint32_t max_vjp_iters,
                            double tol,
                            uint64_t* out_fd_evals);

/*
 * Compute a certified outward-rounded interval enclosure [F_L, F_U] bounding exact -log Z.
 * Uses TRW spanning-tree convex decomposition + outward-rounded tree elimination for F_L,
 * and outward-rounded variational Gibbs free energy (min of mean-field and spanning-tree) for F_U.
 */
bool eshkol_fg_certify_free_energy(eshkol_forney_fg_t* fg,
                                   const eshkol_bp_config_t* trw_cfg,
                                   eshkol_fe_interval_t* out_interval);

/* Validate whether a certified interval [lower, upper] encloses a reference value. */
bool eshkol_fg_interval_contains(const eshkol_fe_interval_t* interval, double value);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_FORNEY_FG_H */
