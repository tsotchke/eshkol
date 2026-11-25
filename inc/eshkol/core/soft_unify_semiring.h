/*
 * Certified Differentiable Unification, 12-Semiring Provenance & Pearl Level-3 SCM
 *
 * Implements:
 *   1. Euclidean x Poincare product-manifold soft-unify with exact analytical
 *      Euclidean and Riemannian gradients (zero finite-difference evaluations)
 *   2. Replayable Merkle-hashed PROOF_DAG for verifiable neuro-symbolic traces
 *   3. 12-Semiring algebraic provenance engine (Boolean, Viterbi, Sum-Product,
 *      Tropical, Max-Plus, Log-Space, Lukasiewicz, Godel, Product-Fuzzy,
 *      Belnap-4, NARS, Certified-Interval) + differentiable diff-top-k pullback
 *   4. Pearl Level-3 Structural Causal Model (SCM) engine supporting exact
 *      observational (Level 1), interventional do-calculus (Level 2), and
 *      counterfactual Abduction -> Action -> Prediction (Level 3) queries
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_SOFT_UNIFY_SEMIRING_H
#define ESHKOL_CORE_SOFT_UNIFY_SEMIRING_H

#include <eshkol/core/nars_truth.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Product-Manifold Soft Unification (R^{d_E} x Poincare Ball B_c^{d_H})
 * ============================================================================ */

typedef struct eshkol_soft_unify_config {
    double curvature;    /* Positive Poincare curvature c > 0 (sectional curvature -c) */
    double weight_eucl;  /* Non-negative Euclidean distance weight w_E */
    double weight_hyp;   /* Non-negative Poincare distance weight w_H */
    double temperature;  /* Positive kernel temperature tau > 0 */
} eshkol_soft_unify_config_t;

typedef struct eshkol_soft_unify_result {
    double score;        /* exp(-(w_E * d_E^2 + w_H * d_H^2) / tau) in (0, 1] */
    double eucl_dist_sq; /* ||e1 - e2||^2 */
    double hyp_dist;     /* Geodesic Poincare distance d_H(h1, h2) */
    uint64_t fd_evals;   /* Always 0 (exact analytical gradients) */
    bool valid;
} eshkol_soft_unify_result_t;

eshkol_soft_unify_config_t eshkol_soft_unify_default_config(void);

/*
 * Evaluate product-manifold soft-unify similarity and exact analytical gradients
 * w.r.t. e1, e2 (Euclidean) and h1, h2 (both ambient Euclidean and Riemannian
 * Poincare gradients when non-NULL). Returns false if any Poincare point lies
 * on or outside the ball boundary (c * ||h||^2 >= 1) or parameters are invalid.
 */
bool eshkol_soft_unify_eval(const double* e1, const double* e2, uint32_t dim_e,
                            const double* h1, const double* h2, uint32_t dim_h,
                            const eshkol_soft_unify_config_t* cfg,
                            eshkol_soft_unify_result_t* out_result,
                            double* grad_e1, double* grad_e2,
                            double* grad_h1_eucl, double* grad_h2_eucl,
                            double* grad_h1_riem, double* grad_h2_riem);

/* ============================================================================
 * 2. 12-Semiring Provenance Engine & Differentiable diff-top-k
 * ============================================================================ */

typedef enum eshkol_semiring_kind {
    ESHKOL_SR_BOOLEAN            = 0,  /* ({0,1}, or, and, 0, 1) */
    ESHKOL_SR_VITERBI            = 1,  /* ([0,1], max, *, 0, 1) */
    ESHKOL_SR_SUM_PRODUCT        = 2,  /* (R>=0, +, *, 0, 1) */
    ESHKOL_SR_TROPICAL           = 3,  /* (R u {+inf}, min, +, +inf, 0) */
    ESHKOL_SR_MAX_PLUS           = 4,  /* (R u {-inf}, max, +, -inf, 0) */
    ESHKOL_SR_LOG_SPACE          = 5,  /* (R u {-inf}, logsumexp, +, -inf, 0) */
    ESHKOL_SR_LUKASIEWICZ        = 6,  /* ([0,1], max, max(0, a+b-1), 0, 1) */
    ESHKOL_SR_GODEL              = 7,  /* ([0,1], max, min, 0, 1) */
    ESHKOL_SR_PRODUCT_FUZZY      = 8,  /* ([0,1], a+b-ab, a*b, 0, 1) */
    ESHKOL_SR_BELNAP4            = 9,  /* 4-valued Belnap bilattice {N,F,T,B} */
    ESHKOL_SR_NARS               = 10, /* NARS AIKR (frequency, confidence) */
    ESHKOL_SR_CERTIFIED_INTERVAL = 11  /* Outward-rounded [lo, hi] >= 0 */
} eshkol_semiring_kind_t;

typedef enum eshkol_belnap_val {
    ESHKOL_BELNAP_NEITHER = 0, /* Unknown / gap */
    ESHKOL_BELNAP_FALSE   = 1,
    ESHKOL_BELNAP_TRUE    = 2,
    ESHKOL_BELNAP_BOTH    = 3  /* Contradiction / glut */
} eshkol_belnap_val_t;

typedef struct eshkol_semiring_val {
    eshkol_semiring_kind_t kind;
    double a;       /* Primary scalar / frequency / interval lower */
    double b;       /* Secondary scalar / confidence / interval upper */
    uint32_t disc;  /* Discrete carrier for Belnap-4 */
} eshkol_semiring_val_t;

eshkol_semiring_val_t eshkol_sr_zero(eshkol_semiring_kind_t kind);
eshkol_semiring_val_t eshkol_sr_one(eshkol_semiring_kind_t kind);
eshkol_semiring_val_t eshkol_sr_scalar(eshkol_semiring_kind_t kind, double v);
eshkol_semiring_val_t eshkol_sr_nars(double frequency, double confidence);
eshkol_semiring_val_t eshkol_sr_interval(double lo, double hi);
eshkol_semiring_val_t eshkol_sr_belnap(eshkol_belnap_val_t v);

bool eshkol_sr_add(eshkol_semiring_val_t x, eshkol_semiring_val_t y,
                   eshkol_semiring_val_t* out);
bool eshkol_sr_mul(eshkol_semiring_val_t x, eshkol_semiring_val_t y,
                   eshkol_semiring_val_t* out);
bool eshkol_sr_equal(eshkol_semiring_val_t x, eshkol_semiring_val_t y, double tol);

/*
 * Smooth differentiable top-k selection:
 * Solves sum_i sigmoid((scores[i] - lambda) / tau) = k to produce soft masks
 * out_probs[i] in (0, 1) summing to k, plus exact analytical implicit pullback
 * dL/d(scores) given upstream cotangents dL/d(probs).
 */
bool eshkol_diff_topk_forward(const double* scores,
                              uint32_t n,
                              double k,
                              double temperature,
                              double* out_probs,
                              double* out_threshold);

bool eshkol_diff_topk_backward(const double* probs,
                               const double* dL_dprobs,
                               uint32_t n,
                               double temperature,
                               double* out_dL_dscores,
                               uint64_t* out_fd_evals);

/* ============================================================================
 * 3. Replayable Merkle-Hashed PROOF_DAG
 * ============================================================================ */

typedef enum eshkol_proof_node_kind {
    ESHKOL_PROOF_AXIOM         = 0,
    ESHKOL_PROOF_SOFT_UNIFY    = 1,
    ESHKOL_PROOF_SEMIRING_ADD  = 2,
    ESHKOL_PROOF_SEMIRING_MUL  = 3,
    ESHKOL_PROOF_DERIVED       = 4
} eshkol_proof_node_kind_t;

typedef struct eshkol_proof_dag eshkol_proof_dag_t;

eshkol_proof_dag_t* eshkol_proof_dag_create(void);
void eshkol_proof_dag_destroy(eshkol_proof_dag_t* dag);

uint32_t eshkol_proof_dag_add_node(eshkol_proof_dag_t* dag,
                                   eshkol_proof_node_kind_t kind,
                                   const char* label,
                                   const uint32_t* parents,
                                   uint32_t num_parents,
                                   eshkol_semiring_val_t value);

uint32_t eshkol_proof_dag_num_nodes(const eshkol_proof_dag_t* dag);
uint64_t eshkol_proof_dag_node_hash(const eshkol_proof_dag_t* dag, uint32_t node_id);
bool eshkol_proof_dag_get_value(const eshkol_proof_dag_t* dag,
                                uint32_t node_id,
                                eshkol_semiring_val_t* out_val);

/* Tamper helper for negative controls: corrupt stored value without updating hash. */
bool eshkol_proof_dag_corrupt_value_for_test(eshkol_proof_dag_t* dag,
                                             uint32_t node_id,
                                             double new_a);

/* Deterministic topological replay verifying every Merkle hash & semiring derivation. */
bool eshkol_proof_dag_verify_replay(const eshkol_proof_dag_t* dag);

/* ============================================================================
 * 4. Pearl Level-3 Structural Causal Model (SCM) & Counterfactual Engine
 * ============================================================================ */

/* Deterministic structural equation callback: given parent values and exogenous values,
 * returns discrete state of this endogenous variable in [0, domain_size - 1]. */
typedef uint32_t (*eshkol_scm_fn_t)(const uint32_t* parent_vals,
                                    uint32_t num_parents,
                                    const uint32_t* exog_vals,
                                    uint32_t num_exog,
                                    void* user_ctx);

typedef struct eshkol_scm eshkol_scm_t;

eshkol_scm_t* eshkol_scm_create(void);
void eshkol_scm_destroy(eshkol_scm_t* scm);

/* Add an independent exogenous noise/confounder variable U with discrete prior probs. */
uint32_t eshkol_scm_add_exogenous(eshkol_scm_t* scm,
                                  const char* name,
                                  const double* prior_probs,
                                  uint32_t domain_size);

/* Add an endogenous variable V_i = fn(parents, exog_vars) in topological order. */
uint32_t eshkol_scm_add_endogenous(eshkol_scm_t* scm,
                                   const char* name,
                                   uint32_t domain_size,
                                   const uint32_t* parent_endog,
                                   uint32_t num_parents,
                                   const uint32_t* exog_vars,
                                   uint32_t num_exog,
                                   eshkol_scm_fn_t fn,
                                   void* user_ctx);

/* Level 1 (Association): exact observational probability P(target = val | cond = cond_vals). */
bool eshkol_scm_observational_query(const eshkol_scm_t* scm,
                                    uint32_t target_var,
                                    uint32_t target_val,
                                    const uint32_t* cond_vars,
                                    const uint32_t* cond_vals,
                                    uint32_t num_cond,
                                    double* out_prob);

/* Level 2 (Intervention): exact interventional probability P(target = val | do(do_vars = do_vals)). */
bool eshkol_scm_do_query(const eshkol_scm_t* scm,
                         uint32_t target_var,
                         uint32_t target_val,
                         const uint32_t* do_vars,
                         const uint32_t* do_vals,
                         uint32_t num_do,
                         double* out_prob);

/* Level 3 (Counterfactual): Abduction -> Action -> Prediction
 * Computes exact P(target_{do(do_vars = do_vals)} = target_val | obs_vars = obs_vals). */
bool eshkol_scm_counterfactual(const eshkol_scm_t* scm,
                               uint32_t target_var,
                               uint32_t target_val,
                               const uint32_t* do_vars,
                               const uint32_t* do_vals,
                               uint32_t num_do,
                               const uint32_t* obs_vars,
                               const uint32_t* obs_vals,
                               uint32_t num_obs,
                               double* out_prob);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_SOFT_UNIFY_SEMIRING_H */
