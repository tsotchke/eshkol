/*
 * Active Inference Runtime for Eshkol Consciousness Engine
 *
 * Implements core probabilistic inference primitives:
 * - Factor graphs with tensor-based CPTs
 * - Loopy belief propagation (sum-product algorithm)
 * - Variational free energy computation
 * - Expected free energy for action selection
 *
 * All objects are arena-allocated with object headers.
 * Tagged value dispatch follows the bignum pattern.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_INFERENCE_H
#define ESHKOL_CORE_INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct arena arena_t;
struct eshkol_tagged_value;
typedef struct eshkol_tagged_value eshkol_tagged_value_t;

/*
 * Factor: connects a set of variables via a conditional probability table (CPT).
 * The CPT is a tensor of log-probabilities indexed by variable states.
 * Layout: [eshkol_object_header_t][eshkol_factor_t][var_indices...]
 *
 * CPT indexing: for variables with dims d0, d1, ..., dn, the CPT is a
 * flattened array of size d0*d1*...*dn. Index = sum(state_i * stride_i).
 */
typedef struct eshkol_factor {
    uint32_t num_vars;         /* Number of connected variables */
    uint32_t cpt_size;         /* Total number of entries in CPT */
    double*  cpt;              /* Pointer to log-probability tensor (arena-allocated) */
    uint32_t* dims;            /* Dimension of each connected variable's state space */
    /* Followed by: uint32_t var_indices[num_vars] */
} eshkol_factor_t;

/* Access the var_indices array (immediately after the struct) */
#define FACTOR_VAR_INDICES(f) ((uint32_t*)((uint8_t*)(f) + sizeof(eshkol_factor_t)))

/*
 * Factor Graph: collection of factors over discrete random variables.
 * Beliefs are vectors of log-probabilities, one per variable.
 *
 * Message passing uses the sum-product algorithm in log-space
 * for numerical stability (log-sum-exp pattern).
 */
typedef struct eshkol_factor_graph {
    uint32_t num_vars;         /* Total number of random variables */
    uint32_t num_factors;      /* Number of factors currently added */
    uint32_t max_factors;      /* Capacity of factors array */
    uint32_t _pad;
    double** beliefs;          /* beliefs[i] = log-probability vector for var i */
    uint32_t* var_dims;        /* var_dims[i] = number of states for var i */
    eshkol_factor_t** factors; /* Array of factor pointers */
    double** msg_fv;           /* Messages from factors to variables */
    double** msg_vf;           /* Messages from variables to factors */
    uint32_t total_messages;   /* Total number of factor-variable edges */
    bool* observed;            /* observed[i] = true → clamped, skip during BP */
} eshkol_factor_graph_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Factor Graph Construction ===== */

/*
 * Create a factor graph with the given number of variables.
 * var_dims[i] = number of discrete states for variable i.
 * Beliefs are initialized to uniform (log(1/dim)).
 */
eshkol_factor_graph_t* eshkol_make_factor_graph(arena_t* arena,
    uint32_t num_vars, const uint32_t* var_dims);

/*
 * Create a factor connecting the specified variables with a CPT.
 * cpt_data is a flat array of log-probabilities.
 * dims[i] matches var_dims[var_indices[i]].
 */
eshkol_factor_t* eshkol_make_factor(arena_t* arena,
    const uint32_t* var_indices, uint32_t num_vars,
    const double* cpt_data, const uint32_t* dims);

/*
 * Add a factor to the graph. Factor graph grows as needed.
 */
void eshkol_fg_add_factor(arena_t* arena, eshkol_factor_graph_t* fg,
    eshkol_factor_t* factor);

/* ===== Inference (Belief Propagation) ===== */

/*
 * Loopy Belief Propagation (sum-product algorithm in log-space).
 * Updates fg->beliefs in-place.
 * Returns true if beliefs converged (max delta < tolerance).
 * max_iterations: upper bound on BP iterations.
 * tolerance: convergence threshold (typically 1e-6).
 */
bool eshkol_fg_infer(arena_t* arena, eshkol_factor_graph_t* fg,
    uint32_t max_iterations, double tolerance);

/* ===== Free Energy ===== */

/*
 * Variational Free Energy: F = E_q[ln q(s)] - E_q[ln p(o,s)]
 *
 * q(s) = current beliefs (product of marginals).
 * p(o,s) = generative model (factor graph potentials).
 * observations: array of (var_index, observed_state) pairs.
 *
 * Returns F as a scalar double. Lower is better (more accurate model).
 */
double eshkol_free_energy(const eshkol_factor_graph_t* fg,
    const double* observations, uint32_t num_obs);

/*
 * Expected Free Energy: G(a) = E_q(o,s|a)[ln q(s|a) - ln p(o,s)]
 *
 * Decomposes into:
 *   pragmatic value: how well action achieves goals
 *   epistemic value: how much action reduces uncertainty
 *
 * action_var: index of the action variable in the factor graph.
 * action_state: which discrete action to evaluate.
 * Returns G(a) as a scalar. Lower = more preferred action.
 */
double eshkol_expected_free_energy(arena_t* arena,
    const eshkol_factor_graph_t* fg,
    uint32_t action_var, uint32_t action_state);

/* ===== Tagged Value Dispatch ===== */
/* Called from LLVM codegen. Same alloca/store/call/load pattern as bignum. */

/**
 * @brief Tagged-value entry point for factor-graph construction, called from LLVM codegen.
 *
 * Unpacks @p num_vars (must be a tagged INT64) and @p var_dims_tensor (a
 * tensor `#(...)` or Scheme vector giving each variable's number of discrete
 * states) and forwards to eshkol_make_factor_graph(). On success @p result
 * is set to a HEAP_PTR tagged value wrapping the new graph; on a type
 * mismatch, missing argument, or allocation failure @p result is set to the
 * tagged NULL value.
 *
 * @param arena Arena used for all graph and belief allocations.
 * @param num_vars Tagged INT64: number of random variables.
 * @param var_dims_tensor Tagged tensor/vector of per-variable state counts.
 * @param[out] result Destination tagged value (HEAP_PTR on success, NULL on failure).
 */
void eshkol_make_factor_graph_tagged(arena_t* arena,
    const eshkol_tagged_value_t* num_vars,
    const eshkol_tagged_value_t* var_dims_tensor,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for adding a factor to a graph, called from LLVM codegen.
 *
 * Unpacks @p var_indices (tensor or vector of variable indices) and @p
 * cpt_tensor (tensor or vector of log-probabilities, with size equal to the
 * product of the connected variables' dimensions) and builds/attaches a new
 * factor to @p fg via eshkol_make_factor() and eshkol_fg_add_factor().
 *
 * @note @p fg is mutated in place; @p result is unconditionally set to the
 * tagged NULL value regardless of success (the Scheme builtin discards the
 * return value and relies on the in-place mutation).
 *
 * @param arena Arena used for factor and CPT allocations.
 * @param fg Tagged HEAP_PTR wrapping the target eshkol_factor_graph_t.
 * @param var_indices Tagged tensor/vector of variable indices the factor connects.
 * @param cpt_tensor Tagged tensor/vector of log-probabilities for the CPT.
 * @param[out] result Always set to the tagged NULL value.
 */
void eshkol_fg_add_factor_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg,
    const eshkol_tagged_value_t* var_indices,
    const eshkol_tagged_value_t* cpt_tensor,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for belief propagation, called from LLVM codegen.
 *
 * Runs eshkol_fg_infer() on @p fg with the given @p max_iters (defaults to
 * 20 if not a tagged INT64) and a fixed convergence tolerance of 1e-6. On
 * success, @p result is set to a HEAP_PTR wrapping a freshly allocated
 * one-dimensional tensor containing every variable's belief vector,
 * concatenated in variable order and converted from log-space back to
 * probabilities. Set to the tagged NULL value if @p fg is not a valid
 * factor-graph HEAP_PTR or allocation fails.
 *
 * @param arena Arena used for the result tensor and any BP scratch state.
 * @param fg Tagged HEAP_PTR wrapping the eshkol_factor_graph_t to run inference on.
 * @param max_iters Tagged INT64 iteration cap, or NULL/non-INT64 to use the default (20).
 * @param[out] result Destination tagged value (HEAP_PTR tensor of beliefs on success, NULL on failure).
 */
void eshkol_fg_infer_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg,
    const eshkol_tagged_value_t* max_iters,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_free_energy(), called from LLVM codegen.
 *
 * Extracts observation pairs (var_index, observed_state) from @p
 * observations_tensor (if given, otherwise treated as no observations) and
 * computes the variational free energy of @p fg_tv's current beliefs.
 * @p result is always set to a tagged (inexact) DOUBLE: 0.0 if @p fg_tv is
 * not a valid factor-graph HEAP_PTR, otherwise the computed free energy.
 *
 * @param fg_tv Tagged HEAP_PTR wrapping the eshkol_factor_graph_t to evaluate.
 * @param observations_tensor Optional tagged tensor of (var_index, state) pairs, or NULL.
 * @param[out] result Destination tagged DOUBLE holding the free-energy value.
 */
void eshkol_free_energy_tagged(
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* observations_tensor,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_expected_free_energy(), called from LLVM codegen.
 *
 * Extracts @p action_var and @p action_state (tagged INT64, defaulting to 0
 * if absent or of the wrong type) and evaluates the expected free energy of
 * that action under @p fg. @p result is always set to a tagged (inexact)
 * DOUBLE: 0.0 if @p fg is not a valid factor-graph HEAP_PTR, otherwise the
 * computed G(a) value.
 *
 * @param arena Arena used for scratch allocations during evaluation.
 * @param fg Tagged HEAP_PTR wrapping the eshkol_factor_graph_t to evaluate.
 * @param action_var Tagged INT64 index of the action variable.
 * @param action_state Tagged INT64 discrete action value to evaluate.
 * @param[out] result Destination tagged DOUBLE holding the expected-free-energy value.
 */
void eshkol_efe_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg,
    const eshkol_tagged_value_t* action_var,
    const eshkol_tagged_value_t* action_state,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for `fg-update-cpt!`, called from LLVM codegen.
 *
 * Replaces the CPT of factor @p factor_idx within @p fg with the values from
 * @p new_cpt (a tensor `#(...)` or Scheme vector, whose element count must
 * equal the factor's existing CPT size). On success, resets the graph's
 * cached factor-to-variable and variable-to-factor messages to NULL so the
 * next eshkol_fg_infer_tagged() call reconverges from scratch, and sets
 * @p result to a HEAP_PTR pointing back at the (mutated in place) @p fg.
 * Sets @p result to the tagged NULL value on any argument-shape mismatch,
 * out-of-range @p factor_idx, or size mismatch (an error is also logged via
 * eshkol_error() in the mismatch cases).
 *
 * @param arena Arena used for any error/side allocations.
 * @param fg Tagged HEAP_PTR wrapping the eshkol_factor_graph_t to update.
 * @param factor_idx Tagged INT64 or DOUBLE index of the factor to update.
 * @param new_cpt Tagged tensor/vector of replacement log-probabilities.
 * @param[out] result Destination tagged value (HEAP_PTR to @p fg on success, NULL on failure).
 */
void eshkol_fg_update_cpt_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg,
    const eshkol_tagged_value_t* factor_idx,
    const eshkol_tagged_value_t* new_cpt,
    eshkol_tagged_value_t* result);

/* ===== Display ===== */

/**
 * @brief Print a short human-readable summary of a factor graph.
 *
 * Writes `#<factor-graph: empty>` if @p fg is NULL, otherwise
 * `#<factor-graph: N factors, M vars>`. Used by the Scheme `display`/`write`
 * machinery for HEAP_SUBTYPE_FACTOR_GRAPH values.
 *
 * @param fg Factor graph to describe, or NULL.
 * @param file Destination `FILE*`, or NULL to write to stdout.
 */
void eshkol_display_factor_graph(const eshkol_factor_graph_t* fg, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_INFERENCE_H */
