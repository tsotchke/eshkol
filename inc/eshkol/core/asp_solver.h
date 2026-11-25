/*
 * Native Answer Set Programming (ASP) Solver (Stable Model Semantics)
 *
 * Implements ground ASP solving atop Eshkol's native CDCL SAT solver:
 *   - Normal rules (head :- pos_body, not neg_body)
 *   - Integrity constraints (:- pos_body, not neg_body)
 *   - Choice rules ({h_1, ..., h_k}[lower..upper] :- pos_body, not neg_body)
 *   - Clark completion + exact Gelfond-Lifschitz unfounded-set detection and
 *     ASSAT loop-formula elimination
 *   - Complete enumeration of all stable models with deterministic budget limits
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_ASP_SOLVER_H
#define ESHKOL_CORE_ASP_SOLVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_asp_status {
    ESHKOL_ASP_SAT             = 0,
    ESHKOL_ASP_UNSAT           = 1,
    ESHKOL_ASP_BUDGET_EXCEEDED = 2,
    ESHKOL_ASP_INVALID_INPUT   = 3
} eshkol_asp_status_t;

typedef struct eshkol_asp_config {
    uint32_t max_models;        /* 0 = enumerate all stable models */
    uint64_t max_sat_conflicts; /* Cumulative CDCL conflict budget (0 = unlimited) */
    uint32_t max_cegar_rounds;  /* Safety bound on loop-formula/enumeration rounds (0 = default 10000) */
} eshkol_asp_config_t;

typedef struct eshkol_asp_model {
    uint32_t num_true_atoms;
    uint32_t* true_atoms;       /* Sorted 1-based atom ids (owned by solver) */
} eshkol_asp_model_t;

typedef struct eshkol_asp_stats {
    uint32_t num_models;
    uint32_t loop_formulas_added;
    uint32_t sat_calls;
    uint64_t total_conflicts;
} eshkol_asp_stats_t;

typedef struct eshkol_asp_solver eshkol_asp_solver_t;

eshkol_asp_config_t eshkol_asp_default_config(void);

/* Create an ASP solver over ground atoms 1..num_atoms. */
eshkol_asp_solver_t* eshkol_asp_create(uint32_t num_atoms,
                                       const eshkol_asp_config_t* cfg);

void eshkol_asp_destroy(eshkol_asp_solver_t* solver);

/* Add a normal rule: `head :- pos[0..num_pos-1], not neg[0..num_neg-1]`. */
bool eshkol_asp_add_normal_rule(eshkol_asp_solver_t* solver,
                                uint32_t head,
                                const uint32_t* pos_body, uint32_t num_pos,
                                const uint32_t* neg_body, uint32_t num_neg);

/* Add an integrity constraint: `:- pos[0..num_pos-1], not neg[0..num_neg-1]`. */
bool eshkol_asp_add_constraint(eshkol_asp_solver_t* solver,
                               const uint32_t* pos_body, uint32_t num_pos,
                               const uint32_t* neg_body, uint32_t num_neg);

/*
 * Add a choice rule: `{heads[0..num_heads-1]}[min_choose..max_choose] :- pos, not neg`.
 * When `max_choose == UINT32_MAX`, upper bound defaults to `num_heads`.
 */
bool eshkol_asp_add_choice_rule(eshkol_asp_solver_t* solver,
                                const uint32_t* heads, uint32_t num_heads,
                                uint32_t min_choose, uint32_t max_choose,
                                const uint32_t* pos_body, uint32_t num_pos,
                                const uint32_t* neg_body, uint32_t num_neg);

/* Enumerate stable models via Clark completion + loop-formula elimination on CDCL SAT. */
eshkol_asp_status_t eshkol_asp_solve(eshkol_asp_solver_t* solver,
                                     eshkol_asp_stats_t* out_stats);

uint32_t eshkol_asp_num_models(const eshkol_asp_solver_t* solver);

const eshkol_asp_model_t* eshkol_asp_get_model(const eshkol_asp_solver_t* solver,
                                               uint32_t model_idx);

/* Independent Gelfond-Lifschitz stability checker for an arbitrary atom subset. */
bool eshkol_asp_verify_stable_model(const eshkol_asp_solver_t* solver,
                                    const uint32_t* true_atoms,
                                    uint32_t num_true_atoms);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_ASP_SOLVER_H */
