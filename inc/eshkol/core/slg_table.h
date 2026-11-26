/*
 * Native SLG-WAM Tabled Resolution & 3-Valued Well-Founded Semantics (WFS)
 *
 * Integrates directly with Eshkol CE-0 logic primitives:
 *   - eshkol_rule_t (HEAP_SUBTYPE_RULE = 14, ESHKOL_RULE_FLAG_TABLED)
 *   - eshkol_fact_t, eshkol_knowledge_base_t, persistent HAMT eshkol_substitution_t
 *
 * Capabilities:
 *   - Goal-directed SLG subgoal variant tabling with producer/consumer resumption
 *   - Guaranteed termination on left-recursive and cyclic positive rules
 *   - Exact 3-valued Well-Founded Semantics (TRUE / FALSE / UNKNOWN) for
 *     stratified and unstratified normal logic programs via Van Gelder's
 *     alternating closure operator with unfounded-set elimination
 *   - Deterministic AIKR step budget returning ESHKOL_SLG_FUEL_EXHAUSTED cleanly
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_SLG_TABLE_H
#define ESHKOL_CORE_SLG_TABLE_H

#include <eshkol/core/logic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_slg_truth {
    ESHKOL_SLG_FALSE   = 0,
    ESHKOL_SLG_UNKNOWN = 1,
    ESHKOL_SLG_TRUE    = 2
} eshkol_slg_truth_t;

typedef enum eshkol_slg_status {
    ESHKOL_SLG_OK             = 0,
    ESHKOL_SLG_FUEL_EXHAUSTED = 1,
    ESHKOL_SLG_INVALID_INPUT  = 2
} eshkol_slg_status_t;

typedef struct eshkol_slg_config {
    uint64_t max_steps;      /* AIKR fuel step budget (0 = unlimited) */
    uint32_t max_subgoals;   /* Safety limit on distinct subgoal entries (0 = unlimited) */
    uint32_t max_answers;    /* Safety limit on answers per subgoal (0 = unlimited) */
    uint32_t max_wfs_iters;  /* Max alternating closure iterations (0 = default 1024) */
} eshkol_slg_config_t;

typedef struct eshkol_slg_answer {
    eshkol_fact_t* fact;           /* Arena-allocated resolved answer fact */
    eshkol_substitution_t* subst;  /* Arena-allocated substitution for query variables */
    eshkol_slg_truth_t truth;      /* ESHKOL_SLG_TRUE or ESHKOL_SLG_UNKNOWN */
} eshkol_slg_answer_t;

typedef struct eshkol_slg_result {
    eshkol_slg_status_t status;
    uint32_t num_answers;
    eshkol_slg_answer_t* answers;  /* Arena-allocated array [num_answers] */
    uint64_t steps_used;
    uint32_t subgoals_tabulated;
    uint32_t consumer_resumptions;
    uint32_t wfs_iterations;
} eshkol_slg_result_t;

typedef struct eshkol_slg_engine eshkol_slg_engine_t;

/* Default SLG configuration (unlimited fuel unless caller sets max_steps). */
eshkol_slg_config_t eshkol_slg_default_config(void);

/* Create an SLG tabled resolution engine over @p arena and EDB @p kb. */
eshkol_slg_engine_t* eshkol_slg_create(arena_t* arena,
                                       const eshkol_knowledge_base_t* kb,
                                       const eshkol_slg_config_t* cfg);

/* Destroy the SLG engine (frees internal C++ tables; arena objects stay arena-owned). */
void eshkol_slg_destroy(eshkol_slg_engine_t* engine);

/* Register a CE-0 eshkol_rule_t with the SLG engine (copies pointer; arena-owned). */
bool eshkol_slg_add_rule(eshkol_slg_engine_t* engine, const eshkol_rule_t* rule);

/* Construct a negated body literal fact `not(inner)` compatible with eshkol_rule_t. */
eshkol_fact_t* eshkol_slg_make_negated_goal(arena_t* arena, const eshkol_fact_t* inner);

/* Inspect whether @p goal is a `not(inner)` negated body literal. */
bool eshkol_slg_is_negated_goal(const eshkol_fact_t* goal, const eshkol_fact_t** out_inner);

/*
 * Evaluate @p query via tabled SLG resolution (and 3-valued Well-Founded
 * Semantics whenever reachable rules use negation).
 * Populates @p out_result with arena-allocated answers and telemetry.
 */
eshkol_slg_status_t eshkol_slg_query(eshkol_slg_engine_t* engine,
                                     const eshkol_fact_t* query,
                                     eshkol_slg_result_t* out_result);

/*
 * Evaluate a single ground atom under the 3-valued Well-Founded Model,
 * setting @p out_truth to TRUE, FALSE, or UNKNOWN.
 */
eshkol_slg_status_t eshkol_slg_query_ground(eshkol_slg_engine_t* engine,
                                            const eshkol_fact_t* ground_atom,
                                            eshkol_slg_truth_t* out_truth,
                                            uint64_t* out_steps_used);

/* Clear cached subgoal tables (call after mutating the underlying KB). */
void eshkol_slg_clear_tables(eshkol_slg_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_SLG_TABLE_H */
