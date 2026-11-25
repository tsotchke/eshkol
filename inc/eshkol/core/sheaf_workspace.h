/*
 * Sheaf-Cohomological & GNWT Global Workspace
 *
 * Implements:
 *   - Unbounded dynamic specialist/chart registry (zero constant-size cap)
 *   - Overlap restriction transport maps rho_{alpha->beta} in GL(d) / O(d)
 *   - Exact Cech 1-cocycle holonomy and H^1 obstruction detection around chart cycles
 *   - Unified H^0 global section gluing when [c] = 0
 *   - Perspective-preserving chart splitting + Dialectic specialist salience boost
 *     when H^1 != 0 (preventing naive single-chart averaging collapse)
 *   - GNWT ignition competition, refractory gating, and Attention Schema (AST)
 *     metacognitive surprise tracking
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_SHEAF_WORKSPACE_H
#define ESHKOL_CORE_SHEAF_WORKSPACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_ws_cohom_status {
    ESHKOL_WS_NO_IGNITION    = 0,
    ESHKOL_WS_GLUED_H0       = 1,
    ESHKOL_WS_H1_OBSTRUCTION = 2
} eshkol_ws_cohom_status_t;

typedef struct eshkol_sheaf_ws_config {
    double h1_threshold;       /* Holonomy / coboundary obstruction threshold, default 1e-4 */
    double ignition_threshold; /* Minimum salience for GNWT ignition, default 0.1 */
    uint32_t refractory_steps; /* Steps winner is gated after igniting, default 1 */
    double dialectic_boost;    /* Additive salience boost to dialectic specialists on H^1 != 0 */
    double ast_learning_rate;  /* EMA update rate for AST predictor, default 0.3 */
} eshkol_sheaf_ws_config_t;

typedef struct eshkol_sheaf_ws_step_result {
    eshkol_ws_cohom_status_t status;
    uint32_t winner_id;               /* Specialist id that ignited (UINT32_MAX if none) */
    double winner_effective_salience; /* Winning salience after any dialectic boost */
    double cech_holonomy_norm;        /* Max cycle holonomy ||(H_C - I) s|| */
    double cech_coboundary_residual;  /* Global coboundary residual ||delta s|| / ||s|| */
    double naive_average_norm;        /* Norm of naive single-chart parallel average */
    double glued_section_norm;        /* Norm of glued H^0 section (when GLUED_H0) */
    uint32_t num_split_charts;        /* Number of split branches preserved on H^1 != 0 */
    double split_branch_norms[2];     /* Norms of top 2 preserved split perspectives */
    double ast_predicted_prob;        /* AST predicted probability of actual winner */
    double ast_surprise;              /* Metacognitive surprise -log(ast_predicted_prob) */
} eshkol_sheaf_ws_step_result_t;

typedef struct eshkol_sheaf_workspace eshkol_sheaf_workspace_t;

eshkol_sheaf_ws_config_t eshkol_sheaf_ws_default_config(void);

/* Create an unbounded sheaf workspace with stalk dimension `stalk_dim`. */
eshkol_sheaf_workspace_t* eshkol_sheaf_ws_create(uint32_t stalk_dim,
                                                 const eshkol_sheaf_ws_config_t* cfg);

void eshkol_sheaf_ws_destroy(eshkol_sheaf_workspace_t* ws);

/* Register a specialist/chart module (unbounded count). Returns chart_id or UINT32_MAX. */
uint32_t eshkol_sheaf_ws_add_specialist(eshkol_sheaf_workspace_t* ws,
                                        const char* name,
                                        bool is_dialectic);

uint32_t eshkol_sheaf_ws_num_specialists(const eshkol_sheaf_workspace_t* ws);

/* Submit a local section proposal `section[0..stalk_dim-1]` and salience for chart `id`. */
bool eshkol_sheaf_ws_propose(eshkol_sheaf_workspace_t* ws,
                             uint32_t specialist_id,
                             const double* section,
                             uint32_t dim,
                             double salience);

/*
 * Define an overlap transport matrix `rho_ab` (row-major `stalk_dim x stalk_dim`)
 * mapping chart `from_id`'s stalk into chart `to_id`'s stalk on overlap U_a cap U_b.
 */
bool eshkol_sheaf_ws_add_transition(eshkol_sheaf_workspace_t* ws,
                                    uint32_t from_id,
                                    uint32_t to_id,
                                    const double* rho_matrix,
                                    uint32_t dim);

/* Execute one sheaf cohomology + chart-splitting + GNWT ignition + AST cycle. */
bool eshkol_sheaf_ws_step(eshkol_sheaf_workspace_t* ws,
                          eshkol_sheaf_ws_step_result_t* out_result);

/* Copy the glued H^0 broadcast section (or split branch `branch_idx` in {0,1} on H^1 != 0). */
bool eshkol_sheaf_ws_get_broadcast(const eshkol_sheaf_workspace_t* ws,
                                   uint32_t branch_idx,
                                   double* out_section,
                                   uint32_t dim);

/* Query remaining refractory countdown for specialist `id`. */
uint32_t eshkol_sheaf_ws_refractory_remaining(const eshkol_sheaf_workspace_t* ws,
                                              uint32_t specialist_id);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_SHEAF_WORKSPACE_H */
