/*
 * Native CDCL SAT Solver, DRAT Proof Emitter, and Independent DRAT Verifier
 *
 * Implements MiniSat-lineage Conflict-Driven Clause Learning (CDCL):
 *   - Two-watched literals with blocker caching
 *   - 1-UIP conflict analysis with recursive clause minimization
 *   - VSIDS activity branching with binary max-heap and exponential decay
 *   - Phase saving and Luby geometric restarts
 *   - LBD-tiered clause database reduction
 *   - Deterministic AIKR fuel budgets (conflicts, propagations, decisions)
 *   - Structured DRAT proof recording (additions and deletions)
 *   - Zero-shared-state independent forward DRAT verifier (RUP + RAT)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_CDCL_SAT_H
#define ESHKOL_CORE_CDCL_SAT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_sat_status {
    ESHKOL_SAT_UNDECIDED       = 0,
    ESHKOL_SAT_SATISFIABLE     = 1,
    ESHKOL_SAT_UNSATISFIABLE   = 2,
    ESHKOL_SAT_BUDGET_EXCEEDED = 3,
    ESHKOL_SAT_INVALID_INPUT   = 4
} eshkol_sat_status_t;

typedef enum eshkol_drat_op {
    ESHKOL_DRAT_ADD    = 0,
    ESHKOL_DRAT_DELETE = 1
} eshkol_drat_op_t;

typedef struct eshkol_drat_step {
    eshkol_drat_op_t op;
    uint32_t num_lits;
    int32_t* lits; /* DIMACS literals (+v / -v, 1-based variable ids) */
} eshkol_drat_step_t;

typedef struct eshkol_drat_proof {
    uint32_t num_steps;
    uint32_t capacity;
    eshkol_drat_step_t* steps;
} eshkol_drat_proof_t;

typedef struct eshkol_cnf_clause {
    const int32_t* lits;
    uint32_t num_lits;
} eshkol_cnf_clause_t;

typedef struct eshkol_cdcl_config {
    uint64_t max_conflicts;     /* 0 = unlimited */
    uint64_t max_propagations;  /* 0 = unlimited */
    uint64_t max_decisions;     /* 0 = unlimited */
    double var_decay;           /* Default 0.95 */
    double clause_decay;        /* Default 0.999 */
    uint32_t restart_base;      /* Luby unit multiplier, default 100 */
    uint32_t max_learnts_init;  /* Initial learnt reduction threshold, default 2000 */
    double learntsize_inc;      /* Growth factor per reduction, default 1.1 */
    uint32_t lbd_tier1;         /* Permanent glue clause LBD threshold, default 2 */
    bool emit_drat;             /* Record DRAT additions & deletions, default true */
    bool phase_saving;          /* Reuse last polarity on backtrack, default true */
    bool minimize_clauses;      /* Recursive 1-UIP clause minimization, default true */
} eshkol_cdcl_config_t;

typedef struct eshkol_cdcl_stats {
    uint64_t conflicts;
    uint64_t propagations;
    uint64_t decisions;
    uint64_t restarts;
    uint64_t learned_clauses;
    uint64_t deleted_clauses;
    uint64_t minimized_lits;
    uint32_t drat_steps;
} eshkol_cdcl_stats_t;

typedef struct eshkol_cdcl_solver eshkol_cdcl_solver_t;

/* Return recommended default CDCL solver configuration. */
eshkol_cdcl_config_t eshkol_cdcl_default_config(void);

/* Create a new CDCL solver for variables 1..num_vars. */
eshkol_cdcl_solver_t* eshkol_cdcl_create(uint32_t num_vars,
                                         const eshkol_cdcl_config_t* cfg);

/* Destroy a CDCL solver and release all internal memory and proof buffers. */
void eshkol_cdcl_destroy(eshkol_cdcl_solver_t* solver);

/*
 * Add a clause of DIMACS literals (+v or -v, with 1 <= |lit| <= num_vars).
 * Returns false on invalid input (NULL solver, out-of-range literal, or 0 literal).
 */
bool eshkol_cdcl_add_clause(eshkol_cdcl_solver_t* solver,
                            const int32_t* lits,
                            uint32_t num_lits);

/* Run CDCL search to satisfiability, unsatisfiability, or budget exhaustion. */
eshkol_sat_status_t eshkol_cdcl_solve(eshkol_cdcl_solver_t* solver);

/*
 * Query model assignment for 1-based variable `var` after SATISFIABLE:
 * returns +1 for true, -1 for false, 0 if unavailable or out of range.
 */
int8_t eshkol_cdcl_model_value(const eshkol_cdcl_solver_t* solver, uint32_t var);

/* Copy full 1-based model into out_model[1..num_vars] (buffer length >= num_vars + 1). */
bool eshkol_cdcl_copy_model(const eshkol_cdcl_solver_t* solver,
                            int8_t* out_model,
                            uint32_t out_len);

/* Access the structured DRAT proof recorded by the solver (owned by solver). */
const eshkol_drat_proof_t* eshkol_cdcl_get_proof(const eshkol_cdcl_solver_t* solver);

/* Retrieve solver telemetry counters. */
void eshkol_cdcl_get_stats(const eshkol_cdcl_solver_t* solver,
                           eshkol_cdcl_stats_t* out_stats);

/*
 * Independent model checker: verifies every clause in `clauses[0..num_clauses-1]`
 * has at least one satisfied literal under 1-based `model[1..num_vars]`.
 */
bool eshkol_cdcl_verify_model(uint32_t num_vars,
                              const eshkol_cnf_clause_t* clauses,
                              uint32_t num_clauses,
                              const int8_t* model,
                              uint32_t model_len);

/*
 * Independent forward DRAT proof verifier (zero shared state with solver).
 * Checks every addition for RUP or RAT validity, applies deletions strictly,
 * and requires an explicit RUP-verified empty clause derivation.
 */
bool eshkol_cdcl_verify_drat(uint32_t num_vars,
                             const eshkol_cnf_clause_t* orig_clauses,
                             uint32_t num_orig_clauses,
                             const eshkol_drat_proof_t* proof);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_CDCL_SAT_H */
