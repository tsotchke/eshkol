/*
 * Stratified Recursive Incremental View Maintenance (DRed / Differential DBSP)
 *
 * Implements:
 *   1. Compile-time stratification graph validator rejecting negative cycles and
 *      ungrounded cyclic dependencies without base cases via
 *      ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION.
 *   2. Exact incremental recursive reachability / transitive closure (DRed
 *      over-deletion + rederivation) handling arbitrary directed cycles and
 *      diamond paths with immediate zero-weight pruning (zero zombie entries).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_RECURSIVE_IVM_H
#define ESHKOL_CORE_RECURSIVE_IVM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_ivm_status {
    ESHKOL_IVM_OK                        = 0,
    ESHKOL_IVM_ERR_INVALID_ARG           = 1,
    ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION = 2
} eshkol_ivm_status_t;

typedef struct eshkol_ivm_strat_compiler eshkol_ivm_strat_compiler_t;
typedef struct eshkol_recursive_ivm eshkol_recursive_ivm_t;

/* ============================================================================
 * Part 1: Compile-Time Stratification Validator
 * ============================================================================ */

eshkol_ivm_strat_compiler_t* eshkol_ivm_strat_create(void);
void eshkol_ivm_strat_destroy(eshkol_ivm_strat_compiler_t* sc);

/* Register a relation ID; `has_base_case` marks an EDB or grounded base rule. */
bool eshkol_ivm_strat_add_relation(eshkol_ivm_strat_compiler_t* sc,
                                   uint32_t rel_id,
                                   bool has_base_case);

/* Register dependency edge: head_rel depends on body_rel (positive or negated). */
bool eshkol_ivm_strat_add_dependency(eshkol_ivm_strat_compiler_t* sc,
                                     uint32_t head_rel,
                                     uint32_t body_rel,
                                     bool is_negated);

/*
 * Compile & validate stratification via Tarjan SCC analysis:
 *   - Rejects any SCC containing a negative edge.
 *   - Rejects any cyclic SCC where no member relation has a grounded base case.
 * Returns ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION immediately on violation.
 */
eshkol_ivm_status_t eshkol_ivm_strat_compile(eshkol_ivm_strat_compiler_t* sc);

/* Query assigned stratum for a relation after successful compilation (UINT32_MAX if absent). */
uint32_t eshkol_ivm_strat_level(const eshkol_ivm_strat_compiler_t* sc, uint32_t rel_id);

/* ============================================================================
 * Part 2: Incremental Recursive Reachability View (Cycle-Safe DRed IVM)
 * ============================================================================ */

eshkol_recursive_ivm_t* eshkol_recursive_ivm_create(void);
void eshkol_recursive_ivm_destroy(eshkol_recursive_ivm_t* ivm);

/* Insert or delete a directed base edge (u -> v) with multiplicity delta (+1 / -1). */
bool eshkol_recursive_ivm_insert_edge(eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v);
bool eshkol_recursive_ivm_delete_edge(eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v);

/* Query derived reachability (non-empty directed path u ->+ v). */
bool eshkol_recursive_ivm_reaches(const eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v);

/* Active counts (all shrink strictly to 0 when all edges are deleted). */
size_t eshkol_recursive_ivm_active_edge_count(const eshkol_recursive_ivm_t* ivm);
size_t eshkol_recursive_ivm_active_node_count(const eshkol_recursive_ivm_t* ivm);
size_t eshkol_recursive_ivm_derived_pair_count(const eshkol_recursive_ivm_t* ivm);

/* Deterministic 64-bit FNV-1a canonical hash of the incremental derived reachability set. */
uint64_t eshkol_recursive_ivm_derived_hash(const eshkol_recursive_ivm_t* ivm);

/*
 * Independent from-scratch BFS reachability oracle computed purely from current
 * base edges; returns canonical hash and writes derived pair count to `*out_pair_count`.
 */
uint64_t eshkol_recursive_ivm_oracle_hash(const eshkol_recursive_ivm_t* ivm,
                                          size_t* out_pair_count);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_RECURSIVE_IVM_H */
