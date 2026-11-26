/*
 * Incremental Distributed DBSP Z-Set Knowledge Graph Engine
 *
 * Implements sharded Z-sets over ground triples (subject, predicate, object) -> int64
 * multiplicity with:
 *   - Immediate zero-weight tombstone pruning (zero zombie nodes across churn)
 *   - Bilinear incremental equijoin Delta(A |><| B) = dA|><|B + A|><|dB + dA|><|dB
 *   - Deterministic canonical serialization & 64-bit hash for bit-identical replay
 *   - Custom merge hook for algebraic commutativity / order-sensitivity verification
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_DBSP_ZSET_KG_H
#define ESHKOL_CORE_DBSP_ZSET_KG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_kg_triple {
    uint64_t subject;
    uint64_t predicate;
    uint64_t object;
    int64_t weight; /* Signed Z-set multiplicity (0 pruned automatically) */
} eshkol_kg_triple_t;

typedef int64_t (*eshkol_zset_merge_fn_t)(int64_t existing_weight,
                                          int64_t delta_weight,
                                          void* user_ctx);

typedef struct eshkol_zset eshkol_zset_t;
typedef struct eshkol_dbsp_join_view eshkol_dbsp_join_view_t;

/* Create/destroy a sharded Z-set partitioned across `num_shards` (>= 1) shards. */
eshkol_zset_t* eshkol_zset_create(uint32_t num_shards);
void eshkol_zset_destroy(eshkol_zset_t* z);

/* Add/retract a single triple by signed multiplicity `delta` (prunes immediately at weight 0). */
bool eshkol_zset_add(eshkol_zset_t* z,
                     uint64_t subject, uint64_t predicate, uint64_t object,
                     int64_t delta);

/* Apply a Z-set delta `state += delta` (standard commutative Abelian group addition). */
bool eshkol_zset_apply_delta(eshkol_zset_t* state, const eshkol_zset_t* delta);

/* Apply a delta with a custom merge operator (e.g. for non-commutative order checks). */
bool eshkol_zset_apply_delta_custom(eshkol_zset_t* state,
                                    const eshkol_zset_t* delta,
                                    eshkol_zset_merge_fn_t merge_fn,
                                    void* user_ctx);

/* Query exact weight (0 if absent), active entry count, and shard count. */
int64_t eshkol_zset_weight(const eshkol_zset_t* z,
                           uint64_t subject, uint64_t predicate, uint64_t object);
size_t eshkol_zset_active_count(const eshkol_zset_t* z);
uint32_t eshkol_zset_num_shards(const eshkol_zset_t* z);

/* Full from-scratch equijoin A.object == B.subject emitting (A.subject, join_pred, B.object). */
eshkol_zset_t* eshkol_zset_join_full(const eshkol_zset_t* a,
                                     const eshkol_zset_t* b,
                                     uint64_t join_predicate,
                                     uint32_t num_shards);

/* Deterministic canonical little-endian serialization & 64-bit FNV-1a hash. */
bool eshkol_zset_canonical_serialize(const eshkol_zset_t* z,
                                     uint8_t** out_buf,
                                     size_t* out_len);
void eshkol_zset_buffer_free(uint8_t* buf);
uint64_t eshkol_zset_canonical_hash(const eshkol_zset_t* z);

/* ============================================================================
 * Incremental Bilinear DBSP Join View: J = A |><|_{A.o = B.s} B
 * ============================================================================ */

eshkol_dbsp_join_view_t* eshkol_dbsp_join_create(uint64_t join_predicate,
                                                 uint32_t num_shards);
void eshkol_dbsp_join_destroy(eshkol_dbsp_join_view_t* view);

/*
 * Step the incremental join view by applying deltas (delta_a, delta_b):
 *   delta_j = (delta_a |><| B) + (A |><| delta_b) + (delta_a |><| delta_b)
 *   A <- A + delta_a, B <- B + delta_b, J <- J + delta_j.
 * Optionally returns the incremental delta_j in `out_delta_j` (caller destroys).
 */
bool eshkol_dbsp_join_step(eshkol_dbsp_join_view_t* view,
                           const eshkol_zset_t* delta_a,
                           const eshkol_zset_t* delta_b,
                           eshkol_zset_t** out_delta_j);

const eshkol_zset_t* eshkol_dbsp_join_state_a(const eshkol_dbsp_join_view_t* view);
const eshkol_zset_t* eshkol_dbsp_join_state_b(const eshkol_dbsp_join_view_t* view);
const eshkol_zset_t* eshkol_dbsp_join_result(const eshkol_dbsp_join_view_t* view);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_DBSP_ZSET_KG_H */
