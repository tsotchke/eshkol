/*
 * Linearizable Quorum-Fenced Distributed Cognitive Checkpoints
 *
 * Pure userspace replicated checkpoint cluster implementing strictly monotonic
 * fencing epochs and strict majority quorum (> N/2) two-phase prepare/commit
 * over an in-process simulated partitionable transport (zero kernel IPC).
 * Guarantees zero split-brain forks under asymmetric (3 vs 2), symmetric
 * (2 vs 2), and stale-leader fault scenarios.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_FENCED_COGNITIVE_CKPT_H
#define ESHKOL_CORE_FENCED_COGNITIVE_CKPT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_ckpt_status {
    ESHKOL_CKPT_OK                     = 0,
    ESHKOL_CKPT_ERR_INVALID_ARG        = 1,
    ESHKOL_CKPT_ERR_NO_QUORUM          = 2,
    ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN = 3
} eshkol_ckpt_status_t;

typedef struct eshkol_fenced_ckpt_cluster eshkol_fenced_ckpt_cluster_t;

/* Create a pure-userspace checkpoint cluster with `num_nodes` replicas (>= 1). */
eshkol_fenced_ckpt_cluster_t* eshkol_ckpt_cluster_create(uint32_t num_nodes);
void eshkol_ckpt_cluster_destroy(eshkol_fenced_ckpt_cluster_t* cluster);

uint32_t eshkol_ckpt_cluster_size(const eshkol_fenced_ckpt_cluster_t* cluster);
uint32_t eshkol_ckpt_cluster_quorum_size(const eshkol_fenced_ckpt_cluster_t* cluster);

/*
 * Partition the userspace transport into two isolated components:
 *   Group A: node_id < split_count
 *   Group B: node_id >= split_count
 * Messages across groups are dropped until `eshkol_ckpt_cluster_heal`.
 */
bool eshkol_ckpt_cluster_partition(eshkol_fenced_ckpt_cluster_t* cluster,
                                   uint32_t split_count);

/* Heal all network partitions and linearly synchronize replicas to highest committed state. */
bool eshkol_ckpt_cluster_heal(eshkol_fenced_ckpt_cluster_t* cluster);

/*
 * Execute a two-phase quorum-fenced cognitive checkpoint proposal from
 * `leader_node` at monotonic fencing epoch `epoch`.
 *   - Rejects stale/fenced epochs with ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN.
 *   - Rejects minority/split clusters with ESHKOL_CKPT_ERR_NO_QUORUM without
 *     committing on any replica.
 *   - Commits linearly only upon strict majority (> N/2) fenced prepare + commit.
 */
eshkol_ckpt_status_t eshkol_ckpt_cluster_propose(
    eshkol_fenced_ckpt_cluster_t* cluster,
    uint32_t leader_node,
    uint64_t epoch,
    const uint8_t* payload,
    size_t payload_len);

/* Inspect a replica's committed epoch, SHA-256 payload digest, and payload length. */
bool eshkol_ckpt_node_state(const eshkol_fenced_ckpt_cluster_t* cluster,
                            uint32_t node_id,
                            uint64_t* out_committed_epoch,
                            uint64_t* out_promised_epoch,
                            uint8_t out_sha256[32],
                            size_t* out_payload_len);

/*
 * Count detected linearizability forks across the cluster's entire commit
 * history (two distinct digests committed at the same epoch, or post-heal
 * divergence). Must remain strictly 0.
 */
uint64_t eshkol_ckpt_cluster_fork_count(const eshkol_fenced_ckpt_cluster_t* cluster);

/* Check whether all replicas currently hold bit-identical committed state. */
bool eshkol_ckpt_cluster_all_converged(const eshkol_fenced_ckpt_cluster_t* cluster);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_FENCED_COGNITIVE_CKPT_H */
