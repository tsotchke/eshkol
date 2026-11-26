/*
 * Linearizable Quorum-Fenced Distributed Cognitive Checkpoints
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/fenced_cognitive_ckpt.h"
#include "eshkol/core/cert_chain_v2.h"

#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct ReplicaNode {
    uint64_t promised_epoch = 0;
    uint64_t committed_epoch = 0;
    std::vector<uint8_t> committed_payload;
    std::array<uint8_t, 32> committed_sha256{};
};

} /* namespace */

struct eshkol_fenced_ckpt_cluster {
    uint32_t n = 0;
    std::vector<ReplicaNode> nodes;
    std::vector<std::vector<bool>> can_reach;

    /* Lifetime history of committed digests indexed by epoch for fork verification. */
    std::unordered_map<uint64_t, std::array<uint8_t, 32>> epoch_history;
    uint64_t fork_count = 0;

    uint32_t quorum() const { return (n / 2u) + 1u; }
};

extern "C" {

eshkol_fenced_ckpt_cluster_t* eshkol_ckpt_cluster_create(uint32_t num_nodes) {
    if (num_nodes == 0) return nullptr;
    auto* c = new eshkol_fenced_ckpt_cluster();
    c->n = num_nodes;
    c->nodes.resize(num_nodes);
    c->can_reach.assign(num_nodes, std::vector<bool>(num_nodes, true));
    return c;
}

void eshkol_ckpt_cluster_destroy(eshkol_fenced_ckpt_cluster_t* cluster) {
    delete cluster;
}

uint32_t eshkol_ckpt_cluster_size(const eshkol_fenced_ckpt_cluster_t* cluster) {
    return cluster ? cluster->n : 0;
}

uint32_t eshkol_ckpt_cluster_quorum_size(const eshkol_fenced_ckpt_cluster_t* cluster) {
    return cluster ? cluster->quorum() : 0;
}

bool eshkol_ckpt_cluster_partition(eshkol_fenced_ckpt_cluster_t* cluster,
                                   uint32_t split_count) {
    if (!cluster || split_count == 0 || split_count >= cluster->n) return false;
    for (uint32_t i = 0; i < cluster->n; ++i) {
        for (uint32_t j = 0; j < cluster->n; ++j) {
            bool same_side = (i < split_count) == (j < split_count);
            cluster->can_reach[i][j] = same_side;
        }
    }
    return true;
}

bool eshkol_ckpt_cluster_heal(eshkol_fenced_ckpt_cluster_t* cluster) {
    if (!cluster) return false;
    for (uint32_t i = 0; i < cluster->n; ++i) {
        for (uint32_t j = 0; j < cluster->n; ++j) {
            cluster->can_reach[i][j] = true;
        }
    }

    /* Locate highest committed state and highest promised fencing epoch across cluster. */
    uint64_t best_epoch = 0;
    uint32_t best_idx = 0;
    uint64_t max_promised = 0;
    for (uint32_t i = 0; i < cluster->n; ++i) {
        if (cluster->nodes[i].promised_epoch > max_promised) {
            max_promised = cluster->nodes[i].promised_epoch;
        }
        if (cluster->nodes[i].committed_epoch > best_epoch) {
            best_epoch = cluster->nodes[i].committed_epoch;
            best_idx = i;
        } else if (cluster->nodes[i].committed_epoch == best_epoch && best_epoch > 0) {
            if (cluster->nodes[i].committed_sha256 != cluster->nodes[best_idx].committed_sha256) {
                ++cluster->fork_count;
            }
        }
    }

    /* Linearly replicate the single winning committed checkpoint to all lagging replicas. */
    const ReplicaNode winner = cluster->nodes[best_idx];
    for (uint32_t i = 0; i < cluster->n; ++i) {
        if (cluster->nodes[i].promised_epoch < max_promised) {
            cluster->nodes[i].promised_epoch = max_promised;
        }
        if (best_epoch > 0 && cluster->nodes[i].committed_epoch < best_epoch) {
            cluster->nodes[i].committed_epoch = winner.committed_epoch;
            cluster->nodes[i].committed_payload = winner.committed_payload;
            cluster->nodes[i].committed_sha256 = winner.committed_sha256;
        }
    }
    return true;
}

eshkol_ckpt_status_t eshkol_ckpt_cluster_propose(
    eshkol_fenced_ckpt_cluster_t* cluster,
    uint32_t leader_node,
    uint64_t epoch,
    const uint8_t* payload,
    size_t payload_len) {
    if (!cluster || leader_node >= cluster->n || epoch == 0 || !payload || payload_len == 0) {
        return ESHKOL_CKPT_ERR_INVALID_ARG;
    }

    const uint32_t q = cluster->quorum();

    /*
     * Phase 1 (Fenced Prepare):
     * First probe all reachable replicas for fencing violations BEFORE mutating
     * any promise state, and verify a strict majority (> N/2) is reachable.
     */
    std::vector<uint32_t> reachable;
    reachable.reserve(cluster->n);
    bool fenced_out = false;
    for (uint32_t j = 0; j < cluster->n; ++j) {
        if (!cluster->can_reach[leader_node][j]) continue;
        reachable.push_back(j);
        if (epoch <= cluster->nodes[j].promised_epoch ||
            epoch <= cluster->nodes[j].committed_epoch) {
            fenced_out = true;
        }
    }

    if (fenced_out) {
        return ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN;
    }
    if (reachable.size() < q) {
        return ESHKOL_CKPT_ERR_NO_QUORUM;
    }

    /* Record promises atomically across the reachable majority. */
    std::vector<uint32_t> promised_nodes;
    promised_nodes.reserve(reachable.size());
    for (uint32_t j : reachable) {
        if (epoch > cluster->nodes[j].promised_epoch) {
            cluster->nodes[j].promised_epoch = epoch;
            promised_nodes.push_back(j);
        }
    }
    if (promised_nodes.size() < q) {
        return ESHKOL_CKPT_ERR_NO_QUORUM;
    }

    /* Compute payload SHA-256 digest and verify lifetime linearizability invariant. */
    std::array<uint8_t, 32> digest{};
    eshkol_cert_sha256(payload, payload_len, digest.data());

    auto hist_it = cluster->epoch_history.find(epoch);
    if (hist_it != cluster->epoch_history.end()) {
        if (hist_it->second != digest) {
            ++cluster->fork_count;
            return ESHKOL_CKPT_ERR_FENCED_SPLIT_BRAIN;
        }
    } else {
        cluster->epoch_history.emplace(epoch, digest);
    }

    /* Phase 2 (Fenced Commit): commit on all promised replicas (strict majority). */
    uint32_t commit_acks = 0;
    for (uint32_t j : promised_nodes) {
        if (cluster->nodes[j].promised_epoch == epoch) {
            cluster->nodes[j].committed_epoch = epoch;
            cluster->nodes[j].committed_payload.assign(payload, payload + payload_len);
            cluster->nodes[j].committed_sha256 = digest;
            ++commit_acks;
        }
    }
    if (commit_acks < q) {
        return ESHKOL_CKPT_ERR_NO_QUORUM;
    }

    return ESHKOL_CKPT_OK;
}

bool eshkol_ckpt_node_state(const eshkol_fenced_ckpt_cluster_t* cluster,
                            uint32_t node_id,
                            uint64_t* out_committed_epoch,
                            uint64_t* out_promised_epoch,
                            uint8_t out_sha256[32],
                            size_t* out_payload_len) {
    if (!cluster || node_id >= cluster->n) return false;
    const ReplicaNode& node = cluster->nodes[node_id];
    if (out_committed_epoch) *out_committed_epoch = node.committed_epoch;
    if (out_promised_epoch) *out_promised_epoch = node.promised_epoch;
    if (out_sha256) std::memcpy(out_sha256, node.committed_sha256.data(), 32);
    if (out_payload_len) *out_payload_len = node.committed_payload.size();
    return true;
}

uint64_t eshkol_ckpt_cluster_fork_count(const eshkol_fenced_ckpt_cluster_t* cluster) {
    return cluster ? cluster->fork_count : 0;
}

bool eshkol_ckpt_cluster_all_converged(const eshkol_fenced_ckpt_cluster_t* cluster) {
    if (!cluster || cluster->n == 0) return false;
    const ReplicaNode& ref = cluster->nodes[0];
    for (uint32_t i = 1; i < cluster->n; ++i) {
        if (cluster->nodes[i].committed_epoch != ref.committed_epoch ||
            cluster->nodes[i].committed_sha256 != ref.committed_sha256 ||
            cluster->nodes[i].committed_payload != ref.committed_payload) {
            return false;
        }
    }
    return true;
}

} /* extern "C" */
