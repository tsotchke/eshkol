/*
 * Quantitative Multiplicity (0, 1, omega) Enforcer, IncrementalizePass DBSP
 * Compiler, and Lob/Cubical Reflective psi_program Self-Modification Verifier
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/qtt_incremental_psi.h"
#include "eshkol/core/cert_chain_v2.h"

#include <cstring>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void append_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void append_u64_le(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
}

void compute_transition_witness(const eshkol_psi_transition_t* tr, uint8_t out_hash[32]) {
    std::vector<uint8_t> buf;
    buf.reserve(128);
    const char tag[] = "PSI_CUBICAL_PATH_V1|";
    buf.insert(buf.end(), tag, tag + sizeof(tag) - 1);
    append_u64_le(buf, tr->from_state_id);
    append_u64_le(buf, tr->to_state_id);
    append_u32_le(buf, tr->from_lob_rank);
    append_u32_le(buf, tr->to_lob_rank);
    append_u32_le(buf, static_cast<uint32_t>(tr->target_guard_kind));
    buf.insert(buf.end(), tr->guard_digest_before, tr->guard_digest_before + 32);
    buf.insert(buf.end(), tr->guard_digest_after, tr->guard_digest_after + 32);
    eshkol_cert_sha256(buf.data(), buf.size(), out_hash);
}

bool reaches_in_graph(const std::unordered_map<uint64_t, std::vector<uint64_t>>& adj,
                      uint64_t src, uint64_t dst) {
    if (src == dst) return true;
    std::unordered_set<uint64_t> visited;
    std::deque<uint64_t> q;
    visited.insert(src);
    q.push_back(src);
    while (!q.empty()) {
        uint64_t cur = q.front();
        q.pop_front();
        auto it = adj.find(cur);
        if (it == adj.end()) continue;
        for (uint64_t nxt : it->second) {
            if (nxt == dst) return true;
            if (visited.insert(nxt).second) q.push_back(nxt);
        }
    }
    return false;
}

} /* namespace */

/* ============================================================================
 * Part 1: QTT + IncrementalizePass DBSP Circuit
 * ============================================================================ */

struct ResourceSlot {
    eshkol_qtt_multiplicity_t mult = ESHKOL_QTT_UNRESTRICTED;
    int64_t value = 0;
    size_t static_term_uses = 0;
};

struct eshkol_qtt_dbsp_circuit {
    std::unordered_map<uint32_t, ResourceSlot> resources;
    std::vector<std::pair<uint32_t, int64_t>> terms;
    /* Compiled O(1)-lookup per-resource effective linear coefficient. */
    std::unordered_map<uint32_t, int64_t> compiled_coeff;
    /* Consumed linear token IDs (prevents duplicate consumption across deltas). */
    std::unordered_set<uint64_t> consumed_linear_tokens;
    int64_t incremental_total = 0;
    bool compiled = false;
};

/* ============================================================================
 * Part 2: Lob / Cubical Reflective psi_program Verifier
 * ============================================================================ */

struct eshkol_psi_verifier {
    uint8_t immutable_guard_digest[32]{};
    uint64_t current_state_id = 0;
    uint32_t current_lob_rank = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> path_adj;
    size_t verified_edges = 0;
};

extern "C" {

eshkol_qtt_dbsp_circuit_t* eshkol_qtt_dbsp_create(void) {
    return new eshkol_qtt_dbsp_circuit();
}

void eshkol_qtt_dbsp_destroy(eshkol_qtt_dbsp_circuit_t* circuit) {
    delete circuit;
}

bool eshkol_qtt_dbsp_add_resource(eshkol_qtt_dbsp_circuit_t* circuit,
                                  uint32_t resource_id,
                                  eshkol_qtt_multiplicity_t mult) {
    if (!circuit) return false;
    ResourceSlot slot;
    slot.mult = mult;
    slot.value = 0;
    slot.static_term_uses = 0;
    circuit->resources[resource_id] = slot;
    circuit->compiled = false;
    return true;
}

bool eshkol_qtt_dbsp_add_term(eshkol_qtt_dbsp_circuit_t* circuit,
                              uint32_t resource_id,
                              int64_t coeff) {
    if (!circuit) return false;
    if (circuit->resources.find(resource_id) == circuit->resources.end()) return false;
    circuit->terms.emplace_back(resource_id, coeff);
    circuit->compiled = false;
    return true;
}

eshkol_ce11_status_t eshkol_qtt_incrementalize_compile(eshkol_qtt_dbsp_circuit_t* circuit) {
    if (!circuit) return ESHKOL_CE11_ERR_INVALID_ARG;
    circuit->compiled = false;
    circuit->compiled_coeff.clear();

    for (auto& kv : circuit->resources) kv.second.static_term_uses = 0;

    for (const auto& term : circuit->terms) {
        auto it = circuit->resources.find(term.first);
        if (it == circuit->resources.end()) return ESHKOL_CE11_ERR_INVALID_ARG;
        ++it->second.static_term_uses;

        if (it->second.mult == ESHKOL_QTT_ZERO) {
            return ESHKOL_QTT_ERR_ERASED_RUNTIME_USE;
        }
        if (it->second.mult == ESHKOL_QTT_LINEAR_ONE && it->second.static_term_uses > 1) {
            return ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE;
        }
        circuit->compiled_coeff[term.first] += term.second;
    }

    /* Synchronize incremental baseline total with current state. */
    circuit->incremental_total = eshkol_qtt_dbsp_eval_full(circuit);
    circuit->compiled = true;
    return ESHKOL_CE11_OK;
}

eshkol_ce11_status_t eshkol_qtt_dbsp_apply_delta(
    eshkol_qtt_dbsp_circuit_t* circuit,
    const eshkol_qtt_delta_update_t* updates,
    size_t n_updates,
    int64_t* out_delta_output,
    int64_t* out_incremental_total) {
    if (!circuit || !circuit->compiled || (!updates && n_updates > 0)) {
        return ESHKOL_CE11_ERR_INVALID_ARG;
    }

    /*
     * Phase 1 (Atomic Validation): verify every update respects QTT multiplicity
     * BEFORE mutating any circuit state (fail-closed, never silently drops).
     */
    std::unordered_set<uint64_t> batch_linear_tokens;
    std::unordered_set<uint32_t> batch_linear_resources;
    for (size_t i = 0; i < n_updates; ++i) {
        const auto& u = updates[i];
        auto it = circuit->resources.find(u.resource_id);
        if (it == circuit->resources.end()) return ESHKOL_CE11_ERR_INVALID_ARG;
        if (it->second.mult == ESHKOL_QTT_ZERO) {
            return ESHKOL_QTT_ERR_ERASED_RUNTIME_USE;
        }
        if (it->second.mult == ESHKOL_QTT_LINEAR_ONE) {
            /* Duplicate consumption in same batch OR reusing a previously consumed token! */
            if (!batch_linear_resources.insert(u.resource_id).second) {
                return ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE;
            }
            if (u.linear_token_id != 0) {
                if (circuit->consumed_linear_tokens.count(u.linear_token_id) > 0 ||
                    !batch_linear_tokens.insert(u.linear_token_id).second) {
                    return ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE;
                }
            }
        }
    }

    /* Phase 2 (O(|Delta|) Incremental Commit): all checks passed. */
    int64_t delta_out = 0;
    for (size_t i = 0; i < n_updates; ++i) {
        const auto& u = updates[i];
        auto& slot = circuit->resources[u.resource_id];
        slot.value += u.delta_value;
        if (slot.mult == ESHKOL_QTT_LINEAR_ONE && u.linear_token_id != 0) {
            circuit->consumed_linear_tokens.insert(u.linear_token_id);
        }
        auto cit = circuit->compiled_coeff.find(u.resource_id);
        if (cit != circuit->compiled_coeff.end()) {
            delta_out += cit->second * u.delta_value;
        }
    }

    circuit->incremental_total += delta_out;
    if (out_delta_output) *out_delta_output = delta_out;
    if (out_incremental_total) *out_incremental_total = circuit->incremental_total;
    return ESHKOL_CE11_OK;
}

int64_t eshkol_qtt_dbsp_eval_full(const eshkol_qtt_dbsp_circuit_t* circuit) {
    if (!circuit) return 0;
    int64_t total = 0;
    for (const auto& term : circuit->terms) {
        auto it = circuit->resources.find(term.first);
        if (it != circuit->resources.end()) {
            total += term.second * it->second.value;
        }
    }
    return total;
}

int64_t eshkol_qtt_dbsp_current_incremental(const eshkol_qtt_dbsp_circuit_t* circuit) {
    return circuit ? circuit->incremental_total : 0;
}

/* ============================================================================
 * Part 2: Lob / Cubical Reflective psi_program Verifier
 * ============================================================================ */

eshkol_psi_verifier_t* eshkol_psi_verifier_create(const uint8_t* immutable_guard_spec,
                                                  size_t spec_len,
                                                  uint64_t initial_state_id,
                                                  uint32_t initial_lob_rank) {
    if (!immutable_guard_spec || spec_len == 0) return nullptr;
    auto* v = new eshkol_psi_verifier();
    eshkol_cert_sha256(immutable_guard_spec, spec_len, v->immutable_guard_digest);
    v->current_state_id = initial_state_id;
    v->current_lob_rank = initial_lob_rank;
    v->path_adj[initial_state_id];
    return v;
}

void eshkol_psi_verifier_destroy(eshkol_psi_verifier_t* verifier) {
    delete verifier;
}

bool eshkol_psi_build_transition(const eshkol_psi_verifier_t* verifier,
                                 uint64_t from_state_id,
                                 uint64_t to_state_id,
                                 uint32_t from_lob_rank,
                                 uint32_t to_lob_rank,
                                 eshkol_psi_guard_kind_t target_guard_kind,
                                 const uint8_t* target_guard_spec,
                                 size_t target_guard_len,
                                 eshkol_psi_transition_t* out_tr) {
    if (!verifier || !target_guard_spec || target_guard_len == 0 || !out_tr) return false;
    std::memset(out_tr, 0, sizeof(*out_tr));
    out_tr->from_state_id = from_state_id;
    out_tr->to_state_id = to_state_id;
    out_tr->from_lob_rank = from_lob_rank;
    out_tr->to_lob_rank = to_lob_rank;
    out_tr->target_guard_kind = target_guard_kind;
    std::memcpy(out_tr->guard_digest_before, verifier->immutable_guard_digest, 32);
    eshkol_cert_sha256(target_guard_spec, target_guard_len, out_tr->guard_digest_after);
    compute_transition_witness(out_tr, out_tr->path_witness_sha256);
    return true;
}

static eshkol_ce11_status_t check_lob_obligations(const eshkol_psi_verifier_t* verifier,
                                                  const eshkol_psi_transition_t* tr) {
    /* 1. Reject unconditional-true guard rewrite bypass (Lob violation). */
    if (tr->target_guard_kind == ESHKOL_PSI_GUARD_CONST_TRUE) {
        return ESHKOL_PSI_ERR_LOB_VIOLATION;
    }
    /* 2. Reject any modification of the immutable safety guard digest. */
    if (std::memcmp(tr->guard_digest_before, verifier->immutable_guard_digest, 32) != 0 ||
        std::memcmp(tr->guard_digest_after, verifier->immutable_guard_digest, 32) != 0) {
        return ESHKOL_PSI_ERR_LOB_VIOLATION;
    }
    /* 3. Require strict modal Lob rank decrease: Box(P) -> P requires well-founded rank descent. */
    if (tr->to_lob_rank >= tr->from_lob_rank) {
        return ESHKOL_PSI_ERR_LOB_VIOLATION;
    }
    return ESHKOL_CE11_OK;
}

eshkol_ce11_status_t eshkol_psi_verify_and_apply(eshkol_psi_verifier_t* verifier,
                                                 const eshkol_psi_transition_t* tr) {
    if (!verifier || !tr) return ESHKOL_CE11_ERR_INVALID_ARG;

    /* 1. Check Lob safety guard obligations first. */
    eshkol_ce11_status_t lob_st = check_lob_obligations(verifier, tr);
    if (lob_st != ESHKOL_CE11_OK) return lob_st;

    /* 2. Check cubical path acyclicity: reject self-loops and any cycle back to an ancestor. */
    if (tr->from_state_id == tr->to_state_id ||
        reaches_in_graph(verifier->path_adj, tr->to_state_id, tr->from_state_id)) {
        return ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP;
    }

    /* 3. Check cubical path witness integrity (tampered path data fails closed). */
    uint8_t expected_witness[32];
    compute_transition_witness(tr, expected_witness);
    if (std::memcmp(expected_witness, tr->path_witness_sha256, 32) != 0) {
        return ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP;
    }

    /* 4. Require continuity from current verified state and Lob rank. */
    if (tr->from_state_id != verifier->current_state_id ||
        tr->from_lob_rank != verifier->current_lob_rank) {
        return ESHKOL_PSI_ERR_LOB_VIOLATION;
    }

    verifier->path_adj[tr->from_state_id].push_back(tr->to_state_id);
    verifier->path_adj[tr->to_state_id];
    verifier->current_state_id = tr->to_state_id;
    verifier->current_lob_rank = tr->to_lob_rank;
    ++verifier->verified_edges;
    return ESHKOL_CE11_OK;
}

eshkol_ce11_status_t eshkol_psi_verify_cubical_path(const eshkol_psi_verifier_t* verifier,
                                                    const eshkol_psi_transition_t* path,
                                                    size_t path_len) {
    if (!verifier || !path || path_len == 0) return ESHKOL_CE11_ERR_INVALID_ARG;

    /* Clone current graph and simulate applying the full candidate cubical path. */
    std::unordered_map<uint64_t, std::vector<uint64_t>> sim_adj = verifier->path_adj;
    for (size_t i = 0; i < path_len; ++i) {
        const auto& tr = path[i];
        eshkol_ce11_status_t lob_st = check_lob_obligations(verifier, &tr);
        if (lob_st != ESHKOL_CE11_OK) return lob_st;

        if (tr.from_state_id == tr.to_state_id ||
            reaches_in_graph(sim_adj, tr.to_state_id, tr.from_state_id)) {
            return ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP;
        }
        uint8_t expected_witness[32];
        compute_transition_witness(&tr, expected_witness);
        if (std::memcmp(expected_witness, tr.path_witness_sha256, 32) != 0) {
            return ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP;
        }
        sim_adj[tr.from_state_id].push_back(tr.to_state_id);
        sim_adj[tr.to_state_id];
    }
    return ESHKOL_CE11_OK;
}

uint64_t eshkol_psi_current_state(const eshkol_psi_verifier_t* verifier) {
    return verifier ? verifier->current_state_id : 0;
}

uint32_t eshkol_psi_current_lob_rank(const eshkol_psi_verifier_t* verifier) {
    return verifier ? verifier->current_lob_rank : 0;
}

size_t eshkol_psi_verified_edge_count(const eshkol_psi_verifier_t* verifier) {
    return verifier ? verifier->verified_edges : 0;
}

} /* extern "C" */
