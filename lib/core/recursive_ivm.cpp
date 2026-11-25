/*
 * Stratified Recursive Incremental View Maintenance (DRed / Differential DBSP)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/recursive_ivm.h"

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct EdgeKey {
    uint64_t u;
    uint64_t v;
    bool operator==(const EdgeKey& o) const { return u == o.u && v == o.v; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        uint64_t h = k.u * 0x9e3779b97f4a7c15ULL;
        h ^= k.v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return static_cast<size_t>(h);
    }
};

uint64_t hash_sorted_pairs(std::vector<std::pair<uint64_t, uint64_t>>& pairs) {
    std::sort(pairs.begin(), pairs.end());
    uint64_t h = 14695981039346656037ULL;
    auto mix_u64 = [&h](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<uint64_t>((v >> (8 * i)) & 0xFFu);
            h *= 1099511628211ULL;
        }
    };
    mix_u64(static_cast<uint64_t>(pairs.size()));
    for (const auto& p : pairs) {
        mix_u64(p.first);
        mix_u64(p.second);
    }
    return h;
}

} /* namespace */

/* ============================================================================
 * Part 1: Compile-Time Stratification Validator
 * ============================================================================ */

struct DepEdge {
    uint32_t body_rel;
    bool is_negated;
};

struct eshkol_ivm_strat_compiler {
    std::unordered_map<uint32_t, bool> has_base;
    std::unordered_map<uint32_t, std::vector<DepEdge>> deps;
    std::unordered_map<uint32_t, uint32_t> stratum;
    bool compiled = false;
};

/* ============================================================================
 * Part 2: Incremental Recursive Reachability View (DRed)
 * ============================================================================ */

struct eshkol_recursive_ivm {
    /* Base edge multiplicity map (pruned immediately on hitting 0). */
    std::unordered_map<EdgeKey, int64_t, EdgeKeyHash> edge_counts;
    /* Active adjacency indices over edges with multiplicity > 0. */
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> out_adj;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> in_adj;
    /* Derived transitive closure R(a, b): reach_out[a] contains b, reach_in[b] contains a. */
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> reach_out;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> reach_in;
    size_t derived_pairs = 0;

    bool insert_reach_pair(uint64_t a, uint64_t b) {
        auto& out_set = reach_out[a];
        if (!out_set.insert(b).second) return false;
        reach_in[b].insert(a);
        ++derived_pairs;
        return true;
    }

    void erase_reach_pair(uint64_t a, uint64_t b) {
        auto it_out = reach_out.find(a);
        if (it_out != reach_out.end()) {
            if (it_out->second.erase(b) > 0) {
                --derived_pairs;
                if (it_out->second.empty()) reach_out.erase(it_out);
                auto it_in = reach_in.find(b);
                if (it_in != reach_in.end()) {
                    it_in->second.erase(a);
                    if (it_in->second.empty()) reach_in.erase(it_in);
                }
            }
        }
    }
};

extern "C" {

eshkol_ivm_strat_compiler_t* eshkol_ivm_strat_create(void) {
    return new eshkol_ivm_strat_compiler();
}

void eshkol_ivm_strat_destroy(eshkol_ivm_strat_compiler_t* sc) {
    delete sc;
}

bool eshkol_ivm_strat_add_relation(eshkol_ivm_strat_compiler_t* sc,
                                   uint32_t rel_id,
                                   bool has_base_case) {
    if (!sc) return false;
    sc->has_base[rel_id] = sc->has_base[rel_id] || has_base_case;
    sc->deps[rel_id];
    sc->compiled = false;
    return true;
}

bool eshkol_ivm_strat_add_dependency(eshkol_ivm_strat_compiler_t* sc,
                                     uint32_t head_rel,
                                     uint32_t body_rel,
                                     bool is_negated) {
    if (!sc) return false;
    if (sc->has_base.find(head_rel) == sc->has_base.end()) sc->has_base[head_rel] = false;
    if (sc->has_base.find(body_rel) == sc->has_base.end()) sc->has_base[body_rel] = false;
    sc->deps[head_rel].push_back({body_rel, is_negated});
    sc->deps[body_rel];
    sc->compiled = false;
    return true;
}

eshkol_ivm_status_t eshkol_ivm_strat_compile(eshkol_ivm_strat_compiler_t* sc) {
    if (!sc) return ESHKOL_IVM_ERR_INVALID_ARG;
    sc->stratum.clear();
    sc->compiled = false;

    std::vector<uint32_t> nodes;
    nodes.reserve(sc->has_base.size());
    for (const auto& kv : sc->has_base) nodes.push_back(kv.first);
    std::sort(nodes.begin(), nodes.end());

    std::unordered_map<uint32_t, int> index_map;
    std::unordered_map<uint32_t, int> lowlink;
    std::unordered_map<uint32_t, bool> on_stack;
    std::vector<uint32_t> stack;
    std::vector<std::vector<uint32_t>> sccs;
    int next_index = 0;

    auto strongconnect = [&](auto&& self, uint32_t v) -> void {
        index_map[v] = next_index;
        lowlink[v] = next_index;
        ++next_index;
        stack.push_back(v);
        on_stack[v] = true;

        auto it = sc->deps.find(v);
        if (it != sc->deps.end()) {
            for (const DepEdge& e : it->second) {
                uint32_t w = e.body_rel;
                if (index_map.find(w) == index_map.end()) {
                    self(self, w);
                    if (lowlink[w] < lowlink[v]) lowlink[v] = lowlink[w];
                } else if (on_stack[w]) {
                    if (index_map[w] < lowlink[v]) lowlink[v] = index_map[w];
                }
            }
        }

        if (lowlink[v] == index_map[v]) {
            std::vector<uint32_t> comp;
            while (true) {
                uint32_t w = stack.back();
                stack.pop_back();
                on_stack[w] = false;
                comp.push_back(w);
                if (w == v) break;
            }
            sccs.push_back(std::move(comp));
        }
    };

    for (uint32_t v : nodes) {
        if (index_map.find(v) == index_map.end()) {
            strongconnect(strongconnect, v);
        }
    }

    std::unordered_map<uint32_t, size_t> node_to_scc;
    for (size_t i = 0; i < sccs.size(); ++i) {
        for (uint32_t v : sccs[i]) node_to_scc[v] = i;
    }

    /* Validate every SCC: reject negative cycles AND ungrounded cycles without base case. */
    for (size_t cid = 0; cid < sccs.size(); ++cid) {
        const auto& comp = sccs[cid];
        bool has_internal_edge = false;
        bool has_internal_neg = false;
        bool any_base_case = false;

        for (uint32_t v : comp) {
            if (sc->has_base[v]) any_base_case = true;
            auto it = sc->deps.find(v);
            if (it == sc->deps.end()) continue;
            for (const DepEdge& e : it->second) {
                if (node_to_scc[e.body_rel] == cid) {
                    has_internal_edge = true;
                    if (e.is_negated) has_internal_neg = true;
                }
            }
        }

        if (has_internal_neg) {
            return ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION;
        }
        if ((comp.size() > 1 || has_internal_edge) && !any_base_case) {
            return ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION;
        }
    }

    /* Tarjan emits SCCs in reverse topological dependency order (leaves first!). */
    std::vector<uint32_t> scc_level(sccs.size(), 0);
    for (size_t cid = 0; cid < sccs.size(); ++cid) {
        uint32_t lvl = 0;
        for (uint32_t v : sccs[cid]) {
            auto it = sc->deps.find(v);
            if (it == sc->deps.end()) continue;
            for (const DepEdge& e : it->second) {
                size_t dep_cid = node_to_scc[e.body_rel];
                if (dep_cid == cid) continue;
                uint32_t req = scc_level[dep_cid] + (e.is_negated ? 1u : 0u);
                if (req > lvl) lvl = req;
            }
        }
        scc_level[cid] = lvl;
        for (uint32_t v : sccs[cid]) sc->stratum[v] = lvl;
    }

    sc->compiled = true;
    return ESHKOL_IVM_OK;
}

uint32_t eshkol_ivm_strat_level(const eshkol_ivm_strat_compiler_t* sc, uint32_t rel_id) {
    if (!sc || !sc->compiled) return UINT32_MAX;
    auto it = sc->stratum.find(rel_id);
    if (it == sc->stratum.end()) return UINT32_MAX;
    return it->second;
}

eshkol_recursive_ivm_t* eshkol_recursive_ivm_create(void) {
    return new eshkol_recursive_ivm();
}

void eshkol_recursive_ivm_destroy(eshkol_recursive_ivm_t* ivm) {
    delete ivm;
}

bool eshkol_recursive_ivm_insert_edge(eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v) {
    if (!ivm) return false;
    EdgeKey key{u, v};
    int64_t& cnt = ivm->edge_counts[key];
    ++cnt;
    if (cnt > 1) return true; /* Multiplicity increased; topology unchanged. */

    ivm->out_adj[u].insert(v);
    ivm->in_adj[v].insert(u);

    /* Collect sources reaching u (including u) and targets reachable from v (including v). */
    std::vector<uint64_t> sources;
    sources.push_back(u);
    auto it_in = ivm->reach_in.find(u);
    if (it_in != ivm->reach_in.end()) {
        for (uint64_t a : it_in->second) {
            if (a != u) sources.push_back(a);
        }
    }

    std::vector<uint64_t> targets;
    targets.push_back(v);
    auto it_out = ivm->reach_out.find(v);
    if (it_out != ivm->reach_out.end()) {
        for (uint64_t b : it_out->second) {
            if (b != v) targets.push_back(b);
        }
    }

    for (uint64_t a : sources) {
        for (uint64_t b : targets) {
            ivm->insert_reach_pair(a, b);
        }
    }
    return true;
}

bool eshkol_recursive_ivm_delete_edge(eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v) {
    if (!ivm) return false;
    EdgeKey key{u, v};
    auto it_e = ivm->edge_counts.find(key);
    if (it_e == ivm->edge_counts.end()) return false;

    --(it_e->second);
    if (it_e->second > 0) return true; /* Edge still supported by remaining multiplicity. */

    /* Remove base edge cleanly with zero zombie entries. */
    ivm->edge_counts.erase(it_e);
    auto it_oa = ivm->out_adj.find(u);
    if (it_oa != ivm->out_adj.end()) {
        it_oa->second.erase(v);
        if (it_oa->second.empty()) ivm->out_adj.erase(it_oa);
    }
    auto it_ia = ivm->in_adj.find(v);
    if (it_ia != ivm->in_adj.end()) {
        it_ia->second.erase(u);
        if (it_ia->second.empty()) ivm->in_adj.erase(it_ia);
    }

    /*
     * DRed Phase 1 (Over-Deletion):
     * Any derived pair (a, b) that could have traversed (u, v) satisfies
     * a in {u} U reach_in[u] and b in {v} U reach_out[v].
     */
    std::vector<uint64_t> affected_sources;
    affected_sources.push_back(u);
    auto it_in = ivm->reach_in.find(u);
    if (it_in != ivm->reach_in.end()) {
        for (uint64_t a : it_in->second) {
            if (a != u) affected_sources.push_back(a);
        }
    }

    std::unordered_set<uint64_t> affected_targets;
    affected_targets.insert(v);
    auto it_out = ivm->reach_out.find(v);
    if (it_out != ivm->reach_out.end()) {
        for (uint64_t b : it_out->second) affected_targets.insert(b);
    }

    std::vector<uint64_t> active_affected_sources;
    active_affected_sources.reserve(affected_sources.size());
    for (uint64_t a : affected_sources) {
        bool removed_any = false;
        for (uint64_t b : affected_targets) {
            auto ro_it = ivm->reach_out.find(a);
            if (ro_it != ivm->reach_out.end() && ro_it->second.count(b) > 0) {
                ivm->erase_reach_pair(a, b);
                removed_any = true;
            }
        }
        if (removed_any) active_affected_sources.push_back(a);
    }

    /*
     * DRed Phase 2 (Alternative-Support Rederivation):
     * For every source `a` that lost pairs in `affected_targets`, re-seed any
     * target y in `affected_targets` reachable in 1 step from `a` itself OR from
     * any surviving unaffected target `w` in reach_out[a], then propagate along
     * remaining base edges within `affected_targets`.
     */
    std::deque<uint64_t> worklist;
    for (uint64_t a : active_affected_sources) {
        worklist.clear();

        /* Seed 1: direct remaining base edges (a -> y). */
        auto oa_a = ivm->out_adj.find(a);
        if (oa_a != ivm->out_adj.end()) {
            for (uint64_t y : oa_a->second) {
                if (affected_targets.count(y) > 0 && ivm->insert_reach_pair(a, y)) {
                    worklist.push_back(y);
                }
            }
        }

        /* Seed 2: surviving unaffected reach_out[a] composed with base edges (w -> y). */
        auto ro_a = ivm->reach_out.find(a);
        if (ro_a != ivm->reach_out.end()) {
            std::vector<uint64_t> surviving(ro_a->second.begin(), ro_a->second.end());
            for (uint64_t w : surviving) {
                auto oa_w = ivm->out_adj.find(w);
                if (oa_w == ivm->out_adj.end()) continue;
                for (uint64_t y : oa_w->second) {
                    if (affected_targets.count(y) > 0 && ivm->insert_reach_pair(a, y)) {
                        worklist.push_back(y);
                    }
                }
            }
        }

        /* Propagate rederivations forward along remaining base edges. */
        while (!worklist.empty()) {
            uint64_t cur = worklist.front();
            worklist.pop_front();
            auto oa_c = ivm->out_adj.find(cur);
            if (oa_c == ivm->out_adj.end()) continue;
            for (uint64_t nxt : oa_c->second) {
                if (affected_targets.count(nxt) > 0 && ivm->insert_reach_pair(a, nxt)) {
                    worklist.push_back(nxt);
                }
            }
        }
    }

    return true;
}

bool eshkol_recursive_ivm_reaches(const eshkol_recursive_ivm_t* ivm, uint64_t u, uint64_t v) {
    if (!ivm) return false;
    auto it = ivm->reach_out.find(u);
    if (it == ivm->reach_out.end()) return false;
    return it->second.count(v) > 0;
}

size_t eshkol_recursive_ivm_active_edge_count(const eshkol_recursive_ivm_t* ivm) {
    if (!ivm) return 0;
    return ivm->edge_counts.size();
}

size_t eshkol_recursive_ivm_active_node_count(const eshkol_recursive_ivm_t* ivm) {
    if (!ivm) return 0;
    std::unordered_set<uint64_t> active;
    for (const auto& kv : ivm->out_adj) active.insert(kv.first);
    for (const auto& kv : ivm->in_adj) active.insert(kv.first);
    for (const auto& kv : ivm->reach_out) active.insert(kv.first);
    for (const auto& kv : ivm->reach_in) active.insert(kv.first);
    return active.size();
}

size_t eshkol_recursive_ivm_derived_pair_count(const eshkol_recursive_ivm_t* ivm) {
    if (!ivm) return 0;
    return ivm->derived_pairs;
}

uint64_t eshkol_recursive_ivm_derived_hash(const eshkol_recursive_ivm_t* ivm) {
    if (!ivm) return 0;
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    pairs.reserve(ivm->derived_pairs);
    for (const auto& kv : ivm->reach_out) {
        for (uint64_t b : kv.second) {
            pairs.emplace_back(kv.first, b);
        }
    }
    return hash_sorted_pairs(pairs);
}

uint64_t eshkol_recursive_ivm_oracle_hash(const eshkol_recursive_ivm_t* ivm,
                                          size_t* out_pair_count) {
    if (!ivm) {
        if (out_pair_count) *out_pair_count = 0;
        return 0;
    }
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    std::unordered_set<uint64_t> visited;
    std::deque<uint64_t> q;

    for (const auto& kv : ivm->out_adj) {
        uint64_t src = kv.first;
        visited.clear();
        q.clear();
        for (uint64_t nxt : kv.second) {
            if (visited.insert(nxt).second) q.push_back(nxt);
        }
        while (!q.empty()) {
            uint64_t cur = q.front();
            q.pop_front();
            pairs.emplace_back(src, cur);
            auto it = ivm->out_adj.find(cur);
            if (it == ivm->out_adj.end()) continue;
            for (uint64_t nxt : it->second) {
                if (visited.insert(nxt).second) q.push_back(nxt);
            }
        }
    }
    if (out_pair_count) *out_pair_count = pairs.size();
    return hash_sorted_pairs(pairs);
}

} /* extern "C" */
