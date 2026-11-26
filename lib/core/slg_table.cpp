/*
 * Native SLG-WAM Tabled Resolution & 3-Valued Well-Founded Semantics (WFS)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/slg_table.h>
#include <eshkol/eshkol.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" {
const char* eshkol_intern_predicate(const char* name);
void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);
}

namespace {

constexpr uint64_t kFreshVarBase = 0x4000000000000000ULL;

inline eshkol_tagged_value_t make_fact_tv(const eshkol_fact_t* f) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_HEAP_PTR;
    tv.data.ptr_val = reinterpret_cast<uint64_t>(f);
    return tv;
}

inline eshkol_tagged_value_t make_var_tv(uint64_t id) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_LOGIC_VAR;
    tv.data.int_val = static_cast<int64_t>(id);
    return tv;
}

inline uint64_t pred_not_id() {
    return reinterpret_cast<uint64_t>(eshkol_intern_predicate("not"));
}

inline const char* predicate_cstr(uint64_t pred) {
    return pred ? reinterpret_cast<const char*>(static_cast<uintptr_t>(pred)) : "";
}

bool is_negated_fact(const eshkol_fact_t* goal, const eshkol_fact_t** out_inner) {
    if (!goal || goal->arity != 1) return false;
    uint64_t p_not = pred_not_id();
    bool match = (goal->predicate == p_not);
    if (!match && goal->predicate != 0) {
        const char* s = predicate_cstr(goal->predicate);
        match = (s && std::strcmp(s, "not") == 0);
    }
    if (!match) return false;

    const eshkol_tagged_value_t* args = FACT_ARGS(goal);
    if (args[0].type != ESHKOL_VALUE_HEAP_PTR || !args[0].data.ptr_val) return false;
    eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(reinterpret_cast<void*>(args[0].data.ptr_val));
    if (!hdr || hdr->subtype != HEAP_SUBTYPE_FACT) return false;
    if (out_inner) {
        *out_inner = reinterpret_cast<const eshkol_fact_t*>(args[0].data.ptr_val);
    }
    return true;
}

/* Canonical structural key encoder for facts and subgoal variants.
 * When `canonicalize_vars` is true, logic variables are renamed to V0, V1, ...
 * in left-to-right discovery order so variant calls share one subgoal table. */
void encode_term_key(const eshkol_tagged_value_t& tv,
                     bool canonicalize_vars,
                     std::unordered_map<uint64_t, uint32_t>& var_map,
                     std::string& out,
                     int depth);

void encode_fact_key_impl(const eshkol_fact_t* f,
                          bool canonicalize_vars,
                          std::unordered_map<uint64_t, uint32_t>& var_map,
                          std::string& out,
                          int depth) {
    if (!f || depth > 64) {
        out.append("F:null;");
        return;
    }
    out.append("F:");
    out.append(predicate_cstr(f->predicate));
    out.push_back('/');
    out.append(std::to_string(f->arity));
    out.push_back('(');
    const eshkol_tagged_value_t* args = FACT_ARGS(f);
    for (uint32_t i = 0; i < f->arity; i++) {
        if (i > 0) out.push_back(',');
        encode_term_key(args[i], canonicalize_vars, var_map, out, depth + 1);
    }
    out.push_back(')');
}

void encode_term_key(const eshkol_tagged_value_t& tv,
                     bool canonicalize_vars,
                     std::unordered_map<uint64_t, uint32_t>& var_map,
                     std::string& out,
                     int depth) {
    if (depth > 64) {
        out.append("D:lim");
        return;
    }
    uint64_t vid = 0;
    if (eshkol_logic_var_id_of(&tv, &vid)) {
        out.append("V:");
        if (canonicalize_vars) {
            auto it = var_map.find(vid);
            uint32_t cid = 0;
            if (it == var_map.end()) {
                cid = static_cast<uint32_t>(var_map.size());
                var_map.emplace(vid, cid);
            } else {
                cid = it->second;
            }
            out.append(std::to_string(cid));
        } else {
            out.append(std::to_string(vid));
        }
        return;
    }

    switch (tv.type) {
        case ESHKOL_VALUE_NULL:
            out.append("N");
            break;
        case ESHKOL_VALUE_INT64:
            out.append("I:");
            out.append(std::to_string(tv.data.int_val));
            break;
        case ESHKOL_VALUE_DOUBLE: {
            uint64_t bits = 0;
            double d = (tv.data.double_val == 0.0) ? 0.0 : tv.data.double_val;
            std::memcpy(&bits, &d, sizeof(bits));
            out.append("D:");
            out.append(std::to_string(bits));
            break;
        }
        case ESHKOL_VALUE_BOOL:
            out.append(tv.data.int_val ? "B:1" : "B:0");
            break;
        case ESHKOL_VALUE_HEAP_PTR: {
            if (!tv.data.ptr_val) {
                out.append("P:0");
                break;
            }
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(reinterpret_cast<void*>(tv.data.ptr_val));
            if (!hdr) {
                out.append("P:u");
                break;
            }
            if (hdr->subtype == HEAP_SUBTYPE_SYMBOL || hdr->subtype == HEAP_SUBTYPE_STRING) {
                const char* s = reinterpret_cast<const char*>(static_cast<uintptr_t>(tv.data.ptr_val));
                out.push_back(hdr->subtype == HEAP_SUBTYPE_SYMBOL ? 'Y' : 'S');
                out.push_back(':');
                out.append(std::to_string(std::strlen(s)));
                out.push_back(':');
                out.append(s);
            } else if (hdr->subtype == HEAP_SUBTYPE_FACT) {
                encode_fact_key_impl(
                    reinterpret_cast<const eshkol_fact_t*>(tv.data.ptr_val),
                    canonicalize_vars, var_map, out, depth + 1);
            } else if (hdr->subtype == HEAP_SUBTYPE_CONS) {
                const auto* cell = reinterpret_cast<const eshkol_tagged_value_t*>(tv.data.ptr_val);
                out.append("C(");
                encode_term_key(cell[0], canonicalize_vars, var_map, out, depth + 1);
                out.push_back('.');
                encode_term_key(cell[1], canonicalize_vars, var_map, out, depth + 1);
                out.push_back(')');
            } else {
                out.append("H:");
                out.append(std::to_string(hdr->subtype));
                out.push_back(':');
                out.append(std::to_string(tv.data.ptr_val));
            }
            break;
        }
        default:
            out.append("T:");
            out.append(std::to_string(tv.type));
            out.push_back(':');
            out.append(std::to_string(tv.data.raw_val));
            break;
    }
}

std::string make_variant_key(const eshkol_fact_t* f) {
    std::unordered_map<uint64_t, uint32_t> var_map;
    std::string out;
    out.reserve(64);
    encode_fact_key_impl(f, true, var_map, out, 0);
    return out;
}

std::string make_answer_key(const eshkol_fact_t* f) {
    std::unordered_map<uint64_t, uint32_t> var_map;
    std::string out;
    out.reserve(64);
    encode_fact_key_impl(f, true, var_map, out, 0);
    return out;
}

/* Standardize a term apart by mapping every logic variable to a fresh internal ID. */
eshkol_tagged_value_t rename_term(arena_t* arena,
                                  const eshkol_tagged_value_t& tv,
                                  std::unordered_map<uint64_t, uint64_t>& rename_map,
                                  uint64_t& next_fresh_var,
                                  int depth);

eshkol_fact_t* rename_fact(arena_t* arena,
                           const eshkol_fact_t* f,
                           std::unordered_map<uint64_t, uint64_t>& rename_map,
                           uint64_t& next_fresh_var,
                           int depth) {
    if (!arena || !f || depth > 64) return nullptr;
    std::vector<eshkol_tagged_value_t> new_args(f->arity);
    const eshkol_tagged_value_t* old_args = FACT_ARGS(f);
    for (uint32_t i = 0; i < f->arity; i++) {
        new_args[i] = rename_term(arena, old_args[i], rename_map, next_fresh_var, depth + 1);
    }
    return eshkol_make_fact(arena, f->predicate,
                            f->arity > 0 ? new_args.data() : nullptr, f->arity);
}

eshkol_tagged_value_t rename_term(arena_t* arena,
                                  const eshkol_tagged_value_t& tv,
                                  std::unordered_map<uint64_t, uint64_t>& rename_map,
                                  uint64_t& next_fresh_var,
                                  int depth) {
    if (depth > 64) return tv;
    uint64_t vid = 0;
    if (eshkol_logic_var_id_of(&tv, &vid)) {
        auto it = rename_map.find(vid);
        uint64_t fresh = 0;
        if (it == rename_map.end()) {
            fresh = next_fresh_var++;
            rename_map.emplace(vid, fresh);
        } else {
            fresh = it->second;
        }
        return make_var_tv(fresh);
    }
    if (tv.type == ESHKOL_VALUE_HEAP_PTR && tv.data.ptr_val) {
        eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(reinterpret_cast<void*>(tv.data.ptr_val));
        if (hdr && hdr->subtype == HEAP_SUBTYPE_FACT) {
            eshkol_fact_t* rf = rename_fact(
                arena, reinterpret_cast<const eshkol_fact_t*>(tv.data.ptr_val),
                rename_map, next_fresh_var, depth + 1);
            return rf ? make_fact_tv(rf) : tv;
        }
    }
    return tv;
}

eshkol_fact_t* walk_deep_fact(arena_t* arena,
                              const eshkol_fact_t* f,
                              const eshkol_substitution_t* subst) {
    if (!arena || !f) return nullptr;
    eshkol_tagged_value_t ftv = make_fact_tv(f);
    eshkol_tagged_value_t walked = eshkol_walk_deep(arena, &ftv, subst);
    if (walked.type == ESHKOL_VALUE_HEAP_PTR && walked.data.ptr_val) {
        return reinterpret_cast<eshkol_fact_t*>(walked.data.ptr_val);
    }
    return nullptr;
}

struct PredArity {
    uint64_t pred;
    uint32_t arity;

    bool operator==(const PredArity& o) const {
        return pred == o.pred && arity == o.arity;
    }
};

struct PredArityHash {
    size_t operator()(const PredArity& p) const {
        return std::hash<uint64_t>{}(p.pred) ^ (std::hash<uint32_t>{}(p.arity) * 0x9e3779b97f4a7c15ULL);
    }
};

/* A consumer continuation suspended on a tabled subgoal. */
struct Consumer {
    uint32_t parent_subgoal = 0;
    eshkol_fact_t* head_template = nullptr;
    std::vector<eshkol_fact_t*> pos_goals;
    std::vector<eshkol_fact_t*> neg_goals;
    uint32_t next_pos_idx = 0;
    eshkol_fact_t* call_goal = nullptr;
    const eshkol_substitution_t* subst = nullptr;
};

struct SubgoalEntry {
    uint32_t id = 0;
    std::string key;
    eshkol_fact_t* goal = nullptr;
    std::vector<eshkol_fact_t*> answers;
    std::unordered_set<std::string> answer_keys;
    std::vector<Consumer> consumers;
    bool expanded = false;
};

struct WakeItem {
    uint32_t subgoal_id = 0;
    uint32_t answer_idx = 0;
    uint32_t consumer_idx = 0;
};

/* Set of derived ground/resolved atoms indexed by predicate/arity for fast negation checks. */
struct AtomInterpretation {
    std::unordered_set<std::string> keys;
    std::unordered_map<PredArity, std::vector<eshkol_fact_t*>, PredArityHash> by_pred;

    void insert(const std::string& key, eshkol_fact_t* fact) {
        if (!fact) return;
        if (keys.insert(key).second) {
            by_pred[{fact->predicate, fact->arity}].push_back(fact);
        }
    }

    bool equals(const AtomInterpretation& other) const {
        return keys == other.keys;
    }

    /* Returns true iff `goal` unifies with any atom in this interpretation. */
    bool matches_any(arena_t* arena, const eshkol_fact_t* goal) const {
        if (!goal) return false;
        if (keys.find(make_answer_key(goal)) != keys.end()) return true;
        auto it = by_pred.find({goal->predicate, goal->arity});
        if (it == by_pred.end()) return false;
        eshkol_substitution_t* empty_s = eshkol_make_substitution(arena, 8);
        if (!empty_s) return false;
        eshkol_tagged_value_t gtv = make_fact_tv(goal);
        for (eshkol_fact_t* atom : it->second) {
            eshkol_tagged_value_t atv = make_fact_tv(atom);
            if (eshkol_unify(arena, &gtv, &atv, empty_s) != nullptr) {
                return true;
            }
        }
        return false;
    }
};

} /* anonymous namespace */

struct eshkol_slg_engine {
    arena_t* arena = nullptr;
    const eshkol_knowledge_base_t* kb = nullptr;
    eshkol_slg_config_t cfg{};
    std::vector<const eshkol_rule_t*> rules;
    std::unordered_map<PredArity, std::vector<const eshkol_rule_t*>, PredArityHash> rules_by_head;
    uint64_t next_fresh_var = kFreshVarBase;

    /* Run telemetry */
    uint64_t steps_used = 0;
    uint32_t subgoals_tabulated = 0;
    uint32_t consumer_resumptions = 0;
    uint32_t wfs_iterations = 0;

    eshkol_slg_engine(arena_t* a, const eshkol_knowledge_base_t* k, const eshkol_slg_config_t& c)
        : arena(a), kb(k), cfg(c) {}

    bool consume_step() {
        steps_used++;
        if (cfg.max_steps > 0 && steps_used > cfg.max_steps) {
            return false;
        }
        return true;
    }

    /* Determine whether any rule reachable from `root_pred` contains negation,
     * and collect all negatively referenced predicate/arity pairs. */
    void analyze_reachability(const eshkol_fact_t* query,
                              bool& has_negation,
                              std::vector<PredArity>& neg_preds) const {
        has_negation = false;
        neg_preds.clear();
        if (!query) return;

        std::unordered_set<PredArity, PredArityHash> visited;
        std::unordered_set<PredArity, PredArityHash> neg_seen;
        std::deque<PredArity> work;

        PredArity root{query->predicate, query->arity};
        visited.insert(root);
        work.push_back(root);

        while (!work.empty()) {
            PredArity cur = work.front();
            work.pop_front();
            auto it = rules_by_head.find(cur);
            if (it == rules_by_head.end()) continue;
            for (const eshkol_rule_t* r : it->second) {
                for (uint32_t i = 0; i < r->num_body; i++) {
                    const eshkol_fact_t* inner = nullptr;
                    if (is_negated_fact(r->body[i], &inner)) {
                        has_negation = true;
                        PredArity np{inner->predicate, inner->arity};
                        if (neg_seen.insert(np).second) {
                            neg_preds.push_back(np);
                        }
                        if (visited.insert(np).second) {
                            work.push_back(np);
                        }
                    } else if (r->body[i]) {
                        PredArity pp{r->body[i]->predicate, r->body[i]->arity};
                        if (visited.insert(pp).second) {
                            work.push_back(pp);
                        }
                    }
                }
            }
        }
    }

    /* Single positive-reduct tabled SLG evaluation (Gamma_P(neg_block)).
     * Negative body literals `not B` succeed iff B does NOT unify with any atom
     * in `neg_block` (and does NOT match an EDB fact when `neg_block` is empty/null). */
    eshkol_slg_status_t eval_tabled_positive(
        const std::vector<eshkol_fact_t*>& seed_goals,
        const AtomInterpretation* neg_block,
        std::vector<SubgoalEntry>& out_tables,
        std::vector<uint32_t>& out_seed_ids) {

        out_tables.clear();
        out_seed_ids.clear();
        std::unordered_map<std::string, uint32_t> key_to_id;
        std::deque<WakeItem> wake_queue;
        std::deque<uint32_t> expand_queue;

        auto get_or_create_subgoal = [&](eshkol_fact_t* goal, uint32_t& out_id) -> eshkol_slg_status_t {
            if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
            std::string key = make_variant_key(goal);
            auto it = key_to_id.find(key);
            if (it != key_to_id.end()) {
                out_id = it->second;
                return ESHKOL_SLG_OK;
            }
            if (cfg.max_subgoals > 0 && out_tables.size() >= cfg.max_subgoals) {
                return ESHKOL_SLG_FUEL_EXHAUSTED;
            }
            uint32_t id = static_cast<uint32_t>(out_tables.size());
            SubgoalEntry entry;
            entry.id = id;
            entry.key = key;
            entry.goal = goal;
            out_tables.push_back(std::move(entry));
            key_to_id.emplace(std::move(key), id);
            subgoals_tabulated++;
            expand_queue.push_back(id);
            out_id = id;
            return ESHKOL_SLG_OK;
        };

        auto add_answer = [&](uint32_t sg_id, eshkol_fact_t* ans) -> eshkol_slg_status_t {
            if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
            if (!ans) return ESHKOL_SLG_OK;
            SubgoalEntry& sg = out_tables[sg_id];
            std::string akey = make_answer_key(ans);
            if (sg.answer_keys.find(akey) != sg.answer_keys.end()) {
                return ESHKOL_SLG_OK;
            }
            if (cfg.max_answers > 0 && sg.answers.size() >= cfg.max_answers) {
                return ESHKOL_SLG_FUEL_EXHAUSTED;
            }
            uint32_t ans_idx = static_cast<uint32_t>(sg.answers.size());
            sg.answer_keys.insert(std::move(akey));
            sg.answers.push_back(ans);
            for (uint32_t ci = 0; ci < sg.consumers.size(); ci++) {
                wake_queue.push_back({sg_id, ans_idx, ci});
            }
            return ESHKOL_SLG_OK;
        };

        auto check_neg_goals = [&](const std::vector<eshkol_fact_t*>& neg_goals,
                                   const eshkol_substitution_t* subst,
                                   bool& passed) -> eshkol_slg_status_t {
            passed = true;
            for (eshkol_fact_t* ng : neg_goals) {
                if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
                eshkol_fact_t* resolved = walk_deep_fact(arena, ng, subst);
                if (!resolved) {
                    passed = false;
                    return ESHKOL_SLG_OK;
                }
                if (neg_block && neg_block->matches_any(arena, resolved)) {
                    passed = false;
                    return ESHKOL_SLG_OK;
                }
                /* Also check EDB directly for negated EDB predicates */
                if (kb) {
                    eshkol_tagged_value_t edb_hits = eshkol_kb_query(arena, kb, resolved, nullptr);
                    if (edb_hits.type == ESHKOL_VALUE_HEAP_PTR && edb_hits.data.ptr_val != 0) {
                        passed = false;
                        return ESHKOL_SLG_OK;
                    }
                }
            }
            return ESHKOL_SLG_OK;
        };

        /* Advance a clause continuation across positive body literals. */
        std::function<eshkol_slg_status_t(
            uint32_t, eshkol_fact_t*, const std::vector<eshkol_fact_t*>&,
            const std::vector<eshkol_fact_t*>&, uint32_t, const eshkol_substitution_t*)> advance_body;

        advance_body = [&](uint32_t parent_sg,
                           eshkol_fact_t* head_tpl,
                           const std::vector<eshkol_fact_t*>& pos_goals,
                           const std::vector<eshkol_fact_t*>& neg_goals,
                           uint32_t pos_idx,
                           const eshkol_substitution_t* subst) -> eshkol_slg_status_t {
            if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;

            if (pos_idx >= pos_goals.size()) {
                bool neg_ok = true;
                eshkol_slg_status_t st = check_neg_goals(neg_goals, subst, neg_ok);
                if (st != ESHKOL_SLG_OK) return st;
                if (!neg_ok) return ESHKOL_SLG_OK;
                eshkol_fact_t* inst_head = walk_deep_fact(arena, head_tpl, subst);
                return add_answer(parent_sg, inst_head);
            }

            eshkol_fact_t* next_goal = walk_deep_fact(arena, pos_goals[pos_idx], subst);
            if (!next_goal) return ESHKOL_SLG_OK;

            /* Pure EDB predicate (no IDB rules define this predicate/arity): query KB directly */
            PredArity pa{next_goal->predicate, next_goal->arity};
            if (rules_by_head.find(pa) == rules_by_head.end()) {
                if (!kb) return ESHKOL_SLG_OK;
                eshkol_tagged_value_t matches = eshkol_kb_query(arena, kb, next_goal, subst);
                eshkol_tagged_value_t cur = matches;
                while (cur.type == ESHKOL_VALUE_HEAP_PTR && cur.data.ptr_val != 0) {
                    if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
                    auto* cell = reinterpret_cast<const eshkol_tagged_value_t*>(cur.data.ptr_val);
                    if (cell[0].type == ESHKOL_VALUE_HEAP_PTR && cell[0].data.ptr_val != 0) {
                        auto* next_s = reinterpret_cast<const eshkol_substitution_t*>(cell[0].data.ptr_val);
                        eshkol_slg_status_t st = advance_body(
                            parent_sg, head_tpl, pos_goals, neg_goals, pos_idx + 1u, next_s);
                        if (st != ESHKOL_SLG_OK) return st;
                    }
                    cur = cell[1];
                }
                return ESHKOL_SLG_OK;
            }

            /* Tabled IDB subgoal: register consumer suspension on subgoal entry */
            uint32_t child_sg = 0;
            eshkol_slg_status_t st = get_or_create_subgoal(next_goal, child_sg);
            if (st != ESHKOL_SLG_OK) return st;

            Consumer cons;
            cons.parent_subgoal = parent_sg;
            cons.head_template = head_tpl;
            cons.pos_goals = pos_goals;
            cons.neg_goals = neg_goals;
            cons.next_pos_idx = pos_idx + 1u;
            cons.call_goal = next_goal;
            cons.subst = subst;

            SubgoalEntry& child_entry = out_tables[child_sg];
            uint32_t cons_idx = static_cast<uint32_t>(child_entry.consumers.size());
            child_entry.consumers.push_back(std::move(cons));
            for (uint32_t ai = 0; ai < child_entry.answers.size(); ai++) {
                wake_queue.push_back({child_sg, ai, cons_idx});
            }
            return ESHKOL_SLG_OK;
        };

        auto expand_subgoal = [&](uint32_t sg_id) -> eshkol_slg_status_t {
            if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
            eshkol_fact_t* sg_goal = out_tables[sg_id].goal;
            out_tables[sg_id].expanded = true;

            /* 1. Direct EDB facts matching the subgoal */
            if (kb) {
                eshkol_tagged_value_t edb = eshkol_kb_query(arena, kb, sg_goal, nullptr);
                eshkol_tagged_value_t cur = edb;
                while (cur.type == ESHKOL_VALUE_HEAP_PTR && cur.data.ptr_val != 0) {
                    if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
                    auto* cell = reinterpret_cast<const eshkol_tagged_value_t*>(cur.data.ptr_val);
                    if (cell[0].type == ESHKOL_VALUE_HEAP_PTR && cell[0].data.ptr_val != 0) {
                        auto* s = reinterpret_cast<const eshkol_substitution_t*>(cell[0].data.ptr_val);
                        eshkol_fact_t* inst = walk_deep_fact(arena, sg_goal, s);
                        eshkol_slg_status_t st = add_answer(sg_id, inst);
                        if (st != ESHKOL_SLG_OK) return st;
                    }
                    cur = cell[1];
                }
            }

            /* 2. IDB rules whose head unifies with the subgoal */
            PredArity pa{sg_goal->predicate, sg_goal->arity};
            auto it = rules_by_head.find(pa);
            if (it == rules_by_head.end()) return ESHKOL_SLG_OK;

            eshkol_tagged_value_t sg_tv = make_fact_tv(sg_goal);
            for (const eshkol_rule_t* rule : it->second) {
                if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
                std::unordered_map<uint64_t, uint64_t> rename_map;
                eshkol_fact_t* r_head = rename_fact(arena, rule->head, rename_map, next_fresh_var, 0);
                if (!r_head) continue;

                eshkol_tagged_value_t rh_tv = make_fact_tv(r_head);
                eshkol_substitution_t* init_s = eshkol_make_substitution(arena, 8);
                eshkol_substitution_t* base_s = eshkol_unify(arena, &sg_tv, &rh_tv, init_s);
                if (!base_s) continue;

                std::vector<eshkol_fact_t*> pos_goals;
                std::vector<eshkol_fact_t*> neg_goals;
                pos_goals.reserve(rule->num_body);
                for (uint32_t b = 0; b < rule->num_body; b++) {
                    const eshkol_fact_t* inner = nullptr;
                    if (is_negated_fact(rule->body[b], &inner)) {
                        eshkol_fact_t* rn = rename_fact(arena, inner, rename_map, next_fresh_var, 0);
                        if (rn) neg_goals.push_back(rn);
                    } else if (rule->body[b]) {
                        eshkol_fact_t* rp = rename_fact(arena, rule->body[b], rename_map, next_fresh_var, 0);
                        if (rp) pos_goals.push_back(rp);
                    }
                }

                eshkol_slg_status_t st = advance_body(
                    sg_id, r_head, pos_goals, neg_goals, 0u, base_s);
                if (st != ESHKOL_SLG_OK) return st;
            }
            return ESHKOL_SLG_OK;
        };

        /* Seed initial subgoals */
        for (eshkol_fact_t* seed : seed_goals) {
            uint32_t sid = 0;
            eshkol_slg_status_t st = get_or_create_subgoal(seed, sid);
            if (st != ESHKOL_SLG_OK) return st;
            out_seed_ids.push_back(sid);
        }

        /* Drain producer expansions and consumer answer resumptions to completion */
        while (!expand_queue.empty() || !wake_queue.empty()) {
            while (!expand_queue.empty()) {
                uint32_t sg_id = expand_queue.front();
                expand_queue.pop_front();
                if (!out_tables[sg_id].expanded) {
                    eshkol_slg_status_t st = expand_subgoal(sg_id);
                    if (st != ESHKOL_SLG_OK) return st;
                }
            }
            while (!wake_queue.empty()) {
                WakeItem item = wake_queue.front();
                wake_queue.pop_front();
                if (!consume_step()) return ESHKOL_SLG_FUEL_EXHAUSTED;
                consumer_resumptions++;

                /* Copy consumer snapshot because advance_body may append new consumers */
                Consumer cons = out_tables[item.subgoal_id].consumers[item.consumer_idx];
                eshkol_fact_t* ans_raw = out_tables[item.subgoal_id].answers[item.answer_idx];
                std::unordered_map<uint64_t, uint64_t> ans_rename;
                eshkol_fact_t* ans = rename_fact(arena, ans_raw, ans_rename, next_fresh_var, 0);
                if (!ans) continue;

                eshkol_tagged_value_t call_tv = make_fact_tv(cons.call_goal);
                eshkol_tagged_value_t ans_tv = make_fact_tv(ans);
                eshkol_substitution_t* next_s = eshkol_unify(arena, &call_tv, &ans_tv, cons.subst);
                if (!next_s) continue;

                eshkol_slg_status_t st = advance_body(
                    cons.parent_subgoal, cons.head_template,
                    cons.pos_goals, cons.neg_goals,
                    cons.next_pos_idx, next_s);
                if (st != ESHKOL_SLG_OK) return st;
            }
        }

        return ESHKOL_SLG_OK;
    }

    eshkol_fact_t* make_open_goal(const PredArity& pa) {
        std::vector<eshkol_tagged_value_t> args(pa.arity);
        for (uint32_t i = 0; i < pa.arity; i++) {
            args[i] = make_var_tv(next_fresh_var++);
        }
        return eshkol_make_fact(arena, pa.pred, pa.arity > 0 ? args.data() : nullptr, pa.arity);
    }

    eshkol_slg_status_t run_query(const eshkol_fact_t* query, eshkol_slg_result_t* out_result) {
        std::memset(out_result, 0, sizeof(*out_result));
        if (!arena || !query || !out_result) {
            if (out_result) out_result->status = ESHKOL_SLG_INVALID_INPUT;
            return ESHKOL_SLG_INVALID_INPUT;
        }

        steps_used = 0;
        subgoals_tabulated = 0;
        consumer_resumptions = 0;
        wfs_iterations = 0;

        bool has_negation = false;
        std::vector<PredArity> neg_preds;
        analyze_reachability(query, has_negation, neg_preds);

        std::vector<std::pair<eshkol_fact_t*, eshkol_slg_truth_t>> collected;

        if (!has_negation) {
            /* Fast path: purely positive goal-directed SLG tabling */
            std::vector<eshkol_fact_t*> seeds = { const_cast<eshkol_fact_t*>(query) };
            std::vector<SubgoalEntry> tables;
            std::vector<uint32_t> seed_ids;
            eshkol_slg_status_t st = eval_tabled_positive(seeds, nullptr, tables, seed_ids);
            if (st != ESHKOL_SLG_OK) {
                out_result->status = st;
                out_result->steps_used = steps_used;
                out_result->subgoals_tabulated = subgoals_tabulated;
                out_result->consumer_resumptions = consumer_resumptions;
                return st;
            }
            const SubgoalEntry& root = tables[seed_ids[0]];
            collected.reserve(root.answers.size());
            for (eshkol_fact_t* ans : root.answers) {
                collected.emplace_back(ans, ESHKOL_SLG_TRUE);
            }
        } else {
            /* Exact 3-valued Well-Founded Semantics via Van Gelder alternating closure */
            const uint32_t max_iters = cfg.max_wfs_iters > 0 ? cfg.max_wfs_iters : 1024u;
            std::vector<eshkol_fact_t*> seeds;
            seeds.reserve(1u + neg_preds.size());
            seeds.push_back(const_cast<eshkol_fact_t*>(query));
            for (const PredArity& np : neg_preds) {
                eshkol_fact_t* open_g = make_open_goal(np);
                if (open_g) seeds.push_back(open_g);
            }

            AtomInterpretation true_set;
            AtomInterpretation poss_set;
            std::vector<SubgoalEntry> last_true_tables;
            std::vector<SubgoalEntry> last_poss_tables;
            std::vector<uint32_t> last_true_seeds;
            std::vector<uint32_t> last_poss_seeds;

            for (uint32_t iter = 0; iter < max_iters; iter++) {
                wfs_iterations++;

                /* Upper bound P_k = Gamma_P(T_k): complement is the greatest unfounded set */
                std::vector<SubgoalEntry> p_tables;
                std::vector<uint32_t> p_seeds;
                eshkol_slg_status_t st = eval_tabled_positive(seeds, &true_set, p_tables, p_seeds);
                if (st != ESHKOL_SLG_OK) {
                    out_result->status = st;
                    out_result->steps_used = steps_used;
                    out_result->subgoals_tabulated = subgoals_tabulated;
                    out_result->consumer_resumptions = consumer_resumptions;
                    out_result->wfs_iterations = wfs_iterations;
                    return st;
                }
                AtomInterpretation next_poss;
                for (uint32_t sidx = 1; sidx < p_seeds.size(); sidx++) {
                    const SubgoalEntry& sg = p_tables[p_seeds[sidx]];
                    for (eshkol_fact_t* ans : sg.answers) {
                        next_poss.insert(make_answer_key(ans), ans);
                    }
                }
                /* Include root answers too if root predicate is itself negatively referenced */
                for (eshkol_fact_t* ans : p_tables[p_seeds[0]].answers) {
                    next_poss.insert(make_answer_key(ans), ans);
                }

                /* Lower bound T_{k+1} = Gamma_P(P_k): definitely true atoms */
                std::vector<SubgoalEntry> t_tables;
                std::vector<uint32_t> t_seeds;
                st = eval_tabled_positive(seeds, &next_poss, t_tables, t_seeds);
                if (st != ESHKOL_SLG_OK) {
                    out_result->status = st;
                    out_result->steps_used = steps_used;
                    out_result->subgoals_tabulated = subgoals_tabulated;
                    out_result->consumer_resumptions = consumer_resumptions;
                    out_result->wfs_iterations = wfs_iterations;
                    return st;
                }
                AtomInterpretation next_true;
                for (uint32_t sidx = 1; sidx < t_seeds.size(); sidx++) {
                    const SubgoalEntry& sg = t_tables[t_seeds[sidx]];
                    for (eshkol_fact_t* ans : sg.answers) {
                        next_true.insert(make_answer_key(ans), ans);
                    }
                }
                for (eshkol_fact_t* ans : t_tables[t_seeds[0]].answers) {
                    next_true.insert(make_answer_key(ans), ans);
                }

                bool converged = (iter > 0 && next_true.equals(true_set) && next_poss.equals(poss_set));
                true_set = std::move(next_true);
                poss_set = std::move(next_poss);
                last_true_tables = std::move(t_tables);
                last_poss_tables = std::move(p_tables);
                last_true_seeds = std::move(t_seeds);
                last_poss_seeds = std::move(p_seeds);
                if (converged) break;
            }

            /* Classify root query answers into TRUE (in T) and UNKNOWN (in P \ T) */
            const SubgoalEntry& t_root = last_true_tables[last_true_seeds[0]];
            const SubgoalEntry& p_root = last_poss_tables[last_poss_seeds[0]];
            std::unordered_set<std::string> true_root_keys;
            for (eshkol_fact_t* ans : t_root.answers) {
                std::string k = make_answer_key(ans);
                true_root_keys.insert(k);
                collected.emplace_back(ans, ESHKOL_SLG_TRUE);
            }
            for (eshkol_fact_t* ans : p_root.answers) {
                std::string k = make_answer_key(ans);
                if (true_root_keys.find(k) == true_root_keys.end()) {
                    collected.emplace_back(ans, ESHKOL_SLG_UNKNOWN);
                }
            }
        }

        /* Materialize arena-allocated result array with query substitutions */
        out_result->status = ESHKOL_SLG_OK;
        out_result->steps_used = steps_used;
        out_result->subgoals_tabulated = subgoals_tabulated;
        out_result->consumer_resumptions = consumer_resumptions;
        out_result->wfs_iterations = wfs_iterations;
        out_result->num_answers = static_cast<uint32_t>(collected.size());

        if (!collected.empty()) {
            auto* arr = static_cast<eshkol_slg_answer_t*>(
                arena_allocate_aligned(arena,
                                       collected.size() * sizeof(eshkol_slg_answer_t),
                                       8));
            if (!arr) {
                out_result->status = ESHKOL_SLG_FUEL_EXHAUSTED;
                out_result->num_answers = 0;
                return out_result->status;
            }
            eshkol_tagged_value_t qtv = make_fact_tv(query);
            for (size_t i = 0; i < collected.size(); i++) {
                eshkol_fact_t* ans = collected[i].first;
                eshkol_tagged_value_t atv = make_fact_tv(ans);
                eshkol_substitution_t* empty_s = eshkol_make_substitution(arena, 8);
                eshkol_substitution_t* bound = eshkol_unify(arena, &qtv, &atv, empty_s);
                arr[i].fact = ans;
                arr[i].subst = bound ? bound : empty_s;
                arr[i].truth = collected[i].second;
            }
            out_result->answers = arr;
        }
        return ESHKOL_SLG_OK;
    }
};

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_slg_config_t eshkol_slg_default_config(void) {
    eshkol_slg_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.max_steps = 0;
    cfg.max_subgoals = 0;
    cfg.max_answers = 0;
    cfg.max_wfs_iters = 1024u;
    return cfg;
}

extern "C" eshkol_slg_engine_t* eshkol_slg_create(arena_t* arena,
                                                  const eshkol_knowledge_base_t* kb,
                                                  const eshkol_slg_config_t* cfg) {
    if (!arena) return nullptr;
    eshkol_slg_config_t effective = cfg ? *cfg : eshkol_slg_default_config();
    return new (std::nothrow) eshkol_slg_engine(arena, kb, effective);
}

extern "C" void eshkol_slg_destroy(eshkol_slg_engine_t* engine) {
    delete engine;
}

extern "C" bool eshkol_slg_add_rule(eshkol_slg_engine_t* engine, const eshkol_rule_t* rule) {
    if (!engine || !rule || !rule->head) return false;
    engine->rules.push_back(rule);
    PredArity pa{rule->head->predicate, rule->head->arity};
    engine->rules_by_head[pa].push_back(rule);
    return true;
}

extern "C" eshkol_fact_t* eshkol_slg_make_negated_goal(arena_t* arena,
                                                       const eshkol_fact_t* inner) {
    if (!arena || !inner) return nullptr;
    eshkol_tagged_value_t arg = make_fact_tv(inner);
    return eshkol_make_fact(arena, pred_not_id(), &arg, 1u);
}

extern "C" bool eshkol_slg_is_negated_goal(const eshkol_fact_t* goal,
                                           const eshkol_fact_t** out_inner) {
    return is_negated_fact(goal, out_inner);
}

extern "C" eshkol_slg_status_t eshkol_slg_query(eshkol_slg_engine_t* engine,
                                                const eshkol_fact_t* query,
                                                eshkol_slg_result_t* out_result) {
    if (!engine || !query || !out_result) {
        if (out_result) {
            std::memset(out_result, 0, sizeof(*out_result));
            out_result->status = ESHKOL_SLG_INVALID_INPUT;
        }
        return ESHKOL_SLG_INVALID_INPUT;
    }
    return engine->run_query(query, out_result);
}

extern "C" eshkol_slg_status_t eshkol_slg_query_ground(eshkol_slg_engine_t* engine,
                                                       const eshkol_fact_t* ground_atom,
                                                       eshkol_slg_truth_t* out_truth,
                                                       uint64_t* out_steps_used) {
    if (!engine || !ground_atom || !out_truth) return ESHKOL_SLG_INVALID_INPUT;
    eshkol_slg_result_t res;
    eshkol_slg_status_t st = engine->run_query(ground_atom, &res);
    if (out_steps_used) *out_steps_used = res.steps_used;
    if (st != ESHKOL_SLG_OK) {
        *out_truth = ESHKOL_SLG_UNKNOWN;
        return st;
    }
    eshkol_slg_truth_t best = ESHKOL_SLG_FALSE;
    for (uint32_t i = 0; i < res.num_answers; i++) {
        if (eshkol_fact_equal(res.answers[i].fact, ground_atom)) {
            if (res.answers[i].truth > best) best = res.answers[i].truth;
        }
    }
    *out_truth = best;
    return ESHKOL_SLG_OK;
}

extern "C" void eshkol_slg_clear_tables(eshkol_slg_engine_t* engine) {
    if (!engine) return;
    engine->steps_used = 0;
    engine->subgoals_tabulated = 0;
    engine->consumer_resumptions = 0;
    engine->wfs_iterations = 0;
}
