/*
 * Native CDCL SAT Solver, DRAT Proof Emitter, and Independent DRAT Verifier
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/cdcl_sat.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kNoReason = UINT32_MAX;

inline uint32_t dimacs_to_lit(int32_t d) {
    uint32_t v = static_cast<uint32_t>(d > 0 ? d : -d) - 1u;
    return (v << 1) | (d < 0 ? 1u : 0u);
}

inline int32_t lit_to_dimacs(uint32_t lit) {
    int32_t v = static_cast<int32_t>((lit >> 1) + 1u);
    return (lit & 1u) ? -v : v;
}

inline uint32_t lit_neg(uint32_t lit) {
    return lit ^ 1u;
}

inline uint32_t lit_var(uint32_t lit) {
    return lit >> 1;
}

/* Classic Luby restart sequence (1-indexed: 1, 1, 2, 1, 1, 2, 4, ...). */
uint64_t luby_seq(uint64_t x) {
    uint64_t size = 1;
    uint64_t seq = 0;
    while (size < x + 1) {
        seq++;
        size = 2 * size + 1;
    }
    while (size - 1 != x) {
        size = (size - 1) >> 1;
        seq--;
        x = x % size;
    }
    return 1ULL << seq;
}

struct Watcher {
    uint32_t cid;
    uint32_t blocker;
};

struct Clause {
    std::vector<uint32_t> lits;
    double activity = 0.0;
    uint32_t lbd = 0;
    bool learnt = false;
    bool deleted = false;
};

/* Binary max-heap keyed by VSIDS variable activity. */
class VarOrderHeap {
public:
    explicit VarOrderHeap(const std::vector<double>& act) : activity_(act) {}

    void init(uint32_t num_vars) {
        heap_.clear();
        pos_.assign(num_vars, -1);
        heap_.reserve(num_vars);
        for (uint32_t v = 0; v < num_vars; v++) {
            insert(v);
        }
    }

    bool empty() const { return heap_.empty(); }

    bool in_heap(uint32_t v) const {
        return v < pos_.size() && pos_[v] >= 0;
    }

    void insert(uint32_t v) {
        if (in_heap(v)) return;
        pos_[v] = static_cast<int32_t>(heap_.size());
        heap_.push_back(v);
        sift_up(static_cast<uint32_t>(pos_[v]));
    }

    void decrease(uint32_t v) {
        if (in_heap(v)) sift_up(static_cast<uint32_t>(pos_[v]));
    }

    uint32_t pop_max() {
        uint32_t top = heap_[0];
        pos_[top] = -1;
        uint32_t last = heap_.back();
        heap_.pop_back();
        if (!heap_.empty()) {
            heap_[0] = last;
            pos_[last] = 0;
            sift_down(0);
        }
        return top;
    }

private:
    bool higher(uint32_t a, uint32_t b) const {
        if (activity_[a] != activity_[b]) return activity_[a] > activity_[b];
        return a < b;
    }

    void sift_up(uint32_t idx) {
        uint32_t v = heap_[idx];
        while (idx > 0) {
            uint32_t p = (idx - 1u) >> 1;
            uint32_t pv = heap_[p];
            if (!higher(v, pv)) break;
            heap_[idx] = pv;
            pos_[pv] = static_cast<int32_t>(idx);
            idx = p;
        }
        heap_[idx] = v;
        pos_[v] = static_cast<int32_t>(idx);
    }

    void sift_down(uint32_t idx) {
        uint32_t v = heap_[idx];
        const uint32_t n = static_cast<uint32_t>(heap_.size());
        while (true) {
            uint32_t left = (idx << 1) + 1u;
            if (left >= n) break;
            uint32_t right = left + 1u;
            uint32_t best = (right < n && higher(heap_[right], heap_[left])) ? right : left;
            if (!higher(heap_[best], v)) break;
            heap_[idx] = heap_[best];
            pos_[heap_[best]] = static_cast<int32_t>(idx);
            idx = best;
        }
        heap_[idx] = v;
        pos_[v] = static_cast<int32_t>(idx);
    }

    const std::vector<double>& activity_;
    std::vector<uint32_t> heap_;
    std::vector<int32_t> pos_;
};

} /* anonymous namespace */

struct eshkol_cdcl_solver {
    uint32_t num_vars = 0;
    eshkol_cdcl_config_t cfg{};
    eshkol_cdcl_stats_t stats{};
    eshkol_sat_status_t status = ESHKOL_SAT_UNDECIDED;
    bool ok = true;
    bool emitted_empty_clause = false;

    std::vector<Clause> clauses;
    std::vector<std::vector<Watcher>> watches;

    /* Assignment state: 0 = unassigned, +1 = true, -1 = false */
    std::vector<int8_t> assign;
    std::vector<uint8_t> saved_phase;
    std::vector<uint32_t> level;
    std::vector<uint32_t> reason;
    std::vector<uint32_t> trail;
    std::vector<uint32_t> trail_lim;
    size_t qhead = 0;

    /* VSIDS & clause activity */
    std::vector<double> var_activity;
    double var_inc = 1.0;
    double cla_inc = 1.0;
    VarOrderHeap order_heap;

    /* Conflict analysis scratch */
    std::vector<uint8_t> seen;
    std::vector<uint32_t> analyze_toclear;
    std::vector<uint32_t> analyze_stack;
    std::vector<uint32_t> learnt_buf;
    std::vector<uint32_t> lbd_mark;
    uint32_t lbd_epoch = 1;

    /* Clause DB management */
    uint32_t active_learnts = 0;
    double max_learnts = 2000.0;

    /* Stored model (1-based: indices 1..num_vars) */
    std::vector<int8_t> model;

    /* Structured DRAT proof */
    eshkol_drat_proof_t proof{};

    eshkol_cdcl_solver(uint32_t nvars, const eshkol_cdcl_config_t& c)
        : num_vars(nvars),
          cfg(c),
          watches(static_cast<size_t>(nvars) * 2u),
          assign(nvars, 0),
          saved_phase(nvars, 0),
          level(nvars, 0),
          reason(nvars, kNoReason),
          var_activity(nvars, 0.0),
          order_heap(var_activity),
          seen(nvars, 0),
          lbd_mark(nvars + 2u, 0),
          max_learnts(static_cast<double>(c.max_learnts_init > 0 ? c.max_learnts_init : 2000u)) {
        order_heap.init(nvars);
        std::memset(&stats, 0, sizeof(stats));
        std::memset(&proof, 0, sizeof(proof));
    }

    ~eshkol_cdcl_solver() {
        for (uint32_t i = 0; i < proof.num_steps; i++) {
            std::free(proof.steps[i].lits);
        }
        std::free(proof.steps);
    }

    inline uint32_t decision_level() const {
        return static_cast<uint32_t>(trail_lim.size());
    }

    inline int8_t lit_value(uint32_t lit) const {
        int8_t v = assign[lit_var(lit)];
        return (lit & 1u) ? static_cast<int8_t>(-v) : v;
    }

    void drat_record(eshkol_drat_op_t op, const std::vector<uint32_t>& lits) {
        if (!cfg.emit_drat) return;
        if (op == ESHKOL_DRAT_ADD && lits.empty()) {
            if (emitted_empty_clause) return;
            emitted_empty_clause = true;
        }
        if (proof.num_steps >= proof.capacity) {
            uint32_t new_cap = proof.capacity == 0 ? 64u : proof.capacity * 2u;
            auto* next = static_cast<eshkol_drat_step_t*>(
                std::realloc(proof.steps, static_cast<size_t>(new_cap) * sizeof(eshkol_drat_step_t)));
            if (!next) return;
            proof.steps = next;
            proof.capacity = new_cap;
        }
        eshkol_drat_step_t& step = proof.steps[proof.num_steps++];
        step.op = op;
        step.num_lits = static_cast<uint32_t>(lits.size());
        if (step.num_lits > 0) {
            step.lits = static_cast<int32_t*>(
                std::malloc(static_cast<size_t>(step.num_lits) * sizeof(int32_t)));
            if (!step.lits) {
                step.num_lits = 0;
            } else {
                for (uint32_t i = 0; i < step.num_lits; i++) {
                    step.lits[i] = lit_to_dimacs(lits[i]);
                }
            }
        } else {
            step.lits = nullptr;
        }
        stats.drat_steps = proof.num_steps;
    }

    void drat_record_empty() {
        std::vector<uint32_t> empty;
        drat_record(ESHKOL_DRAT_ADD, empty);
    }

    bool enqueue(uint32_t lit, uint32_t from_cid) {
        int8_t cur = lit_value(lit);
        if (cur != 0) return cur > 0;
        uint32_t v = lit_var(lit);
        assign[v] = (lit & 1u) ? -1 : 1;
        level[v] = decision_level();
        reason[v] = from_cid;
        trail.push_back(lit);
        return true;
    }

    void attach_clause(uint32_t cid) {
        const Clause& c = clauses[cid];
        watches[c.lits[0]].push_back({cid, c.lits[1]});
        watches[c.lits[1]].push_back({cid, c.lits[0]});
    }

    void var_bump_activity(uint32_t v) {
        var_activity[v] += var_inc;
        if (var_activity[v] > 1e100) {
            for (double& a : var_activity) a *= 1e-100;
            var_inc *= 1e-100;
        }
        order_heap.decrease(v);
    }

    void var_decay_activity() {
        var_inc *= (1.0 / cfg.var_decay);
    }

    void cla_bump_activity(Clause& c) {
        c.activity += cla_inc;
        if (c.activity > 1e20) {
            for (Clause& cl : clauses) {
                if (cl.learnt && !cl.deleted) cl.activity *= 1e-20;
            }
            cla_inc *= 1e-20;
        }
    }

    void cla_decay_activity() {
        cla_inc *= (1.0 / cfg.clause_decay);
    }

    uint32_t compute_lbd(const std::vector<uint32_t>& lits) {
        lbd_epoch++;
        if (lbd_epoch == 0) {
            std::fill(lbd_mark.begin(), lbd_mark.end(), 0u);
            lbd_epoch = 1;
        }
        uint32_t count = 0;
        for (uint32_t lit : lits) {
            uint32_t lvl = level[lit_var(lit)];
            if (lvl == 0) continue;
            if (lvl >= lbd_mark.size()) lbd_mark.resize(lvl + 2u, 0u);
            if (lbd_mark[lvl] != lbd_epoch) {
                lbd_mark[lvl] = lbd_epoch;
                count++;
            }
        }
        return count;
    }

    bool locked(uint32_t cid) const {
        const Clause& c = clauses[cid];
        if (c.deleted || c.lits.empty()) return false;
        uint32_t lit0 = c.lits[0];
        return lit_value(lit0) == 1 && reason[lit_var(lit0)] == cid;
    }

    void cancel_until(uint32_t target_level) {
        if (decision_level() <= target_level) return;
        size_t cut = trail_lim[target_level];
        for (size_t i = trail.size(); i-- > cut; ) {
            uint32_t lit = trail[i];
            uint32_t v = lit_var(lit);
            if (cfg.phase_saving) {
                saved_phase[v] = (assign[v] > 0) ? 1u : 0u;
            }
            assign[v] = 0;
            reason[v] = kNoReason;
            order_heap.insert(v);
        }
        trail.resize(cut);
        trail_lim.resize(target_level);
        qhead = cut;
    }

    /* Two-watched-literal BCP. Returns conflicting cid, or kNoReason. */
    uint32_t propagate(bool& budget_hit) {
        budget_hit = false;
        while (qhead < trail.size()) {
            if (cfg.max_propagations > 0 && stats.propagations >= cfg.max_propagations) {
                budget_hit = true;
                return kNoReason;
            }
            uint32_t p = trail[qhead++];
            stats.propagations++;
            uint32_t false_lit = lit_neg(p);
            auto& ws = watches[false_lit];
            size_t i = 0;
            size_t j = 0;
            uint32_t confl = kNoReason;

            while (i < ws.size()) {
                Watcher w = ws[i++];
                if (clauses[w.cid].deleted) continue;
                if (lit_value(w.blocker) == 1) {
                    ws[j++] = w;
                    continue;
                }
                Clause& c = clauses[w.cid];
                if (c.lits[0] == false_lit) {
                    std::swap(c.lits[0], c.lits[1]);
                }
                uint32_t first = c.lits[0];
                Watcher new_w{w.cid, first};
                if (first != w.blocker && lit_value(first) == 1) {
                    ws[j++] = new_w;
                    continue;
                }

                bool found_watch = false;
                for (size_t k = 2; k < c.lits.size(); k++) {
                    if (lit_value(c.lits[k]) != -1) {
                        std::swap(c.lits[1], c.lits[k]);
                        watches[c.lits[1]].push_back(new_w);
                        found_watch = true;
                        break;
                    }
                }
                if (found_watch) continue;

                ws[j++] = new_w;
                if (lit_value(first) == -1) {
                    confl = w.cid;
                    qhead = trail.size();
                    while (i < ws.size()) ws[j++] = ws[i++];
                    break;
                } else {
                    enqueue(first, w.cid);
                }
            }
            ws.resize(j);
            if (confl != kNoReason) return confl;
        }
        return kNoReason;
    }

    bool lit_redundant(uint32_t p, uint64_t abstract_levels) {
        analyze_stack.clear();
        analyze_stack.push_back(p);
        const size_t top = analyze_toclear.size();

        while (!analyze_stack.empty()) {
            uint32_t r_cid = reason[lit_var(analyze_stack.back())];
            analyze_stack.pop_back();
            const Clause& c = clauses[r_cid];
            for (size_t i = 1; i < c.lits.size(); i++) {
                uint32_t l = c.lits[i];
                uint32_t v = lit_var(l);
                if (!seen[v] && level[v] > 0) {
                    uint64_t lvl_mask = 1ULL << (level[v] & 63u);
                    if (reason[v] != kNoReason && (lvl_mask & abstract_levels) != 0) {
                        seen[v] = 1;
                        analyze_stack.push_back(l);
                        analyze_toclear.push_back(l);
                    } else {
                        for (size_t k = top; k < analyze_toclear.size(); k++) {
                            seen[lit_var(analyze_toclear[k])] = 0;
                        }
                        analyze_toclear.resize(top);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void analyze(uint32_t confl_cid, std::vector<uint32_t>& out_learnt, uint32_t& out_bt_level) {
        out_learnt.clear();
        out_learnt.push_back(0u); /* slot 0 reserved for asserting literal */
        int path_c = 0;
        uint32_t p = UINT32_MAX;
        size_t index = trail.size();
        uint32_t confl = confl_cid;
        const uint32_t cur_lvl = decision_level();

        do {
            Clause& c = clauses[confl];
            if (c.learnt) {
                cla_bump_activity(c);
                uint32_t new_lbd = compute_lbd(c.lits);
                if (new_lbd < c.lbd) c.lbd = new_lbd;
            }
            size_t start = (p == UINT32_MAX) ? 0u : 1u;
            for (size_t k = start; k < c.lits.size(); k++) {
                uint32_t q = c.lits[k];
                uint32_t v = lit_var(q);
                if (!seen[v] && level[v] > 0) {
                    var_bump_activity(v);
                    seen[v] = 1;
                    if (level[v] >= cur_lvl) {
                        path_c++;
                    } else {
                        out_learnt.push_back(q);
                    }
                }
            }
            while (index > 0) {
                index--;
                if (seen[lit_var(trail[index])]) break;
            }
            p = trail[index];
            confl = reason[lit_var(p)];
            seen[lit_var(p)] = 0;
            path_c--;
        } while (path_c > 0);

        out_learnt[0] = lit_neg(p);

        analyze_toclear = out_learnt;
        if (cfg.minimize_clauses && out_learnt.size() > 1) {
            uint64_t abstract_levels = 0;
            for (size_t i = 1; i < out_learnt.size(); i++) {
                abstract_levels |= (1ULL << (level[lit_var(out_learnt[i])] & 63u));
            }
            size_t j = 1;
            for (size_t i = 1; i < out_learnt.size(); i++) {
                uint32_t q = out_learnt[i];
                if (reason[lit_var(q)] == kNoReason || !lit_redundant(q, abstract_levels)) {
                    out_learnt[j++] = q;
                } else {
                    stats.minimized_lits++;
                }
            }
            out_learnt.resize(j);
        }

        for (uint32_t lit : analyze_toclear) {
            seen[lit_var(lit)] = 0;
        }
        analyze_toclear.clear();

        if (out_learnt.size() == 1) {
            out_bt_level = 0;
        } else {
            size_t max_i = 1;
            for (size_t i = 2; i < out_learnt.size(); i++) {
                if (level[lit_var(out_learnt[i])] > level[lit_var(out_learnt[max_i])]) {
                    max_i = i;
                }
            }
            std::swap(out_learnt[1], out_learnt[max_i]);
            out_bt_level = level[lit_var(out_learnt[1])];
        }
    }

    void compact_watches() {
        for (auto& ws : watches) {
            size_t j = 0;
            for (size_t i = 0; i < ws.size(); i++) {
                if (!clauses[ws[i].cid].deleted) {
                    ws[j++] = ws[i];
                }
            }
            ws.resize(j);
        }
    }

    void reduce_db() {
        std::vector<uint32_t> candidates;
        candidates.reserve(active_learnts);
        for (uint32_t cid = 0; cid < clauses.size(); cid++) {
            const Clause& c = clauses[cid];
            if (!c.learnt || c.deleted) continue;
            if (c.lits.size() <= 2 || c.lbd <= cfg.lbd_tier1) continue;
            if (locked(cid)) continue;
            candidates.push_back(cid);
        }
        if (candidates.size() < 2) {
            max_learnts *= cfg.learntsize_inc;
            return;
        }

        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
            const Clause& ca = clauses[a];
            const Clause& cb = clauses[b];
            if (ca.lbd != cb.lbd) return ca.lbd > cb.lbd;
            return ca.activity < cb.activity;
        });

        const size_t n_remove = candidates.size() / 2u;
        for (size_t i = 0; i < n_remove; i++) {
            uint32_t cid = candidates[i];
            Clause& c = clauses[cid];
            if (!c.deleted) {
                drat_record(ESHKOL_DRAT_DELETE, c.lits);
                c.deleted = true;
                if (active_learnts > 0) active_learnts--;
                stats.deleted_clauses++;
            }
        }
        compact_watches();
        max_learnts *= cfg.learntsize_inc;
    }

    uint32_t pick_branch_lit() {
        while (!order_heap.empty()) {
            uint32_t v = order_heap.pop_max();
            if (assign[v] == 0) {
                uint32_t sign = (cfg.phase_saving && saved_phase[v] != 0) ? 0u : 1u;
                return (v << 1) | sign;
            }
        }
        return UINT32_MAX;
    }

    bool add_clause(const int32_t* dimacs_lits, uint32_t num_lits) {
        if (num_lits > 0 && !dimacs_lits) return false;
        std::vector<uint32_t> lits;
        lits.reserve(num_lits);
        for (uint32_t i = 0; i < num_lits; i++) {
            int32_t d = dimacs_lits[i];
            if (d == 0) return false;
            int64_t av = d > 0 ? static_cast<int64_t>(d) : -static_cast<int64_t>(d);
            if (av < 1 || av > static_cast<int64_t>(num_vars)) return false;
            lits.push_back(dimacs_to_lit(d));
        }

        if (!ok) return true;

        std::sort(lits.begin(), lits.end());
        std::vector<uint32_t> cleaned;
        cleaned.reserve(lits.size());
        uint32_t prev = UINT32_MAX;
        for (uint32_t lit : lits) {
            if (lit == prev) continue;
            if (prev != UINT32_MAX && lit == lit_neg(prev)) {
                /* Tautological clause is always satisfied. */
                return true;
            }
            prev = lit;
            int8_t val = lit_value(lit);
            if (val == 1) return true;  /* Already satisfied at level 0. */
            if (val == -1) continue;    /* Falsified at level 0. */
            cleaned.push_back(lit);
        }

        if (cleaned.empty()) {
            ok = false;
            return true;
        }
        if (cleaned.size() == 1) {
            if (!enqueue(cleaned[0], kNoReason)) {
                ok = false;
            }
            return true;
        }

        Clause c;
        c.lits = std::move(cleaned);
        c.learnt = false;
        c.deleted = false;
        c.lbd = 0;
        uint32_t cid = static_cast<uint32_t>(clauses.size());
        clauses.push_back(std::move(c));
        attach_clause(cid);
        return true;
    }

    eshkol_sat_status_t solve() {
        if (!ok) {
            drat_record_empty();
            status = ESHKOL_SAT_UNSATISFIABLE;
            return status;
        }

        /* Initial level-0 propagation */
        bool budget_hit = false;
        uint32_t init_confl = propagate(budget_hit);
        if (budget_hit) {
            status = ESHKOL_SAT_BUDGET_EXCEEDED;
            return status;
        }
        if (init_confl != kNoReason) {
            ok = false;
            drat_record_empty();
            status = ESHKOL_SAT_UNSATISFIABLE;
            return status;
        }

        uint64_t restart_idx = 0;
        uint64_t conflicts_in_restart = 0;
        const uint64_t rbase = cfg.restart_base > 0 ? cfg.restart_base : 100u;
        uint64_t restart_limit = luby_seq(restart_idx) * rbase;

        while (true) {
            uint32_t confl = propagate(budget_hit);
            if (budget_hit) {
                status = ESHKOL_SAT_BUDGET_EXCEEDED;
                return status;
            }

            if (confl != kNoReason) {
                stats.conflicts++;
                conflicts_in_restart++;
                if (decision_level() == 0) {
                    ok = false;
                    drat_record_empty();
                    status = ESHKOL_SAT_UNSATISFIABLE;
                    return status;
                }
                if (cfg.max_conflicts > 0 && stats.conflicts >= cfg.max_conflicts) {
                    status = ESHKOL_SAT_BUDGET_EXCEEDED;
                    return status;
                }

                uint32_t bt_level = 0;
                analyze(confl, learnt_buf, bt_level);
                uint32_t lbd = compute_lbd(learnt_buf);
                drat_record(ESHKOL_DRAT_ADD, learnt_buf);
                stats.learned_clauses++;

                cancel_until(bt_level);
                if (learnt_buf.size() == 1) {
                    if (!enqueue(learnt_buf[0], kNoReason)) {
                        ok = false;
                        drat_record_empty();
                        status = ESHKOL_SAT_UNSATISFIABLE;
                        return status;
                    }
                } else {
                    Clause c;
                    c.lits = learnt_buf;
                    c.learnt = true;
                    c.deleted = false;
                    c.lbd = lbd;
                    c.activity = 0.0;
                    uint32_t cid = static_cast<uint32_t>(clauses.size());
                    clauses.push_back(std::move(c));
                    attach_clause(cid);
                    cla_bump_activity(clauses[cid]);
                    active_learnts++;
                    enqueue(clauses[cid].lits[0], cid);
                }

                var_decay_activity();
                cla_decay_activity();
            } else {
                if (conflicts_in_restart >= restart_limit) {
                    cancel_until(0);
                    restart_idx++;
                    conflicts_in_restart = 0;
                    restart_limit = luby_seq(restart_idx) * rbase;
                    stats.restarts++;
                }

                if (static_cast<double>(active_learnts) > max_learnts) {
                    reduce_db();
                }

                uint32_t next_lit = pick_branch_lit();
                if (next_lit == UINT32_MAX) {
                    model.assign(static_cast<size_t>(num_vars) + 1u, 0);
                    for (uint32_t v = 0; v < num_vars; v++) {
                        model[v + 1u] = (assign[v] >= 0) ? 1 : -1;
                    }
                    status = ESHKOL_SAT_SATISFIABLE;
                    return status;
                }

                if (cfg.max_decisions > 0 && stats.decisions >= cfg.max_decisions) {
                    status = ESHKOL_SAT_BUDGET_EXCEEDED;
                    return status;
                }

                stats.decisions++;
                trail_lim.push_back(trail.size());
                enqueue(next_lit, kNoReason);
            }
        }
    }
};

/* ============================================================================
 * Independent Forward DRAT Verifier (Zero Shared State With CDCL Solver)
 * ============================================================================ */

namespace drat_verifier {

struct VerifierClause {
    std::vector<int32_t> lits; /* Sorted unique DIMACS literals */
    bool active = true;
};

struct VerifierDB {
    uint32_t num_vars = 0;
    std::vector<VerifierClause> clauses;
    std::vector<int8_t> val; /* 1-based: 0=undef, +1=true, -1=false */
    std::vector<uint32_t> assigned_vars;

    explicit VerifierDB(uint32_t nvars) : num_vars(nvars), val(static_cast<size_t>(nvars) + 1u, 0) {}

    inline int8_t eval_lit(int32_t lit) const {
        uint32_t v = static_cast<uint32_t>(lit > 0 ? lit : -lit);
        if (v == 0 || v > num_vars) return 0;
        int8_t a = val[v];
        return (lit > 0) ? a : static_cast<int8_t>(-a);
    }

    bool assign_lit(int32_t lit) {
        uint32_t v = static_cast<uint32_t>(lit > 0 ? lit : -lit);
        if (v == 0 || v > num_vars) return false;
        int8_t target = (lit > 0) ? 1 : -1;
        if (val[v] == 0) {
            val[v] = target;
            assigned_vars.push_back(v);
            return true;
        }
        return val[v] == target;
    }

    void reset_assignments() {
        for (uint32_t v : assigned_vars) val[v] = 0;
        assigned_vars.clear();
    }

    /* Normalize a DIMACS clause: validate bounds, sort, deduplicate.
     * Sets is_tautology = true if complementary literals appear. */
    bool normalize(const int32_t* raw, uint32_t n,
                   std::vector<int32_t>& out, bool& is_tautology) const {
        is_tautology = false;
        out.clear();
        if (n > 0 && !raw) return false;
        out.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            int32_t d = raw[i];
            if (d == 0) return false;
            int64_t av = d > 0 ? static_cast<int64_t>(d) : -static_cast<int64_t>(d);
            if (av < 1 || av > static_cast<int64_t>(num_vars)) return false;
            out.push_back(d);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        for (int32_t d : out) {
            if (d > 0 && std::binary_search(out.begin(), out.end(), -d)) {
                is_tautology = true;
                break;
            }
        }
        return true;
    }

    /* Run pure forward unit propagation from current assumptions.
     * Returns true iff a conflict (falsified active clause) is derived. */
    bool propagate_to_conflict() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const VerifierClause& c : clauses) {
                if (!c.active) continue;
                bool satisfied = false;
                uint32_t undef_count = 0;
                int32_t last_undef = 0;
                for (int32_t lit : c.lits) {
                    int8_t ev = eval_lit(lit);
                    if (ev == 1) {
                        satisfied = true;
                        break;
                    }
                    if (ev == 0) {
                        undef_count++;
                        last_undef = lit;
                    }
                }
                if (satisfied) continue;
                if (undef_count == 0) return true; /* Conflict! */
                if (undef_count == 1) {
                    if (!assign_lit(last_undef)) return true;
                    changed = true;
                }
            }
        }
        return false;
    }

    /* Check whether `clause_lits` is a Reverse Unit Propagation (RUP) clause. */
    bool is_rup(const std::vector<int32_t>& clause_lits) {
        reset_assignments();
        for (int32_t lit : clause_lits) {
            if (!assign_lit(-lit)) {
                reset_assignments();
                return true;
            }
        }
        bool conflict = propagate_to_conflict();
        reset_assignments();
        return conflict;
    }

    /* Check whether `clause_lits` is RAT on pivot `pivot_lit`. */
    bool is_rat(const std::vector<int32_t>& clause_lits, int32_t pivot_lit) {
        if (clause_lits.empty()) return false;
        const int32_t neg_pivot = -pivot_lit;
        for (const VerifierClause& d : clauses) {
            if (!d.active) continue;
            if (!std::binary_search(d.lits.begin(), d.lits.end(), neg_pivot)) {
                continue;
            }
            std::vector<int32_t> resolvent;
            resolvent.reserve(clause_lits.size() + d.lits.size());
            for (int32_t l : clause_lits) {
                if (l != pivot_lit) resolvent.push_back(l);
            }
            for (int32_t l : d.lits) {
                if (l != neg_pivot) resolvent.push_back(l);
            }
            std::sort(resolvent.begin(), resolvent.end());
            resolvent.erase(std::unique(resolvent.begin(), resolvent.end()), resolvent.end());

            bool taut = false;
            for (int32_t l : resolvent) {
                if (l > 0 && std::binary_search(resolvent.begin(), resolvent.end(), -l)) {
                    taut = true;
                    break;
                }
            }
            if (taut) continue;
            if (!is_rup(resolvent)) return false;
        }
        return true;
    }

    bool delete_clause(const std::vector<int32_t>& norm_lits) {
        for (VerifierClause& c : clauses) {
            if (c.active && c.lits == norm_lits) {
                c.active = false;
                return true;
            }
        }
        return false;
    }
};

} /* namespace drat_verifier */

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_cdcl_config_t eshkol_cdcl_default_config(void) {
    eshkol_cdcl_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.max_conflicts = 0;
    cfg.max_propagations = 0;
    cfg.max_decisions = 0;
    cfg.var_decay = 0.95;
    cfg.clause_decay = 0.999;
    cfg.restart_base = 100u;
    cfg.max_learnts_init = 2000u;
    cfg.learntsize_inc = 1.1;
    cfg.lbd_tier1 = 2u;
    cfg.emit_drat = true;
    cfg.phase_saving = true;
    cfg.minimize_clauses = true;
    return cfg;
}

extern "C" eshkol_cdcl_solver_t* eshkol_cdcl_create(uint32_t num_vars,
                                                    const eshkol_cdcl_config_t* cfg) {
    eshkol_cdcl_config_t effective = cfg ? *cfg : eshkol_cdcl_default_config();
    if (!(effective.var_decay > 0.0 && effective.var_decay < 1.0)) {
        effective.var_decay = 0.95;
    }
    if (!(effective.clause_decay > 0.0 && effective.clause_decay < 1.0)) {
        effective.clause_decay = 0.999;
    }
    if (!(effective.learntsize_inc > 1.0)) {
        effective.learntsize_inc = 1.1;
    }
    return new (std::nothrow) eshkol_cdcl_solver(num_vars, effective);
}

extern "C" void eshkol_cdcl_destroy(eshkol_cdcl_solver_t* solver) {
    delete solver;
}

extern "C" bool eshkol_cdcl_add_clause(eshkol_cdcl_solver_t* solver,
                                       const int32_t* lits,
                                       uint32_t num_lits) {
    if (!solver) return false;
    return solver->add_clause(lits, num_lits);
}

extern "C" eshkol_sat_status_t eshkol_cdcl_solve(eshkol_cdcl_solver_t* solver) {
    if (!solver) return ESHKOL_SAT_INVALID_INPUT;
    return solver->solve();
}

extern "C" int8_t eshkol_cdcl_model_value(const eshkol_cdcl_solver_t* solver, uint32_t var) {
    if (!solver || solver->status != ESHKOL_SAT_SATISFIABLE) return 0;
    if (var == 0 || var > solver->num_vars || var >= solver->model.size()) return 0;
    return solver->model[var];
}

extern "C" bool eshkol_cdcl_copy_model(const eshkol_cdcl_solver_t* solver,
                                       int8_t* out_model,
                                       uint32_t out_len) {
    if (!solver || !out_model || solver->status != ESHKOL_SAT_SATISFIABLE) return false;
    if (out_len < solver->num_vars + 1u || solver->model.size() < solver->num_vars + 1u) {
        return false;
    }
    std::memcpy(out_model, solver->model.data(),
                static_cast<size_t>(solver->num_vars + 1u) * sizeof(int8_t));
    return true;
}

extern "C" const eshkol_drat_proof_t* eshkol_cdcl_get_proof(const eshkol_cdcl_solver_t* solver) {
    if (!solver) return nullptr;
    return &solver->proof;
}

extern "C" void eshkol_cdcl_get_stats(const eshkol_cdcl_solver_t* solver,
                                      eshkol_cdcl_stats_t* out_stats) {
    if (!solver || !out_stats) return;
    *out_stats = solver->stats;
}

extern "C" bool eshkol_cdcl_verify_model(uint32_t num_vars,
                                         const eshkol_cnf_clause_t* clauses,
                                         uint32_t num_clauses,
                                         const int8_t* model,
                                         uint32_t model_len) {
    if (!model || model_len < num_vars + 1u) return false;
    if (num_clauses > 0 && !clauses) return false;
    for (uint32_t i = 0; i < num_clauses; i++) {
        const eshkol_cnf_clause_t& cl = clauses[i];
        if (cl.num_lits > 0 && !cl.lits) return false;
        bool sat = false;
        for (uint32_t j = 0; j < cl.num_lits; j++) {
            int32_t lit = cl.lits[j];
            if (lit == 0) return false;
            int64_t av = lit > 0 ? static_cast<int64_t>(lit) : -static_cast<int64_t>(lit);
            if (av < 1 || av > static_cast<int64_t>(num_vars)) return false;
            int8_t mv = model[static_cast<size_t>(av)];
            if (mv != 1 && mv != -1) return false;
            if ((lit > 0 && mv == 1) || (lit < 0 && mv == -1)) {
                sat = true;
                break;
            }
        }
        if (!sat) return false;
    }
    return true;
}

extern "C" bool eshkol_cdcl_verify_drat(uint32_t num_vars,
                                        const eshkol_cnf_clause_t* orig_clauses,
                                        uint32_t num_orig_clauses,
                                        const eshkol_drat_proof_t* proof) {
    if (!proof || (proof->num_steps > 0 && !proof->steps)) return false;
    if (num_orig_clauses > 0 && !orig_clauses) return false;

    drat_verifier::VerifierDB db(num_vars);
    db.clauses.reserve(static_cast<size_t>(num_orig_clauses) + proof->num_steps);

    for (uint32_t i = 0; i < num_orig_clauses; i++) {
        std::vector<int32_t> norm;
        bool taut = false;
        if (!db.normalize(orig_clauses[i].lits, orig_clauses[i].num_lits, norm, taut)) {
            return false;
        }
        if (taut) continue;
        db.clauses.push_back({std::move(norm), true});
    }

    bool derived_empty = false;
    for (uint32_t s = 0; s < proof->num_steps; s++) {
        const eshkol_drat_step_t& step = proof->steps[s];
        std::vector<int32_t> norm;
        bool taut = false;
        if (!db.normalize(step.lits, step.num_lits, norm, taut)) {
            return false;
        }

        if (step.op == ESHKOL_DRAT_DELETE) {
            if (taut) continue;
            if (!db.delete_clause(norm)) return false;
            continue;
        }

        if (step.op != ESHKOL_DRAT_ADD) return false;

        if (norm.empty()) {
            if (!db.is_rup(norm)) return false;
            derived_empty = true;
            return true;
        }

        if (taut) {
            continue;
        }

        if (db.is_rup(norm)) {
            db.clauses.push_back({std::move(norm), true});
            continue;
        }

        /* Fallback to RAT check on the step's leading pivot literal. */
        int32_t pivot = step.lits[0];
        if (db.is_rat(norm, pivot)) {
            db.clauses.push_back({std::move(norm), true});
            continue;
        }

        return false;
    }

    return derived_empty;
}
