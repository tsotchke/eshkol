/*
 * Native Answer Set Programming (ASP) Solver (Stable Model Semantics)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/asp_solver.h>
#include <eshkol/core/cdcl_sat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

enum class RuleKind {
    Normal = 0,
    Constraint = 1,
    Choice = 2
};

struct AspRule {
    RuleKind kind = RuleKind::Normal;
    std::vector<uint32_t> heads; /* 1-based atom ids */
    uint32_t min_choose = 0;
    uint32_t max_choose = 0;
    std::vector<uint32_t> pos;
    std::vector<uint32_t> neg;
};

/* Encode exact cardinality constraint over literals lits[0..n-1] conditioned on guard_lit:
 * guard_lit => (sum_i lits[i] >= min_k && sum_i lits[i] <= max_k)
 * using Sinz sequential counter auxiliary variables. */
void encode_conditional_cardinality(std::vector<std::vector<int32_t>>& clauses,
                                    uint32_t& num_vars,
                                    int32_t guard_lit,
                                    const std::vector<int32_t>& lits,
                                    uint32_t min_k,
                                    uint32_t max_k) {
    const uint32_t n = static_cast<uint32_t>(lits.size());
    if (n == 0) {
        if (min_k > 0) clauses.push_back({-guard_lit});
        return;
    }

    /* Helper: guard => sum(xs) <= K (Sinz sequential counter) */
    auto encode_at_most = [&](const std::vector<int32_t>& xs, uint32_t K) {
        const uint32_t m = static_cast<uint32_t>(xs.size());
        if (K >= m) return;
        if (K == 0) {
            for (int32_t x : xs) clauses.push_back({-guard_lit, -x});
            return;
        }
        /* s(i, j) for i in 0..m-2, j in 0..K-1 means sum(xs[0..i]) >= j+1 */
        std::vector<std::vector<int32_t>> s(m - 1u, std::vector<int32_t>(K, 0));
        for (uint32_t i = 0; i + 1 < m; i++) {
            for (uint32_t j = 0; j < K; j++) {
                s[i][j] = static_cast<int32_t>(++num_vars);
            }
        }
        /* i = 0 */
        clauses.push_back({-guard_lit, -xs[0], s[0][0]});
        for (uint32_t j = 1; j < K; j++) {
            clauses.push_back({-guard_lit, -s[0][j]});
        }
        /* 0 < i < m - 1 */
        for (uint32_t i = 1; i + 1 < m; i++) {
            clauses.push_back({-guard_lit, -xs[i], s[i][0]});
            clauses.push_back({-guard_lit, -s[i - 1u][0], s[i][0]});
            for (uint32_t j = 1; j < K; j++) {
                clauses.push_back({-guard_lit, -xs[i], -s[i - 1u][j - 1u], s[i][j]});
                clauses.push_back({-guard_lit, -s[i - 1u][j], s[i][j]});
            }
            clauses.push_back({-guard_lit, -xs[i], -s[i - 1u][K - 1u]});
        }
        /* i = m - 1 */
        clauses.push_back({-guard_lit, -xs[m - 1u], -s[m - 2u][K - 1u]});
    };

    if (max_k < n) {
        encode_at_most(lits, max_k);
    }
    if (min_k > 0) {
        if (min_k > n) {
            clauses.push_back({-guard_lit});
        } else {
            /* sum(lits) >= min_k  <=>  sum(~lits) <= n - min_k */
            std::vector<int32_t> neg_lits(n);
            for (uint32_t i = 0; i < n; i++) neg_lits[i] = -lits[i];
            encode_at_most(neg_lits, n - min_k);
        }
    }
}

} /* anonymous namespace */

struct eshkol_asp_solver {
    uint32_t num_atoms = 0;
    eshkol_asp_config_t cfg{};
    std::vector<AspRule> rules;
    std::vector<std::vector<uint32_t>> model_storage;
    std::vector<eshkol_asp_model_t> model_views;

    eshkol_asp_solver(uint32_t n, const eshkol_asp_config_t& c)
        : num_atoms(n), cfg(c) {}

    bool valid_atom(uint32_t a) const {
        return a >= 1u && a <= num_atoms;
    }

    /* Compute least Herbrand model of the Gelfond-Lifschitz reduct P^M
     * given candidate truth assignment `in_m[1..num_atoms]`. */
    void compute_reduct_least_model(const std::vector<uint8_t>& in_m,
                                    std::vector<uint8_t>& founded) const {
        founded.assign(static_cast<size_t>(num_atoms) + 1u, 0u);
        bool changed = true;
        while (changed) {
            changed = false;
            for (const AspRule& r : rules) {
                if (r.kind == RuleKind::Constraint) continue;
                bool neg_ok = true;
                for (uint32_t q : r.neg) {
                    if (in_m[q]) { neg_ok = false; break; }
                }
                if (!neg_ok) continue;

                bool pos_ok = true;
                for (uint32_t p : r.pos) {
                    if (!founded[p]) { pos_ok = false; break; }
                }
                if (!pos_ok) continue;

                if (r.kind == RuleKind::Normal) {
                    uint32_t h = r.heads[0];
                    if (!founded[h]) {
                        founded[h] = 1u;
                        changed = true;
                    }
                } else if (r.kind == RuleKind::Choice) {
                    /* In choice reduct P^M, chosen heads h in M become positive facts h :- pos */
                    for (uint32_t h : r.heads) {
                        if (in_m[h] && !founded[h]) {
                            founded[h] = 1u;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    /* Independently verify that `in_m[1..num_atoms]` is an exact Gelfond-Lifschitz stable model. */
    bool verify_model_vec(const std::vector<uint8_t>& in_m) const {
        /* 1. Verify all rules, integrity constraints, and choice cardinality bounds */
        for (const AspRule& r : rules) {
            bool body_holds = true;
            for (uint32_t p : r.pos) {
                if (!in_m[p]) { body_holds = false; break; }
            }
            if (body_holds) {
                for (uint32_t q : r.neg) {
                    if (in_m[q]) { body_holds = false; break; }
                }
            }
            if (!body_holds) continue;

            if (r.kind == RuleKind::Normal) {
                if (!in_m[r.heads[0]]) return false;
            } else if (r.kind == RuleKind::Constraint) {
                return false;
            } else if (r.kind == RuleKind::Choice) {
                uint32_t cnt = 0;
                for (uint32_t h : r.heads) {
                    if (in_m[h]) cnt++;
                }
                if (cnt < r.min_choose || cnt > r.max_choose) return false;
            }
        }

        /* 2. Verify M == lfp(P^M) (exact unfounded-set absence and minimality) */
        std::vector<uint8_t> founded;
        compute_reduct_least_model(in_m, founded);
        for (uint32_t a = 1; a <= num_atoms; a++) {
            if (founded[a] != in_m[a]) return false;
        }
        return true;
    }

    eshkol_asp_status_t solve(eshkol_asp_stats_t* out_stats) {
        model_storage.clear();
        model_views.clear();
        eshkol_asp_stats_t stats{};
        std::memset(&stats, 0, sizeof(stats));

        /* Build Clark completion + rule satisfaction CNF */
        uint32_t total_vars = num_atoms;
        const uint32_t num_rules = static_cast<uint32_t>(rules.size());
        std::vector<int32_t> body_var(num_rules, 0);
        std::vector<std::vector<int32_t>> cnf;

        /* Allocate body variables b_r and define b_r <-> (AND pos AND ~neg) */
        std::vector<std::vector<uint32_t>> supporters(static_cast<size_t>(num_atoms) + 1u);
        for (uint32_t r_idx = 0; r_idx < num_rules; r_idx++) {
            const AspRule& r = rules[r_idx];
            int32_t bv = static_cast<int32_t>(++total_vars);
            body_var[r_idx] = bv;

            std::vector<int32_t> rev_clause;
            rev_clause.reserve(1u + r.pos.size() + r.neg.size());
            rev_clause.push_back(bv);
            for (uint32_t p : r.pos) {
                cnf.push_back({-bv, static_cast<int32_t>(p)});
                rev_clause.push_back(-static_cast<int32_t>(p));
            }
            for (uint32_t q : r.neg) {
                cnf.push_back({-bv, -static_cast<int32_t>(q)});
                rev_clause.push_back(static_cast<int32_t>(q));
            }
            cnf.push_back(std::move(rev_clause));

            if (r.kind == RuleKind::Normal) {
                uint32_t h = r.heads[0];
                cnf.push_back({-bv, static_cast<int32_t>(h)});
                supporters[h].push_back(r_idx);
            } else if (r.kind == RuleKind::Constraint) {
                cnf.push_back({-bv});
            } else if (r.kind == RuleKind::Choice) {
                std::vector<int32_t> hlits;
                hlits.reserve(r.heads.size());
                for (uint32_t h : r.heads) {
                    hlits.push_back(static_cast<int32_t>(h));
                    supporters[h].push_back(r_idx);
                }
                encode_conditional_cardinality(cnf, total_vars, bv, hlits,
                                               r.min_choose, r.max_choose);
            }
        }

        /* Clark completion supportedness: a -> OR_{r in supporters(a)} b_r */
        for (uint32_t a = 1; a <= num_atoms; a++) {
            if (supporters[a].empty()) {
                cnf.push_back({-static_cast<int32_t>(a)});
            } else {
                std::vector<int32_t> cl;
                cl.reserve(1u + supporters[a].size());
                cl.push_back(-static_cast<int32_t>(a));
                for (uint32_t r_idx : supporters[a]) {
                    cl.push_back(body_var[r_idx]);
                }
                cnf.push_back(std::move(cl));
            }
        }

        const uint32_t max_rounds = cfg.max_cegar_rounds > 0 ? cfg.max_cegar_rounds : 10000u;
        for (uint32_t round = 0; round < max_rounds; round++) {
            eshkol_cdcl_config_t scfg = eshkol_cdcl_default_config();
            scfg.emit_drat = false;
            if (cfg.max_sat_conflicts > 0) {
                if (stats.total_conflicts >= cfg.max_sat_conflicts) {
                    if (out_stats) *out_stats = stats;
                    return ESHKOL_ASP_BUDGET_EXCEEDED;
                }
                scfg.max_conflicts = cfg.max_sat_conflicts - stats.total_conflicts;
            }

            eshkol_cdcl_solver_t* sat = eshkol_cdcl_create(total_vars, &scfg);
            if (!sat) {
                if (out_stats) *out_stats = stats;
                return ESHKOL_ASP_INVALID_INPUT;
            }
            for (const auto& cl : cnf) {
                eshkol_cdcl_add_clause(sat, cl.data(), static_cast<uint32_t>(cl.size()));
            }
            eshkol_sat_status_t sst = eshkol_cdcl_solve(sat);
            stats.sat_calls++;
            eshkol_cdcl_stats_t sstats{};
            eshkol_cdcl_get_stats(sat, &sstats);
            stats.total_conflicts += sstats.conflicts;

            if (sst == ESHKOL_SAT_BUDGET_EXCEEDED) {
                eshkol_cdcl_destroy(sat);
                if (out_stats) *out_stats = stats;
                return ESHKOL_ASP_BUDGET_EXCEEDED;
            }
            if (sst == ESHKOL_SAT_UNSATISFIABLE) {
                eshkol_cdcl_destroy(sat);
                break;
            }

            /* Extract candidate atom assignment M */
            std::vector<uint8_t> in_m(static_cast<size_t>(num_atoms) + 1u, 0u);
            for (uint32_t a = 1; a <= num_atoms; a++) {
                in_m[a] = (eshkol_cdcl_model_value(sat, a) > 0) ? 1u : 0u;
            }
            eshkol_cdcl_destroy(sat);

            /* Check for unfounded set U = M \ lfp(P^M) */
            std::vector<uint8_t> founded;
            compute_reduct_least_model(in_m, founded);

            std::vector<uint8_t> in_u(static_cast<size_t>(num_atoms) + 1u, 0u);
            std::vector<uint32_t> unfounded;
            for (uint32_t a = 1; a <= num_atoms; a++) {
                if (in_m[a] && !founded[a]) {
                    in_u[a] = 1u;
                    unfounded.push_back(a);
                }
            }

            if (unfounded.empty()) {
                /* Exact stable model found! */
                std::vector<uint32_t> true_atoms;
                for (uint32_t a = 1; a <= num_atoms; a++) {
                    if (in_m[a]) true_atoms.push_back(a);
                }
                model_storage.push_back(std::move(true_atoms));
                stats.num_models = static_cast<uint32_t>(model_storage.size());

                if (cfg.max_models > 0 && stats.num_models >= cfg.max_models) {
                    break;
                }

                /* Block exact model M to enumerate remaining stable models */
                std::vector<int32_t> block;
                block.reserve(num_atoms);
                for (uint32_t a = 1; a <= num_atoms; a++) {
                    block.push_back(in_m[a] ? -static_cast<int32_t>(a) : static_cast<int32_t>(a));
                }
                cnf.push_back(std::move(block));
            } else {
                /* Emit exact ASSAT loop formula for unfounded set U:
                 * Collect all external supporting rules R^-(U) with head in U and pos cap U == empty. */
                std::vector<int32_t> ext_bodies;
                for (uint32_t r_idx = 0; r_idx < num_rules; r_idx++) {
                    const AspRule& r = rules[r_idx];
                    if (r.kind == RuleKind::Constraint) continue;
                    bool head_in_u = false;
                    for (uint32_t h : r.heads) {
                        if (in_u[h]) { head_in_u = true; break; }
                    }
                    if (!head_in_u) continue;
                    bool pos_hits_u = false;
                    for (uint32_t p : r.pos) {
                        if (in_u[p]) { pos_hits_u = true; break; }
                    }
                    if (!pos_hits_u) {
                        ext_bodies.push_back(body_var[r_idx]);
                    }
                }
                for (uint32_t a : unfounded) {
                    std::vector<int32_t> lf;
                    lf.reserve(1u + ext_bodies.size());
                    lf.push_back(-static_cast<int32_t>(a));
                    for (int32_t eb : ext_bodies) lf.push_back(eb);
                    cnf.push_back(std::move(lf));
                    stats.loop_formulas_added++;
                }
            }
        }

        /* Materialize stable view pointers after model_storage stops growing */
        model_views.resize(model_storage.size());
        for (size_t i = 0; i < model_storage.size(); i++) {
            model_views[i].num_true_atoms = static_cast<uint32_t>(model_storage[i].size());
            model_views[i].true_atoms = model_storage[i].empty() ? nullptr : model_storage[i].data();
        }

        if (out_stats) *out_stats = stats;
        return model_views.empty() ? ESHKOL_ASP_UNSAT : ESHKOL_ASP_SAT;
    }
};

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_asp_config_t eshkol_asp_default_config(void) {
    eshkol_asp_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.max_models = 0;
    cfg.max_sat_conflicts = 0;
    cfg.max_cegar_rounds = 10000u;
    return cfg;
}

extern "C" eshkol_asp_solver_t* eshkol_asp_create(uint32_t num_atoms,
                                                  const eshkol_asp_config_t* cfg) {
    if (num_atoms == 0) return nullptr;
    eshkol_asp_config_t eff = cfg ? *cfg : eshkol_asp_default_config();
    return new (std::nothrow) eshkol_asp_solver(num_atoms, eff);
}

extern "C" void eshkol_asp_destroy(eshkol_asp_solver_t* solver) {
    delete solver;
}

extern "C" bool eshkol_asp_add_normal_rule(eshkol_asp_solver_t* solver,
                                           uint32_t head,
                                           const uint32_t* pos_body, uint32_t num_pos,
                                           const uint32_t* neg_body, uint32_t num_neg) {
    if (!solver || !solver->valid_atom(head)) return false;
    if ((num_pos > 0 && !pos_body) || (num_neg > 0 && !neg_body)) return false;
    for (uint32_t i = 0; i < num_pos; i++) {
        if (!solver->valid_atom(pos_body[i])) return false;
    }
    for (uint32_t i = 0; i < num_neg; i++) {
        if (!solver->valid_atom(neg_body[i])) return false;
    }
    AspRule r;
    r.kind = RuleKind::Normal;
    r.heads = {head};
    if (num_pos > 0) r.pos.assign(pos_body, pos_body + num_pos);
    if (num_neg > 0) r.neg.assign(neg_body, neg_body + num_neg);
    solver->rules.push_back(std::move(r));
    return true;
}

extern "C" bool eshkol_asp_add_constraint(eshkol_asp_solver_t* solver,
                                          const uint32_t* pos_body, uint32_t num_pos,
                                          const uint32_t* neg_body, uint32_t num_neg) {
    if (!solver) return false;
    if ((num_pos > 0 && !pos_body) || (num_neg > 0 && !neg_body)) return false;
    for (uint32_t i = 0; i < num_pos; i++) {
        if (!solver->valid_atom(pos_body[i])) return false;
    }
    for (uint32_t i = 0; i < num_neg; i++) {
        if (!solver->valid_atom(neg_body[i])) return false;
    }
    AspRule r;
    r.kind = RuleKind::Constraint;
    if (num_pos > 0) r.pos.assign(pos_body, pos_body + num_pos);
    if (num_neg > 0) r.neg.assign(neg_body, neg_body + num_neg);
    solver->rules.push_back(std::move(r));
    return true;
}

extern "C" bool eshkol_asp_add_choice_rule(eshkol_asp_solver_t* solver,
                                           const uint32_t* heads, uint32_t num_heads,
                                           uint32_t min_choose, uint32_t max_choose,
                                           const uint32_t* pos_body, uint32_t num_pos,
                                           const uint32_t* neg_body, uint32_t num_neg) {
    if (!solver || (num_heads > 0 && !heads)) return false;
    if ((num_pos > 0 && !pos_body) || (num_neg > 0 && !neg_body)) return false;
    for (uint32_t i = 0; i < num_heads; i++) {
        if (!solver->valid_atom(heads[i])) return false;
    }
    for (uint32_t i = 0; i < num_pos; i++) {
        if (!solver->valid_atom(pos_body[i])) return false;
    }
    for (uint32_t i = 0; i < num_neg; i++) {
        if (!solver->valid_atom(neg_body[i])) return false;
    }
    uint32_t eff_max = (max_choose == UINT32_MAX) ? num_heads : max_choose;
    AspRule r;
    r.kind = RuleKind::Choice;
    if (num_heads > 0) r.heads.assign(heads, heads + num_heads);
    r.min_choose = min_choose;
    r.max_choose = eff_max;
    if (num_pos > 0) r.pos.assign(pos_body, pos_body + num_pos);
    if (num_neg > 0) r.neg.assign(neg_body, neg_body + num_neg);
    solver->rules.push_back(std::move(r));
    return true;
}

extern "C" eshkol_asp_status_t eshkol_asp_solve(eshkol_asp_solver_t* solver,
                                                eshkol_asp_stats_t* out_stats) {
    if (!solver) return ESHKOL_ASP_INVALID_INPUT;
    return solver->solve(out_stats);
}

extern "C" uint32_t eshkol_asp_num_models(const eshkol_asp_solver_t* solver) {
    return solver ? static_cast<uint32_t>(solver->model_views.size()) : 0u;
}

extern "C" const eshkol_asp_model_t* eshkol_asp_get_model(const eshkol_asp_solver_t* solver,
                                                          uint32_t model_idx) {
    if (!solver || model_idx >= solver->model_views.size()) return nullptr;
    return &solver->model_views[model_idx];
}

extern "C" bool eshkol_asp_verify_stable_model(const eshkol_asp_solver_t* solver,
                                               const uint32_t* true_atoms,
                                               uint32_t num_true_atoms) {
    if (!solver || (num_true_atoms > 0 && !true_atoms)) return false;
    std::vector<uint8_t> in_m(static_cast<size_t>(solver->num_atoms) + 1u, 0u);
    for (uint32_t i = 0; i < num_true_atoms; i++) {
        if (!solver->valid_atom(true_atoms[i])) return false;
        in_m[true_atoms[i]] = 1u;
    }
    return solver->verify_model_vec(in_m);
}
