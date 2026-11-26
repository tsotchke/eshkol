/*
 * CE-1 Native Compiled Logic Core Verification Suite
 *
 * Verifies:
 *   1. CDCL SAT solver on SAT formulas with independent model verification
 *   2. CDCL UNSAT + DRAT proof emission + zero-shared-state forward DRAT
 *      verifier on PHP(4,3), PHP(5,4), LBD deletion stress, and Tseitin XOR
 *   3. Provably-failable DRAT negative controls (dropped step, mutated literal,
 *      truncated empty clause, invalid claim) plus explicit RAT verification
 *   4. CDCL deterministic AIKR conflict and propagation budget enforcement
 *   5. SLG-WAM tabled resolution termination on left-recursive cyclic transitive
 *      closure, stratified negation, positive-cycle unfounded sets, and exact
 *      3-valued Well-Founded Semantics on Win-Move games (TRUE/FALSE/UNKNOWN)
 *   6. SLG deterministic AIKR fuel exhaustion with clean arena preservation
 *   7. NARS NAL-1..9 truth-value calculus invariants and domain negative controls
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/cdcl_sat.h>
#include <eshkol/core/logic.h>
#include <eshkol/core/nars_truth.h>
#include <eshkol/core/slg_table.h>
#include <eshkol/eshkol.h>

extern "C" {
arena_t* arena_create(size_t default_block_size);
void arena_destroy(arena_t* arena);
const char* eshkol_intern_predicate(const char* name);
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

static int g_checks = 0;

static void check(bool cond, const char* msg) {
    g_checks++;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static eshkol_tagged_value_t make_int_tv(int64_t v) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_INT64;
    tv.data.int_val = v;
    return tv;
}

static eshkol_tagged_value_t make_var_tv(uint64_t id) {
    eshkol_tagged_value_t tv;
    std::memset(&tv, 0, sizeof(tv));
    tv.type = ESHKOL_VALUE_LOGIC_VAR;
    tv.data.int_val = static_cast<int64_t>(id);
    return tv;
}

struct CnfFormula {
    uint32_t num_vars = 0;
    std::vector<std::vector<int32_t>> clauses;

    void add(std::initializer_list<int32_t> lits) {
        clauses.emplace_back(lits);
    }

    void add_vec(const std::vector<int32_t>& lits) {
        clauses.push_back(lits);
    }

    std::vector<eshkol_cnf_clause_t> views() const {
        std::vector<eshkol_cnf_clause_t> out;
        out.reserve(clauses.size());
        for (const auto& c : clauses) {
            out.push_back({c.empty() ? nullptr : c.data(), static_cast<uint32_t>(c.size())});
        }
        return out;
    }
};

/* Pigeonhole principle PHP(pigeons, holes): UNSAT whenever pigeons > holes. */
static CnfFormula build_php(uint32_t pigeons, uint32_t holes) {
    CnfFormula f;
    f.num_vars = pigeons * holes;
    auto var = [&](uint32_t p, uint32_t h) -> int32_t {
        return static_cast<int32_t>(p * holes + h + 1u);
    };
    /* Every pigeon sits in at least one hole */
    for (uint32_t p = 0; p < pigeons; p++) {
        std::vector<int32_t> cl;
        for (uint32_t h = 0; h < holes; h++) cl.push_back(var(p, h));
        f.add_vec(cl);
    }
    /* No hole holds two pigeons */
    for (uint32_t h = 0; h < holes; h++) {
        for (uint32_t p1 = 0; p1 < pigeons; p1++) {
            for (uint32_t p2 = p1 + 1; p2 < pigeons; p2++) {
                f.add({-var(p1, h), -var(p2, h)});
            }
        }
    }
    return f;
}

/* Encode XOR(vars) == parity into CNF (for small degree <= 4). */
static void add_xor_cnf(CnfFormula& f, const std::vector<int32_t>& vars, int parity) {
    const size_t k = vars.size();
    const uint32_t masks = 1u << k;
    for (uint32_t m = 0; m < masks; m++) {
        int ones = __builtin_popcount(m);
        if ((ones & 1) != (parity & 1)) {
            /* Exclude assignment m where bit i == 1 means vars[i] is true */
            std::vector<int32_t> cl;
            cl.reserve(k);
            for (size_t i = 0; i < k; i++) {
                bool bit = ((m >> i) & 1u) != 0;
                cl.push_back(bit ? -vars[i] : vars[i]);
            }
            f.add_vec(cl);
        }
    }
}

/* Tseitin XOR formula over a 3-regular prism graph (6 vertices, 9 edges, odd charge). */
static CnfFormula build_tseitin_unsat() {
    /* 6 vertices (0..5), 9 edges (vars 1..9):
     * Top triangle: e1=(0,1), e2=(1,2), e3=(2,0)
     * Bottom triangle: e4=(3,4), e5=(4,5), e6=(5,3)
     * Spokes: e7=(0,3), e8=(1,4), e9=(2,5)
     * Sum of charges = 1 (odd) while every edge appears twice => UNSAT! */
    CnfFormula f;
    f.num_vars = 9;
    add_xor_cnf(f, {1, 3, 7}, 1); /* v0 charge 1 */
    add_xor_cnf(f, {1, 2, 8}, 0); /* v1 charge 0 */
    add_xor_cnf(f, {2, 3, 9}, 0); /* v2 charge 0 */
    add_xor_cnf(f, {4, 6, 7}, 0); /* v3 charge 0 */
    add_xor_cnf(f, {4, 5, 8}, 0); /* v4 charge 0 */
    add_xor_cnf(f, {5, 6, 9}, 0); /* v5 charge 0 */
    return f;
}

/* 3-coloring of a 10-vertex Petersen graph (3-colorable -> SAT). */
static CnfFormula build_petersen_3color() {
    CnfFormula f;
    const uint32_t N = 10;
    const uint32_t K = 3;
    f.num_vars = N * K;
    auto v = [&](uint32_t node, uint32_t color) -> int32_t {
        return static_cast<int32_t>(node * K + color + 1u);
    };
    for (uint32_t i = 0; i < N; i++) {
        f.add({v(i, 0), v(i, 1), v(i, 2)});
        f.add({-v(i, 0), -v(i, 1)});
        f.add({-v(i, 0), -v(i, 2)});
        f.add({-v(i, 1), -v(i, 2)});
    }
    const uint32_t edges[15][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0},
        {5, 7}, {7, 9}, {9, 6}, {6, 8}, {8, 5},
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    for (auto& e : edges) {
        for (uint32_t c = 0; c < K; c++) {
            f.add({-v(e[0], c), -v(e[1], c)});
        }
    }
    return f;
}

static void test_cdcl_sat_and_model_verifier() {
    CnfFormula petersen = build_petersen_3color();
    eshkol_cdcl_config_t cfg = eshkol_cdcl_default_config();
    eshkol_cdcl_solver_t* s = eshkol_cdcl_create(petersen.num_vars, &cfg);
    check(s != nullptr, "CDCL solver created for Petersen 3-coloring");

    /* Invalid literal negative controls */
    int32_t bad_zero[1] = {0};
    int32_t bad_oob[1] = {static_cast<int32_t>(petersen.num_vars + 5u)};
    check(!eshkol_cdcl_add_clause(s, bad_zero, 1), "CDCL rejects literal 0");
    check(!eshkol_cdcl_add_clause(s, bad_oob, 1), "CDCL rejects out-of-range variable literal");

    for (const auto& c : petersen.clauses) {
        check(eshkol_cdcl_add_clause(s, c.data(), static_cast<uint32_t>(c.size())),
              "CDCL clause added");
    }

    eshkol_sat_status_t st = eshkol_cdcl_solve(s);
    check(st == ESHKOL_SAT_SATISFIABLE, "Petersen graph is 3-colorable (SAT)");

    std::vector<int8_t> model(petersen.num_vars + 1u, 0);
    check(eshkol_cdcl_copy_model(s, model.data(), static_cast<uint32_t>(model.size())),
          "CDCL model copied");

    auto views = petersen.views();
    check(eshkol_cdcl_verify_model(petersen.num_vars, views.data(),
                                   static_cast<uint32_t>(views.size()),
                                   model.data(), static_cast<uint32_t>(model.size())),
          "independent model verifier accepts Petersen 3-coloring model");

    /* Negative control: corrupt model by forcing all colors of vertex 0 to false */
    std::vector<int8_t> corrupted = model;
    corrupted[1] = -1;
    corrupted[2] = -1;
    corrupted[3] = -1;
    check(!eshkol_cdcl_verify_model(petersen.num_vars, views.data(),
                                    static_cast<uint32_t>(views.size()),
                                    corrupted.data(), static_cast<uint32_t>(corrupted.size())),
          "negative control: model verifier rejects corrupted assignment");

    eshkol_cdcl_destroy(s);
    std::printf("PASS: CDCL SAT + independent model verifier (Petersen 3-coloring)\n");
}

static void run_unsat_drat_oracle(const char* label, const CnfFormula& formula,
                                  const eshkol_cdcl_config_t& cfg,
                                  bool expect_deletions,
                                  bool run_negative_controls) {
    eshkol_cdcl_solver_t* s = eshkol_cdcl_create(formula.num_vars, &cfg);
    check(s != nullptr, "CDCL solver created for UNSAT oracle");
    for (const auto& c : formula.clauses) {
        check(eshkol_cdcl_add_clause(s, c.data(), static_cast<uint32_t>(c.size())),
              "UNSAT clause added");
    }

    eshkol_sat_status_t st = eshkol_cdcl_solve(s);
    check(st == ESHKOL_SAT_UNSATISFIABLE, label);

    eshkol_cdcl_stats_t stats{};
    eshkol_cdcl_get_stats(s, &stats);
    check(stats.conflicts > 0, "UNSAT oracle performed conflict learning");
    if (expect_deletions) {
        check(stats.deleted_clauses > 0, "aggressive LBD reduction emitted DRAT deletions");
    }

    const eshkol_drat_proof_t* proof = eshkol_cdcl_get_proof(s);
    check(proof != nullptr && proof->num_steps > 0, "DRAT proof populated");

    auto views = formula.views();
    bool ok = eshkol_cdcl_verify_drat(formula.num_vars, views.data(),
                                      static_cast<uint32_t>(views.size()), proof);
    check(ok, "independent zero-shared-state DRAT verifier accepts solver proof");

    if (run_negative_controls) {
        /* Copy proof steps so we can construct mutated/dropped negative controls */
        std::vector<std::vector<int32_t>> step_lits(proof->num_steps);
        std::vector<eshkol_drat_step_t> steps(proof->num_steps);
        for (uint32_t i = 0; i < proof->num_steps; i++) {
            step_lits[i].assign(proof->steps[i].lits,
                                proof->steps[i].lits + proof->steps[i].num_lits);
            steps[i].op = proof->steps[i].op;
            steps[i].num_lits = static_cast<uint32_t>(step_lits[i].size());
            steps[i].lits = step_lits[i].empty() ? nullptr : step_lits[i].data();
        }

        /* Negative control 1: truncate final empty clause */
        eshkol_drat_proof_t truncated = {proof->num_steps - 1u, proof->num_steps - 1u, steps.data()};
        check(!eshkol_cdcl_verify_drat(formula.num_vars, views.data(),
                                       static_cast<uint32_t>(views.size()), &truncated),
              "DRAT negative control: truncated proof without empty clause rejected");

        /* Negative control 2: drop all non-empty ADD steps leaving only empty clause */
        eshkol_drat_step_t only_empty = {ESHKOL_DRAT_ADD, 0, nullptr};
        eshkol_drat_proof_t empty_only = {1u, 1u, &only_empty};
        check(!eshkol_cdcl_verify_drat(formula.num_vars, views.data(),
                                       static_cast<uint32_t>(views.size()), &empty_only),
              "DRAT negative control: bare empty clause without derivation rejected");

        /* Negative control 3: mutate a learned ADD step (flip polarity of first literal) */
        bool tested_mutation = false;
        for (uint32_t i = 0; i < proof->num_steps; i++) {
            if (steps[i].op == ESHKOL_DRAT_ADD && steps[i].num_lits >= 2) {
                int32_t orig = step_lits[i][0];
                step_lits[i][0] = -orig;
                eshkol_drat_proof_t mutated = {proof->num_steps, proof->num_steps, steps.data()};
                bool mut_ok = eshkol_cdcl_verify_drat(
                    formula.num_vars, views.data(),
                    static_cast<uint32_t>(views.size()), &mutated);
                step_lits[i][0] = orig;
                if (!mut_ok) {
                    tested_mutation = true;
                    break;
                }
            }
        }
        check(tested_mutation,
              "DRAT negative control: mutated literal in learned step rejected");

        /* Negative control 4: drop a load-bearing learned ADD step */
        bool tested_drop = false;
        for (uint32_t drop_idx = proof->num_steps - 1u; drop_idx-- > 0; ) {
            if (steps[drop_idx].op != ESHKOL_DRAT_ADD || steps[drop_idx].num_lits == 0) continue;
            std::vector<eshkol_drat_step_t> dropped;
            dropped.reserve(proof->num_steps - 1u);
            for (uint32_t k = 0; k < proof->num_steps; k++) {
                if (k != drop_idx) dropped.push_back(steps[k]);
            }
            eshkol_drat_proof_t dropped_proof = {
                static_cast<uint32_t>(dropped.size()),
                static_cast<uint32_t>(dropped.size()),
                dropped.data()
            };
            if (!eshkol_cdcl_verify_drat(formula.num_vars, views.data(),
                                         static_cast<uint32_t>(views.size()), &dropped_proof)) {
                tested_drop = true;
                break;
            }
        }
        check(tested_drop,
              "DRAT negative control: dropping a required learned proof step rejected");
    }

    eshkol_cdcl_destroy(s);
    std::printf("PASS: CDCL UNSAT + DRAT (%s: conflicts=%llu, learnt=%llu, deleted=%llu, drat_steps=%u)\n",
                label,
                static_cast<unsigned long long>(stats.conflicts),
                static_cast<unsigned long long>(stats.learned_clauses),
                static_cast<unsigned long long>(stats.deleted_clauses),
                stats.drat_steps);
}

static void test_cdcl_unsat_drat_and_rat() {
    eshkol_cdcl_config_t cfg = eshkol_cdcl_default_config();
    run_unsat_drat_oracle("PHP(4,3)", build_php(4, 3), cfg, false, true);
    run_unsat_drat_oracle("PHP(5,4)", build_php(5, 4), cfg, false, true);
    run_unsat_drat_oracle("Tseitin-XOR-Prism", build_tseitin_unsat(), cfg, false, true);

    /* Force frequent LBD clause deletions to exercise DRAT DELETE emission + verification */
    eshkol_cdcl_config_t del_cfg = eshkol_cdcl_default_config();
    del_cfg.max_learnts_init = 6;
    del_cfg.learntsize_inc = 1.05;
    del_cfg.restart_base = 16;
    run_unsat_drat_oracle("PHP(5,4)-LBD-Deletion", build_php(5, 4), del_cfg, true, false);

    /* Explicit RAT verification test (pure literal addition is RAT, not RUP):
     * Formula: { (1, 2), (-1, 2), (1, -2), (-1, -2) } with auxiliary fresh var 3.
     * Step 1: ADD (3) -> NOT RUP (assuming -3 does not conflict), IS RAT on pivot 3
     *         (no clause contains -3)!
     * Step 2: ADD (-1) -> RUP (assuming 1 conflicts with (-1,2),(-1,-2))
     * Step 3: ADD {} -> RUP. */
    CnfFormula f;
    f.num_vars = 3;
    f.add({1, 2});
    f.add({-1, 2});
    f.add({1, -2});
    f.add({-1, -2});
    auto views = f.views();

    int32_t s1[1] = {3};
    int32_t s2[1] = {-1};
    eshkol_drat_step_t rat_steps[3] = {
        {ESHKOL_DRAT_ADD, 1, s1},
        {ESHKOL_DRAT_ADD, 1, s2},
        {ESHKOL_DRAT_ADD, 0, nullptr}
    };
    eshkol_drat_proof_t rat_proof = {3, 3, rat_steps};
    check(eshkol_cdcl_verify_drat(f.num_vars, views.data(),
                                  static_cast<uint32_t>(views.size()), &rat_proof),
          "independent DRAT verifier accepts valid non-RUP RAT step");

    /* Invalid RAT negative control on SAT formula { (1, 2) }: adding (-1) is neither RUP nor RAT */
    CnfFormula sat_f;
    sat_f.num_vars = 2;
    sat_f.add({1, 2});
    auto sat_views = sat_f.views();
    int32_t bad_step_lits[1] = {-1};
    eshkol_drat_step_t bad_steps[2] = {
        {ESHKOL_DRAT_ADD, 1, bad_step_lits},
        {ESHKOL_DRAT_ADD, 0, nullptr}
    };
    eshkol_drat_proof_t bad_proof = {2, 2, bad_steps};
    check(!eshkol_cdcl_verify_drat(sat_f.num_vars, sat_views.data(),
                                   static_cast<uint32_t>(sat_views.size()), &bad_proof),
          "DRAT negative control: invalid non-RUP/non-RAT step on SAT formula rejected");

    std::printf("PASS: explicit RAT acceptance and negative controls\n");
}

static void test_cdcl_budget_enforcement() {
    CnfFormula php65 = build_php(6, 5);
    eshkol_cdcl_config_t cfg = eshkol_cdcl_default_config();
    cfg.max_conflicts = 5;
    eshkol_cdcl_solver_t* s = eshkol_cdcl_create(php65.num_vars, &cfg);
    for (const auto& c : php65.clauses) {
        eshkol_cdcl_add_clause(s, c.data(), static_cast<uint32_t>(c.size()));
    }
    eshkol_sat_status_t st = eshkol_cdcl_solve(s);
    eshkol_cdcl_stats_t stats{};
    eshkol_cdcl_get_stats(s, &stats);
    check(st == ESHKOL_SAT_BUDGET_EXCEEDED, "CDCL conflict budget returns BUDGET_EXCEEDED");
    check(stats.conflicts == 5, "CDCL halted exactly at max_conflicts=5");
    eshkol_cdcl_destroy(s);

    eshkol_cdcl_config_t pcfg = eshkol_cdcl_default_config();
    pcfg.max_propagations = 12;
    eshkol_cdcl_solver_t* sp = eshkol_cdcl_create(php65.num_vars, &pcfg);
    for (const auto& c : php65.clauses) {
        eshkol_cdcl_add_clause(sp, c.data(), static_cast<uint32_t>(c.size()));
    }
    st = eshkol_cdcl_solve(sp);
    eshkol_cdcl_get_stats(sp, &stats);
    check(st == ESHKOL_SAT_BUDGET_EXCEEDED, "CDCL propagation budget returns BUDGET_EXCEEDED");
    check(stats.propagations <= 12, "CDCL respected max_propagations=12");
    eshkol_cdcl_destroy(sp);

    std::printf("PASS: CDCL deterministic AIKR conflict/propagation budgets\n");
}

static void test_slg_left_recursive_cyclic_closure() {
    arena_t* arena = arena_create(4 * 1024 * 1024);
    check(arena != nullptr, "arena created for SLG cyclic test");
    eshkol_logic_registry_reset();

    uint64_t p_edge = reinterpret_cast<uint64_t>(eshkol_intern_predicate("edge"));
    uint64_t p_path = reinterpret_cast<uint64_t>(eshkol_intern_predicate("path"));
    uint64_t vx = eshkol_make_logic_var("?x");
    uint64_t vy = eshkol_make_logic_var("?y");
    uint64_t vz = eshkol_make_logic_var("?z");

    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    const int64_t edges[8][2] = {
        {1, 2}, {2, 3}, {3, 1}, /* 3-cycle */
        {3, 4}, {4, 5}, {5, 4}, /* tail 2-cycle */
        {10, 11}, {11, 10}      /* disconnected component */
    };
    for (auto& e : edges) {
        eshkol_tagged_value_t args[2] = {make_int_tv(e[0]), make_int_tv(e[1])};
        eshkol_kb_assert(arena, kb, eshkol_make_fact(arena, p_edge, args, 2));
    }

    /* Rule 1: path(?x, ?y) :- edge(?x, ?y) */
    eshkol_tagged_value_t xy[2] = {make_var_tv(vx), make_var_tv(vy)};
    eshkol_tagged_value_t xz[2] = {make_var_tv(vx), make_var_tv(vz)};
    eshkol_tagged_value_t zy[2] = {make_var_tv(vz), make_var_tv(vy)};
    eshkol_fact_t* r1_head = eshkol_make_fact(arena, p_path, xy, 2);
    eshkol_fact_t* r1_b0 = eshkol_make_fact(arena, p_edge, xy, 2);
    eshkol_fact_t* r1_body[1] = {r1_b0};
    eshkol_rule_t* r1 = eshkol_make_rule(
        arena, r1_head, r1_body, 1, nullptr, 1.0, 1.0,
        ESHKOL_RULE_FLAG_TABLED, "path_base");

    /* Rule 2 (LEFT-RECURSIVE!): path(?x, ?y) :- path(?x, ?z), edge(?z, ?y) */
    eshkol_fact_t* r2_head = eshkol_make_fact(arena, p_path, xy, 2);
    eshkol_fact_t* r2_b0 = eshkol_make_fact(arena, p_path, xz, 2);
    eshkol_fact_t* r2_b1 = eshkol_make_fact(arena, p_edge, zy, 2);
    eshkol_fact_t* r2_body[2] = {r2_b0, r2_b1};
    eshkol_rule_t* r2 = eshkol_make_rule(
        arena, r2_head, r2_body, 2, nullptr, 1.0, 1.0,
        ESHKOL_RULE_FLAG_TABLED, "path_left_rec");

    eshkol_slg_config_t cfg = eshkol_slg_default_config();
    cfg.max_steps = 50000;
    eshkol_slg_engine_t* eng = eshkol_slg_create(arena, kb, &cfg);
    check(eng != nullptr, "SLG engine created");
    check(eshkol_slg_add_rule(eng, r1), "SLG base rule added");
    check(eshkol_slg_add_rule(eng, r2), "SLG left-recursive rule added");

    eshkol_tagged_value_t qargs[2] = {make_int_tv(1), make_var_tv(vy)};
    eshkol_fact_t* query = eshkol_make_fact(arena, p_path, qargs, 2);
    eshkol_slg_result_t res{};
    eshkol_slg_status_t st = eshkol_slg_query(eng, query, &res);
    check(st == ESHKOL_SLG_OK, "SLG terminates on left-recursive cyclic graph");
    check(res.num_answers == 5, "path(1, ?y) yields exactly 5 reachable nodes");
    check(res.consumer_resumptions > 0, "SLG exercised consumer suspension/resumption");

    std::set<int64_t> reached;
    for (uint32_t i = 0; i < res.num_answers; i++) {
        check(res.answers[i].truth == ESHKOL_SLG_TRUE, "positive path answer is TRUE");
        const eshkol_tagged_value_t* bound = eshkol_subst_lookup(res.answers[i].subst, vy);
        check(bound != nullptr && bound->type == ESHKOL_VALUE_INT64, "query var ?y bound");
        reached.insert(bound->data.int_val);
    }
    std::set<int64_t> expected = {1, 2, 3, 4, 5};
    check(reached == expected, "path(1, ?y) reaches exactly {1,2,3,4,5} excluding {10,11}");

    /* Ground negative check for disconnected node */
    eshkol_tagged_value_t g10[2] = {make_int_tv(1), make_int_tv(10)};
    eshkol_fact_t* path_1_10 = eshkol_make_fact(arena, p_path, g10, 2);
    eshkol_slg_truth_t tr = ESHKOL_SLG_UNKNOWN;
    check(eshkol_slg_query_ground(eng, path_1_10, &tr, nullptr) == ESHKOL_SLG_OK &&
          tr == ESHKOL_SLG_FALSE,
          "path(1, 10) is FALSE");

    /* AIKR fuel exhaustion check on tight budget + recovery on fresh engine */
    eshkol_slg_config_t tight_cfg = eshkol_slg_default_config();
    tight_cfg.max_steps = 5;
    eshkol_slg_engine_t* tight_eng = eshkol_slg_create(arena, kb, &tight_cfg);
    eshkol_slg_add_rule(tight_eng, r1);
    eshkol_slg_add_rule(tight_eng, r2);
    eshkol_slg_result_t tight_res{};
    check(eshkol_slg_query(tight_eng, query, &tight_res) == ESHKOL_SLG_FUEL_EXHAUSTED,
          "SLG returns ESHKOL_SLG_FUEL_EXHAUSTED on tight max_steps=5");
    eshkol_slg_destroy(tight_eng);

    /* Confirm arena and main engine stay completely healthy after fuel exhaustion */
    eshkol_slg_result_t post_res{};
    check(eshkol_slg_query(eng, query, &post_res) == ESHKOL_SLG_OK && post_res.num_answers == 5,
          "SLG arena intact after prior fuel exhaustion");

    eshkol_slg_destroy(eng);
    arena_destroy(arena);
    std::printf("PASS: SLG left-recursive cyclic transitive closure + AIKR fuel (resumptions=%u)\n",
                res.consumer_resumptions);
}

static void test_slg_wfs_win_move_and_unfounded_sets() {
    arena_t* arena = arena_create(4 * 1024 * 1024);
    check(arena != nullptr, "arena created for SLG WFS test");
    eshkol_logic_registry_reset();

    uint64_t p_move = reinterpret_cast<uint64_t>(eshkol_intern_predicate("move"));
    uint64_t p_win  = reinterpret_cast<uint64_t>(eshkol_intern_predicate("win"));
    uint64_t vx = eshkol_make_logic_var("?wx");
    uint64_t vy = eshkol_make_logic_var("?wy");

    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    /* Win-Move game graph covering every 3-valued WFS class:
     *   1 -> 2 -> 3 (3 dead-end => win(3)=FALSE, win(2)=TRUE, win(1)=FALSE)
     *   10 -> 11 -> 12 -> 10 (odd 3-cycle => win(10)=win(11)=win(12)=UNKNOWN)
     *   20 <-> 21, plus 20 -> 3 (20 escapes to losing 3 => win(20)=TRUE, win(21)=FALSE)
     *   30 -> 2 (winning) and 30 -> 10 (draw) => win(30)=UNKNOWN
     *   40 <-> 41 (pure even 2-cycle => win(40)=win(41)=UNKNOWN under WFS)
     */
    const int64_t moves[12][2] = {
        {1, 2}, {2, 3},
        {10, 11}, {11, 12}, {12, 10},
        {20, 21}, {21, 20}, {20, 3},
        {30, 2}, {30, 10},
        {40, 41}, {41, 40}
    };
    for (auto& m : moves) {
        eshkol_tagged_value_t args[2] = {make_int_tv(m[0]), make_int_tv(m[1])};
        eshkol_kb_assert(arena, kb, eshkol_make_fact(arena, p_move, args, 2));
    }

    /* Rule: win(?wx) :- move(?wx, ?wy), not(win(?wy)) */
    eshkol_tagged_value_t x_arg[1] = {make_var_tv(vx)};
    eshkol_tagged_value_t y_arg[1] = {make_var_tv(vy)};
    eshkol_tagged_value_t xy[2]    = {make_var_tv(vx), make_var_tv(vy)};

    eshkol_fact_t* head = eshkol_make_fact(arena, p_win, x_arg, 1);
    eshkol_fact_t* b_move = eshkol_make_fact(arena, p_move, xy, 2);
    eshkol_fact_t* inner_win = eshkol_make_fact(arena, p_win, y_arg, 1);
    eshkol_fact_t* b_not_win = eshkol_slg_make_negated_goal(arena, inner_win);
    const eshkol_fact_t* inspected = nullptr;
    check(eshkol_slg_is_negated_goal(b_not_win, &inspected) && inspected == inner_win,
          "negated goal helper constructs inspectable not(win(?wy))");

    eshkol_fact_t* body[2] = {b_move, b_not_win};
    eshkol_rule_t* win_rule = eshkol_make_rule(
        arena, head, body, 2, nullptr, 1.0, 1.0,
        ESHKOL_RULE_FLAG_TABLED, "win_move_wfs");

    eshkol_slg_config_t cfg = eshkol_slg_default_config();
    eshkol_slg_engine_t* eng = eshkol_slg_create(arena, kb, &cfg);
    check(eshkol_slg_add_rule(eng, win_rule), "Win-Move rule registered");

    /* Open query win(?wx) */
    eshkol_fact_t* q_open = eshkol_make_fact(arena, p_win, x_arg, 1);
    eshkol_slg_result_t res{};
    check(eshkol_slg_query(eng, q_open, &res) == ESHKOL_SLG_OK, "WFS Win-Move query succeeded");
    check(res.wfs_iterations >= 2, "WFS ran alternating closure iterations");

    std::map<int64_t, eshkol_slg_truth_t> open_map;
    for (uint32_t i = 0; i < res.num_answers; i++) {
        const eshkol_tagged_value_t* b = eshkol_subst_lookup(res.answers[i].subst, vx);
        check(b != nullptr && b->type == ESHKOL_VALUE_INT64, "win(?wx) answer binds ?wx");
        open_map[b->data.int_val] = res.answers[i].truth;
    }

    const std::pair<int64_t, eshkol_slg_truth_t> expected_nodes[] = {
        {1,  ESHKOL_SLG_FALSE},
        {2,  ESHKOL_SLG_TRUE},
        {3,  ESHKOL_SLG_FALSE},
        {10, ESHKOL_SLG_UNKNOWN},
        {11, ESHKOL_SLG_UNKNOWN},
        {12, ESHKOL_SLG_UNKNOWN},
        {20, ESHKOL_SLG_TRUE},
        {21, ESHKOL_SLG_FALSE},
        {30, ESHKOL_SLG_UNKNOWN},
        {40, ESHKOL_SLG_UNKNOWN},
        {41, ESHKOL_SLG_UNKNOWN}
    };

    for (const auto& kv : expected_nodes) {
        eshkol_slg_truth_t open_t = open_map.count(kv.first) ? open_map[kv.first] : ESHKOL_SLG_FALSE;
        check(open_t == kv.second, "open query WFS truth matches exact Win-Move semantics");

        eshkol_tagged_value_t g_arg[1] = {make_int_tv(kv.first)};
        eshkol_fact_t* g_atom = eshkol_make_fact(arena, p_win, g_arg, 1);
        eshkol_slg_truth_t ground_t = ESHKOL_SLG_UNKNOWN;
        check(eshkol_slg_query_ground(eng, g_atom, &ground_t, nullptr) == ESHKOL_SLG_OK,
              "ground WFS query succeeded");
        check(ground_t == kv.second, "ground query WFS truth matches exact Win-Move semantics");
    }

    /* Positive-cycle unfounded-set elimination + stratified negation check */
    uint64_t p_loop = reinterpret_cast<uint64_t>(eshkol_intern_predicate("loop"));
    uint64_t p_base = reinterpret_cast<uint64_t>(eshkol_intern_predicate("base"));
    uint64_t p_safe = reinterpret_cast<uint64_t>(eshkol_intern_predicate("safe"));
    uint64_t p_node = reinterpret_cast<uint64_t>(eshkol_intern_predicate("node"));

    eshkol_tagged_value_t n1[1] = {make_int_tv(1)};
    eshkol_tagged_value_t n2[1] = {make_int_tv(2)};
    eshkol_kb_assert(arena, kb, eshkol_make_fact(arena, p_base, n1, 1));
    eshkol_kb_assert(arena, kb, eshkol_make_fact(arena, p_node, n1, 1));
    eshkol_kb_assert(arena, kb, eshkol_make_fact(arena, p_node, n2, 1));

    eshkol_fact_t* loop_x = eshkol_make_fact(arena, p_loop, x_arg, 1);
    eshkol_fact_t* base_x = eshkol_make_fact(arena, p_base, x_arg, 1);
    eshkol_fact_t* node_x = eshkol_make_fact(arena, p_node, x_arg, 1);
    eshkol_fact_t* safe_x = eshkol_make_fact(arena, p_safe, x_arg, 1);
    eshkol_fact_t* not_loop_x = eshkol_slg_make_negated_goal(arena, loop_x);

    eshkol_fact_t* lb1[1] = {base_x};
    eshkol_fact_t* lb2[1] = {loop_x};
    eshkol_fact_t* sb[2]  = {node_x, not_loop_x};
    eshkol_slg_add_rule(eng, eshkol_make_rule(arena, loop_x, lb1, 1, nullptr, 1.0, 1.0, ESHKOL_RULE_FLAG_TABLED, "loop_base"));
    eshkol_slg_add_rule(eng, eshkol_make_rule(arena, loop_x, lb2, 1, nullptr, 1.0, 1.0, ESHKOL_RULE_FLAG_TABLED, "loop_self"));
    eshkol_slg_add_rule(eng, eshkol_make_rule(arena, safe_x, sb, 2, nullptr, 1.0, 1.0, ESHKOL_RULE_FLAG_TABLED | ESHKOL_RULE_FLAG_STRATIFIED, "safe_strat"));

    eshkol_slg_truth_t t_loop1, t_loop2, t_safe1, t_safe2;
    check(eshkol_slg_query_ground(eng, eshkol_make_fact(arena, p_loop, n1, 1), &t_loop1, nullptr) == ESHKOL_SLG_OK && t_loop1 == ESHKOL_SLG_TRUE,
          "loop(1) is TRUE");
    check(eshkol_slg_query_ground(eng, eshkol_make_fact(arena, p_loop, n2, 1), &t_loop2, nullptr) == ESHKOL_SLG_OK && t_loop2 == ESHKOL_SLG_FALSE,
          "unfounded positive cycle loop(2) is FALSE");
    check(eshkol_slg_query_ground(eng, eshkol_make_fact(arena, p_safe, n1, 1), &t_safe1, nullptr) == ESHKOL_SLG_OK && t_safe1 == ESHKOL_SLG_FALSE,
          "stratified safe(1) is FALSE");
    check(eshkol_slg_query_ground(eng, eshkol_make_fact(arena, p_safe, n2, 1), &t_safe2, nullptr) == ESHKOL_SLG_OK && t_safe2 == ESHKOL_SLG_TRUE,
          "stratified safe(2) is TRUE via unfounded-set elimination");

    eshkol_slg_destroy(eng);
    arena_destroy(arena);
    std::printf("PASS: SLG 3-valued WFS Win-Move + unfounded-set elimination (iters=%u)\n",
                res.wfs_iterations);
}

static bool near(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol;
}

static void test_nars_truth_calculus() {
    const double k = ESHKOL_NARS_DEFAULT_HORIZON;
    eshkol_nars_truth_t a{}, b{}, c{}, r1{}, r2{};
    check(eshkol_nars_make_truth(0.8, 0.9, &a), "valid truth a constructed");
    check(eshkol_nars_make_truth(0.6, 0.75, &b), "valid truth b constructed");
    check(eshkol_nars_make_truth(0.25, 0.5, &c), "valid truth c constructed");

    /* Expectation */
    double exp_a = 0.0;
    check(eshkol_nars_expectation(a, &exp_a) && near(exp_a, 0.9 * (0.8 - 0.5) + 0.5),
          "NARS expectation exact");
    eshkol_nars_truth_t zero_conf{0.9, 0.0};
    double exp_z = 0.0;
    check(eshkol_nars_expectation(zero_conf, &exp_z) && near(exp_z, 0.5),
          "zero-confidence expectation is 0.5");

    /* Deduction: f = f1*f2, c = f1*f2*c1*c2 */
    check(eshkol_nars_deduction(a, b, &r1), "NARS deduction valid");
    check(near(r1.frequency, 0.8 * 0.6) && near(r1.confidence, 0.8 * 0.6 * 0.9 * 0.75),
          "NARS deduction exact");

    /* Abduction & Induction duality + weak-inference upper bound c < 1/(1+k) */
    check(eshkol_nars_abduction(a, b, k, &r1), "NARS abduction valid");
    check(eshkol_nars_induction(b, a, k, &r2), "NARS induction valid");
    check(near(r1.frequency, b.frequency) &&
          near(r1.confidence, (a.frequency * a.confidence * b.confidence) /
                              (a.frequency * a.confidence * b.confidence + k)),
          "NARS abduction exact");
    check(near(r1.frequency, r2.frequency) && near(r1.confidence, r2.confidence),
          "NARS abduction(a,b) == induction(b,a) duality");
    check(r1.confidence < 1.0 / (1.0 + k), "weak inference confidence strictly below 1/(1+k)");

    /* Revision: commutativity, associativity, strict evidential monotonicity, < 1 */
    check(eshkol_nars_revision(a, b, k, &r1) && eshkol_nars_revision(b, a, k, &r2),
          "NARS revision valid");
    check(near(r1.frequency, r2.frequency) && near(r1.confidence, r2.confidence),
          "NARS revision commutative");
    check(r1.confidence > a.confidence && r1.confidence > b.confidence && r1.confidence < 1.0,
          "NARS revision strictly monotonically increases confidence while staying < 1");

    eshkol_nars_truth_t ab{}, ab_c{}, bc{}, a_bc{};
    check(eshkol_nars_revision(a, b, k, &ab) && eshkol_nars_revision(ab, c, k, &ab_c),
          "NARS revision (a+b)+c");
    check(eshkol_nars_revision(b, c, k, &bc) && eshkol_nars_revision(a, bc, k, &a_bc),
          "NARS revision a+(b+c)");
    check(near(ab_c.frequency, a_bc.frequency, 1e-11) &&
          near(ab_c.confidence, a_bc.confidence, 1e-11),
          "NARS revision associative");

    /* Identity element (c = 0) */
    check(eshkol_nars_revision(a, zero_conf, k, &r1) &&
          near(r1.frequency, a.frequency) && near(r1.confidence, a.confidence),
          "NARS revision with c=0 is identity");

    /* Analogy & Comparison */
    check(eshkol_nars_analogy(a, b, &r1), "NARS analogy valid");
    check(near(r1.frequency, a.frequency * b.frequency) &&
          near(r1.confidence, b.frequency * a.confidence * b.confidence),
          "NARS analogy exact");

    check(eshkol_nars_comparison(a, b, k, &r1) && eshkol_nars_comparison(b, a, k, &r2),
          "NARS comparison valid");
    double f0 = a.frequency + b.frequency - a.frequency * b.frequency;
    double w_cmp = f0 * a.confidence * b.confidence;
    check(near(r1.frequency, (a.frequency * b.frequency) / f0) &&
          near(r1.confidence, w_cmp / (w_cmp + k)) &&
          near(r1.frequency, r2.frequency) && near(r1.confidence, r2.confidence),
          "NARS comparison exact and symmetric");

    /* Negation, Intersection, Union & De Morgan duality */
    eshkol_nars_truth_t na{}, nb{}, inter{}, n_inter{}, u_neg{};
    check(eshkol_nars_negation(a, &na) && eshkol_nars_negation(na, &r1), "NARS negation valid");
    check(near(r1.frequency, a.frequency) && near(r1.confidence, a.confidence),
          "NARS negation is an involution");
    check(eshkol_nars_negation(b, &nb), "negation b");
    check(eshkol_nars_intersection(a, b, &inter) && eshkol_nars_negation(inter, &n_inter),
          "neg(inter(a,b))");
    check(eshkol_nars_union(na, nb, &u_neg), "union(neg(a),neg(b))");
    check(near(n_inter.frequency, u_neg.frequency) && near(n_inter.confidence, u_neg.confidence),
          "NARS De Morgan duality: not(a and b) == (not a) or (not b)");

    /* Evidence round-trip */
    eshkol_nars_evidence_t ev{};
    check(eshkol_nars_to_evidence(a, 2.0, &ev), "to_evidence k=2");
    check(eshkol_nars_from_evidence(ev.positive, ev.total, 2.0, &r1), "from_evidence k=2");
    check(near(r1.frequency, a.frequency) && near(r1.confidence, a.confidence),
          "NARS evidence round-trip exact");

    /* Conversion, Contraposition, Exemplification */
    check(eshkol_nars_conversion(a, k, &r1) && near(r1.frequency, 1.0) && r1.confidence > 0.0,
          "NARS conversion valid");
    check(eshkol_nars_contraposition(a, k, &r1) && near(r1.frequency, 0.0) && r1.confidence > 0.0,
          "NARS contraposition valid");
    check(eshkol_nars_exemplification(a, b, k, &r1) && near(r1.frequency, 1.0) && r1.confidence > 0.0,
          "NARS exemplification valid");

    /* Negative controls: strict rejection of out-of-range, c=1, NaN, Inf, k<=0 */
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    const double inf_v = std::numeric_limits<double>::infinity();
    const eshkol_nars_truth_t bad_truths[] = {
        {-0.01, 0.5},
        {1.01, 0.5},
        {0.5, -0.01},
        {0.5, 1.0},   /* AIKR forbids c == 1.0 */
        {0.5, 1.5},
        {nan_v, 0.5},
        {0.5, nan_v},
        {inf_v, 0.5},
        {0.5, -inf_v}
    };
    for (const auto& bad : bad_truths) {
        check(!eshkol_nars_truth_is_valid(bad), "domain check rejects invalid truth");
        check(!eshkol_nars_make_truth(bad.frequency, bad.confidence, &r1),
              "make_truth rejects invalid truth");
        check(!eshkol_nars_expectation(bad, &exp_a), "expectation rejects invalid truth");
        check(!eshkol_nars_negation(bad, &r1), "negation rejects invalid truth");
        check(!eshkol_nars_deduction(bad, a, &r1), "deduction rejects invalid v1");
        check(!eshkol_nars_deduction(a, bad, &r1), "deduction rejects invalid v2");
        check(!eshkol_nars_revision(bad, a, k, &r1), "revision rejects invalid v1");
        check(!eshkol_nars_abduction(a, bad, k, &r1), "abduction rejects invalid v2");
        check(!eshkol_nars_induction(bad, a, k, &r1), "induction rejects invalid v1");
        check(!eshkol_nars_analogy(bad, a, &r1), "analogy rejects invalid v1");
        check(!eshkol_nars_comparison(a, bad, k, &r1), "comparison rejects invalid v2");
        check(!eshkol_nars_intersection(bad, a, &r1), "intersection rejects invalid v1");
        check(!eshkol_nars_union(a, bad, &r1), "union rejects invalid v2");
    }
    check(!eshkol_nars_revision(a, b, 0.0, &r1), "revision rejects k=0");
    check(!eshkol_nars_revision(a, b, -1.0, &r1), "revision rejects k<0");
    check(!eshkol_nars_revision(a, b, nan_v, &r1), "revision rejects k=NaN");
    check(!eshkol_nars_from_evidence(2.0, 1.0, k, &r1), "from_evidence rejects w_pos > w_total");
    check(!eshkol_nars_from_evidence(-0.1, 1.0, k, &r1), "from_evidence rejects negative w_pos");
    check(!eshkol_nars_deduction(a, b, nullptr), "deduction rejects NULL out");

    std::printf("PASS: NARS NAL-1..9 AIKR truth-value calculus + domain negative controls\n");
}

int main() {
    test_cdcl_sat_and_model_verifier();
    test_cdcl_unsat_drat_and_rat();
    test_cdcl_budget_enforcement();
    test_slg_left_recursive_cyclic_closure();
    test_slg_wfs_win_move_and_unfounded_sets();
    test_nars_truth_calculus();
    std::printf("CE-1 LOGIC CORE TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
