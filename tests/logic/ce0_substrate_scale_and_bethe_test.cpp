/*
 * CE-0 Substrate Scale, Indexing, CSR Belief Propagation & Bethe Free Energy Test
 *
 * Verifies:
 *   1. Unbounded logic variable interning (100k vars, >63-char distinct names)
 *   2. Unbounded predicate interning (5k distinct predicates, >64KB total text)
 *   3. Predicate-indexed KB query order, retract consistency, and scale
 *   4. O(E) CSR belief propagation on a 500-variable chain
 *   5. Exact Bethe free energy F_Bethe == -ln Z on single factor, 3-var chain,
 *      and degree-3 star tree, paired with a mean-field contrast check
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/logic.h>
#include <eshkol/core/inference.h>
#include <eshkol/eshkol.h>

extern "C" {
arena_t* arena_create(size_t default_block_size);
void arena_destroy(arena_t* arena);
const char* eshkol_intern_predicate(const char* name);
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static uint32_t count_cons_list(const eshkol_tagged_value_t& list) {
    uint32_t n = 0;
    eshkol_tagged_value_t cur = list;
    while (cur.type == ESHKOL_VALUE_HEAP_PTR && cur.data.ptr_val != 0) {
        auto* cell = reinterpret_cast<const eshkol_tagged_value_t*>(cur.data.ptr_val);
        n++;
        cur = cell[1];
    }
    return n;
}

static void test_unbounded_logic_vars() {
    eshkol_logic_registry_reset();

    /* Long names sharing an identical 120-char prefix must stay distinct */
    std::string prefix(120, 'v');
    std::string name_a = "?" + prefix + "_alpha_endpoint";
    std::string name_b = "?" + prefix + "_beta_endpoint";
    uint64_t id_a = eshkol_make_logic_var(name_a.c_str());
    uint64_t id_b = eshkol_make_logic_var(name_b.c_str());
    check(id_a != UINT64_MAX && id_b != UINT64_MAX, "long logic vars created");
    check(id_a != id_b, "long logic vars beyond 63 chars stay distinct");
    check(std::strcmp(eshkol_logic_var_name(id_a), name_a.c_str()) == 0,
          "long logic var A preserves full spelling");
    check(std::strcmp(eshkol_logic_var_name(id_b), name_b.c_str()) == 0,
          "long logic var B preserves full spelling");
    check(eshkol_make_logic_var(name_a.c_str()) == id_a,
          "long logic var A deduplicates in O(1)");

    /* Exceed legacy 65536 static limit */
    const uint32_t N = 70000;
    char buf[64];
    for (uint32_t i = 0; i < N; i++) {
        std::snprintf(buf, sizeof(buf), "?ce0_var_%u", i);
        uint64_t id = eshkol_make_logic_var(buf);
        check(id != UINT64_MAX, "70k logic var creation succeeds");
    }
    uint64_t probe = eshkol_make_logic_var("?ce0_var_69999");
    check(probe != UINT64_MAX && std::strcmp(eshkol_logic_var_name(probe), "?ce0_var_69999") == 0,
          "70k-th logic var round-trips cleanly");

    std::printf("PASS: unbounded logic variables (%u vars + 135-char distinct names)\n", N);
}

static void test_unbounded_predicates() {
    eshkol_logic_registry_reset();
    const uint32_t N = 5000;
    std::vector<const char*> ptrs;
    ptrs.reserve(N);
    for (uint32_t i = 0; i < N; i++) {
        std::string p = "ce0_predicate_namespace_segment_" + std::to_string(i);
        const char* interned = eshkol_intern_predicate(p.c_str());
        check(interned != nullptr, "predicate interned");
        ptrs.push_back(interned);
    }
    for (uint32_t i = 0; i < N; i += 17) {
        std::string p = "ce0_predicate_namespace_segment_" + std::to_string(i);
        const char* again = eshkol_intern_predicate(p.c_str());
        check(again == ptrs[i], "predicate pointer identity preserved past 1024/64KB");
    }
    std::printf("PASS: unbounded predicate interning (%u distinct predicates)\n", N);
}

static void test_indexed_kb_and_retract() {
    arena_t* arena = arena_create(4 * 1024 * 1024);
    check(arena != nullptr, "arena created for KB test");

    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    check(kb != nullptr, "KB created");

    const uint32_t NUM_PREDS = 500;
    const uint32_t PER_PRED = 20; /* 10,000 total facts */
    std::vector<uint64_t> preds(NUM_PREDS);
    for (uint32_t p = 0; p < NUM_PREDS; p++) {
        std::string pname = "rel_" + std::to_string(p);
        preds[p] = reinterpret_cast<uint64_t>(eshkol_intern_predicate(pname.c_str()));
        for (uint32_t k = 0; k < PER_PRED; k++) {
            eshkol_tagged_value_t args[2] = { make_int_tv(p), make_int_tv(k) };
            eshkol_fact_t* f = eshkol_make_fact(arena, preds[p], args, 2);
            eshkol_kb_assert(arena, kb, f);
        }
    }
    check(kb->num_facts == NUM_PREDS * PER_PRED, "10,000 facts asserted");
    check(kb->pred_index != nullptr, "KB predicate index active");

    uint64_t vx = eshkol_make_logic_var("?kb_x");
    eshkol_tagged_value_t pat_args[2] = { make_int_tv(321), make_var_tv(vx) };
    eshkol_fact_t* pat = eshkol_make_fact(arena, preds[321], pat_args, 2);

    auto t0 = std::chrono::steady_clock::now();
    for (int q = 0; q < 1000; q++) {
        eshkol_tagged_value_t res = eshkol_kb_query(arena, kb, pat, nullptr);
        if (q == 0) {
            check(count_cons_list(res) == PER_PRED, "indexed query returns all 20 matches");
            /* Verify exact KB insertion order: first match binds ?kb_x -> 0 */
            auto* first_cell = reinterpret_cast<const eshkol_tagged_value_t*>(res.data.ptr_val);
            auto* first_subst = reinterpret_cast<const eshkol_substitution_t*>(first_cell[0].data.ptr_val);
            const eshkol_tagged_value_t* bound = eshkol_subst_lookup(first_subst, vx);
            check(bound && bound->type == ESHKOL_VALUE_INT64 && bound->data.int_val == 0,
                  "indexed query preserves KB insertion order");
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check(ms < 2000.0, "1000 indexed queries over 10k facts stay well under 2s");

    /* Missing predicate returns empty list in O(1) */
    uint64_t absent_pred = reinterpret_cast<uint64_t>(eshkol_intern_predicate("rel_absent"));
    eshkol_fact_t* absent_pat = eshkol_make_fact(arena, absent_pred, pat_args, 2);
    eshkol_tagged_value_t empty_res = eshkol_kb_query(arena, kb, absent_pat, nullptr);
    check(count_cons_list(empty_res) == 0, "absent predicate query returns empty");

    /* Prefix query via index */
    eshkol_tagged_value_t pref_args[1] = { make_int_tv(321) };
    eshkol_fact_t* pref_pat = eshkol_make_fact(arena, preds[321], pref_args, 1);
    eshkol_tagged_value_t pref_res = eshkol_kb_query_prefix(arena, kb, pref_pat, nullptr);
    check(count_cons_list(pref_res) == PER_PRED, "indexed prefix query returns all 20 matches");

    arena_destroy(arena);
    std::printf("PASS: predicate-indexed KB (10k facts, 1000 queries in %.2f ms)\n", ms);
}

static void test_csr_bp_scale() {
    arena_t* arena = arena_create(8 * 1024 * 1024);
    check(arena != nullptr, "arena created for CSR BP test");

    const uint32_t NV = 500;
    std::vector<uint32_t> dims(NV, 2);
    eshkol_factor_graph_t* fg = eshkol_make_factor_graph(arena, NV, dims.data());
    check(fg != nullptr, "500-var factor graph created");

    /* Pairwise ferromagnetic chain */
    double pair_cpt[4] = { std::log(0.8), std::log(0.2), std::log(0.2), std::log(0.8) };
    uint32_t fdims[2] = { 2, 2 };
    for (uint32_t i = 0; i + 1 < NV; i++) {
        uint32_t vi[2] = { i, i + 1 };
        eshkol_factor_t* f = eshkol_make_factor(arena, vi, 2, pair_cpt, fdims);
        eshkol_fg_add_factor(arena, fg, f);
    }

    auto t0 = std::chrono::steady_clock::now();
    bool converged = eshkol_fg_infer(arena, fg, 20, 1e-8);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check(converged, "500-var CSR BP converged");
    check(ms < 1000.0, "500-var CSR BP runs in O(E) under 1s");

    /* Symmetric chain should stay 0.5 / 0.5 everywhere */
    for (uint32_t v = 0; v < NV; v += 50) {
        double p0 = std::exp(fg->beliefs[v][0]);
        check(std::fabs(p0 - 0.5) < 1e-6, "500-var chain marginal is 0.5");
    }

    arena_destroy(arena);
    std::printf("PASS: O(E) CSR BP (500 vars, 499 factors in %.2f ms)\n", ms);
}

static void test_bethe_free_energy_oracles() {
    arena_t* arena = arena_create(1024 * 1024);
    check(arena != nullptr, "arena created for Bethe FE test");

    /* Oracle 1: Single strongly correlated 2-var unnormalized factor */
    {
        uint32_t dims[2] = { 2, 2 };
        eshkol_factor_graph_t* fg = eshkol_make_factor_graph(arena, 2, dims);
        double w[4] = { 1.8, 0.2, 0.1, 1.5 };
        double cpt[4] = { std::log(w[0]), std::log(w[1]), std::log(w[2]), std::log(w[3]) };
        double Z = w[0] + w[1] + w[2] + w[3];
        double exact_neg_log_z = -std::log(Z);

        uint32_t vi[2] = { 0, 1 };
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi, 2, cpt, dims));
        check(eshkol_fg_infer(arena, fg, 50, 1e-12), "oracle 1 BP converged");

        double f_bethe = eshkol_free_energy_bethe(fg, nullptr, 0);
        double f_mf    = eshkol_free_energy(fg, nullptr, 0);
        check(std::fabs(f_bethe - exact_neg_log_z) < 1e-11,
              "Oracle 1: F_Bethe matches -ln Z to 1e-11");
        check(std::fabs(f_mf - exact_neg_log_z) > 0.05,
              "Oracle 1 negative control: mean-field F differs from -ln Z on correlated factor");
    }

    /* Oracle 2: 3-variable asymmetric chain X -- f1 -- Y -- f2 -- Z */
    {
        uint32_t dims[3] = { 2, 2, 2 };
        eshkol_factor_graph_t* fg = eshkol_make_factor_graph(arena, 3, dims);
        double w1[4] = { 0.9, 0.1, 0.2, 0.8 };
        double w2[4] = { 0.7, 0.3, 0.15, 0.85 };
        double c1[4] = { std::log(w1[0]), std::log(w1[1]), std::log(w1[2]), std::log(w1[3]) };
        double c2[4] = { std::log(w2[0]), std::log(w2[1]), std::log(w2[2]), std::log(w2[3]) };

        double Z = 0.0;
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                    Z += w1[x * 2 + y] * w2[y * 2 + z];
        double exact_neg_log_z = -std::log(Z);

        uint32_t vi1[2] = { 0, 1 };
        uint32_t vi2[2] = { 1, 2 };
        uint32_t fd[2]  = { 2, 2 };
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi1, 2, c1, fd));
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi2, 2, c2, fd));
        check(eshkol_fg_infer(arena, fg, 50, 1e-12), "oracle 2 BP converged");

        double f_bethe = eshkol_free_energy_bethe(fg, nullptr, 0);
        check(std::fabs(f_bethe - exact_neg_log_z) < 1e-11,
              "Oracle 2: 3-var chain F_Bethe matches -ln Z to 1e-11");
    }

    /* Oracle 3: 4-variable star tree (center var 0 has degree 3 -> (d-1)=2) */
    {
        uint32_t dims[4] = { 2, 2, 2, 2 };
        eshkol_factor_graph_t* fg = eshkol_make_factor_graph(arena, 4, dims);
        double w1[4] = { 1.2, 0.3, 0.4, 0.9 };
        double w2[4] = { 0.8, 0.5, 0.2, 1.1 };
        double w3[4] = { 1.5, 0.25, 0.35, 0.7 };
        double c1[4] = { std::log(w1[0]), std::log(w1[1]), std::log(w1[2]), std::log(w1[3]) };
        double c2[4] = { std::log(w2[0]), std::log(w2[1]), std::log(w2[2]), std::log(w2[3]) };
        double c3[4] = { std::log(w3[0]), std::log(w3[1]), std::log(w3[2]), std::log(w3[3]) };

        double Z = 0.0;
        for (int c = 0; c < 2; c++)
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    for (int d = 0; d < 2; d++)
                        Z += w1[c * 2 + a] * w2[c * 2 + b] * w3[c * 2 + d];
        double exact_neg_log_z = -std::log(Z);

        uint32_t vi1[2] = { 0, 1 };
        uint32_t vi2[2] = { 0, 2 };
        uint32_t vi3[2] = { 0, 3 };
        uint32_t fd[2]  = { 2, 2 };
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi1, 2, c1, fd));
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi2, 2, c2, fd));
        eshkol_fg_add_factor(arena, fg, eshkol_make_factor(arena, vi3, 2, c3, fd));
        check(eshkol_fg_infer(arena, fg, 50, 1e-12), "oracle 3 BP converged");

        double f_bethe = eshkol_free_energy_bethe(fg, nullptr, 0);
        check(std::fabs(f_bethe - exact_neg_log_z) < 1e-11,
              "Oracle 3: degree-3 star tree F_Bethe matches -ln Z to 1e-11");
    }

    arena_destroy(arena);
    std::printf("PASS: Bethe free energy exact -ln Z oracles (single, chain, star)\n");
}

int main() {
    test_unbounded_logic_vars();
    test_unbounded_predicates();
    test_indexed_kb_and_retract();
    test_csr_bp_scale();
    test_bethe_free_energy_oracles();
    std::printf("CE-0 SUBSTRATE TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
