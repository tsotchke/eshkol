/*
 * CE-3 Native ASP Solver, Colored Differentiable E-Graph (diff-egglog),
 * and Fail-Closed Versioned Binary Persistence Verification Suite
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/asp_solver.h>
#include <eshkol/core/ce_persist.h>
#include <eshkol/core/diff_egglog.h>
#include <eshkol/core/soft_unify_semiring.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

static int g_checks = 0;

static void check(bool cond, const char* msg) {
    g_checks++;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static bool near(double a, double b, double tol = 1e-7) {
    return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol;
}

static std::set<std::vector<uint32_t>> collect_models(const eshkol_asp_solver_t* s) {
    std::set<std::vector<uint32_t>> out;
    uint32_t n = eshkol_asp_num_models(s);
    for (uint32_t i = 0; i < n; i++) {
        const eshkol_asp_model_t* m = eshkol_asp_get_model(s, i);
        check(m != nullptr, "ASP model pointer valid");
        check(eshkol_asp_verify_stable_model(s, m->true_atoms, m->num_true_atoms),
              "independent Gelfond-Lifschitz verifier accepts enumerated model");
        std::vector<uint32_t> vec;
        if (m->num_true_atoms > 0) vec.assign(m->true_atoms, m->true_atoms + m->num_true_atoms);
        out.insert(std::move(vec));
    }
    return out;
}

static void test_asp_solver_stable_models() {
    eshkol_asp_config_t cfg = eshkol_asp_default_config();

    /* 1. Positive self-loop: p :- p => ONLY empty stable model {} (rejects non-founded {p}) */
    {
        eshkol_asp_solver_t* s = eshkol_asp_create(1, &cfg);
        uint32_t pos[1] = {1};
        check(eshkol_asp_add_normal_rule(s, 1, pos, 1, nullptr, 0), "added p :- p");
        eshkol_asp_stats_t st{};
        check(eshkol_asp_solve(s, &st) == ESHKOL_ASP_SAT, "p :- p is SAT");
        auto models = collect_models(s);
        std::set<std::vector<uint32_t>> expected = {{}};
        check(models == expected, "p :- p yields ONLY empty stable model {}");
        uint32_t bogus[1] = {1};
        check(!eshkol_asp_verify_stable_model(s, bogus, 1),
              "negative control: {p} is rejected as non-founded");
        eshkol_asp_destroy(s);
    }

    /* 2. Mutual positive loop with external escape: a :- b; b :- a; a :- not c; {c} */
    {
        eshkol_asp_solver_t* s = eshkol_asp_create(3, &cfg); /* 1=a, 2=b, 3=c */
        uint32_t pa[1] = {1}, pb[1] = {2}, nc[1] = {3}, hc[1] = {3};
        check(eshkol_asp_add_normal_rule(s, 1, pb, 1, nullptr, 0), "a :- b");
        check(eshkol_asp_add_normal_rule(s, 2, pa, 1, nullptr, 0), "b :- a");
        check(eshkol_asp_add_normal_rule(s, 1, nullptr, 0, nc, 1), "a :- not c");
        check(eshkol_asp_add_choice_rule(s, hc, 1, 0, 1, nullptr, 0, nullptr, 0), "{c}");
        eshkol_asp_stats_t st{};
        check(eshkol_asp_solve(s, &st) == ESHKOL_ASP_SAT, "mutual loop + choice is SAT");
        auto models = collect_models(s);
        std::set<std::vector<uint32_t>> expected = {{1, 2}, {3}};
        check(models == expected, "mutual loop yields exact {{a,b}, {c}} (rejects {a,b,c})");
        uint32_t bogus[3] = {1, 2, 3};
        check(!eshkol_asp_verify_stable_model(s, bogus, 3),
              "negative control: {a,b,c} rejected as unfounded loop");
        eshkol_asp_destroy(s);
    }

    /* 3. Even negative cycle: p :- not q; q :- not p => exact 2 stable models {p}, {q} */
    {
        eshkol_asp_solver_t* s = eshkol_asp_create(2, &cfg);
        uint32_t nq[1] = {2}, np[1] = {1};
        check(eshkol_asp_add_normal_rule(s, 1, nullptr, 0, nq, 1), "p :- not q");
        check(eshkol_asp_add_normal_rule(s, 2, nullptr, 0, np, 1), "q :- not p");
        check(eshkol_asp_solve(s, nullptr) == ESHKOL_ASP_SAT, "even cycle is SAT");
        auto models = collect_models(s);
        std::set<std::vector<uint32_t>> expected = {{1}, {2}};
        check(models == expected, "even cycle yields exact 2 stable models {p} and {q}");
        eshkol_asp_destroy(s);
    }

    /* 4. Odd negative cycle: p :- not p => 0 stable models (UNSAT) */
    {
        eshkol_asp_solver_t* s = eshkol_asp_create(1, &cfg);
        uint32_t np[1] = {1};
        check(eshkol_asp_add_normal_rule(s, 1, nullptr, 0, np, 1), "p :- not p");
        check(eshkol_asp_solve(s, nullptr) == ESHKOL_ASP_UNSAT, "odd cycle is UNSAT");
        check(eshkol_asp_num_models(s) == 0, "odd cycle yields 0 stable models");
        eshkol_asp_destroy(s);
    }

    /* 5. Integrity constraint filtering: {a,b,c}, :- a,b, :- not c => {{c}, {a,c}, {b,c}} */
    {
        eshkol_asp_solver_t* s = eshkol_asp_create(3, &cfg);
        uint32_t all[3] = {1, 2, 3};
        uint32_t ab[2] = {1, 2};
        uint32_t c_only[1] = {3};
        check(eshkol_asp_add_choice_rule(s, all, 3, 0, 3, nullptr, 0, nullptr, 0), "{a,b,c}");
        check(eshkol_asp_add_constraint(s, ab, 2, nullptr, 0), ":- a, b");
        check(eshkol_asp_add_constraint(s, nullptr, 0, c_only, 1), ":- not c");
        check(eshkol_asp_solve(s, nullptr) == ESHKOL_ASP_SAT, "constraint filtering SAT");
        auto models = collect_models(s);
        std::set<std::vector<uint32_t>> expected = {{3}, {1, 3}, {2, 3}};
        check(models == expected, "integrity constraints yield exact {{c},{a,c},{b,c}}");
        eshkol_asp_destroy(s);
    }

    /* 6. Graph 3-coloring ASP encoding: K3 has 6 stable models, K4 has 0 (UNSAT) */
    auto test_k_clique_3color = [&](uint32_t n_verts, uint32_t expected_models) {
        eshkol_asp_solver_t* s = eshkol_asp_create(n_verts * 3u, &cfg);
        auto col_atom = [](uint32_t v, uint32_t c) -> uint32_t { return v * 3u + c + 1u; };
        for (uint32_t v = 0; v < n_verts; v++) {
            uint32_t h[3] = {col_atom(v, 0), col_atom(v, 1), col_atom(v, 2)};
            check(eshkol_asp_add_choice_rule(s, h, 3, 1, 1, nullptr, 0, nullptr, 0),
                  "vertex exact-1-of-3 color choice");
        }
        for (uint32_t u = 0; u < n_verts; u++) {
            for (uint32_t v = u + 1; v < n_verts; v++) {
                for (uint32_t c = 0; c < 3; c++) {
                    uint32_t clash[2] = {col_atom(u, c), col_atom(v, c)};
                    check(eshkol_asp_add_constraint(s, clash, 2, nullptr, 0),
                          "adjacent same-color integrity constraint");
                }
            }
        }
        eshkol_asp_status_t st = eshkol_asp_solve(s, nullptr);
        if (expected_models == 0) {
            check(st == ESHKOL_ASP_UNSAT && eshkol_asp_num_models(s) == 0,
                  "K4 3-coloring ASP is UNSAT (0 stable models)");
        } else {
            check(st == ESHKOL_ASP_SAT, "K3 3-coloring ASP is SAT");
            auto models = collect_models(s);
            check(models.size() == expected_models, "K3 3-coloring yields exact 6 stable models");
        }
        eshkol_asp_destroy(s);
    };
    test_k_clique_3color(3, 6);
    test_k_clique_3color(4, 0);

    std::printf("PASS: native ASP solver (self-loop, mutual loop, even/odd cycles, constraints, K3/K4 3-coloring)\n");
}

static eshkol_egraph_t* build_colored_test_egraph(uint32_t& out_c1, uint32_t& out_c2,
                                                  uint32_t& out_c1_sub,
                                                  uint32_t& out_root) {
    /* Ops: 1=sym_x, 2=sym_y, 3=sym_z, 4=const_2, 5=const_1,
     *      10=mul, 11=add, 12=shl, 13=g, 14=sym_u, 15=sym_v */
    eshkol_egraph_t* eg = eshkol_egraph_create();
    uint32_t nx = eshkol_egraph_add_node(eg, 1, nullptr, 0, 1.2);
    uint32_t ny = eshkol_egraph_add_node(eg, 2, nullptr, 0, 1.5);
    uint32_t nz = eshkol_egraph_add_node(eg, 3, nullptr, 0, 0.7);
    uint32_t n2 = eshkol_egraph_add_node(eg, 4, nullptr, 0, 0.8);
    uint32_t n1 = eshkol_egraph_add_node(eg, 5, nullptr, 0, 0.5);

    /* Hash-consing identity check in base color 0 */
    check(eshkol_egraph_add_node(eg, 1, nullptr, 0, 1.2) == nx, "hash-cons deduplicates identical node");

    uint32_t mul_ch[2] = {nx, n2};
    uint32_t add_ch[2] = {nx, nx};
    uint32_t shl_ch[2] = {nx, n1};
    uint32_t e_mul = eshkol_egraph_add_node(eg, 10, mul_ch, 2, 4.0); /* total ~6.0 */
    uint32_t e_add = eshkol_egraph_add_node(eg, 11, add_ch, 2, 2.0); /* total ~4.4 */
    uint32_t e_shl = eshkol_egraph_add_node(eg, 12, shl_ch, 2, 1.1); /* total ~2.8 */

    /* g(x), g(y), g(z) */
    uint32_t gx = eshkol_egraph_add_node(eg, 13, &nx, 1, 2.0);
    uint32_t gy = eshkol_egraph_add_node(eg, 13, &ny, 1, 2.5);
    uint32_t gz = eshkol_egraph_add_node(eg, 13, &nz, 1, 1.0);

    uint32_t nu = eshkol_egraph_add_node(eg, 14, nullptr, 0, 1.0);
    uint32_t nv = eshkol_egraph_add_node(eg, 15, nullptr, 0, 1.0);

    /* Base color 0 equivalences: mul(x,2) == add(x,x) == shl(x,1) */
    check(eshkol_egraph_union(eg, e_mul, e_add, 0), "base union mul==add");
    check(eshkol_egraph_union(eg, e_add, e_shl, 0), "base union add==shl");

    /* Create hierarchical colors */
    out_c1     = eshkol_egraph_create_color(eg, 0);
    out_c2     = eshkol_egraph_create_color(eg, 0);
    out_c1_sub = eshkol_egraph_create_color(eg, out_c1);
    check(out_c1 == 1 && out_c2 == 2 && out_c1_sub == 3, "colored assumption tree created");

    /* Merge x == y ONLY inside color C1 (and e_mul == gz shortcut inside C1_sub!) */
    check(eshkol_egraph_union(eg, nx, ny, out_c1), "C1 union x==y");
    check(eshkol_egraph_union(eg, ny, nz, out_c1_sub), "C1_sub union y==z");
    check(eshkol_egraph_union(eg, e_mul, gz, out_c1_sub), "C1_sub shortcut mul==g(z)");

    /* Base color union AFTER child colors exist must propagate to all colors */
    check(eshkol_egraph_union(eg, nu, nv, 0), "post-color base union u==v");
    check(eshkol_egraph_rebuild(eg), "colored congruence closure rebuilt");

    /* Verify strict color isolation & congruence closure! */
    check(!eshkol_egraph_equiv(eg, nx, ny, 0), "base color 0 has ZERO leakage from C1 (x != y)");
    check(!eshkol_egraph_equiv(eg, gx, gy, 0), "base color 0 congruence isolated (g(x) != g(y))");
    check(!eshkol_egraph_equiv(eg, nx, ny, out_c2), "sibling C2 has ZERO leakage from C1 (x != y)");
    check(!eshkol_egraph_equiv(eg, gx, gy, out_c2), "sibling C2 congruence isolated (g(x) != g(y))");

    check(eshkol_egraph_equiv(eg, nx, ny, out_c1), "C1 sees x == y");
    check(eshkol_egraph_equiv(eg, gx, gy, out_c1), "C1 derives congruence g(x) == g(y)");
    check(!eshkol_egraph_equiv(eg, nx, nz, out_c1), "parent C1 has ZERO leakage from child C1_sub (x != z)");
    check(!eshkol_egraph_equiv(eg, gx, gz, out_c1), "parent C1 congruence isolated from child (g(x) != g(z))");

    check(eshkol_egraph_equiv(eg, nx, nz, out_c1_sub), "descendant C1_sub inherits x==y and derives x==z");
    check(eshkol_egraph_equiv(eg, gx, gz, out_c1_sub), "descendant C1_sub derives congruence g(x)==g(z)");

    /* Base union u==v visible everywhere */
    check(eshkol_egraph_equiv(eg, nu, nv, 0) &&
          eshkol_egraph_equiv(eg, nu, nv, out_c1) &&
          eshkol_egraph_equiv(eg, nu, nv, out_c2) &&
          eshkol_egraph_equiv(eg, nu, nv, out_c1_sub),
          "base color 0 union inherited across all colors");

    out_root = e_mul;
    return eg;
}

static void verify_smoothe_grads(const char* label, eshkol_egraph_t* eg,
                                 uint32_t root, uint32_t color_id,
                                 double tau, uint32_t iters) {
    const uint32_t N = eshkol_egraph_num_nodes(eg);
    std::vector<double> grads(N, 0.0);
    eshkol_smoothe_result_t res{};
    check(eshkol_egraph_smoothe_extract(eg, root, color_id, tau, iters, &res, grads.data(), N),
          "SmoothE extraction succeeded");
    check(res.valid && res.fd_evals == 0, "SmoothE uses 0 finite-difference evals");

    const double h = 1e-6;
    double max_err = 0.0;
    for (uint32_t i = 0; i < N; i++) {
        double orig = 0.0;
        check(eshkol_egraph_inspect_node(eg, i, nullptr, nullptr, nullptr, &orig), "inspect node cost");

        eshkol_egraph_set_node_cost(eg, i, orig + h);
        eshkol_smoothe_result_t rp{};
        check(eshkol_egraph_smoothe_extract(eg, root, color_id, tau, iters, &rp, nullptr, 0), "FD+");

        eshkol_egraph_set_node_cost(eg, i, orig - h);
        eshkol_smoothe_result_t rm{};
        check(eshkol_egraph_smoothe_extract(eg, root, color_id, tau, iters, &rm, nullptr, 0), "FD-");

        eshkol_egraph_set_node_cost(eg, i, orig);
        double fd = (rp.root_cost - rm.root_cost) / (2.0 * h);
        max_err = std::max(max_err, std::fabs(grads[i] - fd));
        check(near(grads[i], fd, 1e-7), "SmoothE analytical gradient matches central FD < 1e-7");
    }
    std::printf("PASS: SmoothE (%s: root_cost=%.6f, max_grad_err=%.2e, fd_evals=0)\n",
                label, res.root_cost, max_err);
}

static void test_colored_diff_egglog_and_smoothe() {
    uint32_t c1 = 0, c2 = 0, c1_sub = 0, root = 0;
    eshkol_egraph_t* eg = build_colored_test_egraph(c1, c2, c1_sub, root);

    /* Low-temperature SmoothE in base color 0 approaches min(6.0, 4.4, 2.8) = 2.8,
     * while in C1_sub shortcut g(z) has cost 1.0 + 0.7 = 1.7! */
    eshkol_smoothe_result_t r0{}, r_sub{};
    check(eshkol_egraph_smoothe_extract(eg, root, 0, 0.05, 12, &r0, nullptr, 0), "SmoothE base low-tau");
    check(eshkol_egraph_smoothe_extract(eg, root, c1_sub, 0.05, 12, &r_sub, nullptr, 0), "SmoothE C1_sub low-tau");
    check(near(r0.root_cost, 2.8, 1e-3), "base color low-tau SmoothE approaches exact min 2.8");
    check(near(r_sub.root_cost, 1.7, 1e-3), "C1_sub low-tau SmoothE approaches shortcut min 1.7");

    verify_smoothe_grads("base-color-0", eg, root, 0, 0.8, 12);
    verify_smoothe_grads("descendant-C1_sub", eg, root, c1_sub, 0.8, 12);

    /* Cyclic e-graph test: a == f(a) with positive node costs */
    eshkol_egraph_t* cyc = eshkol_egraph_create();
    uint32_t a = eshkol_egraph_add_node(cyc, 1, nullptr, 0, 2.0);
    uint32_t fa = eshkol_egraph_add_node(cyc, 2, &a, 1, 1.5);
    check(eshkol_egraph_union(cyc, a, fa, 0) && eshkol_egraph_rebuild(cyc), "cyclic e-graph built");
    verify_smoothe_grads("cyclic-a==f(a)", cyc, a, 0, 0.9, 16);

    eshkol_egraph_destroy(cyc);
    eshkol_egraph_destroy(eg);
}

static void test_versioned_binary_persistence() {
    /* 1. Build a CE-2 PROOF_DAG and test full binary round-trip + fail-closed corruption */
    eshkol_proof_dag_t* dag = eshkol_proof_dag_create();
    uint32_t d0 = eshkol_proof_dag_add_node(dag, ESHKOL_PROOF_AXIOM, "axiom:r1",
                                            nullptr, 0, eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.85));
    uint32_t d1 = eshkol_proof_dag_add_node(dag, ESHKOL_PROOF_SOFT_UNIFY, "soft:u1",
                                            nullptr, 0, eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.72));
    eshkol_semiring_val_t vm{};
    eshkol_sr_mul(eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.85),
                  eshkol_sr_scalar(ESHKOL_SR_VITERBI, 0.72), &vm);
    uint32_t p[2] = {d0, d1};
    uint32_t d2 = eshkol_proof_dag_add_node(dag, ESHKOL_PROOF_SEMIRING_MUL, "derive:p1",
                                            p, 2, vm);
    check(d2 == 2, "PROOF_DAG constructed for persistence");

    uint8_t* dag_bytes = nullptr;
    size_t dag_len = 0;
    check(eshkol_ce_serialize_proof_dag(dag, &dag_bytes, &dag_len) == ESHKOL_PERSIST_OK,
          "PROOF_DAG serialized");
    check(dag_bytes != nullptr && dag_len > ESHKOL_CE_PERSIST_HEADER_BYTES,
          "PROOF_DAG binary non-empty");

    eshkol_proof_dag_t* dag2 = nullptr;
    check(eshkol_ce_deserialize_proof_dag(dag_bytes, dag_len, &dag2) == ESHKOL_PERSIST_OK &&
          dag2 != nullptr,
          "PROOF_DAG deserialized cleanly");
    check(eshkol_proof_dag_num_nodes(dag2) == 3, "PROOF_DAG node count preserved");
    for (uint32_t i = 0; i < 3; i++) {
        check(eshkol_proof_dag_node_hash(dag, i) == eshkol_proof_dag_node_hash(dag2, i),
              "PROOF_DAG Merkle hash preserved across round-trip");
    }

    /* Byte-for-byte re-serialization identity */
    uint8_t* dag_bytes2 = nullptr;
    size_t dag_len2 = 0;
    check(eshkol_ce_serialize_proof_dag(dag2, &dag_bytes2, &dag_len2) == ESHKOL_PERSIST_OK,
          "PROOF_DAG re-serialized");
    check(dag_len == dag_len2 && std::memcmp(dag_bytes, dag_bytes2, dag_len) == 0,
          "PROOF_DAG byte-for-byte deterministic round-trip");
    eshkol_ce_buffer_free(dag_bytes2);
    eshkol_proof_dag_destroy(dag2);

    /* Fail-closed negative controls on PROOF_DAG bytes:
     * (a) 1-bit flip at EVERY single byte offset MUST fail closed! */
    for (size_t pos = 0; pos < dag_len; pos++) {
        std::vector<uint8_t> flipped(dag_bytes, dag_bytes + dag_len);
        flipped[pos] ^= 0x04u;
        eshkol_proof_dag_t* bad = nullptr;
        eshkol_persist_status_t st = eshkol_ce_deserialize_proof_dag(
            flipped.data(), flipped.size(), &bad);
        check(st != ESHKOL_PERSIST_OK && bad == nullptr,
              "fail-closed: 1-bit flip rejected at every byte position");
    }

    /* (b) Truncated payload & trailing byte */
    {
        eshkol_proof_dag_t* bad = nullptr;
        check(eshkol_ce_deserialize_proof_dag(dag_bytes, dag_len - 1u, &bad) == ESHKOL_PERSIST_ERR_TRUNCATED &&
              bad == nullptr,
              "fail-closed: 1-byte truncated PROOF_DAG rejected");
        check(eshkol_ce_deserialize_proof_dag(dag_bytes, 12u, &bad) == ESHKOL_PERSIST_ERR_TRUNCATED &&
              bad == nullptr,
              "fail-closed: truncated header rejected");
        std::vector<uint8_t> trailing(dag_bytes, dag_bytes + dag_len);
        trailing.push_back(0x00u);
        check(eshkol_ce_deserialize_proof_dag(trailing.data(), trailing.size(), &bad) == ESHKOL_PERSIST_ERR_TRUNCATED &&
              bad == nullptr,
              "fail-closed: trailing garbage byte rejected");
    }

    /* (c) Unknown schema version (version = 99 with valid recomputed CRC!) */
    {
        std::vector<uint8_t> v99(dag_bytes, dag_bytes + dag_len);
        v99[4] = 99u; v99[5] = 0u; v99[6] = 0u; v99[7] = 0u;
        std::vector<uint8_t> crc_in;
        crc_in.insert(crc_in.end(), v99.begin(), v99.begin() + 16);
        crc_in.insert(crc_in.end(), v99.begin() + 24, v99.end());
        uint32_t new_crc = eshkol_ce_crc32(crc_in.data(), crc_in.size());
        v99[16] = static_cast<uint8_t>(new_crc & 0xFFu);
        v99[17] = static_cast<uint8_t>((new_crc >> 8) & 0xFFu);
        v99[18] = static_cast<uint8_t>((new_crc >> 16) & 0xFFu);
        v99[19] = static_cast<uint8_t>((new_crc >> 24) & 0xFFu);
        eshkol_proof_dag_t* bad = nullptr;
        check(eshkol_ce_deserialize_proof_dag(v99.data(), v99.size(), &bad) == ESHKOL_PERSIST_ERR_BAD_VERSION &&
              bad == nullptr,
              "fail-closed: unknown schema version 99 rejected");
    }

    /* (d) Wrong payload kind */
    {
        eshkol_egraph_t* bad_eg = nullptr;
        check(eshkol_ce_deserialize_egraph(dag_bytes, dag_len, &bad_eg) == ESHKOL_PERSIST_ERR_BAD_KIND &&
              bad_eg == nullptr,
              "fail-closed: PROOF_DAG bytes rejected by egraph deserializer");
    }

    /* 2. CE-3 Colored E-Graph round-trip + atomic file snapshot */
    uint32_t c1 = 0, c2 = 0, c1_sub = 0, root = 0;
    eshkol_egraph_t* eg = build_colored_test_egraph(c1, c2, c1_sub, root);
    uint8_t* eg_bytes = nullptr;
    size_t eg_len = 0;
    check(eshkol_ce_serialize_egraph(eg, &eg_bytes, &eg_len) == ESHKOL_PERSIST_OK,
          "colored e-graph serialized");

    const char* snap_path = "ce3_test_egraph_snapshot.bin";
    check(eshkol_ce_write_file_atomic(snap_path, eg_bytes, eg_len) == ESHKOL_PERSIST_OK,
          "atomic snapshot write succeeded");
    uint8_t* disk_bytes = nullptr;
    size_t disk_len = 0;
    check(eshkol_ce_read_file(snap_path, &disk_bytes, &disk_len) == ESHKOL_PERSIST_OK,
          "atomic snapshot read succeeded");
    std::remove(snap_path);
    check(disk_len == eg_len && std::memcmp(disk_bytes, eg_bytes, eg_len) == 0,
          "atomic file snapshot byte-exact");

    eshkol_egraph_t* eg2 = nullptr;
    check(eshkol_ce_deserialize_egraph(disk_bytes, disk_len, &eg2) == ESHKOL_PERSIST_OK &&
          eg2 != nullptr,
          "colored e-graph deserialized from atomic snapshot");

    /* Verify identical colored equivalences & SmoothE costs on restored e-graph */
    check(eshkol_egraph_num_colors(eg2) == eshkol_egraph_num_colors(eg) &&
          eshkol_egraph_num_nodes(eg2) == eshkol_egraph_num_nodes(eg),
          "restored e-graph color & node counts match");
    check(!eshkol_egraph_equiv(eg2, 0, 1, 0) &&
          !eshkol_egraph_equiv(eg2, 0, 1, c2) &&
          eshkol_egraph_equiv(eg2, 0, 1, c1) &&
          eshkol_egraph_equiv(eg2, 0, 2, c1_sub),
          "restored e-graph preserves exact color isolation and inheritance");

    eshkol_smoothe_result_t s_orig{}, s_rest{};
    check(eshkol_egraph_smoothe_extract(eg, root, c1_sub, 0.6, 12, &s_orig, nullptr, 0) &&
          eshkol_egraph_smoothe_extract(eg2, root, c1_sub, 0.6, 12, &s_rest, nullptr, 0) &&
          near(s_orig.root_cost, s_rest.root_cost, 1e-13),
          "restored e-graph reproduces exact SmoothE cost");

    /* Byte-for-byte re-serialization of e-graph */
    uint8_t* eg_bytes2 = nullptr;
    size_t eg_len2 = 0;
    check(eshkol_ce_serialize_egraph(eg2, &eg_bytes2, &eg_len2) == ESHKOL_PERSIST_OK &&
          eg_len == eg_len2 && std::memcmp(eg_bytes, eg_bytes2, eg_len) == 0,
          "colored e-graph byte-for-byte deterministic round-trip");

    /* 1-bit flip rejection on e-graph bytes */
    std::vector<uint8_t> eg_flip(eg_bytes, eg_bytes + eg_len);
    eg_flip[eg_len / 2u] ^= 0x01u;
    eshkol_egraph_t* bad_eg = nullptr;
    check(eshkol_ce_deserialize_egraph(eg_flip.data(), eg_flip.size(), &bad_eg) == ESHKOL_PERSIST_ERR_CHECKSUM &&
          bad_eg == nullptr,
          "fail-closed: 1-bit flip in e-graph snapshot rejected with CHECKSUM error");

    eshkol_ce_buffer_free(dag_bytes);
    eshkol_ce_buffer_free(eg_bytes);
    eshkol_ce_buffer_free(eg_bytes2);
    eshkol_ce_buffer_free(disk_bytes);
    eshkol_proof_dag_destroy(dag);
    eshkol_egraph_destroy(eg);
    eshkol_egraph_destroy(eg2);

    std::printf("PASS: fail-closed versioned binary persistence (PROOF_DAG + colored e-graph, 1-bit flip sweep=%zu positions)\n",
                dag_len);
}

int main() {
    test_asp_solver_stable_models();
    test_colored_diff_egglog_and_smoothe();
    test_versioned_binary_persistence();
    std::printf("CE-3 ASP, DIFF-EGGLOG & PERSIST TEST: PASS (%d checks)\n", g_checks);
    return 0;
}
