/*
 * CE-9 Native Verification Suite:
 *   1. Chained eshkol.certificate.v2 Derivation Verifier + Lean 4.12.0 Export (Guards 1, 2, 3)
 *   2. Stratified Recursive IVM with DRed Cycle/Diamond Maintenance (Guards 4, 5, 6)
 *   3. ADR-0005 Differentiable Program-Capsule Foundations (Guards 7, 8, 9)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/cert_chain_v2.h"
#include "eshkol/core/recursive_ivm.h"
#include "eshkol/core/program_capsule.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            ++g_pass;                                                         \
        } else {                                                              \
            ++g_fail;                                                         \
            std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);          \
        }                                                                     \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    double next_range(double lo, double hi) {
        return lo + (hi - lo) * next_unit();
    }
};

std::string resolve_lean4_binary() {
    if (const char* env_bin = std::getenv("ESHKOL_LEAN4_BIN")) {
        if (env_bin[0] != '\0') return std::string(env_bin);
    }
    if (const char* home = std::getenv("HOME")) {
        std::string elan_lean = std::string(home) + "/.elan/bin/lean";
        std::ifstream f(elan_lean);
        if (f.good()) return elan_lean;
    }
    return "lean";
}

std::string temp_dir_prefix() {
    if (const char* tmp = std::getenv("TMPDIR")) {
        if (tmp[0] != '\0') {
            std::string s(tmp);
            if (s.back() != '/') s.push_back('/');
            return s;
        }
    }
    return "/tmp/";
}

/* ============================================================================
 * Part 1: Chained eshkol.certificate.v2 & Lean 4.12.0 Export (Guards 1, 2, 3)
 * ============================================================================ */

void test_cert_chain_v2_and_lean4(const char* export_lean_path) {
    std::printf("\n=== CE-9 Part 1: Chained eshkol.certificate.v2 & Lean 4.12.0 Export ===\n");

    /* Verify SHA-256 against NIST test vector ("abc"). */
    uint8_t abc_hash[32];
    eshkol_cert_sha256(reinterpret_cast<const uint8_t*>("abc"), 3, abc_hash);
    const uint8_t expected_abc[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    CHECK(std::memcmp(abc_hash, expected_abc, 32) == 0, "SHA-256 matches NIST vector for 'abc'");

    /* Build a smooth neural world model (2D -> 2D) certified via CE-8 Interval-Taylor. */
    eshkol_tm_world_model_t* model = eshkol_tm_model_create(2);
    const double W1[4] = { 0.45, -0.35, -0.30, 0.50 };
    const double b1[2] = { 0.05, -0.04 };
    const double W2[4] = { 0.70, -0.40, -0.35, 0.65 };
    const double b2[2] = { 0.02, 0.01 };
    CHECK(eshkol_tm_model_add_layer(model, 2, W1, b1, ESHKOL_TM_ACT_TANH, true),
          "cert model layer 1 (tanh + residual) added");
    CHECK(eshkol_tm_model_add_layer(model, 2, W2, b2, ESHKOL_TM_ACT_IDENTITY, false),
          "cert model layer 2 (identity) added");

    const eshkol_tm_interval_t input_box[2] = {
        { -0.20, 0.20 },
        { -0.15, 0.25 }
    };
    const eshkol_tm_interval_t safe_bounds[2] = {
        { -0.75, 0.75 },
        { -0.75, 0.75 }
    };

    const char program_src[] = "(define (world-step x0 x1) (ce9-certified-controller x0 x1))";
    const char context_id[] = "eshkol.ce9.world_model.safety.v2";
    const uint8_t key[] = "eshkol-ce9-signing-key-v2-2026";
    const uint64_t issued_epoch = 1700000000ULL;
    const uint64_t valid_until = 1800000000ULL;
    const uint64_t eval_epoch = 1750000000ULL;

    eshkol_cert_chain_v2_t chain;
    eshkol_cert_v2_status_t build_st = eshkol_cert_chain_v2_build(
        reinterpret_cast<const uint8_t*>(program_src), std::strlen(program_src),
        model, input_box, safe_bounds,
        context_id, issued_epoch, valid_until,
        key, sizeof(key) - 1, &chain);
    CHECK(build_st == ESHKOL_CERT_OK, "certificate v2 chain built cleanly");

    eshkol_cert_v2_status_t ver_ok = eshkol_cert_chain_v2_verify(
        &chain, context_id, eval_epoch, key, sizeof(key) - 1);
    CHECK(ver_ok == ESHKOL_CERT_OK, "baseline valid certificate v2 verifies as ESHKOL_CERT_OK");

    /* --- GUARD 1: Bit-Exact Certificate Chain Tamper Control --- */
    eshkol_cert_chain_v2_t tampered_mid = chain;
    tampered_mid.taylor_link.link_hash[15] ^= 0x01u; /* Flip 1 single bit in Taylor-model hash */
    eshkol_cert_v2_status_t st_tamper_mid = eshkol_cert_chain_v2_verify(
        &tampered_mid, context_id, eval_epoch, key, sizeof(key) - 1);

    eshkol_cert_chain_v2_t tampered_prev = chain;
    tampered_prev.taylor_link.prev_hash[0] ^= 0x80u;
    eshkol_cert_v2_status_t st_tamper_prev = eshkol_cert_chain_v2_verify(
        &tampered_prev, context_id, eval_epoch, key, sizeof(key) - 1);

    eshkol_cert_chain_v2_t tampered_bound = chain;
    tampered_bound.taylor_link.certified_bounds[0].hi += 1e-9;
    eshkol_cert_v2_status_t st_tamper_bound = eshkol_cert_chain_v2_verify(
        &tampered_bound, context_id, eval_epoch, key, sizeof(key) - 1);

    std::printf("  [Guard 1] 1-bit Taylor hash tamper status=%d (expected %d=HASH_CHAIN_BREAK), "
                "prev_hash tamper=%d, bound tamper=%d\n",
                (int)st_tamper_mid, (int)ESHKOL_CERT_ERR_HASH_CHAIN_BREAK,
                (int)st_tamper_prev, (int)st_tamper_bound);

    CHECK(st_tamper_mid == ESHKOL_CERT_ERR_HASH_CHAIN_BREAK,
          "Guard 1: 1-bit flip in intermediate Taylor hash returns ESHKOL_CERT_ERR_HASH_CHAIN_BREAK");
    CHECK(st_tamper_prev == ESHKOL_CERT_ERR_HASH_CHAIN_BREAK,
          "Guard 1: 1-bit flip in Taylor prev_hash returns ESHKOL_CERT_ERR_HASH_CHAIN_BREAK");
    CHECK(st_tamper_bound == ESHKOL_CERT_ERR_HASH_CHAIN_BREAK,
          "Guard 1: un-resigned bound perturbation returns ESHKOL_CERT_ERR_HASH_CHAIN_BREAK");

    /* --- GUARD 2: Stale/Mismatched Context & Expired Key Rejection Control --- */
    eshkol_cert_chain_v2_t wrong_ctx_chain = chain;
    CHECK(eshkol_cert_chain_v2_resign(
              &wrong_ctx_chain, "eshkol.ce9.mismatched_context.v2",
              issued_epoch, valid_until, key, sizeof(key) - 1) == ESHKOL_CERT_OK,
          "Guard 2: re-signed with mismatched context_id");
    eshkol_cert_v2_status_t st_ctx = eshkol_cert_chain_v2_verify(
        &wrong_ctx_chain, context_id, eval_epoch, key, sizeof(key) - 1);

    eshkol_cert_chain_v2_t expired_chain = chain;
    CHECK(eshkol_cert_chain_v2_resign(
              &expired_chain, context_id,
              issued_epoch, eval_epoch - 1, key, sizeof(key) - 1) == ESHKOL_CERT_OK,
          "Guard 2: re-signed with expired key epoch");
    eshkol_cert_v2_status_t st_exp = eshkol_cert_chain_v2_verify(
        &expired_chain, context_id, eval_epoch, key, sizeof(key) - 1);

    /* Postcondition violation check (re-signed with tighter safety box). */
    eshkol_cert_chain_v2_t tight_safe = chain;
    tight_safe.safety_link.safe_bounds[0] = { -0.01, 0.01 };
    CHECK(eshkol_cert_chain_v2_resign(
              &tight_safe, context_id, issued_epoch, valid_until,
              key, sizeof(key) - 1) == ESHKOL_CERT_OK,
          "postcondition control re-signed");
    eshkol_cert_v2_status_t st_post = eshkol_cert_chain_v2_verify(
        &tight_safe, context_id, eval_epoch, key, sizeof(key) - 1);

    std::printf("  [Guard 2] Mismatched context status=%d (expected %d=CONTEXT_MISMATCH), "
                "expired key status=%d (expected %d=EXPIRED_KEY), postcondition=%d\n",
                (int)st_ctx, (int)ESHKOL_CERT_ERR_CONTEXT_MISMATCH,
                (int)st_exp, (int)ESHKOL_CERT_ERR_EXPIRED_KEY,
                (int)st_post);

    CHECK(st_ctx == ESHKOL_CERT_ERR_CONTEXT_MISMATCH,
          "Guard 2: mismatched context_id returns ESHKOL_CERT_ERR_CONTEXT_MISMATCH");
    CHECK(st_exp == ESHKOL_CERT_ERR_EXPIRED_KEY,
          "Guard 2: expired key epoch returns ESHKOL_CERT_ERR_EXPIRED_KEY");
    CHECK(st_post == ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION,
          "postcondition violation returns ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION");

    /* --- GUARD 3: Real Lean 4.12.0 Kernel Export Check + Tampered Negative Control --- */
    std::vector<char> lean_buf(16384, '\0');
    size_t lean_len = 0;
    bool exp_ok = eshkol_cert_chain_export_lean4(
        &chain, context_id, eval_epoch, "EshkolCE9CertChain",
        lean_buf.data(), lean_buf.size(), &lean_len);
    CHECK(exp_ok && lean_len > 200, "Guard 3: Lean 4 derivation term exported");

    std::string lean_src(lean_buf.data(), lean_len);
    CHECK(lean_src.find("sorry") == std::string::npos, "Guard 3: exported Lean 4 has ZERO sorry");
    CHECK(lean_src.find("\naxiom ") == std::string::npos && lean_src.find("axiom ") == std::string::npos,
          "Guard 3: exported Lean 4 has ZERO custom axioms");

    if (export_lean_path && export_lean_path[0] != '\0') {
        std::ofstream out(export_lean_path, std::ios::binary | std::ios::trunc);
        out.write(lean_src.data(), static_cast<std::streamsize>(lean_src.size()));
        out.close();
    }

    const std::string lean_bin = resolve_lean4_binary();
    const std::string valid_path = temp_dir_prefix() + "eshkol_ce9_valid_cert.lean";
    const std::string bad_path = temp_dir_prefix() + "eshkol_ce9_tampered_cert.lean";

    {
        std::ofstream out(valid_path, std::ios::binary | std::ios::trunc);
        out.write(lean_src.data(), static_cast<std::streamsize>(lean_src.size()));
    }

    /* Build tampered Lean file by mutating 1 hex char in link2.prevHash. */
    std::string bad_lean = lean_src;
    size_t pos = bad_lean.find("prevHash := \"");
    CHECK(pos != std::string::npos, "Guard 3: located prevHash in exported Lean 4");
    if (pos != std::string::npos) {
        size_t hex_idx = pos + std::strlen("prevHash := \"");
        bad_lean[hex_idx] = (bad_lean[hex_idx] == '0') ? '1' : '0';
    }
    {
        std::ofstream out(bad_path, std::ios::binary | std::ios::trunc);
        out.write(bad_lean.data(), static_cast<std::streamsize>(bad_lean.size()));
    }

    std::string cmd_valid = "\"" + lean_bin + "\" \"" + valid_path + "\" > /dev/null 2>&1";
    std::string cmd_bad = "\"" + lean_bin + "\" \"" + bad_path + "\" > /dev/null 2>&1";
    int rc_valid = std::system(cmd_valid.c_str());
    int rc_bad = std::system(cmd_bad.c_str());

    std::remove(valid_path.c_str());
    std::remove(bad_path.c_str());

    std::printf("  [Guard 3] Lean 4.12.0 kernel check: valid_rc=%d (expected 0), "
                "tampered_rc=%d (expected != 0), exported_bytes=%zu\n",
                rc_valid, rc_bad, lean_len);

    CHECK(rc_valid == 0, "Guard 3: Lean 4 kernel accepted valid exported certificate chain");
    CHECK(rc_bad != 0, "Guard 3: Lean 4 kernel rejected tampered certificate chain");

    eshkol_tm_model_destroy(model);
}

/* ============================================================================
 * Part 2: Stratified Recursive IVM (Guards 4, 5, 6)
 * ============================================================================ */

void test_recursive_ivm() {
    std::printf("\n=== CE-9 Part 2: Stratified Recursive IVM (DRed Differential Closure) ===\n");

    /* --- GUARD 4: Compile-Time Cyclic Stratification Rejection --- */
    /* 4a: Ungrounded mutual cycle A <- B, B <- A without base case */
    eshkol_ivm_strat_compiler_t* sc_ungrounded = eshkol_ivm_strat_create();
    eshkol_ivm_strat_add_relation(sc_ungrounded, 1, false);
    eshkol_ivm_strat_add_relation(sc_ungrounded, 2, false);
    eshkol_ivm_strat_add_dependency(sc_ungrounded, 1, 2, false);
    eshkol_ivm_strat_add_dependency(sc_ungrounded, 2, 1, false);
    eshkol_ivm_status_t st_ungrounded = eshkol_ivm_strat_compile(sc_ungrounded);

    /* 4b: Negative cycle A <- B, B <- not A */
    eshkol_ivm_strat_compiler_t* sc_neg = eshkol_ivm_strat_create();
    eshkol_ivm_strat_add_relation(sc_neg, 10, true);
    eshkol_ivm_strat_add_relation(sc_neg, 20, true);
    eshkol_ivm_strat_add_dependency(sc_neg, 10, 20, false);
    eshkol_ivm_strat_add_dependency(sc_neg, 20, 10, true);
    eshkol_ivm_status_t st_neg = eshkol_ivm_strat_compile(sc_neg);

    /* 4c: Valid stratified recursive reachability + negated complement */
    eshkol_ivm_strat_compiler_t* sc_valid = eshkol_ivm_strat_create();
    eshkol_ivm_strat_add_relation(sc_valid, 1, true); /* edge (EDB) */
    eshkol_ivm_strat_add_relation(sc_valid, 2, true); /* reach (base + recursive) */
    eshkol_ivm_strat_add_relation(sc_valid, 3, true); /* unreachable (negated reach) */
    eshkol_ivm_strat_add_dependency(sc_valid, 2, 1, false);
    eshkol_ivm_strat_add_dependency(sc_valid, 2, 2, false);
    eshkol_ivm_strat_add_dependency(sc_valid, 3, 2, true);
    eshkol_ivm_status_t st_valid = eshkol_ivm_strat_compile(sc_valid);

    std::printf("  [Guard 4] Stratification compile: ungrounded_cycle=%d, neg_cycle=%d "
                "(expected %d=CYCLIC_STRATIFICATION), valid=%d (strata: edge=%u, reach=%u, unreach=%u)\n",
                (int)st_ungrounded, (int)st_neg, (int)ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION,
                (int)st_valid,
                eshkol_ivm_strat_level(sc_valid, 1),
                eshkol_ivm_strat_level(sc_valid, 2),
                eshkol_ivm_strat_level(sc_valid, 3));

    CHECK(st_ungrounded == ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION,
          "Guard 4: ungrounded A<->B cycle rejected with ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION");
    CHECK(st_neg == ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION,
          "Guard 4: negative cycle rejected with ESHKOL_IVM_ERR_CYCLIC_STRATIFICATION");
    CHECK(st_valid == ESHKOL_IVM_OK, "Guard 4: valid stratified recursive program compiles");
    CHECK(eshkol_ivm_strat_level(sc_valid, 3) > eshkol_ivm_strat_level(sc_valid, 2),
          "Guard 4: negated consumer assigned strictly higher stratum");

    eshkol_ivm_strat_destroy(sc_ungrounded);
    eshkol_ivm_strat_destroy(sc_neg);
    eshkol_ivm_strat_destroy(sc_valid);

    /* --- GUARD 5: 100-Mutation Differential Oracle (including Cycles & Diamonds) --- */
    eshkol_recursive_ivm_t* ivm = eshkol_recursive_ivm_create();
    std::vector<std::pair<uint64_t, uint64_t>> live_edges;

    /* Seed explicit directed cycle (0->1->2->0) + diamond (3->4->6, 3->5->6) + bridge (2->3). */
    const std::pair<uint64_t, uint64_t> seed_edges[] = {
        {0, 1}, {1, 2}, {2, 0},
        {3, 4}, {4, 6}, {3, 5}, {5, 6},
        {2, 3}
    };
    for (const auto& e : seed_edges) {
        eshkol_recursive_ivm_insert_edge(ivm, e.first, e.second);
        live_edges.push_back(e);
    }

    CHECK(eshkol_recursive_ivm_reaches(ivm, 0, 0), "seeded cycle 0->+0 reachable");
    CHECK(eshkol_recursive_ivm_reaches(ivm, 0, 6), "seeded cycle+diamond 0->+6 reachable");

    SplitMix64 rng(0xCE910055ULL);
    int divergences = 0;
    const int N_MUTATIONS = 100;

    for (int m = 0; m < N_MUTATIONS; ++m) {
        bool do_delete = !live_edges.empty() && ((rng.next_u64() % 100) < 48);
        if (do_delete) {
            size_t idx = static_cast<size_t>(rng.next_u64() % live_edges.size());
            auto e = live_edges[idx];
            live_edges.erase(live_edges.begin() + static_cast<std::ptrdiff_t>(idx));
            eshkol_recursive_ivm_delete_edge(ivm, e.first, e.second);
        } else {
            uint64_t u = rng.next_u64() % 12;
            uint64_t v = rng.next_u64() % 12;
            eshkol_recursive_ivm_insert_edge(ivm, u, v);
            live_edges.emplace_back(u, v);
        }

        size_t oracle_pairs = 0;
        uint64_t h_oracle = eshkol_recursive_ivm_oracle_hash(ivm, &oracle_pairs);
        uint64_t h_inc = eshkol_recursive_ivm_derived_hash(ivm);
        size_t inc_pairs = eshkol_recursive_ivm_derived_pair_count(ivm);

        if (h_inc != h_oracle || inc_pairs != oracle_pairs) {
            ++divergences;
        }
    }

    std::printf("  [Guard 5] 100-mutation differential oracle (cycles + diamonds): "
                "divergences=%d/100, active_edges=%zu, derived_pairs=%zu\n",
                divergences,
                eshkol_recursive_ivm_active_edge_count(ivm),
                eshkol_recursive_ivm_derived_pair_count(ivm));
    CHECK(divergences == 0, "Guard 5: 0/100 incremental vs full BFS oracle divergences");

    /* --- GUARD 6: Zero-Zombie Cleanup Control --- */
    for (const auto& e : live_edges) {
        CHECK(eshkol_recursive_ivm_delete_edge(ivm, e.first, e.second),
              "Guard 6: delete remaining live edge");
    }
    live_edges.clear();

    const size_t rem_edges = eshkol_recursive_ivm_active_edge_count(ivm);
    const size_t rem_nodes = eshkol_recursive_ivm_active_node_count(ivm);
    const size_t rem_pairs = eshkol_recursive_ivm_derived_pair_count(ivm);
    size_t oracle_rem = 999;
    uint64_t h_inc_empty = eshkol_recursive_ivm_derived_hash(ivm);
    uint64_t h_ora_empty = eshkol_recursive_ivm_oracle_hash(ivm, &oracle_rem);

    std::printf("  [Guard 6] Full deletion zero-zombie check: active_edges=%zu, "
                "active_nodes=%zu, derived_pairs=%zu, oracle_pairs=%zu\n",
                rem_edges, rem_nodes, rem_pairs, oracle_rem);

    CHECK(rem_edges == 0, "Guard 6: active_edge_count strictly == 0");
    CHECK(rem_nodes == 0, "Guard 6: active_node_count strictly == 0");
    CHECK(rem_pairs == 0, "Guard 6: derived_pair_count strictly == 0");
    CHECK(h_inc_empty == h_ora_empty && oracle_rem == 0,
          "Guard 6: empty IVM hash matches empty oracle");

    eshkol_recursive_ivm_destroy(ivm);
}

/* ============================================================================
 * Part 3: ADR-0005 Differentiable Program-Capsule Foundations (Guards 7, 8, 9)
 * ============================================================================ */

void test_program_capsule() {
    std::printf("\n=== CE-9 Part 3: ADR-0005 Differentiable Program-Capsule Foundations ===\n");

    /* --- GUARD 7: Discontinuity Control (`if x > 0 then 1 else 0`) --- */
    eshkol_sym_rule_t* step_rule = eshkol_sym_rule_create(1);
    const double wg1[1] = { 1.0 };
    const double wt1[1] = { 0.0 };
    const double we1[1] = { 0.0 };
    CHECK(eshkol_sym_rule_add_step_branch(step_rule, wg1, 0.0, wt1, 1.0, we1, 0.0),
          "Guard 7: step rule `if x > 0 then 1 else 0` constructed");

    const double xp = 0.5, xm = -0.5;
    CHECK(eshkol_sym_rule_eval(step_rule, &xp) == 1.0, "symbolic step(0.5) == 1.0");
    CHECK(eshkol_sym_rule_eval(step_rule, &xm) == 0.0, "symbolic step(-0.5) == 0.0");

    eshkol_program_capsule_t* strict_cap = nullptr;
    eshkol_capsule_status_t st_strict = eshkol_capsule_compile(
        step_rule, ESHKOL_CAPSULE_MODE_STRICT, 0.0, &strict_cap);
    CHECK(st_strict == ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE && strict_cap == nullptr,
          "Guard 7: STRICT mode rejects hard-step rule with ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE");

    const double tau = 0.25;
    eshkol_program_capsule_t* smooth_cap = nullptr;
    eshkol_capsule_status_t st_smooth = eshkol_capsule_compile(
        step_rule, ESHKOL_CAPSULE_MODE_SMOOTH, tau, &smooth_cap);
    CHECK(st_smooth == ESHKOL_CAPSULE_OK && smooth_cap != nullptr,
          "Guard 7: SMOOTH mode compiles relaxed step capsule");

    const double test_xs[5] = { -0.8, -0.2, 0.0, 0.15, 0.6 };
    double max_step_grad_err = 0.0;
    uint64_t step_fd_total = 0;
    const double eps = 1e-6;
    for (double xv : test_xs) {
        double val = 0.0, gx = 0.0;
        uint64_t fde = 999;
        CHECK(eshkol_capsule_eval_value_grad(smooth_cap, &xv, &val, &gx, nullptr, &fde),
              "Guard 7: smooth step eval succeeded");
        step_fd_total += fde;

        double vp = 0.0, vm = 0.0;
        double x_plus = xv + eps, x_minus = xv - eps;
        eshkol_capsule_eval_value_grad(smooth_cap, &x_plus, &vp, nullptr, nullptr, nullptr);
        eshkol_capsule_eval_value_grad(smooth_cap, &x_minus, &vm, nullptr, nullptr, nullptr);
        double fd_gx = (vp - vm) / (2.0 * eps);

        double sig = 1.0 / (1.0 + std::exp(-xv / tau));
        double exact_gx = (sig * (1.0 - sig)) / tau;
        double err = std::max(std::fabs(gx - exact_gx), std::fabs(gx - fd_gx));
        if (err > max_step_grad_err) max_step_grad_err = err;
    }

    std::printf("  [Guard 7] Discontinuity control: strict_status=%d (expected %d=NON_DIFF), "
                "smooth_status=%d, max_grad_err=%.3e (< 1e-6), fd_evals=%llu\n",
                (int)st_strict, (int)ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE,
                (int)st_smooth, max_step_grad_err, (unsigned long long)step_fd_total);
    CHECK(max_step_grad_err < 1e-6, "Guard 7: smooth step gradient error < 1e-6");
    CHECK(step_fd_total == 0, "Guard 7: zero internal FD evals");

    eshkol_capsule_destroy(smooth_cap);
    eshkol_sym_rule_destroy(step_rule);

    /* --- GUARD 8: Multi-Branch External FD Gradient Control --- */
    eshkol_sym_rule_t* multi_rule = eshkol_sym_rule_create(3);
    const double b1_wg[3] = {  0.8, -0.5,  0.3 };
    const double b1_wt[3] = {  1.2,  0.4, -0.7 };
    const double b1_we[3] = { -0.6,  0.9,  0.2 };
    CHECK(eshkol_sym_rule_add_step_branch(multi_rule, b1_wg, 0.1, b1_wt, 0.5, b1_we, -0.3),
          "multi-branch step 1 added");

    const double b2_wg[3] = { -0.4,  0.7,  0.6 };
    const double b2_wt[3] = {  0.3, -1.1,  0.8 };
    const double b2_we[3] = {  0.5,  0.2, -0.4 };
    CHECK(eshkol_sym_rule_add_step_branch(multi_rule, b2_wg, -0.15, b2_wt, -0.2, b2_we, 0.7),
          "multi-branch step 2 added");

    const double b3_w[3] = { 0.5, -0.6, 0.4 };
    CHECK(eshkol_sym_rule_add_smooth_branch(multi_rule, b3_w, 0.05, 1.3, -0.1),
          "multi-branch smooth tanh 3 added");

    eshkol_program_capsule_t* multi_cap = nullptr;
    CHECK(eshkol_capsule_compile(multi_rule, ESHKOL_CAPSULE_MODE_SMOOTH, 0.4, &multi_cap) == ESHKOL_CAPSULE_OK,
          "Guard 8: multi-branch smooth capsule compiled");

    const size_t n_params = eshkol_capsule_param_count(multi_cap);
    CHECK(n_params == 30, "Guard 8: multi-branch capsule has 30 parameters (12+12+6)");

    SplitMix64 rng(0xCA950009ULL);
    double max_in_grad_err = 0.0;
    double max_param_grad_err = 0.0;
    uint64_t total_cap_fd = 0;

    std::vector<double> grad_p(n_params, 0.0);
    for (int pt = 0; pt < 10; ++pt) {
        double x[3] = {
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0)
        };
        double val = 0.0;
        double grad_x[3] = { 0.0, 0.0, 0.0 };
        uint64_t fde = 999;
        CHECK(eshkol_capsule_eval_value_grad(multi_cap, x, &val, grad_x, grad_p.data(), &fde),
              "Guard 8: eval value+grad succeeded");
        total_cap_fd += fde;

        /* External central FD w.r.t. inputs */
        for (int i = 0; i < 3; ++i) {
            double xp_arr[3] = { x[0], x[1], x[2] };
            double xm_arr[3] = { x[0], x[1], x[2] };
            xp_arr[i] += eps;
            xm_arr[i] -= eps;
            double vp = 0.0, vm = 0.0;
            eshkol_capsule_eval_value_grad(multi_cap, xp_arr, &vp, nullptr, nullptr, nullptr);
            eshkol_capsule_eval_value_grad(multi_cap, xm_arr, &vm, nullptr, nullptr, nullptr);
            double fd = (vp - vm) / (2.0 * eps);
            double err = std::fabs(grad_x[i] - fd);
            if (err > max_in_grad_err) max_in_grad_err = err;
        }

        /* External central FD w.r.t. every parameter */
        for (size_t k = 0; k < n_params; ++k) {
            double orig = 0.0;
            eshkol_capsule_get_param(multi_cap, k, &orig);
            double vp = 0.0, vm = 0.0;
            eshkol_capsule_set_param(multi_cap, k, orig + eps);
            eshkol_capsule_eval_value_grad(multi_cap, x, &vp, nullptr, nullptr, nullptr);
            eshkol_capsule_set_param(multi_cap, k, orig - eps);
            eshkol_capsule_eval_value_grad(multi_cap, x, &vm, nullptr, nullptr, nullptr);
            eshkol_capsule_set_param(multi_cap, k, orig);
            double fd = (vp - vm) / (2.0 * eps);
            double err = std::fabs(grad_p[k] - fd);
            if (err > max_param_grad_err) max_param_grad_err = err;
        }
    }

    std::printf("  [Guard 8] Multi-branch external FD check (10 pts x 33 partials): "
                "max_in_grad_err=%.3e, max_param_grad_err=%.3e (< 1e-5), fd_evals=%llu\n",
                max_in_grad_err, max_param_grad_err, (unsigned long long)total_cap_fd);

    CHECK(max_in_grad_err < 1e-5, "Guard 8: input analytical gradients match external FD < 1e-5");
    CHECK(max_param_grad_err < 1e-5, "Guard 8: parameter analytical gradients match external FD < 1e-5");
    CHECK(total_cap_fd == 0, "Guard 8: zero internal FD evals inside capsule engine");

    /* Pure-smooth rule compiles cleanly in STRICT mode as well. */
    eshkol_sym_rule_t* pure_smooth = eshkol_sym_rule_create(3);
    eshkol_sym_rule_add_smooth_branch(pure_smooth, b3_w, 0.05, 1.3, -0.1);
    eshkol_program_capsule_t* strict_smooth_cap = nullptr;
    CHECK(eshkol_capsule_compile(pure_smooth, ESHKOL_CAPSULE_MODE_STRICT, 0.0, &strict_smooth_cap) == ESHKOL_CAPSULE_OK,
          "pure-smooth rule compiles in STRICT mode");
    eshkol_capsule_destroy(strict_smooth_cap);
    eshkol_sym_rule_destroy(pure_smooth);

    /* --- GUARD 9: Program-Capsule Weight Isolation Control --- */
    std::vector<std::vector<double>> probe_pts(20, std::vector<double>(3));
    std::vector<double> sym_before(20, 0.0);
    std::vector<double> cap_before(20, 0.0);
    for (int i = 0; i < 20; ++i) {
        probe_pts[i][0] = rng.next_range(-1.5, 1.5);
        probe_pts[i][1] = rng.next_range(-1.5, 1.5);
        probe_pts[i][2] = rng.next_range(-1.5, 1.5);
        sym_before[i] = eshkol_sym_rule_eval(multi_rule, probe_pts[i].data());
        eshkol_capsule_eval_value_grad(multi_cap, probe_pts[i].data(), &cap_before[i], nullptr, nullptr, nullptr);
    }

    /* Arbitrarily mutate EVERY weight in the compiled capsule. */
    for (size_t k = 0; k < n_params; ++k) {
        double orig = 0.0;
        eshkol_capsule_get_param(multi_cap, k, &orig);
        CHECK(eshkol_capsule_set_param(multi_cap, k, orig * -3.5 + 1.75),
              "Guard 9: mutated compiled capsule weight");
    }

    double max_sym_drift = 0.0;
    double max_cap_delta = 0.0;
    for (int i = 0; i < 20; ++i) {
        double sym_after = eshkol_sym_rule_eval(multi_rule, probe_pts[i].data());
        double cap_after = 0.0;
        eshkol_capsule_eval_value_grad(multi_cap, probe_pts[i].data(), &cap_after, nullptr, nullptr, nullptr);
        double sd = std::fabs(sym_after - sym_before[i]);
        double cd = std::fabs(cap_after - cap_before[i]);
        if (sd > max_sym_drift) max_sym_drift = sd;
        if (cd > max_cap_delta) max_cap_delta = cd;
    }

    /* Unverified relink MUST be rejected and keep symbolic rule untouched. */
    eshkol_capsule_status_t st_unver = eshkol_capsule_relink_to_rule(multi_cap, multi_rule, false);
    double sym_after_unver_drift = 0.0;
    for (int i = 0; i < 20; ++i) {
        double s = eshkol_sym_rule_eval(multi_rule, probe_pts[i].data());
        sym_after_unver_drift = std::max(sym_after_unver_drift, std::fabs(s - sym_before[i]));
    }

    /* Explicit verified relink updates symbolic rule cleanly. */
    eshkol_capsule_status_t st_ver_relink = eshkol_capsule_relink_to_rule(multi_cap, multi_rule, true);
    double max_sym_change_after_relink = 0.0;
    for (int i = 0; i < 20; ++i) {
        double s = eshkol_sym_rule_eval(multi_rule, probe_pts[i].data());
        max_sym_change_after_relink = std::max(max_sym_change_after_relink, std::fabs(s - sym_before[i]));
    }

    std::printf("  [Guard 9] Weight isolation: max_sym_drift=%.3e (must be 0.0), "
                "max_cap_delta=%.3f (> 1.0), unverified_relink=%d, verified_relink=%d (sym_change=%.3f)\n",
                max_sym_drift, max_cap_delta, (int)st_unver, (int)st_ver_relink, max_sym_change_after_relink);

    CHECK(max_sym_drift == 0.0, "Guard 9: mutating capsule weights produces 0.0 drift in symbolic rule");
    CHECK(max_cap_delta > 1.0, "Guard 9: mutated capsule outputs actually changed");
    CHECK(st_unver == ESHKOL_CAPSULE_ERR_UNVERIFIED_RELINK && sym_after_unver_drift == 0.0,
          "Guard 9: unverified relink rejected with ESHKOL_CAPSULE_ERR_UNVERIFIED_RELINK");
    CHECK(st_ver_relink == ESHKOL_CAPSULE_OK && max_sym_change_after_relink > 1.0,
          "Guard 9: explicit verified relink updates symbolic rule cleanly");

    eshkol_capsule_destroy(multi_cap);
    eshkol_sym_rule_destroy(multi_rule);
}

} /* namespace */

int main(int argc, char** argv) {
    const char* export_lean_path = nullptr;
    if (argc >= 3 && std::strcmp(argv[1], "--export-lean") == 0) {
        export_lean_path = argv[2];
    }

    test_cert_chain_v2_and_lean4(export_lean_path);
    test_recursive_ivm();
    test_program_capsule();

    std::printf("\nCE-9 Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
