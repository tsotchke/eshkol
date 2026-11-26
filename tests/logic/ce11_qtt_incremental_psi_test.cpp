/*
 * CE-11 Native Verification Suite:
 *   1. Self-Referential Divergence (Lob Violation Control)
 *   2. Incremental State Corruption (QTT Linear Multiplicity Violation Control)
 *   3. Cubical Path Tampering (Reflective Safety Loop Control)
 *   + Positive QTT IncrementalizePass DBSP & Acyclic Lob Reflective Controls
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/qtt_incremental_psi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

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

void test_qtt_incrementalize_pass() {
    std::printf("\n=== CE-11 Part 1: QTT (0, 1, omega) Enforcer & IncrementalizePass DBSP ===\n");

    /* Positive control: compile valid (0, 1, omega) circuit and verify O(|Delta|) exactness. */
    eshkol_qtt_dbsp_circuit_t* c = eshkol_qtt_dbsp_create();
    CHECK(c != nullptr, "QTT DBSP circuit created");
    CHECK(eshkol_qtt_dbsp_add_resource(c, 0, ESHKOL_QTT_ZERO), "proof-only QTT_ZERO resource added");
    CHECK(eshkol_qtt_dbsp_add_resource(c, 1, ESHKOL_QTT_LINEAR_ONE), "linear QTT_LINEAR_ONE resource added");
    CHECK(eshkol_qtt_dbsp_add_resource(c, 2, ESHKOL_QTT_UNRESTRICTED), "unrestricted QTT_UNRESTRICTED resource added");

    CHECK(eshkol_qtt_dbsp_add_term(c, 1, 5), "linear resource used exactly once");
    CHECK(eshkol_qtt_dbsp_add_term(c, 2, 3), "unrestricted resource term 1 added");
    CHECK(eshkol_qtt_dbsp_add_term(c, 2, 7), "unrestricted resource term 2 added (omega allowed)");

    CHECK(eshkol_qtt_incrementalize_compile(c) == ESHKOL_CE11_OK,
          "valid (0, 1, omega) circuit compiled via IncrementalizePass");

    const eshkol_qtt_delta_update_t batch1[2] = {
        { 1, 1001ULL, 4 },  /* linear token 1001, delta +4 -> +20 */
        { 2, 0ULL,    2 }   /* omega resource, delta +2 -> +20 */
    };
    int64_t d_out = 0, inc_total = 0;
    CHECK(eshkol_qtt_dbsp_apply_delta(c, batch1, 2, &d_out, &inc_total) == ESHKOL_CE11_OK,
          "valid linear+omega delta batch applied");
    CHECK(d_out == 40 && inc_total == 40 && eshkol_qtt_dbsp_eval_full(c) == 40,
          "incremental O(|Delta|) total matches from-scratch evaluation (40)");

    /* --- GUARD 2: Incremental State Corruption (Linear Multiplicity Violation) --- */
    /* 2a: Runtime duplicate consumption of LINEAR_ONE resource within a single delta batch. */
    const eshkol_qtt_delta_update_t dup_batch[2] = {
        { 1, 2001ULL, 3 },
        { 1, 2002ULL, 9 }   /* Duplicate consumption of LINEAR_ONE resource 1! */
    };
    eshkol_ce11_status_t st_dup_res = eshkol_qtt_dbsp_apply_delta(c, dup_batch, 2, &d_out, &inc_total);
    CHECK(st_dup_res == ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE,
          "Guard 2a: duplicate consumption of LINEAR_ONE resource fails closed with ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE");
    CHECK(eshkol_qtt_dbsp_current_incremental(c) == 40 && eshkol_qtt_dbsp_eval_full(c) == 40,
          "Guard 2a: failed batch leaves circuit state completely uncorrupted (40)");

    /* 2b: Runtime replay / re-consumption of an already-consumed linear token 1001. */
    const eshkol_qtt_delta_update_t replay_tok[1] = {
        { 1, 1001ULL, 1 }
    };
    eshkol_ce11_status_t st_dup_tok = eshkol_qtt_dbsp_apply_delta(c, replay_tok, 1, &d_out, &inc_total);
    CHECK(st_dup_tok == ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE,
          "Guard 2b: re-consuming linear token 1001 fails closed with ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE");
    CHECK(eshkol_qtt_dbsp_current_incremental(c) == 40 && eshkol_qtt_dbsp_eval_full(c) == 40,
          "Guard 2b: circuit state strictly uncorrupted after token replay rejection");

    /* 2c: Static compile-time duplicate use of LINEAR_ONE resource in IncrementalizePass. */
    eshkol_qtt_dbsp_circuit_t* c_bad_lin = eshkol_qtt_dbsp_create();
    eshkol_qtt_dbsp_add_resource(c_bad_lin, 1, ESHKOL_QTT_LINEAR_ONE);
    eshkol_qtt_dbsp_add_term(c_bad_lin, 1, 2);
    eshkol_qtt_dbsp_add_term(c_bad_lin, 1, 4); /* Second static use of linear resource! */
    eshkol_ce11_status_t st_static_dup = eshkol_qtt_incrementalize_compile(c_bad_lin);
    CHECK(st_static_dup == ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE,
          "Guard 2c: static IncrementalizePass rejects duplicate LINEAR_ONE use");

    /* 2d: Erased QTT_ZERO resource used at runtime rejected cleanly. */
    eshkol_qtt_dbsp_circuit_t* c_bad_zero = eshkol_qtt_dbsp_create();
    eshkol_qtt_dbsp_add_resource(c_bad_zero, 0, ESHKOL_QTT_ZERO);
    eshkol_qtt_dbsp_add_term(c_bad_zero, 0, 1);
    eshkol_ce11_status_t st_zero_use = eshkol_qtt_incrementalize_compile(c_bad_zero);
    CHECK(st_zero_use == ESHKOL_QTT_ERR_ERASED_RUNTIME_USE,
          "Guard 2d: QTT_ZERO runtime use rejected with ESHKOL_QTT_ERR_ERASED_RUNTIME_USE");

    std::printf("  [Guard 2] Linear duplicate controls: runtime_dup_res=%d, runtime_dup_tok=%d, "
                "static_dup=%d (all expected %d=LINEAR_DUPLICATE_USE), erased_zero=%d\n",
                (int)st_dup_res, (int)st_dup_tok, (int)st_static_dup,
                (int)ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE, (int)st_zero_use);

    eshkol_qtt_dbsp_destroy(c);
    eshkol_qtt_dbsp_destroy(c_bad_lin);
    eshkol_qtt_dbsp_destroy(c_bad_zero);
}

void test_lob_cubical_psi_verifier() {
    std::printf("\n=== CE-11 Part 2: Lob / Cubical Reflective psi_program Verifier ===\n");

    const char guard_spec[] = "(define (safety-guard cert) (verify-cert-chain-v2 cert))";
    const char bypass_spec[] = "(define (safety-guard cert) #t)";

    eshkol_psi_verifier_t* v = eshkol_psi_verifier_create(
        reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec),
        /*initial_state_id=*/100ULL, /*initial_lob_rank=*/10U);
    CHECK(v != nullptr, "Lob/Cubical psi verifier created");

    /* Positive control: valid rank-decreasing acyclic cubical self-modification path 100 -> 101 -> 102. */
    eshkol_psi_transition_t t1, t2;
    CHECK(eshkol_psi_build_transition(v, 100ULL, 101ULL, 10U, 9U,
                                      ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                      reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec),
                                      &t1),
          "valid cubical transition 100->101 built");
    CHECK(eshkol_psi_verify_and_apply(v, &t1) == ESHKOL_CE11_OK,
          "valid Lob transition 100->101 (rank 10->9) accepted");

    CHECK(eshkol_psi_build_transition(v, 101ULL, 102ULL, 9U, 8U,
                                      ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                      reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec),
                                      &t2),
          "valid cubical transition 101->102 built");
    CHECK(eshkol_psi_verify_and_apply(v, &t2) == ESHKOL_CE11_OK,
          "valid Lob transition 101->102 (rank 9->8) accepted");
    CHECK(eshkol_psi_current_state(v) == 102ULL && eshkol_psi_current_lob_rank(v) == 8U &&
          eshkol_psi_verified_edge_count(v) == 2,
          "verifier state advanced cleanly to state=102, rank=8, edges=2");

    /* --- GUARD 1: Self-Referential Divergence (Lob Violation) --- */
    /* 1a: psi_program attempts to rewrite its own guard logic to unconditionally return #t. */
    eshkol_psi_transition_t lob_bypass;
    CHECK(eshkol_psi_build_transition(v, 102ULL, 103ULL, 8U, 7U,
                                      ESHKOL_PSI_GUARD_CONST_TRUE,
                                      reinterpret_cast<const uint8_t*>(bypass_spec), std::strlen(bypass_spec),
                                      &lob_bypass),
          "Guard 1a: adversarial Lob bypass transition built");
    eshkol_ce11_status_t st_lob_bypass = eshkol_psi_verify_and_apply(v, &lob_bypass);

    /* 1b: Subtle guard tamper (claims VERIFIED_PROOF kind, but modified guard_spec bytes). */
    eshkol_psi_transition_t lob_tamper;
    CHECK(eshkol_psi_build_transition(v, 102ULL, 104ULL, 8U, 7U,
                                      ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                      reinterpret_cast<const uint8_t*>(bypass_spec), std::strlen(bypass_spec),
                                      &lob_tamper),
          "Guard 1b: guard-digest tamper transition built");
    eshkol_ce11_status_t st_lob_tamper = eshkol_psi_verify_and_apply(v, &lob_tamper);

    /* 1c: Non-decreasing Lob rank (rank 8 -> 8). */
    eshkol_psi_transition_t lob_rank;
    CHECK(eshkol_psi_build_transition(v, 102ULL, 105ULL, 8U, 8U,
                                      ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                      reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec),
                                      &lob_rank),
          "Guard 1c: non-decreasing Lob rank transition built");
    eshkol_ce11_status_t st_lob_rank = eshkol_psi_verify_and_apply(v, &lob_rank);

    std::printf("  [Guard 1] Lob violation controls: const_true_bypass=%d, guard_tamper=%d, "
                "non_decreasing_rank=%d (all expected %d=LOB_VIOLATION)\n",
                (int)st_lob_bypass, (int)st_lob_tamper, (int)st_lob_rank,
                (int)ESHKOL_PSI_ERR_LOB_VIOLATION);

    CHECK(st_lob_bypass == ESHKOL_PSI_ERR_LOB_VIOLATION,
          "Guard 1a: unconditional-true guard rewrite rejected with ESHKOL_PSI_ERR_LOB_VIOLATION");
    CHECK(st_lob_tamper == ESHKOL_PSI_ERR_LOB_VIOLATION,
          "Guard 1b: guard digest modification rejected with ESHKOL_PSI_ERR_LOB_VIOLATION");
    CHECK(st_lob_rank == ESHKOL_PSI_ERR_LOB_VIOLATION,
          "Guard 1c: non-decreasing Lob rank rejected with ESHKOL_PSI_ERR_LOB_VIOLATION");
    CHECK(eshkol_psi_current_state(v) == 102ULL && eshkol_psi_verified_edge_count(v) == 2,
          "Guard 1: rejected Lob violations left verifier state untouched");

    /* --- GUARD 3: Cubical Path Tampering (Reflective Safety Loop) --- */
    /* 3a: Modify cubical path target back to ancestor 100 (forming cycle 100->101->102->100). */
    eshkol_psi_transition_t loop_edge;
    CHECK(eshkol_psi_build_transition(v, 102ULL, 100ULL, 8U, 7U,
                                      ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                      reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec),
                                      &loop_edge),
          "Guard 3a: cyclic cubical back-edge 102->100 built");
    eshkol_ce11_status_t st_loop_apply = eshkol_psi_verify_and_apply(v, &loop_edge);

    /* 3b: Multi-step tampered cubical path sequence creating a non-trivial loop 102->200->201->200. */
    eshkol_psi_transition_t bad_path[3];
    eshkol_psi_build_transition(v, 102ULL, 200ULL, 8U, 7U, ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec), &bad_path[0]);
    eshkol_psi_build_transition(v, 200ULL, 201ULL, 7U, 6U, ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec), &bad_path[1]);
    eshkol_psi_build_transition(v, 201ULL, 200ULL, 6U, 5U, ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec), &bad_path[2]);
    eshkol_ce11_status_t st_loop_seq = eshkol_psi_verify_cubical_path(v, bad_path, 3);

    /* 3c: Tampered cubical path witness bytes on an otherwise acyclic transition. */
    eshkol_psi_transition_t bad_witness;
    eshkol_psi_build_transition(v, 102ULL, 300ULL, 8U, 7U, ESHKOL_PSI_GUARD_VERIFIED_PROOF,
                                reinterpret_cast<const uint8_t*>(guard_spec), std::strlen(guard_spec), &bad_witness);
    bad_witness.path_witness_sha256[7] ^= 0x01u;
    eshkol_ce11_status_t st_bad_wit = eshkol_psi_verify_and_apply(v, &bad_witness);

    std::printf("  [Guard 3] Cubical path loop controls: back_edge=%d, multi_step_loop=%d, "
                "tampered_witness=%d (all expected %d=CUBICAL_PATH_LOOP)\n",
                (int)st_loop_apply, (int)st_loop_seq, (int)st_bad_wit,
                (int)ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP);

    CHECK(st_loop_apply == ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP,
          "Guard 3a: back-edge 102->100 rejected with ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP");
    CHECK(st_loop_seq == ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP,
          "Guard 3b: non-trivial loop 200->201->200 rejected with ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP");
    CHECK(st_bad_wit == ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP,
          "Guard 3c: tampered cubical path witness rejected with ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP");
    CHECK(eshkol_psi_current_state(v) == 102ULL && eshkol_psi_verified_edge_count(v) == 2,
          "Guard 3: rejected loops left verifier graph strictly acyclic");

    eshkol_psi_verifier_destroy(v);
}

} /* namespace */

int main() {
    test_qtt_incrementalize_pass();
    test_lob_cubical_psi_verifier();

    std::printf("\nCE-11 Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
