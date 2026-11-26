/*
 * Quantitative Multiplicity (0, 1, omega) Enforcer, IncrementalizePass DBSP
 * Compiler, and Lob/Cubical Reflective psi_program Self-Modification Verifier
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_QTT_INCREMENTAL_PSI_H
#define ESHKOL_CORE_QTT_INCREMENTAL_PSI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_qtt_multiplicity {
    ESHKOL_QTT_ZERO         = 0, /* Compile-time proof/erased only (0 runtime uses) */
    ESHKOL_QTT_LINEAR_ONE   = 1, /* Strictly linear cognitive resource (at most 1 use) */
    ESHKOL_QTT_UNRESTRICTED = 2  /* Unrestricted omega resource */
} eshkol_qtt_multiplicity_t;

typedef enum eshkol_ce11_status {
    ESHKOL_CE11_OK                      = 0,
    ESHKOL_CE11_ERR_INVALID_ARG         = 1,
    ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE = 2,
    ESHKOL_QTT_ERR_ERASED_RUNTIME_USE   = 3,
    ESHKOL_PSI_ERR_LOB_VIOLATION        = 4,
    ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP    = 5
} eshkol_ce11_status_t;

typedef enum eshkol_psi_guard_kind {
    ESHKOL_PSI_GUARD_VERIFIED_PROOF = 0, /* Preserves safety proof obligation */
    ESHKOL_PSI_GUARD_CONST_TRUE     = 1  /* Unconditional true bypass (Lob violation!) */
} eshkol_psi_guard_kind_t;

/* ============================================================================
 * Part 1: QTT (0, 1, omega) + IncrementalizePass DBSP Circuit
 * ============================================================================ */

typedef struct eshkol_qtt_dbsp_circuit eshkol_qtt_dbsp_circuit_t;

typedef struct eshkol_qtt_delta_update {
    uint32_t resource_id;
    uint64_t linear_token_id; /* Unique token ID for LINEAR_ONE consumption tracking */
    int64_t delta_value;
} eshkol_qtt_delta_update_t;

eshkol_qtt_dbsp_circuit_t* eshkol_qtt_dbsp_create(void);
void eshkol_qtt_dbsp_destroy(eshkol_qtt_dbsp_circuit_t* circuit);

bool eshkol_qtt_dbsp_add_resource(eshkol_qtt_dbsp_circuit_t* circuit,
                                  uint32_t resource_id,
                                  eshkol_qtt_multiplicity_t mult);

/* Add declarative rule term: output += coeff * value(resource_id). */
bool eshkol_qtt_dbsp_add_term(eshkol_qtt_dbsp_circuit_t* circuit,
                              uint32_t resource_id,
                              int64_t coeff);

/*
 * Run IncrementalizePass compiler: verifies QTT (0, 1, omega) static usage
 * constraints and compiles declarative terms into an O(|Delta|) DBSP circuit.
 */
eshkol_ce11_status_t eshkol_qtt_incrementalize_compile(eshkol_qtt_dbsp_circuit_t* circuit);

/*
 * Apply an O(|Delta|) incremental batch while strictly enforcing runtime linear
 * multiplicity consumption. Duplicate consumption of a LINEAR_ONE resource or
 * token fails closed with ESHKOL_QTT_ERR_LINEAR_DUPLICATE_USE without mutating
 * circuit state.
 */
eshkol_ce11_status_t eshkol_qtt_dbsp_apply_delta(
    eshkol_qtt_dbsp_circuit_t* circuit,
    const eshkol_qtt_delta_update_t* updates,
    size_t n_updates,
    int64_t* out_delta_output,
    int64_t* out_incremental_total);

/* Independent full from-scratch evaluation for differential verification. */
int64_t eshkol_qtt_dbsp_eval_full(const eshkol_qtt_dbsp_circuit_t* circuit);
int64_t eshkol_qtt_dbsp_current_incremental(const eshkol_qtt_dbsp_circuit_t* circuit);

/* ============================================================================
 * Part 2: Lob / Cubical Reflective psi_program Self-Modification Verifier
 * ============================================================================ */

typedef struct eshkol_psi_verifier eshkol_psi_verifier_t;

typedef struct eshkol_psi_transition {
    uint64_t from_state_id;
    uint64_t to_state_id;
    uint32_t from_lob_rank;
    uint32_t to_lob_rank;
    eshkol_psi_guard_kind_t target_guard_kind;
    uint8_t guard_digest_before[32];
    uint8_t guard_digest_after[32];
    uint8_t path_witness_sha256[32];
} eshkol_psi_transition_t;

/* Create a reflective verifier anchored to an immutable safety guard specification. */
eshkol_psi_verifier_t* eshkol_psi_verifier_create(const uint8_t* immutable_guard_spec,
                                                  size_t spec_len,
                                                  uint64_t initial_state_id,
                                                  uint32_t initial_lob_rank);
void eshkol_psi_verifier_destroy(eshkol_psi_verifier_t* verifier);

/* Helper to build a well-formed cubical transition witness. */
bool eshkol_psi_build_transition(const eshkol_psi_verifier_t* verifier,
                                 uint64_t from_state_id,
                                 uint64_t to_state_id,
                                 uint32_t from_lob_rank,
                                 uint32_t to_lob_rank,
                                 eshkol_psi_guard_kind_t target_guard_kind,
                                 const uint8_t* target_guard_spec,
                                 size_t target_guard_len,
                                 eshkol_psi_transition_t* out_tr);

/*
 * Verify and apply a single cubical self-modification path transition:
 *   - Rejects Lob safety bypass / non-decreasing rank / guard tamper with
 *     ESHKOL_PSI_ERR_LOB_VIOLATION.
 *   - Rejects any self-loop or cycle in the cubical self-modification graph
 *     with ESHKOL_PSI_ERR_CUBICAL_PATH_LOOP.
 */
eshkol_ce11_status_t eshkol_psi_verify_and_apply(eshkol_psi_verifier_t* verifier,
                                                 const eshkol_psi_transition_t* tr);

/* Validate an entire candidate cubical path sequence for acyclicity & Lob soundness. */
eshkol_ce11_status_t eshkol_psi_verify_cubical_path(const eshkol_psi_verifier_t* verifier,
                                                    const eshkol_psi_transition_t* path,
                                                    size_t path_len);

uint64_t eshkol_psi_current_state(const eshkol_psi_verifier_t* verifier);
uint32_t eshkol_psi_current_lob_rank(const eshkol_psi_verifier_t* verifier);
size_t eshkol_psi_verified_edge_count(const eshkol_psi_verifier_t* verifier);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_QTT_INCREMENTAL_PSI_H */
