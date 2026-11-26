/*
 * Chained eshkol.certificate.v2 Derivation Verifier & Lean 4.12.0 Exporter
 *
 * Cryptographically links:
 *   Link 1 (ProgramLink) -> Link 2 (TaylorEnclosureLink) -> Link 3 (SafetyPostconditionLink)
 * using self-contained SHA-256 and HMAC-SHA256 bound to an explicit context_id
 * and key validity epoch window. Exports standalone kernel-checkable Lean 4.12.0
 * proof modules with zero sorry and zero custom axioms.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_CERT_CHAIN_V2_H
#define ESHKOL_CORE_CERT_CHAIN_V2_H

#include "eshkol/core/interval_taylor_enclosure.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_CERT_V2_MAGIC         0x45433256U /* "EC2V" */
#define ESHKOL_CERT_V2_VERSION       2U
#define ESHKOL_CERT_V2_MAX_DIMS      16U
#define ESHKOL_CERT_V2_CONTEXT_MAX   64U
#define ESHKOL_CERT_V2_HASH_BYTES    32U

typedef enum eshkol_cert_v2_status {
    ESHKOL_CERT_OK                          = 0,
    ESHKOL_CERT_ERR_HASH_CHAIN_BREAK        = 1,
    ESHKOL_CERT_ERR_CONTEXT_MISMATCH        = 2,
    ESHKOL_CERT_ERR_EXPIRED_KEY             = 3,
    ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION = 4,
    ESHKOL_CERT_ERR_MALFORMED               = 5
} eshkol_cert_v2_status_t;

typedef struct eshkol_cert_program_link {
    uint8_t program_digest[ESHKOL_CERT_V2_HASH_BYTES];
    uint32_t input_dim;
    uint32_t output_dim;
    eshkol_tm_interval_t input_box[ESHKOL_CERT_V2_MAX_DIMS];
    uint8_t link_hash[ESHKOL_CERT_V2_HASH_BYTES];
} eshkol_cert_program_link_t;

typedef struct eshkol_cert_taylor_link {
    uint8_t prev_hash[ESHKOL_CERT_V2_HASH_BYTES];
    eshkol_tm_interval_t certified_bounds[ESHKOL_CERT_V2_MAX_DIMS];
    uint8_t link_hash[ESHKOL_CERT_V2_HASH_BYTES];
} eshkol_cert_taylor_link_t;

typedef struct eshkol_cert_safety_link {
    uint8_t prev_hash[ESHKOL_CERT_V2_HASH_BYTES];
    eshkol_tm_interval_t safe_bounds[ESHKOL_CERT_V2_MAX_DIMS];
    uint8_t link_hash[ESHKOL_CERT_V2_HASH_BYTES];
} eshkol_cert_safety_link_t;

typedef struct eshkol_cert_chain_v2 {
    uint32_t magic;
    uint32_t version;
    char context_id[ESHKOL_CERT_V2_CONTEXT_MAX];
    uint64_t issued_epoch;
    uint64_t valid_until_epoch;
    eshkol_cert_program_link_t program_link;
    eshkol_cert_taylor_link_t taylor_link;
    eshkol_cert_safety_link_t safety_link;
    uint8_t signature_hmac[ESHKOL_CERT_V2_HASH_BYTES];
} eshkol_cert_chain_v2_t;

/* Self-contained SHA-256 & HMAC-SHA256 primitives for certificate v2 chains. */
void eshkol_cert_sha256(const uint8_t* data, size_t len, uint8_t out_hash[32]);
void eshkol_cert_hmac_sha256(const uint8_t* key, size_t key_len,
                             const uint8_t* data, size_t data_len,
                             uint8_t out_mac[32]);

/*
 * Build and sign an eshkol.certificate.v2 derivation chain directly from a
 * neural world model by running CE-8 order-2 Interval-Taylor certification.
 */
eshkol_cert_v2_status_t eshkol_cert_chain_v2_build(
    const uint8_t* program_bytes,
    size_t program_len,
    const eshkol_tm_world_model_t* model,
    const eshkol_tm_interval_t* input_box,
    const eshkol_tm_interval_t* safe_bounds,
    const char* context_id,
    uint64_t issued_epoch,
    uint64_t valid_until_epoch,
    const uint8_t* signing_key,
    size_t key_len,
    eshkol_cert_chain_v2_t* out_chain);

/*
 * Recompute link hashes and HMAC signature on an existing chain structure with
 * a new context_id and epoch validity window (e.g. for context/expiry controls).
 */
eshkol_cert_v2_status_t eshkol_cert_chain_v2_resign(
    eshkol_cert_chain_v2_t* chain,
    const char* new_context_id,
    uint64_t new_issued_epoch,
    uint64_t new_valid_until_epoch,
    const uint8_t* signing_key,
    size_t key_len);

/*
 * Strictly verify an eshkol.certificate.v2 derivation chain:
 *   1. Structural validity -> ESHKOL_CERT_ERR_MALFORMED
 *   2. SHA-256 link chain + HMAC integrity -> ESHKOL_CERT_ERR_HASH_CHAIN_BREAK
 *   3. Context binding -> ESHKOL_CERT_ERR_CONTEXT_MISMATCH
 *   4. Key/epoch validity window -> ESHKOL_CERT_ERR_EXPIRED_KEY
 *   5. Certified Taylor enclosure subset of safe_bounds -> ESHKOL_CERT_ERR_POSTCONDITION_VIOLATION
 */
eshkol_cert_v2_status_t eshkol_cert_chain_v2_verify(
    const eshkol_cert_chain_v2_t* chain,
    const char* expected_context_id,
    uint64_t current_epoch,
    const uint8_t* signing_key,
    size_t key_len);

/*
 * Export a standalone Lean 4.12.0 theorem file encoding the derivation chain
 * hash linkage and outward-rounded scaled-integer safety inclusion proofs
 * (verified by Lean 4 kernel `by decide` with zero sorry and zero axioms).
 */
bool eshkol_cert_chain_export_lean4(
    const eshkol_cert_chain_v2_t* chain,
    const char* expected_context_id,
    uint64_t eval_epoch,
    const char* module_namespace,
    char* out_buf,
    size_t out_cap,
    size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_CERT_CHAIN_V2_H */
