/*
 * Fail-Closed Versioned Binary Persistence for CE-2 PROOF_DAG & CE-3 E-Graphs
 *
 * Wire format (little-endian, 24-byte header + payload):
 *   [0..3]   magic "ECE3"
 *   [4..7]   schema_version (uint32_t, currently 1)
 *   [8..11]  payload_kind (1 = PROOF_DAG, 2 = EGRAPH)
 *   [12..15] payload_bytes (uint32_t)
 *   [16..19] crc32 over header[0..15] || payload
 *   [20..23] fnv1a_32 integrity tag over payload
 *
 * Refuses any 1-bit flip, truncated/trailing bytes, bad magic, unknown schema
 * version, or invalid internal DAG/e-graph invariants.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_CE_PERSIST_H
#define ESHKOL_CORE_CE_PERSIST_H

#include <eshkol/core/diff_egglog.h>
#include <eshkol/core/soft_unify_semiring.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_CE_PERSIST_SCHEMA_V1 1u
#define ESHKOL_CE_PERSIST_HEADER_BYTES 24u

typedef enum eshkol_persist_status {
    ESHKOL_PERSIST_OK                  = 0,
    ESHKOL_PERSIST_ERR_INVALID_ARG     = 1,
    ESHKOL_PERSIST_ERR_TRUNCATED       = 2,
    ESHKOL_PERSIST_ERR_BAD_MAGIC       = 3,
    ESHKOL_PERSIST_ERR_BAD_VERSION     = 4,
    ESHKOL_PERSIST_ERR_BAD_KIND        = 5,
    ESHKOL_PERSIST_ERR_CHECKSUM        = 6,
    ESHKOL_PERSIST_ERR_CORRUPT_PAYLOAD = 7,
    ESHKOL_PERSIST_ERR_IO              = 8
} eshkol_persist_status_t;

/* Compute standard IEEE 802.3 CRC-32 checksum. */
uint32_t eshkol_ce_crc32(const uint8_t* data, size_t len);

/* Free a serialized buffer allocated by eshkol_ce_serialize_*. */
void eshkol_ce_buffer_free(uint8_t* buf);

/* Serialize / deserialize CE-2 PROOF_DAG with fail-closed integrity verification. */
eshkol_persist_status_t eshkol_ce_serialize_proof_dag(const eshkol_proof_dag_t* dag,
                                                      uint8_t** out_buf,
                                                      size_t* out_len);

eshkol_persist_status_t eshkol_ce_deserialize_proof_dag(const uint8_t* buf,
                                                        size_t len,
                                                        eshkol_proof_dag_t** out_dag);

/* Serialize / deserialize CE-3 colored e-graph with fail-closed integrity verification. */
eshkol_persist_status_t eshkol_ce_serialize_egraph(const eshkol_egraph_t* eg,
                                                   uint8_t** out_buf,
                                                   size_t* out_len);

eshkol_persist_status_t eshkol_ce_deserialize_egraph(const uint8_t* buf,
                                                     size_t len,
                                                     eshkol_egraph_t** out_eg);

/* Atomic file snapshot write (write temp + fsync + rename) and full-file read. */
eshkol_persist_status_t eshkol_ce_write_file_atomic(const char* path,
                                                    const uint8_t* data,
                                                    size_t len);

eshkol_persist_status_t eshkol_ce_read_file(const char* path,
                                            uint8_t** out_buf,
                                            size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_CE_PERSIST_H */
