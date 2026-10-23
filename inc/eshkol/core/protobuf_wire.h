/*
 * Protocol Buffers wire primitives — the one native encoder/decoder for the
 * protobuf binary encoding (https://protobuf.dev/programming-guides/encoding/).
 *
 * Every native component that writes or reads protobuf bytes uses these
 * functions: the ONNX checkpoint writer (lib/core/onnx_export.c) and the PJRT
 * compile-options encoder (lib/backend/xla/pjrt_client.cpp). Eshkol programs
 * reach the same encoding, schema-driven and over Eshkol values, through the
 * `core.protobuf` library (lib/core/protobuf.esk); both layers are pinned to
 * the same hand-derived wire vectors (tests/core/protobuf_wire_test.c and
 * tests/protobuf/protobuf_wire_test.esk).
 *
 * The header holds only pure, allocation-free computation, in the same style
 * as inc/eshkol/core/i128.h, so it can be included from C and C++ translation
 * units in any build product (runtime archive, XLA runtime, standalone tests)
 * without adding a link dependency.
 *
 * Encoding contract
 *   - varint: little-endian base-128, at most 10 bytes for a uint64.
 *   - tag: varint of (field_number << 3) | wire_type.
 *   - zigzag: sint32/sint64 map n to (n << 1) ^ (n >> 63) before varint.
 *   - fixed32/fixed64: little-endian, 4 and 8 bytes.
 *
 * Decoding contract
 *   Every reader is bounds-checked against `avail` and returns the number of
 *   bytes consumed, or 0 when the input is truncated or malformed. A varint
 *   longer than 10 bytes, or whose 10th byte carries bits above 2^64, is
 *   malformed. A tag with field number 0, a field number above 2^29 - 1, or a
 *   wire type of 6 or 7 is malformed. No reader ever touches in[avail].
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_PROTOBUF_WIRE_H
#define ESHKOL_CORE_PROTOBUF_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Wire types (the low three bits of a tag). */
#define ESHKOL_PB_WIRE_VARINT      0u
#define ESHKOL_PB_WIRE_FIXED64     1u
#define ESHKOL_PB_WIRE_LEN         2u
#define ESHKOL_PB_WIRE_START_GROUP 3u
#define ESHKOL_PB_WIRE_END_GROUP   4u
#define ESHKOL_PB_WIRE_FIXED32     5u

/* Largest encoded varint (a uint64 needs ceil(64 / 7) bytes). */
#define ESHKOL_PB_VARINT_MAX_BYTES 10u
/* Largest legal field number: 2^29 - 1. */
#define ESHKOL_PB_MAX_FIELD_NUMBER 536870911u

/* ── Encoding ─────────────────────────────────────────────────────────── */

/** Number of bytes the varint encoding of `value` occupies (1..10). */
static inline size_t eshkol_pb_varint_size(uint64_t value) {
    size_t n = 1;
    while (value > 0x7Fu) {
        value >>= 7;
        ++n;
    }
    return n;
}

/**
 * Write the varint encoding of `value` to `out`, which must have room for
 * eshkol_pb_varint_size(value) bytes. Returns the number of bytes written.
 */
static inline size_t eshkol_pb_write_varint(uint8_t* out, uint64_t value) {
    size_t n = 0;
    while (value > 0x7Fu) {
        out[n++] = (uint8_t)((value & 0x7Fu) | 0x80u);
        value >>= 7;
    }
    out[n++] = (uint8_t)value;
    return n;
}

/** The tag key (field_number << 3) | wire_type, before varint encoding. */
static inline uint64_t eshkol_pb_tag_key(uint32_t field_number, uint32_t wire_type) {
    return ((uint64_t)field_number << 3) | (uint64_t)(wire_type & 7u);
}

/** Number of bytes the tag for (field_number, wire_type) occupies. */
static inline size_t eshkol_pb_tag_size(uint32_t field_number, uint32_t wire_type) {
    return eshkol_pb_varint_size(eshkol_pb_tag_key(field_number, wire_type));
}

/** Write the tag for (field_number, wire_type). Returns bytes written. */
static inline size_t eshkol_pb_write_tag(uint8_t* out, uint32_t field_number,
                                         uint32_t wire_type) {
    return eshkol_pb_write_varint(out, eshkol_pb_tag_key(field_number, wire_type));
}

/** Zigzag mapping used by sint32 and sint64: 0,-1,1,-2,... -> 0,1,2,3,... */
static inline uint64_t eshkol_pb_zigzag_encode64(int64_t value) {
    return ((uint64_t)value << 1) ^ (uint64_t)(-(int64_t)((uint64_t)value >> 63));
}

static inline int64_t eshkol_pb_zigzag_decode64(uint64_t value) {
    return (int64_t)((value >> 1) ^ (uint64_t)(-(int64_t)(value & 1u)));
}

/** Write `value` as 4 little-endian bytes. Returns 4. */
static inline size_t eshkol_pb_write_fixed32(uint8_t* out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) out[i] = (uint8_t)(value >> (8 * i));
    return 4;
}

/** Write `value` as 8 little-endian bytes. Returns 8. */
static inline size_t eshkol_pb_write_fixed64(uint8_t* out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) out[i] = (uint8_t)(value >> (8 * i));
    return 8;
}

/* ── Decoding (bounds-checked; 0 = truncated or malformed) ───────────── */

static inline size_t eshkol_pb_read_varint(const uint8_t* in, size_t avail,
                                           uint64_t* value) {
    uint64_t result = 0;
    for (size_t i = 0; i < ESHKOL_PB_VARINT_MAX_BYTES; ++i) {
        if (i >= avail) return 0;
        const uint8_t byte = in[i];
        /* The 10th byte holds bit 63 only; anything more overflows 64 bits. */
        if (i == ESHKOL_PB_VARINT_MAX_BYTES - 1 && byte > 1u) return 0;
        result |= (uint64_t)(byte & 0x7Fu) << (7 * i);
        if ((byte & 0x80u) == 0) {
            *value = result;
            return i + 1;
        }
    }
    return 0;
}

static inline size_t eshkol_pb_read_tag(const uint8_t* in, size_t avail,
                                        uint32_t* field_number, uint32_t* wire_type) {
    uint64_t key = 0;
    const size_t n = eshkol_pb_read_varint(in, avail, &key);
    if (n == 0) return 0;
    const uint64_t field = key >> 3;
    const uint32_t type = (uint32_t)(key & 7u);
    if (field == 0 || field > ESHKOL_PB_MAX_FIELD_NUMBER || type > ESHKOL_PB_WIRE_FIXED32)
        return 0;
    *field_number = (uint32_t)field;
    *wire_type = type;
    return n;
}

static inline size_t eshkol_pb_read_fixed32(const uint8_t* in, size_t avail,
                                            uint32_t* value) {
    if (avail < 4) return 0;
    uint32_t result = 0;
    for (size_t i = 0; i < 4; ++i) result |= (uint32_t)in[i] << (8 * i);
    *value = result;
    return 4;
}

static inline size_t eshkol_pb_read_fixed64(const uint8_t* in, size_t avail,
                                            uint64_t* value) {
    if (avail < 8) return 0;
    uint64_t result = 0;
    for (size_t i = 0; i < 8; ++i) result |= (uint64_t)in[i] << (8 * i);
    *value = result;
    return 8;
}

/**
 * Read a length-delimited payload header: the varint length, checked against
 * the bytes that follow it. On success `*payload_offset` is the offset of the
 * payload inside `in`, `*payload_len` its length, and the return value is the
 * total consumed (header plus payload).
 */
static inline size_t eshkol_pb_read_len(const uint8_t* in, size_t avail,
                                        size_t* payload_offset, size_t* payload_len) {
    uint64_t len = 0;
    const size_t n = eshkol_pb_read_varint(in, avail, &len);
    if (n == 0 || len > (uint64_t)(avail - n)) return 0;
    *payload_offset = n;
    *payload_len = (size_t)len;
    return n + (size_t)len;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_PROTOBUF_WIRE_H */
