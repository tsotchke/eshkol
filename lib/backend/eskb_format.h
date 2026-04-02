/**
 * @file eskb_format.h
 * @brief ESKB (Eshkol Bytecode) binary format specification.
 *
 * Section-based format with LEB128 variable-length encoding.
 * Designed for single-pass validation and qLLM weight loading.
 *
 * Layout:
 *   Header (16 bytes) → Section Table → Sections (CONST, CODE, META, SYMB)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESKB_FORMAT_H
#define ESKB_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ── Magic & Version ── */
#define ESKB_MAGIC   0x45534B42  /* "ESKB" */
#define ESKB_VERSION 1

/* ── Section IDs ── */
#define ESKB_SECTION_CONST  0  /* Constant pool */
#define ESKB_SECTION_CODE   1  /* Bytecode functions */
#define ESKB_SECTION_META   2  /* Debug info (optional) */
#define ESKB_SECTION_SYMB   3  /* Symbol/export table */
#define ESKB_SECTION_COUNT  4

/* ── Constant Types ── */
#define ESKB_CONST_NIL    0
#define ESKB_CONST_INT64  1
#define ESKB_CONST_F64    2
#define ESKB_CONST_BOOL   3
#define ESKB_CONST_STRING 6

/* ── Header Flags ── */
#define ESKB_FLAG_LITTLE_ENDIAN  0x01
#define ESKB_FLAG_DEBUG_INFO     0x02

/* ── Header (16 bytes, fixed) ── */
typedef struct {
    uint32_t magic;     /* ESKB_MAGIC */
    uint32_t version;   /* ESKB_VERSION */
    uint32_t flags;     /* ESKB_FLAG_* */
    uint32_t checksum;  /* CRC32 of everything after header */
} EskbHeader;

/* ── Section descriptor ── */
typedef struct {
    uint8_t  id;
    uint32_t size;  /* size in bytes of section payload */
} EskbSectionDesc;

/* ── Function descriptor (in-memory, not on-disk) ── */
typedef struct {
    char     name[128];
    uint8_t  n_params;
    uint16_t n_locals;
    uint8_t  n_upvalues;
    uint32_t code_offset;  /* offset into code array */
    uint32_t code_len;     /* number of instructions */
} EskbFuncDesc;

/*******************************************************************************
 * LEB128 Encoding/Decoding (unsigned)
 ******************************************************************************/

static inline int eskb_write_leb128(uint8_t* buf, uint64_t val) {
    int n = 0;
    do {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if (val) byte |= 0x80;
        buf[n++] = byte;
    } while (val);
    return n;
}

static inline int eskb_read_leb128(const uint8_t* buf, size_t buf_len, uint64_t* out) {
    *out = 0;
    int shift = 0, n = 0;
    while (n < (int)buf_len && n < 10) {
        uint8_t byte = buf[n];
        *out |= (uint64_t)(byte & 0x7F) << shift;
        n++;
        if (!(byte & 0x80)) return n;
        shift += 7;
    }
    return -1; /* truncated or overflow */
}

/*******************************************************************************
 * CRC32 (simple implementation, no table lookup needed for small files)
 ******************************************************************************/

static inline uint32_t eskb_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

/*******************************************************************************
 * Helpers for writing to a growable buffer
 ******************************************************************************/

typedef struct {
    uint8_t* data;
    size_t   len;
    size_t   cap;
} EskbBuffer;

static inline void eskb_buf_init(EskbBuffer* b) {
    b->data = NULL; b->len = 0; b->cap = 0;
}

static inline void eskb_buf_free(EskbBuffer* b) {
    free(b->data); b->data = NULL; b->len = 0; b->cap = 0;
}

static inline int eskb_buf_ensure(EskbBuffer* b, size_t extra) {
    if (b->len + extra <= b->cap) return 0;
    size_t new_cap = b->cap ? b->cap * 2 : 4096;
    while (new_cap < b->len + extra) new_cap *= 2;
    uint8_t* nd = (uint8_t*)realloc(b->data, new_cap);
    if (!nd) return -1;
    b->data = nd; b->cap = new_cap;
    return 0;
}

static inline int eskb_buf_write(EskbBuffer* b, const void* src, size_t n) {
    if (eskb_buf_ensure(b, n) < 0) return -1;
    memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

static inline int eskb_buf_write_u8(EskbBuffer* b, uint8_t v) {
    return eskb_buf_write(b, &v, 1);
}

static inline int eskb_buf_write_leb128(EskbBuffer* b, uint64_t val) {
    uint8_t tmp[10];
    int n = eskb_write_leb128(tmp, val);
    return eskb_buf_write(b, tmp, n);
}

static inline int eskb_buf_write_f64(EskbBuffer* b, double v) {
    return eskb_buf_write(b, &v, 8);
}

static inline int eskb_buf_write_i64(EskbBuffer* b, int64_t v) {
    return eskb_buf_write(b, &v, 8);
}

static inline int eskb_buf_write_string(EskbBuffer* b, const char* s, size_t len) {
    if (eskb_buf_write_leb128(b, len) < 0) return -1;
    return eskb_buf_write(b, s, len);
}

/*******************************************************************************
 * Helpers for reading from a buffer
 ******************************************************************************/

typedef struct {
    const uint8_t* data;
    size_t         len;
    size_t         pos;
} EskbReader;

static inline int eskb_reader_init(EskbReader* r, const uint8_t* data, size_t len) {
    r->data = data; r->len = len; r->pos = 0;
    return 0;
}

static inline int eskb_read_u8(EskbReader* r, uint8_t* out) {
    if (r->pos >= r->len) return -1;
    *out = r->data[r->pos++];
    return 0;
}

static inline int eskb_read_bytes(EskbReader* r, void* out, size_t n) {
    if (r->pos + n > r->len) return -1;
    memcpy(out, r->data + r->pos, n);
    r->pos += n;
    return 0;
}

static inline int eskb_read_leb(EskbReader* r, uint64_t* out) {
    if (r->pos >= r->len) return -1;
    int n = eskb_read_leb128(r->data + r->pos, r->len - r->pos, out);
    if (n < 0) return -1;
    r->pos += n;
    return 0;
}

static inline int eskb_read_f64(EskbReader* r, double* out) {
    return eskb_read_bytes(r, out, 8);
}

static inline int eskb_read_i64(EskbReader* r, int64_t* out) {
    return eskb_read_bytes(r, out, 8);
}

static inline int eskb_read_string(EskbReader* r, char* out, size_t max_len, size_t* actual_len) {
    uint64_t slen;
    if (eskb_read_leb(r, &slen) < 0) return -1;
    if (slen > max_len - 1) return -1;
    if (eskb_read_bytes(r, out, (size_t)slen) < 0) return -1;
    out[slen] = 0;
    if (actual_len) *actual_len = (size_t)slen;
    return 0;
}

#endif /* ESKB_FORMAT_H */
