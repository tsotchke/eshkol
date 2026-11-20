/* The native and VM Merkle hash primitives share the same 64-bit arithmetic. */
#ifndef ESHKOL_CORE_MERKLE_HASH_H
#define ESHKOL_CORE_MERKLE_HASH_H

#include <stdint.h>
#include <stddef.h>

/* (fnv1a-64-raw bytes len) → int64 (the user sees uint64 bit pattern as
 * an int64 — no precision loss because Eshkol's int64 path is exact). */
static inline int64_t eshkol_merkle_fnv1a(const char* data, int64_t len) {
    if (!data || len <= 0) return (int64_t)UINT64_C(0xcbf29ce484222325);
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    const unsigned char* p = (const unsigned char*)data;
    for (int64_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(0x100000001b3);
    }
    return (int64_t)h;
}

/* Hex-encode the low 64 bits of an int64 into a 16-char buffer (no
 * NUL).  Caller passes a buffer of >=16 bytes.  Returns 16. */
static inline int64_t eshkol_merkle_hex16(int64_t value, char* out, int64_t out_size) {
    if (!out || out_size < 16) return 0;
    static const char hex[] = "0123456789abcdef";
    uint64_t v = (uint64_t)value;
    for (int i = 15; i >= 0; i--) {
        out[i] = hex[v & 0xf];
        v >>= 4;
    }
    /* Optional NUL — only if there's room. */
    if (out_size >= 17) out[16] = '\0';
    return 16;
}

#endif
