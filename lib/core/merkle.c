/*
 * merkle.c — fast non-crypto hash primitives for content addressing.
 *
 * Eshkol-side modulo on bignums >2^53 was returning truncated values
 * (lib/core/merkle.esk's pure-Eshkol FNV-1a tripped this), so the
 * inner loop is in C.  No external dependencies.  64-bit modular
 * arithmetic is just uint64_t overflow.
 *
 * For cryptographic content addressing (signing, distributed
 * adversarial storage), use lib/agent/crypto.esk's sha256 instead —
 * SHA256 is the right tool for that and it's already wrapped.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define FNV1A_OFFSET_64 0xcbf29ce484222325ULL
#define FNV1A_PRIME_64  0x100000001b3ULL

/* (fnv1a-64-raw bytes len) → int64 (the user sees uint64 bit pattern as
 * an int64 — no precision loss because Eshkol's int64 path is exact). */
int64_t eshkol_fnv1a_64(const char* data, int64_t len) {
    if (!data || len <= 0) return (int64_t)FNV1A_OFFSET_64;
    uint64_t h = FNV1A_OFFSET_64;
    const unsigned char* p = (const unsigned char*)data;
    for (int64_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= FNV1A_PRIME_64;
    }
    return (int64_t)h;
}

/* Hex-encode the low 64 bits of an int64 into a 16-char buffer (no
 * NUL).  Caller passes a buffer of >=16 bytes.  Returns 16. */
int64_t eshkol_u64_to_hex16(int64_t value, char* out, int64_t out_size) {
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
