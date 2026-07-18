/*
 * Native 128-bit Integer (i128) — shared computational core for Eshkol.
 *
 * i128 is a DISTINCT, fixed-width, two's-complement signed integer type that
 * sits OFF the exact numeric tower (fixnum -> bignum -> rational -> real ->
 * complex). Unlike bignum — which never overflows — i128 arithmetic WRAPS
 * modulo 2^128 (documented, deterministic). There is no auto-promotion to or
 * from the tower; every crossing is an explicit conversion builtin.
 *
 * This header holds only PURE, allocation-free, non-raising computation over
 * the platform-native `__int128`. It is intentionally shared by both runtime
 * substrates so they compute bit-identical results:
 *   - the native/JIT/AOT runtime (lib/core/i128_runtime.cpp), which boxes the
 *     payload on the arena and signals errors through eshkol_raise();
 *   - the bytecode VM (lib/backend/vm_native.c), which boxes the payload on
 *     the VM region heap and signals errors through vm->error.
 * Each substrate owns its own boxing + error policy; the math lives here once.
 *
 * FFI / storage layout: the heap payload is `eshkol_i128_abi {uint64_t lo, hi}`
 * (little-endian limb order). This layout DELIBERATELY matches the planned
 * `esk_i128_abi` two-u64 marshalling contract (lib/math/fixed_point, owned by
 * a separate effort) so the boxed representation can be unified later without a
 * data migration.
 *
 * The repo already relies on `__int128` as its trusted wide-integer engine
 * (lib/core/bignum.cpp, lib/core/rational.cpp, lib/backend/llvm_codegen.cpp);
 * a first-class i128 type is an exposure of that established primitive.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_I128_H
#define ESHKOL_CORE_I128_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Two-u64 FFI/storage ABI for a boxed i128. Little-endian limb order:
 *   value == ((__int128)(int64 sign-extended hi) << 64) | lo   (two's complement)
 * Deliberately identical to the planned esk_i128_abi so the boxed heap payload
 * unifies with the fixed_point engine's marshalling with no format change.
 */
typedef struct eshkol_i128_abi {
    uint64_t lo;
    uint64_t hi;
} eshkol_i128_abi;

/* Bit-exact widest decimal is -170141183460469231731687303715884105728:
 * sign + 39 digits + NUL = 41 bytes. Callers must supply at least this much. */
#define ESHKOL_I128_STR_MAX 41

/* ---- pack / unpack between the {lo,hi} payload and a native __int128 ---- */

static inline __int128 eshkol_i128_from_abi(eshkol_i128_abi a) {
    return (__int128)(((unsigned __int128)a.hi << 64) | (unsigned __int128)a.lo);
}

static inline eshkol_i128_abi eshkol_i128_to_abi(__int128 v) {
    unsigned __int128 u = (unsigned __int128)v;
    eshkol_i128_abi a;
    a.lo = (uint64_t)u;
    a.hi = (uint64_t)(u >> 64);
    return a;
}

/* ---- wrapping arithmetic (two's-complement, modulo 2^128) ----
 * Computed in the UNSIGNED domain so overflow is well-defined wraparound
 * (signed overflow is UB in C/C++); the bit pattern is then reinterpreted. */

static inline __int128 eshkol_i128_add(__int128 a, __int128 b) {
    return (__int128)((unsigned __int128)a + (unsigned __int128)b);
}
static inline __int128 eshkol_i128_sub(__int128 a, __int128 b) {
    return (__int128)((unsigned __int128)a - (unsigned __int128)b);
}
static inline __int128 eshkol_i128_mul(__int128 a, __int128 b) {
    return (__int128)((unsigned __int128)a * (unsigned __int128)b);
}
static inline __int128 eshkol_i128_neg(__int128 a) {
    return (__int128)(~(unsigned __int128)a + 1u);
}

/* ---- shifts. Caller MUST validate the count is in [0,127] first; this core
 * assumes a valid count and leaves range signalling to the substrate. ---- */

static inline __int128 eshkol_i128_shl(__int128 a, unsigned count) {
    return (__int128)((unsigned __int128)a << count);           /* logical == arithmetic left */
}
static inline __int128 eshkol_i128_ashr(__int128 a, unsigned count) {
    /* Arithmetic (sign-preserving) right shift. Signed >> is
     * implementation-defined for negatives but arithmetic on every target
     * Eshkol supports; make it explicit to avoid relying on that. */
    if (a >= 0) return (__int128)((unsigned __int128)a >> count);
    unsigned __int128 u = (unsigned __int128)a >> count;
    unsigned __int128 sign_fill = (~(unsigned __int128)0) << (count == 0 ? 127 : (128 - count));
    if (count == 0) return a;
    return (__int128)(u | sign_fill);
}
static inline __int128 eshkol_i128_lshr(__int128 a, unsigned count) {
    return (__int128)((unsigned __int128)a >> count);           /* logical (zero-fill) */
}

/* ---- comparisons ---- */

static inline int eshkol_i128_cmp(__int128 a, __int128 b) {
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

/* ---- truncated division (C semantics: quotient truncates toward zero,
 * remainder carries the sign of the dividend). Caller MUST reject a zero
 * divisor first. INT128_MIN / -1 would overflow; it wraps to INT128_MIN,
 * matching two's-complement semantics rather than trapping. ---- */

static inline __int128 eshkol_i128_quotient(__int128 a, __int128 b) {
    /* Guard the single overflowing case so we stay defined. */
    unsigned __int128 amin = (unsigned __int128)1 << 127;   /* INT128_MIN bit pattern */
    if ((unsigned __int128)a == amin && b == (__int128)(-1)) return a;  /* wraps */
    return a / b;
}
static inline __int128 eshkol_i128_remainder(__int128 a, __int128 b) {
    unsigned __int128 amin = (unsigned __int128)1 << 127;
    if ((unsigned __int128)a == amin && b == (__int128)(-1)) return 0;
    return a % b;
}

/* ---- decimal formatting. Writes a NUL-terminated decimal string into `buf`
 * (which must hold ESHKOL_I128_STR_MAX bytes) and returns its length (excluding
 * the NUL). Handles INT128_MIN via unsigned magnitude. ---- */

static inline size_t eshkol_i128_format(__int128 v, char* buf) {
    int neg = (v < 0);
    unsigned __int128 mag = neg ? (~(unsigned __int128)v + 1u) : (unsigned __int128)v;
    char tmp[40];
    int n = 0;
    if (mag == 0) {
        tmp[n++] = '0';
    } else {
        while (mag > 0) {
            tmp[n++] = (char)('0' + (int)(mag % 10));
            mag /= 10;
        }
    }
    size_t pos = 0;
    if (neg) buf[pos++] = '-';
    while (n > 0) buf[pos++] = tmp[--n];
    buf[pos] = '\0';
    return pos;
}

/* ---- decimal parsing over the FULL signed range, including -2^127 (which has
 * no positive counterpart). Accepts an optional leading '+'/'-' then one or
 * more ASCII digits; rejects empty input, stray characters, and any value
 * outside [-2^127, 2^127-1]. Returns true on success. Never raises. ---- */

static inline bool eshkol_i128_parse(const char* s, size_t len, __int128* out) {
    if (!s || len == 0) return false;
    size_t i = 0;
    int neg = 0;
    if (s[i] == '+' || s[i] == '-') {
        neg = (s[i] == '-');
        i++;
    }
    if (i >= len) return false;                 /* sign with no digits */
    const unsigned __int128 UMAX = ~(unsigned __int128)0;
    unsigned __int128 mag = 0;
    for (; i < len; i++) {
        char c = s[i];
        if (c < '0' || c > '9') return false;
        unsigned d = (unsigned)(c - '0');
        if (mag > (UMAX - d) / 10) return false; /* unsigned wrap guard */
        mag = mag * 10u + d;
    }
    unsigned __int128 limit = neg ? ((unsigned __int128)1 << 127)          /* 2^127 */
                                  : (((unsigned __int128)1 << 127) - 1u);  /* 2^127 - 1 */
    if (mag > limit) return false;
    unsigned __int128 bits = neg ? (~mag + 1u) : mag;   /* two's complement for negatives */
    *out = (__int128)bits;
    return true;
}

#endif /* ESHKOL_CORE_I128_H */
