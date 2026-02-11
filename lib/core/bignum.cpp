/*
 * Bignum (Arbitrary-Precision Integer) Implementation for Eshkol
 *
 * Algorithms: Schoolbook addition/subtraction/multiplication, Knuth's Algorithm D
 * for division. Limbs are 64-bit unsigned integers stored little-endian.
 *
 * All allocation goes through the arena — no malloc/free.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/bignum.h"
#include "eshkol/eshkol.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/* Arena functions declared in arena_memory.cpp (extern "C") */
extern "C" {
    void* arena_allocate(arena_t* arena, size_t size);
    void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                      uint8_t subtype, uint8_t flags);
    char* arena_allocate_string_with_header(arena_t* arena, size_t length);
}

/* ===== Internal helpers ===== */

/* Allocate a bignum with n limbs on the arena */
static eshkol_bignum_t* bignum_alloc(arena_t* arena, uint32_t num_limbs) {
    size_t data_size = sizeof(eshkol_bignum_t) + num_limbs * sizeof(uint64_t);
    void* ptr = arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_BIGNUM, 0);
    if (!ptr) return nullptr;
    eshkol_bignum_t* bn = (eshkol_bignum_t*)ptr;
    bn->sign = 0;
    bn->num_limbs = num_limbs;
    memset(BIGNUM_LIMBS(bn), 0, num_limbs * sizeof(uint64_t));
    return bn;
}

/* Normalize: remove trailing zero limbs (but keep at least 1) */
static void bignum_normalize(eshkol_bignum_t* bn) {
    uint64_t* limbs = BIGNUM_LIMBS(bn);
    while (bn->num_limbs > 1 && limbs[bn->num_limbs - 1] == 0) {
        bn->num_limbs--;
    }
    /* Zero is always non-negative */
    if (bn->num_limbs == 1 && limbs[0] == 0) {
        bn->sign = 0;
    }
}

/* Compare absolute values. Returns -1, 0, or 1. */
static int bignum_compare_abs(const eshkol_bignum_t* a, const eshkol_bignum_t* b) {
    if (a->num_limbs != b->num_limbs) {
        return (a->num_limbs > b->num_limbs) ? 1 : -1;
    }
    const uint64_t* la = BIGNUM_LIMBS(a);
    const uint64_t* lb = BIGNUM_LIMBS(b);
    for (int32_t i = (int32_t)a->num_limbs - 1; i >= 0; i--) {
        if (la[i] != lb[i]) {
            return (la[i] > lb[i]) ? 1 : -1;
        }
    }
    return 0;
}

/* Add absolute values: result = |a| + |b|. Caller sets sign. */
static eshkol_bignum_t* bignum_add_abs(arena_t* arena,
                                        const eshkol_bignum_t* a,
                                        const eshkol_bignum_t* b) {
    uint32_t max_limbs = (a->num_limbs > b->num_limbs) ? a->num_limbs : b->num_limbs;
    eshkol_bignum_t* result = bignum_alloc(arena, max_limbs + 1);
    if (!result) return nullptr;

    const uint64_t* la = BIGNUM_LIMBS(a);
    const uint64_t* lb = BIGNUM_LIMBS(b);
    uint64_t* lr = BIGNUM_LIMBS(result);

    uint64_t carry = 0;
    for (uint32_t i = 0; i < max_limbs; i++) {
        uint64_t av = (i < a->num_limbs) ? la[i] : 0;
        uint64_t bv = (i < b->num_limbs) ? lb[i] : 0;
        /* Use 128-bit addition to detect carry */
        __uint128_t sum = (__uint128_t)av + (__uint128_t)bv + carry;
        lr[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    lr[max_limbs] = carry;

    bignum_normalize(result);
    return result;
}

/* Subtract absolute values: result = |a| - |b|. Requires |a| >= |b|. Caller sets sign. */
static eshkol_bignum_t* bignum_sub_abs(arena_t* arena,
                                        const eshkol_bignum_t* a,
                                        const eshkol_bignum_t* b) {
    eshkol_bignum_t* result = bignum_alloc(arena, a->num_limbs);
    if (!result) return nullptr;

    const uint64_t* la = BIGNUM_LIMBS(a);
    const uint64_t* lb = BIGNUM_LIMBS(b);
    uint64_t* lr = BIGNUM_LIMBS(result);

    uint64_t borrow = 0;
    for (uint32_t i = 0; i < a->num_limbs; i++) {
        uint64_t av = la[i];
        uint64_t bv = (i < b->num_limbs) ? lb[i] : 0;
        /* Subtract with borrow */
        uint64_t diff = av - bv - borrow;
        borrow = (av < bv + borrow || (borrow && bv == UINT64_MAX)) ? 1 : 0;
        lr[i] = diff;
    }

    bignum_normalize(result);
    return result;
}

/* Multiply single limb: result += a * b_limb, starting at offset. */
static void bignum_addmul_limb(uint64_t* result, uint32_t result_limbs,
                                const uint64_t* a, uint32_t a_limbs,
                                uint64_t b_limb, uint32_t offset) {
    uint64_t carry = 0;
    for (uint32_t i = 0; i < a_limbs; i++) {
        __uint128_t prod = (__uint128_t)a[i] * (__uint128_t)b_limb +
                           (__uint128_t)result[i + offset] + carry;
        result[i + offset] = (uint64_t)prod;
        carry = (uint64_t)(prod >> 64);
    }
    if (offset + a_limbs < result_limbs) {
        result[offset + a_limbs] += carry;
    }
}

/* Divide bignum by single limb. Returns quotient, stores remainder. */
static eshkol_bignum_t* bignum_div_limb(arena_t* arena,
                                         const eshkol_bignum_t* a,
                                         uint64_t divisor,
                                         uint64_t* remainder) {
    eshkol_bignum_t* result = bignum_alloc(arena, a->num_limbs);
    if (!result) return nullptr;

    const uint64_t* la = BIGNUM_LIMBS(a);
    uint64_t* lr = BIGNUM_LIMBS(result);

    __uint128_t rem = 0;
    for (int32_t i = (int32_t)a->num_limbs - 1; i >= 0; i--) {
        __uint128_t cur = (rem << 64) | la[i];
        lr[i] = (uint64_t)(cur / divisor);
        rem = cur % divisor;
    }
    if (remainder) *remainder = (uint64_t)rem;

    bignum_normalize(result);
    return result;
}

/* Multi-limb division: Knuth's Algorithm D (simplified).
 * Computes quotient and remainder of |a| / |b|. */
static void bignum_divmod_abs(arena_t* arena,
                               const eshkol_bignum_t* a,
                               const eshkol_bignum_t* b,
                               eshkol_bignum_t** quotient,
                               eshkol_bignum_t** remainder) {
    /* Special case: single-limb divisor */
    if (b->num_limbs == 1) {
        uint64_t rem = 0;
        eshkol_bignum_t* q = bignum_div_limb(arena, a, BIGNUM_LIMBS(b)[0], &rem);
        if (quotient) *quotient = q;
        if (remainder) {
            *remainder = bignum_alloc(arena, 1);
            if (*remainder) BIGNUM_LIMBS(*remainder)[0] = rem;
        }
        return;
    }

    int cmp = bignum_compare_abs(a, b);
    if (cmp < 0) {
        /* |a| < |b|: quotient = 0, remainder = a */
        if (quotient) {
            *quotient = bignum_alloc(arena, 1);
        }
        if (remainder) {
            *remainder = bignum_alloc(arena, a->num_limbs);
            if (*remainder) {
                (*remainder)->num_limbs = a->num_limbs;
                memcpy(BIGNUM_LIMBS(*remainder), BIGNUM_LIMBS(a),
                       a->num_limbs * sizeof(uint64_t));
            }
        }
        return;
    }
    if (cmp == 0) {
        /* |a| == |b|: quotient = 1, remainder = 0 */
        if (quotient) {
            *quotient = bignum_alloc(arena, 1);
            if (*quotient) BIGNUM_LIMBS(*quotient)[0] = 1;
        }
        if (remainder) {
            *remainder = bignum_alloc(arena, 1);
        }
        return;
    }

    /* Knuth's Algorithm D for multi-limb division.
     * We use a simplified version: trial division with correction.
     * u = dividend (a), v = divisor (b), n = v->num_limbs, m = u->num_limbs - n */
    uint32_t n = b->num_limbs;
    uint32_t m = a->num_limbs - n;

    /* Normalize: shift so that the high bit of divisor's top limb is set */
    const uint64_t* vn = BIGNUM_LIMBS(b);
    uint64_t d_top = vn[n - 1];
    int shift = 0;
    if (d_top != 0) {
        shift = __builtin_clzll(d_top);
    }

    /* Create shifted copies */
    uint32_t un_len = a->num_limbs + 1;
    uint64_t* un = (uint64_t*)arena_allocate(arena, un_len * sizeof(uint64_t));
    uint64_t* vn_shifted = (uint64_t*)arena_allocate(arena, n * sizeof(uint64_t));
    if (!un || !vn_shifted) {
        if (quotient) *quotient = bignum_alloc(arena, 1);
        if (remainder) *remainder = bignum_alloc(arena, 1);
        return;
    }

    /* Shift divisor left by 'shift' bits */
    if (shift > 0) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < n; i++) {
            vn_shifted[i] = (vn[i] << shift) | carry;
            carry = vn[i] >> (64 - shift);
        }
    } else {
        memcpy(vn_shifted, vn, n * sizeof(uint64_t));
    }

    /* Shift dividend left by 'shift' bits */
    const uint64_t* ua = BIGNUM_LIMBS(a);
    if (shift > 0) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < a->num_limbs; i++) {
            un[i] = (ua[i] << shift) | carry;
            carry = ua[i] >> (64 - shift);
        }
        un[a->num_limbs] = carry;
    } else {
        memcpy(un, ua, a->num_limbs * sizeof(uint64_t));
        un[a->num_limbs] = 0;
    }

    /* Allocate quotient */
    eshkol_bignum_t* q = bignum_alloc(arena, m + 1);
    uint64_t* ql = BIGNUM_LIMBS(q);

    /* Main loop: for each quotient limb from most significant to least */
    for (int32_t j = (int32_t)m; j >= 0; j--) {
        /* Trial quotient: qhat = (un[j+n]*2^64 + un[j+n-1]) / vn_shifted[n-1] */
        __uint128_t uhat = ((__uint128_t)un[j + n] << 64) | un[j + n - 1];
        __uint128_t qhat = uhat / vn_shifted[n - 1];
        __uint128_t rhat = uhat % vn_shifted[n - 1];

        /* Refine trial quotient (Knuth's test) */
        while (qhat >= ((__uint128_t)1 << 64) ||
               (n >= 2 && qhat * vn_shifted[n - 2] > (rhat << 64) + un[j + n - 2])) {
            qhat--;
            rhat += vn_shifted[n - 1];
            if (rhat >= ((__uint128_t)1 << 64)) break;
        }

        /* Multiply and subtract: un[j..j+n] -= qhat * vn_shifted */
        __int128_t borrow_s = 0;
        for (uint32_t i = 0; i < n; i++) {
            __uint128_t prod = (__uint128_t)(uint64_t)qhat * vn_shifted[i];
            __int128_t diff = (__int128_t)un[j + i] - (int64_t)(uint64_t)prod - borrow_s;
            un[j + i] = (uint64_t)diff;
            borrow_s = (int64_t)(uint64_t)(prod >> 64) - (int64_t)(diff >> 64);
        }
        __int128_t diff_top = (__int128_t)un[j + n] - borrow_s;
        un[j + n] = (uint64_t)diff_top;

        ql[j] = (uint64_t)qhat;

        /* If we subtracted too much, add back */
        if (diff_top < 0) {
            ql[j]--;
            uint64_t carry = 0;
            for (uint32_t i = 0; i < n; i++) {
                __uint128_t sum = (__uint128_t)un[j + i] + vn_shifted[i] + carry;
                un[j + i] = (uint64_t)sum;
                carry = (uint64_t)(sum >> 64);
            }
            un[j + n] += carry;
        }
    }

    bignum_normalize(q);
    if (quotient) *quotient = q;

    /* Remainder: un-shift the remainder in un[0..n-1] */
    if (remainder) {
        eshkol_bignum_t* r = bignum_alloc(arena, n);
        if (r) {
            uint64_t* rl = BIGNUM_LIMBS(r);
            if (shift > 0) {
                uint64_t carry = 0;
                for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
                    rl[i] = (un[i] >> shift) | carry;
                    carry = un[i] << (64 - shift);
                }
            } else {
                memcpy(rl, un, n * sizeof(uint64_t));
            }
            bignum_normalize(r);
        }
        *remainder = r;
    }
}


/* ===== Public API ===== */

extern "C" {

eshkol_bignum_t* eshkol_bignum_from_int64(arena_t* arena, int64_t value) {
    eshkol_bignum_t* bn = bignum_alloc(arena, 1);
    if (!bn) return nullptr;
    if (value < 0) {
        bn->sign = 1;
        /* Handle INT64_MIN carefully: -INT64_MIN overflows */
        if (value == INT64_MIN) {
            BIGNUM_LIMBS(bn)[0] = (uint64_t)INT64_MAX + 1;
        } else {
            BIGNUM_LIMBS(bn)[0] = (uint64_t)(-value);
        }
    } else {
        BIGNUM_LIMBS(bn)[0] = (uint64_t)value;
    }
    return bn;
}

eshkol_bignum_t* eshkol_bignum_from_overflow(arena_t* arena, int64_t a, int64_t b, int op) {
    /* Convert both operands to bignum, then perform the operation */
    eshkol_bignum_t* ba = eshkol_bignum_from_int64(arena, a);
    eshkol_bignum_t* bb = eshkol_bignum_from_int64(arena, b);
    if (!ba || !bb) return nullptr;

    switch (op) {
        case 0: return eshkol_bignum_add(arena, ba, bb);
        case 1: return eshkol_bignum_sub(arena, ba, bb);
        case 2: return eshkol_bignum_mul(arena, ba, bb);
        default: return nullptr;
    }
}

eshkol_bignum_t* eshkol_bignum_from_string(arena_t* arena, const char* str, size_t len) {
    if (!str || len == 0) return nullptr;

    int sign = 0;
    size_t start = 0;
    if (str[0] == '-') { sign = 1; start = 1; }
    else if (str[0] == '+') { start = 1; }

    if (start >= len) return nullptr;

    /* Parse decimal digits: multiply by 10 and add each digit */
    eshkol_bignum_t* result = bignum_alloc(arena, 1);
    if (!result) return nullptr;

    for (size_t i = start; i < len; i++) {
        if (str[i] < '0' || str[i] > '9') return nullptr;
        uint8_t digit = str[i] - '0';

        /* result = result * 10 + digit */
        uint64_t* limbs = BIGNUM_LIMBS(result);
        uint64_t carry = digit;
        for (uint32_t j = 0; j < result->num_limbs; j++) {
            __uint128_t prod = (__uint128_t)limbs[j] * 10 + carry;
            limbs[j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        if (carry > 0) {
            /* Need to grow — allocate new bignum */
            eshkol_bignum_t* grown = bignum_alloc(arena, result->num_limbs + 1);
            if (!grown) return nullptr;
            memcpy(BIGNUM_LIMBS(grown), BIGNUM_LIMBS(result),
                   result->num_limbs * sizeof(uint64_t));
            grown->num_limbs = result->num_limbs + 1;
            BIGNUM_LIMBS(grown)[result->num_limbs] = carry;
            result = grown;
        }
    }

    result->sign = sign;
    bignum_normalize(result);
    return result;
}

eshkol_bignum_t* eshkol_bignum_add(arena_t* arena,
                                    const eshkol_bignum_t* a,
                                    const eshkol_bignum_t* b) {
    if (!a || !b) return nullptr;

    if (a->sign == b->sign) {
        /* Same sign: add magnitudes, keep sign */
        eshkol_bignum_t* result = bignum_add_abs(arena, a, b);
        if (result) result->sign = a->sign;
        return result;
    }

    /* Different signs: subtract smaller magnitude from larger */
    int cmp = bignum_compare_abs(a, b);
    if (cmp == 0) {
        return bignum_alloc(arena, 1); /* zero */
    } else if (cmp > 0) {
        eshkol_bignum_t* result = bignum_sub_abs(arena, a, b);
        if (result) result->sign = a->sign;
        return result;
    } else {
        eshkol_bignum_t* result = bignum_sub_abs(arena, b, a);
        if (result) result->sign = b->sign;
        return result;
    }
}

eshkol_bignum_t* eshkol_bignum_sub(arena_t* arena,
                                    const eshkol_bignum_t* a,
                                    const eshkol_bignum_t* b) {
    if (!a || !b) return nullptr;

    /* a - b = a + (-b) */
    if (a->sign != b->sign) {
        /* Different signs: add magnitudes */
        eshkol_bignum_t* result = bignum_add_abs(arena, a, b);
        if (result) result->sign = a->sign;
        return result;
    }

    /* Same sign: subtract magnitudes */
    int cmp = bignum_compare_abs(a, b);
    if (cmp == 0) {
        return bignum_alloc(arena, 1); /* zero */
    } else if (cmp > 0) {
        eshkol_bignum_t* result = bignum_sub_abs(arena, a, b);
        if (result) result->sign = a->sign;
        return result;
    } else {
        eshkol_bignum_t* result = bignum_sub_abs(arena, b, a);
        if (result) result->sign = 1 - a->sign; /* flip sign */
        return result;
    }
}

eshkol_bignum_t* eshkol_bignum_mul(arena_t* arena,
                                    const eshkol_bignum_t* a,
                                    const eshkol_bignum_t* b) {
    if (!a || !b) return nullptr;

    uint32_t result_limbs = a->num_limbs + b->num_limbs;
    eshkol_bignum_t* result = bignum_alloc(arena, result_limbs);
    if (!result) return nullptr;

    /* Schoolbook multiplication: O(n*m) */
    const uint64_t* la = BIGNUM_LIMBS(a);
    const uint64_t* lb = BIGNUM_LIMBS(b);

    for (uint32_t i = 0; i < b->num_limbs; i++) {
        if (lb[i] != 0) {
            bignum_addmul_limb(BIGNUM_LIMBS(result), result_limbs,
                              la, a->num_limbs, lb[i], i);
        }
    }

    result->sign = (a->sign != b->sign && !eshkol_bignum_is_zero(result)) ? 1 : 0;
    bignum_normalize(result);
    return result;
}

eshkol_bignum_t* eshkol_bignum_div(arena_t* arena,
                                    const eshkol_bignum_t* a,
                                    const eshkol_bignum_t* b) {
    if (!a || !b || eshkol_bignum_is_zero(b)) return nullptr;

    eshkol_bignum_t* q = nullptr;
    bignum_divmod_abs(arena, a, b, &q, nullptr);
    if (q) {
        q->sign = (a->sign != b->sign && !eshkol_bignum_is_zero(q)) ? 1 : 0;
    }
    return q;
}

eshkol_bignum_t* eshkol_bignum_mod(arena_t* arena,
                                    const eshkol_bignum_t* a,
                                    const eshkol_bignum_t* b) {
    if (!a || !b || eshkol_bignum_is_zero(b)) return nullptr;

    eshkol_bignum_t* r = nullptr;
    bignum_divmod_abs(arena, a, b, nullptr, &r);
    if (r) {
        r->sign = (a->sign && !eshkol_bignum_is_zero(r)) ? 1 : 0;
    }
    return r;
}

eshkol_bignum_t* eshkol_bignum_neg(arena_t* arena, const eshkol_bignum_t* a) {
    if (!a) return nullptr;
    eshkol_bignum_t* result = bignum_alloc(arena, a->num_limbs);
    if (!result) return nullptr;
    memcpy(BIGNUM_LIMBS(result), BIGNUM_LIMBS(a), a->num_limbs * sizeof(uint64_t));
    result->num_limbs = a->num_limbs;
    result->sign = eshkol_bignum_is_zero(a) ? 0 : 1 - a->sign;
    return result;
}

/* ===== Mixed int64/bignum ===== */

eshkol_bignum_t* eshkol_bignum_add_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b) {
    eshkol_bignum_t* bb = eshkol_bignum_from_int64(arena, b);
    return bb ? eshkol_bignum_add(arena, a, bb) : nullptr;
}

eshkol_bignum_t* eshkol_bignum_sub_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b) {
    eshkol_bignum_t* bb = eshkol_bignum_from_int64(arena, b);
    return bb ? eshkol_bignum_sub(arena, a, bb) : nullptr;
}

eshkol_bignum_t* eshkol_bignum_mul_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b) {
    eshkol_bignum_t* bb = eshkol_bignum_from_int64(arena, b);
    return bb ? eshkol_bignum_mul(arena, a, bb) : nullptr;
}

/* ===== Comparison ===== */

int eshkol_bignum_compare(const eshkol_bignum_t* a, const eshkol_bignum_t* b) {
    if (!a || !b) return 0;

    /* Different signs */
    if (a->sign != b->sign) {
        if (eshkol_bignum_is_zero(a) && eshkol_bignum_is_zero(b)) return 0;
        return a->sign ? -1 : 1;
    }

    /* Same sign: compare magnitudes */
    int cmp = bignum_compare_abs(a, b);
    return a->sign ? -cmp : cmp; /* Negate if both negative */
}

int eshkol_bignum_compare_int64(const eshkol_bignum_t* a, int64_t b) {
    /* Quick path: single-limb bignum */
    if (a->num_limbs == 1) {
        uint64_t av = BIGNUM_LIMBS(a)[0];
        if (a->sign == 0 && b >= 0) {
            if (av > (uint64_t)b) return 1;
            if (av < (uint64_t)b) return -1;
            return 0;
        }
        if (a->sign == 1 && b < 0) {
            uint64_t babs = (b == INT64_MIN) ? ((uint64_t)INT64_MAX + 1) : (uint64_t)(-b);
            if (av > babs) return -1;
            if (av < babs) return 1;
            return 0;
        }
    }
    /* Full comparison: convert b to bignum */
    /* Use a stack-allocated temp for comparison (no arena needed) */
    uint64_t limb_storage;
    eshkol_bignum_t temp;
    temp.num_limbs = 1;
    if (b < 0) {
        temp.sign = 1;
        limb_storage = (b == INT64_MIN) ? ((uint64_t)INT64_MAX + 1) : (uint64_t)(-b);
    } else {
        temp.sign = 0;
        limb_storage = (uint64_t)b;
    }
    /* Hack: place limb right after struct in stack memory.
     * This works because BIGNUM_LIMBS uses offset from bn pointer.
     * We create a correctly-laid-out struct+limb on stack. */
    struct { eshkol_bignum_t hdr; uint64_t limb; } stack_bn;
    stack_bn.hdr = temp;
    stack_bn.limb = limb_storage;
    return eshkol_bignum_compare(a, &stack_bn.hdr);
}

/* ===== Predicates ===== */

bool eshkol_bignum_is_zero(const eshkol_bignum_t* a) {
    if (!a) return true;
    return (a->num_limbs == 1 && BIGNUM_LIMBS(a)[0] == 0);
}

bool eshkol_bignum_is_negative(const eshkol_bignum_t* a) {
    return a && a->sign && !eshkol_bignum_is_zero(a);
}

bool eshkol_bignum_fits_int64(const eshkol_bignum_t* a, int64_t* out) {
    if (!a) return false;
    if (a->num_limbs > 1) return false;

    uint64_t val = BIGNUM_LIMBS(a)[0];
    if (a->sign == 0) {
        /* Non-negative: must fit in [0, INT64_MAX] */
        if (val <= (uint64_t)INT64_MAX) {
            if (out) *out = (int64_t)val;
            return true;
        }
        return false;
    } else {
        /* Negative: magnitude must fit in [0, INT64_MAX+1] (for INT64_MIN) */
        if (val <= (uint64_t)INT64_MAX + 1) {
            if (out) *out = -(int64_t)val;
            return true;
        }
        return false;
    }
}

/* ===== Conversion ===== */

double eshkol_bignum_to_double(const eshkol_bignum_t* a) {
    if (!a) return 0.0;
    const uint64_t* limbs = BIGNUM_LIMBS(a);

    double result = 0.0;
    double base = 1.0;
    /* 2^64 as a double constant */
    const double pow2_64 = 18446744073709551616.0; /* 2^64 */

    for (uint32_t i = 0; i < a->num_limbs; i++) {
        result += (double)limbs[i] * base;
        base *= pow2_64;
    }

    return a->sign ? -result : result;
}

char* eshkol_bignum_to_string(arena_t* arena, const eshkol_bignum_t* a) {
    if (!a) return nullptr;

    /* Special case: zero */
    if (eshkol_bignum_is_zero(a)) {
        char* str = (char*)arena_allocate_string_with_header(arena, 2);
        if (str) { str[0] = '0'; str[1] = '\0'; }
        return str;
    }

    /* Estimate decimal digits: log10(2^64) ~ 19.3 per limb, plus sign */
    size_t max_digits = (size_t)a->num_limbs * 20 + 2;
    char* buf = (char*)arena_allocate(arena, max_digits);
    if (!buf) return nullptr;

    /* Extract digits by repeatedly dividing by 10 */
    /* Work on a copy of the absolute value */
    eshkol_bignum_t* temp = bignum_alloc(arena, a->num_limbs);
    if (!temp) return nullptr;
    memcpy(BIGNUM_LIMBS(temp), BIGNUM_LIMBS(a), a->num_limbs * sizeof(uint64_t));
    temp->num_limbs = a->num_limbs;
    temp->sign = 0;

    size_t pos = 0;
    while (!eshkol_bignum_is_zero(temp)) {
        uint64_t rem = 0;
        eshkol_bignum_t* next = bignum_div_limb(arena, temp, 10, &rem);
        if (!next) break;
        buf[pos++] = '0' + (char)rem;
        temp = next;
    }

    /* Add sign */
    if (a->sign) buf[pos++] = '-';

    /* Reverse the digits */
    for (size_t i = 0; i < pos / 2; i++) {
        char c = buf[i];
        buf[i] = buf[pos - 1 - i];
        buf[pos - 1 - i] = c;
    }
    buf[pos] = '\0';

    /* Allocate proper string with header for the display system */
    char* result = (char*)arena_allocate_string_with_header(arena, pos + 1);
    if (result) {
        memcpy(result, buf, pos + 1);
    }
    return result;
}

/* ===== Tagged Value Dispatch ===== */

static eshkol_bignum_t* tagged_to_bignum(arena_t* arena, const eshkol_tagged_value_t* val) {
    if (val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val != 0)
        return (eshkol_bignum_t*)(void*)val->data.ptr_val;
    if (val->type == ESHKOL_VALUE_INT64)
        return eshkol_bignum_from_int64(arena, val->data.int_val);
    return nullptr;
}

bool eshkol_is_bignum_tagged(const eshkol_tagged_value_t* val) {
    return ESHKOL_IS_BIGNUM(*val);
}

void eshkol_bignum_binary_tagged(arena_t* arena,
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result) {

    /* op 7 = unary neg (right is ignored) */
    if (op == 7) {
        eshkol_bignum_t* a = tagged_to_bignum(arena, left);
        if (!a) { *result = eshkol_make_int64(0, true); return; }
        eshkol_bignum_t* r = eshkol_bignum_neg(arena, a);
        *result = eshkol_make_ptr((uint64_t)(void*)r, ESHKOL_VALUE_HEAP_PTR);
        result->flags = ESHKOL_VALUE_EXACT_FLAG;
        return;
    }

    eshkol_bignum_t* a = tagged_to_bignum(arena, left);
    eshkol_bignum_t* b = tagged_to_bignum(arena, right);
    if (!a || !b) { *result = eshkol_make_int64(0, true); return; }

    eshkol_bignum_t* r = nullptr;
    switch (op) {
        case 0: r = eshkol_bignum_add(arena, a, b); break;
        case 1: r = eshkol_bignum_sub(arena, a, b); break;
        case 2: r = eshkol_bignum_mul(arena, a, b); break;
        case 3: { /* div: exact if mod==0, else inexact double */
            eshkol_bignum_t* m = eshkol_bignum_mod(arena, a, b);
            if (m && eshkol_bignum_is_zero(m)) {
                r = eshkol_bignum_div(arena, a, b);
            } else {
                double ad = eshkol_bignum_to_double(a);
                double bd = eshkol_bignum_to_double(b);
                *result = eshkol_make_double(ad / bd);
                return;
            }
            break;
        }
        case 4: r = eshkol_bignum_mod(arena, a, b); break;
        case 5: r = eshkol_bignum_div(arena, a, b); break;  /* quotient = truncated div */
        case 6: r = eshkol_bignum_mod(arena, a, b); break;  /* remainder = mod */
        default: { *result = eshkol_make_int64(0, true); return; }
    }

    if (!r) { *result = eshkol_make_int64(0, true); return; }

    /* If result fits in int64, demote to avoid unnecessary bignum overhead */
    int64_t fits;
    if (eshkol_bignum_fits_int64(r, &fits)) {
        *result = eshkol_make_int64(fits, true);
    } else {
        *result = eshkol_make_ptr((uint64_t)(void*)r, ESHKOL_VALUE_HEAP_PTR);
        result->flags = ESHKOL_VALUE_EXACT_FLAG;
    }
}

void eshkol_bignum_compare_tagged(
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result) {

    bool left_is_heap = (left->type == ESHKOL_VALUE_HEAP_PTR);
    bool right_is_heap = (right->type == ESHKOL_VALUE_HEAP_PTR);

    /* If either operand is double, fall back to double comparison */
    if (left->type == ESHKOL_VALUE_DOUBLE || right->type == ESHKOL_VALUE_DOUBLE) {
        double ld = (left->type == ESHKOL_VALUE_DOUBLE) ? left->data.double_val
                   : left_is_heap ? eshkol_bignum_to_double((eshkol_bignum_t*)(void*)left->data.ptr_val)
                   : (double)left->data.int_val;
        double rd = (right->type == ESHKOL_VALUE_DOUBLE) ? right->data.double_val
                   : right_is_heap ? eshkol_bignum_to_double((eshkol_bignum_t*)(void*)right->data.ptr_val)
                   : (double)right->data.int_val;
        bool b = false;
        switch (op) {
            case 0: b = ld < rd; break;
            case 1: b = ld > rd; break;
            case 2: b = ld == rd; break;
            case 3: b = ld <= rd; break;
            case 4: b = ld >= rd; break;
        }
        *result = eshkol_make_int64(b ? 1 : 0, true);
        result->type = ESHKOL_VALUE_BOOL;
        return;
    }

    int cmp;
    if (left_is_heap && right_is_heap) {
        cmp = eshkol_bignum_compare(
            (eshkol_bignum_t*)(void*)left->data.ptr_val,
            (eshkol_bignum_t*)(void*)right->data.ptr_val);
    } else if (left_is_heap) {
        cmp = eshkol_bignum_compare_int64(
            (eshkol_bignum_t*)(void*)left->data.ptr_val, right->data.int_val);
    } else {
        cmp = -eshkol_bignum_compare_int64(
            (eshkol_bignum_t*)(void*)right->data.ptr_val, left->data.int_val);
    }

    bool b = false;
    switch (op) {
        case 0: b = cmp < 0; break;   /* lt */
        case 1: b = cmp > 0; break;   /* gt */
        case 2: b = cmp == 0; break;  /* eq */
        case 3: b = cmp <= 0; break;  /* le */
        case 4: b = cmp >= 0; break;  /* ge */
    }
    *result = eshkol_make_int64(b ? 1 : 0, true);
    result->type = ESHKOL_VALUE_BOOL;
}

/* ===== Display ===== */

void eshkol_bignum_display(const eshkol_bignum_t* a, void* file) {
    if (!a || !file) return;
    FILE* f = (FILE*)file;

    if (eshkol_bignum_is_zero(a)) {
        fprintf(f, "0");
        return;
    }

    if (a->sign) fprintf(f, "-");

    /* For small bignums (1 limb), use direct printf */
    if (a->num_limbs == 1) {
        fprintf(f, "%llu", (unsigned long long)BIGNUM_LIMBS(a)[0]);
        return;
    }

    /* For larger bignums, we need to convert to decimal.
     * Use a simple stack buffer for reasonable sizes. */
    char stack_buf[256];
    size_t max_digits = (size_t)a->num_limbs * 20 + 2;

    if (max_digits <= sizeof(stack_buf)) {
        /* Extract digits into stack buffer */
        /* Working copy of limbs — max 12 limbs when max_digits <= 256 */
        uint64_t work[13];
        memcpy(work, BIGNUM_LIMBS(a), a->num_limbs * sizeof(uint64_t));
        uint32_t work_limbs = a->num_limbs;

        size_t pos = 0;
        while (work_limbs > 0) {
            /* Check if all zero */
            bool all_zero = true;
            for (uint32_t i = 0; i < work_limbs; i++) {
                if (work[i] != 0) { all_zero = false; break; }
            }
            if (all_zero) break;

            /* Divide by 10 */
            __uint128_t rem = 0;
            for (int32_t i = (int32_t)work_limbs - 1; i >= 0; i--) {
                __uint128_t cur = (rem << 64) | work[i];
                work[i] = (uint64_t)(cur / 10);
                rem = cur % 10;
            }
            stack_buf[pos++] = '0' + (char)(uint64_t)rem;

            /* Trim leading zero limbs */
            while (work_limbs > 0 && work[work_limbs - 1] == 0) work_limbs--;
        }

        /* Print digits in reverse */
        for (size_t i = pos; i > 0; i--) {
            fputc(stack_buf[i - 1], f);
        }
    } else {
        fprintf(f, "<bignum:%u-limbs>", a->num_limbs);
    }
}

} /* extern "C" */
