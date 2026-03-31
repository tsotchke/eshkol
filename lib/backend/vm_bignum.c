/**
 * @file vm_bignum.c
 * @brief Arbitrary-precision integer runtime for the Eshkol bytecode VM.
 *
 * Sign-magnitude representation with base-2^32 limbs (little-endian).
 * All limb arrays are arena-allocated via vm_arena.h — no GC, no free().
 *
 * Native call IDs 350–369 (VM_NATIVE_BIGNUM_BASE).
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "vm_arena.h"
#include "vm_numeric.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Forward declarations
 * ═══════════════════════════════════════════════════════════════════════════ */

static VmBignum* bignum_create(VmRegionStack* rs, int capacity);
static VmBignum* bignum_from_int64(VmRegionStack* rs, int64_t v);
static VmBignum* bignum_from_uint64(VmRegionStack* rs, uint64_t v);
static VmBignum* bignum_from_string(VmRegionStack* rs, const char* s);
static VmBignum* bignum_copy(VmRegionStack* rs, const VmBignum* src);
static void      bignum_normalize(VmBignum* b);

static VmBignum* bignum_add(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_sub(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_mul(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_div(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_mod(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static void      bignum_divmod(VmRegionStack* rs, const VmBignum* a, const VmBignum* b,
                               VmBignum** q_out, VmBignum** r_out);
static VmBignum* bignum_neg(VmRegionStack* rs, const VmBignum* a);
static VmBignum* bignum_abs_val(VmRegionStack* rs, const VmBignum* a);
static VmBignum* bignum_pow(VmRegionStack* rs, const VmBignum* base, uint64_t exp);
static VmBignum* bignum_gcd(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);

static int       bignum_compare(const VmBignum* a, const VmBignum* b);
static int       bignum_compare_magnitude(const VmBignum* a, const VmBignum* b);
static int       bignum_is_zero(const VmBignum* b);
static int       bignum_sign(const VmBignum* b);

static double    bignum_to_double(const VmBignum* b);
static int64_t   bignum_to_int64(const VmBignum* b, int* overflow);
static char*     bignum_to_string(VmRegionStack* rs, const VmBignum* b);

static VmBignum* bignum_bitwise_and(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_bitwise_or(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_bitwise_xor(VmRegionStack* rs, const VmBignum* a, const VmBignum* b);
static VmBignum* bignum_bitwise_not(VmRegionStack* rs, const VmBignum* a);
static VmBignum* bignum_shift_left(VmRegionStack* rs, const VmBignum* a, int shift);
static VmBignum* bignum_shift_right(VmRegionStack* rs, const VmBignum* a, int shift);

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal helpers: magnitude-only addition/subtraction (unsigned)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Add magnitudes of a and b. Result is always non-negative.
 */
static VmBignum* mag_add(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    int max_n = (a->n_limbs > b->n_limbs ? a->n_limbs : b->n_limbs) + 1;
    VmBignum* r = bignum_create(rs, max_n);
    if (!r) return NULL;

    uint64_t carry = 0;
    int i;
    for (i = 0; i < max_n; i++) {
        uint64_t la = (i < a->n_limbs) ? a->limbs[i] : 0;
        uint64_t lb = (i < b->n_limbs) ? b->limbs[i] : 0;
        uint64_t sum = la + lb + carry;
        r->limbs[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
        carry = sum >> 32;
    }
    r->n_limbs = max_n;
    bignum_normalize(r);
    return r;
}

/**
 * Subtract magnitude of b from a. Precondition: |a| >= |b|.
 * Result magnitude only (sign set by caller).
 */
static VmBignum* mag_sub(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    int max_n = a->n_limbs;
    VmBignum* r = bignum_create(rs, max_n);
    if (!r) return NULL;

    int64_t borrow = 0;
    int i;
    for (i = 0; i < max_n; i++) {
        int64_t la = (i < a->n_limbs) ? (int64_t)a->limbs[i] : 0;
        int64_t lb = (i < b->n_limbs) ? (int64_t)b->limbs[i] : 0;
        int64_t diff = la - lb - borrow;
        if (diff < 0) {
            diff += (int64_t)1 << 32;
            borrow = 1;
        } else {
            borrow = 0;
        }
        r->limbs[i] = (uint32_t)(diff & 0xFFFFFFFFULL);
    }
    r->n_limbs = max_n;
    bignum_normalize(r);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core construction
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Create a zero bignum with the given limb capacity.
 */
static VmBignum* bignum_create(VmRegionStack* rs, int capacity) {
    if (capacity < 1) capacity = 1;
    VmBignum* b = (VmBignum*)vm_alloc(rs, sizeof(VmBignum));
    if (!b) return NULL;
    b->limbs = (uint32_t*)vm_alloc(rs, (size_t)capacity * sizeof(uint32_t));
    if (!b->limbs) return NULL;
    memset(b->limbs, 0, (size_t)capacity * sizeof(uint32_t));
    b->sign = 0;
    b->n_limbs = 0;
    b->capacity = capacity;
    return b;
}

/**
 * Create bignum from int64_t.
 */
static VmBignum* bignum_from_int64(VmRegionStack* rs, int64_t v) {
    VmBignum* b = bignum_create(rs, 2);
    if (!b) return NULL;

    if (v == 0) {
        b->sign = 0;
        b->n_limbs = 0;
        return b;
    }

    uint64_t mag;
    if (v < 0) {
        b->sign = -1;
        /* Handle INT64_MIN carefully */
        if (v == INT64_MIN) {
            mag = (uint64_t)INT64_MAX + 1ULL;
        } else {
            mag = (uint64_t)(-v);
        }
    } else {
        b->sign = 1;
        mag = (uint64_t)v;
    }

    b->limbs[0] = (uint32_t)(mag & 0xFFFFFFFFULL);
    uint32_t hi = (uint32_t)(mag >> 32);
    if (hi) {
        b->limbs[1] = hi;
        b->n_limbs = 2;
    } else {
        b->n_limbs = 1;
    }
    return b;
}

/**
 * Create bignum from uint64_t.
 */
static VmBignum* bignum_from_uint64(VmRegionStack* rs, uint64_t v) {
    VmBignum* b = bignum_create(rs, 3);
    if (!b) return NULL;

    if (v == 0) {
        b->sign = 0;
        b->n_limbs = 0;
        return b;
    }

    b->sign = 1;
    b->limbs[0] = (uint32_t)(v & 0xFFFFFFFFULL);
    uint32_t hi = (uint32_t)(v >> 32);
    if (hi) {
        b->limbs[1] = hi;
        b->n_limbs = 2;
    } else {
        b->n_limbs = 1;
    }
    return b;
}

/**
 * Create bignum from decimal string.
 * Supports optional leading '-' or '+'.
 * Horner's method: result = result * 10 + digit.
 * For efficiency, processes 9 digits at a time (base 10^9).
 */
static VmBignum* bignum_from_string(VmRegionStack* rs, const char* s) {
    if (!s || !*s) return bignum_from_int64(rs, 0);

    int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') { s++; }

    /* Skip leading zeros */
    while (*s == '0' && *(s + 1) != '\0') s++;

    int len = (int)strlen(s);
    if (len == 0) return bignum_from_int64(rs, 0);

    /* Estimate capacity: each decimal digit < log2(10) ≈ 3.32 bits.
     * So n digits needs about n * 3.32 / 32 + 1 limbs. */
    int est_limbs = (len * 34 + 319) / 320 + 2;  /* ceil(len * 3.32 / 32) + slack */
    VmBignum* r = bignum_create(rs, est_limbs);
    if (!r) return NULL;
    r->sign = 0;
    r->n_limbs = 0;

    /* Process 9 digits at a time (10^9 fits in uint32_t: 1,000,000,000 < 2^32) */
    const uint32_t CHUNK_BASE = 1000000000U;
    int pos = 0;

    while (pos < len) {
        /* Determine chunk size for this iteration */
        int remaining = len - pos;
        int chunk_len = remaining % 9;
        if (chunk_len == 0) chunk_len = 9;
        if (pos > 0) chunk_len = 9; /* after first chunk, always 9 */

        /* Parse chunk */
        uint32_t chunk_val = 0;
        uint32_t chunk_multiplier = 1;
        int k;
        for (k = 0; k < chunk_len; k++) {
            char c = s[pos + k];
            if (c < '0' || c > '9') {
                /* Invalid character */
                return bignum_from_int64(rs, 0);
            }
            chunk_val = chunk_val * 10 + (uint32_t)(c - '0');
            if (k > 0 || pos > 0) {
                /* (only to compute multiplier for the actual base) */
            }
        }

        /* Compute the base for this chunk: 10^chunk_len */
        chunk_multiplier = 1;
        for (k = 0; k < chunk_len; k++) chunk_multiplier *= 10;

        /* r = r * chunk_multiplier + chunk_val */
        /* Multiply all limbs by chunk_multiplier and add chunk_val */
        uint64_t carry = chunk_val;
        int i;
        for (i = 0; i < r->n_limbs; i++) {
            uint64_t prod = (uint64_t)r->limbs[i] * chunk_multiplier + carry;
            r->limbs[i] = (uint32_t)(prod & 0xFFFFFFFFULL);
            carry = prod >> 32;
        }
        while (carry > 0) {
            if (r->n_limbs >= r->capacity) {
                /* Grow — arena-allocate new array */
                int new_cap = r->capacity * 2;
                uint32_t* new_limbs = (uint32_t*)vm_alloc(rs, (size_t)new_cap * sizeof(uint32_t));
                if (!new_limbs) return NULL;
                memcpy(new_limbs, r->limbs, (size_t)r->n_limbs * sizeof(uint32_t));
                memset(new_limbs + r->n_limbs, 0, (size_t)(new_cap - r->n_limbs) * sizeof(uint32_t));
                r->limbs = new_limbs;
                r->capacity = new_cap;
            }
            r->limbs[r->n_limbs] = (uint32_t)(carry & 0xFFFFFFFFULL);
            r->n_limbs++;
            carry >>= 32;
        }

        pos += chunk_len;
    }

    bignum_normalize(r);
    if (r->n_limbs > 0) {
        r->sign = neg ? -1 : 1;
    }
    return r;
}

/**
 * Deep copy a bignum (into the active arena).
 */
static VmBignum* bignum_copy(VmRegionStack* rs, const VmBignum* src) {
    if (!src) return NULL;
    VmBignum* b = bignum_create(rs, src->n_limbs > 0 ? src->n_limbs : 1);
    if (!b) return NULL;
    b->sign = src->sign;
    b->n_limbs = src->n_limbs;
    if (src->n_limbs > 0) {
        memcpy(b->limbs, src->limbs, (size_t)src->n_limbs * sizeof(uint32_t));
    }
    return b;
}

/**
 * Strip leading zero limbs. If all limbs are zero, set sign=0 and n_limbs=0.
 */
static void bignum_normalize(VmBignum* b) {
    while (b->n_limbs > 0 && b->limbs[b->n_limbs - 1] == 0) {
        b->n_limbs--;
    }
    if (b->n_limbs == 0) b->sign = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Comparison
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compare magnitudes only: returns -1, 0, or +1.
 */
static int bignum_compare_magnitude(const VmBignum* a, const VmBignum* b) {
    if (a->n_limbs != b->n_limbs) {
        return a->n_limbs > b->n_limbs ? 1 : -1;
    }
    /* Same number of limbs — compare MSB to LSB */
    int i;
    for (i = a->n_limbs - 1; i >= 0; i--) {
        if (a->limbs[i] != b->limbs[i]) {
            return a->limbs[i] > b->limbs[i] ? 1 : -1;
        }
    }
    return 0;
}

/**
 * Full signed comparison: returns -1, 0, or +1.
 */
static int bignum_compare(const VmBignum* a, const VmBignum* b) {
    /* Handle zeros */
    int sa = bignum_is_zero(a) ? 0 : a->sign;
    int sb = bignum_is_zero(b) ? 0 : b->sign;

    if (sa != sb) {
        if (sa > sb) return 1;
        return -1;
    }
    if (sa == 0) return 0; /* both zero */

    int mag = bignum_compare_magnitude(a, b);
    /* If both negative, larger magnitude means smaller value */
    return sa > 0 ? mag : -mag;
}

static int bignum_is_zero(const VmBignum* b) {
    return b->n_limbs == 0 || b->sign == 0;
}

static int bignum_sign(const VmBignum* b) {
    return bignum_is_zero(b) ? 0 : b->sign;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Arithmetic: add, sub, mul, div, mod, neg, abs, pow, gcd
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Signed addition.
 */
static VmBignum* bignum_add(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a)) return bignum_copy(rs, b);
    if (bignum_is_zero(b)) return bignum_copy(rs, a);

    if (a->sign == b->sign) {
        /* Same sign: add magnitudes, keep sign */
        VmBignum* r = mag_add(rs, a, b);
        if (r) r->sign = a->sign;
        return r;
    }

    /* Different signs: subtract smaller magnitude from larger */
    int cmp = bignum_compare_magnitude(a, b);
    if (cmp == 0) {
        /* a + (-a) = 0 */
        return bignum_from_int64(rs, 0);
    }
    if (cmp > 0) {
        VmBignum* r = mag_sub(rs, a, b);
        if (r) r->sign = a->sign;
        return r;
    } else {
        VmBignum* r = mag_sub(rs, b, a);
        if (r) r->sign = b->sign;
        return r;
    }
}

/**
 * Signed subtraction: a - b.
 */
static VmBignum* bignum_sub(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(b)) return bignum_copy(rs, a);
    if (bignum_is_zero(a)) return bignum_neg(rs, b);

    if (a->sign != b->sign) {
        /* Different signs: add magnitudes */
        VmBignum* r = mag_add(rs, a, b);
        if (r) r->sign = a->sign;
        return r;
    }

    /* Same sign: subtract magnitudes */
    int cmp = bignum_compare_magnitude(a, b);
    if (cmp == 0) {
        return bignum_from_int64(rs, 0);
    }
    if (cmp > 0) {
        VmBignum* r = mag_sub(rs, a, b);
        if (r) r->sign = a->sign;
        return r;
    } else {
        VmBignum* r = mag_sub(rs, b, a);
        if (r) r->sign = -(a->sign);
        return r;
    }
}

/**
 * Negation: -a.
 */
static VmBignum* bignum_neg(VmRegionStack* rs, const VmBignum* a) {
    VmBignum* r = bignum_copy(rs, a);
    if (r && !bignum_is_zero(r)) r->sign = -(r->sign);
    return r;
}

/**
 * Absolute value: |a|.
 */
static VmBignum* bignum_abs_val(VmRegionStack* rs, const VmBignum* a) {
    VmBignum* r = bignum_copy(rs, a);
    if (r && r->sign < 0) r->sign = 1;
    return r;
}

/**
 * Schoolbook multiplication: O(n*m).
 */
static VmBignum* bignum_mul(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a) || bignum_is_zero(b)) {
        return bignum_from_int64(rs, 0);
    }

    int rn = a->n_limbs + b->n_limbs;
    VmBignum* r = bignum_create(rs, rn);
    if (!r) return NULL;
    r->n_limbs = rn;

    int i, j;
    for (i = 0; i < a->n_limbs; i++) {
        uint64_t carry = 0;
        for (j = 0; j < b->n_limbs; j++) {
            uint64_t prod = (uint64_t)a->limbs[i] * (uint64_t)b->limbs[j]
                          + (uint64_t)r->limbs[i + j] + carry;
            r->limbs[i + j] = (uint32_t)(prod & 0xFFFFFFFFULL);
            carry = prod >> 32;
        }
        r->limbs[i + j] += (uint32_t)carry;
    }

    r->sign = (a->sign == b->sign) ? 1 : -1;
    bignum_normalize(r);
    return r;
}

/**
 * Single-limb division helper.
 * Divides |a| by a single uint32_t divisor d.
 * Returns quotient; *remainder receives the remainder.
 */
static VmBignum* mag_div_single(VmRegionStack* rs, const VmBignum* a, uint32_t d, uint32_t* remainder) {
    if (d == 0) {
        fprintf(stderr, "ERROR: bignum division by zero\n");
        return NULL;
    }

    VmBignum* q = bignum_create(rs, a->n_limbs);
    if (!q) return NULL;
    q->n_limbs = a->n_limbs;
    q->sign = 1;

    uint64_t rem = 0;
    int i;
    for (i = a->n_limbs - 1; i >= 0; i--) {
        uint64_t cur = (rem << 32) | (uint64_t)a->limbs[i];
        q->limbs[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }

    if (remainder) *remainder = (uint32_t)rem;
    bignum_normalize(q);
    return q;
}

/**
 * Knuth Algorithm D: multi-limb division.
 * Computes |a| / |b| → quotient and remainder.
 * Precondition: b != 0 and b->n_limbs >= 2.
 */
static void mag_div_knuth(VmRegionStack* rs, const VmBignum* a, const VmBignum* b,
                          VmBignum** q_out, VmBignum** r_out) {
    int n = b->n_limbs;
    int m = a->n_limbs - n;

    if (m < 0) {
        /* |a| < |b| → quotient = 0, remainder = a */
        if (q_out) *q_out = bignum_from_int64(rs, 0);
        if (r_out) *r_out = bignum_copy(rs, a);
        return;
    }

    /* D1: Normalize — multiply a and b by d = 2^s where s makes b's MSB >= 2^31 */
    uint32_t msb = b->limbs[n - 1];
    int s = 0;
    {
        uint32_t t = msb;
        while (t < 0x80000000U) { t <<= 1; s++; }
    }

    /* Create normalized copies */
    int un_len = a->n_limbs + 1;
    uint32_t* un = (uint32_t*)vm_alloc(rs, (size_t)(un_len) * sizeof(uint32_t));
    uint32_t* vn = (uint32_t*)vm_alloc(rs, (size_t)n * sizeof(uint32_t));
    if (!un || !vn) {
        if (q_out) *q_out = bignum_from_int64(rs, 0);
        if (r_out) *r_out = bignum_from_int64(rs, 0);
        return;
    }
    memset(un, 0, (size_t)(un_len) * sizeof(uint32_t));

    /* Shift b left by s bits */
    if (s > 0) {
        int i;
        uint32_t carry = 0;
        for (i = 0; i < n; i++) {
            uint64_t tmp = ((uint64_t)b->limbs[i] << s) | carry;
            vn[i] = (uint32_t)(tmp & 0xFFFFFFFFULL);
            carry = (uint32_t)(tmp >> 32);
        }
    } else {
        memcpy(vn, b->limbs, (size_t)n * sizeof(uint32_t));
    }

    /* Shift a left by s bits */
    if (s > 0) {
        int i;
        uint32_t carry = 0;
        for (i = 0; i < a->n_limbs; i++) {
            uint64_t tmp = ((uint64_t)a->limbs[i] << s) | carry;
            un[i] = (uint32_t)(tmp & 0xFFFFFFFFULL);
            carry = (uint32_t)(tmp >> 32);
        }
        un[a->n_limbs] = carry;
    } else {
        memcpy(un, a->limbs, (size_t)a->n_limbs * sizeof(uint32_t));
        un[a->n_limbs] = 0;
    }

    /* Quotient has at most m+1 limbs */
    VmBignum* q = bignum_create(rs, m + 1);
    if (!q) {
        if (q_out) *q_out = bignum_from_int64(rs, 0);
        if (r_out) *r_out = bignum_from_int64(rs, 0);
        return;
    }
    q->n_limbs = m + 1;
    q->sign = 1;

    /* D2-D7: Main loop — compute each quotient digit q[j] for j = m..0 */
    int j;
    for (j = m; j >= 0; j--) {
        /* D3: Estimate q_hat = (un[j+n]*B + un[j+n-1]) / vn[n-1] */
        uint64_t two_digits = ((uint64_t)un[j + n] << 32) | (uint64_t)un[j + n - 1];
        uint64_t q_hat = two_digits / vn[n - 1];
        uint64_t r_hat = two_digits % vn[n - 1];

        /* Refine estimate */
        while (q_hat >= (1ULL << 32) ||
               (n >= 2 && q_hat * (uint64_t)vn[n - 2] > ((r_hat << 32) | (uint64_t)un[j + n - 2]))) {
            q_hat--;
            r_hat += vn[n - 1];
            if (r_hat >= (1ULL << 32)) break;
        }

        /* D4: Multiply and subtract: un[j..j+n] -= q_hat * vn[0..n-1] */
        int64_t borrow = 0;
        int i;
        for (i = 0; i < n; i++) {
            uint64_t prod = q_hat * (uint64_t)vn[i];
            int64_t diff = (int64_t)un[j + i] - (int64_t)(prod & 0xFFFFFFFFULL) - borrow;
            un[j + i] = (uint32_t)(diff & 0xFFFFFFFFULL);
            borrow = (int64_t)(prod >> 32) - (diff >> 32);
        }
        int64_t diff = (int64_t)un[j + n] - borrow;
        un[j + n] = (uint32_t)(diff & 0xFFFFFFFFULL);

        /* D5: Store quotient digit */
        q->limbs[j] = (uint32_t)q_hat;

        /* D6: If we subtracted too much, add back */
        if (diff < 0) {
            q->limbs[j]--;
            uint64_t carry = 0;
            for (i = 0; i < n; i++) {
                uint64_t sum = (uint64_t)un[j + i] + (uint64_t)vn[i] + carry;
                un[j + i] = (uint32_t)(sum & 0xFFFFFFFFULL);
                carry = sum >> 32;
            }
            un[j + n] += (uint32_t)carry;
        }
    }

    bignum_normalize(q);
    if (q_out) *q_out = q;

    /* D8: Unnormalize remainder — shift un right by s bits */
    if (r_out) {
        VmBignum* rem = bignum_create(rs, n);
        if (!rem) { *r_out = bignum_from_int64(rs, 0); return; }
        rem->n_limbs = n;
        rem->sign = 1;
        if (s > 0) {
            int i;
            uint32_t carry = 0;
            for (i = n - 1; i >= 0; i--) {
                uint32_t new_carry = un[i] << (32 - s);
                rem->limbs[i] = (un[i] >> s) | carry;
                carry = new_carry;
            }
        } else {
            memcpy(rem->limbs, un, (size_t)n * sizeof(uint32_t));
        }
        bignum_normalize(rem);
        *r_out = rem;
    }
}

/**
 * Division and modulo (signed, truncated toward zero — R7RS quotient semantics).
 * Signs: quotient sign = sign(a) * sign(b), remainder sign = sign(a).
 */
static void bignum_divmod(VmRegionStack* rs, const VmBignum* a, const VmBignum* b,
                          VmBignum** q_out, VmBignum** r_out) {
    if (bignum_is_zero(b)) {
        fprintf(stderr, "ERROR: bignum division by zero\n");
        if (q_out) *q_out = bignum_from_int64(rs, 0);
        if (r_out) *r_out = bignum_from_int64(rs, 0);
        return;
    }

    if (bignum_is_zero(a)) {
        if (q_out) *q_out = bignum_from_int64(rs, 0);
        if (r_out) *r_out = bignum_from_int64(rs, 0);
        return;
    }

    int result_sign_q = (a->sign == b->sign) ? 1 : -1;
    int result_sign_r = a->sign;

    VmBignum* q = NULL;
    VmBignum* r = NULL;

    if (b->n_limbs == 1) {
        /* Fast path: single-limb divisor */
        uint32_t rem32 = 0;
        q = mag_div_single(rs, a, b->limbs[0], &rem32);
        if (r_out) {
            r = bignum_from_uint64(rs, rem32);
        }
    } else {
        /* Multi-limb: Knuth Algorithm D */
        int cmp = bignum_compare_magnitude(a, b);
        if (cmp < 0) {
            q = bignum_from_int64(rs, 0);
            r = bignum_copy(rs, a);
        } else if (cmp == 0) {
            q = bignum_from_int64(rs, 1);
            r = bignum_from_int64(rs, 0);
        } else {
            mag_div_knuth(rs, a, b, &q, r_out ? &r : NULL);
        }
    }

    /* Apply signs */
    if (q && !bignum_is_zero(q)) q->sign = result_sign_q;
    if (r && !bignum_is_zero(r)) r->sign = result_sign_r;

    if (q_out) *q_out = q;
    if (r_out) *r_out = r;
}

/**
 * Quotient: a / b (truncated toward zero).
 */
static VmBignum* bignum_div(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    VmBignum* q = NULL;
    bignum_divmod(rs, a, b, &q, NULL);
    return q;
}

/**
 * Remainder: a mod b (sign matches a).
 */
static VmBignum* bignum_mod(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    VmBignum* r = NULL;
    bignum_divmod(rs, a, b, NULL, &r);
    return r;
}

/**
 * Power via repeated squaring: base^exp.
 * exp is unsigned — caller handles negative exponents (→ rational).
 */
static VmBignum* bignum_pow(VmRegionStack* rs, const VmBignum* base, uint64_t exp) {
    if (exp == 0) return bignum_from_int64(rs, 1);
    if (exp == 1) return bignum_copy(rs, base);
    if (bignum_is_zero(base)) return bignum_from_int64(rs, 0);

    VmBignum* result = bignum_from_int64(rs, 1);
    VmBignum* b = bignum_copy(rs, base);
    b->sign = 1; /* work with magnitude */

    uint64_t e = exp;
    while (e > 0) {
        if (e & 1) {
            result = bignum_mul(rs, result, b);
            if (!result) return NULL;
        }
        e >>= 1;
        if (e > 0) {
            b = bignum_mul(rs, b, b);
            if (!b) return NULL;
        }
    }

    /* Sign: negative base, odd exponent → negative */
    if (base->sign < 0 && (exp & 1)) {
        result->sign = -1;
    }
    return result;
}

/**
 * GCD via binary algorithm (Stein's algorithm), using magnitude only.
 * Result is always non-negative.
 */
static VmBignum* bignum_gcd(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a)) return bignum_abs_val(rs, b);
    if (bignum_is_zero(b)) return bignum_abs_val(rs, a);

    /* Use Euclidean algorithm: gcd(a, b) = gcd(b, a mod b) */
    VmBignum* x = bignum_abs_val(rs, a);
    VmBignum* y = bignum_abs_val(rs, b);

    while (!bignum_is_zero(y)) {
        VmBignum* r = bignum_mod(rs, x, y);
        if (!r) return x;
        if (!bignum_is_zero(r)) r->sign = 1; /* mod can give negative, force positive */
        x = y;
        y = r;
    }
    return x;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Conversion
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Convert bignum to double (IEEE 754).
 * Extracts the top 53 significant bits with correct exponent.
 */
static double bignum_to_double(const VmBignum* b) {
    if (bignum_is_zero(b)) return 0.0;

    /* Find the bit position of the MSB */
    int top_limb = b->n_limbs - 1;
    uint32_t msb_limb = b->limbs[top_limb];
    int msb_bit = 31;
    while (msb_bit > 0 && !(msb_limb & (1U << msb_bit))) msb_bit--;

    /* Total bit count of the magnitude */
    int total_bits = top_limb * 32 + msb_bit + 1;

    if (total_bits <= 53) {
        /* Fits exactly — reconstruct from limbs */
        double result = 0.0;
        double base = 1.0;
        int i;
        for (i = 0; i < b->n_limbs; i++) {
            result += (double)b->limbs[i] * base;
            base *= 4294967296.0; /* 2^32 */
        }
        return b->sign < 0 ? -result : result;
    }

    /* Extract top 53 bits across limbs */
    /* The top bit is at position total_bits-1 (0-indexed from LSB of entire number).
     * We need bits [total_bits-1 .. total_bits-53].
     * bit position p maps to limb[p/32] bit (p%32).
     */
    uint64_t mantissa = 0;
    int bits_collected = 0;
    int bit_pos = total_bits - 1;

    while (bits_collected < 53 && bit_pos >= 0) {
        int limb_idx = bit_pos / 32;
        int bit_idx = bit_pos % 32;
        if (limb_idx < b->n_limbs && (b->limbs[limb_idx] & (1U << bit_idx))) {
            mantissa |= (1ULL << (52 - bits_collected));
        }
        bits_collected++;
        bit_pos--;
    }

    /* Round to nearest even: check the next bit */
    if (bit_pos >= 0) {
        int limb_idx = bit_pos / 32;
        int bit_idx = bit_pos % 32;
        if (limb_idx < b->n_limbs && (b->limbs[limb_idx] & (1U << bit_idx))) {
            /* Round bit is 1 — check sticky bits and parity */
            int sticky = 0;
            int sp;
            for (sp = bit_pos - 1; sp >= 0 && !sticky; sp--) {
                int sl = sp / 32;
                int sb = sp % 32;
                if (sl < b->n_limbs && (b->limbs[sl] & (1U << sb))) sticky = 1;
            }
            if (sticky || (mantissa & 1)) {
                mantissa++;
                if (mantissa >= (1ULL << 53)) {
                    mantissa >>= 1;
                    total_bits++;
                }
            }
        }
    }

    int exponent = total_bits - 1; /* unbiased exponent */
    /* ldexp(mantissa, exponent - 52) = mantissa * 2^(exponent-52) */
    double result = ldexp((double)mantissa, exponent - 52);
    return b->sign < 0 ? -result : result;
}

/**
 * Convert bignum to int64_t. Sets *overflow=1 if out of range.
 */
static int64_t bignum_to_int64(const VmBignum* b, int* overflow) {
    if (overflow) *overflow = 0;
    if (bignum_is_zero(b)) return 0;

    if (b->n_limbs > 2) {
        if (overflow) *overflow = 1;
        return b->sign > 0 ? INT64_MAX : INT64_MIN;
    }

    uint64_t mag = b->limbs[0];
    if (b->n_limbs == 2) {
        mag |= (uint64_t)b->limbs[1] << 32;
    }

    if (b->sign > 0) {
        if (mag > (uint64_t)INT64_MAX) {
            if (overflow) *overflow = 1;
            return INT64_MAX;
        }
        return (int64_t)mag;
    } else {
        /* INT64_MIN = -9223372036854775808, magnitude = 2^63 */
        if (mag > (uint64_t)INT64_MAX + 1ULL) {
            if (overflow) *overflow = 1;
            return INT64_MIN;
        }
        if (mag == (uint64_t)INT64_MAX + 1ULL) return INT64_MIN;
        return -(int64_t)mag;
    }
}

/**
 * Convert bignum to decimal string.
 * Uses repeated division by 10^9 for efficiency.
 * Returns arena-allocated null-terminated string.
 */
static char* bignum_to_string(VmRegionStack* rs, const VmBignum* b) {
    if (bignum_is_zero(b)) {
        char* s = (char*)vm_alloc(rs, 2);
        if (!s) return NULL;
        s[0] = '0'; s[1] = '\0';
        return s;
    }

    /* Estimate string length: each limb ≤ 32 bits ≈ 9.63 decimal digits.
     * Plus sign and null terminator. */
    int est_len = b->n_limbs * 10 + 2;
    char* buf = (char*)vm_alloc(rs, (size_t)est_len);
    if (!buf) return NULL;

    /* Work with a copy of the magnitude */
    VmBignum* tmp = bignum_abs_val(rs, b);
    if (!tmp) return NULL;

    /* Extract groups of 9 digits using division by 10^9 */
    const uint32_t DIVISOR = 1000000000U; /* 10^9 */
    /* We'll collect chunks in reverse, then write the final string */
    int max_chunks = b->n_limbs * 2 + 2; /* generous estimate */
    uint32_t* chunks = (uint32_t*)vm_alloc(rs, (size_t)max_chunks * sizeof(uint32_t));
    if (!chunks) return NULL;
    int n_chunks = 0;

    while (!bignum_is_zero(tmp)) {
        uint32_t rem = 0;
        VmBignum* q = mag_div_single(rs, tmp, DIVISOR, &rem);
        if (!q) return NULL;
        chunks[n_chunks++] = rem;
        tmp = q;
    }

    /* Write to buffer: most significant chunk first (no leading zeros),
     * then subsequent chunks zero-padded to 9 digits */
    int pos = 0;
    if (b->sign < 0) buf[pos++] = '-';

    int c;
    for (c = n_chunks - 1; c >= 0; c--) {
        uint32_t chunk = chunks[c];
        if (c == n_chunks - 1) {
            /* First chunk: no leading zeros */
            char tmp_buf[16];
            int tlen = sprintf(tmp_buf, "%u", chunk);
            memcpy(buf + pos, tmp_buf, (size_t)tlen);
            pos += tlen;
        } else {
            /* Subsequent chunks: zero-padded to 9 digits */
            char tmp_buf[16];
            sprintf(tmp_buf, "%09u", chunk);
            memcpy(buf + pos, tmp_buf, 9);
            pos += 9;
        }
    }
    buf[pos] = '\0';
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Bitwise operations
 * Operate on magnitude only (unsigned interpretation).
 * For bitwise-not, we compute ~a = -(a+1) per two's complement convention.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Bitwise AND of magnitudes.
 * Result sign: both positive → positive. Otherwise, use two's complement
 * interpretation: for simplicity, we only support non-negative operands
 * (Scheme's bitwise ops are defined for exact integers; negative handled
 * via two's complement).
 */
static VmBignum* bignum_bitwise_and(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a) || bignum_is_zero(b)) return bignum_from_int64(rs, 0);

    /* Two's complement for negative numbers:
     * -n is represented as infinite 1's extension of ~(n-1) */
    /* For simplicity with non-negative, just AND magnitudes */
    if (a->sign > 0 && b->sign > 0) {
        int min_n = a->n_limbs < b->n_limbs ? a->n_limbs : b->n_limbs;
        VmBignum* r = bignum_create(rs, min_n);
        if (!r) return NULL;
        r->n_limbs = min_n;
        r->sign = 1;
        int i;
        for (i = 0; i < min_n; i++) {
            r->limbs[i] = a->limbs[i] & b->limbs[i];
        }
        bignum_normalize(r);
        return r;
    }

    /* Full two's complement support for mixed signs */
    /* -a in two's complement: ~(a-1) with infinite 1-extension */
    /* AND with infinite extension:
     * (+,+): min limbs, AND each
     * (+,-): max a limbs, a & ~(b-1), finite result
     * (-,+): max b limbs, ~(a-1) & b, finite result
     * (-,-): max limbs, ~(a-1) & ~(b-1) = ~((a-1)|(b-1)), result is negative
     */
    int max_n = a->n_limbs > b->n_limbs ? a->n_limbs : b->n_limbs;

    /* Compute two's complement representations */
    /* Helper: compute (mag - 1) for negative numbers */
    VmBignum* am1 = NULL;
    VmBignum* bm1 = NULL;
    VmBignum* one = bignum_from_int64(rs, 1);

    if (a->sign < 0) {
        VmBignum* a_abs = bignum_abs_val(rs, a);
        am1 = mag_sub(rs, a_abs, one);
    }
    if (b->sign < 0) {
        VmBignum* b_abs = bignum_abs_val(rs, b);
        bm1 = mag_sub(rs, b_abs, one);
    }

    if (a->sign > 0 && b->sign < 0) {
        /* a & (-b) = a & ~(b-1) */
        VmBignum* r = bignum_create(rs, a->n_limbs);
        if (!r) return NULL;
        r->n_limbs = a->n_limbs;
        r->sign = 1;
        int i;
        for (i = 0; i < a->n_limbs; i++) {
            uint32_t bval = (i < bm1->n_limbs) ? bm1->limbs[i] : 0;
            r->limbs[i] = a->limbs[i] & ~bval;
        }
        bignum_normalize(r);
        return r;
    }

    if (a->sign < 0 && b->sign > 0) {
        /* (-a) & b = ~(a-1) & b */
        VmBignum* r = bignum_create(rs, b->n_limbs);
        if (!r) return NULL;
        r->n_limbs = b->n_limbs;
        r->sign = 1;
        int i;
        for (i = 0; i < b->n_limbs; i++) {
            uint32_t aval = (i < am1->n_limbs) ? am1->limbs[i] : 0;
            r->limbs[i] = ~aval & b->limbs[i];
        }
        bignum_normalize(r);
        return r;
    }

    /* Both negative: (-a) & (-b) = ~(a-1) & ~(b-1) = ~((a-1)|(b-1)) = -(or_val + 1) */
    VmBignum* or_val = bignum_create(rs, max_n);
    if (!or_val) return NULL;
    or_val->n_limbs = max_n;
    or_val->sign = 1;
    int i;
    for (i = 0; i < max_n; i++) {
        uint32_t aval = (am1 && i < am1->n_limbs) ? am1->limbs[i] : 0;
        uint32_t bval = (bm1 && i < bm1->n_limbs) ? bm1->limbs[i] : 0;
        or_val->limbs[i] = aval | bval;
    }
    bignum_normalize(or_val);
    VmBignum* result = bignum_add(rs, or_val, one);
    if (result) result->sign = -1;
    return result;
}

/**
 * Bitwise OR.
 */
static VmBignum* bignum_bitwise_or(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a)) return bignum_copy(rs, b);
    if (bignum_is_zero(b)) return bignum_copy(rs, a);

    if (a->sign > 0 && b->sign > 0) {
        int max_n = a->n_limbs > b->n_limbs ? a->n_limbs : b->n_limbs;
        VmBignum* r = bignum_create(rs, max_n);
        if (!r) return NULL;
        r->n_limbs = max_n;
        r->sign = 1;
        int i;
        for (i = 0; i < max_n; i++) {
            uint32_t la = (i < a->n_limbs) ? a->limbs[i] : 0;
            uint32_t lb = (i < b->n_limbs) ? b->limbs[i] : 0;
            r->limbs[i] = la | lb;
        }
        bignum_normalize(r);
        return r;
    }

    /* Two's complement for negative:
     * (+,-): a | (-b) = ~(~a & (b-1)) = -(~a & (b-1) + 1)...
     * Easier: use De Morgan's: a|b = ~(~a & ~b) */
    VmBignum* na = bignum_bitwise_not(rs, a);
    VmBignum* nb = bignum_bitwise_not(rs, b);
    VmBignum* and_val = bignum_bitwise_and(rs, na, nb);
    return bignum_bitwise_not(rs, and_val);
}

/**
 * Bitwise XOR.
 */
static VmBignum* bignum_bitwise_xor(VmRegionStack* rs, const VmBignum* a, const VmBignum* b) {
    if (bignum_is_zero(a)) return bignum_copy(rs, b);
    if (bignum_is_zero(b)) return bignum_copy(rs, a);

    if (a->sign > 0 && b->sign > 0) {
        int max_n = a->n_limbs > b->n_limbs ? a->n_limbs : b->n_limbs;
        VmBignum* r = bignum_create(rs, max_n);
        if (!r) return NULL;
        r->n_limbs = max_n;
        r->sign = 1;
        int i;
        for (i = 0; i < max_n; i++) {
            uint32_t la = (i < a->n_limbs) ? a->limbs[i] : 0;
            uint32_t lb = (i < b->n_limbs) ? b->limbs[i] : 0;
            r->limbs[i] = la ^ lb;
        }
        bignum_normalize(r);
        return r;
    }

    /* XOR = (a|b) & ~(a&b) */
    VmBignum* or_val = bignum_bitwise_or(rs, a, b);
    VmBignum* and_val = bignum_bitwise_and(rs, a, b);
    VmBignum* not_and = bignum_bitwise_not(rs, and_val);
    return bignum_bitwise_and(rs, or_val, not_and);
}

/**
 * Bitwise NOT: ~a = -(a+1) in two's complement.
 */
static VmBignum* bignum_bitwise_not(VmRegionStack* rs, const VmBignum* a) {
    VmBignum* one = bignum_from_int64(rs, 1);
    if (a->sign >= 0) {
        /* ~a = -(a + 1) */
        VmBignum* r = bignum_add(rs, a, one);
        if (r) r->sign = -1;
        return r;
    } else {
        /* ~(-a) = |a| - 1 */
        VmBignum* abs_a = bignum_abs_val(rs, a);
        return bignum_sub(rs, abs_a, one);
    }
}

/**
 * Left shift by `shift` bits.
 */
static VmBignum* bignum_shift_left(VmRegionStack* rs, const VmBignum* a, int shift) {
    if (bignum_is_zero(a) || shift == 0) return bignum_copy(rs, a);
    if (shift < 0) return bignum_shift_right(rs, a, -shift);

    int limb_shift = shift / 32;
    int bit_shift = shift % 32;

    int new_n = a->n_limbs + limb_shift + 1;
    VmBignum* r = bignum_create(rs, new_n);
    if (!r) return NULL;
    r->n_limbs = new_n;
    r->sign = a->sign;

    int i;
    if (bit_shift == 0) {
        for (i = 0; i < a->n_limbs; i++) {
            r->limbs[i + limb_shift] = a->limbs[i];
        }
    } else {
        uint32_t carry = 0;
        for (i = 0; i < a->n_limbs; i++) {
            uint64_t val = ((uint64_t)a->limbs[i] << bit_shift) | carry;
            r->limbs[i + limb_shift] = (uint32_t)(val & 0xFFFFFFFFULL);
            carry = (uint32_t)(val >> 32);
        }
        if (carry) r->limbs[a->n_limbs + limb_shift] = carry;
    }

    bignum_normalize(r);
    return r;
}

/**
 * Right shift by `shift` bits (arithmetic: preserves sign).
 */
static VmBignum* bignum_shift_right(VmRegionStack* rs, const VmBignum* a, int shift) {
    if (bignum_is_zero(a) || shift == 0) return bignum_copy(rs, a);
    if (shift < 0) return bignum_shift_left(rs, a, -shift);

    int limb_shift = shift / 32;
    int bit_shift = shift % 32;

    if (limb_shift >= a->n_limbs) {
        /* Shift beyond all bits */
        if (a->sign < 0) return bignum_from_int64(rs, -1); /* floor division rounding */
        return bignum_from_int64(rs, 0);
    }

    int new_n = a->n_limbs - limb_shift;
    VmBignum* r = bignum_create(rs, new_n);
    if (!r) return NULL;
    r->n_limbs = new_n;
    r->sign = a->sign;

    int i;
    if (bit_shift == 0) {
        for (i = 0; i < new_n; i++) {
            r->limbs[i] = a->limbs[i + limb_shift];
        }
    } else {
        for (i = 0; i < new_n; i++) {
            r->limbs[i] = a->limbs[i + limb_shift] >> bit_shift;
            if (i + limb_shift + 1 < a->n_limbs) {
                r->limbs[i] |= a->limbs[i + limb_shift + 1] << (32 - bit_shift);
            }
        }
    }

    bignum_normalize(r);

    /* For negative numbers, arithmetic shift rounds toward -infinity.
     * If any shifted-out bits were 1, subtract 1 from result. */
    if (a->sign < 0) {
        int any_set = 0;
        int j;
        for (j = 0; j < limb_shift && !any_set; j++) {
            if (a->limbs[j]) any_set = 1;
        }
        if (!any_set && bit_shift > 0) {
            uint32_t mask = (1U << bit_shift) - 1;
            if (a->limbs[limb_shift] & mask) any_set = 1;
        }
        if (any_set) {
            VmBignum* one = bignum_from_int64(rs, 1);
            /* r is negative magnitude. We need |r|+1 with negative sign. */
            VmBignum* abs_r = bignum_abs_val(rs, r);
            VmBignum* inc = bignum_add(rs, abs_r, one);
            if (inc) inc->sign = -1;
            return inc;
        }
    }

    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Native call dispatch (IDs 350–369)
 *
 * The VM calls these via OP_NATIVE_CALL with the appropriate ID.
 * Each function pops arguments from the VM stack and pushes the result.
 *
 * 350: bignum_from_int64       — int64 → bignum
 * 351: bignum_from_string      — string → bignum
 * 352: bignum_to_string        — bignum → string
 * 353: bignum_to_double        — bignum → float64
 * 354: bignum_to_int64         — bignum → int64 (clamp on overflow)
 * 355: bignum_add              — bignum × bignum → bignum
 * 356: bignum_sub              — bignum × bignum → bignum
 * 357: bignum_mul              — bignum × bignum → bignum
 * 358: bignum_div              — bignum × bignum → bignum (quotient)
 * 359: bignum_mod              — bignum × bignum → bignum (remainder)
 * 360: bignum_neg              — bignum → bignum
 * 361: bignum_abs              — bignum → bignum
 * 362: bignum_pow              — bignum × uint64 → bignum
 * 363: bignum_compare          — bignum × bignum → int (-1/0/1)
 * 364: bignum_gcd              — bignum × bignum → bignum
 * 365: bignum_bitwise_and      — bignum × bignum → bignum
 * 366: bignum_bitwise_or       — bignum × bignum → bignum
 * 367: bignum_bitwise_xor      — bignum × bignum → bignum
 * 368: bignum_bitwise_not      — bignum → bignum
 * 369: bignum_shift_left       — bignum × int → bignum (negative = right)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define VM_BIGNUM_FROM_INT64    (VM_NATIVE_BIGNUM_BASE + 0)   /* 350 */
#define VM_BIGNUM_FROM_STRING   (VM_NATIVE_BIGNUM_BASE + 1)   /* 351 */
#define VM_BIGNUM_TO_STRING     (VM_NATIVE_BIGNUM_BASE + 2)   /* 352 */
#define VM_BIGNUM_TO_DOUBLE     (VM_NATIVE_BIGNUM_BASE + 3)   /* 353 */
#define VM_BIGNUM_TO_INT64      (VM_NATIVE_BIGNUM_BASE + 4)   /* 354 */
#define VM_BIGNUM_ADD           (VM_NATIVE_BIGNUM_BASE + 5)   /* 355 */
#define VM_BIGNUM_SUB           (VM_NATIVE_BIGNUM_BASE + 6)   /* 356 */
#define VM_BIGNUM_MUL           (VM_NATIVE_BIGNUM_BASE + 7)   /* 357 */
#define VM_BIGNUM_DIV           (VM_NATIVE_BIGNUM_BASE + 8)   /* 358 */
#define VM_BIGNUM_MOD           (VM_NATIVE_BIGNUM_BASE + 9)   /* 359 */
#define VM_BIGNUM_NEG           (VM_NATIVE_BIGNUM_BASE + 10)  /* 360 */
#define VM_BIGNUM_ABS           (VM_NATIVE_BIGNUM_BASE + 11)  /* 361 */
#define VM_BIGNUM_POW           (VM_NATIVE_BIGNUM_BASE + 12)  /* 362 */
#define VM_BIGNUM_COMPARE       (VM_NATIVE_BIGNUM_BASE + 13)  /* 363 */
#define VM_BIGNUM_GCD           (VM_NATIVE_BIGNUM_BASE + 14)  /* 364 */
#define VM_BIGNUM_BITAND        (VM_NATIVE_BIGNUM_BASE + 15)  /* 365 */
#define VM_BIGNUM_BITOR         (VM_NATIVE_BIGNUM_BASE + 16)  /* 366 */
#define VM_BIGNUM_BITXOR        (VM_NATIVE_BIGNUM_BASE + 17)  /* 367 */
#define VM_BIGNUM_BITNOT        (VM_NATIVE_BIGNUM_BASE + 18)  /* 368 */
#define VM_BIGNUM_SHIFT         (VM_NATIVE_BIGNUM_BASE + 19)  /* 369 */

/* ═══════════════════════════════════════════════════════════════════════════
 * Self-test
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef VM_BIGNUM_TEST

static int test_count = 0;
static int pass_count = 0;
static int fail_count = 0;

#define TEST_ASSERT(cond, msg) do { \
    test_count++; \
    if (cond) { pass_count++; printf("  PASS: %s\n", msg); } \
    else { fail_count++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define TEST_ASSERT_STR(got, expected, msg) do { \
    test_count++; \
    if ((got) && strcmp((got), (expected)) == 0) { \
        pass_count++; printf("  PASS: %s  =>  %s\n", msg, got); \
    } else { \
        fail_count++; printf("  FAIL: %s  =>  got '%s', expected '%s'\n", msg, (got) ? (got) : "(null)", expected); \
    } \
} while(0)

int main(void) {
    printf("=== Eshkol VM Bignum Self-Test ===\n\n");

    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* ── Test 1: 2^64 (overflows int64) ── */
    printf("[Test 1] 2^64 arithmetic\n");
    {
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* big = bignum_pow(&rs, two, 64);
        char* s = bignum_to_string(&rs, big);
        TEST_ASSERT_STR(s, "18446744073709551616", "2^64");

        /* 2^64 + 1 */
        VmBignum* one = bignum_from_int64(&rs, 1);
        VmBignum* big_plus_1 = bignum_add(&rs, big, one);
        s = bignum_to_string(&rs, big_plus_1);
        TEST_ASSERT_STR(s, "18446744073709551617", "2^64 + 1");

        /* 2^64 - 1 = UINT64_MAX */
        VmBignum* big_minus_1 = bignum_sub(&rs, big, one);
        s = bignum_to_string(&rs, big_minus_1);
        TEST_ASSERT_STR(s, "18446744073709551615", "2^64 - 1");

        /* 2^64 * 2 = 2^65 */
        VmBignum* doubled = bignum_mul(&rs, big, two);
        s = bignum_to_string(&rs, doubled);
        TEST_ASSERT_STR(s, "36893488147419103232", "2^64 * 2 = 2^65");
    }

    /* ── Test 2: 100! (factorial) ── */
    printf("\n[Test 2] 100!\n");
    {
        VmBignum* fact = bignum_from_int64(&rs, 1);
        int i;
        for (i = 2; i <= 100; i++) {
            VmBignum* iv = bignum_from_int64(&rs, i);
            fact = bignum_mul(&rs, fact, iv);
        }
        char* s = bignum_to_string(&rs, fact);
        const char* expected =
            "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000";
        TEST_ASSERT_STR(s, expected, "100!");
    }

    /* ── Test 3: 2^100 = 1267650600228229401496703205376 ── */
    printf("\n[Test 3] 2^100\n");
    {
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        char* s = bignum_to_string(&rs, p100);
        TEST_ASSERT_STR(s, "1267650600228229401496703205376", "2^100");
    }

    /* ── Test 4: to_string round-trip ── */
    printf("\n[Test 4] bignum_to_string / bignum_from_string round-trip\n");
    {
        const char* val = "123456789012345678901234567890";
        VmBignum* b = bignum_from_string(&rs, val);
        char* s = bignum_to_string(&rs, b);
        TEST_ASSERT_STR(s, val, "round-trip positive");

        const char* neg_val = "-987654321098765432109876543210";
        b = bignum_from_string(&rs, neg_val);
        s = bignum_to_string(&rs, b);
        TEST_ASSERT_STR(s, neg_val, "round-trip negative");

        b = bignum_from_string(&rs, "0");
        s = bignum_to_string(&rs, b);
        TEST_ASSERT_STR(s, "0", "round-trip zero");
    }

    /* ── Test 5: from_string("99999999999999999999") ── */
    printf("\n[Test 5] from_string(\"99999999999999999999\")\n");
    {
        VmBignum* b = bignum_from_string(&rs, "99999999999999999999");
        char* s = bignum_to_string(&rs, b);
        TEST_ASSERT_STR(s, "99999999999999999999", "from_string 20 nines");

        /* Add 1 to verify arithmetic on parsed bignum */
        VmBignum* one = bignum_from_int64(&rs, 1);
        VmBignum* sum = bignum_add(&rs, b, one);
        s = bignum_to_string(&rs, sum);
        TEST_ASSERT_STR(s, "100000000000000000000", "99999999999999999999 + 1");
    }

    /* ── Test 6: comparison: 2^100 > 2^99 ── */
    printf("\n[Test 6] comparison: 2^100 > 2^99\n");
    {
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        VmBignum* p99 = bignum_pow(&rs, two, 99);
        int cmp = bignum_compare(p100, p99);
        TEST_ASSERT(cmp > 0, "2^100 > 2^99");

        cmp = bignum_compare(p99, p100);
        TEST_ASSERT(cmp < 0, "2^99 < 2^100");

        cmp = bignum_compare(p100, p100);
        TEST_ASSERT(cmp == 0, "2^100 == 2^100");

        /* Negative comparisons */
        VmBignum* neg_p100 = bignum_neg(&rs, p100);
        VmBignum* neg_p99 = bignum_neg(&rs, p99);
        cmp = bignum_compare(neg_p100, neg_p99);
        TEST_ASSERT(cmp < 0, "-2^100 < -2^99");
    }

    /* ── Test 7: subtraction: 2^100 - 1 ── */
    printf("\n[Test 7] subtraction: 2^100 - 1\n");
    {
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        VmBignum* one = bignum_from_int64(&rs, 1);
        VmBignum* result = bignum_sub(&rs, p100, one);
        char* s = bignum_to_string(&rs, result);
        TEST_ASSERT_STR(s, "1267650600228229401496703205375", "2^100 - 1");

        /* Verify: (2^100 - 1) + 1 == 2^100 */
        VmBignum* back = bignum_add(&rs, result, one);
        int cmp = bignum_compare(back, p100);
        TEST_ASSERT(cmp == 0, "(2^100 - 1) + 1 == 2^100");
    }

    /* ── Test 8: division: 100! / 99! = 100 ── */
    printf("\n[Test 8] division: 100! / 99! = 100\n");
    {
        VmBignum* fact100 = bignum_from_int64(&rs, 1);
        VmBignum* fact99 = bignum_from_int64(&rs, 1);
        int i;
        for (i = 2; i <= 100; i++) {
            VmBignum* iv = bignum_from_int64(&rs, i);
            fact100 = bignum_mul(&rs, fact100, iv);
            if (i <= 99) fact99 = bignum_mul(&rs, fact99, iv);
        }
        VmBignum* q = bignum_div(&rs, fact100, fact99);
        char* s = bignum_to_string(&rs, q);
        TEST_ASSERT_STR(s, "100", "100! / 99! = 100");

        /* Verify remainder is 0 */
        VmBignum* r = bignum_mod(&rs, fact100, fact99);
        TEST_ASSERT(bignum_is_zero(r), "100! mod 99! = 0");
    }

    /* ── Test 9: pow(2, 100) via repeated squaring ── */
    printf("\n[Test 9] pow(2, 100) via repeated squaring\n");
    {
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p = bignum_pow(&rs, two, 100);
        char* s = bignum_to_string(&rs, p);
        TEST_ASSERT_STR(s, "1267650600228229401496703205376", "pow(2, 100)");

        /* pow(10, 30) */
        VmBignum* ten = bignum_from_int64(&rs, 10);
        p = bignum_pow(&rs, ten, 30);
        s = bignum_to_string(&rs, p);
        TEST_ASSERT_STR(s, "1000000000000000000000000000000", "pow(10, 30)");

        /* pow(-3, 7) = -2187 */
        VmBignum* neg3 = bignum_from_int64(&rs, -3);
        p = bignum_pow(&rs, neg3, 7);
        s = bignum_to_string(&rs, p);
        TEST_ASSERT_STR(s, "-2187", "pow(-3, 7) = -2187");

        /* pow(anything, 0) = 1 */
        p = bignum_pow(&rs, ten, 0);
        s = bignum_to_string(&rs, p);
        TEST_ASSERT_STR(s, "1", "pow(10, 0) = 1");
    }

    /* ── Test 10: bitwise-and, bitwise-or ── */
    printf("\n[Test 10] bitwise-and, bitwise-or\n");
    {
        VmBignum* a = bignum_from_int64(&rs, 0xFF00FF00LL);
        VmBignum* b = bignum_from_int64(&rs, 0x0FF00FF0LL);
        VmBignum* r_and = bignum_bitwise_and(&rs, a, b);
        VmBignum* r_or = bignum_bitwise_or(&rs, a, b);

        /* 0xFF00FF00 & 0x0FF00FF0 = 0x0F000F00 = 251662080 */
        char* s = bignum_to_string(&rs, r_and);
        TEST_ASSERT_STR(s, "251662080", "0xFF00FF00 & 0x0FF00FF0 = 0x0F000F00");

        /* 0xFF00FF00 | 0x0FF00FF0 = 0xFFF0FFF0 = 4293984240 */
        s = bignum_to_string(&rs, r_or);
        TEST_ASSERT_STR(s, "4293984240", "0xFF00FF00 | 0x0FF00FF0 = 0xFFF0FFF0");

        /* XOR */
        VmBignum* r_xor = bignum_bitwise_xor(&rs, a, b);
        /* 0xFF00FF00 ^ 0x0FF00FF0 = 0xF0F0F0F0 = 4042322160 */
        s = bignum_to_string(&rs, r_xor);
        TEST_ASSERT_STR(s, "4042322160", "0xFF00FF00 ^ 0x0FF00FF0 = 0xF0F0F0F0");

        /* Large bitwise AND: 2^100 & (2^100 - 1) = 0 (power of 2 property) */
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        VmBignum* one = bignum_from_int64(&rs, 1);
        VmBignum* p100m1 = bignum_sub(&rs, p100, one);
        VmBignum* bit_and = bignum_bitwise_and(&rs, p100, p100m1);
        TEST_ASSERT(bignum_is_zero(bit_and), "2^100 & (2^100-1) = 0 (power-of-2 check)");

        /* 2^100 | (2^100 - 1) = 2^101 - 1 */
        VmBignum* bit_or = bignum_bitwise_or(&rs, p100, p100m1);
        VmBignum* p101 = bignum_pow(&rs, two, 101);
        VmBignum* p101m1 = bignum_sub(&rs, p101, one);
        int cmp = bignum_compare(bit_or, p101m1);
        TEST_ASSERT(cmp == 0, "2^100 | (2^100-1) = 2^101 - 1");
    }

    /* ── Test 11: to_double precision ── */
    printf("\n[Test 11] bignum_to_double\n");
    {
        VmBignum* b = bignum_from_int64(&rs, 123456789);
        double d = bignum_to_double(b);
        TEST_ASSERT(d == 123456789.0, "int64 to_double exact");

        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        d = bignum_to_double(p100);
        double expected = ldexp(1.0, 100);
        TEST_ASSERT(d == expected, "2^100 to_double = ldexp(1,100)");

        VmBignum* neg = bignum_neg(&rs, p100);
        d = bignum_to_double(neg);
        TEST_ASSERT(d == -expected, "-2^100 to_double");
    }

    /* ── Test 12: to_int64 and overflow detection ── */
    printf("\n[Test 12] bignum_to_int64 with overflow\n");
    {
        VmBignum* small = bignum_from_int64(&rs, 42);
        int ov = 0;
        int64_t v = bignum_to_int64(small, &ov);
        TEST_ASSERT(v == 42 && !ov, "42 fits in int64");

        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        v = bignum_to_int64(p100, &ov);
        TEST_ASSERT(ov == 1, "2^100 overflows int64");

        /* INT64_MIN round-trip */
        VmBignum* min_val = bignum_from_int64(&rs, INT64_MIN);
        v = bignum_to_int64(min_val, &ov);
        TEST_ASSERT(v == INT64_MIN && !ov, "INT64_MIN round-trip");

        /* INT64_MAX round-trip */
        VmBignum* max_val = bignum_from_int64(&rs, INT64_MAX);
        v = bignum_to_int64(max_val, &ov);
        TEST_ASSERT(v == INT64_MAX && !ov, "INT64_MAX round-trip");
    }

    /* ── Test 13: GCD ── */
    printf("\n[Test 13] GCD\n");
    {
        VmBignum* a = bignum_from_int64(&rs, 48);
        VmBignum* b = bignum_from_int64(&rs, 18);
        VmBignum* g = bignum_gcd(&rs, a, b);
        char* s = bignum_to_string(&rs, g);
        TEST_ASSERT_STR(s, "6", "gcd(48, 18) = 6");

        a = bignum_from_int64(&rs, 0);
        b = bignum_from_int64(&rs, 42);
        g = bignum_gcd(&rs, a, b);
        s = bignum_to_string(&rs, g);
        TEST_ASSERT_STR(s, "42", "gcd(0, 42) = 42");

        /* Large GCD: gcd(2^50, 2^30) = 2^30 */
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p50 = bignum_pow(&rs, two, 50);
        VmBignum* p30 = bignum_pow(&rs, two, 30);
        g = bignum_gcd(&rs, p50, p30);
        int cmp = bignum_compare(g, p30);
        TEST_ASSERT(cmp == 0, "gcd(2^50, 2^30) = 2^30");
    }

    /* ── Test 14: Shift operations ── */
    printf("\n[Test 14] shift left / right\n");
    {
        VmBignum* one = bignum_from_int64(&rs, 1);
        VmBignum* shifted = bignum_shift_left(&rs, one, 100);
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        TEST_ASSERT(bignum_compare(shifted, p100) == 0, "1 << 100 == 2^100");

        VmBignum* back = bignum_shift_right(&rs, shifted, 100);
        TEST_ASSERT(bignum_compare(back, one) == 0, "(1 << 100) >> 100 == 1");

        VmBignum* val = bignum_from_int64(&rs, 255);
        VmBignum* sr = bignum_shift_right(&rs, val, 4);
        char* s = bignum_to_string(&rs, sr);
        TEST_ASSERT_STR(s, "15", "255 >> 4 = 15");
    }

    /* ── Test 15: Edge cases ── */
    printf("\n[Test 15] edge cases\n");
    {
        /* zero operations */
        VmBignum* zero = bignum_from_int64(&rs, 0);
        VmBignum* one = bignum_from_int64(&rs, 1);

        VmBignum* r = bignum_add(&rs, zero, zero);
        TEST_ASSERT(bignum_is_zero(r), "0 + 0 = 0");

        r = bignum_sub(&rs, one, one);
        TEST_ASSERT(bignum_is_zero(r), "1 - 1 = 0");

        r = bignum_mul(&rs, zero, one);
        TEST_ASSERT(bignum_is_zero(r), "0 * 1 = 0");

        /* a - a = 0 for large number */
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        r = bignum_sub(&rs, p100, p100);
        TEST_ASSERT(bignum_is_zero(r), "2^100 - 2^100 = 0");

        /* Negation of zero */
        r = bignum_neg(&rs, zero);
        TEST_ASSERT(bignum_is_zero(r), "-0 = 0");

        /* abs(-42) = 42 */
        VmBignum* neg42 = bignum_from_int64(&rs, -42);
        r = bignum_abs_val(&rs, neg42);
        char* s = bignum_to_string(&rs, r);
        TEST_ASSERT_STR(s, "42", "abs(-42) = 42");

        /* Division with remainder */
        VmBignum* dividend = bignum_from_int64(&rs, 17);
        VmBignum* divisor = bignum_from_int64(&rs, 5);
        VmBignum* q = NULL;
        VmBignum* rem = NULL;
        bignum_divmod(&rs, dividend, divisor, &q, &rem);
        s = bignum_to_string(&rs, q);
        TEST_ASSERT_STR(s, "3", "17 / 5 = 3");
        s = bignum_to_string(&rs, rem);
        TEST_ASSERT_STR(s, "2", "17 mod 5 = 2");

        /* Negative division */
        VmBignum* neg17 = bignum_from_int64(&rs, -17);
        bignum_divmod(&rs, neg17, divisor, &q, &rem);
        s = bignum_to_string(&rs, q);
        TEST_ASSERT_STR(s, "-3", "-17 / 5 = -3 (truncated)");
        s = bignum_to_string(&rs, rem);
        TEST_ASSERT_STR(s, "-2", "-17 mod 5 = -2 (remainder has dividend's sign)");
    }

    /* ── Test 16: Large multi-limb division ── */
    printf("\n[Test 16] multi-limb division\n");
    {
        /* (2^100) / (2^50) = 2^50 */
        VmBignum* two = bignum_from_int64(&rs, 2);
        VmBignum* p100 = bignum_pow(&rs, two, 100);
        VmBignum* p50 = bignum_pow(&rs, two, 50);
        VmBignum* q = bignum_div(&rs, p100, p50);
        int cmp = bignum_compare(q, p50);
        TEST_ASSERT(cmp == 0, "2^100 / 2^50 = 2^50");

        VmBignum* r = bignum_mod(&rs, p100, p50);
        TEST_ASSERT(bignum_is_zero(r), "2^100 mod 2^50 = 0");

        /* (2^100 + 7) / (2^50): quotient = 2^50, remainder = 7 */
        VmBignum* seven = bignum_from_int64(&rs, 7);
        VmBignum* n = bignum_add(&rs, p100, seven);
        VmBignum* qq = NULL;
        VmBignum* rr = NULL;
        bignum_divmod(&rs, n, p50, &qq, &rr);
        cmp = bignum_compare(qq, p50);
        TEST_ASSERT(cmp == 0, "(2^100 + 7) / 2^50 = 2^50 (quotient)");
        char* s = bignum_to_string(&rs, rr);
        TEST_ASSERT_STR(s, "7", "(2^100 + 7) mod 2^50 = 7 (remainder)");
    }

    /* ── Summary ── */
    printf("\n=== Results: %d/%d passed", pass_count, test_count);
    if (fail_count > 0) printf(", %d FAILED", fail_count);
    printf(" ===\n");

    vm_arena_print_metrics(&rs);
    vm_region_stack_destroy(&rs);

    return fail_count > 0 ? 1 : 0;
}

#endif /* VM_BIGNUM_TEST */
