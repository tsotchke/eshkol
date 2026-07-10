/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Rational number runtime for Eshkol
 */

#include <eshkol/core/rational.h>
#include <eshkol/core/bignum.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

/* Forward declaration for arena allocation */
extern "C" void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                             uint8_t subtype, uint8_t flags);
extern "C" void* arena_allocate(void* arena, uint64_t size);
extern "C" void* arena_allocate_string_with_header(void* arena, uint64_t size);
/* Runtime thread-local (or global) arena — used by the arena-less rational
 * comparison API for bignum cross-product scratch space. */
extern "C" arena_t* arena_get_thread_local(void);

/* ===== Bignum-path helpers =====
 * The exact rational substrate promotes to arbitrary precision whenever the
 * reduced numerator or denominator leaves int64 range. All bignum work goes
 * through the public bignum API (lib/core/bignum.cpp); this file never
 * degrades an exact rational to double. */

/** True if bignum @p a equals the int64 constant @p k. */
static inline bool bn_equals_i64(const eshkol_bignum_t* a, int64_t k) {
    int64_t v;
    return eshkol_bignum_fits_int64(a, &v) && v == k;
}

/** Non-negative bignum GCD of |a| and |b| (Euclid over magnitudes). */
static eshkol_bignum_t* bn_gcd(arena_t* arena,
                               const eshkol_bignum_t* a,
                               const eshkol_bignum_t* b) {
    eshkol_bignum_t* x = a->sign ? eshkol_bignum_neg(arena, a) : (eshkol_bignum_t*)a;
    eshkol_bignum_t* y = b->sign ? eshkol_bignum_neg(arena, b) : (eshkol_bignum_t*)b;
    while (!eshkol_bignum_is_zero(y)) {
        eshkol_bignum_t* r = eshkol_bignum_mod(arena, x, y);
        if (!r) break;
        if (r->sign) r->sign = 0; /* magnitude */
        x = y;
        y = r;
    }
    if (x && x->sign) x->sign = 0;
    return x;
}

/* Allocate a small (int64 fast-path) rational, assuming num/denom already
 * reduced with denom > 0. */
static void* rational_alloc_small(void* arena, int64_t num, int64_t denom) {
    eshkol_rational_t* r = (eshkol_rational_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_rational_t), HEAP_SUBTYPE_RATIONAL, 0);
    r->numerator = num;
    r->denominator = denom;
    r->is_big = 0;
    r->reserved = 0;
    r->big_num = nullptr;
    r->big_den = nullptr;
    return r;
}

/* Extract a rational operand's numerator as a bignum (promoting the small
 * fast path). */
static eshkol_bignum_t* rat_num_bn(arena_t* arena, const eshkol_rational_t* r) {
    return r->is_big ? r->big_num : eshkol_bignum_from_int64(arena, r->numerator);
}

/* Extract a rational operand's denominator as a bignum. */
static eshkol_bignum_t* rat_den_bn(arena_t* arena, const eshkol_rational_t* r) {
    return r->is_big ? r->big_den : eshkol_bignum_from_int64(arena, r->denominator);
}

/** Euclidean greatest common divisor of two 64-bit integers (operates on absolute values). */
static int64_t gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    if (a == 0) return b;
    if (b == 0) return a;
    while (b != 0) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/* 128-bit GCD for overflow-safe intermediate results */
static __int128_t gcd128(__int128_t a, __int128_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        __int128_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/* Check if a 128-bit value fits in int64_t */
static inline int fits_int64(__int128_t v) {
    return v >= (__int128_t)INT64_MIN && v <= (__int128_t)INT64_MAX;
}

/* Create a rational from 128-bit intermediates (both operands were small).
 * Returns a reduced int64 fast-path rational, or NULL if the reduced result
 * exceeds int64 range — in which case the caller recomputes exactly with the
 * bignum path (it never degrades to double). */
static void* rational_create_safe(void* arena, __int128_t num, __int128_t denom) {
    if (denom == 0) {
        denom = 1;
        num = 0;
    }
    if (denom < 0) {
        num = -num;
        denom = -denom;
    }
    __int128_t g = gcd128(num, denom);
    if (g > 1) {
        num /= g;
        denom /= g;
    }
    if (!fits_int64(num) || !fits_int64(denom)) {
        return NULL; /* Signal: caller falls back to the exact bignum path */
    }
    return rational_alloc_small(arena, (int64_t)num, (int64_t)denom);
}

/** @brief Exact rational from bignum numerator/denominator (ESH-0105/ESH-0123).
 *  Sign-canonicalizes to a positive denominator, reduces by the bignum GCD, and
 *  demotes to the int64 fast path when the reduced pair fits. Never lossy. */
extern "C" void* eshkol_rational_create_bn(void* arena_v,
                                           eshkol_bignum_t* num,
                                           eshkol_bignum_t* denom) {
    arena_t* arena = (arena_t*)arena_v;
    if (!num || !denom || eshkol_bignum_is_zero(denom)) {
        eshkol_error("rational: division by zero (denominator is 0)");
        return rational_alloc_small(arena_v, 0, 1);
    }

    /* Normalize sign so the denominator is positive. */
    if (denom->sign) {
        num = eshkol_bignum_neg(arena, num);
        denom = eshkol_bignum_neg(arena, denom);
    }

    /* Reduce by GCD (skip the divide only when gcd == 1). */
    eshkol_bignum_t* g = bn_gcd(arena, num, denom);
    if (g && !bn_equals_i64(g, 1) && !eshkol_bignum_is_zero(g)) {
        num = eshkol_bignum_div(arena, num, g);
        denom = eshkol_bignum_div(arena, denom, g);
    }

    /* Prefer the int64 fast path whenever the reduced pair fits. */
    int64_t ni, di;
    if (eshkol_bignum_fits_int64(num, &ni) && eshkol_bignum_fits_int64(denom, &di)) {
        return rational_alloc_small(arena_v, ni, di);
    }

    eshkol_rational_t* r = (eshkol_rational_t*)arena_allocate_with_header(
        arena_v, sizeof(eshkol_rational_t), HEAP_SUBTYPE_RATIONAL, 0);
    r->numerator = 0;
    r->denominator = 1;
    r->is_big = 1;
    r->reserved = 0;
    r->big_num = num;
    r->big_den = denom;
    return r;
}

/** @brief Create a normalized rational number in @p arena from a numerator/denominator pair.
 *  Normalizes the sign to a positive denominator and reduces by the GCD. A zero
 *  denominator (or an unsafe INT64_MIN sign-flip) reports an error and yields 0/1.
 *  @return Pointer to a header-tagged eshkol_rational_t. */
extern "C" void* eshkol_rational_create(void* arena, int64_t num, int64_t denom) {
    if (denom == 0) {
        eshkol_error("rational: division by zero (denominator is 0)");
        /* Return 0/1 as safe fallback after error is reported */
        denom = 1;
        num = 0;
    }

    /* Audit C7: the INT64_MIN sign-flip (-INT64_MIN == INT64_MIN) cannot be
     * normalised in int64. Now that the rational substrate is bignum-capable,
     * route these rare cases through the exact bignum constructor instead of
     * erroring. */
    if (denom == INT64_MIN || num == INT64_MIN) {
        eshkol_bignum_t* bn = eshkol_bignum_from_int64((arena_t*)arena, num);
        eshkol_bignum_t* bd = eshkol_bignum_from_int64((arena_t*)arena, denom);
        return eshkol_rational_create_bn(arena, bn, bd);
    }
    if (denom < 0) {
        num = -num;
        denom = -denom;
    }

    /* Reduce by GCD */
    int64_t g = gcd(num, denom);
    if (g > 1) {
        num /= g;
        denom /= g;
    }

    return rational_alloc_small(arena, num, denom);
}

/* Is a rational operand a big (bignum-path) rational? */
static inline bool rat_is_big(const void* r) {
    return ((const eshkol_rational_t*)r)->is_big != 0;
}

/* Bignum-path add/sub/mul: exact, never lossy. sub == 0, add == 1 selector via
 * caller. */
static void* rat_add_sub_big(void* arena_v, const eshkol_rational_t* ra,
                             const eshkol_rational_t* rb, bool subtract) {
    arena_t* arena = (arena_t*)arena_v;
    eshkol_bignum_t* an = rat_num_bn(arena, ra);
    eshkol_bignum_t* ad = rat_den_bn(arena, ra);
    eshkol_bignum_t* bn = rat_num_bn(arena, rb);
    eshkol_bignum_t* bd = rat_den_bn(arena, rb);
    eshkol_bignum_t* left = eshkol_bignum_mul(arena, an, bd);
    eshkol_bignum_t* right = eshkol_bignum_mul(arena, bn, ad);
    eshkol_bignum_t* num = subtract ? eshkol_bignum_sub(arena, left, right)
                                    : eshkol_bignum_add(arena, left, right);
    eshkol_bignum_t* den = eshkol_bignum_mul(arena, ad, bd);
    return eshkol_rational_create_bn(arena_v, num, den);
}

static void* rat_mul_big(void* arena_v, const eshkol_rational_t* ra,
                         const eshkol_rational_t* rb) {
    arena_t* arena = (arena_t*)arena_v;
    eshkol_bignum_t* num = eshkol_bignum_mul(arena, rat_num_bn(arena, ra), rat_num_bn(arena, rb));
    eshkol_bignum_t* den = eshkol_bignum_mul(arena, rat_den_bn(arena, ra), rat_den_bn(arena, rb));
    return eshkol_rational_create_bn(arena_v, num, den);
}

static void* rat_div_big(void* arena_v, const eshkol_rational_t* ra,
                         const eshkol_rational_t* rb) {
    arena_t* arena = (arena_t*)arena_v;
    eshkol_bignum_t* num = eshkol_bignum_mul(arena, rat_num_bn(arena, ra), rat_den_bn(arena, rb));
    eshkol_bignum_t* den = eshkol_bignum_mul(arena, rat_den_bn(arena, ra), rat_num_bn(arena, rb));
    return eshkol_rational_create_bn(arena_v, num, den);
}

/** @brief Add two rationals, returning an exact reduced result. */
extern "C" void* eshkol_rational_add(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (!ra->is_big && !rb->is_big) {
        /* a/b + c/d = (a*d + c*b) / (b*d) — __int128_t guards int64 overflow */
        __int128_t num = (__int128_t)ra->numerator * rb->denominator
                       + (__int128_t)rb->numerator * ra->denominator;
        __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
        void* r = rational_create_safe(arena, num, denom);
        if (r) return r;
    }
    return rat_add_sub_big(arena, ra, rb, /*subtract=*/false);
}

/** @brief Subtract rational @p b from @p a, returning an exact reduced result. */
extern "C" void* eshkol_rational_sub(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (!ra->is_big && !rb->is_big) {
        __int128_t num = (__int128_t)ra->numerator * rb->denominator
                       - (__int128_t)rb->numerator * ra->denominator;
        __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
        void* r = rational_create_safe(arena, num, denom);
        if (r) return r;
    }
    return rat_add_sub_big(arena, ra, rb, /*subtract=*/true);
}

/** @brief Multiply two rationals, returning an exact reduced result. */
extern "C" void* eshkol_rational_mul(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (!ra->is_big && !rb->is_big) {
        __int128_t num = (__int128_t)ra->numerator * rb->numerator;
        __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
        void* r = rational_create_safe(arena, num, denom);
        if (r) return r;
    }
    return rat_mul_big(arena, ra, rb);
}

/** @brief Divide rational @p a by @p b, returning an exact reduced result.
 *  Raises a divide-by-zero exception if @p b is zero. */
extern "C" void* eshkol_rational_div(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    bool b_zero = rb->is_big ? eshkol_bignum_is_zero(rb->big_num)
                             : (rb->numerator == 0);
    if (b_zero) {
        eshkol_exception_t* exc = eshkol_make_exception(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, "rational division by zero");
        eshkol_raise(exc);
        return nullptr;  // unreachable, but satisfies compiler
    }
    if (!ra->is_big && !rb->is_big) {
        __int128_t num = (__int128_t)ra->numerator * rb->denominator;
        __int128_t denom = (__int128_t)ra->denominator * rb->numerator;
        void* r = rational_create_safe(arena, num, denom);
        if (r) return r;
    }
    return rat_div_big(arena, ra, rb);
}

/** @brief Compare two rationals via cross-multiplication.
 *  @return -1 if a < b, 1 if a > b, 0 if equal. */
extern "C" int eshkol_rational_compare(void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (!ra->is_big && !rb->is_big) {
        /* Compare a/b vs c/d → a*d vs c*b — __int128_t guards int64 overflow */
        __int128_t lhs = (__int128_t)ra->numerator * rb->denominator;
        __int128_t rhs = (__int128_t)rb->numerator * ra->denominator;
        if (lhs < rhs) return -1;
        if (lhs > rhs) return 1;
        return 0;
    }
    /* Denominators are positive, so compare a_num*b_den vs b_num*a_den. The
     * cross-products need bignum scratch space; use the runtime thread-local
     * arena (this API has no arena parameter, and the temporaries are dead
     * after the comparison). */
    arena_t* arena = arena_get_thread_local();
    eshkol_bignum_t* lhs = eshkol_bignum_mul(arena, rat_num_bn(arena, ra), rat_den_bn(arena, rb));
    eshkol_bignum_t* rhs = eshkol_bignum_mul(arena, rat_num_bn(arena, rb), rat_den_bn(arena, ra));
    return eshkol_bignum_compare(lhs, rhs);
}

/** Return the numerator of a rational. */
extern "C" int64_t eshkol_rational_numerator(void* r) {
    return ((eshkol_rational_t*)r)->numerator;
}

/** Return the (always positive) denominator of a rational. */
extern "C" int64_t eshkol_rational_denominator(void* r) {
    return ((eshkol_rational_t*)r)->denominator;
}

/** Convert a rational to its nearest double-precision floating-point value. */
extern "C" double eshkol_rational_to_double(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    if (rat->is_big) {
        return eshkol_bignum_to_double(rat->big_num) /
               eshkol_bignum_to_double(rat->big_den);
    }
    return (double)rat->numerator / (double)rat->denominator;
}

/** Return non-zero if the rational represents an integer (denominator == 1). */
extern "C" int eshkol_rational_is_integer(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    if (rat->is_big) return bn_equals_i64(rat->big_den, 1);
    return rat->denominator == 1;
}

/* Convert IEEE 754 double to exact rational (R7RS inexact->exact) */
extern "C" void* eshkol_double_to_rational(void* arena, double d) {
    if (d == 0.0) return eshkol_rational_create(arena, 0, 1);
    // Scale by powers of 2 until integer (IEEE doubles have at most 53 binary digits)
    double abs_d = d < 0 ? -d : d;
    int64_t den = 1;
    while (abs_d != __builtin_floor(abs_d) && den < (1LL << 52)) {
        abs_d *= 2.0;
        den *= 2;
    }
    int64_t num = (int64_t)abs_d;
    if (d < 0) num = -num;
    return eshkol_rational_create(arena, num, den);  // auto-reduces via GCD
}

/* Format rational as "num/denom" string into arena-allocated buffer with
 * string header. `arena_allocate_string_with_header(len)` reserves len+1
 * bytes and handles the NUL — pass bare `len`, not `len + 1`, or
 * string-length sees len+1 chars after the header-aware fix. */
extern "C" char* eshkol_rational_to_string(void* arena, void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    if (rat->is_big) {
        /* Format bignum numerator/denominator into a fresh buffer. */
        char* num_s = eshkol_bignum_to_string((arena_t*)arena, rat->big_num);
        if (bn_equals_i64(rat->big_den, 1)) return num_s;
        char* den_s = eshkol_bignum_to_string((arena_t*)arena, rat->big_den);
        size_t nlen = num_s ? strlen(num_s) : 0;
        size_t dlen = den_s ? strlen(den_s) : 0;
        size_t total = nlen + 1 + dlen;
        char* buf = (char*)arena_allocate_string_with_header(arena, total);
        if (buf) {
            memcpy(buf, num_s, nlen);
            buf[nlen] = '/';
            memcpy(buf + nlen + 1, den_s, dlen);
            buf[total] = '\0';
        }
        return buf;
    }
    if (rat->denominator == 1) {
        char tmp[32];
        int len = snprintf(tmp, sizeof(tmp), "%lld", (long long)rat->numerator);
        char* buf = (char*)arena_allocate_string_with_header(arena, len);
        if (buf) memcpy(buf, tmp, len + 1);  /* copy including NUL */
        return buf;
    }
    char tmp[72];
    int len = snprintf(tmp, sizeof(tmp), "%lld/%lld",
                       (long long)rat->numerator, (long long)rat->denominator);
    char* buf = (char*)arena_allocate_string_with_header(arena, len);
    if (buf) memcpy(buf, tmp, len + 1);  /* copy including NUL */
    return buf;
}

/** @brief Write a rational's "n/d" (or bare "n" when integer) form to a FILE*.
 *  Handles both int64 and bignum paths. Used by the display runtime. */
extern "C" void eshkol_rational_display(void* r, void* file) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    FILE* f = (FILE*)file;
    if (!rat || !f) return;
    if (rat->is_big) {
        eshkol_bignum_display(rat->big_num, f);
        if (!bn_equals_i64(rat->big_den, 1)) {
            fputc('/', f);
            eshkol_bignum_display(rat->big_den, f);
        }
        return;
    }
    if (rat->denominator == 1) {
        fprintf(f, "%lld", (long long)rat->numerator);
    } else {
        fprintf(f, "%lld/%lld", (long long)rat->numerator,
                (long long)rat->denominator);
    }
}

/** @brief Value equality on reduced rationals (int64 and bignum paths).
 *  Both operands are in canonical reduced form, so equality is: same path and
 *  matching fields (int64 pair, or bignum numerator+denominator). */
extern "C" int eshkol_rational_equal(void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (!ra->is_big && !rb->is_big) {
        return ra->numerator == rb->numerator &&
               ra->denominator == rb->denominator;
    }
    if (ra->is_big != rb->is_big) {
        /* Canonical form guarantees a big rational never equals a small one
         * (a big one does not fit int64), so the paths differing => unequal. */
        return 0;
    }
    return eshkol_bignum_compare(ra->big_num, rb->big_num) == 0 &&
           eshkol_bignum_compare(ra->big_den, rb->big_den) == 0;
}

/* Build a tagged exact integer from a bignum, demoting to INT64 when it fits. */
static eshkol_tagged_value_t bignum_to_tagged_int(eshkol_bignum_t* b) {
    eshkol_tagged_value_t t;
    memset(&t, 0, sizeof(t));
    int64_t v;
    if (eshkol_bignum_fits_int64(b, &v)) {
        t.type = ESHKOL_VALUE_INT64;
        t.data.int_val = v;
    } else {
        t.type = ESHKOL_VALUE_HEAP_PTR;
        t.flags = ESHKOL_VALUE_EXACT_FLAG;
        t.data.int_val = (int64_t)(uintptr_t)b;
    }
    return t;
}

/** @brief R7RS numerator as a tagged value (INT64 or bignum HEAP_PTR). */
extern "C" void eshkol_rational_numerator_tagged(
    void* arena, const eshkol_tagged_value_t* v, eshkol_tagged_value_t* result)
{
    (void)arena;
    if (v->type == ESHKOL_VALUE_HEAP_PTR && v->data.int_val) {
        uint8_t subtype = *((uint8_t*)(uintptr_t)v->data.int_val - 8);
        if (subtype == HEAP_SUBTYPE_RATIONAL) {
            eshkol_rational_t* r = (eshkol_rational_t*)(uintptr_t)v->data.int_val;
            if (r->is_big) { *result = bignum_to_tagged_int(r->big_num); return; }
            memset(result, 0, sizeof(*result));
            result->type = ESHKOL_VALUE_INT64;
            result->data.int_val = r->numerator;
            return;
        }
    }
    /* int64, bignum integer, double, or non-number: numerator is the value. */
    *result = *v;
}

/** @brief R7RS denominator as a tagged value (INT64 or bignum HEAP_PTR). */
extern "C" void eshkol_rational_denominator_tagged(
    void* arena, const eshkol_tagged_value_t* v, eshkol_tagged_value_t* result)
{
    (void)arena;
    memset(result, 0, sizeof(*result));
    if (v->type == ESHKOL_VALUE_HEAP_PTR && v->data.int_val) {
        uint8_t subtype = *((uint8_t*)(uintptr_t)v->data.int_val - 8);
        if (subtype == HEAP_SUBTYPE_RATIONAL) {
            eshkol_rational_t* r = (eshkol_rational_t*)(uintptr_t)v->data.int_val;
            if (r->is_big) { *result = bignum_to_tagged_int(r->big_den); return; }
            result->type = ESHKOL_VALUE_INT64;
            result->data.int_val = r->denominator;
            return;
        }
    }
    /* int64, bignum integer, or other: denominator is 1. */
    result->type = ESHKOL_VALUE_INT64;
    result->data.int_val = 1;
}

/* Helper: check if tagged value is a rational (HEAP_PTR with RATIONAL subtype) */
extern "C" int eshkol_is_rational_tagged(eshkol_tagged_value_t val) {
    if (val.type != ESHKOL_VALUE_HEAP_PTR) return 0;
    uint8_t* ptr = (uint8_t*)(uintptr_t)val.data.int_val;
    uint8_t subtype = *(ptr - 8);  /* header is at ptr-8 */
    return subtype == HEAP_SUBTYPE_RATIONAL;
}

/* Pointer-based check for LLVM codegen */
extern "C" int eshkol_is_rational_tagged_ptr(const eshkol_tagged_value_t* val) {
    return eshkol_is_rational_tagged(*val);
}

/* Pointer-based binary dispatch for LLVM codegen */
extern "C" void eshkol_rational_binary_tagged_ptr(
    void* arena, const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b,
    int op, eshkol_tagged_value_t* result)
{
    *result = eshkol_rational_binary_tagged(arena, *a, *b, op);
}

/* Heap subtype of a HEAP_PTR tagged value (0 if not a heap pointer). */
static inline uint8_t tagged_heap_subtype(const eshkol_tagged_value_t* v) {
    if (v->type != ESHKOL_VALUE_HEAP_PTR || !v->data.int_val) return 0xFF;
    return *((uint8_t*)(uintptr_t)v->data.int_val - 8);
}

/* Coerce an exact tagged operand (INT64, bignum HEAP_PTR, or rational HEAP_PTR)
 * to a rational object pointer. Returns NULL if not an exact number. */
static void* tagged_exact_to_rational(void* arena, const eshkol_tagged_value_t* v) {
    if (v->type == ESHKOL_VALUE_INT64) {
        return eshkol_rational_create(arena, v->data.int_val, 1);
    }
    if (v->type == ESHKOL_VALUE_HEAP_PTR && v->data.int_val) {
        void* p = (void*)(uintptr_t)v->data.int_val;
        uint8_t subtype = tagged_heap_subtype(v);
        if (subtype == HEAP_SUBTYPE_RATIONAL) return p;
        if (subtype == HEAP_SUBTYPE_BIGNUM) {
            eshkol_bignum_t* one = eshkol_bignum_from_int64((arena_t*)arena, 1);
            return eshkol_rational_create_bn(arena, (eshkol_bignum_t*)p, one);
        }
    }
    return NULL;
}

/* Demote an exact rational-object result to a canonical tagged value:
 *   - integer that fits int64 -> INT64
 *   - integer too large       -> bare bignum HEAP_PTR (EXACT)
 *   - non-integer             -> rational HEAP_PTR (EXACT) */
static eshkol_tagged_value_t rational_result_to_tagged(void* rr) {
    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));
    eshkol_rational_t* rat = (eshkol_rational_t*)rr;
    if (rat->is_big) {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.flags = ESHKOL_VALUE_EXACT_FLAG;
        result.data.int_val = bn_equals_i64(rat->big_den, 1)
            ? (int64_t)(uintptr_t)rat->big_num   /* huge integer -> bare bignum */
            : (int64_t)(uintptr_t)rr;             /* proper bignum rational */
        return result;
    }
    if (rat->denominator == 1) {
        result.type = ESHKOL_VALUE_INT64;
        result.data.int_val = rat->numerator;
    } else {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.int_val = (int64_t)(uintptr_t)rr;
    }
    return result;
}

/* Extract a double from an exact-or-inexact tagged operand (int64, double,
 * bignum, or rational). */
static double tagged_any_to_double(const eshkol_tagged_value_t* v) {
    if (v->type == ESHKOL_VALUE_INT64) return (double)v->data.int_val;
    if (v->type == ESHKOL_VALUE_DOUBLE) {
        union { int64_t i; double d; } conv; conv.i = v->data.int_val; return conv.d;
    }
    uint8_t subtype = tagged_heap_subtype(v);
    if (subtype == HEAP_SUBTYPE_RATIONAL)
        return eshkol_rational_to_double((void*)(uintptr_t)v->data.int_val);
    if (subtype == HEAP_SUBTYPE_BIGNUM)
        return eshkol_bignum_to_double((eshkol_bignum_t*)(uintptr_t)v->data.int_val);
    return 0.0;
}

/* Tagged value binary dispatch for rationals. Accepts INT64, DOUBLE, bignum,
 * and rational operands. Exact operands stay exact — bignum-capable, never
 * lossy (ESH-0105). Only a genuine DOUBLE operand yields an inexact result. */
extern "C" eshkol_tagged_value_t eshkol_rational_binary_tagged(
    void* arena, eshkol_tagged_value_t a, eshkol_tagged_value_t b, int op)
{
    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));

    /* If either is inexact (double), R7RS forces an inexact result. */
    if (a.type == ESHKOL_VALUE_DOUBLE || b.type == ESHKOL_VALUE_DOUBLE) {
        double da = tagged_any_to_double(&a);
        double db = tagged_any_to_double(&b);
        double dr;
        switch (op) {
            case 0: dr = da + db; break;
            case 1: dr = da - db; break;
            case 2: dr = da * db; break;
            case 3: dr = da / db; break;
            default: dr = 0; break;
        }
        result.type = ESHKOL_VALUE_DOUBLE;
        union { double d; int64_t i; } pack;
        pack.d = dr;
        result.data.int_val = pack.i;
        return result;
    }

    /* Both exact: promote each to a rational object (int64/bignum -> n/1). */
    void* ra = tagged_exact_to_rational(arena, &a);
    void* rb = tagged_exact_to_rational(arena, &b);
    if (!ra || !rb) {
        result.type = ESHKOL_VALUE_INT64;
        result.data.int_val = 0;
        return result;
    }

    void* rr;
    switch (op) {
        case 0: rr = eshkol_rational_add(arena, ra, rb); break;
        case 1: rr = eshkol_rational_sub(arena, ra, rb); break;
        case 2: rr = eshkol_rational_mul(arena, ra, rb); break;
        case 3: rr = eshkol_rational_div(arena, ra, rb); break;
        default: rr = eshkol_rational_create(arena, 0, 1); break;
    }
    return rational_result_to_tagged(rr);
}

/** @brief Build an EXACT tagged number from bignum numerator/denominator.
 *  Reduces and demotes (bare int64/bignum when integer; rational otherwise).
 *  Used by the bignum division dispatch so `(/ 1 (expt 10 19))` stays exact. */
extern "C" void eshkol_rational_from_bignums_tagged(
    void* arena, eshkol_bignum_t* num, eshkol_bignum_t* denom,
    eshkol_tagged_value_t* result)
{
    void* rr = eshkol_rational_create_bn(arena, num, denom);
    *result = rational_result_to_tagged(rr);
}

/* Coerce an INT64 or bignum HEAP_PTR tagged operand to a bignum. */
static eshkol_bignum_t* make_operand_bignum(void* arena, const eshkol_tagged_value_t* v) {
    if (v->type == ESHKOL_VALUE_INT64) {
        return eshkol_bignum_from_int64((arena_t*)arena, v->data.int_val);
    }
    if (tagged_heap_subtype(v) == HEAP_SUBTYPE_BIGNUM) {
        return (eshkol_bignum_t*)(uintptr_t)v->data.int_val;
    }
    /* Fallback: coerce via double->int (should not happen for integer args). */
    return eshkol_bignum_from_int64((arena_t*)arena, 0);
}

/** @brief (make-rational num den) on tagged operands (INT64 or bignum). */
extern "C" void eshkol_rational_make_tagged(
    void* arena, const eshkol_tagged_value_t* num, const eshkol_tagged_value_t* den,
    eshkol_tagged_value_t* result)
{
    eshkol_bignum_t* bn = make_operand_bignum(arena, num);
    eshkol_bignum_t* bd = make_operand_bignum(arena, den);
    eshkol_rational_from_bignums_tagged(arena, bn, bd, result);
}

/* Pointer-based comparison dispatch for LLVM codegen.
 * Handles rational vs rational, rational vs int, int vs rational.
 * If either is a double, promotes to double comparison.
 * op: 0=lt, 1=gt, 2=eq, 3=le, 4=ge */
extern "C" void eshkol_rational_compare_tagged_ptr(
    void* arena, const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b,
    int op, eshkol_tagged_value_t* result)
{
    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_BOOL;

    /* If either is a double, convert to double comparison */
    if (a->type == ESHKOL_VALUE_DOUBLE || b->type == ESHKOL_VALUE_DOUBLE) {
        double da = tagged_any_to_double(a);
        double db = tagged_any_to_double(b);
        int cmp_result = 0;
        switch (op) {
            case 0: cmp_result = (da < db); break;
            case 1: cmp_result = (da > db); break;
            case 2: cmp_result = (da == db); break;
            case 3: cmp_result = (da <= db); break;
            case 4: cmp_result = (da >= db); break;
        }
        result->data.int_val = cmp_result ? 1 : 0;
        return;
    }

    /* Both exact: promote each (int64/bignum/rational) to a rational object. */
    void* ra = tagged_exact_to_rational(arena, a);
    void* rb = tagged_exact_to_rational(arena, b);
    if (!ra || !rb) { result->data.int_val = 0; return; }

    int cmp = eshkol_rational_compare(ra, rb);
    int cmp_result = 0;
    switch (op) {
        case 0: cmp_result = (cmp < 0); break;   /* lt */
        case 1: cmp_result = (cmp > 0); break;   /* gt */
        case 2: cmp_result = (cmp == 0); break;   /* eq */
        case 3: cmp_result = (cmp <= 0); break;   /* le */
        case 4: cmp_result = (cmp >= 0); break;   /* ge */
    }
    result->data.int_val = cmp_result ? 1 : 0;
}

/* Helper: extract a double from a tagged value (int, double, or rational) */
static double tagged_to_double(const eshkol_tagged_value_t* v) {
    if (v->type == ESHKOL_VALUE_INT64) return (double)v->data.int_val;
    if (v->type == ESHKOL_VALUE_DOUBLE) {
        union { int64_t i; double d; } conv;
        conv.i = v->data.int_val;
        return conv.d;
    }
    if (v->type == ESHKOL_VALUE_HEAP_PTR) {
        uint8_t* ptr = (uint8_t*)(uintptr_t)v->data.int_val;
        uint8_t subtype = *(ptr - 8);
        if (subtype == HEAP_SUBTYPE_RATIONAL) {
            return eshkol_rational_to_double((void*)(uintptr_t)v->data.int_val);
        }
    }
    return 0.0;
}

/* R7RS rationalize: find simplest rational p/q such that |x - p/q| <= epsilon.
 * Uses Stern-Brocot mediant search (simplest = smallest denominator). */
extern "C" void eshkol_rationalize_tagged(
    void* arena, const eshkol_tagged_value_t* x, const eshkol_tagged_value_t* epsilon,
    eshkol_tagged_value_t* result)
{
    memset(result, 0, sizeof(*result));
    double xd = tagged_to_double(x);
    double eps = tagged_to_double(epsilon);
    if (eps < 0) eps = -eps;

    /* Handle exact integers: if x is int and eps >= 0, return x */
    if (x->type == ESHKOL_VALUE_INT64 && eps >= 0.0) {
        result->type = ESHKOL_VALUE_INT64;
        result->data.int_val = x->data.int_val;
        return;
    }

    /* Handle special cases */
    if (xd != xd || eps != eps) { /* NaN */
        result->type = ESHKOL_VALUE_DOUBLE;
        union { double d; int64_t i; } pack;
        pack.d = xd;
        result->data.int_val = pack.i;
        return;
    }

    int negative = (xd < 0);
    if (negative) xd = -xd;

    double lo = xd - eps;
    double hi = xd + eps;
    if (lo < 0) lo = 0;

    /* If range includes 0, simplest rational is 0 */
    if (lo <= 0 && hi >= 0) {
        result->type = ESHKOL_VALUE_INT64;
        result->data.int_val = 0;
        return;
    }

    /* Stern-Brocot mediant search */
    int64_t a_num = 0, a_den = 1;  /* left bound: 0/1 */
    int64_t b_num = 1, b_den = 0;  /* right bound: 1/0 = infinity */

    /* First, advance past integers: floor(lo) */
    int64_t int_part = (int64_t)lo;
    if ((double)int_part > lo) int_part--;
    a_num = int_part;
    a_den = 1;
    b_num = int_part + 1;
    b_den = 1;

    /* If an integer is in range, return it */
    if ((double)b_num >= lo && (double)b_num <= hi) {
        int64_t val = negative ? -b_num : b_num;
        result->type = ESHKOL_VALUE_INT64;
        result->data.int_val = val;
        return;
    }
    if ((double)a_num >= lo && (double)a_num <= hi) {
        int64_t val = negative ? -a_num : a_num;
        result->type = ESHKOL_VALUE_INT64;
        result->data.int_val = val;
        return;
    }

    /* Mediant search between a_num/a_den and b_num/b_den */
    for (int iter = 0; iter < 1000; iter++) {
        int64_t m_num = a_num + b_num;
        int64_t m_den = a_den + b_den;

        /* Overflow check */
        if (m_den <= 0 || m_den > 1000000000LL) break;

        double m = (double)m_num / (double)m_den;

        if (m < lo) {
            a_num = m_num;
            a_den = m_den;
        } else if (m > hi) {
            b_num = m_num;
            b_den = m_den;
        } else {
            /* Found: m is in [lo, hi] */
            if (negative) m_num = -m_num;
            if (m_den == 1) {
                result->type = ESHKOL_VALUE_INT64;
                result->data.int_val = m_num;
            } else {
                void* rat = eshkol_rational_create(arena, m_num, m_den);
                result->type = ESHKOL_VALUE_HEAP_PTR;
                result->data.int_val = (int64_t)(uintptr_t)rat;
            }
            return;
        }
    }

    /* Fallback: return closest mediant found */
    double a_val = (double)a_num / (double)a_den;
    double b_val = (double)b_num / (double)b_den;
    int64_t best_num, best_den;
    if (fabs(a_val - xd) < fabs(b_val - xd)) {
        best_num = a_num; best_den = a_den;
    } else {
        best_num = b_num; best_den = b_den;
    }
    if (negative) best_num = -best_num;
    if (best_den == 1) {
        result->type = ESHKOL_VALUE_INT64;
        result->data.int_val = best_num;
    } else {
        void* rat = eshkol_rational_create(arena, best_num, best_den);
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.int_val = (int64_t)(uintptr_t)rat;
    }
}

/* Rounding functions for exact rationals.
 * All return exact integers (INT64 tagged values).
 * Rational invariant: denominator > 0, GCD-reduced. */

/* floor(n/d) — round toward -infinity */
extern "C" int64_t eshkol_rational_floor(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    int64_t n = rat->numerator, d = rat->denominator;
    /* C division truncates toward zero; floor needs toward -inf */
    if (n >= 0 || n % d == 0) return n / d;
    return n / d - 1;
}

/* ceil(n/d) — round toward +infinity */
extern "C" int64_t eshkol_rational_ceil(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    int64_t n = rat->numerator, d = rat->denominator;
    if (n <= 0 || n % d == 0) return n / d;
    return n / d + 1;
}

/* truncate(n/d) — round toward zero (C integer division) */
extern "C" int64_t eshkol_rational_truncate(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    return rat->numerator / rat->denominator;
}

/* round(n/d) — round to nearest, ties to even (banker's rounding) */
extern "C" int64_t eshkol_rational_round(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    int64_t n = rat->numerator, d = rat->denominator;
    /* q = truncated quotient, rem = remainder (same sign as n) */
    int64_t q = n / d;
    int64_t rem = n % d;
    if (rem < 0) rem = -rem;
    /* Compare 2*|rem| to d to decide rounding direction */
    int64_t twice_rem = 2 * rem;
    if (twice_rem > d) {
        /* Round away from zero */
        return (n >= 0) ? q + 1 : q - 1;
    } else if (twice_rem == d) {
        /* Exact half — round to even */
        if (q % 2 != 0) return (n >= 0) ? q + 1 : q - 1;
        return q;
    }
    return q;
}
