/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Rational number runtime for Eshkol
 */

#include <eshkol/core/rational.h>
#include <eshkol/eshkol.h>
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

static int64_t gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
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

/* Create a rational from 128-bit intermediates.
 * Returns NULL if result doesn't fit in int64 after GCD reduction. */
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
        return NULL; /* Signal overflow — caller falls back to double */
    }
    return eshkol_rational_create(arena, (int64_t)num, (int64_t)denom);
}

extern "C" void* eshkol_rational_create(void* arena, int64_t num, int64_t denom) {
    if (denom == 0) {
        /* Division by zero — return 0/1 as fallback */
        denom = 1;
        num = 0;
    }

    /* Normalize: denominator always positive */
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

    /* Allocate with header: 16 bytes for {numerator, denominator} */
    eshkol_rational_t* r = (eshkol_rational_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_rational_t), HEAP_SUBTYPE_RATIONAL, 0);
    r->numerator = num;
    r->denominator = denom;
    return r;
}

extern "C" void* eshkol_rational_add(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    /* a/b + c/d = (a*d + c*b) / (b*d) — uses __int128_t to prevent overflow */
    __int128_t num = (__int128_t)ra->numerator * rb->denominator
                   + (__int128_t)rb->numerator * ra->denominator;
    __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
    return rational_create_safe(arena, num, denom);
}

extern "C" void* eshkol_rational_sub(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    __int128_t num = (__int128_t)ra->numerator * rb->denominator
                   - (__int128_t)rb->numerator * ra->denominator;
    __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
    return rational_create_safe(arena, num, denom);
}

extern "C" void* eshkol_rational_mul(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    __int128_t num = (__int128_t)ra->numerator * rb->numerator;
    __int128_t denom = (__int128_t)ra->denominator * rb->denominator;
    return rational_create_safe(arena, num, denom);
}

extern "C" void* eshkol_rational_div(void* arena, void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    if (rb->numerator == 0) {
        eshkol_exception_t* exc = eshkol_make_exception(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, "rational division by zero");
        eshkol_raise(exc);
        return nullptr;  // unreachable, but satisfies compiler
    }
    __int128_t num = (__int128_t)ra->numerator * rb->denominator;
    __int128_t denom = (__int128_t)ra->denominator * rb->numerator;
    return rational_create_safe(arena, num, denom);
}

extern "C" int eshkol_rational_compare(void* a, void* b) {
    eshkol_rational_t* ra = (eshkol_rational_t*)a;
    eshkol_rational_t* rb = (eshkol_rational_t*)b;
    /* Compare a/b vs c/d → a*d vs c*b — uses __int128_t to prevent overflow */
    __int128_t lhs = (__int128_t)ra->numerator * rb->denominator;
    __int128_t rhs = (__int128_t)rb->numerator * ra->denominator;
    if (lhs < rhs) return -1;
    if (lhs > rhs) return 1;
    return 0;
}

extern "C" int64_t eshkol_rational_numerator(void* r) {
    return ((eshkol_rational_t*)r)->numerator;
}

extern "C" int64_t eshkol_rational_denominator(void* r) {
    return ((eshkol_rational_t*)r)->denominator;
}

extern "C" double eshkol_rational_to_double(void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    return (double)rat->numerator / (double)rat->denominator;
}

extern "C" int eshkol_rational_is_integer(void* r) {
    return ((eshkol_rational_t*)r)->denominator == 1;
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

/* Format rational as "num/denom" string into arena-allocated buffer with string header */
extern "C" char* eshkol_rational_to_string(void* arena, void* r) {
    eshkol_rational_t* rat = (eshkol_rational_t*)r;
    if (rat->denominator == 1) {
        char tmp[32];
        int len = snprintf(tmp, sizeof(tmp), "%lld", (long long)rat->numerator);
        char* buf = (char*)arena_allocate_string_with_header(arena, len + 1);
        if (buf) memcpy(buf, tmp, len + 1);
        return buf;
    }
    char tmp[72];
    int len = snprintf(tmp, sizeof(tmp), "%lld/%lld",
                       (long long)rat->numerator, (long long)rat->denominator);
    char* buf = (char*)arena_allocate_string_with_header(arena, len + 1);
    if (buf) memcpy(buf, tmp, len + 1);
    return buf;
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

/* Tagged value binary dispatch for rationals */
extern "C" eshkol_tagged_value_t eshkol_rational_binary_tagged(
    void* arena, eshkol_tagged_value_t a, eshkol_tagged_value_t b, int op)
{
    int a_is_rational = eshkol_is_rational_tagged(a);
    int b_is_rational = eshkol_is_rational_tagged(b);
    int a_is_int = (a.type == ESHKOL_VALUE_INT64);
    int b_is_int = (b.type == ESHKOL_VALUE_INT64);

    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));

    /* If either is a double, convert to double arithmetic */
    if (a.type == ESHKOL_VALUE_DOUBLE || b.type == ESHKOL_VALUE_DOUBLE) {
        double da, db;
        if (a_is_rational) {
            da = eshkol_rational_to_double((void*)(uintptr_t)a.data.int_val);
        } else if (a_is_int) {
            da = (double)a.data.int_val;
        } else {
            union { int64_t i; double d; } conv;
            conv.i = a.data.int_val;
            da = conv.d;
        }
        if (b_is_rational) {
            db = eshkol_rational_to_double((void*)(uintptr_t)b.data.int_val);
        } else if (b_is_int) {
            db = (double)b.data.int_val;
        } else {
            union { int64_t i; double d; } conv;
            conv.i = b.data.int_val;
            db = conv.d;
        }

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

    /* Both are exact (int or rational) */
    void* ra;
    void* rb;

    /* Convert int64 to rational 1-element if needed */
    if (a_is_rational) {
        ra = (void*)(uintptr_t)a.data.int_val;
    } else {
        ra = eshkol_rational_create(arena, a.data.int_val, 1);
    }

    if (b_is_rational) {
        rb = (void*)(uintptr_t)b.data.int_val;
    } else {
        rb = eshkol_rational_create(arena, b.data.int_val, 1);
    }

    void* rr;
    switch (op) {
        case 0: rr = eshkol_rational_add(arena, ra, rb); break;
        case 1: rr = eshkol_rational_sub(arena, ra, rb); break;
        case 2: rr = eshkol_rational_mul(arena, ra, rb); break;
        case 3: rr = eshkol_rational_div(arena, ra, rb); break;
        default: rr = eshkol_rational_create(arena, 0, 1); break;
    }

    /* NULL means overflow — fall back to double arithmetic */
    if (!rr) {
        double da = eshkol_rational_to_double(ra);
        double db = eshkol_rational_to_double(rb);
        double dr;
        switch (op) {
            case 0: dr = da + db; break;
            case 1: dr = da - db; break;
            case 2: dr = da * db; break;
            case 3: dr = db != 0.0 ? da / db : 0.0; break;
            default: dr = 0; break;
        }
        result.type = ESHKOL_VALUE_DOUBLE;
        union { double d; int64_t i; } pack;
        pack.d = dr;
        result.data.int_val = pack.i;
        return result;
    }

    /* If result is an integer (denom=1), return as INT64 */
    if (eshkol_rational_is_integer(rr)) {
        result.type = ESHKOL_VALUE_INT64;
        result.data.int_val = eshkol_rational_numerator(rr);
    } else {
        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.data.int_val = (int64_t)(uintptr_t)rr;
    }
    return result;
}

/* Pointer-based comparison dispatch for LLVM codegen.
 * Handles rational vs rational, rational vs int, int vs rational.
 * If either is a double, promotes to double comparison.
 * op: 0=lt, 1=gt, 2=eq, 3=le, 4=ge */
extern "C" void eshkol_rational_compare_tagged_ptr(
    void* arena, const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b,
    int op, eshkol_tagged_value_t* result)
{
    int a_is_rational = eshkol_is_rational_tagged(*a);
    int b_is_rational = eshkol_is_rational_tagged(*b);

    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_BOOL;

    /* If either is a double, convert to double comparison */
    if (a->type == ESHKOL_VALUE_DOUBLE || b->type == ESHKOL_VALUE_DOUBLE) {
        double da, db;
        if (a_is_rational) {
            da = eshkol_rational_to_double((void*)(uintptr_t)a->data.int_val);
        } else if (a->type == ESHKOL_VALUE_INT64) {
            da = (double)a->data.int_val;
        } else {
            union { int64_t i; double d; } conv;
            conv.i = a->data.int_val;
            da = conv.d;
        }
        if (b_is_rational) {
            db = eshkol_rational_to_double((void*)(uintptr_t)b->data.int_val);
        } else if (b->type == ESHKOL_VALUE_INT64) {
            db = (double)b->data.int_val;
        } else {
            union { int64_t i; double d; } conv;
            conv.i = b->data.int_val;
            db = conv.d;
        }
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

    /* Both are exact (int or rational) - promote int to rational n/1 */
    void* ra;
    void* rb;

    if (a_is_rational) {
        ra = (void*)(uintptr_t)a->data.int_val;
    } else {
        ra = eshkol_rational_create(arena, a->data.int_val, 1);
    }

    if (b_is_rational) {
        rb = (void*)(uintptr_t)b->data.int_val;
    } else {
        rb = eshkol_rational_create(arena, b->data.int_val, 1);
    }

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
