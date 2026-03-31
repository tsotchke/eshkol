/**
 * @file vm_rational.c
 * @brief Complete rational number runtime for the Eshkol bytecode VM.
 *
 * Implements exact rational arithmetic with the VmRational type from vm_numeric.h.
 * Arena-allocated via vm_arena.h. No GC.
 *
 * Native call IDs 330-349 (VM_NATIVE_RATIONAL_BASE + 0..19):
 *   330  vm_rational_make        (num, denom)     → rational
 *   331  vm_rational_add         (a, b)           → rational
 *   332  vm_rational_sub         (a, b)           → rational
 *   333  vm_rational_mul         (a, b)           → rational
 *   334  vm_rational_div         (a, b)           → rational
 *   335  vm_rational_neg         (a)              → rational
 *   336  vm_rational_abs         (a)              → rational
 *   337  vm_rational_inv         (a)              → rational (reciprocal)
 *   338  vm_rational_compare     (a, b)           → int (-1, 0, +1)
 *   339  vm_rational_equal       (a, b)           → bool
 *   340  vm_rational_to_double   (a)              → double
 *   341  vm_rational_from_int    (n)              → rational
 *   342  vm_rational_floor       (a)              → int64
 *   343  vm_rational_ceil        (a)              → int64
 *   344  vm_rational_truncate    (a)              → int64
 *   345  vm_rational_round       (a)              → int64 (ties to even)
 *   346  vm_rational_numerator   (a)              → int64
 *   347  vm_rational_denominator (a)              → int64
 *   348  vm_rational_gcd         (a, b)           → int64
 *   349  vm_rationalize          (x, tolerance)   → rational (Stern-Brocot)
 *
 * Key design decisions:
 *   - __int128_t intermediates for overflow protection in arithmetic and comparison.
 *   - If result overflows int64 after GCD reduction, return NULL (caller falls back to double).
 *   - Rationals are always normalized: gcd(|num|, denom) == 1, denom > 0.
 *   - Zero is represented as 0/1.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>

#include "vm_numeric.h"
#include "vm_arena.h"

/* ============================================================================
 * Utility: GCD and LCM
 * ============================================================================ */

/**
 * Euclid's algorithm on absolute values. Returns gcd >= 1 for nonzero inputs.
 * gcd(0, 0) = 0.
 */
static int64_t vm_rational_gcd_i64(int64_t a, int64_t b) {
    /* Work with absolute values. Handle INT64_MIN carefully. */
    uint64_t ua = (a < 0) ? (uint64_t)(-(a + 1)) + 1u : (uint64_t)a;
    uint64_t ub = (b < 0) ? (uint64_t)(-(b + 1)) + 1u : (uint64_t)b;

    while (ub != 0) {
        uint64_t t = ub;
        ub = ua % ub;
        ua = t;
    }
    return (int64_t)ua;
}

/**
 * LCM via gcd. Returns 0 if either input is 0.
 * Uses __int128 to detect overflow; returns -1 on overflow.
 */
static int64_t vm_rational_lcm_i64(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    int64_t g = vm_rational_gcd_i64(a, b);

    /* |a / g| * |b| — use uint64 for abs */
    uint64_t ua = (a < 0) ? (uint64_t)(-(a + 1)) + 1u : (uint64_t)a;
    uint64_t ub = (b < 0) ? (uint64_t)(-(b + 1)) + 1u : (uint64_t)b;
    uint64_t ug = (uint64_t)g;

    __uint128_t result = (__uint128_t)(ua / ug) * ub;
    if (result > (uint64_t)INT64_MAX) return -1; /* overflow */
    return (int64_t)result;
}

/* ============================================================================
 * Normalization
 * ============================================================================ */

/**
 * Normalize a rational in-place:
 *   1. denom > 0 (flip signs if denom < 0)
 *   2. gcd(|num|, denom) == 1
 *   3. 0/n → 0/1
 */
static void vm_rational_normalize(int64_t *num, int64_t *denom) {
    if (*denom == 0) {
        /* Division by zero → leave as-is, caller must handle. */
        return;
    }

    /* Ensure positive denominator */
    if (*denom < 0) {
        /* Handle INT64_MIN: we can't negate it, so fail gracefully. */
        if (*denom == INT64_MIN || *num == INT64_MIN) {
            /* Rare edge case. Use __int128 for the flip. */
            __int128_t n128 = -(__int128_t)*num;
            __int128_t d128 = -(__int128_t)*denom;
            if (n128 > INT64_MAX || n128 < INT64_MIN ||
                d128 > INT64_MAX || d128 < INT64_MIN) {
                /* Truly pathological — can't represent. Leave denom negative as signal. */
                return;
            }
            *num = (int64_t)n128;
            *denom = (int64_t)d128;
        } else {
            *num = -*num;
            *denom = -*denom;
        }
    }

    /* Zero numerator → canonical 0/1 */
    if (*num == 0) {
        *denom = 1;
        return;
    }

    /* Divide by GCD */
    int64_t g = vm_rational_gcd_i64(*num, *denom);
    if (g > 1) {
        *num /= g;
        *denom /= g;
    }
}

/* ============================================================================
 * Arena Allocation
 * ============================================================================ */

/**
 * Allocate a VmRational on the arena, pre-normalized. Returns NULL if denom==0.
 */
static VmRational* vm_rational_alloc(VmArena *arena, int64_t num, int64_t denom) {
    if (denom == 0) return NULL;

    vm_rational_normalize(&num, &denom);

    VmRational *r = (VmRational *)vm_arena_alloc_object(arena, VM_SUBTYPE_RATIONAL,
                                                         sizeof(VmRational));
    if (!r) return NULL;
    r->num = num;
    r->denom = denom;
    return r;
}

/* ============================================================================
 * Construction
 * ============================================================================ */

/** Create rational num/denom (native ID 330). */
VmRational* vm_rational_make(VmArena *arena, int64_t num, int64_t denom) {
    return vm_rational_alloc(arena, num, denom);
}

/** Create rational from integer n/1 (native ID 341). */
VmRational* vm_rational_from_int(VmArena *arena, int64_t n) {
    return vm_rational_alloc(arena, n, 1);
}

/* ============================================================================
 * Accessors (native IDs 346, 347)
 * ============================================================================ */

int64_t vm_rational_numerator(const VmRational *r) {
    return r ? r->num : 0;
}

int64_t vm_rational_denominator(const VmRational *r) {
    return r ? r->denom : 1;
}

/* ============================================================================
 * Conversion (native ID 340)
 * ============================================================================ */

double vm_rational_to_double(const VmRational *r) {
    if (!r) return 0.0;
    return (double)r->num / (double)r->denom;
}

/* ============================================================================
 * Arithmetic: __int128_t intermediates, NULL on overflow
 * ============================================================================ */

/**
 * Try to reduce and pack an __int128 numerator/denominator into a new VmRational.
 * Returns NULL if the reduced form doesn't fit in int64.
 */
static VmRational* vm_rational_pack128(VmArena *arena, __int128_t num, __int128_t denom) {
    if (denom == 0) return NULL;

    /* Ensure positive denominator */
    if (denom < 0) {
        num = -num;
        denom = -denom;
    }

    /* Zero → 0/1 */
    if (num == 0) {
        return vm_rational_alloc(arena, 0, 1);
    }

    /* GCD reduction on 128-bit values */
    __int128_t a128 = (num < 0) ? -num : num;
    __int128_t b128 = denom;
    while (b128 != 0) {
        __int128_t t = b128;
        b128 = a128 % b128;
        a128 = t;
    }
    /* a128 = gcd */
    if (a128 > 1) {
        num /= a128;
        denom /= a128;
    }

    /* Check int64 range */
    if (num > (__int128_t)INT64_MAX || num < (__int128_t)INT64_MIN ||
        denom > (__int128_t)INT64_MAX) {
        return NULL; /* overflow — caller falls back to double */
    }

    return vm_rational_alloc(arena, (int64_t)num, (int64_t)denom);
}

/** a + b (native ID 331). */
VmRational* vm_rational_add(VmArena *arena, const VmRational *a, const VmRational *b) {
    if (!a || !b) return NULL;
    /* a.num/a.denom + b.num/b.denom = (a.num*b.denom + b.num*a.denom) / (a.denom*b.denom) */
    __int128_t num = (__int128_t)a->num * b->denom + (__int128_t)b->num * a->denom;
    __int128_t denom = (__int128_t)a->denom * b->denom;
    return vm_rational_pack128(arena, num, denom);
}

/** a - b (native ID 332). */
VmRational* vm_rational_sub(VmArena *arena, const VmRational *a, const VmRational *b) {
    if (!a || !b) return NULL;
    __int128_t num = (__int128_t)a->num * b->denom - (__int128_t)b->num * a->denom;
    __int128_t denom = (__int128_t)a->denom * b->denom;
    return vm_rational_pack128(arena, num, denom);
}

/** a * b (native ID 333). */
VmRational* vm_rational_mul(VmArena *arena, const VmRational *a, const VmRational *b) {
    if (!a || !b) return NULL;
    /* Cross-reduce before multiplying to keep intermediates smaller */
    int64_t g1 = vm_rational_gcd_i64(a->num, b->denom);
    int64_t g2 = vm_rational_gcd_i64(b->num, a->denom);
    __int128_t num = (__int128_t)(a->num / g1) * (b->num / g2);
    __int128_t denom = (__int128_t)(a->denom / g2) * (b->denom / g1);
    return vm_rational_pack128(arena, num, denom);
}

/** a / b (native ID 334). */
VmRational* vm_rational_div(VmArena *arena, const VmRational *a, const VmRational *b) {
    if (!a || !b || b->num == 0) return NULL;
    /* a/b = a * (b.denom / b.num) */
    /* Construct reciprocal of b inline */
    int64_t b_num = b->denom;
    int64_t b_denom = b->num;
    /* Fix sign: if b->num was negative, flip */
    if (b_denom < 0) {
        b_num = -b_num;
        b_denom = -b_denom;
    }
    /* Now multiply a by (b_num / b_denom) using cross-reduction */
    int64_t g1 = vm_rational_gcd_i64(a->num, b_denom);
    int64_t g2 = vm_rational_gcd_i64(b_num, a->denom);
    __int128_t num = (__int128_t)(a->num / g1) * (b_num / g2);
    __int128_t denom = (__int128_t)(a->denom / g2) * (b_denom / g1);
    return vm_rational_pack128(arena, num, denom);
}

/** -a (native ID 335). */
VmRational* vm_rational_neg(VmArena *arena, const VmRational *a) {
    if (!a) return NULL;
    /* Handle INT64_MIN numerator via 128-bit */
    __int128_t num = -(__int128_t)a->num;
    if (num > INT64_MAX || num < INT64_MIN) return NULL;
    return vm_rational_alloc(arena, (int64_t)num, a->denom);
}

/** |a| (native ID 336). */
VmRational* vm_rational_abs(VmArena *arena, const VmRational *a) {
    if (!a) return NULL;
    int64_t n = a->num;
    if (n < 0) {
        __int128_t neg = -(__int128_t)n;
        if (neg > INT64_MAX) return NULL;
        n = (int64_t)neg;
    }
    return vm_rational_alloc(arena, n, a->denom);
}

/** 1/a (native ID 337). */
VmRational* vm_rational_inv(VmArena *arena, const VmRational *a) {
    if (!a || a->num == 0) return NULL;
    /* Reciprocal: denom/num, normalize handles sign */
    return vm_rational_alloc(arena, a->denom, a->num);
}

/* ============================================================================
 * Comparison (native IDs 338, 339)
 * ============================================================================ */

/**
 * Compare a and b. Returns -1, 0, or +1.
 * Uses cross-multiplication with __int128_t to avoid division.
 */
int vm_rational_compare(const VmRational *a, const VmRational *b) {
    if (!a || !b) return 0;
    /* Since denoms are always positive (normalized), we can cross-multiply directly.
     * a/b vs c/d ↔ a*d vs c*b (denom positive, so inequality direction preserved). */
    __int128_t lhs = (__int128_t)a->num * b->denom;
    __int128_t rhs = (__int128_t)b->num * a->denom;
    if (lhs < rhs) return -1;
    if (lhs > rhs) return  1;
    return 0;
}

/** Exact equality (native ID 339). */
bool vm_rational_equal(const VmRational *a, const VmRational *b) {
    if (!a || !b) return (a == b);
    /* Both normalized: direct field comparison suffices. */
    return a->num == b->num && a->denom == b->denom;
}

/* ============================================================================
 * Rounding: floor, ceil, truncate, round (native IDs 342-345)
 *
 * Key: C integer division truncates toward zero. For floor/ceil of negative
 * rationals, we must adjust.
 * ============================================================================ */

/**
 * Floor: largest integer <= a.
 * floor(7/3) = 2, floor(-7/3) = -3.
 */
int64_t vm_rational_floor(const VmRational *a) {
    if (!a) return 0;
    int64_t n = a->num;
    int64_t d = a->denom; /* always > 0 */

    if (n >= 0) {
        return n / d;
    } else {
        /* For negative numerator: floor division.
         * floor(n/d) = -((-n - 1) / d) - 1  when d > 0 and n < 0.
         * This avoids issues with C's truncation-toward-zero. */
        return -(((-n) - 1) / d) - 1;
    }
}

/**
 * Ceiling: smallest integer >= a.
 * ceil(7/3) = 3, ceil(-7/3) = -2.
 */
int64_t vm_rational_ceil(const VmRational *a) {
    if (!a) return 0;
    int64_t n = a->num;
    int64_t d = a->denom;

    if (n >= 0) {
        /* ceil(n/d) = (n + d - 1) / d for positive values. */
        return (n + d - 1) / d;
    } else {
        /* For negative: ceil(n/d) = -((-n) / d) */
        return -((-n) / d);
    }
}

/**
 * Truncate: integer part, rounding toward zero.
 * truncate(7/3) = 2, truncate(-7/3) = -2.
 */
int64_t vm_rational_truncate(const VmRational *a) {
    if (!a) return 0;
    /* C division truncates toward zero — exactly what we want. */
    return a->num / a->denom;
}

/**
 * Round: nearest integer, ties to even (R7RS banker's rounding).
 * round(5/2) = 2, round(7/2) = 4, round(3/2) = 2, round(-5/2) = -2.
 */
int64_t vm_rational_round(const VmRational *a) {
    if (!a) return 0;
    int64_t n = a->num;
    int64_t d = a->denom;

    /* Compute floor and remainder */
    int64_t fl = vm_rational_floor(a);
    /* remainder = n - fl * d  (should be in [0, d)) */
    int64_t rem = n - fl * d;

    /* rem is in [0, d) because fl = floor(n/d). */
    /* Compare 2*rem with d to determine rounding direction. */
    /* Use __int128 to avoid overflow in 2*rem. */
    __int128_t twice_rem = (__int128_t)2 * rem;
    __int128_t dd = (__int128_t)d;

    if (twice_rem < dd) {
        /* Closer to floor */
        return fl;
    } else if (twice_rem > dd) {
        /* Closer to ceil */
        return fl + 1;
    } else {
        /* Exact midpoint: ties to even */
        if (fl % 2 == 0) {
            return fl;       /* fl is even, stay */
        } else {
            return fl + 1;   /* fl is odd, round up to even */
        }
    }
}

/* ============================================================================
 * GCD / LCM on integers (native ID 348 — exposed as builtins)
 * ============================================================================ */

int64_t vm_rational_gcd(int64_t a, int64_t b) {
    return vm_rational_gcd_i64(a, b);
}

int64_t vm_rational_lcm(int64_t a, int64_t b) {
    return vm_rational_lcm_i64(a, b);
}

/* ============================================================================
 * Rationalize: Stern-Brocot mediant search (native ID 349)
 *
 * Find the simplest rational p/q such that |p/q - x| <= tolerance.
 * "Simplest" means smallest denominator, then smallest |numerator|.
 * Uses the Stern-Brocot tree / mediant algorithm, max 1000 iterations.
 * ============================================================================ */

VmRational* vm_rationalize(VmArena *arena, double x, double tolerance) {
    if (!isfinite(x) || !isfinite(tolerance) || tolerance < 0.0) return NULL;

    /* Handle negative x by rationalizing |x| and negating */
    int negate = 0;
    if (x < 0.0) {
        x = -x;
        negate = 1;
    }

    /* Target interval [x_lo, x_hi] */
    double x_lo = x - tolerance;
    double x_hi = x + tolerance;
    if (x_lo < 0.0) x_lo = 0.0;

    /* If the interval includes an integer, that's simplest */
    int64_t lo_int = (int64_t)ceil(x_lo);
    if ((double)lo_int <= x_hi) {
        int64_t result = lo_int;
        if (negate) result = -result;
        return vm_rational_alloc(arena, result, 1);
    }

    /* Stern-Brocot mediant search with binary-search acceleration.
     *
     * Maintain Stern-Brocot ancestors: left = a/b, right = c/d.
     * Invariant: a/b < x_lo and c/d > x_hi.
     *
     * When the mediant lands on the same side k times in a row, the naive
     * algorithm takes k one-step iterations. Instead, we binary-search for
     * the largest k such that (a + k*c) / (b + k*d) is still on that side,
     * then jump. This gives O(log(denom)) convergence — equivalent to
     * continued fraction convergents.
     */
    int64_t a = (int64_t)floor(x_lo);
    int64_t b = 1;
    int64_t c = a + 1;
    int64_t d = 1;

    int64_t best_num = a;
    int64_t best_denom = b;

    for (int iter = 0; iter < 1000; iter++) {
        /* Mediant (a+c)/(b+d) */
        int64_t med_num = a + c;
        int64_t med_denom = b + d;

        /* Overflow guard */
        if (med_num < 0 || med_denom < 0 ||
            med_denom > 1000000000LL) {
            break;
        }

        double med_val = (double)med_num / (double)med_denom;

        if (med_val < x_lo) {
            /* Mediant is below interval. We need to move left bound up.
             * Binary-search: find largest k where (a + k*c)/(b + k*d) < x_lo. */
            int64_t lo_k = 1, hi_k = 1;
            /* Exponential search for upper bound */
            while (1) {
                int64_t tn = a + hi_k * c;
                int64_t td = b + hi_k * d;
                if (td > 1000000000LL || td < 0) { hi_k--; break; }
                double tv = (double)tn / (double)td;
                if (tv >= x_lo) break;
                lo_k = hi_k;
                hi_k *= 2;
            }
            /* Binary search in [lo_k, hi_k] */
            while (lo_k < hi_k) {
                int64_t mid = lo_k + (hi_k - lo_k + 1) / 2;
                int64_t tn = a + mid * c;
                int64_t td = b + mid * d;
                if (td > 1000000000LL || td < 0) { hi_k = mid - 1; continue; }
                double tv = (double)tn / (double)td;
                if (tv < x_lo) {
                    lo_k = mid;
                } else {
                    hi_k = mid - 1;
                }
            }
            a = a + lo_k * c;
            b = b + lo_k * d;
            /* Now check mediant (a+c)/(b+d) — should be >= x_lo */

        } else if (med_val > x_hi) {
            /* Mediant is above interval. Move right bound down.
             * Binary-search: find largest k where (a + c*1)/(b + d*1) variant...
             * Symmetric: find largest k where (k*a + c)/(k*b + d) > x_hi. */
            int64_t lo_k = 1, hi_k = 1;
            while (1) {
                int64_t tn = hi_k * a + c;
                int64_t td = hi_k * b + d;
                if (td > 1000000000LL || td < 0) { hi_k--; break; }
                double tv = (double)tn / (double)td;
                if (tv <= x_hi) break;
                lo_k = hi_k;
                hi_k *= 2;
            }
            while (lo_k < hi_k) {
                int64_t mid = lo_k + (hi_k - lo_k + 1) / 2;
                int64_t tn = mid * a + c;
                int64_t td = mid * b + d;
                if (td > 1000000000LL || td < 0) { hi_k = mid - 1; continue; }
                double tv = (double)tn / (double)td;
                if (tv > x_hi) {
                    lo_k = mid;
                } else {
                    hi_k = mid - 1;
                }
            }
            c = lo_k * a + c;
            d = lo_k * b + d;

        } else {
            /* Mediant is inside the interval — this is the simplest fraction
             * (Stern-Brocot property: first mediant to enter = smallest denom). */
            best_num = med_num;
            best_denom = med_denom;
            break;
        }
    }

    if (negate) best_num = -best_num;
    return vm_rational_alloc(arena, best_num, best_denom);
}

/* ============================================================================
 * Native dispatch: called from VM's OP_NATIVE_CALL handler
 * ============================================================================ */

/**
 * Dispatch rational native calls. The VM pushes arguments onto its value stack.
 * This function is intended to be called from the main VM loop.
 *
 * @param id        Native call ID (330-349)
 * @param arena     Active arena for allocation
 * @param args      Pointer to argument VmRational pointers
 * @param int_args  Pointer to integer arguments (for make, from_int, gcd)
 * @param n_args    Number of arguments
 * @param out_rat   Output rational (for functions returning rational)
 * @param out_int   Output integer (for floor, ceil, etc.)
 * @param out_dbl   Output double (for to_double)
 * @return          0 on success, -1 on error
 */
int vm_rational_dispatch(int id, VmArena *arena,
                         const VmRational **args, const int64_t *int_args,
                         int n_args,
                         VmRational **out_rat, int64_t *out_int, double *out_dbl) {
    int offset = id - VM_NATIVE_RATIONAL_BASE;
    VmRational *result;

    switch (offset) {
    case 0: /* make(num, denom) */
        if (n_args < 2) return -1;
        result = vm_rational_make(arena, int_args[0], int_args[1]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 1: /* add(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        result = vm_rational_add(arena, args[0], args[1]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 2: /* sub(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        result = vm_rational_sub(arena, args[0], args[1]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 3: /* mul(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        result = vm_rational_mul(arena, args[0], args[1]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 4: /* div(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        result = vm_rational_div(arena, args[0], args[1]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 5: /* neg(a) */
        if (n_args < 1 || !args[0]) return -1;
        result = vm_rational_neg(arena, args[0]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 6: /* abs(a) */
        if (n_args < 1 || !args[0]) return -1;
        result = vm_rational_abs(arena, args[0]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 7: /* inv(a) */
        if (n_args < 1 || !args[0]) return -1;
        result = vm_rational_inv(arena, args[0]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 8: /* compare(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        *out_int = vm_rational_compare(args[0], args[1]);
        return 0;

    case 9: /* equal(a, b) */
        if (n_args < 2 || !args[0] || !args[1]) return -1;
        *out_int = vm_rational_equal(args[0], args[1]) ? 1 : 0;
        return 0;

    case 10: /* to_double(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_dbl = vm_rational_to_double(args[0]);
        return 0;

    case 11: /* from_int(n) */
        if (n_args < 1) return -1;
        result = vm_rational_from_int(arena, int_args[0]);
        if (!result) return -1;
        *out_rat = result;
        return 0;

    case 12: /* floor(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_floor(args[0]);
        return 0;

    case 13: /* ceil(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_ceil(args[0]);
        return 0;

    case 14: /* truncate(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_truncate(args[0]);
        return 0;

    case 15: /* round(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_round(args[0]);
        return 0;

    case 16: /* numerator(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_numerator(args[0]);
        return 0;

    case 17: /* denominator(a) */
        if (n_args < 1 || !args[0]) return -1;
        *out_int = vm_rational_denominator(args[0]);
        return 0;

    case 18: /* gcd(a, b) — integer GCD */
        if (n_args < 2) return -1;
        *out_int = vm_rational_gcd(int_args[0], int_args[1]);
        return 0;

    case 19: /* rationalize(x, tolerance) */
        if (n_args < 2) return -1;
        {
            double x_val, tol_val;
            /* Arguments may come as doubles via out_dbl hack, or via int_args. */
            /* We expect the caller to pass doubles via a double array.
             * For flexibility, accept int_args as fixed-point or use the dbl fields. */
            /* The caller is expected to put x in out_dbl[0] and tolerance in out_dbl[1]
             * before calling. We'll use a simpler interface for self-test. */
            /* For the dispatch interface, we reinterpret int_args as double bits. */
            union { int64_t i; double d; } u0, u1;
            u0.i = int_args[0]; x_val = u0.d;
            u1.i = int_args[1]; tol_val = u1.d;
            result = vm_rationalize(arena, x_val, tol_val);
            if (!result) return -1;
            *out_rat = result;
        }
        return 0;

    default:
        return -1;
    }
}

/* ============================================================================
 * Self-Test (compile with -DVM_RATIONAL_TEST)
 * ============================================================================ */

#ifdef VM_RATIONAL_TEST

#include <stdio.h>
#include <assert.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  test %d: %-44s ", test_count, name); } while(0)
#define PASS() do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL — %s\n", msg); } while(0)

#define ASSERT_RAT(r, en, ed, msg) do { \
    if (!(r)) { FAIL("NULL result: " msg); break; } \
    if ((r)->num != (en) || (r)->denom != (ed)) { \
        printf("FAIL — %s: got %lld/%lld, expected %lld/%lld\n", \
               msg, (long long)(r)->num, (long long)(r)->denom, \
               (long long)(en), (long long)(ed)); \
        break; \
    } \
    PASS(); \
} while(0)

#define ASSERT_INT(got, expected, msg) do { \
    if ((got) != (expected)) { \
        printf("FAIL — %s: got %lld, expected %lld\n", msg, \
               (long long)(got), (long long)(expected)); \
        break; \
    } \
    PASS(); \
} while(0)

int main(void) {
    printf("vm_rational self-test\n");
    printf("==================================================\n");

    VmArena arena;
    vm_arena_init(&arena, 0);

    /* ── GCD ── */
    TEST("gcd(12, 18) = 6");
    ASSERT_INT(vm_rational_gcd(12, 18), 6, "gcd(12,18)");

    TEST("gcd(0, 5) = 5");
    ASSERT_INT(vm_rational_gcd(0, 5), 5, "gcd(0,5)");

    TEST("gcd(7, 0) = 7");
    ASSERT_INT(vm_rational_gcd(7, 0), 7, "gcd(7,0)");

    TEST("gcd(0, 0) = 0");
    ASSERT_INT(vm_rational_gcd(0, 0), 0, "gcd(0,0)");

    TEST("gcd(-12, 18) = 6");
    ASSERT_INT(vm_rational_gcd(-12, 18), 6, "gcd(-12,18)");

    /* ── LCM ── */
    TEST("lcm(4, 6) = 12");
    ASSERT_INT(vm_rational_lcm(4, 6), 12, "lcm(4,6)");

    TEST("lcm(0, 5) = 0");
    ASSERT_INT(vm_rational_lcm(0, 5), 0, "lcm(0,5)");

    /* ── Normalization ── */
    TEST("normalize(-6, -9) = (2, 3)");
    {
        VmRational *r = vm_rational_make(&arena, -6, -9);
        ASSERT_RAT(r, 2, 3, "normalize(-6,-9)");
    }

    TEST("normalize(0, 5) = (0, 1)");
    {
        VmRational *r = vm_rational_make(&arena, 0, 5);
        ASSERT_RAT(r, 0, 1, "normalize(0,5)");
    }

    TEST("normalize(6, -9) = (-2, 3)");
    {
        VmRational *r = vm_rational_make(&arena, 6, -9);
        ASSERT_RAT(r, -2, 3, "normalize(6,-9)");
    }

    /* ── Addition ── */
    TEST("1/3 + 1/6 = 1/2");
    {
        VmRational *a = vm_rational_make(&arena, 1, 3);
        VmRational *b = vm_rational_make(&arena, 1, 6);
        VmRational *r = vm_rational_add(&arena, a, b);
        ASSERT_RAT(r, 1, 2, "1/3 + 1/6");
    }

    TEST("1/2 + 1/2 = 1/1");
    {
        VmRational *a = vm_rational_make(&arena, 1, 2);
        VmRational *r = vm_rational_add(&arena, a, a);
        ASSERT_RAT(r, 1, 1, "1/2 + 1/2");
    }

    TEST("-1/3 + 1/3 = 0/1");
    {
        VmRational *a = vm_rational_make(&arena, -1, 3);
        VmRational *b = vm_rational_make(&arena, 1, 3);
        VmRational *r = vm_rational_add(&arena, a, b);
        ASSERT_RAT(r, 0, 1, "-1/3 + 1/3");
    }

    /* ── Subtraction ── */
    TEST("3/4 - 1/4 = 1/2");
    {
        VmRational *a = vm_rational_make(&arena, 3, 4);
        VmRational *b = vm_rational_make(&arena, 1, 4);
        VmRational *r = vm_rational_sub(&arena, a, b);
        ASSERT_RAT(r, 1, 2, "3/4 - 1/4");
    }

    /* ── Multiplication ── */
    TEST("3/4 * 2/3 = 1/2");
    {
        VmRational *a = vm_rational_make(&arena, 3, 4);
        VmRational *b = vm_rational_make(&arena, 2, 3);
        VmRational *r = vm_rational_mul(&arena, a, b);
        ASSERT_RAT(r, 1, 2, "3/4 * 2/3");
    }

    TEST("0/1 * 5/7 = 0/1");
    {
        VmRational *a = vm_rational_make(&arena, 0, 1);
        VmRational *b = vm_rational_make(&arena, 5, 7);
        VmRational *r = vm_rational_mul(&arena, a, b);
        ASSERT_RAT(r, 0, 1, "0 * 5/7");
    }

    /* ── Division ── */
    TEST("1/2 / 3/4 = 2/3");
    {
        VmRational *a = vm_rational_make(&arena, 1, 2);
        VmRational *b = vm_rational_make(&arena, 3, 4);
        VmRational *r = vm_rational_div(&arena, a, b);
        ASSERT_RAT(r, 2, 3, "1/2 / 3/4");
    }

    TEST("div by zero → NULL");
    {
        VmRational *a = vm_rational_make(&arena, 1, 2);
        VmRational *b = vm_rational_make(&arena, 0, 1);
        VmRational *r = vm_rational_div(&arena, a, b);
        if (r != NULL) { FAIL("expected NULL for div-by-zero"); }
        else { PASS(); }
    }

    /* ── Negation ── */
    TEST("neg(3/5) = -3/5");
    {
        VmRational *a = vm_rational_make(&arena, 3, 5);
        VmRational *r = vm_rational_neg(&arena, a);
        ASSERT_RAT(r, -3, 5, "neg(3/5)");
    }

    /* ── Absolute value ── */
    TEST("abs(-7/3) = 7/3");
    {
        VmRational *a = vm_rational_make(&arena, -7, 3);
        VmRational *r = vm_rational_abs(&arena, a);
        ASSERT_RAT(r, 7, 3, "abs(-7/3)");
    }

    /* ── Reciprocal ── */
    TEST("inv(3/7) = 7/3");
    {
        VmRational *a = vm_rational_make(&arena, 3, 7);
        VmRational *r = vm_rational_inv(&arena, a);
        ASSERT_RAT(r, 7, 3, "inv(3/7)");
    }

    TEST("inv(-2/5) = -5/2");
    {
        VmRational *a = vm_rational_make(&arena, -2, 5);
        VmRational *r = vm_rational_inv(&arena, a);
        ASSERT_RAT(r, -5, 2, "inv(-2/5)");
    }

    /* ── Comparison ── */
    TEST("compare(1/3, 1/2) = -1");
    {
        VmRational *a = vm_rational_make(&arena, 1, 3);
        VmRational *b = vm_rational_make(&arena, 1, 2);
        ASSERT_INT(vm_rational_compare(a, b), -1, "1/3 < 1/2");
    }

    TEST("compare(1/2, 1/3) = 1");
    {
        VmRational *a = vm_rational_make(&arena, 1, 2);
        VmRational *b = vm_rational_make(&arena, 1, 3);
        ASSERT_INT(vm_rational_compare(a, b), 1, "1/2 > 1/3");
    }

    TEST("compare(2/4, 1/2) = 0");
    {
        VmRational *a = vm_rational_make(&arena, 2, 4);
        VmRational *b = vm_rational_make(&arena, 1, 2);
        ASSERT_INT(vm_rational_compare(a, b), 0, "2/4 == 1/2");
    }

    TEST("equal(2/4, 1/2) = true");
    {
        VmRational *a = vm_rational_make(&arena, 2, 4);
        VmRational *b = vm_rational_make(&arena, 1, 2);
        if (!vm_rational_equal(a, b)) { FAIL("2/4 should equal 1/2"); }
        else { PASS(); }
    }

    /* ── Conversion ── */
    TEST("to_double(1/4) = 0.25");
    {
        VmRational *a = vm_rational_make(&arena, 1, 4);
        double d = vm_rational_to_double(a);
        if (fabs(d - 0.25) > 1e-15) { FAIL("to_double(1/4) != 0.25"); }
        else { PASS(); }
    }

    TEST("from_int(42) = 42/1");
    {
        VmRational *r = vm_rational_from_int(&arena, 42);
        ASSERT_RAT(r, 42, 1, "from_int(42)");
    }

    /* ── Floor ── */
    TEST("floor(7/3) = 2");
    {
        VmRational *a = vm_rational_make(&arena, 7, 3);
        ASSERT_INT(vm_rational_floor(a), 2, "floor(7/3)");
    }

    TEST("floor(-7/3) = -3");
    {
        VmRational *a = vm_rational_make(&arena, -7, 3);
        ASSERT_INT(vm_rational_floor(a), -3, "floor(-7/3)");
    }

    TEST("floor(6/3) = 2 (exact)");
    {
        VmRational *a = vm_rational_make(&arena, 6, 3);
        ASSERT_INT(vm_rational_floor(a), 2, "floor(6/3)");
    }

    TEST("floor(-6/3) = -2 (exact)");
    {
        VmRational *a = vm_rational_make(&arena, -6, 3);
        ASSERT_INT(vm_rational_floor(a), -2, "floor(-6/3)");
    }

    /* ── Ceil ── */
    TEST("ceil(7/3) = 3");
    {
        VmRational *a = vm_rational_make(&arena, 7, 3);
        ASSERT_INT(vm_rational_ceil(a), 3, "ceil(7/3)");
    }

    TEST("ceil(-7/3) = -2");
    {
        VmRational *a = vm_rational_make(&arena, -7, 3);
        ASSERT_INT(vm_rational_ceil(a), -2, "ceil(-7/3)");
    }

    /* ── Truncate ── */
    TEST("truncate(7/3) = 2");
    {
        VmRational *a = vm_rational_make(&arena, 7, 3);
        ASSERT_INT(vm_rational_truncate(a), 2, "truncate(7/3)");
    }

    TEST("truncate(-7/3) = -2");
    {
        VmRational *a = vm_rational_make(&arena, -7, 3);
        ASSERT_INT(vm_rational_truncate(a), -2, "truncate(-7/3)");
    }

    /* ── Round (ties to even) ── */
    TEST("round(5/2) = 2 (tie → even)");
    {
        VmRational *a = vm_rational_make(&arena, 5, 2);
        ASSERT_INT(vm_rational_round(a), 2, "round(5/2)");
    }

    TEST("round(7/2) = 4 (tie → even)");
    {
        VmRational *a = vm_rational_make(&arena, 7, 2);
        ASSERT_INT(vm_rational_round(a), 4, "round(7/2)");
    }

    TEST("round(3/2) = 2 (tie → even)");
    {
        VmRational *a = vm_rational_make(&arena, 3, 2);
        ASSERT_INT(vm_rational_round(a), 2, "round(3/2)");
    }

    TEST("round(-5/2) = -2 (tie → even)");
    {
        VmRational *a = vm_rational_make(&arena, -5, 2);
        ASSERT_INT(vm_rational_round(a), -2, "round(-5/2)");
    }

    TEST("round(7/4) = 2 (not a tie)");
    {
        VmRational *a = vm_rational_make(&arena, 7, 4);
        ASSERT_INT(vm_rational_round(a), 2, "round(7/4)");
    }

    TEST("round(1/4) = 0");
    {
        VmRational *a = vm_rational_make(&arena, 1, 4);
        ASSERT_INT(vm_rational_round(a), 0, "round(1/4)");
    }

    /* ── Rationalize (Stern-Brocot) ── */
    TEST("rationalize(0.333, 0.01) = 1/3");
    {
        VmRational *r = vm_rationalize(&arena, 0.333, 0.01);
        ASSERT_RAT(r, 1, 3, "rationalize(0.333, 0.01)");
    }

    TEST("rationalize(3.14159, 0.01) = 22/7");
    {
        /* |22/7 - 3.14159| = 0.00127, within 0.01 tolerance.
         * 22/7 has the smallest denominator in the interval. */
        VmRational *r = vm_rationalize(&arena, 3.14159, 0.01);
        ASSERT_RAT(r, 22, 7, "rationalize(3.14159, 0.01)");
    }

    TEST("rationalize(0.5, 0.0) = 1/2");
    {
        VmRational *r = vm_rationalize(&arena, 0.5, 0.0);
        ASSERT_RAT(r, 1, 2, "rationalize(0.5, 0.0)");
    }

    TEST("rationalize(-0.333, 0.01) = -1/3");
    {
        VmRational *r = vm_rationalize(&arena, -0.333, 0.01);
        ASSERT_RAT(r, -1, 3, "rationalize(-0.333, 0.01)");
    }

    TEST("rationalize(0.0, 0.0) = 0/1");
    {
        VmRational *r = vm_rationalize(&arena, 0.0, 0.0);
        ASSERT_RAT(r, 0, 1, "rationalize(0.0, 0.0)");
    }

    TEST("rationalize(1.0, 0.0) = 1/1");
    {
        VmRational *r = vm_rationalize(&arena, 1.0, 0.0);
        ASSERT_RAT(r, 1, 1, "rationalize(1.0, 0.0)");
    }

    /* ── Large number stress ── */
    TEST("large: 999999999/1000000000 + 1/1000000000 = 1/1");
    {
        VmRational *a = vm_rational_make(&arena, 999999999LL, 1000000000LL);
        VmRational *b = vm_rational_make(&arena, 1, 1000000000LL);
        VmRational *r = vm_rational_add(&arena, a, b);
        ASSERT_RAT(r, 1, 1, "large addition");
    }

    /* ── Dispatch interface ── */
    TEST("dispatch: add via native ID 331");
    {
        VmRational *a = vm_rational_make(&arena, 1, 4);
        VmRational *b = vm_rational_make(&arena, 1, 4);
        const VmRational *args[2] = {a, b};
        int64_t int_args[2] = {0, 0};
        VmRational *out_rat = NULL;
        int64_t out_int = 0;
        double out_dbl = 0.0;
        int rc = vm_rational_dispatch(331, &arena, args, int_args, 2,
                                       &out_rat, &out_int, &out_dbl);
        if (rc != 0) { FAIL("dispatch returned error"); }
        else { ASSERT_RAT(out_rat, 1, 2, "dispatch add"); }
    }

    TEST("dispatch: floor via native ID 342");
    {
        VmRational *a = vm_rational_make(&arena, 7, 3);
        const VmRational *args[1] = {a};
        int64_t int_args[1] = {0};
        VmRational *out_rat = NULL;
        int64_t out_int = 0;
        double out_dbl = 0.0;
        int rc = vm_rational_dispatch(342, &arena, args, int_args, 1,
                                       &out_rat, &out_int, &out_dbl);
        if (rc != 0) { FAIL("dispatch returned error"); }
        else { ASSERT_INT(out_int, 2, "dispatch floor(7/3)"); }
    }

    /* ── Summary ── */
    printf("==================================================\n");
    printf("Results: %d/%d passed\n", pass_count, test_count);
    if (pass_count == test_count) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILURES: %d\n", test_count - pass_count);
    }

    vm_arena_destroy(&arena);
    return (pass_count == test_count) ? 0 : 1;
}

#endif /* VM_RATIONAL_TEST */
