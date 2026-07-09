/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * runtime_taylor.c -- arbitrary-order forward-mode AD kernel (ESH-0186, P1).
 *
 * Univariate truncated-Taylor arithmetic ("Taylor tower"): a value is carried
 * as its coefficient array c[0..K] of the truncated series
 *
 *     f(x0 + t) = sum_{k=0..K} c[k] * t^k        =>   f^(n)(x0) = n! * c[n].
 *
 * The differentiation variable is seeded {x0, 1, 0, ...}; a constant seeds
 * {v, 0, ...}. All recurrences are the closed forms proven correct to d=8 in
 * the Phase-0 POC (tests/ad_taylor_poc/taylor_poc.c) and documented in
 * docs/design/AD_TAYLOR_TOWER.md section 5. They are the single computational
 * kernel behind arbitrary-order `derivative`, `(derivative-n k)` and `taylor`.
 *
 * FP-contraction policy (design section 6a): every multiply-accumulate in the
 * convolution recurrences uses an explicit fma() with the reduction order
 * fixed to ascending j, so the runtime kernel and (future, P2) unrolled-IR
 * tier are bit-exact-reconcilable rather than merely tight-ULP.
 *
 * Perturbation-confusion safety (design section 5a): each active
 * differentiation context carries a 16-bit EPOCH tag in esh_taylor_t.flags.
 * A binary op combines the full series of two towers only when their epochs
 * match; a foreign-epoch tower (an outer/inner level) is lifted to a constant
 * (its c[0]) with respect to the current (innermost = highest-epoch) level,
 * exactly as JAX lifts a value from an outer trace.
 *
 * EXACT-COEFFICIENT towers (ESH-0191, P6, design section 9): when the
 * COEFF_MASK field of flags is ESH_TAYLOR_COEFF_RATIONAL, the coefficient
 * storage is reinterpreted as an array of `eshkol_tagged_value_t` (int64 /
 * bignum / rational, produced through Eshkol's existing exact numeric
 * tower -- lib/core/rational.c, lib/core/bignum.cpp) instead of raw
 * doubles. R7RS exactness contagion governs everything: seeding from an
 * exact point yields an exact tower; add/sub/mul/div and non-negative-
 * integer pow stay exact; the moment an operand is inexact, an exact op
 * overflows the int64-limited rational substrate, or a transcendental
 * primitive (exp/log/sin/cos/tan/sqrt/sinh/cosh/tanh) is applied, the
 * WHOLE result tower is rebuilt as COEFF_F64 -- coefficients are never
 * mixed-tagged within one tower (design section 4/12).
 */

/* arena_memory.h declares thread-local storage with the C++/C23 spelling
 * `thread_local`; provide it for the C11 build of this translation unit. */
#if !defined(__cplusplus) && !defined(thread_local)
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#    define thread_local _Thread_local
#  else
#    define thread_local
#  endif
#endif

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/core/rational.h"
#include "../../inc/eshkol/core/bignum.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#include <atomic>
extern "C" {
#endif

/* Op-code constants derived from the shared X-macro table
 * (lib/core/taylor_recurrences.def) so the runtime kernel and the P2 IR
 * emitter (lib/backend/autodiff_codegen.cpp) can never drift on which
 * primitive maps to which op-code -- design section 5b. The integer values
 * remain the ABI mirrored by the codegen dispatch in arithmetic_codegen. */
enum {
#define TAYLOR_BIN(name, opcode, sexpr) ESH_TAYLOR_OP_##name = (opcode),
#include "taylor_recurrences.def"
};
enum {
#define TAYLOR_UN(name, opcode, sexpr, testfn, x0) ESH_TAYLOR_UOP_##name = (opcode),
#include "taylor_recurrences.def"
};

/* ----------------------------------------------------------------------- */
/* allocation                                                              */
/* ----------------------------------------------------------------------- */

/* Allocate a zeroed tower of order K (K+1 coefficients) with the given flags.
 * Returns the DATA pointer (after the 8-byte object header), so it can be
 * stored in a tagged HEAP_PTR exactly like a tensor. When `flags` has
 * ESH_TAYLOR_TANGENT_FLAG set the storage is doubled to 2*(K+1) doubles so a
 * parallel first-order seed-tangent series (P5) rides alongside the value
 * series; both halves are zero-initialised. */
esh_taylor_t* eshkol_taylor_alloc(arena_t* arena, uint32_t order_k, uint32_t flags) {
    if (!arena) arena = get_global_arena();
    if (!arena) return NULL;

    size_t ncoeff = (size_t)order_k + 1;
    size_t nstore = ESH_TAYLOR_HAS_TANGENT(flags) ? (2u * ncoeff) : ncoeff;
    size_t data_size = sizeof(esh_taylor_t) + nstore * sizeof(double);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 15) & ~((size_t)15);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate Taylor tower (order %u)", order_k);
        return NULL;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_TAYLOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    esh_taylor_t* t = (esh_taylor_t*)(mem + sizeof(eshkol_object_header_t));
    t->order_k = order_k;
    t->flags = flags;
    memset(t->c, 0, nstore * sizeof(double));
    return t;
}

/* Pointer to a tower's tangent (seed-derivative) coefficient array, or NULL if
 * the tower does not carry one. The tangent half follows the value half. */
static inline double* taylor_tan(esh_taylor_t* t) {
    if (!t || !ESH_TAYLOR_HAS_TANGENT(t->flags)) return NULL;
    return t->c + ((size_t)t->order_k + 1);
}

/* ----------------------------------------------------------------------- */
/* EXACT-COEFFICIENT tower allocation & scalar helpers (P6, ESH-0191)       */
/* ----------------------------------------------------------------------- */

/* Allocate a zeroed (all exact-0) EXACT tower: coefficient storage is
 * `eshkol_tagged_value_t c[order_k+1]` reinterpreted over the `double c[]`
 * flexible array member -- see the COEFF_RATIONAL comment in eshkol.h. */
esh_taylor_t* eshkol_taylor_alloc_exact(arena_t* arena, uint32_t order_k, uint32_t epoch) {
    if (!arena) arena = get_global_arena();
    if (!arena) return NULL;

    size_t ncoeff = (size_t)order_k + 1;
    size_t data_size = sizeof(esh_taylor_t) + ncoeff * sizeof(eshkol_tagged_value_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 15) & ~((size_t)15);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate exact Taylor tower (order %u)", order_k);
        return NULL;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_TAYLOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    esh_taylor_t* t = (esh_taylor_t*)(mem + sizeof(eshkol_object_header_t));
    t->order_k = order_k;
    t->flags = ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_RATIONAL, epoch);

    eshkol_tagged_value_t* c = (eshkol_tagged_value_t*)(void*)t->c;
    eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
    for (size_t i = 0; i < ncoeff; i++) c[i] = zero;
    return t;
}

/** @brief True iff tower `t` stores EXACT-COEFFICIENT (rational/bignum/int64) coefficients rather than raw doubles. */
static inline int taylor_is_exact(const esh_taylor_t* t) {
    return t != NULL && (t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_RATIONAL;
}

/* Reinterpret an EXACT tower's coefficient storage as tagged values. Only
 * valid when taylor_is_exact(t) -- callers must check first. */
static inline eshkol_tagged_value_t* taylor_exact_c(esh_taylor_t* t) {
    return (eshkol_tagged_value_t*)(void*)t->c;
}
/** @brief Const-qualified counterpart of taylor_exact_c(): reinterpret an EXACT tower's coefficient storage as tagged values. */
static inline const eshkol_tagged_value_t* taylor_exact_c_const(const esh_taylor_t* t) {
    return (const eshkol_tagged_value_t*)(const void*)t->c;
}

/* R7RS exact?: int64, bignum, or rational -- never double. Mirrors the
 * "exact?" builtin's own base-type/subtype check (llvm_codegen.cpp). */
static int tagged_is_exact_number(const eshkol_tagged_value_t* v) {
    uint8_t bt = (uint8_t)(v->type & 0x0F);
    if (bt == ESHKOL_VALUE_INT64) return 1;
    if (bt == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val != 0) {
        const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
        return hdr != NULL && (hdr->subtype == HEAP_SUBTYPE_BIGNUM || hdr->subtype == HEAP_SUBTYPE_RATIONAL);
    }
    return 0;
}
/** @brief True iff tagged value `v` is a heap pointer to a bignum object. */
static inline int tagged_is_bignum(const eshkol_tagged_value_t* v) {
    uint8_t bt = (uint8_t)(v->type & 0x0F);
    if (bt != ESHKOL_VALUE_HEAP_PTR || v->data.ptr_val == 0) return 0;
    const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
    return hdr != NULL && hdr->subtype == HEAP_SUBTYPE_BIGNUM;
}
/** @brief True iff tagged value `v` is a heap pointer to a rational object. */
static inline int tagged_is_rational(const eshkol_tagged_value_t* v) {
    uint8_t bt = (uint8_t)(v->type & 0x0F);
    if (bt != ESHKOL_VALUE_HEAP_PTR || v->data.ptr_val == 0) return 0;
    const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
    return hdr != NULL && hdr->subtype == HEAP_SUBTYPE_RATIONAL;
}
/**
 * @brief True iff exact tagged value `v` (int64, bignum, or rational) is negative.
 *
 * Used to fix the sign of the base point when computing the exact-tower
 * `abs` recurrence.
 */
static inline int tagged_is_negative_exact(const eshkol_tagged_value_t* v) {
    if (tagged_is_bignum(v)) return eshkol_bignum_is_negative((eshkol_bignum_t*)(uintptr_t)v->data.ptr_val);
    if (tagged_is_rational(v)) return eshkol_rational_numerator((void*)(uintptr_t)v->data.ptr_val) < 0;
    return v->data.int_val < 0; /* int64 */
}

/* Convert ANY numeric tagged value (double, int64, bignum, rational) to a
 * plain double -- used when demoting an exact tower to COEFF_F64. */
static double tagged_any_to_double(const eshkol_tagged_value_t* v) {
    uint8_t bt = (uint8_t)(v->type & 0x0F);
    if (bt == ESHKOL_VALUE_DOUBLE) return v->data.double_val;
    if (bt == ESHKOL_VALUE_INT64)  return (double)v->data.int_val;
    if (tagged_is_rational(v)) return eshkol_rational_to_double((void*)(uintptr_t)v->data.ptr_val);
    if (tagged_is_bignum(v))   return eshkol_bignum_to_double((eshkol_bignum_t*)(uintptr_t)v->data.ptr_val);
    return 0.0;
}

/* Exact scalar binary dispatch (op: 0=add,1=sub,2=mul,3=div). Routes pure
 * integer add/sub/mul through the bignum substrate (arbitrary precision --
 * the "bignum-scale" gate), and division / any rational operand through the
 * int64-limited rational substrate; gracefully degrades to double the
 * moment an operand is already inexact, or a genuine (int64-overflowing)
 * bignum meets the rational substrate (which cannot represent it) -- this
 * IS the exactness-contagion rule (design section 9), applied per-scalar so
 * a single coefficient's overflow doesn't require special-casing by the
 * caller: the tower-level contagion check (taylor_materialize_exact_or_demote)
 * notices the resulting inexact entry and demotes the whole tower. */
static eshkol_tagged_value_t exact_binary(arena_t* arena, eshkol_tagged_value_t a,
                                          eshkol_tagged_value_t b, int op) {
    uint8_t abt = (uint8_t)(a.type & 0x0F), bbt = (uint8_t)(b.type & 0x0F);
    if (abt == ESHKOL_VALUE_DOUBLE || bbt == ESHKOL_VALUE_DOUBLE) {
        double da = tagged_any_to_double(&a), db = tagged_any_to_double(&b);
        double r;
        switch (op) { case 0: r = da + db; break; case 1: r = da - db; break;
                      case 2: r = da * db; break; default: r = (db != 0.0) ? da / db : 0.0; break; }
        return eshkol_make_double(r);
    }

    int a_rat = tagged_is_rational(&a), b_rat = tagged_is_rational(&b);
    int a_big = tagged_is_bignum(&a),   b_big = tagged_is_bignum(&b);

    if (op != 3 && !a_rat && !b_rat) {
        /* pure-integer add/sub/mul: arbitrary precision via the bignum
         * substrate (handles int64<->bignum promotion/demotion itself). */
        eshkol_tagged_value_t r;
        eshkol_bignum_binary_tagged(arena, &a, &b, op, &r);
        return r;
    }
    if (a_big || b_big) {
        /* A genuine (int64-overflowing) bignum meeting a rational, or a
         * bignum division -- outside the int64-limited rational substrate.
         * Documented graceful degradation to double, exactly like the
         * transcendental-op promotion (design section 9). */
        double da = tagged_any_to_double(&a), db = tagged_any_to_double(&b);
        double r;
        switch (op) { case 0: r = da + db; break; case 1: r = da - db; break;
                      case 2: r = da * db; break; default: r = (db != 0.0) ? da / db : 0.0; break; }
        return eshkol_make_double(r);
    }
    eshkol_tagged_value_t r;
    eshkol_rational_binary_tagged_ptr((void*)arena, &a, &b, op, &r);
    return r;
}
/** @brief Exact addition: dispatches to exact_binary() with op=0 (add). */
static inline eshkol_tagged_value_t exact_add(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return exact_binary(ar, a, b, 0); }
/** @brief Exact subtraction: dispatches to exact_binary() with op=1 (sub). */
static inline eshkol_tagged_value_t exact_sub(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return exact_binary(ar, a, b, 1); }
/** @brief Exact multiplication: dispatches to exact_binary() with op=2 (mul). */
static inline eshkol_tagged_value_t exact_mul(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return exact_binary(ar, a, b, 2); }
/** @brief Exact division: dispatches to exact_binary() with op=3 (div). */
static inline eshkol_tagged_value_t exact_div(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return exact_binary(ar, a, b, 3); }

/** @brief Allocate an uninitialised array of `n` tagged-value coefficients from `arena`, used as EXACT-tower scratch/result storage. */
static eshkol_tagged_value_t* alloc_exact_series(arena_t* arena, int n) {
    return (eshkol_tagged_value_t*)arena_allocate(arena, (size_t)n * sizeof(eshkol_tagged_value_t));
}

/* If every entry of `s` (length n) is still an exact number, materialize an
 * EXACT tower; otherwise (an intermediate op overflowed the int64-limited
 * rational substrate) rebuild the SAME n entries as a COEFF_F64 tower --
 * a tower's coefficients are never mixed-tagged (design section 4). */
static void taylor_materialize_exact_or_demote(arena_t* arena, const eshkol_tagged_value_t* s, int n,
                                               uint32_t order_k, uint32_t epoch, eshkol_tagged_value_t* result) {
    for (int k = 0; k < n; k++) {
        if (!tagged_is_exact_number(&s[k])) {
            esh_taylor_t* f = eshkol_taylor_alloc(arena, order_k, ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch));
            if (!f) { eshkol_tagged_value_t z; memset(&z, 0, sizeof(z)); *result = z; return; }
            for (int j = 0; j < n; j++) f->c[j] = tagged_any_to_double(&s[j]);
            eshkol_tagged_value_t v; memset(&v, 0, sizeof(v));
            v.type = ESHKOL_VALUE_HEAP_PTR; v.flags = ESHKOL_VALUE_INEXACT_FLAG;
            v.data.ptr_val = (uint64_t)(uintptr_t)f;
            *result = v;
            return;
        }
    }
    esh_taylor_t* out = eshkol_taylor_alloc_exact(arena, order_k, epoch);
    if (!out) { *result = eshkol_make_double(0.0); return; }
    eshkol_tagged_value_t* c = taylor_exact_c(out);
    for (int k = 0; k < n; k++) c[k] = s[k];
    eshkol_tagged_value_t v; memset(&v, 0, sizeof(v));
    v.type = ESHKOL_VALUE_HEAP_PTR; v.flags = ESHKOL_VALUE_INEXACT_FLAG;
    v.data.ptr_val = (uint64_t)(uintptr_t)out;
    *result = v;
}

/**
 * @brief Wrap a Taylor tower pointer `t` in an inexact heap-pointer tagged value.
 *
 * Used to return a freshly allocated (or existing) tower from the
 * binary/unary/seed/shift helpers as an eshkol_tagged_value_t.
 */
static inline eshkol_tagged_value_t taylor_to_tagged(const esh_taylor_t* t) {
    eshkol_tagged_value_t v;
    memset(&v, 0, sizeof(v));
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.flags = ESHKOL_VALUE_INEXACT_FLAG;
    v.data.ptr_val = (uint64_t)(uintptr_t)t;
    return v;
}

/* If tv is a Taylor tower return its data pointer, else NULL. */
static inline esh_taylor_t* tagged_as_taylor(const eshkol_tagged_value_t* tv) {
    if (!tv) return NULL;
    if ((tv->type & 0x0F) != ESHKOL_VALUE_HEAP_PTR) return NULL;
    if (tv->data.ptr_val == 0) return NULL;
    void* ptr = (void*)(uintptr_t)tv->data.ptr_val;
    const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
    if (!hdr || hdr->subtype != HEAP_SUBTYPE_TAYLOR) return NULL;
    return (esh_taylor_t*)ptr;
}

/**
 * @brief Predicate: does tagged value `tv` reference a Taylor tower object?
 * @param tv  Tagged value to inspect.
 * @return    Non-zero if `tv` is a heap pointer to a HEAP_SUBTYPE_TAYLOR object.
 */
int eshkol_is_taylor_tagged(const eshkol_tagged_value_t* tv) {
    return tagged_as_taylor(tv) != NULL;
}

/* Extract a plain scalar (the primal value c[0]) from any operand: a tower's
 * c[0] (exact-aware -- P6), a double, an int, a bignum, or a rational. Used
 * to lift constants and foreign-epoch towers. */
static inline double tagged_scalar_value(const eshkol_tagged_value_t* tv) {
    if (!tv) return 0.0;
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t) {
        if (taylor_is_exact(t)) return tagged_any_to_double(&taylor_exact_c_const(t)[0]);
        return t->c[0];
    }
    return tagged_any_to_double(tv);
}

/* c[0] of a tower value (or the scalar value of a non-tower) — the coercion
 * used when a tower flows into a plain-double numeric context. */
double eshkol_taylor_c0(const eshkol_tagged_value_t* tv) {
    return tagged_scalar_value(tv);
}

/* ----------------------------------------------------------------------- */
/* recurrences (operate on raw coefficient arrays, n = K+1 entries)         */
/* ----------------------------------------------------------------------- */

/** @brief s = u + w, elementwise over n Taylor coefficients. */
static void tr_add(double* s, const double* u, const double* w, int n) {
    for (int k = 0; k < n; k++) s[k] = u[k] + w[k];
}
/** @brief s = u - w, elementwise over n Taylor coefficients. */
static void tr_sub(double* s, const double* u, const double* w, int n) {
    for (int k = 0; k < n; k++) s[k] = u[k] - w[k];
}
/** @brief s = -u, elementwise negation of n Taylor coefficients. */
static void tr_neg(double* s, const double* u, int n) {
    for (int k = 0; k < n; k++) s[k] = -u[k];
}

/* s = u * w : s_k = sum_{j=0..k} u_j * w_{k-j}   (Cauchy convolution, fma). */
static void tr_mul(double* s, const double* u, const double* w, int n) {
    for (int k = 0; k < n; k++) {
        double acc = 0.0;
        for (int j = 0; j <= k; j++) acc = fma(u[j], w[k - j], acc);
        s[k] = acc;
    }
}

/* s = u / w : s_k = ( u_k - sum_{j=1..k} w_j * s_{k-j} ) / w_0. */
static void tr_div(double* s, const double* u, const double* w, int n) {
    for (int k = 0; k < n; k++) {
        double acc = u[k];
        for (int j = 1; j <= k; j++) acc = fma(-w[j], s[k - j], acc);
        s[k] = acc / w[0];
    }
}

/* s = exp(u) : s_0 = exp(u_0); s_k = (1/k) sum_{j=1..k} j*u_j*s_{k-j}. */
static void tr_exp(double* s, const double* u, int n) {
    s[0] = exp(u[0]);
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++) acc = fma((double)j * u[j], s[k - j], acc);
        s[k] = acc / (double)k;
    }
}

/* s = log(u) : s_0 = log(u_0);
 * s_k = ( u_k - (1/k) sum_{j=1..k-1} j*s_j*u_{k-j} ) / u_0. */
static void tr_log(double* s, const double* u, int n) {
    s[0] = log(u[0]);
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k - 1; j++) acc = fma((double)j * s[j], u[k - j], acc);
        s[k] = (u[k] - acc / (double)k) / u[0];
    }
}

/* coupled: so = sin(u), co = cos(u).
 * so_k =  (1/k) sum_{j=1..k} j*u_j*co_{k-j}
 * co_k = -(1/k) sum_{j=1..k} j*u_j*so_{k-j}. */
static void tr_sincos(double* so, double* co, const double* u, int n) {
    so[0] = sin(u[0]);
    co[0] = cos(u[0]);
    for (int k = 1; k < n; k++) {
        double as = 0.0, ac = 0.0;
        for (int j = 1; j <= k; j++) {
            double ju = (double)j * u[j];
            as = fma(ju, co[k - j], as);
            ac = fma(ju, so[k - j], ac);
        }
        so[k] =  as / (double)k;
        co[k] = -ac / (double)k;
    }
}

/* s = u^r (constant real exponent r):
 * s_0 = u_0^r; s_k = (1/(k*u_0)) sum_{j=1..k} (j*r - (k-j))*u_j*s_{k-j}. */
static void tr_pow_const(double* s, const double* u, double r, int n) {
    s[0] = pow(u[0], r);
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++)
            acc = fma(((double)j * r - (double)(k - j)) * u[j], s[k - j], acc);
        s[k] = acc / ((double)k * u[0]);
    }
}

/* ----------------------------------------------------------------------- */
/* dual recurrences: value + first-order seed tangent (P5, ESH-0190)        */
/* ----------------------------------------------------------------------- */
/* A "dual tower" carries, alongside its value series u, a tangent series
 * u' = d(u)/d(reverse-seed). Reverse-over-Taylor (docs/design §8) needs one
 * first-order sensitivity of every high-order coefficient to the outer
 * gradient's seed; this is exactly the tower analogue of the 8-jet's
 * ep-derivative half. Each rule below computes the value with the existing
 * tr_* recurrence and the tangent with the linearised (product/chain) rule,
 * using the same fma / ascending-j reduction order (design §6a). */

/* Small stack buffer avoids a heap alloc for the common orders; falls back
 * to the arena for very high K. */
#define ESH_TAYLOR_STACKN 64

/* s = convolution(a, b): s_k = sum_{j=0..k} a_j * b_{k-j}. */
static void tr_conv(double* s, const double* a, const double* b, int n) {
    for (int k = 0; k < n; k++) {
        double acc = 0.0;
        for (int j = 0; j <= k; j++) acc = fma(a[j], b[k - j], acc);
        s[k] = acc;
    }
}

/* ----------------------------------------------------------------------- */
/* EXACT-COEFFICIENT recurrences (P6, ESH-0191): same algebra as the tr_*   */
/* kernels above, but every multiply-accumulate dispatches through         */
/* exact_{add,sub,mul,div} (the exact numeric tower) instead of fma().     */
/* No FP-contraction-policy concern here (unlike section 6a): exact        */
/* arithmetic has no rounding, so reduction order does not affect the      */
/* result -- only the double fallback triggered by overflow matters, and   */
/* that is identical to ordinary add/sub/mul/div contagion.                */
/* ----------------------------------------------------------------------- */

/** @brief Exact-coefficient counterpart of tr_add(): s = u + w via exact_add() over n coefficients. */
static void tre_add(arena_t* ar, eshkol_tagged_value_t* s,
                    const eshkol_tagged_value_t* u, const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) s[k] = exact_add(ar, u[k], w[k]);
}
/** @brief Exact-coefficient counterpart of tr_sub(): s = u - w via exact_sub() over n coefficients. */
static void tre_sub(arena_t* ar, eshkol_tagged_value_t* s,
                    const eshkol_tagged_value_t* u, const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) s[k] = exact_sub(ar, u[k], w[k]);
}

/* s = u * w : s_k = sum_{j=0..k} u_j * w_{k-j}   (Cauchy convolution). */
static void tre_mul(arena_t* ar, eshkol_tagged_value_t* s,
                    const eshkol_tagged_value_t* u, const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) {
        eshkol_tagged_value_t acc = eshkol_make_int64(0, true);
        for (int j = 0; j <= k; j++) acc = exact_add(ar, acc, exact_mul(ar, u[j], w[k - j]));
        s[k] = acc;
    }
}

/* Tangent of u*w: (u*w)' = u'*w + u*w'. */
static void trd_mul(double* st, const double* uv, const double* ut,
                    const double* wv, const double* wt, int n) {
    for (int k = 0; k < n; k++) {
        double acc = 0.0;
        for (int j = 0; j <= k; j++) {
            acc = fma(ut[j], wv[k - j], acc);
            acc = fma(uv[j], wt[k - j], acc);
        }
        st[k] = acc;
    }
}

/* Tangent of s = u/w, given the already-computed value quotient sv:
 *   s'_k = ( u'_k - sum_{j=1..k}(w'_j s_{k-j} + w_j s'_{k-j}) - s_k w'_0 ) / w_0. */
static void trd_div(double* st, const double* sv,
                    const double* ut, const double* wv, const double* wt, int n) {
    for (int k = 0; k < n; k++) {
        double acc = ut[k];
        for (int j = 1; j <= k; j++) {
            acc = fma(-wt[j], sv[k - j], acc);
            acc = fma(-wv[j], st[k - j], acc);
        }
        acc = fma(-sv[k], wt[0], acc);
        st[k] = acc / wv[0];
    }
}

/* Composable dual ops: each fills value sv and tangent st from operand
 * value/tangent series. Value uses the proven tr_* recurrence; tangent uses
 * the chain rule s' = g'(u)·u' realised as a series convolution/division. */
static void ddual_mul(double* sv, double* st, const double* uv, const double* ut,
                      const double* wv, const double* wt, int n) {
    tr_mul(sv, uv, wv, n);
    trd_mul(st, uv, ut, wv, wt, n);
}
/** @brief Dual (value + tangent) division: sv = uv/wv via tr_div(), then its seed-tangent st via trd_div(). */
static void ddual_div(double* sv, double* st, const double* uv, const double* ut,
                      const double* wv, const double* wt, int n) {
    tr_div(sv, uv, wv, n);
    trd_div(st, sv, ut, wv, wt, n);
}
/** @brief Dual (value + tangent) exp: sv = exp(uv) via tr_exp(); tangent st = (exp u)' = exp(u)*u' via convolution. */
static void ddual_exp(double* sv, double* st, const double* uv, const double* ut, int n) {
    tr_exp(sv, uv, n);
    tr_conv(st, sv, ut, n);                 /* (exp u)' = exp(u)·u' */
}
/** @brief Dual (value + tangent) log: sv = log(uv) via tr_log(); tangent st = (log u)' = u'/u via tr_div(). */
static void ddual_log(double* sv, double* st, const double* uv, const double* ut, int n) {
    tr_log(sv, uv, n);
    tr_div(st, ut, uv, n);                  /* (log u)' = u'/u */
}
/* sin/cos share the coupled recurrence; fills both values + both tangents. */
static void ddual_sincos(double* so, double* sot, double* co, double* cot,
                         const double* uv, const double* ut, int n) {
    tr_sincos(so, co, uv, n);
    tr_conv(sot, co, ut, n);                /* (sin u)' =  cos(u)·u' */
    tr_conv(cot, so, ut, n);
    for (int k = 0; k < n; k++) cot[k] = -cot[k];   /* (cos u)' = -sin(u)·u' */
}
/* u^r, constant real exponent r. (u^r)' = r·u^{r-1}·u' = r·(u^r/u)·u'. */
static void ddual_pow_const(double* sv, double* st, const double* uv, const double* ut,
                            double r, int n, arena_t* arena) {
    tr_pow_const(sv, uv, r, n);
    double qb[ESH_TAYLOR_STACKN];
    double* q = qb; double* hq = NULL;
    if (n > ESH_TAYLOR_STACKN) { hq = (double*)arena_allocate(arena, (size_t)n*sizeof(double)); q = hq; }
    tr_div(q, sv, uv, n);                   /* q = u^r / u = u^{r-1} */
    tr_conv(st, q, ut, n);
    for (int k = 0; k < n; k++) st[k] = r * st[k];
}

/* s = u / w : s_k = ( u_k - sum_{j=1..k} w_j * s_{k-j} ) / w_0. */
static void tre_div(arena_t* ar, eshkol_tagged_value_t* s,
                    const eshkol_tagged_value_t* u, const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) {
        eshkol_tagged_value_t acc = u[k];
        for (int j = 1; j <= k; j++) acc = exact_sub(ar, acc, exact_mul(ar, w[j], s[k - j]));
        s[k] = exact_div(ar, acc, w[0]);
    }
}

/* ----------------------------------------------------------------------- */
/* operand normalisation + epoch (perturbation-confusion) handling          */
/* ----------------------------------------------------------------------- */

/* Materialise operand `tv` as a length-`n` coefficient array into `buf`,
 * relative to the current active epoch `active_epoch`:
 *   - a same-epoch tower  -> its coefficients (zero-extended to n)
 *   - a foreign tower / scalar / int -> a constant series {value, 0, ...}
 * This is the section-5a lift: order->=1 coefficients of a foreign-epoch tower
 * do not participate in the current level's differentiation.
 *
 * P6: a same-epoch EXACT tower's coefficients are converted to double here
 * (never raw-memcpy'd -- they are tagged_value_t, not double, in storage) so
 * this F64 path stays correct whenever exactness has already been decided
 * NOT to propagate for this particular result (see operand_is_exact_for_taylor
 * and eshkol_taylor_binary_tagged/eshkol_taylor_unary_tagged below). */
static void normalise_operand(const eshkol_tagged_value_t* tv, uint32_t active_epoch,
                              double* buf, int n) {
    memset(buf, 0, (size_t)n * sizeof(double));
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t && ESH_TAYLOR_GET_EPOCH(t->flags) == active_epoch) {
        int m = (int)t->order_k + 1;
        if (m > n) m = n;
        if (taylor_is_exact(t)) {
            const eshkol_tagged_value_t* c = taylor_exact_c_const(t);
            for (int i = 0; i < m; i++) buf[i] = tagged_any_to_double(&c[i]);
        } else {
            memcpy(buf, t->c, (size_t)m * sizeof(double));
        }
    } else {
        buf[0] = tagged_scalar_value(tv);
    }
}

/* Exact counterpart of normalise_operand: same shape, but coefficients stay
 * tagged (exact) values. Callers must already know every operand that will
 * flow here is exact at this epoch (operand_is_exact_for_taylor). */
static void normalise_operand_exact(const eshkol_tagged_value_t* tv, uint32_t active_epoch,
                                    eshkol_tagged_value_t* buf, int n) {
    eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
    for (int i = 0; i < n; i++) buf[i] = zero;
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t && ESH_TAYLOR_GET_EPOCH(t->flags) == active_epoch) {
        int m = (int)t->order_k + 1;
        if (m > n) m = n;
        const eshkol_tagged_value_t* c = taylor_exact_c_const(t);
        for (int i = 0; i < m; i++) buf[i] = c[i];
    } else if (t) {
        /* foreign-epoch tower: lift its (already-verified-exact) c[0] */
        buf[0] = taylor_is_exact(t) ? taylor_exact_c_const(t)[0] : zero;
    } else {
        buf[0] = *tv;
    }
}

/* True iff `tv` participates as an EXACT operand at `active_epoch`: a
 * same-epoch exact tower, a plain exact scalar (int64/bignum/rational), or a
 * foreign-epoch tower whose lifted c[0] constant is itself exact. */
static int operand_is_exact_for_taylor(const eshkol_tagged_value_t* tv, uint32_t active_epoch) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) return tagged_is_exact_number(tv);
    if (ESH_TAYLOR_GET_EPOCH(t->flags) != active_epoch) {
        if (!taylor_is_exact(t)) return 0; /* F64 tower's c[0] is a raw double */
        return tagged_is_exact_number(&taylor_exact_c_const(t)[0]);
    }
    return taylor_is_exact(t);
}

/* True iff `right`, at `active_epoch`, reduces to a plain constant
 * non-negative integer (a scalar, or a foreign-epoch tower's c[0]) -- the
 * only exponent shape the exact integer-power recurrence supports. A
 * same-epoch tower (the exponent itself depends on the differentiation
 * variable) is intentionally left unhandled here: pow falls through to the
 * existing (inexact) general recurrence for that rare shape. */
static int exact_pow_exponent_as_nonneg_int(const eshkol_tagged_value_t* right,
                                            uint32_t active_epoch, int64_t* out) {
    esh_taylor_t* t = tagged_as_taylor(right);
    eshkol_tagged_value_t c0;
    if (t) {
        if (ESH_TAYLOR_GET_EPOCH(t->flags) == active_epoch) return 0;
        if (!taylor_is_exact(t)) return 0;
        c0 = taylor_exact_c_const(t)[0];
    } else {
        c0 = *right;
    }
    uint8_t bt = (uint8_t)(c0.type & 0x0F);
    if (bt == ESHKOL_VALUE_INT64) {
        if (c0.data.int_val < 0) return 0;
        *out = c0.data.int_val;
        return 1;
    }
    if (tagged_is_bignum(&c0)) {
        eshkol_bignum_t* bn = (eshkol_bignum_t*)(uintptr_t)c0.data.ptr_val;
        int64_t v;
        if (eshkol_bignum_is_negative(bn) || !eshkol_bignum_fits_int64(bn, &v)) return 0;
        *out = v;
        return 1;
    }
    return 0; /* rational (non-integer) exponent, or inexact -- not integer-power */
}

/* s = u^p for a compile-time-unknown but RUNTIME-constant non-negative
 * integer p, via exact binary exponentiation (repeated series
 * multiplication) -- keeps monomial/polynomial derivatives exact without
 * the general real-exponent recurrence's division (which would force the
 * F64 tier for every pow, even integer ones). */
static void taylor_pow_exact(arena_t* arena, const eshkol_tagged_value_t* left,
                             uint32_t order_k, uint32_t epoch, uint64_t p,
                             eshkol_tagged_value_t* result) {
    int n = (int)order_k + 1;
    eshkol_tagged_value_t* u    = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* acc  = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* base = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* tmp  = alloc_exact_series(arena, n);
    if (!u || !acc || !base || !tmp) { *result = eshkol_make_double(0.0); return; }
    normalise_operand_exact(left, epoch, u, n);

    eshkol_tagged_value_t one = eshkol_make_int64(1, true);
    eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
    acc[0] = one;
    for (int k = 1; k < n; k++) acc[k] = zero;
    for (int k = 0; k < n; k++) base[k] = u[k];

    uint64_t e = p;
    while (e > 0) {
        if (e & 1u) {
            tre_mul(arena, tmp, acc, base, n);
            for (int k = 0; k < n; k++) acc[k] = tmp[k];
        }
        e >>= 1;
        if (e > 0) {
            tre_mul(arena, tmp, base, base, n);
            for (int k = 0; k < n; k++) base[k] = tmp[k];
        }
    }
    taylor_materialize_exact_or_demote(arena, acc, n, order_k, epoch, result);
}

/* add/sub/mul/div dispatch on two EXACT operands (pow is handled by
 * taylor_pow_exact above, gated separately since it needs the exponent's
 * constant-integer shape, not just its exactness). */
static void taylor_binary_exact(arena_t* arena, const eshkol_tagged_value_t* left,
                                const eshkol_tagged_value_t* right, int op,
                                uint32_t order_k, uint32_t epoch, eshkol_tagged_value_t* result) {
    int n = (int)order_k + 1;
    eshkol_tagged_value_t* u = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* w = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* s = alloc_exact_series(arena, n);
    if (!u || !w || !s) { *result = eshkol_make_double(0.0); return; }
    normalise_operand_exact(left, epoch, u, n);
    normalise_operand_exact(right, epoch, w, n);

    switch (op) {
        case ESH_TAYLOR_OP_add: tre_add(arena, s, u, w, n); break;
        case ESH_TAYLOR_OP_sub: tre_sub(arena, s, u, w, n); break;
        case ESH_TAYLOR_OP_mul: tre_mul(arena, s, u, w, n); break;
        case ESH_TAYLOR_OP_div: tre_div(arena, s, u, w, n); break;
        default: tre_add(arena, s, u, w, n); break;
    }
    taylor_materialize_exact_or_demote(arena, s, n, order_k, epoch, result);
}

/* The order and active epoch a result should carry: the max order and max
 * epoch across the tower operands (scalars contribute nothing). */
static void result_shape(const eshkol_tagged_value_t* l, const eshkol_tagged_value_t* r,
                         uint32_t* order_k, uint32_t* epoch) {
    esh_taylor_t* lt = tagged_as_taylor(l);
    esh_taylor_t* rt = tagged_as_taylor(r);
    uint32_t k = 0, e = 0;
    if (lt) { if (lt->order_k > k) k = lt->order_k; }
    if (rt) { if (rt->order_k > k) k = rt->order_k; }
    if (lt) { uint32_t le = ESH_TAYLOR_GET_EPOCH(lt->flags); if (le > e) e = le; }
    if (rt) { uint32_t re = ESH_TAYLOR_GET_EPOCH(rt->flags); if (re > e) e = re; }
    *order_k = k;
    *epoch = e;
}

/* ----------------------------------------------------------------------- */
/* P5 seed-tangent operand extraction (ESH-0190)                            */
/* ----------------------------------------------------------------------- */
/* Reverse-mode hook (defined in runtime_autodiff.cpp): 1.0 iff `node` is the
 * gradient pass's active seed variable. Read here so an AD-node operand flowing
 * into tower arithmetic contributes d(value)/d(seed) = seed_flag into c[0] of
 * its tangent series. */
extern double eshkol_ad_seed_flag(void* node);

/* Does this operand carry (or induce) a first-order seed tangent?
 *   - a tower with ESH_TAYLOR_TANGENT_FLAG            -> yes
 *   - a forward-mode DUAL number (outer gradient seed) -> yes (its e1 tangent)
 *   - a reverse-tape CALLABLE AD node                  -> yes (its seed_flag)
 * Plain scalars / towers-without-tangent do not. */
static int operand_has_tangent(const eshkol_tagged_value_t* tv) {
    if (!tv) return 0;
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t) return ESH_TAYLOR_HAS_TANGENT(t->flags);
    uint8_t bt = (uint8_t)(tv->type & 0x0F);
    if (bt == ESHKOL_VALUE_DUAL_NUMBER) return 1;
    if (bt == ESHKOL_VALUE_CALLABLE) return 1;  /* AD node (subtype not re-checked) */
    return 0;
}

/* Materialise BOTH the value series (epoch-gated, exactly like normalise_operand)
 * and the tangent series (a single global first-order seed dimension, epoch-
 * independent) of an operand into vbuf/tbuf (length n). */
static void normalise_operand_dual(const eshkol_tagged_value_t* tv, uint32_t active_epoch,
                                   double* vbuf, double* tbuf, int n) {
    memset(vbuf, 0, (size_t)n * sizeof(double));
    memset(tbuf, 0, (size_t)n * sizeof(double));
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t) {
        /* value: same-epoch tower contributes its full series; foreign-epoch
         * (outer/inner level) is lifted to its constant c[0] (§5a). */
        if (ESH_TAYLOR_GET_EPOCH(t->flags) == active_epoch) {
            int m = (int)t->order_k + 1;
            if (m > n) m = n;
            memcpy(vbuf, t->c, (size_t)m * sizeof(double));
        } else {
            vbuf[0] = t->c[0];
        }
        /* tangent: the seed dimension is orthogonal to the value epoch, so it
         * always combines. */
        double* tt = taylor_tan(t);
        if (tt) {
            int m = (int)t->order_k + 1;
            if (m > n) m = n;
            memcpy(tbuf, tt, (size_t)m * sizeof(double));
        }
        return;
    }
    uint8_t bt = (uint8_t)(tv->type & 0x0F);
    if (bt == ESHKOL_VALUE_DUAL_NUMBER && tv->data.ptr_val) {
        /* forward-mode jet {primal, e1, ...}: primal is the value, e1 is the
         * outer gradient's first-order perturbation = the seed tangent. */
        const double* d = (const double*)(uintptr_t)tv->data.ptr_val;
        vbuf[0] = d[0];
        tbuf[0] = d[1];
        return;
    }
    if (bt == ESHKOL_VALUE_CALLABLE && tv->data.ptr_val) {
        /* reverse-tape AD node: value is node->value, tangent c[0] is 1.0 iff
         * this IS the active seed (frozen local linearisation, §8). */
        void* node = (void*)(uintptr_t)tv->data.ptr_val;
        vbuf[0] = ((const ad_node_t*)node)->value;
        tbuf[0] = eshkol_ad_seed_flag(node);
        return;
    }
    vbuf[0] = tagged_scalar_value(tv);
}

/* ----------------------------------------------------------------------- */
/* tagged binary / unary dispatch (called from codegen)                     */
/* ----------------------------------------------------------------------- */

/**
 * @brief Apply a binary op (add/sub/mul/div/pow) to two Taylor-tower and/or
 *        scalar operands; entry point called from codegen's numeric dispatch.
 *
 * Determines the result's order and active epoch from whichever operand(s)
 * are towers (result_shape()), then routes to one of three tiers, in
 * priority order:
 *   1. Dual (value + first-order seed-tangent) path, when either operand
 *      carries a tangent (reverse-over-Taylor, P5/ESH-0190): both series are
 *      propagated via the ddual_* recurrences (or tr_add/tr_sub for
 *      add/sub, which are already linear).
 *   2. Exact-coefficient path, when both operands are exact at the active
 *      epoch (P6/ESH-0191): add/sub/mul/div and non-negative-integer pow
 *      stay exact via taylor_binary_exact()/taylor_pow_exact(); any other
 *      op/exponent shape falls through to the next tier.
 *   3. General COEFF_F64 path: normalises both operands to raw double
 *      coefficient arrays (normalise_operand()) and applies the tr_*
 *      recurrence for the op, with pow using tr_pow_const() for a constant
 *      exponent or u^w = exp(w*log(u)) otherwise.
 *
 * `arena` defaults to the global arena when NULL.
 *
 * @param arena   Allocation arena for any new tower/scratch storage (or NULL for global).
 * @param left    Left operand: a Taylor tower or a plain scalar tagged value.
 * @param right   Right operand: a Taylor tower or a plain scalar tagged value.
 * @param op      One of the ESH_TAYLOR_OP_* op-codes (add/sub/mul/div/pow).
 * @param result  Out-parameter receiving the tagged result.
 */
void eshkol_taylor_binary_tagged(arena_t* arena,
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result) {
    if (!arena) arena = get_global_arena();

    uint32_t order_k, epoch;
    result_shape(left, right, &order_k, &epoch);
    int n = (int)order_k + 1;

    /* P5 (ESH-0190): reverse-over-Taylor. If either operand carries a first-
     * order seed tangent (a tangent-tower, a forward jet, or a reverse-tape AD
     * node), propagate the seed derivative alongside the value series so the
     * outer gradient can read d(f^(k))/d(seed) at extraction. */
    if (operand_has_tangent(left) || operand_has_tangent(right)) {
        double uvb[ESH_TAYLOR_STACKN], utb[ESH_TAYLOR_STACKN];
        double wvb[ESH_TAYLOR_STACKN], wtb[ESH_TAYLOR_STACKN];
        double *uv=uvb,*ut=utb,*wv=wvb,*wt=wtb, *h1=NULL,*h2=NULL,*h3=NULL,*h4=NULL;
        if (n > ESH_TAYLOR_STACKN) {
            h1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            h2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            h3=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            h4=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            uv=h1;ut=h2;wv=h3;wt=h4;
        }
        normalise_operand_dual(left,  epoch, uv, ut, n);
        normalise_operand_dual(right, epoch, wv, wt, n);
        esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k,
            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch) | ESH_TAYLOR_TANGENT_FLAG);
        if (!out) { *result = eshkol_make_double(0.0); return; }
        double* ov = out->c;
        double* ot = taylor_tan(out);
        switch (op) {
            case ESH_TAYLOR_OP_add: tr_add(ov, uv, wv, n); tr_add(ot, ut, wt, n); break;
            case ESH_TAYLOR_OP_sub: tr_sub(ov, uv, wv, n); tr_sub(ot, ut, wt, n); break;
            case ESH_TAYLOR_OP_mul: ddual_mul(ov, ot, uv, ut, wv, wt, n); break;
            case ESH_TAYLOR_OP_div: ddual_div(ov, ot, uv, ut, wv, wt, n); break;
            case ESH_TAYLOR_OP_pow: {
                int w_is_const = 1;
                for (int k = 1; k < n; k++) if (wv[k] != 0.0 || wt[k] != 0.0) { w_is_const = 0; break; }
                if (w_is_const && wt[0] == 0.0) {
                    ddual_pow_const(ov, ot, uv, ut, wv[0], n, arena);
                } else {
                    /* u^w = exp(w·log u); compose the dual log/mul/exp. */
                    double lb[ESH_TAYLOR_STACKN], lbt[ESH_TAYLOR_STACKN];
                    double pb[ESH_TAYLOR_STACKN], pbt[ESH_TAYLOR_STACKN];
                    double *lg=lb,*lgt=lbt,*pr=pb,*prt=pbt,*g1=NULL,*g2=NULL,*g3=NULL,*g4=NULL;
                    if (n > ESH_TAYLOR_STACKN) {
                        g1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
                        g2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
                        g3=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
                        g4=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
                        lg=g1;lgt=g2;pr=g3;prt=g4;
                    }
                    ddual_log(lg, lgt, uv, ut, n);
                    ddual_mul(pr, prt, wv, wt, lg, lgt, n);
                    ddual_exp(ov, ot, pr, prt, n);
                }
                break;
            }
            default: tr_add(ov, uv, wv, n); tr_add(ot, ut, wt, n); break;
        }
        *result = taylor_to_tagged(out);
        return;
    }

    /* P6 (ESH-0191): exact-coefficient dispatch. add/sub/mul/div and
     * non-negative-integer pow stay exact when BOTH operands are exact at
     * this epoch (a same-epoch exact tower, an exact scalar, or a
     * foreign-epoch exact tower lifted as a constant, section 5a); any
     * other operand/op shape falls through to the unchanged F64 kernel
     * below (a real/negative pow, or any inexact operand -- the R7RS
     * contagion rule, design section 9). */
    if (operand_is_exact_for_taylor(left, epoch) && operand_is_exact_for_taylor(right, epoch)) {
        if (op == ESH_TAYLOR_OP_add || op == ESH_TAYLOR_OP_sub ||
            op == ESH_TAYLOR_OP_mul || op == ESH_TAYLOR_OP_div) {
            taylor_binary_exact(arena, left, right, op, order_k, epoch, result);
            return;
        }
        if (op == ESH_TAYLOR_OP_pow) {
            int64_t p;
            if (exact_pow_exponent_as_nonneg_int(right, epoch, &p)) {
                taylor_pow_exact(arena, left, order_k, epoch, (uint64_t)p, result);
                return;
            }
            /* non-integer / negative / non-constant exponent: fall through
             * to the general (inexact) pow recurrence below. */
        }
    }

    double sbuf_u[ESH_TAYLOR_STACKN], sbuf_w[ESH_TAYLOR_STACKN];
    double *u = sbuf_u, *w = sbuf_w;
    double *hu = NULL, *hw = NULL;
    if (n > ESH_TAYLOR_STACKN) {
        hu = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        hw = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        u = hu; w = hw;
    }
    normalise_operand(left, epoch, u, n);
    normalise_operand(right, epoch, w, n);

    esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k,
                                            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch));
    if (!out) { *result = eshkol_make_double(0.0); return; }

    switch (op) {
        case ESH_TAYLOR_OP_add: tr_add(out->c, u, w, n); break;
        case ESH_TAYLOR_OP_sub: tr_sub(out->c, u, w, n); break;
        case ESH_TAYLOR_OP_mul: tr_mul(out->c, u, w, n); break;
        case ESH_TAYLOR_OP_div: tr_div(out->c, u, w, n); break;
        case ESH_TAYLOR_OP_pow: {
            /* If the exponent is a plain constant (only c[0] set), use the exact
             * power recurrence; otherwise u^w = exp(w * log(u)). */
            int w_is_const = 1;
            for (int k = 1; k < n; k++) if (w[k] != 0.0) { w_is_const = 0; break; }
            if (w_is_const) {
                tr_pow_const(out->c, u, w[0], n);
            } else {
                double lb[ESH_TAYLOR_STACKN], pb[ESH_TAYLOR_STACKN];
                double *lg = lb, *pr = pb, *hlg = NULL, *hpr = NULL;
                if (n > ESH_TAYLOR_STACKN) {
                    hlg = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
                    hpr = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
                    lg = hlg; pr = hpr;
                }
                tr_log(lg, u, n);
                tr_mul(pr, w, lg, n);
                tr_exp(out->c, pr, n);
            }
            break;
        }
        default: tr_add(out->c, u, w, n); break;
    }
    *result = taylor_to_tagged(out);
}

/**
 * @brief Apply a unary op (neg/exp/log/sin/cos/tan/sqrt/abs/sinh/cosh/tanh) to
 *        a Taylor-tower or scalar operand; entry point called from codegen's
 *        numeric dispatch.
 *
 * Mirrors eshkol_taylor_binary_tagged()'s three-tier dispatch:
 *   1. Dual (value + seed-tangent) path when the operand carries a tangent
 *      (P5/ESH-0190), via the ddual_* recurrences (sin/cos/tan/sinh/cosh/tanh
 *      are composed from ddual_exp/ddual_log/ddual_sincos/ddual_div/
 *      ddual_pow_const, mirroring the F64 tier's composition below).
 *   2. Exact-coefficient path (P6/ESH-0191), applicable only to neg/abs --
 *      the only unary ops that stay exact (every transcendental primitive
 *      here is irrational even at an exact base point, so those always
 *      demote to F64 via normalise_operand()).
 *   3. General COEFF_F64 path: normalises the operand (normalise_operand())
 *      and applies the matching tr_* recurrence; sin/cos/tan/sinh/cosh/tanh
 *      are composed from tr_sincos()/tr_exp()/tr_div()/tr_pow_const().
 *
 * `arena` defaults to the global arena when NULL. A non-tower operand
 * produces an order-0 result.
 *
 * @param arena   Allocation arena for any new tower/scratch storage (or NULL for global).
 * @param in      Operand: a Taylor tower or a plain scalar tagged value.
 * @param op      One of the ESH_TAYLOR_UOP_* op-codes.
 * @param result  Out-parameter receiving the tagged result.
 */
void eshkol_taylor_unary_tagged(arena_t* arena,
    const eshkol_tagged_value_t* in, int op, eshkol_tagged_value_t* result) {
    if (!arena) arena = get_global_arena();

    esh_taylor_t* t = tagged_as_taylor(in);
    uint32_t order_k = t ? t->order_k : 0;
    uint32_t epoch = t ? ESH_TAYLOR_GET_EPOCH(t->flags) : 0;
    int n = (int)order_k + 1;

    /* P5 (ESH-0190): dual (value + seed-tangent) unary path — see the binary
     * dispatch for the rationale. Fires only when the operand carries a seed
     * tangent, so the plain forward-tower path below is byte-for-byte unchanged. */
    if (operand_has_tangent(in)) {
        double uvb[ESH_TAYLOR_STACKN], utb[ESH_TAYLOR_STACKN];
        double *uv=uvb, *ut=utb, *h1=NULL,*h2=NULL;
        if (n > ESH_TAYLOR_STACKN) {
            h1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            h2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
            uv=h1;ut=h2;
        }
        normalise_operand_dual(in, epoch, uv, ut, n);
        esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k,
            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch) | ESH_TAYLOR_TANGENT_FLAG);
        if (!out) { *result = eshkol_make_double(0.0); return; }
        double* ov = out->c;
        double* ot = taylor_tan(out);
        switch (op) {
            case ESH_TAYLOR_UOP_neg: tr_neg(ov, uv, n); tr_neg(ot, ut, n); break;
            case ESH_TAYLOR_UOP_exp: ddual_exp(ov, ot, uv, ut, n); break;
            case ESH_TAYLOR_UOP_log: ddual_log(ov, ot, uv, ut, n); break;
            case ESH_TAYLOR_UOP_sin: {
                double cb[ESH_TAYLOR_STACKN], cbt[ESH_TAYLOR_STACKN];
                double *co=cb,*cot=cbt,*g1=NULL,*g2=NULL;
                if (n>ESH_TAYLOR_STACKN){g1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));co=g1;cot=g2;}
                ddual_sincos(ov, ot, co, cot, uv, ut, n);
                break;
            }
            case ESH_TAYLOR_UOP_cos: {
                double sb[ESH_TAYLOR_STACKN], sbt[ESH_TAYLOR_STACKN];
                double *so=sb,*sot=sbt,*g1=NULL,*g2=NULL;
                if (n>ESH_TAYLOR_STACKN){g1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));so=g1;sot=g2;}
                ddual_sincos(so, sot, ov, ot, uv, ut, n);
                break;
            }
            case ESH_TAYLOR_UOP_tan: {
                double sb[ESH_TAYLOR_STACKN],sbt[ESH_TAYLOR_STACKN],cb[ESH_TAYLOR_STACKN],cbt[ESH_TAYLOR_STACKN];
                double *so=sb,*sot=sbt,*co=cb,*cot=cbt,*g1=NULL,*g2=NULL,*g3=NULL,*g4=NULL;
                if (n>ESH_TAYLOR_STACKN){g1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g3=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g4=(double*)arena_allocate(arena,(size_t)n*sizeof(double));so=g1;sot=g2;co=g3;cot=g4;}
                ddual_sincos(so, sot, co, cot, uv, ut, n);
                ddual_div(ov, ot, so, sot, co, cot, n);
                break;
            }
            case ESH_TAYLOR_UOP_sqrt: ddual_pow_const(ov, ot, uv, ut, 0.5, n, arena); break;
            case ESH_TAYLOR_UOP_abs: {
                double sgn = (uv[0] < 0.0) ? -1.0 : 1.0;
                ov[0] = fabs(uv[0]); ot[0] = sgn * ut[0];
                for (int k = 1; k < n; k++) { ov[k] = sgn * uv[k]; ot[k] = sgn * ut[k]; }
                break;
            }
            case ESH_TAYLOR_UOP_sinh:
            case ESH_TAYLOR_UOP_cosh:
            case ESH_TAYLOR_UOP_tanh: {
                /* sinh/cosh/tanh via dual exp(±u). */
                double epb[ESH_TAYLOR_STACKN],eptb[ESH_TAYLOR_STACKN];
                double emb[ESH_TAYLOR_STACKN],emtb[ESH_TAYLOR_STACKN];
                double nub[ESH_TAYLOR_STACKN],nutb[ESH_TAYLOR_STACKN];
                double *ep=epb,*ept=eptb,*em=emb,*emt=emtb,*nu=nub,*nut=nutb;
                double *g1=NULL,*g2=NULL,*g3=NULL,*g4=NULL,*g5=NULL,*g6=NULL;
                if (n>ESH_TAYLOR_STACKN){g1=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g2=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g3=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g4=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g5=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g6=(double*)arena_allocate(arena,(size_t)n*sizeof(double));ep=g1;ept=g2;em=g3;emt=g4;nu=g5;nut=g6;}
                ddual_exp(ep, ept, uv, ut, n);
                tr_neg(nu, uv, n); tr_neg(nut, ut, n);
                ddual_exp(em, emt, nu, nut, n);
                if (op == ESH_TAYLOR_UOP_sinh) {
                    for (int k=0;k<n;k++){ ov[k]=0.5*(ep[k]-em[k]); ot[k]=0.5*(ept[k]-emt[k]); }
                } else if (op == ESH_TAYLOR_UOP_cosh) {
                    for (int k=0;k<n;k++){ ov[k]=0.5*(ep[k]+em[k]); ot[k]=0.5*(ept[k]+emt[k]); }
                } else { /* tanh = sinh/cosh */
                    double shb[ESH_TAYLOR_STACKN],shtb[ESH_TAYLOR_STACKN],chb[ESH_TAYLOR_STACKN],chtb[ESH_TAYLOR_STACKN];
                    double *sh=shb,*sht=shtb,*ch=chb,*cht=chtb,*g7=NULL,*g8=NULL,*g9=NULL,*g10=NULL;
                    if (n>ESH_TAYLOR_STACKN){g7=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g8=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g9=(double*)arena_allocate(arena,(size_t)n*sizeof(double));g10=(double*)arena_allocate(arena,(size_t)n*sizeof(double));sh=g7;sht=g8;ch=g9;cht=g10;}
                    for (int k=0;k<n;k++){ sh[k]=0.5*(ep[k]-em[k]); sht[k]=0.5*(ept[k]-emt[k]); ch[k]=0.5*(ep[k]+em[k]); cht[k]=0.5*(ept[k]+emt[k]); }
                    ddual_div(ov, ot, sh, sht, ch, cht, n);
                }
                break;
            }
            default: memcpy(ov, uv, (size_t)n*sizeof(double)); memcpy(ot, ut, (size_t)n*sizeof(double)); break;
        }
        *result = taylor_to_tagged(out);
        return;
    }

    /* P6 (ESH-0191): neg/abs are exact-preserving (no genuine division, no
     * transcendental base point); every OTHER unary primitive reachable
     * here is transcendental (exp/log/sin/cos/tan/sqrt/sinh/cosh/tanh) and
     * has no exact rational series even at an exact base point (e.g.
     * exp(1) is irrational) -- those fall through to the unchanged F64
     * kernel below, which is the documented graceful promotion (design
     * section 9); normalise_operand demotes the exact input correctly. */
    if (t && taylor_is_exact(t) && (op == ESH_TAYLOR_UOP_neg || op == ESH_TAYLOR_UOP_abs)) {
        const eshkol_tagged_value_t* u = taylor_exact_c_const(t);
        eshkol_tagged_value_t* s = alloc_exact_series(arena, n);
        if (!s) { *result = eshkol_make_double(0.0); return; }
        if (op == ESH_TAYLOR_UOP_neg) {
            eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
            for (int k = 0; k < n; k++) s[k] = exact_sub(arena, zero, u[k]);
        } else { /* abs: sign is fixed by c[0]'s sign (a truncated series
                  * around x0 -- the classic |x| kink at x0=0 is a pre-existing,
                  * documented limitation of this recurrence, unchanged from
                  * the F64 tr_* version above). */
            int neg0 = tagged_is_negative_exact(&u[0]);
            eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
            for (int k = 0; k < n; k++) s[k] = neg0 ? exact_sub(arena, zero, u[k]) : u[k];
        }
        taylor_materialize_exact_or_demote(arena, s, n, order_k, epoch, result);
        return;
    }

    double sbuf_u[ESH_TAYLOR_STACKN];
    double* u = sbuf_u;
    double* hu = NULL;
    if (n > ESH_TAYLOR_STACKN) {
        hu = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        u = hu;
    }
    normalise_operand(in, epoch, u, n);

    esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k,
                                            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch));
    if (!out) { *result = eshkol_make_double(0.0); return; }

    switch (op) {
        case ESH_TAYLOR_UOP_neg: tr_neg(out->c, u, n); break;
        case ESH_TAYLOR_UOP_exp: tr_exp(out->c, u, n); break;
        case ESH_TAYLOR_UOP_log: tr_log(out->c, u, n); break;
        case ESH_TAYLOR_UOP_sin: {
            double cbuf[ESH_TAYLOR_STACKN]; double* co = cbuf;
            double* hco = NULL;
            if (n > ESH_TAYLOR_STACKN) { hco = (double*)arena_allocate(arena, (size_t)n*sizeof(double)); co = hco; }
            tr_sincos(out->c, co, u, n);
            break;
        }
        case ESH_TAYLOR_UOP_cos: {
            double sbuf[ESH_TAYLOR_STACKN]; double* so = sbuf;
            double* hso = NULL;
            if (n > ESH_TAYLOR_STACKN) { hso = (double*)arena_allocate(arena, (size_t)n*sizeof(double)); so = hso; }
            tr_sincos(so, out->c, u, n);
            break;
        }
        case ESH_TAYLOR_UOP_tan: {
            double sb[ESH_TAYLOR_STACKN], cb[ESH_TAYLOR_STACKN];
            double *so = sb, *co = cb, *hso = NULL, *hco = NULL;
            if (n > ESH_TAYLOR_STACKN) {
                hso = (double*)arena_allocate(arena, (size_t)n*sizeof(double));
                hco = (double*)arena_allocate(arena, (size_t)n*sizeof(double));
                so = hso; co = hco;
            }
            tr_sincos(so, co, u, n);
            tr_div(out->c, so, co, n);
            break;
        }
        case ESH_TAYLOR_UOP_sqrt: tr_pow_const(out->c, u, 0.5, n); break;
        case ESH_TAYLOR_UOP_abs: {
            double sgn = (u[0] < 0.0) ? -1.0 : 1.0;
            out->c[0] = fabs(u[0]);
            for (int k = 1; k < n; k++) out->c[k] = sgn * u[k];
            break;
        }
        case ESH_TAYLOR_UOP_sinh: {
            /* sinh(u) = (exp(u) - exp(-u))/2 */
            double eb[ESH_TAYLOR_STACKN], nb[ESH_TAYLOR_STACKN], mb[ESH_TAYLOR_STACKN];
            double *ep = eb, *nu = nb, *em = mb;
            double *hep=NULL,*hnu=NULL,*hem=NULL;
            if (n > ESH_TAYLOR_STACKN) { hep=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hnu=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hem=(double*)arena_allocate(arena,(size_t)n*sizeof(double));ep=hep;nu=hnu;em=hem;}
            tr_exp(ep, u, n);
            tr_neg(nu, u, n);
            tr_exp(em, nu, n);
            for (int k = 0; k < n; k++) out->c[k] = 0.5 * (ep[k] - em[k]);
            break;
        }
        case ESH_TAYLOR_UOP_cosh: {
            double eb[ESH_TAYLOR_STACKN], nb[ESH_TAYLOR_STACKN], mb[ESH_TAYLOR_STACKN];
            double *ep = eb, *nu = nb, *em = mb;
            double *hep=NULL,*hnu=NULL,*hem=NULL;
            if (n > ESH_TAYLOR_STACKN) { hep=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hnu=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hem=(double*)arena_allocate(arena,(size_t)n*sizeof(double));ep=hep;nu=hnu;em=hem;}
            tr_exp(ep, u, n);
            tr_neg(nu, u, n);
            tr_exp(em, nu, n);
            for (int k = 0; k < n; k++) out->c[k] = 0.5 * (ep[k] + em[k]);
            break;
        }
        case ESH_TAYLOR_UOP_tanh: {
            /* tanh via sinh/cosh */
            double sb[ESH_TAYLOR_STACKN], cbb[ESH_TAYLOR_STACKN], eb[ESH_TAYLOR_STACKN], nb[ESH_TAYLOR_STACKN], mb[ESH_TAYLOR_STACKN];
            double *sh=sb,*ch=cbb,*ep=eb,*nu=nb,*em=mb;
            double *hsh=NULL,*hch=NULL,*hep=NULL,*hnu=NULL,*hem=NULL;
            if (n > ESH_TAYLOR_STACKN) { hsh=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hch=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hep=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hnu=(double*)arena_allocate(arena,(size_t)n*sizeof(double));hem=(double*)arena_allocate(arena,(size_t)n*sizeof(double));sh=hsh;ch=hch;ep=hep;nu=hnu;em=hem;}
            tr_exp(ep, u, n);
            tr_neg(nu, u, n);
            tr_exp(em, nu, n);
            for (int k = 0; k < n; k++) { sh[k] = 0.5*(ep[k]-em[k]); ch[k] = 0.5*(ep[k]+em[k]); }
            tr_div(out->c, sh, ch, n);
            break;
        }
        default: memcpy(out->c, u, (size_t)n * sizeof(double)); break;
    }
    *result = taylor_to_tagged(out);
}

/* ----------------------------------------------------------------------- */
/* seeding, extraction, differentiation (called from codegen for the API)   */
/* ----------------------------------------------------------------------- */

/* Monotonic epoch counter. 0 is reserved for "no active perturbation"
 * (scalars / constants); every fresh differentiation context gets >= 1. */
#ifdef __cplusplus
static std::atomic<uint32_t> g_taylor_epoch{0};
/** @brief Atomically allocate and return the next process-wide Taylor differentiation epoch tag (1..0xFFFF, wrapping, never 0). */
uint32_t eshkol_taylor_next_epoch(void) {
    uint32_t e = g_taylor_epoch.fetch_add(1) + 1;
    /* 16-bit tag: wrap back to 1 (never 0). */
    return ((e - 1) & 0xFFFFu) + 1;
}
#else
static uint32_t g_taylor_epoch = 0;
/** @brief Allocate and return the next process-wide Taylor differentiation epoch tag (1..0xFFFF, wrapping, never 0). */
uint32_t eshkol_taylor_next_epoch(void) {
    uint32_t e = ++g_taylor_epoch;
    return ((e - 1) & 0xFFFFu) + 1;
}
#endif

/* Seed a tower: {x0, is_var ? 1 : 0, 0, ...} of order K under `epoch`. */
void eshkol_taylor_seed(arena_t* arena, double x0, int is_var,
                        uint32_t order_k, uint32_t epoch,
                        eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    esh_taylor_t* t = eshkol_taylor_alloc(arena, order_k,
                                          ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch));
    if (!t) { *out = eshkol_make_double(x0); return; }
    t->c[0] = x0;
    if (is_var && order_k >= 1) t->c[1] = 1.0;
    *out = taylor_to_tagged(t);
}

/* Seed the differentiation variable from a tagged point at a FRESH epoch.
 * Reads x0 as the point's scalar value (a plain double/int, or c[0] of an
 * outer tower) and produces {x0, 1, 0, ...} of order K. Called by codegen for
 * (taylor f x k) / (derivative-n f x k).
 *
 * P6 (ESH-0191), R7RS exactness contagion (design section 9): when the point
 * is itself exact (a plain int64/bignum/rational, or the c[0] of an outer
 * EXACT tower), seed an EXACT tower instead, so every derivative through a
 * polynomial/rational subgraph of an exact point comes back exact. Any other
 * point (a double, or an outer COEFF_F64 tower) seeds the unchanged
 * COEFF_F64 tower. */
void eshkol_taylor_seed_tagged(arena_t* arena, const eshkol_tagged_value_t* point,
                               int32_t order_k, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (order_k < 0) order_k = 0;
    uint32_t epoch = eshkol_taylor_next_epoch();

    esh_taylor_t* outer = tagged_as_taylor(point);
    eshkol_tagged_value_t x0_exact;
    int is_exact_point;
    if (outer) {
        is_exact_point = taylor_is_exact(outer) && tagged_is_exact_number(&taylor_exact_c_const(outer)[0]);
        x0_exact = is_exact_point ? taylor_exact_c_const(outer)[0] : eshkol_make_int64(0, true);
    } else {
        is_exact_point = tagged_is_exact_number(point);
        x0_exact = is_exact_point ? *point : eshkol_make_int64(0, true);
    }

    if (is_exact_point) {
        esh_taylor_t* t = eshkol_taylor_alloc_exact(arena, (uint32_t)order_k, epoch);
        if (!t) { *out = eshkol_make_double(tagged_any_to_double(&x0_exact)); return; }
        eshkol_tagged_value_t* c = taylor_exact_c(t);
        c[0] = x0_exact;
        if (order_k >= 1) c[1] = eshkol_make_int64(1, true);
        eshkol_tagged_value_t v; memset(&v, 0, sizeof(v));
        v.type = ESHKOL_VALUE_HEAP_PTR; v.flags = ESHKOL_VALUE_INEXACT_FLAG;
        v.data.ptr_val = (uint64_t)(uintptr_t)t;
        *out = v;
        return;
    }

    double x0 = tagged_scalar_value(point);
    eshkol_taylor_seed(arena, x0, 1, (uint32_t)order_k, epoch, out);
}

/** @brief Compute n! as a double, used to convert Taylor coefficient c[n] to the n-th derivative f^(n)(x0) = n! * c[n]. */
static double factorial_d(uint32_t n) {
    double f = 1.0;
    for (uint32_t i = 2; i <= n; i++) f *= (double)i;
    return f;
}

/* f^(n)(x0) = n! * c[n]. Non-towers: value at n==0, else 0.
 * Discards exactness (always returns a raw double) -- kept for any caller
 * that only wants the numeric magnitude. Codegen uses the exactness-
 * preserving eshkol_taylor_extract_tagged below (P6, ESH-0191). */
double eshkol_taylor_extract(const eshkol_tagged_value_t* tv, uint32_t n) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) return (n == 0) ? tagged_scalar_value(tv) : 0.0;
    if (n > t->order_k) return 0.0;
    if (taylor_is_exact(t)) return factorial_d(n) * tagged_any_to_double(&taylor_exact_c_const(t)[n]);
    return factorial_d(n) * t->c[n];
}

/* ----------------------------------------------------------------------- */
/* P5 seed-tangent extraction & AD-node lift (ESH-0190)                     */
/* ----------------------------------------------------------------------- */

/* 1 iff the tower value carries a first-order seed tangent series. */
int eshkol_taylor_has_tangent(const eshkol_tagged_value_t* tv) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    return (t && ESH_TAYLOR_HAS_TANGENT(t->flags)) ? 1 : 0;
}

/* d(f^(n)(x0))/d(reverse-seed) = n! * tangent[n]. 0 when the tower has no
 * tangent series or n exceeds the order. This is the dseed the outer gradient's
 * mixed-mode record (or forward jet) reads at the derivative-n return site. */
double eshkol_taylor_extract_tangent(const eshkol_tagged_value_t* tv, uint32_t n) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t || !ESH_TAYLOR_HAS_TANGENT(t->flags)) return 0.0;
    if (n > t->order_k) return 0.0;
    double* tan = taylor_tan(t);
    return factorial_d(n) * tan[n];
}

/* Freeze a reverse-tape AD node into a dual-tower CONSTANT of order K:
 *   value   = {node->value, 0, ..., 0}
 *   tangent = {seed_flag(node), 0, ..., 0}
 * so that when it flows into tower arithmetic the reverse tape does not swallow
 * the tower (withADBinaryDispatch would otherwise convert the tower operand to a
 * scalar AD node) and its first-order dependence on the active gradient seed is
 * propagated as the tower's seed tangent (docs/design/AD_TAYLOR_TOWER.md §8).
 * A constant tower zero-extends, so `order_k` need only be a best-effort upper
 * bound (the current innermost tower order). Called from codegen's
 * maybeJetLiftTapeOperand while a tower differentiation is active. */
void eshkol_taylor_lift_ad_node(arena_t* arena, void* node, int32_t order_k,
                                eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (order_k < 0) order_k = 0;
    double val = node ? ((const ad_node_t*)node)->value : 0.0;
    double flag = node ? eshkol_ad_seed_flag(node) : 0.0;
    esh_taylor_t* t = eshkol_taylor_alloc(arena, (uint32_t)order_k,
        ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, 0u) | ESH_TAYLOR_TANGENT_FLAG);
    if (!t) { *out = eshkol_make_double(val); return; }
    t->c[0] = val;
    taylor_tan(t)[0] = flag;
    *out = taylor_to_tagged(t);
}

/* f^(n)(x0) = n! * c[n], PRESERVING exactness (P6, ESH-0191): when the
 * source tower is COEFF_RATIONAL, n! and the product with c[n] are computed
 * through the exact numeric tower (arbitrary-precision, via exact_mul's
 * bignum dispatch), so `(exact? (derivative-n f x n))` is #t whenever x and
 * every operator f applies are exact. Falls back to a tagged double exactly
 * like eshkol_taylor_extract for COEFF_F64 towers / non-tower operands. */
void eshkol_taylor_extract_tagged(arena_t* arena, const eshkol_tagged_value_t* tv,
                                  uint32_t n, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) { *out = (n == 0) ? *tv : eshkol_make_double(0.0); return; }
    if (n > t->order_k) {
        *out = taylor_is_exact(t) ? eshkol_make_int64(0, true) : eshkol_make_double(0.0);
        return;
    }
    if (taylor_is_exact(t)) {
        eshkol_tagged_value_t fact = eshkol_make_int64(1, true);
        for (uint32_t i = 2; i <= n; i++) fact = exact_mul(arena, fact, eshkol_make_int64((int64_t)i, true));
        *out = exact_mul(arena, fact, taylor_exact_c_const(t)[n]);
        return;
    }
    *out = eshkol_make_double(factorial_d(n) * t->c[n]);
}

/* Differentiate a tower: (f')_k = (k+1) * c_{k+1}. Preserves order/epoch;
 * the top coefficient becomes 0. Non-towers differentiate to 0. P6: an EXACT
 * tower differentiates to an EXACT tower ((k+1) is a plain exact int64). */
void eshkol_taylor_shift(arena_t* arena, const eshkol_tagged_value_t* tv,
                         eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) { *out = eshkol_make_double(0.0); return; }
    if (taylor_is_exact(t)) {
        uint32_t epoch = ESH_TAYLOR_GET_EPOCH(t->flags);
        esh_taylor_t* r = eshkol_taylor_alloc_exact(arena, t->order_k, epoch);
        if (!r) { *out = eshkol_make_double(0.0); return; }
        const eshkol_tagged_value_t* c = taylor_exact_c_const(t);
        eshkol_tagged_value_t* rc = taylor_exact_c(r);
        for (uint32_t k = 0; k < t->order_k; k++)
            rc[k] = exact_mul(arena, eshkol_make_int64((int64_t)(k + 1), true), c[k + 1]);
        rc[t->order_k] = eshkol_make_int64(0, true);
        eshkol_tagged_value_t v; memset(&v, 0, sizeof(v));
        v.type = ESHKOL_VALUE_HEAP_PTR; v.flags = ESHKOL_VALUE_INEXACT_FLAG;
        v.data.ptr_val = (uint64_t)(uintptr_t)r;
        *out = v;
        return;
    }
    esh_taylor_t* r = eshkol_taylor_alloc(arena, t->order_k, t->flags);
    if (!r) { *out = eshkol_make_double(0.0); return; }
    for (uint32_t k = 0; k < t->order_k; k++)
        r->c[k] = (double)(k + 1) * t->c[k + 1];
    r->c[t->order_k] = 0.0;
    *out = taylor_to_tagged(r);
}

/* Build a Scheme list of the K+1 coefficients (c[0] first) for `(taylor f x k)`.
 * Written through `out` (out-param convention, matching the codegen call).
 * P6: each element is the coefficient's OWN tagged value -- an exact
 * int64/bignum/rational for a COEFF_RATIONAL tower, a tagged double
 * otherwise -- so `(exact? (car (taylor f x k)))` reflects the tower's
 * actual coefficient type. */
void eshkol_taylor_coeffs_list(arena_t* arena, const eshkol_tagged_value_t* tv,
                               int32_t order_k_in, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (order_k_in < 0) order_k_in = 0;
    uint32_t order_k = (uint32_t)order_k_in;
    eshkol_tagged_value_t nil;
    memset(&nil, 0, sizeof(nil));
    nil.type = ESHKOL_VALUE_NULL;

    esh_taylor_t* t = tagged_as_taylor(tv);
    int exact = t && taylor_is_exact(t);
    eshkol_tagged_value_t acc = nil;
    /* cons from the tail so element order is c[0], c[1], ..., c[K]. */
    for (int k = (int)order_k; k >= 0; k--) {
        eshkol_tagged_value_t cv;
        if (t) {
            if ((uint32_t)k <= t->order_k) {
                cv = exact ? taylor_exact_c_const(t)[k] : eshkol_make_double(t->c[k]);
            } else {
                cv = exact ? eshkol_make_int64(0, true) : eshkol_make_double(0.0);
            }
        } else {
            cv = (k == 0) ? *tv : eshkol_make_double(0.0);
        }
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!cell) { *out = nil; return; }
        cell->car = cv;
        cell->cdr = acc;
        eshkol_tagged_value_t v;
        memset(&v, 0, sizeof(v));
        v.type = ESHKOL_VALUE_HEAP_PTR;
        v.data.ptr_val = (uint64_t)(uintptr_t)cell;
        acc = v;
    }
    *out = acc;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
