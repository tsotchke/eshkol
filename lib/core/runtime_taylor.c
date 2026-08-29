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
 * exact point yields an exact tower; add/sub/mul/div and integer
 * integer pow stay exact; the moment an operand is inexact, an exact op
 * operation cannot be represented exactly, or a transcendental
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
#include <float.h>
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
    if (tagged_is_rational(v)) {
        const eshkol_rational_t* r =
            (const eshkol_rational_t*)(uintptr_t)v->data.ptr_val;
        return r->is_big ? eshkol_bignum_is_negative(r->big_num)
                         : r->numerator < 0;
    }
    return v->data.int_val < 0; /* int64 */
}

static inline int tagged_is_zero_exact(const eshkol_tagged_value_t* v) {
    if (tagged_is_bignum(v))
        return eshkol_bignum_is_zero((eshkol_bignum_t*)(uintptr_t)v->data.ptr_val);
    if (tagged_is_rational(v)) {
        const eshkol_rational_t* r =
            (const eshkol_rational_t*)(uintptr_t)v->data.ptr_val;
        return r->is_big ? eshkol_bignum_is_zero(r->big_num) : r->numerator == 0;
    }
    return v->data.int_val == 0;
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
 * exact rational substrate; gracefully degrades to double only when the
 * moment an operand is already inexact -- this
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

    if (op != 3 && !a_rat && !b_rat) {
        /* pure-integer add/sub/mul: arbitrary precision via the bignum
         * substrate (handles int64<->bignum promotion/demotion itself). */
        eshkol_tagged_value_t r;
        eshkol_bignum_binary_tagged(arena, &a, &b, op, &r);
        return r;
    }
    /* The rational backend is bignum-capable.  Do not demote a bignum
     * numerator or denominator merely because the fast int64 fields cannot
     * hold it; that was the exactness loss in x/3 at bignum scale. */
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
 * EXACT tower; otherwise (an intermediate op could not be represented by the
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

int32_t eshkol_taylor_order_tagged(void* arena,
                                   const eshkol_tagged_value_t* left,
                                   const eshkol_tagged_value_t* right,
                                   int op) {
    eshkol_tagged_value_t left_c0;
    eshkol_tagged_value_t right_c0;
    const eshkol_tagged_value_t* l = left;
    const eshkol_tagged_value_t* r = right;
    esh_taylor_t* lt = tagged_as_taylor(left);
    esh_taylor_t* rt = tagged_as_taylor(right);
    if (lt) {
        if (taylor_is_exact(lt)) left_c0 = taylor_exact_c_const(lt)[0];
        else left_c0 = eshkol_make_double(lt->c[0]);
        l = &left_c0;
    }
    if (rt) {
        if (taylor_is_exact(rt)) right_c0 = taylor_exact_c_const(rt)[0];
        else right_c0 = eshkol_make_double(rt->c[0]);
        r = &right_c0;
    }

    if (tagged_is_exact_number(l) && tagged_is_exact_number(r)) {
        eshkol_tagged_value_t result;
        eshkol_rational_compare_tagged_ptr(arena, l, r, op, &result);
        return (int32_t)result.data.int_val;
    }

    double a = tagged_any_to_double(l);
    double b = tagged_any_to_double(r);
    switch (op) {
        case 0: return a < b;
        case 1: return a > b;
        case 2: return a == b;
        case 3: return a <= b;
        case 4: return a >= b;
        default: return 0;
    }
}

/* ── AD seed / evaluation-point coercion (ESH-0393) ───────────────────────
 *
 * The forward jet and the reverse tape both carry a RAW DOUBLE per component,
 * so every AD entry point has to turn its evaluation point into a double at
 * the boundary. Doing that by REINTERPRETING the tagged value's data field --
 * `SIToFP` for "anything that isn't a double", or a plain bitcast for
 * "anything at all" -- classifies by what the caller EXPECTED rather than by
 * what the value IS. An exact rational or bignum is HEAP-tagged, so its data
 * field holds a POINTER: SIToFP turns it into the object's ADDRESS (a value of
 * heap magnitude, e.g. 1.0e10) and a bitcast turns it into a denormal near
 * 5e-314. Either way the substrate then differentiates a fabricated number.
 *
 * This is the single authority on that coercion. It dispatches on the value's
 * RUNTIME TAG and converts every numeric representation Eshkol has -- int64,
 * double, bignum, rational, forward jet, Taylor tower -- to the double it
 * actually denotes. `*ok` reports whether the value was a number at all, so
 * the caller can raise instead of inventing one; a non-numeric HEAP object
 * (a vector, a string, a closure, ...) is exactly the misread this function
 * exists to stop, and is never coerced.
 *
 * Immediates whose data field genuinely holds an integer (BOOL, CHAR) keep
 * the historical widening: they are not the pointer-misread family, and the
 * jet path has always accepted them.
 */
double eshkol_ad_seed_to_double(const eshkol_tagged_value_t* v, int32_t* ok) {
    if (ok) *ok = 1;
    if (!v) { if (ok) *ok = 0; return 0.0; }

    uint8_t bt = (uint8_t)(v->type & 0x0F);

    /* Inexact and immediate-exact scalars: the data field is the number. */
    if (bt == ESHKOL_VALUE_DOUBLE) return v->data.double_val;
    if (bt == ESHKOL_VALUE_INT64)  return (double)v->data.int_val;

    /* A forward jet reaching an entry point is a NESTED differentiation's
     * point; its primal is the value being differentiated at. */
    if (bt == ESHKOL_VALUE_DUAL_NUMBER && v->data.ptr_val != 0)
        return ((const double*)(uintptr_t)v->data.ptr_val)[0];

    /* Heap numerics: exact rational / bignum, or a Taylor tower's c[0]. */
    if (bt == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val != 0) {
        const eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
        if (hdr) {
            switch (hdr->subtype) {
                case HEAP_SUBTYPE_RATIONAL:
                    return eshkol_rational_to_double((void*)(uintptr_t)v->data.ptr_val);
                case HEAP_SUBTYPE_BIGNUM:
                    return eshkol_bignum_to_double(
                        (eshkol_bignum_t*)(uintptr_t)v->data.ptr_val);
                case HEAP_SUBTYPE_TAYLOR:
                    return tagged_scalar_value(v);
                default:
                    break;      /* not a number -- fall through to the refusal */
            }
        }
        if (ok) *ok = 0;
        return 0.0;
    }

    /* BOOL / CHAR and other immediates with an integral data field. */
    if (bt == ESHKOL_VALUE_BOOL || bt == ESHKOL_VALUE_CHAR)
        return (double)v->data.int_val;

    if (ok) *ok = 0;
    return 0.0;
}

/* Companion classifier: is this evaluation point a SCALAR (an f: R→R point)
 * rather than a collection (an f: R^n→R point)?
 *
 * Every multivariate AD entry point has to make this choice before it can seed
 * anything, and each did it by enumerating the tags it expected a scalar to
 * have -- DOUBLE, INT64, sometimes DUAL_NUMBER. An exact rational or bignum is
 * a perfectly ordinary scalar that happens to be HEAP-tagged, so it failed
 * every enumeration and was routed down the COLLECTION path, where its object
 * was then dereferenced as [dims][rank][elems]: `(gradient f 1/3)` returned
 * `#()` and `(hessian f 1/3)` segfaulted. Deciding it here keeps the answer in
 * one place for all of them. */
int32_t eshkol_ad_point_is_scalar(const eshkol_tagged_value_t* v) {
    if (!v) return 0;
    uint8_t bt = (uint8_t)(v->type & 0x0F);
    if (bt == ESHKOL_VALUE_DOUBLE || bt == ESHKOL_VALUE_INT64 ||
        bt == ESHKOL_VALUE_DUAL_NUMBER || bt == ESHKOL_VALUE_BOOL ||
        bt == ESHKOL_VALUE_CHAR)
        return 1;
    if (bt == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val != 0) {
        const eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
        if (hdr && (hdr->subtype == HEAP_SUBTYPE_RATIONAL ||
                    hdr->subtype == HEAP_SUBTYPE_BIGNUM  ||
                    hdr->subtype == HEAP_SUBTYPE_TAYLOR))
            return 1;
    }
    return 0;
}

/* Narrow companion: is this point an EXACT HEAP scalar (rational or bignum)?
 *
 * Deliberately NOT the same question as eshkol_ad_point_is_scalar: it excludes
 * DUAL_NUMBER and Taylor towers. Some entry points classify a dual point
 * separately (a dual point means an enclosing differentiation is live), so the
 * `INT64 | DOUBLE` tests those operators use must be widened by exactly the
 * exact-heap-numeric case and nothing else -- otherwise widening the entry
 * classification would silently re-route nested AD as well.
 *
 * Both the ENTRY branch (which promotes a scalar point) and the EXIT unwrap
 * (which returns a bare scalar rather than a 1-vector for a scalar point) must
 * be widened together: widening only the entry made `(gradient f 1/3)` return
 * `#(0.666…)` where `(gradient f 0.333…)` returns `0.666…`. */
int32_t eshkol_ad_point_is_exact_scalar(const eshkol_tagged_value_t* v) {
    if (!v) return 0;
    if ((uint8_t)(v->type & 0x0F) != ESHKOL_VALUE_HEAP_PTR || v->data.ptr_val == 0)
        return 0;
    const eshkol_object_header_t* hdr =
        ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
    return (hdr && (hdr->subtype == HEAP_SUBTYPE_RATIONAL ||
                    hdr->subtype == HEAP_SUBTYPE_BIGNUM)) ? 1 : 0;
}

/* Is this evaluation point an EXACT NUMBER -- an R7RS-exact integer (immediate
 * int64 or heap bignum) or an exact rational?
 *
 * This is the gate on the EXACT TIER: `derivative`/`gradient`/`hessian` carry a
 * raw double per jet/tape component and can only answer exactly by routing the
 * pass through the Taylor tower, which is the compiler's one exact AD carrier
 * (eshkol_taylor_alloc_exact). The tower is seeded exact for exactly the points
 * this predicate accepts -- eshkol_taylor_seed_tagged makes the same
 * tagged_is_exact_number() decision -- so the two agree by construction: the
 * codegen never routes a pass to the exact tier that the seeder would then
 * demote to COEFF_F64.
 *
 * Deliberately broader than eshkol_ad_point_is_exact_scalar (which answers only
 * about HEAP exact scalars, for entry classifications that already enumerate
 * INT64) and deliberately narrower than eshkol_ad_point_is_scalar: a DOUBLE is
 * inexact, and a DUAL_NUMBER or a Taylor tower means an enclosing
 * differentiation is already live, which the exact tier must decline. */
int32_t eshkol_ad_point_is_exact_number(const eshkol_tagged_value_t* v) {
    if (!v) return 0;
    return tagged_is_exact_number(v) ? 1 : 0;
}

/* Raise-on-refusal wrapper: the shape the codegen calls, so an AD entry point
 * never has to branch on `ok` in IR. `what` names the operator for the
 * diagnostic (e.g. "derivative", "gradient"). */
extern void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

double eshkol_ad_point_to_double(const eshkol_tagged_value_t* v, const char* what) {
    int32_t ok = 0;
    double d = eshkol_ad_seed_to_double(v, &ok);
    if (!ok) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                             "%s: evaluation point is not a number "
                             "(tagged type %u)",
                             what ? what : "autodiff",
                             v ? (unsigned)(v->type & 0x0F) : 0u);
        return 0.0;    /* not reached */
    }
    return d;
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

#define ESH_TAYLOR_STACKN 64

static void tr_relu(double* s, const double* u, int n) {
    double sign = u[0] > 0.0 ? 1.0 : 0.0;
    for (int k = 0; k < n; k++) s[k] = sign * u[k];
}

static void tr_sigmoid(double* s, const double* u, int n, arena_t* arena) {
    double nb[ESH_TAYLOR_STACKN], eb[ESH_TAYLOR_STACKN], db[ESH_TAYLOR_STACKN];
    double *nu = nb, *ex = eb, *den = db;
    if (n > ESH_TAYLOR_STACKN) {
        nu = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        ex = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        den = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
    }
    if (u[0] >= 0.0) {
        /* sigma(u) = 1/(1+exp(-u)); this branch never overflows at +inf. */
        tr_neg(nu, u, n);
        tr_exp(ex, nu, n);
        den[0] = 1.0 + ex[0];
        for (int k = 1; k < n; k++) den[k] = ex[k];
        s[0] = 1.0 / den[0];
        for (int k = 1; k < n; k++) {
            double acc = 0.0;
            for (int j = 1; j <= k; j++) acc = fma(-den[j], s[k-j], acc);
            s[k] = acc / den[0];
        }
    } else {
        /* sigma(u) = exp(u)/(1+exp(u)); this branch never overflows at -inf. */
        tr_exp(ex, u, n);
        den[0] = 1.0 + ex[0];
        for (int k = 1; k < n; k++) den[k] = ex[k];
        tr_div(s, ex, den, n);
    }
}

static void tr_tanh(double* s, const double* u, int n, arena_t* arena) {
    double ub[ESH_TAYLOR_STACKN], sb[ESH_TAYLOR_STACKN];
    double* twice = ub;
    double* sig = sb;
    if (n > ESH_TAYLOR_STACKN) {
        twice = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        sig = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
    }
    if (!twice || !sig) return;
    for (int k = 0; k < n; k++) twice[k] = 2.0 * u[k];
    tr_sigmoid(sig, twice, n, arena);
    s[0] = 2.0 * sig[0] - 1.0;
    for (int k = 1; k < n; k++) s[k] = 2.0 * sig[k];
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

static void ddual_relu(double* sv, double* st, const double* uv, const double* ut, int n) {
    tr_relu(sv, uv, n);
    double sign = uv[0] > 0.0 ? 1.0 : 0.0;
    for (int k = 0; k < n; k++) st[k] = sign * ut[k];
}

static void ddual_sigmoid(double* sv, double* st, const double* uv, const double* ut,
                          int n, arena_t* arena) {
    double nb[ESH_TAYLOR_STACKN], ntb[ESH_TAYLOR_STACKN];
    double eb[ESH_TAYLOR_STACKN], etb[ESH_TAYLOR_STACKN];
    double db[ESH_TAYLOR_STACKN], dtb[ESH_TAYLOR_STACKN];
    double *nu=nb, *nut=ntb, *ex=eb, *ext=etb, *den=db, *dent=dtb;
    if (n > ESH_TAYLOR_STACKN) {
        nu=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        nut=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        ex=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        ext=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        den=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        dent=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
    }
    double one[ESH_TAYLOR_STACKN], onet[ESH_TAYLOR_STACKN];
    double *ov=one,*ot=onet;
    if (n > ESH_TAYLOR_STACKN) {
        ov=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
        ot=(double*)arena_allocate(arena,(size_t)n*sizeof(double));
    }
    ov[0]=1.0; ot[0]=0.0;
    for (int k=1;k<n;k++) { ov[k]=0.0; ot[k]=0.0; }
    if (uv[0] >= 0.0) {
        tr_neg(nu, uv, n); tr_neg(nut, ut, n);
        ddual_exp(ex, ext, nu, nut, n);
        den[0] = 1.0 + ex[0]; dent[0] = ext[0];
        for (int k = 1; k < n; k++) { den[k] = ex[k]; dent[k] = ext[k]; }
        ddual_div(sv, st, ov, ot, den, dent, n);
    } else {
        ddual_exp(ex, ext, uv, ut, n);
        den[0] = 1.0 + ex[0]; dent[0] = ext[0];
        for (int k = 1; k < n; k++) { den[k] = ex[k]; dent[k] = ext[k]; }
        ddual_div(sv, st, ex, ext, den, dent, n);
    }
}

static void ddual_tanh(double* sv, double* st, const double* uv, const double* ut,
                       int n, arena_t* arena) {
    double ub[ESH_TAYLOR_STACKN], utb[ESH_TAYLOR_STACKN];
    double sb[ESH_TAYLOR_STACKN], stb[ESH_TAYLOR_STACKN];
    double* twice = ub; double* twicet = utb;
    double* sig = sb; double* sigt = stb;
    if (n > ESH_TAYLOR_STACKN) {
        twice = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        twicet = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        sig = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        sigt = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
    }
    if (!twice || !twicet || !sig || !sigt) return;
    for (int k = 0; k < n; k++) { twice[k] = 2.0 * uv[k]; twicet[k] = 2.0 * ut[k]; }
    ddual_sigmoid(sig, sigt, twice, twicet, n, arena);
    sv[0] = 2.0 * sig[0] - 1.0; st[0] = 2.0 * sigt[0];
    for (int k = 1; k < n; k++) { sv[k] = 2.0 * sig[k]; st[k] = 2.0 * sigt[k]; }
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

/* True iff `right`, at `active_epoch`, reduces to a plain constant integer (a
 * scalar, or a foreign-epoch tower's c[0]) -- the exponent shape the exact
 * integer-power recurrence supports. A
 * same-epoch tower (the exponent itself depends on the differentiation
 * variable) is intentionally left unhandled here: pow falls through to the
 * existing (inexact) general recurrence for that rare shape. */
static int exact_pow_exponent_as_int(const eshkol_tagged_value_t* right,
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
        *out = c0.data.int_val;
        return 1;
    }
    if (tagged_is_bignum(&c0)) {
        eshkol_bignum_t* bn = (eshkol_bignum_t*)(uintptr_t)c0.data.ptr_val;
        int64_t v;
        if (!eshkol_bignum_fits_int64(bn, &v)) return 0;
        *out = v;
        return 1;
    }
    return 0; /* rational (non-integer) exponent, or inexact -- not integer-power */
}

/* s = u^p for a compile-time-unknown but RUNTIME-constant integer p, via
 * exact binary exponentiation (repeated series
 * multiplication) -- keeps monomial/polynomial derivatives exact without
 * the general real-exponent recurrence's division (which would force the
 * F64 tier for every pow, even integer ones). */
static void taylor_pow_exact(arena_t* arena, const eshkol_tagged_value_t* left,
                             uint32_t order_k, uint32_t epoch, int64_t p,
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

    uint64_t e = p < 0 ? (uint64_t)(-(p + 1)) + 1u : (uint64_t)p;
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
    if (p < 0) {
        /* 1/(u^|p|), with the same exact division recurrence used by the
         * ordinary exact tower.  The caller has already established that the
         * base point is nonzero, so this remains an exact rational series. */
        for (int k = 0; k < n; k++) base[k] = eshkol_make_int64(0, true);
        base[0] = eshkol_make_int64(1, true);
        tre_div(arena, tmp, base, acc, n);
        for (int k = 0; k < n; k++) acc[k] = tmp[k];
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
 *      epoch (P6/ESH-0191): add/sub/mul/div and integer pow
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
     * integer pow stay exact when BOTH operands are exact at
     * this epoch (a same-epoch exact tower, an exact scalar, or a
     * foreign-epoch exact tower lifted as a constant, section 5a); any
     * other operand/op shape falls through to the unchanged F64 kernel
     * below (a real/non-constant pow, or any inexact operand -- the R7RS
     * contagion rule, design section 9). */
    if (operand_is_exact_for_taylor(left, epoch) && operand_is_exact_for_taylor(right, epoch)) {
        if (op == ESH_TAYLOR_OP_add || op == ESH_TAYLOR_OP_sub ||
            op == ESH_TAYLOR_OP_mul || op == ESH_TAYLOR_OP_div) {
            taylor_binary_exact(arena, left, right, op, order_k, epoch, result);
            return;
        }
        if (op == ESH_TAYLOR_OP_pow) {
            int64_t p;
            if (exact_pow_exponent_as_int(right, epoch, &p)) {
                eshkol_tagged_value_t base0;
                normalise_operand_exact(left, epoch, &base0, 1);
                /* zero is valid only for non-negative powers; a negative
                 * power remains on the general path at the singularity. */
                if (tagged_is_exact_number(&base0) &&
                    (p >= 0 || !tagged_is_zero_exact(&base0))) {
                    taylor_pow_exact(arena, left, order_k, epoch, p, result);
                    return;
                }
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
                /* An exact zero crossing can arrive through the legacy JET
                 * boundary as a denormal residue. It is still the documented
                 * zero-subgradient point, not a positive smooth-side point. */
                double sgn = fabs(uv[0]) < DBL_MIN ? 0.0
                             : (uv[0] < 0.0 ? -1.0 : 1.0);
                ov[0] = fabs(uv[0]); ot[0] = sgn * ut[0];
                for (int k = 1; k < n; k++) { ov[k] = sgn * uv[k]; ot[k] = sgn * ut[k]; }
                break;
            }
            case ESH_TAYLOR_UOP_sinh:
            case ESH_TAYLOR_UOP_cosh:
            case ESH_TAYLOR_UOP_tanh: {
                if (op == ESH_TAYLOR_UOP_tanh) {
                    ddual_tanh(ov, ot, uv, ut, n, arena);
                    break;
                }
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
            case ESH_TAYLOR_UOP_relu: ddual_relu(ov, ot, uv, ut, n); break;
            case ESH_TAYLOR_UOP_sigmoid: ddual_sigmoid(ov, ot, uv, ut, n, arena); break;
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
    if (t && taylor_is_exact(t) && (op == ESH_TAYLOR_UOP_neg ||
                                    op == ESH_TAYLOR_UOP_abs ||
                                    op == ESH_TAYLOR_UOP_relu)) {
        const eshkol_tagged_value_t* u = taylor_exact_c_const(t);
        eshkol_tagged_value_t* s = alloc_exact_series(arena, n);
        if (!s) { *result = eshkol_make_double(0.0); return; }
        if (op == ESH_TAYLOR_UOP_neg) {
            eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
            for (int k = 0; k < n; k++) s[k] = exact_sub(arena, zero, u[k]);
        } else { /* abs/relu: sign is fixed by c[0]'s sign (a truncated series
                  * around x0 -- the classic |x| kink at x0=0 is a pre-existing,
                  * documented limitation of this recurrence, unchanged from
                  * the F64 tr_* version above). */
            int neg0 = tagged_is_negative_exact(&u[0]);
            int zero0 = tagged_is_zero_exact(&u[0]);
            int active = op == ESH_TAYLOR_UOP_relu;
            eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
            for (int k = 0; k < n; k++)
                s[k] = active ? ((neg0 || zero0) ? zero : u[k])
                              : (zero0 ? zero : (neg0 ? exact_sub(arena, zero, u[k]) : u[k]));
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
            double sgn = (u[0] < 0.0) ? -1.0 : (u[0] > 0.0 ? 1.0 : 0.0);
            out->c[0] = fabs(u[0]);
            for (int k = 1; k < n; k++) out->c[k] = sgn * u[k];
            break;
        }
        case ESH_TAYLOR_UOP_relu: tr_relu(out->c, u, n); break;
        case ESH_TAYLOR_UOP_sigmoid: tr_sigmoid(out->c, u, n, arena); break;
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
        case ESH_TAYLOR_UOP_tanh: tr_tanh(out->c, u, n, arena); break;
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

/* ----------------------------------------------------------------------- */
/* ESH-0402: nested-AD carrier composition (SW-03 / SW-04)                  */
/* ----------------------------------------------------------------------- */
/*
 * Eshkol carries forward-mode derivatives in TWO representations: the 8-jet
 * (three independent FIRST-order perturbations e1/e2/ep, used by `derivative`
 * / `gradient` / `hessian`) and the heap Taylor tower (ONE perturbation to
 * arbitrary order, used by `derivative-n` / `taylor`). Until this change the
 * boundary between them was lossy in BOTH directions:
 *
 *   - eshkol_taylor_seed_tagged() read only the SCALAR value of its point
 *     (`tagged_scalar_value`, i.e. c[0] of an outer tower or the primal of an
 *     outer jet) and seeded a tangent-free tower, so an enclosing pass's
 *     perturbation was dropped at the seed;
 *   - the tower's extraction returned a bare double, which an enclosing pass
 *     then read as "no dependence".
 *
 * The result was a SILENT ZERO for every composition involving `derivative-n`
 * or `taylor` (ledger SW-03/SW-04):
 *
 *     (derivative   (lambda (y) (derivative-n f y 1)) 2.0)     => 0
 *     (derivative-n (lambda (y) (derivative   f y))   2.0 1)   => 0
 *     (derivative-n (lambda (y) (derivative-n f y 1)) 2.0 1)   => 0
 *     (derivative-n (derivative f) 2.0 1)                      => 0
 *
 * while the jet-over-jet spellings of the same mathematics answered correctly.
 *
 * The fix does NOT need a second full series (the deferred "jets of jets"
 * work). A tower already carries a parallel FIRST-ORDER companion series --
 * the P5 seed tangent (ESH_TAYLOR_TANGENT_FLAG) -- and every recurrence in
 * this file already propagates it (see eshkol_taylor_binary_tagged's dual
 * tier and the ddual_* kernels). A first-order companion is exactly what one
 * extra `derivative`-class pass needs. So the two carriers compose whenever
 * ONE of the two passes is first order, by putting that pass on the tangent
 * dimension:
 *
 *   RIDE      the INNER pass is first order: it rides the OUTER tower's
 *             tangent. The seed keeps the outer's value series and epoch
 *             untouched and sets tangent = {1,0,...}; after the body runs, the
 *             tangent series IS d(body)/d(inner argument) as a series in the
 *             outer perturbation, so extraction just promotes the tangent
 *             series to the value series of a tower at the OUTER epoch.
 *             The outer pass then reads it exactly as it reads any tower.
 *             Outer order is unrestricted.
 *
 *   CARRY     the OUTER pass is first order: it rides the INNER tower's
 *             tangent. The seed builds the ordinary fresh-epoch tower
 *             {x0,1,0,...} and additionally sets tangent[0] = the outer's
 *             first-order coefficient, so the tower's tangent series tracks
 *             d(c[k])/d(outer perturbation). Extraction reports
 *             d(f^(k))/d(outer) = k!*tangent[k] back to the outer carrier.
 *             Inner order is unrestricted.
 *
 * When BOTH passes are order >= 2 the composition genuinely exceeds what one
 * value series plus one first-order companion can represent; that case
 * returns ESH_AD_NEST_UNSUPPORTED and the caller raises a LOUD error rather
 * than answering zero.
 *
 * The route is returned PACKED so a single i32 threads from the seed site to
 * the extraction site through codegen: low byte = route, bits 8..23 = the
 * outer tower's epoch (needed only by CARRY_TWR, which must hand its result
 * back in the outer epoch).
 */

/** @brief Pack a route code and an outer epoch into the i32 codegen threads from seed to extract. */
static inline int32_t nest_pack(int route, uint32_t epoch) {
    return (int32_t)((uint32_t)(route & 0xFF) | ((epoch & 0xFFFFu) << 8));
}

/* Copy a tower's value series into a raw double buffer, converting an EXACT
 * (COEFF_RATIONAL) tower coefficient-by-coefficient. Nested composition is
 * always inexact: the tangent companion is a double series, so an exact outer
 * point cannot stay exact through a nested pass. That is a documented
 * narrowing of exactness, not of correctness -- and it replaces a zero. */
static void nest_copy_values(const esh_taylor_t* t, double* dst, uint32_t n) {
    uint32_t m = t->order_k + 1u;
    if (m > n) m = n;
    if (taylor_is_exact(t)) {
        const eshkol_tagged_value_t* c = taylor_exact_c_const(t);
        for (uint32_t i = 0; i < m; i++) dst[i] = tagged_any_to_double(&c[i]);
    } else {
        memcpy(dst, t->c, (size_t)m * sizeof(double));
    }
}

/** @brief Read coefficient `i` of a tower as a double, whatever its coefficient type. */
static double nest_coeff(const esh_taylor_t* t, uint32_t i) {
    if (i > t->order_k) return 0.0;
    if (taylor_is_exact(t)) return tagged_any_to_double(&taylor_exact_c_const(t)[i]);
    return t->c[i];
}

/**
 * @brief Decide and perform the seeding for a differentiation pass whose
 *        evaluation point is ALREADY an enclosing AD pass's carrier.
 *
 * Returns ESH_AD_NEST_NONE (0) and writes nothing when the point is an
 * ordinary value -- the caller then seeds exactly as it did before, so every
 * non-nested pass is byte-for-byte unchanged.
 *
 * @param arena       allocation arena (NULL -> global).
 * @param point       the evaluation point as handed to this pass.
 * @param order_k     this pass's own order (1 for the 8-jet path).
 * @param pert_level  the runtime forward perturbation level on entry.
 * @param tower_pass  1 when the caller is the Taylor-tower arm, 0 for the jet arm.
 * @param out         receives the seeded carrier when the route is not NONE.
 * @return the packed route (see nest_pack), or ESH_AD_NEST_UNSUPPORTED.
 */
int32_t eshkol_ad_nested_seed(arena_t* arena, const eshkol_tagged_value_t* point,
                              int32_t order_k, int64_t pert_level, int32_t tower_pass,
                              eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (!point || !out) return ESH_AD_NEST_NONE;
    if (order_k < 0) order_k = 0;

    const esh_taylor_t* outer = tagged_as_taylor(point);
    if (outer) {
        uint32_t oep = ESH_TAYLOR_GET_EPOCH(outer->flags);

        /* RIDE: this pass is first order, so it becomes the outer tower's
         * tangent dimension. Value series and epoch are the outer's, so the
         * outer's own arithmetic keeps working on the result unchanged. */
        if (order_k <= 1) {
            esh_taylor_t* t = eshkol_taylor_alloc(arena, outer->order_k,
                ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, oep) | ESH_TAYLOR_TANGENT_FLAG);
            if (!t) return ESH_AD_NEST_UNSUPPORTED;
            nest_copy_values(outer, t->c, outer->order_k + 1u);
            taylor_tan(t)[0] = 1.0;
            *out = taylor_to_tagged(t);
            return nest_pack(ESH_AD_NEST_RIDE, oep);
        }

        /* CARRY_TWR: this pass needs its own order-k series, so the OUTER --
         * which must then be first order -- rides this tower's tangent. */
        if (outer->order_k == 1) {
            uint32_t epoch = eshkol_taylor_next_epoch();
            esh_taylor_t* t = eshkol_taylor_alloc(arena, (uint32_t)order_k,
                ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch) | ESH_TAYLOR_TANGENT_FLAG);
            if (!t) return ESH_AD_NEST_UNSUPPORTED;
            t->c[0] = nest_coeff(outer, 0);
            t->c[1] = 1.0;
            taylor_tan(t)[0] = nest_coeff(outer, 1);
            *out = taylor_to_tagged(t);
            return nest_pack(ESH_AD_NEST_CARRY_TWR, oep);
        }

        /* Both passes want order >= 2: beyond one value series plus one
         * first-order companion. The caller raises. */
        return ESH_AD_NEST_UNSUPPORTED;
    }

    /* An outer 8-jet perturbation. Only the tower arm needs help here: the jet
     * arm already nests through the e1/e2/ep slots, and case A of the nesting
     * matrix (jet over jet) has always been correct. */
    if (!tower_pass) return ESH_AD_NEST_NONE;
    if ((uint8_t)(point->type & 0x0F) != ESHKOL_VALUE_DUAL_NUMBER || !point->data.ptr_val)
        return ESH_AD_NEST_NONE;
    if (pert_level < 1) return ESH_AD_NEST_NONE;

    const double* d = (const double*)(uintptr_t)point->data.ptr_val;
    /* The enclosing pass seeded the slot for ITS level, and the level was
     * pushed on the way in, so the live perturbation is slot(pert_level - 1):
     * level 1 -> e1. Deeper jet nesting writes e2/ep, which the tower's single
     * tangent companion cannot be handed back through (the extraction site
     * reports its dependence in e1); those raise rather than answer zero. */
    if (pert_level != 1) return ESH_AD_NEST_UNSUPPORTED;
    if (d[1] == 0.0) return ESH_AD_NEST_NONE;   /* no live dependence to carry */

    uint32_t epoch = eshkol_taylor_next_epoch();
    esh_taylor_t* t = eshkol_taylor_alloc(arena, (uint32_t)order_k,
        ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, epoch) | ESH_TAYLOR_TANGENT_FLAG);
    if (!t) return ESH_AD_NEST_UNSUPPORTED;
    t->c[0] = d[0];
    if (order_k >= 1) t->c[1] = 1.0;
    taylor_tan(t)[0] = d[1];
    *out = taylor_to_tagged(t);
    return nest_pack(ESH_AD_NEST_CARRY_JET, 0u);
}

/**
 * @brief Extraction counterpart of eshkol_ad_nested_seed for the routes whose
 *        result cannot be produced by the caller's ordinary extraction.
 *
 * Only RIDE and CARRY_TWR reach here; CARRY_JET is handled by the tower arm's
 * existing has-tangent branch, which already hands a jet back to the enclosing
 * forward pass, and NONE by the unchanged code.
 *
 * @param arena        allocation arena (NULL -> global).
 * @param result       the differentiated body's raw return value.
 * @param route_packed the value eshkol_ad_nested_seed returned.
 * @param order_k      this pass's own order.
 * @param out          receives the result in the ENCLOSING pass's carrier.
 */
void eshkol_ad_nested_extract(arena_t* arena, const eshkol_tagged_value_t* result,
                              int32_t route_packed, int32_t order_k,
                              eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (!out) return;
    int route = route_packed & 0xFF;
    uint32_t oep = ((uint32_t)route_packed >> 8) & 0xFFFFu;
    if (order_k < 0) order_k = 0;

    if (route == ESH_AD_NEST_RIDE) {
        /* The tangent series holds d(body)/d(this pass's argument) as a series
         * in the OUTER perturbation. Promote it to a value series so the outer
         * pass reads it as an ordinary tower of its own epoch. A body that did
         * not depend on the argument comes back without a tangent -- its
         * derivative is 0, which a plain scalar states correctly. */
        const esh_taylor_t* r = result ? tagged_as_taylor(result) : NULL;
        const double* rt = (r && ESH_TAYLOR_HAS_TANGENT(r->flags))
                         ? (const double*)(r->c + ((size_t)r->order_k + 1)) : NULL;
        if (!r || !rt) { *out = eshkol_make_double(0.0); return; }
        esh_taylor_t* o = eshkol_taylor_alloc(arena, r->order_k,
            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, ESH_TAYLOR_GET_EPOCH(r->flags)));
        if (!o) { *out = eshkol_make_double(0.0); return; }
        memcpy(o->c, rt, ((size_t)r->order_k + 1) * sizeof(double));
        *out = taylor_to_tagged(o);
        return;
    }

    if (route == ESH_AD_NEST_CARRY_TWR) {
        /* f^(k) and d(f^(k))/d(outer perturbation), handed back as the order-1
         * tower of the OUTER epoch that the enclosing tower pass expects. */
        double v  = eshkol_taylor_extract(result, (uint32_t)order_k);
        double dv = eshkol_taylor_extract_tangent(result, (uint32_t)order_k);
        esh_taylor_t* o = eshkol_taylor_alloc(arena, 1u,
            ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_F64, oep));
        if (!o) { *out = eshkol_make_double(v); return; }
        o->c[0] = v;
        o->c[1] = dv;
        *out = taylor_to_tagged(o);
        return;
    }

    *out = result ? *result : eshkol_make_double(0.0);
}

/**
 * @brief Report a nested differentiation the two AD carriers cannot represent.
 *
 * Called from codegen when eshkol_ad_nested_seed returns
 * ESH_AD_NEST_UNSUPPORTED. Raising is deliberate: the answer this replaces was
 * a silent zero, and a wrong number that looks like a derivative is worse than
 * a stopped program (ledger SW-03/SW-04).
 */
void eshkol_ad_nested_unsupported(int32_t order_k) {
    eshkol_error(
        "unsupported nested differentiation: an order-%d `derivative-n`/`taylor` "
        "pass inside another differentiation of order 2 or higher. Eshkol's "
        "forward carriers compose when at least one of the two passes is first "
        "order; rewrite the inner or outer pass as a first-order `derivative`, "
        "or compute the higher-order term with a single `(derivative-n f x k)`.",
        (int)order_k);
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_ERROR,
        "unsupported nested differentiation (both passes order >= 2)");
    eshkol_raise(exc);
}

/**
 * @brief Report a curried/first-class `gradient` reached with reverse-tape
 *        inputs, which the runtime-closure gradient cannot differentiate.
 *
 * The vector arm of emitRuntimeClosureGradient() reads its point's components
 * as raw doubles. When an ENCLOSING reverse pass (`jacobian`, `gradient`) hands
 * it a point whose components are `ad_node_t*` tape nodes, that bitcast turns a
 * heap pointer into a subnormal ~1e-310, so the inner gradient is evaluated at
 * ~0 rather than at the real point, AND the result carries no edge back to the
 * outer tape. Ledger SW-05 (ESH-0096): `(jacobian (gradient f) (vector 2.0))`
 * answered `#((0))` where `(hessian f (vector 2.0))` answers `#((12))`, and for
 * `f(v) = v0*v0` the bogus element bits were classified as a pointer and
 * dereferenced (SIGBUS).
 *
 * Until the curried route learns the forward-over-reverse composition the
 * direct `hessian` already performs, this raises: the alternative is a wrong
 * derivative with no diagnostic, or a crash.
 */
void eshkol_ad_curried_gradient_unsupported(void) {
    eshkol_error(
        "unsupported nested differentiation: a first-class `gradient` closure "
        "differentiated again by an enclosing reverse-mode pass. Use "
        "`(hessian f point)` for the second derivative of a scalar function, or "
        "`(jacobian (lambda (v) (gradient f v)) point)` is NOT a substitute -- "
        "it is the same unsupported composition. Tracked as ESH-0096.");
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_ERROR,
        "unsupported nested differentiation of a first-class gradient (ESH-0096)");
    eshkol_raise(exc);
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
