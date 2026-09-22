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

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif

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
    const size_t lanes = ESH_TAYLOR_HAS_TANGENT(flags) ? 2u : 1u;
    const size_t exact_lanes = ESH_TAYLOR_TANGENT_IS_EXACT(flags) ? 2u : 0u;
    const size_t bytes_per_coefficient = lanes * sizeof(double) +
        exact_lanes * sizeof(eshkol_tagged_value_t);
    if (ncoeff == 0 || ncoeff > (SIZE_MAX - sizeof(esh_taylor_t)) /
                                  bytes_per_coefficient) {
        eshkol_error("Taylor payload is too large for the object header");
        return NULL;
    }
    size_t nstore = lanes * ncoeff;
    size_t exact_value_size = ESH_TAYLOR_TANGENT_IS_EXACT(flags)
        ? ncoeff * sizeof(eshkol_tagged_value_t) : 0u;
    size_t tangent_exact_size = exact_value_size;
    size_t data_size = sizeof(esh_taylor_t) + ncoeff * bytes_per_coefficient;
    if (!eshkol_object_payload_fits(data_size)) {
        eshkol_error("Taylor payload exceeds uint32_t header limit");
        return NULL;
    }
    if (data_size > SIZE_MAX - sizeof(eshkol_object_header_t) - 15) {
        eshkol_error("Taylor allocation size overflow");
        return NULL;
    }
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
    t->reserved_epochs[0] = t->reserved_epochs[1] = t->reserved_epochs[2] = 0;
    t->reserved1 = 0;
    t->exact_c = exact_value_size
        ? (eshkol_tagged_value_t*)(void*)(t->c + nstore) : NULL;
    memset(t->c, 0, nstore * sizeof(double) + exact_value_size +
                     tangent_exact_size);
    if (t->exact_c) {
        eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
        for (size_t i = 0; i < 2u * ncoeff; ++i)
            t->exact_c[i] = zero;
    }
    return t;
}

/* Pointer to a tower's tangent (seed-derivative) coefficient array, or NULL if
 * the tower does not carry one. The tangent half follows the value half. */
static inline double* taylor_tan(esh_taylor_t* t) {
    if (!t || !ESH_TAYLOR_HAS_TANGENT(t->flags)) return NULL;
    return t->c + ((size_t)t->order_k + 1);
}

static inline eshkol_tagged_value_t* taylor_tan_exact(esh_taylor_t* t) {
    if (!t || !ESH_TAYLOR_HAS_TANGENT(t->flags) ||
        !ESH_TAYLOR_TANGENT_IS_EXACT(t->flags)) return NULL;
    const size_t ncoeff = (size_t)t->order_k + 1u;
    return t->exact_c ? t->exact_c + ncoeff : NULL;
}

static inline const eshkol_tagged_value_t* taylor_tan_exact_const(
    const esh_taylor_t* t) {
    return taylor_tan_exact((esh_taylor_t*)(uintptr_t)t);
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
    if (ncoeff > (SIZE_MAX - sizeof(esh_taylor_t)) / sizeof(eshkol_tagged_value_t)) {
        eshkol_error("Exact Taylor payload is too large for the object header");
        return NULL;
    }
    size_t data_size = sizeof(esh_taylor_t) + ncoeff * sizeof(eshkol_tagged_value_t);
    if (!eshkol_object_payload_fits(data_size)) {
        eshkol_error("Exact Taylor payload exceeds uint32_t header limit");
        return NULL;
    }
    if (data_size > SIZE_MAX - sizeof(eshkol_object_header_t) - 15) {
        eshkol_error("Exact Taylor allocation size overflow");
        return NULL;
    }
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
    t->reserved_epochs[0] = t->reserved_epochs[1] = t->reserved_epochs[2] = 0;
    t->reserved1 = 0;
    t->exact_c = (eshkol_tagged_value_t*)(void*)t->c;

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
    return t->exact_c ? t->exact_c
                      : (eshkol_tagged_value_t*)(void*)t->c;
}
/** @brief Const-qualified counterpart of taylor_exact_c(): reinterpret an EXACT tower's coefficient storage as tagged values. */
static inline const eshkol_tagged_value_t* taylor_exact_c_const(const esh_taylor_t* t) {
    return t->exact_c ? t->exact_c
                      : (const eshkol_tagged_value_t*)(const void*)t->c;
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

static int tagged_exact_sign(const eshkol_tagged_value_t* v) {
    return tagged_is_exact_number(v)
        ? (tagged_is_zero_exact(v) ? 0 : (tagged_is_negative_exact(v) ? -1 : 1))
        : 0;
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
    /* ADR-0027: a level carrier's primal is its c[0], itself any number. */
    while (t && (t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_CARRIER) {
        tv = &((const eshkol_tagged_value_t*)(const void*)t->c)[0];
        t = tagged_as_taylor(tv);
    }
    if (!t && (uint8_t)(tv->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && tv->data.ptr_val)
        return ((const double*)(uintptr_t)tv->data.ptr_val)[0];
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
    /* ADR-0027: a level carrier orders by its primal, read recursively. */
    while (lt && (lt->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_CARRIER) {
        left_c0 = ((const eshkol_tagged_value_t*)(const void*)lt->c)[0];
        l = &left_c0;
        lt = tagged_as_taylor(l);
    }
    while (rt && (rt->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_CARRIER) {
        right_c0 = ((const eshkol_tagged_value_t*)(const void*)rt->c)[0];
        r = &right_c0;
        rt = tagged_as_taylor(r);
    }
    if (lt) {
        if (taylor_is_exact(lt)) left_c0 = taylor_exact_c_const(lt)[0];
        else left_c0 = eshkol_make_double(lt->c[0]);
        l = &left_c0;
    } else if ((l->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && l->data.ptr_val) {
        left_c0 = eshkol_make_double(((const double*)(uintptr_t)l->data.ptr_val)[0]);
        l = &left_c0;
    }
    if (rt) {
        if (taylor_is_exact(rt)) right_c0 = taylor_exact_c_const(rt)[0];
        else right_c0 = eshkol_make_double(rt->c[0]);
        r = &right_c0;
    } else if ((r->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && r->data.ptr_val) {
        right_c0 = eshkol_make_double(((const double*)(uintptr_t)r->data.ptr_val)[0]);
        r = &right_c0;
    }

    /* If either side is exact, compare against the exact IEEE-754 rational
     * represented by the other side.  Comparing an exact tiny rational or a
     * bignum against the rounded double is not an ordering operation: distinct
     * values can collapse to the same double and select the wrong Taylor
     * branch.  Non-finite doubles have no exact rational representation and
     * retain the ordinary IEEE comparison below. */
    if (tagged_is_exact_number(l) || tagged_is_exact_number(r)) {
        eshkol_tagged_value_t le = *l;
        eshkol_tagged_value_t re = *r;
        int finite = 1;
        if ((le.type & 0x0F) == ESHKOL_VALUE_DOUBLE) {
            if (!isfinite(le.data.double_val)) finite = 0;
            else eshkol_double_to_exact_tagged(arena, le.data.double_val, &le);
        }
        if ((re.type & 0x0F) == ESHKOL_VALUE_DOUBLE) {
            if (!isfinite(re.data.double_val)) finite = 0;
            else eshkol_double_to_exact_tagged(arena, re.data.double_val, &re);
        }
        if (finite && tagged_is_exact_number(&le) && tagged_is_exact_number(&re)) {
            eshkol_tagged_value_t result;
            eshkol_rational_compare_tagged_ptr(arena, &le, &re, op, &result);
            return (int32_t)result.data.int_val;
        }
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

static uint32_t taylor_dual_primal_sign(arena_t* arena,
                                        const eshkol_tagged_value_t* left,
                                        const eshkol_tagged_value_t* right,
                                        int op) {
    eshkol_tagged_value_t ltmp, rtmp;
    const eshkol_tagged_value_t* l = left;
    const eshkol_tagged_value_t* r = right;
    esh_taylor_t* lt = tagged_as_taylor(left);
    esh_taylor_t* rt = tagged_as_taylor(right);
    if (lt && taylor_is_exact(lt)) l = &taylor_exact_c_const(lt)[0];
    if (rt && taylor_is_exact(rt)) r = &taylor_exact_c_const(rt)[0];
    if (!tagged_is_exact_number(l)) {
        double d;
        if ((l->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && l->data.ptr_val)
            d = ((const double*)(uintptr_t)l->data.ptr_val)[0];
        else if ((l->type & 0x0F) == ESHKOL_VALUE_CALLABLE && l->data.ptr_val)
            d = ((const ad_node_t*)(uintptr_t)l->data.ptr_val)->value;
        else d = tagged_scalar_value(l);
        if (!isfinite(d)) return 0;
        eshkol_double_to_exact_tagged(arena, d, &ltmp);
        l = &ltmp;
    }
    if (!tagged_is_exact_number(r)) {
        double d;
        if ((r->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && r->data.ptr_val)
            d = ((const double*)(uintptr_t)r->data.ptr_val)[0];
        else if ((r->type & 0x0F) == ESHKOL_VALUE_CALLABLE && r->data.ptr_val)
            d = ((const ad_node_t*)(uintptr_t)r->data.ptr_val)->value;
        else d = tagged_scalar_value(r);
        if (!isfinite(d)) return 0;
        eshkol_double_to_exact_tagged(arena, d, &rtmp);
        r = &rtmp;
    }
    if (!tagged_is_exact_number(l) || !tagged_is_exact_number(r)) return 0;
    if (op != ESH_TAYLOR_OP_add && op != ESH_TAYLOR_OP_sub &&
        op != ESH_TAYLOR_OP_mul && op != ESH_TAYLOR_OP_div)
        return 0u;
    eshkol_tagged_value_t value;
    value = (op == ESH_TAYLOR_OP_add) ? exact_add(arena, *l, *r) :
            (op == ESH_TAYLOR_OP_sub) ? exact_sub(arena, *l, *r) :
            (op == ESH_TAYLOR_OP_mul) ? exact_mul(arena, *l, *r) :
                                         exact_div(arena, *l, *r);
    int sign = tagged_exact_sign(&value);
    return sign < 0 ? ESH_TAYLOR_PRIMAL_NEGATIVE_FLAG
         : sign > 0 ? ESH_TAYLOR_PRIMAL_POSITIVE_FLAG : 0u;
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

/* SW-222: a perturbation coefficient that is exactly zero is structurally
 * absent and contributes nothing to a product, even against an infinite
 * factor. IEEE 0 * inf = NaN would otherwise poison the whole series of a
 * carrier that meets a pole: the series of 1/x at 0 is (inf, -inf, inf, ...)
 * and came back (inf, -inf, NaN, ...). Only a PERTURBATION coefficient (a
 * tower's c_k with k >= 1, a jet's non-primal component, a seed tangent)
 * annihilates; a primal that is zero is a value, so 0 * inf there stays NaN
 * -- the derivative of x * (1/x) at 0 is indeterminate and says so.
 * zfma(p, x, c) = c + p x, where `p` is the perturbation factor. The code
 * generator's jets (pertMul) and the compile-time-K emitter (fma3) apply the
 * same rule. */
static inline double zfma(double p, double x, double c) {
    return p == 0.0 ? c : fma(p, x, c);
}
/* The same for a product whose two factors are coefficients i and j of their
 * series: index 0 is the primal. */
static inline double zfma_ij(double a, int i, double b, int j, double c) {
    return ((i > 0 && a == 0.0) || (j > 0 && b == 0.0)) ? c : fma(a, b, c);
}

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
        if (k == 0) { s[0] = u[0] * w[0]; continue; }   /* the primal: IEEE */
        double acc = 0.0;
        for (int j = 0; j <= k; j++) acc = zfma_ij(u[j], j, w[k - j], k - j, acc);
        s[k] = acc;
    }
}

/* s = u / w : s_k = ( u_k - sum_{j=1..k} w_j * s_{k-j} ) / w_0. */
static void tr_div(double* s, const double* u, const double* w, int n) {
    for (int k = 0; k < n; k++) {
        double acc = u[k];
        for (int j = 1; j <= k; j++) acc = zfma(-w[j], s[k - j], acc);
        s[k] = acc / w[0];
    }
}

/* s = exp(u) : s_0 = exp(u_0); s_k = (1/k) sum_{j=1..k} j*u_j*s_{k-j}. */
/* The same recurrence from a given s_0 = exp(u_0): exp2 passes u ln 2 and
 * libm's exp2(u_0) so its primal is the one the plain builtin returns. */
static void tr_exp_seeded(double* s, double s0, const double* u, int n) {
    s[0] = s0;
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++) acc = zfma((double)j * u[j], s[k - j], acc);
        s[k] = acc / (double)k;
    }
}
static void tr_exp(double* s, const double* u, int n) {
    tr_exp_seeded(s, exp(u[0]), u, n);
}

/* s = log(u) : s_0 = log(u_0);
 * s_k = ( u_k - (1/k) sum_{j=1..k-1} j*s_j*u_{k-j} ) / u_0. */
static void tr_log(double* s, const double* u, int n) {
    s[0] = log(u[0]);
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k - 1; j++) acc = zfma((double)j * s[j], u[k - j], acc);
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
            as = zfma(ju, co[k - j], as);
            ac = zfma(ju, so[k - j], ac);
        }
        so[k] =  as / (double)k;
        co[k] = -ac / (double)k;
    }
}

/* s = u^r (constant real exponent r):
 * s_0 = u_0^r; s_k = (1/(k*u_0)) sum_{j=1..k} (j*r - (k-j))*u_j*s_{k-j}. */
/* The same recurrence from a given s_0 = u_0^r: cbrt passes libm's cbrt(u_0),
 * which (unlike pow(u_0, 1/3)) is defined for a negative u_0. */
static void tr_pow_seeded(double* s, double s0, const double* u, double r, int n) {
    s[0] = s0;
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++)
            acc = zfma(((double)j * r - (double)(k - j)) * u[j], s[k - j], acc);
        s[k] = acc / ((double)k * u[0]);
    }
}
static void tr_pow_const(double* s, const double* u, double r, int n) {
    s[0] = pow(u[0], r);
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++)
            acc = zfma(((double)j * r - (double)(k - j)) * u[j], s[k - j], acc);
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
            for (int j = 1; j <= k; j++) acc = zfma(-den[j], s[k-j], acc);
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
/* the integral family (SW-210): asin acos atan asinh acosh atanh log2      */
/* log10 exp2 cbrt, and the piecewise-constant floor ceiling truncate round */
/* ----------------------------------------------------------------------- */
/* One recurrence carries every smooth member. With d = f'(u) as a series,
 * s = f(u) solves ds/dt = d du/dt, so
 *
 *     s_0 = f(u_0),     s_k = (1/k) sum_{j=1..k} j u_j d_{k-j}.
 *
 * For the inverse functions f' is algebraic in u (1/(1+u^2), (1-u^2)^(-1/2),
 * ...), so d comes from the mul/div/power recurrences; for log_b it is
 * 1/(u ln b). exp2 and cbrt have d in terms of s itself (ln 2 s, s/(3u)) and
 * use the exp and power recurrences seeded with the libm primal. The rounding
 * functions are constant between their jumps: s_0 = f(u_0), the rest 0.
 * The same algebra runs over tagged coefficients in level_ext_unary(), so an
 * exact input keeps every coefficient that is rational exact. */

/** @brief True iff `op` is a member of the integral family above. */
static int tr_is_ext_uop(int op) {
    switch (op) {
        case ESH_TAYLOR_UOP_asin: case ESH_TAYLOR_UOP_acos: case ESH_TAYLOR_UOP_atan:
        case ESH_TAYLOR_UOP_asinh: case ESH_TAYLOR_UOP_acosh: case ESH_TAYLOR_UOP_atanh:
        case ESH_TAYLOR_UOP_log2: case ESH_TAYLOR_UOP_log10: case ESH_TAYLOR_UOP_exp2:
        case ESH_TAYLOR_UOP_cbrt: case ESH_TAYLOR_UOP_floor: case ESH_TAYLOR_UOP_ceiling:
        case ESH_TAYLOR_UOP_truncate: case ESH_TAYLOR_UOP_round:
            return 1;
        default:
            return 0;
    }
}

/** @brief The libm primal f(v) of an integral-family op. `round` is R7RS
 *  round-half-to-even, which nearbyint gives in the default rounding mode. */
static double tr_ext_primal(int op, double v) {
    switch (op) {
        case ESH_TAYLOR_UOP_asin:     return asin(v);
        case ESH_TAYLOR_UOP_acos:     return acos(v);
        case ESH_TAYLOR_UOP_atan:     return atan(v);
        case ESH_TAYLOR_UOP_asinh:    return asinh(v);
        case ESH_TAYLOR_UOP_acosh:    return acosh(v);
        case ESH_TAYLOR_UOP_atanh:    return atanh(v);
        case ESH_TAYLOR_UOP_log2:     return log2(v);
        case ESH_TAYLOR_UOP_log10:    return log10(v);
        case ESH_TAYLOR_UOP_exp2:     return exp2(v);
        case ESH_TAYLOR_UOP_cbrt:     return cbrt(v);
        case ESH_TAYLOR_UOP_floor:    return floor(v);
        case ESH_TAYLOR_UOP_ceiling:  return ceil(v);
        case ESH_TAYLOR_UOP_truncate: return trunc(v);
        case ESH_TAYLOR_UOP_round:    return nearbyint(v);
        default:                      return NAN;
    }
}

/* s_k = (1/k) sum_{j=1..k} j u_j d_{k-j}, k >= 1 (s_0 is the caller's). `d`
 * may alias `s`: step k reads only d_0..d_{k-1}. */
static void tr_integrate(double* s, const double* u, const double* d, int n) {
    for (int k = 1; k < n; k++) {
        double acc = 0.0;
        for (int j = 1; j <= k; j++) acc = zfma((double)j * u[j], d[k - j], acc);
        s[k] = acc / (double)k;
    }
}

/* d = c / u: d_k = (c [k = 0] - sum_{j=1..k} u_j d_{k-j}) / u_0. */
static void tr_recip(double* d, double c, const double* u, int n) {
    for (int k = 0; k < n; k++) {
        double acc = k == 0 ? c : 0.0;
        for (int j = 1; j <= k; j++) acc = zfma(-u[j], d[k - j], acc);
        d[k] = acc / u[0];
    }
}

/**
 * @brief s = f(u) and d = f'(u) for an integral-family op, over n doubles.
 *
 * `d` is the series of the derivative, which the value recurrence consumes
 * and the seed-tangent tier reuses (tangent = d * u'). Scratch beyond
 * ESH_TAYLOR_STACKN comes from `ar`. Returns 0 when `op` is not in the family.
 */
static int tr_ext_unary(int op, double* s, double* d, const double* u, int n, arena_t* ar) {
    if (!tr_is_ext_uop(op)) return 0;
    double qb[ESH_TAYLOR_STACKN];
    double* q = qb;
    if (n > ESH_TAYLOR_STACKN) {
        q = (double*)arena_allocate(ar, (size_t)n * sizeof(double));
        if (!q) return 0;
    }
    switch (op) {
        case ESH_TAYLOR_UOP_atan: case ESH_TAYLOR_UOP_atanh:
        case ESH_TAYLOR_UOP_asin: case ESH_TAYLOR_UOP_acos:
        case ESH_TAYLOR_UOP_asinh: case ESH_TAYLOR_UOP_acosh: {
            /* q = 1 + u^2 (atan, asinh), 1 - u^2 (atanh, asin, acos), u^2 - 1 (acosh) */
            int minus = op == ESH_TAYLOR_UOP_atanh || op == ESH_TAYLOR_UOP_asin ||
                        op == ESH_TAYLOR_UOP_acos;
            tr_mul(q, u, u, n);
            if (minus) for (int k = 0; k < n; k++) q[k] = -q[k];
            q[0] += op == ESH_TAYLOR_UOP_acosh ? -1.0 : 1.0;
            if (op == ESH_TAYLOR_UOP_atan || op == ESH_TAYLOR_UOP_atanh) {
                tr_recip(d, 1.0, q, n);
            } else {
                tr_pow_const(d, q, -0.5, n);
                if (op == ESH_TAYLOR_UOP_acos) for (int k = 0; k < n; k++) d[k] = -d[k];
            }
            s[0] = tr_ext_primal(op, u[0]);
            tr_integrate(s, u, d, n);
            return 1;
        }
        case ESH_TAYLOR_UOP_log2: case ESH_TAYLOR_UOP_log10:
            tr_recip(d, op == ESH_TAYLOR_UOP_log2 ? 1.0 / M_LN2 : 1.0 / M_LN10, u, n);
            s[0] = tr_ext_primal(op, u[0]);
            tr_integrate(s, u, d, n);
            return 1;
        case ESH_TAYLOR_UOP_exp2:
            for (int k = 0; k < n; k++) q[k] = M_LN2 * u[k];
            tr_exp_seeded(s, exp2(u[0]), q, n);
            for (int k = 0; k < n; k++) d[k] = M_LN2 * s[k];
            return 1;
        case ESH_TAYLOR_UOP_cbrt: {
            /* d = (1/3) u^(-2/3) by the power recurrence from d_0 =
             * 1/(3 cbrt(u_0)^2), then the integral. The power recurrence on s
             * itself divides s_k by u_0 and is 0/0 at u_0 = 0; on d it is
             * inf/0, the pole the closed form has (SW-222). */
            double c0 = cbrt(u[0]);
            tr_pow_seeded(d, 1.0 / (3.0 * c0 * c0), u, -2.0 / 3.0, n);
            s[0] = c0;
            tr_integrate(s, u, d, n);
            return 1;
        }
        default:                                    /* the rounding functions */
            s[0] = tr_ext_primal(op, u[0]);
            d[0] = 0.0;
            for (int k = 1; k < n; k++) { s[k] = 0.0; d[k] = 0.0; }
            return 1;
    }
}

/**
 * @brief s = atan2(y, x) over n doubles.
 *
 * atan2(y, x) = atan(y/x) + c when |x_0| >= |y_0|, and -atan(x/y) + c
 * otherwise, with c constant (0, +-pi/2 or +-pi) near the point. The constant
 * moves only s_0, which is taken from libm's atan2, so the series is the atan
 * recurrence on the better-conditioned quotient. At the origin the quotient
 * is 0/0 and the series is NaN: atan2 has no derivative there.
 */
static void tr_atan2(double* s, const double* y, const double* x, int n, arena_t* ar) {
    double rb[ESH_TAYLOR_STACKN], db[ESH_TAYLOR_STACKN];
    double *r = rb, *d = db;
    if (n > ESH_TAYLOR_STACKN) {
        r = (double*)arena_allocate(ar, (size_t)n * sizeof(double));
        d = (double*)arena_allocate(ar, (size_t)n * sizeof(double));
        if (!r || !d) return;
    }
    int by_x = fabs(x[0]) >= fabs(y[0]);
    tr_div(r, by_x ? y : x, by_x ? x : y, n);
    tr_ext_unary(ESH_TAYLOR_UOP_atan, s, d, r, n, ar);
    if (!by_x) for (int k = 0; k < n; k++) s[k] = -s[k];
    s[0] = atan2(y[0], x[0]);
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
        for (int j = 0; j <= k; j++) acc = zfma(b[k - j], a[j], acc);   /* b is the tangent */
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
            acc = zfma(ut[j], wv[k - j], acc);
            acc = zfma(wt[k - j], uv[j], acc);
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
            acc = zfma(-wt[j], sv[k - j], acc);
            acc = zfma(-wv[j], st[k - j], acc);
        }
        acc = zfma(-wt[0], sv[k], acc);
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

/* Dual integral-family op: the value by tr_ext_unary(), the tangent by the
 * chain rule f'(u) u' = d * u'. Returns 0 when `op` is not in the family. */
static int ddual_ext(int op, double* sv, double* st, const double* uv, const double* ut,
                     int n, arena_t* arena) {
    double db[ESH_TAYLOR_STACKN];
    double* d = db;
    if (n > ESH_TAYLOR_STACKN) {
        d = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        if (!d) return 0;
    }
    if (!tr_ext_unary(op, sv, d, uv, n, arena)) return 0;
    tr_conv(st, d, ut, n);
    return 1;
}

/* Dual atan2(y, x): the composition of tr_atan2() carried with tangents. */
static void ddual_atan2(double* sv, double* st, const double* yv, const double* yt,
                        const double* xv, const double* xt, int n, arena_t* arena) {
    double rb[ESH_TAYLOR_STACKN], rtb[ESH_TAYLOR_STACKN];
    double *r = rb, *rt = rtb;
    if (n > ESH_TAYLOR_STACKN) {
        r = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        rt = (double*)arena_allocate(arena, (size_t)n * sizeof(double));
        if (!r || !rt) return;
    }
    int by_x = fabs(xv[0]) >= fabs(yv[0]);
    if (by_x) ddual_div(r, rt, yv, yt, xv, xt, n);
    else      ddual_div(r, rt, xv, xt, yv, yt, n);
    ddual_ext(ESH_TAYLOR_UOP_atan, sv, st, r, rt, n, arena);
    if (!by_x) for (int k = 0; k < n; k++) { sv[k] = -sv[k]; st[k] = -st[k]; }
    sv[0] = atan2(yv[0], xv[0]);
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
    if (bt == ESHKOL_VALUE_CALLABLE && tv->data.ptr_val) {
        const eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER((void*)(uintptr_t)tv->data.ptr_val);
        return hdr && hdr->subtype == CALLABLE_SUBTYPE_AD_NODE;
    }
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
        double* tt = taylor_tan(t);
        /* value: same-epoch tower contributes its full series; foreign-epoch
         * (outer/inner level) is lifted to its constant c[0] (§5a). */
        if (ESH_TAYLOR_GET_EPOCH(t->flags) == active_epoch) {
            int m = (int)t->order_k + 1;
            if (m > n) m = n;
            if (taylor_is_exact(t)) {
                const eshkol_tagged_value_t* c = taylor_exact_c_const(t);
                for (int i = 0; i < m; i++) vbuf[i] = tagged_any_to_double(&c[i]);
            } else {
                memcpy(vbuf, t->c, (size_t)m * sizeof(double));
            }
            /* tangent: the companion dimension is orthogonal to the value
             * epoch, so it always combines. */
            if (tt) memcpy(tbuf, tt, (size_t)m * sizeof(double));
        } else {
            /* Another level's tower never reaches this tier (ADR-0027: two
             * levels take the level path); what remains here is the epoch-0
             * lift of a reverse-tape AD node, a constant of this level whose
             * companion is the reverse seed. */
            vbuf[0] = tagged_scalar_value(tv);
            if (tt) tbuf[0] = tt[0];
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
        const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(node);
        if (hdr && hdr->subtype == CALLABLE_SUBTYPE_AD_NODE) {
            vbuf[0] = ((const ad_node_t*)node)->value;
            tbuf[0] = eshkol_ad_seed_flag(node);
            return;
        }
    }
    vbuf[0] = tagged_scalar_value(tv);
}

static int operand_has_exact_tangent(const eshkol_tagged_value_t* tv,
                                     uint32_t active_epoch) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) return 0;
    (void)active_epoch;
    return ESH_TAYLOR_TANGENT_IS_EXACT(t->flags) ? 1 : 0;
}

/* Exact companion-side normalisation.  A foreign exact tower contributes its
 * full opaque series to the orthogonal payload, while the current value pass
 * still sees only its c[0]. */
static void normalise_tangent_exact(const eshkol_tagged_value_t* tv,
                                    uint32_t active_epoch,
                                    eshkol_tagged_value_t* buf, int n,
                                    arena_t* arena) {
    const eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
    for (int i = 0; i < n; ++i) buf[i] = zero;
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (t && ESH_TAYLOR_TANGENT_IS_EXACT(t->flags)) {
        const eshkol_tagged_value_t* tan = taylor_tan_exact_const(t);
        for (int i = 0; i < n && i <= (int)t->order_k; ++i) buf[i] = tan[i];
    }
    (void)active_epoch; (void)arena;
}

static eshkol_tagged_value_t exact_series_op(arena_t* arena,
                                             eshkol_tagged_value_t left,
                                             eshkol_tagged_value_t right,
                                             int op) {
    switch (op) {
        case 0: return exact_add(arena, left, right);
        case 1: return exact_sub(arena, left, right);
        case 2: return exact_mul(arena, left, right);
        default: return exact_div(arena, left, right);
    }
}

static int exact_tangent_binary(arena_t* arena,
                                const eshkol_tagged_value_t* left,
                                const eshkol_tagged_value_t* right,
                                int op, uint32_t order_k, uint32_t epoch,
                                esh_taylor_t* out) {
    const int n = (int)order_k + 1;
    eshkol_tagged_value_t* u = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* w = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* ut = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* wt = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* q = alloc_exact_series(arena, n);
    eshkol_tagged_value_t* qt = alloc_exact_series(arena, n);
    if (!u || !w || !ut || !wt || !q || !qt) return 0;
    normalise_operand_exact(left, epoch, u, n);
    normalise_operand_exact(right, epoch, w, n);
    normalise_tangent_exact(left, epoch, ut, n, arena);
    normalise_tangent_exact(right, epoch, wt, n, arena);
    const eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
    for (int k = 0; k < n; ++k) {
        if (op == ESH_TAYLOR_OP_add || op == ESH_TAYLOR_OP_sub) {
            q[k] = exact_series_op(arena, u[k], w[k],
                                   op == ESH_TAYLOR_OP_add ? 0 : 1);
            qt[k] = exact_series_op(arena, ut[k], wt[k],
                                    op == ESH_TAYLOR_OP_add ? 0 : 1);
            continue;
        }
        if (op == ESH_TAYLOR_OP_mul) {
            q[k] = zero;
            qt[k] = zero;
            for (int i = 0; i <= k; ++i) {
                q[k] = exact_series_op(arena, q[k],
                    exact_series_op(arena, u[i], w[k - i], 2), 0);
                eshkol_tagged_value_t a = exact_series_op(arena, ut[i], w[k - i], 2);
                eshkol_tagged_value_t b = exact_series_op(arena, u[i], wt[k - i], 2);
                qt[k] = exact_series_op(arena, qt[k], a, 0);
                qt[k] = exact_series_op(arena, qt[k], b, 0);
            }
            continue;
        }
        if (op == ESH_TAYLOR_OP_div) {
            q[k] = u[k];
            for (int i = 1; i <= k; ++i) {
                eshkol_tagged_value_t term = exact_series_op(arena, w[i], q[k - i], 2);
                q[k] = exact_series_op(arena, q[k], term, 1);
            }
            q[k] = exact_series_op(arena, q[k], w[0], 3);
            qt[k] = ut[k];
            for (int i = 1; i <= k; ++i) {
                eshkol_tagged_value_t a = exact_series_op(arena, wt[i], q[k - i], 2);
                eshkol_tagged_value_t b = exact_series_op(arena, w[i], qt[k - i], 2);
                qt[k] = exact_series_op(arena, qt[k], a, 1);
                qt[k] = exact_series_op(arena, qt[k], b, 1);
            }
            qt[k] = exact_series_op(arena, qt[k],
                                    exact_series_op(arena, q[k], wt[0], 2), 1);
            qt[k] = exact_series_op(arena, qt[k], w[0], 3);
        } else {
            return 0;
        }
    }
    (void)out;
    (void)zero;
    eshkol_tagged_value_t* target = taylor_tan_exact(out);
    if (!target || !out->exact_c) return 0;
    for (int k = 0; k < n; ++k) {
        target[k] = qt[k];
        out->exact_c[k] = q[k];
    }
    return 1;
}


/* ----------------------------------------------------------------------- */
/* tagged binary / unary dispatch (called from codegen)                     */
/* ----------------------------------------------------------------------- */

/* ======================================================================= */
/* ADR-0027: the recursive Taylor LEVEL carrier                            */
/* ======================================================================= */
/*
 * A pass nested inside another live pass runs as a LEVEL: a truncated series
 *     c[0] + c[1] t_E + ... + c[K] t_E^K
 * in its own perturbation t_E (E = the pass's epoch), whose coefficients are
 * ANY numbers -- exact or inexact scalars, 8-jets of an enclosing
 * `derivative`, classic towers of an enclosing tower pass, or level carriers
 * of enclosing levels (epoch < E). Epochs are drawn from one increasing
 * counter, so a level opened later always has the larger epoch, and a level
 * carrier of epoch E never contains a carrier of epoch >= E.
 *
 * Arithmetic picks the largest epoch among the operands as the active level,
 * reads every other operand as a constant of it (kept whole, never reduced to
 * its primal), and runs the standard recurrences with every coefficient
 * operation dispatched recursively through num_binary()/num_unary(). Depth and
 * order are therefore unbounded and two perturbations meet only when their
 * epochs are equal. See docs/design/adr/0027-recursive-taylor-level-carrier.md.
 */

void eshkol_taylor_binary_tagged(arena_t* arena, const eshkol_tagged_value_t* left,
                                 const eshkol_tagged_value_t* right, int op,
                                 eshkol_tagged_value_t* result);
void eshkol_taylor_unary_tagged(arena_t* arena, const eshkol_tagged_value_t* in,
                                int op, eshkol_tagged_value_t* result);
void eshkol_taylor_lift_ad_node(arena_t* arena, void* node, int32_t order_k,
                                eshkol_tagged_value_t* out);

/** @brief True iff tower `t` is a level carrier (tagged coefficients, ADR-0027). */
static inline int taylor_is_level(const esh_taylor_t* t) {
    return t != NULL &&
           (t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_CARRIER;
}
static inline eshkol_tagged_value_t* level_c(esh_taylor_t* t) {
    return (eshkol_tagged_value_t*)(void*)t->c;
}
static inline const eshkol_tagged_value_t* level_c_const(const esh_taylor_t* t) {
    return (const eshkol_tagged_value_t*)(const void*)t->c;
}


/** @brief Allocate a level carrier of order `order_k` at `epoch`, every coefficient exact 0. */
static esh_taylor_t* level_alloc(arena_t* arena, uint32_t order_k, uint32_t epoch) {
    esh_taylor_t* t = eshkol_taylor_alloc_exact(arena, order_k, epoch);
    if (t) t->flags = ESH_TAYLOR_MK_FLAGS(ESH_TAYLOR_COEFF_CARRIER, epoch);
    return t;
}

static inline eshkol_tagged_value_t num_exact_int(int64_t v) {
    return eshkol_make_int64(v, true);
}
static inline int num_is_exact_zero(const eshkol_tagged_value_t* v) {
    return (uint8_t)(v->type & 0x0F) == ESHKOL_VALUE_INT64 && v->data.int_val == 0;
}
static inline int num_is_jet(const eshkol_tagged_value_t* v) {
    return (uint8_t)(v->type & 0x0F) == ESHKOL_VALUE_DUAL_NUMBER && v->data.ptr_val != 0;
}
static inline int num_is_ad_node(const eshkol_tagged_value_t* v) {
    if ((uint8_t)(v->type & 0x0F) != ESHKOL_VALUE_CALLABLE || !v->data.ptr_val) return 0;
    const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
    return hdr && hdr->subtype == CALLABLE_SUBTYPE_AD_NODE;
}

/* The primal of any number: c[0] taken recursively through towers and jets
 * down to a plain int64/bignum/rational/double. Branches (abs, relu, the
 * comparisons) are decided on it. */
static eshkol_tagged_value_t tagged_primal(const eshkol_tagged_value_t* v) {
    eshkol_tagged_value_t cur = *v;
    for (;;) {
        esh_taylor_t* t = tagged_as_taylor(&cur);
        if (t) {
            if (taylor_is_level(t)) { cur = level_c_const(t)[0]; continue; }
            if (taylor_is_exact(t)) return taylor_exact_c_const(t)[0];
            return eshkol_make_double(t->c[0]);
        }
        if (num_is_jet(&cur))
            return eshkol_make_double(((const double*)(uintptr_t)cur.data.ptr_val)[0]);
        if (num_is_ad_node(&cur))
            return eshkol_make_double(((const ad_node_t*)(uintptr_t)cur.data.ptr_val)->value);
        return cur;
    }
}

/* -1 / 0 / +1: the sign of a number's primal, decided exactly when it is exact. */
static int tagged_primal_sign(const eshkol_tagged_value_t* v) {
    eshkol_tagged_value_t p = tagged_primal(v);
    if (tagged_is_exact_number(&p)) return tagged_exact_sign(&p);
    double d = tagged_any_to_double(&p);
    return d > 0.0 ? 1 : (d < 0.0 ? -1 : 0);
}

/* ── the 8-jet as a coefficient ─────────────────────────────────────────
 * An enclosing first-order `derivative` carries its perturbation in the flat
 * 8-jet (value, e1, e2, e1e2, ep, ep*e1, ep*e2, ep*e1e2). Inside a level it is
 * an ordinary coefficient with its own algebra: component s of a product is
 * the sum over the subsets a of s of x[a]*y[s\a], and a unary f is
 * f(x0 + n) = sum_{k<=3} f^(k)(x0)/k! n^k for the nilpotent part n, with the
 * f^(k)(x0)/k! read off the same double recurrences as the tower. */
typedef struct { double v[8]; } level_jet_t;

static level_jet_t jet_of(const eshkol_tagged_value_t* x) {
    level_jet_t j;
    memset(&j, 0, sizeof(j));
    if (num_is_jet(x)) memcpy(j.v, (const void*)(uintptr_t)x->data.ptr_val, sizeof(j.v));
    else j.v[0] = tagged_any_to_double(x);
    return j;
}
static eshkol_tagged_value_t jet_tagged(arena_t* ar, const level_jet_t* j) {
    eshkol_dual_number_t* d = arena_allocate_dual_number(ar);
    eshkol_tagged_value_t r;
    memset(&r, 0, sizeof(r));
    if (!d) { r = eshkol_make_double(j->v[0]); return r; }
    memcpy((void*)d, j->v, sizeof(j->v));
    r.type = ESHKOL_VALUE_DUAL_NUMBER;
    r.flags = ESHKOL_VALUE_INEXACT_FLAG;
    r.data.ptr_val = (uint64_t)(uintptr_t)d;
    return r;
}
static void jet_mul_raw(double* r, const double* x, const double* y) {
    r[0] = x[0] * y[0];                          /* the primal: IEEE */
    for (int s = 1; s < 8; s++) {
        double acc = 0.0;
        for (int a = 0; a < 8; a++)
            if ((a & s) == a) acc = zfma_ij(x[a], a, y[s ^ a], s ^ a, acc);   /* SW-222 */
        r[s] = acc;
    }
}
/* r = x / y by the division recurrence over the jet (SW-222), the recurrence
 * tr_div runs over a series: r_S = (x_S - sum_{T nonempty subset of S}
 * y_T r_{S\T}) / y_0, with a zero coefficient of y contributing nothing. At a
 * simple pole this is the closed form (d/dx 1/x at 0 is -inf); at a pole of
 * higher order it ends in 0/0. */
static void jet_div_raw(double* r, const double* x, const double* y) {
    for (int s = 0; s < 8; s++) {
        double acc = x[s];
        for (int t = 1; t < 8; t++)
            if ((t & s) == t) acc = zfma_ij(-y[t], t, r[s ^ t], s ^ t, acc);
        r[s] = acc / y[0];
    }
}
/* r = g0 + g1 n + g2 n^2 + g3 n^3 with n = x - x[0]. */
static void jet_compose(double* r, const double* x, const double* g) {
    double n[8], n2[8], n3[8];
    memcpy(n, x, sizeof(n));
    n[0] = 0.0;
    jet_mul_raw(n2, n, n);
    jet_mul_raw(n3, n2, n);
    /* IEEE products: g_k is a precomputed number, and an infinite one times a
     * zero coefficient is indeterminate here; poles enter soundly only
     * through the division recurrence (jet_div_raw). */
    for (int s = 0; s < 8; s++) r[s] = g[1] * n[s] + g[2] * n2[s] + g[3] * n3[s];
    r[0] = g[0];                                 /* n[0] = 0: the primal is g_0 */
}
/* g[k] = f^(k)(x0)/k!, k = 0..3, for the unary op `uop` (or x^r when uop < 0). */
static void jet_unary_coeffs(arena_t* ar, int uop, double x0, double r, double* g) {
    double u[4] = {x0, 1.0, 0.0, 0.0}, a[4], b[4];
    switch (uop) {
        case ESH_TAYLOR_UOP_neg: g[0] = -x0; g[1] = -1.0; g[2] = g[3] = 0.0; return;
        case ESH_TAYLOR_UOP_exp: tr_exp(g, u, 4); return;
        case ESH_TAYLOR_UOP_log: tr_log(g, u, 4); return;
        case ESH_TAYLOR_UOP_sin: tr_sincos(g, a, u, 4); return;
        case ESH_TAYLOR_UOP_cos: tr_sincos(a, g, u, 4); return;
        case ESH_TAYLOR_UOP_tan: tr_sincos(a, b, u, 4); tr_div(g, a, b, 4); return;
        case ESH_TAYLOR_UOP_sqrt: tr_pow_const(g, u, 0.5, 4); return;
        case ESH_TAYLOR_UOP_abs: {
            double s = x0 > 0.0 ? 1.0 : (x0 < 0.0 ? -1.0 : 0.0);
            g[0] = fabs(x0); g[1] = s; g[2] = g[3] = 0.0; return;
        }
        case ESH_TAYLOR_UOP_relu: tr_relu(g, u, 4); return;
        case ESH_TAYLOR_UOP_sigmoid: tr_sigmoid(g, u, 4, ar); return;
        case ESH_TAYLOR_UOP_tanh: tr_tanh(g, u, 4, ar); return;
        case ESH_TAYLOR_UOP_sinh:
        case ESH_TAYLOR_UOP_cosh: {
            double m[4] = {-x0, -1.0, 0.0, 0.0};
            tr_exp(a, u, 4); tr_exp(b, m, 4);
            for (int k = 0; k < 4; k++)
                g[k] = 0.5 * (uop == ESH_TAYLOR_UOP_sinh ? a[k] - b[k] : a[k] + b[k]);
            return;
        }
        default: {
            double d[4];
            if (uop < 0) { tr_pow_const(g, u, r, 4); return; }   /* x^r */
            if (tr_ext_unary(uop, g, d, u, 4, ar)) return;
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                "internal: Taylor unary op %d has no recurrence", uop);
            return;
        }
    }
}
static eshkol_tagged_value_t jet_unary(arena_t* ar, const eshkol_tagged_value_t* x, int uop, double r) {
    level_jet_t j = jet_of(x), out;
    double g[4];
    jet_unary_coeffs(ar, uop, j.v[0], r, g);
    jet_compose(out.v, j.v, g);
    return jet_tagged(ar, &out);
}

/* ── the scalar ring every coefficient lives in ─────────────────────────── */
static eshkol_tagged_value_t num_unary(arena_t* ar, eshkol_tagged_value_t x, int uop);

/* op: ESH_TAYLOR_OP_add/sub/mul/div/pow. */
static eshkol_tagged_value_t num_binary(arena_t* ar, eshkol_tagged_value_t a,
                                        eshkol_tagged_value_t b, int op) {
    /* identities that keep a structural zero from costing an allocation */
    if (op == ESH_TAYLOR_OP_add) {
        if (num_is_exact_zero(&a)) return b;
        if (num_is_exact_zero(&b)) return a;
    } else if (op == ESH_TAYLOR_OP_sub) {
        if (num_is_exact_zero(&b)) return a;
    } else if (op == ESH_TAYLOR_OP_mul) {
        if (num_is_exact_zero(&a) || num_is_exact_zero(&b)) return num_exact_int(0);
        if ((uint8_t)(a.type & 0x0F) == ESHKOL_VALUE_INT64 && a.data.int_val == 1) return b;
        if ((uint8_t)(b.type & 0x0F) == ESHKOL_VALUE_INT64 && b.data.int_val == 1) return a;
    } else if (op == ESH_TAYLOR_OP_div) {
        if ((uint8_t)(b.type & 0x0F) == ESHKOL_VALUE_INT64 && b.data.int_val == 1) return a;
        if (num_is_exact_zero(&a) && !num_is_exact_zero(&b)) return num_exact_int(0);
    }
    if (num_is_ad_node(&a)) {
        eshkol_tagged_value_t l;
        eshkol_taylor_lift_ad_node(ar, (void*)(uintptr_t)a.data.ptr_val, 0, &l);
        a = l;
    }
    if (num_is_ad_node(&b)) {
        eshkol_tagged_value_t l;
        eshkol_taylor_lift_ad_node(ar, (void*)(uintptr_t)b.data.ptr_val, 0, &l);
        b = l;
    }
    if (tagged_as_taylor(&a) || tagged_as_taylor(&b)) {
        eshkol_tagged_value_t r;
        eshkol_taylor_binary_tagged(ar, &a, &b, op, &r);
        return r;
    }
    if (op == ESH_TAYLOR_OP_atan2) {
        /* sigma*atan(quotient) + const, as tr_atan2() does for a series. */
        eshkol_tagged_value_t pa = tagged_primal(&a), pb = tagged_primal(&b);
        double y0 = tagged_any_to_double(&pa), x0 = tagged_any_to_double(&pb);
        if (!num_is_jet(&a) && !num_is_jet(&b)) return eshkol_make_double(atan2(y0, x0));
        int by_x = fabs(x0) >= fabs(y0);
        eshkol_tagged_value_t q = by_x ? num_binary(ar, a, b, ESH_TAYLOR_OP_div)
                                       : num_binary(ar, b, a, ESH_TAYLOR_OP_div);
        level_jet_t j = jet_of(&q), out;
        double g[4];
        jet_unary_coeffs(ar, ESH_TAYLOR_UOP_atan, j.v[0], 0.0, g);
        jet_compose(out.v, j.v, g);
        if (!by_x) for (int i = 0; i < 8; i++) out.v[i] = -out.v[i];
        out.v[0] = atan2(y0, x0);
        return jet_tagged(ar, &out);
    }
    if (num_is_jet(&a) || num_is_jet(&b)) {
        level_jet_t x = jet_of(&a), y = jet_of(&b), r;
        switch (op) {
            case ESH_TAYLOR_OP_add: for (int s = 0; s < 8; s++) r.v[s] = x.v[s] + y.v[s]; break;
            case ESH_TAYLOR_OP_sub: for (int s = 0; s < 8; s++) r.v[s] = x.v[s] - y.v[s]; break;
            case ESH_TAYLOR_OP_mul: jet_mul_raw(r.v, x.v, y.v); break;
            case ESH_TAYLOR_OP_div: jet_div_raw(r.v, x.v, y.v); break;
            default: {   /* pow */
                int y_const = 1;
                for (int s = 1; s < 8; s++) if (y.v[s] != 0.0) { y_const = 0; break; }
                if (y_const) return jet_unary(ar, &a, -1, y.v[0]);
                eshkol_tagged_value_t lg = jet_unary(ar, &a, ESH_TAYLOR_UOP_log, 0.0);
                eshkol_tagged_value_t pr = num_binary(ar, b, lg, ESH_TAYLOR_OP_mul);
                return jet_unary(ar, &pr, ESH_TAYLOR_UOP_exp, 0.0);
            }
        }
        return jet_tagged(ar, &r);
    }
    if ((uint8_t)(a.type & 0x0F) == ESHKOL_VALUE_COMPLEX ||
        (uint8_t)(b.type & 0x0F) == ESHKOL_VALUE_COMPLEX) {
        /* ADR-0027 section 2: a complex value is the OUTERMOST structure
         * (its parts are carriers); a complex coefficient is never built. */
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
            "internal: a complex value reached a Taylor level coefficient");
        return eshkol_make_double(0.0);
    }
    int exact = tagged_is_exact_number(&a) && tagged_is_exact_number(&b);
    if (op == ESH_TAYLOR_OP_pow) {
        if (exact && (uint8_t)(b.type & 0x0F) == ESHKOL_VALUE_INT64 &&
            (b.data.int_val >= 0 || !num_is_exact_zero(&a))) {
            int64_t p = b.data.int_val;
            uint64_t m = p < 0 ? (uint64_t)(-(p + 1)) + 1u : (uint64_t)p;
            eshkol_tagged_value_t acc = num_exact_int(1), base = a;
            while (m) {
                if (m & 1u) acc = exact_mul(ar, acc, base);
                m >>= 1;
                if (m) base = exact_mul(ar, base, base);
            }
            return p < 0 ? exact_div(ar, num_exact_int(1), acc) : acc;
        }
        return eshkol_make_double(pow(tagged_any_to_double(&a), tagged_any_to_double(&b)));
    }
    if (exact) return exact_binary(ar, a, b, op);   /* exact / exact 0 raises, as R7RS requires */
    double x = tagged_any_to_double(&a), y = tagged_any_to_double(&b);
    switch (op) {
        case ESH_TAYLOR_OP_add: return eshkol_make_double(x + y);
        case ESH_TAYLOR_OP_sub: return eshkol_make_double(x - y);
        case ESH_TAYLOR_OP_mul: return eshkol_make_double(x * y);
        default:                return eshkol_make_double(x / y);
    }
}
static inline eshkol_tagged_value_t num_add(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return num_binary(ar, a, b, ESH_TAYLOR_OP_add); }
static inline eshkol_tagged_value_t num_sub(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return num_binary(ar, a, b, ESH_TAYLOR_OP_sub); }
static inline eshkol_tagged_value_t num_mul(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return num_binary(ar, a, b, ESH_TAYLOR_OP_mul); }
static inline eshkol_tagged_value_t num_div(arena_t* ar, eshkol_tagged_value_t a, eshkol_tagged_value_t b) { return num_binary(ar, a, b, ESH_TAYLOR_OP_div); }

static eshkol_tagged_value_t num_unary(arena_t* ar, eshkol_tagged_value_t x, int uop) {
    if (num_is_ad_node(&x)) {
        eshkol_tagged_value_t l;
        eshkol_taylor_lift_ad_node(ar, (void*)(uintptr_t)x.data.ptr_val, 0, &l);
        x = l;
    }
    if (tagged_as_taylor(&x)) {
        eshkol_tagged_value_t r;
        eshkol_taylor_unary_tagged(ar, &x, uop, &r);
        return r;
    }
    if (num_is_jet(&x)) return jet_unary(ar, &x, uop, 0.0);
    if (tagged_is_exact_number(&x)) {
        int s = tagged_exact_sign(&x);
        switch (uop) {
            case ESH_TAYLOR_UOP_neg: return exact_mul(ar, num_exact_int(-1), x);
            case ESH_TAYLOR_UOP_abs: return s < 0 ? exact_mul(ar, num_exact_int(-1), x) : x;
            case ESH_TAYLOR_UOP_relu: return s > 0 ? x : num_exact_int(0);
            case ESH_TAYLOR_UOP_floor: case ESH_TAYLOR_UOP_ceiling:
            case ESH_TAYLOR_UOP_truncate: case ESH_TAYLOR_UOP_round: {
                /* an exact integer is its own rounding; a rational rounds exactly */
                if (!tagged_is_rational(&x)) return x;
                eshkol_tagged_value_t r;
                void* rp = (void*)(uintptr_t)x.data.ptr_val;
                if (uop == ESH_TAYLOR_UOP_floor)        eshkol_rational_floor_tagged(ar, rp, &r);
                else if (uop == ESH_TAYLOR_UOP_ceiling) eshkol_rational_ceil_tagged(ar, rp, &r);
                else if (uop == ESH_TAYLOR_UOP_truncate) eshkol_rational_truncate_tagged(ar, rp, &r);
                else                                    eshkol_rational_round_tagged(ar, rp, &r);
                return r;
            }
            case ESH_TAYLOR_UOP_asin: case ESH_TAYLOR_UOP_atan:
            case ESH_TAYLOR_UOP_asinh: case ESH_TAYLOR_UOP_atanh:
                if (s == 0) return num_exact_int(0);          /* odd, f(0) = 0 */
                break;
            case ESH_TAYLOR_UOP_acos: case ESH_TAYLOR_UOP_acosh:
            case ESH_TAYLOR_UOP_log2: case ESH_TAYLOR_UOP_log10:
                if ((uint8_t)(x.type & 0x0F) == ESHKOL_VALUE_INT64) {
                    int64_t v = x.data.int_val;
                    if (v == 1) return num_exact_int(0);       /* f(1) = 0 */
                    int64_t b = uop == ESH_TAYLOR_UOP_log2 ? 2 : uop == ESH_TAYLOR_UOP_log10 ? 10 : 0;
                    if (b && v > 1) {                          /* log_b of an exact power of b */
                        int64_t e = 0;
                        while (v % b == 0) { v /= b; e++; }
                        if (v == 1) return num_exact_int(e);
                    }
                }
                break;
            case ESH_TAYLOR_UOP_exp2:
                if ((uint8_t)(x.type & 0x0F) == ESHKOL_VALUE_INT64 &&
                    x.data.int_val > -63 && x.data.int_val < 63) {
                    int64_t e = x.data.int_val;
                    eshkol_tagged_value_t p = num_exact_int((int64_t)1 << (e < 0 ? -e : e));
                    return e < 0 ? exact_div(ar, num_exact_int(1), p) : p;
                }
                break;
            case ESH_TAYLOR_UOP_cbrt:
                /* the cube root of a perfect int64 cube is exact */
                if ((uint8_t)(x.type & 0x0F) == ESHKOL_VALUE_INT64) {
                    int64_t v = x.data.int_val;
                    int64_t c = (int64_t)llround(cbrt((double)v));
                    for (int64_t t = c - 1; t <= c + 1; t++)
                        if (t > -2097152 && t < 2097152 && t * t * t == v) return num_exact_int(t);
                }
                break;
            default: break;
        }
    }
    double v = tagged_any_to_double(&x), r;
    switch (uop) {
        case ESH_TAYLOR_UOP_neg:  r = -v; break;
        case ESH_TAYLOR_UOP_exp:  r = exp(v); break;
        case ESH_TAYLOR_UOP_log:  r = log(v); break;
        case ESH_TAYLOR_UOP_sin:  r = sin(v); break;
        case ESH_TAYLOR_UOP_cos:  r = cos(v); break;
        case ESH_TAYLOR_UOP_tan:  r = tan(v); break;
        case ESH_TAYLOR_UOP_sqrt: r = sqrt(v); break;
        case ESH_TAYLOR_UOP_abs:  r = fabs(v); break;
        case ESH_TAYLOR_UOP_sinh: r = sinh(v); break;
        case ESH_TAYLOR_UOP_cosh: r = cosh(v); break;
        case ESH_TAYLOR_UOP_tanh: r = tanh(v); break;
        case ESH_TAYLOR_UOP_relu: r = v > 0.0 ? v : 0.0; break;
        case ESH_TAYLOR_UOP_sigmoid: r = 1.0 / (1.0 + exp(-v)); break;
        default:
            if (!tr_is_ext_uop(uop)) {
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                    "internal: Taylor unary op %d has no scalar rule", uop);
            }
            r = tr_ext_primal(uop, v);
            break;
    }
    return eshkol_make_double(r);
}

/* ── series over the scalar ring (n = K+1 coefficients) ───────────────── */
static eshkol_tagged_value_t* level_series_alloc(arena_t* ar, int n) {
    eshkol_tagged_value_t* s = (eshkol_tagged_value_t*)arena_allocate(
        ar, (size_t)n * sizeof(eshkol_tagged_value_t));
    if (s) for (int i = 0; i < n; i++) s[i] = num_exact_int(0);
    return s;
}

/* s = u * w (Cauchy product). */
static void level_ser_mul(arena_t* ar, eshkol_tagged_value_t* s,
                          const eshkol_tagged_value_t* u,
                          const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) {
        eshkol_tagged_value_t acc = num_exact_int(0);
        for (int j = 0; j <= k; j++) acc = num_add(ar, acc, num_mul(ar, u[j], w[k - j]));
        s[k] = acc;
    }
}
/* s = u / w: s_k = (u_k - sum_{j=1..k} w_j s_{k-j}) / w_0. */
static void level_ser_div(arena_t* ar, eshkol_tagged_value_t* s,
                          const eshkol_tagged_value_t* u,
                          const eshkol_tagged_value_t* w, int n) {
    for (int k = 0; k < n; k++) {
        eshkol_tagged_value_t acc = u[k];
        for (int j = 1; j <= k; j++) acc = num_sub(ar, acc, num_mul(ar, w[j], s[k - j]));
        s[k] = num_div(ar, acc, w[0]);
    }
}
/* s = exp(u): s_0 = exp(u_0), s_k = (1/k) sum_{j=1..k} j u_j s_{k-j}. */
static void level_ser_exp(arena_t* ar, eshkol_tagged_value_t* s,
                          const eshkol_tagged_value_t* u, int n) {
    s[0] = num_unary(ar, u[0], ESH_TAYLOR_UOP_exp);
    for (int k = 1; k < n; k++) {
        eshkol_tagged_value_t acc = num_exact_int(0);
        for (int j = 1; j <= k; j++)
            acc = num_add(ar, acc, num_mul(ar, num_mul(ar, num_exact_int(j), u[j]), s[k - j]));
        s[k] = num_div(ar, acc, num_exact_int(k));
    }
}
/* s = log(u): s_k = (u_k - (1/k) sum_{j=1..k-1} j s_j u_{k-j}) / u_0. */
static void level_ser_log(arena_t* ar, eshkol_tagged_value_t* s,
                          const eshkol_tagged_value_t* u, int n) {
    s[0] = num_unary(ar, u[0], ESH_TAYLOR_UOP_log);
    for (int k = 1; k < n; k++) {
        eshkol_tagged_value_t acc = num_exact_int(0);
        for (int j = 1; j < k; j++)
            acc = num_add(ar, acc, num_mul(ar, num_mul(ar, num_exact_int(j), s[j]), u[k - j]));
        acc = num_div(ar, acc, num_exact_int(k));
        s[k] = num_div(ar, num_sub(ar, u[k], acc), u[0]);
    }
}
/* coupled sin/cos: sn_k = (1/k) sum j u_j cs_{k-j}, cs_k = -(1/k) sum j u_j sn_{k-j}. */
static void level_ser_sincos(arena_t* ar, eshkol_tagged_value_t* sn, eshkol_tagged_value_t* cs,
                             const eshkol_tagged_value_t* u, int n) {
    sn[0] = num_unary(ar, u[0], ESH_TAYLOR_UOP_sin);
    cs[0] = num_unary(ar, u[0], ESH_TAYLOR_UOP_cos);
    for (int k = 1; k < n; k++) {
        eshkol_tagged_value_t as = num_exact_int(0), ac = num_exact_int(0);
        for (int j = 1; j <= k; j++) {
            eshkol_tagged_value_t ju = num_mul(ar, num_exact_int(j), u[j]);
            as = num_add(ar, as, num_mul(ar, ju, cs[k - j]));
            ac = num_add(ar, ac, num_mul(ar, ju, sn[k - j]));
        }
        sn[k] = num_div(ar, as, num_exact_int(k));
        cs[k] = num_div(ar, num_unary(ar, ac, ESH_TAYLOR_UOP_neg), num_exact_int(k));
    }
}
/* s_k = (1/(k u_0)) sum_{j=1..k} (r j - (k-j)) u_j s_{k-j} for k >= 1: the
 * power recurrence from a given s_0 (cbrt seeds it with the real cube root). */
static void level_ser_pow_tail(arena_t* ar, eshkol_tagged_value_t* s,
                               const eshkol_tagged_value_t* u,
                               eshkol_tagged_value_t r, int n) {
    for (int k = 1; k < n; k++) {
        eshkol_tagged_value_t acc = num_exact_int(0);
        for (int j = 1; j <= k; j++) {
            eshkol_tagged_value_t f = num_sub(ar, num_mul(ar, r, num_exact_int(j)),
                                              num_exact_int(k - j));
            acc = num_add(ar, acc, num_mul(ar, num_mul(ar, f, u[j]), s[k - j]));
        }
        s[k] = num_div(ar, acc, num_mul(ar, num_exact_int(k), u[0]));
    }
}

/* s = u^r for a constant r (any number of an enclosing level). An exact
 * integer r is repeated multiplication: exact, and defined at a zero base.
 * Otherwise s_0 = u_0^r, s_k = (1/(k u_0)) sum_{j=1..k} (r j - (k-j)) u_j s_{k-j}. */
static void level_ser_pow_const(arena_t* ar, eshkol_tagged_value_t* s,
                                const eshkol_tagged_value_t* u,
                                eshkol_tagged_value_t r, int n) {
    if ((uint8_t)(r.type & 0x0F) == ESHKOL_VALUE_INT64) {
        int64_t p = r.data.int_val;
        uint64_t m = p < 0 ? (uint64_t)(-(p + 1)) + 1u : (uint64_t)p;
        eshkol_tagged_value_t* acc = level_series_alloc(ar, n);
        eshkol_tagged_value_t* base = level_series_alloc(ar, n);
        eshkol_tagged_value_t* tmp = level_series_alloc(ar, n);
        if (!acc || !base || !tmp) return;
        acc[0] = num_exact_int(1);
        for (int i = 0; i < n; i++) base[i] = u[i];
        while (m) {
            if (m & 1u) { level_ser_mul(ar, tmp, acc, base, n); memcpy(acc, tmp, (size_t)n * sizeof(*acc)); }
            m >>= 1;
            if (m) { level_ser_mul(ar, tmp, base, base, n); memcpy(base, tmp, (size_t)n * sizeof(*base)); }
        }
        if (p < 0) {
            eshkol_tagged_value_t* one = level_series_alloc(ar, n);
            if (!one) return;
            one[0] = num_exact_int(1);
            level_ser_div(ar, s, one, acc, n);
        } else {
            memcpy(s, acc, (size_t)n * sizeof(*s));
        }
        return;
    }
    s[0] = num_binary(ar, u[0], r, ESH_TAYLOR_OP_pow);
    level_ser_pow_tail(ar, s, u, r, n);
}

/* s_k = (1/k) sum_{j=1..k} j u_j d_{k-j} for k >= 1 over the scalar ring:
 * the integral-family recurrence of tr_integrate(). `d` may alias `s`. */
static void level_ser_integrate(arena_t* ar, eshkol_tagged_value_t* s,
                                const eshkol_tagged_value_t* u,
                                const eshkol_tagged_value_t* d, int n) {
    for (int k = 1; k < n; k++) {
        eshkol_tagged_value_t acc = num_exact_int(0);
        for (int j = 1; j <= k; j++)
            acc = num_add(ar, acc, num_mul(ar, num_mul(ar, num_exact_int(j), u[j]), d[k - j]));
        s[k] = num_div(ar, acc, num_exact_int(k));
    }
}

/**
 * @brief The integral family (tr_ext_unary()) over the scalar ring.
 *
 * Every coefficient operation dispatches through num_binary()/num_unary(), so
 * the coefficients may be any numbers, and an exact input keeps each
 * coefficient that is rational exact: at an exact point the series of atan
 * and atanh are exact past s_0 (their derivatives are rational functions),
 * and so is every coefficient of cbrt at a perfect cube. Returns 0 when `op`
 * is not in the family.
 */
static int level_ext_unary(arena_t* ar, int op, eshkol_tagged_value_t* s,
                           const eshkol_tagged_value_t* u, int n) {
    if (!tr_is_ext_uop(op)) return 0;
    eshkol_tagged_value_t* q = level_series_alloc(ar, n);
    eshkol_tagged_value_t* d = level_series_alloc(ar, n);
    if (!q || !d) return 0;
    /* At an exact point where f' is unbounded (atanh/asin/acos at +-1, acosh
     * at 1, cbrt/log2/log10 at 0) the recurrence would divide an exact number
     * by exact 0 and raise. The derivative there is infinite, as the inexact
     * kernel answers: read the point as inexact. */
    if (tagged_is_exact_number(&u[0])) {
        int sgn = tagged_exact_sign(&u[0]);
        eshkol_tagged_value_t a = sgn < 0 ? exact_mul(ar, num_exact_int(-1), u[0]) : u[0];
        int at_one = (uint8_t)(a.type & 0x0F) == ESHKOL_VALUE_INT64 && a.data.int_val == 1;
        int singular =
            ((op == ESH_TAYLOR_UOP_atanh || op == ESH_TAYLOR_UOP_asin ||
              op == ESH_TAYLOR_UOP_acos) && at_one) ||
            (op == ESH_TAYLOR_UOP_acosh && at_one && sgn > 0) ||
            ((op == ESH_TAYLOR_UOP_cbrt || op == ESH_TAYLOR_UOP_log2 ||
              op == ESH_TAYLOR_UOP_log10) && sgn == 0);
        if (singular) {
            eshkol_tagged_value_t* v = level_series_alloc(ar, n);
            if (!v) return 0;
            for (int k = 0; k < n; k++) v[k] = u[k];
            v[0] = eshkol_make_double(tagged_any_to_double(&u[0]));
            u = v;
        }
    }
    for (int k = 0; k < n; k++) s[k] = num_exact_int(0);
    switch (op) {
        case ESH_TAYLOR_UOP_atan: case ESH_TAYLOR_UOP_atanh:
        case ESH_TAYLOR_UOP_asin: case ESH_TAYLOR_UOP_acos:
        case ESH_TAYLOR_UOP_asinh: case ESH_TAYLOR_UOP_acosh: {
            int minus = op == ESH_TAYLOR_UOP_atanh || op == ESH_TAYLOR_UOP_asin ||
                        op == ESH_TAYLOR_UOP_acos;
            level_ser_mul(ar, q, u, u, n);
            if (minus) for (int k = 0; k < n; k++) q[k] = num_unary(ar, q[k], ESH_TAYLOR_UOP_neg);
            q[0] = num_add(ar, q[0], num_exact_int(op == ESH_TAYLOR_UOP_acosh ? -1 : 1));
            if (op == ESH_TAYLOR_UOP_atan || op == ESH_TAYLOR_UOP_atanh) {
                eshkol_tagged_value_t* one = level_series_alloc(ar, n);
                if (!one) return 0;
                one[0] = num_exact_int(1);
                level_ser_div(ar, d, one, q, n);
            } else {
                level_ser_pow_const(ar, d, q, exact_div(ar, num_exact_int(-1), num_exact_int(2)), n);
                if (op == ESH_TAYLOR_UOP_acos)
                    for (int k = 0; k < n; k++) d[k] = num_unary(ar, d[k], ESH_TAYLOR_UOP_neg);
            }
            s[0] = num_unary(ar, u[0], op);
            level_ser_integrate(ar, s, u, d, n);
            return 1;
        }
        case ESH_TAYLOR_UOP_log2: case ESH_TAYLOR_UOP_log10:
            q[0] = eshkol_make_double(op == ESH_TAYLOR_UOP_log2 ? 1.0 / M_LN2 : 1.0 / M_LN10);
            level_ser_div(ar, d, q, u, n);
            s[0] = num_unary(ar, u[0], op);
            level_ser_integrate(ar, s, u, d, n);
            return 1;
        case ESH_TAYLOR_UOP_exp2:
            for (int k = 0; k < n; k++) q[k] = num_mul(ar, eshkol_make_double(M_LN2), u[k]);
            s[0] = num_unary(ar, u[0], op);
            level_ser_integrate(ar, s, q, s, n);
            return 1;
        case ESH_TAYLOR_UOP_cbrt:
            /* d = (1/3) u^(-2/3) from d_0 = 1/(3 s_0^2), then the integral:
             * exact at a perfect cube, the pole of the closed form at 0. */
            s[0] = num_unary(ar, u[0], op);
            d[0] = num_div(ar, num_exact_int(1),
                           num_mul(ar, num_exact_int(3), num_mul(ar, s[0], s[0])));
            level_ser_pow_tail(ar, d, u, exact_div(ar, num_exact_int(-2), num_exact_int(3)), n);
            level_ser_integrate(ar, s, u, d, n);
            return 1;
        default:                                    /* the rounding functions */
            s[0] = num_unary(ar, u[0], op);
            return 1;
    }
}

/* ── operands as series of the active level ───────────────────────────── */
/* The series of `v` in level `epoch`: its own coefficients when it is a tower
 * of that epoch, otherwise the constant (v, 0, ..., 0) with v kept whole. */
static int level_series_of(arena_t* ar, const eshkol_tagged_value_t* v, uint32_t epoch,
                           eshkol_tagged_value_t* s, int n) {
    for (int i = 0; i < n; i++) s[i] = num_exact_int(0);
    esh_taylor_t* t = tagged_as_taylor(v);
    if (!t || ESH_TAYLOR_GET_EPOCH(t->flags) != epoch || epoch == 0u) {
        s[0] = *v;
        return 0;
    }
    int m = (int)t->order_k + 1;
    if (m > n) m = n;
    if (taylor_is_level(t)) {
        for (int i = 0; i < m; i++) s[i] = level_c_const(t)[i];
    } else {
        if (ESH_TAYLOR_HAS_TANGENT(t->flags)) {
            /* A classic tower carrying the reverse-seed companion is never
             * nested inside a level (ADR-0027 section 3); dropping the
             * companion here would silently lose the gradient. */
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                "internal: a reverse-seed Taylor carrier met a nested level");
            return 1;
        }
        for (int i = 0; i < m; i++)
            s[i] = taylor_is_exact(t) ? taylor_exact_c_const(t)[i]
                                      : eshkol_make_double(t->c[i]);
    }
    (void)ar;
    return 1;
}

static eshkol_tagged_value_t level_make(arena_t* ar, uint32_t epoch, uint32_t order_k,
                                        const eshkol_tagged_value_t* s) {
    esh_taylor_t* t = level_alloc(ar, order_k, epoch);
    if (!t) return s[0];
    for (uint32_t i = 0; i <= order_k; i++) level_c(t)[i] = s[i];
    return taylor_to_tagged(t);
}

/* Does this operation belong to the level path? A level carrier among the
 * operands, or two towers of different levels. */
static int level_route_binary(const eshkol_tagged_value_t* l, const eshkol_tagged_value_t* r) {
    esh_taylor_t* lt = tagged_as_taylor(l);
    esh_taylor_t* rt = tagged_as_taylor(r);
    if (taylor_is_level(lt) || taylor_is_level(rt)) return 1;
    if (lt && rt) {
        uint32_t le = ESH_TAYLOR_GET_EPOCH(lt->flags), re = ESH_TAYLOR_GET_EPOCH(rt->flags);
        return le != 0u && re != 0u && le != re;
    }
    return 0;
}

static void level_binary(arena_t* ar, const eshkol_tagged_value_t* left,
                         const eshkol_tagged_value_t* right, int op,
                         eshkol_tagged_value_t* result) {
    esh_taylor_t* lt = tagged_as_taylor(left);
    esh_taylor_t* rt = tagged_as_taylor(right);
    uint32_t le = lt ? ESH_TAYLOR_GET_EPOCH(lt->flags) : 0u;
    uint32_t re = rt ? ESH_TAYLOR_GET_EPOCH(rt->flags) : 0u;
    uint32_t epoch = le > re ? le : re;
    uint32_t order_k = 0;
    if (lt && le == epoch && lt->order_k > order_k) order_k = lt->order_k;
    if (rt && re == epoch && rt->order_k > order_k) order_k = rt->order_k;
    int n = (int)order_k + 1;
    eshkol_tagged_value_t* u = level_series_alloc(ar, n);
    eshkol_tagged_value_t* w = level_series_alloc(ar, n);
    eshkol_tagged_value_t* s = level_series_alloc(ar, n);
    if (!u || !w || !s) { *result = eshkol_make_double(0.0); return; }
    int u_active = level_series_of(ar, left, epoch, u, n);
    int w_active = level_series_of(ar, right, epoch, w, n);
    switch (op) {
        case ESH_TAYLOR_OP_add:
            for (int k = 0; k < n; k++) s[k] = num_add(ar, u[k], w[k]);
            break;
        case ESH_TAYLOR_OP_sub:
            for (int k = 0; k < n; k++) s[k] = num_sub(ar, u[k], w[k]);
            break;
        case ESH_TAYLOR_OP_mul:
            if (!u_active)      for (int k = 0; k < n; k++) s[k] = num_mul(ar, *left, w[k]);
            else if (!w_active) for (int k = 0; k < n; k++) s[k] = num_mul(ar, u[k], *right);
            else level_ser_mul(ar, s, u, w, n);
            break;
        case ESH_TAYLOR_OP_div:
            if (!w_active) for (int k = 0; k < n; k++) s[k] = num_div(ar, u[k], *right);
            else level_ser_div(ar, s, u, w, n);
            break;
        case ESH_TAYLOR_OP_atan2: {
            /* sigma*atan(quotient) + const, as tr_atan2() does for doubles;
             * s_0 recurses to the coefficients' own atan2. */
            eshkol_tagged_value_t py = tagged_primal(&u[0]), px = tagged_primal(&w[0]);
            double y0 = tagged_any_to_double(&py), x0 = tagged_any_to_double(&px);
            int by_x = fabs(x0) >= fabs(y0);
            eshkol_tagged_value_t* r = level_series_alloc(ar, n);
            if (!r) break;
            if (by_x) level_ser_div(ar, r, u, w, n);
            else      level_ser_div(ar, r, w, u, n);
            level_ext_unary(ar, ESH_TAYLOR_UOP_atan, s, r, n);
            if (!by_x) for (int k = 0; k < n; k++) s[k] = num_unary(ar, s[k], ESH_TAYLOR_UOP_neg);
            s[0] = num_binary(ar, u[0], w[0], ESH_TAYLOR_OP_atan2);
            break;
        }
        default: {   /* pow */
            if (!w_active) {
                level_ser_pow_const(ar, s, u, *right, n);
            } else {
                eshkol_tagged_value_t* lg = level_series_alloc(ar, n);
                eshkol_tagged_value_t* pr = level_series_alloc(ar, n);
                if (!lg || !pr) break;
                level_ser_log(ar, lg, u, n);
                level_ser_mul(ar, pr, w, lg, n);
                level_ser_exp(ar, s, pr, n);
            }
            break;
        }
    }
    *result = level_make(ar, epoch, order_k, s);
}

static void level_unary(arena_t* ar, const eshkol_tagged_value_t* in, int op,
                        eshkol_tagged_value_t* result) {
    esh_taylor_t* t = tagged_as_taylor(in);
    uint32_t epoch = ESH_TAYLOR_GET_EPOCH(t->flags);
    uint32_t order_k = t->order_k;
    int n = (int)order_k + 1;
    eshkol_tagged_value_t* u = level_series_alloc(ar, n);
    eshkol_tagged_value_t* s = level_series_alloc(ar, n);
    eshkol_tagged_value_t* a = level_series_alloc(ar, n);
    eshkol_tagged_value_t* b = level_series_alloc(ar, n);
    if (!u || !s || !a || !b) { *result = eshkol_make_double(0.0); return; }
    level_series_of(ar, in, epoch, u, n);
    switch (op) {
        case ESH_TAYLOR_UOP_neg:
            for (int k = 0; k < n; k++) s[k] = num_unary(ar, u[k], ESH_TAYLOR_UOP_neg);
            break;
        case ESH_TAYLOR_UOP_exp: level_ser_exp(ar, s, u, n); break;
        case ESH_TAYLOR_UOP_log: level_ser_log(ar, s, u, n); break;
        case ESH_TAYLOR_UOP_sin: level_ser_sincos(ar, s, a, u, n); break;
        case ESH_TAYLOR_UOP_cos: level_ser_sincos(ar, a, s, u, n); break;
        case ESH_TAYLOR_UOP_tan: level_ser_sincos(ar, a, b, u, n); level_ser_div(ar, s, a, b, n); break;
        case ESH_TAYLOR_UOP_sqrt: level_ser_pow_const(ar, s, u, eshkol_make_double(0.5), n); break;
        case ESH_TAYLOR_UOP_abs:
        case ESH_TAYLOR_UOP_relu: {
            /* Decided on the exact primal; at the kink the documented
             * zero-subgradient convention of the classic tower applies. */
            int sgn = tagged_primal_sign(&u[0]);
            if (sgn > 0) {
                for (int k = 0; k < n; k++) s[k] = u[k];
            } else if (sgn < 0) {
                if (op == ESH_TAYLOR_UOP_abs)
                    for (int k = 0; k < n; k++) s[k] = num_unary(ar, u[k], ESH_TAYLOR_UOP_neg);
                else
                    s[0] = num_unary(ar, u[0], ESH_TAYLOR_UOP_relu);
            } else {
                s[0] = num_unary(ar, u[0], op);
            }
            break;
        }
        case ESH_TAYLOR_UOP_sinh:
        case ESH_TAYLOR_UOP_cosh:
        case ESH_TAYLOR_UOP_tanh: {
            eshkol_tagged_value_t* m = level_series_alloc(ar, n);
            if (!m) break;
            for (int k = 0; k < n; k++) m[k] = num_unary(ar, u[k], ESH_TAYLOR_UOP_neg);
            level_ser_exp(ar, a, u, n);
            level_ser_exp(ar, b, m, n);
            eshkol_tagged_value_t half = exact_div(ar, num_exact_int(1), num_exact_int(2));
            for (int k = 0; k < n; k++) {
                m[k] = num_mul(ar, half, num_sub(ar, a[k], b[k]));    /* sinh */
                a[k] = num_mul(ar, half, num_add(ar, a[k], b[k]));    /* cosh */
            }
            if (op == ESH_TAYLOR_UOP_sinh) memcpy(s, m, (size_t)n * sizeof(*s));
            else if (op == ESH_TAYLOR_UOP_cosh) memcpy(s, a, (size_t)n * sizeof(*s));
            else level_ser_div(ar, s, m, a, n);
            break;
        }
        case ESH_TAYLOR_UOP_sigmoid: {
            /* 1 / (1 + exp(-u)) */
            for (int k = 0; k < n; k++) b[k] = num_unary(ar, u[k], ESH_TAYLOR_UOP_neg);
            level_ser_exp(ar, a, b, n);
            a[0] = num_add(ar, a[0], num_exact_int(1));
            for (int k = 0; k < n; k++) b[k] = num_exact_int(0);
            b[0] = num_exact_int(1);
            level_ser_div(ar, s, b, a, n);
            break;
        }
        default:
            if (!level_ext_unary(ar, op, s, u, n))
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                    "internal: Taylor unary op %d has no recurrence", op);
            break;
    }
    *result = level_make(ar, epoch, order_k, s);
}

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

    /* No tower among the operands (atan2 of forward jets, SW-210): the
     * carrier ring's own rule, in the full 8-jet algebra. */
    if (!tagged_as_taylor(left) && !tagged_as_taylor(right)) {
        *result = num_binary(arena, *left, *right, op);
        return;
    }

    /* ADR-0027: nested levels take the level path. */
    if (level_route_binary(left, right)) {
        level_binary(arena, left, right, op, result);
        return;
    }

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
        int exact_tangent = (op == ESH_TAYLOR_OP_add ||
                             op == ESH_TAYLOR_OP_sub ||
                             op == ESH_TAYLOR_OP_mul ||
                             op == ESH_TAYLOR_OP_div) &&
            operand_is_exact_for_taylor(left, epoch) &&
            operand_is_exact_for_taylor(right, epoch) &&
            (operand_has_exact_tangent(left, epoch) ||
             operand_has_exact_tangent(right, epoch));
        uint32_t out_flags = ESH_TAYLOR_MK_FLAGS(
            exact_tangent ? ESH_TAYLOR_COEFF_RATIONAL
                          : ESH_TAYLOR_COEFF_F64, epoch) |
            ESH_TAYLOR_TANGENT_FLAG |
            (exact_tangent ? ESH_TAYLOR_TANGENT_EXACT_FLAG : 0u);
        esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k, out_flags);
        if (!out) { *result = eshkol_make_double(0.0); return; }
        out->flags |= taylor_dual_primal_sign(arena, left, right, op);
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
            case ESH_TAYLOR_OP_atan2: ddual_atan2(ov, ot, uv, ut, wv, wt, n, arena); break;
            default:
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                    "internal: Taylor binary op %d has no recurrence", op);
                break;
        }
        if (exact_tangent && !exact_tangent_binary(arena, left, right, op,
                                                   order_k, epoch, out)) {
            out->flags &= ~ESH_TAYLOR_TANGENT_EXACT_FLAG;
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
        case ESH_TAYLOR_OP_atan2: tr_atan2(out->c, u, w, n, arena); break;
        default:
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                "internal: Taylor binary op %d has no recurrence", op);
            break;
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
    /* A forward jet with no tower (abs of a jet, SW-212): the carrier ring's
     * own rule, in the full 8-jet algebra. */
    if (!t && num_is_jet(in)) {
        *result = num_unary(arena, *in, op);
        return;
    }
    /* ADR-0027: a level carrier takes the level path. */
    if (taylor_is_level(t)) {
        level_unary(arena, in, op, result);
        return;
    }
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
        int exact_linear = (op == ESH_TAYLOR_UOP_neg ||
                            op == ESH_TAYLOR_UOP_abs ||
                            op == ESH_TAYLOR_UOP_relu) &&
            operand_is_exact_for_taylor(in, epoch) &&
            operand_has_exact_tangent(in, epoch);
        uint32_t unary_flags = ESH_TAYLOR_MK_FLAGS(
            exact_linear ? ESH_TAYLOR_COEFF_RATIONAL
                         : ESH_TAYLOR_COEFF_F64, epoch) |
            ESH_TAYLOR_TANGENT_FLAG |
            (exact_linear ? ESH_TAYLOR_TANGENT_EXACT_FLAG : 0u);
        esh_taylor_t* out = eshkol_taylor_alloc(arena, order_k, unary_flags);
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
                int hinted_sign = (t->flags & ESH_TAYLOR_PRIMAL_NEGATIVE_FLAG) ? -1
                                : (t->flags & ESH_TAYLOR_PRIMAL_POSITIVE_FLAG) ? 1 : 0;
                double sgn = uv[0] == 0.0
                    ? (double)hinted_sign : (uv[0] < 0.0 ? -1.0 : 1.0);
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
            default:
                if (!ddual_ext(op, ov, ot, uv, ut, n, arena))
                    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                        "internal: Taylor unary op %d has no recurrence", op);
                break;
        }
        if (exact_linear) {
            eshkol_tagged_value_t* eu = alloc_exact_series(arena, n);
            eshkol_tagged_value_t* eut = alloc_exact_series(arena, n);
            eshkol_tagged_value_t* eo = out->exact_c;
            eshkol_tagged_value_t* eot = taylor_tan_exact(out);
            if (!eu || !eut || !eo || !eot) {
                *result = eshkol_make_double(0.0);
                return;
            }
            normalise_operand_exact(in, epoch, eu, n);
            normalise_tangent_exact(in, epoch, eut, n, arena);
            eshkol_tagged_value_t zero = eshkol_make_int64(0, true);
            int sign = tagged_exact_sign(&eu[0]);
            for (int k = 0; k < n; ++k) {
                if (op == ESH_TAYLOR_UOP_neg) {
                    eo[k] = exact_sub(arena, zero, eu[k]);
                    eot[k] = exact_sub(arena, zero, eut[k]);
                } else if (op == ESH_TAYLOR_UOP_relu) {
                    eo[k] = sign > 0 ? eu[k] : zero;
                    eot[k] = sign > 0 ? eut[k] : zero;
                } else {
                    eo[k] = sign > 0 ? eu[k]
                          : sign < 0 ? exact_sub(arena, zero, eu[k]) : zero;
                    eot[k] = sign > 0 ? eut[k]
                           : sign < 0 ? exact_sub(arena, zero, eut[k]) : zero;
                }
            }
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
    /* SW-210: the integral family of an exact tower runs its recurrence over
     * the exact coefficients (level_ext_unary), and the tower stays exact
     * when every coefficient came out exact -- a rounding function anywhere,
     * atan/asin/asinh/atanh at 0, acos/acosh/log2/log10 where the primal is
     * exact, cbrt at a perfect cube. Otherwise the whole tower demotes (its
     * coefficients are never mixed-tagged; a level carrier keeps an exact
     * tail past an irrational s_0). */
    if (t && taylor_is_exact(t) && tr_is_ext_uop(op)) {
        eshkol_tagged_value_t* u = alloc_exact_series(arena, n);
        eshkol_tagged_value_t* s = alloc_exact_series(arena, n);
        if (!u || !s) { *result = eshkol_make_double(0.0); return; }
        normalise_operand_exact(in, epoch, u, n);
        if (level_ext_unary(arena, op, s, u, n)) {
            taylor_materialize_exact_or_demote(arena, s, n, order_k, epoch, result);
            return;
        }
    }
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
        default: {
            double db[ESH_TAYLOR_STACKN];
            double* d = n > ESH_TAYLOR_STACKN
                ? (double*)arena_allocate(arena, (size_t)n * sizeof(double)) : db;
            if (!d || !tr_ext_unary(op, out->c, d, u, n, arena))
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                    "internal: Taylor unary op %d has no recurrence", op);
            break;
        }
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

/* Exact counterpart of eshkol_taylor_extract_tangent().  A tangent companion
 * produced by a foreign exact epoch is stored as tagged values, so factorial
 * scaling must stay in the same exact numeric tower before a reverse tape sees
 * the value. */
void eshkol_taylor_extract_tangent_tagged(arena_t* arena,
                                          const eshkol_tagged_value_t* tv,
                                          uint32_t n,
                                          eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (!out) return;
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t || !ESH_TAYLOR_HAS_TANGENT(t->flags) || n > t->order_k) {
        *out = eshkol_make_double(0.0);
        return;
    }
    if (ESH_TAYLOR_TANGENT_IS_EXACT(t->flags) &&
        (taylor_is_exact(t) || t->exact_c)) {
        eshkol_tagged_value_t fact = eshkol_make_int64(1, true);
        for (uint32_t i = 2; i <= n; ++i)
            fact = exact_mul(arena, fact, eshkol_make_int64((int64_t)i, true));
        *out = exact_mul(arena, fact, taylor_tan_exact_const(t)[n]);
        return;
    }
    *out = eshkol_make_double(factorial_d(n) * taylor_tan(t)[n]);
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
    const ad_node_t* ad_node = (const ad_node_t*)node;
    double val = ad_node ? ad_node->value : 0.0;
    double flag = node ? eshkol_ad_seed_flag(node) : 0.0;
    int exact = ad_node && ad_node->exact_value &&
        tagged_is_exact_number(ad_node->exact_value);
    esh_taylor_t* t = eshkol_taylor_alloc(arena, (uint32_t)order_k,
        ESH_TAYLOR_MK_FLAGS(exact ? ESH_TAYLOR_COEFF_RATIONAL
                                  : ESH_TAYLOR_COEFF_F64, 0u) |
        ESH_TAYLOR_TANGENT_FLAG |
        (exact ? ESH_TAYLOR_TANGENT_EXACT_FLAG : 0u));
    if (!t) { *out = eshkol_make_double(val); return; }
    t->c[0] = val;
    taylor_tan(t)[0] = flag;
    if (exact) {
        taylor_exact_c(t)[0] = *ad_node->exact_value;
        taylor_tan_exact(t)[0] = eshkol_make_int64(flag != 0.0 ? 1 : 0, true);
    }
    *out = taylor_to_tagged(t);
}

/* ----------------------------------------------------------------------- */
/* ESH-0402: nested-AD carrier composition (SW-03 / SW-04)                  */
/* ----------------------------------------------------------------------- */
/* Nested passes (ADR-0027)                                                 */
/* ----------------------------------------------------------------------- */

/**
 * @brief Guard at a jet pass's extraction: a Taylor carrier never reaches it.
 *
 * A pass nested inside a tower or level runs as a level itself and is
 * extracted by eshkol_ad_nested_extract, so a carrier arriving at an un-nested
 * 8-jet extraction escaped the pass that owns it (a carrier stored and read
 * back outside its pass). Reading it as a constant would answer 0; raise.
 */
void eshkol_ad_jet_result_check(const eshkol_tagged_value_t* result) {
    if (!result || !tagged_as_taylor(result)) return;
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
        "derivative: a Taylor carrier escaped the differentiation pass that owns it");
}

/**
 * @brief Decide whether a differentiation pass runs as a LEVEL (ADR-0027) and,
 *        when it does, seed it.
 *
 * A `derivative-n`/`taylor` pass (tower_pass = 1) runs as a level when any
 * forward pass is live (pert_level > 0 or tower_depth > 0) or its point is
 * already a carrier. A first-order jet pass (tower_pass = 0) runs as an
 * order-1 level when a tower or level pass is open, its point is a tower, or
 * two jet levels are already live (the 8-jet has no third forward slot);
 * otherwise it keeps the 8-jet, which nests with itself. Every other pass is
 * un-nested and seeds exactly as before (ESH_AD_NEST_NONE).
 *
 * The level's seed is (point, 1, 0, ..., 0): the point is kept whole as c[0],
 * and the unit is exact when the point's primal is exact.
 *
 * @param arena       allocation arena (NULL -> global).
 * @param point       the evaluation point as handed to this pass.
 * @param order_k     this pass's own order (1 for the jet arm).
 * @param pert_level  the runtime forward (8-jet) perturbation level on entry.
 * @param tower_pass  1 for the Taylor-tower arm, 0 for the jet arm.
 * @param tower_depth the number of open tower/level passes on entry.
 * @param out         receives the seeded level carrier when nested.
 * @return ESH_AD_NEST_NONE, or ESH_AD_NEST_LEVEL | epoch << 8 | inexact << 24.
 */
int32_t eshkol_ad_nested_seed(arena_t* arena, const eshkol_tagged_value_t* point,
                              int32_t order_k, int64_t pert_level, int32_t tower_pass,
                              int64_t tower_depth, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (!point || !out) return ESH_AD_NEST_NONE;
    if (order_k < 0) order_k = 0;
    const int point_is_tower = tagged_as_taylor(point) != NULL;
    const int point_is_jet = num_is_jet(point);
    /* The 8-jet holds two forward directions (e1, e2) plus the reverse-seed
     * slot ep, so a third live jet level has no slot of its own: it runs as a
     * level too (SW-206). */
    const int nested = tower_pass
        ? (pert_level > 0 || tower_depth > 0 || point_is_tower || point_is_jet)
        : (tower_depth > 0 || point_is_tower || pert_level >= 2);
    if (!nested) return ESH_AD_NEST_NONE;

    uint32_t epoch = eshkol_taylor_next_epoch();
    eshkol_tagged_value_t primal = tagged_primal(point);
    const int exact = tagged_is_exact_number(&primal);
    int32_t route = (int32_t)((uint32_t)ESH_AD_NEST_LEVEL |
                              ((epoch & 0xFFFFu) << 8) |
                              (exact ? 0u : (1u << 24)));
    if (order_k == 0) {        /* order 0 is evaluation: no new perturbation */
        *out = *point;
        return route;
    }
    esh_taylor_t* t = level_alloc(arena, (uint32_t)order_k, epoch);
    if (!t) return ESH_AD_NEST_UNSUPPORTED;
    level_c(t)[0] = *point;
    level_c(t)[1] = exact ? num_exact_int(1) : eshkol_make_double(1.0);
    *out = taylor_to_tagged(t);
    return route;
}

/* A coefficient read out of a level: a structural (exact) zero follows the
 * exactness of the point, exactly like an un-nested pass's vanishing
 * derivative (ESH-0411). */
static eshkol_tagged_value_t level_extracted(eshkol_tagged_value_t v, int inexact_seed,
                                             const eshkol_tagged_value_t* result) {
    if (!num_is_exact_zero(&v)) return v;
    eshkol_tagged_value_t p = tagged_primal(result);
    if (inexact_seed || !tagged_is_exact_number(&p)) return eshkol_make_double(0.0);
    return v;
}

/**
 * @brief Read a level pass's answer in its own epoch (ADR-0027 section 4).
 *
 * `derivative`/`derivative-n` of order k answer k! * c[k]; `taylor` of order K
 * (want_list != 0) answers the list (c[0] ... c[K]). A coefficient is a number
 * of the enclosing levels, so the enclosing passes keep their perturbations. A
 * result that is not a carrier of this level does not depend on it.
 *
 * @param arena        allocation arena (NULL -> global).
 * @param result       the differentiated body's raw return value.
 * @param route_packed the value eshkol_ad_nested_seed returned.
 * @param order_k      this pass's own order.
 * @param want_list    1 for `taylor` (coefficient list), 0 for a derivative.
 * @param out          receives the answer.
 */
void eshkol_ad_nested_extract(arena_t* arena, const eshkol_tagged_value_t* result,
                              int32_t route_packed, int32_t order_k, int32_t want_list,
                              eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    if (!out) return;
    if (!result) { *out = eshkol_make_double(0.0); return; }
    uint32_t epoch = ((uint32_t)route_packed >> 8) & 0xFFFFu;
    int inexact_seed = (((uint32_t)route_packed >> 24) & 1u) != 0;
    if (order_k < 0) order_k = 0;
    int n = order_k + 1;
    eshkol_tagged_value_t* s = level_series_alloc(arena, n);
    if (!s) { *out = eshkol_make_double(0.0); return; }
    if (order_k == 0) s[0] = *result;
    else level_series_of(arena, result, epoch, s, n);

    if (!want_list && order_k > 0 &&
        (uint8_t)(result->type & 0x0F) == ESHKOL_VALUE_HEAP_PTR && result->data.ptr_val &&
        ESHKOL_GET_HEADER((void*)(uintptr_t)result->data.ptr_val)->subtype == HEAP_SUBTYPE_VECTOR) {
        /* R -> R^n: the derivative of a vector is the vector of derivatives. */
        const uint8_t* src = (const uint8_t*)(uintptr_t)result->data.ptr_val;
        int64_t len = *(const int64_t*)(const void*)src;
        const eshkol_tagged_value_t* elems = (const eshkol_tagged_value_t*)(const void*)(src + 8);
        uint8_t* dst = (uint8_t*)arena_allocate_vector_with_header(arena, (size_t)(len > 0 ? len : 0));
        if (!dst) { *out = eshkol_make_double(0.0); return; }
        *(int64_t*)(void*)dst = len;
        eshkol_tagged_value_t* delems = (eshkol_tagged_value_t*)(void*)(dst + 8);
        for (int64_t i = 0; i < len; i++)
            eshkol_ad_nested_extract(arena, &elems[i], route_packed, order_k, 0, &delems[i]);
        memset(out, 0, sizeof(*out));
        out->type = ESHKOL_VALUE_HEAP_PTR;
        out->data.ptr_val = (uint64_t)(uintptr_t)dst;
        return;
    }
    if (!want_list) {
        eshkol_tagged_value_t fact = num_exact_int(1);
        for (int i = 2; i <= order_k; i++) fact = exact_mul(arena, fact, num_exact_int(i));
        *out = level_extracted(num_mul(arena, fact, s[order_k]), inexact_seed, result);
        return;
    }
    eshkol_tagged_value_t acc;
    memset(&acc, 0, sizeof(acc));
    acc.type = ESHKOL_VALUE_NULL;
    for (int k = order_k; k >= 0; k--) {
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!cell) { *out = acc; return; }
        cell->car = level_extracted(s[k], inexact_seed, result);
        cell->cdr = acc;
        eshkol_tagged_value_t v;
        memset(&v, 0, sizeof(v));
        v.type = ESHKOL_VALUE_HEAP_PTR;
        v.data.ptr_val = (uint64_t)(uintptr_t)cell;
        acc = v;
    }
    *out = acc;
}

/* Defined below. */
void eshkol_taylor_extract_tagged(arena_t* arena, const eshkol_tagged_value_t* tv,
                                  uint32_t n, eshkol_tagged_value_t* out);


/* ── Vector-point operators nested in a live pass (ADR-0027) ─────────────
 * A `jacobian` whose point carries an enclosing carrier, or which runs while a
 * forward pass is live, is computed column by column as forward passes of its
 * own: each column seeds one coordinate through the one forward-pass protocol
 * (so it runs as a level when nested) and the columns are assembled into a
 * tensor whose elements are tagged numbers of the enclosing levels. These
 * helpers read and build the points and the result. */

static int tagged_is_heap_subtype(const eshkol_tagged_value_t* v, uint8_t subtype) {
    if (!v || (uint8_t)(v->type & 0x0F) != ESHKOL_VALUE_HEAP_PTR || !v->data.ptr_val) return 0;
    const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
    return hdr && hdr->subtype == subtype;
}

/** @brief Number of coordinates of a vector/list/tensor point (1 for a scalar). */
int64_t eshkol_ad_point_length(const eshkol_tagged_value_t* point) {
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_VECTOR))
        return *(const int64_t*)(uintptr_t)point->data.ptr_val;
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_TENSOR))
        return (int64_t)((const eshkol_tensor_t*)(uintptr_t)point->data.ptr_val)->total_elements;
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_CONS)) {
        int64_t n = 0;
        eshkol_tagged_value_t cur = *point;
        while (tagged_is_heap_subtype(&cur, HEAP_SUBTYPE_CONS)) {
            n++;
            cur = ((const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->cdr;
        }
        return n;
    }
    return 1;
}

/** @brief Coordinate `j` of a vector/list/tensor point as a tagged number. */
void eshkol_ad_point_element(const eshkol_tagged_value_t* point, int64_t j,
                             eshkol_tagged_value_t* out) {
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_VECTOR)) {
        *out = ((const eshkol_tagged_value_t*)(uintptr_t)(point->data.ptr_val + 8))[j];
        return;
    }
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_TENSOR)) {
        const eshkol_tensor_t* t = (const eshkol_tensor_t*)(uintptr_t)point->data.ptr_val;
        if (eshkol_tensor_dtype_is_tagged(t->dtype)) {
            *out = ((const eshkol_tagged_value_t*)(void*)t->elements)[j];
        } else {
            double d;
            memcpy(&d, &t->elements[j], sizeof(d));
            *out = eshkol_make_double(d);
        }
        return;
    }
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_CONS)) {
        eshkol_tagged_value_t cur = *point;
        for (int64_t i = 0; i < j && tagged_is_heap_subtype(&cur, HEAP_SUBTYPE_CONS); i++)
            cur = ((const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->cdr;
        *out = ((const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->car;
        return;
    }
    *out = *point;
}

/** @brief Does the point (or any coordinate of it) carry a differentiation carrier? */
int32_t eshkol_ad_point_has_carrier(const eshkol_tagged_value_t* point) {
    if (!point) return 0;
    if (tagged_as_taylor(point) || num_is_jet(point)) return 1;
    if (!tagged_is_heap_subtype(point, HEAP_SUBTYPE_VECTOR) &&
        !tagged_is_heap_subtype(point, HEAP_SUBTYPE_CONS) &&
        !tagged_is_heap_subtype(point, HEAP_SUBTYPE_TENSOR)) return 0;
    if (tagged_is_heap_subtype(point, HEAP_SUBTYPE_TENSOR) &&
        !eshkol_tensor_dtype_is_tagged(
            ((const eshkol_tensor_t*)(uintptr_t)point->data.ptr_val)->dtype)) return 0;
    int64_t n = eshkol_ad_point_length(point);
    for (int64_t j = 0; j < n; j++) {
        eshkol_tagged_value_t e;
        eshkol_ad_point_element(point, j, &e);
        if (tagged_as_taylor(&e) || num_is_jet(&e)) return 1;
    }
    return 0;
}

/** @brief A Scheme vector copy of the point with coordinate `j` replaced by `v`. */
void eshkol_ad_point_with(arena_t* arena, const eshkol_tagged_value_t* point, int64_t j,
                          const eshkol_tagged_value_t* v, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    int64_t n = eshkol_ad_point_length(point);
    uint8_t* dst = (uint8_t*)arena_allocate_vector_with_header(arena, (size_t)(n > 0 ? n : 0));
    if (!dst) { *out = *v; return; }
    *(int64_t*)(void*)dst = n;
    eshkol_tagged_value_t* el = (eshkol_tagged_value_t*)(void*)(dst + 8);
    for (int64_t i = 0; i < n; i++) {
        if (i == j) el[i] = *v;
        else eshkol_ad_point_element(point, i, &el[i]);
    }
    memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_HEAP_PTR;
    out->data.ptr_val = (uint64_t)(uintptr_t)dst;
}

/**
 * @brief Store column `j` (the forward derivative of every output along
 *        coordinate j) into the m x n Jacobian held in `*jac`, allocating it on
 *        the first column. The Jacobian's elements are tagged numbers, so an
 *        entry that is a carrier of an enclosing level stays one.
 */
void eshkol_ad_jacobian_store_column(arena_t* arena, eshkol_tagged_value_t* jac,
                                     int64_t n, int64_t j,
                                     const eshkol_tagged_value_t* column) {
    if (!arena) arena = get_global_arena();
    int64_t m = (tagged_is_heap_subtype(column, HEAP_SUBTYPE_VECTOR) ||
                 tagged_is_heap_subtype(column, HEAP_SUBTYPE_TENSOR) ||
                 tagged_is_heap_subtype(column, HEAP_SUBTYPE_CONS))
                ? eshkol_ad_point_length(column) : 1;
    eshkol_tensor_t* t = tagged_is_heap_subtype(jac, HEAP_SUBTYPE_TENSOR)
        ? (eshkol_tensor_t*)(uintptr_t)jac->data.ptr_val : NULL;
    if (!t) {
        t = arena_allocate_tensor_with_header(arena);
        if (!t) eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "jacobian: allocation failed");
        t->dimensions = (uint64_t*)arena_allocate(arena, 2 * sizeof(uint64_t));
        t->elements = (int64_t*)arena_allocate(arena, (size_t)(m * n) * sizeof(eshkol_tagged_value_t));
        if (!t->dimensions || !t->elements)
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "jacobian: allocation failed");
        t->dimensions[0] = (uint64_t)m;
        t->dimensions[1] = (uint64_t)n;
        t->num_dimensions = 2;
        t->total_elements = (uint64_t)(m * n);
        t->dtype = ESHKOL_TENSOR_DTYPE_DUAL;
        eshkol_tagged_value_t* el = (eshkol_tagged_value_t*)(void*)t->elements;
        for (int64_t i = 0; i < m * n; i++) el[i] = eshkol_make_double(0.0);
        memset(jac, 0, sizeof(*jac));
        jac->type = ESHKOL_VALUE_HEAP_PTR;
        jac->data.ptr_val = (uint64_t)(uintptr_t)t;
    }
    if ((int64_t)t->dimensions[0] != m)
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
            "jacobian: the function returned outputs of different lengths");
    eshkol_tagged_value_t* el = (eshkol_tagged_value_t*)(void*)t->elements;
    for (int64_t i = 0; i < m; i++) {
        eshkol_tagged_value_t e;
        if (m == 1 && !tagged_is_heap_subtype(column, HEAP_SUBTYPE_VECTOR) &&
            !tagged_is_heap_subtype(column, HEAP_SUBTYPE_TENSOR) &&
            !tagged_is_heap_subtype(column, HEAP_SUBTYPE_CONS)) e = *column;
        else eshkol_ad_point_element(column, i, &e);
        el[i * n + j] = e;
    }
}


/* Element (i, j) of a matrix: a rank-2 tensor (f64 or tagged) or a vector of
 * rows. Out of range reads an exact 0 (a field that does not depend on a
 * coordinate has a vanishing partial). */
static eshkol_tagged_value_t level_matrix_ref(const eshkol_tagged_value_t* m, int64_t i, int64_t j) {
    if (tagged_is_heap_subtype(m, HEAP_SUBTYPE_TENSOR)) {
        const eshkol_tensor_t* t = (const eshkol_tensor_t*)(uintptr_t)m->data.ptr_val;
        if (t->num_dimensions == 2) {
            int64_t rows = (int64_t)t->dimensions[0], cols = (int64_t)t->dimensions[1];
            if (i < 0 || j < 0 || i >= rows || j >= cols) return num_exact_int(0);
            eshkol_tagged_value_t e;
            eshkol_ad_point_element(m, i * cols + j, &e);
            return e;
        }
    }
    if (i < 0 || i >= eshkol_ad_point_length(m)) return num_exact_int(0);
    eshkol_tagged_value_t row;
    eshkol_ad_point_element(m, i, &row);
    if (j < 0 || j >= eshkol_ad_point_length(&row)) return num_exact_int(0);
    eshkol_tagged_value_t e;
    eshkol_ad_point_element(&row, j, &e);
    return e;
}

/** @brief The trace of a matrix whose entries may be carriers (divergence, Laplacian). */
void eshkol_ad_generic_trace(arena_t* arena, const eshkol_tagged_value_t* m,
                             eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    int64_t n = eshkol_ad_point_length(m);
    if (tagged_is_heap_subtype(m, HEAP_SUBTYPE_TENSOR)) {
        const eshkol_tensor_t* t = (const eshkol_tensor_t*)(uintptr_t)m->data.ptr_val;
        if (t->num_dimensions == 2) n = (int64_t)t->dimensions[0];
    }
    eshkol_tagged_value_t acc = num_exact_int(0);
    for (int64_t i = 0; i < n; i++) acc = num_add(arena, acc, level_matrix_ref(m, i, i));
    *out = acc;
}

/** @brief The dot product of two vectors whose entries may be carriers (directional derivative). */
void eshkol_ad_generic_dot(arena_t* arena, const eshkol_tagged_value_t* a,
                           const eshkol_tagged_value_t* b, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    int64_t n = eshkol_ad_point_length(a);
    int64_t nb = eshkol_ad_point_length(b);
    if (nb < n) n = nb;
    eshkol_tagged_value_t acc = num_exact_int(0);
    for (int64_t i = 0; i < n; i++) {
        eshkol_tagged_value_t x, y;
        eshkol_ad_point_element(a, i, &x);
        eshkol_ad_point_element(b, i, &y);
        acc = num_add(arena, acc, num_mul(arena, x, y));
    }
    *out = acc;
}

/** @brief The curl (J21 - J12, J02 - J20, J10 - J01) of a Jacobian whose entries may be carriers. */
void eshkol_ad_generic_curl(arena_t* arena, const eshkol_tagged_value_t* j,
                            eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    uint8_t* dst = (uint8_t*)arena_allocate_vector_with_header(arena, 3);
    if (!dst) eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "curl: allocation failed");
    *(int64_t*)(void*)dst = 3;
    eshkol_tagged_value_t* el = (eshkol_tagged_value_t*)(void*)(dst + 8);
    el[0] = num_sub(arena, level_matrix_ref(j, 2, 1), level_matrix_ref(j, 1, 2));
    el[1] = num_sub(arena, level_matrix_ref(j, 0, 2), level_matrix_ref(j, 2, 0));
    el[2] = num_sub(arena, level_matrix_ref(j, 1, 0), level_matrix_ref(j, 0, 1));
    memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_HEAP_PTR;
    out->data.ptr_val = (uint64_t)(uintptr_t)dst;
}

/**
 * @brief Store entry (i, j) of the rows x cols matrix held in `*m`, allocating a
 *        tagged-element (dual) tensor on the first store (the nested Hessian).
 */
void eshkol_ad_matrix_store(arena_t* arena, eshkol_tagged_value_t* m, int64_t rows,
                            int64_t cols, int64_t i, int64_t j,
                            const eshkol_tagged_value_t* value) {
    if (!arena) arena = get_global_arena();
    eshkol_tensor_t* t = tagged_is_heap_subtype(m, HEAP_SUBTYPE_TENSOR)
        ? (eshkol_tensor_t*)(uintptr_t)m->data.ptr_val : NULL;
    if (!t) {
        t = arena_allocate_tensor_with_header(arena);
        if (!t) eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "hessian: allocation failed");
        t->dimensions = (uint64_t*)arena_allocate(arena, 2 * sizeof(uint64_t));
        t->elements = (int64_t*)arena_allocate(arena, (size_t)(rows * cols) * sizeof(eshkol_tagged_value_t));
        if (!t->dimensions || !t->elements)
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "hessian: allocation failed");
        t->dimensions[0] = (uint64_t)rows;
        t->dimensions[1] = (uint64_t)cols;
        t->num_dimensions = 2;
        t->total_elements = (uint64_t)(rows * cols);
        t->dtype = ESHKOL_TENSOR_DTYPE_DUAL;
        eshkol_tagged_value_t* el = (eshkol_tagged_value_t*)(void*)t->elements;
        for (int64_t k = 0; k < rows * cols; k++) el[k] = eshkol_make_double(0.0);
        memset(m, 0, sizeof(*m));
        m->type = ESHKOL_VALUE_HEAP_PTR;
        m->data.ptr_val = (uint64_t)(uintptr_t)t;
    }
    ((eshkol_tagged_value_t*)(void*)t->elements)[i * cols + j] = *value;
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
 *        inputs that the enclosing pass has not published a seed among.
 *
 * ESH-0096 / ledger SW-05. The runtime-closure gradient's vector arm now
 * detects a point whose components are an ENCLOSING reverse pass's
 * `ad_node_t*` tape nodes and evaluates the inner gradient FORWARD instead --
 * the forward-over-reverse composition the direct `hessian` performs -- so
 * `(jacobian (gradient f) point)` answers the same matrix as
 * `(hessian f point)`.
 *
 * That route needs one thing from the enclosing pass: the component it is
 * currently differentiating with respect to must be published as the active
 * reverse seed (`eshkol_ad_seed_swap`) AND must be one of the point's own
 * components, so the inner jet's ep dimension can carry the dependence and
 * `eshkol_ad_mixed_record` can write it back. When the components are tape
 * nodes but none of them is the active seed -- the enclosing pass computed the
 * point through intermediate tape nodes, or published no seed at all -- the
 * edge cannot be established, and this raises rather than answering a
 * disconnected (silently zero) derivative.
 */
void eshkol_ad_curried_gradient_unsupported(void) {
    eshkol_error(
        "unsupported nested differentiation: a first-class `gradient` closure "
        "reached with reverse-tape inputs that the enclosing pass is not "
        "differentiating directly. `(jacobian (gradient f) point)` and "
        "`(jacobian (lambda (v) ((gradient f) v)) point)` are supported and "
        "answer the Hessian; a point COMPUTED from the enclosing pass's "
        "variables (e.g. `(jacobian (lambda (v) ((gradient f) (vector (* 2.0 "
        "(vector-ref v 0))))) point)`) is not. Use `(hessian f point)` for the "
        "second derivative of a scalar function. Tracked as ESH-0096.");
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
    /* ESH-0411: a body that never touches the seed returns a plain value, not a
     * tower, so every derivative of it VANISHES. That zero used to be hard-coded
     * INEXACT, while the in-tower zero two lines below (n > order_k) is exact --
     * so adding a vanishing partial into an otherwise exact sum (a divergence, a
     * Laplacian, a residual) silently demoted the whole assembly to a double.
     * The zero derivative of an exact constant is exactly 0; R7RS contagion is
     * read off the value that is actually there, exactly as the tower case does. */
    if (!t) {
        *out = (n == 0) ? *tv
             : (tagged_is_exact_number(tv) ? eshkol_make_int64(0, true)
                                           : eshkol_make_double(0.0));
        return;
    }
    if (n > t->order_k) {
        *out = taylor_is_exact(t) ? eshkol_make_int64(0, true) : eshkol_make_double(0.0);
        return;
    }
    if (taylor_is_exact(t) || t->exact_c) {
        eshkol_tagged_value_t fact = eshkol_make_int64(1, true);
        for (uint32_t i = 2; i <= n; i++) fact = exact_mul(arena, fact, eshkol_make_int64((int64_t)i, true));
        *out = exact_mul(arena, fact, taylor_exact_c_const(t)[n]);
        return;
    }
    *out = eshkol_make_double(factorial_d(n) * t->c[n]);
}

extern void* eshkol_ad_mixed_record_tagged(
    void* arena, void* tape, const eshkol_tagged_value_t* value,
    const eshkol_tagged_value_t* dseed);

/* Coefficient k of a tower that carries the reverse-seed companion (P5),
 * restated as a jet {c[k], tangent[k]} so the enclosing forward gradient
 * reads d(c[k])/d(seed) from the jet's first slot. A tower without the
 * companion answers the plain coefficient. */
static void taylor_coefficient_with_seed(arena_t* arena,
                                         const eshkol_tagged_value_t* tv,
                                         uint32_t k,
                                         eshkol_tagged_value_t* out) {
    esh_taylor_t* t = tagged_as_taylor(tv);
    if (!t) { *out = k == 0 ? *tv : eshkol_make_double(0.0); return; }
    const int exact = taylor_is_exact(t) || t->exact_c;
    eshkol_tagged_value_t value = k <= t->order_k
        ? (exact ? taylor_exact_c_const(t)[k] : eshkol_make_double(t->c[k]))
        : (exact ? eshkol_make_int64(0, true) : eshkol_make_double(0.0));
    if (!ESH_TAYLOR_HAS_TANGENT(t->flags)) { *out = value; return; }
    const eshkol_tagged_value_t* exact_tangent = taylor_tan_exact_const(t);
    eshkol_tagged_value_t tangent = k <= t->order_k
        ? (exact_tangent ? exact_tangent[k] : eshkol_make_double(taylor_tan(t)[k]))
        : eshkol_make_double(0.0);
    double* dual = (double*)arena_allocate_aligned(
        arena, ESHKOL_DUAL_HEAP_PAYLOAD_SIZE, 16u);
    if (!dual) { *out = eshkol_make_double(0.0); return; }
    memset(dual, 0, ESHKOL_DUAL_HEAP_PAYLOAD_SIZE);
    dual[0] = tagged_any_to_double(&value);
    dual[1] = tagged_any_to_double(&tangent);
    memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_DUAL_NUMBER;
    out->data.ptr_val = (uint64_t)(uintptr_t)dual;
}

/* Differentiate a tower: (f')_k = (k+1) * c_{k+1}. Preserves order/epoch;
 * the top coefficient becomes 0. Non-towers differentiate to 0. P6: an EXACT
 * tower differentiates to an EXACT tower ((k+1) is a plain exact int64). */
void eshkol_taylor_shift(arena_t* arena, const eshkol_tagged_value_t* tv,
                         eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    esh_taylor_t* t = tagged_as_taylor(tv);
    /* ESH-0411: d/dx of a value that does not carry the perturbation is 0 -- and
     * exactly 0 when that value is exact (see eshkol_taylor_extract_tagged). */
    if (!t) {
        *out = tagged_is_exact_number(tv) ? eshkol_make_int64(0, true)
                                          : eshkol_make_double(0.0);
        return;
    }
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
                               int32_t order_k_in, void* tape,
                               eshkol_tagged_value_t* out) {
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
        if (t && taylor_is_level(t)) {
            /* ADR-0027: a level's coefficients are numbers of the enclosing
             * levels, returned as they are. */
            cv = (uint32_t)k <= t->order_k ? level_c_const(t)[k] : eshkol_make_int64(0, true);
        } else if (t && ESH_TAYLOR_HAS_TANGENT(t->flags) &&
            (uint32_t)k <= t->order_k) {
            /* P5: the reverse-seed companion. With a live tape each
             * coefficient is recorded as an exact local linearisation;
             * without one it is restated as a jet for the forward gradient. */
            void* node = NULL;
            if (tape) {
                eshkol_tagged_value_t value = exact
                    ? taylor_exact_c_const(t)[k] : eshkol_make_double(t->c[k]);
                const eshkol_tagged_value_t* exact_tangent =
                    taylor_tan_exact_const(t);
                eshkol_tagged_value_t tangent = exact_tangent
                    ? exact_tangent[k] : eshkol_make_double(taylor_tan(t)[k]);
                node = eshkol_ad_mixed_record_tagged(arena, tape, &value, &tangent);
            }
            if (node) {
                memset(&cv, 0, sizeof(cv));
                cv.type = ESHKOL_VALUE_CALLABLE;
                cv.data.ptr_val = (uint64_t)(uintptr_t)node;
            } else {
                taylor_coefficient_with_seed(arena, tv, (uint32_t)k, &cv);
            }
        } else if (t) {
            if ((uint32_t)k <= t->order_k) {
                cv = exact ? taylor_exact_c_const(t)[k] : eshkol_make_double(t->c[k]);
            } else {
                cv = exact ? eshkol_make_int64(0, true) : eshkol_make_double(0.0);
            }
        } else {
            /* ESH-0411: same rule as eshkol_taylor_extract_tagged -- the
             * vanishing coefficients of a seed-independent value stay exact
             * when the value is exact. */
            cv = (k == 0) ? *tv
               : (tagged_is_exact_number(tv) ? eshkol_make_int64(0, true)
                                             : eshkol_make_double(0.0));
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

/* Zip independently extracted real/imaginary coefficient carriers without
 * erasing their enclosing derivatives. Both lists come from the same order. */
void eshkol_complex_coefficients_zip(arena_t* arena,
                                     const eshkol_tagged_value_t* real_list,
                                     const eshkol_tagged_value_t* imag_list,
                                     eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    eshkol_tagged_value_t nil = {0};
    nil.type = ESHKOL_VALUE_NULL;
    *out = nil;
    eshkol_tagged_value_t r = *real_list, i = *imag_list;
    arena_tagged_cons_cell_t* tail = NULL;
    while ((r.type & 0x0F) != ESHKOL_VALUE_NULL ||
           (i.type & 0x0F) != ESHKOL_VALUE_NULL) {
        if ((r.type & 0x0F) != ESHKOL_VALUE_HEAP_PTR ||
            (i.type & 0x0F) != ESHKOL_VALUE_HEAP_PTR ||
            !r.data.ptr_val || !i.data.ptr_val) {
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "complex Taylor coefficient lists disagree");
            return;
        }
        const arena_tagged_cons_cell_t* rc = (const arena_tagged_cons_cell_t*)(uintptr_t)r.data.ptr_val;
        const arena_tagged_cons_cell_t* ic = (const arena_tagged_cons_cell_t*)(uintptr_t)i.data.ptr_val;
        eshkol_complex_carrier_t* value = (eshkol_complex_carrier_t*)arena_allocate(arena, sizeof(*value));
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!value || !cell) {
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "complex Taylor coefficient allocation failed");
            return;
        }
        int32_t real_ok = 0, imag_ok = 0;
        value->primal.real = eshkol_ad_seed_to_double(&rc->car, &real_ok);
        value->primal.imag = eshkol_ad_seed_to_double(&ic->car, &imag_ok);
        if (!real_ok || !imag_ok) {
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "complex Taylor coefficient is not numeric");
            return;
        }
        value->real = rc->car;
        value->imag = ic->car;
        cell->car = eshkol_make_complex((uint64_t)(uintptr_t)value);
        cell->car.flags |= ESHKOL_COMPLEX_CARRIER_FLAG;
        cell->cdr = nil;
        eshkol_tagged_value_t link = nil;
        link.type = ESHKOL_VALUE_HEAP_PTR;
        link.data.ptr_val = (uint64_t)(uintptr_t)cell;
        if (tail) tail->cdr = link;
        else *out = link;
        tail = cell;
        r = rc->cdr;
        i = ic->cdr;
    }
}

#ifdef __cplusplus
}  /* extern "C" */
#endif
