/**
 * @file vm_numeric.h
 * @brief Unified numeric type definitions for the Eshkol bytecode VM.
 *
 * Defines the full numeric tower: int64 → bignum → rational → float64 → complex → dual.
 * All types arena-allocated via vm_arena.h. No GC.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_NUMERIC_H
#define VM_NUMERIC_H

#include <stdint.h>
#include <math.h>
#include "vm_arena.h"
#include "eshkol/core/i128.h"   /* shared pure i128 math (same core as native runtime) */

/* ── Extended Value Types (beyond base VM's VAL_INT/FLOAT/BOOL/PAIR/CLOSURE) ──
 *
 * Values 8-9 and 14 fill gaps in the base ValType enum (0-7, 15).
 * Values 16+ extend beyond VAL_CONTINUATION (15).
 *
 * Each heap-allocated opaque type gets its own ValType so that print_value
 * and type predicates can dispatch on .type without inspecting the heap.
 * The .as.ptr field (union with .as.i) is the heap object index in all cases.
 */
#define VAL_TENSOR       8   /* heap-allocated VmTensor   (opaque)           */
#define VAL_KB           9   /* heap-allocated VmKB       (opaque)           */
#define VAL_COMPLEX     10   /* heap-allocated VmComplex  (opaque)           */
#define VAL_RATIONAL    11   /* heap-allocated VmRational (opaque)           */
#define VAL_BIGNUM      12   /* heap-allocated VmBignum   (opaque)           */
#define VAL_DUAL        13   /* heap-allocated VmDual     (opaque)           */
#define VAL_FACTOR_GRAPH 14  /* heap-allocated VmFactorGraph (opaque)       */
/* 15 = VAL_CONTINUATION (defined in vm_core.c ValType enum)                */
#define VAL_WORKSPACE   16   /* heap-allocated VmWorkspace   (opaque)       */
#define VAL_SUBST       17   /* heap-allocated VmSubstitution (opaque)      */
#define VAL_HASH        18   /* heap-allocated hash table    (opaque)       */
#define VAL_BYTEVECTOR  19   /* heap-allocated bytevector    (opaque)       */
#define VAL_PARAMETER_OBJ 20 /* heap-allocated dynamic parameter (opaque)  */
#define VAL_AD_TAPE     21   /* heap-allocated AD tape       (opaque)       */
#define VAL_ERROR_OBJ   22   /* heap-allocated error object  (opaque)      */
#define VAL_MANIFOLD    23   /* heap-allocated Riemannian manifold (opaque) */
#define VAL_PORT        24   /* heap-allocated I/O port      (opaque)       */
#define VAL_VOID        25   /* unspecified return value (display, newline)  */
#define VAL_HYPER_DUAL  26   /* heap-allocated VmHyperDual (opaque)          */
#define VAL_RIEMANNIAN_ADAM_STATE 27 /* heap-allocated optimizer state       */
#define VAL_FUTURE      28   /* heap-allocated standalone VM future handle   */
#define VAL_CHAR        29   /* immediate Unicode character (codepoint in .as.i) */
#define VAL_MULTI_VALUE 30   /* opaque R7RS multiple-values packet            */
#define VAL_SYMBOL      31   /* heap-backed internable Scheme symbol spelling */
#define VAL_EOF         32   /* distinct end-of-file object                    */
#define VAL_I128        33   /* heap-allocated native 128-bit integer (opaque) */

/* ── Heap Subtypes ── */
#define VM_SUBTYPE_COMPLEX   5
#define VM_SUBTYPE_RATIONAL  6
#define VM_SUBTYPE_BIGNUM    7
#define VM_SUBTYPE_DUAL      8
#define VM_SUBTYPE_TENSOR    9
#define VM_SUBTYPE_LOGIC_VAR 10
#define VM_SUBTYPE_SUBST     11
#define VM_SUBTYPE_FACT      12
#define VM_SUBTYPE_KB        13
#define VM_SUBTYPE_FG        14  /* factor graph */
#define VM_SUBTYPE_WORKSPACE 15
#define VM_SUBTYPE_PORT      16
#define VM_SUBTYPE_AD_TAPE   17
#define VM_SUBTYPE_PROMISE   18
#define VM_SUBTYPE_HASH      20
#define VM_SUBTYPE_ERROR     21
#define VM_SUBTYPE_BYTEVEC   22
#define VM_SUBTYPE_PARAMETER 23
#define VM_SUBTYPE_MULTI_VAL 4
#define VM_SUBTYPE_FUTURE    26
#define VM_SUBTYPE_I128      27

/* ── Complex Number ── */
typedef struct {
    double real;
    double imag;
} VmComplex;

/* ── Bignum (sign-magnitude, base 2^32 limbs, little-endian) ── */
typedef struct {
    int sign;           /* -1, 0, or 1 */
    uint32_t* limbs;    /* arena-allocated */
    int n_limbs;
    int capacity;
} VmBignum;

/* ── Rational Number (always normalized: gcd(|num|,denom)=1, denom>0) ──
 *
 * SW-18 / ESH-0105: this used to be a bare int64/int64 pair, so the moment an
 * exact operation produced a numerator or denominator outside int64 the VM
 * pushed the correctly-rounded DOUBLE instead — `(/ (expt 2 100) 3)` printed
 * 4.2255020007607644e+29 where the native engine prints the exact
 * 1267650600228229401496703205376/3, with exit 0 and no diagnostic.  The
 * representation now mirrors the native runtime's eshkol_rational_t
 * (inc/eshkol/core/rational.h), which PR #247 already made bignum-capable:
 *
 *   is_big == 0  fast path — num/denom hold the reduced int64 pair
 *   is_big == 1  exact path — big_num/big_den hold the reduced bignum pair
 *
 * The representation is CANONICAL: a value is stored big only when the reduced
 * pair genuinely does not fit int64, so equality and eqv? stay a field
 * comparison and every existing int64 fast path keeps working unchanged. */
typedef struct {
    int64_t num;        /* valid iff is_big == 0 */
    int64_t denom;      /* valid iff is_big == 0; > 0 */
    int32_t is_big;     /* 0 = int64 fast path, 1 = bignum path */
    VmBignum* big_num;  /* valid iff is_big == 1 */
    VmBignum* big_den;  /* valid iff is_big == 1; > 0 */
} VmRational;

/* ── Dual Number (forward-mode AD: primal + tangent*epsilon) ──
 *
 * SW-85: the two `double` fields used to be the WHOLE carrier, and that made
 * the VM answer inexactly at an exact point where native answers exactly:
 * `(derivative (lambda (x) (* x x)) 1/3)` was 0.6666666666666666 on the VM and
 * 2/3 on native. The exactness was lost at the SEED — a rational point has
 * nowhere to live in a double — not in the arithmetic.
 *
 * The fix is a HYBRID carrier rather than a second dual type. `eprimal` and
 * `etangent` are the exact halves; NULL means "this half is inexact", which is
 * the R7RS exactness-contagion state and also the state every pre-existing
 * construction site produces for free. The doubles are ALWAYS maintained, so
 * the ~40 sites that read `d->primal` / `d->tangent` keep working untouched and
 * a caller that does not know about exactness cannot observe the change.
 *
 * VmRational already spans int64 and bignum through its `is_big` field, so one
 * pointer type covers exact integers, big integers and rationals alike.
 *
 * INVARIANT: if a half is non-NULL, its VmRational value EQUALS the double in
 * the same slot up to double rounding, and the exact half is authoritative.
 * Only the exactness-preserving ops (+ - * / and integer expt) propagate the
 * exact halves; every transcendental leaves them NULL, which is exactly the
 * demotion native's COEFF_F64 tower performs.
 *
 * Taylor carriers additionally carry a value perturbation epoch. Nested Taylor
 * operations combine coefficients only when epochs match; a foreign epoch is
 * lifted as a constant. While a nested pass is active, the optional
 * tangent_coeff array carries the orthogonal outer first-order perturbation.
 *
 * NOTE for the region evacuator: a dual carrying exact halves owns INTERIOR
 * arena pointers, so it is no longer a leaf — see vm_region_evac.c. */
#define VM_DUAL_KIND_SCALAR  0u
#define VM_DUAL_KIND_TAYLOR  1u

typedef struct {
    double primal;
    double tangent;
    VmRational* eprimal;   /* NULL = primal is inexact  */
    VmRational* etangent;  /* NULL = tangent is inexact */
    /* A Taylor tower uses the same VAL_DUAL/HEAP_DUAL envelope so the VM's
     * existing arithmetic dispatch remains one closed carrier family. */
    uint32_t kind;         /* VM_DUAL_KIND_SCALAR or VM_DUAL_KIND_TAYLOR */
    uint32_t order;        /* highest coefficient index for a Taylor tower */
    uint32_t epoch;        /* perturbation epoch; 0 for scalar duals */
    int32_t primal_sign;   /* exact sign hint when the double primal underflows */
    double* coeff;         /* c[0..order], present for VM_DUAL_KIND_TAYLOR */
    VmRational** exact_coeff; /* optional exact c[0..order] parallel array */
    double* tangent_coeff; /* optional d(c[k])/d(seed), for nested Taylor */
} VmDual;

/* ── Exact-arithmetic surface shared by the rational tower and the AD dual ──
 * (SW-85) These were file-static in vm_rational.c and reachable only from the
 * native-call dispatcher; the forward-mode dual needs the SAME arithmetic, so
 * that "exact" means one thing on this substrate rather than two. Declared
 * here rather than duplicated, so vm_dual.c cannot drift from the tower. */
VmRational* vm_rational_op_exact(VmRegionStack *rs, const VmRational *a,
                                 const VmRational *b, char op);
VmRational* vm_rational_negate_exact(VmRegionStack *rs, const VmRational *a);
VmRational* vm_rational_absolute_exact(VmRegionStack *rs, const VmRational *a);
VmRational* vm_rational_from_bignum(VmRegionStack *rs, VmBignum *n);
VmRational* vm_rational_from_double_exact(VmRegionStack *rs, double d);
VmRational* vm_rational_from_int(VmArena *arena, int64_t n);
VmRational* vm_rational_make(VmArena *arena, int64_t num, int64_t denom);
double      vm_rational_to_double(const VmRational *r);
int         vm_rational_is_zero(const VmRational *r);
int         vm_rational_sign(const VmRational *r);
int         vm_rational_compare_exact_values(VmRegionStack *rs,
                                              const VmRational *a,
                                              const VmRational *b);

/* ── Forward-mode dual: exact seed and exact extraction (SW-85) ──
 * The only cross-translation-unit surface the exact halves need. Everything
 * else about exactness is decided inside vm_dual.c. */
VmDual*     vm_dual_make_exact_seed(VmRegionStack* rs, VmRational* point);
VmRational* vm_dual_exact_tangent(const VmDual* d);
VmRational* vm_dual_exact_primal(const VmDual* d);
VmDual*     vm_dual_make_taylor_seed(VmRegionStack* rs, VmRational* point,
                                     double point_value, uint32_t order,
                                     int exact, uint32_t epoch);
uint32_t    vm_dual_next_taylor_epoch(void);
VmDual*     vm_dual_make_taylor_ride_seed(VmRegionStack* rs,
                                           const VmDual* outer);
VmDual*     vm_dual_make_taylor_carry_seed(VmRegionStack* rs,
                                            const VmDual* outer,
                                            uint32_t order);
VmDual*     vm_dual_taylor_promote_tangent(VmRegionStack* rs,
                                           const VmDual* result);
VmDual*     vm_dual_taylor_carry_result(VmRegionStack* rs,
                                         const VmDual* result,
                                         uint32_t order,
                                         uint32_t outer_epoch);
int         vm_dual_is_taylor(const VmDual* d);
int         vm_dual_taylor_is_exact(const VmDual* d);
double      vm_dual_taylor_coeff(const VmDual* d, uint32_t n);
VmRational* vm_dual_taylor_exact_coeff(const VmDual* d, uint32_t n);
VmRational* vm_dual_taylor_exact_derivative(VmRegionStack* rs,
                                             const VmDual* d, uint32_t n);

/* ── Hyper-Dual Number (exact second derivatives via ε₁, ε₂)  ──
 * h = f + f₁·ε₁ + f₂·ε₂ + f₁₂·ε₁ε₂
 * where ε₁² = ε₂² = 0.
 * f₁₂ gives the exact mixed partial ∂²f/∂x₁∂x₂.
 * For scalar hessian: seed (x, 1, 1, 0) → f₁₂ = f''(x). */
typedef struct {
    double f;      /* function value */
    double f1;     /* ∂f/∂ε₁ */
    double f2;     /* ∂f/∂ε₂ */
    double f12;    /* ∂²f/∂ε₁∂ε₂ = exact second derivative */
} VmHyperDual;

/* ── Numeric Tower Promotion ──
 * int64 < bignum < rational < float64 < complex < dual
 * Promotion rules:
 *   int + int    → int (overflow → bignum)
 *   int + float  → float
 *   int + rational → rational
 *   int + complex → complex
 *   float + rational → float (R7RS: exact + inexact → inexact)
 *   any + dual → dual
 *   complex + dual → dual-of-complex (future)
 */

/* ── Native Call ID Ranges ── */
#define VM_NATIVE_COMPLEX_BASE   300
#define VM_NATIVE_RATIONAL_BASE  330
#define VM_NATIVE_BIGNUM_BASE    350
#define VM_NATIVE_DUAL_BASE      370
#define VM_NATIVE_HYPER_DUAL_BASE 1900
#define VM_SUBTYPE_HYPER_DUAL    14
#define VM_NATIVE_AD_BASE        390
#define VM_NATIVE_TENSOR_BASE    410
#define VM_NATIVE_TENSOR_OP_BASE 440
#define VM_NATIVE_LOGIC_BASE     500
#define VM_NATIVE_INFERENCE_BASE 520
#define VM_NATIVE_WORKSPACE_BASE 540
#define VM_NATIVE_STRING_BASE    550
#define VM_NATIVE_IO_BASE        580
#define VM_NATIVE_PARALLEL_BASE  620
#define VM_NATIVE_MULTIVAL_BASE  650
#define VM_NATIVE_HASH_BASE      660
#define VM_NATIVE_BYTEVEC_BASE   680
#define VM_NATIVE_PARAM_BASE     700
#define VM_NATIVE_ERROR_BASE     710
#define VM_NATIVE_I128_BASE      2100  /* native fixed-width 128-bit integer (2100-2118) */

#endif /* VM_NUMERIC_H */
