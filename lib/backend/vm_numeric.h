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

/* ── Extended Value Types (beyond base VM's VAL_INT/FLOAT/BOOL/PAIR/CLOSURE) ── */
#define VAL_COMPLEX   10
#define VAL_RATIONAL  11
#define VAL_BIGNUM    12
#define VAL_DUAL      13

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

/* ── Complex Number ── */
typedef struct {
    double real;
    double imag;
} VmComplex;

/* ── Rational Number (always normalized: gcd(|num|,denom)=1, denom>0) ── */
typedef struct {
    int64_t num;
    int64_t denom;
} VmRational;

/* ── Bignum (sign-magnitude, base 2^32 limbs, little-endian) ── */
typedef struct {
    int sign;           /* -1, 0, or 1 */
    uint32_t* limbs;    /* arena-allocated */
    int n_limbs;
    int capacity;
} VmBignum;

/* ── Dual Number (forward-mode AD: primal + tangent*ε) ── */
typedef struct {
    double primal;
    double tangent;
} VmDual;

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

#endif /* VM_NUMERIC_H */
