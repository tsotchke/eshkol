/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Rational number support for Eshkol (R7RS exact rationals)
 *
 * Rationals are stored as heap-allocated pairs (numerator, denominator)
 * with HEAP_SUBTYPE_RATIONAL header. Always kept in reduced form with
 * positive denominator.
 */
#ifndef ESHKOL_CORE_RATIONAL_H
#define ESHKOL_CORE_RATIONAL_H

#include <eshkol/eshkol.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rational struct: 16 bytes, always GCD-reduced, denominator > 0 */
typedef struct {
    int64_t numerator;
    int64_t denominator;
} eshkol_rational_t;

/* Create a rational from numerator/denominator, auto-reduces via GCD */
void* eshkol_rational_create(void* arena, int64_t num, int64_t denom);

/* Arithmetic: returns arena-allocated rational pointer */
void* eshkol_rational_add(void* arena, void* a, void* b);
void* eshkol_rational_sub(void* arena, void* a, void* b);
void* eshkol_rational_mul(void* arena, void* a, void* b);
void* eshkol_rational_div(void* arena, void* a, void* b);

/* Comparison: returns -1, 0, or 1 */
int eshkol_rational_compare(void* a, void* b);

/* Extract fields */
int64_t eshkol_rational_numerator(void* r);
int64_t eshkol_rational_denominator(void* r);

/* Convert to double (exact->inexact) */
double eshkol_rational_to_double(void* r);

/* Check if integer (denominator == 1) */
int eshkol_rational_is_integer(void* r);

/* Format rational as "num/denom" string (arena-allocated with string header) */
char* eshkol_rational_to_string(void* arena, void* r);

/* Tagged value dispatch: binary op on two tagged values that may be int/rational/double
 * op: 0=add, 1=sub, 2=mul, 3=div
 * Returns tagged value (rational if both exact, double if either inexact)
 */
eshkol_tagged_value_t eshkol_rational_binary_tagged(
    void* arena, eshkol_tagged_value_t a, eshkol_tagged_value_t b, int op);

/* Check if a tagged value is a rational */
int eshkol_is_rational_tagged(eshkol_tagged_value_t val);

/* Pointer-based versions for LLVM codegen (avoids ABI issues with by-value structs) */
int eshkol_is_rational_tagged_ptr(const eshkol_tagged_value_t* val);
void eshkol_rational_binary_tagged_ptr(
    void* arena, const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b,
    int op, eshkol_tagged_value_t* result);

/* Pointer-based comparison for LLVM codegen
 * op: 0=lt, 1=gt, 2=eq, 3=le, 4=ge
 * Handles int/rational mixed operands, writes boolean tagged value to result */
void eshkol_rational_compare_tagged_ptr(
    void* arena, const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b,
    int op, eshkol_tagged_value_t* result);

/* R7RS rationalize: find simplest rational within epsilon of x.
 * x and epsilon are tagged values (int, double, or rational).
 * Writes result tagged value (rational or integer). */
void eshkol_rationalize_tagged(
    void* arena, const eshkol_tagged_value_t* x, const eshkol_tagged_value_t* epsilon,
    eshkol_tagged_value_t* result);

/* Rounding functions for exact rationals — return exact int64 results.
 * floor: toward -infinity, ceil: toward +infinity,
 * truncate: toward zero, round: nearest (ties to even) */
int64_t eshkol_rational_floor(void* r);
int64_t eshkol_rational_ceil(void* r);
int64_t eshkol_rational_truncate(void* r);
int64_t eshkol_rational_round(void* r);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_RATIONAL_H */
