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
#include <eshkol/core/bignum.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bignum-capable exact rational.
 *
 * Two representations, chosen canonically so equal values share one form:
 *   - is_big == 0 (fast path): numerator/denominator hold the reduced int64
 *     pair, denominator > 0. Used whenever both fit in int64.
 *   - is_big == 1 (bignum path): big_num/big_den hold the reduced bignum
 *     pair (big_den > 0). Used ONLY when the reduced numerator or denominator
 *     does not fit in int64. In this case numerator/denominator are unused
 *     (set to 0/1).
 *
 * Invariant (both paths): the fraction is fully GCD-reduced with a positive
 * denominator, and the int64 fast path is preferred whenever representable —
 * so value equality reduces to comparing (is_big, and the active fields).
 *
 * The numerator/denominator int64 fields remain the first two members so the
 * common small-rational read paths keep working unchanged.
 */
typedef struct {
    int64_t numerator;    /* fast-path numerator   (valid iff is_big == 0) */
    int64_t denominator;  /* fast-path denominator (valid iff is_big == 0; > 0) */
    int32_t is_big;       /* 0 = int64 fast path, 1 = bignum path */
    int32_t reserved;
    eshkol_bignum_t* big_num;  /* bignum numerator   (valid iff is_big == 1) */
    eshkol_bignum_t* big_den;  /* bignum denominator (valid iff is_big == 1; > 0) */
} eshkol_rational_t;

/* Create a rational from int64 numerator/denominator, auto-reduces via GCD.
 * Promotes to the bignum path only for the INT64_MIN sign-flip corner. */
void* eshkol_rational_create(void* arena, int64_t num, int64_t denom);

/* Create a rational from two bignum numerator/denominator pointers.
 * Sign-canonicalizes (positive denominator), GCD-reduces via bignum gcd,
 * and demotes to the int64 fast path when the reduced pair fits.
 * Never degrades to double — this is the exact bignum-rational constructor. */
void* eshkol_rational_create_bn(void* arena, eshkol_bignum_t* num, eshkol_bignum_t* denom);

/* Build an EXACT tagged number from bignum numerator/denominator: an INT64 or
 * bare bignum when the reduced denominator is 1, otherwise a rational HEAP_PTR.
 * Used by the bignum division dispatch to preserve exactness (ESH-0105). */
void eshkol_rational_from_bignums_tagged(
    void* arena, eshkol_bignum_t* num, eshkol_bignum_t* denom,
    eshkol_tagged_value_t* result);

/* (make-rational num den) with tagged operands: each of num/den may be an
 * INT64 or a bignum HEAP_PTR (bignum-magnitude literal). Produces an exact
 * INT64/bignum (when the reduced denominator is 1) or a rational HEAP_PTR. */
void eshkol_rational_make_tagged(
    void* arena, const eshkol_tagged_value_t* num, const eshkol_tagged_value_t* den,
    eshkol_tagged_value_t* result);

/* Value equality on reduced rationals (handles both int64 and bignum paths).
 * Used by eqv?/equal? via the deep-equality runtime (ESH-0114). */
int eshkol_rational_equal(void* a, void* b);

/* Write a rational's "n/d" (or bare "n" when integer) form to a FILE*. */
void eshkol_rational_display(void* r, void* file);

/* Tagged numerator/denominator: return an exact INT64 or bignum HEAP_PTR so
 * bignum-magnitude rationals report their true numerator/denominator. */
void eshkol_rational_numerator_tagged(
    void* arena, const eshkol_tagged_value_t* v, eshkol_tagged_value_t* result);
void eshkol_rational_denominator_tagged(
    void* arena, const eshkol_tagged_value_t* v, eshkol_tagged_value_t* result);

/* Arithmetic: returns arena-allocated rational pointer */
void* eshkol_rational_add(void* arena, void* a, void* b);
/**
 * @brief Subtract two rationals: a - b.
 *
 * Computed via 128-bit intermediates to guard against int64 overflow before
 * GCD-reduction.
 *
 * @param arena Arena to allocate the result from.
 * @param a Minuend, a rational pointer (eshkol_rational_t*) produced by
 *          eshkol_rational_create() or another rational operation.
 * @param b Subtrahend, likewise a rational pointer.
 * @return Newly allocated, GCD-reduced rational pointer, or NULL if the
 *         exact result overflows int64 (caller should fall back to double arithmetic).
 */
void* eshkol_rational_sub(void* arena, void* a, void* b);
/**
 * @brief Multiply two rationals: a * b.
 *
 * Computed via 128-bit intermediates to guard against int64 overflow before
 * GCD-reduction.
 *
 * @param arena Arena to allocate the result from.
 * @param a Left rational operand.
 * @param b Right rational operand.
 * @return Newly allocated, GCD-reduced rational pointer, or NULL on int64 overflow.
 */
void* eshkol_rational_mul(void* arena, void* a, void* b);
/**
 * @brief Divide two rationals: a / b.
 *
 * Raises an ESHKOL_EXCEPTION_DIVIDE_BY_ZERO exception (via eshkol_raise) if
 * @p b's numerator is zero.
 *
 * @param arena Arena to allocate the result from.
 * @param a Dividend rational.
 * @param b Divisor rational.
 * @return Newly allocated, GCD-reduced rational pointer, or NULL on int64 overflow.
 */
void* eshkol_rational_div(void* arena, void* a, void* b);

/* Comparison: returns -1, 0, or 1 */
int eshkol_rational_compare(void* a, void* b);

/* Extract fields */
int64_t eshkol_rational_numerator(void* r);
/**
 * @brief Extract the denominator of a rational.
 *
 * @param r Rational pointer (an eshkol_rational_t*).
 * @return The denominator, always strictly positive in the normalized representation.
 */
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
/**
 * @brief Pointer-based binary dispatch for LLVM codegen.
 *
 * Equivalent to eshkol_rational_binary_tagged() but takes and writes tagged
 * values by pointer, avoiding ABI issues with by-value eshkol_tagged_value_t
 * struct passing/returning from generated code.
 *
 * @param arena Arena to allocate any resulting rational from.
 * @param a Left operand (tagged int, double, or rational).
 * @param b Right operand (tagged int, double, or rational).
 * @param op Operation selector: 0=add, 1=sub, 2=mul, 3=div.
 * @param[out] result Tagged value written with the operation's result
 *             (rational or int if both operands are exact, double otherwise).
 */
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
/**
 * @brief Round a rational toward +infinity (ceiling).
 *
 * @param r Rational pointer (an eshkol_rational_t*).
 * @return Exact int64 result of ceil(numerator/denominator).
 */
int64_t eshkol_rational_ceil(void* r);
/**
 * @brief Round a rational toward zero (truncation).
 *
 * @param r Rational pointer (an eshkol_rational_t*).
 * @return Exact int64 result of numerator/denominator, using C truncating division.
 */
int64_t eshkol_rational_truncate(void* r);
/**
 * @brief Round a rational to the nearest integer, with ties rounding to even (banker's rounding).
 *
 * @param r Rational pointer (an eshkol_rational_t*).
 * @return Nearest exact int64 to numerator/denominator, per R7RS round semantics.
 */
int64_t eshkol_rational_round(void* r);

/* Rounding for the FULL rational substrate (SW-29 / ESH-0105).
 *
 * The int64-returning variants above read `numerator`/`denominator` directly.
 * On the bignum path those fields are the inert 0/1 shadow, so every one of
 * them answered 0 for a bignum-backed rational: `(floor (/ (expt 2 100) 3))`
 * printed 0 instead of 422550200076076467165567735125, with exit 0 and no
 * diagnostic. The exact result can itself exceed int64, so it cannot be
 * signalled through an `int64_t` return at all — these write a TAGGED value
 * instead, exactly as eshkol_rational_from_bignums_tagged() already does, and
 * so can answer with an INT64 or a bignum as the value requires.
 *
 * The int64 variants are kept for callers that have already established the
 * operand is on the int64 fast path. */
void eshkol_rational_floor_tagged(void* arena, void* r, eshkol_tagged_value_t* result);
void eshkol_rational_ceil_tagged(void* arena, void* r, eshkol_tagged_value_t* result);
void eshkol_rational_truncate_tagged(void* arena, void* r, eshkol_tagged_value_t* result);
void eshkol_rational_round_tagged(void* arena, void* r, eshkol_tagged_value_t* result);

/**
 * @brief Exact IEEE-754 double to rational (R7RS `inexact->exact`).
 *
 * A finite double is exactly `mantissa * 2^exponent`, and this returns that
 * value — never a nearby approximation. `(inexact->exact 0.1)` is
 * 3602879701896397/36028797018963968, as documented in
 * ESHKOL_LANGUAGE_GUIDE.md; `(exact->inexact (inexact->exact x))` reproduces
 * `x` bit-for-bit for every finite `x`, subnormals included. Whole values come
 * back with denominator 1 and a bignum numerator when they exceed int64.
 *
 * @param arena Arena to allocate the result from.
 * @param d Value to convert; must be finite (a non-finite input reports an
 *          error and yields 0/1, since infinities and NaN have no exact value).
 * @return Newly allocated rational pointer.
 */
void* eshkol_double_to_rational(void* arena, double d);

/**
 * @brief Exact `inexact->exact` producing a tagged value.
 *
 * The entry point both back ends call. Exactness conversion changes the value's
 * SHAPE — a whole double becomes an exact integer (int64, or a bignum beyond
 * int64 range), a fractional one becomes a rational — so the choice is made
 * here once rather than duplicated in each engine's dispatch, where an
 * `fptosi` on a value past 2^63 was undefined behaviour.
 *
 * @param arena Arena to allocate any heap result from.
 * @param d Value to convert.
 * @param out Receives the exact result.
 */
void eshkol_double_to_exact_tagged(void* arena, double d,
                                   eshkol_tagged_value_t* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_RATIONAL_H */
