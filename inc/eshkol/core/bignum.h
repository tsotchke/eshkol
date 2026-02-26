/*
 * Bignum (Arbitrary-Precision Integer) for Eshkol
 *
 * R7RS requires exact integers of arbitrary size. This implementation
 * uses an array of 64-bit "limbs" stored little-endian (least significant
 * first). Bignums are arena-allocated and immutable — all operations
 * return new bignums.
 *
 * Representation:
 *   [eshkol_object_header_t][eshkol_bignum_t][limbs...]
 *   subtype = HEAP_SUBTYPE_BIGNUM (11)
 *
 * The tagged value stores the pointer to eshkol_bignum_t (after header).
 * Type tag = ESHKOL_VALUE_HEAP_PTR (8).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_BIGNUM_H
#define ESHKOL_CORE_BIGNUM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct arena arena_t;

/*
 * Bignum structure. Immediately followed in memory by `num_limbs` uint64_t limbs.
 * Limbs are stored little-endian: limbs[0] is least significant.
 * The number is: sign * sum(limbs[i] * 2^(64*i), i=0..num_limbs-1)
 * Zero is represented as num_limbs=1, limbs[0]=0, sign=0.
 */
typedef struct eshkol_bignum {
    int32_t  sign;       /* 0 = non-negative, 1 = negative */
    uint32_t num_limbs;  /* Number of 64-bit limbs */
    /* uint64_t limbs[] follows in memory */
} eshkol_bignum_t;

/* Access limbs array (immediately after the struct) */
#define BIGNUM_LIMBS(bn) ((uint64_t*)((uint8_t*)(bn) + sizeof(eshkol_bignum_t)))

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Construction ===== */

/* Create bignum from int64 */
eshkol_bignum_t* eshkol_bignum_from_int64(arena_t* arena, int64_t value);

/* Create bignum from two int64 operands that overflowed.
 * Computes the correct result for: a op b where op is +, -, or *.
 * op: 0=add, 1=sub, 2=mul */
eshkol_bignum_t* eshkol_bignum_from_overflow(arena_t* arena, int64_t a, int64_t b, int op);

/* Create bignum from string (decimal). Returns NULL on parse error. */
eshkol_bignum_t* eshkol_bignum_from_string(arena_t* arena, const char* str, size_t len);

/* ===== Arithmetic ===== */

eshkol_bignum_t* eshkol_bignum_add(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_sub(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_mul(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_div(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_mod(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_neg(arena_t* arena, const eshkol_bignum_t* a);

/* ===== Mixed int64/bignum arithmetic ===== */
/* These are called from codegen when one operand is int64 and the other is bignum */

eshkol_bignum_t* eshkol_bignum_add_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);
eshkol_bignum_t* eshkol_bignum_sub_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);
eshkol_bignum_t* eshkol_bignum_mul_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);

/* ===== Comparison ===== */

/* Returns -1, 0, or 1 */
int eshkol_bignum_compare(const eshkol_bignum_t* a, const eshkol_bignum_t* b);
int eshkol_bignum_compare_int64(const eshkol_bignum_t* a, int64_t b);

/* ===== Predicates ===== */

bool eshkol_bignum_is_zero(const eshkol_bignum_t* a);
bool eshkol_bignum_is_negative(const eshkol_bignum_t* a);
bool eshkol_bignum_is_positive(const eshkol_bignum_t* a);
bool eshkol_bignum_is_even(const eshkol_bignum_t* a);
bool eshkol_bignum_is_odd(const eshkol_bignum_t* a);

/* Check if bignum fits in int64. If so, store the value and return true. */
bool eshkol_bignum_fits_int64(const eshkol_bignum_t* a, int64_t* out);

/* ===== Conversion ===== */

/* Convert to double (may lose precision for very large values) */
double eshkol_bignum_to_double(const eshkol_bignum_t* a);

/* Convert to decimal string. Returns arena-allocated null-terminated string.
 * The returned pointer points past an object header. */
char* eshkol_bignum_to_string(arena_t* arena, const eshkol_bignum_t* a);

/* ===== Tagged Value Dispatch ===== */
/* Called from LLVM codegen. Accepts tagged value pointers, handles type
 * checking and int64->bignum promotion internally.
 * op: 0=add, 1=sub, 2=mul, 3=div, 4=mod, 5=quotient, 6=remainder, 7=neg */
struct eshkol_tagged_value;
typedef struct eshkol_tagged_value eshkol_tagged_value_t;

void eshkol_bignum_binary_tagged(arena_t* arena,
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result);

/* op: 0=lt, 1=gt, 2=eq, 3=le, 4=ge */
void eshkol_bignum_compare_tagged(
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result);

bool eshkol_is_bignum_tagged(const eshkol_tagged_value_t* val);

/* ===== Bitwise Operations ===== */
/* R7RS two's complement semantics for negative bignums */

eshkol_bignum_t* eshkol_bignum_bitwise_and(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_bitwise_or(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_bitwise_xor(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
eshkol_bignum_t* eshkol_bignum_bitwise_not(arena_t* arena, const eshkol_bignum_t* a);
eshkol_bignum_t* eshkol_bignum_shift(arena_t* arena, const eshkol_bignum_t* a, int64_t count);

/* Tagged value dispatch for bitwise ops.
 * op: 0=and, 1=or, 2=xor, 3=not, 4=arithmetic-shift */
void eshkol_bignum_bitwise_tagged(arena_t* arena,
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result);

/* ===== Exponentiation ===== */

/* Exact bignum exponentiation via repeated squaring: O(log n) multiplications */
eshkol_bignum_t* eshkol_bignum_pow(arena_t* arena, const eshkol_bignum_t* base, uint64_t exp);

/* Tagged value dispatch for expt.
 * If both operands are exact integers and exponent >= 0, returns exact bignum.
 * Otherwise falls back to double pow(). */
void eshkol_bignum_pow_tagged(arena_t* arena,
    const eshkol_tagged_value_t* base, const eshkol_tagged_value_t* exponent,
    eshkol_tagged_value_t* result);

/* ===== String Conversion ===== */

/* Parse a string to a tagged number value.
 * Handles integers (int64 or bignum for overflow) and doubles.
 * Result is written to *result as a tagged value. */
void eshkol_string_to_number_tagged(arena_t* arena, const char* str,
    eshkol_tagged_value_t* result);

/* ===== Display ===== */

/* Write decimal representation to file (for display/write) */
void eshkol_bignum_display(const eshkol_bignum_t* a, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_BIGNUM_H */
