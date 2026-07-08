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
/**
 * @brief Opaque handle to the arena allocator used for all bignum allocations.
 *
 * Bignums are never allocated with malloc/free; every constructor and
 * arithmetic operation in this API takes an arena_t* and allocates the
 * result (object header + limb array) from it. The full definition lives
 * in the arena allocator implementation.
 */
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

/**
 * @brief Compute a + b for two arbitrary-precision integers.
 *
 * @param arena Arena to allocate the result from.
 * @param a Augend.
 * @param b Addend.
 * @return Newly allocated sum, or NULL if @p a or @p b is NULL.
 */
eshkol_bignum_t* eshkol_bignum_add(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Compute a - b for two arbitrary-precision integers.
 *
 * @param arena Arena to allocate the result from.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return Newly allocated difference, or NULL if @p a or @p b is NULL.
 */
eshkol_bignum_t* eshkol_bignum_sub(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Compute a * b for two arbitrary-precision integers.
 *
 * Uses schoolbook O(n*m) multiplication over the limb arrays.
 *
 * @param arena Arena to allocate the result from.
 * @param a Multiplicand.
 * @param b Multiplier.
 * @return Newly allocated product, or NULL if @p a or @p b is NULL.
 */
eshkol_bignum_t* eshkol_bignum_mul(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Compute truncating integer division a / b.
 *
 * Uses Knuth's Algorithm D for multi-limb divisors. Raises an
 * ESHKOL_EXCEPTION_DIVIDE_BY_ZERO exception (via eshkol_raise) if @p b is zero.
 *
 * @param arena Arena to allocate the result from.
 * @param a Dividend.
 * @param b Divisor.
 * @return Newly allocated quotient, truncated toward zero.
 */
eshkol_bignum_t* eshkol_bignum_div(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Compute a mod b (remainder of truncating division).
 *
 * Raises an ESHKOL_EXCEPTION_DIVIDE_BY_ZERO exception (via eshkol_raise) if
 * @p b is zero. The result's sign follows the sign of @p a, matching C's
 * truncating-division remainder and eshkol_bignum_div().
 *
 * @param arena Arena to allocate the result from.
 * @param a Dividend.
 * @param b Divisor.
 * @return Newly allocated remainder.
 */
eshkol_bignum_t* eshkol_bignum_mod(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Negate an arbitrary-precision integer.
 *
 * @param arena Arena to allocate the result from.
 * @param a Value to negate.
 * @return Newly allocated bignum equal to -a (negating zero yields zero), or NULL if @p a is NULL.
 */
eshkol_bignum_t* eshkol_bignum_neg(arena_t* arena, const eshkol_bignum_t* a);

/* ===== Mixed int64/bignum arithmetic ===== */
/* These are called from codegen when one operand is int64 and the other is bignum */

/**
 * @brief Compute a + b, where b is a plain machine int64.
 *
 * Internally promotes @p b to a bignum and delegates to eshkol_bignum_add().
 *
 * @param arena Arena to allocate the result from.
 * @param a Bignum operand.
 * @param b Int64 operand.
 * @return Newly allocated sum, or NULL on failure.
 */
eshkol_bignum_t* eshkol_bignum_add_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);
/**
 * @brief Compute a - b, where b is a plain machine int64.
 *
 * Internally promotes @p b to a bignum and delegates to eshkol_bignum_sub().
 *
 * @param arena Arena to allocate the result from.
 * @param a Bignum operand.
 * @param b Int64 operand.
 * @return Newly allocated difference, or NULL on failure.
 */
eshkol_bignum_t* eshkol_bignum_sub_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);
/**
 * @brief Compute a * b, where b is a plain machine int64.
 *
 * Internally promotes @p b to a bignum and delegates to eshkol_bignum_mul().
 *
 * @param arena Arena to allocate the result from.
 * @param a Bignum operand.
 * @param b Int64 operand.
 * @return Newly allocated product, or NULL on failure.
 */
eshkol_bignum_t* eshkol_bignum_mul_int64(arena_t* arena, const eshkol_bignum_t* a, int64_t b);

/* ===== Comparison ===== */

/* Returns -1, 0, or 1 */
int eshkol_bignum_compare(const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Compare a bignum against a machine int64 value.
 *
 * Takes a fast path with no allocation for single-limb bignums; otherwise
 * constructs a temporary single-limb bignum for @p b on the stack.
 *
 * @param a Bignum operand.
 * @param b Int64 operand to compare against.
 * @return -1 if a < b, 0 if a == b, 1 if a > b.
 */
int eshkol_bignum_compare_int64(const eshkol_bignum_t* a, int64_t b);

/* ===== Predicates ===== */

/**
 * @brief Test whether a bignum represents zero.
 *
 * A NULL pointer is treated as zero.
 *
 * @param a Bignum to test.
 * @return true if @p a is zero or NULL.
 */
bool eshkol_bignum_is_zero(const eshkol_bignum_t* a);
/**
 * @brief Test whether a bignum is strictly negative.
 *
 * @param a Bignum to test.
 * @return true if @p a is non-NULL, has its sign bit set, and is not zero.
 */
bool eshkol_bignum_is_negative(const eshkol_bignum_t* a);
/**
 * @brief Test whether a bignum is strictly positive.
 *
 * @param a Bignum to test.
 * @return true if @p a is non-NULL, has its sign bit clear, and is not zero.
 */
bool eshkol_bignum_is_positive(const eshkol_bignum_t* a);
/**
 * @brief Test whether a bignum is even.
 *
 * Checks only the least significant bit of the lowest limb. A NULL pointer is treated as even.
 *
 * @param a Bignum to test.
 * @return true if @p a is even.
 */
bool eshkol_bignum_is_even(const eshkol_bignum_t* a);
/**
 * @brief Test whether a bignum is odd.
 *
 * Checks only the least significant bit of the lowest limb. A NULL pointer is treated as not odd.
 *
 * @param a Bignum to test.
 * @return true if @p a is odd.
 */
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
/**
 * @brief Forward declaration of the tagged runtime value type.
 *
 * Declared locally instead of including eshkol.h, to keep this header
 * lightweight for callers that only need bignum construction/arithmetic.
 * The full definition lives in inc/eshkol/eshkol.h.
 */
struct eshkol_tagged_value;
typedef struct eshkol_tagged_value eshkol_tagged_value_t;

/**
 * @brief Dispatch a binary arithmetic operation on two tagged numeric values, using bignum arithmetic.
 *
 * Called from LLVM-generated code. Accepts tagged value pointers (INT64 or
 * bignum HEAP_PTR) and internally promotes int64 operands to bignum as
 * needed. If either operand is a double, the operation is instead computed
 * in double precision and an inexact result is returned (R7RS exact +
 * inexact -> inexact). Exact integer results that fit back into int64 are
 * demoted to avoid unnecessary bignum overhead.
 *
 * @param arena Arena to allocate any intermediate/result bignum from.
 * @param left Left operand (tagged INT64, DOUBLE, or bignum HEAP_PTR).
 * @param right Right operand (tagged INT64, DOUBLE, or bignum HEAP_PTR); ignored when op is unary negation.
 * @param op Operation selector: 0=add, 1=sub, 2=mul, 3=div, 4=mod, 5=quotient, 6=remainder, 7=neg.
 * @param[out] result Tagged value written with the operation's result.
 */
void eshkol_bignum_binary_tagged(arena_t* arena,
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result);

/* op: 0=lt, 1=gt, 2=eq, 3=le, 4=ge */
void eshkol_bignum_compare_tagged(
    const eshkol_tagged_value_t* left, const eshkol_tagged_value_t* right,
    int op, eshkol_tagged_value_t* result);

/**
 * @brief Test whether a tagged value currently holds a bignum.
 *
 * Checks the HEAP_PTR type tag together with the HEAP_SUBTYPE_BIGNUM object
 * header subtype (see ESHKOL_IS_BIGNUM).
 *
 * @param val Tagged value to check.
 * @return true if @p val is a bignum-backed HEAP_PTR value.
 */
bool eshkol_is_bignum_tagged(const eshkol_tagged_value_t* val);

/* ===== Bitwise Operations ===== */
/* R7RS two's complement semantics for negative bignums */

/**
 * @brief Bitwise AND of two bignums, using two's-complement semantics.
 *
 * Negative operands are treated as an infinite-precision two's-complement
 * bit string (R7RS bitwise-and semantics), not as sign+magnitude.
 *
 * @param arena Arena to allocate the result from.
 * @param a Left operand.
 * @param b Right operand.
 * @return Newly allocated bignum, or NULL if any argument is NULL.
 */
eshkol_bignum_t* eshkol_bignum_bitwise_and(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Bitwise OR of two bignums, using two's-complement semantics.
 *
 * @param arena Arena to allocate the result from.
 * @param a Left operand.
 * @param b Right operand.
 * @return Newly allocated bignum, or NULL if any argument is NULL.
 */
eshkol_bignum_t* eshkol_bignum_bitwise_or(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Bitwise XOR of two bignums, using two's-complement semantics.
 *
 * @param arena Arena to allocate the result from.
 * @param a Left operand.
 * @param b Right operand.
 * @return Newly allocated bignum, or NULL if any argument is NULL.
 */
eshkol_bignum_t* eshkol_bignum_bitwise_xor(arena_t* arena, const eshkol_bignum_t* a, const eshkol_bignum_t* b);
/**
 * @brief Bitwise NOT (one's-complement negation) of a bignum.
 *
 * Computed as -(a + 1), matching two's-complement bitwise-not semantics.
 *
 * @param arena Arena to allocate the result from.
 * @param a Operand.
 * @return Newly allocated bignum equal to ~a, or NULL if @p a is NULL.
 */
eshkol_bignum_t* eshkol_bignum_bitwise_not(arena_t* arena, const eshkol_bignum_t* a);
/**
 * @brief Arithmetic bit shift of a bignum by a signed bit count.
 *
 * A positive @p count shifts left (multiplies by 2^count); a negative
 * @p count performs an arithmetic right shift, rounding toward -infinity
 * for negative values (matching R7RS arithmetic-shift).
 *
 * @param arena Arena to allocate the result from.
 * @param a Value to shift.
 * @param count Signed shift amount in bits.
 * @return Newly allocated shifted bignum, or NULL if @p a is NULL.
 */
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

/* Parse a string to a tagged number in an explicit radix (R7RS 6.2.6,
 * (string->number string radix)). Radix 2..36; #b/#o/#d/#x prefixes
 * override `radix`. Result is #f for malformed input. */
void eshkol_string_to_number_radix_tagged(arena_t* arena, const char* str,
    int64_t radix, eshkol_tagged_value_t* result);

/* ===== Display ===== */

/* Write decimal representation to file (for display/write) */
void eshkol_bignum_display(const eshkol_bignum_t* a, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_BIGNUM_H */
