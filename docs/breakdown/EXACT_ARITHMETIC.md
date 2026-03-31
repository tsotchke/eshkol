# Exact Arithmetic in Eshkol: Bignums and Rationals

This document describes the design and implementation of Eshkol's exact arithmetic
subsystem, covering arbitrary-precision integers (bignums) and exact rational numbers.
The system implements the R7RS numeric tower with automatic promotion, demotion, and
mixed-exactness dispatch.

---

## 1. Numeric Tower

Eshkol implements a five-level numeric tower, ordered by increasing generality:

```
int64 < bignum < rational < double < complex
```

Each level has a distinct runtime representation:

| Level    | Type Tag                  | Heap Subtype             | Width    |
|----------|---------------------------|--------------------------|----------|
| int64    | `ESHKOL_VALUE_INT64` (2)  | N/A (inline)             | 8 bytes  |
| bignum   | `ESHKOL_VALUE_HEAP_PTR` (8) | `HEAP_SUBTYPE_BIGNUM` (11)  | variable |
| rational | `ESHKOL_VALUE_HEAP_PTR` (8) | `HEAP_SUBTYPE_RATIONAL` (19) | 16 bytes |
| double   | `ESHKOL_VALUE_DOUBLE` (3) | N/A (inline)             | 8 bytes  |
| complex  | `ESHKOL_VALUE_HEAP_PTR` (8) | heap-allocated           | 16 bytes |

All values are transported as tagged values: a 16-byte struct
`{type:i8, flags:i8, reserved:i16, padding:i32, data:i64}`. The `data` field holds
either an inline int64/double (bit-punned) or a pointer to heap-allocated storage.

### 1.1. Exactness Semantics (R7RS 6.2.3)

The fundamental invariant governing arithmetic dispatch:

- **Exact + Exact = Exact.** An int64 added to a bignum yields a bignum (or int64 if
  the result fits). A rational divided by a rational yields a rational.

- **Exact + Inexact = Inexact.** Any operation involving a double converts the exact
  operand to double first, and the result is a double. This is a one-way gate:
  exactness, once lost, is not recovered.

- **Promotion preserves value.** An int64 promoted to bignum represents the identical
  integer. A bignum promoted to double may lose precision (this is the
  exact-to-inexact conversion).

- **Demotion recovers compactness.** After an exact operation, if the bignum result
  fits in the range `[-2^63, 2^63 - 1]`, it is demoted back to int64. This avoids
  unnecessary heap allocation for values that transiently exceeded int64 range.

---

## 2. Bignum Implementation

### 2.1. Representation

Bignums use a sign-magnitude representation with a dynamic array of 64-bit limbs
stored in little-endian order (least significant limb first). The layout in memory is:

```
[eshkol_object_header_t][eshkol_bignum_t][limbs[0], limbs[1], ..., limbs[n-1]]
```

The `eshkol_bignum_t` structure (`inc/eshkol/core/bignum.h:36-40`):

```c
typedef struct eshkol_bignum {
    int32_t  sign;       /* 0 = non-negative, 1 = negative */
    uint32_t num_limbs;  /* Number of 64-bit limbs */
    /* uint64_t limbs[] follows in memory */
} eshkol_bignum_t;
```

The numeric value represented is:

```
value = (-1)^sign * sum(limbs[i] * 2^(64*i), i = 0..num_limbs-1)
```

Zero is canonicalized as `{sign=0, num_limbs=1, limbs[0]=0}`. The macro
`BIGNUM_LIMBS(bn)` accesses the limb array at the fixed offset past the struct
(`bignum.h:43`).

All bignums are arena-allocated and immutable: every arithmetic operation returns a
freshly allocated bignum. There are no in-place mutations. This simplifies reasoning
about aliasing and is consistent with the functional semantics of Scheme numerics.

### 2.2. Construction

Three entry points create bignums:

- **`eshkol_bignum_from_int64(arena, value)`** (`bignum.cpp:331`): Creates a
  single-limb bignum. Handles `INT64_MIN` specially since `-INT64_MIN` overflows
  signed int64 (it stores the magnitude as `(uint64_t)INT64_MAX + 1`).

- **`eshkol_bignum_from_overflow(arena, a, b, op)`** (`bignum.cpp:348`): Called from
  LLVM-generated overflow handlers. Promotes both int64 operands to bignums, then
  performs the operation. Op codes: 0=add, 1=sub, 2=mul.

- **`eshkol_bignum_from_string(arena, str, len)`** (`bignum.cpp:362`): Parses
  arbitrary-length decimal integers. Processes digits left-to-right via repeated
  multiply-by-10-and-add, growing the limb array as needed.

### 2.3. Arithmetic Algorithms

The core algorithms in `lib/core/bignum.cpp`:

**Addition** (`bignum_add_abs`, line 72): Schoolbook addition with 128-bit
intermediate sums. Uses `__uint128_t` to detect carry without branching:

```c
__uint128_t sum = (__uint128_t)av + (__uint128_t)bv + carry;
lr[i] = (uint64_t)sum;
carry = (uint64_t)(sum >> 64);
```

Signed addition (`eshkol_bignum_add`, line 405) dispatches based on sign agreement:
same signs add magnitudes; different signs subtract the smaller from the larger and
take the sign of the larger.

**Subtraction** (`bignum_sub_abs`, line 98): Schoolbook subtraction with borrow
propagation. Requires `|a| >= |b|`; the caller ensures correct ordering.

**Multiplication** (`eshkol_bignum_mul`, line 460): Schoolbook O(n*m) multiplication.
For each limb of the multiplier, `bignum_addmul_limb` (line 124) performs a
multiply-accumulate pass using `__uint128_t` products. The result is allocated with
`a.num_limbs + b.num_limbs` limbs.

**Division** (`bignum_divmod_abs`, line 164): Implements Knuth's Algorithm D with
trial quotient refinement. For single-limb divisors, a fast path
(`bignum_div_limb`, line 140) uses 128-bit division. For multi-limb divisors,
the algorithm normalizes the divisor (left-shifts until the high bit is set),
computes trial quotients, and corrects via the add-back step. The complexity is
O(n*m) where n and m are the limb counts of dividend and divisor.

**Negation** (`eshkol_bignum_neg`, line 521): Copies the limb array and flips the
sign bit. Zero remains non-negative.

**Exponentiation** (`eshkol_bignum_pow`, line 862): Repeated squaring in O(log n)
multiplications. For each bit of the exponent, the base is squared; if the bit is
set, the accumulator is multiplied by the current base.

### 2.4. Normalization

After every operation, `bignum_normalize` (line 44) trims trailing zero limbs and
ensures zero has a non-negative sign. This maintains the canonical form invariant.

### 2.5. Comparison

`eshkol_bignum_compare` (line 550) implements three-way comparison:
1. Different signs: negative < positive (with a special case for both being zero).
2. Same sign: compare magnitudes via `bignum_compare_abs` (line 57), then negate
   the result if both are negative.

`eshkol_bignum_compare_int64` (line 564) provides a fast path for comparing against
a single int64, using a stack-allocated temporary bignum to avoid arena allocation.

### 2.6. Predicates and Conversion

- `eshkol_bignum_is_zero/negative/positive/even/odd` (lines 603-624): Direct
  inspection of sign and least-significant limb.

- `eshkol_bignum_fits_int64` (line 626): Returns true if the bignum is a single limb
  whose magnitude fits in `[0, INT64_MAX]` (non-negative) or `[0, INT64_MAX+1]`
  (negative, for `INT64_MIN`). This is the demotion check.

- `eshkol_bignum_to_double` (line 650): Accumulates `(double)limbs[i] * 2^(64*i)`.
  This is intentionally lossy for values exceeding 2^53.

- `eshkol_bignum_to_string` (line 667): Extracts decimal digits by repeated division
  by 10, then reverses the digit buffer. The result is arena-allocated with a string
  header for the display system.

### 2.7. Bitwise Operations (R7RS)

Bitwise operations (`bignum.cpp:1092-1335`) use two's complement semantics for
negative bignums, as required by R7RS. The implementation converts sign-magnitude to
two's complement limbs (`to_twos_complement`, line 1097), performs the bitwise
operation, and converts back (`from_twos_complement`, line 1115). Operations
supported: `bitwise-and`, `bitwise-or`, `bitwise-xor`, `bitwise-not`, and
`arithmetic-shift` (positive count = left shift, negative = right shift with
sign extension).

---

## 3. Overflow Detection and Promotion

The critical bridge between int64 and bignum arithmetic is the overflow detection
path, implemented in `lib/backend/arithmetic_codegen.cpp`.

### 3.1. LLVM Overflow Intrinsics

For each integer arithmetic operation, the codegen emits LLVM's checked arithmetic
intrinsics rather than plain `add`/`sub`/`mul`. For addition
(`arithmetic_codegen.cpp:767-768`):

```cpp
llvm::Function* sadd_ovf = ESHKOL_GET_INTRINSIC(
    &ctx_.module(), llvm::Intrinsic::sadd_with_overflow, {ctx_.int64Type()});
llvm::Value* add_ovf_result = ctx_.builder().CreateCall(sadd_ovf, {left_int, right_int});
llvm::Value* add_int_val = ctx_.builder().CreateExtractValue(add_ovf_result, 0);
llvm::Value* add_overflowed = ctx_.builder().CreateExtractValue(add_ovf_result, 1);
```

These intrinsics return a struct `{i64, i1}` where the second element is the
overflow flag. The generated code branches on this flag:

- **No overflow**: Pack the i64 result as a tagged int64.
- **Overflow**: Call `emitBignumPromotion` (line 60).

The same pattern applies to subtraction (`llvm::Intrinsic::ssub_with_overflow`)
and multiplication (`llvm::Intrinsic::smul_with_overflow`).

### 3.2. The Promotion Path

`emitBignumPromotion` (`arithmetic_codegen.cpp:60-84`) calls the runtime function
`eshkol_bignum_from_overflow(arena, a, b, op)`, which:

1. Promotes both int64 operands to single-limb bignums via `eshkol_bignum_from_int64`.
2. Performs the operation using bignum arithmetic (which cannot overflow).
3. Returns an `eshkol_bignum_t*`.

The codegen packs this pointer into a tagged value with type
`ESHKOL_VALUE_HEAP_PTR` (8). If the arena is unavailable (should not happen in
normal execution), a fallback path promotes both operands to double.

### 3.3. The Demotion Path

After every bignum binary operation, `eshkol_bignum_binary_tagged`
(`bignum.cpp:797-804`) checks whether the result fits in int64:

```c
int64_t fits;
if (eshkol_bignum_fits_int64(r, &fits)) {
    *result = eshkol_make_int64(fits, true);
} else {
    *result = eshkol_make_ptr((uint64_t)(void*)r, ESHKOL_VALUE_HEAP_PTR);
    result->flags = ESHKOL_VALUE_EXACT_FLAG;
}
```

This ensures that temporary excursions into bignum space (e.g., computing
`INT64_MAX + 1 - 1`) return to int64 representation, avoiding unnecessary
heap pressure.

---

## 4. Rational Implementation

### 4.1. Representation

Rationals are stored as a pair of int64 values (`inc/eshkol/core/rational.h:23-26`):

```c
typedef struct {
    int64_t numerator;
    int64_t denominator;
} eshkol_rational_t;
```

Heap layout:

```
[eshkol_object_header_t (subtype=HEAP_SUBTYPE_RATIONAL=19)][eshkol_rational_t]
```

Two invariants are maintained at all times:
1. **Positive denominator**: If the input denominator is negative, both numerator and
   denominator are negated.
2. **GCD-reduced**: The numerator and denominator are divided by their greatest
   common divisor. This ensures a unique canonical form for each rational value.

### 4.2. Construction and Reduction

`eshkol_rational_create` (`rational.cpp:72-98`) enforces both invariants:

```c
if (denom < 0) { num = -num; denom = -denom; }
int64_t g = gcd(num, denom);
if (g > 1) { num /= g; denom /= g; }
```

The GCD is computed via the Euclidean algorithm (`rational.cpp:22-31`). For
intermediate results that may exceed int64 range, a 128-bit GCD function
(`gcd128`, line 34) is used by `rational_create_safe` (line 52), which falls back
to double arithmetic if the reduced result still does not fit in int64.

### 4.3. Arithmetic

All four operations are implemented with 128-bit intermediate arithmetic to prevent
overflow (`rational.cpp:100-138`):

- **Addition**: `a/b + c/d = (a*d + c*b) / (b*d)` using `__int128_t` products.
- **Subtraction**: `a/b - c/d = (a*d - c*b) / (b*d)`.
- **Multiplication**: `a/b * c/d = (a*c) / (b*d)`.
- **Division**: `a/b / c/d = (a*d) / (b*c)`. Raises a divide-by-zero exception if
  `c == 0`.

Each result is passed through `rational_create_safe`, which reduces by GCD128 and
checks whether the reduced numerator and denominator fit in int64. If they do not
fit, the function returns `NULL`, signaling the caller to fall back to double
arithmetic. This overflow-to-double fallback preserves the R7RS guarantee that
arithmetic never fails, at the cost of exactness.

### 4.4. Comparison

`eshkol_rational_compare` (`rational.cpp:140-149`) uses cross-multiplication to
avoid division:

```c
__int128_t lhs = (__int128_t)ra->numerator * rb->denominator;
__int128_t rhs = (__int128_t)rb->numerator * ra->denominator;
```

Since both denominators are positive (by invariant), the sign of `lhs - rhs`
directly gives the comparison result. The 128-bit multiplication prevents overflow
for all int64 numerator/denominator pairs.

### 4.5. Rounding Functions

Four rounding modes are implemented for exact rationals (`rational.cpp:528-570`):

- **floor** (toward negative infinity): `n / d`, adjusted by `-1` when the
  remainder is negative.
- **ceil** (toward positive infinity): `n / d`, adjusted by `+1` when the
  remainder is positive.
- **truncate** (toward zero): Plain C integer division `n / d`.
- **round** (nearest, ties to even): Compares `2 * |remainder|` against `d`.
  When equal (exact half), rounds to even quotient (banker's rounding).

### 4.6. Rationalize (R7RS)

`eshkol_rationalize_tagged` (`rational.cpp:408-522`) implements the R7RS
`rationalize` procedure using a Stern-Brocot mediant search. Given a value `x` and
tolerance `epsilon`, it finds the simplest rational `p/q` (smallest denominator)
such that `|x - p/q| <= epsilon`. The search terminates when the mediant falls
within the target interval or the denominator exceeds 10^9.

---

## 5. Mixed-Exactness Dispatch

### 5.1. Tagged Value Runtime Dispatch

The runtime dispatch functions accept tagged value pointers and handle all type
combinations internally. The bignum dispatch function
`eshkol_bignum_binary_tagged` (`bignum.cpp:732-805`) follows this priority order:

1. **Unary negation** (op 7): Convert operand to bignum, negate, return.
2. **Double operand check** (R7RS exact+inexact rule): If either operand has type
   `ESHKOL_VALUE_DOUBLE`, convert both to double, perform the operation as double
   arithmetic, and return a double result. Bignums are converted via
   `eshkol_bignum_to_double`.
3. **Exact path**: Convert both operands to bignum (promoting int64 if necessary via
   `tagged_to_bignum`, line 720), perform the operation, and attempt demotion.

For division (op 3), the exact path first checks whether the remainder is zero. If
so, it returns the exact quotient; otherwise, it converts both operands to double
and returns an inexact result.

### 5.2. Rational Tagged Dispatch

`eshkol_rational_binary_tagged` (`rational.cpp:208-311`) follows an analogous
pattern:

1. If either operand is a double, convert both to double and return double.
2. Otherwise, promote int64 operands to rational (denominator 1) and perform
   exact rational arithmetic.
3. If the rational result overflows int64 (NULL return from `rational_create_safe`),
   fall back to double.
4. If the result has denominator 1, demote to int64.

### 5.3. Codegen-Level Type Dispatch

At the LLVM IR level, `ArithmeticCodegen::extractAsDouble`
(`arithmetic_codegen.cpp:1586`) provides a 4-way dispatch that converts any numeric
tagged value to a double:

1. `ESHKOL_VALUE_CALLABLE` with AD node subtype: extract primal double from the AD
   node struct.
2. `ESHKOL_VALUE_DOUBLE`: unpack directly.
3. `ESHKOL_VALUE_HEAP_PTR`: call `eshkol_bignum_to_double` runtime function.
4. `ESHKOL_VALUE_INT64`: `SIToFP` conversion.

This function is used whenever a numeric value must be converted to floating-point
for inexact operations (e.g., trigonometric functions, mixed exact+inexact
arithmetic at the codegen level).

---

## 6. String Conversion

### 6.1. Number to String

`eshkol_bignum_to_string` (`bignum.cpp:667-716`) converts a bignum to its decimal
representation by repeated division by 10. Each division extracts the least
significant digit; the digits are reversed to produce the final string. The string
is allocated with a header via `arena_allocate_string_with_header` for integration
with the display and `write` subsystems.

For rationals, `eshkol_rational_to_string` (`rational.cpp:169-184`) formats as
`"num/denom"` when the denominator is not 1, or as a plain integer string when it
is 1.

### 6.2. String to Number

`eshkol_string_to_number_tagged` (`bignum.cpp:950-1028`) parses numeric strings
with a multi-strategy approach:

1. **Syntax scan**: Scan for `/` (rational), `.` or `e`/`E` (floating-point).
2. **Rational syntax** (`"num/denom"`): Parse numerator and denominator as int64 via
   `strtoll`, then create a rational via `eshkol_rational_create`. Falls through to
   double if parsing fails.
3. **Floating-point syntax**: Parse via `strtod`, return as double.
4. **Integer syntax**: Attempt `strtoll`. If it succeeds without `ERANGE`, return as
   int64. If `ERANGE` is set (overflow), fall to bignum parsing via
   `eshkol_bignum_from_string`.
5. **Final fallback**: Parse as double via `strtod`.

This cascade ensures that `(string->number "99999999999999999999999999999")` returns
an exact bignum rather than losing precision through double conversion.

---

## 7. Codegen Architecture

### 7.1. Runtime Dispatch vs. Inline LLVM IR

The codegen uses a hybrid approach:

**Inline LLVM IR** (fast path): For int64 arithmetic, the codegen emits native LLVM
instructions (`add`, `sub`, `mul`) with overflow intrinsics. This path avoids
function call overhead for the common case where both operands are small integers
and no overflow occurs.

**Runtime dispatch** (slow path): When overflow is detected or when either operand is
a bignum/rational, the codegen calls into C runtime functions. These functions handle
type-checking, promotion, the actual arbitrary-precision arithmetic, and demotion.

### 7.2. Codegen Helper Functions

`ArithmeticCodegen` in `arithmetic_codegen.cpp` provides three key helpers that
emit the calls to the runtime dispatch layer:

- **`emitIsBignumCheck(left, right)`** (line 480): Allocates two tagged values on
  the stack, stores the operands, calls `eshkol_is_bignum_tagged` on each, and
  returns the logical OR. Used to branch into the bignum path before attempting
  int64 arithmetic.

- **`emitBignumBinaryCall(left, right, op_code)`** (line 494): Allocates stack space
  for left, right, and result tagged values. Stores operands, calls
  `eshkol_bignum_binary_tagged`, loads and returns the result. All allocas are placed
  in the function's entry block to avoid stack growth in loops (critical for TCO).

- **`emitBignumCompareCall(left, right, op_code)`** (line 520+): Same alloca pattern
  as the binary call, but invokes `eshkol_bignum_compare_tagged`. Op codes:
  0=lt, 1=gt, 2=eq, 3=le, 4=ge.

For rationals, `emitRationalCompareCall` follows the same pattern, calling
`eshkol_rational_compare_tagged_ptr` with pointer-based arguments to avoid ABI
issues with passing 16-byte structs by value on different architectures.

### 7.3. Entry Block Alloca Pattern

A critical implementation detail: all `alloca` instructions used by the dispatch
helpers are placed in the function's entry block, not at the current insertion point.
This is achieved by creating a temporary `IRBuilder` positioned at the entry block:

```cpp
llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), ...);
```

If allocas were placed inside loop bodies, each iteration would grow the stack,
eventually causing a stack overflow in tight loops. Entry block placement ensures
constant stack usage regardless of iteration count.

---

## 8. Code Examples

### 8.1. Large Factorials

```scheme
(define (factorial n)
  (if (<= n 1) 1
      (* n (factorial (- n 1)))))

(display (factorial 50))
;; => 30414093201713378043612608166979581188299763898377856820553615673507270386838265
;;    2522168640000000000000
```

The multiplication `(* n ...)` uses the int64 fast path for small values of `n`. When
the accumulator exceeds `2^63 - 1`, the `smul.with.overflow.i64` intrinsic triggers
bignum promotion. Subsequent multiplications proceed through
`eshkol_bignum_binary_tagged` with op code 2 (mul).

### 8.2. Rational Arithmetic

```scheme
(display (+ 1/3 1/6))    ;; => 1/2
(display (* 2/3 3/4))    ;; => 1/2
(display (- 1/2 1/3))    ;; => 1/6
(display (/ 5/7 10/21))  ;; => 3/2
```

Each operation produces an exact rational in canonical form. The GCD reduction
ensures `1/3 + 1/6 = 3/6 = 1/2` rather than `3/6`.

### 8.3. Numeric Tower Promotion

```scheme
(display (+ 1 0.5))            ;; => 1.5 (int64 + double = double)
(display (exact? (* 2 3)))     ;; => #t  (int64 * int64 = int64)
(display (exact? (expt 2 100)));; => #t  (exact exponentiation = bignum)
(display (+ (expt 2 100) 0.0));; => 1.2676506002282294e+30 (bignum + double = double)
```

The `(expt 2 100)` call enters `eshkol_bignum_pow_tagged`, which detects that both
operands are exact integers with a non-negative exponent, and delegates to
`eshkol_bignum_pow` for exact repeated squaring. Adding `0.0` triggers the
inexact promotion path in `eshkol_bignum_binary_tagged`, converting the bignum to
double via `eshkol_bignum_to_double`.

---

## References

- `inc/eshkol/core/bignum.h` -- Bignum type definition, public API declarations
- `lib/core/bignum.cpp` -- Bignum arithmetic, tagged dispatch, bitwise operations
- `inc/eshkol/core/rational.h` -- Rational type definition, rounding functions
- `lib/core/rational.cpp` -- Rational arithmetic, comparison, rationalize
- `lib/backend/arithmetic_codegen.cpp` -- LLVM IR emission for overflow detection and runtime dispatch
- R7RS Section 6.2 -- Numerical operations specification
