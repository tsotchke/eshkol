# Exact Arithmetic in Eshkol — Bignum, Rational, Complex, Double and Dual

This document is the canonical reference for the numeric tower implemented in
Eshkol v1.2.1-scale. Its scope is the *exact* and *inexact* numeric types —
`int64`, arbitrary-precision integers (bignums), exact rationals, IEEE 754
doubles, dual numbers (forward-mode automatic differentiation) and complex
numbers — together with the dispatch architecture that ties them into a single
polymorphic arithmetic surface compatible with R7RS §6.2 *Numbers*.

The treatment is post-doctoral in register and faithful to the source. Every
function name, dispatch order, layout constant and demotion rule cited below
maps to a concrete line in the v1.2.1-scale source tree. Where the v1.2 release
closed gaps that were latent in earlier branches, the gap and its fix are
described with file-level citations.

---

## 1. Numeric tower overview

Eshkol implements a six-level numeric tower. Five of the levels are standard
R7RS numbers; the sixth is the forward-mode automatic-differentiation type
`dual` (R7RS-compatible because a dual reduces to its primal under
`extractAsDouble`). The tower, ordered by increasing generality of the values
it can represent (not by storage size), is

$$\texttt{int64} \;\subseteq\; \texttt{bignum} \;\subseteq\; \texttt{rational}
  \;\subseteq\; \texttt{double} \;\subseteq\; \texttt{complex}$$

with $\texttt{dual}$ as a side branch isomorphic to $\mathbb{R}[\varepsilon]
/(\varepsilon^{2})$ — that is, ordered pairs $(v, v') \in \mathbb{R}^{2}$
treated as truncated jets.

The following table maps every type to its runtime representation as the source
defines it. The numeric constants are taken from `inc/eshkol/eshkol.h` §70–113
and §337–360.

| Type     | Type tag (value)                  | Heap subtype                          | Payload layout                                                                 | Exactness            | AD-aware |
|----------|-----------------------------------|---------------------------------------|--------------------------------------------------------------------------------|----------------------|----------|
| int64    | `ESHKOL_VALUE_INT64` = 1          | n/a (inline in `data.int_val`)        | 64-bit two's complement, range $[-2^{63}, 2^{63}-1]$                            | exact                | yes      |
| bignum   | `ESHKOL_VALUE_HEAP_PTR` = 8       | `HEAP_SUBTYPE_BIGNUM` = 11            | sign-magnitude, `int32_t sign + uint32_t num_limbs + uint64_t limbs[]`         | exact                | lossy    |
| rational | `ESHKOL_VALUE_HEAP_PTR` = 8       | `HEAP_SUBTYPE_RATIONAL` = 19          | `int64_t numerator + int64_t denominator`, GCD-reduced, denominator > 0        | exact                | lossy    |
| double   | `ESHKOL_VALUE_DOUBLE` = 2         | n/a (inline in `data.double_val`)     | IEEE 754 binary64                                                              | inexact              | yes      |
| complex  | `ESHKOL_VALUE_COMPLEX` = 7        | n/a (pointer in `data.ptr_val`)       | `double real + double imag`, heap-allocated 16-byte block                      | always inexact       | no       |
| dual     | `ESHKOL_VALUE_DUAL_NUMBER` = 6    | n/a (inline pair, 16 bytes)           | `double value + double derivative`                                             | inexact              | native   |

The "lossy" entry in the AD column means that `extractAsDouble` on a bignum or
rational performs an `eshkol_bignum_to_double` / `eshkol_rational_to_double`
call, which is exact only when the value fits in the 53-bit IEEE 754
significand. The hand-off is documented in
`arithmetic_codegen.cpp §extractAsDouble` (lines 1613–1742).

The static-assertion blocks at `eshkol.h`, `eshkol.h`, and `eshkol.h`
constrain the tagged value, dual number and complex number to fit in 16 bytes,
which the codegen relies upon when emitting `alloca` / `getelementptr`
sequences.

### 1.1 Exactness flags

`eshkol.h` defines two flag bits that flow alongside the type tag:

* `ESHKOL_VALUE_EXACT_FLAG = 0x10`
* `ESHKOL_VALUE_INEXACT_FLAG = 0x20`

R7RS-mandated predicates `exact?` / `inexact?` are implemented in terms of
these flag bits via `ESHKOL_IS_EXACT(type)` / `ESHKOL_IS_INEXACT(type)`
(`eshkol.h`). The runtime tagged-value constructors enforce the rule:

* `eshkol_make_int64(val, exact=true)` sets `flags = ESHKOL_VALUE_EXACT_FLAG`
  (`eshkol.h`).
* `eshkol_make_double(val)` sets `flags = ESHKOL_VALUE_INEXACT_FLAG`
  (`eshkol.h`).
* `eshkol_make_complex(ptr)` sets `flags = ESHKOL_VALUE_INEXACT_FLAG`
  (`eshkol.h`) because Eshkol complex numbers are stored as a pair of
  doubles — there is no exact-complex representation.

The bignum binary dispatch (`bignum.cpp`) explicitly stamps
`result->flags = ESHKOL_VALUE_EXACT_FLAG` on the heap-pointer result; this
matters for `exact?` / `inexact?` predicates that downstream code uses to
decide whether to call `exact->inexact` before printing.

---

## 2. Tagged value layout

All runtime values flow through a single 16-byte structure defined at
`eshkol.h`:

```c
typedef struct eshkol_tagged_value {
    uint8_t  type;        // value type (eshkol_value_type_t)
    uint8_t  flags;       // exactness and other flags
    uint16_t reserved;    // reserved for future use
    union {
        int64_t  int_val;
        double   double_val;
        uint64_t ptr_val;
        uint64_t raw_val;
    } data;
} eshkol_tagged_value_t;
```

Field offsets (assuming default packing on the platforms Eshkol targets, all of
which align an `int64_t` on an 8-byte boundary):

| Offset | Field      | Size | Notes                                                  |
|--------|------------|------|--------------------------------------------------------|
| 0      | `type`     | 1    | one of `ESHKOL_VALUE_*` (0–63 reserved for base types) |
| 1      | `flags`    | 1    | `EXACT_FLAG`, `INEXACT_FLAG`, port-direction flags     |
| 2      | `reserved` | 2    | currently unused; future GC bits                       |
| 4      | padding    | 4    | implicit alignment padding before the 8-byte union     |
| 8      | `data`     | 8    | int64, double bit pattern, or heap pointer             |

When the LLVM codegen wants to read the data union it uses
`CreateExtractValue(tagged, {4})` — index 4, not 3, because the LLVM struct
layout is `{i8, i8, i16, i32, i64}`. The padding word at index 3 was the source
of a long-standing class of off-by-one bugs in v1.1 (`MEMORY.md`: "Tagged value
data field index").

### 2.1 What `data` holds per type

The mapping from `type` to which union field is semantically active is:

| Type tag                       | Active field     | Interpretation                                                                                  |
|--------------------------------|------------------|-------------------------------------------------------------------------------------------------|
| `ESHKOL_VALUE_NULL`            | `int_val = 0`    | `()`, the empty list / null                                                                     |
| `ESHKOL_VALUE_INT64`           | `int_val`        | signed 64-bit integer, two's complement                                                         |
| `ESHKOL_VALUE_DOUBLE`          | `double_val`     | IEEE 754 binary64                                                                               |
| `ESHKOL_VALUE_BOOL`            | `int_val`        | 0 or 1                                                                                          |
| `ESHKOL_VALUE_CHAR`            | `int_val`        | Unicode codepoint as int64                                                                      |
| `ESHKOL_VALUE_SYMBOL`          | `int_val`        | symbol-table id, interned                                                                       |
| `ESHKOL_VALUE_DUAL_NUMBER`     | `int_val`        | bit-cast of the `eshkol_dual_number_t` (forward AD)                                             |
| `ESHKOL_VALUE_COMPLEX`         | `ptr_val`        | pointer to heap `{double real, double imag}` block                                              |
| `ESHKOL_VALUE_HEAP_PTR`        | `ptr_val`        | pointer to heap object, subtype distinguished by 8-byte header at `ptr - 8`                     |
| `ESHKOL_VALUE_CALLABLE`        | `ptr_val`        | pointer to closure/lambda-sexpr/ad-node, subtype distinguished by header                        |
| `ESHKOL_VALUE_LOGIC_VAR`       | `int_val`        | logic variable id `?x` (neuro-symbolic engine)                                                  |

The crucial invariant for the numeric tower is that **bignum and rational both
sit under `ESHKOL_VALUE_HEAP_PTR`**, distinguished only by the heap object
header subtype byte at `ptr - 8`. The complex type is *not* a HEAP_PTR — it has
its own dedicated tag, partly to avoid loading the header on every complex
operation, and partly because complex numbers have no exact form.

### 2.2 Heap object header

Every heap-allocated object is prefixed by an 8-byte
`eshkol_object_header_t` (`eshkol.h`):

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;      // distinguishes types within HEAP_PTR / CALLABLE
    uint8_t  flags;        // GC marks, linear status, etc.
    uint16_t ref_count;    // 0 = not ref-counted
    uint32_t size;         // object size excluding header
} eshkol_object_header_t;
```

The numeric subtypes are:

* `HEAP_SUBTYPE_BIGNUM = 11` (`eshkol.h`)
* `HEAP_SUBTYPE_RATIONAL = 19` (`eshkol.h`)

Subtype lookup from a tagged value is:

```c
#define ESHKOL_GET_HEADER(data_ptr) \
    ((eshkol_object_header_t*)((uint8_t*)(data_ptr) - sizeof(eshkol_object_header_t)))
#define ESHKOL_GET_SUBTYPE(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->subtype)
```

(`eshkol.h`).

The macro `ESHKOL_IS_BIGNUM(val)` (`eshkol.h`) packages the heap-ptr
check with the subtype dereference:

```c
#define ESHKOL_IS_BIGNUM(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_BIGNUM)
```

There is no symmetrical `ESHKOL_IS_RATIONAL` macro at the header level; the
runtime exposes `eshkol_is_rational_tagged_ptr` (`rational.cpp`) which
the codegen calls from `emitIsRationalCheck` (`arithmetic_codegen.cpp`)
to keep the rational-vs-bignum subtype decision in one place.

---

## 3. Bignum implementation

### 3.1 Representation

Bignums use a sign-magnitude representation with a dynamic array of 64-bit
limbs stored little-endian (least significant limb first). The struct
(`inc/eshkol/core/bignum.h`) is

```c
typedef struct eshkol_bignum {
    int32_t  sign;       // 0 = non-negative, 1 = negative
    uint32_t num_limbs;  // number of 64-bit limbs
    // uint64_t limbs[] follows in memory
} eshkol_bignum_t;
```

with the limb array placed in-band immediately after the struct, accessed via
the `BIGNUM_LIMBS(bn)` macro (`bignum.h`). The full heap layout for a
bignum with $n$ limbs is

```
[eshkol_object_header_t : 8 B] [eshkol_bignum_t : 8 B] [limb_0 : 8 B] ... [limb_{n-1} : 8 B]
```

so a bignum of $n$ limbs occupies $16 + 8n$ bytes on the arena. The numeric
value represented is

$$\mathrm{val}(b) \;=\; (-1)^{b.\mathrm{sign}} \cdot \sum_{i=0}^{b.\mathrm{num\_limbs}-1}
    b.\mathrm{limbs}[i] \cdot 2^{64 i}.$$

Zero is canonicalised as `{sign=0, num_limbs=1, limbs[0]=0}`. The invariant is
maintained by `bignum_normalize` (`bignum.cpp`), which trims trailing
zero limbs (keeping at least one) and forces the sign back to zero when the
trimmed value is zero. Every public bignum function returns a normalised
result.

All bignums are arena-allocated and immutable. There are no in-place mutations;
every arithmetic operation returns a freshly allocated bignum. This decision
simplifies aliasing analysis, eliminates a class of double-free bugs that
plagued earlier sign-magnitude libraries, and is consistent with the functional
semantics of Scheme numerics.

The allocator helper `bignum_alloc(arena, num_limbs)` (`bignum.cpp`)
calls `arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_BIGNUM, 0)`
and zeroes the limb array.

### 3.2 Construction

Three entry points create bignums (`bignum.h`, `bignum.cpp`):

* `eshkol_bignum_from_int64(arena, value)` — single-limb construction with a
  special case for `INT64_MIN`, because `-INT64_MIN` overflows signed int64.
  The magnitude is stored as `(uint64_t)INT64_MAX + 1`. (`bignum.cpp`)

* `eshkol_bignum_from_overflow(arena, a, b, op)` — called from LLVM-generated
  overflow handlers. Promotes both int64 operands to bignums, then performs
  the operation. Op codes: `0=add`, `1=sub`, `2=mul`. (`bignum.cpp`)

* `eshkol_bignum_from_string(arena, str, len)` — parses arbitrary-length
  decimal integers. Processes digits left-to-right via repeated multiply-by-10
  and add-digit. When the limb array overflows on the multiplication step the
  function allocates a wider bignum and copies the limbs. (`bignum.cpp`)

### 3.3 Arithmetic algorithms

The five core arithmetic operations live in `lib/core/bignum.cpp`:

| Operation       | Function                              | Algorithm                                                       |
|-----------------|---------------------------------------|-----------------------------------------------------------------|
| addition        | `eshkol_bignum_add` (lines 405–430)   | sign dispatch onto `bignum_add_abs` / `bignum_sub_abs`          |
| subtraction     | `eshkol_bignum_sub` (lines 432–458)   | identical sign dispatch                                         |
| multiplication  | `eshkol_bignum_mul` (lines 460–483)   | schoolbook $O(nm)$ via `bignum_addmul_limb`                     |
| division        | `eshkol_bignum_div` (lines 485–501)   | wraps `bignum_divmod_abs`, sign handled by caller               |
| modulo          | `eshkol_bignum_mod` (lines 503–519)   | wraps `bignum_divmod_abs`, sign follows dividend                |
| negation        | `eshkol_bignum_neg` (lines 521–529)   | copies limbs, flips sign bit, special-cases zero                |

The 128-bit primitives all rest on GCC/Clang's `__uint128_t` and `__int128_t`
intrinsics:

* `bignum_add_abs` (lines 72–96) uses
  $\texttt{sum} = (\texttt{av} + \texttt{bv} + \texttt{carry})$ in 128-bit
  width to detect carry without branching.
* `bignum_sub_abs` (lines 99–121) propagates borrow via two integer comparisons
  per limb.
* `bignum_addmul_limb` (lines 124–137) computes
  $\texttt{result}[i+\mathrm{offset}] \mathrel{+}= a[i] \cdot b_{\mathrm{limb}} + \mathrm{carry}$
  in 128-bit width per limb.

Division (`bignum_divmod_abs`, lines 164–324) implements Knuth's Algorithm D
with trial-quotient refinement. The single-limb fast path lives in
`bignum_div_limb` (lines 140–160) and uses 128-bit division. For multi-limb
divisors the algorithm

1. normalises the divisor by left-shifting so its top limb's high bit is set
   (computed via `__builtin_clzll`),
2. allocates shifted copies of both operands on the arena,
3. iterates from the most significant quotient limb downward, computing a
   trial quotient $\hat{q} = \lfloor (u_{j+n} \cdot 2^{64} + u_{j+n-1}) / v_{n-1} \rfloor$,
4. refines $\hat{q}$ until Knuth's test
   $\hat{q} \cdot v_{n-2} \le \hat{r} \cdot 2^{64} + u_{j+n-2}$ holds,
5. multiplies and subtracts in place, with an add-back step when the
   subtraction goes negative.

Complexity is $O(nm)$ where $n$ and $m$ are the limb counts of dividend and
divisor.

### 3.4 Comparison

`eshkol_bignum_compare(a, b)` (`bignum.cpp`) is three-way:

1. Different signs: the negative one is smaller, with a special case for both
   being zero.
2. Same sign: `bignum_compare_abs` (lines 57–69) compares magnitudes by limb
   count first, then by limbs from most significant to least; the result is
   negated when both are negative.

The fast path `eshkol_bignum_compare_int64` (lines 564–599) handles
single-limb bignums inline without arena allocation; for multi-limb bignums it
constructs a stack-resident `{eshkol_bignum_t hdr; uint64_t limb;}` aggregate
and delegates to `eshkol_bignum_compare`.

### 3.5 Predicates and conversion

* `eshkol_bignum_is_zero/negative/positive/even/odd` (`bignum.cpp`)
  inspect the sign field and `limbs[0]`.
* `eshkol_bignum_fits_int64(bn, out)` (`bignum.cpp`) returns true
  when `num_limbs == 1` and the magnitude lies in
  $[0, \texttt{INT64\_MAX}]$ for non-negative, or in
  $[0, \texttt{INT64\_MAX}+1]$ for negative (allowing `INT64_MIN`).
  This is the demotion oracle.
* `eshkol_bignum_to_double(bn)` (`bignum.cpp`) accumulates
  $\sum_i b_i \cdot 2^{64i}$ in double precision. Intentionally lossy for
  $|b| > 2^{53}$.
* `eshkol_bignum_to_string(arena, bn)` (`bignum.cpp`) extracts decimal
  digits by repeated division by 10 (using the limb-divisor fast path) and
  reverses the buffer. The result is allocated via
  `arena_allocate_string_with_header`, so it is a first-class Eshkol string
  with header byte count.

### 3.6 Two's-complement bitwise operations (R7RS)

R7RS requires bitwise operations on integers to follow two's complement
semantics, including for negative bignums. The implementation
(`bignum.cpp`) converts each operand to a two's complement limb
buffer via `to_twos_complement` (lines 1126–1141), performs the bitwise
operation, and converts back via `from_twos_complement` (lines 1144–1169).

Operations supported:

* `bitwise-and`, `bitwise-or`, `bitwise-xor` — element-wise on the two's
  complement representation, sign-extended to the longer operand
  (`bignum.cpp`).
* `bitwise-not` — special-cased to $a \in \mathbb{Z} \mapsto -(a+1)$, since
  $\sim a$ in two's complement equals $-(a+1)$ regardless of width
  (`bignum.cpp`).
* `arithmetic-shift` — positive count is left shift (limb-aware), negative
  count is right shift with sign extension to $-\infty$
  (`bignum.cpp`).

The dispatch entry point is `eshkol_bignum_bitwise_tagged`
(`bignum.cpp`), which uses op codes `0=and, 1=or, 2=xor, 3=not,
4=arithmetic-shift` and demotes the result to int64 when it fits.

### 3.7 Exponentiation

`eshkol_bignum_pow(arena, base, exp)` (`bignum.cpp`) implements
repeated squaring:

$$\text{base}^{\,\text{exp}} = \prod_{i : \text{exp}_i = 1} \text{base}^{2^i}$$

requiring $O(\log_2 \text{exp})$ multiplications. Special cases at lines
869–879 handle `exp == 0` (returns 1) and `exp == 1` (returns a copy of the
base via `bignum_mul(base, 1)`).

The tagged-value entry point `eshkol_bignum_pow_tagged`
(`bignum.cpp`) dispatches into the exact path only when

* the base is `ESHKOL_VALUE_INT64` or `ESHKOL_VALUE_HEAP_PTR` (which is
  necessarily a bignum because rational `expt` is not in this code path),
* the exponent is `ESHKOL_VALUE_INT64`, and
* the exponent is non-negative.

If any of those fail, the function falls back to libm `pow()` on
double-precision extracted from both operands. The exact path attempts to
demote the result to int64 via `eshkol_bignum_fits_int64`.

---

## 4. Overflow detection and promotion

The critical bridge between int64 and bignum arithmetic is the overflow
detection path in `lib/backend/arithmetic_codegen.cpp`.

### 4.1 LLVM overflow intrinsics

For each integer arithmetic operation the codegen emits LLVM's checked
arithmetic intrinsics rather than plain `add`, `sub`, `mul`. The addition
case at `arithmetic_codegen.cpp`:

```cpp
llvm::Function* sadd_ovf = ESHKOL_GET_INTRINSIC(
    &ctx_.module(), llvm::Intrinsic::sadd_with_overflow, {ctx_.int64Type()});
llvm::Value* add_ovf_result =
    ctx_.builder().CreateCall(sadd_ovf, {left_int, right_int});
llvm::Value* add_int_val   = ctx_.builder().CreateExtractValue(add_ovf_result, 0);
llvm::Value* add_overflowed = ctx_.builder().CreateExtractValue(add_ovf_result, 1);
```

These intrinsics return `{i64, i1}` where the second element is the overflow
flag. The generated code branches on the flag:

* **no overflow** — pack the i64 result as a tagged int64
  (`tagged_.packInt64(add_int_val, /*exact=*/true)`);
* **overflow** — call `emitBignumPromotion` (`arithmetic_codegen.cpp`).

The same pattern applies to subtraction (`llvm::Intrinsic::ssub_with_overflow`)
and multiplication (`llvm::Intrinsic::smul_with_overflow`).

### 4.2 The promotion path

`emitBignumPromotion(ctx, tagged, left_int, right_int, op_code)`
(`arithmetic_codegen.cpp`):

1. Loads the global arena pointer through `getArenaPtr(ctx)`
   (`arithmetic_codegen.cpp`). If the arena is unavailable (an
   instantiation-time failure that should not happen at runtime) the function
   falls back to a double computation with a debug log message.
2. Calls `eshkol_bignum_from_overflow(arena, a, b, op)`
   (`bignum.cpp`), which promotes both int64 operands to single-limb
   bignums via `eshkol_bignum_from_int64` and dispatches into the bignum
   operation.
3. Packs the returned `eshkol_bignum_t*` into a tagged value with type
   `ESHKOL_VALUE_HEAP_PTR` via `tagged.packPtr(bignum_ptr, ESHKOL_VALUE_HEAP_PTR)`.

### 4.3 The demotion path

Every bignum binary operation funnels through `eshkol_bignum_binary_tagged`
(`bignum.cpp`), which after computing the result performs

```c
int64_t fits;
if (eshkol_bignum_fits_int64(r, &fits)) {
    *result = eshkol_make_int64(fits, /*exact=*/true);
} else {
    *result = eshkol_make_ptr((uint64_t)(void*)r, ESHKOL_VALUE_HEAP_PTR);
    result->flags = ESHKOL_VALUE_EXACT_FLAG;
}
```

(`bignum.cpp`). This ensures that temporary excursions into bignum
space — for example computing $(\texttt{INT64\_MAX} + 1) - 1$ — return to int64
representation and avoid persistent heap pressure. The flag bit is set
explicitly so downstream `exact?` queries report `#t` for the demoted int64
(consistent with the explicit `exact=true` parameter to `eshkol_make_int64`).

### 4.4 The entry-block alloca pattern

A critical implementation detail: every `alloca` used by the dispatch helpers
(`emitIsBignumCheck`, `emitBignumBinaryCall`, `emitBignumCompareCall`,
`emitIsRationalCheck`, `emitRationalBinaryCall`, `emitRationalCompareCall`)
is placed in the function's *entry block*, not at the current insertion point.
This is achieved via a fresh `IRBuilder` positioned at
`fn->getEntryBlock().begin()`:

```cpp
llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
llvm::Value* left_alloca  = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_l");
```

(`arithmetic_codegen.cpp`).

If allocas were placed inside loop bodies, each iteration would grow the stack,
eventually causing stack overflow. Entry-block placement ensures constant stack
usage regardless of iteration count. This matters because tail-call
optimisation in Eshkol depends on stable stack frames across loop iterations.

---

## 5. Rational implementation

### 5.1 Representation

Rationals are stored as a pair of int64 values
(`inc/eshkol/core/rational.h`):

```c
typedef struct {
    int64_t numerator;
    int64_t denominator;
} eshkol_rational_t;
```

with the heap layout

```
[eshkol_object_header_t : 8 B] [eshkol_rational_t : 16 B]
```

— a total of 24 bytes. The header subtype byte is set to
`HEAP_SUBTYPE_RATIONAL = 19` at construction time
(`rational.cpp`).

Crucially, the rational's numerator and denominator are *int64*, not bignums.
This means that an exact rational arithmetic that overflows int64 cannot be
represented and must demote (see §5.5 on overflow-to-double fallback). The
design choice trades full closure under rational arithmetic for cheap
$O(1)$ component access; arbitrary-precision rationals are a documented
future-work item.

Two invariants are maintained at all times:

1. **Positive denominator.** If the input denominator is negative, both
   numerator and denominator are negated.
2. **GCD-reduced.** Numerator and denominator are divided by their greatest
   common divisor, giving a unique canonical form for each rational value.

### 5.2 Construction and reduction

`eshkol_rational_create(arena, num, denom)` (`rational.cpp`) enforces
both invariants. The implementation includes an audit-driven safety check
(Audit C7, `rational.cpp`) for the case `num == INT64_MIN` or
`denom == INT64_MIN`: because `-INT64_MIN == INT64_MIN` in two's complement,
the naive negation `denom = -denom` would silently produce a
non-normalised rational. The check raises a runtime error and returns `0/1` as
the safe fallback:

```c
if (denom == INT64_MIN || num == INT64_MIN) {
    eshkol_error("rational: INT64_MIN numerator/denominator cannot be "
                 "sign-normalised without overflow");
    denom = 1;
    num = 0;
}
```

The GCD is computed via the Euclidean algorithm
(`rational.cpp`). For 128-bit intermediates (the result of multiplying
two int64s) there is a separate `gcd128` (`rational.cpp`).

A division-by-zero in the denominator triggers `eshkol_error` and returns the
safe rational `0/1` (`rational.cpp`).

### 5.3 Arithmetic and the safe-create wrapper

All four binary operations are implemented with `__int128_t` intermediates to
prevent overflow on the cross-multiplication step (`rational.cpp`):

$$\frac{a}{b} + \frac{c}{d} = \frac{ad + cb}{bd},\quad
  \frac{a}{b} - \frac{c}{d} = \frac{ad - cb}{bd},\quad
  \frac{a}{b} \cdot \frac{c}{d} = \frac{ac}{bd},\quad
  \frac{a}{b} \div \frac{c}{d} = \frac{ad}{bc}.$$

Each result is passed through `rational_create_safe`
(`rational.cpp`), which:

1. Normalises the denominator sign (negates both if `denom < 0`).
2. Reduces by `gcd128(num, denom)`.
3. Checks whether the reduced 128-bit values fit in int64 via
   `fits_int64` (`rational.cpp`).
4. Returns `NULL` if either doesn't fit, signalling overflow.

Division (`rational.cpp`) explicitly checks `rb->numerator == 0`
and raises an `ESHKOL_EXCEPTION_DIVIDE_BY_ZERO` exception via
`eshkol_make_exception` and `eshkol_raise`. The unreachable `return nullptr`
exists to satisfy the compiler.

### 5.4 Comparison

`eshkol_rational_compare(a, b)` (`rational.cpp`) uses
cross-multiplication to avoid division. Because both denominators are positive
by invariant, the sign of $(an \cdot bd - bn \cdot ad)$ directly gives the
comparison result:

$$\frac{a_n}{a_d} \lessgtr \frac{b_n}{b_d} \iff a_n b_d \lessgtr b_n a_d \quad (a_d, b_d > 0).$$

The 128-bit multiplication prevents overflow for all int64 numerator /
denominator pairs.

### 5.5 The rational binary tagged dispatch

`eshkol_rational_binary_tagged_ptr(arena, a, b, op, result)`
(`rational.cpp`) is the entry point called from the codegen. It
dereferences both pointers and delegates to `eshkol_rational_binary_tagged`
(`rational.cpp`), whose dispatch order is:

1. **Double check (R7RS exact + inexact $\to$ inexact).** If either operand
   is `ESHKOL_VALUE_DOUBLE`, both operands are converted to double (via
   `eshkol_rational_to_double` for rationals, `SIToFP` for ints, bit-cast for
   doubles) and the operation is performed as IEEE arithmetic
   (`rational.cpp`).
2. **Promote int to rational.** Integers are wrapped as `n/1` via
   `eshkol_rational_create` (`rational.cpp`).
3. **Operate.** Dispatch on op code 0=add, 1=sub, 2=mul, 3=div
   (`rational.cpp`).
4. **Overflow fallback.** If `rational_create_safe` returned `NULL`, fall back
   to double arithmetic on the rationals' double approximations
   (`rational.cpp`). This preserves the R7RS guarantee that
   arithmetic never silently fails, at the documented cost of exactness.
5. **Demote on integer result.** If the result has `denominator == 1`, return
   as `ESHKOL_VALUE_INT64` (`rational.cpp`); otherwise wrap as a
   `HEAP_PTR` (`rational.cpp`).

### 5.6 Rational comparison tagged dispatch

`eshkol_rational_compare_tagged_ptr(arena, a, b, op, result)`
(`rational.cpp`) mirrors the binary dispatch:

1. Double check (R7RS exact + inexact $\to$ inexact for comparison: both
   sides become doubles, `FCmp*` performed at double precision).
2. Promote int to rational `n/1`.
3. `eshkol_rational_compare` cross-multiplies and the sign decides the result.
4. Result is written into `result->data.int_val` as 0 / 1 and
   `result->type = ESHKOL_VALUE_BOOL`.

Op codes are `0=lt, 1=gt, 2=eq, 3=le, 4=ge` — identical to the bignum compare
op encoding, so the codegen can use a single op-code computation site
(`arithmetic_codegen.cpp`) for both.

### 5.7 Rounding functions

Four rounding modes are implemented for exact rationals
(`rational.cpp`), returning exact int64:

* **`eshkol_rational_floor`** (lines 566–572) — toward $-\infty$.
  C division truncates toward zero, so for a negative non-integer
  rational the truncated quotient is one too high; the function detects
  `n < 0 && n % d != 0` and subtracts 1.

* **`eshkol_rational_ceil`** (lines 575–580) — toward $+\infty$, symmetrical
  to floor.

* **`eshkol_rational_truncate`** (lines 583–586) — toward zero. Plain C
  integer division.

* **`eshkol_rational_round`** (lines 589–607) — to nearest, ties to even
  (banker's rounding, R7RS-mandated). The implementation compares
  $2 \cdot |\text{rem}|$ to $d$:

  - $2 |\text{rem}| > d$: round away from zero.
  - $2 |\text{rem}| = d$ (exact half): round to even quotient.
  - $2 |\text{rem}| < d$: truncate.

These functions are called from `codegenMathFunction` in `llvm_codegen.cpp`
when the operand is a rational; see §8 for the dispatch.

### 5.8 Rationalize (R7RS)

`eshkol_rationalize_tagged(arena, x, eps, result)`
(`rational.cpp`) implements the R7RS `rationalize` procedure: given
$x$ and $\varepsilon$, find the simplest rational $p/q$ (smallest denominator)
such that $|x - p/q| \le \varepsilon$.

The implementation uses a Stern–Brocot mediant search. The loop:

1. Sets initial bounds $a = \lfloor \text{lo} \rfloor / 1$ and
   $b = (\lfloor \text{lo} \rfloor + 1) / 1$.
2. If either bound is in $[\text{lo}, \text{hi}]$, return it as an integer.
3. Otherwise iterate (up to 1000 times, or until $m_d > 10^{9}$) computing the
   mediant $m = (a_n + b_n) / (a_d + b_d)$ and tightening the appropriate
   bound.

The denominator cap at $10^{9}$ (line 516) and the iteration cap at 1000
(line 511) ensure termination even for hostile inputs (irrationals or
$\varepsilon = 0$ on a transcendental); the fallback returns the closer of the
two final bounds.

### 5.9 Inexact-to-exact

`eshkol_double_to_rational(arena, d)` (`rational.cpp`) converts an
IEEE 754 double to a rational by scaling. The loop multiplies `abs_d` by 2 and
doubles the denominator until either `abs_d` is integral or the denominator
hits $2^{52}$. After the loop, `num = (int64_t)abs_d` and the rational is
constructed via `eshkol_rational_create` (which then GCD-reduces). This gives
the exact rational representation of the double's bit pattern up to the IEEE
754 53-bit significand.

---

## 6. Complex implementation

### 6.1 Representation

The complex type uses its own tag `ESHKOL_VALUE_COMPLEX = 7`
(`eshkol.h`), distinct from `HEAP_PTR`. The struct
(`eshkol.h`) is

```c
typedef struct eshkol_complex_number {
    double real;        // ℜ
    double imag;        // 𝕴
} eshkol_complex_number_t;
```

— 16 bytes (statically asserted at `eshkol.h`). The tagged value's
`data.ptr_val` points to a heap-allocated `eshkol_complex_number_t`. The
allocator is `mem_.getArenaAllocate()` and the heap pointer is then stored
into via `CreateStore(complex, complex_heap_ptr)`
(`complex_codegen.cpp`).

There is no exact-complex representation: complex is always
`ESHKOL_VALUE_INEXACT_FLAG`, as the constructor `eshkol_make_complex`
explicitly states (`eshkol.h`).

### 6.2 Basic arithmetic

The four basic operations are implemented in `complex_codegen.cpp` lines
142–263. Letting $z_1 = a + bi$ and $z_2 = c + di$:

* **Addition** (lines 142–153):
  $z_1 + z_2 = (a+c) + (b+d)i$.

* **Subtraction** (lines 155–166):
  $z_1 - z_2 = (a-c) + (b-d)i$.

* **Multiplication** (lines 168–186):
  $z_1 \cdot z_2 = (ac - bd) + (ad + bc)i$.

* **Negation** (lines 254–263) and **conjugation** (lines 265–273) are
  elementwise.

### 6.3 Smith's division formula

Naive complex division $z_1 / z_2 = (z_1 \bar{z_2}) / |z_2|^2$ overflows when
$|z_2|^2$ exceeds the double range (i.e. when $|c|$ or $|d|$ approaches
$\sqrt{\texttt{DBL\_MAX}} \approx 1.34 \times 10^{154}$). The implementation
(`complex_codegen.cpp`) uses Smith's formula, which scales by the
larger-magnitude denominator component:

If $|d| \le |c|$:

$$r = d/c,\qquad \text{denom} = c + dr,\qquad
  \Re(z_1/z_2) = \frac{a + br}{\text{denom}},\qquad
  \Im(z_1/z_2) = \frac{b - ar}{\text{denom}}.$$

Else $(|d| > |c|)$:

$$r = c/d,\qquad \text{denom} = d + cr,\qquad
  \Re(z_1/z_2) = \frac{ar + b}{\text{denom}},\qquad
  \Im(z_1/z_2) = \frac{br - a}{\text{denom}}.$$

The branch is selected via an `FCmpOLE` on `abs_c` vs `abs_d`. Each branch
performs five FP operations; the two branches converge through PHI nodes
on the real and imaginary results.

### 6.4 Overflow-safe magnitude

`complexMagnitude(z)` (`complex_codegen.cpp`) computes
$|z| = \sqrt{a^2 + b^2}$ with overflow protection:

$$|z| = \max(|a|, |b|) \cdot \sqrt{\left(\frac{a}{\max(|a|,|b|)}\right)^{2}
  + \left(\frac{b}{\max(|a|,|b|)}\right)^{2}}.$$

A guard $\max(|a|, |b|) = 0$ (when both components are zero) is detected via
`FCmpOEQ` and short-circuits to 0.0 to avoid `0/0` propagating NaN. Without
the scaling, computing $a^2 + b^2$ would overflow when $|a|$ or $|b|$ exceeds
$\sqrt{\texttt{DBL\_MAX}}$.

### 6.5 Transcendentals

* `complexAngle(z)` = $\text{atan2}(b, a)$ (`complex_codegen.cpp`),
  delegating to libm `atan2` which is declared on demand
  (`getAtan2Intrinsic`, lines 482–495).

* `complexExp(z)` = $e^{a}(\cos b + i \sin b)$ (lines 321–338).

* `complexLog(z)` = $\log |z| + i \cdot \text{arg}(z)$ (lines 340–349).

* `complexSqrt(z)` = $\sqrt{|z|} \cdot (\cos(\text{arg}/2) + i\sin(\text{arg}/2))$
  (lines 351–371).

* `complexSin(z)` = $\sin a \cosh b + i \cos a \sinh b$ (lines 373–403).

* `complexCos(z)` = $\cos a \cosh b - i \sin a \sinh b$ (lines 405–433).

The hyperbolic intermediates use the identity
$\cosh x = (e^{x} + e^{-x})/2$, $\sinh x = (e^{x} - e^{-x})/2$ rather than
calling libm `cosh`/`sinh` directly — this keeps the SLP vectorizer's job
easier and avoids one round trip through the libm symbol table.

### 6.6 Polar form

`makeFromPolar(magnitude, angle)` (lines 439–451) implements
$z = r \cdot e^{i\theta} = r\cos\theta + ir\sin\theta$.

### 6.7 Interaction with the AD types

Complex numbers and dual numbers are mutually exclusive in the dispatch tree
of `ArithmeticCodegen::add` / `sub` / `mul` / `div`: dual is checked first
(line 711 of `arithmetic_codegen.cpp`), and if no dual is detected, complex is
checked next (line 733). The conversion `convertToComplex`
(`arithmetic_codegen.cpp`) promotes an integer, bignum or double
operand to $(v, 0)$, including the lossy bignum → double conversion for
HEAP_PTR operands. Bignum to complex is intentionally lossy because complex is
already inexact; the conversion logs `eshkol_debug("complex: bignum->double
conversion is lossy")` at line 399.

---

## 7. Dispatch architecture

### 7.1 Polymorphic arithmetic surface

The polymorphic arithmetic surface is `class ArithmeticCodegen`
(`inc/eshkol/backend/arithmetic_codegen.h`). Its public method set is:

* Binary: `add`, `sub`, `mul`, `div`, `mod`
* Unary: `neg`, `abs`
* Comparison: `compare(left, right, op_str)` where `op_str ∈ {lt, gt, eq, le, ge}`
* Math: `mathFunc(operand, func_name)`, `pow(base, exp)`, `min`, `max`,
  `remainder`, `quotient`
* Coercion: `intToDouble`, `doubleToInt`, `extractAsDouble`
* AD entry points: `withADBinaryDispatch`, `withADUnaryDispatch`
* Bignum / rational helpers (public so `equal?` can use them):
  `emitBignumBinaryCall`, `emitBignumCompareCall`, `emitIsBignumCheck`,
  `emitIsRationalCheck`, `emitRationalBinaryCall`,
  `emitRationalCompareCall`

### 7.2 The dispatch tree of `add`

The binary additions all follow the same dispatch shape; `add` is canonical
(`arithmetic_codegen.cpp`). Reading the actual block sequence
top-down:

1. **AD wrapping.** The body is wrapped in `withADBinaryDispatch(left, right,
   AD_NODE_ADD=2, regular_fn)` (lines 644). If either operand is a
   `CALLABLE` whose subtype is `CALLABLE_SUBTYPE_AD_NODE`, both operands are
   promoted to AD nodes (via `convertToADNode`) and a binary op is recorded
   on the autodiff tape. Otherwise control falls through to `regular_fn`.

2. **Bignum check.** Inside `regular_fn`, the first runtime check is
   `emitIsBignumCheck(left, right)` (line 672). This calls
   `eshkol_is_bignum_tagged` on each operand and ORs the results. If true,
   control enters `bignum_path` which calls `emitBignumBinaryCall(left, right,
   0)` — the same call site that handles INT64+BIGNUM, BIGNUM+INT64 and
   BIGNUM+BIGNUM combinations because the runtime promotes int operands via
   `tagged_to_bignum` (`bignum.cpp`).

3. **Rational check.** If no bignum, the second runtime check is
   `emitIsRationalCheck(left, right)` (line 685). If true, dispatch to
   `emitRationalBinaryCall(left, right, 0)`.

4. **Vector / tensor check.** If either operand has `ESHKOL_VALUE_HEAP_PTR`
   type with a non-bignum, non-rational subtype, control enters `vector_path`
   which invokes the tensor codegen.

5. **Dual check.** If either operand has `ESHKOL_VALUE_DUAL_NUMBER` type,
   both operands are promoted to dual via `convertToDual` and `dualAdd` is
   used.

6. **Complex check.** If either operand has `ESHKOL_VALUE_COMPLEX` type,
   both operands are promoted to complex via `convertToComplex` and
   `complexAdd` is used.

7. **Double check.** If either operand is `ESHKOL_VALUE_DOUBLE`, both are
   extracted to double (via `unpackDouble` or `SIToFP`) and `FAdd` is used.

8. **Int path with overflow.** Final fallback: `sadd.with.overflow.i64`
   intrinsic, with the overflow branch entering `emitBignumPromotion`.

The PHI merge (`arithmetic_codegen.cpp`) has eight incoming edges,
one per dispatch path.

This dispatch order is dictated by *type ownership*: a bignum operand cannot
appear in any of the other paths' fast paths (because `unpackInt64` on a
HEAP_PTR returns the pointer-as-int64, not the numeric value), so the bignum
check must be first.

### 7.3 The dispatch tree of `compare`

`compare(left, right, op_str)` (`arithmetic_codegen.cpp`) follows
the same shape:

1. Extract base types.
2. Validate that both operands are numbers (or chars) — R7RS-ish: char is
   accepted because stdlib code historically writes `(= c 32)` to test for a
   space byte (lines 1788–1816). Non-numeric operands enter
   `type_error_path` which raises `eshkol_type_error`.
3. **Bignum check** via `emitIsBignumCheck`. If true,
   `emitBignumCompareCall(left, right, cmp_op)` where `cmp_op ∈ {0..4}` is
   the int encoding of `lt/gt/eq/le/ge`.
4. **Rational check** via `emitIsRationalCheck`. If true,
   `emitRationalCompareCall(left, right, cmp_op)`.
5. **Double or AD-callable check**: if any operand is `DOUBLE` or
   `CALLABLE`, both operands are extracted to double via `extractAsDouble`
   and `FCmpO{LT,GT,EQ,LE,GE}` is used. `extractAsDouble` itself dispatches
   into AD-node, bignum, rational and int paths internally.
6. **Int path**: `ICmpS{LT,GT,EQ,LE,GE}` on `unpackInt64`.

The PHI has five incoming edges (error, bignum, rational, double, int).

### 7.4 The dispatch tree of `abs`

`abs(operand)` (`arithmetic_codegen.cpp`) is wrapped in
`withADUnaryDispatch(operand, AD_NODE_ABS=42, regular_fn)`. Inside the
regular path:

1. **Heap check.** If the type is `ESHKOL_VALUE_HEAP_PTR` (which means
   bignum here — rational `abs` goes through a different code path that
   constructs the negated numerator), compare the operand to 0 via
   `emitBignumCompareCall(operand, zero, 0=lt)`. If `is_negative` is true,
   call `emitBignumBinaryCall(operand, operand, 7=neg)`; otherwise return
   the operand unchanged.

2. **Double check.** If `ESHKOL_VALUE_DOUBLE`, use `FNeg` and `Select` based
   on `FCmpOLT(val, 0.0)`.

3. **Int path.** Standard `Neg`+`Select`, with a special case at
   `val == INT64_MIN` that promotes to bignum (calls
   `eshkol_bignum_from_int64` then `eshkol_bignum_neg`) — because
   `-INT64_MIN` overflows signed int64.

The PHI has three incoming edges (heap, double, int).

### 7.5 The dispatch tree of `min` / `max`

`min` (`arithmetic_codegen.cpp`) and `max`
(`arithmetic_codegen.cpp`) both follow

1. **Bignum check** via `emitIsBignumCheck`. If true,
   `emitBignumCompareCall(left, right, 0=lt)` (for min) /
   `(left, right, 1=gt)` (for max), then `Select` between the *original*
   tagged operands based on the comparison result.

2. **Double path** via `extractAsDouble` + `FCmpOLE` / `FCmpOGE`.

The key correctness point: the bignum check exists so that `min`/`max` on
bignums $> 2^{53}$ preserves precision. Without it, the double path's
`extractAsDouble → eshkol_bignum_to_double` would lose bits and pick the
wrong operand for adversarial inputs near $2^{63}$.

The pattern of selecting between *original tagged operands* (lines 2073–2076,
2118–2121) — rather than between extracted-double values — is essential to
prevent precision loss on the output: even though the bignum compare is exact,
returning the operand verbatim avoids any double round-trip.

### 7.6 The dispatch tree of `pow`

`pow(base, exp)` (`arithmetic_codegen.cpp`) is wrapped in
`withADBinaryDispatch(base, exp, AD_NODE_POW=10, regular_fn)`. Inside:

1. **Dual check** (lines 1947–1965). If either operand is dual, both are
   promoted to dual and `autodiff_.dualPow` is used.

2. **Exact-int test** (lines 1968–1981):
   - `base_is_exact = (base_base == INT64) || (base_base == HEAP_PTR)` (i.e.
     int64 or bignum),
   - `exp_is_int = (exp_base == INT64)`,
   - `exp_non_neg = (exp_val >= 0)`,
   - `use_exact = base_is_exact && exp_is_int && exp_non_neg`.

3. **Exact path** (lines 1984–2002). Call `eshkol_bignum_pow_tagged(arena,
   base, exp, result)`, which uses repeated squaring and demotes via
   `eshkol_bignum_fits_int64`.

4. **Regular path** (lines 2005–2019). Both operands to double via
   `extractAsDouble`, then libm `pow()`.

### 7.7 `extractAsDouble` and the unified 6-way dispatch

`extractAsDouble(tagged)` (`arithmetic_codegen.cpp`) is the single
choke point through which every "I need a double" caller flows. The dispatch
tree has six leaves:

1. **Raw `double` LLVM value** — returned as-is (line 1620).
2. **Raw `i64` LLVM value** — `SIToFP` to double (lines 1623–1625).
3. **Tagged value, type = `CALLABLE` with `CALLABLE_SUBTYPE_AD_NODE`** —
   load primal value from field 1 of the AD node struct
   (lines 1647–1662).
4. **Tagged value, type = `DOUBLE`** — `unpackDouble` (lines 1665–1672).
5. **Tagged value, type = `HEAP_PTR`, subtype = `RATIONAL`** — call
   `eshkol_rational_to_double` (lines 1692–1699).
6. **Tagged value, type = `HEAP_PTR`, subtype = `BIGNUM`** — call
   `eshkol_bignum_to_double` (lines 1701–1717).
7. **Tagged value, type = `HEAP_PTR`, non-numeric subtype** — return 0.0
   instead of crashing (lines 1719–1723). This is a defensive fallback that
   shows up when a user passes a string to a numeric operation; the
   `compare` path raises a type error upstream, so this branch is reached
   only from places like `mathFunc` that have already validated their
   inputs.
8. **Tagged value, type = `INT64`** — `unpackInt64` then `SIToFP`
   (lines 1726–1730).

The PHI at line 1734 has six incoming edges, one per non-fallthrough leaf.

This function is the architectural reason the rest of the numeric tower stays
clean: every place that needs to do a numeric computation in double-precision
calls `extractAsDouble` and stops worrying about which exact-arithmetic type
it received.

---

## 8. R7RS exact + inexact dispatch — the precise rules

R7RS §6.2.3 mandates that an operation involving any inexact operand produces
an inexact result. The implementation enforces this at every dispatch level:

### 8.1 Bignum + double → double

`eshkol_bignum_binary_tagged` (`bignum.cpp`) has an explicit
double-operand check at the top of the dispatch tree, *before* the
`tagged_to_bignum` promotion attempt:

```c
if (left->type == ESHKOL_VALUE_DOUBLE || right->type == ESHKOL_VALUE_DOUBLE) {
    double ld = (left->type == ESHKOL_VALUE_DOUBLE) ? left->data.double_val
               : (left->type == ESHKOL_VALUE_HEAP_PTR && left->data.ptr_val != 0)
                 ? eshkol_bignum_to_double((eshkol_bignum_t*)(void*)left->data.ptr_val)
               : (double)left->data.int_val;
    double rd = ...;
    double r;
    switch (op) {
        case 0: r = ld + rd; break;
        case 1: r = ld - rd; break;
        case 2: r = ld * rd; break;
        case 3: r = (rd != 0.0) ? ld / rd : 0.0; break;
    }
    *result = eshkol_make_double(r);
    return;
}
```

This is the bug class documented in `MEMORY.md` as "bignum+double returns 0":
before this dispatch was hoisted above the exact path, `tagged_to_bignum`
returned NULL for type DOUBLE (because doubles don't have a HEAP_PTR field to
unwrap), and the function fell through the `if (!a || !b)` guard and returned
int64 zero. The fix places the double check first so the bignum path is never
entered when an inexact operand is present.

### 8.2 Rational + double → double

The same rule applies in `eshkol_rational_binary_tagged`
(`rational.cpp`): the double check is at the top of the dispatch.

### 8.3 Examples (concrete behaviour)

* `(+ (expt 10 30) 0.5)` — `expt` returns a bignum. The `+` enters the bignum
  path's runtime, sees a DOUBLE operand, falls through to the double
  computation. Result: a double close to $10^{30}$.

* `(* 1/3 2.0)` — the `*` enters the rational binary tagged dispatch, sees a
  DOUBLE operand, falls through to double. Result: `0.6666666666666666`.

* `(< (- (expt 2 100) 1) 1e30)` — `expt` returns a bignum, `-` keeps it a
  bignum (the runtime demotes only if `fits_int64`, which it doesn't here).
  The `<` enters `compare`'s bignum path because the LHS is bignum; the
  bignum compare path's runtime sees `1e30` is DOUBLE and falls through to
  the double compare branch (`bignum.cpp`). The bignum is converted
  to double via `eshkol_bignum_to_double`; result: `#f` because
  $2^{100} - 1 \approx 1.27 \times 10^{30} > 10^{30}$.

* `(modulo (* (expt 2 64) 3) (expt 2 64))` — `expt` and `*` both stay in
  bignum. `modulo` reaches `codegenModulo` in `llvm_codegen.cpp`,
  which (since the fix at commit 51ec814) checks `emitIsBignumCheck` first
  and dispatches into `emitBignumBinaryCall(op=4)`. Without that check, the
  raw `SRem` would have been applied to the heap pointer bits, returning a
  small multiple of the pointer — the original symptom of the 35-gap audit's
  modulo bug.

### 8.4 Exactness flag propagation

The bignum binary dispatch explicitly sets `result->flags =
ESHKOL_VALUE_EXACT_FLAG` on the HEAP_PTR return path
(`bignum.cpp`). The `eshkol_make_double` constructor sets
`ESHKOL_VALUE_INEXACT_FLAG` (`eshkol.h`). The
`eshkol_make_int64(val, /*exact=*/true)` constructor used on the demoted
return path sets `ESHKOL_VALUE_EXACT_FLAG` (`eshkol.h`).

This means the chain of `(exact? (+ 1/3 1/6))` $\to$ `#t`, `(exact? (+ 1/3
0.5))` $\to$ `#f`, `(exact? (expt 2 100))` $\to$ `#t`, `(exact? (+ (expt 2
100) 0.0))` $\to$ `#f` all flow correctly through the dispatch tree.

---

## 9. Floor / ceil / round / truncate

The four R7RS rounding functions live in `codegenMathFunction`
(`llvm_codegen.cpp`). The dispatch keys
`func_name ∈ {"floor", "ceil", "trunc", "round"}` enter the rounding-specific
branch (line 16356) and follow a three-way path:

1. **Exact-int / bignum identity.** If the operand is INT64 or HEAP_PTR with
   subtype BIGNUM (`is_exact_int = arg_is_int || arg_is_bignum`, line 16360),
   return the operand unchanged. Floor/ceil/round/truncate of an integer are
   the integer.

2. **Rational path.** If the operand is HEAP_PTR with subtype RATIONAL
   (lines 16376–16399), call the appropriate runtime:
   - `floor` $\to$ `eshkol_rational_floor`
   - `ceil` $\to$ `eshkol_rational_ceil`
   - `trunc` $\to$ `eshkol_rational_truncate`
   - `round` $\to$ `eshkol_rational_round`
   each of which returns a raw int64 that is packed as exact int64.

3. **Float path.** Otherwise (operand is DOUBLE) call the libm function, with
   the special case `round` $\to$ `llvm.roundeven` (the LLVM IEEE 754 round-half-to-even
   intrinsic) at line 16407.

The PHI at line 16416 merges three incoming edges: exact-identity, rational,
float.

### 9.1 The PHI predecessor mismatch fix

A subtle correctness lesson is documented in `MEMORY.md` as the
"floor/ceil/round/truncate PHI predecessor mismatch": when
`extractDoubleFromTagged` is called inside `float_path`, it internally
creates basic blocks (specifically, the bignum / rational subdispatch in
`extractAsDouble`), shifting the builder's insertion point. The PHI for the
outer `regular_merge` therefore must capture the *actual* exit block of
`float_path` via `builder->GetInsertBlock()` after the `CreateBr(merge)`, not
the block originally allocated as `float_path`:

```cpp
BasicBlock* float_exit = builder->GetInsertBlock();  // line 16413
```

(`llvm_codegen.cpp`). The same precaution applies to `rational_exit`
(line 16399). Without these recaptures, the PHI claims a predecessor that
isn't its actual predecessor and the LLVM verifier rejects the module.

This is the same lesson the AD dispatch uses: any helper that may create
basic blocks (`extractAsDouble`, `extractDoubleFromTagged`,
`packDoubleToTaggedValue`, `isHeapSubtype`, `emitBignumCompareCall`,
`emitBignumBinaryCall`, etc.) requires the caller to recapture the insertion
block before adding incoming edges.

### 9.2 Two-argument round

`codegenRound(op)` (`llvm_codegen.cpp`) handles two-argument
`(round x precision)` separately, always in double:

```cpp
Value* scaled = builder->CreateFDiv(val, precision);
Function* roundeven_fn = ESHKOL_GET_INTRINSIC(module.get(), Intrinsic::roundeven, {double_type});
Value* rounded = builder->CreateCall(roundeven_fn, {scaled});
Value* result = builder->CreateFMul(rounded, precision);
return packDoubleToTaggedValue(result);
```

Two-argument round is never exact, even when both arguments are integers; the
output is always a double. This is consistent with the IEEE 754 round-half-to-even
semantic exposed by `llvm.roundeven`.

---

## 10. Number ↔ string conversion

### 10.1 Number to string

`numberToString(op)` (`string_io_codegen.cpp`) dispatches on the
operand's runtime type:

* **Two-argument form `(number->string n radix)`** — radix in $[2, 36]$,
  calls the C-level helper `eshkol_number_to_string_radix_raw`
  (`system_builtins.c`). This handler is int64-only; it does not
  yet support bignums or rationals in a non-decimal radix. The character set
  is `0-9a-z` (line 4633).

* **One-argument form** with runtime-typed tagged value:
  - HEAP_PTR with subtype = RATIONAL: call `eshkol_rational_to_string`
    (`rational.cpp`), which uses `snprintf("%lld/%lld", num, denom)`
    and returns an arena-allocated string with header. The denominator-1
    case (integer rational) elides the `/denom` portion.
  - HEAP_PTR with subtype = BIGNUM: call `eshkol_bignum_to_string`
    (`bignum.cpp`), which extracts decimal digits by repeated
    division by 10 using the limb-divisor fast path and reverses the buffer.
  - DOUBLE: `snprintf("%g", val)`.
  - INT64: `snprintf("%lld", val)`.

The PHI at line 832 merges four buffer pointers; double/int reuse a 64-byte
pre-allocated buffer, while rational and bignum allocate their own via the
runtime helpers. The header byte count is patched to the exact written length
via `truncate_header(written)` (line 730–743), because the allocator stamps
the header size to `buf_size + 1` regardless of how many bytes `snprintf`
actually wrote.

### 10.2 String to number

`stringToNumber(op)` (`string_io_codegen.cpp`) is a thin LLVM wrapper
that calls the runtime `eshkol_string_to_number_tagged(arena, str, result)`.
The runtime (`bignum.cpp`) implements the multi-strategy parse:

1. **Sentinel.** Empty string, whitespace-only, or leading non-digit
   non-`+/-/.` $\to$ return `#f` (the R7RS sentinel for "not a number"). The
   `#f` value is constructed as `eshkol_tagged_value_t{type=BOOL, data.int_val=0}`
   (lines 958–963).

2. **Syntax scan.** Walk the string looking for `/` (rational marker),
   `.`, `e`, or `E` (float marker). The presence flags `is_rational` and
   `is_float` direct the parse strategy.

3. **Rational syntax** `"num/denom"` (lines 998–1017):
   - Parse numerator via `strtoll`. If the endpoint is the slash and no
     `ERANGE`, parse denominator via `strtoll`. If neither overflowed and
     denom $\ne 0$, construct the rational via `eshkol_rational_create`.
   - Falls back to `#f` if any step fails.

4. **Float syntax** (lines 1019–1031):
   - `strtod`; the entire string must be consumed (modulo trailing
     whitespace). Returns DOUBLE on success, `#f` otherwise.

5. **Integer syntax** (lines 1033–1043):
   - `strtoll`; entire string must be consumed.
   - If `errno != ERANGE` and parse succeeded $\to$ return INT64.

6. **Integer overflow fallback** (lines 1045–1053):
   - If `strtoll` set `ERANGE`, parse as bignum via
     `eshkol_bignum_from_string(arena, start, strlen(start))`. If parse
     succeeds, return HEAP_PTR (BIGNUM) with EXACT flag set.

7. **Final `#f`** (line 1056) — none of the strategies matched.

This cascade ensures that `(string->number "99999999999999999999999999999")`
returns an exact bignum rather than losing precision through double
conversion. The fix is documented in `MEMORY.md` as
"string->number missing bignum".

### 10.3 What's *not* yet supported

The string-to-number parser intentionally does not handle:

* Radix prefixes `#b`, `#o`, `#d`, `#x` (binary, octal, decimal, hex). The
  R7RS-mandated prefixes are recognised by Eshkol's *lexer* in the source
  parser, but `string->number` operates on already-Eshkol-string inputs and
  currently rejects them.
* Exactness prefixes `#e`, `#i`. Same reason — `string->number` does not
  observe these markers.
* Complex syntax `a+bi`. The complex type has no string parser.

These are documented gaps; user code that needs e.g. hex parsing must wrap
the input with `(string-append "#x" hex)` and route through the lexer via
`read`, not through `string->number`.

---

## 11. The 35-gap audit closure

The v1.2 release closed a 35-gap audit of the bignum / rational dispatch
surface. The CHANGELOG entry at `CHANGELOG.md:723–726` records the closure:

> Bignum arithmetic: full 35-gap audit closed, including rational
> comparison, `abs`, `min`/`max` precision, `expt` with exact integer
> exponents, `number->string` / `string->number` bignum round-trip,
> and `bignum + double` → double per R7RS exact+inexact semantics.

The gaps fell into four classes, each illustrated by representative source
fixes:

### 11.1 Class A — dispatch fall-through (operand path skips bignum check)

The pattern: a primitive operates on an int64 fast path without first checking
whether either operand is a bignum HEAP_PTR. Because `unpackInt64` on a
HEAP_PTR returns the pointer-as-int64 (not the numeric value), the fast path
silently produces garbage.

**Representative fixes:**

* **`codegenModulo`** (`llvm_codegen.cpp`). The safe path now
  checks `emitIsBignumCheck` before the `SRem` (line 16634). Without this,
  `(modulo (* big-int 2) (expt 2 64))` returned a small multiple of the
  pointer.

* **`ArithmeticCodegen::compare`** (`arithmetic_codegen.cpp`). The
  bignum check (line 1856) and rational check (line 1871) precede the
  double/int dispatch. Before the fix, rationals fell through to the int
  path, which compared their *pointer addresses* — non-deterministic ordering
  depending on arena allocation order.

* **`ArithmeticCodegen::abs`** (`arithmetic_codegen.cpp`). The
  heap-pointer path (lines 1489–1529) handles bignums by `emitBignumCompareCall`
  + `emitBignumBinaryCall(op=7=neg)`. Before the fix, `codegenAbs` always
  routed through `extractAsDouble` and packed via `packDouble`, returning an
  inexact result for negative bignums.

### 11.2 Class B — precision loss on bignum → double

The pattern: a primitive normalises both operands to double via
`extractAsDouble` and operates on the doubles. For bignums larger than
$2^{53}$ this loses bits and can produce a *wrong* result, not merely an
inexact one.

**Representative fixes:**

* **`min` / `max`** (`arithmetic_codegen.cpp`). The bignum check
  dispatches to `emitBignumCompareCall` and selects between the *original*
  tagged operands, preserving exactness. Without the check, `(min (expt 2
  100) (+ (expt 2 100) 1))` would return whichever happens to round to the
  smaller double.

* **`compare`** for bignum-rational mixed operands. The rational path is
  reached only when neither operand is a bignum; if one is a bignum and the
  other a rational, the bignum-compare runtime promotes the rational to its
  double approximation via the `tagged_to_bignum` fallback — a documented
  precision loss for adversarial inputs. Closing this fully would require a
  bignum-rational comparison path; it's a documented but as-yet-unfilled
  gap.

### 11.3 Class C — exact-result-loss on integer exponents

* **`expt`** (`arithmetic_codegen.cpp` + `bignum.cpp`).
  Before the fix, `(expt 2 100)` always went through libm `pow`, returning a
  double approximation. After the fix:
  - codegen tests `base_is_exact && exp_is_int && exp_non_neg`,
  - dispatches into `eshkol_bignum_pow_tagged`,
  - runtime does repeated squaring in bignum arithmetic,
  - result is demoted via `eshkol_bignum_fits_int64`.
  This makes `(exact? (expt 2 100))` $\to$ `#t` and the printed value is
  exact.

### 11.4 Class D — ABI mismatch on tagged-value passing

* **`emitRationalCompareCall`** (`arithmetic_codegen.cpp`). The
  runtime entry point `eshkol_rational_compare_tagged_ptr`
  (`rational.cpp`) takes pointers to tagged values, not tagged values
  by value. The codegen `alloca`s the tagged values in the entry block and
  passes the pointers — avoiding ABI issues on platforms where 16-byte
  structs are passed in two registers or split across registers and the
  stack.

  The accompanying helper `eshkol_is_rational_tagged_ptr`
  (`rational.cpp`) wraps the pass-by-value
  `eshkol_is_rational_tagged` (`rational.cpp`) for the same reason.

### 11.5 Class E — number↔string round trip

* **`numberToString`** (`string_io_codegen.cpp`). The runtime-type
  dispatch added explicit HEAP_PTR subtype check at lines 775–805 (rational
  vs bignum). Before the fix, `(number->string (expt 2 100))` printed raw
  pointer bits.

* **`string->number`** (`bignum.cpp`). The cascade
  `strtoll` $\to$ `ERANGE` $\to$ bignum parse $\to$ `strtod` ensures that
  `(string->number "<long integer>")` produces exact bignum, not lossy
  double. Before the fix, the function used `strtod` only.

### 11.6 Class F — conversion-to-AD types missing bignum

* **`convertToComplex`** (`arithmetic_codegen.cpp`). The HEAP_PTR
  branch dispatches into `eshkol_bignum_to_double` (lines 400–408). Before
  the fix, bignums were treated as int64 via `unpackInt64` + `SIToFP`, which
  applied `SIToFP` to the pointer bits — producing a meaningless double.

* **`convertToDual`** (`arithmetic_codegen.cpp`). Same fix: the
  HEAP_PTR branch (lines 124–139) calls `eshkol_bignum_to_double`.

### 11.7 Five concrete fixes with citations

To make the audit's surface concrete, here are five representative fixes with
exact source locations:

1. **`codegenAbs` missing bignum path** — fixed by delegating to
   `ArithmeticCodegen::abs` at `llvm_codegen.cpp`, which handles bignum
   via `emitBignumCompareCall` / `emitBignumBinaryCall` at
   `arithmetic_codegen.cpp`.

2. **`ArithmeticCodegen::compare` missing rational path** — fixed by adding
   `emitIsRationalCheck` at `arithmetic_codegen.cpp` and
   `emitRationalCompareCall` at lines 1875–1877. Without this, rationals
   fell through to integer comparison on raw pointer bits.

3. **`min/max` precision loss for bignums > $2^{53}$** — fixed at
   `arithmetic_codegen.cpp` (min) and 2095–2102 (max). The bignum
   check uses exact 128-bit comparison and selects between the original
   operands.

4. **`bignum + double` returning 0** — fixed at `bignum.cpp`. The
   double check was hoisted above the `tagged_to_bignum` promotion attempt
   in `eshkol_bignum_binary_tagged`.

5. **`number->string` missing bignum** — fixed at
   `string_io_codegen.cpp`. The HEAP_PTR subtype is read at
   `ptr - 8`, then dispatches to `eshkol_bignum_to_string` or
   `eshkol_rational_to_string`.

A sixth fix worth mentioning, although technically Class A: the
**`floor/ceil/round/truncate` PHI predecessor mismatch** at
`llvm_codegen.cpp` — recapturing `float_exit` via
`builder->GetInsertBlock()` after `CreateBr(merge)`. This one was an LLVM
verifier failure rather than a silent wrong-answer bug.

### 11.8 Regression test coverage

`tests/v1_2_edge_cases/bignum_modulo_test.esk` is the documented regression
template (`MEMORY.md`: "Bignum dispatch in arithmetic codegen"). The pattern
extends: any new integer op added inline in `llvm_codegen.cpp` must include
a bignum-modulo-style regression test.

---

## 12. Performance characteristics

### 12.1 Int64 fast path

The int64 fast path is the cheapest by an order of magnitude. For `(+ a b)`
where both are int64:

* No allocation.
* Three LLVM instructions: `sadd.with.overflow.i64`, `extractvalue 0`,
  `extractvalue 1`, plus a `br` on the overflow flag (which the branch
  predictor handles trivially because overflow is rare).
* No function call into the runtime.

The `sadd.with.overflow.i64` intrinsic emits a single `addq` plus a
conditional move on x86, and a single `adds` on AArch64.

### 12.2 Bignum demotion

The demotion check `eshkol_bignum_fits_int64` (`bignum.cpp`) is
roughly four integer comparisons and a sign-conditional cast — cheaper than
the bignum allocation itself. Demotion ensures that "transient bignum"
patterns like

```scheme
(- (+ INT64_MAX 1) 1)
```

return to int64 representation. The intermediate `(+ INT64_MAX 1)` is a
bignum, but the final `-` step demotes back. Without demotion, the result
would be a HEAP_PTR with all the downstream allocator and dispatch costs.

### 12.3 Rational normalisation

Each rational arithmetic op pays for:

* One `__int128_t` multiplication per numerator (and one per denominator).
* One `gcd128` call on the result, which is $O(\log \min(n, d))$ iterations
  of 128-bit modulo.
* One `arena_allocate_with_header` if the result is a HEAP_PTR (skipped if
  the result is integer and is demoted to INT64).

For "small" rationals (numerator and denominator below $2^{32}$) the
`gcd128` is dominated by branch overhead rather than arithmetic. For larger
operands the 128-bit modulo becomes the hot path.

### 12.4 Complex arithmetic

Complex add/sub/mul/conj are four to eight FP ops with no allocation in the
arithmetic itself, but `packComplexToTagged` allocates 16 bytes on the arena
to store the result. Operations that chain (e.g. `(* z1 z2 z3)`) therefore
allocate $n - 1$ complex blocks for $n$ operands. The arena is bump-allocated
so per-block cost is two integer adds and a bounds check, but the cumulative
allocator pressure is visible in profiling for tight complex loops.

Complex division goes through Smith's formula's two-way branch; on modern
out-of-order cores the branch is well-predicted because the two paths have
similar latency and the decision is data-driven.

### 12.5 Double fast path

For pure-double code the dispatch tree of `add` evaluates seven `ICmpEQ`
predicates before reaching the double path. LLVM's instruction combiner
folds these into a single switch when the operand types are statically
known via the type-inference layer, but for fully dynamic code (e.g. values
flowing through `hash-table-ref`) the predicates remain.

---

## 13. Limitations and future work

The source explicitly defers four features:

### 13.1 Arbitrary-precision rationals

The rational struct holds int64 numerator and denominator. When
`rational_create_safe` (`rational.cpp`) cannot fit the reduced
fraction in int64, it returns `NULL`, triggering the
`eshkol_rational_binary_tagged` overflow-to-double fallback
(`rational.cpp`). The fall-back is documented as preserving R7RS's
"arithmetic never fails" guarantee at the cost of exactness. A future
extension would replace the int64 fields with bignum pointers; the
`rational.cpp` Audit C7 comment notes:

> legitimate rationals at INT64_MIN magnitude are rare and can be
> re-expressed once bignum rationals land.

### 13.2 Bignum rationals (transitively)

Same scope as 13.1, listed separately because the upgrade is non-trivial:
every arithmetic on a bignum-rational must dispatch through the bignum
runtime, requiring O(1) allocation for the numerator and denominator
intermediate even for tiny operands. The performance cost would be
significant for the common case.

### 13.3 Radix prefixes in `string->number`

The R7RS `#b`, `#o`, `#d`, `#x`, `#e`, `#i` prefixes are recognised by the
*lexer* but not by `string->number`. See §10.3.

### 13.4 Complex `string->number` / `number->string`

Complex numbers have no built-in string parser or printer in
`stringToNumber` / `numberToString`. User code can construct
`a + bi` strings manually via concatenation, but round-tripping through
`string->number` returns `#f`. The display system in
`lib/core/printer.cpp` does print complex numbers as `<real>+<imag>i`,
which is asymmetric.

### 13.5 Big-radix `number->string`

`eshkol_number_to_string_radix_raw` (`system_builtins.c`) is
int64-only. Calling `(number->string (expt 2 100) 16)` falls back to the
single-argument path because the codegen rejects non-int64 operands in the
two-argument form (`string_io_codegen.cpp`).

### 13.6 Quaternions / hypercomplex

Not present in the codebase. The `ESHKOL_VALUE_*` enum reserves space for
future type tags but no hypercomplex implementation exists.

### 13.7 IEEE 754 binary128 / extended

Eshkol is IEEE 754 binary64 throughout. There is no extended-precision FP
type. The codegen uses LLVM's `Double` type uniformly.

### 13.8 Decimal floating point

Not present. R7RS does not require it.

---

## 14. R7RS conformance summary

The R7RS numeric tower requirements that Eshkol fully implements:

* §6.2.1 — exact / inexact distinction with `EXACT_FLAG` / `INEXACT_FLAG`.
* §6.2.2 — number types: int, rational, real, complex (exact and inexact
  variants where applicable).
* §6.2.3 — exact + inexact $\to$ inexact (universal rule).
* §6.2.5 — predicates: `integer?`, `rational?`, `real?`, `complex?`,
  `number?`, `exact?`, `inexact?`.
* §6.2.6 — `+`, `-`, `*`, `/`, `abs`, `quotient`, `remainder`, `modulo`,
  `gcd`, `lcm`, `numerator`, `denominator`, `floor`, `ceiling`, `truncate`,
  `round`, `rationalize`, `exact`, `inexact`, `expt`.
* §6.2.7 — `<`, `>`, `<=`, `>=`, `=` with R7RS-mandated type validation.
* §6.2.10 — `string->number`, `number->string` (single-argument, plus
  int64-only two-argument with radix).

Partial:

* §6.2.6 — `rationalize` works for finite inputs via Stern–Brocot; for
  irrationals the iteration cap of 1000 may produce a near-but-not-exact
  result.
* §6.2.10 — radix prefixes for `string->number` not honoured.

---

## 15. References

### Headers

* `inc/eshkol/eshkol.h` — tagged value layout (lines 130–150),
  type/subtype constants (lines 70–113, 337–360), helper macros
  (lines 446–558).
* `inc/eshkol/core/bignum.h` — bignum API.
* `inc/eshkol/core/rational.h` — rational API.
* `inc/eshkol/backend/arithmetic_codegen.h` — codegen surface for
  polymorphic arithmetic.

### Runtime

* `lib/core/bignum.cpp` — sign-magnitude limbs, Knuth division, tagged
  dispatch, two's complement bitwise, string ↔ number.
* `lib/core/rational.cpp` — GCD reduction, 128-bit intermediates,
  tagged dispatch, rounding, rationalize.

### Codegen

* `lib/backend/arithmetic_codegen.cpp` — `add`, `sub`, `mul`, `div`,
  `mod`, `neg`, `abs`, `compare`, `pow`, `min`, `max`, `remainder`,
  `quotient`, `extractAsDouble`, the AD dispatch entry points, and the
  bignum / rational helpers.
* `lib/backend/complex_codegen.cpp` — complex arithmetic with Smith's
  division and overflow-safe magnitude.
* `lib/backend/string_io_codegen.cpp` — `numberToString` and
  `stringToNumber` codegen entry points.
* `lib/backend/llvm_codegen.cpp §codegenModulo`, `§codegenAbs`,
  `§codegenRound`, `§codegenMathFunction` (the rounding-function branch),
  `§codegenMinMax` — the dispatch sites the 35-gap audit fixed.

### Standards

* R7RS, §6.2 *Numbers*. Eshkol implements the R7RS-small numeric tower
  with the partial-conformance notes documented in §14.

### Project memory

* `~/.claude/projects/-Users-tyr-Desktop-eshkol/memory/MEMORY.md` — index
  of bug-and-fix entries, including the bignum-dispatch lessons documented
  here. The entries are point-in-time observations; this document supersedes
  them where source verification is required.
* `~/.claude/projects/-Users-tyr-Desktop-eshkol/memory/project_bignum_dispatch_pattern.md`
  — the dispatch pattern for new integer ops; cites the
  `codegenModulo` fix at commit `51ec814`.
