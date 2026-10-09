---
kind: reference
status: current
owner-area: runtime
since: v1.3.3-evolve
sources:
  - lib/math/fixed_point/eshkol_fixed_point.h
  - lib/math/fixed_point/eshkol_fixed_point.c
  - docs/api/public_surface.tsv
---

# Fixed-Point, i128, and Exact Accumulation

A self-contained C ABI — `lib/math/fixed_point/eshkol_fixed_point.h` /
`eshkol_fixed_point.c` — that gives the compression stack and any other FFI
caller three building blocks with EXPLICIT, documented exactness and overflow
behavior instead of implicit floating-point rounding:

- **`esk_i128`** — a fixed-width, two's-complement signed 128-bit integer.
  Distinct from Eshkol's arbitrary-precision bignum tower, which never
  overflows; `esk_i128` wraps modulo 2^128, deterministically.
- **`esk_fixed`** — parametric fixed-point, `fixed<W,F>` (W total bits in
  `{32,64,128}`, F fractional bits), with the rounding mode and overflow
  policy passed explicitly on every operation rather than held as global
  state.
- **`esk_dot_exact_*` / `esk_idot_*` / `esk_imatmul_i32`** — block-scaled
  integer dot products and a row-major integer matmul with an `esk_i128`
  accumulator: the summation step is pure integer addition, which is
  associative and commutative, so the result is bit-for-bit
  ORDER-INDEPENDENT regardless of accumulation order. A dedicated contract
  test shuffles accumulation order and checks for a byte-identical result
  (see [Verification](#verification) below).

This is a C-level ABI, not yet an Eshkol (`.esk`) module — there is no
`(require ...)` surface for it. `inc/eshkol/core/i128.h` is a separate,
independently-maintained `__int128` core shared by the native/JIT/AOT runtime
and the bytecode VM for Eshkol's own boxed `i128` value type; its heap ABI
(`eshkol_i128_abi`) is deliberately laid out to match `esk_i128_abi` so the two
can unify later without a data migration, but the two headers are not
currently the same code path. See
[`docs/api/core/i128.md`](../../api/core/i128.md) for that runtime-facing type.

Primary source: [`lib/math/fixed_point/eshkol_fixed_point.h`](../../../lib/math/fixed_point/eshkol_fixed_point.h),
[`lib/math/fixed_point/eshkol_fixed_point.c`](../../../lib/math/fixed_point/eshkol_fixed_point.c).
Curated entries for every function and constant below are tracked in
[`docs/api/public_surface.tsv`](../../api/public_surface.tsv) and rendered at
[`docs/api/public_surface.md`](../../api/public_surface.md); follow a symbol's
link there for its one-line description and exact source line.

## FFI marshalling

`esk_i128` (native `__int128`) is exposed to foreign callers as the stable
two-`u64` struct `esk_i128_abi` (little-endian limb order: `lo` then `hi`),
converted with
[`esk_i128_to_abi`](../../api/public_surface.md#esk_i128_to_abi) /
[`esk_i128_from_abi`](../../api/public_surface.md#esk_i128_from_abi). An
`esk_fixed` value carries its own `(W, F)` alongside the raw `esk_i128`, so a
single ABI instance is enough for any `fixed<W,F>` instantiation:

```c
typedef struct { uint64_t lo, hi; } esk_i128_abi;     /* two-u64, lo then hi */
typedef struct { esk_i128 raw; uint8_t W, F; } esk_fixed;  /* value == raw / 2^F */
typedef struct { esk_i128 sum; uint64_t count; } esk_accum128;  /* running sum */
typedef struct { uint64_t state; } esk_rng;           /* SplitMix64 stream */
```

Two enums carry the explicit-per-operation policy that replaces global
floating-point rounding state:

| Type | Values | Meaning |
|---|---|---|
| `esk_round_mode` | `ESK_ROUND_TRUNCATE`, `ESK_ROUND_NEAREST_EVEN`, `ESK_ROUND_STOCHASTIC` | How a fixed-point operation rounds its fractional bits; `ESK_ROUND_STOCHASTIC` draws from the caller's `esk_rng` stream, rounding up with probability equal to the fractional part. |
| `esk_overflow_mode` | `ESK_OF_WRAP`, `ESK_OF_SATURATE` | What a W-bit-range-exceeding result does: wrap (two's-complement) or saturate to `[min, max]`. |

Requires a compiler with `__int128` (every Eshkol LLVM target — arm64,
x86-64 — provides it; a portable soft-i128 fallback is not yet built, per the
header's own `#error` guard).

## i128 engine

Construction, conversion, wrapping arithmetic, overflow-checked arithmetic,
decimal formatting/parsing. Compare: `esk_i128_from_i64`, `esk_i128_add`,
`esk_i128_shl` etc. are `static inline` (header-only, no exported symbol);
`esk_i128_from_parts`, `esk_i128_cmp`, `esk_i128_add_overflow` etc. are
genuine `extern` entry points defined in `eshkol_fixed_point.c`. Both kinds
are part of the public ABI — see each symbol's manifest row for which.

Limits: `ESK_I128_MAX` (2^127 - 1) and `ESK_I128_MIN` (-2^127) are exported
`const esk_i128` globals.

Full symbol list: [`public_surface.md`](../../api/public_surface.md) (search
`esk_i128_`).

## fixed<W,F>

`esk_fixed_make`/`esk_fixed_wmin`/`esk_fixed_wmax` construct and bound a
value; `esk_fixed_add`/`sub`/`neg`/`mul`/`div` operate on two values that
must share `(W, F)`, each taking an explicit `esk_overflow_mode` and
(for `mul`/`div`) `esk_round_mode`; `esk_fixed_from_i64`/`from_double`/`from_f32`
and `esk_fixed_to_i64`/`to_double`/`to_f32` convert at the boundary, each
reporting exactness through an optional `bool *exact` out-parameter so a
caller can tell a lossless conversion from a rounded one without re-deriving
it; `esk_fixed_convert` requantizes an existing value to a different
`(W, F)`, also with an exactness report; `esk_fixed_to_string` prints the
value as an exact decimal (every `fixed<W,F>` value is a dyadic rational, so
this never loses information).

`esk_fixed_mul` forms the full `2W`-bit product exactly (a 256-bit path when
`W=128`) before rounding and narrowing — the rounding/narrowing step is the
only place precision can be lost, and it is controlled by the caller's
`esk_round_mode`/`esk_overflow_mode`, never implicit.

Full symbol list: [`public_surface.md`](../../api/public_surface.md) (search
`esk_fixed_`).

## Exact accumulation and dot products

`esk_accum128` is a running-sum type (`sum: esk_i128`, `count: u64`) for
layer-reduction-style accumulation (e.g. LayerNorm/softmax denominators):
`esk_accum128_init` resets it, `esk_accum128_add_i64`/`add_i128` add one
element (`static inline`), `esk_accum128_merge` combines two partial
accumulators (summing both `sum` and `count` — the operation partial
reductions need to recombine without re-summing from scratch), and
`esk_accum128_value` reads the current sum.

`esk_dot_exact_i8`/`i16` compute the block-scaled dot product
`scale_a * scale_b * sum(a[i]*b[i])` as a `fixed<128,F>`: every
`int8*int8`/`int16*int16` product is widened into `esk_i128` before
accumulating, so the summation is EXACT and order-independent; only the
final `scale_a * scale_b` multiply can introduce rounding, reported through
the `bool *exact` out-parameter. `esk_idot_i8`/`i16`/`i32` expose that same
order-independent integer core directly, with no scaling at all.

`esk_imatmul_i32` is a row-major reference integer matmul,
`A[rows,inner] x B[inner,cols]` into `out[rows,cols]`, with every output
cell the stable two-`u64` ABI form of an exact `esk_i128` sum. It returns
`false` without reading or writing any matrix element if a required pointer
is `NULL` or a row-major size/index product would overflow `size_t`; an
empty output shape is a valid no-op, and when `inner == 0` the `A`/`B`
pointers may be `NULL` and every output cell is written as exact zero.

Full symbol list: [`public_surface.md`](../../api/public_surface.md) (search
`esk_dot_exact_`, `esk_idot_`, `esk_accum128_`, `esk_imatmul_i32`).

## Example

Compiled and run directly against `eshkol_fixed_point.c` at release SHA
`60f345def` (no Eshkol build needed — this module has no `.esk` binding
yet):

```sh
clang -std=c11 -O2 -I lib/math/fixed_point demo.c \
  lib/math/fixed_point/eshkol_fixed_point.c -lm -o demo && ./demo
```

```c
esk_fixed a = esk_fixed_from_double(3.25, 16, 8, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, NULL, &exact);
esk_fixed b = esk_fixed_from_double(1.5, 16, 8, ESK_ROUND_NEAREST_EVEN, ESK_OF_WRAP, NULL, &exact);
esk_fixed sum = esk_fixed_add(a, b, ESK_OF_WRAP, &overflow);
/* ... */
esk_fixed dot = esk_dot_exact_i8(x, y, 4, 1.0, 1.0, 8, &exact);  /* x={1,2,3,4}, y={4,3,2,1} */
esk_i128 raw = esk_idot_i8(x, y, 4);
```

Actual output:

```
3.25 + 1.5 = 4.75 (exact=1, overflow=0)
dot_exact_i8([1,2,3,4],[4,3,2,1]) = 20 (exact=1)
idot_i8 raw sum = 20
```

## Verification

The module ships its own standalone test suite, independent of the Eshkol
build tree: `tests/fixed_point/run_fixed_point_tests.sh` (no CMake, no
dependency on the Eshkol build). Re-run at release SHA `60f345def`:

```
$ tests/fixed_point/run_fixed_point_tests.sh
test_i128: 15/15 checks passed
test_fixed: 28/28 checks passed
test_dot_exact: 23/23 checks passed
ALL SUITES PASSED (3 suites)
```

`test_dot_exact.c`'s contract tests are the order-independence claim made
executable: `idot_i32` is checked against an independent wide-accumulator
reference, and a 200-shuffle and a separate 20-shuffle-at-max-magnitude
pass both assert a byte-identical `esk_i128` result across every shuffled
accumulation order, for both the raw dot products and `esk_imatmul_i32`
(50 contraction orders). `test_fixed.c` covers rounding-mode and
overflow-mode combinations, conversion exactness flags, and division by
zero. `test_i128.c` covers wraparound, widening multiply, shift, and
decimal string round-trips.
