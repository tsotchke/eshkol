# The Numeric Tower

Eshkol implements a Scheme-style numeric tower with several representations that
arithmetic automatically promotes between:

- **fixnums** — machine integers,
- **bignums** — arbitrary-precision integers (automatic on overflow),
- **rationals** — exact `p/q` (written `1/3`),
- **inexact reals** — IEEE-754 doubles (written `1.5`, `3.0`),
- **complex** — `a+bi` (type tag 7, heap-allocated `{real, imag}`).

> **Off the tower: `i128`.** Eshkol also provides a native fixed-width 128-bit
> integer, [`i128`](i128.md). It is a *distinct type* that is **not** part of
> this tower: it does not auto-promote, and its arithmetic **wraps** at ±2¹²⁷
> (two's complement) instead of growing to a bignum. Use it for deterministic
> machine-word semantics; use the tower for ordinary exact arithmetic.

## Exactness

`exact?` and `inexact?` classify a number. Integers and rationals are exact;
doubles are inexact.

```scheme
(display (list (exact? 1/2) (inexact? 1.5))) (newline)
(display (/ 6 3)) (display " exact=") (display (exact? (/ 6 3))) (newline)
```
```
(#t #t)
2 exact=#t
```

Conversions: `exact->inexact` (a.k.a. `inexact`), `inexact->exact` (a.k.a. `exact`).
```scheme
(display (exact->inexact 1/4)) (newline)
```
```
0.25
```

`inexact->exact` preserves the exact IEEE-754 value of a finite double,
including values whose numerator or denominator requires the arbitrary-
precision representation. For example, `0.1` becomes
`3602879701896397/36028797018963968`, and the conversion round-trips through
`exact->inexact` for large values and subnormals as well.

Complex values use one external representation on both engines: a zero real
part is omitted and an imaginary part of `+1` or `-1` is printed as `+i` or
`-i`; other imaginary parts include their explicit sign (for example,
`2+3i`).

> **Display convention:** an inexact value with no fractional part prints without a
> decimal point — `(+ 1.0 2)` prints `3`, and `(+ 1/2 0.5)` prints `1` — but it is
> still inexact (`(inexact? (+ 1/2 0.5))` ⇒ `#t`).
>
> **Negative zero is the one exception:** `-0.0` prints `-0.0`, not `-0`. `-0`
> would read back as the *exact* integer zero, which has no sign, so both the
> inexactness and the sign bit would be lost — and the sign is observable
> (`(/ 1.0 -0.0)` ⇒ `-inf.0`, `(/ 1.0 0.0)` ⇒ `+inf.0`). Positive zero prints
> `0` like every other integral-valued double, since reading `0` back recovers
> the same numeric value.

> **Shortest round-trip printing (R7RS 6.2.6):** `display`, `write`, and
> `number->string` emit the **shortest decimal string that reads back as the
> identical `double`**. No digits are lost and none are fabricated: `(sqrt 2.0)`
> prints `1.4142135623730951`, and reading that string back yields the same bit
> pattern. Integral doubles keep the no-`.0` form described above. The native
> compiler and the bytecode VM share one portable-C conversion routine, so their
> printed output is byte-identical.

## Integers and bignums

Integer arithmetic promotes to bignum automatically; there is no overflow.

```scheme
(display (expt 2 100)) (newline)
(display (* 99999999999999999999 99999999999999999999)) (newline)
(display (list (quotient 17 5) (remainder 17 5) (modulo -7 3) (gcd 12 18) (lcm 4 6))) (newline)
```
```
1267650600228229401496703205376
9999999999999999999800000000000000000001
(3 2 2 6 12)
```

### Integer division (R7RS 6.2.6)

Three sign conventions, each with its R7RS synonym. `modulo` **is**
`floor-remainder`, `remainder` **is** `truncate-remainder`, and `quotient`
**is** `truncate-quotient` — the same procedure under two names, so they agree
on every representation.

| procedure | synonym | sign of result |
|---|---|---|
| `quotient` | `truncate-quotient` | truncates toward zero |
| `remainder` | `truncate-remainder` | sign of the **dividend** |
| `floor-quotient` | — | floors toward −∞ |
| `modulo` | `floor-remainder` | sign of the **divisor** |

`floor/` and `truncate/` return both halves as two values.

```scheme
(display (list (quotient -7 3) (remainder -7 3) (modulo -7 3))) (newline)
(display (list (floor-quotient -7 3) (floor-remainder -7 3))) (newline)
(display (call-with-values (lambda () (floor/ -7 3)) list)) (newline)
```
```
(-2 -1 2)
(-3 2)
(-3 2)
```

All of them accept bignums, and — because an integral flonum *is* an integer
(`(integer? 7.0)` ⇒ `#t`) — inexact operands too, where exactness contagion
makes the result inexact:

```scheme
(display (list (modulo (expt 2 100) 3) (floor-quotient (expt 2 100) 3))) (newline)
(display (list (remainder 7.0 2.0) (modulo -7.0 3.0) (modulo 5 2.0))) (newline)
(display (list (remainder 5.5 2.0) (modulo 5.5 2.0))) (newline)
(display (list (exact? (quotient 7 2)) (exact? (quotient 7.0 2.0)))) (newline)
```
```
(1 422550200076076467165567735125)
(1 2 1)
(1.5 1.5)
(#t #f)
```

> **Not C's `remainder()`.** The C library's `remainder(5.5, 2.0)` is `-0.5`
> (IEEE-754 round-to-*nearest* remainder). Scheme's `remainder` is the
> *truncated* remainder — `fmod` — so `(remainder 5.5 2.0)` is `1.5`.

An inexact result stays a flonum rather than being narrowed to a machine
integer, so magnitudes past 2^63 are carried rather than saturated —
`(quotient 1e20 3.0)` displays as `33333333333333330000` (the double
`3.3333333333333332e19`), not the largest machine integer.

A zero divisor is an error for `quotient`, `remainder`, `modulo` and their
synonyms (R7RS 6.2.6) and raises, for every operand representation — fixnum,
flonum and bignum alike. `/` is different: only an *exact* zero divisor is an
error there, and an inexact one is ordinary IEEE-754 division, so
`(/ (expt 2 100) 0.0)` is `+inf.0`.

## Rationals

Division of exact integers that do not divide evenly yields an exact rational,
kept in lowest terms.

```scheme
(display (/ 7 2)) (newline)
(display (+ 1/3 1/6)) (newline)        ; => 1/2
(display (* 2/3 3/4)) (newline)        ; => 1/2
(display (list (numerator 3/4) (denominator 3/4))) (newline)
```
```
7/2
1/2
1/2
(3 4)
```

### Rationals carry bignum components

A rational whose numerator or denominator is past the machine-integer range is
still an exact rational — on **both** engines. Nothing degrades to a `double`
at the bignum boundary, and nothing is clamped.

```scheme
(display (* 1/3 99999999999999999999)) (newline)
(display (+ 1/3 (expt 10 30))) (newline)
(display (/ 1 (expt 10 19))) (newline)
(display (+ 1/3 (expt 2 70))) (newline)
(display (exact? (+ 1/3 (expt 2 70)))) (newline)
(display (number->string (/ (expt 7 30) (expt 11 25)))) (newline)
```
```
33333333333333333333
3000000000000000000000000000001/3
1/10000000000000000000
3541774862152233910273/3
#t
22539340290692258087863249/108347059433883722041830251
```

Every line above is byte-identical under `eshkol-run -r`, under AOT and under
the bytecode VM; see [Engine agreement](#engine-agreement) for what changed
there in v1.3.5.

### Exact-rational temporaries are reclaimed

Exact-rational temporaries in a loop or a recursion are reclaimed on the same
terms as bignum-integer temporaries: resident memory tracks the values a
computation *keeps*, not the work it *does*. A long-running exact-rational
loop therefore holds a flat resident set, the way the identical loop over
bignum integers already did. See
[memory model](../runtime/memory-model.md).

## Exact roots and exact `expt`

Exactness is decided by the **value**, not by which operator was called. When
an exact result exists over the rationals, it is the answer; the inexact path
is used only when it does not.

```scheme
(display (list (sqrt 4/9) (sqrt 16) (expt 8 1/3) (expt 2/3 -3))) (newline)
(display (list (exact? (sqrt 4/9)) (exact? (expt 8 1/3)) (exact? (expt 2/3 -3)))) (newline)
(display (expt 1/3 50)) (newline)
(display (expt 2 -10)) (newline)
(display (sqrt 2)) (newline)
```
```
(2/3 4 2 27/8)
(#t #t #t)
1/717897987691852588770249
1/1024
1.4142135623730951
```

- `sqrt` of a perfect square — integer or rational — is exact.
- `expt` with an **exact rational exponent** is exact when the root exists
  (`(expt 8 1/3)` ⇒ `2`); otherwise it falls back to the correctly-rounded
  inexact result.
- `expt` with a **negative exponent** takes the reciprocal exactly rather than
  reconstructing it from a float, for a fixnum, bignum or rational base.
- Repeated exact powers stay exact at any magnitude: `(expt 1/3 50)` is the
  exact rational above, not `0` and not a double.

`sqrt 2` has no exact rational value, so it is the correctly-rounded double.
When you need a *proof-backed* bracket around an inexact elementary function
rather than a nearest neighbour, see
[certified enclosures](../stdlib/certified-enclosures.md), which builds
outward-rounded intervals and rigorous Taylor models on top of the
`fl-next-up` / `fl-next-down` directed-rounding builtins.

## Exactness in literals, vectors and tensors

An exact value survives every literal position it can appear in, quoted or
evaluated, on both engines.

```scheme
(define v #(1/2 3 1.5 123456789012345678901234567890))
(display v) (newline)
(display (list (exact? (vector-ref v 0)) (exact? (vector-ref v 3)))) (newline)
(display (tensor 1/2 2/3 1.5)) (newline)
(display '123456789012345678901234567890) (newline)
(display (quote 1/123456789012345678901234567890)) (newline)
(display `(x ,(/ 1 3) 123456789012345678901234567890)) (newline)
```
```
#(1/2 3 1.5 123456789012345678901234567890)
(#t #t)
#(0.5 0.6666666666666666 1.5)
123456789012345678901234567890
1/123456789012345678901234567890
(x 1/3 123456789012345678901234567890)
```

- A flat `#(…)` numeric literal keeps an exact rational or bignum element as
  the value it is.
- A **tensor** is a dense `f64` carrier by construction, so an exact element is
  converted once, explicitly, at construction — `1/2` becomes `0.5` there. That
  is a property of the tensor element type, not a loss of exactness in the
  tower: the same element in a `#(…)` vector stays `1/2`.
- A quoted or quasiquoted bignum / bignum-rational literal is the same value
  its evaluated form is, and prints identically.

## Exactness under differentiation

Exactness is a property of the runtime value the differentiation carrier holds,
not of the shape of the source. `derivative`, `gradient` and `hessian` return
an exact integer or rational whenever the arithmetic they perform is exact —
including when the constant comes from a top-level `define`, when the body is a
several-deep composed call, and when the point argument is an expression whose
runtime value happens to be exact.

```scheme
(define c 1/5)
(display (derivative (lambda (x) (* x x)) 1/3)) (newline)
(display (exact? (derivative (lambda (x) (* x x)) 1/3))) (newline)
(display (derivative (lambda (x) (* c x x)) 1/3)) (newline)
```
```
2/3
#t
2/15
```

See [the AD reference](../ad/INDEX.md) for the exactness tier in full, and for
the one nesting shape that is **not** supported in v1.3.5.

## Heap accounting is a fail-closed contract

The numeric tower allocates: a bignum, an exact rational and a complex are heap
values. Crossing a heap ceiling you asked for reports the breach **once**, in
bytes, and exits nonzero without completing — it never prints per arena block
and finishes with exit 0. With no ceiling requested the default is an
accounting reference and says nothing. A malformed `ESHKOL_MAX_HEAP` names
itself, the offending value and the accepted grammar before falling back to its
default. See
[environment variables](../runtime/environment-variables.md#resource-limits).

## Inexact reals

```scheme
(display (list (floor 3.7) (ceiling 3.2) (round 2.5) (truncate -3.7))) (newline)
(display (sqrt 16)) (display " ") (display (sqrt 2)) (newline)
```
```
(3 4 2 -3)
4 1.4142135623730951
```
`(round 2.5)` ⇒ `2`: rounding is round-half-to-even (banker's rounding).

## Complex numbers

```scheme
(display (make-rectangular 3 4)) (newline)
(display (magnitude (make-rectangular 3 4))) (newline)
(display (sqrt -1)) (newline)
```
```
3+4i
5
+i
```

## Contagion (mixed-type arithmetic)

Combining an exact operand with an inexact one yields an inexact result (R7RS
contagion): exact + inexact → inexact.

```scheme
(display (+ 1.0 2)) (newline)     ; 3 (inexact)
(display (+ 1/2 0.5)) (newline)   ; 1 (inexact)
```
```
3
1
```

## Engine agreement

The native back end and the bytecode VM answer identically across the whole
tower: fixnum, bignum, exact rational (bignum components included), inexact
real and complex — in literals, in arithmetic, and in `number->string` /
`display` / `write`. Mixed exact/inexact arithmetic agrees too:
`(* 0.5 1/3)` is `0.16666666666666666` on both.

Earlier releases carried a documented VM gap here (ESH-0105): the VM's rational
was an `int64` numerator over an `int64` denominator, its digit-token reader
fell through to `atof()` past `INT64_MAX`, and `number->string` coerced every
non-fixnum through a double, so `(* 1/3 99999999999999999999)` answered
`33333333333333330000` and `(/ 1 (expt 10 19))` answered `1e-19`. All three are
closed for v1.3.5: the VM shares one bignum/rational runtime between its
reader, its arithmetic and its printer, and the exact values shown under
[Rationals carry bignum components](#rationals-carry-bignum-components) are
what both engines print.

The differential is pinned by
`tests/vm_parity/corpus/79_bignum_rational_literals.esk` (native versus VM, on
both the source and the ESKB axis) and by
`tests/vm/bignum_rational_literals_test.esk`, which is compiled to `.eskb` and
run from there so the bytecode round trip is covered as well. A bignum or
bignum-rational literal compiles to ordinary `OP_CONST`/`OP_NATIVE_CALL`
bytecode, so it needed no ESKB format change and an existing module keeps
loading.
