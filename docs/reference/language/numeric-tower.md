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

> **Display convention:** an inexact value with no fractional part prints without a
> decimal point — `(+ 1.0 2)` prints `3`, and `(+ 1/2 0.5)` prints `1` — but it is
> still inexact (`(inexact? (+ 1/2 0.5))` ⇒ `#t`).

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

## Inexact reals

```scheme
(display (list (floor 3.7) (ceiling 3.2) (round 2.5) (truncate -3.7))) (newline)
(display (sqrt 16)) (display " ") (display (sqrt 2)) (newline)
```
```
(3 4 2 -3)
4 1.41421
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

## Known issue — rationals degrade near the bignum boundary (ESH-0105)

Exact rational arithmetic silently loses exactness — or returns a wrong value —
once a **bignum** operand is involved, instead of producing an exact result or
signalling an error.

```scheme
(display (* 1/3 99999999999999999999)) (newline)   ; expected 33333333333333333333
(display (+ 1/3 (expt 10 30))) (newline)            ; expected 3000…0001/3
(display (/ 1 (expt 10 19))) (newline)              ; expected exact 1/10000000000000000000
```
```
0
1000000000000000000000000000000
1e-19
```
Observed: `(* 1/3 <bignum>)` returns `0`; `(+ 1/3 <bignum>)` drops the fraction;
`(/ 1 (expt 10 19))` degrades to an inexact double (`1e-19`) whereas
`(/ 1 (expt 10 18))` is still an exact rational. Keep rational computations within
the fixnum range for exact results, or convert deliberately with
`exact->inexact` when you want doubles.
