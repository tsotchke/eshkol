---
kind: reference
status: current
owner-area: stdlib
since: v1.3.2-evolve
sources:
  - lib/core/blc.esk
---

# `core.blc` — Binary Lambda Calculus

**Source**: [`lib/core/blc.esk`](../../../lib/core/blc.esk)
**Require**: `(require core.blc)` — **must be required individually**; there is no `(require core.blc)` line in `lib/stdlib.esk`, so `(require stdlib)` alone does **not** load it. Calling `blc-encode` etc. after only `(require stdlib)` fails with *"called undefined function 'blc-encode'"*.

A pure-Eshkol implementation of John Tromp's Binary Lambda Calculus: a bit-level
encoding of De Bruijn-indexed lambda terms, plus a faithful normal-order
(leftmost-outermost) beta-reducer. For the tutorial-style walkthrough — term
representation rationale, the encoding grammar, the universal machine, BLC8
byte I/O and lambda diagrams with worked examples — see the
[Binary Lambda Calculus guide](../../guide/BINARY_LAMBDA_CALCULUS.md). This
page is the per-export contract: exact signature, return shape, step limits
and error text, taken from reading `lib/core/blc.esk` directly. All samples
below were run under both the JIT (`eshkol-run -r`) and AOT (`eshkol-run -o`)
engines; JIT and AOT outputs are byte-identical for every sample on this page.

## Term representation

Terms are homoiconic s-expressions, tagged with a symbol as the `car`:

| Form        | Meaning                        |
| ----------- | ------------------------------- |
| `(var i)`   | variable, `i` a **1-based** De Bruijn index (index `1` = innermost binder) |
| `(lam B)`   | abstraction over body `B`       |
| `(app M N)` | application of `M` to `N`       |

None of the constructors or accessors validate their argument's shape beyond
what `list`/`cadr`/`caddr` do implicitly; `blc-term?` is the only structural
validator (see below).

## Functions

### `(blc-var i)`
Constructs a variable term `(var i)`. Does not check that `i` is a positive
integer — `blc-term?` and `blc-encode` are what enforce that.

```scheme
(require core.blc)
(display (blc-var 1)) (newline)
```
```
(var 1)
```

### `(blc-lam b)`
Constructs an abstraction term `(lam b)` over body `b`.

```scheme
(require core.blc)
(display (blc-lam (blc-var 1))) (newline)
```
```
(lam (var 1))
```

### `(blc-app m n)`
Constructs an application term `(app m n)`.

```scheme
(require core.blc)
(display (blc-app (blc-var 1) (blc-var 2))) (newline)
```
```
(app (var 1) (var 2))
```

### `(blc-var? t)`
`#t` iff `t` is a pair whose `car` is `'var`.

```scheme
(require core.blc)
(display (blc-var? (blc-var 1))) (newline)
(display (blc-var? (blc-lam (blc-var 1)))) (newline)
```
```
#t
#f
```

### `(blc-lam? t)`
`#t` iff `t` is a pair whose `car` is `'lam`.

```scheme
(require core.blc)
(display (blc-lam? (blc-lam (blc-var 1)))) (newline)
(display (blc-lam? (blc-var 1))) (newline)
```
```
#t
#f
```

### `(blc-app? t)`
`#t` iff `t` is a pair whose `car` is `'app`.

```scheme
(require core.blc)
(display (blc-app? (blc-app (blc-var 1) (blc-var 2)))) (newline)
(display (blc-app? (blc-var 1))) (newline)
```
```
#t
#f
```

### `(blc-var-index t)`
Accessor: `(cadr t)`. Assumes `t` is a `(var i)` term; no shape check.

```scheme
(require core.blc)
(display (blc-var-index (blc-var 3))) (newline)
```
```
3
```

### `(blc-lam-body t)`
Accessor: `(cadr t)`. Assumes `t` is a `(lam b)` term.

```scheme
(require core.blc)
(display (blc-lam-body (blc-lam (blc-var 1)))) (newline)
```
```
(var 1)
```

### `(blc-app-fun t)`
Accessor: `(cadr t)`. The function (left) operand of `(app m n)`.

```scheme
(require core.blc)
(display (blc-app-fun (blc-app (blc-var 1) (blc-var 2)))) (newline)
```
```
(var 1)
```

### `(blc-app-arg t)`
Accessor: `(caddr t)`. The argument (right) operand of `(app m n)`.

```scheme
(require core.blc)
(display (blc-app-arg (blc-app (blc-var 1) (blc-var 2)))) (newline)
```
```
(var 2)
```

### `(blc-term? t)`
Recursively validates that `t` is a well-formed term: every `(var i)` has an
integer index `>= 1`, and `(lam ...)`/`(app ...)` recurse into their
subterms. Anything else (including a bare symbol, an improper tag, or a
`(var i)` with `i < 1` or non-integer `i`) yields `#f` rather than erroring.

```scheme
(require core.blc)
(display (blc-term? (blc-lam (blc-var 1)))) (newline)
(display (blc-term? (blc-var 0))) (newline)
(display (blc-term? '(quux 1))) (newline)
```
```
#t
#f
#f
```

### `(blc-encode term)`
Encodes `term` to a bit string of `#\0`/`#\1` characters per Tromp's grammar:
`(lam M)` → `"00"` + `blc(M)`; `(app M N)` → `"01"` + `blc(M)` + `blc(N)`;
`(var i)` → `i` copies of `"1"` followed by `"0"`. Signals
`"blc-encode: variable index must be a positive integer"` if a `(var i)` has
`i < 1` or a non-integer index, and `"blc-encode: not a valid BLC term"` for
any node that is not a `var`/`lam`/`app` tag (see [Errors](#blc-encode-errors)
below).

```scheme
(require core.blc)
(display (blc-encode blc-I)) (newline)
(display (blc-encode blc-K)) (newline)
(display (blc-encode blc-pair)) (newline)
```
```
0010
0000110
0000000101101110110
```

<a id="blc-encode-errors"></a>Malformed input:

```scheme
(require core.blc)
(display (blc-encode (blc-var 0))) (newline)
```
```
Unhandled exception: blc-encode: variable index must be a positive integer
```

### `(blc-decode bits)`
Decodes one complete term from the front of bit string `bits` and **silently
ignores any trailing bits** (the encoding is self-delimiting, so a decoder
does not need to know where the string ends). Round-trips with `blc-encode`.
Errors rather than looping or crashing on malformed input — see
[Errors](#blc-decode-errors).

```scheme
(require core.blc)
(display (blc-decode "0010")) (newline)
(display (equal? blc-K (blc-decode (blc-encode blc-K)))) (newline)
```
```
(lam (var 1))
#t
```

<a id="blc-decode-errors"></a>Error text, read directly from the source (each
signalled by `error`, so `guard`/`with-exception-handler` can distinguish
them by message):

| Condition                                        | Message |
| ------------------------------------------------- | ------- |
| `bits` is not a string, or `""`                   | `"blc-decode: empty input, no term to decode"` |
| position runs off the end while decoding a term   | `"blc-decode: unexpected end of input (truncated term)"` |
| string ends right after a leading `"0"`           | `"blc-decode: unexpected end of input after '0'"` |
| a character other than `#\0`/`#\1` is seen        | `"blc-decode: invalid bit character (expected 0 or 1)"` |
| a run of `1`s runs off the end (missing terminating `0`) | `"blc-decode: unexpected end of input in variable (missing terminating 0)"` |
| a `"0"` immediately terminates a variable with 0 leading `1`s | `"blc-decode: zero-length variable"` |

```scheme
(require core.blc)
(display (blc-decode "11")) (newline)
```
```
Unhandled exception: blc-decode: unexpected end of input in variable (missing terminating 0)
```

### `(blc-shift d cutoff term)`
De Bruijn shift: adds `d` to every index of `term` that is `>= cutoff` (i.e.
every variable *free* relative to `cutoff`); indices below `cutoff` are
untouched. Descending under a `lam` raises the cutoff by 1, so a binder's own
newly-bound variable is never shifted.

```scheme
(require core.blc)
;; both indices are >= cutoff 1, so both shift by 1
(display (blc-shift 1 1 (blc-app (blc-var 1) (blc-var 2)))) (newline)
;; under one binder: the binder's own var (index 1, cutoff becomes 2) stays;
;; the outer free var (index 2 at the outer cutoff) shifts
(display (blc-shift 1 1 (blc-lam (blc-app (blc-var 1) (blc-var 2))))) (newline)
```
```
(app (var 2) (var 3))
(lam (app (var 1) (var 3)))
```

### `(blc-subst term j s)`
De Bruijn substitution: replaces every `(var j)` in `term` with `s`, raising
`j` by 1 and shifting `s` by `(blc-shift 1 1 s)` each time the walk descends
under a `lam` (so `s`'s free variables stay correctly indexed once captured
under the new binder).

```scheme
(require core.blc)
;; replace (var 1) with K inside (app (var 1) (var 2))
(display (blc-subst (blc-app (blc-var 1) (blc-var 2)) 1 blc-K)) (newline)
```
```
(app (lam (lam (var 2))) (var 2))
```

### `(blc-step term)`
Performs **one** leftmost-outermost reduction step. Returns a pair
`(reduced? . term')`: `reduced?` is `#t` if a redex was contracted (`term'`
is the new term) or `#f` if `term` was already in normal form (`term'` is
`term` unchanged). Because `term'` is itself a list, `display` on the
returned pair prints it flattened, e.g. `(#t app ...)` rather than
`(#t . (app ...))`. At an application whose function position is already an
abstraction, that (outermost) redex is contracted immediately; otherwise the
function is reduced first, then the argument; reduction also proceeds under
`lam` so that repeated stepping reaches full beta normal form.

```scheme
(require core.blc)
;; one step of (K I S): the outer redex (K I) contracts to I
(display (blc-step (blc-app (blc-app blc-K blc-I) blc-S))) (newline)
;; a normal-form term does not reduce: reduced? is #f, term' is term
(display (blc-step blc-I)) (newline)
```
```
(#t app (lam (lam (var 1))) (lam (lam (lam (app (app (var 3) (var 1)) (app (var 2) (var 1)))))))
(#f lam (var 1))
```

### `(blc-eval term)`
Repeatedly applies `blc-step` until it reports no further reduction
(`reduced?` is `#f`), returning the beta normal form. Uses **normal-order**
(leftmost-outermost) reduction, so it reaches a normal form whenever one
exists even when a divergent subterm is discarded before it would ever be
reduced (applicative order would loop on such terms). Step-counts against
[`blc-max-steps`](#blc-max-steps); once the count exceeds the bound, signals
`"blc-eval: no normal form within step bound"` instead of looping forever.

```scheme
(require core.blc)
;; K I S = I
(display (blc-eval (blc-app (blc-app blc-K blc-I) blc-S))) (newline)
;; S K K I = I
(display (blc-eval (blc-app (blc-app (blc-app blc-S blc-K) blc-K) blc-I))) (newline)
```
```
(lam (var 1))
(lam (var 1))
```

Exceeding the step bound (`blc-omega` has no normal form by construction):

```scheme
(require core.blc)
(display (blc-eval blc-omega)) (newline)
```
```
Unhandled exception: blc-eval: no normal form within step bound
```

### `(blc-apply program arg)`
`(blc-eval (blc-app program arg))` — apply `program` to `arg` and reduce to
normal form. Subject to the same `blc-max-steps` bound as `blc-eval`.

```scheme
(require core.blc)
(display (blc-apply blc-K blc-I)) (newline)
```
```
(lam (lam (var 1)))
```

### `(blc->debruijn-string term)`
Pretty-prints `term` in compact De Bruijn notation: `λ` per abstraction (only
the innermost abstraction in a run gets the `.` before its body), a bare
digit per variable, application by juxtaposition of the rendered operands.
Parenthesization rule: the left (function) operand of an application is
parenthesized only if it is itself an abstraction; the right (argument)
operand is parenthesized whenever it is compound (abstraction or
application) — a bare variable never needs parens.

```scheme
(require core.blc)
(display (blc->debruijn-string blc-I)) (newline)
(display (blc->debruijn-string blc-K)) (newline)
(display (blc->debruijn-string blc-S)) (newline)
```
```
λ.1
λλ.2
λλλ.31(21)
```

## Predefined terms and constants

### `blc-I`
`I = λx.x` = `(lam (var 1))`.

```scheme
(require core.blc)
(display blc-I) (newline)
```
```
(lam (var 1))
```

### `blc-K`
`K = λx.λy.x` = `(lam (lam (var 2)))`.

```scheme
(require core.blc)
(display blc-K) (newline)
```
```
(lam (lam (var 2)))
```

### `blc-S`
`S = λx.λy.λz.((x z)(y z))`, i.e. `(lam (lam (lam (app (app (var 3) (var 1)) (app (var 2) (var 1))))))`.

```scheme
(require core.blc)
(display blc-S) (newline)
```
```
(lam (lam (lam (app (app (var 3) (var 1)) (app (var 2) (var 1))))))
```

### `blc-I-bits`
`(blc-encode blc-I)`, given as a literal for convenience.

```scheme
(require core.blc)
(display blc-I-bits) (newline)
```
```
0010
```

### `blc-K-bits`
`(blc-encode blc-K)`, given as a literal.

```scheme
(require core.blc)
(display blc-K-bits) (newline)
```
```
0000110
```

### `blc-S-bits`
`(blc-encode blc-S)`, given as a literal (24 bits).

```scheme
(require core.blc)
(display blc-S-bits) (newline)
```
```
00000001011110100111010
```

### `blc-true`
Church boolean `True = λx.λy.x`; structurally identical to `blc-K` (both are
`(lam (lam (var 2)))`), defined separately for readability at call sites.

```scheme
(require core.blc)
(display blc-true) (newline)
```
```
(lam (lam (var 2)))
```

### `blc-false`
Church boolean `False = λx.λy.y` = `(lam (lam (var 1)))`.

```scheme
(require core.blc)
(display blc-false) (newline)
```
```
(lam (lam (var 1)))
```

### `blc-omega`
`Ω = (λx.x x)(λx.x x)` — the classic divergent term, with no beta normal
form. Used to exercise the `blc-max-steps` cap (see
[`blc-eval`](#blc-eval-term)); `blc-eval` on it always signals the step-bound
error rather than hanging.

```scheme
(require core.blc)
(display blc-omega) (newline)
```
```
(app (lam (app (var 1) (var 1))) (lam (app (var 1) (var 1))))
```

### `blc-max-steps`
Upper bound (`1000000`) on the number of `blc-step` contractions `blc-eval`
(and hence `blc-apply`) will perform before giving up on a term. It is an
ordinary top-level `define`, not a parameter — there is no supported way to
rebind it for a single call; changing the bound means editing this constant
in `lib/core/blc.esk`. `blc-eval-loop` compares the running step count
against it with `>` (strictly greater), so a term that finishes in exactly
`blc-max-steps` steps still succeeds — the bound is only exceeded, and only
then does evaluation abort, once one more step would have been attempted.

```scheme
(require core.blc)
(display blc-max-steps) (newline)
```
```
1000000
```

## Bit streams (Scott-list encoding)

Tromp's I/O convention: bit `0` is encoded as `blc-true`, bit `1` as
`blc-false`; a stream of bits is a Scott list built from the pairing
combinator `blc-pair`, terminated by `blc-nil` when finite.

### `blc-pair`
The pairing combinator `λh.λt.λf. f h t` (De Bruijn `λλλ.((1 3) 2)`), whose
`blc-encode` is `"0000000101101110110"` (19 bits). `blc-cons` below builds
cons *values* by partially applying this combinator's body directly, not by
calling `blc-pair` itself.

```scheme
(require core.blc)
(display blc-pair) (newline)
(display (blc-encode blc-pair)) (newline)
```
```
(lam (lam (lam (app (app (var 1) (var 3)) (var 2)))))
0000000101101110110
```

### `blc-nil`
List terminator for finite streams; defined as `blc-false` (`(equal? blc-nil blc-false)` is `#t`), so any code that only checks a stream head against `blc-false` also accepts `blc-nil`.

```scheme
(require core.blc)
(display blc-nil) (newline)
(display (equal? blc-nil blc-false)) (newline)
```
```
(lam (lam (var 1)))
#t
```

### `(blc-cons h t)`
Builds a cons **value** `⟨h,t⟩ = λf. f h t`. `h` and `t` must already be
closed terms (booleans or sub-lists) — `blc-cons` does no shifting, so a term
with free variables passed as `h`/`t` will have those variables silently
misindexed once the result is placed under further binders.

```scheme
(require core.blc)
(display (blc-cons blc-true blc-nil)) (newline)
```
```
(lam (app (app (var 1) (lam (lam (var 2)))) (lam (lam (var 1)))))
```

### `(blc-list-of-bits bits tail)`
Converts bit string `bits` (characters `#\0`/`#\1`) to a Scott list of
booleans (`#\0` → `blc-true`, `#\1` → `blc-false`), terminated by `tail`
(pass `blc-nil` for a finite, `blc-nil`-terminated list). Signals
`"blc: bit character must be 0 or 1"` for any other character.

```scheme
(require core.blc)
(display (blc-list-of-bits "10" blc-nil)) (newline)
```
```
(lam (app (app (var 1) (lam (lam (var 1)))) (lam (app (app (var 1) (lam (lam (var 2)))) (lam (lam (var 1)))))))
```

### `(blc-encode-input program-bits input-bits)`
`(blc-list-of-bits (string-append program-bits input-bits) blc-nil)` — builds
the single `blc-nil`-terminated Scott list that [`blc-U`](#blc-u) consumes:
a self-delimiting program encoding immediately followed by input bits.

```scheme
(require core.blc)
(display (blc-encode-input (blc-encode blc-I) "10")) (newline)
```
```
(lam (app (app (var 1) (lam (lam (var 2)))) (lam (app (app (var 1) (lam (lam (var 2)))) (lam (app (app (var 1) (lam (lam (var 1)))) (lam (app (app (var 1) (lam (lam (var 2)))) (lam (app (app (var 1) (lam (lam (var 1)))) (lam (app (app (var 1) (lam (lam (var 2)))) (lam (lam (var 1)))))))))))))))
```

### `(blc-list-null? t)`
`(equal? t blc-nil)`. Since `blc-nil` is `blc-false`, this is also `#t` for
`blc-false` itself, not just for terms actually built by this module's list
functions.

```scheme
(require core.blc)
(display (blc-list-null? blc-nil)) (newline)
(display (blc-list-null? (blc-cons blc-true blc-nil))) (newline)
```
```
#t
#f
```

### `(blc-list-cons? t)`
Structural check: `t` is a `lam` whose body is an `app` of an `app` whose
function is `(var 1)` — i.e. `t` has the exact shape `blc-cons` produces.
Does not check that the head/tail contents are themselves valid; only the
outer cons shape.

```scheme
(require core.blc)
(display (blc-list-cons? (blc-cons blc-true blc-nil))) (newline)
(display (blc-list-cons? blc-nil)) (newline)
```
```
#t
#f
```

### `(blc-list-head t)`
Extracts the head of a cons value produced by `blc-cons`:
`(blc-app-arg (blc-app-fun (blc-lam-body t)))`. Assumes `t` has that exact
shape (check with `blc-list-cons?` first); does not validate.

```scheme
(require core.blc)
(display (equal? (blc-list-head (blc-cons blc-true blc-nil)) blc-true)) (newline)
```
```
#t
```

### `(blc-list-tail t)`
Extracts the tail of a cons value: `(blc-app-arg (blc-lam-body t))`. Same
no-validation caveat as `blc-list-head`.

```scheme
(require core.blc)
(display (equal? (blc-list-tail (blc-cons blc-true blc-nil)) blc-nil)) (newline)
```
```
#t
```

## The universal machine U

### `blc-U-bits`
Tromp's 232-bit self-interpreter, as a literal bit string. Cross-checked in
the source by re-encoding Tromp's published De Bruijn term for `U` with this
module's own `blc-encode` and confirming byte-for-byte agreement.

```scheme
(require core.blc)
(display (string-length blc-U-bits)) (newline)
(display blc-U-bits) (newline)
```
```
232
0101000110100000000101011000000000011110000101111110011110000101110011110000001111000010110110111001111100001111100001011110100111010010110011100001101100001011111000011111000011100110111101111100111101110110000110010001101000011010
```

### `(blc-U)`
`(blc-decode blc-U-bits)` — the universal machine as a decoded term (a
thunk, not a value, so it must be called: `(blc-U)`, not `blc-U`). Applied
via `blc-eval`/`blc-apply` to a Scott list built by
[`blc-encode-input`](#blc-encode-input-program-bits-input-bits) — a
self-delimiting program encoding `M` followed by input bits — `U` reduces to
`M` applied to the remaining input list.

```scheme
(require core.blc)
(display (equal? (blc-encode (blc-U)) blc-U-bits)) (newline)
```
```
#t
```

## BLC8 byte I/O

A byte is a `blc-nil`-terminated Scott list of 8 booleans in **big-endian**
order (most significant bit first); a byte *string* is a `blc-nil`-terminated
list of such byte-lists.

### `(blc-byte->term b)`
Integer byte `b` (0–255) → `blc-nil`-terminated list of 8 booleans, MSB
first. Built with `(modulo (quotient b (expt 2 k)) 2)` for `k` from 7 down to
0; out-of-range `b` is not checked and simply produces whatever bits that
arithmetic yields.

```scheme
(require core.blc)
(display (blc-term->byte (blc-byte->term 72))) (newline)
```
```
72
```

### `(blc-bytes->term bytes)`
List of integer bytes → BLC8 term (a Scott list of byte-lists), via
`blc-byte->term` per element and `blc-nil` at `'()`.

```scheme
(require core.blc)
(display (blc-term->bytes (blc-bytes->term (list 72 105)))) (newline)
```
```
(72 105)
```

### `(blc-term->byte t)`
Reads exactly 8 bits (MSB first) off Scott-list `t` via `blc-list-cons?`/
`blc-list-head`/`blc-list-tail`, returning the integer byte. Signals
`"blc-term->byte: byte has fewer than 8 bits"` if the list ends (fails
`blc-list-cons?`) before 8 bits are read.

```scheme
(require core.blc)
(display (blc-term->byte (blc-byte->term 72))) (newline)
```
```
72
```

### `(blc-term->bytes t)`
BLC8 term (Scott list of byte-lists) → list of integer bytes, by
`blc-term->byte` on `blc-list-head` at each step until `blc-list-null?`.

```scheme
(require core.blc)
(display (blc-term->bytes (blc-string->term "Hi"))) (newline)
```
```
(72 105)
```

### `(blc-string->term s)`
Scheme string `s` → BLC8 term, one byte per character via
`(char->integer (string-ref s i))`. ASCII/Latin-1 range only follows from
`char->integer`'s own range; this function does no separate validation.

```scheme
(require core.blc)
(display (blc-term->string (blc-string->term "Hi"))) (newline)
```
```
Hi
```

### `(blc-term->string t)`
Inverse of `blc-string->term`: BLC8 term → Scheme string, one character per
byte via `(integer->char ...)`.

```scheme
(require core.blc)
(display (blc-term->string (blc-string->term "Hi"))) (newline)
```
```
Hi
```

## Lambda diagrams

### `(blc-diagram term)`
Renders `term` as a Tromp-style ASCII lambda diagram
(<https://tromp.github.io/cl/diagrams.html>): each abstraction is a
horizontal bar; each variable occurrence is a vertical line rising to the
bar of the lambda that binds it; each application is a horizontal link
joining the leftmost variable columns of its two subterms. Variable columns
are spaced 4 apart, so a term with `V` variable occurrences renders `4V-1`
columns wide. Returns a single (possibly multi-line, `\n`-joined,
right-trimmed) string; the caller must `display` or `newline` it themselves
to see the line breaks.

```scheme
(require core.blc)
(display (blc-diagram blc-I)) (newline)
(display "---") (newline)
(display (blc-diagram blc-K)) (newline)
(display "---") (newline)
(display (blc-diagram blc-S)) (newline)
```
```
---
 |
---
---
-|-
 |
---
---------------
-|-------------
-|-------|-----
 |   |   |   |
 |----   |----
 |--------
```

## Known issues

None.
