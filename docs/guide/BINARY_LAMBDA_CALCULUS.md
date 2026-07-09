# Binary Lambda Calculus in Eshkol

*The `core.blc` module — a pure-Eshkol implementation of John Tromp's
Binary Lambda Calculus (BLC).*

Eshkol is a compiled R7RS Scheme, and Scheme is lambda calculus with a
concrete syntax. The `core.blc` module makes that lineage explicit: it
encodes De Bruijn-indexed lambda terms as bit strings, decodes them back,
and reduces them to normal form — all in ordinary Eshkol, with terms
represented as plain s-expressions.

Load it on demand:

```scheme
(require core.blc)
```

## Term representation

Terms are homoiconic s-expressions. De Bruijn indices are **1-based**:
index `1` refers to the innermost enclosing binder.

| Form        | Meaning                          | Constructor         |
| ----------- | -------------------------------- | ------------------- |
| `(var i)`   | variable, `i >= 1` (De Bruijn)   | `(blc-var i)`       |
| `(lam B)`   | abstraction over body `B`        | `(blc-lam B)`       |
| `(app M N)` | application of `M` to `N`        | `(blc-app M N)`     |

Predicates `blc-var?`, `blc-lam?`, `blc-app?`, `blc-term?` and accessors
`blc-var-index`, `blc-lam-body`, `blc-app-fun`, `blc-app-arg` are provided.
Because terms are just lists, a quoted literal such as `'(lam (var 1))` is
structurally equal to `(blc-lam (blc-var 1))`.

## The bit encoding

BLC encodes a term as bits (Tromp):

| Term          | Encoding                                  |
| ------------- | ----------------------------------------- |
| `(lam M)`     | `00` · `blc(M)`                           |
| `(app M N)`   | `01` · `blc(M)` · `blc(N)`                |
| `(var i)`     | (`1` repeated `i` times) · `0`  (1-based) |

The encoding is **self-delimiting**: a term's bit string carries its own
end, so terms can be concatenated unambiguously. `blc-decode` therefore
parses exactly one term from the front of the string and ignores any
trailing bits.

Verified reference encodings (reproduced exactly by this module):

| Term                                    | Bits                        |
| --------------------------------------- | --------------------------- |
| `I = λx.x` = `(lam (var 1))`            | `0010`                      |
| `K = λx.λy.x` = `(lam (lam (var 2)))`   | `0000110`                   |
| pairing `λλλ.((1 3) 2)`                 | `0000000101101110110`       |
| `S = λx.λy.λz.((x z)(y z))`             | `00000001011110100111010`   |

## API

| Procedure                     | Description                                                        |
| ----------------------------- | ----------------------------------------------------------------- |
| `(blc-encode term)`           | Encode a term to a bit string of `#\0`/`#\1` characters.          |
| `(blc-decode bits)`           | Decode one term; trailing bits ignored. Errors on truncated/invalid input. |
| `(blc-eval term)`             | Reduce to beta normal form via **normal-order** reduction.        |
| `(blc-apply program arg)`     | `(blc-eval (blc-app program arg))`.                               |
| `(blc-shift d cutoff term)`   | De Bruijn shift (raise free indices by `d`).                      |
| `(blc-subst term j s)`        | De Bruijn substitution of `s` for `(var j)`.                     |
| `(blc-step term)`             | One leftmost-outermost step; returns `(reduced? . term')`.        |
| `(blc->debruijn-string term)` | Pretty-printer, e.g. `"λλλ.132"`.                                 |

Predefined terms: `blc-I`, `blc-K`, `blc-S`, the Church booleans
`blc-true` / `blc-false`, and the divergent `blc-omega`. Their known bit
strings are `blc-I-bits`, `blc-K-bits`, `blc-S-bits`. The reduction bound
is `blc-max-steps`.

### Normal-order reduction

`blc-eval` uses **normal-order** (leftmost-outermost) reduction, which is
required for BLC faithfulness: it reaches a normal form whenever one
exists, even when a subterm that would diverge is ultimately discarded.
Eager/applicative order is *wrong* for BLC — it can loop on terms that
have a perfectly good normal form. Divergent terms are capped at
`blc-max-steps` reductions, after which `blc-eval` signals
`"no normal form within step bound"` rather than hanging.

The reducer implements De Bruijn `shift`/`subst` directly; a single beta
contraction of `(app (lam B) N)` is
`shift(-1, 1, subst(B, 1, shift(1, 1, N)))`.

## Examples

Exact encoding and round-trip:

```scheme
(require core.blc)

(blc-encode blc-I)              ; => "0010"
(blc-encode blc-K)              ; => "0000110"
(blc-decode "0010")            ; => (lam (var 1))   [= blc-I]
(equal? blc-K (blc-decode (blc-encode blc-K)))   ; => #t
```

Evaluation — combinators reduce as expected:

```scheme
;; K I S = I
(blc-eval (blc-app (blc-app blc-K blc-I) blc-S))   ; => (lam (var 1))

;; S K K I = I
(blc-eval (blc-app (blc-app (blc-app blc-S blc-K) blc-K) blc-I))
                                                   ; => (lam (var 1))
```

Normal-order in action — a divergent argument is discarded, not evaluated,
so evaluation terminates (applicative order would loop forever):

```scheme
;; K I Omega = I   (Omega = (λx.x x)(λx.x x) never gets reduced)
(blc-eval (blc-app (blc-app blc-K blc-I) blc-omega))   ; => (lam (var 1))
```

## Reference

John Tromp, *Binary Lambda Calculus*,
<https://tromp.github.io/cl/Binary_lambda_calculus.html>.

The self-interpreter / universal machine `U` (232 bits) and BLC8 byte-list
I/O are natural next steps built on this module; they are not included here.
