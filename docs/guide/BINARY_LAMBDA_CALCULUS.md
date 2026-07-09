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

### Bit streams, the universal machine, BLC8, and diagrams

| Procedure / value                | Description                                                                 |
| -------------------------------- | --------------------------------------------------------------------------- |
| `blc-pair`                       | Pairing combinator `λλλ.((1 3) 2)`, encoding `0000000101101110110`.          |
| `blc-nil`                        | List terminator for finite streams (`= blc-false`).                         |
| `(blc-cons h t)`                 | Cons value `⟨h,t⟩ = λf. f h t` (h, t closed).                               |
| `(blc-list-of-bits bits tail)`   | Bit string `bits` → Scott list of booleans, terminated by `tail`.           |
| `(blc-encode-input prog input)`  | `prog` bits then `input` bits as one `blc-nil`-terminated list (U's input).  |
| `blc-U-bits`                     | Tromp's 232-bit universal machine as a bit string.                          |
| `(blc-U)`                        | The decoded universal-machine term.                                        |
| `(blc-byte->term b)`             | Integer byte (0–255) → big-endian delimited list of 8 bits.                 |
| `(blc-bytes->term bytes)`        | List of integer bytes → BLC8 term.                                         |
| `(blc-term->byte t)` / `(blc-term->bytes t)` | Inverse of the two above.                                       |
| `(blc-string->term s)` / `(blc-term->string t)` | Scheme string ⇄ BLC8 term (one byte per char).             |
| `(blc-diagram term)`             | Render a term as a Tromp-style ASCII lambda diagram.                        |

**Bit-stream convention** (Tromp): a bit `0` is `True = λx.λy.x`, a bit `1` is
`False = λx.λy.y`, and a stream is a Scott list built with `blc-pair`
(`⟨h,t⟩ = λf. f h t`), terminated by `blc-nil` when finite.

### The universal machine U

`U` is Tromp's 232-bit (29-byte) self-interpreter. Applied to a Scott list
that begins with the self-delimiting BLC encoding of a closed program `M`
followed by input bits, `U` reduces to `M` applied to the remaining list:

```scheme
;; Identity program: U consumes encode(I) off the front, then applies I to
;; the rest of the stream (here a marker K), yielding K unchanged.
(blc-eval (blc-app (blc-U) (blc-list-of-bits (blc-encode blc-I) blc-K)))
                                                   ; => blc-K   (37 steps)

;; Constant-output program (lambda i. True) ignores its input:
(blc-eval (blc-app (blc-U)
                   (blc-encode-input (blc-encode (blc-lam blc-true)) "")))
                                                   ; => blc-true (89 steps)

;; Identity on nonempty input "10" returns the input list:
(blc-eval (blc-app (blc-U) (blc-encode-input (blc-encode blc-I) "10")))
                                    ; => (blc-list-of-bits "10" blc-nil)
```

The exact 232 bits (`blc-U-bits`) are:

```
0101000110100000000101011000000000011110000101111110011110000101110011
1100000011110000101101101110011111000011111000010111101001110100101100
1110000110110000101111100001111100001110011011110111110011110111011000
0110010001101000011010
```

These were cross-checked by parsing Tromp's De Bruijn term for `U` and
re-encoding it with `blc-encode`: the result reproduces the bit string
exactly. Round-tripping `(blc-encode (blc-U))` also returns `blc-U-bits`.

**Step-cap note.** All demonstrations above reach normal form in well under
100 reduction steps, far below `blc-max-steps`. `U` is a genuine general
interpreter, so running it on programs that themselves loop, or on large
inputs, can blow up term size and exceed the step cap — in which case
`blc-eval` signals `"no normal form within step bound"` rather than hanging,
exactly as for any divergent term.

### BLC8 byte I/O

BLC8 operates on byte streams: a byte is a delimited list of its 8 bits in
**big-endian** order (most significant bit first), and a byte string is a
list of such byte-lists.

```scheme
(blc-term->bytes  (blc-string->term "Hi"))   ; => (72 105)
(blc-term->string (blc-string->term "Hi"))   ; => "Hi"
(blc-term->byte   (blc-byte->term 72))        ; => 72
```

### Lambda diagrams

`(blc-diagram term)` renders a term as a Tromp lambda diagram
(<https://tromp.github.io/cl/diagrams.html>): abstractions are horizontal
bars, a variable is a vertical line rising to the bar of its binding lambda,
and an application is a horizontal link joining the leftmost variables of its
two subterms (variables are spaced 4 columns apart, so a term with `V`
variable occurrences is `4V-1` columns wide).

```
(blc-diagram blc-I)      (blc-diagram blc-K)      (blc-diagram blc-S)

---                      ---                      ---------------
 |                       -|-                      -|-------------
                          |                       -|-------|-----
                                                   |   |   |   |
                                                   |----   |----
                                                   |--------
```

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

John Tromp, *Binary Lambda Calculus and Combinatory Logic*,
<https://tromp.github.io/cl/Binary_lambda_calculus.html>, and the lambda
diagrams page <https://tromp.github.io/cl/diagrams.html>.

The 232-bit universal machine `U`, the BLC8 byte-list I/O convention, and the
lambda-diagram algorithm are all taken from those sources. The exact `U` bit
string used here was cross-checked by re-encoding Tromp's De Bruijn term for
`U` with this module's own `blc-encode`.
