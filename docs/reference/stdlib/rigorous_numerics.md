# Rigorous numerics — interval jets, compact-set bounds, torus fields, element tensors

**Sources**: [`lib/core/ad/interval_jet.esk`](../../../lib/core/ad/interval_jet.esk), [`lib/core/ad/compact_bounds.esk`](../../../lib/core/ad/compact_bounds.esk), [`lib/core/torus.esk`](../../../lib/core/torus.esk), [`lib/core/ad/interval_tensor.esk`](../../../lib/core/ad/interval_tensor.esk)
**Require**: none — all four modules are in the standard library closure (native and bytecode VM). `(require core.torus)` etc. also works.
**Design record**: [ADR-0050](../../design/adr/0050-rigorous-numerics.md)

Every result below is an **enclosure**: an interval `(lo . hi)` proven to
contain the true real value, built on the directed-rounding arithmetic of
[certified-enclosures.md](certified-enclosures.md) (the default of
`core.ad.interval` since v1.5.0). Exact inputs give exact results wherever
the mathematics is rational.

## 1. One function, every carrier — `core.ad.interval_jet`

Write a function once with the generic operators and evaluate it on any
carrier:

| operator | meaning |
|---|---|
| `rn+ rn- rn* rn/ rn-neg` | arithmetic |
| `rn-sqr`, `(rn-expt x n)` | square (tight), exact nonnegative integer power |
| `rn-exp rn-log rn-sqrt rn-sin rn-cos rn-atan` | elementary functions |
| `rn->interval` | the enclosure carried by any value |

| argument | result |
|---|---|
| number (exact or binary64) | the native operator — so `derivative`, `gradient`, `derivative-n`, `taylor` differentiate the function, exactly at exact points |
| interval `(lo . hi)` | directed-rounding enclosure (`ia+` …) |
| interval Taylor jet | coefficient enclosures `Cₙ ⊇ f⁽ⁿ⁾(X)/n!` |
| interval gradient | value and partial-derivative enclosures over a box |
| rigorous Taylor model | `tm+`, `tm*`, `tm-exp` … |

A binary64 literal denotes the binary64 number it rounds to: write `1/10`,
not `0.1`, for one tenth.

```scheme
(define (f x) (rn+ (rn-sqr x) (rn-sin x)))
(derivative f 1.0)                          ; native AD: 2 + cos 1
(interval-eval f (cons 0 1/2))              ; range enclosure over [0, 1/2]
(interval-derivative-n f (cons 0 1/2) 3)    ; encloses f'''(x) = -cos x on [0, 1/2]
(interval-taylor-coeffs f (cons 1/3 1/3) 4) ; C0..C4 at the point 1/3
(interval-gradient (lambda (v) (rn* (vector-ref v 0) (rn-exp (vector-ref v 1))))
                   (vector (cons 1 2) (cons 0 1)))
;; => (value-enclosure . #(d/dx-enclosure d/dy-enclosure))
(taylor-model-certified f 1/2 1/4 6)        ; rigorous Taylor model, proven remainder
```

`ijet-var`, `ijet-const`, `ijet-coeff`, `ijet-coeffs`, `igrad-var`,
`igrad-const`, `igrad-value`, `igrad-partials` expose the carriers directly.
Jets follow the Taylor-coefficient recurrences (product, quotient, exp, log,
sqrt, coupled sin/cos, atan through `1/(1+x²)`); each recurrence is an exact
identity at every point of the box, so its interval evaluation encloses the
coefficients over the box. `taylor-model-certified` returns the
`core.ad.rigorous_taylor_models` representation: exact midpoint coefficients
at the center and a remainder collecting the coefficient radii plus
`C_{k+1}(D)·[-r, r]^{k+1}` with `C_{k+1}(D)` enclosing `f⁽ᵏ⁺¹⁾/(k+1)!` over
the whole domain.

## 2. Certified sup / inf over a box — `core.ad.compact_bounds`

A box is a Scheme vector of intervals; `f` takes one vector and uses the
generic operators.

| call | result |
|---|---|
| `(box-enclose f box)` | enclosure of `f` over the box (natural extension ∩ mean-value form) |
| `(box-sup f box tol [max-boxes])` | certificate with `[L, U] ∋ sup f` |
| `(box-inf f box tol [max-boxes])` | certificate with `[L, U] ∋ inf f` |
| `(box-certify-bound f box lo hi [max-boxes])` | `'proven`, `'refuted` (with a witness point) or `'undecided` |

Certificate accessors: `compact-bound-kind`, `compact-bound-enclosure`,
`compact-bound-witness`, `compact-bound-converged?`, `compact-bound-boxes`,
`compact-bound-pruned`, `compact-bound-monotone`.

Branch and bound: the lower bound `L` of the supremum is the largest
certified lower endpoint of `f` at an evaluated midpoint; boxes whose upper
bound is below `L` are discarded; a coordinate on which the partial
derivative enclosure is `≥ 0` (or `≤ 0`) collapses to the face that holds the
supremum; otherwise the box with the largest upper bound is bisected along
the coordinate maximizing `width × |∂f|`, at the exact midpoint when its
endpoints are exact. The search ends at `U − L ≤ tol` or when `max-boxes`
(default 4000) boxes are spent; `[L, U]` encloses the supremum either way and
`compact-bound-converged?` says which.

```scheme
(box-sup (lambda (v) (let ((x (vector-ref v 0))) (rn* x (rn- 1 x))))
         (vector (cons 0 1)) 1/1000000)
;; enclosure (1/4 . 1/4), exact
```

A compactness constant can be produced rather than asserted: take the
certified lower endpoint of `box-inf` for a lower bound `c_K`, the certified
upper endpoint of `box-sup` for `B_K`, and combine them with upward-rounded
interval arithmetic (the test file builds `P_K = max{(B_K+2)/c_K, 8B_K/γ_K}`
this way, matching row 19 of the
[Navier–Stokes mechanization](../../design/NAVIER_STOKES_RESIDUAL_MECHANIZATION.md)).

## 3. Torus fields — `core.torus`

Coordinates are angles, `Tᵈ = ℝᵈ/(2πℤ)ᵈ`; the Haar measure is normalized.

**Trigonometric polynomials** (exact): `(torus-trig d '((k a b) ...))` is
`Σ a cos(k·θ) + b sin(k·θ)`; `torus-cos`, `torus-sin`, `torus-constant`,
`torus+`, `torus-`, `torus-neg`, `torus-scale`, `torus*` (product-to-sum),
`torus-mean` (Haar mean), `torus-angular-mean` (mean over one angle),
`torus-coefficient`, `torus-deriv`, `torus-directional-deriv` (`v·∇`),
`torus-directional-inverse` (the zero-mean preimage of `v·∇`; an error names
a resonant frequency `v·k = 0` or a nonzero mean), `torus-degree`,
`torus-zero?`, `torus-terms`.

```scheme
(torus-mean (torus* (torus-cos '(3)) (torus-cos '(3))))   ; => 1/2
(define F (torus+ (torus-scale (torus-cos '(1 0)) 3) (torus-sin '(0 2))))
(torus-directional-inverse F '(1 1/3))
;; => terms ((0 2) -3/2 0) ((1 0) 0 3): exact
```

**Evaluation at a phase map**: `(torus-eval F theta)` with `theta` a list of
angles; `(torus-at-phase F phi)` is `x ↦ F(phi(x))`. On numbers it uses
`+ * cos sin`, so `(derivative (torus-at-phase F phi) x)` carries the full
chain rule through the phase; on intervals it returns an enclosure.

**Supported fields**: `(torus-field d support proc)`, `proc` a procedure of a
list of angles written with the generic operators, `support` a list of
boxes, each a list of `d` closed arcs `(a . b)` in turns (exact rationals;
`b < a` wraps through 0, `[7/8, 9/8]` is the arc `(7/8 . 1/8)`).
`torus-field*` intersects supports exactly; `torus-field-zero?` certifies a
product of disjointly supported fields is zero; `torus-supports-disjoint?`,
`torus-support-intersect`, `torus-field+`, `torus-field-deriv`,
`torus-field-eval`, `torus-field-from-trig`.

`(torus-field-mean field tol [max-cells])` encloses the Haar mean: on `T¹`
each cell of width `h` contributes `f(c) + C₂(c)h²/12 + C₄(X)h⁴/80`
(fourth order), on `Tᵈ` `f(c) + Σ C₂,ᵢᵢ(X)h²/12 ± Σ|Hᵢⱼ(X)|h²/16` (second
order), with all angles enclosed through the rigorous π; the grid doubles
until the enclosure is narrower than `tol` or the next grid would exceed
`max-cells` (default 4096).

```scheme
(torus-field-mean (torus-field 1 (list (list (cons 0 1)))
                               (lambda (th) (rn-exp (rn-cos (car th)))))
                  1e-9)
;; encloses I0(1) = 1.2660658777520083..., width < 1e-9
```

## 4. Interval and Taylor-model element tensors — `core.ad.interval_tensor`

An element tensor is `(itensor-from-list shape entries)` (row-major) or
`(tensor->itensor T)`; entries are any carrier. `itensor+ itensor- itensor*`
(Hadamard), `itensor-scale`, `itensor-neg`, `itensor-exp/log/sin/cos/sqrt/
sqr/atan`, `itensor-matmul`, `itensor-transpose`, `itensor-sum`,
`itensor-map`, `itensor-map2`, `itensor-ref`, `itensor-intervals`,
`itensor-contains-tensor?`. Dense boundary: `(itensor->bounds X)` gives
outward-rounded f64 `(lower . upper)` tensors, `(bounds->itensor lo hi)`
reads them back.

```scheme
(define A (itensor-from-list '(2 2) (list 1 2 3 4)))
(define B (itensor-from-list '(2 2) (list 1/2 -1 0 1/3)))
(define (P t) (itensor-sum (itensor-matmul (itensor-sqr (itensor-scale A t)) B)))
(P 3/5)                              ; 3/5, exact
(derivative P 3/5)                   ; 2, native AD through the tensor program
(P (ijet-var (cons 3/5 3/5) 2))      ; jet: coefficient 2 is exactly 5/3
(P (cons -1/2 1))                    ; interval enclosure of the range
(tm-bound (P (tm-var 0 1/2 3)))      ; tensor-valued Taylor model, bounded
```

## 5. Tests

`tests/stdlib/rigorous_numerics_test.esk` (CTest
`rigorous_numerics_runtime_smoke` and `rigorous_numerics_aot_smoke`)
compares every facility with exact rational ground truth computed in the
test itself from series with proven tails, including randomized containment
of exp, log, sin, cos, atan, sqrt and a polynomial on exact and binary64
intervals (the required rate is 100%).
`tests/vm_parity/corpus/102_rigorous_numerics.esk` is the native/VM parity
program.
