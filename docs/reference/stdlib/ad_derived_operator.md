---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ad/derived_operator.esk
  - lib/core/symbolic.esk
  - tests/stdlib/derived_operator_test.esk
  - tests/stdlib/certified_nonlinear_test.esk
  - examples/mathematics_derived_operator.esk
---

# `core.ad.derived_operator`

`(require core.ad.derived_operator)` defines a scalar polynomial once and uses
`core.symbolic` to derive numeric and temporal rules.

```scheme
(define op (make-derived-operator '(x y) '(+ (* x x) (* x y))))
(derived-operator-value op (vector 2 3))              ; 10
(derived-operator-jvp op (vector 2 3) (vector 1 -2))   ; 3
(derived-operator-vjp op (vector 2 3) 1)              ; (vector 7 2)
(derived-operator-delta op (vector 2 3) (vector 1 -2)) ; 2
(derived-operator-info op)
```

The definition admits distinct named variables, exact rational constants,
binary `+ - *`, unary minus, division by a literal nonzero rational, and
`pow`/`expt` with a literal nonnegative integer exponent. Arbitrary calls,
effects, transcendentals and variable denominators refuse. Limits are 1–8
variables, degree eight, 256 syntax nodes, depth 32, and conservative 4096-term
work checked before expansion, including shifted inputs. Cancellation can make
that conservative bound reject an otherwise small result.

The canonical forward polynomial and `poly-deriv` supply forward/JVP/VJP.
Temporal change is the simplified exact polynomial `F(x+delta)-F(x)` and retains
all mixed/higher-order delta terms. No hand-derived VJP or finite-difference
fallback is installed. This is a restricted library protocol, not a compiler-wide
rule-registry refactor, tensor lowering or arbitrary-callback differentiation.

The operator closure owns compiled data. Caller syntax/names and returned
cotangents/metadata cannot mutate later evaluations. Inputs have fixed width.
Generic arithmetic validates values while preserving native AD carriers; forward
evaluation composes with compiler differentiation. Exact rational evaluations
check exact result retention. Floating evaluations retain ordinary rounding;
finite-update identities are exact-domain claims. Complex VJP means formal
polynomial transpose, not conjugate adjoint.

Info is `#(arity degree forward-term-work shifted-term-work)`. Tests check
independent analytic derivatives, JVP/VJP pairing, exact full-update parity over
a rational corpus, cubic change terms, compiler AD composition and refusals.
## Exact interval enclosures

`(derived-operator-enclosure op boxes)` returns `#(value-interval partial-interval-vector)`.
Boxes are a vector of closed exact rational `(lo . hi)` pairs in input order.
The existing canonical polynomial and symbolic partials are evaluated with the
shared rigorous interval arithmetic. Every endpoint must retain exactness.
Bounds enclose the entire box; dependency can make them conservative. Returned
intervals own their pairs. This is the shared enclosure seam used by
[certified polynomial roots](certified_nonlinear.md).
