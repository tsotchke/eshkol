---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/polynomial_exact.esk
  - tests/stdlib/polynomial_exact_test.esk
---

# `core.polynomial_exact` — checked rational polynomials

Coefficients are ordinary generic vectors, low degree first. `(exact-poly x)`
constructs an owned normalized vector from exact rational scalar/list/vector
data; zero is `(vector 0)`. Floating inputs and demoted arithmetic refuse.

`exact-poly-zero?`, `exact-poly-sub`, `exact-poly-scale`, `exact-poly-mul`,
`exact-poly-divmod`, `exact-poly-inverse-mod` and `exact-poly-eval` provide
checked arithmetic. Division returns `(quotient . remainder)`; inversion
requires a nonconstant modulus and a unit. Inputs remain unchanged.

`(exact-poly-real-root-count p lower upper)` counts distinct roots using Sturm
variations, with ordered rational nonroot endpoints. `(exact-poly-isolate p
[split-budget])` returns sorted open rational interval pairs, one per distinct
real root. Default split budget is 256; exhaustion raises without publishing a
partial list. Zero polynomials refuse, repeated roots count once and midpoint
roots are avoided. These routines share the arithmetic used by cyclotomic and
real algebraic fields. Run CTest with `-R polynomial_exact`.
