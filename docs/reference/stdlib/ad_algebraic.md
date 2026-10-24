---
kind: reference
status: current
owner-area: ad
sources:
  - lib/core/ad/algebraic.esk
  - tests/stdlib/algebraic_ad_test.esk
  - examples/algebraic_ad.esk
---
# `core.ad.algebraic` — exact field-valued finite AD

This optional module interprets finite differentiation carriers over one
explicitly selected [`core.algebraic_field`](algebraic_field.md). It uses the
existing exact field operations, with no floating approximation or finite
differences. It does not change the compiler's numeric `derivative` or tape APIs.

## Carriers and ownership

`(make-algebraic-taylor field order)` constructs a perturbation context for
`Q(alpha)[t]/(t^(order+1))`. Order is an exact integer from 0 through 16; order
one is a dual number. Coefficient `k` stores the normalized Taylor coefficient,
so `(algebraic-ad-derivative value k)` returns `k!` times that coefficient.
Order zero is evaluation, with no retained perturbation.

`(make-algebraic-hyperdual field)` constructs the four-coefficient carrier
`(1, e1, e2, e1*e2)`, with `e1^2=e2^2=0`. Its last coefficient is the mixed
derivative for unit first seeds and zero mixed input seed. General seeds retain
the additional first-derivative contribution from a nonzero mixed input seed.

`(algebraic-ad-value context coefficients)` accepts a vector or proper list of
exactly the carrier's dimension. Each coefficient is an exact rational or an
element of that context's field. `(algebraic-ad-constant context coefficient)`
sets only the primal. `(algebraic-ad-seed context point)` supplies a unit first
Taylor seed, or two unit first hyperdual seeds and zero mixed seed.

`(algebraic-ad-coefficients value)` returns a fresh vector of copied field
elements. Constructors copy both the outer coefficient container and nested
field coefficient storage; arithmetic produces fresh results. Contexts, field
descriptors and AD values themselves are immutable by contract. Operations
require the same context object, not merely equal layout or field polynomials.
Independent perturbations and untransported foreign fields refuse.

## Arithmetic and implicit roots

`algebraic-ad-add`, `algebraic-ad-sub` and `algebraic-ad-mul` take two AD values.
Use `algebraic-ad-constant` for scalar coefficients. `algebraic-ad-equal?`
compares every retained coefficient exactly. `algebraic-ad-inverse` requires a
nonzero primal, evaluates the finite nilpotent geometric series and checks the
whole inverse residual. Zero-primal values are nonunits and refuse.

`(algebraic-ad-root context coefficients root)` lifts a supplied simple root of
an ascending polynomial of degree 1 through 32. Polynomial coefficients may
be context-owned AD values, exact rationals or elements of the selected field.
The supplied root must already be a coefficient in that field. The lift checks
the primal equation and nonzero derivative with respect to the root, derives
that derivative by Horner evaluation and runs bounded nilpotent Newton steps.
It returns only after the **entire retained polynomial residual** is exactly
zero. Wrong roots, singular roots and foreign coefficients raise. This is local
branch lifting; it neither finds scalar roots nor adjoins a new field extension.

```scheme
(require stdlib)
(require core.ad.algebraic)
(define field (make-algebraic-field '(-2 0 0 1) 7 0))
(define alpha (algebraic-generator field))
(define context (make-algebraic-taylor field 4))
(define target (algebraic-ad-seed context 2))
(define root (algebraic-ad-root context
  (list (algebraic-ad-sub (algebraic-ad-constant context 0) target) 0 0 1) alpha))
(algebraic-ad-derivative root 1) ; alpha/6
(algebraic-ad-derivative root 2) ; -alpha/18
(algebraic-ad-derivative root 4) ; -5*alpha/81
```

The [runnable example](../../../examples/algebraic_ad.esk) also checks the full
equation `root^3 = 2+t` and a zero-primal field relation with nonzero derivatives.
The latter distinction is essential: reducing coefficients modulo the field
polynomial must not reduce a perturbed variable modulo that polynomial.

## Limits and verification

Exactness inherits the existing field admission, scalar arithmetic and budget
limits. Arithmetic demotion raises rather than returning inexact coefficients.
There is no arbitrary-function transformation, automatic nesting, reverse tape,
transcendental lift, implicit field coercion or native tensor-cell integration.
These are small reference kernels; high-order cost is not a performance claim.

The focused suite uses independent closed-form derivatives, rational inverse
recurrences and exact block linear solves, plus full defining residuals on both
real cubic/quartic branches, storage ownership and refusal controls. Run:

```sh
ctest --test-dir build --output-on-failure -R 'algebraic_ad|algebraic_field|polynomial_exact|cyclotomic'
```
