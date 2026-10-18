---
kind: reference
status: current
owner-area: ad
sources:
  - lib/core/ad/algebraic.esk
  - tests/stdlib/algebraic_ad_test.esk
  - tests/stdlib/algebraic_ad_nested_test.esk
  - examples/algebraic_ad.esk
  - examples/nested_algebraic_fields.esk
---
# `core.ad.algebraic` — exact field-valued finite AD

This optional module interprets finite differentiation carriers over one
explicitly selected [`core.algebraic_field`](algebraic_field.md). It uses the
existing exact field operations, with no floating approximation or finite
differences. It does not change the compiler's numeric `derivative` or tape APIs.

## Carriers and ownership

`(make-algebraic-taylor coefficient-domain order)` constructs a perturbation context for
`Q(alpha)[t]/(t^(order+1))`. Order is an exact integer from 0 through 16; order
one is a dual number. Coefficient `k` stores the normalized Taylor coefficient,
so `(algebraic-ad-derivative value k)` returns `k!` times that coefficient.
Order zero is evaluation, with no retained perturbation.

`(make-algebraic-hyperdual coefficient-domain)` constructs the four-coefficient carrier
`(1, e1, e2, e1*e2)`, with `e1^2=e2^2=0`. Its last coefficient is the mixed
derivative for unit first seeds and zero mixed input seed. General seeds retain
the additional first-derivative contribution from a nonzero mixed input seed.

`(algebraic-ad-value context coefficients)` accepts a vector or proper list of
exactly the carrier's dimension. Each coefficient is an exact rational or an
element of that context's field, or a value in its declared enclosing AD domain. `(algebraic-ad-constant context coefficient)`
sets only the primal. `(algebraic-ad-seed context point)` supplies a unit first
Taylor seed, or two unit first hyperdual seeds and zero mixed seed.

`(algebraic-ad-coefficients value)` returns a fresh vector with recursively copied
coefficients, including every enclosing AD layer. Constructors copy both the outer coefficient container and nested
field coefficient storage; arithmetic produces fresh results. Contexts, field
descriptors and AD values themselves are immutable by contract. Operations
require the same context object, not merely equal layout or field polynomials.
Independent perturbations and untransported foreign fields refuse.

## Arithmetic and implicit roots

`algebraic-ad-add`, `algebraic-ad-sub` and `algebraic-ad-mul` take two AD values.
Use `algebraic-ad-constant` for scalar coefficients. `algebraic-ad-equal?`
compares every retained coefficient exactly. `algebraic-ad-inverse` requires a
unit primal in its coefficient domain, evaluates the finite nilpotent geometric series and checks the
whole inverse residual. A nonunit primal refuses, including a nonzero nilpotent in an enclosing carrier.

`(algebraic-ad-root context coefficients root)` lifts a supplied simple root of
an ascending polynomial of degree 1 through 32. Polynomial coefficients may
be context-owned AD values or coefficients in the declared domain.
The supplied root must already be a coefficient in that domain. The lift checks
the primal equation and a unit derivative with respect to the root, derives
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

## Explicit nested scopes

A coefficient domain is either a selected exact field or an enclosing Taylor or
hyperdual context. At most three perturbation levels are admitted. The same
finite arithmetic kernels recurse through declared enclosing domains, including
inverse residuals, root lifting and factorial scaling; there is no second
differentiator. `(algebraic-ad-context value)` exposes the shared, immutable by
contract owner so a caller can construct the next level explicitly.

`(call-with-algebraic-derivative domain function point order)` creates a fresh
Taylor context and calls `function` with `(seed lift)`. `seed` represents the
evaluation point with unit perturbation; `lift` embeds captured enclosing values
as constants in this fresh scope. Extraction returns a value in the coefficient
domain, retaining outer derivatives. Evaluation points may themselves move in
the enclosing domain. Order zero preserves those outer coefficients. Constant
results still undergo domain validation, even when their positive derivative is
zero. Peers and siblings refuse; only the declared enclosing chain can enter.

```scheme
(define mixed
  (call-with-algebraic-derivative field
    (lambda (a lift-a)
      (call-with-algebraic-derivative (algebraic-ad-context a)
        (lambda (b lift-b)
          (algebraic-ad-inverse (algebraic-ad-add (lift-b a) b)))
        3 2))
    2 2))
(algebraic-rational mixed) ; D_a^2 D_b^2 1/(a+b) = 24/3125
```

For an implicit root, the primal equation must hold in the **entire enclosing
coefficient domain**, not merely at its bottom scalar value. The derivative with
respect to the root must be a unit there, even if the whole supplied polynomial
residual already vanishes. This rejects a nonzero nilpotent derivative such as
the coefficient of `epsilon*y`; nonzero alone is insufficient for implicit
inversion. Newton lifting still verifies the entire retained polynomial residual.

The [combined field example](../../../examples/nested_algebraic_fields.esk)
checks independent mixed reciprocal and moving cubic-root formulas over the
certified degree-six field `Q(sqrt(2),cuberoot(2))`.

## Limits and verification

Exactness inherits the existing field admission, scalar arithmetic and budget
limits. Arithmetic demotion raises rather than returning inexact coefficients.
There is no arbitrary-function transformation, implicit nesting, reverse tape,
transcendental lift, implicit field coercion or native tensor-cell integration.
Storage grows with the product of retained carrier dimensions and field degree.
These are small reference kernels; high-order cost is not a performance claim.

The focused suite uses independent closed-form derivatives, rational inverse
recurrences and exact block linear solves, plus full defining residuals on both
real cubic/quartic branches, storage ownership and refusal controls. Run:

```sh
ctest --test-dir build --output-on-failure -R 'algebraic_ad|algebraic_field|algebraic_compositum|nested_algebraic_fields|polynomial_exact|cyclotomic'
```
