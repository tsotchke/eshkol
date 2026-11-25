---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ad/algebraic_implicit.esk
  - tests/stdlib/algebraic_implicit_test.esk
  - examples/algebraic_implicit.esk
---

# Coupled exact implicit differentiation

`core.ad.algebraic_implicit` extends supplied-branch lifting to polynomial
systems with 1–8 unknowns. An equation vector contains one proper term list per
unknown; each term is `(list coefficient exponent-vector)`. Coefficients belong
to the supplied AD context or its enclosing domain. Exponents are nonnegative
exact integers, total degree is at most 16 and each equation admits 64 terms.

`(algebraic-ad-system-root context equations roots [iteration-budget])` accepts
a supplied root vector in the **enclosing coefficient domain**. It verifies
every primal equation across that full domain, derives the Jacobian by formal
polynomial differentiation and factors the primal Jacobian once through
`core.algebraic_linalg`. A singular bottom Jacobian refuses even when the whole
initial residual is already zero. No extra AD level or user Jacobian is needed.

Fixed-Jacobian nilpotent corrections reuse the same factorization for every
coefficient and iteration. The default budget is Taylor dimension or three for
hyperduals; an explicit budget is an exact integer from 0 through 32. Return
requires every coefficient of every equation to vanish exactly. Exhaustion
raises without publishing a partial root. Moving evaluation points and supplied
branches retain enclosing derivatives, with the existing three-level AD limit.

`(algebraic-ad-system-residual context equations values)` returns owned full
residual values using the same bounded polynomial interpretation. Constructors
copy input coefficients/exponents and returned roots own their storage.

This is local exact differentiation around a supplied simple branch, not a
scalar root finder, global nonlinear solver, arbitrary callback transformer or
uniqueness proof outside the retained nilpotent neighborhood. Exactness inherits
certified field arithmetic and its refusal on scalar demotion.

Run `ctest --test-dir build --output-on-failure -R algebraic_implicit`.
Independent binomial derivatives, a nonlinear two-parameter mixed derivative,
moving outer branches, full vector residuals and singularity/budget controls
are covered by the executable suite.

`algebraic-system-linearization` exposes the existing formal polynomial
linearization as `(residual Jacobian coefficient-monomials)` over a field or
nested coefficient owner, with the same shape, degree and exactness admission.
