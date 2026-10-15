---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/certified_nonlinear.esk
  - lib/core/ad/derived_operator.esk
  - tests/stdlib/certified_nonlinear_test.esk
---

# `core.certified_nonlinear`

`(require core.certified_nonlinear)` validates a supplied approximate root of a
small coupled polynomial system and encloses its implicit parameter sensitivities.

```scheme
(define system (make-certified-polynomial-system '(x y) '(p)
  '((- (* x x) p) (- y x))))
(define certificate (certify-polynomial-root system (vector 3/2 3/2)
  (vector 1/4 1/4) (vector 2) (vector (vector 1/3 0) (vector 1/3 1))))
(verify-polynomial-root-certificate certificate system (vector 3/2 3/2)
  (vector 1/4 1/4) (vector 2) (vector (vector 1/3 0) (vector 1/3 1)))
```

The system has one equation per unknown and at most eight total unknown/parameter
variables, with at least one unknown. Expressions use the existing bounded
`core.ad.derived_operator` scalar-polynomial grammar. Source compilation owns the
expressions. Centers, strictly positive trial radii, fixed parameter values and
square preconditioner R are exact rational data. Numeric vector literals may
become floating tensors; use the `vector` procedure for exact containers.

The shared `derived-operator-enclosure` extension evaluates canonical polynomial
terms and their existing symbolic derivatives over exact rational boxes using
`core.ad.rigorous_interval`. It generates whole-box Jacobian bounds from the same
admitted source; a sampled/point Jacobian is never substituted for that bound.

For the trial box B, residual F(c), Jacobian anchor J and radii dJ, the existing
linear certificate supplies `N=abs(I-RJ)+abs(R)*dJ`, `y=abs(R*F(c))` and the
weighted contraction `q=max((N*r)/r)`. Strict q<1 and componentwise
`y+N*r<=r` prove that `x-RF(x)` maps B into itself and is a contraction. This
proves one unique root in B. Existing uniform linear error transfer tightens its
reported enclosure. Failed admission is UNKNOWN, not a root-absence certificate.

For each parameter, the same uniformly invertible Jacobian box and source-derived
parameter partials certify `dx/dp=-J^-1*F_p`. The received nominal derivative is
`R` times the midpoint RHS; exact linear validation encloses its error uniformly
on the trial box. No separate nonlinear or sensitivity solver is trusted.

The result is `#(polynomial-root-certificate-v1 proved|unknown root-boxes
sensitivity-box-matrix receipt)`. UNKNOWN has no root/sensitivity boxes. Receipt
fields bind the compiled system identity, copied center/radii/parameters/R,
residual, Jacobian anchor/radii, majorant, forcing, q, inclusion margins and the
nested linear certificate. The verifier recomputes the full source-bound result;
mutable receipt vectors are not sealed proofs or portable serialized owners.

This is an a-posteriori validator, not an automatic root search or a global
solution census. Interval dependency can refuse valid roots. Transcendentals,
complex roots, singular branches, uncertain parameter families and arbitrary
callback enclosure premises are outside this slice. Exact arithmetic overflow
refuses. Tests include independent irrational and rational roots/derivatives,
whole-box and strict-boundary controls, singular systems, source/receipt tampering,
input ownership and independent value/Jacobian containment probes.
