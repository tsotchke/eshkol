# Certified native linear-solve pullbacks

`(require core.certified_solve_pullback)` exports two operations:

```scheme
(solve-certified-native-pullback A b seed tolerance
  [primal-weights [adjoint-weights [matrix-radii [rhs-radii [seed-radii]]]]])
(verify-certified-native-pullback result A b seed tolerance
  [primal-weights [adjoint-weights [matrix-radii [rhs-radii [seed-radii]]]]])
```

For `x=A^-1 b` and a scalar objective with solution cotangent `seed`, the
adjoint satisfies `A^T lambda=seed`. The cotangents are
`bbar=lambda` and `Abar=-lambda*x^T`. This module obtains both candidates
through `solve-certified-native`, including its bounded rational reconstruction
and up to three exact defect corrections. Each solve must independently prove
its uniform enclosure and meet the supplied componentwise solve tolerance.
There is no new differentiation or linear-solver implementation.

All original coefficients, seeds, weights and uncertainty radii are exact
rationals in plain vectors; matrices are vectors of row vectors. Dimension is
1..64. Weights are strictly positive, radii nonnegative. Omitted weights are
ones; omitted radii are zero. The adjoint uses transposed matrix radii and its
own norm weights: primal contraction alone does not authorize an adjoint.
Seed radii describe a family of linear functionals, rather than uncertainty in
the primal RHS. Invalid seeds are checked before native work even when the
primal cannot certify.

The result is `#(certified-native-pullback-v1 status primal adjoint Abar bbar)`.
`status` is `proved` only when both native solve certificates admit. `Abar` is
a matrix of closed rational interval pairs and `bbar` a vector of pairs. Exact
endpoint products enclose every `-lambda_i*x_j`; they remain conservative when
both solves depend on the same uncertain matrix. Correlation is not assumed
absent, but is not exploited for sharper bounds. The tolerance bounds each
solve's error, not the width of the final product intervals. With unknown
status, derivative fields are `#f`; the adjoint may also be `#f` if the primal
failed. Malformed original data raises rather than proving anything.

The verifier recomputes both existing native-solve admissions against the
caller's original matrix, RHS, seed, tolerance and all premises, then recomputes
the product intervals. A status flag, floating residual or stored derivative
box is never sufficient. Returned certificates are independently owned mutable
data, so edits are detectable by reverification. Native candidate cost is
recorded by each nested solve receipt; a fully executed pair makes four native
candidate calls, not one factorization with a reused adjoint. No factorization
reuse, GPU qualification, dense Jacobian, transcendental rule or higher-order
pullback is claimed. Matrix cotangents use independent entries; a symmetric
parameterization combines the corresponding off-diagonal entries.

Tests use independent 2x2 formulas, correlated uncertain families, signed scalar
products, independently weighted transpose admission, unknown primal/adjoint
controls, omitted/changed premises and forged gradient boxes. Both JIT and AOT
registrations require a strict `RESULT: PASS` verdict.
