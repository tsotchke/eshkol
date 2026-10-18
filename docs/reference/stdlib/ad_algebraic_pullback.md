# Exact equation-defined solver pullbacks

`(require core.ad.algebraic_pullback)` differentiates the equations defining
linear solutions and supplied polynomial branches. It reuses exact coefficient
owners, the existing LU representation, formal polynomial evaluation and the
custom-op reverse recorder.

`(algebraic-solve-with-pullback owner matrix rhs)` returns
`#(solution pullback factor-summary)`. Given a solution cotangent `g`,
`(pullback g)` solves `A^T lambda=g` and returns `(A-bar b-bar)` where
`A-bar=-lambda*x^T` and `b-bar=lambda`. The same retained LU serves every seed;
no inverse matrix or elimination tape is formed. Full primal and adjoint
residuals are checked. Public values and each returned cotangent own storage.

`algebraic-factor-solve-transpose` extends the existing factor API. It solves
`U^T z=b`, `L^T y=z`, and `P x=y`, then applies the same transposed coefficient
recurrences through nested Taylor/hyperdual owners. One bottom-field LU is
retained. Matrix/RHS shape, exactness and singular-primal guards remain active.

`(algebraic-system-with-pullback owner equations roots)` admits the existing
bounded polynomial representation and a supplied full-domain zero-residual
root. It factors the formal Jacobian once. A seed solves `J^T lambda=g` and
returns coefficient cotangents in the equation vector's term-list order:
`coefficient-bar_(i,t)=-lambda_i*monomial_t(roots)`. It rejects singular roots
and invalid enclosing derivatives even if the bottom scalar root is correct.
It does not discover a scalar root or choose a branch.

`algebraic-system-linearization` shares the existing polynomial admission and
formal evaluator, returning `(residual Jacobian coefficient-monomials)`.

`algebraic-reverse-solve` accepts a matrix and RHS of exact reverse nodes;
`algebraic-reverse-system` accepts polynomial terms with coefficient nodes plus
supplied full-domain roots. Both return a vector of solution nodes. Their
`record-multi-op!` recorder entry gathers every output cotangent before one
backward callback, so a vector solution pays one adjoint solve per sweep.
Shared coefficient nodes accumulate normally. Root values in nested carriers
must already describe the moving branch; `algebraic-ad-root` and
`algebraic-ad-system-root` provide the existing checked lifting paths.

The acceptance suite uses nonsymmetric row-permuted matrices, analytic
reciprocal-square gradients and HVPs, coupled polynomial coefficient adjoints,
owned observations, grouped callback counts and nested full-domain residuals.
No finite differences, compiler transform or performance speedup is claimed.
