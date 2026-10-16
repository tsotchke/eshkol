# Live exact ridge regression

`(require core.ad.algebraic_ridge)` maintains the unique minimizer of
`sum weight*(x^T beta-y)^2 + lambda*beta^T beta` over an explicit exact field
or nested AD coefficient owner. Rows have feature coordinates followed by the
target. Dimension is bounded to 1–7, matching the eight-coordinate row boundary.

`(algebraic-ridge-new owner dimension lambda [initial-zset])` compiles one
existing polynomial moment plan. It retains the upper-triangular Gram moments,
feature/target moments, target square and signed row count: four moments for
one feature, seven for two, and 37 for seven. It retains no relation or history.
All changed rows cross the existing validated/normalized signed DBSP boundary.

`(algebraic-ridge-step snapshot row-delta lambda-delta)` applies both changes
atomically and returns `(next-snapshot coefficient-delta receipt)`. Receipt is
`(ridge-moments-v1 changed-row-count row-monomial-evaluations dimension 1)`.
A parameter-only tick visits zero rows and row monomials. The bounded normal
matrix is refactored once per successful tick; no low-rank update speedup is
claimed. Failed ticks preserve the prior snapshot.

Regularization must have a nonnegative selected-real primal. The actual normal
matrix must have a positive-definite primal, even with signed/retracted rows.
`algebraic-factor-positive-definite?` checks full-domain symmetry and uses the
already-verified LU's identity permutation and strictly positive bottom-field
Schur pivots (Sylvester admission). It adds no second factorization. Exact sign
refinement can refuse on budget exhaustion. Singular/indefinite fits refuse;
there is no unstable or partial fit disguised as a minimizer.

`(algebraic-ridge-output snapshot)` returns owned
`(coefficients d-coefficients/d-lambda minimized-objective)`. The sensitivity is
`-A^-1 beta`; the optimum objective is `sum weight*y^2-b^T beta`, justified by
the checked normal residual. Nested carriers retain enclosing derivatives.
`algebraic-ridge-info` returns `(dimension moment-count lambda bottom-LU-count)`.

`(algebraic-ridge-pullback snapshot seed)` reuses the retained solver pullback
and returns `(normal-matrix-cotangent RHS-cotangent lambda-cotangent)`. The last
entry is the diagonal trace of the matrix cotangent. The matrix cotangent is
unconstrained; callers composing symmetric Gram entries must combine both
symmetric positions. No per-row gradient scan is performed.

`algebraic-incremental-moment-ref` is the shared owned observation of an existing
compiled moment, addressed by its row-exponent vector. Uncompiled powers refuse.

The acceptance suite compares every streamed prefix to independent two-by-two
closed-form rational fits, including simultaneous row replacements and lambda
changes, objective/sensitivity parity, exact coefficient deltas, irrational
rows, nested second derivatives, retained snapshots, owned observations and
positive-definiteness refusals. No native staged optimizer or general timing
claim is implied.
