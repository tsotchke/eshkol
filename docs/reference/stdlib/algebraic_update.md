# Guarded exact low-rank solves

`(require core.algebraic_update)` adds bounded Woodbury corrections over the
existing checked exact LU owner. Fields and nested AD contexts use the same
coefficient-domain protocol, ownership and exactness admission as
`core.algebraic_linalg`.

`(algebraic-update-factor owner matrix [positive-definite?])` creates a retained
factor. Matrices are square, size 1–32. Positive-definite admission checks
full-domain symmetry and exact selected-real primal Schur pivots.

`(algebraic-update factor left-columns right-columns [policy])` updates
`A := A + U V^T`. Each column collection is a list of owned vectors; rank is
0–8. This general interface accepts factors without a positivity declaration.
`(algebraic-update-symmetric factor columns weights [policy])` requires a
positive-definite base and applies `A := A + sum weight_i*u_i*u_i^T` atomically.
Signed weights and enclosing derivative coefficients are admitted explicitly.

Policies are `auto` (default), `correction`, and `refactor`. The correction path
retains `W=A^-1 U`, `Z=A^-T V`, and a checked factor of `S=I+V^T W`:
`x=A^-1 b-W*S^-1*V^T*A^-1 b`. Transpose solves exchange the two factors and
use `S^-T`. All returned solves verify the complete updated `A*x=b` or
`A^T*x=b` residual in the full coefficient domain.

Symmetric corrections check `1+weight*u^T*A^-1*u > 0` at the selected primal
before each rank-one correction. If an intermediate matrix is inadmissible,
`auto` factors and checks the final simultaneous matrix instead. Thus a signed
batch can succeed even when its chosen intermediate order loses positivity.
Forced correction refuses that intermediate. Indefinite/singular final matrices
refuse; no partial state escapes. Correction depth is bounded by eight; auto
refactors at the bound, while forced correction refuses budget exhaustion.

`algebraic-update-solve` and `algebraic-update-solve-transpose` reuse retained
state. `(algebraic-update-pullback factor rhs)` returns `#(solution pullback)`;
the pullback takes a vector seed and returns unconstrained matrix/RHS cotangents
`#(-lambda*x^T lambda)`, with `lambda=A^-T*seed`. Symmetric parameterizations
combine both off-diagonal entries. Observations own their storage.
`algebraic-update-info` returns
`#(route rank full-factorization-attempts small-factorization-attempts depth)`.
Counts include attempted corrections before a successful fallback.

The first contribution supplies an exact update backend suitable for maintained
normal systems. It does not change `algebraic-ridge-step`'s existing default,
provide a sparse LU, claim a wall-clock speedup or eliminate coefficient growth.
Full residual checks and bounded chains still cost work.

Run `ctest --test-dir build -R algebraic_update --output-on-failure`.
Independent rational formulas and fresh-solve comparisons cover nonsymmetric
transpose orientation, rank-two updates, signed SPD fallback, chain refactoring,
irrational columns, nested derivatives, ownership and pullbacks.
