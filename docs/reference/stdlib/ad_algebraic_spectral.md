# Certified simple eigenvalue branches

`(require core.ad.algebraic_spectral)` composes exact matrix arithmetic,
characteristic polynomials and implicit AD. Matrices are square, size 1–8.

`(algebraic-characteristic owner matrix)` returns owned coefficients in ascending
degree order. Faddeev-LeVerrier computes `B0=I`, `ck=-trace(A*B[k-1])/k`,
`Bk=A*B[k-1]+ck*I`; the terminal matrix must satisfy its exact Cayley-Hamilton
residual. The same retained matrices evaluate `adj(tI-A)`.

`(algebraic-eigenvalue-with-pullback owner matrix eigenvalue)` requires full-domain
symmetry, an exactly satisfying selected eigenvalue and an invertible
characteristic derivative. It forms
`P=adj(lambda*I-A)/p'(lambda)` and checks `A*P=lambda*P`, `P*P=P`, and
`trace(P)=1` exactly. The result is `#(value projector pullback summary)`;
projector observations own their storage. The scalar-seeded pullback returns
`seed*P^T`, the unconstrained matrix cotangent. Symmetric parameters combine
both off-diagonal entries.

`(algebraic-eigenvalue-lift context matrix enclosing-root)` derives the moving
characteristic polynomial, reuses `algebraic-ad-root`, and rechecks the full
eigenvalue/projector certificate. It preserves enclosing Taylor or hyperdual
derivatives under the existing owner rules. Root selection is explicit: the
caller supplies an admitted field element/enclosing branch. This does not
automatically factor characteristic polynomials, discover all eigenvalues,
normalize eigenvectors, sort eigenvalue branches or handle repeated-root
projectors. Collisions, unresolved units, foreign owners and inexact data refuse.

Run `ctest --test-dir build -R algebraic_spectral --output-on-failure`.
Tests compare characteristic coefficients, irrational two-by-two projectors,
fourth-order spectral Taylor coefficients, derivatives of sensitivities and
three-dimensional diagonal branches with independent formulas. Collision and
ownership refusals are explicit; no floating eigensolver or finite differences
enter the result.
