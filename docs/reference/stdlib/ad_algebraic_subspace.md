# `core.ad.algebraic_subspace`

`(require core.ad.algebraic_subspace)` certifies and differentiates a caller-selected
real symmetric invariant subspace, even when eigenvalues within it repeat/cross.

```scheme
(define A (vector (vector 1 0 0) (vector 0 1 0) (vector 0 0 4)))
(define P (vector (vector 1 0 0) (vector 0 1 0) (vector 0 0 0)))
(define space (make-algebraic-subspace field A P))
(algebraic-subspace-jvp space direction)
(algebraic-subspace-vjp space projector-cotangent)
```

The owner is the existing exact algebraic field or nested coefficient domain.
Matrices have size 2–4. Construction copies A and P and checks full-domain
symmetry, P^2=P, AP=PA, and integer trace/rank strictly between zero and dimension.
It does not discover a projector or sort eigenvalue branches.

Let Q=I-P. The flattened operator
`L(X)=[A,[P,X]]+PXP+QXQ` acts as identity on within-subspace blocks and as Sylvester
maps on the cross blocks. The existing exact LU must admit L as a unit over the
primal owner and verify its full factorization residual. This certifies separation
between selected and complementary spectra. Internal repetitions are allowed;
a spectral collision across the selection boundary refuses. This is an exact
separation admission, not a numerical lower bound on a gap size.

For symmetric H=dA, the JVP solves `L(dP)=PHQ+QHP` with the retained factor.
It rechecks `P*dP+dP*P=dP` and `[A,dP]=[P,H]` exactly. The VJP symmetrizes its seed,
uses the same factor's transposed solve and projects onto cross blocks. Returned
cotangents pair with symmetric matrix perturbations; tied off-diagonal parameters
combine their two entries. This is real/formal transpose calculus, not a complex
Hermitian API.

`algebraic-subspace-projector` returns an owned matrix;
`algebraic-subspace-info` returns `#(dimension rank flattened-size owner-depth)`.
`verify-algebraic-subspace-jvp owner A P H dP` re-admits the caller's original
problem, including separation, and checks the defining linearized identities.
Observations and source matrices own coefficient storage; the declared field/AD
owner remains immutable under the existing contract.

The solve is dense in dimension-squared (at most 16), with exact coefficient
costs. There is no sparse eigensolver, automatic clustered decomposition, global
continuation certificate or permission to cross an external collision. Tests
use independent infinitesimal rotations, rational basis changes, internal
crossings, complement identities, maximum size, irrational separation, nested
Taylor derivatives, JVP/VJP pairing, ownership and explicit refusal controls.
