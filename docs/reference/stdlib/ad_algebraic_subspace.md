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
The direct constructor does not discover a projector or sort eigenvalue branches.

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

## Projectors from checked polynomial splits

`(make-algebraic-subspace-from-factors owner A selected complement)` constructs
and admits the projector instead of requiring an eigenbasis. Both supplied
factors are nonconstant monic exact rational coefficient vectors/lists, low
degree first. Their product must equal the original matrix characteristic
polynomial in the full declared owner. Matrix size remains 2..4.

The shared `exact-poly-inverse-mod` computes `u*g = 1 mod f`; matrix Horner
evaluation gives `P=u(A)g(A)`. This is one on selected-factor roots and zero
on complementary-factor roots. Noncoprime factors refuse. The existing
subspace constructor rechecks symmetry, idempotence, invariance and separation,
and the selected rank must equal `degree(f)`. Repeated roots within a factor
are allowed. The resulting object uses the existing JVP/VJP without another
spectral derivative path. Inputs/observations retain the existing ownership.

Factor selection is explicit. This does not factor arbitrary polynomials,
sort individual eigenvalues, or accept arbitrary algebraic factor coefficients.
A rational irreducible factor selects its conjugate eigenvalues together;
algebraic-entry matrices are admitted when their full characteristic polynomial
has the supplied rational split. Existing nested owners are also checked in
full, rather than only at their primal coefficients. Exact coefficient growth
and the small dense Sylvester solve retain their existing costs/refusals.
