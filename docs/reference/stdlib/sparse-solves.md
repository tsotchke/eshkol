# Add CSR Krylov solves with exact implicit sensitivity enclosures

CG and full-orthogonalization symmetric minimum residual over existing CSR, plus explicit implicit JVP/VJP and exact diagonal-dominance enclosures of original-model solution and cotangent errors.

```scheme
(require core.ml.sparse_solve)
(define a (sparse-from-triplets 2 2 '((0 0 4) (0 1 -1) (1 0 -1) (1 1 4))))
(define b (vector 1 2))
(define r (sparse-cg a b 1e-10 64))
(define c (sparse-enclose a b (vector-ref r 3)))
(verify-sparse-enclosure c a b)
```

`sparse-cg` / `sparse-minres` return
`#(sparse-solve-v1 method status x actual-residual-2-norm iterations tolerance)`.
Convergence uses the recomputed original residual, relative to `max(1,||b||2)`.
CSR columns must be sorted and unique; existing constructors sum duplicates.
Neither method materializes the full matrix.

`sparse-enclose` snaps a received candidate to the explicit 2^-20 grid and uses
exact arithmetic on the original A,b. With
`eta = min_i(|A_ii| - sum_{j!=i}|A_ij|) > 0`, the inverse infinity norm is at
most `1/eta`. The certificate `#(sparse-enclosure-v1 center eta radius)` states
`||x*-center||inf <= ||b-A*center||inf/eta`. Every rational arithmetic result is
checked for exactness. Snapping is a proposal, never an assumed approximation.

`sparse-solve-jvp method A b dvalues db tolerance limit` returns
`#(sparse-jvp-v1 status primal-raw tangent-raw primal-cert tangent-cert center radius)`.
It solves `A dx = db - dA x`; the final radius also propagates uncertainty in
the primal center by `||dA||inf * primal-radius / eta`.

`sparse-solve-vjp method A b seed tolerance limit` returns
`#(sparse-vjp-v1 status primal-raw adjoint-raw primal-cert adjoint-cert bars)`.
`bars` is `#(value-centers value-radii rhs-centers rhs-radius)`. For each stored
slot `(i,j)`, the center is `-lambda_i*x_j` and its radius is
`|lambda_i| rx + |x_j| rlambda + rx*rlambda`. Symmetric A makes the adjoint solve
use A itself. These are ambient matrix-slot cotangents; parameters tied across
two symmetric slots must gather both slot contributions by the chain rule.

Both sensitivity APIs require successful numerical solves and exact enclosures.
They return `unknown` without centers/radii if either admission fails. Numerical
tolerance controls iterative residuals; certificate radii are reported separately.

## Scope

CG requires SPD; a negative-curvature guard is not a complete SPD test. MINRES uses a full orthogonal basis with O(n*k+k^2) storage. Exact certificates require strict row diagonal dominance and exact rational model inputs; unadmitted arithmetic, nonconvergence or missing bounds produce unknown with no gradient artifact. Explicit JVP/VJP APIs are supplied; no new automatic tape primitive or performance speedup is claimed.

## Independent replay

Run `python3 repro/sparse-solves/oracle.py repro/sparse-solves/results.json` to replay saved solution and sensitivity enclosures without a native solver. The independent checker uses Python Fractions; small cases also compare to exact dense Gaussian elimination.

Selected MIT contribution; see `repro/sparse-solves/LICENSE`.
