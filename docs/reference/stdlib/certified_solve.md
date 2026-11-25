---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/certified_solve.esk
  - tests/stdlib/certified_solve_test.esk
  - examples/mathematics_certified_linear_solve.esk
---

# `core.certified_solve`

`(require core.certified_solve)` checks received approximate solves using exact
rational arithmetic. It runs no solver. Decode finite received floats explicitly
with `inexact->exact`; model uncertainty requires separate bounds.

```scheme
(certify-linear-solve A b xhat R [weights [matrix-radii [rhs-radii]]])
(verify-linear-solve-certificate certificate A b xhat R [weights [matrix-radii [rhs-radii]]])
```

Matrices are plain vectors of row vectors, dimension 1–64. Use the `vector`
procedure instead of numeric vector literals, which may become floating tensors.
Entries are exact rationals, weights positive, uncertainty radii nonnegative.
Shapes and every intermediate arithmetic result are checked. Inputs are copied.

The result is `#(linear-solve-certificate-v1 proved|unknown boxes receipt)`.
UNKNOWN proves neither singularity nor inaccurate output. PROVED boxes enclose
the unique solution, uniformly over any declared coefficient/RHS uncertainty.

For `r=b-A*xhat`, `E=I-R*A`, form
`N=abs(E)+abs(R)*matrix-radii` and
`c=abs(R*r)+abs(R)*(rhs-radii+matrix-radii*abs(xhat))`.
Positive weights w give `q=max((N*w)/w)` and `eta=max(c/w)`.
Only `q<1` admits the contraction. Component radii are
`c+(N*w)*eta/(1-q)`. No floating residual or condition estimate is trusted.

Receipt fields are `#(A b xhat R weights matrix-radii rhs-radii residual
signed-inverse-defect N c q nominal-backward-error radii)`. The nominal
componentwise backward error is
`max_i abs(r_i)/(abs(b_i)+sum_j abs(A_ij*xhat_j))`, with zero-over-zero zero.
It permits independent coefficient/RHS perturbations, without symmetry,
positivity or shared-parameter constraints. It stays nominal under uncertainty.

The verifier recomputes the complete source-bound receipt and returns true when
the supplied record matches, including for an `unknown` result. Verification
checks record integrity; callers must separately require status `proved` before
using boxes as an enclosure. A mutable vector is not a sealed proof. Tests use independent Cramer's-rule answers, weighted and
strict-boundary cases, singular zero-residual controls, uncertainty corners,
copied observations and tampering. Exact verification can be expensive; a poor
preconditioner may refuse. No speedup, automatic refinement, sparse solver or
certified AD error propagation is claimed.
