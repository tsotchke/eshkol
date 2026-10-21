---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/certified_native_solve.esk
  - tests/stdlib/certified_native_solve_test.esk
  - tests/stdlib/certified_native_solve_vm_test.esk
---

# Certified solutions from the native solver

`(require core.certified_native_solve)` connects the existing native
`linear-solve` and `tensor-inverse` candidates to `core.certified_solve`.

```scheme
(define A (vector (vector 2/3 1/7) (vector -1/5 3/2)))
(define b (vector 641/441 -323/210))
(define result (solve-certified-native A b 1/1000000000000))
(verify-certified-native-solve result A b 1/1000000000000)
```

The model is an exact rational matrix and RHS, dimension 1..64. Tolerance is
an exact nonnegative absolute component error bound. Optional arguments are
correction cap (default 3, 0..8), then the checker's positive weights,
nonnegative matrix radii and RHS radii. The verifier takes only those checker
premises after tolerance, without the correction cap.

Inputs/premises are validated and copied before native work. The original
coefficients stay exact; conversion to native f64 is candidate generation.
Finite native solution/inverse entries propose rationals by at most 32
continued-fraction steps and denominator at most 4096. This reconstruction
need not recover the intended number: the existing exact inverse-defect
checker alone admits a unique solution enclosure for the original problem.
No measurement uncertainty is inferred from conversion or reconstruction.

If the admitted error exceeds tolerance, bounded exact defect corrections
`x <- x + R*(b-A*x)` reuse the reconstructed preconditioner. Every iteration
revalidates the original problem, including uncertainty. This is separate
from the existing native solver's own platform-dependent refinement policy.

Result is `#(certified-native-solve-v1 proved|unknown candidate boxes certificate
receipt)`. PROVED requires contraction and every certified component radius
at most tolerance. UNKNOWN has no public boxes; a last linear certificate
may still contain a valid but wider enclosure. Candidate generation/arithmetic
failure returns UNKNOWN without candidate/certificate. The receipt reports
reason, completed exact corrections and two native candidate calls for a
completed candidate path; unavailable candidates report zero completed work.
It is not a complete count of work spent before an exception.

`verify-certified-native-solve` verifies only PROVED results: it recomputes
the underlying certificate against the caller's original inputs, checks the
returned boxes and tolerance, and trusts no native residual or status flag.
It does not attest the descriptive cost receipt or rerun native generation.
Mutable receipt vectors are not sealed proofs. Arithmetic loss refuses
certification. Exact verification/reconstruction/correction costs are additional;
Native numerical qualification covers JIT and AOT. The current VM route is unsupported for numerical candidate generation and has a refusal test; when both primitives become available, that route runs an exact identity-solve assertion. Current VM numerical certification remains unsupported. There is no unconditional speedup, sparse solve or GPU qualification claim.
