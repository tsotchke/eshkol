# Adaptive certified coupled continuation

`(require core.ad.certified_continuation)` tracks a selected nonsingular real
root of an existing compiled polynomial system with exactly one scalar parameter.

```scheme
(define equations (make-certified-polynomial-system '(x y) '(t)
  '((- (* x x) (* (+ 1 t) (+ 1 t))) (- y (* x x)))))
(define start (make-certified-continuation equations 0 (vector 1 1)
  (vector 1/32 1/16) (vector (vector 1/2 0) (vector 1 1))))
(define result (certified-continuation-advance start 1/2 1/2 128))
```

Construction validates the caller's initial center, positive radii and inverse
through `certify-polynomial-root`. The existing total-variable/degree budgets
remain. Times, maximum step and all proof arithmetic are exact rationals.

Advance revalidates its starting root certificate and takes state, target, positive maximum step and attempt budget 0..256.
The old implicit sensitivity predicts a midpoint center; four bounded Newton
proposal corrections round to a 1/4096 rational grid. These are proposals,
not trusted roots. The driver chooses a tube from initial radii and predicted
movement, derives residual and Jacobian intervals from the original polynomial
source over the entire tube times the entire proposed time interval, and uses
`certify-linear-solve` for the uniform inverse-defect/forcing bound. Strict
contraction and self-inclusion must hold, and the old root enclosure must be
inside the tube. Polynomial smoothness and uniform invertibility connect the
selected root continuously throughout that interval.

An independently revalidated endpoint proposal must have its certified root
box inside the admitted tube before publication. Failed attempts halve the
step; accepted attempts reset the requested maximum step. Forward and reverse
traversal use the same proof. Arithmetic or proposal failures spend an attempt
and cannot establish a root-absence or collision claim.

Result is `#(certified-continuation-result-v1 complete|partial successor
accepted-step-receipts attempts)`. PARTIAL preserves a certified prefix and
makes no claim that the requested endpoint is reachable. Enclosure returns
owned `#(time boxes)`. The source-bound verifier replays the entire bounded
advance from the supplied start/target/policy and compares the result; it is
not a portable sealed proof. Retained states and forks remain valid under the
existing immutable-source contract.

This adds the compiled system's `family-box` action, evaluating both unknown
and parameter interval vectors through the same derived-operator enclosure
owner. Point/fixed-parameter certificate behavior is preserved. No singular
branch traversal, complex paths, global root census or arbitrary-function
tracker is claimed. Conservative interval dependency, fixed tube policy,
exact arithmetic growth and proposal resolution can stop valid paths.
