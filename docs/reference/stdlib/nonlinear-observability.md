# Certify polynomial observation fibers and all-order ambiguity symmetries

Exact polynomial ODE Lie jets, task-relative observation-fiber proofs, and invariant-family symmetry witnesses. Proof replay uses supplied multipliers and re-expansion; Groebner membership is a separate discovery hint.

```scheme
(require core.nonlinear_observability)
(define c (certify-observation-task '(x) '(0) '(x) 0 'x 1 '(1)))
(verify-observation-task c '(x) '(0) '(x) 0 'x) ; #t
```

`observation-jet` orders derivatives before sensors. `observation-fiber` returns
`#(P constraints task-difference)`, using mirror symbols prefixed with
`observation$`. A certificate is `#(observation-task-v1 power multipliers)`;
replay reconstructs the constraints from the supplied original model and checks
`sum multiplier_i * constraint_i = task-difference^power`. Powers are positive.

`observation-task-hint` returns `ideal-consequence` or `unknown`; it is not a
proof artifact. `observation-symmetry?` checks an invariant parameterized family,
an involutive parameter permutation, equivariant reduced dynamics, invariant
sensors, and a noninvariant task. These finite polynomial identities imply an
all-order observation ambiguity. No finite sample of Lie jets substitutes for
that symmetry proof.

## Scope

Polynomial ODEs over exact Q; no restricted-domain, real-radical completeness, global-existence, or automatic Nullstellensatz-search claim. A positive task-difference power proves vanishing. A failed hint or witness yields unknown. Symmetry means indistinguishability for intervals on which the trajectories exist.

## Independent replay

Run `python3 repro/nonlinear-observability/oracle.py repro/nonlinear-observability/results.json` for source replay, then `python3 tools/certcheck/eshkol_certcheck.py --strict repro/nonlinear-observability/certificates.json` for the standard polynomial-identity claims. The standard checker establishes the identities; source-to-fiber binding is checked separately by the source oracle.

Selected MIT contribution; see `repro/nonlinear-observability/LICENSE`.
