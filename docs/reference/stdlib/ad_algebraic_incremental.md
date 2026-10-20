---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ad/algebraic_incremental.esk
  - tests/stdlib/algebraic_incremental_test.esk
  - bench/algebraic_incremental.esk
  - examples/algebraic_incremental.esk
---

# Live incremental exact polynomial derivatives

`core.ad.algebraic_incremental` maintains signed polynomial row reductions and
their exact parameter gradient. `(make-algebraic-incremental-plan owner
row-width parameter-width expression)` admits widths 0–8 and a prefix IR of
exact constants, `(row i)`, `(param i)`, `+`, unary/binary `-`, `*` and `(pow e n)`.
Total degree is at most eight, expanded terms 128, syntax nodes 256 and depth
32. There is no host `eval`, division, transcendental operation or arbitrary
loss callback. One spare Taylor level is required for the live gradient.

Compilation combines equal monomials and separates row powers from parameter
powers. Each distinct row monomial has one maintained exact moment. Existing
exact Taylor carriers evaluate the parameter gradient; enclosing derivatives
remain intact. The squared affine-loss witness has six row moments regardless
of relation size.

`(algebraic-incremental-new plan parameters [initial-zset])` creates a persistent
snapshot. `(algebraic-incremental-step snapshot row-delta parameter-delta)`
accepts a DBSP Z-set of row vectors and an additive parameter vector, returning
`(vector successor output-delta receipt)`. Both edits take effect in one tick;
finite cross terms are retained. Signed exact integer weights, including
negative multiplicities, have the existing Z-set group semantics. This is not
a positive-table constraint checker or an exactly-once changefeed adapter.

`algebraic-incremental-output` returns owned `(vector value gradient-vector)`
data. `algebraic-incremental-info` returns `#(row-width parameter-width terms
moments)`. The receipt is `#(polynomial-moments-v1 consolidated-rows
row-monomial-evaluations terms parameter-evaluations row-replays)`. Parameter-only
ticks visit no rows. Failed ticks publish no successor; old snapshots and forks
retain independent coefficients. There is no stored relation or row-tape replay.

State size depends on the finite plan, while row updates cost changed rows times
retained monomials. Parameter edits reevaluate the plan and its gradients.
Runtime allocation and arena retention still need measured limits; fixed logical
state does not imply flat process RSS. Scalar demotion refuses through the
certified field owner.

Run CTest with `-R 'algebraic_incremental|algebraic_frontiers'`.
The suite compares 48 simultaneous row/parameter prefixes with an independent
closed-form full scan and checks reported deltas, historical/forked snapshots,
cross terms, signed cancellation, ownership, nested gradients and refusals.
`bench/algebraic_incremental.esk` compares four parameter-only ticks with full
closed-form scans, excluding initialization/oracle from the inner timing and
counting actual row visits. Set `ESHKOL_POLYNOMIAL_MODE=incremental` or `full`
and `ESHKOL_POLYNOMIAL_ROWS=16`, `128` or `512`.
