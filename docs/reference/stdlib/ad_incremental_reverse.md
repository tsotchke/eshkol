# `core.ad.incremental_reverse`

`(require core.ad.incremental_reverse)` maintains exact scalar-DAG values and a
seeded reverse pullback across sparse input and cotangent-seed edits.

```scheme
(define state (make-incremental-pullback field '(x y)
  '((* square x x) (* product square y)) '(product)
  (vector 2 3) (vector 1)))
(define step (incremental-pullback-step state '((0 1)) '((0 1))))
(incremental-pullback-output (vector-ref step 0)) ; outputs and new cotangents
```

The owner is the existing exact field or nested Taylor/hyperdual domain. Inputs
and outputs number 1–64; at most 1024 declarations form one fixed topological
DAG. Declarations are `(constant name coefficient)`, `(+|-|* name left right)`
and `(inverse name input)`. Names are distinct symbols; references must point to
preceding inputs/nodes. Outputs may select multiple nodes, including shared or
repeated targets. There is no host evaluation, effectful callback or changing
control-flow trace.

Every local forward evaluation and partial uses the existing
`algebraic-value-and-pullback` recorder and its exact operation builders. The
module does not introduce another differentiation engine. It retains owned
primal values, local partial vectors, node cotangents and per-edge contributions.

Changes are lists of `(zero-based-index additive-delta)`, with at most 4096 entries
per input/seed batch. Repeated indices consolidate exactly. A tick refreshes dirty
primal operations in forward order, then changes backward edge contributions in
reverse order. An unchanged primal can still have changed partials. Updating
primal data and seeds together retains every finite cross term. Inverse operations
require a unit in the declared owner; failure publishes no successor.

Step returns `#(successor output-delta input-cotangent-delta work)`.
`incremental-pullback-output` returns owned `#(outputs cotangents)`;
`incremental-pullback-info` returns `#(input-count node-count output-count)`.
Snapshots and forks remain valid. The owner itself retains the existing immutable
owner contract; public coefficient observations are copied.

Work is `#(refreshed-forward-operations refreshed-reverse-operations changed-edges
copied-vector-slots scanned-nodes)`. Each refreshed forward operation builds a
small local tape. State vectors are shallow-copied and both graph orders are
scanned, so total bookkeeping remains linear in graph size even when refreshed
arithmetic is local. Dense output observation also costs its output width.
There is no flat-RSS, sparse-total-time, arbitrary-function or wall-clock speedup
claim. Exact coefficients can grow; arithmetic demotion refuses through the
existing owner checks.

Tests compare every simultaneous input/seed prefix with a fresh existing reverse
sweep and independent analytic gradients, integrate both reported deltas, retain
history/forks, check nested derivatives and inverse refusal, and distinguish
constant refreshed arithmetic from growing container/scan work.
