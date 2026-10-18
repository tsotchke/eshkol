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
copied-vector-slots processed-work-nodes trie-visits unique-enqueues)`.
The first three fields retain their meanings. Field 3 now counts constructed or
copied state/queue trie slots, four-slot node records, and stored partial/edge
vectors. Field 4 counts work-set removals, including input leaves in the reverse
set. Fields 5 and 6 expose trie traversal and distinct scheduling work. This is
an explicit receipt format extension from five to seven fields; callers which
assumed the previous length must adapt.

Construction still evaluates/scans the full graph and builds an immutable
baseline. Successors share it and path-copy private 11-level binary tries for
node overrides and work sets. The fixed height covers the admitted 1088 nodes.
Forward work removes the smallest scheduled index; reverse work removes the
largest, so all changed parents/children are ready before an operation runs.
No tick copies or scans an N-slot state or dirty vector. Only touched nodes,
consumer edges, bounded input/seed batches and public output widths are visited.
An unchanged value still schedules its refreshed local partial for reverse work.
There is no unbounded overlay-chain lookup or periodic hidden compaction.

The counters exclude coefficient arithmetic/size, short-lived local tape
allocations, change consolidation lists, consumer-list iteration and returned
observation/delta vectors. They are neither total allocations nor wall-clock
predictions. Each refreshed forward operation builds the existing small local
tape. Dense observation costs its declared output/input width. Growing fan-out
and exact coefficient sizes can still dominate; there is no flat-RSS or general
speedup claim. Arithmetic demotion/nonunit refusal uses the existing owner.

Tests compare every simultaneous input/seed prefix with a fresh existing reverse
sweep and independent analytic gradients, integrate both deltas, retain
history/forks, check nested derivatives and inverse refusal, and compare one hot
branch surrounded by 1 versus 1000 cold nodes. Cold graph growth adds no copied
state slots or processed work nodes; trie lookup counts remain bounded by their
fixed height. A 64-tick sparse witness checks values, cotangents and bounded
bookkeeping at every prefix, including shared/repeated parents and targets.
