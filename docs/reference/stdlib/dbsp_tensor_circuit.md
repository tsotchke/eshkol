---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/dbsp/tensor_circuit.esk
  - tests/stdlib/tensor_circuit_test.esk
  - examples/mathematics_dbsp_tensor_circuit.esk
---

# Maintained named-axis tensor expressions

`(require core.dbsp.tensor_circuit)` compiles a checked, fixed-shape expression
into a standing computation. It uses the sparse changes and shared-block
snapshots from `core.dbsp.tensor` and `core.dbsp.tensor_snapshot`.

`(make-tensor-circuit inputs expression)` accepts bindings `(name axes tensor)`
and `(name rows Z-set)`. Tensor axes are distinct symbols; row values are lists.
Expressions are Scheme data, with named inputs and these forms:

- `(contract a b (axes ...))`: sum the named shared axes; other shared axes
  remain batch axes. An empty contraction list gives elementwise/outer products.
- `(+ a b)` and `(- a b)`: identical named axes and shape are required.
- `(transpose a (axes ...))`: an explicit permutation of every axis.
- `(sum a (axes ...))`: reduce the named axes.
- `(project rows (axes ...) (shape ...) (coordinate-columns ...) value-column)`:
  signed weighted scatter from stable zero-based row coordinates.

For example, `(contract (project trades (i k) (2 2) (0 1) 2) weights (k))`
maintains a table-to-matrix contraction. No host evaluation or effects occur.
Structurally equal expression terms share one node.

`(tensor-circuit-step snapshot changes)` takes `(name delta)` bindings and
returns `#(next-snapshot output-delta receipt)`. Repeated input names consolidate
before execution; all inputs are installed before consumers run in topological
order. Failed attempts publish no successor. Consumers with a zero net change
do not schedule their descendants. Historical versions and forks remain valid.

`tensor-circuit-ref` reads one logical output coordinate. Full
`tensor-circuit-output` materializes an owned native tensor. Logical scalars use
coordinate `()` and materialize as a one-cell tensor. `tensor-circuit-work`
returns an owned `#(executed-consumers products copied-cells)` receipt.

Ordinary matrix products call the existing `tensor-matmul-delta` kernel through
read views. General contractions enumerate only unfixed coordinates in the two
linear terms and match sparse shared coordinates in the cross term. Every
finite product retains `dA B + A dB + dA dB`. Construction uses a direct dense
contraction, rather than a dense sparse-pair cross product. Row inputs are not
retained after their projections have been initialized.

Limits: 64 inputs, 4096 nodes, expression nesting 64, rank 8, positive fixed
dimensions, 1,048,576 cells per tensor, and 16,777,216 initial contraction
iterations. Native cells are floating point; rounding/drift follow the existing
tensor bridge. Dense changes and fan-out can be expensive, sparse cross matching
can be quadratic in delta size, and dirty-node queue insertion is ordered-list
work. There is no whole-update allocation/RSS, GPU, arbitrary AD, automatic
compiler transformation, broadcasting or full Rayfall-language claim.

Run CTest with `-R 'tensor_circuit|tensor_snapshot|dbsp_tensor'`. The acceptance
suite checks every streamed prefix against independent dense products,
simultaneous updates, shared operands, batch axes, reductions, retractions,
ownership, failed ticks and fixed sparse work at growing payload sizes.

## Structural visitor

`tensor-circuit-visit` calls a callback for each node in topological order with
its ID, operation, fresh child/parameter/axis/shape lists, associated input
names, and a read-only scalar reader. The reader refuses on row nodes, so row
relations remain discrete and inaccessible. It returns `#(node-count root-id)`
and does not expose internal graph or snapshot storage.

## Explicit execution strategies

`(tensor-circuit-step snapshot changes [strategy])` accepts `incremental`
(the compatible default), `recompute`, or `auto`. Strategies apply to affected
contractions and do not rescan the graph. Forced incremental uses the existing
finite-delta kernels. Recompute uses native `tensor-matmul` for ordinary matrix
products and direct named-axis contraction for shared batch axes and general
ranks. A recomputed node installs its fresh snapshot directly. Downstream nodes
rebuild from fresh child values, avoiding `old+(fresh-old)` cancellation when a
delta cannot represent the fresh value. Snapshots and reverse consumers remain
compatible.

`tensor-circuit-work` and the step's three-field receipt retain their layout.
`tensor-circuit-plan-work` returns an owned
`#(incremental-contractions recomputed-contractions sparse-estimate
recompute-estimate materialized-cells)` for the latest tick; initial states return
zeros. Estimates sum executed contractions only. Materialized cells count dense
operands for native matrix multiplication and output construction/snapshot; rebuilt
linear descendants count operand passes and zero/baseline/applied snapshots.
Sparse snapshot block-copy cells remain in the legacy receipt.

For each contraction, the sparse proxy is
`|dA|*freeB + |dB|*freeA + |dA|*|dB|`. The last term charges every sparse
cross-pair examination, including unmatched pairs. The recompute proxy counts all
union-axis products plus dense materialization. Auto recomputes when its proxy is
no greater than the sparse proxy, or when a recomputed upstream operand requires
propagation from its authoritative value. These dimensionless estimates omit
queue/list handling, delta consolidation, allocation, cache effects and BLAS
throughput. They are inspectable work estimates, not per-machine timing claims or
guarantees of the fastest strategy.

Native floating-point deltas need not integrate bit-for-bit after cancellation.
Recomputed snapshots are authoritative, while incremental paths retain ordinary
floating drift. Tests compare all strategies with independent native products at
every prefix, cover both planner choices and shared batch axes, and test old
snapshots and large cancellation through reductions and another contraction. The
informational CPU benchmark is not an admission gate; crossovers depend on host and
batch size.
