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

`tensor-circuit-visit state visitor` is the read-only graph consumer seam.
The callback receives `(id op children parameters axes shape input-names ref)`
in topological order; structural lists are fresh and `ref` reads one logical
coordinate as a scalar. Row nodes have no tensor reader. The return value is
`#(node-count root-id)`. It does not expose internal graph/snapshot storage.
The [maintained pullback](ad_tensor_circuit_reverse.md) uses this seam.

## Explicit execution strategies

`(tensor-circuit-step snapshot changes [strategy])` accepts `incremental`
(the compatible default), `recompute`, or `auto`. Strategies apply to affected
contractions; they do not rescan the whole graph. Forced incremental uses the
existing finite-delta kernels. Recompute uses the existing native
`tensor-matmul` for ordinary matrices and direct named-axis contraction for
shared batch axes/general ranks. A recomputed node installs its fresh snapshot
itself. Downstream operations rebuild from fresh child values, avoiding
`old+(fresh-old)` cancellation when a delta cannot represent the fresh value.
Input integration and row projections retain their existing delta semantics.
Snapshots and the read-only visitor remain compatible with maintained reverse
consumers. Reverse sweeps do not automatically become sparse.

`tensor-circuit-work` and the step's three-field receipt retain their layout.
Dense products there are logical contraction product counts, not measured BLAS
instructions. New `tensor-circuit-plan-work` returns an owned
`#(incremental-contractions recomputed-contractions sparse-estimate
recompute-estimate materialized-cells)` for the most recent tick. Initial states
return zeros. Estimates are sums over executed contractions only. Materialized
cells include dense operands for native matrix multiplication and two output
passes (construction plus snapshot copy); rebuilt linear descendants add their
operand passes and three output passes (zero tensor, baseline snapshot and
applied snapshot). Sparse snapshot block-copy cells remain in the legacy
receipt, independently of these dense materializations.

For each contraction, the sparse operation proxy is
`|dA|*freeB + |dB|*freeA + |dA|*|dB|`. The last term charges every sparse
cross-pair examination, including unmatched pairs. The recompute proxy counts
all union-axis products plus dense materializations. Auto recomputes when its
proxy is no greater than the sparse proxy, or when a freshly recomputed
upstream operand requires propagation from its authoritative value. These dimensionless estimates omit
queue/list handling, delta consolidation, allocation, cache effects and BLAS
throughput. They are deliberately inspectable, untuned estimates, not a
per-machine runtime optimizer or a guarantee of the fastest strategy.

Native f64 deltas cannot always integrate bit-for-bit after cancellation. The
fresh snapshot is authoritative on a recomputed path; applications requiring
its full value should observe that snapshot. Incremental paths retain ordinary
floating drift. Tests compare all three modes with independent native products
under an explicit absolute tolerance, verify both choices, general batch axes,
old snapshots, strict failure and a large cancellation through reduction.
The existing forward and reverse regression suites remain required.

`tests/benchmarks/tensor_circuit_planner_bench.esk` is an informational CPU
witness with warm kernels, 8x8 matrices, four densities and eight ticks per
sample. It reports elapsed nanoseconds, strategy and estimates. Run it through
a configured native compiler in JIT or AOT mode with the source-tree library
path and one BLAS thread. The observed crossover depends on host and batch;
automatic mode can miss a profitable intermediate-density recomputation.
Timings are never an admission gate, and no GPU/performance-portability claim
is made.
