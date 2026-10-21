# Maintained named-axis tensor pullbacks

`(require core.ad.tensor_circuit_reverse)` pairs the existing maintained
forward tensor circuit with a seeded structural reverse sweep per update.

```scheme
(define state (make-tensor-circuit-pullback inputs
  '(contract (contract A w (k)) (contract A w (k)) (i)) seed))
(define step (tensor-circuit-pullback-step state named-input-deltas seed-delta))
(tensor-circuit-pullback-output (vector-ref step 0))
```

Inputs/expressions retain `core.dbsp.tensor_circuit`'s fixed-shape, named-axis,
node/rank/cell/iteration limits. Seed is a native tensor with the output's
physical shape (logical scalar uses `(1)`). Seed edits are additive tensor
deltas; `#f` means unchanged. Inputs and seed snapshots own caller tensor data.
All arithmetic is native floating point; nonfinite observed values/adjoint
arithmetic refuse publication. This is not the exact algebraic coefficient API.

The forward circuit first installs all input changes. A fresh custom-op tape
then records its current tensor nodes, using the existing `core.ad.tape`
recorder and reverse accumulation. Addition/subtraction, named transpose,
axis reduction and contraction have structural pullbacks. Contraction sums
over matching named coordinates, including shared free batch axes. Shared
operands accumulate both contributions. No dense Jacobian is materialized.
The output seed is an explicit final linear functional on the same tape.

Every tensor input has a cotangent, including disconnected inputs (zero).
Rows and their multiplicities are discrete inputs: their projection changes
can alter tensor-parameter gradients, but no row cotangent is returned.
Projections enter the reverse tape as fixed values of the current invocation.
Simultaneous data/parameter/seed changes use the new forward state and seed,
retaining all finite cross terms without a first-order approximation.

Step returns `#(successor value-delta named-cotangent-deltas work)`.
Output returns `#(owned-output-tensor ((name owned-cotangent-tensor) ...))`.
History/forks remain valid; failed updates publish no successor. Forward work
is the existing circuit receipt. Reverse work is `#(recorded-ops
materialized-primal-cells contraction-adjoint-products copied-gradient-cells)`.
Full cotangent differencing also scans the gradient cells; seed scaling and
other adjoint arithmetic are additional work. Every tick rebuilds/reverses the
full tensor graph and materializes flat primal/adjoint vectors. This provides
maintained gradient semantics, not sparse reverse update cost or flat RSS.

The forward owner's `tensor-circuit-visit` visits nodes in topological order
with copied structural lists and a scalar-only read closure, returning
`#(node-count root-id)`. It exposes no internal graph node or mutable tensor.
The API reference for that owner specifies the callback arguments. No arbitrary
AD tracing, compiler transformation, broadcasting, GPU or HVP API is added.
