# Compiler differentiation graph

The native frontend supports an explicit numeric interpretation of a pure scalar
arithmetic computation:

```scheme
(require stdlib)
(differentiate 'numeric
  (lambda (x y) (+ (* x x) (* 3 x y)))
  (vector 2.0 4.0)) ; gradient: #(16.0 6.0)
```

The point is evaluated once. The result is a gradient vector in lambda parameter
order, produced by the existing `gradient` engine. This form does not introduce a
second automatic differentiation engine. Native JIT and AOT use the same lowering;
the separate bytecode frontend explicitly refuses this syntax.

## Admission and representation

The computation must be an inline lambda with 1–64 distinct, fixed scalar
parameters and one expression. Its expression may contain finite numeric
literals, parameter references, `+`, `-`, `*`, `/`, and ordinary `let`/`let*`.
Exact integer, bignum and rational literals retain their spelling/value; floating
literals retain a round-trip representation. Runtime point slots must be finite
real numbers. Numeric differentiation retains the existing engine's numerical
and exactness behavior; this form does not certify floating error.

Parallel `let` initializers read the outer environment. `let*` initializers read
preceding bindings. Variadic arithmetic is normalized in source order, including
unary negation/reciprocal and the zero-operand identities of `+` and `*`.
Identical ordered expressions share graph nodes; operands are not reassociated.
Every admitted binding is validated, including unused initializers.

Free captures must become explicit lambda inputs. Effects, opaque calls, recursion,
control flow, typed lambdas/bindings and nested callable forms are refused at
compile time. Extraction follows macro expansion and binding resolution, and
arithmetic operators and generated helper references must resolve to their
primitive bindings. Shadowing a required helper is a diagnostic, even if that
helper is needed only by the generated wrapper.

The compiler-owned `DifferentiationGraph` has topologically ordered input,
constant and binary arithmetic nodes, ordered edges, owned literal storage and
source identities/locations. It has no LLVM or runtime-value dependency. The
frontend caps the graph at 1,024 nodes, traversal at 8,192 expressions and syntax
depth at 128. These are admission limits, not silent truncation.

The graph's rule table records both numeric and finite arithmetic interpretations.
`IncrementalizePass` consumes this same admitted graph for persistent execution.

## Finite incremental interpretation

```scheme
(define state
  (differentiate 'incremental (lambda (x y) (+ (* x y) (* x x)))
    (vector 2 3)))
(state 'output #f) ; 10
(define step (state 'step (vector 1 2)))
(define next (vector-ref step 0))
(next 'output #f) ; 24
(vector-ref step 1) ; 14, the finite output change
(state 'output #f) ; 10, the original state remains readable
```

The result is an opaque two-argument state closure. Its actions are:

| Action | Payload | Result |
|---|---|---|
| `'output` | ignored | Current scalar output |
| `'step` | Dense delta vector in input order | `#(successor output-change work)` |
| `'work` | ignored | Fresh copy of the last work receipt |
| `'info` | ignored | Fresh `#(nodes operand-edges inputs trie-height)` |

The initializer evaluates the point once and stores its own baseline. A step
adds each delta slot to the corresponding retained input, processes affected
arithmetic nodes in topological order, and returns an immutable successor.
Neither the point/delta vectors nor observation vectors alias private state.
Histories may be forked and used independently. Invalid width, nonnumeric values,
division by zero, nonfinite results or a nonfinite output change raise without
modifying the input state. Ordinary integers, bignums, rationals and finite
floating values are admitted; complex values, library-defined carriers and
active AD/dual/Taylor values are refused. This pass does not persist AD tapes.

Each arithmetic node becomes a native primitive closure with compiled parent
indices. The initializer is a compiled primal let spine. The runtime does not
interpret source expressions or graph opcodes. Updated values use a persistent
binary trie over the immutable baseline, and the work queue is another trie
whose minimum index is always the next topologically ready node. Consumer lists
deduplicate repeated edges; queued nodes execute once after their parents.
There is no per-tick graph-wide array copy or graph-order scan.

Change detection uses `eqv?`, retaining exactness changes and floating signed
zero. An unchanged intermediate stops propagation. Touched arithmetic computes
its fresh value from updated cached parents, which retains finite product cross
terms and avoids rebuilding a floating snapshot as `old + output-change`.
For exact arithmetic the output changes telescope exactly. For floating
arithmetic the returned snapshot is authoritative: rounded output changes need
not telescope bit-for-bit, and no certified error bound is claimed.

The six work fields are `#(kernels changed-values trie-visits copied-trie-slots
enqueue-attempts input-slots)`. Trie visits include value reads, path writes and
queue minimum traversal; copied slots count newly allocated two-slot internal
trie nodes for values and queue. Each tick scans the bounded input width, not
the graph width. Kernel/edge work is proportional to touched nodes/consumer
lists, and trie operations add a logarithmic graph-size factor. These counters
exclude scalar/rational arithmetic cost, constant-size receipt/closure/tick
allocation, context initialization, allocator behavior and global RSS. The
initializer receipt counts initial kernels/values/inputs and zero trie work.
Scalar output observation takes one trie lookup.

The context, retained values and closures use existing native region escape
promotion. No new ownership ABI, native index, checkpoint format or background
scheduler is introduced. Graph storage is bounded by admission; retained history
consumes additional path storage until its owning region is released.

## Existing symbolic syntax

`(differentiate expression variable)` remains an alias of `diff` and returns the
existing symbolic derivative representation. Quoted symbolic expressions retain
their existing catchable unsupported-input error. The new mode is selected only
by the explicit quoted `numeric`/`incremental` first operand.

This is a scalar frontend slice of [ADR 0009](../../design/adr/0009-incremental-dataflow-dbsp.md).
General array/tensor lowering, arbitrary closure extraction, staged dense-kernel
ABIs and a bytecode graph frontend remain separate work.

## Verification

### Eshkol-written planning and shared state runtime

Native tools use an ahead-of-time compiled Eshkol pass to turn the validated
scalar graph into its consumer relation, ordered scheduling metadata and trie
shape. The pass is [`core.compiler.incremental_plan`](../../../lib/core/compiler/incremental_plan.esk),
available with `(require core.compiler.incremental_plan)`. Its version-1 input
is `(vector 1 input-count output-index nodes)`, where each node is
`(vector kind input-index operands)` and kind is 0 (input), 1 (constant), or 2
(binary primitive). Its result is
`(vector 1 capacity height operand-edges kernel-count consumers)`.
Consumer vectors are increasing and duplicate-free; repeated operands still
count separately in the operand-edge total. The native adapter independently
checks the complete returned relation and metadata against the original graph.

This is a structural compiler pass. Literal values, primitive binding admission,
source identities and arithmetic lowering remain with the existing frontend.
The pass neither evaluates the program nor changes arithmetic order. Its
temporary values live in a private compiler arena; only owned C++ plan data
survives the call, and the caller's memory context is restored after success or
refusal.

CMake builds `eshkol-bootstrap` using the C++ seed planner, compiles the Eshkol
planner and shared state runtime, and packages those objects before linking the
final tools. The packager derives a pointer-based C++ calling thunk from the
compiled pass's actual LLVM signature, checks its tagged layout, and gives each
library initializer a distinct name. Final native tools explicitly install the
Eshkol pass at startup. Once installed, a failed pass is an error; it does not
retry through the seed planner. `ESHKOL_SELF_HOSTED_INCREMENTAL=OFF` explicitly
selects the seed planner in final tools for reference builds. The standalone
compiler archive retains the seed; this milestone concerns the native tools,
not a self-hosted embedding SDK or a compiler that compiles its entire source.
The bootstrap currently requires a 64-bit native host build.

The persistent state helpers are ordinary Eshkol in
[`incremental_runtime.esk`](../../../lib/core/compiler/incremental_runtime.esk),
compiled once into the generated-program runtime archive. Each graph emits only
its kernels, initial values, consumer metadata and private runtime calls. A tick
executes the same persistent trie/ordered queue algorithm; planning is not
repeated per tick. Private node identities select the calls, and ambient user
macros and same-named source bindings cannot replace them.

The bootstrap gate compares the seed and Eshkol planners on the same graph
corpus, recompiles both Eshkol modules with the seed and final compilers, and
requires byte-identical packaged objects. It also executes mixed numeric and
incremental programs with deliberately colliding source names and macros.
This is a reproducible bootstrap of the selected pass, not a Futamura projection.

Build `eshkol-run`, `stdlib`, `differentiation_graph_test`, `incrementalize_test`,
`seed_incremental_plan_test`, `selfhost_incremental_test` and
`eshkol-vm-standalone-test`, then run:

```sh
ctest --test-dir build --output-on-failure \
  -R '^(differentiation_graph_test|incrementalize_test|seed_incremental_plan_test|selfhost_incremental_test|selfhost_incremental_bootstrap|incremental_plan_(jit|aot|vm)|compiler_differentiation_(jit|aot|diagnostics)|compiler_incremental_(jit|aot|locality))$'
```

The standalone graph test checks topology, sharing, lexical scope, literal
ownership, source coordinates and refusal budgets without LLVM. The native test
compares independent analytic partials and the existing gradient engine, and
checks point evaluation, generated-name isolation and legacy symbolic behavior.
Diagnostics tests require compile failure with no executable for inadmissible
programs and explicitly check bytecode refusal.

The incremental native suite compares every rational prefix with an independent
batch formula, checks forks/ownership, failure atomicity, floating cancellation,
signed zero/exactness changes, macros and region escape. The locality witness
grows cold branches to 36/132/516 graph nodes and checks every one of 64 ticks in
both native modes: exactly two kernels, three changed values, two enqueue
attempts, two inspected inputs, and explicit logarithmic trie-work/copy bounds.
