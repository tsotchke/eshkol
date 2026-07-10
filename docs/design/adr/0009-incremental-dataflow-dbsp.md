# ADR 0009: Native DBSP-Style Incremental Dataflow and Unified Differentiation

- Status: Proposed
- Date: 2026-07-09
- Decision owners: Eshkol compiler/runtime maintainers; `core.memory` and Selene maintainers
- Research cluster: 9

## Context

Selene is a resident Eshkol computation: it accepts a small amount of new
information on each tick, updates long-lived knowledge-base, workspace, memory,
and session state, and continues indefinitely. Eshkol's canonical resident-loop
shape already makes this state transition explicit by threading `state` to the
next tail call (`docs/LONG_RUNNING_LOOPS.md:12-26`), and the durable memory module
identifies itself as the first persistent link in Selene's memory substrate
(`lib/core/memory_store.esk:1-16`). The current implementation solves two
necessary lifetime problems, but not the recomputation problem:

- ESH-0214b places a reclaimable arena scope around eligible loop iterations;
  the runtime pops the scope when no loop-carried value points into it and
  conservatively commits it otherwise (`lib/core/runtime_arena_core.cpp:348-432`).
  Code generation opens that scope at the loop header and closes it on every
  back edge and exit (`lib/backend/llvm_codegen.cpp:23752-23788`,
  `lib/backend/llvm_codegen.cpp:26803-26904`).
- ESH-0214c deep-promotes an escaping reachable subgraph from a dying region,
  preserving cycles and sharing with a forwarding map
  (`lib/core/runtime_regions.cpp:623-643`,
  `lib/core/runtime_regions.cpp:821-858`). Mutation paths invoke a region write
  barrier before storing into longer-lived state
  (`lib/core/runtime_regions.cpp:1064-1124`).

Those mechanisms can make a tick's transient allocation bounded, but a tick can
still scan or reconstruct all accumulated state. Memory reclamation changes the
space curve; it does not change an `O(state)` state transition into an
`O(change)` transition.

This distinction is visible in `core.memory`. Its events are immutable,
content-addressed, hash-linked values, and its log is an RGA CRDT whose merge is
commutative, associative, and idempotent (`lib/core/memory.esk:1-20`). A local
append advances a vector clock and appends one new event
(`lib/core/memory.esk:61-79`), while merge combines the RGA and vector clocks
(`lib/core/memory.esk:81-90`). However, a derived LWW value is currently obtained
by folding the complete event sequence (`lib/core/memory.esk:114-129`), and the
durable open path reconstructs state by reading every persisted line
(`lib/core/memory_store.esk:188-221`). The streaming audit was introduced
because materializing a 6.8K-link log for verification took roughly sixteen
minutes, whereas a one-pass audit took seconds
(`lib/core/memory_store.esk:389-405`). Separately, the iterative reader was
required to boot a resident from a persisted state containing roughly 46K
entries (`tests/stdlib/reader_long_list_test.esk:1-18`). These are concrete
incremental-maintenance and continuity problems, not merely database concerns.

The consciousness runtime has the same shape. Knowledge-base assertion appends
to a growable fact array (`lib/core/logic.cpp:625-647`), while each query walks
that array from the beginning (`lib/core/logic.cpp:649-690`). A workspace step
invokes every registered module (`lib/backend/logic_workspace_codegen.cpp:597-649`)
and then overwrites the winning content and increments persistent step state
(`lib/core/workspace.cpp:244-249`, `lib/core/workspace.cpp:314-338`). DBSP does
not eliminate work whose output is genuinely affected, but it gives facts,
matches, proposals, and summaries explicit indexed views so an unrelated input
change need not rebuild all of them.

Eshkol already treats differentiation as a compiler concern. Numeric derivative,
gradient, Jacobian, Hessian, and related AST operations dispatch directly into
`AutodiffCodegen` (`lib/backend/llvm_codegen.cpp:10486-10511`). Reverse-mode
nodes record an operation, primal value, accumulated gradient, and input links
(`inc/eshkol/eshkol.h:958-1013`); tensor nodes similarly record their inputs,
saved values, shape, and tape membership
(`lib/backend/autodiff_codegen.cpp:2311-2424`). Backpropagation walks the tape in
reverse topological order and applies each local chain rule
(`lib/backend/autodiff_codegen.cpp:9965-10096`). The forward/Taylor side seeds a
fresh perturbation level while preserving enclosing perturbations, which is the
mechanism behind nested levels in the AD tower
(`lib/backend/autodiff_codegen.cpp:446-529`). The staged-kernel design already
calls for refactoring this machinery into reusable, one-pass computations
rather than repeatedly reconstructing tapes
(`docs/design/AD_STAGED_KERNEL_HANDOFF.md:19-35`,
`docs/design/AD_STAGED_KERNEL_HANDOFF.md:202-244`).

DBSP supplies the corresponding calculus for computations over changing data.
For an abelian group `(A, +, 0, -)`, a stream is a function `s : N -> A`.
DBSP defines:

```text
(z^-1 s)[0] = 0                 one-tick delay
(z^-1 s)[t] = s[t-1], t > 0

D(s) = s - z^-1(s)             discrete differentiation: snapshots -> changes
I(s)[t] = sum(i=0..t, s[i])     discrete integration: changes -> snapshots

D . I = identity
I . D = identity
```

For a batch query `Q : A -> B`, lifting, written `Q^`, applies it independently
at every logical time: `(Q^(s))[t] = Q(s[t])`. Its incremental form is

```text
Q^D = D . Q^ . I
```

It maps input changes to output changes. The transformation composes:

```text
(Q1 . Q2)^D = Q1^D . Q2^D
```

This is the DBSP chain rule. Linear operators such as map, filter, projection,
and Z-set union are their own incremental forms. A join is bilinear and follows
the discrete product rule. With previous snapshots `A0`, `B0` and current
changes `dA`, `dB`:

```text
d(A join B) = (dA join B0) + (A0 join dB) + (dA join dB)
```

Feedback is legal only through `z^-1`. Giving a recursive query an inner
logical clock and differentiating its feedback circuit yields semi-naive
fixed-point evaluation; differentiating that circuit again at the outer tick
clock incrementally maintains recursive results as external inputs change.

Relations use Z-sets, `Z[A]`: finite maps from rows in `A` to exact integer
weights. Weight `+1` inserts an occurrence, weight `-1` retracts one, addition
consolidates changes, and negative weights make retraction a group operation.
This algebra is a direct fit for event-sourced state, provided CRDT delivery
deduplication and Z-set addition are not confused.

## Decision

Eshkol will adopt DBSP-style incremental computation as a native semantic and
compiler capability.

1. A pure-Eshkol module named `core.dbsp` will establish the reference
   semantics: Z-sets, streams and circuits, `z^-1`, `D`, `I`, incremental
   relational operators, aggregates, distinct, and nested fixed points. It will
   not bind to the Rust `dbsp` crate.
2. Selene's tick will be expressible as one standing DBSP circuit. Input events
   are delta batches; knowledge, workspace, session, and other derived
   collections are maintained views. A tick processes the delta and affected
   matches, rather than rescanning every persisted row.
3. The algebraic data plane of `core.memory` will become a Z-set stream while
   retaining its content hashes, causal metadata, RGA ordering, durability, and
   idempotent CRDT merge envelope. `I` is replay; `D` is the unseen delta to
   replicate or feed to a view.
4. DBSP circuit state will be OALR-native. Persistent traces and delay state
   live in circuit-owned regions; transient work lives in a per-tick nursery
   reclaimed at the tick barrier. No garbage collector will be introduced.
5. A later compiler pass will transform a pure batch query into its incremental
   circuit. The pass will share a differentiation graph, rule registry, AST
   traversal, capture/effect analysis, staging, diagnostics, and test-oracle
   infrastructure with numeric AD.
6. At v2.0, Eshkol will expose one compiler primitive, `differentiate`, with two
   explicit interpretations: numeric differentiation and incremental temporal
   differentiation. Existing numeric operators and a new `incrementalize`
   convenience form will remain as surface desugarings.
7. Feldera's batch-equivalence, checkpoint/restart, and spill-to-storage lessons
   will be adopted as concepts and reimplemented under Eshkol's semantics and
   OALR constraints. Eshkol will not adopt Feldera's SQL/Calcite front end or a
   Rust ABI dependency.

The target cost of a non-recursive tick is
`O(input delta + affected index matches + deterministic compaction)`, not a scan
of the complete logical state. Recursive ticks additionally pay for the inner
delta iterations needed to reach the new fixed point. This is an asymptotic
contract, not a claim that every update is constant-time or that persistent
state stops growing.

## Technical design

### 1. Semantic domains and invariants

`core.dbsp` will use exact Eshkol integers for weights, including bignums. A
Z-set is always consolidated at an operator boundary: equal rows have one
summed weight and zero-weight rows are absent. Algebraically negative snapshot
weights are valid; an API boundary that declares a value to be a relation or bag
may additionally require non-negative weights.

The reference external representation is an ordinary Eshkol value:

```scheme
#(dbsp-zset-v1 ((weight row) ...))
```

Entries are ordered deterministically by the canonical S-expression rendering
of `row`; equality remains `equal?`, and canonical strings are only ordering and
indexing aids. Implementations may use a hash table internally, but observable
iteration, serialization, checkpoint, and merge order is canonical. This keeps
results reproducible across native, JIT, bytecode, and WASM execution.

One circuit step is one logical transaction `t`. All input batches for `t` are
installed before any node runs, nodes run in stable topological order, every
edge is consolidated, recursive nodes run their inner clock to quiescence, and
outputs become visible only at the step barrier. Wall-clock arrival order inside
a batch is not semantic order.

The four primitive stream constructions are:

- `lift`: apply a batch function pointwise;
- `z^-1`: retain exactly the preceding logical value, with group zero at `t=0`;
- group addition/negation: combine streams pointwise;
- feedback: permitted only when every cycle crosses a `z^-1` node.

`D` and `I` are library-visible nodes even when optimization later cancels an
adjacent pair. Keeping them explicit makes the semantics, test oracle,
checkpoint state, and compiler transformation inspectable.

### 2. `core.dbsp` API

The first implementation follows the precedent of `core.blc`: an on-demand,
pure-Eshkol module with ordinary S-expression terms and constructors. The BLC
module demonstrates that such a calculus can be represented and evaluated in
Eshkol itself (`lib/core/blc.esk:1-13`, `lib/core/blc.esk:51-90`). DBSP terms,
however, use Eshkol's native named binding representation, not BLC's local
De Bruijn encoding. It will ship as `lib/core/dbsp.esk` and load with
`(require core.dbsp)`; `core.incremental` may later be a descriptive alias, not
a second implementation.

#### Z-set operations

| Procedure | Contract |
|---|---|
| `(zset-empty)` | Additive identity. |
| `(zset-singleton row weight)` | One weighted row; zero produces the identity. |
| `(zset-from-weighted entries)` | Validate exact integer weights, combine duplicates, remove zeros, and canonicalize. |
| `(zset-weight zs row)` | Return the consolidated weight, defaulting to zero. |
| `(zset-add a b)` / `(zset-negate a)` / `(zset-sub a b)` | Commutative-group operations. |
| `(zset-scale k zs)` | Multiply every weight by exact integer `k`. |
| `(zset-consolidate zs)` | Canonical normal form. |
| `(zset-empty? zs)` / `(zset-positive? zs)` / `(zset=? a b)` | Algebraic predicates. |
| `(zset-entries zs)` | Canonically ordered `(weight row)` entries. |

#### Circuit and stream operations

| Procedure | Contract |
|---|---|
| `(dbsp-circuit term env)` | Validate a circuit term and resolve its named procedures from `env`. |
| `(dbsp-input circuit name)` | Return the stream handle for a declared input. |
| `(dbsp-lift circuit name proc inputs...)` | Add pointwise batch query `proc`. |
| `(dbsp-delay circuit name input)` | Add `z^-1`, initialized to the input group's zero. `dbsp-z^-1` is an alias. |
| `(dbsp-D circuit name input)` | Add discrete differentiation. |
| `(dbsp-I circuit name input)` | Add discrete integration. |
| `(dbsp-map circuit name proc input)` | Weight-preserving row map, consolidating collisions. |
| `(dbsp-filter circuit name pred input)` | Weight-preserving selection. |
| `(dbsp-project circuit name proc input)` | Projection; semantically map plus consolidation. |
| `(dbsp-union circuit name inputs...)` | Z-set addition. |
| `(dbsp-join circuit name left right left-key right-key emit)` | Indexed equijoin with multiplied weights and the bilinear delta rule. |
| `(dbsp-aggregate circuit name input key lift group emit)` | Incremental keyed homomorphism; `group` supplies zero, add, and negate. |
| `(dbsp-distinct circuit name input)` | Maintain per-row counts and emit only zero/nonzero boundary changes. |
| `(dbsp-fixed-point circuit name inputs step)` | Nested-clock fixed point; `step` is a named-parameter Eshkol lambda. |
| `(dbsp-incrementalize query-term)` | Reference syntax-directed `D . lift(query) . I` transformation. |
| `(dbsp-start circuit)` | Allocate an empty runtime and initialize delay/operator state. |
| `(dbsp-step! runtime input-alist)` | Atomically apply one tick; return canonically ordered output delta Z-sets. |
| `(dbsp-view runtime name)` | Read a materialized snapshot for inspection or batch-equivalence testing. |

`dbsp-aggregate` treats `count` and `sum` as group homomorphisms. A composable
grouped aggregate keeps an accumulator per key and emits `-old` and `+new`
singleton rows when its rendered result changes. `dbsp-distinct` similarly emits
`+1` on a `0 -> positive` count transition and `-1` on a
`positive -> 0` transition; it does not rescan the input collection.

### 3. Circuit term representation

The serializable term is a versioned, pointer-free S-expression:

```scheme
(dbsp-circuit-v1
  (inputs (events zset))
  (nodes
    (events-delta (input events))
    (events-now   (I events-delta))
    (facts-now
      (map (lambda (event) (event->fact event)) events-now))
    (facts-prev   (z^-1 facts-now)))
  (outputs facts-now))
```

The grammar is:

```text
circuit ::= (dbsp-circuit-v1 (inputs input*) (nodes binding*) (outputs name*))
input   ::= (name group-type)
binding ::= (name term)
term    ::= (input name)
          | (lift fn name*) | (z^-1 name) | (D name) | (I name)
          | (add name*) | (neg name)
          | (map fn name) | (filter fn name) | (project fn name)
          | (join left right left-key right-key emit)
          | (aggregate input key lift group emit)
          | (distinct input)
          | (fixed-point relation seed step-circuit)
fn      ::= identifier | (lambda (named-parameter*) body)
```

Every binding name is unique, references resolve lexically, and a cycle is
rejected unless it crosses `z^-1`. Inline lambdas retain named parameters.
This matches the actual Eshkol AST: lambdas are `ESHKOL_LAMBDA_OP` nodes whose
`lambda_op` contains parameter ASTs and a body
(`inc/eshkol/eshkol.h:1992-2001`, `inc/eshkol/eshkol.h:2183-2194`). The
homoiconic lowering reconstructs a parameter list from the variables' names and
emits `(lambda (params...) body)` (`lib/backend/homoiconic_codegen.cpp:532-570`,
`lib/backend/homoiconic_codegen.cpp:576-625`). DBSP terms will not translate
these bindings to De Bruijn indices.

The runtime representation is separately versioned:

```text
dbsp-runtime-v1 {
  term-hash,
  outer-tick,
  node-state[name],
  delay-state[name],
  trace-spines[name],
  input-frontiers[name],
  output-frontiers[name]
}
```

Executable closures are rebound from the validated term and are not serialized
as raw code pointers. A checkpoint is accepted only when its term hash, operator
schema versions, equality/canonicalization version, and target numeric policy
match the running circuit.

### 4. Operator lowering

The reference transformer applies these rules recursively:

- **Map, filter, project, and union:** linear over Z-sets, so their incremental
  form is the same operator applied directly to the delta batch.
- **Join:** retain indexed prior snapshots of both inputs and compute the three
  product-rule terms. Weights multiply at a match and add during consolidation.
  A two-term implementation may use `dA join B_current + A_previous join dB`;
  it is accepted only if batch equivalence proves it identical to the explicit
  three-term rule.
- **Aggregate:** sum and count apply the incoming weights directly. General
  keyed groups must provide inverse as well as addition; non-invertible
  aggregates require a separate maintained multiset and are not silently
  treated as homomorphisms.
- **Distinct:** retain one exact count per row. The count trace, not the entire
  upstream relation, determines the output boundary delta.
- **Composition:** incrementalize each child and compose the results. The
  compiler must never reconstruct an `I` snapshot between two operators when
  the chain rule can push deltas through both.
- **Recursion:** build a nested circuit with outer time `t` and fixed-point time
  `k`. The recursive back edge crosses `z^-1` at the inner clock. Each inner
  round consumes only the preceding round's new facts and stops on an empty
  consolidated delta. This is semi-naive evaluation derived from the cycle
  rule, not a special transitive-closure algorithm.

### 5. Selene as a standing circuit

The resident loop becomes a thin driver around `dbsp-step!`:

```text
external inputs for tick t
        |
        v
  canonical delta Z-sets
        |
        v
  persistent Selene DBSP circuit
   | event relation
   | knowledge-base views
   | inference/workspace views
   | session and continuity views
        |
        v
  output deltas + effect commands
```

Effects are outside the algebraic circuit. File writes, network sends, actuator
commands, and mutation of legacy consciousness objects occur after the circuit
has produced and committed a deterministic output delta. Their acknowledgements
return as later input events. This makes retry and checkpoint boundaries visible
and prevents the compiler from reordering an untracked effect as if it were a
pure relation operator.

For each cognitive view, the migration requires a batch definition `Q` and an
incremental definition mechanically derived or library-composed from it. During
the migration both execute in tests. For every input prefix:

```text
I(incremental-output-deltas)[t] == Q(I(input-deltas)[t])
```

That invariant is the authority. Performance work is rejected if it changes the
batch meaning.

### 6. `core.memory` mapping

The canonical logical contents of the event-log CRDT will be `Z[Event]`. Its
CRDT causal envelope and its Z-set data algebra are complementary layers of the
same state transition, not interchangeable merge algorithms.

| `core.memory` concept | DBSP interpretation |
|---|---|
| Immutable event with content hash | Z-set row; event id becomes the stable delivery identity. |
| Successful local append | Delta singleton `{event -> +1}`. |
| Logical correction or retraction | A new immutable operation whose derived relation contributes `-1` for the retracted row; history is not erased. |
| RGA/vector-clock merge | Idempotent causal envelope that identifies newly observed operation ids. |
| Newly observed operations after merge | Delta Z-set delivered once to the circuit. |
| Replay of the accepted operation stream | `I`: integrate deltas into the current event relation and its derived views. |
| Delta to send to a peer | `D` relative to that peer's causal frontier: the operations it has not accepted. |
| `memory-fold-lww` | Keyed incremental view over value events. |

This mapping preserves the existing facts that event ids cover the full event
body (`lib/core/memory.esk:45-59`), the RGA merge is an idempotent entry union
(`lib/core/memory.esk:84-90`), and durable append is persist-before-advance
(`lib/core/memory_store.esk:223-249`). The refactor additionally consolidates
RGA entries by content event id before emitting a circuit delta.

There is one critical law: Z-set addition is commutative and invertible, but it
is not idempotent. Therefore replica snapshots must not be merged by blindly
adding their Z-sets; a duplicate event would acquire weight two. The existing
RGA entry identity, tombstones, and vector clocks remain the CRDT envelope. It
performs idempotent union first, and only previously unseen operation ids are
emitted as a delta. The RGA already represents deletion with tombstones and
prevents a stale replica from reintroducing a deleted entry
(`lib/core/distributed.esk:480-487`, `lib/core/distributed.esk:529-555`).

The durable event log remains the source of historical truth. A DBSP checkpoint
is a cache of derived circuit state anchored to a verified event id and vector
clock. On reboot, Selene restores the checkpoint and replays only the event
suffix after that frontier. This is cleaner and bounded relative to new work,
while retaining the ability to audit the complete history. The current head
sidecar already separates an append-optimized session from full replay
(`lib/core/memory_store.esk:260-321`); a circuit checkpoint extends that idea to
all derived views rather than only the latest hash and clock.

### 7. OALR-native state and determinism

Eshkol's memory law is unchanged: arenas and lexical regions, deterministic
bulk reclamation, and no garbage collector. The existing region control block
already has a single-owner lifetime and is
deliberately allocated/freed outside the arena so entering `with-region` in a
hot loop has `O(1)` steady-state space (`lib/core/runtime_regions.cpp:234-247`).
The DBSP implementation uses explicit ownership tiers:

1. **Circuit arena:** owns the runtime descriptor, operator metadata, indexes,
   and current trace-spine roots for the circuit's lifetime.
2. **Immutable batch regions:** each sorted trace batch owns a region. Appending
   a delta creates a new sealed batch. A fixed binary-carry spine policy merges
   equal levels in a deterministic order; after the replacement root is
   published at the tick barrier, superseded batch regions are destroyed
   together. No individual object tracing is required.
3. **Delay ring:** `z^-1` retains exactly one completed tick. Two alternating
   owned regions permit the older value to be destroyed deterministically once
   no node can reference it.
4. **Tick nursery:** input normalization, join probes, temporary output batches,
   and inner fixed-point round scratch live here. It is popped after surviving
   batches and state roots have been moved or promoted to their owners.
5. **Checkpoint/spill regions:** serialization buffers and file-backed batch
   handles have explicit open/close lifetimes and never become anonymous heap
   objects.

The executor will not depend on the existing generic
`iterScopeSafeExpr` analysis to infer this boundary. That analysis correctly
rejects mutators, unknown callees, and unhandled AD/consciousness operations
(`lib/backend/llvm_codegen.cpp:24034-24061`,
`lib/backend/llvm_codegen.cpp:24294-24299`), whereas a DBSP step intentionally
mutates circuit-owned roots. `dbsp-step!` instead establishes a compiler/runtime
known nursery and exposes only typed promotion/publication operations at the
barrier.

ESH-0214c is the baseline for publishing state out of a tick region. At current
master it deep-walks conses, vectors, hashes, tensors, exceptions, and closures,
but explicitly documents knowledge bases, factor graphs, and workspaces as
shallow-copy limitations (`lib/core/runtime_regions.cpp:671-745`). ESH-0214d is
a prerequisite for a Selene DBSP pilot: commit `0843fe43` adds dedicated
evacuation kinds for substitutions, facts, knowledge bases, factor graphs, and
workspaces (commit `0843fe43`, `lib/core/runtime_regions.cpp:686-758`) and tests
promoted workspace closures, knowledge-base facts, and factor-graph buffers
under arena poison (commit `0843fe43`,
`tests/memory/region_evac_subtype_coverage_test.esk:1-36`). No persistent circuit
state containing those subtypes may be published from a tick nursery until that
coverage is on the integration branch and its gate passes.

Trace size is proportional to retained logical state, not merely to the latest
delta. Compaction removes zero weights and coalesces batches, but cannot erase
state that a correct future update may need. When a configured resident-memory
budget would be exceeded, sealed immutable batches spill in canonical order to
checksummed files; the circuit retains an owned indexed handle. This extends the
existing bounded-arena seam, which fails allocation rather than silently growing
past its capacity (`lib/core/runtime_arena_core.cpp:112-124`,
`lib/core/runtime_arena_core.cpp:221-229`). Spill is a deterministic overflow
policy for persistent traces, not a hidden garbage collector and not a mechanism
for allowing unbounded tick scratch.

### 8. Compiler incrementalization and the unified primitive

DBSP's `D`/`I` boundary is structurally the same move as the AD tower's
seed/propagate/readout boundary: change the representation of a primal
computation, apply local differentiation rules compositionally, and obtain a
computation on changes. The meanings remain distinct—numeric AD computes a
local linearization, whereas DBSP computes exact differences along logical
time—but the compiler transformation backbone is the same.

The library implementation is the executable specification. A later
`IncrementalizePass` will accept the same validated query/circuit term and
produce an optimized state machine before LLVM or bytecode lowering. It will:

1. normalize the named-variable AST and resolve captures;
2. reject effects, unstable equality, and unsupported opaque calls inside the
   incremental region;
3. build an operation graph with stable node ids, source spans, input edges, and
   operator attributes;
4. select a differentiation rule for each node;
5. apply the composition, linear, bilinear, and cycle rules;
6. cancel adjacent `D`/`I`, share indexes, and place trace/delay state;
7. emit OALR ownership metadata, checkpoint schema, and cost counters; and
8. lower the resulting circuit identically for AOT, JIT, and the bytecode/WASM
   runtime, subject to each target's persistence facilities.

The shared compiler backbone will be factored from, rather than bolted beside,
the AD path:

```text
DifferentiationGraph node
  id, operation, inputs, captures, type/effect, source-span, attributes

DifferentiationRule
  numeric:     JVP/VJP or Taylor recurrence
  incremental: delta rule plus retained-state requirements
```

Current AD nodes already provide the operation/input graph shape
(`inc/eshkol/eshkol.h:971-1013`), `recordADNodeTensor` already records a dense
operation and its saved state (`lib/backend/autodiff_codegen.cpp:2334-2424`),
and `propagateGradient` dispatches local scalar or tensor rules
(`lib/backend/autodiff_codegen.cpp:10099-10164`). The refactor lifts graph
construction, rule lookup, composition, capture analysis, and diagnostics into
a compiler-neutral layer. Numeric lowering continues to emit dual/Taylor values
or `ad_node_t` tapes; incremental lowering emits Z-set operators, indexes,
delays, and traces. It does not encode Z-sets as numeric AD nodes or make DBSP
state part of the reverse tape.

The eventual surface is one compiler primitive:

```scheme
(differentiate 'numeric computation options ...)
(differentiate 'incremental computation options ...)
```

The interpretation is explicit; it is never guessed from a runtime value.
Numeric differentiation computes a local linear map, tangent, or adjoint over a
numeric domain. Incremental differentiation computes an exact temporal
difference over a commutative group. They are not mathematically identical.
They are the same compiler move at the structural level: turn a primal
computation into a computation on changes, compose local rules, and retain the
minimal state required by those rules.

The existing forms remain source-compatible:

```text
derivative/gradient/jacobian/... -> differentiate 'numeric + seed/readout policy
incrementalize                  -> differentiate 'incremental
```

This unification makes the parallel exact:

- Numeric AD chain rule: differentiate each composed subcomputation and compose
  its local differential.
- DBSP chain rule: `(Q1 . Q2)^D = Q1^D . Q2^D`.
- Numeric product rule: a bilinear primitive propagates both input tangents.
- DBSP join rule: retain prior inputs and propagate `dA`, `dB`, and their cross
  term.
- Numeric nested AD: distinct perturbation/tape levels.
- Recursive IVM: distinct outer-tick and inner-fixed-point clocks.

The current AD composition oracle already tests operator, binding, capture,
nesting, and loop axes against an independent baseline under both JIT and AOT
(`tests/ad_oracle/README.md:1-35`). DBSP batch equivalence will use the same
campaign shape: generate compositions and update schedules mechanically, run
both semantics, and retain every reduced disagreement as a permanent cell.

### 9. Production lessons borrowed from Feldera

Only concepts are adopted.

#### Batch equivalence is the semantic oracle

After every prefix of every generated update stream, integrating the circuit's
output deltas must equal running the batch query over the integrated inputs.
Tests include duplicate inserts, deletions, same-tick insert/delete
cancellation, empty deltas, collisions after projection, negative intermediate
weights, and nested recursion. There is no tolerance for relational results;
numeric payload comparison follows Eshkol's existing exact/inexact policy.

#### Checkpoint/restart is resident continuity

A checkpoint taken only at a completed tick barrier contains circuit tick,
delay slots, trace-spine roots, aggregate/distinct state, nested-circuit state,
input/output frontiers, and the anchoring `core.memory` event id/vector clock.
Restore must produce the same future output deltas as uninterrupted execution.
The event log remains available for audit and post-checkpoint suffix replay; it
is no longer necessary to rebuild every derived cognitive view from the entire
46K-entry history on each ordinary reboot.

Checkpoint publication is atomic: write data, verify checksums and schema, fsync,
then atomically advance a small manifest. A checkpoint never contains transient
tick-nursery addresses or code pointers.

#### Spill-to-storage is an OALR overflow tier

Trace batches are already immutable and append-oriented, so a sealed batch can
move from an owned memory region to an owned file region without changing its
logical identity. A deterministic size threshold, merge order, file naming,
checksum, and cache-eviction order are part of the semantics profile. The
in-memory and forced-spill paths must be batch-equivalent and byte-reproducible
for the same target profile.

## Acceptance tests

All gates are falsifiable and run with deterministic seeds. The initial library
gates run under JIT and AOT; VM/WASM parity becomes blocking before the API is
declared native across all Eshkol targets.

| Gate | Test and failure condition |
|---|---|
| Algebraic inversion | For generated finite Z-set streams, `D(I(s)) = s` at every tick after consolidation. Also test `I(D(s)) = s` with the mandated zero initial snapshot. Any differing row or weight fails. |
| Join product rule | Feed two keyed relations interleaved inserts and deletes, including `dA=-a` and `dB=+b2` in the same tick. Assert incremental join delta equals `(dA join B0) + (A0 join dB) + (dA join dB)` and that its integrated output equals batch join after every tick. The cross term must prevent a spurious `a join b2`. |
| Aggregate homomorphisms | For weighted duplicate values, assert `count(zs)=sum(weights)` and `sum(zs)=sum(value*weight)`. For every delta, `agg(state+d)=agg(state)+agg(d)` for scalar sum/count, including retractions. Grouped output must emit exactly `-old,+new`. |
| Distinct boundaries | Exercise counts `0->1->2->1->0`, cancellation within one tick, and invalid negative relational snapshots. Output must be `+1,0,0,-1` for the valid transitions and strict mode must reject the invalid snapshot. |
| Incremental transitive closure | Maintain non-reflexive reachability over edge insertions, deletions, cycles, and disconnected components. After every outer tick, the nested circuit must quiesce and equal a fresh batch transitive closure exactly. An iteration cap hit, residual nonzero delta, or row mismatch fails. |
| General batch equivalence | Generate typed compositions of map/filter/project/union/join/aggregate/distinct and update prefixes. `I(outputs)[t] == Q(I(inputs)[t])` after every prefix. Preserve and shrink every counterexample. |
| CRDT delivery | Merge the same replica log repeatedly and in all replica orders. Each event id enters the Z-set stream once; duplicate delivery changes no weight; tombstone/retraction views converge; the existing RGA merge remains commutative, associative, and idempotent. |
| OALR resident bound | Run at least one million small-delta ticks with a fixed live relation and arena poison enabled. After warm-up, RSS and arena block counts must remain within a declared plateau; delay state must retain one tick only; no stale pointer may survive nursery pop. |
| Checkpoint continuity | For every cut point in a generated stream, compare uninterrupted execution with checkpoint, process teardown, restore, and suffix replay. Output delta sequence, final views, frontiers, and canonical checkpoint state must match exactly. |
| Forced spill | Run the same stream once in memory and once with a threshold small enough to spill every eligible spine level. Output deltas and checkpoints must match; resident memory must stay below the configured ceiling plus declared scratch allowance. |
| Compiler/library equivalence | For every supported query term, compare the pure-Eshkol executor with `IncrementalizePass` output under AOT, JIT, and VM. Any semantic mismatch or untracked fallback fails. |
| Unified differentiation regression | Run the complete numeric AD oracle and DBSP oracle after introducing `differentiate`. Neither interpretation may change the other's graph, state, counters, or results. |

## Consequences

### Positive

- Selene's cognitive loop becomes an explicit standing computation whose normal
  work is proportional to new and affected information rather than accumulated
  history.
- `core.memory` gains incremental materialized views without surrendering its
  immutable audit log or CRDT convergence.
- Retraction is first-class. Deletes, corrections, forgotten hypotheses, and
  invalidated inferences travel through the same algebra as insertions.
- Recursive knowledge queries obtain semi-naive evaluation from a general
  circuit rule rather than one-off caches.
- Circuit ownership, checkpoint state, and spill state have explicit OALR
  lifetimes; latency does not depend on a tracing collector.
- Numeric AD and IVM converge on one compiler architecture for differentiating
  named, homoiconic computations while retaining domain-specific runtimes.
- Batch equivalence supplies a strong executable specification throughout the
  library-to-compiler migration.

### Costs

- Stateful indexes and traces add persistent memory even when a batch query
  previously had no explicit cache.
- The compiler needs effect analysis and a stable equality/canonicalization
  contract before it can incrementalize arbitrary source expressions safely.
- Correct deletion support makes joins, distinct, aggregation, and recursive
  queries more complex than insert-only streaming.
- Checkpoint schema evolution, deterministic compaction, and file-backed batch
  ownership become long-term runtime compatibility surfaces.
- During migration, batch and incremental implementations intentionally coexist,
  increasing test and maintenance cost.

## Staged trajectory

These are additive slices within the existing release themes, not replacements
for those releases. Every stage is independently shippable and has a blocking
gate; no later compiler primitive is required to use the earlier library.

| Target | Shippable scope | Blocking gate |
|---|---|---|
| v1.4-connection | Freeze Z-set equality, weight, term-version, tick-atomicity, and batch-equivalence contracts in this ADR; add the deterministic batch-vs-delta oracle harness and corpus format. | Reference fixtures cover insert, delete, duplicate, cancellation, and canonical serialization; corpus regeneration is byte-stable. |
| v1.5.0-intelligence | Ship pure-Eshkol `core.dbsp` Z-sets, finite stream runner, `z^-1`, `D`, `I`, add/negate, map, filter, project, and union. | `D.I` and `I.D` inversion gates pass under JIT and AOT; Z-set group laws pass generated tests. |
| v1.5.1-intelligence | Ship the circuit builder/runtime, atomic `dbsp-step!`, indexed join, explicit circuit arena/tick nursery, deterministic node scheduling, and inspection counters. | Interleaved insert/delete join product-rule gate and one-million-tick arena-poison/RSS gate pass. |
| v1.6.0-reasoning | Ship incremental sum/count, general group aggregate, grouped result replacement, distinct, and the full batch-equivalence composition matrix. | Aggregate, distinct-boundary, and generated batch-equivalence gates pass with no untracked cell. |
| v1.6.1-reasoning | Replace reference flat traces with immutable sorted batches and deterministic binary-carry spines; add consolidation and trace metrics. | Reference and spine executors emit identical deltas; adversarial compaction schedules produce one canonical state. |
| v1.7.0-synthesis | Ship nested-clock fixed points and incremental recursive relations, initially stratified positive Datalog-shaped terms with deletion support. | Incremental transitive closure equals batch closure after every insertion/deletion prefix and quiesces within the declared bound. |
| v1.8.0-platform | Refactor `core.memory` so newly accepted CRDT operations feed a Z-set delta stream; maintain LWW and selected knowledge/session views incrementally while retaining RGA/hash-chain compatibility. | Replica-order, duplicate-delivery, tombstone, durable append, audit, and old-log migration gates pass. |
| v1.8.1-platform | Pilot Selene's event, knowledge, workspace, and session derivations as one library-built circuit; require integrated ESH-0214d subtype evacuation. | Shadow mode matches existing derived state for a sustained resident run; ESH-0214d poison/RSS coverage and per-tick batch equivalence remain green. |
| v1.9.0-types | Ship `IncrementalizePass`, shared differentiation graph/rule registry, purity/effect rejection, OALR state placement, and native/VM lowering for the supported query subset. | Compiler output equals `core.dbsp` on the complete oracle; unsupported effects fail at compile time with source spans, never by silent batch fallback. |
| v1.9.1-types | Ship tick-barrier checkpoint/restart with circuit schema hashes and `core.memory` frontiers. | Crash-and-restore at every generated cut point is identical to uninterrupted execution; a 46K-entry history restores derived state from checkpoint plus suffix. |
| v1.9.2-types | Ship deterministic file-backed trace batches and bounded-arena spill policy on hosted targets. | Forced-spill equivalence, checksum-corruption rejection, file-lifetime, and resident-memory ceiling gates pass. |
| v2.0-starlight | Ship `differentiate` with explicit `numeric` and `incremental` interpretations; lower existing AD forms and `incrementalize` through the shared compiler backbone. | Numeric AD and DBSP composition oracles both pass across AOT, JIT, and VM/WASM; no interpretation-specific runtime representation leaks into the other. |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CRDT merge is mistaken for Z-set addition, so duplicate delivery doubles weights. | Keep RGA/vector-clock deduplication as the causal envelope; emit only newly accepted operation ids as deltas; make duplicate-delivery a blocking gate. |
| “O(change)” is overstated for high-fanout joins, recursive updates, or compaction. | Publish counters for input weights, index probes, output weights, inner rounds, and merge work; specify cost as change plus affected results and scheduled compaction. |
| Persistent traces grow without a collector. | Consolidate zeros, compact immutable spine levels deterministically, bound retention where semantics permit, checkpoint, and spill sealed batches before bounded-arena exhaustion. |
| A tick publishes a pointer into its nursery. | Use typed publication APIs, ESH-0214c deep promotion, integrated ESH-0214d coverage for consciousness subtypes, arena poison, and million-tick lifetime gates. |
| Effects are duplicated or reordered by incrementalization. | Admit only pure deterministic query regions; represent external effects as output commands and acknowledgements as later input events. |
| Numeric and temporal differentiation are conflated. | Require an explicit interpretation, share only compiler structure, retain distinct rule sets and runtime representations, and run both oracle suites together. |
| Distinct or an aggregate receives an invalid negative relation snapshot. | Preserve algebraic Z-sets internally, validate positivity at declared relation boundaries, and fail strict mode with the offending row and tick. |
| Deletion through recursive queries causes excessive work or nontermination. | Require per-tick quiescence metrics and caps, preserve a batch fallback only as an explicit diagnostic tool, and do not declare a recursive operator supported until insert/delete equivalence passes. |
| Checkpoints become incompatible with changed code or equality rules. | Hash the canonical circuit term and all operator/schema policies; reject incompatible state and fall back explicitly to verified log replay. |
| Spill makes latency nondeterministic. | Use fixed thresholds and synchronous step-boundary publication initially; make asynchronous prefetch/merge a later opt-in with the same logical ordering. |

## Non-goals

- No SQL syntax, SQL planner, or Apache Calcite front end.
- No Rust `dbsp` crate binding, Feldera embedding, or cross-language circuit ABI.
- No tracing, generational, or stop-the-world garbage collector.
- No replacement of the content-addressed event log, its hash chain, RGA order,
  vector clocks, or audit path.
- No automatic incrementalization of arbitrary I/O, mutation, continuations,
  nondeterministic procedures, or opaque foreign calls.
- No promise that update time is independent of affected output size.
- No distributed consensus, transactional connector protocol, or general
  exactly-once external effects in this ADR.
- No wall-clock/window semantics in the first module; logical tick time is the
  only clock defined here.
- No automatic parallel execution. Deterministic single-thread semantics are
  established first.

## Deferred

- Native specialized kernels beyond the pure-Eshkol reference executor.
- Windowing, watermarks, late data, and bounded-history stream operators.
- Distributed/sharded traces and cross-node fixed points.
- Browser-specific durable checkpoint and spill backends.
- Non-group aggregates that require order statistics or retained value bags.
- Incremental maintenance across arbitrary schema/circuit upgrades.
- Asynchronous compaction and I/O scheduling.
- Proof-carrying validation of the unified differentiation rule registry.

## References

### DBSP and Feldera

- Mihai Budiu, Tej Chajed, Frank McSherry, Leonid Ryzhyk, and Val Tannen,
  [“DBSP: Automatic Incremental View Maintenance for Rich Query Languages”](https://www.vldb.org/pvldb/vol16/p1601-budiu.pdf),
  *Proceedings of the VLDB Endowment* 16(7), 2023,
  doi:10.14778/3587136.3587137. Definitions 2.1, 2.5, 2.15, 2.17,
  and 3.1 establish streams, delay, `D`, `I`, and `Q^D`; Proposition 3.2 and
  Theorems 3.3-3.4 establish composition, linear, bilinear, and cycle rules;
  Sections 5-6 establish nested-clock recursive evaluation.
- [`dbsp` 0.314.0 Rust crate documentation](https://docs.rs/dbsp/0.314.0/dbsp/) and
  [trace/spine documentation](https://docs.rs/dbsp/0.314.0/dbsp/trace/), consulted
  for the production concepts of immutable batches, appendable traces, spines,
  logical time, and file-backed batch implementations. These are references,
  not dependencies.
- Feldera,
  [“Feldera, DBSP and incremental view maintenance”](https://docs.feldera.com/sql/intro/),
  consulted for standing-query and input/output change semantics.
- Feldera, [documentation overview](https://docs.feldera.com/), consulted for
  the production consistency contract that an incremental view agrees with its
  batch result for the same input.
- Feldera,
  [“Checkpoints & Fault Tolerance”](https://docs.feldera.com/pipelines/fault-tolerance/),
  consulted for consistent operator-state checkpoints, connector frontiers, and
  checkpoint-plus-journal restart.
- Feldera,
  [“Memory Usage”](https://docs.feldera.com/operations/memory/), consulted for
  immutable index batches, background merging, storage caches, and spilling
  state beyond resident memory.

### Eshkol source anchors

- `lib/core/memory.esk:1-129` — content-addressed RGA event log, append, merge,
  verification, and full-sequence LWW fold.
- `lib/core/memory_store.esk:188-321`, `lib/core/memory_store.esk:389-443` —
  durable replay, persist-before-advance, fast head restore, and streaming audit.
- `lib/core/distributed.esk:480-654` — RGA entries, tombstones, deterministic
  traversal, and idempotent merge.
- `lib/core/logic.cpp:625-690`, `lib/core/workspace.cpp:244-338`, and
  `lib/backend/logic_workspace_codegen.cpp:597-649` — mutable knowledge-base
  append/query and one persistent workspace step.
- `lib/core/runtime_arena_core.cpp:273-432` — arena scopes and automatic
  per-iteration pop-or-commit.
- `lib/core/runtime_regions.cpp:623-1124` — ESH-0214c deep evacuation and region
  write barrier; commit `0843fe43`, `lib/core/runtime_regions.cpp:686-1195` —
  ESH-0214d consciousness-subtype coverage.
- `lib/backend/llvm_codegen.cpp:23752-24383`,
  `lib/backend/llvm_codegen.cpp:26803-26904` — static iteration-scope safety,
  runtime scope release, and loop lowering.
- `inc/eshkol/eshkol.h:958-1023`,
  `lib/backend/autodiff_codegen.cpp:2311-2424`,
  `lib/backend/autodiff_codegen.cpp:9965-10164` — AD graph/tape representation,
  dense node recording, reverse traversal, and local chain-rule dispatch.
- `docs/design/AD_STAGED_KERNEL_HANDOFF.md:19-35`,
  `docs/design/AD_STAGED_KERNEL_HANDOFF.md:104-172`,
  `docs/design/AD_STAGED_KERNEL_HANDOFF.md:202-244` — current AD substrate and
  staged one-pass direction.
- `tests/ad_oracle/README.md:1-67` — composition-matrix differential oracle and
  JIT/AOT gate shape.
