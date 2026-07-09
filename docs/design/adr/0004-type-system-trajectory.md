# ADR 0004: One quantitative dependent type system for Eshkol

- Status: Proposed
- Date: 2026-07-09
- Decision owners: language, frontend, runtime, and LLVM backend
- Scope: v1.3.2 through v2.0

## Decision

Eshkol will have one typed elaboration pipeline, not separate gradual,
dependent, ownership, effect, refinement, session, and quantum checkers. Every
expression is checked by the same judgment:

```text
Σ ; Φ ; Γ ; Δin ⊢ e ⇒ A ! ε ⊣ Δout ↝ h
Σ ; Φ ; Γ ; Δin ⊢ e ⇐ A ! ε ⊣ Δout ↝ h
```

Here `A` is the value type, `ε` is an effect row, `Φ` is the set of index,
refinement, and outlives facts, `Γ` contains duplicable bindings, `Δ` is the
flow-sensitive affine/linear ownership context, and `h` is typed HIR with all
casts, checks, moves, loans, cleanups, region promotions, and handlers made
explicit. The output ownership context is as much a result of checking as the
value type.

The extensions compose because they occupy distinct parts of that judgment:

| Concern | Meaning | Static home |
|---|---|---|
| Dependent types | Later types may mention earlier *pure, stable* values | `A`, binders, and `Φ` |
| Refinements | A value satisfies a decidable proposition | `A` and entailment from `Φ` |
| Affine/linear types | Whether a value may be dropped or duplicated | binders and `Δ` |
| OALR ownership | Who owns storage, how long it lives, and which loans exist | region-indexed types, `Φ`, and `Δ` |
| Effects | What evaluating an expression may do | `ε` |
| Session types | Which communication action a unique endpoint permits next | a dependent protocol index in `A`, advanced through `Δ` |
| Quantum types | A non-clonable resource indexed by a quantum region and size/state | a linear family in `A`, advanced through `Δ`, with `Quantum ρ` in `ε` |
| Gradual typing | Where static information is absent and evidence must be checked | explicit unknowns and coercions in `h` |

No axis is allowed to erase another. In particular, an unknown value is not
silently duplicable, an unknown effect is not pure, an unknown lifetime does
not outlive the current region, and an unproved refinement is not true.

## Context and implementation evidence

The present implementation contains useful pieces, but their representations
do not yet form a type system.

### Nominal IDs currently carry too much meaning and too little structure

`TypeId` packs a 16-bit numeric identity, universe, and flags into the AST, but
its equality and ordering compare only the numeric identity
([`inc/eshkol/types/hott_types.h:102-141`](../../../inc/eshkol/types/hott_types.h#L102-L141)).
Linearity is a flag on that nominal identity
([`inc/eshkol/types/hott_types.h:57-63`](../../../inc/eshkol/types/hott_types.h#L57-L63)),
which cannot express that one instantiation or one binding is affine while
another use of the same constructor is unrestricted.

There are structural side objects for parameterized types and Pi types, but
the main checker still returns a `TypeId`. A parameterized instantiation is
cached back to its *base* ID rather than receiving structural identity
([`lib/types/hott_types.cpp:702-716`](../../../lib/types/hott_types.cpp#L702-L716)).
The Pi representation stores only a `TypeId` result plus a Boolean claiming
whether it is dependent
([`inc/eshkol/types/hott_types.h:370-405`](../../../inc/eshkol/types/hott_types.h#L370-L405));
ordinary function construction always creates a non-dependent Pi
([`lib/types/hott_types.cpp:884-896`](../../../lib/types/hott_types.cpp#L884-L896)).
Subtype, dimension, function, and pair caches are consequently keyed by
numeric IDs or parallel side tables
([`inc/eshkol/types/hott_types.h:422-448`](../../../inc/eshkol/types/hott_types.h#L422-L448)).

The C surface type AST reserves `HOTT_TYPE_DEPENDENT` and `HOTT_TYPE_PATH` for
future use, but its payload union has no representation for either
([`inc/eshkol/eshkol.h:1681-1717`](../../../inc/eshkol/eshkol.h#L1681-L1717),
[`inc/eshkol/eshkol.h:1723-1775`](../../../inc/eshkol/eshkol.h#L1723-L1775)).
The packed `inferred_hott_type` field likewise has room only for the nominal
ID, universe byte, and flags
([`inc/eshkol/eshkol.h:2453-2456`](../../../inc/eshkol/eshkol.h#L2453-L2456)).
That field is an erased ABI hint, not a viable home for a complete type.

### Dependent checking is detached from ordinary checking

There are two compile-time value representations: `CTValueSimple` in the
nominal type layer and `CTValue` in the dependent layer. The latter can retain
a non-owning raw AST pointer as a symbolic expression
([`inc/eshkol/types/dependent.h:41-58`](../../../inc/eshkol/types/dependent.h#L41-L58)).
`DependentType` is a separate base/type-index/value-index object
([`inc/eshkol/types/dependent.h:218-272`](../../../inc/eshkol/types/dependent.h#L218-L272)),
not a case of the type returned by `synthesize`.

Unknown dependent equality is treated as inequality
([`lib/types/dependent.cpp:389-395`](../../../lib/types/dependent.cpp#L389-L395)),
and an unknown bound produces a “proof required” failure
([`lib/types/dependent.cpp:403-449`](../../../lib/types/dependent.cpp#L403-L449)),
but there is no evidence language, constraint context, or gradual contract to
discharge that proof. Even successful matrix dimension checking returns the
generic `Tensor` type rather than its result shape
([`lib/types/type_checker.cpp:3188-3208`](../../../lib/types/type_checker.cpp#L3188-L3208)).

### “Gradual” currently means imprecision plus warnings, not evidence

The checker has synthesis and checking entry points, but checking is just
“synthesize, then nominal subtype”
([`lib/types/type_checker.cpp:969-1008`](../../../lib/types/type_checker.cpp#L969-L1008)).
Lambda checking does not use the expected function type
([`lib/types/type_checker.cpp:2933-2945`](../../../lib/types/type_checker.cpp#L2933-L2945)),
and unification is equality or subtyping without metavariables or substitution
([`lib/types/type_checker.cpp:2962-2973`](../../../lib/types/type_checker.cpp#L2962-L2973)).
At calls, `Value` on either side suppresses compatibility checking
([`lib/types/type_checker.cpp:2613-2633`](../../../lib/types/type_checker.cpp#L2613-L2633)).
`Value` is therefore simultaneously the runtime tagged-value supertype, an
inference hole, and a gradual escape hatch. These must be separate concepts.

`reportTypeIssue` changes printed severity, records the issue, and returns
immediately in unsafe mode
([`lib/types/type_checker.cpp:3115-3143`](../../../lib/types/type_checker.cpp#L3115-L3143)).
Program checking constructs the checker with default modes even when its own
`strict` parameter is set
([`lib/types/type_checker.cpp:3465-3479`](../../../lib/types/type_checker.cpp#L3465-L3479)).
The replacement modes defined below control obligations and elaboration, not
just diagnostic wording.

### Ownership utilities are not integrated with expression typing

The current `Context` tracks ordinary types by source name and keeps a second,
non-scoped set and counter map for linear variables
([`inc/eshkol/types/type_checker.h:88-166`](../../../inc/eshkol/types/type_checker.h#L88-L166)).
The borrow checker is another name-indexed state machine
([`inc/eshkol/types/type_checker.h:204-310`](../../../inc/eshkol/types/type_checker.h#L204-L310)).
It is not threaded through branches or returned by a typing judgment. In the
main checker, its only expression-level query is the special case for a
`set!` target
([`lib/types/type_checker.cpp:2494-2511`](../../../lib/types/type_checker.cpp#L2494-L2511)).

Most importantly, `with-region`, `owned`, `move`, `borrow`, `shared`, and
`weak-ref` all synthesize unqualified `Value`
([`lib/types/type_checker.cpp:1308-1315`](../../../lib/types/type_checker.cpp#L1308-L1315)).
The unsafe context is documented as allowing linear duplication and use after
move
([`inc/eshkol/types/type_checker.h:312-324`](../../../inc/eshkol/types/type_checker.h#L312-L324)).
That is incompatible with using the same mechanism to prove quantum
no-cloning.

### The OALR runtime is a strong lowering target

OALR already has lexical region syntax and distinct `owned`, `move`, and
`borrow` AST payloads
([`inc/eshkol/eshkol.h:2019-2025`](../../../inc/eshkol/eshkol.h#L2019-L2025),
[`inc/eshkol/eshkol.h:2227-2250`](../../../inc/eshkol/eshkol.h#L2227-L2250)).
At runtime, `with-region` routes allocations to a region arena, promotes the
result, destroys the arena, and restores the parent allocation target
([`lib/backend/llvm_codegen.cpp:29940-30020`](../../../lib/backend/llvm_codegen.cpp#L29940-L30020)).
The region stack is thread-local and records the parent arena used for escape
promotion
([`lib/core/arena_memory.h:320-342`](../../../lib/core/arena_memory.h#L320-L342)),
while the runtime performs deterministic bulk destruction
([`lib/core/runtime_regions.cpp:291-336`](../../../lib/core/runtime_regions.cpp#L291-L336)).
Object headers already have linear, borrowed, consumed, shared, and external
flags
([`inc/eshkol/eshkol.h:450-474`](../../../inc/eshkol/eshkol.h#L450-L474)).

Those runtime features remain as lowering operations and dynamic/debug guards.
They do not define source typing. In particular, toggling a header bit after
code generation cannot prove that every control-flow path moved, closed, or
borrowed a value correctly.

## Non-negotiable invariants

1. **Safety obligations never degrade to warnings.** Use after move, duplicate
   use of a linear value, illegal aliasing, escaping a dead region, an invalid
   session transition, and quantum cloning are compile-time errors in typed
   code. A gradual boundary may insert a stateful runtime guard, but it may not
   merely warn and continue.
2. **Unknown is per axis.** Missing value information is `Dyn`; missing effects
   are an open effect tail; missing indices are existential metavariables;
   missing protocols are monitored protocol variables. None grants
   `Clone`, `Pure`, an outlives relation, or a proof.
3. **Only pure, total, stable terms occur in types.** Type indices and
   refinement predicates cannot perform I/O, mutate state, throw, capture a
   continuation, borrow a mutable place, or consume a linear value.
4. **Proof erasure cannot erase a resource.** Proofs are irrelevant and
   duplicable only in `Prop`; elimination from `Prop` into runtime data cannot
   smuggle a linear or region-owned witness into an erased term.
5. **All implicit runtime work is explicit in typed HIR.** A source-level
   implicit cast, region escape, cleanup, dynamic check, or handler becomes a
   node in `h` before LLVM lowering.
6. **Control flow checks resources path-sensitively.** Branches, loops,
   exceptions, continuations, closures, async suspension, and effect handlers
   all transform or capture `Δ`; there is no global name-use counter.
7. **Erasure is the last step.** ABI representation is selected after typing.
   Runtime representation does not participate in definitional equality.

## Core static language

The user syntax can remain Scheme-like. The following is the normalized core
the parser and resolver elaborate into.

```text
u       ::= 0 | succ u | universe-meta
mode    ::= unrestricted | affine | linear

A, B    ::= Dyn[mode] | Any | Never | α
          | Type u | Prop
          | Π (mode x : A). Comp ε B
          | Σ (x : A). B
          | A + B | A × B | F A index...
          | {x : A | φ}
          | Own ρ A | Shared ρ A | Weak ρ A
          | Borrow shared ℓ ρ A | Borrow mut ℓ ρ A
          | Handle kind ρ state
          | Chan ρ protocol
          | Qubit ρ | QReg ρ n

ε       ::= <label..., effect-meta | open-tail>
label   ::= IO | Alloc ρ | State ρ | Throw E | Async
          | Control mode | Quantum ρ | Unsafe

protocol ::= End
           | !(x : A).protocol | ?(x : A).protocol
           | ⊕{label : protocol...} | &{label : protocol...}
           | μ p.protocol | p
```

`Type u : Type (succ u)` is predicative. `Prop` is proof-irrelevant and erased.
Eshkol remains HoTT-inspired, but v2.0 does not depend on implementing
univalence or higher inductive types. Path/equality types are intensional
proofs with transport. This keeps the kernel small enough to audit while
supporting the dependent, refinement, session, and quantum use cases on the
roadmap.

### Four relations, not one overloaded subtype test

The kernel keeps these relations distinct:

- `A ≡ B`: definitional equality by normalization and structural equality.
- `A <: B`: static subtyping, including refinement implication and effect-row
  inclusion.
- `A ~ B`: gradual consistency, which produces coercion evidence and may
  insert a runtime contract.
- `Φ ⊨ φ`: proposition entailment, discharged by normalization, a decidable
  theory, an SMT solver, or explicit proof evidence.

Consistency is deliberately not transitive and must never be cached as
subtyping. A failed known refinement or known dimension is an error; an
unknown but runtime-reifiable fact can become a checked coercion; a
non-reifiable unknown requires proof or rejection.

Static subtyping is structural and conservative: Pi inputs are contravariant,
results are covariant under the bound variable, and a computation with fewer
effects is a subtype of one with more declared effects. Refinement subtyping is
predicate implication. `Own`, mutable borrows, handles, and channels are
invariant in the indices that govern mutation or protocol state; a shared
borrow may shorten its loan and be covariant in an immutable payload. Session
subtyping is initially disabled.

Usage modes do not participate in an unsound “qualifier cast.” A duplicable
value can be passed to a one-use binder; an affine value becomes duplicable
only through an explicit `Clone` implementation; and a strict-linear value can
be discarded only through an explicit consuming operation. Any such adaptation
is evidence in HIR rather than a flag rewrite.

### Usage modes belong to bindings and capabilities

The existing `TYPE_FLAG_LINEAR` treats linearity as an intrinsic Boolean
property of a nominal type. The new system uses three usage modes:

- **Unrestricted** values have `Clone` and `Drop`; they may be used any number
  of times.
- **Affine** values do not have `Clone`; they may be explicitly consumed once
  or deterministically dropped.
- **Linear** values have neither `Clone` nor implicit `Drop`; every path must
  consume them exactly once.

Internally, the checker can represent the obligation as a usage interval:
`[0,∞]`, `[0,1]`, or `[1,1]`. A resource constructor constrains the mode of its
bindings, but the mode is not a mutable flag on the nominal constructor.

`owned` means unique ownership and is therefore affine by default. A socket or
file becomes affine when it has deterministic close/drop glue. A protocol
endpoint is linear unless its protocol defines an explicit `Abort` transition.
A qubit is strictly linear: it has no `Clone` and no implicit `Drop`.
Compiler-inserted drop glue is an explicit consumption in HIR, so “guaranteed
close” remains true even when source code does not spell `close`.

### The one checking algorithm

Checking an expression performs these operations together:

1. Resolve surface types into hash-consed core terms and create metavariables
   for omitted value, index, region, effect, and mode information.
2. Bidirectionally synthesize or check `A`; thread `Δ` through evaluation
   order; collect effects into `ε`; add equations, predicates, and outlives
   facts to `Φ`.
3. At a branch, require compatible value types and join effects, but also join
   residual ownership states. A linear resource must be consumed on every
   branch or remain in the same state on every branch.
4. At a loop, infer or check an invariant over types, refinements, effects, and
   the ownership state. Back edges must reproduce that invariant.
5. Solve kind/universe constraints, definitional equality and unification,
   usage/region constraints, effect rows, and finally refinement entailment.
6. Elaborate every accepted non-identity relation into explicit HIR evidence.
   Reject any unsolved hard obligation before code generation.

Facts in `Φ` refer to SSA values or stable places, never just source names.
Mutation creates a new version and invalidates facts about the old value.
Returning a mutable borrow invalidates facts not guaranteed by its
postcondition. This is how refinement checking remains sound in a language
with `set!` and state effects.

Generalization follows an effect-and-ownership value restriction: a binding
may generalize type, index, region, or effect variables only when its right
hand side is a value, has a closed empty effect row, leaves `Δ` unchanged, and
captures no local region or affine/linear place. Higher-rank types remain
explicit.

## Gradual typing as evidence, not permissiveness

`Any` and `Dyn` are different:

- `Any` is the static top type. Eliminating it exposes only operations valid
  for every value.
- `Dyn` means the producer withheld type information. Eliminating it requires
  a coercion with a runtime descriptor and a blame label.

Untyped R7RS code is checked as `Dyn[unrestricted]` and remains source
compatible. It cannot directly contain a unique resource. A resource crossing
that boundary is sealed in `Dyn[affine]` or `Dyn[linear]`; the wrapper itself is
tracked in `Δ` and carries a runtime generation/consumption token. Untyped code
may copy the opaque wrapper as ordinary Scheme data, but every alias shares one
token and therefore cannot duplicate the underlying authority. It can pass the
wrapper or invoke a checked operation, but cannot inspect or serialize the
resource. A strict-linear export additionally requires a delimited linear
contract that observes one consumption before the untyped call returns; if the
lifetime cannot be delimited, that export is rejected.

| Unknown information | Representation | Conservative rule |
|---|---|---|
| Ordinary value type | `Dyn[unrestricted]` | runtime type contract with blame |
| Unique/linear value type | sealed `Dyn[affine/linear]` | one shared authority token; consume/scope monitor |
| Dependent index | existential index plus reifier | unpack or check a decidable equality |
| Refinement | predicate coercion | run a pure predicate, or require proof |
| Effect row | open effect tail | propagate it; never infer `Pure` |
| Region | existential region package | cannot expose a local region name or raw loan |
| Session protocol | `Chan ρ ?p` plus monitor | endpoint remains linear; transitions checked dynamically |

The compiler modes become:

- `--types=gradual`: permit `Dyn` and reifiable unknown obligations, inserting
  contracts. Ownership, lifetime, and no-cloning violations remain errors.
- `--types=strict`: additionally reject unresolved `Dyn`, open exported effect
  rows, and runtime-only proof obligations at typed public boundaries.
- `unsafe`: introduce explicitly typed trusted primitives with an `Unsafe`
  effect. It does not turn off the checker. It may manufacture a postcondition
  or raw pointer under a declared contract, but cannot clone a qubit or make a
  dangling loan well-typed.

For the pure unrestricted fragment, Eshkol should satisfy the gradual
guarantee: removing an annotation changes static knowledge into casts, not
program meaning before blame. Across effects and resources, the stronger
required property is safety monotonicity: losing precision can add a dynamic
guard, never add `Clone`, `Pure`, or `Outlives` authority.

## Dependent and refinement types share one term and proof layer

The two current compile-time value classes are replaced by one immutable
`IndexTerm` representation. It supports variables, literals, constructors,
projection, and calls to functions accepted by the totality checker. Raw AST
pointers are never stored in types or cache keys.

The type-level fragment is phased and total:

- only unrestricted values may be quoted into an index;
- the producing expression must have the empty closed effect row;
- recursion must be structurally decreasing or separately proved total;
- normalization is fuel-independent for accepted definitions;
- opaque runtime computations can appear only through existential indices and
  explicit equality evidence.

Refinement `{x : A | φ}` is semantically an erased dependent pair
`Σ (x : A). proof(φ)`, with the proof in `Prop` and runtime representation
equal to `A`. Its predicate language initially contains equality, ordering,
linear integer arithmetic, Boolean connectives, constructor tests, lengths,
shapes, protocol labels, and region outlives facts. Arbitrary Scheme predicates
are not refinements unless separately proved pure and total.

Obligation discharge is ordered and deterministic:

1. normalize both sides and solve definitional equality;
2. substitute unification metavariables;
3. use local constructors, arithmetic, shape, and outlives solvers;
4. ask SMT only for the decidable refinement fragment, with a time and memory
   budget;
5. on `unknown`, insert a runtime check only if `φ` is reifiable and pure;
6. otherwise require an explicit proof term or reject.

An SMT `sat` result is not proof of a subtype; refinement subtyping asks whether
`Φ ∧ source ⇒ target`, implemented as unsatisfiability of its negation. Solver
timeouts never become success. Until proof certificates are checked, the SMT
adapter is part of the trusted computing base and must be version-pinned in
typed interface metadata.

Dependent function and pair types retain binders in the core term. A function
such as matrix multiplication can therefore have the actual type:

```text
matmul : Π (m n p : Nat).
         Tensor Float64 [m,n] -> Tensor Float64 [n,p]
         -> Comp <Alloc ρ> (Tensor Float64 [m,p])
```

When a dynamic tensor supplies a shape, the boundary elaborates it to
`Σ (shape : Vec Nat rank). Tensor A shape`, validates the descriptor, and makes
the witness available to later checking. It does not discard the shape and
return generic `Tensor`.

## Effects are the computation dimension of the same type

The canonical function result is `Comp ε A`; arrow syntax is sugar for a Pi
whose body is a computation. `Pure` is the empty *closed* row, not a label.
Rows are ordered sets with optional tail variables, support inclusion, and are
part of schemes and interface hashes.

Initial labels are:

- `IO` for host I/O and network operations;
- `Alloc ρ` for allocation or promotion into region `ρ`;
- `State ρ` for mutation visible through region `ρ`;
- `Throw E` for exceptions of type `E`;
- `Async` for suspension or scheduling;
- `Control mode` for continuation capture/resumption;
- `Quantum ρ` for quantum state transitions;
- `Unsafe` for trusted low-level operations.

Builtin behavior moves out of the long name-dispatch in
`synthesizeApplication` and into a versioned signature registry. Each entry
has a `TypeScheme`, effect row, ownership transfer, refinement pre/postcondition,
and lowering symbol. Native, VM, JIT, and FFI paths consume the same registry.
An unannotated FFI call has `Dyn` arguments/results plus `<IO, Throw Foreign,
Unsafe | ?e>` and may not accept a raw affine/linear value. FFI ownership must
say `borrow`, `take`, `return-owned ρ`, or `shared`.

Effects interact with ownership at control boundaries:

- Throwing and early return elaborate cleanup edges. Live affine values with
  `Drop` are consumed there; strict-linear values must already have an explicit
  transition valid on that path.
- A multi-shot continuation can capture only a `Δ` containing duplicable
  values. Capturing resources yields a one-shot `Control linear` continuation.
- A continuation carries the free regions and loans of its captured
  environment just like a closure. It cannot escape those bounds; an untyped
  `call/cc` inside a resource scope needs a delimiting contract or is rejected.
- An algebraic handler's resumption has a usage mode. Resuming twice is legal
  only when the captured `Δ` is unrestricted; a one-shot handler can carry
  sockets, session endpoints, or qubits.
- A shared or mutable loan cannot cross `Async` suspension unless its lifetime
  outlives the task and its type satisfies the required thread-safety traits.
- Forking splits `Δ`; a unique endpoint or owner goes to exactly one task.

Thus algebraic effects cannot be added later as a syntactic transform that is
oblivious to linear captures.

## OALR as region-indexed affine ownership

OALR gets a static model with two different lifetime concepts:

- `ρ : Region` names an allocation arena and participates in `Outlives` facts.
- `ℓ : Loan` names a borrow interval inferred over the control-flow graph.

They are not interchangeable. A value can be allocated in `ρ` and borrowed
for a much shorter `ℓ`.

The principal types are:

```text
Cap ρ                    -- scoped capability that opens/closes region ρ
Own ρ A                  -- unique value stored in ρ
Borrow shared ℓ ρ A      -- duplicable read loan, bounded by ℓ and ρ
Borrow mut ℓ ρ A         -- exclusive mutable loan, bounded by ℓ and ρ
Shared ρ A               -- ref-counted alias whose storage still depends on ρ
Weak ρ A                 -- non-owning observation, upgraded to Option (Shared ρ A)
```

`with-region` introduces a fresh, generative `ρ`, a scoped `Cap ρ`, and the fact
`Outlives(parent, ρ)`. It cannot unify `ρ` with a region outside the construct.
Closing it performs, in typed HIR order:

1. end every loan into `ρ`;
2. run reverse-order cleanup for live affine external resources;
3. reject any remaining strict-linear resource that has no legal cleanup
   transition;
4. consume `Cap ρ` and bulk-free arena storage.

After non-escape is proved, `with-region` handles the local `Alloc ρ` and
private `State ρ` labels, so they do not leak into the outward effect row.
Effects of acquisition or cleanup that are externally observable—such as
`IO`, `Throw E`, or `Unsafe`—remain in that row.

Region allocation yields `Own ρ A` and effect `Alloc ρ`. `move` consumes the
source place and creates a new owner of the same value; it does not copy and it
does not rely on a transient object-header flag. A shared borrow freezes moves
and mutation of its owner for `ℓ`; a mutable borrow is exclusive and temporarily
grants mutation. The owner reappears in `Δ` when the loan ends.

A type mentioning local `ρ` cannot escape directly. Existing source behavior
is preserved by elaborating return and outliving-store escapes to an explicit
HIR coercion:

```text
promote ρ parent : Own ρ A
                 ⊸ Comp <Alloc parent> (Own parent A)
```

Promotion requires `Promotable A`. For arena data it performs the existing
deep graph promotion while preserving sharing; for an external handle a
separate `Rehome` implementation transfers bookkeeping without copying the OS
resource; a loan is never promotable. The current runtime already has the
right parent-target concept for escape copies
([`lib/core/runtime_regions.cpp:500-556`](../../../lib/core/runtime_regions.cpp#L500-L556)).
Typed HIR makes when and why that cost occurs visible to optimization and
diagnostics.

`Shared ρ A` does not erase `ρ`. It may be duplicated only while `ρ` outlives
all aliases, or it must be promoted to a longer-lived shared heap. A weak
reference is similarly region-indexed and always upgrades through a checked
`Option`; it is never evidence that the allocation is alive.

Closures derive their usage mode and region bound from captures. Capturing an
affine value makes an affine/one-shot closure; capturing a loan bounds the
closure by `ℓ`; capturing `Own ρ A` prevents the closure from escaping `ρ`
unless the capture is promoted. This replaces source-name counts with actual
ownership of places.

## Resource typestate and session types

The v1.4 handle model is the first useful instance of the same dependent
linear machinery:

```text
Socket kind ρ state
state ::= Fresh | Bound | Listening | Connected peer | HalfClosed | Closed

listen : Socket TCP ρ Bound
         ⊸ Comp <IO, State ρ, Throw NetError>
                 (Socket TCP ρ Listening)

close  : Socket k ρ s
         ⊸ Comp <IO, State ρ> Unit
         where s != Closed
```

`close` consumes rather than mutates an alias. Borrowed read/write operations
return the owner to its original typestate after the loan. A state transition
consumes the old endpoint and returns a new indexed endpoint. Dynamic code gets
a sealed handle and runtime state guard; statically known double-close and
use-after-close are errors.

General session types make the `state` index a protocol. For example:

```text
Echo = μ p. &{
  Data  : ?(n : Nat). ?(payload : Bytes n). !Ack n. p,
  Close : End
}

recv-label : Chan ρ Echo
             ⊸ Comp <IO, Async, Throw NetError>
                     (Σ label. Branch label Echo)
```

This example composes all axes rather than invoking four checkers: `Chan` is
linear, `ρ` controls its lifetime, `Echo` is a dependent index, `Bytes n` and
`Ack n` use refinements/dependency, and receive has an effect row. Each send,
receive, or choice consumes `Chan ρ P` and returns `Chan ρ P'`. `End` must be
closed; cancellation consumes an endpoint only through a declared `Abort`
transition.

Duality is checked when two local endpoints are paired or a typed service
interface is imported. A remote peer is untrusted: codecs and a boundary
protocol monitor validate its labels, lengths, and refinements. Session types
prove the local program follows its protocol; they do not prove an arbitrary
network peer behaves honestly.

Public session subtyping is invariant in its first release. Output covariance,
input contravariance, and asynchronous queue subtyping can be introduced only
after their metatheory and monitor behavior agree.

## Quantum types are a specialization, not an escape hatch

Quantum values use the same region, dependent, effect, and usage machinery:

```text
qalloc  : Π (ρ : QRegion) (n : Nat).
          Comp <Quantum ρ, Alloc ρ> (QReg ρ n)

h       : Qubit ρ ⊸ Comp <Quantum ρ> (Qubit ρ)
cnot    : Qubit ρ ⊗ Qubit ρ
          ⊸ Comp <Quantum ρ> (Qubit ρ ⊗ Qubit ρ)
measure : Qubit ρ ⊸ Comp <Quantum ρ> Bit
split   : QReg ρ (m + n) ⊸ (QReg ρ m ⊗ QReg ρ n)
```

`QRegion ρ` is generative like an OALR region but owns a quantum runtime/device
context and has `Quantum ρ` effects. A quantum region may close only after all
qubits have been measured, reset-and-released, or otherwise consumed by a
declared physical operation. `QReg ρ n` is a linear dependent family; shape
arithmetic uses the same index solver as tensors.

There is no `Clone Qubit`, no proof-erased container that can hide a qubit, and
no unsafe primitive that duplicates one. A quantum value may cross dynamic
code only through the scoped linear contract above; otherwise the boundary is
rejected. Simulator internals may copy a state vector as an implementation
detail, but the source-level logical qubit capability remains unique.

This directly realizes the roadmap's qubit, `qreg<n>`, and quantum-region goals
([`ROADMAP.md:482-491`](../../../ROADMAP.md#L482-L491)) without creating a
second “quantum borrow checker.”

## Compiler representation and pipeline

The architectural boundary is a new typed HIR. The existing C AST remains the
parse/macro/compatibility representation.

```text
R7RS source + annotations
          │
          ▼
surface AST / hott_type_expr_t
          │ resolve binders and names
          ▼
interned core terms + schemes + primitive signatures
          │ one bidirectional, quantitative elaborator
          ▼
typed HIR
  (types, effects, FlowEnv transitions, evidence, cleanup, promotion)
          │ verify: no hard obligations remain
          ▼
erasure / specialization / monomorphization
          │
          ▼
LLVM 21 IR, VM bytecode, or JIT IR
```

### Core identities

The implementation separates nominal constructors from full terms:

```text
NominalTypeId   -- stable identity of Integer, Tensor, user record, Socket, ...
TypeRef         -- interned structural term, including every argument/binder
IndexRef        -- interned total term used in types and predicates
PredicateRef    -- canonical proposition
EffectRowRef    -- canonical row with optional tail metavariable
RegionVar       -- generative region identity
LoanVar         -- inferred borrow interval
PlaceId         -- resolved storage location, independent of source spelling
TypeScheme      -- quantified type/index/region/effect/mode variables
```

`TypeRef` equality is structural/hash-consed after alpha-normalization. Binder
identity uses de Bruijn levels or stable binder IDs, not strings. Runtime
representation is computed by erasure and stored separately. Caches key on the
complete term and an explicit solver-context fingerprint; no cache key is just
a 16-bit nominal ID when indices, effects, or regions matter.

The AST's packed `inferred_hott_type` remains populated with the erased
nominal head for old tools and ABI consumers. A compilation-owned side table,
keyed by stable AST/HIR node ID, stores:

```text
TypedExprInfo {
  TypeRef type;
  EffectRowRef effects;
  PredicateRef facts;
  FlowStateId ownership_after;
  EvidenceList evidence;
}
```

No pointer to an arena AST node is interned in a type. Typed interface files
serialize normalized schemes, effect rows, region quantifiers, refinement
solver version, and a content hash. They never serialize process-local
`TypeRef` integers.

### Flow environment

`Context`, `LinearContext`, and `BorrowChecker` become one persistent
`FlowEnv`. Each `PlaceId` entry contains its `TypeRef`, usage mode, owner,
allocation region, typestate/protocol index, active loans, and source origin.
Persistent snapshots make branches cheap and diagnostics reproducible.

The join operation is explicit:

- unrestricted values join by ordinary type/refinement rules;
- an affine/linear place must be live in compatible state on all incoming
  edges or consumed on all incoming edges;
- effects are row union;
- facts true on only one edge are dropped unless represented by a sum/existential;
- active loans must agree or end before the join.

Pattern matching refines both `A` and `Φ` per arm while splitting `Δ` according
to the pattern. Destructuring a linear product moves its components; it does
not implicitly copy them.

### Builtin and FFI signatures

Every builtin is declared once in machine-readable form. Representative
signatures are:

```text
vector-ref : Π (A : Type) (n : Nat).
             (Borrow shared ℓ ρ (Vector A n), {i : Int | 0 <= i < n})
             -> Comp <> A

set!       : Π (A : Type).
             (Borrow mut ℓ ρ A, A) -> Comp <State ρ> Unit

display    : Dyn[unrestricted] -> Comp <IO> Unit

call/cc    : mode-polymorphic signature whose continuation mode is derived
             from the captured FlowEnv
```

The registry removes ad hoc return-type cases, gives effect inference complete
coverage, and makes backend parity testable. Lowering may still special-case
intrinsics, but typing may not.

## Enforcement and erasure policy

After elaboration, HIR verification checks:

- all type, universe, and mode metavariables required by codegen are solved;
- no local region or loan appears free in an escaping type or closure;
- each strict-linear place has exactly one consuming path;
- each affine resource has one consuming or generated cleanup path;
- every session transition is legal and every endpoint reaches `End`/`Abort`;
- each effect is declared, propagated, or handled;
- every refinement obligation is proved or represented by a checked coercion;
- no proof term contains runtime-owned payload;
- no `Dyn[unrestricted]` box contains an affine/linear capability.

Then erasure removes universes, proofs, refinements already proved, region and
loan names, and static effect rows. Ownership and borrows are zero-cost in
fully typed code except for real cleanup/promotion operations. Dynamic casts,
unproved refinement checks, boundary protocol monitors, and unique capability
tokens remain. Existing object-header flags remain enabled in debug and at
untyped/FFI boundaries as defense in depth.

## Release trajectory

The existing roadmap assigns v1.4 linear network handles
([`ROADMAP.md:386-396`](../../../ROADMAP.md#L386-L396)), groups dependent,
refinement, effects, and sessions in v1.9
([`ROADMAP.md:468-478`](../../../ROADMAP.md#L468-L478)), and makes v2.0 depend
on linear dependent types for quantum work
([`ROADMAP.md:228-235`](../../../ROADMAP.md#L228-L235)). This ADR keeps those
product promises but moves the common semantic spine earlier so v1.4 does not
ship a disposable resource-only checker.

The current mention of v1.3.2 is only a superseded AD staging note
([`ROADMAP.md:379-382`](../../../ROADMAP.md#L379-L382)). For this ADR,
“v1.3.2” names the source-compatible type-kernel tranche. If release management
does not reuse that number, the identical gate must land in the next v1.3.x
before the v1.4 branch.

### v1.3.2: the semantic spine

v1.3.2 is intentionally an internal architecture release, feature-gated where
necessary and compatible with unannotated R7RS.

It ships:

1. `NominalTypeId`, interned `TypeRef`, `IndexRef`, `PredicateRef`,
   `EffectRowRef`, `TypeScheme`, and normalized binder representation.
2. Typed HIR and the full-expression side table; the packed AST type remains an
   erased compatibility hint.
3. Real bidirectional checking with metavariables, distinct equality/subtyping/
   consistency/entailment, and explicit gradual coercions with blame.
4. `Dyn` separated from `Any`/`Value`; typed/untyped higher-order boundaries
   and sealed unique dynamic capabilities.
5. `FlowEnv` keyed by `PlaceId`, including branch joins, loop invariants,
   closure capture modes, and hard move/borrow/lifetime errors.
6. One index term language and actual Pi/Sigma representation. Tensor/vector
   shape checking uses symbolic results and runtime contracts when decidable
   facts are dynamic.
7. Effect-row inference infrastructure and a signature registry seeded for
   core arithmetic, allocation, mutation, exceptions, continuations, and I/O.
   Rows may remain mostly inferred/internal; general handler syntax is deferred.
8. Elaboration of existing OALR forms into ownership HIR behind the new checker,
   while retaining current runtime lowering.

It explicitly defers external SMT, public general effect handlers, public
session syntax, and quantum types.

Exit gates:

- no new feature stores semantic type structure solely in a `TypeId` flag or
  side cache;
- a linear/affine safety violation cannot be converted to a warning by gradual
  mode;
- removing an ordinary annotation produces an explicit cast in HIR;
- branch, loop, exception, and one-shot continuation tests exercise the same
  `FlowEnv` implementation;
- native, VM, and JIT primitive signatures are generated or validated against
  one registry;
- the unannotated R7RS regression suite is behaviorally unchanged.

### v1.4: the resource-sound systems profile

v1.4 is not “a linear flag for sockets.” It is the first production profile of
the unified system.

It ships:

1. Generative `Region`, `Cap ρ`, `Own`, `Borrow`, `Shared`, `Weak`, outlives
   solving, non-lexical loans, and typed promotion/rehome.
2. Deterministic drop/close HIR on normal return, exception, cancellation, and
   early exit; cleanup occurs before region bulk destruction.
3. Affine typed-state handles for files, sockets, TLS contexts, event-loop
   registrations, threads, mutex guards, and buffers. Operations consume and
   return states; double-close and use-after-close are static errors when typed.
4. Production effects `<IO, Alloc ρ, State ρ, Throw E, Async, Unsafe>` in
   signatures and inference. Unknown calls taint callers rather than appearing
   pure.
5. The first decidable refinements for buffer length, vector/tensor shape,
   integer bounds, port openness, and socket preconditions, with runtime
   contracts at dynamic boundaries.
6. FFI ownership/effect annotations and sealed dynamic resource wrappers.
7. Concurrency rules for splitting `Δ`, task outlives constraints, and loans
   across suspension.
8. The internal session kernel used for handle typestate. General recursive
   protocol syntax and subtyping remain a later language feature.

The compiler roadmap already requires socket handles to close on normal and
exceptional exits
([`docs/COMPILER_ROADMAP.md:403-409`](../../COMPILER_ROADMAP.md#L403-L409))
and calls for owning and borrowed forms
([`docs/COMPILER_ROADMAP.md:464-472`](../../COMPILER_ROADMAP.md#L464-L472)).
The v1.4 exit gate strengthens “obvious” checking: all typed CFG paths are
checked statically; genuinely dynamic paths use stateful guards with blame.

Exit gates:

- every resource acquisition has a statically verified consume/cleanup path;
- no loan or region-local owner crosses an invalid return, store, closure, or
  async boundary;
- a 10K-connection server can use resource types without per-operation dynamic
  ownership checks in fully typed modules;
- forced exceptions and cancellation prove and test exactly-once cleanup;
- typed-to-typed socket operations are monitor-free; untyped and remote
  boundaries retain guards.

### v1.5 through v1.9: expose the general proof features

This interval generalizes the shipped spine rather than building another
checker. It adds the complete refinement surface plus budgeted SMT, user effect
rows and one-shot-aware algebraic handlers, row polymorphism, higher-rank
schemes, and public recursive/dependent session protocols. The v1.9 “types”
release is the language-complete preview of these features, as already listed
in the compiler roadmap
([`docs/COMPILER_ROADMAP.md:634-649`](../../COMPILER_ROADMAP.md#L634-L649)).

Its gate is integration: dependent protocols, refinement contracts, handler
resumptions, regions, and dynamic modules must all elaborate through the same
core judgment and typed HIR. None may introduce a parallel context or solver
that can accept a program rejected by the ownership kernel.

### v2.0: quantitative dependent types proven on quantum resources

v2.0 is the stability and research-frontier release, not the point at which the
core is rewritten.

It ships:

1. Stable dependent, refinement, effect, region, and session interfaces with
   serialized normalized schemes.
2. `QRegion`, `Qubit`, `QReg ρ n`, gate transitions, measurement, reset/release,
   and sealed dynamic quantum capabilities.
3. Quantitative combinators for splitting/merging registers and proving index
   arithmetic, with no `Clone` or implicit `Drop` for logical qubits.
4. `Quantum ρ` effects integrated with ordinary control, AD parameter-shift
   operations, exceptions, and one-shot handlers.
5. A small documented kernel judgment export suitable for Lean checking; the
   export contains normalized types, usage derivations, effects, and discharged
   proof obligations rather than trusting a pretty-printed source type.
6. Mechanized or machine-checked metatheory for the core fragment needed by
   quantum code: substitution, preservation, region non-escape, linear usage,
   and handler/continuation usage.

This matches the compiler roadmap's expectation that v2.0 qubits reuse the
linear dependent substrate
([`docs/COMPILER_ROADMAP.md:653-662`](../../COMPILER_ROADMAP.md#L653-L662)).

Exit gates:

- cloning, dropping, proof-erasing, serializing, or multi-shot-capturing a
  qubit is rejected through every typed and gradual path;
- register split/merge and gate APIs preserve dependent sizes;
- a quantum region cannot close with live logical qubits;
- effect handlers cannot resume a continuation containing quantum state more
  than once;
- Lean export/recheck agrees with the compiler on the normative core corpus;
- the quantum simulator and hardware backends share the same source typing.

## Migration sequence

The implementation order is constrained; later work must not create temporary
public semantics that conflict with the core.

1. Introduce interned core terms and adapters from existing `hott_type_expr_t`
   and builtin `TypeId`s. Keep all old lowering behavior.
2. Add stable node/place identities and typed HIR side tables.
3. Move primitive signatures from application-name conditionals into the
   registry, beginning with arithmetic, collections, mutation, I/O, and OALR.
4. Implement metavariable-based bidirectional checking and explicit `Dyn`
   coercions; stop using `Value` as an inference hole.
5. Replace the three ownership trackers with persistent `FlowEnv`; wire every
   control-flow form before making resource enforcement default.
6. Replace both CT value forms with `IndexTerm`; make tensors/vectors the first
   dependent families.
7. Add region/loan solving and explicit promotion/cleanup HIR; lower it to the
   existing OALR runtime and object-header defenses.
8. Turn on v1.4 resource signatures and FFI contracts.
9. Add refinement SMT, general effects/handlers, and session syntax on the same
   term and solver infrastructure.
10. Add quantum constructors and operations only after the dynamic-linear,
    continuation, exception, and region gates pass.

During migration, legacy nominal checking may run in comparison mode for
diagnostics, but its result is never combined with the new result by “accept if
either accepts.” Typed HIR verification is the sole authorization to lower a
typed feature.

## Verification strategy

Tests are organized by interactions rather than feature directories alone.

### Kernel and property tests

- alpha-equivalence, substitution, normalization, and hash-cons stability;
- separation of equality, subtyping, consistency, and entailment;
- effect row union/inclusion/generalization;
- usage algebra and `FlowEnv` join laws;
- region outlives and loan non-escape;
- refinement implication and deterministic solver timeout behavior;
- session duality and protocol advancement;
- erasure preserves runtime representation for legacy values.

### Adversarial composition tests

- a linear value through `if`, loops, pattern matches, exceptions, `call/cc`,
  an algebraic handler, an async task, and a dynamic module;
- a refinement invalidated by `set!` or mutable borrow;
- dependent tensor shapes crossing typed/untyped boundaries;
- nested regions with shared graph promotion, closures, weak references, and
  failed FFI calls;
- socket/session cleanup on normal, throw, cancellation, and peer violation;
- qubits hidden in products, existentials, closures, handlers, and `Dyn`;
- attempts to eliminate resource-bearing evidence through `Prop` erasure.

### Backend parity

The same typed HIR corpus runs through LLVM AOT, JIT, and VM backends. Tests
compare observable values, blame sites, cleanup traces, protocol transitions,
and effect-handler behavior. Sanitizers and runtime header checks remain on for
negative dynamic/FFI cases.

## Consequences

### Positive

- v1.4 resource safety is reusable substrate for sessions and quantum code,
  not a throwaway socket checker.
- Gradual R7RS interoperability remains possible without weakening ownership,
  purity, or proof claims.
- Regions become visible to types and optimization while retaining OALR's
  deterministic runtime behavior.
- Dependent shapes, refinements, typestate, protocols, and quantum register
  sizes share one term language and obligation pipeline.
- Effects make exceptional cleanup, continuations, handlers, async, and
  quantum state explicit at exactly the points where linearity needs them.
- Proof erasure and runtime erasure have a clear, auditable boundary.

### Costs and risks

- Typed HIR and structural interning are substantial frontend work before new
  syntax becomes visible.
- Flow-sensitive ownership plus refinements can increase compile time. The
  mitigation is persistent environments, canonical interning, stratified
  solvers, per-obligation budgets, and interface caching—not semantic shortcuts.
- Higher-order gradual contracts and sealed resource wrappers add allocation
  at typed/untyped boundaries. Fully typed calls remain zero-cost.
- External SMT expands the trusted computing base until certificate checking
  exists.
- Some programs that currently compile with warnings will become errors when
  they claim ownership, lifetime, purity, session, or quantum guarantees. They
  can remain untyped or cross an explicit checked/unsafe boundary, but cannot
  retain a false static guarantee.
- Full HoTT features such as univalence are deferred. “HoTT-inspired” will no
  longer be used to imply capabilities absent from the kernel.

## Rejected alternatives

### Add one checker per roadmap feature

Rejected because independent counters, effect scans, SMT checks, and protocol
automata disagree at branches, continuations, dynamic boundaries, and erasure.
The current detached dependent and ownership utilities demonstrate this
failure mode.

### Treat `Value` as gradual unknown forever

Rejected because top typing, missing information, runtime tagged
representation, and dynamic evidence have different rules. In particular,
top/unknown compatibility cannot safely decide linearity or purity.

### Model OALR solely with lexical lifetime inference

Rejected because allocation region, borrow interval, owner transfer, shared
aliases, external cleanup, and deep promotion are distinct. Both `Region ρ`
and `Loan ℓ` are required.

### Make all unique resources affine

Rejected because affine ownership is correct for deterministically droppable
memory and handles, but session completion and quantum no-cloning/no-discard
need exact linear obligations. The system needs both modes.

### Make all resources exactly linear

Rejected because it makes ordinary region data and exception-safe handles
unnecessarily brittle. Generated, explicit HIR drop glue gives deterministic
cleanup without forcing every source path to spell a destructor.

### Use runtime object flags as the ownership semantics

Rejected because flags cannot prove path coverage, region non-escape, closure
capture, async outlives, or legal protocol transition. They remain valuable
defense in depth.

### Let `unsafe` disable linear and borrow checking

Rejected because it would make no-cloning and region safety optional exactly
where FFI and low-level code are most dangerous. `Unsafe` is an effect with
trusted contracts, not absence of typing.

### Implement full HoTT before systems types

Rejected because univalence and higher inductives are not prerequisites for
regions, dependent shapes, refinements, effects, sessions, or quantum
no-cloning. A small predicative dependent kernel with intensional equality can
ship, be tested, and be exported for proof checking sooner.
