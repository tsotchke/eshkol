---
kind: project
status: current
owner-area: docs
since: v1.3.5
sources:
  - .icc/ledger/meta.yaml
  - tests/coverage/release_record.json
  - scripts/doc_audit/extract_examples.py
---
# Glossary

Terms the Eshkol documentation uses without defining them where they appear.
Each entry ends with the page that is the term's home. Terms are grouped by
first letter.

## A

**ADR (Architecture Decision Record).** A numbered design record under
`docs/design/adr/`, one number per record, with a status of Proposed,
Accepted (fully or partially implemented) or Superseded.
[ADR index](design/adr/README.md)

**AD (automatic differentiation).** Derivatives computed by the compiler from
the program itself, exactly where the arithmetic allows, through `derivative`,
`gradient`, `jacobian`, `hessian` and related operators.
[AD guide](guide/AUTOMATIC_DIFFERENTIATION.md)

**AD carrier.** The runtime representation that carries a value together with
its derivative information through a computation: a dual number, a jet, a Taylor
tower or a tape node. [AD architecture](reference/ad/architecture.md)

**Agent FFI.** The foreign-function layer that gives Eshkol programs HTTP,
subprocess, SQLite, cryptography and platform services under a capability
policy. [Agent reference](reference/agent/INDEX.md)

**AOT (ahead of time).** Compiling a program to a native binary or object with
`eshkol-run file.esk -o out`, as opposed to running it on the JIT.
[`eshkol-run`](reference/runtime/eshkol-run.md)

**Arena.** A region of memory from which objects are allocated by bumping a
pointer and released all at once. [Memory model](reference/runtime/memory-model.md)

**Ascription.** `(the type expr)`: a statement to the type checker that `expr`
has `type`. It generates no code, and is reported only when no value could have
both types. [Gradual typing](guide/GRADUAL_TYPING.md#8-telling-the-checker-what-you-know)

**Assurance gates.** The CI job that runs the build-free gates and their
self-tests on every pull request. [Testing](TESTING.md)

## B

**Birth location.** The source line and column every AST node receives when it
is created, inherited from the form it was generated from, so synthesised nodes
report a real position. [Architecture](ESHKOL_V1_ARCHITECTURE.md)

**BLC (binary lambda calculus).** A lambda-calculus encoding in which programs
are bit strings; `core.blc` implements it with a universal machine.
[Binary lambda calculus guide](guide/BINARY_LAMBDA_CALCULUS.md)

**Bignum.** An exact integer of unbounded size, the member of the numeric tower
an exact integer becomes when it no longer fits 64 bits.
[Numeric tower](reference/language/numeric-tower.md)

**Borrow.** Temporary, non-owning access to a value under OALR, checked by the
type checker's ownership analysis. [Type system](breakdown/TYPE_SYSTEM.md#ownership-and-linearity)

**Builtin.** A procedure provided by the compiler and runtime rather than a
library module; the count is taken from `tests/coverage/language_surface.json`.
[Language reference](reference/language/INDEX.md)

**Bytecode VM.** Eshkol's second execution engine, which runs compiled ESKB
bytecode; it is used by the browser REPL and embedded profiles and is checked
against the native engine for parity. [Bytecode VM](breakdown/BYTECODE_VM.md)

## C

**Capability policy.** The rules that decide which effects (network, files,
subprocesses) a program may use. [Capabilities](reference/language/capabilities.md)

**Capture.** A variable from an enclosing scope that a closure refers to. Under
differentiation a capture is resolved where the closure was created.
[AD capture rules](reference/ad/operators.md#where-a-capture-is-resolved)

**Certified enclosure.** An interval or Taylor model whose bounds are proved
with directed rounding rather than sampled. [Certified enclosures](reference/stdlib/certified-enclosures.md)

**Change class.** How CI classifies a pull request by what it can affect:
`docs`, `tests-only` or `full`. A change to a page under the example gate is
`tests-only`. [Testing](TESTING.md)

**Closure ABI.** The calling convention for a procedure value: its code pointer
plus its captured environment. A procedure binding that can change is always
called through it. [Architecture](ESHKOL_V1_ARCHITECTURE.md)

**Completion oracle.** The set of criteria in `.icc/completion-oracles.yaml`
that a release target must meet; each criterion is graded from evidence.
[Release process](platform/RELEASE_PROCESS.md)

**Consistency.** The gradual-typing relation under which the dynamic type
`Value` fits every type. [Gradual typing](guide/GRADUAL_TYPING.md#4-fitting-one-type-to-another)

**Consistent subtyping.** The rule the checker applies at arguments, returns and
annotated bindings: a static subtype in which every unknown component is
acceptable. [Type system](breakdown/TYPE_SYSTEM.md#the-type-relation)

**Construct.** One entry of the language surface: a special form, builtin or
syntax the coverage gate must see executed. [Test coverage](TEST_COVERAGE.md)

**Continuation.** The rest of a computation captured as a value by `call/cc`;
Eshkol's continuations are re-entrant and multi-shot. [Continuations](reference/language/continuations.md)

**Custom VJP.** A user-supplied vector-Jacobian product that tells reverse-mode
AD how to differentiate through an operation. [AD tape](reference/stdlib/ad_tape.md)

## D

**DBSP.** Incremental dataflow over Z-sets, in which a query updates its answer
from changes rather than recomputing. [DBSP](reference/stdlib/dbsp.md)

**Deep walk.** Evacuating an object together with everything it points to when
it leaves a region; heap subtypes are marked `[DEEPWALK]` or `[LEAF]`.
[Memory management](breakdown/MEMORY_MANAGEMENT.md)

**Differential testing.** Running the same program on two engines, or against a
reference Scheme, and comparing their output. [Testing](TESTING.md)

**Doc-example gate.** `scripts/doc_audit/check_doc_examples.py`, which runs
every example in a gated documentation scope on the JIT and AOT and compares
what it prints with the page. [Writing documentation](DOCUMENTATION.md#5-examples-are-executed)

**Dual number.** A value paired with one derivative, the carrier of first-order
forward-mode AD. [AD architecture](reference/ad/architecture.md)

## E

**Engine parity floor.** The recorded minimum share of language constructs with
differential evidence on both engines, stored as exact counts.
[VM parity](VM_PARITY.md)

**EREPL machine mode.** The REPL's line protocol for driving it from another
program. [`eshkol-repl`](reference/runtime/eshkol-repl.md)

**ESHKOL_PATH.** The environment variable naming the library directory the
compiler searches for modules and the standard library.
[Environment variables](reference/runtime/environment-variables.md)

**ESKM.** The validated file format for saved tensors and models.
[ESKM v1](reference/tensors/eskm-v1.md)

**Evacuation.** Copying the objects that escape a region into the enclosing
arena when the region ends. [Memory management](breakdown/MEMORY_MANAGEMENT.md)

**Evergreen page.** A tutorial, guide, reference or explanation page, which
states what is true now and carries no release narrative.
[Writing documentation](DOCUMENTATION.md#3-evergreen-wording)

**Evidence path.** A directory where a gate writes or reads its trace records
(`TRACE_DIR`, `ICC_TRACE_DIR`); a relative value is relative to the repository
root. [Release process](platform/RELEASE_PROCESS.md)

**Exactness.** Whether a number is exact (integer, rational) or inexact
(floating point); Eshkol preserves exactness through arithmetic and, where
possible, through differentiation. [Numeric tower](reference/language/numeric-tower.md)

## F

**Fail closed.** A gate that reports failure, never success, when its input is
missing or unreadable. [Testing](TESTING.md)

**Forward mode.** AD that propagates derivatives alongside values from inputs to
outputs. [AD architecture](reference/ad/architecture.md#forward-mode--the-4-component-jet)

**Front matter.** The YAML block at the top of a documentation page giving its
kind, status, owning area, first release and sources.
[Writing documentation](DOCUMENTATION.md#2-front-matter)

**Function type.** A procedure signature written `(-> argument ... result)`,
contravariant in its parameters and covariant in its result.
[Gradual typing](guide/GRADUAL_TYPING.md#5-function-types)

## G

**Gated scope.** A named set of documentation paths, listed in `GATED_SCOPES`,
whose examples the doc-example gate executes. [Writing documentation](DOCUMENTATION.md#5-examples-are-executed)

**Gradual typing.** Optional static types in which anything unannotated is
dynamic and a mismatch is a warning unless `--strict-types` is given.
[Gradual typing](guide/GRADUAL_TYPING.md)

## H

**Heap subtype.** The kind of a heap-allocated object (bignum, rational,
closure, tensor, ...), recorded in its header; all subtypes are defined in one
place. [Type system](breakdown/TYPE_SYSTEM.md)

**High-risk construct.** A language construct the parity gate weights
separately because a divergence there is most likely to change an answer.
[VM parity](VM_PARITY.md)

**HoTT.** Homotopy type theory, the foundation of the compile-time type system's
universe levels and dependent types. [Type system](breakdown/TYPE_SYSTEM.md#hott-compile-time-type-system)

## I

**i128.** Eshkol's native 128-bit wrapping integer, distinct from the exact
integers. [i128](reference/language/i128.md)

**ICC.** The repository's code-index and audit tool: it indexes the tree and
grades documentation claims, architecture invariants and release readiness.
[Contributing](../CONTRIBUTING.md)

**Import glue.** The JavaScript functions a WebAssembly build imports from its
host; Eshkol generates the flat-AD part from one source and checks it for
freshness. [Web platform](breakdown/WEB_PLATFORM.md)

**Inferred slot.** A variable whose type the checker infers from what flows into
it: a named-`let` parameter, a `do` variable, a recursive procedure's result.
[Type system](breakdown/TYPE_SYSTEM.md#inferred-slots-loop-parameters-and-recursive-results)

## J

**Jet.** A value with a truncated series of derivatives, the forward-mode
carrier for higher orders. [AD architecture](reference/ad/architecture.md)

**JIT.** Compiling and running a program in one step with `eshkol-run -r`, the
engine the REPL uses. [JIT internals](reference/runtime/jit-internals.md)

**Join.** The most specific type that each of several types fits, used for the
result of a branching form and for loop variables. [Gradual typing](guide/GRADUAL_TYPING.md#6-branches-join)

## K

**Known-defect marker.** `<!-- doc-example: known-defect ID: ... -->`: the page
states the designed behaviour, the build does not deliver it yet, and the ledger
entry `ID` is open. [Writing documentation](DOCUMENTATION.md#markers)

## L

**Language surface.** The complete list of constructs and builtins, generated
into `tests/coverage/language_surface.json`. [Test coverage](TEST_COVERAGE.md)

**Ledger.** The flaw ledger under `.icc/ledger/entries/`. Entry prefixes:
`SW-` (mostly silent-wrong results), `LE-` (loud errors), `DD-` (documentation
debt), `PR-` (parity ratchet), `IF-` (in flight), `VA-` (vacuous assurance).
Only open silent-wrong entries block a release. [Contributing](../CONTRIBUTING.md)

**Linear type.** A type whose values must be used exactly once, such as `Qubit`;
violations are compile-time errors. [Language guide](ESHKOL_LANGUAGE_GUIDE.md)

## M

**Metamorphic testing.** Checking that a transformation of a program which should
not change its answer does not. [Testing](TESTING.md)

**Multi-shot continuation.** A continuation that can be resumed more than once.
[Continuations](reference/language/continuations.md)

**musttail.** The LLVM marker that guarantees a tail call reuses its frame;
Eshkol uses it where the target allows and the tail-transfer dispatcher
elsewhere. [Tail calls](reference/language/tail-calls.md)

## N

**Named let.** `(let loop ((var init) ...) body)`, a local recursive procedure
used for loops. [Special forms](reference/language/special-forms.md)

**Narrowing.** The checker's refinement of a variable's type inside a branch
guarded by a type predicate. [Gradual typing](guide/GRADUAL_TYPING.md#8-telling-the-checker-what-you-know)

**Node identity.** The `eshkol_node_id_t` key that identifies an AST node across
the parser, the node allocator and semantic queries. [Architecture](ESHKOL_V1_ARCHITECTURE.md)

**Numeric tower.** Integer, bignum, rational, real and complex, with exactness
tracked across operations. [Numeric tower](reference/language/numeric-tower.md)

**Nursery.** The short-lived allocation space reclaimed on each iteration of a
resident loop. [Memory model](reference/runtime/memory-model.md)

## O

**OALR (ownership-aware lexical regions).** Eshkol's memory model: arenas tied
to lexical regions, with ownership and borrowing checked at compile time and
escaping objects evacuated. [Memory management](breakdown/MEMORY_MANAGEMENT.md)

## P

**Parity.** Agreement between the native engine and the bytecode VM on the same
program, tracked row by row in `tests/vm_parity/PARITY.tsv`. [VM parity](VM_PARITY.md)

**Perturbation level.** The tag that keeps nested differentiations apart so an
inner derivative's perturbation is not confused with an outer one.
[AD architecture](reference/ad/architecture.md)

**Pillar.** One of the families of adversarial tests (differential, edge
matrix, AD oracle, stress, VM parity, depth, external oracles, escape matrix)
that CI and the nightly runs organise around. [Pillar CI inventory](design/PILLAR_CI_INVENTORY.md)

## R

**Ratchet.** A recorded count that a gate lets move only in one direction, such
as the number of marked documentation examples. [Writing documentation](DOCUMENTATION.md)

**Rational.** An exact fraction such as `2/3`, kept exact through arithmetic.
[Exact arithmetic](breakdown/EXACT_ARITHMETIC.md)

**Readiness.** The graded verdict that a commit meets its completion oracle,
bound to that exact commit. [Release process](platform/RELEASE_PROCESS.md)

**Receipt.** A trace record a probe or test writes as evidence that it ran and
what it measured. [Release process](platform/RELEASE_PROCESS.md)

**Region.** A lexical scope whose allocations are released together when it
ends, written with `with-region`. [Memory model](reference/runtime/memory-model.md)

**Release record.** `tests/coverage/release_record.json`, the single source for
the release tag, previous tag, date, status and evidence totals.
[Release process](platform/RELEASE_PROCESS.md)

**Release-record span.** `<!-- release-record:KEY -->...<!-- /release-record -->`,
text in a document that must equal the record's rendering of `KEY`.
[Writing documentation](DOCUMENTATION.md#4-facts-come-from-sources)

**Reverse mode.** AD that records operations on a tape and propagates
derivatives from outputs back to inputs. [AD tape](reference/ad/tape.md)

## S

**Self-test.** A gate's `--self-test` mode, which proves the gate fails on
planted faults and passes on clean input. [Testing](TESTING.md)

**Silent-wrong.** A wrong value, derivative or memory outcome produced with no
diagnostic and a zero exit status; an open silent-wrong ledger entry blocks a
release. [Contributing](../CONTRIBUTING.md)

**Static callee binding.** The compiler's record that a name denotes a specific
compiled procedure, tied to the binding that owns it, so a direct call is used
only while the name cannot change. [ADR 0015](design/adr/0015-static-callee-binding-identity.md)

**stdlib.o.** The precompiled standard library object that the JIT and AOT
drivers load instead of recompiling the library. [JIT internals](reference/runtime/jit-internals.md)

**Strict types.** `--strict-types`: every type diagnostic is an error and no
code is generated. [`eshkol-run`](reference/runtime/eshkol-run.md)

## T

**Tagged value.** The runtime representation of every Eshkol value: a type tag,
flags and a payload. [Type system](breakdown/TYPE_SYSTEM.md#tagged-value-representation)

**Tail-transfer dispatcher.** The portable lowering of mutual tail calls on
targets where `musttail` cannot be used. [Tail calls](reference/language/tail-calls.md)

**Tape.** The record of operations that reverse-mode AD replays backwards.
[AD tape](reference/ad/tape.md)

**Taylor model.** A polynomial plus an interval remainder that encloses a
function over a whole domain. [Certified enclosures](reference/stdlib/certified-enclosures.md)

**Taylor tower.** The truncated Taylor-series carrier used for arbitrary-order
derivatives. [AD architecture](reference/ad/architecture.md)

**Tensor.** Eshkol's homogeneous numeric array, distinct from a Scheme vector,
which can hold any values. [Tensors](reference/tensors/INDEX.md)

**Type relation.** The single module, `TypeRelation`, that owns subtyping,
consistency, joins, casts and type printing. [ADR 0013](design/adr/0013-gradual-type-relation.md)

**Typed claim.** A checkable statement in a document (a count, a line figure, a
file path) that the audit tooling grades against the tree.
[Writing documentation](DOCUMENTATION.md#9-the-gates-and-how-to-run-them)

## V

**Value.** The checker's dynamic type: the type of anything it does not know,
consistent with every type. [Gradual typing](guide/GRADUAL_TYPING.md#4-fitting-one-type-to-another)

**VM.** See *Bytecode VM*.

## W

**WASM.** The WebAssembly build of Eshkol that runs in the browser REPL and the
website. [Web platform](breakdown/WEB_PLATFORM.md)

## Z

**Z-set.** A collection whose elements carry integer weights, so insertions and
deletions are both changes; the data type of DBSP. [DBSP](reference/stdlib/dbsp.md)
