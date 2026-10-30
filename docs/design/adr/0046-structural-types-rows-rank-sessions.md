# ADR-0046: Structural types: row polymorphism, arbitrary rank and session types

- **Status:** Proposed — implemented for v1.9.0, awaiting the root's ruling (`inc/eshkol/types/structural_types.h`, `lib/types/structural_types.cpp`, `lib/types/structural_checker.cpp`, `lib/core/records.esk`, `lib/core/sessions.esk`)
- **Date:** 2026-10-30
- **Decision owners:** Eshkol type-system maintainers
- **Builds on:** ADR-0004 (one quantitative dependent type system), ADR-0013 (one gradual type relation)

## Context

The v1.9.0 roadmap lists row polymorphism for records, higher-rank types and
session types. ADR-0004 places all three in the one checking judgment: records
and effect rows are rows, a rank-N type is a polytype in argument position, and
a session protocol is an index of a linear endpoint advanced through the
ownership context `Δ`. ADR-0004 also asks for "hash-consed core terms with
metavariables" beneath the packed `TypeId` layer, and forbids any extension
from bringing its own context or solver that could accept a program the
ownership kernel rejects.

The packed `TypeId` layer has no type variables, no substitution and no rows;
`forall` is erased to its body. None of the three features can be stated in
it. The linear judgment (`TypeChecker::analyzeLinearUses`) already decides
exactly-once use over a well-defined fragment of the AST and is wired into the
native driver and, through `linear_check_bridge.cpp`, into the bytecode VM.

## Decision

### One structural term language and one solver

`eshkol::hott::st` defines the structural terms: metavariables, rigid
variables, bound variables, `any` (Dyn), nominal constructors, n-ary arrows
with a usage mode (`->` unrestricted, `-o` one-shot), `forall`, records over
rows, rows (`RowEmpty`, `RowExtend label field rest`), channel endpoints
`(chan P)`, protocols (`end`, `(! T P)`, `(? T P)`, `(choose (l P) ...)`,
`(offer (l P) ...)`, `(rec x P)`, `(dual P)`) and multiple values. Terms are
kinded: `Type`, `Row` (with a field kind) and `Session`.

One `Solver` decides equality. Every metavariable is solved in one place,
`bindMeta`, which enforces all side conditions together: kinds, the occurs
check, predicativity (a metavariable is never solved with a type containing
`forall`), level adjustment and skolem escape (a metavariable may not capture
a rigid variable introduced at a deeper level), row `lacks` sets, and the
usage restriction (a variable that ranges over unrestricted values may not be
solved with a linear type).

### Rows: Rémy/Gaster–Jones rows unified by rewriting

Rows are duplicate-free. Every row metavariable and rigid row variable carries
the set of labels it must lack. Unification rewrites the other row so a label
is at its head (Leijen, "Extensible records with scoped labels", without the
scoped duplicates), solving a row metavariable tail with a fresh field and a
fresh tail that lacks the label, and checks Leijen's side condition that the
rewrite did not solve the row's own tail. This gives principal types for the
row fragment: `record-ref` has the principal type
`(forall (a r) (-> (record (x a) r) a))` for label `x`, `record-extend` requires
the absence of its label, `record-remove` and `record-update` require its
presence.

Record operations take their label as a quoted symbol literal:
`(record 'x 1 'y 2)`, `(record-ref r 'x)`, `(record-extend r 'z v)`,
`(record-remove r 'x)`, `(record-update r 'x v)`. They are ordinary library
procedures of `core.records` (so untyped code can call them and the runtime
checks labels); the checker gives them their row typing rules when the
program requires the library.

**Structural subtyping is width subtyping at known closed record types.** The
subsumption judgment (`subsCheckRho`) accepts a record with more fields where a
closed record type is expected, field by field (covariant, since records are
immutable). Inference never invents a subtyping step: metavariables are solved
by unification, so principality is kept, and row polymorphism is what lets an
unannotated function accept wider records. Depth subtyping follows from the
field-wise check; there is no subtyping between a nominal `define-record-type`
type and a structural record.

**Nominal records stay nominal.** `define-record-type` creates a nominal type
whose procedures the gradual `TypeId` layer types; a structural record is built
with `record`. A nominal record converts to a structural one only by explicit
construction. Records may not hold linear values (a record is duplicable).

**Effect rows reuse these rows** (ADR-0045): an effect row is a `RowExtend`
chain whose labels are effect names and whose fields are payload types;
`Solver::rowInclusion` is "fewer effects is a subtype of more".

### Arbitrary rank: predicative bidirectional checking

The checker follows Peyton Jones, Vytiniotis, Weirich and Shields, "Practical
type inference for arbitrary-rank types" (JFP 2007): `inferRho` and `checkRho`
over rho-types, `checkSigma` by deep skolemisation, `subsCheck` with
contravariant arguments and deep instantiation of results, and generalisation
of top-level definitions only (local `let` bindings are monomorphic, as in
"Let should not be generalised", Vytiniotis et al. 2010). Polytypes enter only
through annotations: parameter annotations `(f : (forall (a) (-> a a)))`, return
annotations, `(: name T)` declarations, `let` binding annotations and `(the T e)`.
Free type variables of a top-level signature are quantified at its outermost
level and scope over the body. Skolem escape is detected by levels, which is
what makes the `runST` encapsulation hold: `(run-st (lambda (tok) (st-new tok 0)))`
is rejected because the reference type mentions the run's own state variable.

### Session types: protocols indexed on linear endpoints

A protocol is a `Session`-kinded term. `(chan P)` is a linear type. Each
operation of `core.sessions` consumes an endpoint and returns the endpoint for
the rest of the protocol:

| Operation | Type |
|---|---|
| `session-send` | `(forall (t p) (-> (chan (! t p)) t (chan p)))` |
| `session-recv` | `(forall (t p) (-> (chan (? t p)) (values t (chan p))))` |
| `session-select` | `(chan (choose (l p) . r))`, label literal -> `(chan p)` |
| `session-offer` | `(chan (offer (l1 p1) ... (ln pn)))`, handlers `(-o (chan pi) a)` -> `a` |
| `session-close` | `(-> (chan end) null)` |
| `make-session-pair` | `(forall (p) (-> (values (chan p) (chan (dual p)))))` |
| `session-fork` | `(forall (p) (-> (-o (chan p) any) (chan (dual p))))` |
| `tcp-session` | `(forall (p) (-> any (chan p)))` |

Choices are rows of protocols, so selecting from a choice is row-polymorphic in
the other branches and an offer must handle exactly the offered labels.
Recursive protocols are equi-recursive and are compared coinductively. Duality
is an involution on protocols that also reaches unsolved protocol variables
(`(dual p)` stays symbolic until `p` is known), and the two ends made by
`make-session-pair` or `session-fork` have dual protocols by construction.
Session subtyping is invariant, as ADR-0004 requires for its first release.

**Linearity reuses the one linear judgment.** Every binder whose type is
linear — an endpoint, a one-shot closure, or multiple values containing one —
is enforced by `TypeChecker::enforceLinearRegions` in *strict* mode. The judgment
now computes a usage interval `[lo, hi]` per binder (ADR-0004's usage
intervals): `hi` is the worst-case count the no-cloning rule always read, and
strict mode also requires `lo = 1` (consumed on every path) and treats a body
the analysis cannot decide as a violation. `let-values`, `values`, and named
`let` loops whose parameter is itself an enforced linear binder are in the
decidable fragment. Qubit verdicts are unchanged. Further rules of the
structural layer: a linear value may not be discarded (a non-final expression
of a sequence, a test, a one-armed branch), may not flow into code whose type
is unknown (`any`), may not be stored in a record, and may not be captured by a
closure unless the closure is one-shot (`-o`). Written and generalised type
variables range over unrestricted types only.

**Transports.** Local endpoints are a pair of bounded `core.channels` queues; a
forked peer runs on a `core.threads` worker. A TCP endpoint carries one
message per line in `write` syntax over a `core.net` handle. The remote peer is
untrusted (ADR-0004): `session-offer` checks every received label against the
labels it handles and reports a protocol violation otherwise. Session types
prove that the local program follows its protocol; they do not prove what a
remote peer sends.

### Where the structural layer runs

`TypeChecker::synthesize()` runs the structural layer on every top-level form
after the `TypeId` layer. A form is structural when an annotation in it uses a
structural former (`record`, `chan`, a protocol, `-o`, `forall`, `values`, a
user type constructor such as `(st-ref s a)`), when it mentions an operation of
a required structural library (`core.records`, `core.sessions`), or when it
mentions a definition that is structural; drivers call
`TypeChecker::beginProgram()` first so forward references resolve. Everything
else stays with the gradual `TypeId` layer, and its values are `any` here.
Forms outside the structural fragment (`do`, `match`, `guard`, `call/cc`,
quasiquote and others) are untyped islands whose value is `any`; an endpoint
used inside one is rejected because its use cannot be decided.

Violations are reported through `TypeChecker::reportStructuralViolation()`,
which is fatal in every mode except `--unsafe`, like a linearity violation.
The native driver refuses code generation, and the VM's linear bridge returns
linear plus structural violations, so the bytecode VM refuses the same
programs. `ESHKOL_DUMP_STRUCTURAL_TYPES=1` prints the scheme of every
structurally checked top-level definition.

## Consequences

- Records, rank-N polymorphism and session protocols are compile-time
  guarantees on both engines, with no runtime cost for typed code.
- The type-expression parser keeps named type applications
  (`HOTT_TYPE_APP`) instead of skipping them; the `TypeId` layer resolves an
  application's head as it resolved the bare name before.
- Generalised and written type variables are unrestricted, so a polymorphic
  function cannot be instantiated at an endpoint type; protocol code is
  written against `(chan P)` types. Quantification over linear types is
  planned for v2.0 with ADR-0004's quantitative binders.
- Delegation (sending an endpoint over an endpoint) is planned for v1.9.1.
  Validation of received payload types at the TCP boundary (beyond labels) is
  planned for v1.9.1.
- When the bidirectional core of ADR-0042 lands, its checking mode and the
  structural layer's `checkRho` are to be one rule set over one term language;
  this ADR's terms are the representation offered for that.

## Verification

- `tests/types/structural_types_test.cpp`: the solver contract (row rewriting,
  lacks sets, missing labels, equi-recursive protocols, duality of unsolved
  protocols, skolem escape, generalisation, the usage restriction,
  predicativity, row inclusion).
- `tests/types_ext/types_ext_gate.sh` (CTest `types_ext_gate`): positive
  fixtures run on JIT and AOT (and the VM where the program is VM-visible) with
  exact output and the reported principal types; every reject fixture must be
  refused by the native compiler with its diagnostic and no binary, and by the
  VM.
