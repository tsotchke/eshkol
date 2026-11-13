---
kind: design
status: current
owner-area: types
since: v1.4.0-connection
sources:
  - lib/types/type_checker.cpp
  - lib/types/hott_types.cpp
  - lib/types/linear_check_bridge.cpp
  - lib/backend/eshkol_vm.c
  - lib/backend/vm_compiler.c
  - lib/frontend/parser.cpp
  - scripts/run_stdlib_aot.cmake
---
# Type-system integration, 2026-11-08

This note records how the type-system work of the v1.9.0-types track sits
together on the v1.4.x integration branch: dependent type enforcement
(ADR-0042), HoTT identity types (ADR-0043), refinement types (ADR-0044),
algebraic effects with effect rows (ADR-0045), structural types (ADR-0046),
quantum register types (ADR-0048) and `??` synthesis holes (ADR-0052). Where
two designs needed the same hook, the hook now has one dispatch that hands
each form to the system that owns it. Each decision is listed with the files
it touches.

## 1. One checking entry point

`TypeChecker::checkTask` (Γ ⊢ e ⇐ τ) is the only checking judgment:

1. an unbound `??` takes the expected type and is reported to the hole
   observer (ADR-0052);
2. every other expression is synthesized, and the tail rule of ADR-0042
   judges each expression whose value becomes the value of `e` against τ;
3. a lambda is additionally checked against an expected function type by
   `checkLambdaAgainst()`.

The separate lambda checking rule that only re-ran synthesis is gone; step 3
replaces it. The checking passes of ADR-0042 synthesize a body a second time
with AST writes suppressed. A hole is reported by the synthesis pass that
types the program and never by those passes, so a hole gets exactly one
report, as before (`TypeChecker::reportHole`).

The structural layer (ADR-0046) keeps its own bidirectional judgment for
rank-n polymorphism, rows and session protocols over its own type terms. It
is reached from the same public entry, `TypeChecker::synthesize`, after the
core judgment, and judges only what the `TypeId` layer cannot represent.

## 2. One signature dispatch for a function define

A function define resolves its signature once (`resolveSignature`) and closes
it through one choice:

- a signature that mentions a quantum family (`qreg<n>`, `qidx<n>`, a quantum
  region) is bound under its registered quantum identity (ADR-0048), so calls
  unify register dimensions;
- every other signature is closed as a telescope (`closeSignature`,
  ADR-0042), which is the plain nominal function type when nothing in it is
  dependent.

A quantum signature is a registered function type, never a term-backed one,
so at most one of the two application rules applies to a call: the quantum
rule (`checkApplication`) or the dependent rule (`applyDependent`). A
declared result is judged by the quantum dimension check first, then by the
tail rule. A `(: name type)` declaration is checked against whichever type
the define closed to.

## 3. One type-id range map

| Range | Owner |
|---|---|
| 300-499 | pair types |
| 500-999 | function types, including dependent quantum signatures |
| 1000+ | user types |
| 40000-57999 | term-backed dependent types (ADR-0042) |
| 58000-59999 | indexed family instances, `qreg<e>` / `qidx<e>` (ADR-0048) |
| 60000+ | sum types |

Both allocators previously claimed 58000-59999. The term store now stops at
57999 and falls back to the erased type when its range is full, exactly as
it did at the old limit. The type-expression kinds are appended in merge
order (`HOTT_TYPE_REFINE`, then `HOTT_TYPE_SIGMA`, `HOTT_TYPE_APP`,
`HOTT_TYPE_INDEX_LIT`, `HOTT_TYPE_INDEX_OP`); no code depends on their
numeric values.

## 4. One front-end type gate in the VM

The bytecode VM refuses the programs the native engine refuses, through
hooks the drivers install. All three source-level judgments now share one
hook type (`EshkolSourceCheckFn`) and one gate,
`vm_reject_type_violations()`, called by both VM routes (run and emit) in the
native order: dependent obligations, then linearity, then refinements. Each
judgment is consulted only when the source writes a form it governs. The
structural layer's violations are proofs of ill-typing like a clone, so they
travel with the linear verdict and the gate also consults it for sources that
use structural types.

On the front-end side, `lib/types/linear_check_bridge.cpp` reaches every
verdict through one helper, `checkSource(source, name, verdict)`, which runs
the parser, the macro expander and the type checker exactly as the native
engine does and hands the checker to a per-judgment verdict. The Windows stub
accepts every hook, including the refinement one.

## 5. One parse dispatch for type forms and ascriptions

- `parseTypeExpression` reads `(qreg e)` and `(qidx e)` as indexed quantum
  families before the general rule; every other head is a type-family
  application `(F arg ...)` read as a trial, which keeps the historical bare
  head name when the arguments are not type-language phrases.
- In the VM compiler, `define-type` leaves the unspecified value, `(: name
  type)` is the unspecified value and `(the T e)` compiles to `e`; both
  special forms yield to a user binding of the same name. `hott` and
  `hott-eval` go to the cubical kernel hook.
- `--types=strict` and `--strict-types` set the same flag; effect-row
  declarations (`declare-effects`) and type errors are fatal under it.

## 6. Certificates on the merged transcript

`core.certificate` (ADR-0047) builds on `core.transcript` as merged: the
UTF-8 FNV-1a digest, the canonical claim encoding and the document layout are
unchanged. The certificate work adds `tv-float`, the `transcript-open-as` /
`transcript-open-path` handles, `transcript-record-claim!` and
`transcript-seal-claim`, and a per-handle schema. A four-field transcript
claim hashes to the same text as before; `replay_check.py` hashes every field
before `"hash"` as UTF-8 text and reads both document schemas.

## 7. One verdict contract for library AOT suites

`scripts/run_stdlib_aot.cmake` had three verdict configurations on the
merged histories: a `PASS`-only script, a test that expected `ESHKOL_VERDICT`
and a local standard-library check, and a third variable,
`ESHKOL_EXPECTED_VERDICT`, from the signed-curvature work. They are now one
contract, documented in `docs/TESTING.md` ("Library AOT suites: one verdict
contract"): `ESHKOL_VERDICT` is `PASS` (default) or `ALL PASS` and nothing
else, a native compile requires the build tree's own `stdlib.o` and
`stdlib.bc`, and the VM route and the compile and run timeouts are kept.
Every registration keeps the verdict its suite prints: the DBSP acceptance
programs and `signed_curvature_aot` select `ALL PASS`, every other suite the
default.

## 8. Effect handlers and coroutines share one dynamic-state record

Coroutines (ADR-0030) keep the dynamic state of each computation in one
record, `eshkol_exec_context_t`, saved and loaded on every switch. The
algebraic-effect handler context (ADR-0045) is dynamic state of the same
kind, so it is a field of that record (`effect_context`), saved and loaded
by `lib/core/runtime_continuations.cpp` together with the dynamic-wind chain
it lives beside. A coroutine that suspends inside `handle` resumes with that
handler installed; the context it switches to runs with its own handlers; a
new coroutine starts with none, because its resumer's handlers are not on its
stack and a resumption cannot span two stacks. An operation with no handler
on the coroutine's stack raises `unhandled-effect-error?`, which reaches the
resumer as any raise in a body does. `tests/fibers/effects_coroutine_test.esk`
checks handlers inside and outside a coroutine across yield and resume, on
JIT and AOT, in the fibers gate.
