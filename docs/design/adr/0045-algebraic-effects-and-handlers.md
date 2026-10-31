---
kind: explanation
status: current
owner-area: language
since: v1.9.0
sources:
  - inc/eshkol/frontend/effect_syntax_core.h
  - lib/frontend/parser.cpp
  - lib/backend/vm_parser.c
  - lib/core/control/effects.esk
  - lib/core/effects.esk
  - inc/eshkol/types/effect_check.h
  - lib/types/effect_check.cpp
  - scripts/run_effects_gate.py
---
# ADR-0045: Algebraic effects and handlers, with effect rows

**Status:** Accepted (v1.9.0-types; for root review, lands in rung order)
**Relates to:** ADR-0004 (effects are the `ε` axis of the one typed
judgment), ADR-0026 (one syntax engine shared by both engines), ADR-0003
(codegen/VM parity), the continuation design in
`docs/reference/language/continuations.md`.

## Context

The v1.9.0 roadmap lists effect types (`Pure`, `IO`, `State`, `Exception`) and
algebraic effects with handlers. ADR-0004 fixes their static home: a
computation type `Comp ε A` where `ε` is an ordered row of labels with an
optional open tail, `Pure` is the empty closed row, and an unknown effect is an
open tail, never `Pure`. The runtime already has re-entrant, multi-shot
`call/cc` on all three engines (native JIT, native AOT, bytecode VM) and
`dynamic-wind`, `guard` and `raise` that interoperate with it.

Prior art this design follows: Plotkin and Pretnar, *Handlers of Algebraic
Effects* (ESOP 2009) and *Handling Algebraic Effects* (LMCS 2013) for the
semantics and the handler typing rule; Bauer and Pretnar's Eff for deep and
shallow handlers as a language feature; Leijen's Koka (*Type Directed
Compilation of Row-Typed Algebraic Effects*, POPL 2017) for row-typed effects,
effect polymorphism and the treatment of tail-resumptive operations as plain
calls; OCaml 5 effects (Sivaramakrishnan et al., *Retrofitting Effect Handlers
onto OCaml*, PLDI 2021) for one-shot handlers whose operations do not unwind;
Filinski, *Representing Monads* (POPL 1994), for delimited control built from
`call/cc` and one mutable cell.

## Decision

### 1. Surface

`(define-effect Name (op formals...) ...)` declares an effect and one
first-class procedure per operation. `(handle expr clause...)` installs a deep
handler; `(handle-shallow expr clause...)` a shallow one. Clauses are
`(return (x) body...)`, `((op formals...) k body...)` and
`(fun (op formals...) body...)` (sugar for resuming with the body's value).
`(declare-effects f row)` bounds a procedure's latent row. Full reference:
`docs/reference/language/effects.md`.

### 2. One lowering for both engines

The forms are rewritten by `inc/eshkol/frontend/effect_syntax_core.h`, a
header-only C lowering over the neutral syntax tree of
`syntax_rules_core.h` (ADR-0026). The native parser converts its reader
syntax to that tree and parses the result in place of the form; the VM reader
lowers each complete top-level datum. There is no second copy of the rules, so
the engines cannot disagree about what a form means. The lowering emits only
symbols and lists (atoms are rebuilt from reader tokens by each engine), skips
quoted data, and leaves a `handle` whose operands are not clause-shaped as an
ordinary call.

### 3. Runtime: delimited control from `call/cc`

`core.control.effects` (in the stdlib closure, so the forms work without a
`require`) keeps the handler context as a stack of frames. The context is
thread dynamic state, read and written through `%effect-context` and
`%effect-context-set!`: native holds it in the runtime next to the
dynamic-wind chain, because a store into ordinary program storage from inside
`with-region` crosses the region store boundary (ADR-0020) and is copied, which
would separate a handler from the computation it handles; the VM, whose
regions are evacuated only at exit, holds it in one prelude global. Each frame has a
*delimiter*: the continuation that receives the result of the handled
computation — the `handle` form, or the call site of the resumption currently
running it. An operation finds the innermost frame with a clause, captures its
own continuation, transfers to the delimiter and runs the clause there with
the frame's outer handlers in scope; a resumption reinstalls the frames that
were between the operation and the handler (the handler itself unless
shallow) on top of the context at the resumption site, makes its own call site
the delimiter, and re-enters. This is Filinski's construction with one
delimiter cell per frame, made re-entrant by saving and restoring the cell
around each resumption, which gives multi-shot and stored resumptions.

### 4. Strategy by resumption use

The lowering classifies each clause by how its body uses `k` and picks the
cheapest strategy with the same meaning:

- **tail** (every tail position is `(k e)`, `k` nowhere else; every `fun`):
  the clause runs in place at the operation and `(k e)` becomes `e`. No
  continuation is captured. This is Koka's tail-resumptive optimization made a
  semantic rule: such an operation is a call.
- **abort** (`k` unused): unwind to the delimiter and run the clause there.
- **general-tail** (`k` only in tail calls): capture the operation's
  continuation; the resumption jumps into it without capturing its own site.
- **general**: capture both.

A handler whose clauses are all *tail* is *light*: it installs its frame with
`dynamic-wind` and captures nothing. The hot path of an in-place operation
allocates nothing (arguments travel in fixed slots up to arity 3).

### 5. Interaction rules

- `dynamic-wind`: in-place clauses do not leave the computation's extent;
  unwinding clauses run after-thunks before the clause and before-thunks on
  resumption; while a resumption runs a computation, the extents enclosing the
  resumption call but not the original `handle` are exited (a consequence of
  undelimited capture, stated as the rule).
- `guard`/`raise`: an exception raised by a handled computation goes to the
  context running it — the `handle` form, or the resumption site (a guard in
  the full handler forwards it there). Exceptions from in-place clauses
  propagate through the computation's guards; from unwinding clauses, from the
  `handle` form. Foreign exits (raise to an outer guard, escape continuations)
  and re-entries keep the handler stack consistent through `dynamic-wind`.
- Regions: captures follow the continuation rules — a capture inside
  `with-region` pins the open regions; light handlers capture nothing and pin
  nothing.
- AD: in-place operations are ordinary calls, transparent to forward and
  reverse mode; a resumed computation re-executes and records its own tape
  nodes per execution.
- Resumptions are multi-shot. ADR-0004's usage modes (a one-shot `Control
  linear` resumption may carry linear resources) are a static property; the
  runtime places no limit on the number of resumptions.

### 6. Effect rows in the checker

`lib/types/effect_check.cpp` infers a latent row for every top-level
procedure by a least fixpoint over the program: labels `IO`, `State`,
`Exception`, `Control` and every declared effect; tails `param:i` for the
procedure's i-th argument (effect polymorphism, instantiated with the actual
argument's row at each call) and `?` for calls the analysis cannot see.
Typing rules: an operation adds its effect; a deep handler with clauses for all
of an effect's operations discharges it from its body (Plotkin-Pretnar
handler rule); a catch-all `guard` discharges `Exception`; a resumption call
adds nothing (its effects are the handled body's). Builtins and library
procedures come from one signature registry (table of effectful and
higher-order builtins plus the standard library's naming rules). The pass runs
after the type checker and reports through `TypeChecker::reportTypeIssue`, so a
violated `declare-effects` is an error under `--strict-types` and a warning in
gradual mode — the same enforcement mode as the rest of the checker, which
ADR-0042's bidirectional core extends. The row representation (label set plus
tails with excluded labels) is local to this pass and is to be expressed in
the shared `RowType` of ADR-0046 when that lands.

## Consequences

- The three engines share one meaning for every effect form; the gate
  (`scripts/run_effects_gate.py`, CTest `v19_effects_gate`) compares exact
  transcripts of the semantics corpus on all three and runs the
  effect-typing corpus under `--strict-types`.
- Common effects (state, reader, writer, iteration) cost a few procedure
  calls (0.62 µs per operation against a 0.10 µs loop iteration, AOT x86-64)
  and allocate nothing. Effects that genuinely resume from elsewhere (pulling
  generators, nondeterminism, task switching) cost one or two stack captures
  per operation (hundreds of µs, retained as continuation images are).
- Non-local exits restore the recursion-depth counter: `guard`,
  `with-exception-handler` and `call/cc` save it in a frame slot before
  `setjmp` and restore it on the landing path, so a program that leaves deep
  code by `raise` or a continuation many times does not reach
  `ESHKOL_MAX_STACK` without being deep.
- `handle`, `handle-shallow`, `define-effect` and `declare-effects` become
  keywords (a clause-shape check keeps existing procedures named `handle`
  callable; user macros take precedence).
- Planned for v1.9.1: resumptions on a separate stack segment so that
  one-shot resumptions do not copy the stack (building on the coroutine
  substrate); usage-mode checking of resumptions in the ADR-0004 linear
  context; effect rows in function types written with `:` annotations.
