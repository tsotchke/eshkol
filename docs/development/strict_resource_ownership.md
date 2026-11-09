# Native strict resource ownership

`eshkol-run --strict-resources` is the strict profile of the ownership
checker. It tracks every native handle family (TCP and UDP sockets, TLS
contexts, listeners and sessions, file ports, the event loop, threads,
mutexes and condition variables, subprocesses) through the typestate
`open -> borrowed -> open -> closed`. The language-level description is
[docs/reference/language/linear-handles.md](../reference/language/linear-handles.md);
the rules are [ADR-0056](../design/adr/0056-linear-handle-typestate.md).

The profile is available for native compilation and `--run`; it does not
change the default compiler path. The bytecode VM profile, `--eval`, WebAssembly
output, opaque `.o` inputs, and explicit `-l` libraries are refused with this
profile, because their initializers or runtime effects are outside the checked
source AST.

## Contracts and provenance

Every handle operation is a row in `inc/eshkol/strict_resource_contracts.h`.
Module rows (`core.net`, `core.tls`, `core.threads`, `agent.subprocess`) are
trusted only for the exact stock SDK source they were reviewed against: the
module's SHA-256 is pinned, and its provide list must equal the contract
names. `scripts/strict_resource_pins.py --check` (CTest
`strict_resource_contract_pins`) verifies the pins; after reviewing a module
change, `--update` refreshes them. Builtin rows (file ports, the event loop,
higher-order builtins) apply only when the name is not lexically rebound.

Factories and handle-consuming calls are accepted only through verified module
bindings or unshadowed builtins; a same-spelled local function, unknown
builtin, raw extern, or mutated alias does not gain a contract. Required module
initializers are checked along with user top-level forms, because the compiler
and JIT loader execute those forms.

## Analysis

The checker is a work-list abstract interpreter over the AST. Its state is
the typestate of every lexical binding plus the active loans; branches fork
and join it, loops check a back-edge invariant, and non-local exits (raise,
continuation calls, loop back edges) are collected by guard, dynamic-wind,
and call/cc frames. User functions are analyzed at each call site against
the caller's handle states, memoized by those states; recursion iterates to a
fixed point from "does not return". A function that refers to no handle,
contract, or handle-relevant function is classified without flow analysis.

Calls whose effects cannot be proved in the supported fragment are rejected
when they could affect resource safety. The diagnostic names the boundary
(`case`, `match`, multiple values, quasiquote, computed procedures, escaping
continuations or closures) rather than treating it as safe.

## Corpus

`tests/v1_4_resource_ownership/` holds compile-time positives and falsifiers
(`test_compiled.py`; each new falsifier pins its expected diagnostic) and JIT
and AOT runs (`test_runtime.py`). The loopback cases allocate a fresh local
TCP port for each JIT and AOT run; the file, thread and event-loop cases run
in a fresh working directory.
