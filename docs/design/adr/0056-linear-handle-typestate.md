---
kind: explanation
status: current
owner-area: types
since: v1.4.0-connection
sources:
  - inc/eshkol/strict_resource_contracts.h
  - exe/eshkol-run.cpp
  - lib/core/runtime_continuations.cpp
  - tests/v1_4_resource_ownership/test_compiled.py
---
# ADR-0056: Linear handle typestate for the strict resource profile

- **Status:** Proposed
- **Date:** 2026-11-08
- **Decision owners:** Eshkol type system and runtime
- **Applies to:** `eshkol-run --strict-resources` (the ownership checker in
  `exe/eshkol-run.cpp`), the handle contract table
  `inc/eshkol/strict_resource_contracts.h`, the borrow-scope builtins
  (`call-with-port`, `call-with-input-file`, `call-with-output-file`) in
  `lib/backend/llvm_codegen.cpp`, and the dynamic-wind stack in
  `lib/core/runtime_continuations.cpp`
- **Relates to:** ADR-0004 ("Resource typestate and session types" and "v1.4:
  the resource-sound systems profile"), the runtime handle registry
  (`inc/eshkol/core/handle.h`), and the local IPC ownership model
  (acquire, borrow, release, close; close on normal return, exception and
  continuation escape)

## Context

ADR-0004 fixes the model: a handle is an affine owner with a typestate index,
`close` consumes its owner, a borrowed operation returns the owner to its
state when the loan ends, and statically known double close and use after
close are errors. It does not fix how a checker follows that typestate through
Eshkol's control forms, which handle families it covers, or what the borrow
pattern means at run time.

Before this decision the opt-in strict profile covered TCP and UDP sockets
from `core.net` in straight-line code, `if` with two arms, `borrow`, and
`move`. Loops, user functions, closures, `cond`/`when`/`unless`, exceptions,
`dynamic-wind`, and `call/cc` were refused wholesale, and files, threads, the
event loop, TLS, and subprocesses had no contract. The runtime registry
(`handle.h`) already enforces `open -> borrowed -> open -> closed` dynamically.

## Decision

### 1. One contract table

Every operation that creates, consumes, or uses a native handle is described
once, in `inc/eshkol/strict_resource_contracts.h`:

| Role | Meaning |
|---|---|
| `FACTORY` | returns a fresh owned handle of one kind |
| `CLOSE` | consumes the handle at its handle argument |
| `OBSERVE` | borrows its handle arguments for the call only |
| `ERROR_PREDICATE` | refines a failure value of one family |
| `ERROR_OBSERVE` | reads a failure value on the failed branch |
| `LEND` | opens or takes a handle, lends it to a procedure, closes it on every exit |
| `SAFE` | no handle effect (it may take a procedure argument) |

Each row also names the handle kinds it accepts at each handle argument and
how it calls procedure arguments: exactly once (`ONCE`), zero or more times
(`MANY`), or on another thread (`THREAD`). Module rows are trusted only for
the exact SDK source they were reviewed against (a SHA-256 pin per module);
builtin rows apply only when the name is not lexically rebound. The checker
contains no operation names of its own.

Families covered: TCP listeners and streams, UDP sockets, TLS contexts,
listeners and sessions, input and output file ports, the event loop, threads,
mutexes, condition variables, and subprocess handles.

### 2. The typestate lattice

A tracked binding is in one of:

```text
ERROR_OR_LIVE  acquisition not yet refined (failure families only)
ERROR_ONLY     refined to the failure value; owns nothing
LIVE           open
CLOSED         consumed by its close operation
MOVED          ownership transferred (move, call-with-port, return)
MAYBE_LIVE     open on some paths only
```

A borrowed binding is `LIVE` with an active loan. A factory whose failure
raises yields `LIVE` directly. A factory whose failure is a value
(`network-error?`, `tls-error?`, or `#f`) yields `ERROR_OR_LIVE`, and only its
refiner (`(if h ...)`/`(not h)` for `#f`) turns it into `LIVE` or `ERROR_ONLY`.

Joins are pointwise. `LIVE` joined with `CLOSED` is `MAYBE_LIVE`, which is an
error at every use, at every close, and at scope exit. Unreachable paths (after
`exit`, `raise`, a loop back edge, or a continuation call) are the identity of
the join.

### 3. Obligations

- **Scope exit.** A binding that leaves its scope `LIVE`, `ERROR_OR_LIVE`, or
  `MAYBE_LIVE` is an unclosed handle.
- **Close.** Close requires `LIVE` (or `ERROR_ONLY`, which owns nothing). Close
  of `CLOSED` is a double close; of `MOVED` or `MAYBE_LIVE`, a use after close.
- **Use.** Every handle argument requires `LIVE` and an accepted kind. A handle
  used anywhere else (stored, copied, compared, passed to an operation with no
  contract) is rejected.
- **Loans.** A borrowed or lent binding cannot be closed, moved, returned,
  stored, or captured by a closure that may outlive the loan.

### 4. Control forms

- **Branches.** `if` (with one or two arms), `cond`, `when`, and `unless` fork
  the state and join it. `case`, `match`, `=>` clauses, and multiple values are
  refused with a diagnostic naming the form.
- **Loops.** A named-let or `do` loop starts from its entry state. A tail call
  to the loop name is a back edge: every handle in scope must be in its entry
  state there, and a handle acquired in the iteration must be closed. This is
  the fixed point, reached in one pass. A non-tail call to the loop name, or
  the loop name used as a value, is refused while handles are in scope.
- **Functions.** A user function is checked at each call site, against the
  caller's handle states (context-sensitive summaries, memoized). Its result
  updates the caller: a parameter the body closes is closed in the caller, a
  captured handle the body uses keeps its state, and a handle the body acquires
  and returns in tail position is a fresh owner for the caller's binding. A
  function that returns a handle on some paths only, or returns its own handle
  parameter, is refused. Recursion (including mutual recursion) iterates from
  "does not return" to a fixed point; a tail-recursive drain closes its
  parameter only in its base case and converges in two passes. A function
  that names no handle, contract, or handle-relevant function needs no flow
  analysis; one that calls an operation without a proved effect is refused
  with that operation as the reason.
- **Closures.** A lambda bound to a name is a function. A lambda passed to a
  `ONCE` or `MANY` procedure argument runs inline: `ONCE` applies its effects,
  `MANY` must leave every handle as it found it. A `THREAD` body may not
  capture any handle. Any other lambda that captures a handle, a loan, a
  continuation, or a loop is refused: the checker cannot see when it runs.
  A function that uses handles may not escape as a value.
- **Exceptions.** Every operation that may raise is a raise point. Inside a
  `guard` body, the handler starts from the join of the states at all raise
  points, and the guard's result joins the body's normal exit with each
  clause. A guard without a catch-all clause re-raises on its fall-through
  path. A handle bound inside the guard body that is open at a raise point is
  an unclosed handle (the exception leaves its scope); the remedy is a borrow
  scope or a `dynamic-wind` cleanup. An explicit `raise` or `error` with an
  open handle and no handler in scope is refused. A callee that raises while a
  caller's handler is active is held to the same rule for its own handles.
- **dynamic-wind.** The before thunk runs inline. Raise points and
  continuation escapes inside the body are collected. The after thunk is
  checked twice: from the normal exit, and from the join of every exceptional
  and escaping exit; the second result continues to the outer handler or the
  continuation's target. A body that closes a handle the after thunk also
  closes is a double close.
- **call/cc.** With a literal one-parameter lambda, the continuation is an
  escape-only continuation: calling it leaves with the current state (open
  handles bound inside the extent are unclosed handles), and the call/cc result
  joins the normal exit with every escape. A continuation used as a value or
  captured by a closure while handles are open is refused: the checker cannot
  follow a re-entry.

### 5. The borrow pattern

`(borrow h body ...)` lends `h` for its body. A `LEND` contract
(`call-with-input-file`, `call-with-output-file`, `call-with-port`) lends a
handle to its procedure argument: inside, the parameter is borrowed (no
close, no escape); `call-with-port` consumes its owner. `with-mutex` borrows a
mutex for its thunk. A named function may be the procedure; its summary must
leave the loan open.

The runtime matches the static rule: a `LEND` builtin pushes a native cleanup
frame on the dynamic-wind stack before calling its procedure. A raise or a
continuation escape that unwinds the frame closes the port; a normal return
pops the frame and closes it. The cleanup runs at most once.

### 6. Profiles

The default profile is unchanged: programs compile without the typestate
rules, and real OS resources keep their dynamic checks (the registry refuses
double close and use after close at run time, as a catchable condition).
`--strict-resources` is the strict profile. It is available to native
compilation and `-r`; the bytecode VM profile, `--eval`, WebAssembly output,
opaque `.o` inputs, and `-l` libraries are refused with an explicit diagnostic.

## Alternatives rejected

- **A second checker in the HoTT type checker.** ADR-0004 and the IPC design
  both require one checker. The ownership checker already owns `borrow`,
  `move`, and the module provenance rules.
- **Whole-program inlining of function bodies.** Context-sensitive summaries
  give the same precision with memoization, terminate on recursion, and name
  the function in every diagnostic.
- **Rejecting every implicit exception path at top level.** Any operation that
  may raise would then need a cleanup scope, which is the right end state but
  not a usable profile today. Raise points are tracked wherever a handler can
  observe them; an exception that no handler catches ends the program.

## Staging

- **v1.4.0-connection (this record):** the contract table and pins, every
  family above, the control-form rules, the borrow pattern, and the runtime
  cleanup frames, all under `--strict-resources`.
- **Planned for v1.4.1:** the strict rules on by default for modules that opt
  in, Unix-domain socket contracts once the local IPC core lands, event-loop
  registration typestate (`add-fd!` creates a watch the loop must drop before
  close), mutex lock-guard typestate, and a cleanup requirement for implicit
  exceptions that no handler catches.
- **Planned for v1.4.2:** VM-route checking, once the VM shares the native
  checker's front end.
