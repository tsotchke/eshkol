---
kind: reference
status: current
owner-area: types
since: v1.4.0-connection
sources:
  - inc/eshkol/strict_resource_contracts.h
  - exe/eshkol-run.cpp
  - tests/v1_4_resource_ownership/test_compiled.py
  - tests/v1_4_resource_ownership/test_runtime.py
  - docs/design/adr/0056-linear-handle-typestate.md
---
# Linear handles and the borrow pattern

Every native handle (a socket, a TLS context or session, a file port, the
event loop, a thread, a mutex or condition variable, a subprocess) is an
**owned, linear value**. It moves through one typestate:

```text
open ──borrow──▶ borrowed ──loan ends──▶ open ──close──▶ closed
```

Under `eshkol-run --strict-resources`, the compiler checks that typestate on
every path it can see and refuses to produce an artifact when a rule is broken:

- a handle that is still open when its scope ends (an **unclosed handle**);
- a **double close**;
- a **use after close**, including a use on a path where it may be closed;
- a **close while borrowed**, and a borrowed handle that **escapes** its loan;
- a handle passed to an operation of the wrong kind.

The default profile does not run these rules. There, the runtime handle
registry still refuses a double close or a use after close when it happens,
as a catchable condition. The rules and their rationale are in
[ADR-0056](../../design/adr/0056-linear-handle-typestate.md).

## Using the strict profile

```bash
eshkol-run --strict-resources -r server.esk        # JIT
eshkol-run --strict-resources -o server server.esk # AOT
```

A rejected program exits with status 1, prints `Strict resource error:` lines,
and writes no artifact. The bytecode VM profile (`--profile hosted-vm`),
`--eval`, `--wasm`, opaque `.o` inputs, and `-l` libraries are refused under
this flag with an explicit diagnostic.

## Handle families

| Family | Acquire | Close | Failed acquisition | Borrow scope |
|---|---|---|---|---|
| TCP listener / stream | `tcp-listen`, `tcp-connect`, `tcp-accept` | `tcp-close` | `network-error?` | `borrow` |
| UDP socket | `udp-bind` | `udp-close` | `network-error?` | `borrow` |
| TLS context / listener / session | `tls-client-context`, `tls-server-context`, `tls-listen`, `tls-accept`, `tls-connect` | `tls-context-close`, `tls-listener-close`, `tls-session-close` | `tls-error?` | `borrow` |
| Input port | `open-input-file`, `open-binary-input-file` | `close-port`, `close-input-port` | raises | `call-with-input-file`, `call-with-port` |
| Output port | `open-output-file`, `open-binary-output-file`, `open-output-file-append` | `close-port`, `close-output-port` | raises | `call-with-output-file`, `call-with-port` |
| Event loop | `make-event-loop` | `event-loop-close` | `#f` | `borrow` |
| Thread | `make-thread` | `thread-join` | raises | — |
| Mutex / condition variable | `make-mutex`, `make-condvar` | `mutex-destroy!`, `condvar-destroy!` | raises | `with-mutex` |
| Process | `process-spawn`, `process-spawn-argv`, … | `process-destroy` | `#f` | `borrow` |

Every other operation of a family (reading, writing, polling, waiting)
borrows its handle for the duration of the call. The full table, with the
accepted kind of every argument, is `inc/eshkol/strict_resource_contracts.h`.
Module operations are trusted only for the reviewed SDK source of
`core.net`, `core.tls`, `core.threads`, and `agent.subprocess`; an edited
module is refused until its contract is reviewed
(`scripts/strict_resource_pins.py`).

A factory whose failure is a value must be refined before its handle is used:

```scheme
(define loop (make-event-loop 8))
(if loop
    (begin (event-loop-poll loop 0) (event-loop-close loop))
    (display "no event loop"))
```

## The borrow pattern

A borrow scope lends a handle for one dynamic extent and closes it itself.
Inside the scope the handle cannot be closed, stored, returned, or captured by
a closure that outlives the scope:

```scheme
(call-with-output-file "log.txt"
  (lambda (out) (write-string "ready" out)))

(define first-line
  (call-with-input-file "log.txt" (lambda (in) (read-line in))))
```

`call-with-port` takes an owner you already hold and closes it, so the owner
is consumed by the call. `(borrow h body ...)` lends any handle for its body
and gives it back open. `with-mutex` borrows a mutex for its thunk.

A borrow scope closes its port on normal return, when an exception leaves the
scope, and when a continuation escapes it. The close runs once.

## What the checker follows

**Branches.** `if` (one or two arms), `cond`, `when`, and `unless` are joined
path by path. A handle closed on one branch and open on another is open on
some paths only; every later use and the scope exit report it.

**Loops.** A named `let` or `do` loop must leave every handle in scope as it
found it when it starts the next iteration, and a handle acquired in an
iteration must be closed before the next one. The loop may close a handle on
its exit path:

```scheme
(define in (open-input-file "data.txt"))
(let loop ((line (read-line in)))
  (if (eof-object? line)
      (close-port in)
      (begin (display line) (loop (read-line in)))))
```

A call to the loop name that is not a tail call is refused while handles are
in scope.

**Functions.** A function is checked at each call site with the caller's
handles. It may borrow a parameter, close it, or return a handle it acquired;
the caller sees the effect. A tail-recursive function may close its parameter
on its base case only:

```scheme
(define (drain in count)
  (let ((line (read-line in)))
    (if (eof-object? line)
        (begin (close-port in) count)
        (drain in (+ count 1)))))
```

A function that returns a handle on some paths and another value on others,
or that returns its own handle parameter, is refused.

**Closures.** A lambda passed to `for-each`, `map`, `vector-for-each`,
`string-for-each`, or a handler runs inline and must leave every handle as it
found it. A lambda passed to a borrow scope, `dynamic-wind`, or applied
immediately runs exactly once. A thread body may not capture any handle. Any
other closure that captures a handle is refused, because the checker cannot see
when it runs.

**Exceptions.** Inside `guard`, the handler sees the state at every operation
that may raise. A handle bound inside the guard body that is open at such a
point would be lost when the exception leaves its scope, so it is refused;
acquire it with a borrow scope instead. An explicit `raise` or `error` while a
handle is open and no handler is in scope is refused. An exception that no
handler catches ends the program.

**dynamic-wind.** The after thunk is checked from the normal exit and from
every exceptional or escaping exit of the body:

```scheme
(define in (open-input-file "data.txt"))
(dynamic-wind
  (lambda () #f)
  (lambda () (display (read-line in)))
  (lambda () (close-port in)))
```

**call/cc.** With a literal `(lambda (k) ...)`, `k` is an escape: calling it
leaves with the current handle states, and the result joins the normal exit
with every escape. A continuation stored, returned, or captured while handles
are open is refused: the checker cannot follow a re-entry.

The checker names the form it does not follow (`case`, `match`, `=>` clauses,
multiple values, quasiquote, computed procedures) instead of accepting it.

## Tests

`tests/v1_4_resource_ownership/` holds the corpus: `test_compiled.py` compiles
every positive fixture and checks that every negative fails with its expected
diagnostic and no artifact; `test_runtime.py` runs the positives on JIT and
AOT and observes that a borrow scope closes its port on a raise and on a
continuation escape. CTest entries: `strict_resource_profile_compiled`,
`strict_resource_profile_runtime`, `strict_resource_handles_runtime`,
`strict_resource_contract_pins`.
