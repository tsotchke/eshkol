# Algebraic effects and handlers

Eshkol has algebraic effects in the sense of Plotkin and Pretnar: a program
declares *operations*, performs them like ordinary procedure calls, and gives
them meaning with *handlers* that receive the rest of the computation as a
resumption. Effects are also part of the type checker: every procedure has an
inferred *effect row*, handlers remove what they handle, and `declare-effects`
states what a procedure may do. Design: [ADR-0045](../../design/adr/0045-algebraic-effects-and-handlers.md).

The forms work identically on the native JIT (`eshkol-run -r`), native AOT, and
the bytecode VM. One lowering (`inc/eshkol/frontend/effect_syntax_core.h`)
serves both compilers, and the runtime is `core.control.effects`, which is part
of the standard library closure. The standard effects (state, reader, writer,
exceptions, choice, generators, async) are in `core.effects`.

## `define-effect`

```scheme
(define-effect Name (op formals ...) ...)
```

Declares the effect `Name` and one procedure per operation. `formals` may be a
dotted list for a variadic operation. Performing an operation is calling it:

```scheme
(define-effect Store (get) (put v))
(define-effect Log (log-all . items))
(get)            ; performs Store.get
(put 5)          ; performs Store.put
```

Operations are first-class procedures. Performing one with no handler in
scope raises a condition satisfying `unhandled-effect-error?` whose irritants
are the effect and operation names.

## `handle` and `handle-shallow`

```scheme
(handle expr clause ...)
(handle-shallow expr clause ...)

clause ::= (return (x) body ...)          ; applied to a normal result of expr
         | ((op formals ...) k body ...)  ; k is the resumption
         | (fun (op formals ...) body ...) ; resumes with the value of body
```

`expr` runs with the handler installed. When it performs an operation that has
a clause, the clause runs with the operation's arguments and a resumption `k`;
`(k v)` continues `expr` from the operation, with `v` as the operation's value,
and returns what that continuation produces. The value of the clause is the
value of the `handle` form. A `return` clause, when present, transforms a
normal result of `expr`; without one the result is unchanged.

```scheme
(handle (+ 1 (get))
  (return (x) (* x 10))
  ((get) k (k 4)))                  ; => 50

(handle (list (if (decide) 1 2) (if (decide) 'a 'b))
  (return (x) (list x))
  ((decide) k (append (k #t) (k #f))))
; => ((1 a) (1 b) (2 a) (2 b))
```

- **Deep and shallow.** A `handle` stays installed around every resumption, so
  each later operation of the resumed computation reaches it again. A
  `handle-shallow` handles one operation: the resumption runs without it, and
  its `return` clause does not apply to the resumed result.
- **Clause scope.** A clause runs with the handler's *enclosing* handlers in
  scope; an operation performed inside a clause goes to an outer handler.
- **Forwarding.** An operation with no clause in a handler goes to the next
  handler out.
- **Resumptions.** `k` may be called zero times (the computation is
  abandoned), once, or many times (each call resumes the same point), and may
  be stored and called after the `handle` has returned, from anywhere.
  `(fun (op x) e)` is the same as `((op x) k (k e))`.
- **Recognition.** The form is a `handle` only when every operand after the
  first is clause-shaped; otherwise it is an ordinary call, so a program's own
  procedure named `handle` keeps working. `define-effect`, `handle`,
  `handle-shallow` and `declare-effects` are otherwise reserved keywords, and a
  user macro with the same name takes precedence.

## How each clause runs

The lowering looks at how a clause uses its resumption and chooses the
cheapest strategy with the same meaning:

| Clause uses `k` | Strategy | Captures |
|---|---|---|
| every tail position is `(k e)`, nowhere else (and every `fun`) | runs in place at the operation | nothing |
| never | unwinds to the `handle`, runs there | the `handle`'s delimiter, once per evaluation |
| only as a tail call, but some path does not resume | unwinds, runs, jumps back into the computation | the operation's continuation |
| anything else | as above, and `(k v)` returns to its call site | the operation's continuation and the resumption site |

A `handle` whose clauses all run in place captures no continuation at all;
`with-store`, `with-reader`, `run-writer`, `for-each-yield`, `generator->list`
and `generator-fold` are of this kind. Measured on AOT (x86-64 and arm64), an
operation handled in place costs about 0.2 µs and allocates nothing.

Captured continuations use the language's re-entrant `call/cc`, so the rules of
[continuations.md](continuations.md) apply to them: a capture copies the live
stack into the process arena, and capturing inside `with-region` pins the open
regions. Programs that perform millions of non-tail operations (pulling
generators, nondeterminism, task switches) retain one stack image per capture
for the life of the process.

## Dynamic extent

- **In place.** A clause that runs in place does not leave the dynamic extent
  of the handled computation: no `dynamic-wind` after-thunk runs, and an
  exception the clause raises propagates through the computation's `guard`s.
- **Unwinding clauses.** A clause that unwinds leaves the extent first: the
  after-thunks between the operation and the `handle` run, innermost first,
  before the clause body. Resuming re-enters the extent (before-thunks,
  outermost first), and leaving it again runs the after-thunks again.
- **Exceptions.** An exception raised by the handled computation propagates to
  the context that is running it: the `handle` form, or, while a resumption is
  running it, the resumption's call site. An exception raised by an unwinding
  clause propagates from the `handle` form.
- **Other exits.** Leaving a handled computation by `raise` to an outer
  `guard`, or by an escape continuation, removes its handlers; re-entering a
  continuation captured inside it reinstalls them.
- **Resumption from elsewhere.** While a resumption runs a computation, the
  `dynamic-wind` extents that enclose the resumption call but not the original
  `handle` are exited, and re-entered when the computation returns to the
  resumption.
- **Automatic differentiation.** Operations handled in place are ordinary
  calls, so derivatives flow through their arguments and results
  (`(derivative (lambda (x) (with-reader 3 (lambda () (* (ask) x x)))) 2.0)`
  is `12.0`). A resumed computation re-executes its code; on a reverse-mode
  tape each execution records its own nodes.

## Effect rows and `declare-effects`

The checker infers a *latent effect row* for every top-level procedure: the
labels its calls may perform, and open tails.

| Label | Introduced by |
|---|---|
| `IO` | host I/O: `display`, `write`, `read-…`, files, sockets, the event loop, time, randomness |
| `State` | `set!` of a variable not bound in the procedure; procedures whose name ends in `!` |
| `Exception` | `raise`, `error`, `assert` |
| `Control` | `call/cc` |
| *Name* | an operation of an effect declared with `define-effect` |

- A deep `handle` with a clause for **every** operation of an effect removes
  that effect from its body's row. A partial handler removes nothing.
- A `guard` with a catch-all clause (`else` or `#t`) removes `Exception`.
- Calling a procedure argument adds a tail `param:i`; at each call site the
  tail becomes the row of the actual argument (`map`, `for-each`, `filter`,
  `fold`, `apply`, `sort`, `call-with-values`, `dynamic-wind`, ... are
  effect-polymorphic in the same way).
- Calling something the analysis cannot see adds the tail `?`. An open tail is
  never pure.

```scheme
(declare-effects f (IO State))   ; f performs at most IO and State
(declare-effects g Pure)         ; g performs nothing (also written ())
(declare-effects h (IO . e))     ; IO, plus whatever h's procedure arguments do
(declare-effects m e)            ; only what m's procedure arguments do
```

A procedure whose inferred row is not contained in its declaration is a type
error under `--strict-types` and a warning in gradual mode, reported through
the same enforcement point as other type errors:

```
'noisy' is declared Pure but may perform IO (display at line 4)
```

`ESHKOL_DUMP_EFFECTS=1` prints every inferred row while compiling:

```
effects: with-tally : <| param:0 - Tally>
effects: tally-then-print : <IO>
```

The bytecode VM runs programs without the type checker; rows are checked when
compiling with the native compiler.

## `core.control.effects` (in the standard library)

| Procedure | Meaning |
|---|---|
| `(effect? x)`, `(effect-name e)`, `(effect-operation-names e)` | effect values made by `define-effect` |
| `(effect-operation? x)`, `(effect-operation-name op)`, `(effect-operation-effect op)`, `(effect-operation-arity op)` | operations (arity `-1` = variadic) |
| `(effect-handled? op)` | whether a handler for `op` is in scope |
| `(effect-handler-depth)` | number of installed handlers |
| `(unhandled-effect-error? x)` | the condition raised by an unhandled operation |

## `core.effects`

```scheme
(require core.effects)
```

| Effect | Operations | Handlers |
|---|---|---|
| `Store` | `(get)` `(put v)` | `(with-store init thunk)`; `(run-store init thunk)` => `(result . state)`; `(run-store/pure init thunk)`: state-passing, each resumed branch has its own state |
| `Reader` | `(ask)` | `(with-reader env thunk)`, `(local-reader f thunk)` |
| `Writer` | `(tell x)` | `(run-writer thunk)` => `(result . items)` |
| `Throw` | `(throw x)` | `(catch-throw thunk on-throw)`, `(throw->result thunk)` => `(ok . v)` or `(thrown . x)` |
| `Choice` | `(decide)` `(fail)`; `(choose list)`, `(choose-range lo hi)` | `(all-choices thunk)`, `(first-choice thunk none)` |
| `Yield` | `(yield v)` | `(for-each-yield f thunk)`, `(generator->list thunk)`, `(generator-fold f init thunk)`, `(make-generator thunk)` with `generator-done?` |
| `Async` | `(async-spawn thunk)` `(async-yield)` `(async-sleep ms)` `(await-fd fd events)` `(await task)`; `await-readable`, `await-writable` | `(run-async main)`: cooperative tasks; descriptor and timer waits go through one event loop |

```scheme
(all-choices (lambda ()
  (let* ((a (choose-range 1 13)) (b (choose-range a 13)) (c (choose-range b 13)))
    (if (= (+ (* a a) (* b b)) (* c c)) (list a b c) (fail)))))
; => ((3 4 5) (5 12 13) (6 8 10))

(run-async (lambda ()
  (let* ((p (make-pipe))
         (reader (async-spawn (lambda () (await-readable (car p)) 'readable)))
         (writer (async-spawn (lambda () (async-sleep 20) (fd-write (cdr p) "x") 'wrote))))
    (list (await writer) (await reader)))))
; => (wrote readable)
```

Fixtures: `tests/effects/` (handler laws, resumption counts, unwinding order,
the standard effects, regions and AD) run on all three engines, and
`tests/effects/typing/` holds the accepted and rejected effect-typing programs;
`scripts/run_effects_gate.py` (CTest `v19_effects_gate`) runs both.
