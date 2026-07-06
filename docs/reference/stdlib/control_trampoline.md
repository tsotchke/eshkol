# `core.control.trampoline` — bounded-stack recursion via thunks

**Source**: [`lib/core/control/trampoline.esk`](../../../lib/core/control/trampoline.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.control.trampoline)`

A trampoline drives CPS-style recursion in constant stack space. Instead of calling itself directly, a function returns a **thunk** (a zero-argument procedure) in tail position; `trampoline` loops, invoking each returned thunk until a non-procedure value appears. This sidesteps deep native call stacks. Note that direct mutual tail calls are now proper R7RS tail calls (emitted as LLVM `musttail`, ESH-0102 resolved) and already run in O(1) stack, so a trampoline is only needed for continuation-passing patterns that pass thunks through data structures or across a higher-order boundary — e.g. a tail call that forwards a freshly built closure argument, which still falls back to a bounded call.

## Functions

### `(trampoline thunk)`
Calls `thunk`, then repeatedly re-invokes the result as long as it is a procedure, returning the first non-procedure value. Runs as a `do` loop, so stack depth stays constant regardless of logical recursion depth.

```scheme
;; trampoline.esk
(require stdlib)
(define (even-cps n)
  (if (= n 0) (done #t)
      (bounce (lambda () (odd-cps (- n 1))))))
(define (odd-cps n)
  (if (= n 0) (done #f)
      (bounce (lambda () (even-cps (- n 1))))))
(display (trampoline (lambda () (even-cps 100000)))) (newline)
(display (trampoline (lambda () (odd-cps 7)))) (newline)
```
```
#t
#t
```

Edge cases: because the loop condition is `(not (procedure? result))`, a computation whose genuine final value is itself a procedure would be re-invoked instead of returned — wrap such a value so the terminal value is not a procedure.

### `(bounce thunk)`
Marks a thunk as the next trampoline step. It is the **identity function** — it simply returns the thunk it is given. Provided for readability at recursion sites; `(bounce (lambda () ...))` is equivalent to `(lambda () ...)`.

```scheme
(require stdlib)
(display (procedure? (bounce (lambda () 1)))) (newline)
```
```
#t
```

### `(done value)`
Marks a final value. Also the **identity function** — `(done x)` is exactly `x`. Provided so recursion sites read symmetrically with `bounce`; the trampoline stops on any non-procedure, whether or not it was wrapped with `done`.

```scheme
(require stdlib)
(display (done 42)) (newline)
```
```
42
```

Edge cases: since `done` returns its argument verbatim, `(done some-procedure)` does **not** shield a procedure value from being trampolined.
