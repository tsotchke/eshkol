# `core.fibers` — cooperative fibers over the event loop

**Source**: [`lib/core/fibers.esk`](../../../lib/core/fibers.esk)
**Require**: `(require core.fibers)` — not auto-loaded via `stdlib`.
**Runs on**: native JIT and AOT (Linux, macOS, FreeBSD on x86-64 and AArch64;
Windows through the system Fibers API). The bytecode VM: Planned for v1.4.1.
In the browser the coroutine builtins answer `#f`.

A fiber is a lightweight task: a procedure running on a stackful coroutine
(see [coroutines](../runtime/coroutines.md)) that this module's scheduler
resumes. A fiber runs until it yields, sleeps, waits for a file descriptor, a
future, a condition or another fiber, or finishes. While it waits, other fibers
run; when every fiber waits, the scheduler blocks in the
[event loop](../runtime/event-loop.md) until a descriptor is ready or the next
sleeper is due. Ten thousand live fibers cost ten thousand small records and as
many stacks, whose pages are committed only as they are touched and are given
back when each fiber finishes.

The design is recorded in
[ADR-0030](../../design/adr/0030-stackful-coroutines-for-fibers.md).

## Semantics

- **Ordering.** The run queue is FIFO: fibers start in spawn order, and a fiber
  that becomes ready goes to the back, so every ready fiber runs once before
  any runs twice. Scheduling is cooperative: a fiber that never suspends keeps
  the thread.
- **Dynamic extent.** Each fiber has its own dynamic-wind chain, exception
  handlers, `parameterize` bindings and open regions. Switching fibers runs no
  dynamic-wind thunk. A fiber's `after` thunks run exactly once, when its body
  leaves the extent — by returning, by a raise, or by cancellation.
- **Raises.** A raise that escapes a fiber's body finishes the fiber as
  `failed`; `fiber-join` re-raises the same object in the joiner.
- **Cancellation.** `fiber-cancel!` raises a fiber-cancelled condition inside
  the fiber at its current suspension point (or before it first runs). It
  unwinds like any raise, so `after` thunks run once. A body may intercept it
  with `guard`. A fiber that ends with its own condition is `cancelled`.
- **Joining from outside.** `fiber-join`, `fiber-sleep` and `fiber-wait-until`
  called outside every fiber run the scheduler until they can return, so a
  program can `spawn` and then `fiber-join` without calling `run-fibers`.
- **Threads.** The scheduler belongs to the thread that runs it. A coroutine
  refuses to be resumed by another thread.
- **Channels.** Inside a fiber, a blocking `channel-send!` or `channel-recv!`
  from [`core.channels`](channels.md) parks the fiber, not the thread.

## Procedures

### `(spawn thunk)`
Create a fiber that will run `(thunk)` and put it at the back of the run queue.
Returns the fiber. Nothing runs until the scheduler does.

### `(run-fibers)`
Run fibers until none is left. Raises when the remaining fibers all wait on
something no fiber can provide (for example two fibers joining each other).

### `(fiber-yield)`
Let every other ready fiber run, then continue. Outside a fiber it does nothing.

### `(fiber-sleep ms)`
Park the calling fiber for at least `ms` milliseconds. Outside a fiber, run the
scheduler for that long.

### `(fiber-await-readable fd)` / `(fiber-await-writable fd)`
Park the calling fiber until `fd` is readable (or writable) through the event
loop. Returns the ready bits: 1 readable, 2 writable, 4 error or hangup.

### `(fiber-await future)`
Park until a future made by `(future thunk)`, or a `core.threads` handle, is
ready, then return its value (`force` re-raises its raise).

### `(fiber-wait-until pred)`
Park until `(pred)` is true. Conditions are checked when the run queue drains
and, while it stays busy, at least once every max(64, waiters) scheduling steps.

### `(fiber-join f)`
Wait for `f` to finish and return its value. Re-raises its raise; raises its
fiber-cancelled condition when it was cancelled. A fiber cannot join itself.

### `(fiber-cancel! f)`
Request cancellation. Returns `#t`, or `#f` when `f` had already finished.
Cancelling the running fiber raises at once.

### `(fiber? x)`, `(fiber-status f)`, `(fiber-result f)`, `(fiber-id f)`
Predicate; state (`ready`, `running`, `waiting`, `done`, `failed`,
`cancelled`); the finished fiber's value (raising for a failed or cancelled
fiber); a small integer id.

### `(current-fiber)`
The fiber whose body is running, or `#f`.

### `(fiber-cancelled? x)`, `(fiber-cancelled-fiber c)`
Recognise a cancellation condition and name the fiber it cancelled.

### `(fiber-scheduler-stats)`
An association list: `live`, `ready`, `sleeping`, `io-waiting` (descriptors
registered), `condition-waiting`.

## Example

```scheme
;; fibers_example.esk
(require core.fibers)

(define (worker name n)
  (lambda ()
    (let loop ((i 0))
      (if (< i n)
          (begin (display (list name i)) (newline)
                 (fiber-yield)
                 (loop (+ i 1)))
          name))))

(define a (spawn (worker 'a 2)))
(define b (spawn (worker 'b 2)))
(display (list (fiber-join a) (fiber-join b))) (newline)

(define cleanups 0)
(define sleeper
  (spawn (lambda ()
           (dynamic-wind (lambda () #t)
                         (lambda () (fiber-sleep 100000))
                         (lambda () (set! cleanups (+ cleanups 1)))))))
(fiber-yield)
(spawn (lambda () (fiber-cancel! sleeper)))
(run-fibers)
(display (list (fiber-status sleeper) cleanups)) (newline)

(define p (make-pipe))
(define reader (spawn (lambda () (fiber-await-readable (car p)))))
(spawn (lambda () (fiber-sleep 10) (fd-write (cdr p) "x")))
(display (fiber-join reader)) (newline)
```
```
(a 0)
(b 0)
(a 1)
(b 1)
(a b)
(cancelled 1)
1
```

## Evidence

`tests/fibers/fibers_test.esk` (ordering, round-robin fairness, sleep order,
raises crossing fibers, cancellation with exactly-once `after` thunks, a fiber
parked on a pipe, channel ping-pong, futures) and
`tests/fibers/fibers_stress_test.esk` (10,000 live fibers, 100,000 switches a
wave, 5,000 channel ping-pong pairs) run through JIT and O1 AOT in
`scripts/run_fibers_gate.py`, which also compares peak RSS between one and four
stress waves. The gate feeds the `v14_fibers_coroutines` criterion of the
v1.4-connection oracle.
