# `core.threads` — mutexes, condition variables, semaphores, barriers, and thread handles

**Source**: [`lib/core/threads.esk`](../../../lib/core/threads.esk)
**Require**: `(require core.threads)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

Thread-coordination primitives in five families: **recursive POSIX mutexes** (`pthread_mutex_t`), **condition variables** (`pthread_cond_t`), **counting semaphores**, **reusable barriers**, and **thread handles** built on top of `(future thunk)` so a thunk runs concurrently on a thread-pool worker while `thread-join` blocks for its result. Mutex/condvar objects are opaque C pointers; a thread handle is a length-2 vector `#('eshkol-thread <future>)`.

Semaphores and barriers are one facility with the mutex and condition variable: each is a tagged vector holding its counters plus one mutex and one condvar from this module, and every wait re-checks its condition under the mutex, so spurious wakeups cannot release a waiter early. Timed waits are measured on the monotonic clock.

**Routes.** Native JIT (`eshkol-run -r`) and AOT run this module on real pool threads. The bytecode VM does not load `core.threads`: the VM admits only the `agent.sqlite`/`agent.crypto` extern signatures, and it offers its own single-threaded mutex/condition-variable builtins (`make-mutex`, `make-condition-variable`, …) without a `require`. Semaphores and barriers on the VM are planned for v1.4.1.

> Concurrency note: the closure-capture correctness that makes `make-thread` safe (a `set!`-mutated variable shared across the spawner and worker) was the subject of **ESH-0074** ("set!-mutated variable captured by MULTIPLE closures is not shared", status: done — general assignment-conversion fix following PR #50). Keep worker thunks self-contained and prefer returning values over mutating shared captures.

## Functions

### `(make-mutex)`
Create a recursive POSIX mutex. Returns an opaque pointer. Recursive means re-locking from the *same* thread does not deadlock (SRFI-18 semantics). You own the lifetime — call `mutex-destroy!` or rely on process exit.

### `(mutex-lock! m)`
Block until the mutex is acquired. Returns `0` on success or an errno on failure.

### `(mutex-trylock! m)`
Non-blocking lock attempt. Returns `0` if acquired, `EBUSY` (16 on macOS/Linux) if held by another thread. Because the mutex is recursive, a re-lock from the same thread succeeds with `0`.

### `(mutex-unlock! m)`
Release the mutex. Returns `0` on success.

### `(mutex-destroy! m)`
Destroy and free the mutex. Do not use `m` afterward.

### `(with-mutex m thunk)`
Acquire `m`, invoke `(thunk)`, release `m` before returning — even if `thunk` raises (the error is re-raised after unlock). Returns the thunk's value.

```scheme
;; threads.esk
(require core.threads)
(define m (make-mutex))
(display (mutex-lock! m)) (newline)
(display (mutex-trylock! m)) (newline)   ; recursive re-lock, same thread
(display (mutex-unlock! m)) (newline)
(display (mutex-unlock! m)) (newline)
(display (with-mutex m (lambda () (+ 1 2)))) (newline)
(mutex-destroy! m)
```
```
0
0
0
0
3
```

### `(make-condvar)`
Create a POSIX condition variable. Returns an opaque pointer.

### `(condvar-wait! c m)`
Atomically release mutex `m`, block until `c` is signalled, reacquire `m` before returning. `m` MUST be held by this thread on entry. (Used internally by `core.channels`; a bare producer/consumer example would block, so none is shown here.)

### `(condvar-wait-timeout! c m timeout-ms)`
Like `condvar-wait!`, but gives up after `timeout-ms` milliseconds of monotonic time. Returns `#t` when woken and `#f` when the timeout elapsed; either way `m` is held again on return. A wakeup can be spurious, so wait in a loop that re-checks your condition. `m` must be held exactly once by the calling thread.

### `(condvar-signal! c)`
Wake one thread waiting on `c`.

### `(condvar-broadcast! c)`
Wake all threads waiting on `c`.

### `(condvar-destroy! c)`
Destroy and free the condition variable.

### `(make-semaphore permits)`
Create a counting semaphore with `permits` free permits (an exact integer ≥ 0). Raises for a negative or non-integer count.

### `(semaphore? s)`
Predicate: is `s` a semaphore from `make-semaphore`?

### `(semaphore-acquire! s)`
Take one permit, blocking until one is free. Returns `#t`.

### `(semaphore-try-acquire! s [timeout-ms])`
Take one permit if one is free and return `#t`, otherwise return `#f` at once. With `timeout-ms`, wait up to that many milliseconds (monotonic clock) for a permit before returning `#f`; spurious wakeups do not extend the bound.

### `(semaphore-release! s [count])`
Return one permit (or `count` permits, an exact integer ≥ 1) and wake one waiter per permit. Returns the number of free permits after the release.

### `(semaphore-value s)`
Number of free permits at the moment of the call.

### `(semaphore-destroy! s)`
Free the semaphore's mutex and condvar. No thread may be waiting on `s`; do not use it afterward.

```scheme
;; threads.esk
(require core.threads)
(define s (make-semaphore 2))
(display (semaphore-acquire! s)) (newline)
(display (semaphore-try-acquire! s)) (newline)
(display (semaphore-try-acquire! s)) (newline)      ; no permit left
(display (semaphore-try-acquire! s 20)) (newline)   ; waits 20 ms, still none
(display (semaphore-release! s)) (newline)          ; free permits now
(display (semaphore-value s)) (newline)
(semaphore-destroy! s)
```
```
#t
#t
#f
#f
1
1
```

### `(make-barrier parties)`
Create a reusable barrier for `parties` threads (an exact integer ≥ 1).

### `(barrier? b)`
Predicate: is `b` a barrier from `make-barrier`?

### `(barrier-wait! b)`
Block until all parties have called `barrier-wait!` for the current generation. Returns `#t` in exactly one party per generation (the one whose arrival completed it) and `#f` in the others. Each completed meeting advances the generation, so the same barrier serves any number of phases; a waiter is released only when its own generation completes.

Every party must be running at the same time, so start parties with `make-thread` (each thunk goes to its own pool worker) and keep `parties` within `(thread-pool-size)`. `parallel-map` is a data-parallel map and may run some elements on the calling thread first, so it is not a way to start parties that wait on each other.

### `(barrier-parties b)`
Number of parties the barrier waits for.

### `(barrier-generation b)`
Number of generations the barrier has completed.

### `(barrier-destroy! b)`
Free the barrier's mutex and condvar. No thread may be waiting on `b`; do not use it afterward.

```scheme
;; threads.esk
(require core.threads)
(define b (make-barrier 3))
(define (party) (list (barrier-wait! b) (barrier-wait! b)))
(define ts (list (make-thread party) (make-thread party) (make-thread party)))
(define rs (map thread-join ts))
(display (length (filter car rs))) (newline)     ; one serial party in phase 1
(display (length (filter cadr rs))) (newline)    ; one serial party in phase 2
(display (barrier-generation b)) (newline)
(barrier-destroy! b)
```
```
1
1
2
```

### `(make-thread thunk)`
Spawn `thunk` on a thread-pool worker; returns a thread handle. The thunk runs concurrently with the spawner. Pass the *thunk itself* (a 0-arg callable) — do not call it.

### `(thread? h)`
Predicate: is `h` a handle from `make-thread`? (A length-2 vector tagged `'eshkol-thread`.)

### `(thread-result-ready? h)`
Non-blocking check: has the worker finished? Returns `#f` for non-thread values.

### `(thread-join h)`
Block until the thunk completes; return its result. Memoised (a second join returns the cached result). Joining a non-thread returns the value unchanged (per R7RS `force`).

```scheme
;; threads.esk
(require core.threads)
(define t (make-thread (lambda () (* 6 7))))
(display (thread? t)) (newline)
(display (thread? 5)) (newline)
(display (thread-join t)) (newline)
(display (thread-result-ready? t)) (newline)
(display (thread-join 99)) (newline)     ; non-thread -> returned as-is
```
```
#t
#f
42
#t
99
```

Edge cases: `mutex-lock!`/`unlock!` return errno-style integers, not booleans; the semaphore, barrier and `condvar-wait-timeout!` procedures return booleans or counts as documented above. `thread-result-ready?` and `thread-join` on non-thread values are total (no error). Do not `condvar-wait!` without holding the paired mutex.
