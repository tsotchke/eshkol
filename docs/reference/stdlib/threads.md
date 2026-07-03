# `core.threads` — mutexes, condition variables, and thread handles

**Source**: [`lib/core/threads.esk`](../../../lib/core/threads.esk)
**Require**: `(require core.threads)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

Thread-coordination primitives in three families: **recursive POSIX mutexes** (`pthread_mutex_t`), **condition variables** (`pthread_cond_t`), and **thread handles** built on top of `(future thunk)` so a thunk runs concurrently on a thread-pool worker while `thread-join` blocks for its result. Mutex/condvar objects are opaque C pointers; a thread handle is a length-2 vector `#('eshkol-thread <future>)`.

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

### `(condvar-signal! c)`
Wake one thread waiting on `c`.

### `(condvar-broadcast! c)`
Wake all threads waiting on `c`.

### `(condvar-destroy! c)`
Destroy and free the condition variable.

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

Edge cases: `mutex-lock!`/`unlock!` return errno-style integers, not booleans. `thread-result-ready?` and `thread-join` on non-thread values are total (no error). Do not `condvar-wait!` without holding the paired mutex.
