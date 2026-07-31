# Event loop reference

Status: SHIPPED (ESH-0011, v1.4 async foundation)

A portable readiness-style multiplexer over the host's I/O facility. This is the
primitive v1.4's sockets, fibers, timers and async I/O register their
descriptors against; it is deliberately small and does not grow a scheduler of
its own.

| Platform | Backend | Source |
| --- | --- | --- |
| macOS, *BSD | `kqueue` / `kevent` | `lib/core/event_loop_kqueue.c` |
| Linux, Android | `epoll_create1` / `epoll_wait` | `lib/core/event_loop_epoll.c` |
| Windows | IOCP + `WSAPoll` / `PeekNamedPipe` | `lib/core/event_loop_iocp.c` |
| WebAssembly, other | fail-closed stub | `lib/core/event_loop_stub.c` |

Exactly one backend is compiled, selected in `CMakeLists.txt` the same way the
GPU layer picks its `gpu_memory` implementation. The portable half — handle
registry, argument validation, the registration table, result coalescing —
lives in `lib/core/event_loop.c` and is shared by all of them. Configure output
names the choice:

```
-- Event-loop backend: kqueue (lib/core/event_loop_kqueue.c)
```

`(event-loop-backend)` reports the same name at run time, so a program or test
can state which implementation it exercised instead of inferring one from the
host.

## Surface

### `(make-event-loop max-events)`

Creates a loop able to hold and report up to `max-events` descriptors.
`max-events` bounds both the registration table and the number of notifications
a single poll returns, so one poll can always drain every registered descriptor
and starvation is impossible by construction.

Returns an integer handle, or `#f` where the platform has no event loop (the
WebAssembly sandbox has no descriptors) or the kernel refused to create one.
Raises if `max-events` is outside `1 … 65536`.

The handle carries a generation counter in its high bits. A handle whose slot
has since been recycled by a later `make-event-loop` is rejected rather than
silently addressing the newer loop.

### `(event-loop-add-fd! loop fd events)`

Watches `fd`. `events` is an interest bitmask:

| Bit | Value | Meaning |
| --- | --- | --- |
| 0 | `1` | readable |
| 1 | `2` | writable |
| 2 | `4` | error / hangup — **reported, never requested** |

At least one of `1` or `2` must be set. Bit `2` is always reported when the
descriptor errors or hangs up, so you never ask for it.

Calling `add` twice for the same descriptor is a modify, not an error: the new
interest wins. Returns `#t`.

Raises on a closed or unknown loop handle, on a negative descriptor, on an empty
interest mask, when the registration table is full, and when the platform cannot
watch that kind of object. The last case matters for portability: **Linux epoll
refuses regular files** (`EPERM`, surfaced as a "cannot watch that kind of
object" condition) because they are always ready, while kqueue accepts them. The
portable intersection of the three backends is pipes, sockets and terminals.

### `(event-loop-remove-fd! loop fd)`

Stops watching `fd`. Returns `#t` if it was being watched, `#f` if it was not —
"not watched" is an answer, not a failure. Raises on a closed or unknown handle.

The registration is dropped even when the kernel reports the descriptor as
already gone, so a descriptor closed behind the loop's back cannot pin table
capacity forever.

### `(event-loop-poll loop timeout-ms)`

Waits for readiness and returns a list of `(fd . events)` pairs, one entry per
ready descriptor, in the order the kernel reported them. A timeout returns the
empty list `'()`, not `#f` — an ordinary result that `for-each` handles without
a special case.

`timeout-ms` of `0` polls without blocking; a negative value blocks until an
event arrives or a signal interrupts the wait.

Readiness is **level-triggered**: buffered bytes are re-reported by every poll
until they are consumed. Hangup rides *alongside* the readiness bit rather than
replacing it, so a reader still drains what is buffered before observing EOF.

A signal that interrupts the wait is reported as a timeout, not an error. That
is what keeps an unbounded wait breakable — `Ctrl-C` interrupts the syscall and
the call returns rather than wedging the REPL. Polling a loop with nothing
registered returns immediately instead of blocking forever on a queue no one can
wake.

Raises on a closed or unknown handle, or on a kernel error.

### `(event-loop-close loop)`

Releases the underlying kqueue descriptor, epoll descriptor or completion port
and retires the handle. Registered descriptors are **not** closed — the loop
never owned them.

Returns `#t` on success and `#f` when the handle was already closed or unknown.
This is the one asymmetry in the surface, and it is deliberate: **close is
idempotent-observable, use fails closed.** A double close is reportable without
being fatal (the contract ESH-0010 specifies for handle lifecycles), while
`add`, `remove` and `poll` on a closed handle raise a catchable condition.

### `(event-loop-backend)`

Returns `"kqueue"`, `"epoll"`, `"iocp"` or `"none"`.

## Example

```scheme
(define loop (make-event-loop 64))
(define p (make-pipe))
(define read-fd (car p))
(define write-fd (cdr p))

(event-loop-add-fd! loop read-fd 1)      ; watch for readable
(fd-write write-fd "ping")

(for-each
  (lambda (entry)
    (display "fd ") (display (car entry))
    (display " ready, bits ") (display (cdr entry)) (newline))
  (event-loop-poll loop 1000))

(event-loop-close loop)
(fd-close read-fd)
(fd-close write-fd)
```

## The Windows adaptation, and its limits

The surface above is a *readiness* model, which is what kqueue and epoll
provide. Windows IOCP is a *completion* model: you associate a handle opened
with `FILE_FLAG_OVERLAPPED`, start an operation, and the port reports that the
operation finished. There is no "is it readable?" query. The adaptation is
therefore explicit, and worth stating plainly rather than glossing.

**What the completion port really does.** The loop owns a genuine port from
`CreateIoCompletionPort`, and it is load-bearing on three counts: every blocking
wait in the backend is a `GetQueuedCompletionStatus` on that port rather than a
bare `Sleep`, which makes the wait interruptible by `PostQueuedCompletionStatus`;
it is the association point v1.4's overlapped sockets will use, so their
completions are dequeued by the same wait; and `event-loop-close` closes it, so
"releases the underlying resource" is literally true on Windows.

**Where readiness actually comes from.** Not from `GetQueuedCompletionStatus`.
Each descriptor is classified once at registration:

- **Sockets** use `WSAPoll` — genuine kernel readiness with a genuine timeout. A
  loop watching only sockets performs exactly one `WSAPoll` per poll call and
  does not spin.
- **Anonymous pipes, consoles and files** use `PeekNamedPipe` /
  `GetNamedPipeInfo` / `GetNumberOfConsoleInputEvents` — documented
  non-consuming readiness queries — driven by a bounded exponential backoff
  (1→8 ms) whose sleep is the completion-port wait above.

The second bullet is the honest limit: for non-socket descriptors this is
*emulated* level-triggered readiness with latency bounded by the 8 ms backoff
cap, not a zero-latency kernel notification. The thread is blocked in the kernel
between probes, so it is not a busy-wait — but it is a polling loop, and calling
it anything else would be false. `lib/agent/c/agent_poll.c` has taken the same
approach since v1.2 for the same reason: anonymous Win32 pipe handles are not
waitable objects and cannot be associated with a completion port at all.

**Why not the standard tricks.** Zero-byte `WSARecv` works only for sockets —
exactly the case `WSAPoll` already handles correctly and simply — and buys
nothing for anonymous pipes, while costing a pending `OVERLAPPED` per descriptor
whose cancellation is a classic use-after-free source. `\Device\Afd` +
`IOCTL_AFD_POLL` (libuv's and mio's approach) is the only route to true
IOCP-delivered readiness, but it is undocumented internal NT interface and still
sockets-only — not an acceptable dependency for a primitive whose entire job is
portability. Replacing `make-pipe`'s anonymous pipes with overlapped *named*
pipes would make pipes completion-capable, but that is a change to the
process-plumbing surface and belongs with the socket work.

## Memory model

An event loop is `malloc`'d, never arena-allocated, and the Scheme-visible value
is a plain integer handle. This is the convention every other kernel-resource
holder in the tree already uses (the sqlite, line-reader and watcher tables in
`lib/agent/c/`), and here it is the only safe choice:
`lib/core/runtime_regions.cpp` evacuates a region by `memcpy`-ing objects to a
new address, so an arena-resident struct owning a kernel descriptor would be
byte-duplicated on escape — two structs, one descriptor, two closes. Because a
`malloc`'d pointer lies outside every region arena the evacuator leaves it alone
unconditionally, and because the Scheme value is an integer it is not a heap
value to the evacuator at all. No `evac_kind_for` entry is required.

The handle table is not synchronised, exactly like the other handle tables in
the tree: a loop is owned by the thread that opened it. v1.4's fiber scheduler
is single-threaded per loop by design; sharing one loop across OS threads needs
a lock that lands with that work.

## Substrate coverage

| Substrate | Status |
| --- | --- |
| JIT (`-r`) | Full support |
| AOT | Full support |
| Bytecode VM (`vm-src`, `vm-eskb`) | Full support — the VM calls the same `lib/core/event_loop.c`, so there is no parity gap. Byte-diffed against native by `tests/vm_parity/corpus/42_event_loop.esk`; recorded as `vm-supported` in `tests/vm_parity/PARITY.tsv`. |
| WebAssembly | Fails closed. The browser sandbox has no descriptors, so the whole surface returns `#f` and `(event-loop-backend)` answers `"none"` — the same degradation `make-pipe`, `fd-write` and `fd-close` already use there. |

`tests/v1_3_edge_cases/event_loop_test.esk` is the acceptance battery (27
checks) and the `event_loop_works` ICC probe runs it on both native substrates.

## See also

- [Memory model](memory-model.md) — arenas, regions and evacuation.
- [Parallelism & threading](parallelism.md) — the work-stealing pool the v1.4
  fiber scheduler will sit beside.
- `inc/eshkol/core/event_loop.h` — the C API v1.4 sockets link against.
