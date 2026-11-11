# Decision: Unix-domain stream sockets

**Status:** Socket integration contract agreed; scaffold amendments and implementation pending. **Updated:** 2026-11-11.
**Basis:** [IPC design](https://github.com/Gabriel-Kahen/eshkol/blob/93c02a9679b9476c58296bfeb6e229d251a21224/weeks/IPC_DESIGN-2026-11-02.md); [#916 at `da7005c7f`](https://github.com/tsotchke/eshkol/pull/916). Agreed design behavior; runtime validation will accompany implementation.

**Companion review:** [#916 scaffold findings and proposed resolutions](SOCKET_DESIGN_REVIEW-2026-11-10.md) explain the contract gaps and acceptance-test issues behind this decision.

## Context and decision

Provide filesystem `AF_UNIX/SOCK_STREAM` sockets using existing epoll/kqueue and the shared resource-lease machinery proposed in [#917](https://github.com/tsotchke/eshkol/pull/917). Extend that shared machinery for caller-supplied loops and sticky cancellation; keep one lease implementation and coordinate with PROCESS watches. Datagrams, descriptor passing and SSE remain outside this slice.

## Agreed surface and progress

Prefix: `unix-stream-socket-`. Align scaffold declarations and tests with this contract; implementation evidence is pending.

| Operation | Contract |
|---|---|
| `listen loop path`; `accept listener deadline-ms` | Require an explicit, live event-loop handle (`0` is invalid); peers inherit it. Listen is setup-only; descriptors are nonblocking/close-on-exec before publication. |
| `connect-begin loop path deadline-ms`; `connect-finish! id` | Return a connecting SOCKET before waiting, enabling cancellation; finish uses the deadline captured at begin. |
| `read!/write! id buffer start count` | Nonblocking, positive checked ranges; exact-count `data`, `would-block`, read `eof`, or operation-labelled error with portable/OS codes. |
| `poll id interests timeout-ms`; `cancel! id`; `close! id` | Checked readiness/timeout/cancellation/error; cancellation request; ownership release. |

Timeouts are exact 0..INT32_MAX milliseconds. Absolute monotonic deadlines never renew on retry/interruption. Accept/poll zero never wait; zero-budget begin returns connected or rolls back with would-block. Begin returns the published owner before readiness waiting; failures before publication roll back acquisition. Its deadline caps finish/poll **while connecting**; connected I/O has independent poll budgets. Reject read/write until connected. Missing/refused destinations return immediately; retry backlog-full `EAGAIN` in finish/poll with bounded loop waits, after begin has returned the handle. Verify completion using `SO_ERROR`/peer state.[4] Clock errors preserve published ownership; failed published attempts still require close.

Both backends are level-triggered: arm requested interests and remove temporary interests after waiting. After `would-block`, callers use `poll` with read/write interests and a bounded timeout, then retry from the remaining byte offset; readiness can still be followed by another `would-block`. The caller retains the bytevector and advances its offset only by a successful byte count. Yield after bounded work. Caller buffers preserve NULs/large payloads without retained output. Drain buffered input before EOF. Report observed peer-close/EPIPE errors; suppress SIGPIPE per socket/send (`SO_NOSIGPIPE`/`MSG_NOSIGNAL`), preserving process signal disposition.[2], [5]

## Ownership, cancellation and cleanup

Each listener/peer has one generation-checked SOCKET owner in the shared lease layer. Operations borrow exclusively; a second borrower gets busy. The caller supplies a loop; each listener/socket retains it through watch removal and socket teardown. Loop close refuses while resources retain it. Ordinary I/O and watch teardown run on the loop owner thread; cross-thread requests only update lifetime-safe state and signal that thread. Shared wait machinery must support that loop and preserve readiness for other resources, including PROCESS watches. Validate readiness identities across fd reuse. Runtime ownership prepares future compile-time linear handles; it does not implement them.

Idempotent, lifetime-safe cross-thread cancellation remains set for the socket, wakes pending work and prevents further I/O; the caller still closes it. Close during a borrow is accepted: reject new uses immediately, wake the operation, and close the descriptor exactly once after release. The low-level registry's ordinary close can return busy for a borrow; the public socket lease uses shared close-after-release instead. A close request is not proof teardown finished. Use the shared lease layer; define observable watch-removal and path-cleanup outcomes before implementation, retaining safe ownership until cleanup resolves. Scoped `dynamic-wind` cleanup rejects re-entry. Checked/native acquisition and I/O require `network`; owned-resource cancel/close need neither renewed capability nor a clock. They still validate the numeric identity, kind, tag and generation. Scoped exception cleanup follows the same release path. Cancellation cannot retract bytes already transferred; an operation completed before the request may report its successful byte count.

## Implementation ownership

Gabe owns the common lease/wait extensions and Unix socket core. The scaffold lane owns the corresponding API declarations and acceptance-test amendments in #916. Extend one common facility while preserving existing documented TCP/UDP/TLS behavior. Before implementation, synchronize the socket work with the current relevant Eshkol integration base and amended scaffold. Start with listen, connect/accept, a byte-counted round trip and complete close/path cleanup.

## Paths and platforms

Require an absolute, NUL-free path fitting `sun_path` in encoded bytes, inside a verified owned private 0700 directory. Bind, chmod 0600, then listen/publish; roll back setup errors without global umask changes. Refuse existing entries, including restart leftovers; require explicit owner removal. Only the listener unlinks its recorded entry during close/rollback, never replacements/client destinations. Report unlink errors for repair; never retry uncertain fd close.[1]

**Abstract namespace: no**, preserving portable permissions/cleanup.[1] **Windows:** after Winsock initialization, probe `socket(AF_UNIX, SOCK_STREAM, 0)`, close it, and distinguish unsupported-family from other errors. This slice remains explicitly unsupported without a readiness adapter.[3] **VM:** explicitly reject the bridge until adapted.

## Acceptance

Native/JIT/AOT: round trips; actual slow-peer partial I/O; byte-for-byte NUL/>16 MiB transfer; peer close with default SIGPIPE; measured zero/positive deadlines and pending-connect cancellation; measured descriptor counts over 1,000 cycles; path restart/replacement; borrow/deferred-close/expired-ID, sticky cancellation and exception/re-entry/revocation cleanup. Record Linux/macOS evidence separately. Shared-layer tests cover loop lifetime, cancellation/wakeup and deferred close. Socket tests cover readiness after `would-block`, pending-connect cancellation and cleanup after capability revocation, including native/JIT/AOT cases. Corrected tests and their implementation land together; enable passing cases in the default suite as each capability lands. Unfinished cases may retain the explicit pending label; disabled cases do not establish acceptance.

[1]: https://man7.org/linux/man-pages/man7/unix.7.html
[2]: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/setsockopt.2.html
[3]: https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-socket
[4]: https://man7.org/linux/man-pages/man2/connect.2.html
[5]: https://man7.org/linux/man-pages/man2/send.2.html
