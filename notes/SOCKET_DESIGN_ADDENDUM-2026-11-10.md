# Proposed decision: Unix-domain stream sockets

**Status:** Proposed design; scaffold amendments await owner review. **Date:** 2026-11-10.
**Basis:** [IPC design](https://github.com/Gabriel-Kahen/eshkol/blob/93c02a9679b9476c58296bfeb6e229d251a21224/weeks/IPC_DESIGN-2026-11-02.md); [#916 at `da7005c7f`](https://github.com/tsotchke/eshkol/pull/916). Proposed behavior, not implementation evidence.

**Companion review:** [#916 scaffold findings and proposed resolutions](SOCKET_DESIGN_REVIEW-2026-11-10.md) explain the contract gaps and acceptance-test issues behind this proposal.

## Context and decision

Provide filesystem `AF_UNIX/SOCK_STREAM` sockets using existing epoll/kqueue, registry and policy facilities. Extend managed-watch dispatch for SOCKET owners alongside PROCESS. Datagrams, descriptor passing and SSE remain outside this slice.

## Proposed surface and progress

Prefix: `unix-stream-socket-`. Revise scaffold declarations/tests to match these proposed amendments.

| Operation | Contract |
|---|---|
| `listen loop path`; `accept listener deadline-ms` | Retain an explicit loop; peers inherit it. Listen is setup-only; descriptors are nonblocking/close-on-exec before publication. |
| `connect-begin loop path deadline-ms`; `connect-finish! id` | Publish a connecting SOCKET before waiting, enabling cancellation; finish uses its stored deadline. |
| `read!/write! id buffer start count` | Nonblocking, positive checked ranges; exact-count `data`, `would-block`, read `eof`, or operation-labelled error with portable/OS codes. |
| `poll id interests timeout-ms`; `cancel! id`; `close! id` | Checked readiness/timeout/cancellation/error; cancellation request; ownership release. |

Timeouts are exact 0..INT32_MAX milliseconds. Absolute monotonic deadlines never renew on retry/interruption. Accept/poll zero never wait; zero-budget begin returns connected or rolls back with would-block. Positive begin publishes an owner before waiting. Its deadline caps finish/poll **while connecting**; connected I/O has independent poll budgets. Reject read/write until connected. Missing/refused destinations return immediately; retry backlog-full `EAGAIN` with bounded loop waits. Verify completion using `SO_ERROR`/peer state.[4] Clock errors preserve published ownership; failed published attempts still require close.

Both backends are level-triggered: arm requested interests, remove writable interest after waiting, resume at the remaining byte offset, and yield after bounded work. Caller buffers preserve NULs/large payloads without retained output. Drain buffered input before EOF. Report observed peer-close/EPIPE errors; suppress SIGPIPE per socket/send (`SO_NOSIGPIPE`/`MSG_NOSIGNAL`), preserving process signal disposition.[2], [5]

## Ownership, cancellation and cleanup

Each listener/peer has one generation-checked SOCKET owner: acquire, exclusively borrow, release, close; borrowed close returns busy. Retain the loop until unregister/close; shutdown rejects active ownership. Validate readiness cookies across fd reuse. Runtime ownership prepares future compile-time linear handles; it does not implement them.

I/O stays owner-thread confined. Idempotent, lifetime-safe cross-thread cancellation sets a sticky flag and wakes the loop without borrowing/closing the fd. Cancellation ends further I/O; ownership remains. Unregister before registry close; removal errors retain ownership for retry. Once fd close is attempted, the identity is terminal; repeated close returns invalid. Scoped `dynamic-wind` cleanup rejects re-entry. Checked/native acquisition and I/O require `network`; owned-resource cancel/close need neither renewed capability nor a clock.

## Paths and platforms

Require an absolute, NUL-free path fitting `sun_path` in encoded bytes, inside a verified owned private 0700 directory. Bind, chmod 0600, then listen/publish; roll back setup errors without global umask changes. Refuse existing entries, including restart leftovers; require explicit owner removal. Only the listener unlinks its recorded entry during close/rollback, never replacements/client destinations. Report unlink errors for repair; never retry uncertain fd close.[1]

**Abstract namespace: no**, preserving portable permissions/cleanup.[1] **Windows:** after Winsock initialization, probe `socket(AF_UNIX, SOCK_STREAM, 0)`, close it, and distinguish unsupported-family from other errors. This slice remains explicitly unsupported without a readiness adapter.[3] **VM:** explicitly reject the bridge until adapted.

## Acceptance

Native/JIT/AOT: round trips; actual slow-peer partial I/O; byte-for-byte NUL/>16 MiB transfer; peer close with default SIGPIPE; measured zero/positive deadlines and pending-connect cancellation; measured descriptor counts over 1,000 cycles; path restart/replacement; borrow/expired-ID and exception/re-entry/revocation cleanup. Record Linux/macOS evidence separately; correct scaffold assertions before promotion.

[1]: https://man7.org/linux/man-pages/man7/unix.7.html
[2]: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/setsockopt.2.html
[3]: https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-socket
[4]: https://man7.org/linux/man-pages/man2/connect.2.html
[5]: https://man7.org/linux/man-pages/man2/send.2.html
