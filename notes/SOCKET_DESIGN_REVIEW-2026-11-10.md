# Review of #916: socket scaffold and design alignment

**Date:** 2026-11-10. **Scope:** static review only; no socket implementation or test execution.
**Draft:** [Socket design addendum](SOCKET_DESIGN_ADDENDUM-2026-11-10.md).
**Scaffold reviewed:** [#916](https://github.com/tsotchke/eshkol/pull/916), `da7005c7f23ee12f80e2f2fd07994b518d3f5532`. This document records the findings; it does not constitute a submitted GitHub PR review.

## Assessment

The draft covers the November 9 plan's required topics. It proposes changes to the scaffold's public API and acceptance tests; it does not claim those changes already exist. The three decisions below are agreed; the pending-connect API and implementation details still need alignment with the component owner and scaffold. The original IPC ownership and cleanup rules remain the architectural basis.

## Agreed direction — 2026-11-11

The caller supplies the loop and each socket retains it; the shared wait machinery must support it. Cancellation stays set on the socket, wakes pending work and blocks future I/O; the caller still closes it. Close during active I/O is accepted, immediately invalidates the identity, wakes the borrower, and closes the descriptor exactly once after release. A second borrower still receives busy. These decisions supersede the original busy-close proposal below.

Use the common adoption, borrow/release, wait, cancellation, close and outcome machinery from [#917](https://github.com/tsotchke/eshkol/pull/917); extend it where needed rather than introducing a second lease implementation. At #917 head `12d437bb6b9fedce0c6e822d15ecb2db473f560b`, wait uses a per-thread loop and cancellation is consumed, so caller-supplied loop and persistent cancellation require integration work. The shared close accepts deferred teardown; close acceptance and completed cleanup are distinct. The exact reporting/recovery of watch-removal and socket-path unlink failures remains to be designed and tested in the shared facility. Pending-connect cancellation still requires a reachable identity or equivalent cancellation target; begin/finish is one proposed API.

The findings below describe the original #916 scaffold at the reviewed head. The scaffold lane owns its amendments; this record does not claim they have landed.

## Original contract gaps and proposed resolutions

1. **Define loop ownership and readiness waiting.** The current Scheme listen/connect wrappers pass literal zero, with no definition of the implicit loop's lifetime. There is no socket poll/wait operation. The draft proposes passing a real loop at acquisition, retaining it per socket, and adding `poll` for callers receiving would-block. A documented runtime-owned loop could also satisfy the lifetime requirement; the explicit-loop API is a proposal. Tests should verify the chosen loop lifetime and deferred-close behavior while sockets remain owned. Socket readiness must coexist with #914's managed PROCESS watches rather than bypass their cookie/ownership checks. Sources: `lib/agent/unix_stream_socket.esk:44–65`; `inc/eshkol/core/event_loop.h`; `lib/agent/c/agent_subprocess.c`.

2. **Make pending-connect cancellation reachable.** The current C `connect` returns the identity only after it completes; `cancel(id)` cannot address a pending connect. The draft proposes `connect-begin`/`connect-finish!` and one deadline captured in begin; another interface with a reachable cancellation target could also resolve the gap. Under the proposed interface, begin publishes without a readiness wait and finish does not reset the budget. If begin cannot publish, undo fd/watch/lease acquisition. Once an identity is returned, the caller retains its close obligation even after timeout/cancellation. Sources: `inc/eshkol/core/unix_socket_stream.h:49–61`; [connect(2)](https://man7.org/linux/man-pages/man2/connect.2.html).

3. **Support a cancellation request without concurrent descriptor mutation.** The native acceptance test calls cancel from a second pthread. PROCESS's owner-thread confinement cannot be copied unchanged. The draft permits only a synchronized, generation-validated cancellation request and loop wakeup from that thread; I/O, watch changes and close remain on the owner thread. Lifetime protection must cover the cancellation record until the request returns. Source: `tests/agent/unix_socket_stream_test.c:144–161`.

4. **Preserve cleanup after policy revocation.** The scaffold requires `network` even for cancel/close. That can strand an already-owned socket when policy is revoked. Gate acquisition and I/O, including real native acquisition, but permit owned-resource cancellation/close without reacquiring capability. Verify the identity still belongs to this runtime. Sources: `lib/agent/unix_stream_socket.esk:110–121`; existing IPC design, “Policy, platforms, and evidence.”

5. **Confirmed backend behavior.** The current epoll backend has no `EPOLLET`; kqueue uses `EV_ADD | EV_ENABLE` without `EV_CLEAR`. The draft therefore uses level-triggered retry with bounded work and temporary writable interest. This is architectural context, not a requested correction to the existing backends. Any later backend change needs corresponding progress tests. Sources: `lib/core/event_loop_epoll.c:74–80`; `lib/core/event_loop_kqueue.c:101–127`.

## Acceptance cases to repair before implementing against them

| Case | Current assertion or setup | Required correction |
|---|---|---|
| Large transfer / slow peer | Native writes all >16 MiB before the receiver reads and rejects would-block. Scheme checks one write status without checking received bytes. | Interleave a slow receiver and sender; handle readiness and short counts, enforce a finite test deadline, compare every received byte, and measure retained memory. |
| Connect deadline | Native accepts either timeout or a generic system error. Scheme expects a successful identity when the destination is absent. | Keep absent/refused-destination tests separate. Use a controlled pending connection/backlog or deterministic test hook to exercise the actual deadline; assert the specific result and elapsed time. Cover time between begin and finish. |
| Cancellation | Scheme expects the request itself to raise; it starts no blocked operation and leaves the listener open. | Require a successful idempotent request, observe canceled accept/connect/readiness separately, and close every owner. Test repeated cancellation and a request arriving from another thread. |
| Path lifecycle | Scheme expects listen to fail immediately after successful listener close has removed the path. | Verify successful restart after normal close. Independently create a leftover socket entry and verify refusal without removal; test replacement protection and unlink-error reporting. |
| SIGPIPE / peer close | Native harness globally ignores SIGPIPE; one immediate write is required to report EPIPE. | Run with default signal disposition in an isolated child. Drive I/O until peer closure is observed within a bounded interval; verify the operation error and process survival. Earlier successful sends do not imply peer receipt. |
| Descriptor counts | The fixture repeats 1,000 open/close cycles but never measures counts. | Compare actual descriptor counts before/after, allowing only documented baseline changes. Include acquisition rollback and cancellation paths. |
| Error identity | Scheme `raises?` accepts any exception, potentially accepting unrelated/planned results. | Assert the intended operation and portable error code; do not treat the planned sentinel or another exception as contract evidence. |

Locations: native `tests/agent/unix_socket_stream_test.c:78–99,104–142,164–215`; Scheme `tests/agent/unix_socket_stream_pending.esk:31–108,124–130`.

## Draft decisions and remaining review questions

- **Direction and proposals:** caller-supplied loops, sticky cancellation and deferred close are agreed above; begin/finish connect, private pathname sockets, no abstract namespace, and explicit unsupported Windows/VM adapters remain design proposals.
- **Independent review corrections incorporated:** zero-budget begin either connects immediately or rolls back without publishing an ID; a positive pending begin publishes its owner before waiting. The acquisition deadline applies only while connecting; connected I/O gets independent poll budgets. Failed published attempts remain owned until close, and reads/writes require connected state. Clock errors preserve ownership, while cleanup needs neither a clock nor renewed capability.
- **Cleanup ordering:** A close request immediately invalidates the identity and defers teardown until the borrower releases. Ensure watches are safely removed before descriptor disposal, and define how cleanup failures remain observable; never retry a descriptor after an uncertain close. Validate directory ownership/privacy, bind, set 0600, then listen/publish; every setup error must undo unpublished resources.
- **Tradeoff:** the agreed explicit loop and proposed begin/finish API require scaffold changes. Begin/finish is one proposed way to make pending-connect cancellation reachable; its exact interface has not been settled.
- **Platform boundary:** Windows socket-family probing must preserve non-support errors as distinct from resource/initialization errors. An available family does not establish readiness-backend support. Linux/macOS runtime tests are separate evidence; no platform support is inferred from static review.
- **Filesystem boundary:** parent-directory privacy is part of the contract. Checking an inode and then unlinking is not a race-free deletion primitive in an adversarial shared directory; the draft does not support such a directory. Socket permissions alone are not a portable access-control boundary.
- **Evidence boundary:** these documents present a reviewed design proposal and source-review findings. They do not claim scaffold changes, a socket implementation, runtime test results, or sign-off on the remaining API details. The original #916 review was submitted separately.

## Sources checked

- [Existing IPC design](https://github.com/Gabriel-Kahen/eshkol/blob/93c02a9679b9476c58296bfeb6e229d251a21224/weeks/IPC_DESIGN-2026-11-02.md).
- [Weekly plan](https://drive.google.com/file/d/1aI3gm47ZRgFjmWCUv8cbctsVUTxKQsuG/view), socket addendum row; required topics checked against the previously fetched text.
- Scaffold files at the exact head above, including header, Scheme wrapper, native/Eshkol fixtures, common handle contract and both event-loop backends.
- [unix(7)](https://man7.org/linux/man-pages/man7/unix.7.html), [connect(2)](https://man7.org/linux/man-pages/man2/connect.2.html), [send(2)](https://man7.org/linux/man-pages/man2/send.2.html), [Darwin setsockopt(2)](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/setsockopt.2.html), [Windows socket](https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-socket).
