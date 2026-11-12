---
kind: reference
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - inc/eshkol/core/resource_lease.h
  - lib/core/resource_lease.c
  - inc/eshkol/core/handle.h
  - lib/core/runtime_handle.cpp
---

# Resource leases

Hosted I/O resources (TCP and UDP sockets, TLS sessions' streams, and the local
IPC kinds that follow the same contract) are owned through one process-wide
registry of generation-tagged handles (`eshkol/core/handle.h`). The small layer
in `eshkol/core/resource_lease.h` is the contract every resource kind shares:

| Concern | Rule |
|---|---|
| Acquire | `eshkol_resource_adopt(kind, tag, native, close_fn, data)` is the one acquire site: it moves sole ownership of the native resource into the registry and returns an exact identity (at most 2^53 - 1, never reused). On failure ownership stays with the caller. |
| Kinds | The registry kind (`SOCKET`, `FILE`, `EVENT_LOOP`, ...) plus an immutable tag (TCP listener, TCP stream, UDP socket, ...) let every operation check the handle it receives. |
| Borrow | Each operation borrows exclusively (`eshkol_resource_borrow`) and releases when done. A second borrower gets `ESHKOL_IO_BUSY` instead of waiting. |
| Deadline | One monotonic deadline per operation (`eshkol_resource_deadline`), shared across retries and phases; a zero timeout is one nonblocking pass. |
| Wait | `eshkol_resource_wait` waits for readiness on the calling thread's event loop (kqueue, epoll), together with a wake pipe. Each thread creates its loop on first use and releases it at thread exit. |
| Cancel | `eshkol_resource_cancel` from any thread wakes the borrower, whose wait returns `ESHKOL_IO_CANCELLED`; with no borrower, the next wait observes it once. |
| Close | `eshkol_resource_close` is the one close site. An open handle closes now. A borrowed handle becomes unusable at once, its borrower is woken, and the close callback runs exactly once when that borrow ends. Retired, repeated and other-kind closes return `ESHKOL_IO_INVALID`. |
| Bytes | `eshkol_resource_bytes` validates a `[start, start + count)` range of an Eshkol bytevector for byte-counted I/O. |

## Outcome codes

A nonnegative result is an exact byte count or a new identity; a stream read
of zero bytes for a positive request is end of stream. Failures are shared by
every kind:

| Code | Name | Meaning |
|---|---|---|
| `-1` | `ESHKOL_IO_SYSTEM` | Operating-system failure. |
| `-2` | `ESHKOL_IO_TIMEOUT` | Deadline reached (would-block for a zero deadline). |
| `-3` | `ESHKOL_IO_UNSUPPORTED` | Not available on this build or platform. |
| `-4` | `ESHKOL_IO_INVALID` | Invalid argument or a retired, closed or other-kind handle. |
| `-5` | `ESHKOL_IO_BUSY` | Another operation holds the lease. |
| `-6` | `ESHKOL_IO_CANCELLED` | Cancelled, or closed during the operation. |
| `-7` | `ESHKOL_IO_RESET` | The peer reset the connection or closed its reader. |
| `-8` | `ESHKOL_IO_TRUNCATED` | A datagram exceeded the receive capacity. |
| `-9` | `ESHKOL_IO_REFUSED` | Connection refused. |
| `-10` | `ESHKOL_IO_UNTRUSTED` | TLS peer certificate failed verification. |
| `-11` | `ESHKOL_IO_IDENTITY` | TLS peer certificate names another host. |

## Registry operations behind the layer

`eshkol/core/handle.h` provides the primitives: `create_tagged` and `get_kind`
for tags, `borrow`/`release`, `close` (which returns `EBUSY` for a borrowed
handle), `close_after_release` (which schedules the close for the borrower's
release and wakes it), and `interrupt`, `set_waker` and `take_interrupt` for
cancellation. Wakers run under the registry lock and only signal. Exact
registries (`eshkol_handle_registry_create_exact`) retire a slot after
2^21 - 1 generations so identities stay exact in binary64.

## Scheme scopes

Resource modules expose a scoped form for each acquire site, built on
`dynamic-wind`: the handle closes exactly once on return, exception or
continuation escape, and re-entry into a finished scope raises before the
receiver runs. See [core.net](../stdlib/net.md) and [core.tls](../stdlib/tls.md).
These runtime guarantees are independent of the static checks in
`eshkol-run --strict-resources`.


## Explicit-loop owners and cleanup results

Adapters opt into `eshkol_resource_adopt_on_loop`, retaining a
caller-supplied live EVENT_LOOP identity and enforcing owner-thread I/O.
`eshkol_resource_wait_on_loop` uses the same event-loop backends, validates
readiness cookies and preserves unrelated PROCESS/SOCKET registrations.
An adopting adapter can select sticky cancellation; existing TCP/UDP/TLS operations
keep their documented consumable requests and per-thread waits.

The registry accepts distinct EVENT_LOOP, SOCKET, PROCESS and CLEANUP kinds.
Every shared-layer operation checks kind and tag. An adapter's shutdown must
retire only its own records, never destroy the process-lifetime registry or
another adapter's resources. Published platform kind values remain WINDOW4,
AUDIO5 and MIDI6; PROCESS is7 and CLEANUP is8. PROCESS streaming migration and
the Unix socket adapter are separate changes built on this facility.

`eshkol_resource_close_request` retires a managed owner's identity and returns a
separate CLEANUP ticket. Ticket capacity is reserved before publishing the
owner, so accepting close does not allocate. The registry close callback queues
owner-thread work after outstanding borrows end. Loop maintenance and
`eshkol_resource_cleanup_result` on the owner thread drive teardown without a
clock. Foreign-thread result inspection only observes.

A result records pending/complete/failed, the watch/descriptor/path/loop stage,
OS detail and whether retry is safe. `eshkol_resource_cleanup_retry` explicitly
retries recoverable failed teardown after repair; `eshkol_resource_cleanup_release`
releases a terminal result. An uncertain descriptor close is never retried.
The convenience `eshkol_resource_close` detaches this result: success is released
automatically and a failed result is recoverable through
`eshkol_resource_cleanup_ticket` using the retired owner identity. It grants no
new I/O authority. The cleanup callback owns terminality: the shared layer never
times out or automatically retries a recoverable failure. Each adapter must
provide a documented route to a terminal result, including a persistent-failure
case. A terminal ticket may outlive loop destruction because it retains only
result data; pending/retryable cleanup retains the loop.

Routine legacy TCP/UDP/TLS operations dispatch through immutable registry flags
without the managed-resource lock. Managed identities and tickets use a full-ID
hash index, preserving outstanding failed results across registry-slot reuse.
Release pumps only an owner's newly ready cleanup; loop maintenance traverses
queued work rather than all active resources.
