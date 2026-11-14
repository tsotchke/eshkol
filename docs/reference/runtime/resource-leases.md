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
| `-12` | `ESHKOL_IO_DENIED` | The capability policy does not permit the operation. |

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
