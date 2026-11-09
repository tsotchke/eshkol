---
kind: reference
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - lib/core/net.esk
  - lib/core/net.c
  - lib/core/resource_lease.c
  - inc/eshkol/core/net.h
  - inc/eshkol/core/resource_lease.h
---

# TCP and UDP

`(require core.net)` exposes native sockets for numeric IPv4 and IPv6
addresses. Every socket is a SOCKET-kind handle in the runtime's shared
resource registry (see [resource leases](../runtime/resource-leases.md)):

- one operation holds a handle at a time (an exclusive lease);
- every operation has one monotonic deadline, shared by all of its retries;
- waits go through the calling thread's event loop (kqueue on macOS, epoll on
  Linux), so a close or cancellation from another thread wakes them at once;
- every handle closes exactly once, at a single close site per kind, and the
  scoped forms close it on every exit path.

DNS is not performed: addresses are numeric. TLS lives in [core.tls](tls.md).
Native JIT and AOT programs on Linux and macOS run sockets. Windows sockets are
Planned for v1.4.1 and return error code `-3` (unsupported) until then, as do
WebAssembly builds. The bytecode VM refuses `core.net` at compile time with
`core.net and core.tls are not supported by the bytecode VM`.

## Handles and their kinds

| Kind | Acquire | Close | Scoped form |
|---|---|---|---|
| TCP listener | `(tcp-listen address port)` | `tcp-close` | `(call-with-tcp-listener address port proc)` |
| TCP stream | `(tcp-connect address port timeout-ms)`, `(tcp-accept listener timeout-ms)` | `tcp-close` | `(call-with-tcp-connection address port timeout-ms proc)`, `(call-with-tcp-accept listener timeout-ms proc)` |
| UDP socket | `(udp-bind address port)` | `udp-close` | `(call-with-udp-socket address port proc)` |

`tcp-close` refuses a UDP handle and `udp-close` refuses a TCP handle. A second
close of the same handle, or a close of a stale handle, returns error `-4`.
Handles are exact positive integers no larger than 2^53 - 1, distinct from OS
descriptors, and never reused, so a stale handle cannot reach a later socket
that received the same descriptor. They cannot survive serialization or move
to another process. The descriptor limit of the process bounds how many are
open at once; reaching it returns error `-1` and closing a handle frees a slot.

A scoped form acquires the handle, calls `(proc handle)`, and closes the handle
exactly once when `proc` returns, raises, or escapes through a continuation.
It returns `proc`'s value; an exception or continuation escape keeps its
original value. Re-entering a finished scope through a saved continuation
raises a network error with code `-4` before `proc` runs again. If acquisition
fails, the scoped form returns that network error without calling `proc`.

## Byte-counted I/O

These procedures read and write a range `[start, start + count)` of a caller
bytevector and never copy through strings.

| Procedure | Result |
|---|---|
| `(tcp-read! stream bv start count timeout-ms)` | Bytes received (at least one), the eof object at end of stream, or an error. `count` must be positive. |
| `(tcp-write! stream bv start count timeout-ms)` | Bytes accepted by the kernel, possibly fewer than `count`. |
| `(tcp-write-all! stream bv start count timeout-ms)` | `count` once every byte is sent within one deadline, or an error. |
| `(tcp-shutdown stream how)` | Zero; `how` is `'read`, `'write` or `'both`. The peer of a `'write` shutdown reads end of stream. |
| `(udp-send-to! socket address port bv start count timeout-ms)` | Datagram byte count (at most 65,507). |
| `(udp-receive-from! socket bv start count timeout-ms)` | `#(byte-count sender truncated?)`. A longer datagram is consumed: its first `count` bytes are stored, `byte-count` is `count` and `truncated?` is `#t`. |
| `(net-cancel! handle)` | Zero. Cancels the operation waiting on `handle` from any thread, or else the next operation to wait; that operation returns error `-6`. Cancellation is consumed once. |

The string procedures remain available: `(tcp-read handle capacity timeout-ms)`
returns a string (`""` at end of stream), `(tcp-write handle data timeout-ms)`
a partial byte count, `(udp-sendto handle address port data timeout-ms)` a
byte count, and `(udp-recvfrom handle capacity timeout-ms)` a vector
`#(payload sender)`; it refuses a datagram longer than `capacity` with error
`-8` after consuming it. String reads take at most 1 MiB per call.

Endpoint queries: `(net-local-address h)`, `(net-peer-address h)` return a
numeric address; `(net-local-port h)`, `(net-peer-port h)` an exact port.
Senders render as `address:port`, with brackets for IPv6. `(net-nonblocking! h
#t)` makes every later wait on `h` a single readiness pass.

## Deadlines

A timeout is an exact integer from zero to 2,147,483,647 milliseconds. It
becomes one monotonic deadline when the operation starts; retries after a
signal or a partial transfer use the time that remains and never restart it. A
zero timeout performs one nonblocking readiness pass and returns error `-2`
(would-block) instead of sleeping. A read or write that reaches its deadline
leaves the handle usable.

## Outcomes

Failures return an error object; inspect it with `network-error?`,
`network-error-code`, `network-error-operation` and `network-error-reason`:

| Code | Reason | Meaning |
|---|---|---|
| `-1` | `system` | Operating-system failure, including descriptor exhaustion. |
| `-2` | `timeout` | Deadline reached; a zero deadline reports would-block. |
| `-3` | `unsupported` | No socket support on this platform or build. |
| `-4` | `invalid` | Invalid argument, out-of-range bytevector range, or a stale, closed or wrong-kind handle. |
| `-5` | `busy` | Another operation (or a TLS session) holds the handle. |
| `-6` | `cancelled` | Cancelled with `net-cancel!`, or closed by another thread during the operation. |
| `-7` | `reset` | The peer reset the connection or stopped reading. |
| `-8` | `truncated` | A datagram exceeded the string receive capacity. |
| `-9` | `refused` | The peer refused the connection. |

## Concurrency

Handles may be used from any thread, one operation at a time. A second
operation on a handle that is in use returns `-5` instead of waiting. `tcp-close`
or `udp-close` from another thread makes the handle unusable at once, wakes the
operation in progress (which returns `-6`), and closes the descriptor when that
operation ends. Use `net-cancel!` to stop an operation and keep the handle.

## Capabilities

Creating sockets and moving bytes require the `network` capability from
`core.capabilities`; acquisition also checks the address and port. Endpoint
queries, cancellation and close remain available after the policy changes, so
owned sockets can always be released. Addresses are explicit: `127.0.0.1` and
`::1` are loopback, while `0.0.0.0` and `::` expose a listener on every
interface.

## Static ownership checking

`eshkol-run --strict-resources` recognizes the acquire and close sites above
for the stock `core.net` module (see
[strict resource ownership](../../development/strict_resource_ownership.md)).
The runtime guarantees hold without it.

## Evidence

- `net_lease_test` (C): partial writes against a slow peer, peer reset, read,
  accept and connect deadlines, cancellation from another thread, busy leases,
  shutdown, kind-checked close, truncation, and an unchanged descriptor count
  after 1,000 connection cycles.
- `net_core_jit`, `net_core_aot`: the same behaviours through the Scheme surface,
  plus guaranteed close on return, exception and continuation escape, re-entry
  refusal, and network capability denial.
- `net_close_concurrency_test`: close during a blocked accept, read and receive
  cancels the operation promptly; descriptor exhaustion refuses and recovers.
- `net_runtime_test`, `net_roundtrip_jit`, `net_roundtrip_aot`,
  `net_clock_failure_test`: independent POSIX peers, IPv6, and clock failure.
- `net_vm_refusal`: the bytecode VM refusal.
