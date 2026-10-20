# TCP and UDP

`(require core.net)` exposes bounded native POSIX sockets for numeric IPv4 and
IPv6 addresses. DNS, TLS and standalone VM FFI integration are not implemented
by this module. Windows and WASM return error code `-3` (unsupported) rather
than pretending to create a socket. Native Linux and macOS use nonblocking
kernel sockets with monotonic deadlines.

| Procedure | Result |
|---|---|
| `(tcp-listen address port)` | Listener handle; port zero chooses an ephemeral port. Backlog is 64. |
| `(tcp-connect address port timeout-ms)` | Connected handle. |
| `(tcp-accept listener timeout-ms)` | A separate connected handle. |
| `(tcp-read handle capacity timeout-ms)` | Up to capacity bytes as a string; `""` means EOF. |
| `(tcp-write handle data timeout-ms)` | Bytes sent, potentially fewer than the input byte length. |
| `(udp-bind address port)` | Datagram handle. |
| `(udp-sendto handle address port data timeout-ms)` | Datagram byte count. |
| `(udp-recvfrom handle capacity timeout-ms)` | `#(payload sender-endpoint)`. IPv6 endpoints use `[address]:port`. |
| `(net-local-address handle)`, `(net-peer-address handle)` | Numeric address string. |
| `(net-local-port handle)`, `(net-peer-port handle)` | Exact integer port. Peer queries require a connected socket. |
| `(net-nonblocking! handle enabled)` | Zero on success; enabled handles ignore wait budgets and return immediately. |
| `(tcp-close handle)`, `(udp-close handle)` | Zero on success. Close applies to either handle family. |

Failures return an opaque error object. Inspect it with `network-error?`,
`network-error-code`, and `network-error-operation`:

| Code | Meaning |
|---|---|
| `-1` | System/network failure, including connection refusal or allocation failure. |
| `-2` | Timeout or would-block; distinct from TCP EOF and a zero-length UDP datagram. |
| `-3` | Unsupported platform. |
| `-4` | Invalid argument, stale/wrong-kind handle, or oversized received datagram. |

Counts, ports and timeout budgets must be exact integers in range. Timeouts
range from zero through 2,147,483,647 milliseconds. Reads require positive
capacity (TCP at most 1 MiB; UDP at most 65,535 bytes); writes accept empty
strings and preserve embedded NUL bytes. TCP writes are partial: use the
returned **byte count** to send the remaining bytes. A short TCP read is normal.
UDP sends are limited to 65,507 bytes; a datagram exceeding receive capacity
is consumed and rejected, never silently returned as a truncated payload.

Handles are process-local, opaque positive integers, distinct from OS file
descriptors. There are 256 simultaneously allocated slots and at most 2^53 - 1 handle
allocations per process; exhaustion refuses further creation. Handles are never
reused, so a stale close cannot close a later socket that reused its OS
file descriptor. Close invalidates a handle immediately; in-flight operations
retain the underlying descriptor until completion. It is not a cancellation
API: blocked operations may return EOF/error or wait until their own deadline.
The caller must close all handles before process exit. They cannot survive
serialization or be transferred to another process.

The Scheme network operations enforce `core.capabilities` network policy;
endpoint queries and cleanup remain available after revocation. Addresses are
explicit: `127.0.0.1`/`::1` are loopback, while `0.0.0.0`/`::` expose listeners
on every interface. No address or service is contacted implicitly.

`python3 scripts/run_net_gate.py` runs public Scheme TCP/UDP roundtrips with a
30-second process deadline. `net_runtime_test` separately uses independent
POSIX peers, checks IPv6, real timeout timing and 1,000 close/reopen cycles.
Unsupported-platform test builds verify refusal; they do not certify Windows
or WASM networking support.
