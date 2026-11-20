# ADR-0055: Sockets and TLS sessions as resource leases

- **Status:** Proposed
- **Date:** 2026-11-08
- **Decision owners:** Eshkol hosted runtime
- **Applies to:** `inc/eshkol/core/resource_lease.h`, `lib/core/resource_lease.c`,
  `inc/eshkol/core/handle.h`, `lib/core/runtime_handle.cpp`, `lib/core/net.c`,
  `lib/core/tls.c`, `lib/core/net.esk`, `lib/core/tls.esk`
- **Relates to:** the v1.4.0 handle registry (`docs/design/V1_4_CONNECTION_START_2026_10_19.md`),
  the event loop (`docs/reference/runtime/event-loop.md`), the local IPC design
  for subprocess streams and Unix-domain sockets, and the compile-time linear
  handle work in v1.4.0

## Context

`core.net` kept its own table of 256 socket slots with reference counts and
waited with `poll(2)` per call. `core.tls` layered sessions over those handles.
The runtime also had a generation-tagged handle registry with exclusive
borrows, unused by any resource. Three needs met here:

1. The v1.4.0 roadmap asks for sockets with guaranteed close, on the shipped
   event loop, and for TLS through system libraries.
2. Local IPC (subprocess streams, Unix-domain sockets) follows the same
   ownership rules and must not grow a second contract.
3. Compile-time linear handle tracking needs one acquire site and one close
   site per handle kind.

## Decision

**One registry, one small lease layer.** Every hosted I/O resource is adopted
into one process-wide exact registry (identities at most 2^53 - 1, never
reused). `resource_lease.h` is the contract every kind uses: adopt (the acquire
site), kind- and tag-checked exclusive borrow and release, close (the close
site), cancel, one monotonic deadline per operation, readiness waits on the
calling thread's event loop, byte-counted bytevector ranges, and one table of
outcome codes (`ESHKOL_IO_*`, -1 through -11) shared by sockets, TLS and IPC.

**Close never invalidates an active borrower.** The registry keeps its rule
that a plain close of a borrowed handle is busy. It adds close-after-release:
the identity becomes unusable at once, the borrower is woken through a waker it
installed for its wait, its operation returns `ESHKOL_IO_CANCELLED`, and the
close callback runs exactly once when it releases. Cancellation uses the same
interruption without closing; an interruption with no borrower applies once to
the next wait. A second concurrent operation on a handle is refused as busy
instead of sharing the descriptor.

**Waits go through the event loop.** Each thread owns one event loop plus a
wake pipe, created on first use and released at thread exit, so descriptor use
is constant across any number of operations. A zero deadline is one
nonblocking readiness pass.

**TLS sessions own their stream's lease.** A session borrows its TCP stream
for its whole life, so the stream cannot be read or written around the TLS
record layer; closing the session closes the stream. Handshakes can start on a
TCP stream the program already owns, in either role. Verification is on by
default with SNI and hostname or IP checks, failures report untrusted versus
identity mismatch, and only an explicit option disables verification. OpenSSL
is the system library on Linux and macOS; Windows Schannel is Planned for
v1.4.1 and refuses explicitly until then.

**Guaranteed close in Scheme.** Each acquire site has a scoped form built on
`dynamic-wind` that closes exactly once on return, exception or continuation
escape and refuses re-entry before the receiver runs. Byte-counted procedures
operate on caller bytevectors.

**The bytecode VM refuses sockets explicitly** at compile time instead of
running a partial program.

## Consequences

- The 256-slot ceiling is gone; the process descriptor limit bounds sockets
  and exhaustion refuses cleanly.
- A close from another thread now returns the blocked operation promptly with
  `-6` instead of after its deadline. Datagram truncation and connection
  refusal have their own codes (`-8`, `-9`) instead of sharing `-4` and `-1`.
- `tcp-close` and `udp-close` check the handle kind.
- Local IPC kinds adopt the same layer; a `PROCESS` registry kind and its
  pending-cleanup records are added by that work.
- The static checker's trusted `core.net` contract covers the new procedures;
  it remains a separate, opt-in profile.
