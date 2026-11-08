# `agent.unix_socket` — Unix-Domain Stream Sockets (blueprint)

**Status: Planned for v1.4.0.** This page is a skeleton for the paired build
described in Selene `apprentice/gabe-kahen/notes/IPC_DESIGN-2026-11-02.md`
("Later Unix streams" row): a lane writes the acceptance tests and the API
scaffolding below, and Gabe Kahen writes the socket core that makes them
pass. Nothing on this page is implemented yet. `lib/agent/unix_socket.esk`
currently raises `"Planned for v1.4.0"` from every operation, via the
scaffold in `lib/core/unix_socket_stream_stub.c`.

```scheme
(require agent.unix_socket)
```

Source: `lib/agent/unix_socket.esk`. C header: `inc/eshkol/core/unix_socket_stream.h`.
C symbols: `eshkol_unix_socket_*`.

## Surface (blueprint)

| Operation | Contract |
|---|---|
| `unix-socket-listen path` | Bind and listen on a filesystem path on the existing event loop. Owns the path: close unlinks it. Refuses a pre-existing, non-stale path rather than unlinking and reusing it. |
| `unix-socket-accept listener deadline-ms` | Accept one connection within one bounded, monotonic deadline. Zero makes one nonblocking pass and never sleeps. |
| `unix-socket-connect path deadline-ms` | Connect within one monotonic deadline shared across retries. |
| `unix-socket-read! id buffer start count` | Byte-counted nonblocking read into a caller bytevector. Returns a result vector distinguishing data, EOF, would-block, and an operation-labelled error. NUL bytes and payloads over 16 MiB survive unmodified; reads may be partial. |
| `unix-socket-write! id buffer start count` | Byte-counted nonblocking write, possibly partial. A closed peer reports an operation-labelled error; SIGPIPE is suppressed by the write path itself. |
| `unix-socket-cancel! id` | Request cancellation of an outstanding accept, connect, read, or write. |
| `unix-socket-close! id` | Close a listener, an accepted peer, or an outbound connection exactly once. A listener close unlinks the path it bound. |

Result accessors: `unix-socket-read-status`/`-count`/`-error` and
`unix-socket-write-status`/`-count`/`-error`, mirroring the subprocess stream
contract in [subprocess](subprocess.md).

## Ownership rules (blueprint)

Following the same pattern as the `PROCESS` session in
[subprocess](subprocess.md#ownership--cleanup-contract-94):

- Each socket identity is a `SOCKET`-kind lease in the existing handle
  registry (`inc/eshkol/core/handle.h`): acquire, exclusive borrow, release,
  close — the same rules as the `PROCESS` kind, and the shape the v1.4.0
  compile-time linear handles will consume.
- A listener owns the filesystem path it bound; its close unlinks that path.
  A stale path left on disk with nothing listening must be refused by a new
  `unix-socket-listen`, never silently unlinked and reused.
- Close with an active borrow returns busy, the same as every other leased
  resource; release the borrow before closing.
- A closed peer on the write side is reported through the byte-counted
  result (an operation-labelled error), never a raised `SIGPIPE`.
- `network` is the capability that gates every operation above, reusing the
  existing policy (`docs/reference/agent/capabilities.md`), the same gate
  TCP/UDP sockets use in `lib/core/net.esk`.
- The VM needs a matching adapter or must report an explicit unsupported
  result; it must not silently admit the ABI.

## Acceptance list (blueprint)

The scaffold registers one native C fixture plus JIT and AOT `.esk` cases per
item below, under CTest label `ipc-unix-pending` (excluded from the default
suite until the core implementation lands):

1. connect/accept round trip
2. partial reads/writes with a slow peer
3. peer close mid-stream (an operation-labelled error, with `SIGPIPE`
   suppressed)
4. deadline expiry on accept and on connect
5. cancellation
6. NUL and >16 MiB payloads
7. a descriptor-count check over 1,000 open/close cycles
8. `network` capability denial
9. socket-path lifecycle (unlink on close, refusal on an existing stale path)
10. VM: an explicit unsupported result

A small check in the default suite (`agent-unix-socket-stub-contract`)
confirms these cases are registered and that every scaffold entry point
still reports the planned result.

## Status

Planned for v1.4.0 (ladder: v1.4.0-connection, v1.4.1, v1.4.2, v1.4.3,
v1.4.4, v1.4.5-accelerate, v1.5.0). This page moves from blueprint to shipped
text in the same change that moves the acceptance cases out of
`ipc-unix-pending` and into the default suite.
