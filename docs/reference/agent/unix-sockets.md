# `agent.unix_stream_socket` — Unix-domain stream sockets

Filesystem Unix stream sockets use the shared resource lease and explicit
caller-owned event loop. Native Linux JIT/AOT support is implemented; the
implementation evidence records the exact source and platform exercised.
macOS requires separate qualification. Windows and the bytecode VM explicitly
refuse this bridge until their adapters are available.

```scheme
(require agent.unix_stream_socket)
```

Sources: `lib/agent/unix_stream_socket.esk`,
`inc/eshkol/core/unix_socket_stream.h`, and
`lib/agent/c/agent_unix_socket_stream.c`. See also
[resource leases](../runtime/resource-leases.md) and the
[design addendum](../../../notes/SOCKET_DESIGN_ADDENDUM-2026-11-10.md).

## Surface

| Operation | Contract |
|---|---|
| `unix-stream-socket-listen loop path` | Explicit live loop (`0` invalid); bind in an owned private directory and return a listener. |
| `unix-stream-socket-accept listener timeout-ms` | One monotonic budget. Accepted peers inherit and retain the listener's loop. |
| `unix-stream-socket-connect-begin loop path timeout-ms` | Return a cancellable connecting owner before waiting; capture one deadline. Zero returns an immediately connected owner or rolls back with would-block. |
| `unix-stream-socket-connect-finish! id` | Complete within begin's remaining budget, without renewal. A published failed attempt remains owned until close. |
| `unix-stream-socket-read! id bv start count` | Nonblocking positive checked range; `#(data count #f)`, `#(would-block 0 #f)`, `#(eof 0 #f)`, or an operation-labelled error. |
| `unix-stream-socket-write! id bv start count` | Exact partial byte count, would-block, or an operation-labelled error. SIGPIPE is suppressed per socket/send. |
| `unix-stream-socket-poll id interests timeout-ms` | Mask 1=read, 2=write, 3=both. Returns a ready mask, timeout or error. Pending connect is capped by begin's deadline. |
| `unix-stream-socket-cancel! id` | Idempotent thread-safe sticky cancellation. Ownership remains until close. |
| `unix-stream-socket-close! id` | Accept close, retire the identity and queue teardown after active borrows release. |
| `unix-stream-socket-close-request! id` | Accept close and return an independently owned cleanup-result ticket. |

Read/write result accessors end in `-status`, `-count` and `-error`. Poll uses
`-status`, `-interests` and `-error`. Error vectors expose `-error-operation`,
`-error-code` and `-error-os-code`; portable codes use the common `ESHKOL_IO_*`
table. Native OS detail is calling-thread metadata read immediately after the
negative outcome. A timeout is an exact integer from zero to 2,147,483,647 ms;
zero never waits. Native I/O returns exact counts, with read zero meaning EOF
for a positive request.

The caller retains the bytevector and current byte offset. Advance by the
successful count, poll after would-block, then retry the remaining range.
Readiness permits retrying; another would-block result is still possible.
No output-sized buffer is retained inside the socket implementation.

## Ownership and cancellation

Every socket retains its caller-supplied loop through watch removal and
teardown. Closing that loop while retained reports busy. Ordinary I/O and
watch mutation run on the loop owner thread. A second active borrower reports
busy. Cross-thread cancellation and close requests only mark state and wake
the owner; they never mutate its watches or close a borrowed descriptor.
Cancellation prevents later I/O, cannot retract bytes already transferred,
and does not release ownership.

PROCESS, EVENT_LOOP, SOCKET and cleanup-ticket identities share one exact
process-lifetime registry. Kind and tag checks reject unrelated identities.
PROCESS shutdown retires only PROCESS records and never destroys another
resource's registry. Temporary socket watches preserve unrelated PROCESS
readiness and cookies. Existing TCP/UDP/TLS consumable cancellation remains
unchanged.

Acquisition and I/O require `network` in both native and Scheme paths.
Already-owned cancellation, close, cleanup inspection, retry and ticket release
need neither a renewed capability nor a working clock. Identity checks still
apply after revocation.

## Observable teardown

Close acceptance retires the socket identity immediately. It does not promise
that watch removal, descriptor close, pathname removal and loop release have
already finished. Use `unix-stream-socket-close-request!` for an observable
result:

| Operation | Result |
|---|---|
| `unix-stream-socket-cleanup-result ticket` | `#(state stage os-error retryable?)`; owner-thread inspection also drives queued cleanup. |
| `unix-stream-socket-cleanup-state result` | `pending`, `complete` or `failed`. |
| `unix-stream-socket-cleanup-stage result` | `none`, `watch`, `descriptor`, `path` or `loop`. |
| `unix-stream-socket-cleanup-os-error result` | OS error number, or zero. |
| `unix-stream-socket-cleanup-retryable? result` | Whether repair followed by explicit retry is allowed. |
| `unix-stream-socket-cleanup-retry! ticket` | Retry a recoverable failed cleanup on the owner thread. |
| `unix-stream-socket-cleanup-release! ticket` | Release a terminal result after ownership has been resolved. Pending or retryable results remain owned. |
| `unix-stream-socket-cleanup-ticket retired-id` | Recover a retained failed result from convenience close, or `#f`. |

Release outstanding borrows before expecting completion. Foreign-thread result
queries observe state without running teardown. Watch-removal failures keep
safe ownership and the loop retain. Pathname failures can be retried after
repair. A descriptor is marked attempted before close, and an uncertain close
is never retried, even if the OS later reuses its number. Successful detached
convenience-close results are discarded automatically; failed results remain
recoverable. The old socket identity never regains I/O authority.

`call-with-unix-stream-listener`, `call-with-unix-stream-connection` and
`call-with-unix-stream-accept` close on normal return, exception and
continuation escape, preserving the body result. Re-entering a finished scope
raises before the receiver runs again. Failed detached cleanup can be observed
using a retained socket identity and `unix-stream-socket-cleanup-ticket`.

## Paths and platforms

Paths are absolute, NUL-free and fit `sun_path` in encoded bytes. Listen and
connect require an owned parent directory with mode 0700. Listen refuses every
existing entry, binds and sets mode 0600 before publication, and never changes
the process umask. Only the listener removes its recorded socket entry, using
its retained parent directory and recorded device/inode. Replacement entries
are preserved and reported through cleanup results.

Linux abstract names and descriptor passing are outside this slice. The VM
refuses the imported bridge before bytecode emission. Native Linux and macOS
results are reported separately. Darwin pending-connect controls use labelled
private-path EAGAIN injection; Linux fixtures use a real full backlog. The
fixture helper provides observations and those labelled controls, not a
replacement socket implementation or agent archive. The JIT loads that helper
into its process to resolve observation symbols; AOT links the same helper.

## Qualification

The implementation evidence records native, JIT/AOT, mixed PROCESS/socket,
cleanup-ticket and sanitizer checks. Passing cases are enabled in default
CTest; any unfinished public case remains explicitly labelled pending.
Fixture or wrapper-only checks do not establish product execution. No portable
release-package or Windows/macOS execution is implied by Linux evidence.
