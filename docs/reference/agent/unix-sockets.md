# `agent.unix_stream_socket` — Unix-domain stream socket contract candidate

**Status: agreed integration contract; declarations and core remain under
review.** The [agreed addendum](https://github.com/Gabriel-Kahen/eshkol/blob/1e8d900b359128354b086ff334e7267a9a02561e/notes/SOCKET_DESIGN_ADDENDUM-2026-11-10.md)
and [shared-layer decision](https://github.com/tsotchke/eshkol/pull/917#issuecomment-6806485060)
record the ownership split: Gabe Kahen implements the common lease/wait
extensions and Unix socket core; the scaffold lane amends declarations and
acceptance tests. Well-formed operational calls still report `Planned for
v1.4.0`; this scaffold implements no socket I/O.

```scheme
(require agent.unix_stream_socket)
```

Source: `lib/agent/unix_stream_socket.esk`. Native declarations:
`inc/eshkol/core/unix_socket_stream.h`. The common lease/outcome layer is
[`resource_lease.h`](../../../inc/eshkol/core/resource_lease.h), from the
shared network lease work. This candidate combines the public scaffold and
shared-lease sources for review; it does not select a release baseline.

## Surface

| Operation | Contract |
|---|---|
| `unix-stream-socket-listen loop path` | Require an explicit live loop; zero is invalid. Retain the loop, bind an owned private pathname and publish the listener. |
| `unix-stream-socket-accept listener deadline-ms` | Accept within one monotonic budget; zero makes one nonblocking pass. The accepted peer inherits and retains the listener's loop. |
| `unix-stream-socket-connect-begin loop path deadline-ms` | Capture one acquisition deadline and publish a connecting owner before any readiness wait. Zero returns an immediately connected owner or rolls back with would-block. |
| `unix-stream-socket-connect-finish! id` | Complete using begin's remaining budget; do not renew it. The returned identity remains owned after timeout/cancellation until close. |
| `unix-stream-socket-read! id buffer start count` | Positive checked range; exact partial count, would-block, EOF, or operation-labelled error. Requires connected state. |
| `unix-stream-socket-write! id buffer start count` | Positive checked range and exact partial count. Preserve NULs and large payloads; suppress SIGPIPE per socket/send. |
| `unix-stream-socket-poll id interests timeout-ms` | Interests1=read,2=write,3=both; result distinguishes readiness, timeout and error. Zero never waits. While connecting, begin's deadline caps the wait. Connected I/O has independent poll budgets. |
| `unix-stream-socket-cancel! id` | Idempotent sticky cancellation plus shared wakeup. Further I/O observes cancellation; ownership remains. |
| `unix-stream-socket-close! id` | Public deferred close through the common lease layer: after acceptance the identity is unusable; descriptor close runs once after active borrows release. |

Read/write result accessors are `-status`, `-count`, `-error`; poll has
`-status`, `-interests`, `-error`. Error records expose `-error-operation`,
`-error-code`, `-error-os-code`. Portable codes come from the common
`ESHKOL_IO_*` table. Native `last_os_error()` is calling-thread metadata read
immediately after the corresponding negative outcome. The planned sentinel
is separate from every shared outcome and is never acceptance evidence.

## Loop, ownership and cleanup

- The caller supplies the loop. Every listener and peer retains it until
  watch removal and descriptor teardown complete. Loop close refuses active
  retains; releasing a socket's retain does not close the caller's loop.
- Use common lease adoption, exclusive borrow/release, close and outcome
  machinery. **The common wait facility still needs explicit-loop support**
  that dispatches SOCKET and PROCESS readiness together with generation/cookie
  validation. This is a core integration prerequisite, not a facility supplied
  by these stubs. Do not add a private socket registry or polling loop.
- A generation-validated, lifetime-safe cancellation request may arrive from
  another thread. It sets sticky socket state and signals the shared wakeup;
  I/O and watch mutation remain on the owner thread. The common layer's
  consumable interrupt is the wake transport, not the socket's lifetime state.
- Distinguish low-level registry close, which can report a busy borrow, from
  public lease close, which retires the identity and defers teardown until
  release. Acceptance is not completed teardown: the shared extension must
  provide observable watch-removal and pathname-cleanup outcomes before core
  implementation. That completion interface remains to be specified by the
  common-layer owner; this scaffold does not invent another ownership table.
  Watch-removal failure retains safe ownership for retry. Once descriptor
  close is attempted, never retry an uncertain descriptor identity.
- Acquisition and I/O require `network` in both native and Scheme paths.
  Already-owned cancel/close do not reacquire that capability or consult a
  clock. Runtime, kind, tag and generation checks remain mandatory after
  revocation. Scoped cleanup must use the same cleanup path.

## Paths and platforms

Use absolute NUL-free paths fitting `sun_path` in encoded bytes, inside a
verified owned private directory with mode 0700. Refuse all existing entries, including
restart leftovers. Bind, set mode 0600, then listen/publish; roll back unpublished
setup errors. Only remove the listener's recorded entry; preserve replacement
entries and report cleanup errors. Never change the process umask.

Linux abstract namespace and descriptor passing are outside this slice.
Windows requires a distinct AF_UNIX availability check and a readiness adapter;
family availability alone is insufficient. The VM must explicitly reject the
bridge before bytecode emission until an adapter exists. Linux and macOS need
separate runtime evidence.

## Acceptance and integration

The public list has 15 native/JIT/AOT case groups, plus a native-only direct
borrow/deferred-close case and one explicit VM-refusal case: 47 pending cases
when the agent FFI and runner are enabled,16 native cases with the FFI disabled.
The groups cover round trips; real slow-peer partial transfers; default
SIGPIPE peer closure; measured accept/original-connect deadlines; cancellation
including a published pending connect; binary payloads beyond 16 MiB; measured
1,000-cycle descriptor counts; native/Scheme capability denial; cleanup after
revocation; pathname restart/replacement; loop lifetime; and zero budgets.
Native cases also exercise direct active-borrow teardown. Scheme cases verify
public errors by operation/code, never by an arbitrary exception.

The raw peer executable creates only controlled external backlogs/leftover
paths; it does not implement the socket API. The helper DSO observes
monotonic time, descriptor counts and platform errno constants. For Darwin's
pending-connect controls it also injects EAGAIN at exactly the designated
private fixture path; Darwin's real full-queue outcome is immediate refusal.
Linux uses a real full backlog. The receipt distinguishes this deterministic
syscall-error control from an OS-pending connection. All other connect calls
reach the real syscall. The DSO supplies no socket core, registry or replacement
agent archive. The public runner
retains unique fixture/log directories, bounds execution and stops only its
owned process groups. SIGPIPE uses the normal default disposition.

Pending cases remain explicitly disabled under `ipc-unix-pending`. Default
checks cover argument/planned behavior, the actual controlled peer fixture,
and registration counts. Neither those checks nor a passing metadata-only
inventory qualifies the socket core. As each facility lands, replace its
planned assertions with the corresponding enabled default-suite acceptance
cases in the same change. Required acceptance failures then block integration.

Before promotion, mixed PROCESS/SOCKET identity and readiness tests are also
required. The subprocess stream source still has its own registry while the
shared network layer has another. A source-representative allocation probe can
produce the same integer in both domains. Main integration must reconcile
those domains so a token from one cannot address an unrelated resource in the
other. This candidate does not migrate Gabe's subprocess implementation.

The scaffold does not certify compiler-enforced linearity, packaged consumers,
portable release artifacts, or the complete v1.4.0 release.
