# Socket review revision

The [manifest](unix-socket-review-revision-2026-11-12.json) records **68/68 selected Linux CTests at eight-way parallelism**, **148/148 native O0 ownership checks**, and the [hosted native run](https://github.com/Gabriel-Kahen/eshkol/actions/runs/43312363510). Hosted Linux normal, Linux ASan/UBSan/leak, and macOS 15 arm64 kqueue all passed the socket/common/PROCESS suites. Original raw logs are retained locally or as hosted workflow artifacts; only portable summaries and original-byte hashes are repository content.

Production code was compiled at `85588a3dd3b433f152f921cf2659786d04c016cd`. The hosted run used `11d5707964479a61850529d09e5da55b3df5e10c`, which changes qualification fixtures/registration, not production code. Later runner pathname-budget checks also include an actual JIT round trip with a supplied short TMPDIR and successful scratch removal. The manifest records these revisions separately.

## Review changes

| Review items | Result |
|---|---|
| 1–2: source pins and ownership contracts | All 40 PROCESS and 34 socket exports have reviewed rows. Connecting/open/cancelled owners, nullable recovery, consuming close-request, ticket release and scoped loans are checked; aggregate PROCESS cleanup ownership is explicitly unsupported. |
| 3, 16–18: common loop contract | Genuine watch-removal errors retain bookkeeping for repair/retry; already-gone registrations normalize to success. Owner-thread and BUSY/refusal behavior, opaque loop identities and level-triggered rediscovery are documented and tested. Legacy private waits retire their backend safely on a removal error. |
| 4: common-path cost | Immutable registry flags skip managed locking for legacy I/O; managed IDs use a full-ID hash index. Release pumps only this owner's pending queue. Controls hold 128 managed owners while legacy operations acquire zero managed locks/pumps. |
| 5: terminal path cleanup | The callback owns terminality. Initial pathname failure permits one repair/retry; a second failed attempt reports terminal PATH failure, releases the loop and permits ticket release. |
| 6: creation flags | Linux sockets, accepted descriptors and wake pipes use atomic close-on-exec/nonblocking creation. Portable fallback paths remain where those interfaces are unavailable. |
| 7–8: connect and interruption | Full-backlog retries use bounded backoff capped by the original deadline; cancellation remains observable. Interrupted reads/writes report retryable would-block. A real 180 ms backlog control used 25 attempts and 0–1 ms thread CPU. |
| 9: unpublished pathname cleanup | The primary setup error and residual pathname error remain separately observable; the Scheme exception includes the affected path. Fault controls verify both. |
| 10–11: fixtures | Short TMPDIR values are honored; longer prefixes fall back to a documented short directory with room for fixture leaf names. Successful runs remove their socket directories. The observation DSO has no connect interposer; Darwin pending controls are separate and explicitly labelled. |
| 12–13, 19: repository records | Raw machine logs are removed, summaries use portable placeholders, and unrelated generated backend API/index pages are deferred to integration. The cache publishes completed objects from process-unique files. One all-target parallel bootstrap configuration is outside this qualification. |
| 14–15: outcome names | Scheme uses named shared outcome constants with native alignment assertions; poll reports cancellation explicitly. |
| 20: PR shape | The four integration changes and the common lease/loop foundation have separate branches/PRs. #914 carries PROCESS contracts and its shared-registry adapter. Socket work remains dependent on those and the amended scaffold. |
| 21–22: descriptor/path controls | A child lowers RLIMIT_NOFILE, observes accept exhaustion and recovers after freeing a descriptor. The private-parent permission rationale and abstract-name refusal are explicit. |

## Additional review controls

Scoped finalization privately owns its ticket, attempts one bounded retry and releases terminal results. Normal return reports a plain failure snapshot with context/retired identity. Exceptions and continuation escape retain their original control; manual ticket forms are required for guaranteed cleanup-error inspection on those exits. Genuinely pending/watch failures remain safely detached and retained. Actual public JIT and AOT lifecycle runs passed 44 checks each, including real non-root permission failures, loop release, original exception preservation and cleanup with network permission revoked. Filesystem probes receive their own file-read capability and do not masquerade as cleanup results.

The peer publishes readiness atomically only after its controlled state is established, and stays alive across compilation. Large public transfer budgets derive from measured compile time. Native pacing measures three real naps and captures one finite deadline from that measured granularity. The first hosted native Mac run used a constant 20-second fixture allowance; the calibrated run verifies all 16,777,253 bytes and 4,095 partial writes within its measured allowance. No byte, chunking, pacing or partial-I/O assertion was removed.

The [design answers](../../../notes/SOCKET_DESIGN_ANSWERS-2026-11-12.md) explain the per-resource ticket choice. A shared test retains 128 independent failed results on one loop and recovers all of them after slot reuse; serial callback execution does not eliminate simultaneous unresolved responsibilities.

## Scope

Native macOS socket/common/PROCESS execution is established here. macOS net/TLS, public JIT/AOT and packaged-runtime qualification remain separate. Host-compiled Windows/WASM source branches explicitly refuse the bridge and have no native socket/readiness references; these controls do not claim Windows/WASM execution. Bytecode VM pre-emission refusal is separately tested. Full repository and release/package qualification are not claimed. Ownership checks qualify O0 compilation; O2 optimization latency for large SDK modules is outside this record. Maintainer review of the cleanup API remains open.
