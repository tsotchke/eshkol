# Scoped HTTP SSE streams

The bounded SSE writer builds on the accepted-connection adapter at `dc38e82e939a73967d7a72994b182cc421abe3ed` and the shared lease revision `667c382d5f2e8a992c4c6e9053d734110c987773`. [Base approval](https://github.com/tsotchke/eshkol/pull/926#issuecomment-7095401085)

## Scheme interface

Require `agent.http_sse`. After accepting and routing a valid HTTP/1.1 GET, call `(call-with-http-sse-stream server loop watch-cookie procedure)` on the loop owner thread. The scope takes the accepted socket only after lease adoption succeeds. The HTTP server retains its listener and can serve ordinary complete responses on other connections.

| Operation | Result |
| --- | --- |
| `(sse-send! stream data [event-type [id]])` | Queue one UTF-8 event. Returns 0 on admission, -5 if a write is pending, -8 beyond the encoded 16 KiB limit, or -4 for invalid input. Omit metadata with `#f`. An empty event type is omitted and an empty ID resets the client ID. |
| `(sse-progress! stream)` | Send the next portion of the headers or pending chunk. Returns 0 when ready for another event, 1 while pending, or 2 after writing the final zero chunk and requesting close. A negative result reports a transport or deadline outcome. |
| `(sse-finish! stream)` | Queue the final zero chunk after the pending event has drained. Drive progress until it returns 2 for a clean finish. |
| `(sse-cancel! stream)` | Retire immediately without emitting further bytes. |
| `(sse-deadline-ms stream)` | Absolute monotonic deadline for the current write or idle period. Drive progress after polling the loop and when that deadline becomes due. |

The adapter owns the write registration. The writer retains the admitted chunk, offset, and original five-second write deadline across retries. Between writes, a separate 60-second idle deadline applies. The native writer checks loop ownership, pending work, and idle expiry before validating field bytes. Metadata rejects CR, LF, and NUL before admission.

Native acquisition and progression require the network capability. Revoking it retires the stream without advancing a buffered write. Cleanup remains available.

Scope exit retires without advancing any buffered write on return, exception, or continuation escape. A continuation cannot reenter a scope that has already adopted a socket. The after thunk hands the cleanup ticket to the shared lease manager so pending cleanup survives an unwind. The manager releases successful cleanup and retains terminal errors. A caller retaining the Scheme stream can claim a failed ticket with `sse-cleanup-ticket` and use the shared cleanup API on the owner loop. A retryable watch failure keeps the descriptor and original registration owned until retry. An uncertain descriptor close is recorded and never retried.

## Repeat the loopback

Configure the source with agent FFI and tests enabled, then build the compiler and the native SSE suites. The following uses LLVM 21.

```sh
cmake -S . -B build-sse -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_REQUIRED_LLVM_MAJOR=21 -DESHKOL_GPU_ENABLED=OFF -DESHKOL_BUILD_AGENT_FFI=ON
cmake --build build-sse --target eshkol-run http_sse_adapter_scaffold_test http_sse_stream_test
ctest --test-dir build-sse -R '^http_sse_(adapter_scaffold_test|stream_test|scope_.*)$' --output-on-failure
# Optional one-minute check of the idle deadline
build-sse/http_sse_stream_test --idle
```

The Scheme loopback drives the public API over a live TCP connection in JIT and AOT. The native peer reads the canonical 37-byte chunk across a split UTF-8 character and the final blank line. The receiver dispatches one intact event. The scaffold also forces a short native send, write expiry, disconnect and watch-removal recovery. The native core suite covers denied acquisition, policy revocation, and status precedence. The idle run verifies that expired state takes precedence over an oversized event.

The mixed-loop suite needs a composed checkout with the PROCESS and Unix socket implementations. Call the test runner from this checkout with that source as its argument.

```sh
tests/agent/run_http_sse_mixed_loop_test.sh "$composed_source"
ESHKOL_SSE_MIXED_SANITIZE=1 tests/agent/run_http_sse_mixed_loop_test.sh "$composed_source"
```

The mixed-loop test injects one watch-removal error and checks that fresh subprocess and Unix socket bytes arrive on their original registrations before, during, and after SSE teardown. It also checks the exact 14 bytes sent before cancellation, EOF, and one descriptor close on retry. Wave 9 integration must retain the native network capability gate. The review and platform qualification of that landing tree remain open.
