# Native HTTP SSE draft

The core layers the bounded SSE write state over the accepted-connection adapter at `dc38e82e939a73967d7a72994b182cc421abe3ed`. Tsotchke approved its parent `667c382d5f2e8a992c4c6e9053d734110c987773` as the shared SSE source. [Base approval](https://github.com/tsotchke/eshkol/pull/926#issuecomment-7095401085)

After routing a valid HTTP/1.1 GET on the loopback HTTP server, the loop owner can begin a stream. Successful adoption consumes the current client slot, while the original server retains its listener. Progress writes the HTTP response headers. Once those drain, one event can be admitted at a time. Admission copies the encoded chunk into a 16 KiB buffer and sets a five-second monotonic deadline. Busy admission preserves the frame, offset, and deadline. Once a frame drains, the loop owner uses the next deadline to enforce a separate 60-second idle limit.

The adapter handles native sends and the write watch. The core retains the matching lease and adapter together and clears both after a close request is accepted. The loop owner calls progress on a matching readiness notification or when the deadline comes due. On clean finish it writes the zero chunk. On early disposal, timeout, or failed transport I/O it retires without another send. The caller receives the cleanup ticket on disposal and observes, retries if appropriate, and releases it before closing the loop. A successful send whose watch removal fails now reports an error so the core retires through the same ticket path.

## Repeat the native loopback

With the project's LLVM dependencies installed, configure the checkout with tests enabled and build the two focused targets. On the tested Linux host, LLVM 22 was selected explicitly and GPU support was disabled.

```sh
cmake -S . -B build-sse -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_REQUIRED_LLVM_MAJOR=22 -DESHKOL_GPU_ENABLED=OFF -DESHKOL_BUILD_AGENT_FFI=OFF
cmake --build build-sse --target http_sse_adapter_scaffold_test http_sse_stream_test
ctest --test-dir build-sse -R '^http_sse_(adapter_scaffold|stream)_test$' --output-on-failure
# Optional one-minute check of the idle deadline
build-sse/http_sse_stream_test --idle
```

The scaffold forces short progress, a would-block and a late writable attempt. The core loopback sends real HTTP response headers and chunked UTF-8 over TCP, then hands the captured payload to the existing SSE parser. It verifies one intact event, a second ordinary response on the original listener, busy preserving the admitted event and deadline, the exact encoded-size boundary, and zero body writes after abandonment or a spent write deadline. The tests use the native SSE slice directly with agent FFI disabled.

The interface in `lib/agent/c/agent_http_sse_stream.h` is internal. Joint review still needs to settle the public scoped interface, confirm same-descriptor watch ownership, and exercise the product PROCESS/SOCKET/SSE combination, scope exceptions and continuation exits on the integrated revision. This branch uses the approved adapter base. The current Wave 9 composition has a CMake conflict and includes a native capability gate that integration must preserve.
