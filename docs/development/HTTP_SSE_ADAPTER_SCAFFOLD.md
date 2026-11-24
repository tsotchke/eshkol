# Accepted-HTTP-Connection Ownership Adapter and SSE Scaffold

Base revision: `667c382d5f2e8a992c4c6e9053d734110c987773` (shared explicit-loop resource leases, PR #926).

## Scope and ownership contract

This slice supplies the minimum accepted-HTTP-connection ownership adapter and native acceptance scaffold for server-side SSE over loopback HTTP/1.1:

- **HTTP server to lease handoff** (`eshkol_http_sse_adapter_adopt`): validates the active server client slot and owner-thread explicit event loop, switches the descriptor to nonblocking mode, and adopts sole ownership via `eshkol_resource_adopt_on_loop(..., ESHKOL_HANDLE_KIND_SOCKET, ESHKOL_HTTP_SSE_LEASE_TAG, ...)`. Failed adoption leaves `srv->client_fd` intact and writes nothing; successful adoption clears `srv->client_fd` so `eshkol_http_server_respond` and `eshkol_http_server_close` never touch the transferred descriptor.
- **Write-watch and borrow contract**: `eshkol_http_sse_adapter_write_step` takes a short exclusive borrow (`eshkol_resource_borrow` / `eshkol_resource_release`), checks cancellation and the original monotonic deadline immediately before native `send()`, advances `*offset_inout` on partial progress, and arms or disarms the write watch on the caller's loop.
- **Bounds and deadlines**:
  - `ESHKOL_SSE_MAX_CHUNK_BYTES`: 16 KiB (`16384` bytes), including HTTP/1.1 chunk framing overhead.
  - `ESHKOL_SSE_WRITE_BUDGET_MS`: 5,000 ms monotonic per-frame write budget, held across retries and checked before every send.
  - `ESHKOL_SSE_IDLE_TIMEOUT_MS`: 60,000 ms idle budget between frames.
- **Teardown and cleanup ticket**: `eshkol_http_sse_adapter_request_close` delegates to `eshkol_resource_close_request`. Owner-thread cleanup removes any armed write watch before closing the descriptor once. A watch-removal failure returns `ESHKOL_CLEANUP_FAILED` at `ESHKOL_CLEANUP_STAGE_WATCH` with `retryable = 1`, retaining descriptor and loop lifetime until retry reaches a terminal outcome. An uncertain descriptor close is reported at `ESHKOL_CLEANUP_STAGE_DESCRIPTOR` with `retryable = 0` and is never retried.

## Local verification

```sh
# macOS (kqueue)
clang++ -std=c++17 -O2 -Iinc -c lib/core/runtime_handle.cpp -o /tmp/runtime_handle.o
clang -std=c11 -O2 -DESHKOL_SSE_ADAPTER_TESTING -Iinc \
  tests/agent/http_sse_adapter_scaffold_test.c \
  lib/agent/c/agent_http_server.c \
  lib/core/resource_lease.c \
  lib/core/event_loop.c \
  lib/core/event_loop_kqueue.c \
  /tmp/runtime_handle.o -lpthread -lc++ -o /tmp/http_sse_adapter_scaffold_test
/tmp/http_sse_adapter_scaffold_test
```

Replace `event_loop_kqueue.c` with `event_loop_epoll.c` and `-lc++` with `-lstdc++` on Linux.
