# Runtime Reference

Reference documentation for the Eshkol runtime and toolchain
(v1.4.0-connection).

## Tools

- [`eshkol-run`](eshkol-run.md) — the compiler & JIT driver: every CLI flag
  (AOT, `-r`/`-e` JIT, `-c`/`--emit-object`, `-s` shared lib, `-w` WASM,
  `--profile`, `--target`, `-O`, `--dump-ast`/`--dump-ir`, `--debug-info`, …).
- [`eshkol-repl`](eshkol-repl.md) — interactive REPL and the `--machine`
  warm-worker **EREPL v1** protocol: the original READY/DONE/FAIL bare-line
  framing, plus a versioned JSON request/response layer on top of it
  (`ready` handshake, `"op":"eval"` with the form's value and stdout reported
  separately from the stdout pipe, structured error kinds). `tools/erepl_client.py`
  is the stdlib-only Python reference client and the executable form of the
  contract.
- [`eshkol-vm-standalone`](eshkol-vm-standalone.md) — the bytecode VM, the
  **ESKB** binary format, `--emit-eskb`, and `--require-vm-entry[-zero-arg]`.

## Runtime concepts

- [Environment variables](environment-variables.md) — the full user-facing set:
  JIT/run cache, search paths, resource limits, parallelism, backends, subprocess
  sandbox.
- [Memory model](memory-model.md) — 16-byte tagged values, the arena allocator,
  the hybrid global/per-thread model, and `with-region` semantics (incl. the
  PR #81 reclamation change). Both engines reclaim through `with-region`; the
  AUTOMATIC per-loop nursery and the `region-open`/`region-close` handles are
  native-engine capabilities.
- [Parallelism & threading](parallelism.md) — `parallel-map`/`-fold`/`-filter`/
  `-execute`, `future`/`force`, the work-stealing pool, the serialized-state
  pattern, and the AD-mode-flag limitation.
- [JIT internals](jit-internals.md) — the run cache, the stdlib object cache, and
  the Large code model / arm64 Branch26 veneer.
- [Coroutines](coroutines.md) — stackful asymmetric coroutines:
  `make-coroutine`, `coroutine-resume`, `coroutine-yield`, `coroutine-raise`,
  `coroutine-status`, `coroutine-release!`, `current-coroutine`, `coroutine?`,
  `coroutine-stats`; the per-coroutine execution context, and the substrate of
  [`core.fibers`](../stdlib/fibers.md).
- [Compilation server](compilation-server.md) — how `eshkol-server` frames,
  bounds and answers an HTTP request: one reading of the header section for
  framing, the repeated-`Content-Length` and `Transfer-Encoding` rules,
  byte-counted accumulation, the header/body/deadline budgets and their
  environment variables, per-request isolation of a compilation, and the
  status each refusal carries.
- [Event loop](event-loop.md) — the portable readiness multiplexer over
  kqueue / epoll / IOCP: `make-event-loop`, `event-loop-add-fd!`,
  `event-loop-remove-fd!`, `event-loop-poll`, `event-loop-close`, the Windows
  completion-vs-readiness adaptation and its stated limits, and why the loop
  lives outside the arena.
- [Resource leases](resource-leases.md) — the lease that a linear handle
  carries: acquisition, transfer, release and what the runtime refuses when a
  lease outlives its scope ([ADR 0055](../../design/adr/0055-socket-resource-leases.md)).
- [Fixed-point, i128, and exact accumulation](fixed-point.md) — the
  `esk_i128` / `esk_fixed<W,F>` / `esk_dot_exact_*` / `esk_imatmul_i32` C ABI
  for the compression stack: explicit per-operation rounding and overflow
  policy, exactness-reporting conversions, and the order-independent
  i128-accumulator dot products and matmul.

## Connection and systems surface

New in v1.4.0-connection, documented under the stdlib reference:

- [`core.net`](../stdlib/net.md) — TCP/UDP sockets as linear, typestate-checked
  handles, closed deterministically at region boundaries, with the lease
  contract above.
- [`core.tls`](../stdlib/tls.md) — OpenSSL TLS over those handles (Linux and
  macOS).
- [`core.http_server`](../stdlib/http_server.md) — HTTP/1.1 request parsing,
  responses and routing. The HTTP/WebSocket client and server run on the
  [event loop](event-loop.md) above, and the agent runtime adds a streaming
  Server-Sent Events client and server.
- [Unix domain sockets](../agent/unix-sockets.md) — local-transport handles.
- [Subprocess execution](../agent/subprocess.md) — structured spawn with
  ownership-driven cleanup and sandboxing.
- [Agent memory faculty](../agent/memory-faculty.md) — durable
  write-ahead-logged (WAL) agent memory and its atomic persistence path.
- [`core.protobuf`](../stdlib/protobuf.md) — wire format for inter-process and
  inter-agent messages.
- Browser VM dispatch over WebGPU is specified in
  [ADR 0029](../../design/adr/0029-browser-vm-webgpu-dispatch.md); exact-rational
  tensors (`'exact`) are in the [tensor reference](../tensors/INDEX.md), and the
  shared-library target (`-s`) and its tail transfer in
  [`eshkol-run`](eshkol-run.md).

## See also

- [Agent / FFI reference](../agent/INDEX.md)
- [Platform reference](../../platform/) — [CI lanes](../../platform/CI_LANES.md),
  [build notes](../../platform/BUILD_NOTES.md),
  [target matrix](../../platform/TARGET_SUPPORT_MATRIX.md).

## Notes

- The binary reports `v1.4.0-connection` (`eshkol-run --version`), matching
  this documentation; the string is derived from `project("Eshkol" VERSION …)`
  and `ESHKOL_RELEASE_NAME` in `CMakeLists.txt`.
- `-e` does not auto-print a bare value expression; `--help` says so
  ("output is shown via `(display …)`"), and `eshkol-run -e '(+ 1 2)'` prints
  nothing while `eshkol-run -e '(display (+ 1 2))'` prints `3`. (See
  [`eshkol-run`](eshkol-run.md).)
