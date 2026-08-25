# Runtime Reference

Reference documentation for the Eshkol runtime and toolchain (v1.3.4-evolve).

## Tools

- [`eshkol-run`](eshkol-run.md) — the compiler & JIT driver: every CLI flag
  (AOT, `-r`/`-e` JIT, `-c`/`--emit-object`, `-s` shared lib, `-w` WASM,
  `--profile`, `--target`, `-O`, `--dump-ast`/`--dump-ir`, `--debug-info`, …).
- [`eshkol-repl`](eshkol-repl.md) — interactive REPL and the `--machine`
  warm-worker **EREPL** protocol (READY/DONE/FAIL framing).
- [`eshkol-vm-standalone`](eshkol-vm-standalone.md) — the bytecode VM, the
  **ESKB** binary format, `--emit-eskb`, and `--require-vm-entry[-zero-arg]`.

## Runtime concepts

- [Environment variables](environment-variables.md) — the full user-facing set:
  JIT/run cache, search paths, resource limits, parallelism, backends, subprocess
  sandbox.
- [Memory model](memory-model.md) — 16-byte tagged values, the arena allocator,
  the hybrid global/per-thread model, and `with-region` semantics (incl. the
  PR #81 reclamation fix). Both engines reclaim through `with-region`; the
  AUTOMATIC per-loop nursery and the `region-open`/`region-close` handles are
  native-engine capabilities.
- [Parallelism & threading](parallelism.md) — `parallel-map`/`-fold`/`-filter`/
  `-execute`, `future`/`force`, the work-stealing pool, the serialized-state
  pattern, and the AD-mode-flag limitation.
- [JIT internals](jit-internals.md) — the run cache, the stdlib object cache, and
  the Large code model / arm64 Branch26 veneer.
- [Event loop](event-loop.md) — the portable readiness multiplexer over
  kqueue / epoll / IOCP: `make-event-loop`, `event-loop-add-fd!`,
  `event-loop-remove-fd!`, `event-loop-poll`, `event-loop-close`, the Windows
  completion-vs-readiness adaptation and its stated limits, and why the loop
  lives outside the arena.

## See also

- [Agent / FFI reference](../agent/INDEX.md)
- [Platform reference](../../platform/) — [CI lanes](../../platform/CI_LANES.md),
  [build notes](../../platform/BUILD_NOTES.md),
  [target matrix](../../platform/TARGET_SUPPORT_MATRIX.md).

## Notes

- The binary reports `v1.3.4-evolve` (`eshkol-run --version`), matching this
  documentation.
- `-e` does not auto-print a bare value expression; `--help` says so
  ("output is shown via `(display …)`"), and `eshkol-run -e '(+ 1 2)'` prints
  nothing while `eshkol-run -e '(display (+ 1 2))'` prints `3`. (See
  [`eshkol-run`](eshkol-run.md).)
