# Runtime Reference

Reference documentation for the Eshkol runtime and toolchain (v1.3.0-evolve).

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
  PR #81 reclamation fix).
- [Parallelism & threading](parallelism.md) — `parallel-map`/`-fold`/`-filter`/
  `-execute`, `future`/`force`, the work-stealing pool, the serialized-state
  pattern, and the AD-mode-flag limitation.
- [JIT internals](jit-internals.md) — the run cache, the stdlib object cache, and
  the Large code model / arm64 Branch26 veneer.

## See also

- [Agent / FFI reference](../agent/INDEX.md)
- [Platform reference](../../platform/) — [CI lanes](../../platform/CI_LANES.md),
  [build notes](../../platform/BUILD_NOTES.md),
  [target matrix](../../platform/TARGET_SUPPORT_MATRIX.md).

## Notes / discrepancies observed

- The binary reports `v1.2.4-scale` (`eshkol-run --version`) while this
  documentation campaign targets **v1.3.0-evolve**.
- `--help` says `-e` "prints the result," but a bare value expression is **not**
  auto-printed — use `display`. (See [`eshkol-run`](eshkol-run.md).)
