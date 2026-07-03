# JIT Internals

Both `eshkol-run -r`/`-e` and `eshkol-repl` run code in-process through an LLVM
ORC JIT. This page documents the parts a user needs to understand and control:
the persistent **run cache**, the **stdlib object cache**, and platform-specific
**code-model** behavior.

## Run cache (`-r`)

When `ESHKOL_JIT_CACHE` is enabled (the default), `eshkol-run -r <file>` compiles
the file once to a standalone native binary, stores it, and on subsequent runs
re-execs the cached binary — turning repeat `-r` invocations into near-instant
launches.

- **Location**: `ESHKOL_JIT_CACHE_DIR`, else `$XDG_CACHE_HOME/eshkol/jit` /
  `$HOME/.cache/eshkol/jit` (Unix), `%LOCALAPPDATA%\eshkol\jit` (Windows), else a
  temp dir. Cached binaries are named `run-<key><exe-suffix>`.
- **Cache key / invalidation**: a SHA-256 over a schema tag, the canonical source
  path **and source bytes**, the Eshkol version, the LLVM version, the
  `eshkol-run` binary fingerprint (size + mtime), the relevant flags
  (`--no-stdlib`, `--strict-types`, `--unsafe`, `-O`, `--target`), the
  `stdlib.bc`/`stdlib.o` fingerprints, and all include/lib/linked-lib flags. Any
  change to any of these misses the cache and recompiles.
- **Bypass**: the cache is skipped when disabled, when the source uses `eval` or
  `compile` (these need the in-process JIT bridge that a plain AOT binary lacks),
  when the self path can't be found, or when the cache dir can't be created.
- **Pruning**: entries older than 30 days are removed, then the oldest are
  evicted until the cache is ≤ 1 GiB.
- **Tracing**: `ESHKOL_JIT_CACHE_TRACE=1` prints `[jit-cache] hit|miss|bypass`
  to stderr.
- Cached binaries inherit the parent's actually-loaded library directories on
  their library search path so they can find `libLLVM` etc.

## Stdlib object cache

The standard library is ~55 MB of IR. Compiling it through SelectionDAG on every
launch cost ~58 s. Instead, the JIT SelectionDAG-compiles the stdlib to an object
file **exactly once** — keyed on the `stdlib.bc` content hash plus the process
triple (`stdlib-jit-v2-<hash>-<triple>.o`) — using the JIT's exact
`TargetMachine` configuration (host CPU, PIC, `CodeModel::Large`,
`OptLevel::None`, per-function/per-data sections). Later runs `addObjectFile` the
cached object, skipping IR parse/clone/SelectionDAG entirely. It is emitted with
`FunctionSections`/`DataSections` so the JIT linker can insert range-extension
stubs. This is why warm `-r`/REPL startup is fast (~2-3 s) despite the stdlib
size.

> The `OptLevel::None` and exact ABI match are deliberate: an earlier mismatch
> produced 3-argument struct corruption. Do not "optimize" the stdlib object
> emission without re-checking the ABI.

## Large code model & arm64 Branch26 veneer

The stdlib (~58 MB of IR) can be JIT-linked more than 128 MB away from runtime
symbols, exceeding the AArch64 `Branch26` (`bl`, ±128 MB) range. Two mechanisms
handle this:

1. The JIT sets `CodeModel::Large` so calls are emitted as far calls
   (`movz`/`movk` + `blr`). This fixed a regression where the growing stdlib
   broke *all* `-r` on arm64 (even `(+ 2 3)`) while AOT stayed fine.
2. On ELF/COFF arm64 the JIT installs a `Branch26RangeExtensionPlugin` on the
   JITLink object-linking layer that veneers every out-of-range Branch26 edge
   through an inline absolute-jump stub. It is a no-op off arm64 and on Mach-O
   (where `CodeModel::Large` already emits correct far calls). Disable it with
   `ESHKOL_JIT_NO_BRANCH26_VENEER=1`.

## Compile threads

`ESHKOL_JIT_COMPILE_THREADS` sets the ORC compile-thread count (default
`hardware_concurrency()/2`, clamped to [1,16]; accepts 1-64). Higher counts
reduce the materialization-lock contention that otherwise serializes
`parallel-map` workers behind a single compile thread — at the cost of memory
(each thread holds its own `LLVMContext` and a cloned module). See
[parallelism](parallelism.md).

## Related

- [Environment variables](environment-variables.md)
- [`eshkol-run`](eshkol-run.md) / [`eshkol-repl`](eshkol-repl.md)
