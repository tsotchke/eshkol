# Environment Variables

User-facing environment variables read by the Eshkol runtime and toolchain.
Boolean flags accept truthy/falsey spellings (`1`/`0`, `true`/`false`,
`on`/`off`, `yes`/`no`) unless noted.

## JIT & run cache

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_JIT_CACHE` | Persistent AOT run cache for `-r`; disable with `0`/`false`/`off`/`no`. When on, `-r` compiles the file once to a standalone binary and re-execs the cached one on later runs. | enabled |
| `ESHKOL_JIT_CACHE_DIR` | Run-cache directory. | `$XDG_CACHE_HOME/eshkol/jit` or `$HOME/.cache/eshkol/jit` (Unix); `%LOCALAPPDATA%\eshkol\jit` (Windows); else temp |
| `ESHKOL_JIT_CACHE_TRACE` | Print `[jit-cache] <hit\|miss\|bypass>` to stderr. | off |
| `ESHKOL_JIT_COMPILE_THREADS` | ORC compile-thread count (accepts 1-64). More threads reduce materialization-lock contention (which serializes parallel-map workers) at higher memory cost. | `hardware_concurrency()/2`, clamped to [1,16] |
| `ESHKOL_JIT_NO_BRANCH26_VENEER` | Disable the arm64 Branch26 range-extension veneer in the JIT linker (escape hatch). | off (veneer on) |

See [JIT internals](jit-internals.md) for details, including cache-key
invalidation and the stdlib object cache.

## Module & library search paths

| Variable | Effect |
|----------|--------|
| `ESHKOL_PATH` | Module/include search path for `require`. Searched after the requiring file's own directory and the project root, and **before** the installed `lib/` tree, so a path you name overrides a module that ships with the compiler. `-I` directories are merged into it. |
| `ESHKOL_LIB_DIR` | Directory holding the precompiled stdlib (`stdlib.o`, `stdlib.bc`) and the runtime archives (`libeshkol-runtime.a`, or the legacy `libeshkol-static.a`). Highest precedence of all: it is searched — together with its `eshkol/` subdirectory — before `-L` paths, before the compiler's own install, and before any system location. |
| `ESHKOL_SYSTEM_PREFIXES` | Replaces the built-in list of system install prefixes searched as a last resort (`/usr/local`, `/usr`, `/opt/homebrew`). Path-list syntax; each entry contributes `<prefix>/lib`, `<prefix>/lib/eshkol` and `<prefix>/share/eshkol/lib`. For unusual installs and for packaging tests that must not read the host's real system directories. |
| `ESHKOL_PROJECT_ROOT` | Project root used for relative paths in exception/backtrace reporting. |

### Resolution precedence

Every install artifact the toolchain resolves at run time — the runtime
archive, the agent-FFI archives beside it, `stdlib.o`, `stdlib.bc`, and the
`lib/**.esk` module tree — is looked up in one order, highest precedence
first:

1. `$ESHKOL_LIB_DIR` (native artifacts) / `$ESHKOL_PATH` (module sources).
2. Directories named by `-L` (native) or `-I` (modules) flags.
3. The install the running compiler belongs to: the directory holding its
   **real** path (symlinks resolved, so a `bin/eshkol-run` symlink into a
   Homebrew Cellar keg resolves inside the keg), plus that directory's
   `../lib`, `../lib/eshkol` and `../share/eshkol/lib`.
4. The working directory and its `build/` trees — for running out of a build
   tree during development.
5. The system prefixes.

Within one directory the split runtime archive (`libeshkol-runtime.a`) is
preferred over the legacy aggregate (`libeshkol-static.a`); the search never
moves to a lower-precedence directory while the current one can satisfy the
request. A compiler therefore always links its own runtime, and an install
that ships only the legacy archive name is not overtaken by an unrelated
`libeshkol-runtime.a` elsewhere on the machine.

If an artifact does come from a system location, `eshkol-run` says so on
stderr with the path it used. Archives carry the Eshkol version they were
built from; when that disagrees with the running compiler, the message is a
warning, because such an archive can satisfy every symbol and still have been
built against a different runtime layout.

## Resource limits

Read by `lib/core/resource_limits.cpp`. Size vars accept `K`/`M`/`G` suffixes.

| Variable | Effect | Default | Exit status when exceeded |
|----------|--------|---------|---------------------------|
| `ESHKOL_MAX_HEAP` | Max heap bytes (soft limit at 80%). | 1 GiB | 120 |
| `ESHKOL_MAX_STACK` | Max interpreter stack depth. | 100000 | 121 |
| `ESHKOL_STACK_SIZE` | OS `RLIMIT_STACK` target (min 1 MiB). | 512 MB | — |
| `ESHKOL_MAX_STRING_LEN` | Max string length. | 100 MiB | 123 |
| `ESHKOL_MAX_TENSOR_ELEMS` | Max tensor element count. | 1e9 | 122 |
| `ESHKOL_TIMEOUT_MS` | Max execution time (ms); `0` = unlimited. | 30000 | 124 |
| `ESHKOL_VM_MAX_INSN` | Bytecode-VM runaway-instruction guard; `0` = unlimited. | 10000000 | 125 |
| `ESHKOL_ENFORCE_LIMITS` | Enforce hard limits (terminate on exceed). | true | — |
| `ESHKOL_LIMIT_WARNINGS` | Emit soft-limit warnings. | true | — |

### What "enforced" means

With `ESHKOL_ENFORCE_LIMITS=true` (the default), exceeding a hard limit ends
the run immediately. The runtime flushes whatever the program has already
written, prints one line to stderr naming the limit, the configured ceiling and
the variable that set it —

```
eshkol: fatal: Heap hard limit exceeded (limit 1048576 bytes, set by ESHKOL_MAX_HEAP): arena block
```

— and exits with the status in the table above. The statuses are distinct per
limit so a supervising process can tell which ceiling was hit without parsing
the message; `124` for the execution timeout matches GNU coreutils `timeout(1)`
and the convention already used by `run-command` / `run-argv`. They are defined
as `ESHKOL_EXIT_LIMIT_*` in `inc/eshkol/core/resource_limits.h`.

With `ESHKOL_ENFORCE_LIMITS=false` the ceilings become advisory: a breach is
recorded (readable from C via `eshkol_get_last_limit_error()`), reported as a
warning when `ESHKOL_LIMIT_WARNINGS` is on, and the program continues to
completion.

Enforcement is placed so that staying under a limit costs nothing measurable:
the heap ceiling is checked once per arena *block* (a megabyte at a time), not
per allocation, leaving the bump-pointer path untouched; the tensor and string
ceilings are checked once per object created; the VM's instruction guard and
the execution-timeout poll run once per 4096 instructions and once per tail-call
loop back-edge respectively. No check reads or writes a program value, so
enabling limits cannot change a computed result.

`ESHKOL_MAX_STACK` bounds the recursion the runtime's depth guard observes.
Codegen does not yet emit that guard at the entry of every top-level `define`d
function (tracked as ESH-0101, `tests/stress/found/deep_recursion_270k_no_diagnostic.esk`),
so deep non-tail recursion in such a function can still exhaust the native stack
before the ceiling is consulted.

## Parallelism & threading

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_PARALLEL_DISABLE` | `1` forces sequential fallback for parallel primitives. | off |
| `ESHKOL_PARALLEL_ENABLE` | Legacy toggle; `0` disables parallelism. | on |
| `ESHKOL_PARALLEL_NO_WARMUP` | `1` skips the single-item ORC warmup before dispatching workers. | off |
| `ESHKOL_DISABLE_WORK_STEALING` | Set (non-`0`) to use the legacy queue instead of per-worker work-stealing deques. | work-stealing on |
| `ESHKOL_WORKER_STACK_BYTES` | Per-worker pthread stack size (floored at `PTHREAD_STACK_MIN`). | 16 MB |
| `ESHKOL_DEBUG_PAR` | Print pool/task metrics. | off |

See [parallelism & threading](parallelism.md).

## Native link / object emission (AOT)

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_CXX_COMPILER` | C++ driver used for AOT and persistent-cache native links. Useful when LLVM is installed outside PATH or the package builder's original prefix. On ClangCL/MSVC Windows this must belong to a complete matching LLVM toolchain because Eshkol resolves its architecture-specific compiler-rt builtins from that consumer installation. | build-time driver if present; otherwise `clang++`/`c++` discovery |
| `ESHKOL_LINK_TIMEOUT_SECONDS` | AOT native-link timeout (`0` = unbounded). | 300 |
| `ESHKOL_OBJECT_EMIT_TIMEOUT_SECONDS` | Object-emit timeout. | 0 (unbounded) |

## GPU / BLAS / XLA backends

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_GPU_THRESHOLD` / `ESHKOL_GPU_MATMUL_THRESHOLD` | Min element count to dispatch to GPU (set `1` to force GPU). | 100000 |
| `ESHKOL_GPU_PRECISION` | `exact` (sf64) / `high` (df64) / `fast` (f32). | `exact` |
| `ESHKOL_GPU_VERBOSE` | GPU dispatch logging. | off |
| `ESHKOL_BLAS_THRESHOLD` | Min size to use the CPU BLAS backend. | 64 |
| `ESHKOL_XLA_THRESHOLD` | Min size to use the XLA backend. | 100000 |

More GPU tuning vars (`ESHKOL_GPU_PEAK_GFLOPS`, `ESHKOL_GPU_WAIT_TIMEOUT`,
`ESHKOL_ENABLE_TENSORCORE`, `ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_OZAKI_*`) exist for
backend benchmarking — see [platform build notes](../../platform/BUILD_NOTES.md).

## Agent subprocess sandbox

Resource caps applied to children spawned by [`agent.subprocess`](../agent/subprocess.md).

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_SUBPROC_CPU_SEC` | `RLIMIT_CPU` seconds. | 300 |
| `ESHKOL_SUBPROC_MEM_MB` | `RLIMIT_AS` (virtual memory) MB. | 4096 |
| `ESHKOL_SUBPROC_NOFILE` | `RLIMIT_NOFILE` (file descriptors). | 1024 |
| `ESHKOL_SUBPROC_NPROC` | `RLIMIT_NPROC` (processes per user). | 512 |

## Server & misc

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_SERVER_TOKEN` | Auth token for `agent.http-server` / `eshkol-server`. | unset |
| `ESHKOL_VERBOSE` | Verbose logging (`=1`). | off |
| `ESHKOL_ARENA_POISON` | Poison freed arena memory (debug); set non-`0`. | off |
| `ESHKOL_VM_NO_DISASM` | Suppress the VM disassembly dump in `eshkol-vm-standalone`. | off |
| `ESHKOL_DUMP_BC` / `ESHKOL_DUMP_REPL_IR` | Dump bitcode / REPL IR (debug). | off |
