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
| `ESHKOL_PATH` | Module/include search path for `require`. |
| `ESHKOL_LIB_DIR` | Directory to locate the precompiled stdlib / runtime libraries. |
| `ESHKOL_PROJECT_ROOT` | Project root used for relative paths in exception/backtrace reporting. |

## Resource limits

Read by `lib/core/resource_limits.cpp`. Size vars accept `K`/`M`/`G` suffixes.

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_MAX_HEAP` | Max heap bytes (soft limit at 80%). | 1 GiB |
| `ESHKOL_MAX_STACK` | Max interpreter stack depth. | 100000 |
| `ESHKOL_STACK_SIZE` | OS `RLIMIT_STACK` target (min 1 MiB). | 512 MB |
| `ESHKOL_MAX_STRING_LEN` | Max string length. | 100 MiB |
| `ESHKOL_MAX_TENSOR_ELEMS` | Max tensor element count. | 1e9 |
| `ESHKOL_TIMEOUT_MS` | Max execution time (ms). | 30000 |
| `ESHKOL_VM_MAX_INSN` | VM runaway-instruction guard. | 10000000 |
| `ESHKOL_ENFORCE_LIMITS` | Enforce hard limits (abort on exceed). | true |
| `ESHKOL_LIMIT_WARNINGS` | Emit soft-limit warnings. | true |

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
| `ESHKOL_CXX_COMPILER` | C++ driver used for AOT and persistent-cache native links. Useful when LLVM is installed outside PATH or the package builder's original prefix. | build-time driver if present; otherwise `clang++`/`c++` discovery |
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
