# Environment Variables

User-facing environment variables read by the Eshkol runtime and toolchain.
Boolean flags accept `1`/`0` unless noted. Only `ESHKOL_JIT_CACHE`,
`ESHKOL_JIT_CACHE_TRACE`, `ESHKOL_ENFORCE_LIMITS` and `ESHKOL_LIMIT_WARNINGS`
additionally accept `true`/`false`, `on`/`off` and `yes`/`no`. Every other flag
below either tests the value's first byte against `'0'` or parses a base-10
integer, so for example `ESHKOL_VM_REGION_EVAC=false` leaves the evacuator ON
rather than disabling it. Where a variable is presence-based — any value,
including `0`, takes effect — the row says so.

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

Read by `lib/core/resource_limits.cpp`, except `ESHKOL_STACK_SIZE`, which is
read by `lib/core/runtime_stack_hosted.cpp`. The `resource_limits.cpp` size
vars accept `K`/`M`/`G` suffixes; `ESHKOL_STACK_SIZE` does **not** — it is
parsed with `strtoull` base 0 and no suffix handling, so `512M` reads as 512
bytes, falls below the 1 MiB floor, and silently falls back to the 512 MB
default. Give it a plain byte count.

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

### Limits are opt-in

A ceiling binds a run **only when that run asks for it** — by setting the
variable (or setting the corresponding `ESHKOL_LIMIT_ACTIVE_*` bit before
`eshkol_set_limits()`). The defaults in the table are the values a limit takes
*when you turn it on*; they are not ceilings every program is silently held to.

This is a deliberate distinction, not an omission, and it is the ruled v1.3.4
behaviour: ceilings are opt-in, so shipping behaviour is unchanged for every
existing program. Whether the documented defaults should also bind an
unconfigured run is deferred as a v1.3.5 policy question. The defaults are real
numbers that real programs pass: `tests/features/blc_test.esk` in this
repository allocates past 1 GiB, and the bytecode VM's computed-goto dispatch
never had an instruction guard at all. Applying every documented default to
every run would not be enforcing what the docs say — it would impose a new
ceiling on every existing program. Whether the defaults should also bind an
unconfigured run is a release decision, not a bug fix.

So: `eshkol-run prog.esk` is unbounded, exactly as before.
`ESHKOL_MAX_HEAP=512M eshkol-run prog.esk` is bounded at 512 MiB and will be
terminated if it exceeds that.

### What "enforced" means

With `ESHKOL_ENFORCE_LIMITS=true` (the default), exceeding an active limit ends
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

The timeout poll is emitted for hosted native codegen only. The watchdog that
raises the interrupt lives in the hosted runtime, which a standalone
freestanding object and a `--wasm` module do not link at all — so in those
profiles there is nothing that could request an interrupt, and the back-edge
poll is not emitted rather than left calling a symbol the profile does not
have. The environment variables above describe hosted `eshkol-run` execution.

### Two limits that are narrower than the table suggests

`ESHKOL_MAX_STACK` bounds the recursion the runtime's depth guard observes.
Codegen does not yet emit that guard at the entry of every top-level `define`d
function (tracked as ESH-0101, `tests/stress/found/deep_recursion_270k_no_diagnostic.esk`),
so deep non-tail recursion in such a function can still exhaust the native stack
before the ceiling is consulted.

The execution timer follows the same opt-in rule: the watchdog is armed only
when `ESHKOL_TIMEOUT_MS` is present in the environment (`eshkol_runtime_init()`),
so a run that does not set it is not on a clock. Arming a 30-second wall-clock
kill on every invocation would also bound AOT compilation and interactive REPL
sessions.

The `ESHKOL_MAX_STACK` gap above (ESH-0101) is likewise a ledgered v1.3.5 item:
wiring the variable where the guard already runs is the v1.3.4 scope, extending
guard coverage to every top-level `define` is not.

### Bytecode-VM region reclamation and heap growth watchdog

`(with-region ...)` reclaims on the bytecode VM as of the Stage-1 region
evacuator. **Outside** a region the VM heap has no reclamation at all — no
garbage collector, no per-loop nursery — so a resident VM workload that never
opens a region still grows monotonically. See [Memory model](memory-model.md)
and `docs/KNOWN_ISSUES.md`. None of these knobs changes any answer; the coverage
gate re-runs its fixture with reclamation on and off and requires identical
results.

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_VM_HEAP_BUDGET_MB` | VM arena size past which a diagnostic names the growth and the mechanism that reclaims. `0` disables the watchdog. | 1024 |
| `ESHKOL_VM_HEAP_BUDGET_FATAL` | Make crossing the budget exit nonzero instead of advisory, so a lane can gate on it. | off |
| `ESHKOL_VM_REGION_QUIET` | Suppress the one-time note that a VM `region-close` reclaims no heap (the handle surface is still bookkeeping-only), and the note a pinned region prints. | off |
| `ESHKOL_VM_REGION_EVAC` | `0` disables region reclamation entirely and restores the pre-Stage-1 pass-through. Present so a gate can measure the same program with and without it. | on |
| `ESHKOL_ARENA_POISON` | The same variable the native arena reads. On the VM it makes a region pop keep dead blocks mapped and stamped `0xCB`, and stops retired heap indices from being recycled, so a dangling reference faults instead of aliasing a fresh object. Diagnostic use: it retains all the memory a pop would have freed. Set it to exactly `1`: the native arena and the VM evacuator test the first byte, while `lib/backend/vm_arena.h` compares the whole string against `"0"`, so a value like `01` arms only one of the three readers. | off |
| `ESHKOL_VM_REGION_VERIFY` | After each region pop, run an audit independent of the mark: scan the object table and the root set for any surviving reference to an index the pop retired, and report it on stderr. Implied by `ESHKOL_ARENA_POISON`. | off |
| `ESHKOL_VM_REGION_VERIFY_FATAL` | Make that audit exit nonzero, so a lane can gate on it. | off |
| `ESHKOL_VM_REGION_COMPACT` | `0` stops a surviving object's fixed-size header from being copied out of the dying region, so its whole arena block is retained instead. Diagnostic only — it keeps every address stable. Forced off under `ESHKOL_ARENA_POISON`. | on |
| `ESHKOL_VM_REGION_RECYCLE` | `0` stops retired heap indices from being handed out again. Costs 8 bytes per reclaimed object in the object table; makes a stale reference read as an invalid heap pointer forever. Forced off under `ESHKOL_ARENA_POISON`. | on |

## Parallelism & threading

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_PARALLEL_DISABLE` | `1` forces sequential fallback for parallel primitives. | off |
| `ESHKOL_PARALLEL_ENABLE` | Legacy toggle; `0` disables parallelism. | on |
| `ESHKOL_PARALLEL_NO_WARMUP` | Skips the single-item ORC warmup before dispatching workers. Presence-based: any value, including `0`, takes effect. | off |
| `ESHKOL_DISABLE_WORK_STEALING` | Set to anything whose first byte is not `0` — the empty string included — to use the legacy queue instead of per-worker work-stealing deques. | work-stealing on |
| `ESHKOL_WORKER_STACK_BYTES` | Per-worker pthread stack size (floored at `PTHREAD_STACK_MIN`). | 16 MB |
| `ESHKOL_DEBUG_PAR` | Print pool/task metrics. Presence-based: any value, including `0`, takes effect. | off |

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
| `ESHKOL_GPU_THRESHOLD` | Min element count for the Metal/CUDA backends to dispatch to GPU (set `1` to force GPU). Only values greater than 0 apply. | 100000 |
| `ESHKOL_GPU_MATMUL_THRESHOLD` | Min matmul element count for the BLAS backend's GPU path. A separate knob from `ESHKOL_GPU_THRESHOLD`, read through `atoll` and applied unconditionally, so `0` is accepted. | 1000000000 |
| `ESHKOL_GPU_PRECISION` | `exact` (sf64) / `high` (df64) / `fast` (f32). | `exact` |
| `ESHKOL_GPU_VERBOSE` | CUDA dispatch logging. Presence-based: any value, including `0`, enables it. The Metal backend ignores this variable; use `ESHKOL_VERBOSE=1` there. | off |
| `ESHKOL_BLAS_THRESHOLD` | Min size to use the CPU BLAS backend. | 64 |
| `ESHKOL_XLA_THRESHOLD` | Min size to use the XLA backend. | 100000 |

More GPU tuning vars (`ESHKOL_GPU_PEAK_GFLOPS`, `ESHKOL_GPU_WAIT_TIMEOUT`,
`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_OZAKI_*`, `ESHKOL_MATMUL_ACCURACY`,
`ESHKOL_CUDA_F64_KERNEL`, `ESHKOL_OZAKI_CUDA_T`, and the Metal kernel-tiling
family `ESHKOL_SF64_*` / `ESHKOL_DF64_*` / `ESHKOL_F32S*` / `ESHKOL_FP*`) exist
for backend benchmarking — see
[platform build notes](../../platform/BUILD_NOTES.md).
`ESHKOL_ENABLE_TENSORCORE` is a documented capability that no code currently
reads; Tensor Core selection is a CMake option (`ESHKOL_TENSORCORE_ENABLED`)
today, and wiring the runtime variable is an open build item.

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
| `ESHKOL_VERBOSE` | Metal GPU per-call logging. Requires a leading `1`. Despite the name it affects nothing outside the Metal backend; the CUDA equivalent is `ESHKOL_GPU_VERBOSE`. | off |
| `ESHKOL_ARENA_POISON` | Poison freed arena memory (debug). See the VM region table above for the accepted-value caveat; set it to exactly `1`. | off |
| `ESHKOL_ARENA_REPORT` | Set to exactly `1` to print the process-global arena's own byte total once at exit, on stderr, as `[eshkol-arena] global_total_allocated_bytes=N`. Diagnostic only — it changes no allocation behaviour. This is the retention signal `tests/memory/resident_longrun_flat_gate.sh` gates on, because it is deterministic to the byte, whereas peak RSS is a high-water mark of *instantaneous* residency and reads low on a loaded host. | off |
| `ESHKOL_VM_NO_DISASM` | Suppress the VM disassembly dump in `eshkol-vm-standalone`. | off |
| `ESHKOL_DUMP_BC` / `ESHKOL_DUMP_REPL_IR` | Dump bitcode / REPL IR (debug). | off |

## Compiler and codegen diagnostics

Read by the compiler front end and LLVM back end. Diagnostic and
build-reproduction knobs rather than user-facing configuration, but they change
observable behaviour and are listed here so nothing the runtime reads is
undocumented.

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_TARGET_CPU` | Override the LLVM target CPU used for codegen (`lib/backend/llvm_codegen.cpp`, `lib/backend/tensor_codegen.cpp`). | host CPU |
| `ESHKOL_TARGET_FEATURES` | Override the LLVM target feature string. | host features |
| `ESHKOL_TAIL_TRANSFER_ONLY` | Force every mutual tail call onto the tail-transfer dispatcher, bypassing the `musttail` lowering. Used by the TCO gates to exercise the portable path on a target that could have used `musttail`. | off |
| `ESHKOL_NO_ITER_SCOPE` | Disable the ESH-0214b per-iteration nursery, restoring pre-v1.3.1 loop allocation. Presence-based. Diagnostic only: it turns off a shipped memory-reclamation feature. | off (nursery on) |
| `ESHKOL_AOT_PHASE_TRACE` | Emit per-phase AOT compile tracing. | off |
| `ESHKOL_PHASE_TIME` | Print per-phase wall-clock timings from `eshkol-run`. | off |
| `ESHKOL_NODE_IDENTITY_STATS` | Print `eshkol-node-identity: allocated=N queried=N resolved=N located=N extent=N` at process exit. Read by `scripts/run_node_identity_gate.py` (ADR-0000 Stage 1). | off |
| `ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR` | Directory for executable language-coverage traces. Setting it changes which execution path `eshkol-run` takes, which is the divergence the `module_load_path_engine_parity_gate` pins. | unset |
| `ESHKOL_DEBUG_DL` | REPL/JIT dynamic-loader debug output. | off |
| `ESHKOL_INTERNAL_CACHE_BUILD` | Set and cleared by `eshkol-run` around its own internal cache build; not intended to be set by hand. | unset |
| `ESHKOL_LINSOLVE_FORCE_DGESV` | Force `linear-solve` onto the LAPACK `dgesv` path instead of the mixed-precision solver. | off |
| `ESHKOL_WEIGHTS_OUT` / `ESHKOL_BC` | Output paths for the weight-matrix transformer's QLMW and bytecode artifacts. | unset |

## Config-layer variables

Read by `lib/core/config.cpp` and declared in `inc/eshkol/core/config.h`. These
are the environment overrides for the TOML configuration file; see
[runtime configuration](../../breakdown/RUNTIME_CONFIGURATION.md).

| Variable | Effect |
|----------|--------|
| `ESHKOL_LOG_LEVEL` / `ESHKOL_LOG_FORMAT` / `ESHKOL_LOG_FILE` | Logging level, format and destination. |
| `ESHKOL_OPT_LEVEL` | Default optimisation level. |
| `ESHKOL_ENABLE_SIMD` / `ESHKOL_ENABLE_XLA` / `ESHKOL_ENABLE_GPU` | Feature toggles applied to the loaded configuration. |
| `ESHKOL_DEBUG` | Debug mode. |
| `ESHKOL_LIB_PATH` | Library search path used by the config layer. Distinct from `ESHKOL_LIB_DIR` above, which the module loader reads — setting only one of the pair is a common source of confusion. |

## Non-`ESHKOL_` environment the toolchain consults

`eshkol-run` **writes** `LD_LIBRARY_PATH` and `PATH` when it re-execs a
compiled artifact. It **reads** `LLVM_HOME` / `LLVM_ROOT` / `LLVM_DIR`,
`CUDAToolkit_ROOT` / `CUDA_HOME` / `CUDA_PATH`, `LIBRARY_PATH`, and
`HOME` / `USERPROFILE` / `APPDATA` for cache and config locations. The REPL
honours `NO_COLOR`, `TERM`, `COLORTERM`, `WT_SESSION` and `ANSICON` for colour
detection.
