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
| `ESHKOL_AOT_MODULE_CACHE_DIR` | Directory of the content-addressed AOT module cache: the reusable object `-c` produces for a single source file, keyed on the source, its transitive dependencies, the compiler, target, options, libraries and object ABI. A directory under `/tmp` or `/private/tmp` is refused with a warning. Not used with `-d`, `-g`, `-i` or `-a`. | `$XDG_CACHE_HOME/eshkol/modules` or `$HOME/.cache/eshkol/modules` (Unix); `%LOCALAPPDATA%\eshkol\modules` (Windows); else `.eshkol-aot-cache/modules` relative to the working directory |
| `ESHKOL_AOT_MODULE_CACHE_TRACE` | Print `ESH-0089: AOT module cache <hit\|miss> key=<key>` to stderr. Any value other than empty or `0` enables it. | off |

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

Read by `lib/core/resource_limits.cpp` (`ESHKOL_STACK_SIZE` is read in
`lib/core/runtime_stack_hosted.cpp`, but parsed by the same shared
`eshkol_parse_size()`). Size vars accept `K`/`M`/`G` suffixes, optionally
followed by `i`/`I` and/or `B`/`b` — so `512M`, `512MB`, and `512MiB` are all
equivalent. A value below a variable's own floor (`ESHKOL_STACK_SIZE`'s is
1 MiB) falls back to the default silently.

A value that fails to parse at all is **reported, then ignored** (SW-165).
Every size variable names itself, the offending value and the accepted
grammar on stderr before falling back to its default (since v1.3.5), so an
operator who set a bound is never left believing one is in force when it is
not.

**The heap ceiling is a fail-closed contract (SW-165).** With no
`ESHKOL_MAX_HEAP` set, the default is an accounting reference only: nothing is
printed and no run is stopped. With one set, crossing it is reported **once**,
in the unit it was given, and the process exits nonzero without completing — a
one-shot warning at 80% may precede it, and only for a ceiling that was asked
for. Under `ESHKOL_ENFORCE_LIMITS=false` the breach is recorded and warned about
instead (since v1.3.5). The diagnostic is not repeated per arena block, it
never fires on a default ceiling nobody asked for, and a breach under
enforcement never finishes with exit 0.

| Variable | Effect | Default | Exit status when exceeded |
|----------|--------|---------|---------------------------|
| `ESHKOL_MAX_HEAP` | Max heap bytes; fail-closed when set (soft warning at 80%). | 1 GiB, accounting only | 120 |
| `ESHKOL_MAX_STACK` | Max interpreter stack depth. | 100000 | 121 |
| `ESHKOL_STACK_SIZE` | Native stack target and guard ceiling (min 1 MiB). | 512 MiB | 121 |
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

This is a deliberate distinction, not an omission, and it is the ruled
behaviour through v1.3.5: ceilings are opt-in, so shipping behaviour is
unchanged for every existing program. The defaults are real numbers that real
programs pass: `tests/features/blc_test.esk` in this repository allocates past
1 GiB, and the bytecode VM's computed-goto dispatch never had an instruction
guard at all. Applying every documented default to every run would not be
enforcing what the docs say — it would impose a new ceiling on every existing
program. Whether the defaults should also bind an unconfigured run stays a
release decision rather than a bug fix, and it is not taken in this release.

What v1.3.5 **does** change is what happens once a ceiling you asked for is
crossed: enforcement is fail-closed (see above), and a malformed value is
reported rather than silently discarded.

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

### Native stack guard

`ESHKOL_STACK_SIZE` is applied by hosted native JIT and AOT entry code. The
runtime raises the process's soft `RLIMIT_STACK` when the operating system
allows it, then emits a cheap, stateless headroom check at every generated user
function entry. A check that reaches the reserved guard margin prints

```
eshkol: stack overflow: recursion depth exceeded the 512 MiB stack (ESHKOL_STACK_SIZE); use tail recursion, or raise ESHKOL_STACK_SIZE and the OS stack limit to allow deeper recursion
```

and exits 121 (the size named is whatever `ESHKOL_STACK_SIZE` resolved to).
Raising `ESHKOL_STACK_SIZE` — and the OS stack limit, see below — completes the
same program. This closes ESH-0101 / ESH-0112 (ledger SW-81): the former
failure was the guard-page trap with no handler installed, which surfaced as a
bare SIGILL with no message. If a large frame steps over the margin, the POSIX
SIGSEGV/SIGBUS backstop runs on `sigaltstack` and identifies faults in the
thread's guard region. Runtime-created parallel workers install their own
alternate signal stack because `sigaltstack` is per thread; their stack size is
controlled separately by `ESHKOL_WORKER_STACK_BYTES`.

On Linux the initial thread's reachable stack extent is fixed from the soft
`ulimit -s` at process launch. To test a 1 GiB completion leg, launch the gate
with a generous inherited limit, for example `ulimit -s 1048576`, then set
`ESHKOL_STACK_SIZE=1G`. Raising `RLIMIT_STACK` after launch cannot enlarge an
initial stack that was created smaller. The 512 MiB default is therefore a
guard target, not a promise that every host grants 512 MiB.

`ESHKOL_MAX_STACK` remains the optional software recursion-depth ceiling for
the runtime paths that use the frame counter. It is complementary to the
native-byte guard: the byte guard covers plain user recursion and reports a
stack failure before the process can fall into an undiagnosed signal.

The execution timer follows the same opt-in rule: the watchdog is armed only
when `ESHKOL_TIMEOUT_MS` is present in the environment (`eshkol_runtime_init()`),
so a run that does not set it is not on a clock. Arming a 30-second wall-clock
kill on every invocation would also bound AOT compilation and interactive REPL
sessions.

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
| `ESHKOL_WORKER_STACK_BYTES` | Per-worker pthread stack size (size grammar accepted; floored at `PTHREAD_STACK_MIN`). | 16 MiB |
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
| `ESHKOL_SF64_KERNEL` | Metal exact-tier (float64) matmul kernel: `fp53` (fixed-point exact, the default tier-0 kernel), `legacy` or `v2` (software float64), `ozaki` (Ozaki-II exact, for matrices of at least 512 in every dimension), `ozaki-fast` (enables the reduced-precision Ozaki-II fast tier, which falls back to the exact tier on any failure). | fp53 |
| `ESHKOL_F32_KERNEL` | Metal `fast` tier: `simd` selects the SIMD-group float32 kernel instead of Metal Performance Shaders. | MPS |
| `ESHKOL_CUDA_LIBRARY_PATH` | Path list (`:`, `;` on Windows) searched first for the CUDA runtime libraries, ahead of `CUDAToolkit_ROOT`, `CUDA_HOME`, `CUDA_PATH`, `LIBRARY_PATH` and `LD_LIBRARY_PATH`. | unset |

More GPU tuning vars (`ESHKOL_GPU_PEAK_GFLOPS`, `ESHKOL_GPU_WAIT_TIMEOUT`,
`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_OZAKI_*`) exist for backend benchmarking —
see [platform build notes](../../platform/BUILD_NOTES.md).

TensorCore support is a **build-time** choice, not a runtime environment
variable: the CMake option `ESHKOL_TENSORCORE_ENABLED` (off by default) links
the canonical Eshkol adapter against an installed TensorCore package
(`CMakeLists.txt`). There is no `ESHKOL_ENABLE_TENSORCORE` runtime knob — no
code reads it — so setting it in the environment has no effect either way.
`ESHKOL_BLAS_PEAK_GFLOPS`, `ESHKOL_OZAKI_*`, `ESHKOL_MATMUL_ACCURACY`,
`ESHKOL_CUDA_F64_KERNEL`, `ESHKOL_OZAKI_CUDA_T`, and the Metal kernel-tiling
family `ESHKOL_SF64_*` / `ESHKOL_DF64_*` / `ESHKOL_F32S*` / `ESHKOL_FP*`) exist
for backend benchmarking — see
[platform build notes](../../platform/BUILD_NOTES.md).

## Agent subprocess sandbox

Resource caps applied to children spawned by [`agent.subprocess`](../agent/subprocess.md).

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_SUBPROC_CPU_SEC` | `RLIMIT_CPU` seconds. | 300 |
| `ESHKOL_SUBPROC_MEM_MB` | `RLIMIT_AS` (virtual memory) MB. | 4096 |
| `ESHKOL_SUBPROC_NOFILE` | `RLIMIT_NOFILE` (file descriptors). | 1024 |
| `ESHKOL_SUBPROC_NPROC` | `RLIMIT_NPROC` (processes per user). | 512 |
| `ESHKOL_SUBPROC_MAX_CONCURRENT` | Maximum number of concurrently running spawned children; a spawn beyond it is refused. A value that is not a positive base-10 integer falls back to the default; values above 4096 are clamped to 4096. | 64 |

## Server & misc

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_SERVER_TOKEN` | Auth token for `agent.http_server` / `eshkol-server`. | unset |
| `ESHKOL_VERBOSE` | Metal GPU per-call logging. Requires a leading `1`. Despite the name it affects nothing outside the Metal backend; the CUDA equivalent is `ESHKOL_GPU_VERBOSE`. | off |
| `ESHKOL_ARENA_POISON` | Poison freed arena memory (debug). See the VM region table above for the accepted-value caveat; set it to exactly `1`. | off |
| `ESHKOL_ARENA_REPORT` | Set to exactly `1` to print the process-global arena's own byte total once at exit, on stderr, as `[eshkol-arena] global_total_allocated_bytes=N`. Diagnostic only — it changes no allocation behaviour. This is the retention signal `tests/memory/resident_longrun_flat_gate.sh` gates on, because it is deterministic to the byte, whereas peak RSS is a high-water mark of *instantaneous* residency and reads low on a loaded host. | off |
| `ESHKOL_VM_NO_DISASM` | Suppress the VM disassembly dump in `eshkol-vm-standalone`. | off |
| `ESHKOL_REGISTRY` | Git URL of the package registry `eshkol-pkg` uses. | the project registry |
| `ESHKOL_COMPILER` | Compiler `eshkol-pkg build` invokes. | `eshkol-run` |
| `ESHKOL_DUMP_BC` / `ESHKOL_DUMP_REPL_IR` | Dump bitcode / REPL IR (debug). | off |

## Compiler and codegen diagnostics

Read by the compiler front end and LLVM back end. Diagnostic and
build-reproduction knobs rather than user-facing configuration, but they change
observable behaviour and are listed here so nothing the runtime reads is
undocumented.

| Variable | Effect | Default |
|----------|--------|---------|
| `ESHKOL_LLVM_REMARKS` | Let LLVM optimization remarks reach stderr. Off by default: the optimizer runs behind `eshkol-run`, so its stderr is the compiled program's stderr and a remark about a stdlib loop is not part of the program's output. Covers both the AOT and the `-r` JIT pipeline. Errors and module-verification diagnostics are never suppressed. | off |
| `ESHKOL_TARGET_CPU` | Override the LLVM target CPU used for codegen (`lib/backend/llvm_codegen.cpp`, `lib/backend/tensor_codegen.cpp`). | host CPU |
| `ESHKOL_TARGET_FEATURES` | Override the LLVM target feature string. | host features |
| `ESHKOL_TAIL_TRANSFER_ONLY` | Force every mutual tail call onto the tail-transfer dispatcher, bypassing the `musttail` lowering. Used by the TCO gates to exercise the portable path on a target that could have used `musttail`. | off |
| `ESHKOL_NO_ITER_SCOPE` | Disable the ESH-0214b per-iteration nursery, restoring pre-v1.3.1 loop allocation. Presence-based. Diagnostic only: it turns off a shipped memory-reclamation feature. | off (nursery on) |
| `ESHKOL_AOT_PHASE_TRACE` | Emit per-phase AOT compile tracing. | off |
| `ESHKOL_PHASE_TIME` | Print per-phase wall-clock timings from `eshkol-run`. | off |
| `ESHKOL_NODE_IDENTITY_STATS` | Print `eshkol-node-identity: allocated=N queried=N resolved=N located=N extent=N` at process exit. Read by `scripts/run_node_identity_gate.py` (ADR-0000 Stage 1). | off |
| `ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR` | Directory for executable language-coverage traces. When set, native codegen instruments every executed construct and the VM reports its native-dispatch, call and form markers. Each process appends to its own `language-coverage-<pid>.tsv` in the directory, created if absent, and writes each distinct record once. Each instrumented site in generated code owns a guard, so the runtime is entered once per site however often a loop runs it; the VM consults direct-mapped first-sighting tables before entering the runtime. Fork is detected through a `pthread_atfork` child handler, so a forked child writes its own trace stream rather than sharing the parent's buffer. Records are flushed every 256 records and at normal exit, and `emergency-exit` flushes before `_exit`. Setting it also bypasses the `-r` run cache, so `eshkol-run` takes the in-process path; that divergence is what the `module_load_path_engine_parity_gate` pins. | unset |
| `ESHKOL_LANGUAGE_COVERAGE_HOOK_STATS` | Print `eshkol-language-coverage: exec-hook-entries=N` at process exit: how many times generated code or the VM entered a language-coverage execution hook. Each instrumented site and VM marker is guarded to enter once, so N is bounded by the number of distinct sites, not by iteration counts. Read by `scripts/test_language_coverage_hook_guard.py`. | off |
| `ESHKOL_DEBUG_DL` | REPL/JIT dynamic-loader debug output. | off |
| `ESHKOL_DUMP_IR_ON_VERIFY_FAIL` | When LLVM module verification fails, print the whole module to stderr before the error. Presence-based. | off |
| `ESHKOL_DENSE_TENSOR_AD_NODES` | `0`, `off`, `false` or `no` (any case) makes `matmul`, `tensor-sum` and `tensor-mean` record scalarized tape nodes instead of one dense node each. Gradients are the same either way; `scripts/run_dense_tensor_ad_gate.sh` runs both and compares. | on (dense) |
| `ESHKOL_AD_STRICT` | Strict AD validation. Read by `eshkol_ad_strict_enabled()` (`lib/core/config.cpp`); any value other than empty, `0`, `false` or `FALSE` sets it. No code path consults the predicate yet, so setting it changes nothing; the intended behaviour is in [ADR-0002](../../design/adr/0002-ad-staged-dense-kernels.md). | off |
| `ESHKOL_COMMAND_DEFINES` | Set by `eshkol-run` from its `-D NAME[=VALUE]` options (comma separated) so the bytecode-VM compiler and a cold-cache child see the same `cond-expand` features; not intended to be set by hand. | unset |
| `ESHKOL_TEST_MODEL_IO_FAIL` | Failure-injection point for the atomic model-checkpoint writer. Read only in builds compiled with `ESHKOL_MODEL_IO_TEST_HOOKS` (the model-I/O tests); release binaries do not read it. | unset |
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
`CUDAToolkit_ROOT` / `CUDA_HOME` / `CUDA_PATH` (and, on Windows,
`ProgramFiles` / `ProgramFiles(x86)` to find an installed CUDA toolkit),
`LIBRARY_PATH` and `LD_LIBRARY_PATH` for library directories, `XDG_CACHE_HOME`
and `LOCALAPPDATA` for the run and module cache roots, and
`HOME` / `USERPROFILE` / `APPDATA` for cache and config locations. The REPL
honours `NO_COLOR`, `TERM`, `COLORTERM`, `WT_SESSION` and `ANSICON` for colour
detection, and the terminal-capability builtin also reads `LANG` for UTF-8
support. The system builtins read `TMPDIR`, then `TMP`, then `TEMP` for the
temporary directory, and `USER` (`USERNAME` on Windows) for the current user.

## Test, gate and release harness variables

Read by the scripts under `scripts/` and `tests/`, not by the compiler or the
runtime. How the release uses them is in
[RELEASE_PROCESS.md](../../platform/RELEASE_PROCESS.md).

**Evidence paths.** A relative `TRACE_DIR` or `ICC_TRACE_DIR` is relative to
the repository root. Every script that reads one from its environment makes it
absolute before first use (`scripts/lib/evidence_paths.sh`), because producers
change directory: `ctest --test-dir build --output-junit P` would otherwise
resolve a relative `P` inside `build/`. The path does not have to exist yet, and
symlinks are not resolved.

| Variable | Effect | Default |
|----------|--------|---------|
| `TRACE_DIR` | Completion-oracle evidence directory the gates write to and read from. | `scripts/icc_traces` |
| `ICC_TRACE_DIR` | Evidence directory for `scripts/run_language_coverage.sh`; the readiness recipe and the smoke battery export it equal to `TRACE_DIR`. | `scripts/icc_traces` |
| `ESHKOL_TRACE_DIR`, `ESH0103_TRACE_DIR` | Exported equal to `TRACE_DIR` by the readiness recipe for producers that read these names. | `TRACE_DIR` |
| `BUILD_DIR` | Build tree the harnesses test; a relative value is read against the repository root. | `build` |
| `QUANTUM_BUILD_DIR` | Quantum-enabled build tree that complete language coverage uses. | `build-quantum` |
| `ICC_BIN`, `ICC_REPO` | The ICC binary and the registered repository name the readiness recipe queries. | `icc`, `eshkol_lang` |
| `ARCH_MODEL`, `ARCH_TRACE_GLOB` | Architecture model graded by `icc architecture-verify`, and the glob of its verification traces read at the readiness step. | `.icc/architecture-model.yaml`, `.icc/runtime-traces-oracle-view/*architecture-model-verify-*.jsonl` |
| `ESHKOL_RELEASE_PHASE_ID` | Identity binding the split readiness phases together. Required for a single `--phase`; derived from the workflow run id and attempt under GitHub Actions, and generated for a full run. | derived |
| `ESHKOL_RELEASE_TRACE_ARCHIVE_ROOT` | Where the recipe moves a previous trace cohort, outside the active trace root. | `.scratch/release-readiness-history` |
| `ESHKOL_RELEASE_SANITIZER_ROOT`, `ESHKOL_RELEASE_SANITIZER_REPORT` | Work directory and report of the release sanitizer corpus run. | under `.scratch/v1-3-readiness/` |
| `ESHKOL_DURABLE_WORK_ROOT` | Absolute directory for deterministic evidence roots. Each gate claims a fresh child directory and refuses one that already exists, so a run cannot consume earlier evidence. | unset (scratch space) |
| `ESHKOL_LANGUAGE_COVERAGE_ALREADY_RUN` | `1` stops the smoke battery from re-running language coverage that the baseline phase already produced. | `0` |
| `ESHKOL_EVIDENCE_MAX_AGE_DAYS` | Staleness window of `scripts/check_evidence_staleness.py` (also `--max-age-days`). | 14 |
| `ESHKOL_PYTHON_MODULE_DIR` | Directory holding the built `eshkol` Python module; CTest sets it for `python_bindings_capsule_lifetime`. | `build` |
| `ESHKOL_TEST_TMP_ROOT` | Parent of the per-run scratch directories of the shell suites (`scripts/lib/test_isolation.sh`). | `$TMPDIR` |
| `ESHKOL_TEST_TMP_MAX_AGE_MIN`, `ESHKOL_TEST_TMP_MAX_DIRS`, `ESHKOL_TEST_TMP_MAX_MB` | Pruning bounds for leftover scratch directories from interrupted runs. | 720, 64, 2048 |
| `ESHKOL_TEST_KEEP_TMPDIR` | Keep a suite's scratch directory after it exits, for inspection. | off |
| `ESHKOL_GATE_DISK_CAP_MB` | Disk cap of the long-running memory gates. | 512 or 1024, per gate |
| `ESHKOL_FUZZ_GATE_LIMIT`, `ESHKOL_FUZZ_FULL` | Corpus size of `scripts/run_sanitizer_fuzz.sh`, and `1` for the full generated and test corpus. | 150, `0` |
| `ESHKOL_FUZZ_MAX_MB`, `ESHKOL_FUZZ_MAX_GB` | Artifact disk cap of the fuzz run; the MB value wins. | 300 MB |
| `ESHKOL_RUN_BIN_OVERRIDE`, `ESHKOL_ROSETTE_BUILD_DIR` | Compiler and build tree used by the Rosette Wire oracle. | `BUILD_DIR` |
| `ESHKOL_HTTP_SERVER_SMOKE_TIMEOUT`, `ESHKOL_REPL_MACHINE_READY_TIMEOUT`, `ESHKOL_REPL_MACHINE_DONE_TIMEOUT` | Timeouts, in seconds, of the HTTP-server smoke and the machine-mode REPL tests. | 120, 120, 30 |
| `ESHKOL_EDGE_FAILURE_LINES`, `ESHKOL_TEST_FAILURE_LINES` | Lines of a failing test's output the suites print. | 40 |

Variables local to one script (for example the continuation suite's
`ESHKOL_CONT_*` or the TensorCore integration test's `ESHKOL_TENSORCORE_*`
paths) are documented in that script's header. Self-hosted runner provisioning
variables are in [SELF_HOSTED_RUNNERS.md](../../platform/SELF_HOSTED_RUNNERS.md).
