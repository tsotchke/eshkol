# Platform Program Decisions

This file is an append-only log of architecture and governance decisions for the freestanding / platform program.

## Status Values

- `Accepted`
- `Provisional`
- `Superseded`
- `Deferred`

---

## D-0001

- Date: 2026-04-15
- Status: Accepted
- Title: Start the platform program during `v1.2-scale`

### Context

The public roadmap places embedded cross-compilation in `v1.8-platform`, but the compiler, runtime, and language work needed for that milestone is too large to begin only at `v1.8`.

### Decision

Start the platform program during `v1.2-scale` as a parallel infrastructure effort that converges into the public roadmap by `v1.8-platform`.

### Consequences

- platform work can proceed now
- public roadmap wording does not need to change drastically
- only merge-safe infrastructure lands early

---

## D-0002

- Date: 2026-04-15
- Status: Accepted
- Title: Use a long-lived platform branch

### Context

The current active branch is `feature/v1.2-scale`, and there is already active work in central compiler/runtime files.

### Decision

Use:

- `feature/v1.2-scale` as the current release integration branch
- `feature/platform-freestanding` as the long-lived platform branch

### Consequences

- platform work can advance without destabilizing the release stream
- regular merge discipline is required

---

## D-0003

- Date: 2026-04-15
- Status: Accepted
- Title: The first deliverable is a freestanding toolchain, not a kernel

### Context

The temptation is to start immediately with a bootloader or kernel implementation.

### Decision

The first success metric is a stable freestanding Eshkol platform:

- profiles
- runtime split
- low-level language surface
- freestanding LLVM path
- BSP contract

Kernel work begins downstream only after that foundation exists.

### Consequences

- effort stays focused on reusable infrastructure
- `eshkol-kernel` remains downstream

---

## D-0004

- Date: 2026-04-15
- Status: Accepted
- Title: The VM is a first-class part of the platform strategy

### Context

Eshkol already has a production VM and ESKB format. It is useful beyond browser deployment.

### Decision

The platform program explicitly includes a `freestanding-vm` profile and a VM host-hook architecture.

### Consequences

- platform portability is not limited to LLVM-native targets
- the VM becomes a candidate runtime for monitors, recovery, and constrained systems

---

## D-0005

- Date: 2026-04-15
- Status: Accepted
- Title: First reference targets are QEMU-friendly general-purpose ISAs

### Context

The initial goal is to validate the platform architecture quickly and repeatably.

### Decision

Use these reference targets in order:

1. x86_64 QEMU PC
2. AArch64 QEMU virt
3. RISC-V QEMU virt

Real MCU support comes later once the freestanding contract is stable.

### Consequences

- first bring-up stays debuggable and automatable
- MCU-specific work does not block the core platform architecture

---

## D-0006

- Date: 2026-04-15
- Status: Accepted
- Title: Governance is founder-led with bounded delegation

### Context

The founder/maintainer is doing most of the work, but some bounded tasks may be delegated.

### Decision

Core architecture, syntax, runtime boundary, and repo-boundary decisions remain centralized. Delegation is implementation-focused and topic-branch bounded.

### Consequences

- design coherence is preserved
- helpers can still contribute effectively on decision-complete tasks

---

## D-0007

- Date: 2026-04-15
- Status: Accepted
- Title: Keep the platform integration branch in its own dedicated worktree

### Context

The roadmap branch, the platform integration branch, and short-lived topic branches need to move in parallel without stash churn or accidental branch contamination.

### Decision

Use:

- `~/Desktop/eshkol` for the active release/mainline branch
- `~/Desktop/eshkol-platform` for `feature/platform-freestanding`
- additional short-lived worktrees for active `topic/platform-*` branches when needed

### Consequences

- branch roles stay physically separated
- the platform integration branch remains continuously available for sync and validation
- topic branches remain disposable after merge

---

## D-0008

- Date: 2026-04-15
- Status: Accepted
- Title: Use explicit merge commits for platform syncs and topic-branch integration

### Context

The platform program is long-running and will repeatedly reconcile roadmap syncs, topic slices, and eventual merge-back into the release stream.

### Decision

Use explicit merge commits for:

- syncing mainline into `feature/platform-freestanding`
- merging accepted `topic/platform-*` branches into `feature/platform-freestanding`

### Consequences

- integration history remains legible
- platform slice boundaries stay visible
- later audit, bisect, and rollback are safer

---

## D-0009

- Date: 2026-04-15
- Status: Accepted
- Title: Runtime ownership is responsibility-based, not directory-based

### Context

The current implementation mixes allocator substrate, signal/process/runtime state, configuration, logging, image and ONNX helpers, knowledge-base persistence, and higher-level language services under `lib/core/`. Treating the entire directory as "the runtime" would produce a bad split and make freestanding support harder.

### Decision

The runtime split is responsibility-based:

- `runtime-core` owns the value ABI, allocator substrate, and profile-independent runtime contracts
- `runtime-hosted` owns process, filesystem, terminal, env, temp path, host compiler/linker, and hosted libc wrapper behavior
- higher-level language services remain outside the runtime family split even if they live under `lib/core/`

The concrete baseline for this classification is `docs/platform/RUNTIME_INVENTORY.md`.

### Consequences

- runtime extraction work is driven by documented ownership, not by directory moves
- `platform_runtime.h` and `runtime_exports.h` are treated as hosted-runtime surfaces from the start
- files such as `logic.cpp`, `workspace.cpp`, `inference.cpp`, and `introspection.cpp` are not silently folded into `runtime-core`

---

## D-0010

- Date: 2026-04-15
- Status: Accepted
- Title: Runtime decomposition begins with internal source sets, not immediate archive changes

### Context

The runtime is still delivered as part of `eshkol-static`, and too many core files still straddle freestanding and hosted concerns to make a clean archive split in one step.

### Decision

Introduce explicit internal build buckets first:

- `runtime-core`
- `runtime-hosted`
- `runtime-split-pending`

These are represented in CMake as internal object libraries while `eshkol-static` remains the delivered aggregate archive.

### Consequences

- build ownership becomes explicit without destabilizing downstream link behavior
- the remaining mixed files are visible instead of being silently misclassified

---

## D-0028

- Date: 2026-05-23
- Status: Accepted
- Title: Extract tensor fill helpers from hosted runtime state

### Context

`lib/core/runtime.cpp` mixed hosted runtime lifecycle code with generated-code
helpers that are freestanding-safe. The native tensor fill primitives do not
need signals, process state, files, environment variables, allocation, or host
threading.

### Decision

Move `eshkol_tensor_rect_fill` and `eshkol_tensor_disk_fill` into
`lib/core/runtime_tensor_fill.cpp` and classify that file as `runtime-core`.
Keep `runtime.cpp` in `runtime-split-pending` until its remaining hosted and
core responsibilities are separated.

### Consequences

- runtime-core now owns a concrete extracted piece of the former runtime.cpp
- tensor fill helpers are covered by the runtime boundary test
- the aggregate `eshkol-static` link contract remains unchanged

---

## D-0029

- Date: 2026-05-23
- Status: Accepted
- Title: Extract tensor index helpers from hosted runtime state

### Context

`lib/core/runtime.cpp` still contains several generated-code helper groups
that do not depend on hosted runtime lifecycle behavior. The tensor index
helpers normalize tagged scalar/list indices and compute tensor row-major
offsets using only tagged value layout, cons-cell accessors, tensor metadata,
and raw dimension arrays.

### Decision

Move `eshkol_unwrap_list_index`, `eshkol_tensor_linear_from_index_arg`,
`eshkol_vref_unwrap_index`, and `eshkol_tensor_index_arg_count` into
`lib/core/runtime_tensor_index.cpp` and classify that file as `runtime-core`.
Keep `runtime.cpp` in `runtime-split-pending` until its remaining hosted and
core responsibilities are separated.

### Consequences

- runtime-core now owns tensor index normalization alongside tensor fill helpers
- generated tensor-ref/tensor-set code keeps the same symbol names and link path
- the aggregate `eshkol-static` link contract remains unchanged
- later runtime archive extraction can proceed incrementally from an already-structured build graph

---

## D-0030

- Date: 2026-05-23
- Status: Accepted
- Title: Extract string and UTF-8 helpers into runtime core

### Context

`lib/core/runtime.cpp` still mixes hosted lifecycle behavior with generated-code
helpers. The string byte-length and UTF-8 helpers only inspect Eshkol string
headers, walk byte sequences, and allocate substring results through the arena
string allocator. They do not need host files, environment variables, process
state, signals, or threads.

### Decision

Move `eshkol_string_byte_length`, `eshkol_utf8_strlen`, `eshkol_utf8_ref`, and
`eshkol_utf8_substring` into `lib/core/runtime_string.cpp` and classify that
file as `runtime-core`. Keep `runtime.cpp` in `runtime-split-pending` until its
remaining hosted lifecycle and core parameter/bytevector helpers are split.

### Consequences

- generated string code keeps the same exported helper symbol names
- runtime-core now owns the header-backed string/UTF-8 helper surface
- `runtime.cpp` is smaller and more focused on hosted lifecycle plus remaining
  unsplit helper groups

---

## D-0031

- Date: 2026-05-23
- Status: Accepted
- Title: Extract bytevector helpers into runtime core

### Context

`lib/core/runtime.cpp` still mixes hosted lifecycle state with generated-code
helper groups. The bytevector helpers allocate payload storage through the arena
object allocator, read and write bytes by offset, and copy byte ranges. That
data-path behavior is independent of host files, process control, environment
variables, signals, and threads.

### Decision

Move `eshkol_make_bytevector`, `eshkol_bytevector_u8_ref`,
`eshkol_bytevector_u8_set`, `eshkol_bytevector_length`, and
`eshkol_bytevector_copy` into `lib/core/runtime_bytevector.cpp` and classify the
file as `runtime-core`. Keep the existing fatal error behavior by exporting the
shared runtime fatal helper from `runtime.cpp`; the fatal/panic hook boundary
will be split as a later platform slice.

### Consequences

- generated bytevector code keeps the same exported helper symbol names
- runtime-core now owns the bytevector storage helper surface
- the remaining `runtime.cpp` work is narrowed to hosted lifecycle, fatal-error,
  and parameter-object boundaries

---

## D-0032

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted runtime error handling

### Context

The current fatal/type-error implementation is a generated-code ABI surface, but
its behavior is hosted: it writes diagnostics through stderr/logger state, builds
hosted exception objects, raises through the hosted handler path, and finally
terminates the process. Keeping that implementation in `runtime.cpp` obscures
the remaining lifecycle state work and makes the future freestanding panic hook
harder to see.

### Decision

Move `eshkol_runtime_fatal`, `eshkol_type_error`, and
`eshkol_type_error_with_value` into `lib/core/runtime_errors_hosted.cpp` and
classify that file as `runtime-hosted`. Preserve the symbol names and aggregate
`eshkol-static` link behavior. Runtime-core bytevector helpers can keep
delegating error paths to this symbol until the freestanding panic/error hook
ABI replaces the hosted sink.

### Consequences

- `runtime.cpp` is narrower and no longer owns the stderr/exit fatal path
- runtime-hosted now explicitly owns the current fatal/type-error sink
- the remaining runtime split can focus on lifecycle/signal state and parameter
  object storage

---

## D-0033

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted dynamic parameter storage

### Context

The current `make-parameter` / `parameterize` implementation is a generated-code
ABI surface, but its storage path is not the final freestanding design. Parameter
objects are arena allocated, while each dynamic binding stack grows through
`malloc` / `realloc` and reports stack-growth failures through the hosted logger.
Keeping this block in `runtime.cpp` makes the remaining lifecycle split harder
to review.

### Decision

Move `eshkol_make_parameter`, `eshkol_parameter_push`,
`eshkol_parameter_pop`, `eshkol_parameter_ref`, and the pointer ABI wrappers
into `lib/core/runtime_parameters_hosted.cpp` and classify that file as
`runtime-hosted`. Preserve the symbol names and aggregate `eshkol-static` link
behavior. Treat a freestanding arena-backed parameter stack as a later
implementation replacement behind the same ABI.

### Consequences

- `runtime.cpp` no longer owns generated-code parameter storage helpers
- runtime-hosted explicitly owns the current malloc/realloc-backed parameter
  implementation
- the remaining `runtime.cpp` split can focus on lifecycle, signal handling,
  in-flight operations, and shutdown hooks

---

## D-0044

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted exception runtime

### Context

`arena_memory.cpp` still carried R7RS exception state, `longjmp` handler
dispatch, hosted stderr fallback printing, and REPL forward-reference provider
diagnostics. The provider diagnostic path scans project `.esk` files using the
host filesystem and is explicitly not allocator substrate.

### Decision

Move exception allocation, raised-value state, handler stack management,
forward-reference diagnostics, and exception display into
`lib/core/runtime_exceptions_hosted.cpp`, classified as `runtime-hosted`. Keep
the existing public ABI symbols for generated code and REPL JIT registration.

### Consequences

- `arena_memory.cpp` no longer owns exception state or hosted provider-file scans
- runtime-hosted explicitly owns the current `longjmp`/stderr exception sink
- the remaining split-pending arena file no longer includes filesystem/fstream
  just for exception diagnostics

---

## D-0043

- Date: 2026-05-23
- Status: Accepted
- Title: Extract closure reflection and lambda registry

### Context

`arena_memory.cpp` still carried the generated-code ABI helpers for closure
reflection and the homoiconic lambda registry. These helpers read closure
metadata, allocate procedure-name strings into the arena, and maintain a
function-pointer to S-expression registry for display/JIT integration, but they
are not allocator implementation.

### Decision

Move the closure reflection helpers and lambda registry into
`lib/core/runtime_closure_reflection.cpp`, classified as `runtime-core`. Keep the
existing ABI names for `procedure-arity`, `procedure-name`, variadic checks, and
lambda registry registration/lookup unchanged.

### Consequences

- `arena_memory.cpp` no longer owns closure reflection or lambda registry state
- runtime-core explicitly owns the generated-code closure reflection ABI
- hosted display can depend on the registry without keeping that registry in the
  split-pending arena file

---

## D-0042

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted display and current-port runtime

### Context

`arena_memory.cpp` still carried the display/write implementation and the
runtime cells for `current-output-port`, `current-input-port`, and
`current-error-port`. That code is not allocator substrate: it formats values to
hosted `FILE*` streams, uses `stdout` / `stdin` / `stderr` defaults, emits
UTF-8, and renders higher-level runtime values through hosted output sinks.

### Decision

Move the display/write and current-port implementation into
`lib/core/runtime_display_hosted.cpp`, classified as `runtime-hosted`. Keep the
existing public ABI names for display/write, current-port accessors, list/vector
display, and UTF-8 string construction unchanged.

### Consequences

- `arena_memory.cpp` no longer owns hosted display/write or current-port state
- runtime-hosted explicitly owns the current `FILE*` display sink behavior
- a later freestanding profile can provide target-specific display sinks behind
  the same generated-code ABI

---

## D-0041

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted S-expression reader

### Context

`arena_memory.cpp` still carried the implementation of `eshkol_read_sexpr`.
That reader is not allocator substrate: it tokenizes hosted `FILE*` streams with
`fgetc` / `ungetc`, applies reader-specific depth and token guards, allocates
tagged values into the arena, and interns symbols through the process-global
symbol table.

### Decision

Move the reader implementation into `lib/core/runtime_reader_hosted.cpp`,
classified as `runtime-hosted`. Keep the exported ABI name
`eshkol_read_sexpr` unchanged for generated code and REPL/JIT callers.

### Consequences

- `arena_memory.cpp` no longer owns the FILE-backed S-expression reader
- hosted reader behavior is explicit in the runtime-hosted source set
- a later freestanding profile can provide a target-specific reader stream
  behind the same generated-code ABI

---

## D-0040

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted arena synchronization primitives

### Context

`arena_memory.cpp` still owned platform synchronization setup for
thread-safe arenas, hash-table mutation/iteration guards, and global arena
initialization. That code selected between Windows `std::mutex` /
`std::call_once` and POSIX `pthread_mutex_t` / `pthread_once_t`, keeping direct
hosted thread primitives in the allocator split-pending file even though the
arena code only needs opaque lock handles and a once gate.

### Decision

Move arena lock creation/destruction/lock/unlock helpers, the hash-table lock,
and the global arena once gate into `lib/core/runtime_arena_sync_hosted.cpp`,
classified as `runtime-hosted`. Keep `arena_create_threadsafe`, `arena_lock`,
`arena_unlock`, hash-table APIs, and global arena selection in
`arena_memory.cpp` so existing runtime callers keep the same public ABI.

### Consequences

- `arena_memory.cpp` no longer includes `pthread.h` or `<mutex>`
- hosted arena and hash-table synchronization is explicit in the
  runtime-hosted source set
- a later freestanding profile can replace the internal sync helpers with
  target-specific critical-section and once primitives

---

## D-0039

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted string-port helpers

### Context

`arena_memory.cpp` still carried the runtime helpers for `open-input-string`,
`open-output-string`, and `get-output-string`. These helpers are not arena
allocation substrate: they allocate hosted `FILE*` ports using `tmpfile`,
`fmemopen`, and `open_memstream`, plus a hosted side table for output buffers.
Keeping them in the arena split-pending file tied string-port I/O and temporary
file behavior to the core allocator slice.

### Decision

Move the string-port helper ABI into `lib/core/runtime_string_ports_hosted.cpp`
and classify that unit as `runtime-hosted`. Keep the exported symbol names
unchanged for generated code and REPL/JIT callers.

### Consequences

- `arena_memory.cpp` no longer owns FILE-backed string-port construction
- hosted string-port behavior is explicit in the runtime-hosted source set
- a later freestanding profile can replace this implementation with target
  string streams behind the same generated-code ABI

---

## D-0038

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted process stack setup

### Context

`arena_memory.cpp` still carried `eshkol_init_stack_size`, which is not arena
allocation substrate. Its current implementation reads `ESHKOL_STACK_SIZE` and,
on POSIX hosts, calls `getrlimit` / `setrlimit` for `RLIMIT_STACK` before
generated code runs. That is hosted process startup behavior and keeps
`sys/resource.h` coupled to the arena split-pending file.

### Decision

Move `eshkol_init_stack_size` into `lib/core/runtime_stack_hosted.cpp`,
classified as `runtime-hosted`. Keep the ABI name unchanged for generated code
and the REPL JIT symbol registry.

### Consequences

- `arena_memory.cpp` no longer includes `sys/resource.h`
- hosted stack-limit setup is explicit in the runtime-hosted source set
- freestanding profiles can later provide target startup or no-op stack setup
  behind the same ABI

---

## D-0037

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted runtime lifecycle

### Context

After generated-code helpers, error sinks, dynamic parameters, in-flight
operations, shutdown hooks, and signal handlers were split out, `runtime.cpp`
still owned hosted lifecycle state: runtime/shutdown atomics, interrupt
request/clear behavior, stdout buffering, operation draining, hook dispatch,
signal restore sequencing, and lifecycle logging. The only freestanding-safe
piece left in that file was the public interrupt flag used by the inline hot
path in `runtime.h`.

### Decision

Move `eshkol_runtime_init`, `eshkol_runtime_shutdown`, interrupt request/clear,
shutdown reason access, and runtime state access into
`lib/core/runtime_lifecycle_hosted.cpp`, classified as `runtime-hosted`.
Keep `lib/core/runtime.cpp` as a tiny runtime-core translation unit defining
`g_eshkol_interrupt_flag`.

### Consequences

- `runtime.cpp` no longer owns hosted lifecycle state or shutdown sequencing
- runtime-hosted explicitly owns the current process-oriented runtime lifecycle
  implementation
- the split-pending runtime family is narrowed to `arena_memory.cpp`

---

## D-0036

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted runtime signal handlers

### Context

The remaining `runtime.cpp` signal block owns host signal and exception
installation, saved signal dispositions, Windows unhandled-exception filtering,
signal-safe write/exit helpers, and volatile signal-shadow state. That code is
needed by the hosted runtime, but it is not the lifecycle state machine itself
and is not a freestanding-safe runtime-core implementation.

### Decision

Move hosted signal and exception handler installation/restoration into
`lib/core/runtime_signals_hosted.cpp`, classified as `runtime-hosted`.
Keep the public interrupt flag ABI in `runtime.cpp`, and update signal-shadow
state through private hosted helpers from `runtime_hosted_internal.h`.

### Consequences

- runtime-hosted explicitly owns the current POSIX/Windows signal and exception
  handler implementation
- `runtime.cpp` no longer owns saved signal dispositions, fatal-signal handlers,
  or signal-safe shadow variables directly
- a later freestanding runtime profile can provide target-specific trap,
  interrupt, or polling hooks behind the same public interrupt ABI

---

## D-0035

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted runtime shutdown hooks

### Context

Runtime shutdown hooks are registered through a public ABI but currently use
hosted C++ storage: `std::mutex`, `std::vector`, `std::string`, reverse-order
iteration, and hosted logging. Keeping that registry in `runtime.cpp` leaves
threading and STL ownership beside signal/lifecycle state even after the
operation tracker was split into hosted code.

### Decision

Move `eshkol_register_shutdown_hook`, `eshkol_unregister_shutdown_hook`, and
reverse-order shutdown-hook dispatch into
`lib/core/runtime_shutdown_hooks_hosted.cpp`, classified as `runtime-hosted`.
Keep `runtime.cpp` responsible for lifecycle sequencing and call a private
hosted helper to run registered hooks during shutdown.

### Consequences

- runtime-hosted explicitly owns the current mutex/vector-backed hook registry
- `runtime.cpp` no longer owns hook vectors, hook mutexes, or reverse-order hook
  dispatch details
- a later freestanding runtime profile can replace hook storage and dispatch
  behind the same public registration ABI

---

## D-0034

- Date: 2026-05-23
- Status: Accepted
- Title: Extract hosted runtime operation tracking

### Context

The runtime shutdown path tracks in-flight hosted operations so shutdown can
wait for active work to drain. The current implementation uses `std::mutex`,
`std::condition_variable`, `std::vector`, `std::string`, and steady-clock
durations. Those choices are appropriate for the hosted runtime, but they keep
threading and STL synchronization state in `runtime.cpp` while the remaining
file is being narrowed toward lifecycle and signal behavior.

### Decision

Move `eshkol_runtime_begin_operation`, `eshkol_runtime_end_operation`,
`eshkol_runtime_drain_operations`, and
`eshkol_runtime_get_operation_count` into
`lib/core/runtime_operations_hosted.cpp`, classified as `runtime-hosted`.
Preserve the public ABI and keep `eshkol_runtime_shutdown` calling those APIs
rather than reaching into operation storage directly.

### Consequences

- runtime-hosted explicitly owns the current condition-variable-backed
  operation-drain implementation
- `runtime.cpp` no longer owns operation vectors, operation mutexes, or
  condition-variable state
- a later freestanding runtime profile can replace the operation tracking
  implementation behind the same public symbols

---

## D-0011

- Date: 2026-04-15
- Status: Accepted
- Title: Machine integer surface starts as a type-system and ABI slice, not a full arithmetic semantics split

### Context

The freestanding roadmap needs `u8`/`u16`/`u32`/`u64`/`usize` and signed counterparts, but the current Eshkol numeric pipeline still assumes generic integer arithmetic normalizes through the existing `Int64` / `Float64` paths. Forcing width-specific arithmetic semantics into this slice would spread through the parser, checker, runtime representation, arithmetic optimizer, and polymorphic runtime at once.

### Decision

Add the machine integer family to the HoTT builtin type environment, builtin-name resolution, closure type metadata, and typed-value extraction paths now, while keeping generic arithmetic promotion normalized to `Int64` for the integer family. Treat this first slice as the low-level type and ABI surface required for annotations and future freestanding lowering.

### Consequences

- Eshkol can now parse and resolve `i8`/`i16`/`i32`/`i64`/`isize` and `u8`/`u16`/`u32`/`u64`/`usize` in type annotations
- closure metadata and typed tagged-value extraction preserve those builtin type IDs instead of collapsing them immediately to `Value`
- generic arithmetic still widens machine integer annotations through the existing `Int64` path until dedicated width-aware lowering is introduced
- the next low-level language slices can build pointers, volatility, and native ABI lowering on top of a stable machine integer vocabulary

---

## D-0012

- Date: 2026-04-15
- Status: Accepted
- Title: `runtime_exports.h` wrappers live in dedicated hosted runtime glue

### Context

`lib/core/platform_runtime.cpp` previously mixed host/process/toolchain services from `platform_runtime.h` with generated-code and JIT ABI wrappers exported through `runtime_exports.h`. That hid the boundary between platform runtime services and hosted wrapper glue needed by generated code.

### Decision

Keep `platform_runtime.cpp` focused on `platform_runtime.h`, and move the `runtime_exports.h` implementation into a dedicated hosted runtime unit: `lib/core/runtime_exports_hosted.cpp`.

### Consequences

- the hosted wrapper surface now has an explicit build boundary
- later `runtime-core` and `runtime-freestanding` work can reason about `platform_runtime` and `runtime_exports` independently
- generated-code ABI glue is not implicitly bundled into future freestanding runtime layers

---

## D-0013

- Date: 2026-04-15
- Status: Accepted
- Title: Raw pointer types start as an annotation surface before pointer operations

### Context

The platform roadmap needs `ptr<T>`-style vocabulary for MMIO, foreign calls, buffer views, and later freestanding ABI lowering. Landing pointer conversions, address-of, volatile memory access, and pointer arithmetic all at once would couple parser syntax, type checking, codegen, memory safety policy, and runtime representation in one large change.

### Decision

Add a raw pointer type constructor to the HoTT type surface first. The accepted spelling is `(ptr T)`, with `pointer` as a parser alias and `ptr` / `pointer` as builtin type aliases. The initial runtime representation is the existing pointer representation, and type resolution preserves the `Ptr` builtin family while later slices add conversions and address-producing operations.

### Consequences

- Eshkol can now parse and resolve pointer annotations such as `(ptr u8)`
- pointer type expressions copy and print round-trip as `(ptr T)`
- the pointer surface depends on the machine integer vocabulary already present in `master`
- pointer conversions, address-of, volatile load/store, and stricter ABI semantics remain separate follow-up slices

---

## D-0014

- Date: 2026-04-15
- Status: Accepted
- Title: Pointer conversion builtins use tracked `Ptr` bindings, while the procedure ABI stays tagged

### Context

The next low-level slice needs value-level pointer conversions, but the current compiler still routes ordinary procedure arguments, closure captures, and most generic runtime plumbing through tagged values. A full raw-pointer procedure ABI would cut across function lowering, closure dispatch, captures, REPL/module boundaries, and runtime tagging all at once.

### Decision

Add the first pointer conversion builtins now:

- `null-ptr`
- `ptr->usize`
- `usize->ptr`

Lower them through the typed codegen path and make tracked `Ptr` locals/globals recoverable as raw pointers during typed code generation. Do not broaden the general procedure ABI in this slice; generic call/capture paths remain tagged-first until a later low-level ABI pass exists.

### Consequences

- Eshkol now has the first value-level raw-pointer primitives required for MMIO and address manipulation
- typed codegen can recover raw pointers from tracked `Ptr` bindings instead of collapsing them into generic heap-object handling
- low-level pointer work can proceed without forcing an immediate closure/function ABI rewrite
- passing raw pointers through fully generic higher-order call paths remains a later slice, alongside volatility and explicit low-level calling-convention work

---

## D-0015

- Date: 2026-04-15
- Status: Accepted
- Title: `addr-of` is limited to storage-backed bindings in the initial low-level surface

### Context

After landing pointer conversions, the next missing primitive was direct address-taking. The current compiler still treats many values as by-value tagged procedure arguments rather than true addressable lvalues. A full `addr-of` that worked over arbitrary expressions, temporaries, fields, and by-value parameters would require a broader lvalue model and ABI rewrite.

### Decision

Add `addr-of` now, but limit it to bindings that already have concrete storage in the current compiler model:

- local `alloca`-backed bindings
- global variables
- external variables
- existing capture-storage pointers used by closure lowering

Reject non-variable expressions and non-addressable bindings in this slice.

### Consequences

- Eshkol gains a real address-taking primitive for low-level work and linker/global symbol integration
- the primitive is honest about current compiler boundaries instead of fabricating lvalue semantics the runtime does not yet support
- future work on volatility, MMIO, and linker-defined symbols can build on a stable address-of baseline
- a broader `addr-of` semantics for temporaries, fields, and by-value parameters remains deferred until the low-level ABI surface expands

---

## D-0016

- Date: 2026-04-16
- Status: Accepted
- Title: Fence builtins use explicit ordering designators and LLVM sync scopes

### Context

After address-taking and pointer conversions, the low-level surface needs barrier emission for startup, HAL, and MMIO-adjacent code. The backend can already express LLVM fence instructions, but the language lacked a bounded way to distinguish compiler-only ordering barriers from system-scope memory fences. A broader atomic/RMW or target-specific barrier surface would widen this slice beyond the immediate need.

### Decision

Add two builtins now:

- `compiler-fence`
- `memory-fence`

The first operand is an explicit fence-ordering designator, not a normal synthesized runtime expression. Supported orderings in this slice are `acquire`, `release`, `acq-rel`, and `seq-cst`.

`compiler-fence` lowers to an LLVM `fence` with `singlethread` sync scope. `memory-fence` lowers to a normal system-scope LLVM `fence`. Both builtins return `Null`.

Do not add atomic load/store/RMW builtins, target-specific barrier intrinsics, or inline assembly in this slice.

### Consequences

- Eshkol can express the first explicit compiler-only and system-scope barrier forms needed for low-level startup and HAL code
- ordering tokens are treated as source-level designators rather than undefined runtime variables
- the fence surface stays portable across the existing LLVM path without committing yet to a larger atomic or intrinsic language design
- richer atomics, architecture-specific barriers, and intrinsic escape hatches remain later slices

---

## D-0017

- Date: 2026-04-16
- Status: Accepted
- Title: Volatile operations are typed MMIO primitives over raw pointers

### Context

With pointer conversions, `addr-of`, and fence builtins in place, the platform surface still needed a direct way to express MMIO-style loads and stores. Treating this as a general memory model or atomic/RMW design would force a much larger language decision than the current HAL/bootstrap slice needs.

### Decision

Add two builtins now:

- `volatile-load`
- `volatile-store!`

The first operand is a low-level type designator, not a runtime expression. Supported designators in this slice are machine integer types and `ptr`. The address operand must be a `Ptr`; pointer loads return `Ptr`, integer loads return the requested machine integer type, and stores return `Null`.

Lower the builtins to LLVM `load volatile` and `store volatile` directly. Do not add atomic memory ordering, address spaces, target intrinsics, or inline assembly in this slice.

### Consequences

- Eshkol can now express typed volatile MMIO reads and writes through the LLVM backend
- type designators stay explicit and do not require undefined runtime variables
- the primitive stays narrow enough to compose with the existing pointer/fence surface
- richer atomics, address-space-aware pointers, and architecture-specific intrinsics remain later platform-language work

---

## D-0018

- Date: 2026-04-17
- Status: Accepted
- Title: Target intrinsics are explicit typed calls to LLVM intrinsic declarations

### Context

The low-level platform surface needs an escape hatch for compiler-recognized operations such as byte swaps, traps, frame addresses, and later target-specific hooks. Adding inline assembly or a broad foreign-call facility would couple this slice to freestanding runtime and target-linker work that is not ready to merge.

### Decision

Add `target-intrinsic` as a typed LLVM intrinsic form:

```scheme
(target-intrinsic return-type "llvm.intrinsic.name" arg-type arg ...)
```

The return type and every argument type are explicit low-level type designators. Supported designators in this slice are machine integer types, `ptr`, and `null` for void-returning intrinsics. The backend resolves the LLVM intrinsic by base name, checks the requested signature against LLVM's intrinsic type table, declares the intrinsic in the module, and lowers a direct typed call.

Do not add inline assembly, arbitrary external symbol calls, target feature dispatch, or freestanding linker integration in this slice.

### Consequences

- Eshkol can express selected LLVM intrinsics without inventing a new backend-specific surface for each operation
- intrinsic signatures are checked against LLVM instead of being trusted stringly-typed calls
- the feature composes with existing machine integer, pointer, fence, and volatile primitives
- architecture-specific policy, target feature gating, and external symbol/linker integration remain later platform work

---

## D-0019

- Date: 2026-05-20
- Status: Accepted
- Title: Declaration attributes are bounded tail modifiers for low-level symbols

### Context

The platform surface needs enough symbol control for startup code, linker-script sections, weak target hooks, and external ABI names. A generic attribute framework would pull in target-specific validation, packed layouts, interrupt handlers, naked functions, and broader ABI policy before the freestanding pipeline is ready.

### Decision

Add bounded declaration-tail modifiers now:

- `define`: `:link-section`, `:align`, `:used`, `:weak`, `:export-symbol`, and function-only `:no-return`
- `extern`: `:extern-symbol` / `:real`, `:weak`, and `:no-return`
- `extern-var`: `:extern-symbol` / `:real`

Modifiers are tail-only. `:align` accepts only positive power-of-two integers. `:export-symbol` may force public linkage with the source name or accept an explicit emitted symbol name. The LLVM backend lowers these to section names, alignment, weak/extern-weak linkage, `llvm.used`, exported symbol renaming, and `noreturn` metadata.

Do not add `packed`, `interrupt-handler`, `naked`, arbitrary attribute bags, inline assembly, or object/linker policy in this slice.

### Consequences

- freestanding and HAL-facing code can name exported/imported symbols without C shims
- linker-script sections and retention roots can be expressed directly in Eshkol source
- the feature remains narrow enough to verify through parser and LLVM IR tests
- target-specific ABI attributes and object-level layout validation remain later platform work

---

## D-0020

- Date: 2026-05-21
- Status: Accepted
- Title: Pointer arithmetic starts as byte-offset `Ptr` addition

### Context

The low-level platform surface needs address arithmetic for MMIO register windows, linker-defined memory ranges, and freestanding startup code. A C-style pointer arithmetic model would require stable pointee layout, scaled element semantics, address spaces, provenance policy, and broader raw-pointer ABI rules that are not ready to merge.

### Decision

Add `ptr-add` as the first pointer arithmetic primitive:

```scheme
(ptr-add base offset)
```

`base` must be a `Ptr`. `offset` must be an integer byte count. The result is a `Ptr`. The backend lowers the operation as byte-offset LLVM `getelementptr i8`, not as element-scaled pointer arithmetic.

Do not add pointer subtraction, comparisons, address-space-aware pointers, field offsets, scaled element semantics, or generic pointer arithmetic operators in this slice.

### Consequences

- MMIO and startup code can compute byte-offset addresses without round-tripping through ad hoc integer arithmetic at every use site
- the pointer model stays narrow and explicit while the low-level ABI continues to mature
- richer pointer semantics remain later platform-language work once layout and address-space policy are defined

---

## D-0021

- Date: 2026-05-21
- Status: Accepted
- Title: Atomic memory access starts as typed load/store over raw `Ptr`

### Context

After fences, volatile memory access, and byte-offset pointer arithmetic, the platform surface needs a minimal atomic memory access vocabulary for shared flags, bootstrap handoff state, and low-level runtime coordination. A full atomic/RMW model would require compare-exchange semantics, failure orderings, typed pointer provenance, and target policy that are not needed for the first usable slice.

### Decision

Add two typed atomic memory primitives:

```scheme
(atomic-load type ptr ordering)
(atomic-store! type ptr value ordering)
```

The `type` operand is a low-level type designator, not a runtime expression. Supported designators are machine integer types and `ptr`. The address operand must be a `Ptr`; pointer loads return `Ptr`, integer loads return the requested machine integer type, and stores return `Null`.

Load orderings are `relaxed`, `acquire`, and `seq-cst`. Store orderings are `relaxed`, `release`, and `seq-cst`. The backend lowers `relaxed` to LLVM `monotonic`, uses explicit ABI alignment, and emits LLVM atomic load/store instructions directly.

Do not add read-modify-write operations, compare-exchange, address spaces, volatile+atomic combined forms, inline assembly, or target-specific atomic policy in this slice.

### Consequences

- Eshkol can express the first typed atomic memory accesses needed for low-level platform code
- the operation set stays explicit about access size and memory ordering
- richer atomic operations and target memory-model policy remain later v1.8 platform-language work

---

## D-0022

- Date: 2026-05-21
- Status: Accepted
- Title: Execution profiles are selected through the compiler driver

### Context

The profile resolver existed as a toolchain model, but `eshkol-run` still
exposed only ad hoc hosted flags such as `--wasm`, `--compile-only`, and
`--no-stdlib`. That made profile behavior hard to test from the actual user
entrypoint and left freestanding object work dependent on implicit flag
combinations.

### Decision

Add driver-level profile and target selection:

```text
eshkol-run --profile NAME [--target TRIPLE] ...
```

`--profile` accepts the documented profile names from the execution profile
model. `--target` sets the LLVM target triple before code generation. The
driver resolves the selected profile once after option parsing and applies the
resulting compile-only, no-stdlib, WASM, and target settings through the same
resolver used by the unit tests.

Freestanding native profiles require `--target <triple>`, reject hosted-only
JIT/linking combinations, imply compile-only object output, and imply
`--no-stdlib`. `--wasm` remains supported as a compatibility alias for hosted
WASM output.

### Consequences

- the platform profile model is now reachable from the production compiler
  entrypoint
- invalid profile/target combinations fail before parsing and code generation
- the next freestanding object slice can depend on an explicit driver contract
  instead of layering more behavior onto legacy flags

---

## D-0023

- Date: 2026-05-21
- Status: Accepted
- Title: Atomic exchange is the first read-modify-write primitive

### Context

The initial atomic surface deliberately stopped at typed loads and stores.
Low-level runtime and bootstrap code still needs one operation that can publish
a new value and observe the previous value without introducing the full
compare-exchange contract.

### Decision

Add a typed exchange primitive:

```scheme
(atomic-exchange! type ptr value ordering)
```

The `type` operand uses the same low-level machine designators as
`atomic-load` and `atomic-store!`. The address operand must be a `Ptr`, the
value must match the requested machine type, and the result is the previous
value at that address. Supported orderings are `relaxed`, `acquire`,
`release`, `acq-rel`, and `seq-cst`; `relaxed` lowers to LLVM `monotonic`.

Do not add compare-exchange, fetch-add/sub families, weak/strong CAS policy, or
target-specific memory-model rules in this slice.

### Consequences

- Eshkol now has a minimal typed RMW operation for lock-free handoff patterns
- compare-exchange remains a later v1.8 slice with explicit failure-ordering
  design
- the atomic test surface now checks both load/store and RMW IR lowering

---

## D-0024

- Date: 2026-05-21
- Status: Accepted
- Title: Atomic fetch-add/sub are bounded integer RMW primitives

### Context

After `atomic-exchange!`, low-level runtime and bootstrap code still needs
simple counters and reference-style handoff state. Compare-exchange remains a
larger semantic commitment because it needs success/failure orderings, weak vs.
strong policy, and a value/result shape for the observed load.

### Decision

Add two typed arithmetic read-modify-write primitives:

```scheme
(atomic-fetch-add! type ptr value ordering)
(atomic-fetch-sub! type ptr value ordering)
```

The `type` operand uses the existing low-level machine integer designators.
The address operand must be a `Ptr`, the value must be integer-compatible with
the requested type, and the result is the previous value at that address.
Supported orderings match `atomic-exchange!`: `relaxed`, `acquire`, `release`,
`acq-rel`, and `seq-cst`; `relaxed` lowers to LLVM `monotonic`.

Do not support pointer designators for these arithmetic RMW forms. Do not add
compare-exchange, fetch-and/or/xor families, weak/strong CAS policy, or
target-specific memory-model rules in this slice.

### Consequences

- Eshkol can express atomic integer counters without resorting to CAS loops in
  source
- pointer arithmetic remains explicit through `ptr-add` instead of being hidden
  inside arithmetic atomic operations
- compare-exchange remains a later v1.8 slice with a separately documented
  failure-ordering contract

---

## D-0025

- Date: 2026-05-21
- Status: Accepted
- Title: Atomic bitwise fetch operations are integer-only RMW primitives

### Context

Arithmetic fetch-add/sub supports counters, but low-level platform code also
needs flag-word updates for device state, scheduler masks, and bootstrap
coordination. Compare-exchange still has a larger contract because it needs a
structured observed-value/success result and success/failure ordering rules.

### Decision

Add three typed bitwise read-modify-write primitives:

```scheme
(atomic-fetch-and! type ptr value ordering)
(atomic-fetch-or! type ptr value ordering)
(atomic-fetch-xor! type ptr value ordering)
```

The `type` operand uses the existing low-level machine integer designators.
The address operand must be a `Ptr`, the value must be integer-compatible with
the requested type, and the result is the previous value at that address.
Supported orderings match the other RMW forms: `relaxed`, `acquire`,
`release`, `acq-rel`, and `seq-cst`; `relaxed` lowers to LLVM `monotonic`.

Do not support pointer designators for these bitwise RMW forms. Do not add
compare-exchange, fetch-nand, weak/strong CAS policy, or target-specific
memory-model rules in this slice.

### Consequences

- Eshkol can express atomic flag-word updates without CAS loops in source
- the RMW family now covers exchange, arithmetic counters, and bitwise masks
- compare-exchange remains a later v1.8 slice with a separately documented
  result and failure-ordering contract

---

## D-0026

- Date: 2026-05-22
- Status: Accepted
- Title: Atomic compare-exchange returns the observed value

### Context

The atomic RMW family now covers exchange, arithmetic counters, and bitwise
flag updates. Freestanding runtime code still needs a bounded CAS primitive for
lock-free state transitions, but introducing a new structured return type would
expand the low-level surface more than this slice requires.

### Decision

Add a strong typed compare-exchange primitive:

```scheme
(atomic-compare-exchange! type ptr expected desired success-order failure-order)
```

The `type` operand uses the existing low-level machine designators, including
`ptr`. The address operand must be a `Ptr`; `expected` and `desired` must match
the requested machine type. The result is the observed value read from memory:
on success it equals `expected`, and on failure it is the current value at the
address. Source code can test success by comparing the returned observed value
with `expected`.

Success orderings match the other RMW forms: `relaxed`, `acquire`, `release`,
`acq-rel`, and `seq-cst`. Failure orderings are limited to `relaxed`,
`acquire`, and `seq-cst`, and must not be stronger than the success ordering.
The backend lowers `relaxed` to LLVM `monotonic` and emits LLVM `cmpxchg`
directly with explicit ABI alignment.

Do not add weak compare-exchange, retry-loop helpers, fetch-nand, address-space
policy, or a structured CAS result type in this slice.

### Consequences

- Eshkol can now express lock-free state transitions without relying on opaque
  target intrinsics
- CAS success remains explicit in source through ordinary value comparison
- a future structured result type can be added later without changing the
  observed-value primitive

---

## D-0027

- Date: 2026-05-23
- Status: Accepted
- Title: Classify current logger and resource-limit implementations as hosted runtime

### Context

The runtime source sets already separate direct core runtime files from direct
hosted runtime files, but `runtime-split-pending` still mixed several different
kinds of work. `logger.cpp` and `resource_limits.cpp` are current hosted
implementations: they depend on process stderr/files, platform backtraces,
environment variables, timers, logging sinks, and hosted runtime interrupts.
`printer.cpp` is not runtime substrate at all; it is an AST pretty-printer and
debugging surface for compiler data structures.

### Decision

Move the current implementations of `lib/core/logger.cpp` and
`lib/core/resource_limits.cpp` from `runtime-split-pending` to
`runtime-hosted`. Remove `lib/core/printer.cpp` from the runtime family source
sets entirely so it remains part of the aggregate library as out-of-runtime
tooling code.

Keep `arena_memory.cpp` and `runtime.cpp` in `runtime-split-pending` until
their host-dependent seams are split. Do not introduce a freestanding logger
hook ABI or resource-limit policy object in this slice.

### Consequences

- the remaining split-pending runtime set is smaller and more honest
- hosted logger/resource-limit behavior remains unchanged because
  `eshkol-static` still aggregates all internal source sets
- future slices can focus on the real hard seams: arena/string-port/threading
  in `arena_memory.cpp` and signal/process/shutdown behavior in `runtime.cpp`

---

## D-0045

- Date: 2026-05-23
- Status: Accepted
- Title: Extract continuation and dynamic-wind runtime helpers

### Context

`arena_memory.cpp` still carried the runtime helpers for first-class
continuations and `dynamic-wind`: continuation state allocation, continuation
closure construction, the global wind stack, and thunk dispatch used while
unwinding non-local exits. These helpers use arena allocation, closure metadata,
and generated-code ABI conventions, but they do not depend on hosted files,
process state, environment variables, or thread/process APIs.

Keeping this block in the split-pending arena file made the remaining
allocator/data-structure split less clear and hid a freestanding-safe runtime
surface inside a file that still contains unrelated high-level helpers.

### Decision

Move the continuation and dynamic-wind helper ABI into
`lib/core/runtime_continuations.cpp`, classified as `runtime-core`. Preserve the
existing exported symbol names and the `g_dynamic_wind_stack` global so
generated code, REPL JIT symbol registration, and hosted behavior remain
unchanged.

Do not change the single-shot continuation semantics, dynamic-wind thunk
dispatch ABI, or exception handler runtime in this slice.

### Consequences

- `arena_memory.cpp` no longer owns continuation state or dynamic-wind runtime
  machinery
- runtime-core explicitly owns the current continuation helper implementation
- the remaining split-pending arena file is narrowed further toward allocator,
  AD, region/shared-memory, hash-table, tensor-allocation, and math helper
  groups

---

## D-0046

- Date: 2026-05-24
- Status: Accepted
- Title: Extract automatic differentiation runtime helpers

### Context

`arena_memory.cpp` still carried the generated-code ABI for dual numbers,
AD nodes, AD tapes, and nested-gradient thread-local state. These helpers use
arena allocation and the tagged object layout, but they do not depend on
filesystem, process, environment, socket, or platform thread APIs.

Keeping them in the split-pending arena file hid a freestanding-safe runtime
surface inside the remaining allocator/data-structure implementation.

### Decision

Move the AD helper ABI into `lib/core/runtime_autodiff.cpp`, classified as
`runtime-core`. Preserve the existing exported symbol names and TLS globals so
LLVM codegen, REPL JIT registration, and generated executables keep the same
ABI.

### Consequences

- `arena_memory.cpp` no longer owns AD tape/node allocation or nested-gradient
  TLS state
- runtime-core explicitly owns the current AD runtime helper implementation
- the remaining split-pending arena file is narrowed further toward allocator,
  region/shared-memory, hash-table, tensor-allocation, and math helper groups

---

## D-0047

- Date: 2026-05-24
- Status: Accepted
- Title: Extract hash-table runtime helpers

### Context

`arena_memory.cpp` still carried the generated-code ABI for hash-table
allocation, tagged-key hashing/equality, open-addressing mutation/lookup, and
key/value list materialization. These helpers use arena allocation, tagged
object layout, bignum comparison, and the existing abstract hash-table lock
hooks, but they do not depend directly on filesystem, process, environment,
socket, or host thread APIs.

Keeping them in the split-pending arena file made the remaining arena split
less precise and hid another freestanding-safe data-structure surface behind
unrelated allocator and tensor-allocation code.

### Decision

Move the hash-table helper ABI into `lib/core/runtime_hash_table.cpp`,
classified as `runtime-core`. Preserve the exported symbol names used by LLVM
codegen, REPL JIT registration, generated executables, and existing hash-table
tests.

Keep the current lock/unlock calls behind the `eshkol_hash_table_lock` /
`eshkol_hash_table_unlock` ABI. The hosted implementation still lives in
`runtime_arena_sync_hosted.cpp`; a later freestanding profile can provide a
target-specific critical-section implementation behind the same symbols.

### Consequences

- `arena_memory.cpp` no longer owns hash-table hashing, equality, allocation,
  lookup, mutation, or key/value list materialization
- runtime-core explicitly owns the current hash-table runtime helper
  implementation
- the remaining split-pending arena file is narrowed further toward allocator,
  region/shared-memory, tensor-allocation, and math helper groups

---

## D-0048

- Date: 2026-05-24
- Status: Accepted
- Title: Extract tensor allocation runtime helpers

### Context

`arena_memory.cpp` still carried the exported tensor allocation ABI:
`arena_allocate_tensor_with_header` and `arena_allocate_tensor_full`. These
helpers allocate header-backed tensor objects plus dimensions/elements arrays
through the arena allocator, initialize tensor metadata, and zero element
storage for generated code, FFI, model IO, BLAS/XLA stubs, and REPL JIT
programs.

The helpers use arena allocation, tagged object layout, and raw memory
initialization. They do not depend on filesystem, process, environment, socket,
or host thread APIs.

### Decision

Move the tensor allocation helper ABI into
`lib/core/runtime_tensor_alloc.cpp`, classified as `runtime-core`. Preserve the
existing exported symbol names so codegen, REPL JIT registration, generated
executables, and hosted tensor consumers keep the same link contract.

### Consequences

- `arena_memory.cpp` no longer owns tensor object or tensor payload allocation
- runtime-core explicitly owns tensor allocation next to tensor indexing and
  tensor fill helpers
- the remaining split-pending arena file is narrowed further toward allocator,
  region/shared-memory, list/error helper, and math helper groups

---

## D-0049

- Date: 2026-05-24
- Status: Accepted
- Title: Extract tensor math runtime helpers

### Context

`arena_memory.cpp` still carried generated-code ABI helpers for tensor linear
algebra, broadcast, shape conversion, concatenation, and batched matrix
multiplication. These helpers implement symbols such as `eshkol_lu_decompose`,
`eshkol_tensor_svd`, `eshkol_broadcast_elementwise_f64`,
`eshkol_cons_list_to_dims`, `eshkol_tensor_to_dims`, `eshkol_concat_strided`,
and `eshkol_batch_matmul_f64`.

The helpers operate on raw tensor dimensions/elements, arena tagged cons-cell
accessors, and C math/memory primitives. They do not depend on filesystem,
process, environment, socket, or host thread APIs.

### Decision

Move the tensor math helper ABI into `lib/core/runtime_tensor_math.cpp`,
classified as `runtime-core`. Preserve the exported symbol names used by LLVM
codegen, REPL JIT registration, generated executables, and tensor examples.

### Consequences

- `arena_memory.cpp` no longer owns tensor linalg, broadcast, shape-conversion,
  concat, or batched-matmul helper implementations
- runtime-core explicitly owns the tensor math surface next to tensor
  allocation, indexing, and fill helpers
- the remaining split-pending arena file is narrowed further toward allocator,
  region/shared-memory, and list/error helper groups

---

## D-0050

- Date: 2026-05-24
- Status: Accepted
- Title: Extract list helper runtime ABI

### Context

`arena_memory.cpp` still carried generated-code ABI helpers for tagged-list
reverse, quasiquote append/splice, recursion-depth accounting, and list/vector
error guards. These helpers operate on tagged cons cells and arena allocation,
but they are not allocator substrate. Keeping them in the split-pending arena
file obscured the remaining platform work and kept a dynamic-array buffering
path in quasiquote append.

### Decision

Move the list helper ABI into `lib/core/runtime_list_helpers.cpp`, classified
as `runtime-core`. Preserve the exported symbol names used by LLVM codegen and
REPL JIT registration. Replace the quasiquote append helper's temporary dynamic
buffer with a single forward arena walk that copies cons cells in order and
attaches the right-hand tail.

Keep list/vector guard errors behind `eshkol_runtime_fatal` for now, matching
the existing runtime-core pattern where generated-code helpers delegate to the
hosted fatal sink until a freestanding panic hook ABI exists.

### Consequences

- `arena_memory.cpp` no longer owns list reverse, quasiquote append/splice,
  recursion-depth, or list/vector guard helper implementations
- runtime-core explicitly owns the current generated-code list helper ABI
- quasiquote append no longer allocates a temporary host-side dynamic buffer
- the remaining split-pending arena file is narrowed further toward allocator
  and region/shared-memory groups

---

## D-0051

- Date: 2026-05-24
- Status: Accepted
- Title: Extract shared memory runtime helpers

### Context

`arena_memory.cpp` still carried the shared allocation and weak-reference ABI:
`shared_allocate`, `shared_allocate_typed`, `shared_retain`,
`shared_release`, `shared_ref_count`, `shared_get_header`, and the
`weak_ref_*` helpers registered with the REPL JIT. These helpers manage
reference-count metadata around process-independent allocations; they are not
arena block/scope mechanics or region stack behavior.

Leaving them in the split-pending arena file made the remaining allocator work
less precise and left this ABI without a focused regression test.

### Decision

Move the shared/weak-reference helper implementation into
`lib/core/runtime_shared_memory.cpp`, classified as `runtime-core`. Preserve
the exported symbol names used by REPL JIT registration and ownership paths.
Add `runtime_shared_memory_test` to exercise typed allocation, ref-count
retain/release, weak upgrade, final destructor dispatch, and dead weak-ref
behavior.

### Consequences

- `arena_memory.cpp` no longer owns shared allocation or weak-reference helper
  implementations
- runtime-core explicitly owns the current shared/weak-reference ABI
- shared memory behavior has a focused CTest regression
- the remaining split-pending arena file is narrowed further toward allocator
  and region stack/escape groups

---

## D-0052

- Date: 2026-05-24
- Status: Accepted
- Title: Extract closure allocation runtime helpers

### Context

`arena_memory.cpp` still carried the closure allocation ABI used by generated
closure code and continuation setup: `arena_allocate_closure_env`,
`arena_allocate_closure`, and `arena_allocate_closure_with_header`. These
helpers depend on arena allocation and closure metadata packing, but they are
not arena block/scope mechanics.

Keeping them in the split-pending arena file made it harder to tell raw arena
substrate from higher-level callable object construction.

### Decision

Move the closure allocation helpers into `lib/core/runtime_closure_alloc.cpp`,
classified as `runtime-core`. Preserve the exported ABI and the legacy/headered
allocation behavior, including variadic and named closure metadata, HoTT return
type bits, lambda-sexpr subtype selection for zero-capture header allocations,
and null-initialized capture slots.

Add `runtime_closure_alloc_test` to cover closure environment initialization,
packed metadata preservation, legacy closure allocation, zero-capture
header-backed lambda allocation, and capturing header-backed closure
allocation.

### Consequences

- `arena_memory.cpp` no longer owns closure object/environment construction
- runtime-core explicitly owns closure allocation above the raw arena substrate
- closure allocation behavior has a focused CTest regression
- the remaining split-pending arena file is narrowed toward raw allocator,
  region, thread-local arena, deep-equality, and C++ wrapper groups

---

## D-0053

- Date: 2026-05-24
- Status: Accepted
- Title: Extract region and thread-local arena runtime helpers

### Context

`arena_memory.cpp` still carried the OALR region stack, global arena
selection, per-worker thread-local arena lifecycle, worker TLS reset logic,
arena merge ownership transfer, and region escape helpers. These routines use
the raw arena allocator, but they are not arena block/scope mechanics.

This grouping also contained two cleanup hazards: destroying an active region
could continue after `region_pop()` had already destroyed the region arena, and
region destruction logged the arena-owned name after freeing the region arena.

### Decision

Move the region/thread-local arena helper implementation into
`lib/core/runtime_regions.cpp`, classified as `runtime-core`. Preserve the
exported ABI used by generated `with-region` code and REPL JIT registration:
`get_global_arena`, `get_global_arena_shared`, worker arena lifecycle,
`arena_merge_to_parent`, `arena_is_worker_thread`, `region_*`, and
`region_escape_*`.

Keep raw arena creation/allocation/scope functions in `arena_memory.cpp`.
Tighten region cleanup so active-region destruction returns after popping, and
so region destruction logs before freeing arena-owned region names.

Add `runtime_regions_test` to cover worker TLS arena override/shutdown, active
region destruction, nested-region escape into a parent arena, header-backed
string/tagged-value escape, cons-cell escape, and arena merge ownership
transfer.

### Consequences

- `arena_memory.cpp` no longer owns region stack/lifecycle/escape behavior or
  per-worker thread-local arena setup
- runtime-core explicitly owns the OALR/generated-code region ABI
- active region destruction no longer risks double-destroy or use-after-free
  logging of region names
- the remaining split-pending arena file is narrowed toward raw allocator,
  tagged object allocation/accessors, deep equality, and C++ wrapper groups
