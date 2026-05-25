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

---

## D-0054

- Date: 2026-05-24
- Status: Accepted
- Title: Extract deep structural equality runtime helper

### Context

`arena_memory.cpp` still carried `eshkol_deep_equal`, the recursive structural
comparison helper used by generated `equal?` lowering. The helper depends on
tagged object layout, cons accessors, bignum comparison, vector layout, and
tensor metadata, but it is not arena block/scope mechanics.

Keeping this comparison logic in the split-pending arena file obscured the
remaining raw allocator substrate and made equality regressions share the arena
test surface.

### Decision

Move `eshkol_deep_equal` into `lib/core/runtime_deep_equal.cpp`, classified as
`runtime-core`. Preserve the exported ABI and comparison behavior for nulls,
immediate numbers/booleans, legacy and header-backed strings, header-backed
symbols, recursive cons cells, vectors, bignums, tensors, and callable pointer
identity.

Add `runtime_deep_equal_test` to cover pointer-null handling, numeric
cross-type equality, legacy/header string equality, symbol equality, nested cons
comparison, vector recursion, and tensor value comparison.

### Consequences

- `arena_memory.cpp` no longer owns generated `equal?` structural comparison
- runtime-core explicitly owns the deep-equality ABI used by generated code
- equality behavior now has a focused CTest regression independent of arena
  block/scope tests
- the remaining split-pending arena file is narrowed toward raw allocator,
  tagged object allocation/accessors, and C++ wrapper groups

---

## D-0055

- Date: 2026-05-24
- Status: Accepted
- Title: Extract header-aware object allocation helpers

### Context

`arena_memory.cpp` still carried the header-aware allocation ABI for tagged heap
objects: `arena_allocate_with_header`, `arena_allocate_with_header_zeroed`,
`arena_allocate_multi_value`, `arena_allocate_cons_with_header`,
`arena_allocate_string_with_header`, `arena_allocate_vector_with_header`, and
`arena_allocate_symbol_with_header`. These helpers use the raw arena allocator
and object header layout, but they are object-construction wrappers rather than
raw arena block/scope mechanics.

Keeping them in the split-pending arena file made the remaining allocator work
less clear and forced object-header regressions through broad arena or language
tests.

### Decision

Move the header-aware object allocation helpers into
`lib/core/runtime_object_alloc.cpp`, classified as `runtime-core`. Preserve the
exported ABI, object-header layout, payload initialization, and existing size
guards for generic, zeroed, multi-value, cons, string, vector, and symbol
allocation.

Add `runtime_object_alloc_test` to cover null/zero-size rejection, header
metadata, zeroed payloads, multi-value count storage, initialized cons cells,
NUL-terminated string payloads, vector payload sizing, and symbol payload
sizing.

### Consequences

- `arena_memory.cpp` no longer owns header-aware tagged-object construction
- runtime-core explicitly owns object-header allocation wrappers above the raw
  arena substrate
- object-header allocation behavior has a focused CTest regression independent
  of broad arena/language tests
- the remaining split-pending arena file is narrowed toward raw allocator,
  tagged object accessors, and C++ wrapper groups

---

## D-0056

- Date: 2026-05-24
- Status: Accepted
- Title: Extract tagged cons allocation and accessor helpers

### Context

`arena_memory.cpp` still owned the exported tagged cons ABI used by generated
list code, tensor index helpers, hash-table materialization, REPL/JIT symbol
registration, and FFI paths. This group includes raw tagged cons allocation,
batch allocation, convenience constructors, typed getters/setters, type/flag
queries, and full tagged-value copy helpers.

These helpers are freestanding-safe and depend only on raw arena allocation and
the tagged-value layout. Keeping them in the split-pending arena file obscured
the remaining raw allocator/C++ wrapper work.

### Decision

Move tagged cons allocation and accessor helpers into
`lib/core/runtime_tagged_cons.cpp`, classified as `runtime-core`. Preserve the
exported symbol names and exact tagged-value initialization/copy behavior.

Add `runtime_tagged_cons_test` to cover allocation initialization, typed
set/get helpers, pointer/null behavior, flags, whole tagged-value copies, batch
allocation initialization, and convenience constructors.

### Consequences

- generated-code and REPL/JIT tagged cons symbols now live in an explicit
  runtime-core unit
- `arena_memory.cpp` no longer owns tagged cons accessor behavior
- the remaining split-pending arena file is narrowed toward raw arena
  block/scope/statistics helpers and the C++ `Arena` wrapper

---

## D-0057

- Date: 2026-05-24
- Status: Accepted
- Title: Extract raw arena core and C++ wrapper seam

### Context

After the object-allocation and tagged-cons extractions, `arena_memory.cpp`
contained the raw arena block/scope/statistics substrate, legacy list-node
allocation, the weak generated-code argument globals, the REPL shared-arena
global, and the C++ `Arena` RAII wrapper.

The raw allocator is freestanding-critical, but its diagnostic poison path still
read `ESHKOL_ARENA_POISON` directly from the process environment. The C++ RAII
wrapper is not the generated-code runtime ABI and has different policy questions
because it exposes C++ exception behavior.

### Decision

Move raw arena creation/destruction, aligned allocation, zeroed allocation,
scope push/pop/reset, statistics, weak generated-code argument globals, the REPL
shared-arena global, and legacy list-node allocation into
`lib/core/runtime_arena_core.cpp`, classified as `runtime-core`.

Move the C++ `Arena` RAII wrapper into `lib/core/runtime_arena_cpp.cpp`, kept as
the remaining `runtime-split-pending` source until its final family is decided.

Move hosted arena poison policy into
`lib/core/runtime_arena_diagnostics_hosted.cpp`. Runtime-core now calls an
abstract `eshkol_arena_poison_enabled` hook instead of reading process
environment variables directly.

While extracting the allocator, fix over-aligned allocation so
`arena_allocate_aligned` aligns the returned absolute pointer address, not just
the offset within the current block. Also reject non-power-of-two alignments and
overflowing aligned/list-node allocation sizes.

Add `runtime_arena_core_test` and `runtime_arena_cpp_test` to cover raw arena
statistics, block/scope behavior, over-aligned pointers, invalid alignment,
overflow rejection, legacy list allocation, RAII scope restoration, typed array
allocation, move semantics, and wrapper reset behavior.

### Consequences

- the old `lib/core/arena_memory.cpp` implementation file is removed
- runtime-core owns the raw allocator substrate needed by freestanding profiles
- process environment access for arena poison diagnostics is isolated in a
  hosted policy hook
- the remaining split-pending runtime surface is a single explicit C++ wrapper
  file rather than the arena implementation monolith
- allocator alignment now satisfies over-aligned C++ wrapper allocations instead
  of relying on the base address returned by `malloc`

---

## D-0058

- Date: 2026-05-24
- Status: Accepted
- Title: Retire runtime-split-pending source set

### Context

After extracting raw arena mechanics and hosted arena diagnostics, the only
remaining `runtime-split-pending` source was `runtime_arena_cpp.cpp`, the C++
`Arena` RAII wrapper around the C arena ABI.

That wrapper is useful for C++ consumers and tests, but it is not part of the
generated-code runtime ABI. It also exposes C++ exception and convenience-wrapper
semantics that should not be treated as freestanding runtime-core substrate.

### Decision

Retire `ESHKOL_RUNTIME_SPLIT_PENDING_SRC` and the
`eshkol-runtime-split-pending-obj` object library. Keep `runtime_arena_cpp.cpp`
compiled through the aggregate `eshkol-static` archive via the non-runtime
source set, and explicitly keep it out of both `runtime-core` and
`runtime-hosted`.

Update the runtime boundary test so it fails if a split-pending source set or
object target is reintroduced, and so it verifies that the C++ `Arena` adapter
stays outside runtime source families.

### Consequences

- the runtime implementation is now classified into explicit `runtime-core` and
  `runtime-hosted` source sets, with no split-pending bucket
- the C++ `Arena` wrapper remains available to existing consumers without
  widening freestanding runtime-core policy
- future runtime decomposition work can move to hosted-leakage enforcement,
  freestanding hook definitions, VM runtime decomposition, and target ABI work

---

## D-0059

- Date: 2026-05-24
- Status: Accepted
- Title: Classify the bytecode VM unity hub into explicit source families

### Context

`lib/backend/eshkol_vm.c` is still compiled as a unity-build hub because the VM
submodules share internal static types, include-order assumptions, and the
compiler/parser/native dispatch table. That arrangement kept desktop VM behavior
working, but it also hid the boundary needed for freestanding and embedded
profiles.

The Tamatsotchke scripting use case needs a much smaller VM profile: static
memory limits, no desktop native table, fixed host calls, flash/content-pack
loading, and budget checks. That cannot be built safely while `eshkol_vm.c`
remains an unclassified blob inside the aggregate archive.

### Decision

Keep `eshkol_vm.c` as the temporary compile-order hub, but compile it through a
dedicated `eshkol-vm-unity-obj` object target instead of appending it directly to
`LIB_SRC`.

Add explicit CMake component families for the files included by the hub:

- `ESHKOL_VM_CORE_COMPONENT_SRC`
- `ESHKOL_VM_HOSTED_COMPONENT_SRC`
- `ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC`
- `ESHKOL_VM_TEST_COMPONENT_SRC`

Add `vm_source_boundary_test` so the build fails if a VM component is included
by the unity hub without being classified, if component families overlap, if the
hub is silently appended back into `LIB_SRC`, or if VM core files grow direct
hosted-only dependencies such as files, processes, sockets, dynamic loading, or
host threads.

### Consequences

- the shipped aggregate archive and desktop VM behavior remain unchanged
- VM decomposition now has a checked build-graph boundary before physical file
  extraction begins
- embedded VM work can start from concrete families: keep core, replace hosted
  natives with a static host-call table, and constrain toolchain/loader behavior
  for firmware scripts
- `vm_native.c` remains classified as hosted until its broad desktop native
  table is split by capability

---

## D-0060

- Date: 2026-05-25
- Status: Accepted
- Title: Add deterministic VM host-native table installation

### Context

The public bytecode VM C ABI already allowed embedders to register host-native
callbacks dynamically by name. That is sufficient for desktop tools and tests,
but it is not deterministic enough for firmware/product runtimes such as
Tamatsotchke, where bytecode needs stable native-call fids and a fixed host-call
surface.

The embedded profile work also needs a migration path that does not require the
full desktop native table to be physically split before any product-runtime
experiments can begin.

### Decision

Extend `inc/eshkol/backend/vm.h` with `EshkolVmHostNative` and a deterministic
host-native table API:

- `eshkol_vm_install_host_natives`
- `eshkol_vm_clear_host_natives`
- `eshkol_vm_host_native_capacity`
- `eshkol_vm_host_native_count`

Installed entries map directly to slots by array index, so bytecode calls use
`ESHKOL_VM_HOST_NATIVE_BASE + index`. Table installation validates the whole
input before mutating the current registry, rejecting null callbacks, invalid
names, duplicate names, and over-capacity tables without partially changing
dispatch state.

Keep `eshkol_vm_register_host_native` and `eshkol_vm_unregister_host_native` for
desktop embedders and tests. Dynamic registrations append after the installed
fixed table, while duplicate names remain rejected across both paths.

### Consequences

- product runtimes can define fixed host-call slots before ESKB export-table and
  embedded loader work lands
- the public VM C API tests now execute bytecode against fixed host-native slots
  and verify all-or-nothing table validation
- this does not make `vm_native.c` freestanding; the broad desktop native table
  still needs capability partitioning before a small VM target can omit hosted
  subsystems

---

## D-0061

- Date: 2026-05-25
- Status: Accepted
- Title: Preserve ESKB function tables for named VM entry points

### Context

The public VM C ABI could load an in-memory ESKB chunk and run its first
function, but product runtimes need stable script entry points such as `init`,
`tick`, `input`, and `render`. The ESKB code section already carries function
names and per-function code bodies, but the reader validated and then discarded
every function after the first one.

Tamatsotchke-style firmware can use fixed host-call slots only if it can also
dispatch a known script function repeatedly from the host loop.

### Decision

Keep the existing ESKB section format, but preserve the decoded function table
inside `EskbModule`. Concatenate function bodies into the loaded VM instruction
array and store each function's code offset/length and name.

Extend `inc/eshkol/backend/vm.h` with:

- `eshkol_vm_has_function`
- `eshkol_vm_call`

`eshkol_vm_run` now dispatches the first function after resetting instruction,
stack, frame, handler, wind, halt, error, and autodiff-tracking state.
`eshkol_vm_call` applies the same execution reset and starts at the named
function's decoded code offset while preserving heap and host resources owned by
the VM handle.

### Consequences

- embedded/product hosts can drive stable named script hooks without relinking
  the full desktop VM or inventing a side-channel entry table
- the public VM C API tests now verify function lookup, missing-entry rejection,
  named helper execution, and repeat named dispatch
- a future product profile should still add compiler-side manifest checks so
  required entries are declared explicitly and unsupported entries fail before
  bytecode deployment

---

## D-0062

- Date: 2026-05-25
- Status: Accepted
- Title: Add a host-native-only VM dispatch policy

### Context

The deterministic host-native table gives product runtimes stable native-call
slots, but it does not by itself prevent bytecode from calling the broad desktop
native table in `vm_native.c`. That table still owns files, processes, sockets,
dynamic loading, terminal behavior, and other hosted subsystems that firmware
profiles must not expose.

Physically splitting `vm_native.c` by capability remains necessary, but
Tamatsotchke-style runtime work needs an immediate guardrail that can run
bytecode with fixed host calls while rejecting desktop native fids.

### Decision

Keep desktop behavior as the default native-call policy for loaded VM handles.
Add a per-handle policy API to `inc/eshkol/backend/vm.h`:

- `ESHKOL_VM_NATIVE_POLICY_DESKTOP`
- `ESHKOL_VM_NATIVE_POLICY_HOST_ONLY`
- `eshkol_vm_set_native_policy`
- `eshkol_vm_get_native_policy`

When a VM handle is set to `ESHKOL_VM_NATIVE_POLICY_HOST_ONLY`,
`vm_dispatch_native` rejects any fid below `ESHKOL_VM_HOST_NATIVE_BASE` and
continues to allow only deterministic host-native table slots.

### Consequences

- embedded/product hosts can load normal ESKB chunks and run them with a checked
  host-call-only native surface
- desktop VM behavior and existing native-call tests remain unchanged by default
- the public VM C API tests now prove that fixed host calls still execute under
  host-only policy and desktop native fids fail under that policy
- this is a runtime guardrail, not a replacement for compiler-side embedded
  target checks or the eventual physical split of `vm_native.c`

---

## D-0063

- Date: 2026-05-25
- Status: Accepted
- Title: Materialize ESKB string constants in the public VM loader

### Context

`eskb_reader.c` decoded `ESKB_CONST_STRING` entries into owned strings and
recorded their lengths, but `eshkol_vm_load_chunk` did not turn those constants
into VM string values. The default constant conversion path silently converted
unknown constant types into integers, so public VM embedders saw string
constants as their lengths instead of as `VAL_STRING` heap objects.

Tamatsotchke-style firmware may still avoid dynamic script strings on device and
route user-facing text through a read-only content pack, but the desktop/public
VM loader must faithfully execute ESKB chunks that contain strings.

### Decision

Add a shared ESKB constant materialization helper for VM loading. The helper
maps:

- `ESKB_CONST_NIL` to `NIL_VAL`
- `ESKB_CONST_INT64` to `INT_VAL`
- `ESKB_CONST_F64` to `FLOAT_VAL`
- `ESKB_CONST_BOOL` to `BOOL_VAL`
- `ESKB_CONST_STRING` to a VM string allocated in the VM heap/region state

Use the helper from both `eshkol_vm_load_chunk` and the standalone `.eskb`
execution path. If a string constant cannot be materialized, loading fails
instead of producing an integer placeholder.

### Consequences

- public VM embedders can execute bytecode that uses ESKB string constants
- the VM C API test now exercises a string constant through `OP_STR_LEN`
- embedded product profiles still need compiler/profile policy for whether
  dynamic script strings are allowed or replaced with content-pack IDs

---

## D-0064

- Date: 2026-05-25
- Status: Accepted
- Title: Add load-time VM policy options for embedded profiles

### Context

The public VM loader now materializes ESKB string constants correctly, and VM
handles can be switched to host-native-only dispatch after loading. That is
right for desktop compatibility, but embedded/product hosts need policy to be
visible before bytecode runs.

Tamatsotchke-style firmware should be able to load scripts directly into a
fixed host-call surface and reject dynamic script strings so user-facing text
can come from a read-only content pack. Requiring callers to load first and then
apply those policies leaves a window where the runtime state does not describe
the intended product profile.

### Decision

Add `EshkolVmLoadOptions` to the public VM C ABI with:

- `native_policy`
- `reject_string_constants`

Add `eshkol_vm_default_load_options` and `eshkol_vm_load_chunk_with_options`.
The existing `eshkol_vm_load_chunk` remains the desktop-compatible default:
desktop native policy and ESKB string constants enabled.

`eshkol_vm_load_chunk_with_options` validates the requested native policy before
decoding completes, applies it to the loaded VM handle, and can reject
`ESKB_CONST_STRING` constants during constant materialization.

### Consequences

- embedded/product hosts can load bytecode directly into host-native-only mode
- product profiles can reject dynamic ESKB strings at load time while desktop
  embedders continue to receive VM string objects
- the public VM C API tests now cover default option initialization, invalid
  native-policy rejection, host-only load policy, string materialization, and
  embedded string rejection
- compiler-side embedded target checks are still needed so unsupported strings
  and desktop native calls fail before bytecode is deployed

---

## D-0065

- Date: 2026-05-25
- Status: Accepted
- Title: Reject desktop native calls during embedded VM loading

### Context

`ESHKOL_VM_NATIVE_POLICY_HOST_ONLY` rejects desktop native fids when bytecode
executes, but that still lets a product host accept bytecode containing desktop
native dependencies. For firmware profiles, admission should fail before the VM
handle is exposed to the host loop.

The deterministic host-native range already gives product runtimes a stable
boundary: embedded bytecode should call `ESHKOL_VM_HOST_NATIVE_BASE + slot` and
should not contain direct calls into the broad desktop native table.

### Decision

Extend `EshkolVmLoadOptions` with `reject_desktop_native_calls`.

When this option is enabled, `eshkol_vm_load_chunk_with_options` scans decoded
ESKB instructions after structural/profile validation and rejects any
`OP_NATIVE_CALL` operand below `ESHKOL_VM_HOST_NATIVE_BASE`. The default loader
continues to allow desktop native calls so existing desktop VM behavior remains
unchanged.

### Consequences

- embedded/product hosts can reject desktop native dependencies at load time
  instead of waiting for execution to reach them
- fixed host-native slots remain valid under the embedded admission policy
- the VM C API tests now prove both acceptance of host-native slots and
  rejection of desktop native fids through load options
- compiler-side target checks are still needed to report source-level reasons
  before bytecode is emitted

---

## D-0066

- Date: 2026-05-25
- Status: Accepted
- Title: Enforce required VM entries during product-script loading

### Context

The public VM ABI can query and run named ESKB function entries, which is enough
for a host loop to call `init`, `tick`, `input`, or `render`. But before this
decision, the loader still accepted chunks that lacked those entries. Product
runtimes then had to discover the mistake after load, or encode entry checks as
out-of-band convention.

Tamatsotchke-style firmware needs script admission to prove that the bytecode
contains the hooks the device loop will call repeatedly.

### Decision

Extend `EshkolVmLoadOptions` with:

- `required_functions`
- `required_function_count`

`eshkol_vm_load_chunk_with_options` now validates those names against the
decoded ESKB function table after structural/profile validation. Missing,
empty, or null required names reject the chunk. The default loader continues to
require no named entries.

### Consequences

- product hosts can fail missing script-entry contracts at load time
- existing desktop VM loading remains unchanged
- the public VM C API tests now cover required-entry success, missing entry
  rejection, null required-name rejection, missing required-name array rejection,
  and negative count rejection
- compiler-side embedded target checks should still produce source-level
  diagnostics before ESKB emission

---

## D-0067

- Date: 2026-05-25
- Status: Accepted
- Title: Expose guarded ESKB emission through the VM header

### Context

`embedded-vm` now has a guarded ESKB emission path: it omits the desktop VM
prelude and rejects bytecode that still contains desktop-native
`OP_NATIVE_CALL` operands. The CLI and VM C API tests were using that symbol as
an ad hoc cross-language declaration, and native Windows builds use
`eshkol_vm_stub.c` instead of the full VM unity hub.

Jack's Windows x86 validation makes this a real deployment boundary: every
symbol used by `eshkol-run` must exist in both the full VM build and the native
Windows stub build.

### Decision

Declare both ESKB emitters in `inc/eshkol/backend/vm.h`:

- `eshkol_emit_eskb`
- `eshkol_emit_eskb_embedded`

Make `eshkol-run` include that header instead of locally redeclaring the
symbols, and add an `eshkol_emit_eskb_embedded` stub beside the existing native
Windows `eshkol_emit_eskb` stub.

### Consequences

- C and C++ callers now share one public declaration for desktop and embedded
  ESKB emission
- native Windows builds stay link-complete even though full ESKB emission still
  returns an unavailable diagnostic there
- the embedded emitter remains backed by the full VM/toolchain path on hosted
  non-Windows builds

---

## D-0068

- Date: 2026-05-25
- Status: Accepted
- Title: Emit named VM entries from source ESKB output

### Context

The public VM loader can preserve, query, and call named ESKB functions, and
load options can require entries such as `init` or `tick`. Before this
decision, compiler-produced ESKB still emitted a single synthetic `main`
function record, so product entry checks were only useful for hand-authored
or post-processed bytecode.

`embedded-vm` needs the normal source-to-bytecode path to produce the same
entry table that product runtimes validate at load time.

### Decision

Extend the ESKB writer with a multi-function CODE-section path while keeping
the existing single-function writer API as a compatibility wrapper. The VM
source compiler records closed top-level function definitions as named entries
alongside synthetic `main`, and the embedded emitter writes those entries into
the ESKB function table.

Only closed top-level functions are exported as independent entries in this
slice. Functions that capture upvalues remain reachable through normal VM
execution but are not advertised as product hooks yet.

### Consequences

- compiler-produced embedded ESKB can satisfy required-entry load options for
  hooks such as `tick`
- product hosts can call emitted source hooks through `eshkol_vm_call`
- the public VM C API test now loads compiler-emitted embedded ESKB requiring
  both `main` and `tick`, then calls `tick` and verifies its result
- export-manifest diagnostics are still a compiler/profile responsibility above
  this conservative function-table emission

---

## D-0069

- Date: 2026-05-25
- Status: Accepted
- Title: Gate VM ESKB output with required entry checks

### Context

The compiler now emits named ESKB function records for closed top-level source
functions, and the VM loader can reject chunks that lack required entries. But
without a CLI gate, product build scripts still had to run a separate loader
check after `eshkol-run --profile embedded-vm --emit-eskb`.

Firmware builds need the normal bytecode-emission command to fail atomically
when a script lacks hooks such as `init` or `tick`.

### Decision

Add `--require-vm-entry NAME` to `eshkol-run` for VM profiles. The option can
be repeated. After ESKB emission, the CLI reloads the emitted bytecode through
the public VM loader and checks every requested entry. For `embedded-vm`, the
admission load uses the product policy: host-native-only dispatch, rejected
ESKB string constants, and rejected desktop-native calls.

If admission fails, `eshkol-run` reports the missing entry or profile admission
failure, removes the emitted `.eskb` path, and exits non-zero.

### Consequences

- product builds can fail missing VM hook contracts in the ESKB emission step
- rejected bytecode artifacts are not left behind for firmware packaging
- hosted VM profiles can also use explicit required-entry checks while retaining
  desktop-compatible load policy
- richer manifest formats can build on top of this explicit entry list

---

## D-0070

- Date: 2026-05-25
- Status: Accepted
- Title: Expose VM function table enumeration

### Context

Product hosts and build tools can already ask whether a specific VM entry is
present and can call it by name. That is enough for a fixed hook contract, but
inspection tools still had to know the requested hook names up front or reach
into decoded ESKB internals to show what a bytecode artifact actually exports.

The compiler now emits closed top-level source functions as named ESKB entries,
so the public ABI needs a way to list those entries without exposing
`EskbModule`.

### Decision

Add two public VM C ABI calls:

- `eshkol_vm_function_count`
- `eshkol_vm_function_name`

The count call reports the decoded function-table size for a loaded VM handle.
The name call returns a borrowed pointer for an in-range function index, valid
until the VM handle is destroyed. Invalid handles and out-of-range indices fail
without exposing VM internals.

### Consequences

- product tooling can list emitted hooks before packaging or deployment
- embedders can build richer manifest checks above the existing required-entry
  loader and `--require-vm-entry` CLI gate
- the VM C API test now covers count, ordered names, null handles, and
  out-of-range index rejection

---

## D-0071

- Date: 2026-05-25
- Status: Accepted
- Title: Expose VM function metadata

### Context

Function-table enumeration lets tooling list emitted VM hooks, but product
admission often needs more than names. A firmware loop may need to verify hook
arity, reject closures for fixed entry contracts, or estimate bytecode budget
before packaging.

The ESKB CODE section already stores function metadata during decoding, and the
VM loader validates that metadata before creating a handle. Keeping that data
private forced tools to either duplicate ESKB parsing or limit themselves to
name-only checks.

### Decision

Add `EshkolVmFunctionInfo` and `eshkol_vm_function_info` to the public VM C ABI.
The accessor fills borrowed function name, parameter count, local count,
upvalue count, code offset, and code length for an indexed decoded function.
Invalid handles, invalid indices, and null output pointers return `-1`.

### Consequences

- product tooling can inspect hook arity and closedness without decoding ESKB
  directly
- bytecode packaging can use code-length and local-count metadata for early
  budget checks
- native Windows stubs expose the same symbol while the full bytecode VM remains
  disabled there
- the VM C API tests now cover metadata success and invalid-input rejection
