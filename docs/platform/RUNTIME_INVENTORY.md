# Runtime Inventory

## Purpose

This document is the concrete inventory baseline for Workstream 2: runtime stratification.

Its job is to answer three questions before any file movement begins:

1. Which files and headers are actually part of the runtime substrate?
2. Which of those are intrinsically hosted today?
3. Which `lib/core` files are language services or tooling surfaces and should stay outside the runtime family split?

This inventory is intentionally specific. It is meant to support bounded implementation slices such as:

- `topic/platform-runtime-core-split`
- `topic/platform-runtime-hosted-split`
- `topic/platform-runtime-leakage-tests`
- `topic/platform-vm-runtime-split`

## Current Build Reality

Today the shipped runtime is still a wide aggregate inside `eshkol-static`, but the internal build graph is no longer fully monolithic.

Current build behavior:

- `CMakeLists.txt` globs most of `lib/**/*.cpp` and `lib/**/*.c`
- backend standalone `.c` files are filtered out
- `lib/backend/eshkol_vm.c` is compiled through `eshkol-vm-unity-obj`
  while its included VM components are classified into explicit source
  families
- `lib/core/image_io.c` is appended explicitly
- `lib/ffi/eshkol_ffi.cpp` is appended explicitly
- internal object libraries now exist for:
  - `eshkol-runtime-core-obj`
  - `eshkol-runtime-hosted-obj`
  - `eshkol-vm-unity-obj`
- REPL, toolchain binaries, and generated programs still rely on `eshkol-static`

This means the runtime split has started as an internal source-set decomposition while preserving the current aggregate output and link contract.

## Runtime Family Targets

This inventory uses the following target families:

- `runtime-core`
  - tagged value ABI
  - object headers
  - allocator substrate
  - fundamental numeric/runtime helpers
  - profile-independent runtime contracts
- `runtime-hosted`
  - signals
  - process model
  - files and directories
  - environment variables
  - stdio and terminal behavior
  - temp paths
  - host compiler/linker/process helpers
- `runtime-freestanding`
  - future hook-based allocator bootstrap, console, panic, timer, interrupt, and startup support
  - mostly a planned target family today
- `out-of-runtime`
  - code that currently lives near the runtime but is actually a language service, toolchain service, or optional product feature

## Header Inventory

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `inc/eshkol/eshkol.h` | `runtime-core` | direct | Canonical tagged value ABI, object layout, and low-level value helpers shared by codegen, runtime, VM, and FFI. |
| `inc/eshkol/core/runtime.h` | `runtime-core` API + profile implementations | split required | Public lifecycle API is shared, but the current implementation is signal/thread/process oriented and therefore hosted. |
| `inc/eshkol/platform_runtime.h` | `runtime-hosted` | direct | Explicitly host-facing: executable paths, shell quoting, compiler lookup, temp files, command execution, host linker args. |
| `inc/eshkol/runtime_exports.h` | `runtime-hosted` | direct | Generated-code ABI wrappers around `FILE*`, env, filesystem, directory, and libc/process behavior. |
| `inc/eshkol/core/config.h` | `runtime-hosted` control plane | direct | Environment variables, config files, optimization flags, logging configuration. |
| `inc/eshkol/core/resource_limits.h` | hosted implementation API + future core policy | partially classified | Policy surface may remain shared later, but the current implementation is env/timer/logging oriented and is classified with `runtime-hosted`. |
| `inc/eshkol/logger.h` | hosted implementation API + future core sink contract | partially classified | Formatting and severity vocabulary may survive as a freestanding hook contract later, but the current implementation is hosted and is classified with `runtime-hosted`. |
| `inc/eshkol/eshkol_ffi.h` | `out-of-runtime` hosted consumer | direct | Embedding/JIT API; not part of the freestanding runtime families. |

## Native Runtime Inventory

### Core substrate and split candidates

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `lib/core/arena_memory.h` | `runtime-core` | split required | Public arena substrate and object-allocation helpers are core, but the implementation still includes higher-level runtime services that need separate freestanding-safe units. |
| `lib/core/runtime_arena_core.cpp` | `runtime-core` | direct current implementation | Raw arena block creation/destruction, aligned allocation, zeroed allocation, scope push/pop/reset, statistics, and legacy list-node allocation. Uses abstract arena mutex and poison-diagnostic hooks so the core allocator does not read process environment variables or own hosted threading primitives. |
| `lib/core/runtime_arena_cpp.cpp` | `out-of-runtime` C++ adapter | direct | C++ `Arena` RAII wrapper around the C arena ABI. It is compiled into the aggregate archive for existing C++ consumers, but it is not part of generated-code runtime ABI and is kept outside `runtime-core` / `runtime-hosted` source families because it exposes C++ exception and convenience-wrapper semantics. |
| `lib/core/runtime.cpp` | `runtime-core` | direct | Public interrupt flag definition used by the inline `eshkol_runtime_interrupt_requested` hot path. Hosted lifecycle, operation, shutdown-hook, signal, and generated-code helper implementations have been split into dedicated units. |
| `lib/core/runtime_autodiff.cpp` | `runtime-core` | direct current implementation | Dual-number, AD-node, AD-tape, and nested-gradient TLS state used by generated autodiff code and REPL/JIT symbol registration. Uses arena allocation and tagged object layout without filesystem/process/thread APIs. |
| `lib/core/runtime_bytevector.cpp` | `runtime-core` | direct | Bytevector allocation, length, ref/set, and copy helpers used by generated bytevector code. Uses tagged object layout and arena allocation; error paths delegate to the hosted runtime fatal helper until panic hooks split. |
| `lib/core/runtime_closure_alloc.cpp` | `runtime-core` | direct current implementation | Closure environment allocation, legacy closure allocation, and header-backed callable allocation used by generated closure code, higher-order functions, continuations, and REPL/JIT execution. Uses arena allocation, tagged value initialization, closure metadata packing, and callable object headers without filesystem/process/thread APIs. |
| `lib/core/runtime_closure_reflection.cpp` | `runtime-core` | direct current implementation | Closure reflection helpers for `procedure-arity`, `procedure-name`, variadic checks, and the lambda registry used by homoiconic display/JIT registration. Uses closure metadata, arena string allocation, and the existing logger/error hooks. |
| `lib/core/runtime_continuations.cpp` | `runtime-core` | direct current implementation | First-class continuation state allocation, continuation closure construction, dynamic-wind stack management, and thunk dispatch helpers. Uses arena allocation, closure metadata, and the generated-code continuation ABI without filesystem/process/thread dependencies. |
| `lib/core/runtime_deep_equal.cpp` | `runtime-core` | direct current implementation | Deep structural equality helper for tagged values used by generated `equal?` lowering. Compares nulls, numbers, strings, symbols, cons cells, vectors, bignums, tensors, and pointer-like callable values using tagged object layout, arena cons accessors, and bignum/tensor metadata without filesystem/process/thread APIs. |
| `lib/core/runtime_hash_table.cpp` | `runtime-core` | direct current implementation | Hash-table allocation, tagged-key hashing/equality, open-addressing mutation/lookup, and key/value list materialization used by generated hash-table code. Uses arena allocation, tagged object layout, bignum comparison, and the abstract hash-table lock/unlock ABI without filesystem/process/thread APIs. |
| `lib/core/runtime_list_helpers.cpp` | `runtime-core` | direct current implementation | Generated-code list helper ABI for tagged-list reverse, quasiquote append/splice, recursion-depth checks, and list/vector error guards. Uses tagged cons cells and arena allocation; error paths delegate to the hosted runtime fatal helper until panic hooks split. |
| `lib/core/runtime_object_alloc.cpp` | `runtime-core` | direct current implementation | Header-aware allocation helper ABI for tagged heap objects, including generic header-backed allocation, zeroed allocation, multi-values, cons cells, strings, vectors, and symbols. Uses raw arena allocation, tagged object headers, and initialization of object payloads without filesystem/process/thread APIs. |
| `lib/core/runtime_regions.cpp` | `runtime-core` | direct current implementation | Global arena selection, worker thread-local arena lifecycle, arena merge ownership transfer, OALR region stack/lifecycle/allocation helpers, and region escape helpers registered with the REPL JIT. Uses arena allocation, tagged object headers, AD TLS reset symbols, and platform abstract once/mutex hooks without filesystem/process APIs. |
| `lib/core/runtime_shared_memory.cpp` | `runtime-core` | direct current implementation | Shared allocation, retain/release, weak-reference create/upgrade/release, and shared-header lookup ABI registered with REPL JIT and used by ownership paths. Uses process-independent reference-count metadata and malloc/free; no filesystem/process/thread APIs. |
| `lib/core/runtime_string.cpp` | `runtime-core` | direct | Header-backed string byte length, UTF-8 length/ref, and substring helpers used by generated string code. Uses only tagged object layout, byte walking, and arena string allocation. |
| `lib/core/runtime_tagged_cons.cpp` | `runtime-core` | direct current implementation | Tagged cons allocation, batch allocation, constructors, typed get/set helpers, type/flag query helpers, and full tagged-value copy helpers used by generated list, tensor index, hash-table, REPL/JIT, and FFI paths. Uses raw arena allocation and tagged value layout without filesystem/process/thread APIs. |
| `lib/core/runtime_tensor_alloc.cpp` | `runtime-core` | direct current implementation | Header-backed tensor object allocation plus dimensions/elements array allocation used by codegen, FFI, model IO, and tensor backends. Uses arena allocation, tagged object layout, and raw memory initialization without filesystem/process/thread APIs. |
| `lib/core/runtime_tensor_index.cpp` | `runtime-core` | direct | Tensor index normalization and row-major offset helpers used by generated tensor-ref/tensor-set code. Operates only on tagged values, cons cells, tensor layout, and raw dimensions. |
| `lib/core/runtime_tensor_fill.cpp` | `runtime-core` | direct | Native `tensor-rect-fill!` / `tensor-disk-fill!` helpers. Operate only on tagged tensor layout and raw memory, with no allocation or hosted process/thread/file dependency. |
| `lib/core/runtime_tensor_math.cpp` | `runtime-core` | direct current implementation | Tensor linalg, broadcast, shape-conversion, concat, and batched-matmul helper ABI used by generated tensor code and REPL JIT registration. Uses raw tensor dimensions/elements, arena tagged cons accessors, and C math/memory primitives without filesystem/process/thread APIs. |
| `lib/core/bignum.cpp` | `runtime-core` | direct | Numeric runtime support with no obvious host/process dependency. |
| `lib/core/rational.cpp` | `runtime-core` | direct | Numeric runtime support with no obvious host/process dependency. |
| `lib/core/ad_tape_builtins.c` | `runtime-core` optional | direct | Runtime numeric/AD helper surface. |

### Hosted runtime and hosted runtime-adjacent code

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `lib/core/platform_runtime.cpp` | `runtime-hosted` | direct | Explicit host/process/toolchain implementation for `platform_runtime.h`. |
| `lib/core/runtime_arena_sync_hosted.cpp` | `runtime-hosted` | direct current implementation | Current thread-safe arena locks, hash-table lock, and global-arena once initialization are backed by `std::mutex` on Windows and `pthread_mutex_t` / `pthread_once_t` on POSIX hosts. A later freestanding profile can provide target-specific critical-section and once primitives behind the same internal ABI. |
| `lib/core/runtime_display_hosted.cpp` | `runtime-hosted` | direct current implementation | Current display/write, current input/output/error port cells, UTF-8 character/string emission, and FILE-backed rendering for lists, vectors, tensors, closures, ports, bignums, logic values, and workspace values. A later freestanding profile can keep the display ABI with a target-specific output sink. |
| `lib/core/runtime_errors_hosted.cpp` | `runtime-hosted` | direct current implementation | Current fatal/type-error implementation writes to stderr/logger state, raises hosted exception objects, and exits the process. A later freestanding panic/error hook can keep the ABI names without this sink implementation. |
| `lib/core/runtime_exceptions_hosted.cpp` | `runtime-hosted` | direct current implementation | Current R7RS exception state, `longjmp` handler dispatch, unhandled-exception stderr printing, and REPL forward-reference provider diagnostics. The diagnostic path scans hosted project files with filesystem/fstream and can later be replaced by a freestanding diagnostic hook behind the same ABI. |
| `lib/core/runtime_exports_hosted.cpp` | `runtime-hosted` | direct | Dedicated hosted implementation for the `runtime_exports.h` generated-code ABI wrappers. |
| `lib/core/runtime_lifecycle_hosted.cpp` | `runtime-hosted` | direct current implementation | Current `eshkol_runtime_init`, shutdown, interrupt request/clear, runtime-state, stdout buffering, operation draining, hook dispatch, and signal restore sequencing are hosted lifecycle behavior. A later freestanding profile can provide target-specific lifecycle state behind the same ABI. |
| `lib/core/runtime_operations_hosted.cpp` | `runtime-hosted` | direct current implementation | Current in-flight operation tracking uses mutexes, condition variables, vectors, strings, and wall-clock durations to drain hosted work during shutdown. A later freestanding profile can keep the public operation ABI with a target-specific implementation. |
| `lib/core/runtime_parameters_hosted.cpp` | `runtime-hosted` | direct current implementation | Current `make-parameter` / `parameterize` storage uses malloc/realloc for binding stacks and hosted warnings on stack-growth failure. A later freestanding parameter stack can replace this implementation behind the same ABI symbols. |
| `lib/core/runtime_reader_hosted.cpp` | `runtime-hosted` | direct current implementation | Current `eshkol_read_sexpr` parses S-expressions from hosted `FILE*` streams with `fgetc` / `ungetc`, allocates tagged values into the arena, and interns symbols through the process-global symbol table. A later freestanding profile can keep the ABI with a target-specific reader stream. |
| `lib/core/runtime_shutdown_hooks_hosted.cpp` | `runtime-hosted` | direct current implementation | Current shutdown-hook registration and reverse-order dispatch use mutexes, vectors, strings, and hosted logging. A later freestanding profile can keep the public hook ABI with a target-specific registry. |
| `lib/core/runtime_signals_hosted.cpp` | `runtime-hosted` | direct current implementation | Current SIGINT/SIGTERM/SIGPIPE/fatal-signal and Windows unhandled-exception installation uses host signal/exception APIs and signal-safe shadow state. A later freestanding profile can keep the interrupt ABI with target-specific trap or poll hooks. |
| `lib/core/runtime_stack_hosted.cpp` | `runtime-hosted` | direct current implementation | Current `eshkol_init_stack_size` uses hosted process limits and `ESHKOL_STACK_SIZE` environment parsing to raise stack limits for deep recursion. A later freestanding profile can provide target startup or no-op behavior behind the same ABI. |
| `lib/core/runtime_string_ports_hosted.cpp` | `runtime-hosted` | direct current implementation | Current `open-input-string`, `open-output-string`, and `get-output-string` helpers are backed by hosted `FILE*`, `tmpfile`, `fmemopen`, and `open_memstream` behavior. A later freestanding profile can provide target-specific string streams behind the same ABI. |
| `lib/core/system_builtins.c` | `runtime-hosted` | direct | Heavy OS dependency surface: env, path, temp files, directory traversal, fork/exec, wait, symlink, file copy, process spawn. |
| `lib/core/config.cpp` | `runtime-hosted` | direct | Reads env, discovers home/config files, and binds host-facing optimization and logging controls. |
| `lib/core/resource_limits.cpp` | `runtime-hosted` | direct current implementation | Current implementation reads env, owns timeout/timer behavior, emits warning logs, and requests hosted runtime interrupts. A later core policy object can be extracted under a new slice. |
| `lib/core/runtime_arena_diagnostics_hosted.cpp` | `runtime-hosted` | direct current implementation | Hosted policy hook for arena diagnostics. Currently maps `ESHKOL_ARENA_POISON` to the allocator poison sentinel path while keeping process environment reads out of `runtime-core`. |
| `lib/core/logger.cpp` | `runtime-hosted` | direct current implementation | Current implementation depends on stderr/files, ANSI colors, OS-specific backtraces, platform symbolization, and hosted timestamps. A later freestanding logger hook can share the severity vocabulary without this sink implementation. |
| `lib/backend/thread_pool.cpp` | `runtime-hosted` adjunct | direct | Uses `std::thread` and env-driven host tuning. Important to the current hosted parallel runtime, but not part of freestanding v1. |

### Hosted services outside the runtime family split

These files are real hosted features, but they should not be renamed into runtime families just because they currently live in `lib/core`.

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `lib/core/kb_persistence.cpp` | `out-of-runtime` hosted service | direct | File persistence for the knowledge-base subsystem. Hosted and file-based, but not runtime substrate. |
| `lib/core/image_io.c` | `out-of-runtime` hosted service | direct | Optional image functionality, not runtime substrate. |
| `lib/core/onnx_export.c` | `out-of-runtime` hosted service | direct | Optional model export functionality, not runtime substrate. |
| `lib/ffi/eshkol_ffi.cpp` | `out-of-runtime` hosted service | direct | Embedding and JIT service; depends on `ReplJITContext` and hosted stdout pipe capture. |

### Language services and toolchain support currently living in `lib/core`

These files consume the runtime, but they are not the runtime and should not move into `runtime-core`, `runtime-hosted`, or `runtime-freestanding`.

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `lib/core/ast.cpp` | `out-of-runtime` | direct | Front-end / AST layer. |
| `lib/core/sexp_to_ast.cpp` | `out-of-runtime` | direct | Front-end lowering, even though it uses arena helpers. |
| `lib/core/execution_profile.cpp` | `out-of-runtime` | direct | Toolchain configuration, not runtime substrate. |
| `lib/core/printer.cpp` | `out-of-runtime` | direct | AST pretty-printer/debug surface. It prints compiler data structures and should not be counted as runtime substrate. |
| `lib/core/logic.cpp` | `out-of-runtime` | direct | Language service built on top of the core runtime. |
| `lib/core/logic_builtins.cpp` | `out-of-runtime` | direct | Language service built on top of the core runtime. |
| `lib/core/inference.cpp` | `out-of-runtime` | direct | Higher-level logic/inference engine. |
| `lib/core/workspace.cpp` | `out-of-runtime` | direct | Higher-level workspace subsystem. |
| `lib/core/introspection.cpp` | `out-of-runtime` hosted consumer | direct | Uses `arena_memory`, `tmpfile` / `open_memstream`, and REPL JIT integration; hosted language tooling/service. |

## VM Inventory

`lib/backend/eshkol_vm.c` is still a unity-build hub for compilation-order
reasons, but its build boundary is no longer implicit. `CMakeLists.txt` now
owns explicit VM source families and builds the hub through
`eshkol-vm-unity-obj`, with `vm_source_boundary_test` checking that every
included VM component remains classified.

### VM core family

- `vm_core.c`
- `vm_run.c`
- `vm_numeric.h`
- `vm_complex.c`
- `vm_rational.c`
- `vm_bignum.c`
- `vm_dual.c`
- `vm_hyperdual.c`
- `vm_autodiff.c`
- `vm_tensor.c`
- `vm_tensor_ops.c`
- `vm_logic.c`
- `vm_inference.c`
- `vm_workspace.c`
- `vm_string.c`
- `vm_hashtable.c`
- `vm_bytevector.c`
- `vm_multivalue.c`
- `vm_error.c`
- `vm_parameter.c`
- `vm_geometric.c`
- `vm_symbolic_ad.c`

These are classified as `ESHKOL_VM_CORE_COMPONENT_SRC` because they define the
VM value model, data structures, opcode execution, numeric/runtime semantics,
and core tensor/logic/workspace behavior without direct file/process/thread
ownership.

### VM hosted/runtime-adjacent family

- `vm_io.c`
- `vm_model_io.c`
- `vm_parallel.c`
- `vm_native.c`
- `eskb_reader.c`
- `eskb_writer.c`

These pieces are classified as `ESHKOL_VM_HOSTED_COMPONENT_SRC`. They are the
current hosted runtime surface inside the VM path and depend on files,
directories, time, polling, signals, processes, dynamic loading, sockets, or
host threading. `vm_native.c` remains intentionally mixed here until its
native-call table can be partitioned by capability.

`vm_native.c` now exposes a deterministic host-native install API through
`inc/eshkol/backend/vm.h`. Embedders can install a fixed table of host calls
whose slots map directly to `ESHKOL_VM_HOST_NATIVE_BASE + index`; the existing
dynamic registration path remains available for desktop tests and tools.

VM handles also expose a native-call policy switch. The default policy preserves
the desktop native table; `ESHKOL_VM_NATIVE_POLICY_HOST_ONLY` rejects desktop
native fids at dispatch time and permits only fixed host-native slots. This is
the current product-runtime guardrail until `vm_native.c` is physically split by
capability.

`eskb_reader.c` now preserves the ESKB function table instead of discarding
non-zero function records. The public VM ABI can query and execute a named
zero-argument function entry, which gives embedded/product runtimes a stable
path for `init`/`tick`/`render`-style scripts before a dedicated export manifest
format lands.

`eskb_writer.c` can now write multiple CODE function records. The source VM
emitter keeps its synthetic `main` entry and also records closed top-level
function definitions as named ESKB entries, allowing compiler-produced embedded
bytecode to satisfy load-time required-entry policies.

The public VM loader and standalone `.eskb` path now materialize
`ESKB_CONST_STRING` entries as VM string heap objects. Firmware profiles may
still choose to reject dynamic script strings and route text through read-only
content packs, but the desktop/public loader no longer loses decoded string
constants.

`eshkol_vm_load_chunk_with_options` exposes the first load-time product profile
controls through the public C ABI. Embedders can start a loaded VM in
`ESHKOL_VM_NATIVE_POLICY_HOST_ONLY` and can reject `ESKB_CONST_STRING` entries
before bytecode runs. They can also reject `OP_NATIVE_CALL` operands below
`ESHKOL_VM_HOST_NATIVE_BASE`, which makes desktop-native dependencies fail at
load time instead of at first execution, including calls in helper function
bodies that the host has not invoked yet. Product hosts can also provide a list
of required function entries so scripts missing `init`/`tick`/`render`-style
hooks fail during loading. The default `eshkol_vm_load_chunk` path remains
desktop-compatible: desktop native calls are allowed, no entries are required,
and string constants are materialized.

`eshkol-run` also exposes build-time VM entry admission for VM profiles through
`--require-vm-entry NAME`. After ESKB emission, the CLI reloads the bytecode
through the public VM loader, checks the requested entry names, and removes the
output file on admission failure. For `embedded-vm`, that validation uses the
same host-native-only, no-string, no-desktop-native policy as product loading.
Tooling can enumerate the decoded function table through
`eshkol_vm_function_count` and `eshkol_vm_function_name` without depending on
`EskbModule` internals. `eshkol_vm_function_info` exposes borrowed names plus
parameter count, locals, upvalue count, code offset, and code length for
signature and budget inspection. Load options can also require that named
entries match exact arity, have no upvalues, and stay under local/code-length
ceilings before a VM handle is admitted.

### VM toolchain/compiler family

- `vm_parser.c`
- `vm_macro.c`
- `vm_compiler.c`
- `vm_peephole.c`
- `vm_prelude_source.h`
- `vm_prelude_cache.c`
- `vm_prelude_cache.h`

The `.c` components are classified as `ESHKOL_VM_TOOLCHAIN_COMPONENT_SRC`.
The prelude headers/cache generator remain VM toolchain adjuncts rather than
runtime families. `inc/eshkol/backend/vm.h` now declares both desktop ESKB
emission and the embedded emitter that omits the desktop VM prelude and rejects
desktop-native bytecode during emission. Native Windows builds still stub these
symbols while the full bytecode VM remains disabled for that target. The
emitter's product-entry table is intentionally conservative: only closed
top-level functions are exported as independent named entries in this slice.
The `eshkol-run --require-vm-entry-zero-arg` gate lets embedded VM builds reject
required hooks whose decoded function metadata is incompatible with the current
zero-argument public dispatch ABI: entries must declare zero parameters and no
upvalues. The runtime dispatch path itself also rejects entries that declare
upvalues, because the public call surface does not provide a closure environment
for direct entry execution. The VM loader also rejects empty or duplicate decoded
function names, making named hook admission and dispatch unambiguous. Known ESKB
sections are strict: CONST, CODE, and META must be fully consumed by their
decoders, and the declared section table must cover the full payload, so
trailing section or payload bytes fail before VM loading. Duplicate known
sections are also rejected, giving CONST, CODE, and META one authoritative
section instance each.

### VM test family

- `vm_tests.c`

This is classified as `ESHKOL_VM_TEST_COMPONENT_SRC` and remains test-only.

## Current Dependency Pressure

The following couplings matter for the runtime split:

- `llvm_codegen.cpp` includes `platform_runtime.h` and `runtime_exports.h`
- `string_io_codegen.cpp` includes `runtime_exports.h`
- `autodiff_codegen.cpp` includes `runtime_exports.h`
- `eshkol_ffi.cpp` depends on `core/runtime.h`, `arena_memory.h`, and `repl_jit.h`
- `introspection.cpp` depends on `arena_memory.h` and `repl_jit.h`
- REPL and JIT binaries force-load `eshkol-static` for symbol availability

This means the first extraction slices must preserve symbol names and link behavior even if the physical source grouping changes internally.

## Initial Extraction Order

The runtime split should proceed in this order:

1. Keep `runtime-core` and `runtime-hosted` explicit in CMake while still producing `eshkol-static`; the temporary `runtime-split-pending` bucket has been retired.
2. Keep `platform_runtime.cpp` and `runtime_exports_hosted.cpp` as distinct hosted runtime units instead of a fused host/runtime-export file.
3. Classify the current hosted logger and resource-limit implementations as
   `runtime-hosted`, and keep the AST pretty-printer out of the runtime source
   families.
4. Keep the decomposed arena implementation classified: raw arena mechanics in
   `runtime-core`, hosted diagnostics policy in `runtime-hosted`, and the C++
   `Arena` wrapper as a non-runtime adapter around the C ABI.
5. Add hosted-leakage tests that fail if `runtime-core` depends on env, files, temp streams, process control, or host threading primitives.
6. Compile the VM unity hub through its own object target and classify its
   included components into `vm-core`, `vm-hosted`, VM toolchain, and VM test
   buckets.
7. Physically split `eshkol_vm.c` behind the now-explicit VM source families.
8. Introduce the first real `runtime-freestanding` hooks and stub implementation.

## Non-Goals of This Slice

This inventory/source-set slice does not yet:

- move any files
- create new runtime archives
- define the final freestanding hook ABI
- make the VM freestanding

Its purpose is to make the next implementation slices bounded and unambiguous.
