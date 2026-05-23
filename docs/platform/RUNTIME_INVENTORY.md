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
- `lib/backend/eshkol_vm.c` is appended as a unity-build hub
- `lib/core/image_io.c` is appended explicitly
- `lib/ffi/eshkol_ffi.cpp` is appended explicitly
- internal object libraries now exist for:
  - `eshkol-runtime-core-obj`
  - `eshkol-runtime-hosted-obj`
  - `eshkol-runtime-split-pending-obj`
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
| `lib/core/arena_memory.h` | `runtime-core` | split required | Public arena substrate and object-allocation helpers are core, but the implementation currently includes pthread locking and string-port helpers that are not freestanding-safe. |
| `lib/core/arena_memory.cpp` | `runtime-core` | split required | Core allocator and tagged-object substrate. Must be split to remove pthread mutex setup and `tmpfile` / `open_memstream` string-port plumbing from the core slice. |
| `lib/core/runtime.cpp` | `runtime-core` API + `runtime-hosted` backend | split required | Interface is foundational, but current behavior depends on signals, condition variables, threads, stderr, and OS shutdown semantics. |
| `lib/core/bignum.cpp` | `runtime-core` | direct | Numeric runtime support with no obvious host/process dependency. |
| `lib/core/rational.cpp` | `runtime-core` | direct | Numeric runtime support with no obvious host/process dependency. |
| `lib/core/ad_tape_builtins.c` | `runtime-core` optional | direct | Runtime numeric/AD helper surface. |

### Hosted runtime and hosted runtime-adjacent code

| Path | Target family | Status | Notes |
| --- | --- | --- | --- |
| `lib/core/platform_runtime.cpp` | `runtime-hosted` | direct | Explicit host/process/toolchain implementation for `platform_runtime.h`. |
| `lib/core/runtime_exports_hosted.cpp` | `runtime-hosted` | direct | Dedicated hosted implementation for the `runtime_exports.h` generated-code ABI wrappers. |
| `lib/core/system_builtins.c` | `runtime-hosted` | direct | Heavy OS dependency surface: env, path, temp files, directory traversal, fork/exec, wait, symlink, file copy, process spawn. |
| `lib/core/config.cpp` | `runtime-hosted` | direct | Reads env, discovers home/config files, and binds host-facing optimization and logging controls. |
| `lib/core/resource_limits.cpp` | `runtime-hosted` | direct current implementation | Current implementation reads env, owns timeout/timer behavior, emits warning logs, and requests hosted runtime interrupts. A later core policy object can be extracted under a new slice. |
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

`lib/backend/eshkol_vm.c` is currently a unity-build hub, not a runtime family boundary. It mixes four different concerns into one compilation unit.

### VM core candidates

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

These are the natural `vm-core` candidates because they define the VM value model, data structures, opcode execution, and language-runtime semantics.

### VM hosted/runtime-adjacent candidates

- `vm_io.c`
- `vm_parallel.c`
- hosted portions of `vm_native.c`
- file-oriented parts of `eskb_reader.c`
- file-oriented parts of `eskb_writer.c`

These pieces are the current hosted runtime surface inside the VM path. They depend on files, directories, time, polling, signals, processes, or host threading.

### VM toolchain/compiler components

- `vm_parser.c`
- `vm_macro.c`
- `vm_compiler.c`
- `vm_peephole.c`
- `vm_prelude_source.h`
- `vm_prelude_cache.c`
- `vm_prelude_cache.h`

These belong to the VM toolchain path rather than the runtime families.

### VM tests

- `vm_tests.c`

This remains test-only and should be isolated when the VM hub is decomposed.

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

1. Keep `runtime-core`, `runtime-hosted`, and `runtime-split-pending` explicit in CMake while still producing `eshkol-static`.
2. Keep `platform_runtime.cpp` and `runtime_exports_hosted.cpp` as distinct hosted runtime units instead of a fused host/runtime-export file.
3. Classify the current hosted logger and resource-limit implementations as
   `runtime-hosted`, and keep the AST pretty-printer out of the runtime source
   families.
4. Split `runtime.cpp` and `arena_memory.cpp` along host-dependent seams so
   they can leave `runtime-split-pending`.
5. Add hosted-leakage tests that fail if `runtime-core` depends on env, files, temp streams, process control, or host threading primitives.
6. Decompose `eshkol_vm.c` into `vm-core`, `vm-hosted`, and VM toolchain buckets.
7. Introduce the first real `runtime-freestanding` hooks and stub implementation.

## Non-Goals of This Slice

This inventory/source-set slice does not yet:

- move any files
- create new runtime archives
- define the final freestanding hook ABI
- make the VM freestanding

Its purpose is to make the next implementation slices bounded and unambiguous.
