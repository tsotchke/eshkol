# Freestanding / Platform Architecture

## Purpose

This document is the canonical technical architecture for the Eshkol freestanding / kernel / embedded program.

It describes:

- the current Eshkol architecture
- what blocks bare-metal and embedded execution today
- the target profile-based architecture
- the runtime and backend splits required
- the boundary between the compiler repo and the downstream kernel repo

## 1. Current Eshkol Architecture

## 1.1 Language front end

Eshkol already has a strong portable language core:

- Scheme-family syntax
- hygienic macro expansion
- AST-based compiler pipeline
- HoTT-inspired gradual typing
- homoiconic semantics
- module system
- standard library largely written in Eshkol

This front-end architecture should remain shared across all execution profiles.

## 1.2 Native LLVM backend

The LLVM backend is currently the primary production path. It already supports:

- native object generation
- hosted executable generation
- WebAssembly generation
- REPL and ORC JIT support

The backend is modular and already broad enough to serve as the primary native path for systems work. The main issue is not capability but policy: too many host assumptions remain embedded in the current pipeline.

## 1.3 Bytecode VM backend

The bytecode VM is already a real execution backend, not just a toy interpreter. It provides:

- a 64-opcode core VM
- ESKB bytecode format
- portable execution
- browser deployment
- a native-call bridge

This is strategically important for freestanding and embedded work because it offers a second path for portability and constrained deployments.

## 1.4 Runtime and stdlib

The runtime currently mixes:

- tagged value operations
- arena memory
- closures and dynamic features
- system services
- file and process utilities
- tensor, ML, and GPU-related helpers
- REPL support glue
- thread-pool and parallel features

The stdlib similarly assumes a broad hosted environment.

This is the main architectural barrier to freestanding support.

## 2. Current Architecture Problems

The current system is blocked on the following issues:

- executable generation assumes hosted entrypoints and host linker behavior
- native linking assumes host runtime archives and host libraries/frameworks
- runtime boundaries are implicit rather than layered
- no stable low-level language surface exists for MMIO, pointers, sections, or barriers
- no explicit BSP or memory-map contract exists
- the VM has not been formalized as a freestanding target
- stdlib compatibility is not expressed per capability or profile

## 3. Target Architecture

The target architecture is a profiled language platform:

- one front end
- multiple execution profiles
- multiple runtime families
- multiple backends
- explicit BSP contracts
- a downstream kernel consumer

## 3.1 Execution profiles

Introduce the following first-class profiles:

- `hosted-native`
- `hosted-wasm`
- `hosted-vm`
- `freestanding-kernel-native`
- `freestanding-mcu-native`
- `freestanding-vm`

The driver surface is `eshkol-run --profile NAME` with optional
`--target TRIPLE`. Freestanding native profiles require an explicit target
triple, imply compile-only object output, and imply `--no-stdlib` until the
freestanding runtime and stdlib partitions are available. `--wasm` remains a
compatibility alias for the hosted WASM profile.

Each profile must define:

- backend
- runtime family
- allowed stdlib surface
- startup model
- entrypoint semantics
- artifact outputs
- diagnostics policy

## 3.2 Runtime families

The runtime must be split into:

- `runtime-core`
  - tagged values
  - object headers
  - arena primitives
  - minimal closures
  - panic/assert hooks
  - essential arithmetic helpers
- `runtime-hosted`
  - filesystem
  - process environment
  - stdio
  - OS clocks
  - sockets
  - threads
  - dynamic services
  - hosted-specific glue
- `runtime-freestanding`
  - allocator bootstrap
  - console hooks
  - timer hooks
  - trap/panic hooks
  - no implicit libc dependency
- `vm-core`
  - opcode engine
  - bytecode loader
  - VM value model
  - native-call bridge
- `vm-freestanding`
  - hook-driven runtime services
  - no implicit OS assumptions

The concrete starting point for this split is documented in [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md).

Two architectural clarifications matter here:

- `lib/core` is not the runtime boundary. It currently contains runtime substrate, hosted control-plane code, and higher-level language services.
- `platform_runtime.h` and `runtime_exports.h` are already effectively hosted-runtime boundaries, even though they are still linked through the monolithic `eshkol-static` archive.

## 3.3 Native freestanding architecture

The LLVM-native freestanding path must support:

- generic entry symbol generation instead of implicit hosted `main()`
- explicit target selection
- explicit relocation and code models
- object output
- ELF linking
- linker-script-driven image layout
- later flat binary or hex output

Hosted-only link behavior must not leak into freestanding profiles.

## 3.4 VM freestanding architecture

The VM should be formalized as a non-hosted runtime profile with:

- explicit hook tables
- configurable allocator and console support
- optional timer integration
- explicit native-call registration

This enables:

- monitor shells
- embedded scripting
- recovery environments
- unsupported-target fallback paths

## 3.5 Stdlib architecture

The stdlib must be capability-partitioned rather than monolithic.

Recommended partitions:

- `core.freestanding`
- `core.hosted`
- `kernel`
- `embedded`
- `numeric`
- `vm.portable`
- `desktop`
- `web`
- `ai`

Profile-aware import and gating rules must determine what is legal in each build.

## 3.6 BSP and platform layer

The BSP layer must become the boundary between Eshkol and specific boards or execution environments.

Each BSP must define:

- startup objects
- linker script
- memory map symbols
- console/UART implementation
- timer support
- interrupt hooks or stubs
- target-specific helper bindings

Initial BSP targets should be:

- x86_64 QEMU PC
- AArch64 QEMU virt
- RISC-V QEMU virt

## 4. Low-Level Language Surface

The language needs new machine-facing primitives.

## 4.1 Required types

- `u8`, `u16`, `u32`, `u64`
- `i8`, `i16`, `i32`, `i64`
- `usize`, `isize`
- `ptr<T>`

## 4.2 Required operations

- pointer/integer casts
- byte-offset pointer arithmetic
- `addr-of`
- `volatile-load`
- `volatile-store!`
- `atomic-load`
- `atomic-store!`
- `atomic-exchange!`
- compiler and memory fences

## 4.3 Required attributes

- `link-section`
- `used`
- `weak`
- `align`
- `packed`
- `extern-symbol`
- `export-symbol`
- `interrupt-handler`
- `naked`
- `no-return`

Initial declaration-attribute support has landed for `link-section`, `used`, `weak`, `align`, `extern-symbol`, `export-symbol`, and `no-return`. `packed`, `interrupt-handler`, and `naked` remain separate target/ABI policy work.

## 4.4 Required escape hatch

One of the following must exist before deeper systems work:

- target intrinsic surface
- inline assembly
- both

The initial program only requires one stable escape hatch, not the entire final design space.

## 5. Kernel and Embedded Programming Model

The intended programming model is layered:

- reset/startup path uses low-level, tightly controlled forms
- HAL and BSP code uses pointers, volatile ops, sections, and explicit layout
- higher-level kernel code may use richer Eshkol features once the runtime is established

This means Eshkol does not need to force tagged, dynamic, or reflective features into the earliest boot path.

## 6. Non-Goals for Early Phases

The early phases do not attempt to solve:

- a fully custom boot chain from reset vector on every architecture
- support for non-LLVM arbitrary CPUs
- full hosted feature parity in freestanding mode
- a large driver ecosystem
- a full OS tree inside the compiler repo

## 7. Repository Boundary

## 7.1 What belongs in `~/Desktop/eshkol`

- profiles
- compiler driver changes
- low-level language surface
- runtime family splits
- stdlib partitioning
- VM freestanding mode
- BSP contracts
- reference BSPs
- tests and CI

## 7.2 What belongs in `~/Desktop/eshkol-kernel`

- kernel entry and bootstrap consumer code
- memory manager
- interrupt subsystem
- scheduler
- drivers
- kernel demos and platform experiments

The compiler repo owns the platform. The kernel repo consumes it.

## 8. Artifact Model

The program must make these artifact types explicit:

- `.o`
- `.bc`
- `.elf`
- `.bin`
- `.hex`
- ESKB payloads

Not every profile produces every artifact, but the expectations must be documented and testable.

## 9. Architectural Success Criteria

The platform architecture is complete enough to support downstream kernel work when all of the following are true:

- execution profiles are implemented and documented
- runtime-core and runtime-freestanding exist
- the LLVM backend can produce freestanding objects and ELF artifacts
- low-level language primitives exist for pointers, volatility, and sections
- a BSP contract exists and at least one reference BSP works
- the VM can run with explicit non-hosted hooks
- `eshkol-kernel` can consume the platform without local compiler hacks
