# Workstreams

## Purpose

This document breaks the platform program into implementation tracks that can be worked on in parallel, delegated selectively, and merged incrementally.

## Workstream 1: Execution Profile Architecture

### Goal

Make execution profiles a first-class part of the compiler and documentation model.

### Deliverables

- profile enum and profile object
- driver-level profile parsing
- profile-aware diagnostics
- profile-aware artifact selection

### Dependencies

- none

### Primary areas

- `exe/eshkol-run.cpp`
- compiler configuration headers
- command-line docs

### Delegation scope

Safe to delegate once profile names and semantics are fixed.

### Merge criteria

- hosted behavior unchanged
- all profile resolution tests pass
- driver-level profile CLI smoke tests pass

## Workstream 2: Runtime Stratification

### Goal

Split runtime responsibilities into core, hosted, and freestanding layers.

### Deliverables

- runtime symbol inventory
- explicit internal source sets for `runtime-core` and `runtime-hosted`
- `runtime-core` boundary
- `runtime-hosted` boundary
- `runtime-freestanding` boundary
- hosted leakage tests

### Dependencies

- profile architecture

### Primary areas

- `lib/core/`
- selected runtime-adjacent code in `lib/backend/`
- build system and archive definitions
- [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md)

### Delegation scope

Delegate only file- or subsystem-bounded split work after the boundary is written down.

### Merge criteria

- hosted builds still pass
- minimal freestanding runtime links without hosted-only symbols
- runtime inventory stays current as boundaries move
- internal source grouping is explicit in the build graph

## Workstream 3: LLVM Freestanding Native Path

### Goal

Turn LLVM-native compilation into a real freestanding pipeline.

### Deliverables

- generic entrypoint generation
- freestanding object output
- freestanding link path
- linker-script support
- explicit linker-driver abstraction

### Dependencies

- profile architecture
- runtime stratification foundation

### Primary areas

- `lib/backend/llvm_codegen.cpp`
- `inc/eshkol/llvm_backend.h`
- `exe/eshkol-run.cpp`

### Delegation scope

Delegate carefully; this is central code with high merge pressure.

### Merge criteria

- freestanding `.o` output works
- freestanding ELF link works for a trivial program
- hosted link path is unchanged

## Workstream 4: Low-Level Language Surface

### Goal

Give Eshkol enough language surface to express HAL, startup, and MMIO code.

### Deliverables

- fixed-width machine types
- pointer type and casts
- volatile operations
- barriers/fences
- section/linkage/layout attributes
- target intrinsic escape hatch

### Dependencies

- profile architecture

### Primary areas

- parser
- AST definitions
- type checker
- LLVM lowering
- language docs

### Delegation scope

Delegate implementation only after syntax and semantics are recorded in `DECISIONS.md`.

### Merge criteria

- low-level samples compile
- IR and object output reflect correct semantics
- docs are updated

## Workstream 5: Stdlib Partitioning

### Goal

Replace the single hosted-oriented stdlib assumption with capability-aware partitions.

### Deliverables

- freestanding-safe core modules
- profile-aware module gating
- hosted-only module diagnostics
- separate build artifacts or packaging for profile-specific stdlib subsets

### Dependencies

- profile architecture
- module system improvements from the main roadmap where useful

### Primary areas

- `lib/*.esk`
- stdlib build rules
- module loader behavior

### Delegation scope

Good delegation candidate once capability boundaries are fixed.

### Merge criteria

- freestanding build can import a core module without hosted leakage
- hosted stdlib behavior remains stable

## Workstream 6: VM Freestanding Mode

### Goal

Make the VM a valid non-hosted runtime target.

### Deliverables

- `freestanding-vm` and `embedded-vm` profiles
- VM host-hook ABI
- hook-driven allocator and console behavior
- freestanding VM build target
- ESKB smoke tests in non-hosted mode
- embedded profile limits for static RAM, stack, frames, constants, and
  bytecode size
- deterministic host-call table ABI for product runtimes such as Tamatsotchke
- host-native-only VM dispatch policy that rejects desktop native calls
- named VM entry-point dispatch for product runtime loops
- ESKB string constant loading plus explicit embedded string rejection/content-pack
  policy at load time
- load-time rejection for desktop native fids in embedded bytecode profiles
- load-time required entry checks for product loops such as `init` and `tick`

### Dependencies

- profile architecture
- runtime split groundwork

### Primary areas

- `lib/backend/eshkol_vm.c`
- VM submodules
- `inc/eshkol/backend/vm.h`
- `lib/backend/vm_core.c`
- `lib/backend/vm_native.c`
- `lib/backend/eskb_reader.c`
- ESKB tooling
- build rules

### Delegation scope

Good delegation candidate for a VM-focused contributor with a fixed host-hook contract.

### Merge criteria

- VM builds in both hosted and freestanding modes
- hook-based smoke tests pass
- embedded VM rejects unsupported desktop native calls and enforces static
  memory/budget limits
- fixed host-call slots are exercised by the public VM C API tests
- host-native-only policy is exercised by the public VM C API tests
- named ESKB entry points are exercised by the public VM C API tests
- ESKB string constant materialization and embedded rejection behavior are
  exercised by the public VM C API tests
- embedded bytecode admission rejects desktop native calls before VM execution
  in the public VM C API tests
- required named VM entries are enforced by load options in the public VM C API
  tests
- compiler-produced embedded ESKB exposes closed top-level hooks such as `tick`
  through the function table and required-entry loader path
- `--require-vm-entry` is exercised by the profile CLI test for both accepted
  and rejected embedded ESKB outputs

## Workstream 7: BSP Contract and Reference Targets

### Goal

Define and validate the boundary between the toolchain and specific targets.

### Deliverables

- BSP contract
- startup object conventions
- linker-script conventions
- first reference BSP
- later second and third BSPs

### Dependencies

- freestanding LLVM path
- low-level language surface
- freestanding runtime core

### Primary areas

- `docs/platform/`
- new BSP directories
- build scripts and target metadata

### Delegation scope

Implementing a BSP from a fixed contract is a strong delegation candidate.

### Merge criteria

- reference BSP can link and boot a minimal image
- smoke tests exist

## Workstream 8: Downstream Kernel Bootstrap

### Goal

Start `eshkol-kernel` as a real consumer once the upstream contract is stable.

### Deliverables

- kernel build using the freestanding toolchain
- serial console
- early allocator usage
- trap placeholders
- memory bootstrap scaffolding

### Dependencies

- at least one working reference BSP
- stable upstream freestanding contract

### Primary areas

- `~/Desktop/eshkol-kernel`

### Delegation scope

Limited early. This work should remain closely coordinated with the founder/maintainer.

### Merge criteria

- the kernel builds without patching the compiler repo locally
- boot smoke test works on the first reference target

## Cross-Workstream Dependency Order

Preferred order:

1. Workstream 1
2. Workstream 2
3. Workstream 4
4. Workstream 3
5. Workstream 5
6. Workstream 6
7. Workstream 7
8. Workstream 8

Workstreams 3 and 4 can overlap once the profile model is defined. Workstreams 5 and 6 can proceed in parallel after the runtime boundaries are clear.
