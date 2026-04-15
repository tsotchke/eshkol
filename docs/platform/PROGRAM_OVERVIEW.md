# Program Overview

## Purpose

The freestanding / platform program turns Eshkol from a hosted native compiler with a portable VM into a profiled systems language platform with explicit support for:

- kernels
- firmware
- boot stages
- embedded targets
- non-hosted VM deployments

The program is intentionally longer-running than any single release. It starts during `v1.2-scale`, advances in parallel with existing roadmap work, and converges publicly at `v1.8-platform`.

## Core Thesis

Eshkol already contains most of the architectural ingredients needed for systems work:

- a mature front end
- a modular LLVM backend
- a real bytecode VM
- deterministic memory management
- explicit tagged-value and object-header layouts
- FFI and external symbol mechanisms

The missing piece is not “more language power.” The missing piece is a clean execution architecture that distinguishes:

- hosted execution
- freestanding native execution
- freestanding VM execution
- board and SoC support
- downstream kernel consumers

## What This Program Delivers

By the end of the program, Eshkol should have:

- explicit execution profiles
- separate hosted and freestanding runtime families
- native freestanding object and ELF generation
- a freestanding VM mode with explicit host hooks
- low-level machine-facing language primitives
- capability-based stdlib partitioning
- BSP contracts for reference targets
- a stable handoff path to `eshkol-kernel`

## Current State

Today Eshkol is strongest in these areas:

- LLVM-native compilation
- REPL and ORC JIT
- bytecode VM and ESKB
- web platform / WASM
- tensor, AD, ML, and advanced runtime facilities

Today Eshkol is weakest for systems work in these areas:

- startup and entrypoint control
- linker-script-driven outputs
- runtime separation
- volatile memory operations
- pointer and section semantics
- interrupt-safe and boot-safe language features
- BSP and memory map contracts

## End-State Architecture

The target architecture is:

- one language front end
- multiple execution profiles
- multiple backends
- multiple runtime families
- board-specific support layers
- a downstream kernel consumer repo

The end-state decomposition is:

- Language front end
  - parser
  - macro expander
  - AST
  - type checker
  - module resolver
- Execution profiles
  - hosted-native
  - hosted-wasm
  - hosted-vm
  - freestanding-kernel-native
  - freestanding-mcu-native
  - freestanding-vm
- Backends
  - LLVM native
  - bytecode VM / ESKB
  - weight-matrix backend
- Runtime families
  - runtime-core
  - runtime-hosted
  - runtime-freestanding
  - vm-core
  - vm-freestanding
- Platform layer
  - BSPs
  - linker scripts
  - startup objects
  - target-specific hooks
- Downstream consumer
  - `eshkol-kernel`

## Program Principles

### 1. Hosted Eshkol must remain healthy

The platform program cannot break the main user-facing language or release stream. Hosted compilation remains the default until freestanding capability is proven.

### 2. Platform infrastructure lands before kernel code

The first success metric is a stable freestanding toolchain, not a booting OS.

### 3. Profiles are the architectural boundary

The compiler, runtime, stdlib, tests, and docs must all agree on profile semantics. Profile boundaries must be explicit rather than implied by ad hoc flags.

### 4. The VM is a first-class systems asset

The bytecode VM is not a browser-only side subsystem. It is part of the long-term portability story.

### 5. Mergeability matters more than ambition

Every change should be made in a form that can converge into the existing roadmap without creating a permanent fork.

## Program Phases

### Phase 1: Foundation

- execution profile model
- compiler driver refactor
- runtime stratification groundwork
- docs, governance, and CI scaffolding

### Phase 2: Freestanding language and runtime

- machine-facing types
- pointers, volatile operations, barriers
- runtime-core and runtime-freestanding split
- stdlib partition groundwork

### Phase 3: Freestanding native output

- generic entry symbol generation
- freestanding object pipeline
- ELF linking with linker scripts
- first reference BSP contract

### Phase 4: Freestanding VM

- VM host-hook ABI
- freestanding VM build
- ESKB execution in non-hosted mode

### Phase 5: Reference targets

- first QEMU BSP
- later second and third reference targets
- documentation and smoke tests for each

### Phase 6: Downstream kernel bootstrap

- `eshkol-kernel` becomes active
- kernel entry, serial, memory bootstrap, traps, and platform demos

## Success Definition

The program is successful when all of the following are true:

- Eshkol can build freestanding objects and linked ELF images without host runtime leakage.
- Eshkol has a documented low-level language surface for hardware-facing code.
- At least one reference target can execute an Eshkol-generated freestanding image.
- The VM can run in a hook-driven freestanding configuration.
- The compiler and runtime changes are merged into the main development stream without breaking the release cadence.
- `eshkol-kernel` can consume the platform with stable interfaces rather than local compiler hacks.

## Founder-Led Operating Model

This program assumes:

- the founder/maintainer is the primary architect and primary implementer
- delegation happens through bounded topic branches
- syntax and architecture boundary decisions remain centralized

This is intentional. The documentation exists to preserve context and make selective delegation safe, not to decentralize core design authority.
