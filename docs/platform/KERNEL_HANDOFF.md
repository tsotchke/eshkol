# Kernel Handoff

## Purpose

This document defines when and how `~/Desktop/eshkol-kernel` becomes the downstream consumer of the freestanding/platform work in `~/Desktop/eshkol`.

## Why Kernel Work Is Downstream

The compiler repo must first provide:

- stable execution profiles
- a freestanding runtime
- low-level language primitives
- a native freestanding link pipeline
- a BSP contract

Without that foundation, kernel work becomes a shadow compiler fork and loses mergeability.

## Preconditions For Real Kernel Work

`eshkol-kernel` should remain minimal until all of the following exist upstream:

- profile model implemented
- `runtime-core` and `runtime-freestanding` exist
- low-level machine types and pointer/volatile primitives exist
- freestanding object and ELF output work
- first reference BSP exists
- at least one smoke-tested target can execute a minimal freestanding image

## What Belongs In `eshkol`

- compiler driver
- syntax and type changes
- runtime families
- VM freestanding support
- stdlib partitioning
- BSP contract definitions
- reference BSP implementations
- target metadata and linker conventions
- integration tests and CI for the platform itself

## What Belongs In `eshkol-kernel`

- kernel entry and boot consumer code
- serial console implementation using the upstream language/runtime surface
- memory manager
- trap and interrupt subsystem
- scheduler
- drivers
- platform experiments that do not change the compiler contract

## Early Kernel Bootstrap Plan

Phase 1 in `eshkol-kernel`:

- trivial kernel entry
- serial write
- memory map and bootstrap diagnostics

Phase 2:

- early allocator integration
- trap vector placeholders
- page or frame allocator scaffolding

Phase 3:

- scheduler skeleton
- interrupt enablement
- platform abstractions for drivers

## Handoff Checklist

Before upstream platform work is consumed by `eshkol-kernel`, verify:

- compiler flags and profile names are stable enough
- runtime boundaries are documented
- first BSP contract is documented
- output artifact expectations are documented
- smoke tests already exist upstream
- no local compiler patches are needed for the downstream build

## Ongoing Contract Between Repos

- `eshkol` owns platform semantics
- `eshkol-kernel` owns kernel behavior
- compiler changes needed by the kernel must land upstream first
- downstream work may prototype needs, but upstream remains the source of truth for language and runtime contracts

## Success Condition

The handoff is considered successful when `eshkol-kernel` can build and boot a minimal kernel against stable upstream interfaces without local compiler/runtime modifications.
