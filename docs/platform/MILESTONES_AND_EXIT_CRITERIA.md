# Milestones and Exit Criteria

## Purpose

This document defines the program milestones, what each milestone must deliver, what does not need to be complete yet, and what counts as done.

All milestone sign-off is owned by the founder/maintainer unless explicitly delegated.

## Milestone A: Program Bootstrap

### Release window

- `v1.2-scale`

### Required deliverables

- `docs/platform/` doc set exists
- branch and governance model documented
- workstreams documented
- milestone and CI expectations documented

### Not required yet

- any code changes
- any freestanding artifacts

### Exit criteria

- docs are checked in and linked from the existing developer docs

## Milestone B: Profile Architecture Landed

### Release window

- `v1.2-scale`

### Required deliverables

- profile enum and configuration object
- profile-aware CLI parsing
- profile-aware diagnostics
- tests for valid and invalid profile combinations
- driver-level smoke coverage for `eshkol-run --profile` and `--target`

### Not required yet

- real freestanding linking
- new low-level syntax

### Exit criteria

- hosted builds remain green
- freestanding profiles can be selected even if some are still stubs

## Milestone C: Runtime Split Foundation

### Release window

- late `v1.2` into `v1.3`

### Required deliverables

- runtime-core boundary defined and implemented
- hosted-only services separated structurally
- freestanding runtime surface defined
- symbol leakage tests introduced

### Not required yet

- complete BSP support
- first bootable image

### Exit criteria

- minimal freestanding runtime links without hosted-only services

## Milestone D: First Freestanding Object

### Release window

- `v1.3`

### Required deliverables

- generic entry symbol generation
- freestanding object output
- low-level machine types implemented
- pointers and volatile operations implemented

### Not required yet

- full ELF image
- BSP boot demo

### Exit criteria

- a minimal freestanding source file produces a correct object file

## Milestone E: First Freestanding ELF

### Release window

- `v1.3`

### Required deliverables

- linker-script support
- ELF linking path
- explicit runtime selection
- artifact smoke tests

### Not required yet

- multiple targets
- downstream kernel repo integration

### Exit criteria

- a trivial freestanding program links to ELF with no hosted runtime leakage

## Milestone F: First Reference BSP Boot

### Release window

- `v1.3-v1.4`

### Required deliverables

- first BSP
- startup object
- linker script
- serial or console output demo
- QEMU smoke test

### Not required yet

- second architecture
- full interrupt subsystem

### Exit criteria

- reference image runs in the target environment and emits a verifiable output signal

## Milestone G: Freestanding VM

### Release window

- `v1.4-v1.5`

### Required deliverables

- VM hook ABI
- `freestanding-vm` profile
- hook-driven freestanding build
- ESKB smoke test in non-hosted mode

### Not required yet

- VM parity with all hosted features

### Exit criteria

- VM runs under an explicit freestanding hook table with no hidden hosted assumptions

## Milestone H: Multi-Target Platform Layer

### Release window

- `v1.5-v1.7`

### Required deliverables

- second and third reference target support
- BSP contract maturity
- stronger target support matrix
- profile-aware stdlib partitioning is stable

### Not required yet

- complete driver ecosystem
- broad hardware coverage

### Exit criteria

- at least two architecture families have validated reference support

## Milestone I: Downstream Kernel Bootstrap

### Release window

- `v1.5-v1.7`

### Required deliverables

- `eshkol-kernel` build consumes upstream freestanding toolchain
- kernel entry and serial path
- early memory bootstrap
- trap placeholders

### Not required yet

- scheduler
- paging
- userland

### Exit criteria

- `eshkol-kernel` builds and boots on the first reference target without patching the compiler repo

## Milestone J: Public Convergence at v1.8

### Release window

- `v1.8-platform`

### Required deliverables

- embedded cross-compilation is a documented, real capability
- platform branch infrastructure has merged or is in final merge-ready shape
- support matrix is published and accurate
- public docs describe the platform surface

### Exit criteria

- `v1.8-platform` reflects a delivered capability, not an architectural research plan

## Required Sign-Off Inputs for Every Milestone

Each milestone must include:

- code state summary
- test state summary
- known gaps
- updated docs
- explicit founder/maintainer sign-off note or recorded acceptance in `DECISIONS.md`
