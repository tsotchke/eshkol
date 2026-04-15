# Branching and Governance

## Purpose

This document defines how the freestanding / platform program is run as a founder-led effort that still supports bounded delegation and regular convergence with the main roadmap.

## Branch Model

## Primary branches

- `feature/v1.2-scale`
  - current roadmap integration branch
  - remains focused on shipping current release work
- `feature/platform-freestanding`
  - long-lived incubator for freestanding, embedded, BSP, and toolchain work
  - all platform topic branches start here

## Topic branch model

All implementation work should happen on short-lived topic branches off `feature/platform-freestanding`.

Recommended naming:

- `topic/platform-profile-model`
- `topic/platform-runtime-split`
- `topic/platform-llvm-freestanding`
- `topic/platform-language-machine-types`
- `topic/platform-language-pointers-volatile`
- `topic/platform-stdlib-freestanding`
- `topic/platform-vm-hook-abi`
- `topic/platform-bsp-x86_64-qemu`
- `topic/platform-bsp-aarch64-qemu`
- `topic/platform-bsp-riscv-qemu`
- `topic/platform-kernel-bootstrap`

Topic branch rule:

- one clear subsystem boundary
- one reviewable set of changes
- one merge target

## Merge Flow

### Mainline into platform

- merge or rebase `feature/v1.2-scale` into `feature/platform-freestanding` every 1-3 days while churn is high
- never allow the platform branch to drift far from the main release line

### Platform into mainline

Merge only when all are true:

- hosted mode behavior is preserved
- relevant tests pass
- the change is within the current release window's acceptable scope
- no kernel-only artifacts are required to justify the change

### Kernel repo relationship

Changes should not be merged into `~/Desktop/eshkol-kernel` until the upstream compiler/runtime contract is stable enough for downstream consumption. `eshkol-kernel` is not the place to prototype unsettled compiler semantics.

## Founder-Led Authority Model

The founder/maintainer is the program lead and final decision-maker for:

- execution profile semantics
- low-level language syntax
- runtime family boundaries
- BSP contract shape
- `eshkol` / `eshkol-kernel` repo boundary
- support tier promotion
- milestone sign-off

This is deliberate. Delegation is implementation-focused, not architecture-authority focused.

## Delegation Rules

Delegation is appropriate when the task is bounded and decision-complete.

Good delegated tasks:

- add tests for a defined profile behavior
- split a runtime file along a specified boundary
- implement a BSP from an already-defined contract
- wire CI lanes from an already-defined matrix
- implement a low-level primitive whose syntax and semantics are already accepted

Poor delegated tasks:

- define the low-level type system
- choose profile names and semantics
- design interrupt attributes without a prior decision
- invent the `eshkol` / `eshkol-kernel` boundary

## Ownership Model

Because this is a founder-led program, ownership is best expressed as “primary owner plus optional delegate,” not as equal subsystem co-owners.

### Required roles

- Program lead
  - primary owner: founder/maintainer
- Compiler driver and integration
  - primary owner: founder/maintainer or delegated compiler engineer
- LLVM and codegen
  - primary owner: founder/maintainer or delegated codegen engineer
- Runtime and memory
  - primary owner: founder/maintainer or delegated runtime engineer
- VM
  - primary owner: founder/maintainer or delegated VM engineer
- Stdlib and module partitioning
  - primary owner: founder/maintainer or delegated stdlib engineer
- BSP and reference targets
  - primary owner: founder/maintainer or delegated systems engineer
- Downstream kernel bootstrap
  - primary owner: founder/maintainer

## Decision Process

## Decision classes

### Class A: architecture-boundary decisions

Examples:

- execution profile names
- runtime families
- first reference targets
- stdlib partition rules
- repo boundary between `eshkol` and `eshkol-kernel`

Process:

- record context in `docs/platform/DECISIONS.md`
- founder/maintainer approval required

### Class B: subsystem implementation decisions

Examples:

- a particular file split
- CI lane naming
- a linker helper implementation

Process:

- document in the relevant topic branch or PR notes
- founder review required only if it changes a public or cross-subsystem contract

## RFC threshold

Create a short design note before implementation if a change affects:

- syntax
- profile semantics
- runtime family membership
- BSP contract
- kernel handoff criteria
- target support tier definitions

These notes can begin as entries or references in `DECISIONS.md`; they do not need a heavyweight RFC system unless the program grows significantly.

## Conflict Handling

When platform work conflicts with active roadmap work:

- prefer keeping the platform change on `feature/platform-freestanding`
- do not block release-critical work on `feature/v1.2-scale`
- rebase after the roadmap slice stabilizes
- if the conflict touches a core boundary, record the issue in `DECISIONS.md`

## Documentation Obligations

A change is not ready to merge if it changes any of the following without updating the docs:

- profile semantics
- runtime family structure
- BSP contract
- milestone exit criteria
- support tier definitions
- test/CI requirements

The doc set under `docs/platform/` is part of the program implementation, not optional commentary.
