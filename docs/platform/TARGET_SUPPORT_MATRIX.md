# Target Support Matrix

## Purpose

This document makes “arbitrary processor” precise by defining support tiers and the expected meaning of support.

## Support Tiers

## Tier 1: Supported

Definition:

- profile is documented
- compiler output is tested
- artifact generation is tested
- reference boot or execution smoke test exists
- failures are considered regressions

## Tier 2: Experimental

Definition:

- implementation exists
- basic compile or link path works
- partial smoke testing exists
- behavior may change rapidly

## Tier 3: Research

Definition:

- architecture direction is known
- no stable contract is promised
- support may exist only as notes, prototypes, or downstream experiments

## Initial Target Classes

## Hosted targets

- macOS host builds
- Linux host builds
- Windows host builds

Status:

- Tier 1

## Freestanding native reference targets

### x86_64 QEMU PC

- intended role: first kernel-class reference target
- initial status: Tier 2 target for early bring-up
- target state by `v1.8`: Tier 1 candidate

### AArch64 QEMU virt

- intended role: second kernel-class reference target
- initial status: Tier 3
- target state by `v1.8`: Tier 2 candidate

### RISC-V QEMU virt

- intended role: third kernel-class or embedded-adjacent reference target
- initial status: Tier 3
- target state by `v1.8`: Tier 2 candidate

## MCU-class targets

Recommended first family:

- ARM Cortex-M or RISC-V MCU

Initial status:

- Tier 3

Target state by `v1.8`:

- at least one experimental Tier 2 MCU path

## VM target classes

### Hosted VM

- Tier 1

### Freestanding VM

- initial status: Tier 3
- target state by `v1.8`: Tier 2

## What “Support” Means

Support must always be qualified by:

- profile
- backend
- target class
- artifact type

Examples:

- “supported” does not mean every stdlib module works
- “supported” does not mean every advanced language feature is available
- “supported” does mean the documented profile/target combination compiles, links, and passes its required tests

## Artifact Expectations by Profile

### hosted-native

- `.o`
- executable
- optional `.bc`

### hosted-wasm

- `.wasm`
- related web artifacts

### hosted-vm

- ESKB
- hosted VM execution

### freestanding-kernel-native

- `.o`
- `.elf`
- later optional `.bin` or `.hex`

### freestanding-mcu-native

- `.o`
- `.elf`
- later `.bin` or `.hex`

### freestanding-vm

- ESKB
- hook-driven VM execution

## Feature Availability Rules

Hosted-only subsystems remain unavailable in freestanding profiles unless explicitly supported:

- filesystem
- sockets
- dynamic loading
- hosted clocks and environment
- REPL/JIT
- thread pool
- GPU/XLA

Freestanding support is profile- and target-specific, not universal by default.
