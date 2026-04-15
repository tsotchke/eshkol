# Roadmap Alignment

## Purpose

This document explains how the freestanding / platform program fits into the existing release schedule without derailing the current product roadmap.

The public roadmap currently places embedded cross-compilation in `v1.8-platform`. The platform program starts much earlier because the necessary compiler, runtime, and language infrastructure is too large to fit inside a single release window.

## Alignment Rule

The public feature milestone remains `v1.8-platform`.

The internal implementation rule is:

- start now during `v1.2-scale`
- merge only infrastructure that is safe and timely for the current release
- keep more disruptive work on the platform branch until the mainline is ready
- converge fully by `v1.8-platform`

## Release-by-Release Convergence

## v1.2-scale

Public focus:

- model serialization
- Python bindings
- per-thread arenas
- image I/O
- CSV/DataFrame
- improved error spans
- terminal plotting

Platform work allowed to merge during `v1.2`:

- execution profile model
- build-system and driver groundwork for profile-aware compilation
- runtime split scaffolding that does not change hosted behavior
- diagnostics improvements that also help profile and target errors
- stdlib capability audit and documentation

Platform work that should remain on `feature/platform-freestanding` during `v1.2`:

- linker scripts
- startup objects
- BSP directories
- freestanding ELF images
- low-level syntax that is still unsettled
- VM freestanding runtime hooks

## v1.3-evolve

Public focus:

- full R7RS library system
- string interpolation
- keyword arguments
- let destructuring
- PGO
- whole-program optimization

Strong convergence points:

- `define-library` / import work directly helps stdlib partitioning
- module capability boundaries become easier to express cleanly
- profile and target diagnostics can become stricter

Platform merges to target in this window:

- stdlib partitioning
- profile-aware module gating
- stable low-level type additions
- safe compiler attributes if syntax is settled

## v1.4-connection

Public focus:

- sockets
- TLS
- non-blocking I/O
- event loop
- linear types for handles
- borrow pattern

Strong convergence points:

- hardware resources are naturally modeled as linear or borrowed handles
- device ownership, DMA buffers, and IRQ registrations can eventually reuse the same discipline

Platform merges to target in this window:

- generalized handle/resource typing that helps both networking and hardware
- freestanding runtime hook refinement
- stricter capability separation between hosted and non-hosted services

## v1.5-intelligence to v1.7-synthesis

Public focus:

- neuro-symbolic intelligence
- logic engine maturity
- program synthesis

Relationship to the platform program:

- these releases do not depend heavily on freestanding work
- they provide a relatively good window for platform work to continue in parallel
- platform merges should remain small and low-risk

Platform work to advance mostly on the side branch:

- additional BSPs
- freestanding VM maturity
- reference-target expansion
- kernel bootstrap in `eshkol-kernel`

## v1.8-platform

Public focus:

- cross-platform windowing
- event system
- audio
- Vulkan
- multi-GPU
- embedded cross-compilation

Required convergence by `v1.8`:

- execution profiles merged
- freestanding runtime merged
- low-level language surface merged
- freestanding object and ELF generation merged
- BSP contract merged
- at least one reference embedded or bare-metal target demonstrated

This lets `v1.8-platform` be a productization and public-surface milestone rather than the point where infrastructure first begins.

## Merge Policy by Release Window

Use these rules when deciding whether platform work can merge into the active roadmap branch.

### Merge now if the change is:

- profile-aware but behavior-preserving in hosted mode
- build or driver infrastructure
- documentation or tests
- runtime separation that does not alter current public semantics
- syntax or type support that is already stable and harmless for hosted users

### Keep on the platform branch if the change is:

- architecture-specific bring-up
- linker-script or startup-object heavy
- likely to cause large conflicts in active roadmap files
- still open-ended in syntax or semantics
- kernel-consumer specific

## Scheduling Guidance

The platform program should consume only a bounded share of the mainline attention during `v1.2-v1.4`.

Recommended operating model:

- keep `feature/v1.2-scale` as the active release stream
- use `feature/platform-freestanding` for the hardware program
- merge mainline into platform frequently
- merge back only stable slices

## Status Labels

Every platform work item should carry one of these labels in planning docs or issues:

- `merge-safe-now`
- `platform-branch-only`
- `downstream-kernel-only`
- `public-v1.8-surface`

These labels make it easier to keep the program aligned with the release schedule.
