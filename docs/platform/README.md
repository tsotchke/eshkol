# Eshkol Freestanding / Platform Program

This directory documents the long-running platform program that makes Eshkol a profiled systems language capable of producing freestanding kernels, firmware, loaders, and embedded artifacts without disrupting the existing release schedule.

## Why This Program Exists

Eshkol already has the right high-level shape for systems work:

- a real compiler front end
- a production LLVM backend
- a production bytecode VM
- deterministic arena-based memory
- explicit runtime value layouts
- substantial standard library and module infrastructure

What it does not yet have is a clean freestanding architecture. Today the implementation is still hosted-first. This program documents the work required to add:

- execution profiles
- freestanding runtime families
- native freestanding LLVM output
- freestanding VM mode
- low-level machine primitives
- BSP and linker-script contracts
- a downstream kernel consumer path

## Current Program Model

- Active roadmap integration branch: `feature/v1.2-scale`
- Recommended long-lived platform branch: `feature/platform-freestanding`
- Primary repo for platform/toolchain work: `~/Desktop/eshkol`
- Downstream kernel repo: `~/Desktop/eshkol-kernel`

This program starts during `v1.2-scale`, runs in parallel with the roadmap, and converges publicly at `v1.8-platform`.

## How To Use These Docs

Read in this order:

1. [PROGRAM_OVERVIEW.md](PROGRAM_OVERVIEW.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [ROADMAP_ALIGNMENT.md](ROADMAP_ALIGNMENT.md)
4. [BRANCHING_AND_GOVERNANCE.md](BRANCHING_AND_GOVERNANCE.md)
5. [SYNC_POLICY.md](SYNC_POLICY.md)
6. [IMPLEMENTATION_WORKFLOW.md](IMPLEMENTATION_WORKFLOW.md)
7. [SLICE_CHECKLIST.md](SLICE_CHECKLIST.md)
8. [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md)
9. [WORKSTREAMS.md](WORKSTREAMS.md)
10. [MILESTONES_AND_EXIT_CRITERIA.md](MILESTONES_AND_EXIT_CRITERIA.md)

Use the remaining docs as operational references while implementation proceeds.

## Document Index

- [PROGRAM_OVERVIEW.md](PROGRAM_OVERVIEW.md)
  - program vision, end-state architecture, phases, success definition
- [ROADMAP_ALIGNMENT.md](ROADMAP_ALIGNMENT.md)
  - how the platform program fits `v1.2` through `v1.8`
- [BRANCHING_AND_GOVERNANCE.md](BRANCHING_AND_GOVERNANCE.md)
  - branch roles, merge policy, founder-led decision process, delegation rules
- [SYNC_POLICY.md](SYNC_POLICY.md)
  - roadmap-to-platform sync cadence, hotspot files, merge-back criteria, full vs selective sync
- [IMPLEMENTATION_WORKFLOW.md](IMPLEMENTATION_WORKFLOW.md)
  - worktree layout, topic-branch lifecycle, merge flow, validation expectations
- [SLICE_CHECKLIST.md](SLICE_CHECKLIST.md)
  - release-quality checklist for every platform change slice
- [RUNTIME_INVENTORY.md](RUNTIME_INVENTORY.md)
  - concrete file, header, and ownership baseline for the runtime split
- [ARCHITECTURE.md](ARCHITECTURE.md)
  - technical architecture for profiles, runtimes, backends, BSPs, and kernel handoff
- [WORKSTREAMS.md](WORKSTREAMS.md)
  - subsystem-by-subsystem implementation tracks and dependencies
- [MILESTONES_AND_EXIT_CRITERIA.md](MILESTONES_AND_EXIT_CRITERIA.md)
  - milestone definitions, tests, exit criteria, and sign-off rules
- [RISKS_AND_NON_GOALS.md](RISKS_AND_NON_GOALS.md)
  - scope protection, risk register, deferred questions
- [TEST_AND_CI_PLAN.md](TEST_AND_CI_PLAN.md)
  - testing strategy, CI lanes, promotion rules, invariants
- [TARGET_SUPPORT_MATRIX.md](TARGET_SUPPORT_MATRIX.md)
  - support tiers, reference targets, target-class expectations
- [KERNEL_HANDOFF.md](KERNEL_HANDOFF.md)
  - contract between `eshkol` and `eshkol-kernel`
- [DECISIONS.md](DECISIONS.md)
  - append-only architecture and governance decision log

## Current Status

Program phase:

- documentation and planning complete enough to guide implementation
- execution profile architecture is merged on `master`
- runtime inventory baseline is documented on `master`
- runtime-core and runtime-hosted internal source sets are explicit in CMake
- hosted `runtime_exports.h` wrappers now live in a dedicated `runtime_exports_hosted.cpp` unit
- machine integer and raw pointer annotation surfaces exist in the HoTT parser/type-checker path
- pointer conversion and byte-offset arithmetic builtins exist for the low-level surface: `null-ptr`, `ptr->usize`, `usize->ptr`, and `ptr-add`
- tracked `Ptr` bindings round-trip through variable storage and typed codegen without collapsing back into generic heap-object handling
- `addr-of` exists for storage-backed bindings, giving the low-level surface a direct address-taking primitive without widening the general procedure ABI
- `compiler-fence` and `memory-fence` exist with explicit ordering operands for compiler-only and system-scope barrier emission
- `volatile-load` and `volatile-store!` exist for typed MMIO-style load/store lowering over raw pointers
- `atomic-load` and `atomic-store!` exist for typed raw-pointer memory accesses with explicit load/store orderings
- `target-intrinsic` exists as a typed LLVM intrinsic escape hatch for low-level codegen
- declaration attributes exist for section placement, alignment, symbol export/remapping, weak linkage, used retention, and no-return metadata
- deeper runtime/freestanding implementation remains staged on `feature/platform-freestanding`
- embedded targets remain a public `v1.8-platform` milestone
- early program work should remain merge-safe during roadmap releases before `v1.8`

Immediate priorities:

- split the current `runtime-split-pending` files along host-dependent seams
- hosted leakage checks for runtime-core
- low-level machine-facing language surface
- freestanding LLVM object and ELF pipeline
- stdlib capability partitioning

## Program Rules

- Hosted behavior must not regress while platform infrastructure is added.
- Freestanding work lands in small mergeable slices.
- Kernel and bootloader work remain downstream until the toolchain can produce stable freestanding artifacts.
- Architectural decisions are recorded in [DECISIONS.md](DECISIONS.md).

## Related Docs

- [../architecture/README.md](../architecture/README.md)
- [../development/README.md](../development/README.md)
- [../future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md](../future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md)
- [../../ROADMAP.md](../../ROADMAP.md)
