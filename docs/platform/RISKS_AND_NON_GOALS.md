# Risks and Non-Goals

## Non-Goals

The early platform program explicitly does not attempt to solve:

- a fully custom boot chain on every architecture
- universal support for non-LLVM arbitrary CPUs
- full hosted feature parity in freestanding mode
- large-scale driver development inside the compiler repo
- a complete operating system inside `~/Desktop/eshkol`
- making every advanced Eshkol subsystem available in the earliest boot path

These are deferred, downstream, or separate research efforts.

## Primary Risks

## Risk 1: Core-file merge pressure

High-risk areas:

- `lib/backend/llvm_codegen.cpp`
- runtime and allocator code
- build system and toolchain helpers

Why it matters:

- active roadmap work already touches these files
- large platform refactors can collide with `v1.2` and later work

Mitigation:

- narrow topic branches
- frequent merges from the roadmap branch
- isolate policy changes behind profiles

## Risk 2: Runtime split regressions

Why it matters:

- the current runtime is broad and entangled
- accidental hosted behavior regressions would hurt the main product

Mitigation:

- establish a symbol inventory before large refactors
- introduce leakage tests
- move boundaries incrementally

## Risk 3: VM/native semantic drift

Why it matters:

- Eshkol already has multiple execution backends
- freestanding work can create silent divergence if not tested carefully

Mitigation:

- document profile semantics centrally
- add backend-specific smoke tests
- record contract decisions in `DECISIONS.md`

## Risk 4: Syntax churn

Why it matters:

- low-level types, pointer forms, attributes, and intrinsics can be bikeshedded endlessly

Mitigation:

- founder-led decision process
- use early accepted syntax decisions
- separate “provisional” from “accepted” in `DECISIONS.md`

## Risk 5: BSP sprawl

Why it matters:

- adding too many targets too early diffuses effort and increases maintenance cost

Mitigation:

- one BSP first
- one or two follow-up reference targets only after the contract is stable
- use the support tier model strictly

## Risk 6: Platform program overwhelms the roadmap

Why it matters:

- the mainline still has `v1.2-v1.7` deliverables to ship

Mitigation:

- keep the public milestone at `v1.8-platform`
- merge only release-safe infrastructure early
- keep bring-up work on the side branch until stable

## Deferred Questions

These questions are real but should not block early implementation:

- exact inline assembly design
- full interrupt ABI surface
- packed struct and bitfield syntax details
- kernel-safe subset policy for continuations and dynamic features
- how much of the VM should be feature-gated in freestanding mode

## Scope-Protection Rules

- If a task does not improve the freestanding compiler/runtime contract, it should not land in the platform branch by default.
- If a task only matters once a kernel already exists, it belongs in `eshkol-kernel`, not the compiler repo.
- If a change cannot be explained in terms of execution profiles, runtime families, BSPs, or downstream kernel handoff, it is probably out of scope.
