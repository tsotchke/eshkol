---
kind: explanation
status: current
owner-area: platform
since: v1.4.0-connection
sources:
  - inc/eshkol/kernel/k3_smp_cap_ipc.h
  - lib/kernel/k3_smp_cap_ipc.c
  - proofs/lean/EshkolKernelK3.lean
  - tests/kernel/k3_smp_cap_ipc_test.cpp
---

# Eshkol OS Phase K3: Verified Zero-Alloc SMP Scheduler, Capability Derivation Tree, and Session-Typed Ring IPC

## Overview

Phase K3 completes the formally verified microkernel core specified in [OS_ENABLEMENT_PLAN.md](OS_ENABLEMENT_PLAN.md), building directly on Phase K1 ([KERNEL_K1_BOOT_TRAPS.md](KERNEL_K1_BOOT_TRAPS.md)) and Phase K2 ([KERNEL_K2_MMU_FRAMES.md](KERNEL_K2_MMU_FRAMES.md)).

## Subsystems

### 1. Cache-Line Isolated Per-CPU SMP Scheduler (`k3_smp_sched_t`)
- Per-CPU runqueues (`K3_MAX_CPUS = 8`) are strictly 64-byte aligned (`_Alignas(64)` / `alignas(64)`) and padded to an exact multiple of 64 bytes (`sizeof(k3_cpu_runqueue_t) == 192`), enforced by compile-time static assertions and runtime address checks to eliminate cross-core false sharing.
- Deterministic priority/FIFO preemptive dispatch (`k3_smp_schedule_next`, `k3_smp_preempt_tick`) and isolated task fault containment (`k3_smp_fault_task`), purging only the faulting task while sibling processes continue running uninterrupted.

### 2. Capability Derivation Tree with Immediate Atomic Revocation (`k3_cdt_t`)
- Static boot-sized capability slot pool (`K3_MAX_CAPS = 256`) with atomic 32-bit liveness/generation tracking and kernel-minted unforgeable badges.
- **Monotonic Rights Attenuation**: `k3_cdt_derive` enforces `child_rights ⊆ parent_rights`, rejecting escalation fail-closed (`K3_ERR_RIGHTS_ESCALATION`).
- **Bounded Depth & Cycle Rejection**: `k3_cdt_derive` caps depth at `K3_MAX_CDT_DEPTH = 16` (`K3_ERR_CDT_DEPTH_OVERFLOW`), and `k3_cdt_reparent` walks ancestors to reject self-parenting and multi-hop cycles (`K3_ERR_CDT_CYCLE`).
- **Immediate Atomic Subtree Revocation**: `k3_cdt_revoke_subtree` atomically invalidates the target node and every transitive descendant via a bounded static DFS stack (zero recursion, zero heap) and increments the global revocation epoch.

### 3. Zero-Copy Session-Typed `submit`/`complete` Ring IPC & Shared-Memory Confinement
- Static ring descriptors (`K3_IPC_RING_CAP = 64`) enforce an explicit session-type protocol (`INIT -> REQUEST_SENT -> RESPONSE_SENT -> CLOSED`), rejecting out-of-protocol operations (`K3_ERR_SESSION_PROTOCOL`).
- `k3_ipc_complete` re-validates the in-flight `sender_cap` atomically against the live CDT **before** copying payload bytes or mutating receiver state; revoking a capability or any CDT ancestor mid-flight aborts completion with `K3_ERR_CAP_REVOKED` and zero side effects.
- Compartment-tagged shared data pages (`k3_shm_page_t`) enforce capability-gated data reads/writes while rejecting raw capability smuggling or cross-compartment forgery fail-closed (`K3_ERR_CAP_CONFINEMENT`).

## Formal Verification (`proofs/lean/EshkolKernelK3.lean`)

Machine-checked in Lean 4.12.0 with zero `sorry` and zero custom axioms:
- **Monotonic Rights & Acyclicity**: `cdt_empty_valid`, `mintRoot_preserves_valid`, `derive_preserves_valid`, `child_capability_never_exceeds_parent`, `strict_ancestor_increases_depth`, `cdt_depth_bounded_and_acyclic`.
- **Revocation Completeness & Mid-Flight Zero Side-Effects**: `revocation_completeness`, `zero_side_effects_on_revoked_complete`.
- **Confinement & Non-Interference**: `shm_step_preserves_cspaces`, `capability_confinement_and_non_interference`.

## Verification Gates

1. **Runtime & Adversarial Suite (`tests/kernel/k3_smp_cap_ipc_test.cpp`)**: Exercises CDT rights escalation, depth overflow, cycle rejection, deterministic + multi-threaded mid-flight `submit`/`revoke`/`complete` races, 64-byte cache-line isolation, shared-memory capability smuggling rejection, 10,000 session-typed IPC exchanges (20,000 messages), protocol violation controls, userspace fault isolation, and strict `heap_allocs == 0`.
2. **Freestanding 5-Target Object & Symbol Audit**: Compiles `lib/kernel/k3_smp_cap_ipc.c` across all 5 target triples (`x86_64-unknown-none-elf`, `aarch64-unknown-none-elf`, `riscv64-unknown-none-elf`, `riscv32-unknown-none-elf`, `thumbv7em-none-eabi`) with zero warnings, zero external symbols (`llvm-nm -u` empty), and zero unwind sections.
3. **Lean 4.12.0 Proof Replay**: `lean proofs/lean/EshkolKernelK3.lean` exits `0` with zero warnings and zero custom axioms.
