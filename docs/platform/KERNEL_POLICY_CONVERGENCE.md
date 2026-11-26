---
kind: explanation
status: current
owner-area: platform
since: v1.4.0-connection
sources:
  - inc/eshkol/kernel/k_kernel_policy.h
  - lib/kernel/k_kernel_policy.c
  - proofs/lean/EshkolKernelPolicy.lean
  - tests/kernel/k_convergence_policy_test.cpp
---

# Eshkol OS Phase K5/K6: Formal Policy Verification, Bounded Model Checking, and Unified K1–K3 Convergence

## Overview

This layer unifies the Phase K1 ([KERNEL_K1_BOOT_TRAPS.md](KERNEL_K1_BOOT_TRAPS.md)), Phase K2 ([KERNEL_K2_MMU_FRAMES.md](KERNEL_K2_MMU_FRAMES.md)), and Phase K3 ([KERNEL_K3_SMP_CAP_IPC.md](KERNEL_K3_SMP_CAP_IPC.md)) microkernel cores under `k_microkernel_t` and adds formally verified supervisor restart policy and compositor/audio resource arbitration (`inc/eshkol/kernel/k_kernel_policy.h`, `lib/kernel/k_kernel_policy.c`).

## Verified Policy Subsystems

### 1. Supervisor Restart Policy and Zombie-Lock Elimination
- **Atomic Lock Release on Fault**: `k_policy_service_fault` atomically releases every lock owned by the faulting service (`lock_owner[l] = K_POL_NO_OWNER`, `held_lock_mask = 0`), preventing zombie lock accumulation across faults and restarts.
- **Strictly Monotonic Token Generations**: `k_policy_service_restart` mints a strictly greater `token_gen > last_faulted_gen`, permanently invalidating any token held prior to the fault (`K_POL_ERR_STALE_TOKEN`).
- **Fail-Closed Revoked-Capability Rejection**: Revoking a service's root capability (`k_policy_revoke_service_cap`) transitions it to `K_POL_SVC_STOPPED` and causes all subsequent restart attempts to fail closed (`K_POL_ERR_REVOKED_CAP`).

### 2. Compositor Surface and Real-Time Audio Arbitration
- **Spatial Non-Overlap Across Distinct Badges**: `k_policy_surface_register` and `k_policy_surface_write` enforce pairwise spatial disjointness across active surfaces and fail closed on cross-badge ownership conflicts (`K_POL_ERR_SURFACE_CONFLICT`) without mutating the framebuffer.
- **Read-Only Surface Trap**: Any write through a read-only surface capability traps explicitly (`K_POL_ERR_RO_TRAP`), increments `ro_trap_count`, and preserves every framebuffer cell identically.
- **Bounded Audio Ring Degradation**: `k_policy_audio_step` enforces `count <= capacity` across arbitrary overrun and underrun bursts, accounting for dropped incoming frames (`overrun_drops`) and silence fills (`underrun_silences`) while keeping `supervisor_halted == 0`.

## Formal Verification (`proofs/lean/EshkolKernelPolicy.lean`)

Machine-checked in Lean 4.12.0 with zero `sorry` and zero custom axioms:
- `supervisor_restart_never_leaks_prior_token`
- `supervisor_revoked_cap_rejects_restart`
- `supervisor_no_zombie_lock_on_restart`
- `registerSurface_preserves_disjoint` and `compositor_surface_spatial_disjointness`
- `compositor_readonly_write_always_traps`
- `audio_step_preserves_valid` and `compositor_audio_bounds_preserved`

## Bounded State-Space Model Checking and Convergence Verification

`tests/kernel/k_convergence_policy_test.cpp` executes two complementary verification layers:
1. **Exhaustive Bounded State-Space BFS Model Checker**: Explores thousands of reachable interleaved states of `(service_fault, service_restart, cap_revoke, lock_acquire, surface_write, audio_produce, audio_consume)`, asserting `k_policy_invariants_hold` plus RO-trap, cross-badge conflict, and stale-token negative controls at every reachable state.
2. **End-to-End Unified K1+K2+K3+Policy Convergence Test**: Boots the complete unified microkernel with zero heap allocations (`heap_allocs == 0`), exercises W^X MMU mappings, 1,000 multi-CPU timer IRQs, live RV64 illegal-instruction fault containment across K1/K3/Policy, mid-flight revoked IPC rejection, monotonic restart token validation, compositor/audio arbitration, and leak-free frame teardown.
