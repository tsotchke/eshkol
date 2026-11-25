---
kind: design
status: current
owner-area: platform
since: v1.4.0-connection
sources:
  - inc/eshkol/kernel/k2_mmu_frames.h
  - lib/kernel/k2_mmu_frames.c
  - proofs/lean/EshkolKernelK2.lean
  - tests/kernel/k2_mmu_frames_test.cpp
---

# Eshkol OS Phase K2: Verified Physical Frame Allocator, 3-ISA Page Tables, and MCU MPU Manager

## Overview

Phase K2 builds on Phase K1 ([KERNEL_K1_BOOT_TRAPS.md](KERNEL_K1_BOOT_TRAPS.md)) to deliver the zero-allocation physical memory allocator, bit-accurate 3-ISA hardware page-table managers, TLB shootdown engine, MCU MPU region manager, and machine-checked Lean 4.12.0 proofs specified in [OS_ENABLEMENT_PLAN.md](OS_ENABLEMENT_PLAN.md).

## Subsystems

### 1. Zero-Alloc Physical Frame Allocator (`k2_frame_pool_t`)
- Static boot-sized pool (`K2_MAX_FRAMES = 2048`, 4 KiB pages) backed by an intrusive O(1) free-list and an independent 64-bit word allocation bitmap (`alloc_bitmap`).
- Linear `k2_page_frame_t` handles carry `(phys_addr, frame_idx, generation, owner)`.
- Boot-reserved and runtime-allocated kernel frames (`K2_OWNER_KERNEL`) are immutable against user deallocation (`K2_ERR_KERNEL_FRAME_LOCKED`) and user mapping (`K2_ERR_KERNEL_ALIAS`).
- Double-free attempts fail closed with `K2_ERR_DOUBLE_FREE`, stale-generation handles fail closed with `K2_ERR_STALE_HANDLE`, and freeing a still-mapped frame fails closed with `K2_ERR_FRAME_STILL_MAPPED`.

### 2. 3-ISA Hardware Page-Table Managers and TLB Shootdown (`k2_aspace_t`)
Supported hardware PTE formats:

| Mode | Architecture | Hardware Bits Enforced |
|---|---|---|
| `K2_MMU_RV64_SV39` | RISC-V 64 Sv39 | `V`, `R`, `W`, `X`, `U`, `G`, `A`, `D`, 44-bit PPN, canonical 39-bit VA |
| `K2_MMU_RV64_SV48` | RISC-V 64 Sv48 | `V`, `R`, `W`, `X`, `U`, `G`, `A`, `D`, 44-bit PPN, canonical 48-bit VA |
| `K2_MMU_AARCH64_4K` | AArch64 4 KiB L3 | Valid `0b11`, `AP[2:1]`, `AF`, `nG`, `PXN`, `UXN`, 48-bit OA |
| `K2_MMU_X86_64_4L` | x86_64 4-Level | `P`, `R/W`, `U/S`, `A`, `D`, `G`, `NX` (bit 63), 52-bit PFN |

Invariants enforced on every `map`, `remap`, raw PTE insertion, and `translate` walk:
- **Strict W^X**: Simultaneous `WRITE` and `EXEC` fail closed with `K2_ERR_WX_VIOLATION`.
- **Privilege Separation**: Simultaneous `USER` and `KERNEL_ONLY`, supervisor PTEs in user spaces, or `PXN=0` user pages on AArch64 fail closed with `K2_ERR_PRIV_VIOLATION`.
- **Kernel Non-Aliasing & Conflicting-Permission Alias Rejection**: Kernel-owned frames never enter user address spaces (`K2_ERR_KERNEL_ALIAS`); aliasing any physical frame as both `RW` and `RX` or across distinct non-shared user ASIDs fails closed (`K2_ERR_PERM_ALIAS`).
- **Immediate TLB Shootdown**: Every permission flip or unmap invalidates matching ASID/VA TLB slots immediately and increments `tlb_shootdowns` and `tlb_invalidated_entries`.

### 3. MCU MPU Region Manager (`k2_mpu_table_t`)
- Manages up to 8 regions (`K2_MAX_MPU_REGIONS = 8`) for Cortex-M4 and RV32 PMP/MPU targets.
- Enforces power-of-two sizes (`>= 32` bytes), natural base alignment, strict W^X, pairwise non-overlap (`K2_ERR_MPU_OVERLAP`), and unconditional exclusion of the hardware vector table and kernel stack guard (`K2_ERR_MPU_RESERVED_GUARD`).

## Formal Verification (`proofs/lean/EshkolKernelK2.lean`)

Machine-checked in Lean 4.12.0 with zero `sorry` and zero custom axioms:
- **Frame Conservation**: `pool_init_conserved`, `pool_step_preserves_conservation`, `double_free_rejected_preserves_state`, `kernel_frame_free_rejected`, `no_frame_leak`.
- **W^X & Kernel Non-Aliasing**: `pte_checkSafe_iff`, `empty_aspace_wx_safe`, `empty_aspace_kernel_non_aliasing`, `aspace_step_preserves_wx`, `w_x_enforcement`, `aspace_step_preserves_kernel_non_aliasing`, `kernel_non_aliasing`.
- **Address-Space Disjointness**: `permConflict_symm`, `pte_self_no_conflict`, `mapFrame_preserves_disjoint`, `address_space_disjointness`.
- **MCU MPU Safety**: `mpu_overlaps_symm`, `mpu_add_preserves_safe`, `mpu_overlap_and_guard_exclusion`.

## Verification Gates

1. **Runtime & Adversarial Suite (`tests/kernel/k2_mmu_frames_test.cpp`)**: Exercises W^X and privilege-bit bypasses, raw PTE injection rejection, TLB shootdown invalidation, kernel & permission aliasing rejection, 100,000-op randomized frame conservation against an independent shadow bitmap, 10,000 map/unmap cycles across RV64/AArch64/x86_64, MCU MPU overlap/guard controls, and strict `heap_allocs == 0`.
2. **Freestanding 5-Target Object & Symbol Audit**: Compiles `lib/kernel/k2_mmu_frames.c` for all 5 freestanding target triples (`x86_64-unknown-none-elf`, `aarch64-unknown-none-elf`, `riscv64-unknown-none-elf`, `riscv32-unknown-none-elf`, `thumbv7em-none-eabi`) with zero warnings, zero hosted/allocator/libc/unwind symbols, and zero unwind sections.
3. **Lean 4.12.0 Proof Replay**: `lean proofs/lean/EshkolKernelK2.lean` exits `0` with zero warnings and zero custom axioms.
