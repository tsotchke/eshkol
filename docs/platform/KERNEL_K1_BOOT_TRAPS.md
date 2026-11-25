---
kind: design
status: current
owner-area: platform
since: v1.4.0-connection
sources:
  - inc/eshkol/kernel/k1_boot_traps.h
  - lib/kernel/k1_boot_traps.c
  - lib/kernel/arch/k1_arch_x86_64.c
  - lib/kernel/arch/k1_arch_aarch64.c
  - lib/kernel/arch/k1_arch_riscv64.c
  - lib/kernel/arch/k1_arch_riscv32.c
  - lib/kernel/arch/k1_arch_cortex_m4.c
  - proofs/lean/EshkolKernelK1.lean
  - tests/kernel/k1_boot_traps_test.cpp
---

# Eshkol OS Phase K1: Verified Zero-Alloc Microkernel Early Boot and 5-ISA Trap Vectors

## Overview

Phase K1 establishes the zero-allocation microkernel early-boot, hardware handoff validation, per-CPU static IRQ ring buffer, and 5-ISA trap dispatch foundation specified in [OS_ENABLEMENT_PLAN.md](OS_ENABLEMENT_PLAN.md).

All kernel core routines compile strictly as freestanding C11 (`-ffreestanding -fno-builtin -fno-stack-protector -Wall -Wextra -Werror`) with zero heap allocator calls, zero C library symbols, and zero unwind tables across all five target architectures:

| ISA | Target Triple | Hardware Boot Handoff | Known-Bad Instruction | Native Illegal Cause | Valid NOP Control |
|---|---|---|---|---|---|
| `x86_64` | `x86_64-unknown-none-elf` | ACPI RSDP (`"RSD PTR "`) | `0F 0B` (`UD2`) | IDT vector `6` (`#UD`) | `90` (`NOP`) |
| `aarch64` | `aarch64-unknown-none-elf` | DTB (`0xD00DFEED`) / Apple ADT | `00 00 00 00` (`UDF #0`) | `ESR_EL1.EC = 0x00` | `1F 20 03 D5` (`0xD503201F`) |
| `riscv64` | `riscv64-unknown-none-elf` | DTB (`0xD00DFEED`) | `00 00 00 00` | `scause/mcause = 2` | `13 00 00 00` (`0x00000013`) |
| `riscv32` | `riscv32-unknown-none-elf` | DTB / MCU Vectors | `00 00 00 00` | `mcause = 2` | `13 00 00 00` (`0x00000013`) |
| `cortex-m4` | `thumbv7em-none-eabi` | Cortex-M Vector Table | `00 DE` (`0xDE00` `UDF #0`) | UsageFault `UNDEFINSTR` (`0x00060001`) | `00 BF` (`0xBF00` `NOP`) |

## Architecture and Invariants

### 1. Boot Handoff Validation (`k1_validate_boot_info`)
- **Flattened Device Tree (`K1_BOOT_DTB`)**: Verifies big-endian magic `0xD00DFEED`, `totalsize` within payload bounds, FDT version `>= 16`, and non-overflowing structure, strings, and memory-reservation offsets.
- **ACPI RSDP (`K1_BOOT_ACPI_RSDP`)**: Verifies the 8-byte `"RSD PTR "` signature, ACPI 1.0 20-byte checksum, and (for revision `>= 2`) extended table length and full checksum.
- **Apple Device Tree (`K1_BOOT_APPLE_ADT`)**: Verifies root node property/child count bounds, NUL-terminated first property name, and strictly in-bounds property payload length.
- **MCU Vector Handoff (`K1_BOOT_MCU_VECTORS`)**: On Cortex-M4 verifies at least 16 32-bit vectors, 8-byte aligned initial stack pointer, and Thumb-bit-set reset vector; on RV32 verifies non-zero entry instruction and aligned `mtvec` base.

### 2. Per-CPU Static IRQ Ring Buffers (`k1_irq_ring_t`)
- Compile-time static arrays: `K1_MAX_CPUS = 8`, `K1_IRQ_RING_CAPACITY = 256` per CPU.
- Power-of-two masking (`& 255u`) eliminates 64-bit division/modulo helper emissions on 32-bit targets.
- Ring saturation increments `dropped_total` and returns `K1_ERR_RING_FULL` without overwriting any live entry.

### 3. Trap Vector Isolation and Kernel Integrity Canaries
- `k1_kernel_context_t` surrounds the 16 kernel GPRs, stack pointer, program counter, and status register with dual 64-bit guard canaries (`K1_STACK_CANARY_LO` and `K1_STACK_CANARY_HI`).
- `k1_dispatch_user_trap` verifies kernel canaries before and after dispatch, enforces strictly decreasing bounded trap fuel, classifies every user fault (`K1_TRAP_DIV_ZERO`, `K1_TRAP_ILLEGAL_INSN`, `K1_TRAP_ACCESS_FAULT`, `K1_TRAP_UNKNOWN`) with its ISA-native cause, terminates only the faulting user task (`K1_TASK_FAULTED`), leaves all sibling user tasks untouched, and never mutates `kernel_ctx`.

## Formal Verification (`proofs/lean/EshkolKernelK1.lean`)

Machine-checked in Lean 4.12.0 with zero `sorry` and zero custom axioms:
- `ring_init_valid`, `ring_push_write_in_bounds`, `ring_pop_read_in_bounds`, `ring_push_overflow_preserves_entries`, `ring_push_preserves_valid`, `ring_pop_preserves_valid`, `ring_pop_decrements_count`.
- `all_five_isas_known_bad_traps_illegal` and `all_five_isas_known_nop_never_illegal`.
- `step_preserves_no_alloc`, `trace_preserves_no_alloc`, `step_strictly_decreases_fuel`, `step_preserves_ring_valid`, `trace_preserves_ring_valid`.
- `step_preserves_kernel_context`, `trace_preserves_kernel_context`, `user_fault_isolates_sibling_tasks`, `user_fault_marks_faulting_task`, and `all_five_isas_illegal_insn_isolation`.

## Verification Gates

1. **Runtime & Isolation Suite (`tests/kernel/k1_boot_traps_test.cpp`)**: Exercises all 4 boot handoffs, 5-ISA negative/positive instruction controls, trap vector isolation across all 5 ISAs, 1,600+ multi-CPU timer IRQs, ring overflow drop safety, canary corruption fail-closed, and strict runtime `heap_allocs == 0` via global `operator new` instrumentation.
2. **Freestanding 5-Target Object & Symbol Audit**: Compiles `lib/kernel/k1_boot_traps.c` and all five `lib/kernel/arch/k1_arch_*.c` units with `-ffreestanding -fno-builtin -fno-stack-protector -O2 -Wall -Wextra -Werror` for `x86_64-unknown-none-elf`, `aarch64-unknown-none-elf`, `riscv64-unknown-none-elf`, `riscv32-unknown-none-elf`, and `thumbv7em-none-eabi`, verifying zero undefined hosted, allocator, libc, or unwind symbols via `llvm-nm -u`.
3. **Lean 4.12.0 Proof Replay**: `lean proofs/lean/EshkolKernelK1.lean` exits `0` with zero warnings and zero axioms.
