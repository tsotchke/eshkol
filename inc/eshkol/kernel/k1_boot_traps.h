/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K1: Verified Zero-Alloc Microkernel Early Boot,
 * 5-ISA Trap Vectors, and Per-CPU Static IRQ Ring Buffers.
 *
 * Pure freestanding C11/C++17 ABI with zero heap allocation, zero libc
 * dependencies, and zero hosted runtime symbols across x86_64, AArch64,
 * RISC-V 64, RISC-V 32, and Cortex-M4.
 */

#ifndef ESHKOL_KERNEL_K1_BOOT_TRAPS_H
#define ESHKOL_KERNEL_K1_BOOT_TRAPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version and compile-time static capacities (zero dynamic allocation). */
#define K1_ABI_VERSION          1u
#define K1_MAX_CPUS             8u
#define K1_IRQ_RING_CAPACITY    256u
#define K1_MAX_USER_TASKS       16u
#define K1_KERNEL_GPRS          16u
#define K1_DEFAULT_TRAP_FUEL    1024u

/* Boot handoff magic constants and minimum structural header sizes. */
#define K1_DTB_MAGIC            0xD00DFEEDu
#define K1_DTB_HEADER_BYTES     40u
#define K1_ACPI_RSDP_V1_BYTES   20u
#define K1_ACPI_RSDP_V2_BYTES   36u
#define K1_APPLE_ADT_HDR_BYTES  8u
#define K1_APPLE_ADT_PROP_NAME  32u
#define K1_MCU_VECTOR_MIN_BYTES 64u

/* Kernel context integrity guard canaries. */
#define K1_STACK_CANARY_LO      0x5A3C96E1D2B4780FULL
#define K1_STACK_CANARY_HI      0xC3A5691E2D4B87F0ULL

/* Per-ISA canonical hardware trap causes / vectors. */
/* x86_64 IDT vectors */
#define K1_X86_64_VEC_DIV_ZERO      0ULL
#define K1_X86_64_VEC_ILLEGAL_INSN  6ULL
#define K1_X86_64_VEC_GP_FAULT      13ULL
#define K1_X86_64_VEC_PAGE_FAULT    14ULL
#define K1_X86_64_VEC_TIMER_IRQ     32ULL
#define K1_X86_64_VEC_SYSCALL       128ULL

/* AArch64 ESR_EL1 exception class (EC) codes and generic timer PPI */
#define K1_AARCH64_EC_ILLEGAL_INSN  0x00ULL
#define K1_AARCH64_EC_SVC64         0x15ULL
#define K1_AARCH64_EC_IABT_EL0      0x20ULL
#define K1_AARCH64_EC_IABT_EL1      0x21ULL
#define K1_AARCH64_EC_DABT_EL0      0x24ULL
#define K1_AARCH64_EC_DABT_EL1      0x25ULL
#define K1_AARCH64_EC_BRK_DIV_ZERO  0x3CULL
#define K1_AARCH64_IRQ_TIMER_PPI    0x1000001EULL

/* RISC-V 64/32 mcause / scause codes */
#define K1_RISCV_CAUSE_INSN_ACCESS  1ULL
#define K1_RISCV_CAUSE_ILLEGAL_INSN 2ULL
#define K1_RISCV_CAUSE_BREAK_DIV0   3ULL
#define K1_RISCV_CAUSE_LOAD_ACCESS  5ULL
#define K1_RISCV_CAUSE_STORE_ACCESS 7ULL
#define K1_RISCV_CAUSE_ECALL_U      8ULL
#define K1_RISCV_CAUSE_INSN_PAGE    12ULL
#define K1_RISCV_CAUSE_LOAD_PAGE    13ULL
#define K1_RISCV_CAUSE_STORE_PAGE   15ULL
#define K1_RISCV64_IRQ_STIMER       (0x8000000000000000ULL | 5ULL)
#define K1_RISCV64_IRQ_MTIMER       (0x8000000000000000ULL | 7ULL)
#define K1_RISCV32_IRQ_STIMER       (0x80000000ULL | 5ULL)
#define K1_RISCV32_IRQ_MTIMER       (0x80000000ULL | 7ULL)

/* Cortex-M4 exception number (high 16 bits) + UsageFault UFSR mask (low 16 bits) */
#define K1_CORTEX_M4_CAUSE_MEMMANAGE  0x00040000ULL
#define K1_CORTEX_M4_CAUSE_BUSFAULT   0x00050000ULL
#define K1_CORTEX_M4_CAUSE_UNDEFINSTR 0x00060001ULL
#define K1_CORTEX_M4_CAUSE_DIVBYZERO  0x00060200ULL
#define K1_CORTEX_M4_CAUSE_SVCALL     0x000B0000ULL
#define K1_CORTEX_M4_CAUSE_SYSTICK    0x000F0000ULL

typedef enum k1_isa {
    K1_ISA_X86_64    = 0,
    K1_ISA_AARCH64   = 1,
    K1_ISA_RISCV64   = 2,
    K1_ISA_RISCV32   = 3,
    K1_ISA_CORTEX_M4 = 4,
    K1_ISA_COUNT     = 5
} k1_isa_t;

typedef enum k1_boot_kind {
    K1_BOOT_NONE        = 0,
    K1_BOOT_DTB         = 1,
    K1_BOOT_ACPI_RSDP   = 2,
    K1_BOOT_APPLE_ADT   = 3,
    K1_BOOT_MCU_VECTORS = 4
} k1_boot_kind_t;

typedef enum k1_status {
    K1_OK                    = 0,
    K1_ERR_INVALID_ARG       = 1,
    K1_ERR_UNSUPPORTED_ISA   = 2,
    K1_ERR_BAD_BOOT_HANDOFF  = 3,
    K1_ERR_CPU_OUT_OF_RANGE  = 4,
    K1_ERR_RING_FULL         = 5,
    K1_ERR_RING_EMPTY        = 6,
    K1_ERR_TASK_OUT_OF_RANGE = 7,
    K1_ERR_TASK_NOT_ACTIVE   = 8,
    K1_ERR_CANARY_CORRUPT    = 9,
    K1_ERR_FUEL_EXHAUSTED    = 10,
    K1_ERR_NOT_BOOTED        = 11
} k1_status_t;

typedef enum k1_trap_code {
    K1_TRAP_NONE         = 0,
    K1_TRAP_TIMER_IRQ    = 1,
    K1_TRAP_DIV_ZERO     = 2,
    K1_TRAP_ILLEGAL_INSN = 3,
    K1_TRAP_ACCESS_FAULT = 4,
    K1_TRAP_SYSCALL      = 5,
    K1_TRAP_UNKNOWN      = 6
} k1_trap_code_t;

typedef enum k1_priv_mode {
    K1_MODE_KERNEL = 0,
    K1_MODE_USER   = 1
} k1_priv_mode_t;

typedef enum k1_task_state {
    K1_TASK_UNUSED   = 0,
    K1_TASK_RUNNABLE = 1,
    K1_TASK_FAULTED  = 2,
    K1_TASK_EXITED   = 3
} k1_task_state_t;

typedef struct k1_boot_info {
    k1_isa_t isa;
    k1_boot_kind_t boot_kind;
    uint32_t cpu_count;
    uint32_t boot_cpu_id;
    uint64_t ram_base;
    uint64_t ram_size;
    uint64_t console_mmio_base;
    const uint8_t* handoff_payload;
    size_t handoff_size;
} k1_boot_info_t;

typedef struct k1_irq_event {
    uint64_t tick;
    uint32_t cpu_id;
    uint32_t hw_irq;
    uint32_t trap_code;
    uint32_t flags;
} k1_irq_event_t;

typedef struct k1_irq_ring {
    k1_irq_event_t entries[K1_IRQ_RING_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint32_t capacity;
    uint64_t enqueued_total;
    uint64_t drained_total;
    uint64_t dropped_total;
} k1_irq_ring_t;

typedef struct k1_kernel_context {
    uint64_t stack_canary_lo;
    uint64_t gpr[K1_KERNEL_GPRS];
    uint64_t kernel_sp;
    uint64_t kernel_pc;
    uint64_t status_reg;
    uint64_t stack_canary_hi;
} k1_kernel_context_t;

typedef struct k1_user_task {
    uint32_t task_id;
    k1_task_state_t state;
    uint64_t user_pc;
    uint64_t user_sp;
    uint64_t fault_addr;
    uint64_t last_raw_cause;
    k1_trap_code_t last_trap;
    uint32_t fault_count;
} k1_user_task_t;

typedef struct k1_trap_frame {
    k1_isa_t isa;
    k1_priv_mode_t origin_mode;
    uint32_t cpu_id;
    uint32_t task_id;
    uint64_t raw_cause;
    uint64_t fault_pc;
    uint64_t fault_addr;
    uint8_t insn_bytes[8];
    uint8_t insn_len;
    uint8_t reserved[7];
} k1_trap_frame_t;

typedef struct k1_insn_decode_result {
    k1_isa_t isa;
    k1_trap_code_t trap_code;
    uint64_t native_cause;
    uint8_t is_valid_insn;
    uint8_t is_nop;
    uint8_t consumed_len;
    uint8_t reserved;
} k1_insn_decode_result_t;

typedef struct k1_trap_outcome {
    k1_status_t status;
    k1_trap_code_t trap_code;
    uint64_t native_cause;
    uint32_t affected_task_id;
    uint32_t task_terminated;
    uint32_t kernel_intact;
    uint32_t fuel_remaining;
} k1_trap_outcome_t;

typedef struct k1_arch_vector_desc {
    k1_isa_t isa;
    const char* isa_name;
    const char* target_triple;
    uint32_t pointer_bits;
    uint32_t vector_align;
    uint64_t timer_irq_cause;
    uint64_t div_zero_cause;
    uint64_t illegal_insn_cause;
    uint64_t access_fault_cause;
    uint64_t syscall_cause;
    uint8_t known_bad_insn[4];
    uint8_t known_bad_len;
    uint8_t known_nop_insn[4];
    uint8_t known_nop_len;
} k1_arch_vector_desc_t;

typedef struct k1_kernel_state {
    uint32_t booted;
    k1_isa_t isa;
    k1_boot_kind_t boot_kind;
    uint32_t cpu_count;
    uint32_t boot_cpu_id;
    uint32_t trap_fuel_limit;
    uint64_t ram_base;
    uint64_t ram_size;
    uint64_t console_mmio_base;
    uint64_t total_irqs_handled;
    uint64_t total_user_traps;
    k1_kernel_context_t kernel_ctx;
    k1_irq_ring_t cpu_irq_rings[K1_MAX_CPUS];
    k1_user_task_t tasks[K1_MAX_USER_TASKS];
} k1_kernel_state_t;

/* Per-ISA static vector table descriptors. */
const k1_arch_vector_desc_t* k1_arch_descriptor(k1_isa_t isa);

/* Boot handoff validation and zero-alloc kernel initialization. */
k1_status_t k1_validate_boot_info(const k1_boot_info_t* info);
void k1_kernel_context_init(k1_kernel_context_t* ctx,
                            uint64_t kernel_sp,
                            uint64_t kernel_pc,
                            uint64_t status_reg);
int k1_kernel_context_check_canaries(const k1_kernel_context_t* ctx);
k1_status_t k1_kernel_boot(k1_kernel_state_t* state,
                           const k1_boot_info_t* info,
                           const k1_kernel_context_t* initial_kernel_ctx);

/* User task slot registration (static table, zero heap allocation). */
k1_status_t k1_user_task_register(k1_kernel_state_t* state,
                                  uint32_t task_id,
                                  uint64_t entry_pc,
                                  uint64_t user_sp);

/* Per-CPU static IRQ ring buffer operations. */
void k1_irq_ring_init(k1_irq_ring_t* ring);
k1_status_t k1_irq_ring_push(k1_irq_ring_t* ring, const k1_irq_event_t* ev);
k1_status_t k1_irq_ring_pop(k1_irq_ring_t* ring, k1_irq_event_t* out_ev);
int k1_irq_ring_invariants_hold(const k1_irq_ring_t* ring);

/* 5-ISA raw hardware trap cause and instruction decoders. */
k1_trap_code_t k1_decode_raw_cause(k1_isa_t isa, uint64_t raw_cause);
k1_status_t k1_decode_isa_instruction(k1_isa_t isa,
                                      const uint8_t* insn,
                                      size_t len,
                                      k1_insn_decode_result_t* out_res);

/* Timer IRQ enqueue/drain and isolated user-to-kernel trap dispatch. */
k1_status_t k1_dispatch_timer_irq(k1_kernel_state_t* state,
                                  uint32_t cpu_id,
                                  uint64_t tick,
                                  uint32_t hw_irq);
size_t k1_drain_cpu_irqs(k1_kernel_state_t* state,
                         uint32_t cpu_id,
                         k1_irq_event_t* out_buf,
                         size_t max_events);
k1_trap_outcome_t k1_dispatch_user_trap(k1_kernel_state_t* state,
                                        const k1_trap_frame_t* frame);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_KERNEL_K1_BOOT_TRAPS_H */
