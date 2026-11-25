/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K1: Verified Zero-Alloc Microkernel Early Boot,
 * 5-ISA Trap Vector Dispatch, and Per-CPU Static IRQ Ring Buffers.
 *
 * Pure freestanding C11 implementation:
 *   - ZERO dynamic allocation (no malloc/calloc/realloc/free).
 *   - ZERO libc, POSIX, unwind, or LLVM dependencies.
 *   - Strictly static caller-supplied buffers with bounded fuel termination.
 */

#include <eshkol/kernel/k1_boot_traps.h>
#include "arch/k1_arch_internal.h"

/* Compiler-barriered freestanding byte loops prevent -O2 loop idiom
 * lowering into external memset/memcpy symbols across all 5 target ISAs. */
static void k1_zero_bytes(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; ++i) {
        p[i] = 0u;
    }
}

static void k1_copy_bytes(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0u; i < n; ++i) {
        d[i] = s[i];
    }
}

static uint32_t k1_read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24u) |
           ((uint32_t)p[1] << 16u) |
           ((uint32_t)p[2] << 8u) |
           ((uint32_t)p[3]);
}

static uint32_t k1_read_le32(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

const k1_arch_vector_desc_t* k1_arch_descriptor(k1_isa_t isa) {
    switch (isa) {
        case K1_ISA_X86_64:
            return &k1_arch_desc_x86_64;
        case K1_ISA_AARCH64:
            return &k1_arch_desc_aarch64;
        case K1_ISA_RISCV64:
            return &k1_arch_desc_riscv64;
        case K1_ISA_RISCV32:
            return &k1_arch_desc_riscv32;
        case K1_ISA_CORTEX_M4:
            return &k1_arch_desc_cortex_m4;
        default:
            return (const k1_arch_vector_desc_t*)0;
    }
}

static int k1_validate_dtb_payload(const uint8_t* p, size_t len) {
    if (len < K1_DTB_HEADER_BYTES) {
        return 0;
    }
    uint32_t magic = k1_read_be32(p + 0u);
    if (magic != K1_DTB_MAGIC) {
        return 0;
    }
    uint32_t totalsize = k1_read_be32(p + 4u);
    uint32_t off_dt_struct = k1_read_be32(p + 8u);
    uint32_t off_dt_strings = k1_read_be32(p + 12u);
    uint32_t off_mem_rsvmap = k1_read_be32(p + 16u);
    uint32_t version = k1_read_be32(p + 20u);
    uint32_t last_comp_version = k1_read_be32(p + 24u);
    uint32_t size_dt_strings = k1_read_be32(p + 32u);
    uint32_t size_dt_struct = k1_read_be32(p + 36u);

    if (totalsize < K1_DTB_HEADER_BYTES || (size_t)totalsize > len) {
        return 0;
    }
    if (version < 16u || last_comp_version > version) {
        return 0;
    }
    if (off_dt_struct < K1_DTB_HEADER_BYTES || off_dt_struct > totalsize) {
        return 0;
    }
    if (off_dt_strings < K1_DTB_HEADER_BYTES || off_dt_strings > totalsize) {
        return 0;
    }
    if (off_mem_rsvmap < K1_DTB_HEADER_BYTES || off_mem_rsvmap > totalsize) {
        return 0;
    }
    if (size_dt_strings > totalsize - off_dt_strings) {
        return 0;
    }
    if (size_dt_struct > totalsize - off_dt_struct) {
        return 0;
    }
    return 1;
}

static int k1_validate_acpi_rsdp_payload(const uint8_t* p, size_t len) {
    static const uint8_t k_sig[8] = {'R', 'S', 'D', ' ', 'P', 'T', 'R', ' '};
    if (len < K1_ACPI_RSDP_V1_BYTES) {
        return 0;
    }
    for (size_t i = 0u; i < 8u; ++i) {
        if (p[i] != k_sig[i]) {
            return 0;
        }
    }

    uint8_t sum_v1 = 0u;
    for (size_t i = 0u; i < K1_ACPI_RSDP_V1_BYTES; ++i) {
        sum_v1 = (uint8_t)(sum_v1 + p[i]);
    }
    if (sum_v1 != 0u) {
        return 0;
    }

    uint8_t revision = p[15u];
    if (revision >= 2u) {
        if (len < K1_ACPI_RSDP_V2_BYTES) {
            return 0;
        }
        uint32_t ext_len = k1_read_le32(p + 20u);
        if (ext_len < K1_ACPI_RSDP_V2_BYTES || (size_t)ext_len > len) {
            return 0;
        }
        uint8_t sum_v2 = 0u;
        for (size_t i = 0u; i < (size_t)ext_len; ++i) {
            sum_v2 = (uint8_t)(sum_v2 + p[i]);
        }
        if (sum_v2 != 0u) {
            return 0;
        }
    }
    return 1;
}

static int k1_validate_apple_adt_payload(const uint8_t* p, size_t len) {
    if (len < (K1_APPLE_ADT_HDR_BYTES + K1_APPLE_ADT_PROP_NAME + 4u)) {
        return 0;
    }
    uint32_t n_properties = k1_read_le32(p + 0u);
    uint32_t n_children = k1_read_le32(p + 4u);
    if (n_properties == 0u || n_properties > 256u || n_children > 128u) {
        return 0;
    }

    /* Validate first property header: non-empty NUL-terminated 32B name + in-bounds length. */
    const uint8_t* prop0 = p + K1_APPLE_ADT_HDR_BYTES;
    if (prop0[0u] == 0u) {
        return 0;
    }
    int has_nul = 0;
    for (size_t i = 0u; i < K1_APPLE_ADT_PROP_NAME; ++i) {
        if (prop0[i] == 0u) {
            has_nul = 1;
            break;
        }
    }
    if (!has_nul) {
        return 0;
    }

    uint32_t prop0_len = k1_read_le32(prop0 + K1_APPLE_ADT_PROP_NAME) & 0x7FFFFFFFu;
    size_t remaining = len - (K1_APPLE_ADT_HDR_BYTES + K1_APPLE_ADT_PROP_NAME + 4u);
    if ((size_t)prop0_len > remaining) {
        return 0;
    }
    return 1;
}

static int k1_validate_mcu_vectors_payload(k1_isa_t isa, const uint8_t* p, size_t len) {
    if (isa == K1_ISA_CORTEX_M4) {
        if (len < K1_MCU_VECTOR_MIN_BYTES) {
            return 0;
        }
        uint32_t initial_sp = k1_read_le32(p + 0u);
        uint32_t reset_vec = k1_read_le32(p + 4u);
        if (initial_sp == 0u || (initial_sp & 0x7u) != 0u) {
            return 0;
        }
        if (reset_vec == 0u || (reset_vec & 0x1u) == 0u) {
            return 0;
        }
        return 1;
    }
    if (isa == K1_ISA_RISCV32) {
        if (len < 16u) {
            return 0;
        }
        uint32_t entry_word = k1_read_le32(p + 0u);
        uint32_t mtvec_word = k1_read_le32(p + 4u);
        if (entry_word == 0u || entry_word == 0xFFFFFFFFu) {
            return 0;
        }
        if (mtvec_word == 0u || (mtvec_word & 0x3u) > 1u) {
            return 0;
        }
        return 1;
    }
    return 0;
}

k1_status_t k1_validate_boot_info(const k1_boot_info_t* info) {
    if (!info) {
        return K1_ERR_INVALID_ARG;
    }
    if (k1_arch_descriptor(info->isa) == (const k1_arch_vector_desc_t*)0) {
        return K1_ERR_UNSUPPORTED_ISA;
    }
    if (info->cpu_count == 0u || info->cpu_count > K1_MAX_CPUS) {
        return K1_ERR_CPU_OUT_OF_RANGE;
    }
    if (info->boot_cpu_id >= info->cpu_count) {
        return K1_ERR_CPU_OUT_OF_RANGE;
    }
    if (info->ram_size == 0ULL || info->ram_base > (UINT64_MAX - info->ram_size)) {
        return K1_ERR_BAD_BOOT_HANDOFF;
    }
    if (!info->handoff_payload || info->handoff_size == 0u) {
        return K1_ERR_BAD_BOOT_HANDOFF;
    }

    int ok = 0;
    switch (info->boot_kind) {
        case K1_BOOT_DTB:
            ok = k1_validate_dtb_payload(info->handoff_payload, info->handoff_size);
            break;
        case K1_BOOT_ACPI_RSDP:
            ok = k1_validate_acpi_rsdp_payload(info->handoff_payload, info->handoff_size);
            break;
        case K1_BOOT_APPLE_ADT:
            ok = k1_validate_apple_adt_payload(info->handoff_payload, info->handoff_size);
            break;
        case K1_BOOT_MCU_VECTORS:
            ok = k1_validate_mcu_vectors_payload(info->isa, info->handoff_payload, info->handoff_size);
            break;
        default:
            ok = 0;
            break;
    }

    return ok ? K1_OK : K1_ERR_BAD_BOOT_HANDOFF;
}

void k1_kernel_context_init(k1_kernel_context_t* ctx,
                            uint64_t kernel_sp,
                            uint64_t kernel_pc,
                            uint64_t status_reg) {
    if (!ctx) {
        return;
    }
    k1_zero_bytes(ctx, sizeof(*ctx));
    ctx->stack_canary_lo = K1_STACK_CANARY_LO;
    ctx->kernel_sp = kernel_sp;
    ctx->kernel_pc = kernel_pc;
    ctx->status_reg = status_reg;
    ctx->stack_canary_hi = K1_STACK_CANARY_HI;
}

int k1_kernel_context_check_canaries(const k1_kernel_context_t* ctx) {
    if (!ctx) {
        return 0;
    }
    return (ctx->stack_canary_lo == K1_STACK_CANARY_LO) &&
           (ctx->stack_canary_hi == K1_STACK_CANARY_HI);
}

void k1_irq_ring_init(k1_irq_ring_t* ring) {
    if (!ring) {
        return;
    }
    k1_zero_bytes(ring, sizeof(*ring));
    ring->capacity = K1_IRQ_RING_CAPACITY;
}

int k1_irq_ring_invariants_hold(const k1_irq_ring_t* ring) {
    if (!ring) {
        return 0;
    }
    if (ring->capacity != K1_IRQ_RING_CAPACITY) {
        return 0;
    }
    if (ring->head >= ring->capacity || ring->tail >= ring->capacity) {
        return 0;
    }
    if (ring->count > ring->capacity) {
        return 0;
    }
    uint32_t expected_tail = (ring->head + ring->count) & (K1_IRQ_RING_CAPACITY - 1u);
    if (ring->count < ring->capacity && ring->tail != expected_tail) {
        return 0;
    }
    if (ring->count == ring->capacity && ring->tail != ring->head) {
        return 0;
    }
    return 1;
}

k1_status_t k1_irq_ring_push(k1_irq_ring_t* ring, const k1_irq_event_t* ev) {
    if (!ring || !ev || !k1_irq_ring_invariants_hold(ring)) {
        return K1_ERR_INVALID_ARG;
    }
    if (ring->count == ring->capacity) {
        ring->dropped_total += 1ULL;
        return K1_ERR_RING_FULL;
    }
    ring->entries[ring->tail] = *ev;
    ring->tail = (ring->tail + 1u) & (K1_IRQ_RING_CAPACITY - 1u);
    ring->count += 1u;
    ring->enqueued_total += 1ULL;
    return K1_OK;
}

k1_status_t k1_irq_ring_pop(k1_irq_ring_t* ring, k1_irq_event_t* out_ev) {
    if (!ring || !out_ev || !k1_irq_ring_invariants_hold(ring)) {
        return K1_ERR_INVALID_ARG;
    }
    if (ring->count == 0u) {
        return K1_ERR_RING_EMPTY;
    }
    *out_ev = ring->entries[ring->head];
    ring->head = (ring->head + 1u) & (K1_IRQ_RING_CAPACITY - 1u);
    ring->count -= 1u;
    ring->drained_total += 1ULL;
    return K1_OK;
}

k1_status_t k1_kernel_boot(k1_kernel_state_t* state,
                           const k1_boot_info_t* info,
                           const k1_kernel_context_t* initial_kernel_ctx) {
    if (!state || !info || !initial_kernel_ctx) {
        return K1_ERR_INVALID_ARG;
    }
    k1_status_t boot_st = k1_validate_boot_info(info);
    if (boot_st != K1_OK) {
        return boot_st;
    }
    if (!k1_kernel_context_check_canaries(initial_kernel_ctx)) {
        return K1_ERR_CANARY_CORRUPT;
    }

    k1_zero_bytes(state, sizeof(*state));
    state->booted = 1u;
    state->isa = info->isa;
    state->boot_kind = info->boot_kind;
    state->cpu_count = info->cpu_count;
    state->boot_cpu_id = info->boot_cpu_id;
    state->trap_fuel_limit = K1_DEFAULT_TRAP_FUEL;
    state->ram_base = info->ram_base;
    state->ram_size = info->ram_size;
    state->console_mmio_base = info->console_mmio_base;

    k1_copy_bytes(&state->kernel_ctx, initial_kernel_ctx, sizeof(state->kernel_ctx));

    for (uint32_t cpu = 0u; cpu < K1_MAX_CPUS; ++cpu) {
        k1_irq_ring_init(&state->cpu_irq_rings[cpu]);
    }
    for (uint32_t tid = 0u; tid < K1_MAX_USER_TASKS; ++tid) {
        state->tasks[tid].task_id = tid;
        state->tasks[tid].state = K1_TASK_UNUSED;
    }

    return K1_OK;
}

k1_status_t k1_user_task_register(k1_kernel_state_t* state,
                                  uint32_t task_id,
                                  uint64_t entry_pc,
                                  uint64_t user_sp) {
    if (!state) {
        return K1_ERR_INVALID_ARG;
    }
    if (!state->booted) {
        return K1_ERR_NOT_BOOTED;
    }
    if (task_id >= K1_MAX_USER_TASKS) {
        return K1_ERR_TASK_OUT_OF_RANGE;
    }
    k1_user_task_t* task = &state->tasks[task_id];
    k1_zero_bytes(task, sizeof(*task));
    task->task_id = task_id;
    task->state = K1_TASK_RUNNABLE;
    task->user_pc = entry_pc;
    task->user_sp = user_sp;
    task->last_trap = K1_TRAP_NONE;
    return K1_OK;
}

k1_trap_code_t k1_decode_raw_cause(k1_isa_t isa, uint64_t raw_cause) {
    switch (isa) {
        case K1_ISA_X86_64:
            return k1_arch_decode_cause_x86_64(raw_cause);
        case K1_ISA_AARCH64:
            return k1_arch_decode_cause_aarch64(raw_cause);
        case K1_ISA_RISCV64:
            return k1_arch_decode_cause_riscv64(raw_cause);
        case K1_ISA_RISCV32:
            return k1_arch_decode_cause_riscv32(raw_cause);
        case K1_ISA_CORTEX_M4:
            return k1_arch_decode_cause_cortex_m4(raw_cause);
        default:
            return K1_TRAP_UNKNOWN;
    }
}

k1_status_t k1_decode_isa_instruction(k1_isa_t isa,
                                      const uint8_t* insn,
                                      size_t len,
                                      k1_insn_decode_result_t* out_res) {
    if (!insn || !out_res) {
        return K1_ERR_INVALID_ARG;
    }
    switch (isa) {
        case K1_ISA_X86_64:
            k1_arch_decode_insn_x86_64(insn, len, out_res);
            return K1_OK;
        case K1_ISA_AARCH64:
            k1_arch_decode_insn_aarch64(insn, len, out_res);
            return K1_OK;
        case K1_ISA_RISCV64:
            k1_arch_decode_insn_riscv64(insn, len, out_res);
            return K1_OK;
        case K1_ISA_RISCV32:
            k1_arch_decode_insn_riscv32(insn, len, out_res);
            return K1_OK;
        case K1_ISA_CORTEX_M4:
            k1_arch_decode_insn_cortex_m4(insn, len, out_res);
            return K1_OK;
        default:
            return K1_ERR_UNSUPPORTED_ISA;
    }
}

k1_status_t k1_dispatch_timer_irq(k1_kernel_state_t* state,
                                  uint32_t cpu_id,
                                  uint64_t tick,
                                  uint32_t hw_irq) {
    if (!state) {
        return K1_ERR_INVALID_ARG;
    }
    if (!state->booted) {
        return K1_ERR_NOT_BOOTED;
    }
    if (!k1_kernel_context_check_canaries(&state->kernel_ctx)) {
        return K1_ERR_CANARY_CORRUPT;
    }
    if (cpu_id >= state->cpu_count || cpu_id >= K1_MAX_CPUS) {
        return K1_ERR_CPU_OUT_OF_RANGE;
    }

    k1_irq_event_t ev;
    ev.tick = tick;
    ev.cpu_id = cpu_id;
    ev.hw_irq = hw_irq;
    ev.trap_code = (uint32_t)K1_TRAP_TIMER_IRQ;
    ev.flags = 0u;

    state->total_irqs_handled += 1ULL;
    return k1_irq_ring_push(&state->cpu_irq_rings[cpu_id], &ev);
}

size_t k1_drain_cpu_irqs(k1_kernel_state_t* state,
                         uint32_t cpu_id,
                         k1_irq_event_t* out_buf,
                         size_t max_events) {
    if (!state || !state->booted || !out_buf || max_events == 0u) {
        return 0u;
    }
    if (cpu_id >= state->cpu_count || cpu_id >= K1_MAX_CPUS) {
        return 0u;
    }

    size_t drained = 0u;
    k1_irq_ring_t* ring = &state->cpu_irq_rings[cpu_id];
    while (drained < max_events) {
        k1_irq_event_t ev;
        if (k1_irq_ring_pop(ring, &ev) != K1_OK) {
            break;
        }
        out_buf[drained] = ev;
        drained += 1u;
    }
    return drained;
}

k1_trap_outcome_t k1_dispatch_user_trap(k1_kernel_state_t* state,
                                        const k1_trap_frame_t* frame) {
    k1_trap_outcome_t out;
    k1_zero_bytes(&out, sizeof(out));

    if (!state || !frame) {
        out.status = K1_ERR_INVALID_ARG;
        return out;
    }
    if (!state->booted) {
        out.status = K1_ERR_NOT_BOOTED;
        return out;
    }
    if (!k1_kernel_context_check_canaries(&state->kernel_ctx)) {
        out.status = K1_ERR_CANARY_CORRUPT;
        out.kernel_intact = 0u;
        return out;
    }

    uint32_t fuel = state->trap_fuel_limit;
    if (fuel == 0u) {
        out.status = K1_ERR_FUEL_EXHAUSTED;
        out.kernel_intact = 1u;
        return out;
    }

    if (frame->isa != state->isa) {
        out.status = K1_ERR_UNSUPPORTED_ISA;
        out.kernel_intact = 1u;
        out.fuel_remaining = fuel;
        return out;
    }
    if (frame->cpu_id >= state->cpu_count || frame->cpu_id >= K1_MAX_CPUS) {
        out.status = K1_ERR_CPU_OUT_OF_RANGE;
        out.kernel_intact = 1u;
        out.fuel_remaining = fuel;
        return out;
    }
    if (frame->origin_mode != K1_MODE_USER) {
        out.status = K1_ERR_INVALID_ARG;
        out.kernel_intact = 1u;
        out.fuel_remaining = fuel;
        return out;
    }
    if (frame->task_id >= K1_MAX_USER_TASKS) {
        out.status = K1_ERR_TASK_OUT_OF_RANGE;
        out.kernel_intact = 1u;
        out.fuel_remaining = fuel;
        return out;
    }

    k1_user_task_t* task = &state->tasks[frame->task_id];
    if (task->state != K1_TASK_RUNNABLE) {
        out.status = K1_ERR_TASK_NOT_ACTIVE;
        out.affected_task_id = frame->task_id;
        out.kernel_intact = 1u;
        out.fuel_remaining = fuel;
        return out;
    }

    /* Step 1: Decode instruction bytes (if present) and raw hardware cause. */
    fuel -= 1u;
    k1_trap_code_t resolved_trap = K1_TRAP_UNKNOWN;
    uint64_t resolved_cause = frame->raw_cause;

    if (frame->insn_len > 0u) {
        size_t bounded_len = (frame->insn_len <= 8u) ? (size_t)frame->insn_len : 8u;
        k1_insn_decode_result_t dec;
        if (k1_decode_isa_instruction(frame->isa, frame->insn_bytes, bounded_len, &dec) == K1_OK &&
            dec.trap_code == K1_TRAP_ILLEGAL_INSN) {
            resolved_trap = K1_TRAP_ILLEGAL_INSN;
            resolved_cause = dec.native_cause;
        }
    }

    if (resolved_trap == K1_TRAP_UNKNOWN) {
        resolved_trap = k1_decode_raw_cause(frame->isa, frame->raw_cause);
    }

    /* Step 2: Apply isolated user-task transition; kernel_ctx is read-only. */
    if (fuel > 0u) {
        fuel -= 1u;
    }

    state->total_user_traps += 1ULL;
    out.affected_task_id = frame->task_id;
    out.trap_code = resolved_trap;
    out.native_cause = resolved_cause;

    if (resolved_trap == K1_TRAP_SYSCALL) {
        task->last_trap = K1_TRAP_SYSCALL;
        task->last_raw_cause = resolved_cause;
        task->user_pc = frame->fault_pc;
        out.task_terminated = 0u;
    } else if (resolved_trap == K1_TRAP_TIMER_IRQ) {
        (void)k1_dispatch_timer_irq(state, frame->cpu_id, state->total_irqs_handled, (uint32_t)resolved_cause);
        task->last_trap = K1_TRAP_TIMER_IRQ;
        task->last_raw_cause = resolved_cause;
        out.task_terminated = 0u;
    } else {
        /* Fault isolation: terminate ONLY the faulting user task. */
        task->state = K1_TASK_FAULTED;
        task->user_pc = frame->fault_pc;
        task->fault_addr = frame->fault_addr;
        task->last_raw_cause = resolved_cause;
        task->last_trap = resolved_trap;
        task->fault_count += 1u;
        out.task_terminated = 1u;
    }

    out.kernel_intact = (uint32_t)k1_kernel_context_check_canaries(&state->kernel_ctx);
    out.fuel_remaining = fuel;
    out.status = out.kernel_intact ? K1_OK : K1_ERR_CANARY_CORRUPT;
    return out;
}
