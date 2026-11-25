/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K1 Verification Suite:
 *   1. Boot handoff validation (DTB 0xD00DFEED, ACPI RSDP v1/v2, Apple ADT,
 *      Cortex-M4 and RV32 MCU vector tables, plus strict negative controls).
 *   2. Trap Vector Isolation: live kernel context (16 GPRs + SP/PC/status +
 *      dual stack canaries) + sibling user task under div-zero, illegal
 *      instruction, and access fault across all 5 ISAs.
 *   3. Zero-Alloc Runtime Verification: global operator new hook asserts
 *      strictly 0 heap allocations across full boot + 1,000+ multi-CPU timer
 *      IRQs + ring overflow + drain + all user fault traps.
 *   4. 5-ISA Known-Bad Instruction Negative Controls & Positive NOP Controls.
 *   5. Per-CPU Static IRQ Ring Buffer bounds, overflow drop safety, FIFO
 *      ordering, and bounded trap fuel termination.
 */

#include <eshkol/kernel/k1_boot_traps.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

std::atomic<uint64_t> g_heap_alloc_count{0};
std::atomic<bool> g_track_allocs{false};

void write_be32(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>((v >> 24u) & 0xFFu);
    dst[1] = static_cast<uint8_t>((v >> 16u) & 0xFFu);
    dst[2] = static_cast<uint8_t>((v >> 8u) & 0xFFu);
    dst[3] = static_cast<uint8_t>(v & 0xFFu);
}

void write_le32(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xFFu);
    dst[1] = static_cast<uint8_t>((v >> 8u) & 0xFFu);
    dst[2] = static_cast<uint8_t>((v >> 16u) & 0xFFu);
    dst[3] = static_cast<uint8_t>((v >> 24u) & 0xFFu);
}

void build_valid_dtb(uint8_t buf[64]) {
    std::memset(buf, 0, 64);
    write_be32(buf + 0, K1_DTB_MAGIC);
    write_be32(buf + 4, 64u);   /* totalsize */
    write_be32(buf + 8, 48u);   /* off_dt_struct */
    write_be32(buf + 12, 56u);  /* off_dt_strings */
    write_be32(buf + 16, 40u);  /* off_mem_rsvmap */
    write_be32(buf + 20, 17u);  /* version */
    write_be32(buf + 24, 16u);  /* last_comp_version */
    write_be32(buf + 28, 0u);   /* boot_cpuid_phys */
    write_be32(buf + 32, 8u);   /* size_dt_strings */
    write_be32(buf + 36, 8u);   /* size_dt_struct */
}

void build_valid_acpi_rsdp_v2(uint8_t buf[36]) {
    std::memset(buf, 0, 36);
    static const char k_sig[8] = {'R', 'S', 'D', ' ', 'P', 'T', 'R', ' '};
    std::memcpy(buf, k_sig, 8);
    std::memcpy(buf + 9, "ESHKOL", 6);
    buf[15] = 2u; /* ACPI revision 2.0 */
    write_le32(buf + 20, 36u); /* extended length */

    /* Compute v1 checksum over bytes [0..19] */
    uint8_t sum_v1 = 0u;
    for (size_t i = 0; i < 20u; ++i) {
        if (i != 8u) sum_v1 = static_cast<uint8_t>(sum_v1 + buf[i]);
    }
    buf[8] = static_cast<uint8_t>(0u - sum_v1);

    /* Compute v2 extended checksum over bytes [0..35] */
    uint8_t sum_v2 = 0u;
    for (size_t i = 0; i < 36u; ++i) {
        if (i != 32u) sum_v2 = static_cast<uint8_t>(sum_v2 + buf[i]);
    }
    buf[32] = static_cast<uint8_t>(0u - sum_v2);
}

void build_valid_apple_adt(uint8_t buf[64]) {
    std::memset(buf, 0, 64);
    write_le32(buf + 0, 1u); /* n_properties */
    write_le32(buf + 4, 2u); /* n_children */
    std::memcpy(buf + 8, "compatible", 11);
    write_le32(buf + 8 + 32, 12u); /* prop0_len */
    std::memcpy(buf + 8 + 36, "apple,arm64", 12);
}

void build_valid_cortex_m4_vectors(uint8_t buf[64]) {
    std::memset(buf, 0, 64);
    write_le32(buf + 0, 0x20010000u); /* 8-byte aligned SRAM top */
    write_le32(buf + 4, 0x00000101u); /* Reset handler with Thumb bit set */
}

void build_valid_rv32_vectors(uint8_t buf[16]) {
    std::memset(buf, 0, 16);
    write_le32(buf + 0, 0x00000013u); /* Valid entry instruction */
    write_le32(buf + 4, 0x80000100u); /* 4-byte aligned direct mtvec */
}

k1_kernel_context_t make_canary_context() {
    k1_kernel_context_t ctx;
    k1_kernel_context_init(&ctx, 0xFFFF800000080000ULL, 0xFFFF800000001234ULL, 0x0000000000000246ULL);
    for (uint32_t i = 0; i < K1_KERNEL_GPRS; ++i) {
        ctx.gpr[i] = 0xA5A5000000000000ULL | (static_cast<uint64_t>(i + 1u) * 0x0123456789ABCDEFULL);
    }
    return ctx;
}

int g_checks_passed = 0;
int g_checks_failed = 0;

void check_impl(bool cond, const char* name, int line) {
    if (cond) {
        g_checks_passed += 1;
    } else {
        g_checks_failed += 1;
        std::fprintf(stderr, "FAIL [line %d]: %s\n", line, name);
    }
}

#define CHECK(expr) check_impl((expr), #expr, __LINE__)

/* Test 1: Boot handoff validation (all 4 boot handoff kinds + negative controls). */
void test_boot_handoff_validation() {
    uint8_t dtb[64];
    uint8_t acpi[36];
    uint8_t adt[64];
    uint8_t m4_vec[64];
    uint8_t rv32_vec[16];

    build_valid_dtb(dtb);
    build_valid_acpi_rsdp_v2(acpi);
    build_valid_apple_adt(adt);
    build_valid_cortex_m4_vectors(m4_vec);
    build_valid_rv32_vectors(rv32_vec);

    k1_boot_info_t info{};
    info.cpu_count = 4u;
    info.boot_cpu_id = 0u;
    info.ram_base = 0x80000000ULL;
    info.ram_size = 0x10000000ULL;
    info.console_mmio_base = 0x10000000ULL;

    /* Valid DTB on RISC-V 64 */
    info.isa = K1_ISA_RISCV64;
    info.boot_kind = K1_BOOT_DTB;
    info.handoff_payload = dtb;
    info.handoff_size = sizeof(dtb);
    CHECK(k1_validate_boot_info(&info) == K1_OK);

    /* Negative: corrupt DTB magic */
    dtb[0] ^= 0xFFu;
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
    dtb[0] ^= 0xFFu;

    /* Negative: truncated DTB size */
    info.handoff_size = 32u;
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
    info.handoff_size = sizeof(dtb);

    /* Valid ACPI RSDP v2 on x86_64 */
    info.isa = K1_ISA_X86_64;
    info.boot_kind = K1_BOOT_ACPI_RSDP;
    info.handoff_payload = acpi;
    info.handoff_size = sizeof(acpi);
    CHECK(k1_validate_boot_info(&info) == K1_OK);

    /* Negative: corrupt ACPI extended checksum */
    acpi[35] ^= 0x01u;
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
    acpi[35] ^= 0x01u;

    /* Valid Apple ADT on AArch64 */
    info.isa = K1_ISA_AARCH64;
    info.boot_kind = K1_BOOT_APPLE_ADT;
    info.handoff_payload = adt;
    info.handoff_size = sizeof(adt);
    CHECK(k1_validate_boot_info(&info) == K1_OK);

    /* Negative: ADT property length exceeds payload bounds */
    write_le32(adt + 8 + 32, 4096u);
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
    write_le32(adt + 8 + 32, 12u);

    /* Valid Cortex-M4 MCU vector table */
    info.isa = K1_ISA_CORTEX_M4;
    info.boot_kind = K1_BOOT_MCU_VECTORS;
    info.cpu_count = 1u;
    info.boot_cpu_id = 0u;
    info.handoff_payload = m4_vec;
    info.handoff_size = sizeof(m4_vec);
    CHECK(k1_validate_boot_info(&info) == K1_OK);

    /* Negative: Cortex-M4 reset vector missing Thumb bit */
    write_le32(m4_vec + 4, 0x00000100u);
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
    write_le32(m4_vec + 4, 0x00000101u);

    /* Valid RV32 MCU vector table */
    info.isa = K1_ISA_RISCV32;
    info.boot_kind = K1_BOOT_MCU_VECTORS;
    info.handoff_payload = rv32_vec;
    info.handoff_size = sizeof(rv32_vec);
    CHECK(k1_validate_boot_info(&info) == K1_OK);

    /* Negative: CPU count > K1_MAX_CPUS and RAM overflow */
    info.cpu_count = K1_MAX_CPUS + 1u;
    CHECK(k1_validate_boot_info(&info) == K1_ERR_CPU_OUT_OF_RANGE);
    info.cpu_count = 1u;
    info.ram_base = UINT64_MAX - 16ULL;
    info.ram_size = 64ULL;
    CHECK(k1_validate_boot_info(&info) == K1_ERR_BAD_BOOT_HANDOFF);
}

/* Test 2: 5-ISA Known-Bad Instruction Negative Controls & Positive NOP Controls. */
void test_five_isa_instruction_controls() {
    struct IsaControlCase {
        k1_isa_t isa;
        uint8_t bad_bytes[4];
        size_t bad_len;
        uint64_t expected_bad_cause;
        uint8_t nop_bytes[4];
        size_t nop_len;
    };

    static const IsaControlCase k_cases[5] = {
        {K1_ISA_X86_64,    {0x0Fu, 0x0Bu, 0x00u, 0x00u}, 2u, K1_X86_64_VEC_ILLEGAL_INSN,   {0x90u, 0x00u, 0x00u, 0x00u}, 1u},
        {K1_ISA_AARCH64,   {0x00u, 0x00u, 0x00u, 0x00u}, 4u, K1_AARCH64_EC_ILLEGAL_INSN,   {0x1Fu, 0x20u, 0x03u, 0xD5u}, 4u},
        {K1_ISA_RISCV64,   {0x00u, 0x00u, 0x00u, 0x00u}, 4u, K1_RISCV_CAUSE_ILLEGAL_INSN,  {0x13u, 0x00u, 0x00u, 0x00u}, 4u},
        {K1_ISA_RISCV32,   {0x00u, 0x00u, 0x00u, 0x00u}, 4u, K1_RISCV_CAUSE_ILLEGAL_INSN,  {0x13u, 0x00u, 0x00u, 0x00u}, 4u},
        {K1_ISA_CORTEX_M4, {0x00u, 0xDEu, 0x00u, 0x00u}, 2u, K1_CORTEX_M4_CAUSE_UNDEFINSTR,{0x00u, 0xBFu, 0x00u, 0x00u}, 2u},
    };

    for (size_t i = 0; i < 5u; ++i) {
        const IsaControlCase& c = k_cases[i];
        const k1_arch_vector_desc_t* desc = k1_arch_descriptor(c.isa);
        CHECK(desc != nullptr);
        CHECK(desc->isa == c.isa);
        CHECK(desc->illegal_insn_cause == c.expected_bad_cause);
        CHECK(desc->known_bad_len == c.bad_len);
        CHECK(std::memcmp(desc->known_bad_insn, c.bad_bytes, c.bad_len) == 0);
        CHECK(desc->known_nop_len == c.nop_len);
        CHECK(std::memcmp(desc->known_nop_insn, c.nop_bytes, c.nop_len) == 0);

        /* Negative control: known-bad instruction MUST decode to K1_TRAP_ILLEGAL_INSN with exact native cause */
        k1_insn_decode_result_t bad_res{};
        CHECK(k1_decode_isa_instruction(c.isa, c.bad_bytes, c.bad_len, &bad_res) == K1_OK);
        CHECK(bad_res.trap_code == K1_TRAP_ILLEGAL_INSN);
        CHECK(bad_res.native_cause == c.expected_bad_cause);
        CHECK(bad_res.is_valid_insn == 0u);
        CHECK(bad_res.is_nop == 0u);

        /* Positive control: valid NOP MUST decode cleanly and NOT trigger K1_TRAP_ILLEGAL_INSN */
        k1_insn_decode_result_t nop_res{};
        CHECK(k1_decode_isa_instruction(c.isa, c.nop_bytes, c.nop_len, &nop_res) == K1_OK);
        CHECK(nop_res.trap_code == K1_TRAP_NONE);
        CHECK(nop_res.is_valid_insn == 1u);
        CHECK(nop_res.is_nop == 1u);
    }
}

/* Test 3: Trap Vector Isolation + Non-Interference across ALL 5 ISAs. */
void test_trap_vector_isolation_all_five_isas() {
    uint8_t dtb[64];
    uint8_t acpi[36];
    uint8_t adt[64];
    uint8_t m4_vec[64];
    uint8_t rv32_vec[16];

    build_valid_dtb(dtb);
    build_valid_acpi_rsdp_v2(acpi);
    build_valid_apple_adt(adt);
    build_valid_cortex_m4_vectors(m4_vec);
    build_valid_rv32_vectors(rv32_vec);

    struct IsaBootSpec {
        k1_isa_t isa;
        k1_boot_kind_t kind;
        const uint8_t* payload;
        size_t payload_len;
        uint32_t cpus;
    };

    const IsaBootSpec specs[5] = {
        {K1_ISA_X86_64,    K1_BOOT_ACPI_RSDP,   acpi,     sizeof(acpi),     4u},
        {K1_ISA_AARCH64,   K1_BOOT_APPLE_ADT,   adt,      sizeof(adt),      4u},
        {K1_ISA_RISCV64,   K1_BOOT_DTB,         dtb,      sizeof(dtb),      4u},
        {K1_ISA_RISCV32,   K1_BOOT_MCU_VECTORS, rv32_vec, sizeof(rv32_vec), 1u},
        {K1_ISA_CORTEX_M4, K1_BOOT_MCU_VECTORS, m4_vec,   sizeof(m4_vec),   1u},
    };

    static k1_kernel_state_t state;

    for (size_t i = 0; i < 5u; ++i) {
        const IsaBootSpec& s = specs[i];
        const k1_arch_vector_desc_t* desc = k1_arch_descriptor(s.isa);
        CHECK(desc != nullptr);

        k1_boot_info_t info{};
        info.isa = s.isa;
        info.boot_kind = s.kind;
        info.cpu_count = s.cpus;
        info.boot_cpu_id = 0u;
        info.ram_base = 0x80000000ULL;
        info.ram_size = 0x04000000ULL;
        info.console_mmio_base = 0x10000000ULL;
        info.handoff_payload = s.payload;
        info.handoff_size = s.payload_len;

        k1_kernel_context_t baseline_ctx = make_canary_context();
        CHECK(k1_kernel_boot(&state, &info, &baseline_ctx) == K1_OK);

        /* Register three faulting tasks (0, 1, 2) + one healthy sibling task (3) */
        CHECK(k1_user_task_register(&state, 0u, 0x400000ULL, 0x7FFF0000ULL) == K1_OK);
        CHECK(k1_user_task_register(&state, 1u, 0x401000ULL, 0x7FFE0000ULL) == K1_OK);
        CHECK(k1_user_task_register(&state, 2u, 0x402000ULL, 0x7FFD0000ULL) == K1_OK);
        CHECK(k1_user_task_register(&state, 3u, 0x403000ULL, 0x7FFC0000ULL) == K1_OK);

        const k1_user_task_t sibling_snapshot = state.tasks[3u];

        /* Fault A: Divide-by-zero in task 0 */
        k1_trap_frame_t div_frame{};
        div_frame.isa = s.isa;
        div_frame.origin_mode = K1_MODE_USER;
        div_frame.cpu_id = 0u;
        div_frame.task_id = 0u;
        div_frame.raw_cause = desc->div_zero_cause;
        div_frame.fault_pc = 0x400020ULL;
        k1_trap_outcome_t out_div = k1_dispatch_user_trap(&state, &div_frame);
        CHECK(out_div.status == K1_OK);
        CHECK(out_div.trap_code == K1_TRAP_DIV_ZERO);
        CHECK(out_div.native_cause == desc->div_zero_cause);
        CHECK(out_div.task_terminated == 1u);
        CHECK(out_div.kernel_intact == 1u);
        CHECK(state.tasks[0u].state == K1_TASK_FAULTED);
        CHECK(state.tasks[0u].last_trap == K1_TRAP_DIV_ZERO);
        CHECK(std::memcmp(&state.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
        CHECK(std::memcmp(&state.tasks[3u], &sibling_snapshot, sizeof(sibling_snapshot)) == 0);

        /* Fault B: ISA-specific known-bad illegal instruction in task 1 */
        k1_trap_frame_t ill_frame{};
        ill_frame.isa = s.isa;
        ill_frame.origin_mode = K1_MODE_USER;
        ill_frame.cpu_id = 0u;
        ill_frame.task_id = 1u;
        ill_frame.raw_cause = desc->illegal_insn_cause;
        ill_frame.fault_pc = 0x401040ULL;
        std::memcpy(ill_frame.insn_bytes, desc->known_bad_insn, desc->known_bad_len);
        ill_frame.insn_len = desc->known_bad_len;
        k1_trap_outcome_t out_ill = k1_dispatch_user_trap(&state, &ill_frame);
        CHECK(out_ill.status == K1_OK);
        CHECK(out_ill.trap_code == K1_TRAP_ILLEGAL_INSN);
        CHECK(out_ill.native_cause == desc->illegal_insn_cause);
        CHECK(out_ill.task_terminated == 1u);
        CHECK(out_ill.kernel_intact == 1u);
        CHECK(state.tasks[1u].state == K1_TASK_FAULTED);
        CHECK(state.tasks[1u].last_trap == K1_TRAP_ILLEGAL_INSN);
        CHECK(std::memcmp(&state.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
        CHECK(std::memcmp(&state.tasks[3u], &sibling_snapshot, sizeof(sibling_snapshot)) == 0);

        /* Fault C: Access/protection fault in task 2 */
        k1_trap_frame_t acc_frame{};
        acc_frame.isa = s.isa;
        acc_frame.origin_mode = K1_MODE_USER;
        acc_frame.cpu_id = 0u;
        acc_frame.task_id = 2u;
        acc_frame.raw_cause = desc->access_fault_cause;
        acc_frame.fault_pc = 0x402080ULL;
        acc_frame.fault_addr = 0xDEAD000000000010ULL;
        k1_trap_outcome_t out_acc = k1_dispatch_user_trap(&state, &acc_frame);
        CHECK(out_acc.status == K1_OK);
        CHECK(out_acc.trap_code == K1_TRAP_ACCESS_FAULT);
        CHECK(out_acc.native_cause == desc->access_fault_cause);
        CHECK(out_acc.task_terminated == 1u);
        CHECK(out_acc.kernel_intact == 1u);
        CHECK(state.tasks[2u].state == K1_TASK_FAULTED);
        CHECK(state.tasks[2u].last_trap == K1_TRAP_ACCESS_FAULT);
        CHECK(state.tasks[2u].fault_addr == 0xDEAD000000000010ULL);

        /* Kernel context & canaries 100% bit-identical; healthy sibling task 3 still runnable */
        CHECK(k1_kernel_context_check_canaries(&state.kernel_ctx) == 1);
        CHECK(std::memcmp(&state.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
        CHECK(state.tasks[3u].state == K1_TASK_RUNNABLE);
        CHECK(std::memcmp(&state.tasks[3u], &sibling_snapshot, sizeof(sibling_snapshot)) == 0);

        /* Syscall from healthy sibling task 3 keeps task 3 runnable */
        k1_trap_frame_t sys_frame{};
        sys_frame.isa = s.isa;
        sys_frame.origin_mode = K1_MODE_USER;
        sys_frame.cpu_id = 0u;
        sys_frame.task_id = 3u;
        sys_frame.raw_cause = desc->syscall_cause;
        sys_frame.fault_pc = 0x403010ULL;
        k1_trap_outcome_t out_sys = k1_dispatch_user_trap(&state, &sys_frame);
        CHECK(out_sys.status == K1_OK);
        CHECK(out_sys.trap_code == K1_TRAP_SYSCALL);
        CHECK(out_sys.task_terminated == 0u);
        CHECK(state.tasks[3u].state == K1_TASK_RUNNABLE);
        CHECK(std::memcmp(&state.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
    }

    /* Negative control: corrupted kernel stack canary fails closed immediately */
    state.kernel_ctx.stack_canary_hi ^= 1ULL;
    k1_trap_frame_t canary_frame{};
    canary_frame.isa = state.isa;
    canary_frame.origin_mode = K1_MODE_USER;
    canary_frame.cpu_id = 0u;
    canary_frame.task_id = 3u;
    canary_frame.raw_cause = K1_CORTEX_M4_CAUSE_DIVBYZERO;
    k1_trap_outcome_t canary_out = k1_dispatch_user_trap(&state, &canary_frame);
    CHECK(canary_out.status == K1_ERR_CANARY_CORRUPT);
    CHECK(canary_out.kernel_intact == 0u);
    CHECK(k1_dispatch_timer_irq(&state, 0u, 1ULL, 15u) == K1_ERR_CANARY_CORRUPT);

    /* Negative control: zero trap fuel fails closed cleanly */
    state.kernel_ctx.stack_canary_hi ^= 1ULL;
    state.trap_fuel_limit = 0u;
    k1_trap_outcome_t fuel_out = k1_dispatch_user_trap(&state, &canary_frame);
    CHECK(fuel_out.status == K1_ERR_FUEL_EXHAUSTED);
    CHECK(fuel_out.kernel_intact == 1u);
}

/* Test 4: 1,000+ Multi-CPU Timer IRQs, Ring Overflow Safety, FIFO Drain, and Zero Heap Allocs. */
void test_timer_irqs_and_ring_overflow_zero_alloc() {
    uint8_t dtb[64];
    build_valid_dtb(dtb);

    k1_boot_info_t info{};
    info.isa = K1_ISA_RISCV64;
    info.boot_kind = K1_BOOT_DTB;
    info.cpu_count = K1_MAX_CPUS; /* All 8 CPUs */
    info.boot_cpu_id = 0u;
    info.ram_base = 0x80000000ULL;
    info.ram_size = 0x20000000ULL;
    info.console_mmio_base = 0x10000000ULL;
    info.handoff_payload = dtb;
    info.handoff_size = sizeof(dtb);

    static k1_kernel_state_t state;
    static k1_irq_event_t drain_buf[K1_IRQ_RING_CAPACITY];
    k1_kernel_context_t baseline_ctx = make_canary_context();

    CHECK(k1_kernel_boot(&state, &info, &baseline_ctx) == K1_OK);

    /* Fire 1,600 timer IRQs across 8 CPUs (200 per CPU), draining cleanly */
    for (uint32_t round = 0u; round < 200u; ++round) {
        for (uint32_t cpu = 0u; cpu < K1_MAX_CPUS; ++cpu) {
            uint64_t tick = static_cast<uint64_t>(round) * K1_MAX_CPUS + cpu;
            CHECK(k1_dispatch_timer_irq(&state, cpu, tick, 7u) == K1_OK);
            CHECK(k1_irq_ring_invariants_hold(&state.cpu_irq_rings[cpu]) == 1);
        }
    }
    CHECK(state.total_irqs_handled == 1600ULL);

    for (uint32_t cpu = 0u; cpu < K1_MAX_CPUS; ++cpu) {
        size_t n = k1_drain_cpu_irqs(&state, cpu, drain_buf, K1_IRQ_RING_CAPACITY);
        CHECK(n == 200u);
        CHECK(state.cpu_irq_rings[cpu].count == 0u);
        CHECK(state.cpu_irq_rings[cpu].dropped_total == 0ULL);
        CHECK(k1_irq_ring_invariants_hold(&state.cpu_irq_rings[cpu]) == 1);
        for (size_t k = 0u; k < n; ++k) {
            CHECK(drain_buf[k].cpu_id == cpu);
            CHECK(drain_buf[k].tick == static_cast<uint64_t>(k) * K1_MAX_CPUS + cpu);
            CHECK(drain_buf[k].trap_code == K1_TRAP_TIMER_IRQ);
        }
    }

    /* Ring overflow test on CPU 0: push 300 IRQs into 256-slot ring -> 44 cleanly dropped, live 256 intact */
    for (uint32_t i = 0u; i < 300u; ++i) {
        k1_status_t st = k1_dispatch_timer_irq(&state, 0u, 10000ULL + i, 7u);
        if (i < K1_IRQ_RING_CAPACITY) {
            CHECK(st == K1_OK);
        } else {
            CHECK(st == K1_ERR_RING_FULL);
        }
        CHECK(k1_irq_ring_invariants_hold(&state.cpu_irq_rings[0u]) == 1);
    }
    CHECK(state.cpu_irq_rings[0u].count == K1_IRQ_RING_CAPACITY);
    CHECK(state.cpu_irq_rings[0u].dropped_total == 44ULL);

    size_t drained = k1_drain_cpu_irqs(&state, 0u, drain_buf, K1_IRQ_RING_CAPACITY);
    CHECK(drained == K1_IRQ_RING_CAPACITY);
    for (size_t k = 0u; k < K1_IRQ_RING_CAPACITY; ++k) {
        CHECK(drain_buf[k].tick == 10000ULL + k);
    }
    CHECK(std::memcmp(&state.kernel_ctx, &baseline_ctx, sizeof(baseline_ctx)) == 0);
}

} /* namespace */

/* Runtime zero-allocation guard: intercept all C++ global operator new calls. */
void* operator new(std::size_t sz) {
    if (g_track_allocs.load(std::memory_order_relaxed)) {
        g_heap_alloc_count.fetch_add(1ULL, std::memory_order_relaxed);
    }
    void* p = std::malloc(sz == 0u ? 1u : sz);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void* operator new[](std::size_t sz) {
    return ::operator new(sz);
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}

int main() {
    g_heap_alloc_count.store(0ULL, std::memory_order_relaxed);
    g_track_allocs.store(true, std::memory_order_release);

    test_boot_handoff_validation();
    test_five_isa_instruction_controls();
    test_trap_vector_isolation_all_five_isas();
    test_timer_irqs_and_ring_overflow_zero_alloc();

    g_track_allocs.store(false, std::memory_order_release);
    uint64_t allocs = g_heap_alloc_count.load(std::memory_order_acquire);
    CHECK(allocs == 0ULL);

    std::printf("K1 Boot & Traps Test Suite: %d passed, %d failed (heap_allocs=%llu)\n",
                g_checks_passed,
                g_checks_failed,
                static_cast<unsigned long long>(allocs));
    return (g_checks_failed == 0 && allocs == 0ULL) ? 0 : 1;
}
