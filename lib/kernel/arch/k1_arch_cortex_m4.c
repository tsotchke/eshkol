/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Phase K1 Arm Cortex-M4 (Thumb-2 / ARMv7E-M) trap vector descriptor
 * and instruction decoder.
 */

#include "k1_arch_internal.h"

const k1_arch_vector_desc_t k1_arch_desc_cortex_m4 = {
    K1_ISA_CORTEX_M4,
    "cortex-m4",
    "thumbv7em-none-eabi",
    32u,
    128u, /* VTOR minimum 128-byte vector table alignment */
    K1_CORTEX_M4_CAUSE_SYSTICK,
    K1_CORTEX_M4_CAUSE_DIVBYZERO,
    K1_CORTEX_M4_CAUSE_UNDEFINSTR,
    K1_CORTEX_M4_CAUSE_MEMMANAGE,
    K1_CORTEX_M4_CAUSE_SVCALL,
    {0x00u, 0xDEu, 0x00u, 0x00u}, /* Thumb UDF #0 (0xDE00 little-endian) */
    2u,
    {0x00u, 0xBFu, 0x00u, 0x00u}, /* Thumb NOP (0xBF00 little-endian) */
    2u
};

k1_trap_code_t k1_arch_decode_cause_cortex_m4(uint64_t raw_cause) {
    switch (raw_cause) {
        case K1_CORTEX_M4_CAUSE_UNDEFINSTR:
        case 0x00010000ULL: /* Raw CFSR.UFSR UNDEFINSTR bit */
            return K1_TRAP_ILLEGAL_INSN;
        case K1_CORTEX_M4_CAUSE_DIVBYZERO:
        case 0x02000000ULL: /* Raw CFSR.UFSR DIVBYZERO bit */
            return K1_TRAP_DIV_ZERO;
        case K1_CORTEX_M4_CAUSE_MEMMANAGE:
        case K1_CORTEX_M4_CAUSE_BUSFAULT:
        case 4ULL: /* MemManage exception number */
        case 5ULL: /* BusFault exception number */
            return K1_TRAP_ACCESS_FAULT;
        case K1_CORTEX_M4_CAUSE_SVCALL:
        case 11ULL: /* SVCall exception number */
            return K1_TRAP_SYSCALL;
        case K1_CORTEX_M4_CAUSE_SYSTICK:
        case 15ULL: /* SysTick exception number */
            return K1_TRAP_TIMER_IRQ;
        default:
            return K1_TRAP_UNKNOWN;
    }
}

void k1_arch_decode_insn_cortex_m4(const uint8_t* insn,
                                   size_t len,
                                   k1_insn_decode_result_t* out_res) {
    out_res->isa = K1_ISA_CORTEX_M4;
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_CORTEX_M4_CAUSE_UNDEFINSTR;
    out_res->is_valid_insn = 0u;
    out_res->is_nop = 0u;
    out_res->consumed_len = 0u;
    out_res->reserved = 0u;

    if (len < 2u) {
        return;
    }

    uint16_t hw = (uint16_t)((uint16_t)insn[0] | ((uint16_t)insn[1] << 8u));
    out_res->consumed_len = 2u;

    /* 16-bit Thumb permanently undefined UDF #imm8: 0xDE00..0xDEFF */
    if ((hw & 0xFF00u) == 0xDE00u) {
        out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
        out_res->native_cause = K1_CORTEX_M4_CAUSE_UNDEFINSTR;
        return;
    }

    /* 32-bit Thumb-2 permanently undefined UDF.W: 0xF7F0..0xF7FF + 0xAxxx */
    if ((hw & 0xFFF0u) == 0xF7F0u && len >= 4u) {
        uint16_t hw2 = (uint16_t)((uint16_t)insn[2] | ((uint16_t)insn[3] << 8u));
        if ((hw2 & 0xF000u) == 0xA000u) {
            out_res->consumed_len = 4u;
            out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
            out_res->native_cause = K1_CORTEX_M4_CAUSE_UNDEFINSTR;
            return;
        }
    }

    /* Canonical 16-bit Thumb NOP: 0xBF00 */
    if (hw == 0xBF00u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        return;
    }

    /* Thumb SVC #imm8: 0xDF00..0xDFFF */
    if ((hw & 0xFF00u) == 0xDF00u) {
        out_res->trap_code = K1_TRAP_SYSCALL;
        out_res->native_cause = K1_CORTEX_M4_CAUSE_SVCALL;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* Thumb BX LR (0x4770) */
    if (hw == 0x4770u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* Default fail-closed to UsageFault UNDEFINSTR */
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_CORTEX_M4_CAUSE_UNDEFINSTR;
}
