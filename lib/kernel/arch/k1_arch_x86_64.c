/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Phase K1 x86_64 IDT trap vector descriptor and instruction decoder.
 */

#include "k1_arch_internal.h"

const k1_arch_vector_desc_t k1_arch_desc_x86_64 = {
    K1_ISA_X86_64,
    "x86_64",
    "x86_64-unknown-none-elf",
    64u,
    16u,
    K1_X86_64_VEC_TIMER_IRQ,
    K1_X86_64_VEC_DIV_ZERO,
    K1_X86_64_VEC_ILLEGAL_INSN,
    K1_X86_64_VEC_PAGE_FAULT,
    K1_X86_64_VEC_SYSCALL,
    {0x0Fu, 0x0Bu, 0x00u, 0x00u}, /* UD2 */
    2u,
    {0x90u, 0x00u, 0x00u, 0x00u}, /* NOP */
    1u
};

k1_trap_code_t k1_arch_decode_cause_x86_64(uint64_t raw_cause) {
    switch (raw_cause) {
        case K1_X86_64_VEC_DIV_ZERO:
            return K1_TRAP_DIV_ZERO;
        case K1_X86_64_VEC_ILLEGAL_INSN:
            return K1_TRAP_ILLEGAL_INSN;
        case K1_X86_64_VEC_GP_FAULT:
        case K1_X86_64_VEC_PAGE_FAULT:
            return K1_TRAP_ACCESS_FAULT;
        case K1_X86_64_VEC_TIMER_IRQ:
            return K1_TRAP_TIMER_IRQ;
        case K1_X86_64_VEC_SYSCALL:
            return K1_TRAP_SYSCALL;
        default:
            return K1_TRAP_UNKNOWN;
    }
}

void k1_arch_decode_insn_x86_64(const uint8_t* insn,
                                size_t len,
                                k1_insn_decode_result_t* out_res) {
    out_res->isa = K1_ISA_X86_64;
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_X86_64_VEC_ILLEGAL_INSN;
    out_res->is_valid_insn = 0u;
    out_res->is_nop = 0u;
    out_res->consumed_len = 0u;
    out_res->reserved = 0u;

    if (len == 0u) {
        return;
    }

    /* Single-byte NOP (0x90) */
    if (insn[0] == 0x90u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        out_res->consumed_len = 1u;
        return;
    }

    /* Two-byte UD2 (0x0F 0x0B) or UD0/UD1 (0x0F 0xFF / 0x0F 0xB9) */
    if (insn[0] == 0x0Fu) {
        if (len >= 2u && (insn[1] == 0x0Bu || insn[1] == 0xB9u || insn[1] == 0xFFu)) {
            out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
            out_res->native_cause = K1_X86_64_VEC_ILLEGAL_INSN;
            out_res->consumed_len = 2u;
            return;
        }
        /* Two-byte SYSCALL (0x0F 0x05) */
        if (len >= 2u && insn[1] == 0x05u) {
            out_res->trap_code = K1_TRAP_SYSCALL;
            out_res->native_cause = K1_X86_64_VEC_SYSCALL;
            out_res->is_valid_insn = 1u;
            out_res->consumed_len = 2u;
            return;
        }
        return;
    }

    /* Multi-byte NOP (0x66 0x90) */
    if (len >= 2u && insn[0] == 0x66u && insn[1] == 0x90u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        out_res->consumed_len = 2u;
        return;
    }

    /* Valid RET (0xC3) */
    if (insn[0] == 0xC3u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->consumed_len = 1u;
        return;
    }

    /* Legacy 32-bit-only invalid-in-64-bit opcodes (e.g. 0x06, 0x07, 0x0E, 0x27, 0x2F, 0x60, 0x61, 0xCE, 0xD4, 0xD5) */
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_X86_64_VEC_ILLEGAL_INSN;
    out_res->consumed_len = 1u;
}
