/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Phase K1 RISC-V 32 (RV32IMAC) trap vector descriptor and instruction decoder.
 */

#include "k1_arch_internal.h"

static uint32_t k1_riscv32_read_le32(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

const k1_arch_vector_desc_t k1_arch_desc_riscv32 = {
    K1_ISA_RISCV32,
    "riscv32",
    "riscv32-unknown-none-elf",
    32u,
    4u, /* mtvec direct-mode 4-byte alignment */
    K1_RISCV32_IRQ_MTIMER,
    K1_RISCV_CAUSE_BREAK_DIV0,
    K1_RISCV_CAUSE_ILLEGAL_INSN,
    K1_RISCV_CAUSE_LOAD_ACCESS,
    K1_RISCV_CAUSE_ECALL_U,
    {0x00u, 0x00u, 0x00u, 0x00u}, /* Illegal 0-word */
    4u,
    {0x13u, 0x00u, 0x00u, 0x00u}, /* ADDI x0, x0, 0 (0x00000013 little-endian) */
    4u
};

k1_trap_code_t k1_arch_decode_cause_riscv32(uint64_t raw_cause) {
    if (raw_cause == K1_RISCV32_IRQ_MTIMER || raw_cause == K1_RISCV32_IRQ_STIMER) {
        return K1_TRAP_TIMER_IRQ;
    }

    switch (raw_cause) {
        case K1_RISCV_CAUSE_ILLEGAL_INSN:
            return K1_TRAP_ILLEGAL_INSN;
        case K1_RISCV_CAUSE_BREAK_DIV0:
            return K1_TRAP_DIV_ZERO;
        case K1_RISCV_CAUSE_INSN_ACCESS:
        case K1_RISCV_CAUSE_LOAD_ACCESS:
        case K1_RISCV_CAUSE_STORE_ACCESS:
        case K1_RISCV_CAUSE_INSN_PAGE:
        case K1_RISCV_CAUSE_LOAD_PAGE:
        case K1_RISCV_CAUSE_STORE_PAGE:
            return K1_TRAP_ACCESS_FAULT;
        case K1_RISCV_CAUSE_ECALL_U:
            return K1_TRAP_SYSCALL;
        default:
            return K1_TRAP_UNKNOWN;
    }
}

void k1_arch_decode_insn_riscv32(const uint8_t* insn,
                                 size_t len,
                                 k1_insn_decode_result_t* out_res) {
    out_res->isa = K1_ISA_RISCV32;
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
    out_res->is_valid_insn = 0u;
    out_res->is_nop = 0u;
    out_res->consumed_len = 0u;
    out_res->reserved = 0u;

    if (len < 2u) {
        return;
    }

    uint16_t lo = (uint16_t)((uint16_t)insn[0] | ((uint16_t)insn[1] << 8u));

    /* All-zero instruction is architecturally defined illegal on RV32 */
    if (lo == 0x0000u) {
        out_res->consumed_len = (len >= 4u) ? 4u : 2u;
        out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
        out_res->native_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
        return;
    }

    /* Compressed RVC C.NOP (0x0001) */
    if (lo == 0x0001u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        out_res->consumed_len = 2u;
        return;
    }

    if ((lo & 0x3u) != 0x3u || len < 4u) {
        out_res->consumed_len = 2u;
        out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
        out_res->native_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
        return;
    }

    uint32_t word = k1_riscv32_read_le32(insn);
    out_res->consumed_len = 4u;

    if (word == 0xFFFFFFFFu) {
        out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
        out_res->native_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
        return;
    }

    /* Canonical NOP: ADDI x0, x0, 0 (0x00000013) */
    if (word == 0x00000013u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        return;
    }

    /* ECALL (0x00000073) */
    if (word == 0x00000073u) {
        out_res->trap_code = K1_TRAP_SYSCALL;
        out_res->native_cause = K1_RISCV_CAUSE_ECALL_U;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* EBREAK (0x00100073) */
    if (word == 0x00100073u) {
        out_res->trap_code = K1_TRAP_DIV_ZERO;
        out_res->native_cause = K1_RISCV_CAUSE_BREAK_DIV0;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* RET: JALR x0, 0(x1) (0x00008067) */
    if (word == 0x00008067u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* Reject RV64-only 64-bit opcodes (OP-IMM-32 0x1B, OP-32 0x3B) and unrecognized words */
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_RISCV_CAUSE_ILLEGAL_INSN;
}
