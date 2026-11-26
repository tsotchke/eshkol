/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Phase K1 AArch64 EL1 trap vector descriptor and instruction decoder.
 */

#include "k1_arch_internal.h"

static uint32_t k1_aarch64_read_le32(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

const k1_arch_vector_desc_t k1_arch_desc_aarch64 = {
    K1_ISA_AARCH64,
    "aarch64",
    "aarch64-unknown-none-elf",
    64u,
    2048u, /* VBAR_EL1 requires 2 KiB alignment */
    K1_AARCH64_IRQ_TIMER_PPI,
    K1_AARCH64_EC_BRK_DIV_ZERO,
    K1_AARCH64_EC_ILLEGAL_INSN,
    K1_AARCH64_EC_DABT_EL0,
    K1_AARCH64_EC_SVC64,
    {0x00u, 0x00u, 0x00u, 0x00u}, /* UDF #0 */
    4u,
    {0x1Fu, 0x20u, 0x03u, 0xD5u}, /* NOP (0xD503201F little-endian) */
    4u
};

k1_trap_code_t k1_arch_decode_cause_aarch64(uint64_t raw_cause) {
    if (raw_cause == K1_AARCH64_IRQ_TIMER_PPI) {
        return K1_TRAP_TIMER_IRQ;
    }

    /* Accept either direct 6-bit EC or full 32-bit ESR_EL1 with EC in [31:26]. */
    uint64_t ec = (raw_cause > 0x3FULL) ? ((raw_cause >> 26u) & 0x3FULL) : raw_cause;

    switch (ec) {
        case K1_AARCH64_EC_ILLEGAL_INSN:
            return K1_TRAP_ILLEGAL_INSN;
        case K1_AARCH64_EC_SVC64:
            return K1_TRAP_SYSCALL;
        case K1_AARCH64_EC_IABT_EL0:
        case K1_AARCH64_EC_IABT_EL1:
        case K1_AARCH64_EC_DABT_EL0:
        case K1_AARCH64_EC_DABT_EL1:
            return K1_TRAP_ACCESS_FAULT;
        case K1_AARCH64_EC_BRK_DIV_ZERO:
            return K1_TRAP_DIV_ZERO;
        default:
            return K1_TRAP_UNKNOWN;
    }
}

void k1_arch_decode_insn_aarch64(const uint8_t* insn,
                                 size_t len,
                                 k1_insn_decode_result_t* out_res) {
    out_res->isa = K1_ISA_AARCH64;
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_AARCH64_EC_ILLEGAL_INSN;
    out_res->is_valid_insn = 0u;
    out_res->is_nop = 0u;
    out_res->consumed_len = 0u;
    out_res->reserved = 0u;

    if (len < 4u) {
        return;
    }

    uint32_t word = k1_aarch64_read_le32(insn);
    out_res->consumed_len = 4u;

    /* Permanently undefined UDF #imm16 space: 0x00000000..0x0000FFFF */
    if ((word & 0xFFFF0000u) == 0x00000000u || word == 0xFFFFFFFFu) {
        out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
        out_res->native_cause = K1_AARCH64_EC_ILLEGAL_INSN;
        return;
    }

    /* Canonical A64 NOP: 0xD503201F */
    if (word == 0xD503201Fu) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        out_res->is_nop = 1u;
        return;
    }

    /* SVC #imm16: 0xD4000001 mask 0xFFE0001F */
    if ((word & 0xFFE0001Fu) == 0xD4000001u) {
        out_res->trap_code = K1_TRAP_SYSCALL;
        out_res->native_cause = K1_AARCH64_EC_SVC64;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* BRK #imm16: 0xD4200000 mask 0xFFE0001F */
    if ((word & 0xFFE0001Fu) == 0xD4200000u) {
        out_res->trap_code = K1_TRAP_DIV_ZERO;
        out_res->native_cause = K1_AARCH64_EC_BRK_DIV_ZERO;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* RET (0xD65F03C0) */
    if (word == 0xD65F03C0u) {
        out_res->trap_code = K1_TRAP_NONE;
        out_res->native_cause = 0ULL;
        out_res->is_valid_insn = 1u;
        return;
    }

    /* Default fail-closed for unrecognized A64 words */
    out_res->trap_code = K1_TRAP_ILLEGAL_INSN;
    out_res->native_cause = K1_AARCH64_EC_ILLEGAL_INSN;
}
