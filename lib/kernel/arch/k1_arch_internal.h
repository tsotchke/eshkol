/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Internal per-ISA trap vector descriptor and decoder hooks for Phase K1.
 */

#ifndef ESHKOL_KERNEL_K1_ARCH_INTERNAL_H
#define ESHKOL_KERNEL_K1_ARCH_INTERNAL_H

#include <eshkol/kernel/k1_boot_traps.h>

extern const k1_arch_vector_desc_t k1_arch_desc_x86_64;
extern const k1_arch_vector_desc_t k1_arch_desc_aarch64;
extern const k1_arch_vector_desc_t k1_arch_desc_riscv64;
extern const k1_arch_vector_desc_t k1_arch_desc_riscv32;
extern const k1_arch_vector_desc_t k1_arch_desc_cortex_m4;

k1_trap_code_t k1_arch_decode_cause_x86_64(uint64_t raw_cause);
k1_trap_code_t k1_arch_decode_cause_aarch64(uint64_t raw_cause);
k1_trap_code_t k1_arch_decode_cause_riscv64(uint64_t raw_cause);
k1_trap_code_t k1_arch_decode_cause_riscv32(uint64_t raw_cause);
k1_trap_code_t k1_arch_decode_cause_cortex_m4(uint64_t raw_cause);

void k1_arch_decode_insn_x86_64(const uint8_t* insn,
                                size_t len,
                                k1_insn_decode_result_t* out_res);
void k1_arch_decode_insn_aarch64(const uint8_t* insn,
                                 size_t len,
                                 k1_insn_decode_result_t* out_res);
void k1_arch_decode_insn_riscv64(const uint8_t* insn,
                                 size_t len,
                                 k1_insn_decode_result_t* out_res);
void k1_arch_decode_insn_riscv32(const uint8_t* insn,
                                 size_t len,
                                 k1_insn_decode_result_t* out_res);
void k1_arch_decode_insn_cortex_m4(const uint8_t* insn,
                                   size_t len,
                                   k1_insn_decode_result_t* out_res);

#endif /* ESHKOL_KERNEL_K1_ARCH_INTERNAL_H */
