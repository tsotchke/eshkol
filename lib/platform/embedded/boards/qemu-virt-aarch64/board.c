/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Reference board: `qemu-system-aarch64 -M virt -cpu cortex-a53`, entered
 * at EL1 from the ELF entry point.  Console: PL011 UART at 0x09000000.
 * Exit: Arm semihosting SYS_EXIT (run QEMU with -semihosting).
 */
#include "eshkol_embedded.h"

#define PL011_BASE 0x09000000u
#define PL011_DR   (*(volatile uint32_t*)(PL011_BASE + 0x000u))
#define PL011_FR   (*(volatile uint32_t*)(PL011_BASE + 0x018u))
#define PL011_FR_TXFF 0x20u

extern uint64_t __eshkol_stack_top[];

__attribute__((naked, noreturn, section(".text.boot")))
void eshkol_aarch64_entry(void) {
    __asm__ volatile(
        "ldr x0, =__eshkol_stack_top\n\t"
        "mov sp, x0\n\t"
        /* CPACR_EL1.FPEN = 0b11: the AAPCS64 code uses FP/SIMD registers. */
        "mov x0, #(3 << 20)\n\t"
        "msr cpacr_el1, x0\n\t"
        "isb\n\t"
        "b eshkol_embedded_start\n\t");
}

void eshkol_board_init(void) {}

void eshkol_board_putc(uint32_t c) {
    while (PL011_FR & PL011_FR_TXFF) {
    }
    PL011_DR = c & 0xffu;
}

volatile uint8_t* eshkol_board_console_register(void) {
    return (volatile uint8_t*)(uintptr_t)PL011_BASE;
}

void eshkol_board_exit(uint32_t code) {
    /* SYS_EXIT (0x18) on AArch64 takes {reason, subcode}. */
    const uint64_t block[2] = {0x20026u, code};
    register uint64_t x0 __asm__("x0") = 0x18u;
    register const uint64_t* x1 __asm__("x1") = block;
    __asm__ volatile("hlt #0xf000" : "+r"(x0) : "r"(x1) : "memory");
    for (;;) {
    }
}
