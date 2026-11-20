/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Reference board: `qemu-system-riscv32/riscv64 -M virt -bios none`,
 * entered in machine mode at 0x80000000.  Console: NS16550A UART at
 * 0x10000000.  Exit: SiFive test finisher at 0x100000 (0x5555 = pass,
 * (code << 16) | 0x3333 = fail with code), so the emulator's exit status is
 * the value eshkol_main returned.
 */
#include "eshkol_embedded.h"

#define UART_BASE 0x10000000u
#define UART_THR  (*(volatile uint8_t*)(UART_BASE + 0u))
#define UART_LSR  (*(volatile uint8_t*)(UART_BASE + 5u))
#define UART_LSR_THRE 0x20u
#define TEST_FINISHER (*(volatile uint32_t*)0x100000u)

__attribute__((naked, noreturn, section(".text.boot")))
void eshkol_riscv_entry(void) {
    __asm__ volatile(
        ".option push\n\t"
        ".option norelax\n\t"
        "la gp, __global_pointer$\n\t"
        ".option pop\n\t"
        "la sp, __eshkol_stack_top\n\t"
#if defined(__riscv_flen)
        /* mstatus.FS = Initial: the lp64d/ilp32d ABI uses the F/D registers. */
        "li t0, 0x2000\n\t"
        "csrs mstatus, t0\n\t"
#endif
        "j eshkol_embedded_start\n\t");
}

void eshkol_board_init(void) {}

void eshkol_board_putc(uint32_t c) {
    while ((UART_LSR & UART_LSR_THRE) == 0) {
    }
    UART_THR = (uint8_t)c;
}

volatile uint8_t* eshkol_board_console_register(void) {
    return (volatile uint8_t*)(uintptr_t)UART_BASE;
}

void eshkol_board_exit(uint32_t code) {
    TEST_FINISHER = code == 0 ? 0x5555u : ((code & 0xffffu) << 16) | 0x3333u;
    for (;;) {
    }
}
