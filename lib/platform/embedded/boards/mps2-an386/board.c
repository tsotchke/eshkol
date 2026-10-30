/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Reference board: Arm MPS2+ AN386 (Cortex-M4), as modelled by
 * `qemu-system-arm -M mps2-an386`.  Console: CMSDK APB UART0.  Exit: Arm
 * semihosting SYS_EXIT_EXTENDED, so the emulator's exit status is the
 * value eshkol_main returned (run QEMU with -semihosting-config
 * enable=on,target=native).
 */
#include "eshkol_embedded.h"

#define UART0_BASE 0x40004000u
#define UART_DATA  (*(volatile uint32_t*)(UART0_BASE + 0x00u))
#define UART_STATE (*(volatile uint32_t*)(UART0_BASE + 0x04u))
#define UART_CTRL  (*(volatile uint32_t*)(UART0_BASE + 0x08u))
#define UART_BAUD  (*(volatile uint32_t*)(UART0_BASE + 0x10u))
#define UART_STATE_TX_FULL 0x1u
#define UART_CTRL_TX_EN    0x1u

void eshkol_board_init(void) {
    UART_BAUD = 16u;
    UART_CTRL = UART_CTRL_TX_EN;
}

void eshkol_board_putc(uint32_t c) {
    while (UART_STATE & UART_STATE_TX_FULL) {
    }
    UART_DATA = c & 0xffu;
}

volatile uint8_t* eshkol_board_console_register(void) {
    return (volatile uint8_t*)(uintptr_t)UART0_BASE;
}

static uint32_t semihost(uint32_t op, const void* arg) {
    register uint32_t r0 __asm__("r0") = op;
    register const void* r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void eshkol_board_exit(uint32_t code) {
    /* SYS_EXIT_EXTENDED (0x20): {ADP_Stopped_ApplicationExit, exit code}. */
    const uint32_t block[2] = {0x20026u, code};
    semihost(0x20u, block);
    for (;;) {
    }
}
