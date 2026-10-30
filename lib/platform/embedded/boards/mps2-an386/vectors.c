/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Cortex-M vector table and reset handler.  The core loads the initial
 * stack pointer and the reset vector from the first two words of
 * .isr_vector; every fault parks in a loop the debugger can inspect.
 */
#include "eshkol_embedded.h"

extern uint32_t __eshkol_stack_top[];

void eshkol_reset_handler(void) __attribute__((noreturn));
static void eshkol_fault_handler(void) {
    for (;;) {
    }
}

__attribute__((section(".isr_vector"), used))
void (*const eshkol_vector_table[16])(void) = {
    (void (*)(void))__eshkol_stack_top,
    eshkol_reset_handler,
    eshkol_fault_handler, /* NMI */
    eshkol_fault_handler, /* HardFault */
    eshkol_fault_handler, /* MemManage */
    eshkol_fault_handler, /* BusFault */
    eshkol_fault_handler, /* UsageFault */
    0, 0, 0, 0,
    eshkol_fault_handler, /* SVCall */
    eshkol_fault_handler, /* DebugMonitor */
    0,
    eshkol_fault_handler, /* PendSV */
    eshkol_fault_handler, /* SysTick */
};

void eshkol_reset_handler(void) {
#if defined(__ARM_FP)
    /* Hard-float images: grant CP10/CP11 (the FPU) full access first. */
    *(volatile uint32_t*)0xE000ED88u |= (0xFu << 20);
    __asm__ volatile("dsb\n\tisb" ::: "memory");
#endif
    eshkol_embedded_start();
}
