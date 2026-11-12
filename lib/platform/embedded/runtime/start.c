/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Common C startup for freestanding Eshkol images.  Each board's reset or
 * entry code sets up a stack (and enables the FPU where the ABI uses it)
 * and then calls eshkol_embedded_start().  Link scripts define the symbols
 * below.
 */
#include "eshkol_embedded.h"

extern uint8_t __eshkol_data_load[];
extern uint8_t __eshkol_data_start[];
extern uint8_t __eshkol_data_end[];
extern uint8_t __eshkol_bss_start[];
extern uint8_t __eshkol_bss_end[];

void eshkol_embedded_start(void) {
    /* Link-script symbols: compare addresses, not arrays. */
    uint8_t* load = __eshkol_data_load;
    uint8_t* start = __eshkol_data_start;
    if (load != start) {
        memcpy(start, load, (size_t)(__eshkol_data_end - start));
    }
    memset(__eshkol_bss_start, 0, (size_t)(__eshkol_bss_end - __eshkol_bss_start));
    eshkol_board_init();
    eshkol_board_exit(eshkol_main());
}

void eshkol_board_puts(const char* s) {
    while (*s) eshkol_board_putc((uint8_t)*s++);
}

void eshkol_board_put_u32(uint32_t value) {
    char digits[10];
    int n = 0;
    do {
        digits[n++] = (char)('0' + value % 10u);
        value /= 10u;
    } while (value != 0);
    while (n > 0) eshkol_board_putc((uint8_t)digits[--n]);
}
