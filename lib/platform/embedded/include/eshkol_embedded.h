/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Eshkol freestanding runtime subset: the contract between a bare-metal
 * Eshkol image, its board support, and the startup code.
 *
 * An image consists of
 *   - the object eshkol-run emits for a freestanding profile, which exports
 *       uint32_t eshkol_main(void);
 *     (an `(define (eshkol-main) : u32 ... :export-symbol eshkol_main)`;
 *     machine-typed exports use the platform C ABI),
 *   - this runtime subset (memory primitives and startup), and
 *   - one board: reset/entry code, console and exit hooks, link script.
 *
 * Startup zeroes .bss, copies .data from its load address, runs
 * eshkol_board_init(), calls eshkol_main() and passes its result to
 * eshkol_board_exit().  Nothing here uses a C library, an allocator, an
 * operating system or the hosted Eshkol runtime.
 */
#ifndef ESHKOL_EMBEDDED_H
#define ESHKOL_EMBEDDED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provided by the Eshkol program. */
uint32_t eshkol_main(void);

/* Provided by the runtime subset. */
void eshkol_embedded_start(void) __attribute__((noreturn));
void* memcpy(void* dst, const void* src, size_t n);
void* memmove(void* dst, const void* src, size_t n);
void* memset(void* dst, int c, size_t n);
int memcmp(const void* a, const void* b, size_t n);
void eshkol_board_puts(const char* s);
void eshkol_board_put_u32(uint32_t value);

/* Provided by each board. */
void eshkol_board_init(void);
void eshkol_board_putc(uint32_t c);
/* Address of the console transmit-data register, for programs that write
 * the console directly with volatile-store! (the reference boards' QEMU
 * models accept back-to-back writes; eshkol_board_putc waits for space). */
volatile uint8_t* eshkol_board_console_register(void);
void eshkol_board_exit(uint32_t code) __attribute__((noreturn));

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_EMBEDDED_H */
