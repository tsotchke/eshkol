/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime interrupt flag definition + readable accessor.
 *
 * The flag itself is declared `extern volatile sig_atomic_t` in
 * inc/eshkol/core/runtime.h.  Signal handlers in
 * runtime_signals_hosted.cpp store into it; long-running loops in
 * the codegen test it via `eshkol_runtime_interrupt_requested()`
 * (also declared in runtime.h, defined inline there).  This file
 * is the single storage definition.
 */

#include <eshkol/core/runtime.h>

volatile sig_atomic_t g_eshkol_interrupt_flag = 0;

/*
 * Programmatic reset for the interrupt flag.  Useful when an
 * interactive REPL absorbs a SIGINT and then wants to continue
 * accepting new input without inheriting the previous interrupt
 * state.  Naming mirrors `eshkol_runtime_interrupt_requested()`.
 */
extern "C" void eshkol_runtime_clear_interrupt_flag(void) {
    g_eshkol_interrupt_flag = 0;
}
