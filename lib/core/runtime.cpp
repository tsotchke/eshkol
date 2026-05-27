/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime interrupt flag definition + non-inline readable accessor.
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
 * Non-inline reader for tools and FFI surfaces that cannot use the
 * header-only `eshkol_runtime_interrupt_requested()` helper.
 */
extern "C" bool eshkol_runtime_interrupt_flag_is_set(void) {
    return g_eshkol_interrupt_flag != 0;
}
