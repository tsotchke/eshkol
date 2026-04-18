/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * eval bridge — runtime-registered accessors for ReplJITContext access.
 *
 * Design: eshkol-static holds writable function pointers that default to
 * nullptr. Binaries that link eshkol-repl-lib also pull in a static
 * constructor (in eval_bridge_impl.cpp) that populates those pointers at
 * program start. Binaries without the REPL runtime keep the nullptrs
 * and produce a clear runtime error when eval is invoked.
 *
 * Why registration over weak-symbol override:
 *   macOS ld64 does not reliably replace weak data or function symbols
 *   with strong versions when both live in separate static archives —
 *   even with -force_load on the archive containing the strong
 *   symbol. The ABI says it should work; in practice it doesn't for
 *   our build. Registration sidesteps that entirely: the strong TU
 *   runs a constructor with a priority high enough that it fires
 *   before any eval call.
 */

#include <eshkol/eshkol.h>
#include <eshkol/logger.h>
#include <eshkol/core/eval_bridge.h>

#include <atomic>

/* Runtime-mutable function pointers. Initially nullptr. If
 * eshkol-repl-lib is linked, its constructor (in eval_bridge_impl.cpp)
 * flips these to the real implementations before main() runs. */
static eshkol_eval_jit_acquire_fn_t s_acquire = nullptr;
static eshkol_eval_jit_execute_fn_t s_execute = nullptr;
static eshkol_eval_jit_lookup_fn_t  s_lookup  = nullptr;

extern "C" {

void eshkol_eval_jit_register(eshkol_eval_jit_acquire_fn_t acquire,
                              eshkol_eval_jit_execute_fn_t execute,
                              eshkol_eval_jit_lookup_fn_t  lookup) {
    s_acquire = acquire;
    s_execute = execute;
    s_lookup  = lookup;
}

eshkol_eval_jit_acquire_fn_t eshkol_eval_jit_get_acquire(void) { return s_acquire; }
eshkol_eval_jit_execute_fn_t eshkol_eval_jit_get_execute(void) { return s_execute; }
eshkol_eval_jit_lookup_fn_t  eshkol_eval_jit_get_lookup (void) { return s_lookup;  }

int eshkol_eval_jit_available(void) {
    return s_acquire != nullptr;
}

void eshkol_eval_jit_warn_missing(const char* caller) {
    static std::atomic<bool> s_warned{false};
    bool expected = false;
    if (s_warned.compare_exchange_strong(expected, true)) {
        eshkol_error("%s: JIT runtime not linked — rebuild with "
                     "eshkol-repl-lib to enable eval/compile",
                     caller ? caller : "eval");
    }
}

} /* extern "C" */
