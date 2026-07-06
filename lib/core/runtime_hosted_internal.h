/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Private hosted runtime coordination helpers.
 */
#ifndef ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H
#define ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H

#include <eshkol/core/runtime.h>

namespace eshkol::runtime_hosted {

void set_signal_runtime_state(eshkol_runtime_state_t state);
void set_signal_shutdown_reason(eshkol_shutdown_reason_t reason);
void run_shutdown_hooks(eshkol_shutdown_reason_t reason);

// ESH-0216: stop/join every runtime-owned worker-thread pool (the
// parallel-map/parallel-execute thread pool and the bytecode VM's parallel
// pool, if either was ever created). Must run before run_shutdown_hooks()
// and before eshkol_runtime_restore_signals() in eshkol_runtime_shutdown() —
// arbitrary shutdown hooks and the fatal-signal teardown may free or reset
// shared arena-backed state, and a still-running worker thread that touches
// that state after it is freed is exactly the SIGSEGV-after-"graceful
// shutdown" race this closes. Idempotent and safe to call even if no pool
// was ever created.
void shutdown_all_thread_pools();

}  // namespace eshkol::runtime_hosted

#endif  // ESHKOL_CORE_RUNTIME_HOSTED_INTERNAL_H
