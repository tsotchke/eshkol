/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime lifecycle and interrupt state management.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/core/resource_limits.h>
#include <eshkol/backend/thread_pool.h>
#include <eshkol/logger.h>

#include "runtime_hosted_internal.h"

#include <atomic>
#include <cstdlib>
#include <cstdio>

namespace {

// Runtime state (std::atomic for thread-safe API outside signal handlers)
std::atomic<eshkol_runtime_state_t> g_runtime_state{ESHKOL_RUNTIME_INITIALIZING};
std::atomic<eshkol_shutdown_reason_t> g_shutdown_reason{ESHKOL_SHUTDOWN_NONE};

}  // namespace

namespace eshkol::runtime_hosted {

/**
 * @brief Stop and join every runtime-owned worker-thread pool during shutdown.
 *
 * Currently calls thread_pool_global_shutdown() (idempotent — safe even if
 * the pool was never created, and safe to call more than once). Deliberately
 * does NOT touch the bytecode VM's separate parallel-map pool
 * (eshkol_vm_parallel_shutdown_global in lib/backend/vm_parallel.c); see the
 * inline comment in the body for why pulling that in here would force-link
 * VM/GPU dependencies into every AOT binary. Called from
 * eshkol_runtime_shutdown() before shutdown hooks run and before signal
 * handlers are restored, so no Eshkol worker thread can race a teardown step
 * that frees shared arena-backed state.
 */
void shutdown_all_thread_pools() {
    // thread_pool_global_shutdown() is idempotent — safe to call even if the
    // global pool was never created (no-op), and safe to call more than once
    // (guarded by mutex + null check) — so this can run unconditionally and
    // be reached from more than one shutdown path with no double-join risk.
    //
    // Deliberately NOT wired here: the bytecode VM's own parallel-map pool
    // (eshkol_vm_parallel_shutdown_global, lib/backend/vm_parallel.c). That
    // pool backs `-r`/interpreted and eval-mode parallel-map, not the
    // LLVM-codegen'd thread_pool.cpp path AOT-compiled binaries use. Calling
    // it from here would force-link the whole bytecode-VM unity object (and
    // its GPU/Metal/ObjC dependencies) into every AOT-compiled Eshkol
    // program via eshkol-runtime, even ones that never touch the VM —
    // confirmed experimentally: a minimal program linking only
    // libeshkol-runtime.a failed to link with undefined Metal/ObjC runtime
    // symbols once something referenced eshkol_vm_parallel_shutdown_global.
    // The interpreter/eval drivers (exe/eshkol-run.cpp, exe/eshkol-repl.cpp)
    // already link the full VM regardless and already call
    // eshkol_runtime_shutdown() before returning; closing the analogous race
    // for vm_parallel.c's pool belongs in those drivers directly (or behind
    // a self-registered shutdown hook from vm_parallel_ensure_pool()), not
    // in this always-linked TU. Tracked as a follow-up, out of scope here.
    thread_pool_global_shutdown();
}

}  // namespace eshkol::runtime_hosted

extern "C" {

/**
 * @brief Request a graceful runtime interrupt/shutdown from any thread.
 *
 * Sets the global interrupt flag, records `reason` and transitions
 * g_runtime_state to ESHKOL_RUNTIME_SHUTTING_DOWN (both via
 * release-ordered atomics), and mirrors the reason/state into the
 * signal-safe shadow variables so a subsequent signal handler observes a
 * consistent view. Does not itself perform teardown — code polling
 * g_eshkol_interrupt_flag / eshkol_runtime_get_shutdown_reason is expected
 * to unwind and call eshkol_runtime_shutdown().
 *
 * @param reason  Shutdown reason to record.
 */
void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason) {
    g_eshkol_interrupt_flag = 1;
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN, std::memory_order_release);
    // Update signal-safe shadow variables.
    eshkol::runtime_hosted::set_signal_shutdown_reason(reason);
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_SHUTTING_DOWN);
}

/**
 * @brief Clear a previously requested interrupt without changing runtime state.
 *
 * Resets g_eshkol_interrupt_flag and the shutdown reason (both the atomic
 * and the signal-safe shadow) back to "none". Does not touch
 * g_runtime_state, so this only cancels the interrupt signal itself, not
 * an in-progress shutdown state transition.
 */
void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
    // Update signal-safe shadow variables.
    eshkol::runtime_hosted::set_signal_shutdown_reason(ESHKOL_SHUTDOWN_NONE);
}

/** @brief Return the most recently recorded shutdown reason (acquire-ordered load). */
eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void) {
    return g_shutdown_reason.load(std::memory_order_acquire);
}

/**
 * @brief One-time initialization of the hosted Eshkol runtime.
 *
 * Atomically transitions g_runtime_state from ESHKOL_RUNTIME_INITIALIZING to
 * ESHKOL_RUNTIME_RUNNING; if the state is already RUNNING this is a no-op
 * success (idempotent), and any other unexpected state is logged and
 * rejected. On the successful path: makes stdout unbuffered (Bug AA — so
 * `(display ...)` output is never lost to buffering if a later form
 * crashes), reads resource limits from the environment and starts the
 * execution-time-limit timer if `ESHKOL_TIMEOUT_MS` is set, and installs
 * runtime signal handlers.
 *
 * @return  0 on success (including "already running"), -1 if called from an
 *          unexpected runtime state.
 */
int eshkol_runtime_init(void) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_INITIALIZING;
    if (!g_runtime_state.compare_exchange_strong(expected, ESHKOL_RUNTIME_RUNNING)) {
        // Already initialized or in wrong state.
        if (expected == ESHKOL_RUNTIME_RUNNING) {
            return 0;
        }
        eshkol_warn("Runtime init called in unexpected state: %d", (int)expected);
        return -1;
    }
    // Sync signal-safe shadow.
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_RUNNING);

    // Bug AA: make stdout unbuffered so EVERY `(display ...)` reaches the
    // terminal immediately, even without a trailing newline and even when the
    // next form crashes. setvbuf is only safe before any I/O on the stream;
    // runtime init runs before user-level Eshkol code.
    setvbuf(stdout, NULL, _IONBF, 0);

    eshkol_resource_limits_t limits = eshkol_init_limits_from_env();
    if (std::getenv("ESHKOL_TIMEOUT_MS") && limits.max_execution_time_ms > 0) {
        eshkol_start_timer(limits.max_execution_time_ms);
    }

    eshkol_runtime_init_signals();

    eshkol_info("Eshkol runtime initialized");
    return 0;
}

/**
 * @brief Tear down the hosted Eshkol runtime, once, in a fixed safe order.
 *
 * Atomically transitions g_runtime_state from RUNNING to SHUTTING_DOWN;
 * returns immediately (no-op) if shutdown/termination is already underway
 * or complete, so this is safe to call from multiple shutdown paths. On the
 * first call: records the shutdown reason (atomic + signal-safe shadow),
 * sets the interrupt flag, logs the reason, waits (up to 5s) for in-flight
 * operations to drain, then — in this order — stops every runtime worker
 * pool (shutdown_all_thread_pools, ESH-0216: must happen before anything
 * that can free shared arena state, to avoid a racing worker touching freed
 * memory), runs registered shutdown hooks, stops the execution timer, and
 * restores the original signal handlers. Finally marks the runtime
 * TERMINATED (atomic + shadow).
 *
 * @param reason  Reason for the shutdown, recorded for later inspection.
 */
void eshkol_runtime_shutdown(eshkol_shutdown_reason_t reason) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_RUNNING;
    if (!g_runtime_state.compare_exchange_strong(expected, ESHKOL_RUNTIME_SHUTTING_DOWN)) {
        if (expected == ESHKOL_RUNTIME_SHUTTING_DOWN ||
            expected == ESHKOL_RUNTIME_TERMINATED) {
            return;
        }
    }
    // Sync signal-safe shadow.
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_SHUTTING_DOWN);

    g_shutdown_reason.store(reason, std::memory_order_release);
    eshkol::runtime_hosted::set_signal_shutdown_reason(reason);
    g_eshkol_interrupt_flag = 1;

    const char* reason_str = "unknown";
    switch (reason) {
        case ESHKOL_SHUTDOWN_NONE: reason_str = "none"; break;
        case ESHKOL_SHUTDOWN_REQUESTED: reason_str = "user request"; break;
        case ESHKOL_SHUTDOWN_TIMEOUT: reason_str = "timeout"; break;
        case ESHKOL_SHUTDOWN_MEMORY: reason_str = "memory limit"; break;
        case ESHKOL_SHUTDOWN_ERROR: reason_str = "error"; break;
    }
    eshkol_info("Shutting down (reason: %s)...", reason_str);

    uint32_t op_count = eshkol_runtime_get_operation_count();
    if (op_count > 0) {
        eshkol_info("Waiting for %u in-flight operations to complete...", op_count);
        if (!eshkol_runtime_drain_operations(5000)) {
            eshkol_warn("Timeout waiting for operations, proceeding with shutdown");
        }
    }

    // ESH-0216: stop/join every runtime-owned worker-thread pool BEFORE
    // running shutdown hooks or restoring signal handlers. Shutdown hooks
    // (and future teardown steps) are free to reset or free shared
    // arena-backed state; if a parallel-map/parallel-execute worker is still
    // mid-task at that point it dereferences that state out from under
    // itself and SIGSEGVs deep in teardown — well after "graceful shutdown"
    // has already been logged, and the fatal-signal handler is still
    // installed (eshkol_runtime_restore_signals() hasn't run yet), so the
    // crash surfaces as "[Eshkol] fatal signal: SIGSEGV" at whatever address
    // each racing worker happened to be touching. Stopping every pool here,
    // ahead of everything else in this function, guarantees no Eshkol
    // worker thread is running for the remainder of shutdown or after it
    // returns.
    eshkol::runtime_hosted::shutdown_all_thread_pools();

    eshkol::runtime_hosted::run_shutdown_hooks(reason);
    eshkol_stop_timer();
    eshkol_runtime_restore_signals();

    g_runtime_state.store(ESHKOL_RUNTIME_TERMINATED, std::memory_order_release);
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_TERMINATED);

    eshkol_info("Shutdown complete");
}

/** @brief Return the current runtime lifecycle state (acquire-ordered load). */
eshkol_runtime_state_t eshkol_runtime_get_state(void) {
    return g_runtime_state.load(std::memory_order_acquire);
}

}  // extern "C"
