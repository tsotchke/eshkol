/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime lifecycle and interrupt state management.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/core/resource_limits.h>
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

extern "C" {

void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason) {
    g_eshkol_interrupt_flag = 1;
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN, std::memory_order_release);
    // Update signal-safe shadow variables.
    eshkol::runtime_hosted::set_signal_shutdown_reason(reason);
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_SHUTTING_DOWN);
}

void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
    // Update signal-safe shadow variables.
    eshkol::runtime_hosted::set_signal_shutdown_reason(ESHKOL_SHUTDOWN_NONE);
}

eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void) {
    return g_shutdown_reason.load(std::memory_order_acquire);
}

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

    eshkol::runtime_hosted::run_shutdown_hooks(reason);
    eshkol_stop_timer();
    eshkol_runtime_restore_signals();

    g_runtime_state.store(ESHKOL_RUNTIME_TERMINATED, std::memory_order_release);
    eshkol::runtime_hosted::set_signal_runtime_state(ESHKOL_RUNTIME_TERMINATED);

    eshkol_info("Shutdown complete");
}

eshkol_runtime_state_t eshkol_runtime_get_state(void) {
    return g_runtime_state.load(std::memory_order_acquire);
}

}  // extern "C"
