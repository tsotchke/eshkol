/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Runtime State Management Implementation
 *
 * Provides signal handling, interrupt checking, shutdown hooks,
 * and in-flight operation tracking for graceful shutdown.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/logger.h>

#include <unistd.h>    // STDERR_FILENO, write(), _exit()
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <cstring>
#include <csignal>

// ============================================================================
// Global State
// ============================================================================

// Volatile flag for signal-safe interrupt checking
volatile sig_atomic_t g_eshkol_interrupt_flag = 0;

namespace {

// Runtime state
std::atomic<eshkol_runtime_state_t> g_runtime_state{ESHKOL_RUNTIME_INITIALIZING};
std::atomic<eshkol_shutdown_reason_t> g_shutdown_reason{ESHKOL_SHUTDOWN_NONE};

// Shutdown hooks
struct ShutdownHook {
    uint32_t id;
    eshkol_shutdown_hook_t callback;
    void* context;
    std::string name;
};

std::mutex g_hooks_mutex;
std::vector<ShutdownHook> g_shutdown_hooks;
std::atomic<uint32_t> g_next_hook_id{1};

// In-flight operations
struct InFlightOperation {
    uint32_t id;
    std::string name;
    std::chrono::steady_clock::time_point start_time;
};

std::mutex g_operations_mutex;
std::condition_variable g_operations_cv;
std::vector<InFlightOperation> g_in_flight_operations;
std::atomic<uint32_t> g_next_operation_id{1};

// Original signal handlers (for restoration)
struct sigaction g_old_sigint_handler;
struct sigaction g_old_sigterm_handler;
struct sigaction g_old_sigpipe_handler;
bool g_signals_installed = false;

// ============================================================================
// Signal Handler
// ============================================================================

void eshkol_signal_handler(int signum) {
    // Set the volatile flag atomically (signal-safe)
    g_eshkol_interrupt_flag = 1;

    // Set shutdown reason based on signal
    eshkol_shutdown_reason_t reason = ESHKOL_SHUTDOWN_REQUESTED;

    // Store reason (atomic operations are signal-safe for sig_atomic_t-sized types)
    g_shutdown_reason.store(reason, std::memory_order_relaxed);

    // Log message (use only async-signal-safe functions)
    // write() is async-signal-safe, fprintf is not
    const char* msg = nullptr;
    size_t msg_len = 0;

    switch (signum) {
        case SIGINT:
            msg = "\n[Eshkol] Interrupt received (SIGINT), initiating graceful shutdown...\n";
            msg_len = 64;
            break;
        case SIGTERM:
            msg = "\n[Eshkol] Termination requested (SIGTERM), initiating graceful shutdown...\n";
            msg_len = 70;
            break;
        default:
            msg = "\n[Eshkol] Signal received, initiating graceful shutdown...\n";
            msg_len = 58;
            break;
    }

    // Use write() which is async-signal-safe
    (void)write(STDERR_FILENO, msg, msg_len);

    // If we get a second signal during shutdown, force exit
    if (g_runtime_state.load(std::memory_order_relaxed) == ESHKOL_RUNTIME_SHUTTING_DOWN) {
        const char* force_msg = "[Eshkol] Second interrupt, forcing exit!\n";
        (void)write(STDERR_FILENO, force_msg, 42);
        _exit(128 + signum);  // Standard Unix exit code for signal termination
    }

    // Mark as shutting down
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN, std::memory_order_relaxed);
}

} // anonymous namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Signal Handling
// ----------------------------------------------------------------------------

void eshkol_runtime_init_signals(void) {
    if (g_signals_installed) {
        return;
    }

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = eshkol_signal_handler;
    sa.sa_flags = 0;  // No SA_RESTART - we want syscalls to be interrupted
    sigemptyset(&sa.sa_mask);

    // Install SIGINT handler (Ctrl+C)
    if (sigaction(SIGINT, &sa, &g_old_sigint_handler) != 0) {
        eshkol_warn("Failed to install SIGINT handler");
    }

    // Install SIGTERM handler (kill)
    if (sigaction(SIGTERM, &sa, &g_old_sigterm_handler) != 0) {
        eshkol_warn("Failed to install SIGTERM handler");
    }

    // Ignore SIGPIPE (broken pipe) - handle errors in application code instead
    struct sigaction sa_ignore;
    std::memset(&sa_ignore, 0, sizeof(sa_ignore));
    sa_ignore.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &sa_ignore, &g_old_sigpipe_handler) != 0) {
        eshkol_warn("Failed to install SIGPIPE handler");
    }

    g_signals_installed = true;
    eshkol_debug("Signal handlers installed");
}

void eshkol_runtime_restore_signals(void) {
    if (!g_signals_installed) {
        return;
    }

    sigaction(SIGINT, &g_old_sigint_handler, nullptr);
    sigaction(SIGTERM, &g_old_sigterm_handler, nullptr);
    sigaction(SIGPIPE, &g_old_sigpipe_handler, nullptr);

    g_signals_installed = false;
    eshkol_debug("Signal handlers restored");
}

// ----------------------------------------------------------------------------
// Interrupt Checking
// ----------------------------------------------------------------------------

void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason) {
    g_eshkol_interrupt_flag = 1;
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN, std::memory_order_release);
}

void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
}

eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void) {
    return g_shutdown_reason.load(std::memory_order_acquire);
}

// ----------------------------------------------------------------------------
// Shutdown Hooks
// ----------------------------------------------------------------------------

uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                        void* context,
                                        const char* name) {
    if (!hook) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_hooks_mutex);

    uint32_t id = g_next_hook_id.fetch_add(1, std::memory_order_relaxed);

    g_shutdown_hooks.push_back({
        .id = id,
        .callback = hook,
        .context = context,
        .name = name ? name : "(unnamed)"
    });

    eshkol_debug("Registered shutdown hook %u: %s", id, name ? name : "(unnamed)");
    return id;
}

bool eshkol_unregister_shutdown_hook(uint32_t hook_id) {
    if (hook_id == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_hooks_mutex);

    auto it = std::find_if(g_shutdown_hooks.begin(), g_shutdown_hooks.end(),
                           [hook_id](const ShutdownHook& h) { return h.id == hook_id; });

    if (it != g_shutdown_hooks.end()) {
        eshkol_debug("Unregistered shutdown hook %u: %s", hook_id, it->name.c_str());
        g_shutdown_hooks.erase(it);
        return true;
    }

    return false;
}

// ----------------------------------------------------------------------------
// Runtime Lifecycle
// ----------------------------------------------------------------------------

int eshkol_runtime_init(void) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_INITIALIZING;
    if (!g_runtime_state.compare_exchange_strong(expected, ESHKOL_RUNTIME_RUNNING)) {
        // Already initialized or in wrong state
        if (expected == ESHKOL_RUNTIME_RUNNING) {
            return 0;  // Already running, that's fine
        }
        eshkol_warn("Runtime init called in unexpected state: %d", (int)expected);
        return -1;
    }

    // Install signal handlers
    eshkol_runtime_init_signals();

    eshkol_info("Eshkol runtime initialized");
    return 0;
}

void eshkol_runtime_shutdown(eshkol_shutdown_reason_t reason) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_RUNNING;
    if (!g_runtime_state.compare_exchange_strong(expected, ESHKOL_RUNTIME_SHUTTING_DOWN)) {
        if (expected == ESHKOL_RUNTIME_SHUTTING_DOWN ||
            expected == ESHKOL_RUNTIME_TERMINATED) {
            // Already shutting down or terminated
            return;
        }
    }

    // Store shutdown reason
    g_shutdown_reason.store(reason, std::memory_order_release);
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

    // Wait for in-flight operations (with timeout)
    uint32_t op_count = eshkol_runtime_get_operation_count();
    if (op_count > 0) {
        eshkol_info("Waiting for %u in-flight operations to complete...", op_count);
        if (!eshkol_runtime_drain_operations(5000)) {  // 5 second timeout
            eshkol_warn("Timeout waiting for operations, proceeding with shutdown");
        }
    }

    // Call shutdown hooks in reverse registration order
    std::vector<ShutdownHook> hooks_copy;
    {
        std::lock_guard<std::mutex> lock(g_hooks_mutex);
        hooks_copy = g_shutdown_hooks;
    }

    std::reverse(hooks_copy.begin(), hooks_copy.end());

    for (const auto& hook : hooks_copy) {
        eshkol_debug("Calling shutdown hook: %s", hook.name.c_str());
        int result = hook.callback(hook.context, reason);
        if (result != 0) {
            eshkol_warn("Shutdown hook '%s' returned error: %d", hook.name.c_str(), result);
        }
    }

    // Restore signal handlers
    eshkol_runtime_restore_signals();

    // Mark as terminated
    g_runtime_state.store(ESHKOL_RUNTIME_TERMINATED, std::memory_order_release);

    eshkol_info("Shutdown complete");
}

eshkol_runtime_state_t eshkol_runtime_get_state(void) {
    return g_runtime_state.load(std::memory_order_acquire);
}

// ----------------------------------------------------------------------------
// In-flight Operation Tracking
// ----------------------------------------------------------------------------

uint32_t eshkol_runtime_begin_operation(const char* name) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);

    uint32_t id = g_next_operation_id.fetch_add(1, std::memory_order_relaxed);

    g_in_flight_operations.push_back({
        .id = id,
        .name = name ? name : "(unnamed)",
        .start_time = std::chrono::steady_clock::now()
    });

    eshkol_debug("Started operation %u: %s", id, name ? name : "(unnamed)");
    return id;
}

void eshkol_runtime_end_operation(uint32_t operation_id) {
    if (operation_id == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_operations_mutex);

    auto it = std::find_if(g_in_flight_operations.begin(), g_in_flight_operations.end(),
                           [operation_id](const InFlightOperation& op) {
                               return op.id == operation_id;
                           });

    if (it != g_in_flight_operations.end()) {
        auto duration = std::chrono::steady_clock::now() - it->start_time;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        eshkol_debug("Completed operation %u: %s (took %lld ms)",
                     operation_id, it->name.c_str(), (long long)ms);
        g_in_flight_operations.erase(it);
    }

    // Notify waiters
    g_operations_cv.notify_all();
}

bool eshkol_runtime_drain_operations(int timeout_ms) {
    std::unique_lock<std::mutex> lock(g_operations_mutex);

    if (timeout_ms == 0) {
        return g_in_flight_operations.empty();
    }

    if (timeout_ms < 0) {
        // Wait forever
        g_operations_cv.wait(lock, []() {
            return g_in_flight_operations.empty();
        });
        return true;
    }

    // Wait with timeout
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    return g_operations_cv.wait_until(lock, deadline, []() {
        return g_in_flight_operations.empty();
    });
}

uint32_t eshkol_runtime_get_operation_count(void) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);
    return static_cast<uint32_t>(g_in_flight_operations.size());
}

// ----------------------------------------------------------------------------
// Type Errors (R7RS Compliance)
// ----------------------------------------------------------------------------

void eshkol_type_error(const char* proc_name, const char* expected_type) {
    // Format: "Error in <proc>: expected <type>"
    eshkol_error("Type error in %s: expected %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>");

    // Type errors are fatal in Scheme - abort execution
    std::abort();
}

void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                   const char* actual_type) {
    // Format: "Error in <proc>: expected <type>, got <actual>"
    eshkol_error("Type error in %s: expected %s, got %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>",
                 actual_type ? actual_type : "<unknown>");

    // Type errors are fatal in Scheme - abort execution
    std::abort();
}

} // extern "C"
