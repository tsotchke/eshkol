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
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>    // STDERR_FILENO, write(), _exit()
#endif
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <cstdio>

// ----------------------------------------------------------------------------
// Fatal error helper
// ----------------------------------------------------------------------------
//
// Used by runtime functions that detect a non-recoverable user error (type
// error, OOB index, OOM, null bytevector, etc). Constructs an exception
// object and calls eshkol_raise() so any installed `with-exception-handler`
// can intercept it. If no handler is installed, eshkol_raise exits the
// process (we still print to stderr first so embedding hosts see the
// message even if eshkol_raise's exit path is muted).
//
// Replaces the prior pattern of `fprintf(stderr, ...); std::abort();` which
// bypassed the exception handler entirely and produced SIGABRT in embedding
// scenarios.
static void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    fprintf(stderr, "%s\n", buf);

    eshkol_exception_t* exc = eshkol_make_exception(type, buf);
    if (exc) {
        eshkol_raise(exc);
        // eshkol_raise normally does not return. If it does (very rare —
        // means the handler ran a longjmp back out and returned cleanly),
        // we still must not proceed. Fall through to exit(1).
    }
    std::exit(1);
}

// ============================================================================
// Global State
// ============================================================================

// Volatile flag for signal-safe interrupt checking
volatile sig_atomic_t g_eshkol_interrupt_flag = 0;

namespace {

#ifdef _WIN32
constexpr int kEshkolStderrFd = 2;
constexpr DWORD kEshkolStatusBadStack = 0xC0000028UL;
using eshkol_exception_filter_t = LPTOP_LEVEL_EXCEPTION_FILTER;
#endif

// Runtime state (std::atomic for thread-safe API outside signal handlers)
std::atomic<eshkol_runtime_state_t> g_runtime_state{ESHKOL_RUNTIME_INITIALIZING};
std::atomic<eshkol_shutdown_reason_t> g_shutdown_reason{ESHKOL_SHUTDOWN_NONE};

// Signal-safe shadow variables: std::atomic is NOT guaranteed async-signal-safe
// in C++. These volatile sig_atomic_t variables are the ONLY ones the signal
// handler reads/writes. Normal API functions update both atomics and shadows.
volatile sig_atomic_t g_sig_runtime_state = ESHKOL_RUNTIME_INITIALIZING;
volatile sig_atomic_t g_sig_shutdown_reason = ESHKOL_SHUTDOWN_NONE;

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
#ifdef _WIN32
using eshkol_signal_handler_t = void (*)(int);
eshkol_signal_handler_t g_old_sigint_handler = SIG_DFL;
eshkol_signal_handler_t g_old_sigterm_handler = SIG_DFL;
eshkol_exception_filter_t g_old_exception_filter = nullptr;
std::atomic<bool> g_handling_unhandled_exception{false};
#ifdef SIGPIPE
eshkol_signal_handler_t g_old_sigpipe_handler = SIG_DFL;
#endif
#else
struct sigaction g_old_sigint_handler;
struct sigaction g_old_sigterm_handler;
struct sigaction g_old_sigpipe_handler;
#endif
bool g_signals_installed = false;

inline void eshkol_signal_safe_write(const char* msg, size_t msg_len) {
#ifdef _WIN32
    (void)_write(kEshkolStderrFd, msg, (unsigned int)msg_len);
#else
    (void)write(STDERR_FILENO, msg, msg_len);
#endif
}

[[noreturn]] inline void eshkol_signal_safe_exit(int code) {
    _exit(code);
}

#ifdef _WIN32
const char* eshkol_exception_code_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION: return "EXCEPTION_ACCESS_VIOLATION";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "EXCEPTION_ARRAY_BOUNDS_EXCEEDED";
        case EXCEPTION_BREAKPOINT: return "EXCEPTION_BREAKPOINT";
        case EXCEPTION_DATATYPE_MISALIGNMENT: return "EXCEPTION_DATATYPE_MISALIGNMENT";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO: return "EXCEPTION_FLT_DIVIDE_BY_ZERO";
        case EXCEPTION_ILLEGAL_INSTRUCTION: return "EXCEPTION_ILLEGAL_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR: return "EXCEPTION_IN_PAGE_ERROR";
        case EXCEPTION_INT_DIVIDE_BY_ZERO: return "EXCEPTION_INT_DIVIDE_BY_ZERO";
        case EXCEPTION_STACK_OVERFLOW: return "EXCEPTION_STACK_OVERFLOW";
        case kEshkolStatusBadStack: return "STATUS_BAD_STACK";
        default: return "UNKNOWN_EXCEPTION";
    }
}

LONG WINAPI eshkol_unhandled_exception_filter(EXCEPTION_POINTERS* exception_info) {
    bool expected = false;
    if (!g_handling_unhandled_exception.compare_exchange_strong(expected, true)) {
        TerminateProcess(GetCurrentProcess(), exception_info && exception_info->ExceptionRecord
            ? exception_info->ExceptionRecord->ExceptionCode
            : 1);
    }

    DWORD exception_code = exception_info && exception_info->ExceptionRecord
        ? exception_info->ExceptionRecord->ExceptionCode
        : 1;
    void* exception_address = exception_info && exception_info->ExceptionRecord
        ? exception_info->ExceptionRecord->ExceptionAddress
        : nullptr;

    std::fprintf(stderr,
        "\n[Eshkol] Unhandled Windows exception 0x%08lX (%s) at %p\n",
        static_cast<unsigned long>(exception_code),
        eshkol_exception_code_name(exception_code),
        exception_address);
    eshkol_stacktrace(ESHKOL_ERROR);
    eshkol_log_flush();
    std::fflush(stderr);

    TerminateProcess(GetCurrentProcess(), exception_code);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// ============================================================================
// Signal Handler
// ============================================================================

void eshkol_signal_handler(int signum) {
    // Set the volatile flag (signal-safe: volatile sig_atomic_t)
    g_eshkol_interrupt_flag = 1;

    // Store shutdown reason using signal-safe variable only
    g_sig_shutdown_reason = (sig_atomic_t)ESHKOL_SHUTDOWN_REQUESTED;

    // Log message (use only async-signal-safe functions)
    // write() is async-signal-safe, fprintf is NOT
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
    eshkol_signal_safe_write(msg, msg_len);

    // If we get a second signal during shutdown, force exit
    // Read from volatile sig_atomic_t (NOT std::atomic — not async-signal-safe)
    if (g_sig_runtime_state == (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN) {
        const char* force_msg = "[Eshkol] Second interrupt, forcing exit!\n";
        eshkol_signal_safe_write(force_msg, 42);
        eshkol_signal_safe_exit(128 + signum);  // Standard Unix exit code for signal termination
    }

    // Mark as shutting down (signal-safe variable)
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN;
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

#ifdef _WIN32
    g_old_sigint_handler = std::signal(SIGINT, eshkol_signal_handler);
    if (g_old_sigint_handler == SIG_ERR) {
        eshkol_warn("Failed to install SIGINT handler");
        g_old_sigint_handler = SIG_DFL;
    }

    g_old_sigterm_handler = std::signal(SIGTERM, eshkol_signal_handler);
    if (g_old_sigterm_handler == SIG_ERR) {
        eshkol_warn("Failed to install SIGTERM handler");
        g_old_sigterm_handler = SIG_DFL;
    }

#ifdef SIGPIPE
    g_old_sigpipe_handler = std::signal(SIGPIPE, SIG_IGN);
    if (g_old_sigpipe_handler == SIG_ERR) {
        eshkol_warn("Failed to install SIGPIPE handler");
        g_old_sigpipe_handler = SIG_DFL;
    }
#endif

    g_old_exception_filter = SetUnhandledExceptionFilter(eshkol_unhandled_exception_filter);
#else
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
#endif

    g_signals_installed = true;
    eshkol_debug("Signal handlers installed");
}

void eshkol_runtime_restore_signals(void) {
    if (!g_signals_installed) {
        return;
    }

#ifdef _WIN32
    std::signal(SIGINT, g_old_sigint_handler);
    std::signal(SIGTERM, g_old_sigterm_handler);
    SetUnhandledExceptionFilter(g_old_exception_filter);
    g_old_exception_filter = nullptr;
    g_handling_unhandled_exception.store(false, std::memory_order_release);
#ifdef SIGPIPE
    std::signal(SIGPIPE, g_old_sigpipe_handler);
#endif
#else
    sigaction(SIGINT, &g_old_sigint_handler, nullptr);
    sigaction(SIGTERM, &g_old_sigterm_handler, nullptr);
    sigaction(SIGPIPE, &g_old_sigpipe_handler, nullptr);
#endif

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
    // Update signal-safe shadow variables
    g_sig_shutdown_reason = (sig_atomic_t)reason;
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN;
}

void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
    // Update signal-safe shadow variables
    g_sig_shutdown_reason = (sig_atomic_t)ESHKOL_SHUTDOWN_NONE;
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
    // Sync signal-safe shadow
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_RUNNING;

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
    // Sync signal-safe shadow
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN;

    // Store shutdown reason
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_sig_shutdown_reason = (sig_atomic_t)reason;
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
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_TERMINATED;

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

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "Type error in %s: expected %s",
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>");
}

void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                   const char* actual_type) {
    // Format: "Error in <proc>: expected <type>, got <actual>"
    eshkol_error("Type error in %s: expected %s, got %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>",
                 actual_type ? actual_type : "<unknown>");

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "Type error in %s: expected %s, got %s",
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>",
                         actual_type ? actual_type : "<unknown>");
}

// ----------------------------------------------------------------------------
// Parameter Objects (R7RS make-parameter / parameterize)
// ----------------------------------------------------------------------------

// Forward declaration for arena allocation with GC-tracked header
extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

#define HEAP_SUBTYPE_PARAMETER 20

// Parameter object: a stack of tagged values implementing dynamic binding.
// The bottom of the stack (index 0) holds the default value.
// Each parameterize pushes a new value; exiting parameterize pops it.
typedef struct {
    eshkol_tagged_value_t* stack;
    int top;
    int capacity;
} eshkol_param_t;

// Create a new parameter object with the given default value.
// The parameter struct is arena-allocated (GC-tracked); the internal
// value stack uses malloc since it may need to grow.
void* eshkol_make_parameter(void* arena, eshkol_tagged_value_t default_val) {
    eshkol_param_t* param = (eshkol_param_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_param_t), HEAP_SUBTYPE_PARAMETER, 0);
    if (!param) {
        return nullptr;
    }

    const int initial_capacity = 8;
    param->stack = (eshkol_tagged_value_t*)std::malloc(
        initial_capacity * sizeof(eshkol_tagged_value_t));
    if (!param->stack) {
        // Allocation failure — return the param with no stack
        // (eshkol_parameter_ref will return null tagged value)
        param->top = -1;
        param->capacity = 0;
        return (void*)param;
    }

    param->capacity = initial_capacity;
    param->top = 0;
    param->stack[0] = default_val;
    return (void*)param;
}

// Push a new binding onto the parameter's value stack.
// Called at entry to a parameterize block.
void eshkol_parameter_push(void* param_ptr, eshkol_tagged_value_t val) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    // Grow stack if at capacity
    if (param->top + 1 >= param->capacity) {
        int new_capacity = param->capacity * 2;
        if (new_capacity < 8) new_capacity = 8;
        eshkol_tagged_value_t* new_stack = (eshkol_tagged_value_t*)std::realloc(
            param->stack, new_capacity * sizeof(eshkol_tagged_value_t));
        if (!new_stack) {
            // OOM on parameter-stack growth. Crashing mid-dynamic-bind
            // would unwind through every `parameterize` form and leak
            // the old stack, so we keep the current binding — but log
            // loudly now rather than silently drop the push (which
            // looked to the caller like `parameterize` had worked).
            // #181 error-propagation follow-up.
            eshkol_warn("parameter-push: realloc(%d -> %d entries) failed; "
                        "new binding dropped, previous value remains",
                        param->capacity, new_capacity);
            return;
        }
        param->stack = new_stack;
        param->capacity = new_capacity;
    }

    param->top++;
    param->stack[param->top] = val;
}

// Pop the most recent binding from the parameter's value stack.
// Called at exit from a parameterize block (including unwinding).
// Never pops below index 0 — the default value is always preserved.
void eshkol_parameter_pop(void* param_ptr) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    // Never pop below 1 — always keep the default value at index 0
    if (param->top > 0) {
        param->top--;
    }
}

// Return the current (topmost) binding of the parameter.
// If the parameter has no stack (allocation failure), returns ESHKOL_VALUE_NULL.
eshkol_tagged_value_t eshkol_parameter_ref(void* param_ptr) {
    if (!param_ptr) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top < 0 || !param->stack) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }

    return param->stack[param->top];
}

// Pointer-based wrappers for LLVM codegen (avoids by-value struct ABI issues)
void* eshkol_make_parameter_ptr(void* arena, const eshkol_tagged_value_t* default_val) {
    if (!default_val) return nullptr;
    return eshkol_make_parameter(arena, *default_val);
}

void eshkol_parameter_push_ptr(void* param, const eshkol_tagged_value_t* val) {
    if (!val) return;
    eshkol_parameter_push(param, *val);
}

void eshkol_parameter_ref_ptr(void* param, eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_ref(param);
}

// ----------------------------------------------------------------------------
// Bytevector Runtime (R7RS)
// ----------------------------------------------------------------------------
// Layout: ptr[-8] = header (subtype byte), ptr[0..7] = int64_t length, ptr[8..] = byte data
// HEAP_SUBTYPE_BYTEVECTOR = 8 (defined in eshkol/eshkol.h)

/* Forward declaration for arena allocation with header */
extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

/* Create a new bytevector of given length, filled with fill_byte */
void* eshkol_make_bytevector(void* arena, int64_t len, int64_t fill_byte) {
    if (!arena || len < 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in make-bytevector: invalid arguments (len=%lld)",
                             (long long)len);
    }

    // Allocate: 8 bytes for length + len bytes for data
    uint64_t data_size = (uint64_t)(8 + len);
    void* ptr = arena_allocate_with_header(arena, data_size, 8 /* HEAP_SUBTYPE_BYTEVECTOR */, 0);
    if (!ptr) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in make-bytevector: out of memory (len=%lld)",
                             (long long)len);
    }

    // Store length at ptr[0..7]
    int64_t* length_ptr = (int64_t*)ptr;
    *length_ptr = len;

    // Fill byte data at ptr[8..]
    uint8_t* data = (uint8_t*)ptr + 8;
    uint8_t fill = (uint8_t)(fill_byte & 0xFF);
    memset(data, fill, (size_t)len);

    return ptr;
}

/* Get byte at index k (returns int64_t for tagged value compatibility) */
int64_t eshkol_bytevector_u8_ref(void* bv, int64_t k) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-u8-ref: null bytevector");
    }

    int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-ref: index %lld out of range [0, %lld)",
                             (long long)k, (long long)len);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    return (int64_t)data[k];
}

/* Set byte at index k */
void eshkol_bytevector_u8_set(void* bv, int64_t k, int64_t byte_val) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-u8-set!: null bytevector");
    }

    int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-set!: index %lld out of range [0, %lld)",
                             (long long)k, (long long)len);
    }

    if (byte_val < 0 || byte_val > 255) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-set!: byte value %lld out of range [0, 255]",
                             (long long)byte_val);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    data[k] = (uint8_t)byte_val;
}

/* Get length of bytevector */
int64_t eshkol_bytevector_length(void* bv) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-length: null bytevector");
    }

    return *((int64_t*)bv);
}

/* Copy bytevector (or sub-range) into a new arena-allocated bytevector */
void* eshkol_bytevector_copy(void* arena, void* bv, int64_t start, int64_t end) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: null bytevector");
    }
    if (!arena) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: null arena");
    }

    int64_t len = *((int64_t*)bv);

    // Clamp end to length if -1 (sentinel for "copy to end")
    if (end < 0) end = len;

    if (start < 0 || start > len || end < start || end > len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-copy: range [%lld, %lld) out of bounds [0, %lld)",
                             (long long)start, (long long)end, (long long)len);
    }

    int64_t copy_len = end - start;

    // Allocate new bytevector: 8 bytes for length + copy_len bytes for data
    uint64_t data_size = (uint64_t)(8 + copy_len);
    void* new_bv = arena_allocate_with_header(arena, data_size, 8 /* HEAP_SUBTYPE_BYTEVECTOR */, 0);
    if (!new_bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: out of memory (len=%lld)",
                             (long long)copy_len);
    }

    // Store length
    *((int64_t*)new_bv) = copy_len;

    // Copy byte data
    if (copy_len > 0) {
        uint8_t* src_data = (uint8_t*)bv + 8 + start;
        uint8_t* dst_data = (uint8_t*)new_bv + 8;
        memcpy(dst_data, src_data, (size_t)copy_len);
    }

    return new_bv;
}

// ----------------------------------------------------------------------------
// UTF-8 Helpers
// ----------------------------------------------------------------------------

/*
 * All Scheme string values in Eshkol live behind an 8-byte
 * ESHKOL_OBJECT_HEADER whose `size` field stores the data-byte count
 * (data = chars + NUL terminator, so the visible payload is size - 1
 * bytes). That header is created by both string literals
 * (`internStringWithHeader`) and runtime allocations
 * (`arena_allocate_string_with_header`), so every string passed from
 * LLVM IR to these helpers has it.
 *
 * Using the header gives correct length for strings that contain
 * embedded NUL bytes (e.g. `(make-string 4096 #\nul)` for FFI scratch
 * buffers in lib/agent/regex.esk). Without it, we'd fall back to
 * strlen semantics which stops at the first NUL and reports 0 — which
 * is why `(string-length (make-string N #\nul))` was returning 0
 * instead of N.
 *
 * Fallback: a null/non-header pointer still uses strlen semantics so
 * callers passing a bare C string (none in tree, but defensive)
 * continue to work.
 */
static int64_t eshkol_string_byte_length(const char* s) {
    if (!s) return 0;
    /* Sanity: the header sits at s-8 and its subtype byte is the first
     * byte of the 8-byte header. Validate it's HEAP_SUBTYPE_STRING
     * before trusting size. */
    const eshkol_object_header_t* hdr =
        (const eshkol_object_header_t*)((const uint8_t*)s - sizeof(eshkol_object_header_t));
    if (hdr->subtype == HEAP_SUBTYPE_STRING && hdr->size > 0) {
        /* size = data_size including NUL terminator. */
        return (int64_t)hdr->size - 1;
    }
    /* No header: fall back to C strlen. */
    return (int64_t)strlen(s);
}

/* Count UTF-8 codepoints over the full byte range (includes bytes
 * after embedded NULs, unlike plain strlen). */
int64_t eshkol_utf8_strlen(const char* s) {
    if (!s) return 0;
    int64_t byte_len = eshkol_string_byte_length(s);
    int64_t count = 0;
    for (int64_t i = 0; i < byte_len; i++) {
        if ((s[i] & 0xC0) != 0x80) count++;
    }
    return count;
}

/* Decode a single UTF-8 codepoint at s, advance *s past it */
static int64_t decode_utf8_codepoint(const char** s) {
    const unsigned char* p = (const unsigned char*)*s;
    int64_t cp;
    if (*p < 0x80) { cp = *p; *s += 1; }
    else if ((*p & 0xE0) == 0xC0) { cp = (*p & 0x1F) << 6 | (p[1] & 0x3F); *s += 2; }
    else if ((*p & 0xF0) == 0xE0) { cp = (*p & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F); *s += 3; }
    else if ((*p & 0xF8) == 0xF0) { cp = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 | (p[2] & 0x3F) << 6 | (p[3] & 0x3F); *s += 4; }
    else { cp = 0xFFFD; *s += 1; } /* replacement char for invalid */
    return cp;
}

/* Get k-th codepoint from string, walking the full header-backed byte
 * range so embedded NULs don't truncate the walk early. */
int64_t eshkol_utf8_ref(const char* s, int64_t k) {
    if (!s || k < 0) return -1;
    int64_t byte_len = eshkol_string_byte_length(s);
    int64_t cp_idx = 0;
    int64_t i = 0;
    while (i < byte_len && cp_idx < k) {
        if ((s[i] & 0xC0) != 0x80) cp_idx++;
        i++;
    }
    if (i >= byte_len) return -1;
    const char* p = s + i;
    return decode_utf8_codepoint(&p);
}

/* Substring by codepoint indices → new arena-allocated string.
 * Walks the full header-backed byte range (not strlen-limited) so
 * strings containing embedded NULs substring correctly. */
char* eshkol_utf8_substring(const char* s, int64_t start, int64_t end, void* arena) {
    if (!s || !arena || start < 0 || end < start) return nullptr;
    extern void* arena_allocate_string_with_header(void*, uint64_t);
    int64_t byte_len_total = eshkol_string_byte_length(s);
    /* Bug H (2026-04-20): advance by FULL codepoints each step —
     * skip all continuation bytes of each codepoint as a unit —
     * so start_off lands on the first byte of the start-th
     * codepoint and end i lands one-past the last byte of the
     * (end-1)-th codepoint.
     *
     * The previous `cp_idx++` was off by one codepoint's
     * continuation-byte tail on both start and end:
     *
     *   "⟨ab" (e2 9f a9 61 62) substring(1,3):
     *     WANT: bytes 3..4 = "ab"
     *     OLD:  start_off=1 (inside ⟨'s continuation bytes),
     *           end stopped at i=3 (just before 'a' — so byte_len=2)
     *
     *   "ab⟩" (61 62 e2 9f a9) substring(1,3):
     *     WANT: bytes 1..4 = "b⟩"
     *     OLD:  start_off=1 (correct by luck — only ASCII before),
     *           end stopped at i=3 (on start of ⟩'s cont bytes), byte_len=2 */
    auto advance_one_codepoint = [&](int64_t& i) {
        if (i >= byte_len_total) return;
        unsigned char b = (unsigned char)s[i];
        if ((b & 0x80) == 0)          i += 1;   /* ASCII */
        else if ((b & 0xE0) == 0xC0)  i += 2;   /* 2-byte */
        else if ((b & 0xF0) == 0xE0)  i += 3;   /* 3-byte */
        else if ((b & 0xF8) == 0xF0)  i += 4;   /* 4-byte */
        else                          i += 1;   /* malformed — best effort */
        /* Clamp — truncated final codepoint. */
        if (i > byte_len_total) i = byte_len_total;
    };

    int64_t i = 0;
    int64_t cp_idx = 0;
    while (i < byte_len_total && cp_idx < start) {
        advance_one_codepoint(i);
        cp_idx++;
    }
    int64_t start_off = i;
    while (i < byte_len_total && cp_idx < end) {
        advance_one_codepoint(i);
        cp_idx++;
    }
    int64_t byte_len = i - start_off;
    /* allocator adds +1 for NUL itself — pass the payload byte count. */
    char* buf = (char*)arena_allocate_string_with_header(arena, byte_len);
    if (buf) {
        memcpy(buf, s + start_off, byte_len);
        buf[byte_len] = '\0';
    }
    return buf;
}

/*
 * Normalise a tensor-ref index argument to a raw int64.
 *
 * Callers write `(tensor-ref t i)` with either a literal integer index or,
 * the NumPy / JAX / Noesis-Sigma idiom, `(tensor-ref t (list i))`. The
 * second form passes a cons cell; the old scalar-int path read its
 * pointer bits as an integer and GEP'd at a wild address — SEGV. This
 * helper is called from the LLVM codegen for tensor-ref/vref to produce
 * a scalar int in either case.
 *
 * Rules:
 *   - tv.type == ESHKOL_VALUE_HEAP_PTR and the object at tv.data is a
 *     cons cell → return (int)car. Car is itself a tagged_value (int or
 *     double); we extract the scalar consistently with what the scalar
 *     path below would have done.
 *   - tv.type == ESHKOL_VALUE_INT64 → tv.data is already the int.
 *   - tv.type == ESHKOL_VALUE_DOUBLE → cast the double to int64.
 *   - anything else → return tv.data as-is (back-compat with the old
 *     safeExtractInt64 behaviour; keeps the AD-aware tensor path happy
 *     since that path does its own type-tag dispatch downstream).
 */
int64_t eshkol_unwrap_list_index(const eshkol_tagged_value_t* tv_in) {
    if (!tv_in) return 0;
    eshkol_tagged_value_t tv = *tv_in;
    uint8_t base_type = tv.type;
    /* Masks for exactness-flag types (0-7); types >= 8 are not immediate. */
    if (base_type < 8) base_type &= 0x0F;

    if (base_type == ESHKOL_VALUE_HEAP_PTR && tv.data.ptr_val != 0) {
        const eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER((void*)tv.data.ptr_val);
        if (hdr->subtype == HEAP_SUBTYPE_CONS) {
            /* Cons cell layout is arena_tagged_cons_cell_t — extract car.
             * Use the public accessor rather than struct layout so future
             * rearrangements don't silently break this path. */
            extern eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(
                const void* cell, bool is_cdr);
            eshkol_tagged_value_t car =
                arena_tagged_cons_get_tagged_value((const void*)tv.data.ptr_val, false);
            uint8_t car_base = car.type;
            if (car_base < 8) car_base &= 0x0F;
            if (car_base == ESHKOL_VALUE_INT64) return (int64_t)car.data.int_val;
            if (car_base == ESHKOL_VALUE_DOUBLE) {
                double d;
                memcpy(&d, &car.data, sizeof(double));
                return (int64_t)d;
            }
            return (int64_t)car.data.int_val;
        }
    }
    if (base_type == ESHKOL_VALUE_INT64) return (int64_t)tv.data.int_val;
    if (base_type == ESHKOL_VALUE_DOUBLE) {
        double d;
        memcpy(&d, &tv.data, sizeof(double));
        return (int64_t)d;
    }
    return (int64_t)tv.data.int_val;
}

} // extern "C"
