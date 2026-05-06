/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime lifecycle implementation.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/logger.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstddef>
#include <csignal>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

volatile sig_atomic_t g_eshkol_interrupt_flag = 0;

namespace {

#ifdef _WIN32
constexpr int kEshkolStderrFd = 2;
constexpr DWORD kEshkolStatusBadStack = 0xC0000028UL;
using eshkol_exception_filter_t = LPTOP_LEVEL_EXCEPTION_FILTER;
#endif

std::atomic<eshkol_runtime_state_t> g_runtime_state{ESHKOL_RUNTIME_INITIALIZING};
std::atomic<eshkol_shutdown_reason_t> g_shutdown_reason{ESHKOL_SHUTDOWN_NONE};
std::atomic<eshkol_runtime_operation_drain_hook_t> g_operation_drain_hook{nullptr};
std::atomic<void*> g_operation_drain_hook_context{nullptr};

// Only these shadow values are touched in signal handlers.
volatile sig_atomic_t g_sig_runtime_state = ESHKOL_RUNTIME_INITIALIZING;
volatile sig_atomic_t g_sig_shutdown_reason = ESHKOL_SHUTDOWN_NONE;

struct ShutdownHook {
    uint32_t id;
    eshkol_shutdown_hook_t callback;
    void* context;
    std::string name;
};

std::mutex g_hooks_mutex;
std::vector<ShutdownHook> g_shutdown_hooks;
std::atomic<uint32_t> g_next_hook_id{1};

struct InFlightOperation {
    uint32_t id;
    std::string name;
    std::chrono::steady_clock::time_point start_time;
};

std::mutex g_operations_mutex;
std::condition_variable g_operations_cv;
std::vector<InFlightOperation> g_in_flight_operations;
std::atomic<uint32_t> g_next_operation_id{1};

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
#ifndef ESHKOL_SANITIZER_BUILD
struct sigaction g_old_sigsegv_handler;
struct sigaction g_old_sigbus_handler;
#endif
#endif

bool g_signals_installed = false;

inline void eshkol_signal_safe_write(const char* msg, size_t msg_len) {
#ifdef _WIN32
    (void)_write(kEshkolStderrFd, msg, static_cast<unsigned int>(msg_len));
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

    const DWORD exception_code = exception_info && exception_info->ExceptionRecord
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

#ifndef ESHKOL_SANITIZER_BUILD
void eshkol_fatal_signal_handler(int signum) {
    const char* name = "unknown";
    switch (signum) {
        case SIGSEGV:
            name = "SIGSEGV (segmentation fault)";
            break;
#ifdef SIGBUS
        case SIGBUS:
            name = "SIGBUS (bus error)";
            break;
#endif
        default:
            break;
    }

    std::fflush(stdout);
    std::fflush(stderr);

    static const char prefix[] = "\n[Eshkol] fatal signal: ";
    static const char suffix[] =
        " -- terminating; output above is what reached stdout before the crash\n";
    eshkol_signal_safe_write(prefix, sizeof(prefix) - 1);
    eshkol_signal_safe_write(name, std::strlen(name));
    eshkol_signal_safe_write(suffix, sizeof(suffix) - 1);

#ifndef _WIN32
    struct sigaction sa_dfl;
    std::memset(&sa_dfl, 0, sizeof(sa_dfl));
    sa_dfl.sa_handler = SIG_DFL;
    sigemptyset(&sa_dfl.sa_mask);
    sigaction(signum, &sa_dfl, nullptr);
#endif
    std::raise(signum);
    eshkol_signal_safe_exit(128 + signum);
}
#endif

void eshkol_signal_handler(int signum) {
    g_eshkol_interrupt_flag = 1;
    g_sig_shutdown_reason = static_cast<sig_atomic_t>(ESHKOL_SHUTDOWN_REQUESTED);

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

    eshkol_signal_safe_write(msg, msg_len);

    if (g_sig_runtime_state == static_cast<sig_atomic_t>(ESHKOL_RUNTIME_SHUTTING_DOWN)) {
        const char* force_msg = "[Eshkol] Second interrupt, forcing exit!\n";
        eshkol_signal_safe_write(force_msg, 42);
        eshkol_signal_safe_exit(128 + signum);
    }

    g_sig_runtime_state = static_cast<sig_atomic_t>(ESHKOL_RUNTIME_SHUTTING_DOWN);
}

} // namespace

extern "C" {

bool eshkol_runtime_default_monotonic_time_source(uint64_t* out_time_ns) {
    if (!out_time_ns) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    if (ns < 0) {
        return false;
    }

    *out_time_ns = static_cast<uint64_t>(ns);
    return true;
}

bool eshkol_runtime_default_delay(uint64_t duration_ns) {
    using nanoseconds_rep = std::chrono::nanoseconds::rep;

    constexpr uint64_t kMaxDelayNs =
        static_cast<uint64_t>(std::numeric_limits<nanoseconds_rep>::max());
    const uint64_t clamped_duration =
        duration_ns > kMaxDelayNs ? kMaxDelayNs : duration_ns;

    std::this_thread::sleep_for(
        std::chrono::nanoseconds(static_cast<nanoseconds_rep>(clamped_duration)));
    return true;
}

bool eshkol_runtime_profile_has_default_monotonic_clock(void) { return true; }

bool eshkol_runtime_profile_has_default_delay(void) { return true; }

bool eshkol_runtime_profile_has_operation_drain_hook_installed(void) {
    return g_operation_drain_hook.load(std::memory_order_acquire) != nullptr;
}

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
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGINT, &sa, &g_old_sigint_handler) != 0) {
        eshkol_warn("Failed to install SIGINT handler");
    }
    if (sigaction(SIGTERM, &sa, &g_old_sigterm_handler) != 0) {
        eshkol_warn("Failed to install SIGTERM handler");
    }

    struct sigaction sa_ignore;
    std::memset(&sa_ignore, 0, sizeof(sa_ignore));
    sa_ignore.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &sa_ignore, &g_old_sigpipe_handler) != 0) {
        eshkol_warn("Failed to install SIGPIPE handler");
    }

#ifndef ESHKOL_SANITIZER_BUILD
    struct sigaction sa_fatal;
    std::memset(&sa_fatal, 0, sizeof(sa_fatal));
    sa_fatal.sa_handler = eshkol_fatal_signal_handler;
    sa_fatal.sa_flags = SA_RESETHAND;
    sigemptyset(&sa_fatal.sa_mask);
    if (sigaction(SIGSEGV, &sa_fatal, &g_old_sigsegv_handler) != 0) {
        eshkol_warn("Failed to install SIGSEGV handler");
    }
#ifdef SIGBUS
    if (sigaction(SIGBUS, &sa_fatal, &g_old_sigbus_handler) != 0) {
        eshkol_warn("Failed to install SIGBUS handler");
    }
#endif
#endif
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
#ifndef ESHKOL_SANITIZER_BUILD
    sigaction(SIGSEGV, &g_old_sigsegv_handler, nullptr);
#ifdef SIGBUS
    sigaction(SIGBUS, &g_old_sigbus_handler, nullptr);
#endif
#endif
#endif

    g_signals_installed = false;
    eshkol_debug("Signal handlers restored");
}

void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason) {
    g_eshkol_interrupt_flag = 1;
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_runtime_state.store(ESHKOL_RUNTIME_SHUTTING_DOWN, std::memory_order_release);
    g_sig_shutdown_reason = static_cast<sig_atomic_t>(reason);
    g_sig_runtime_state = static_cast<sig_atomic_t>(ESHKOL_RUNTIME_SHUTTING_DOWN);
}

void eshkol_runtime_clear_interrupt(void) {
    g_eshkol_interrupt_flag = 0;
    g_shutdown_reason.store(ESHKOL_SHUTDOWN_NONE, std::memory_order_release);
    g_sig_shutdown_reason = static_cast<sig_atomic_t>(ESHKOL_SHUTDOWN_NONE);
}

void eshkol_runtime_set_operation_drain_hook(
    eshkol_runtime_operation_drain_hook_t hook,
    void* context) {
    g_operation_drain_hook_context.store(context, std::memory_order_relaxed);
    g_operation_drain_hook.store(hook, std::memory_order_release);
}

void eshkol_runtime_clear_operation_drain_hook(void) {
    g_operation_drain_hook.store(nullptr, std::memory_order_release);
    g_operation_drain_hook_context.store(nullptr, std::memory_order_relaxed);
}

eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void) {
    return g_shutdown_reason.load(std::memory_order_acquire);
}

uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                       void* context,
                                       const char* name) {
    if (!hook) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_hooks_mutex);
    const uint32_t id = g_next_hook_id.fetch_add(1, std::memory_order_relaxed);
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
    const auto it = std::find_if(g_shutdown_hooks.begin(), g_shutdown_hooks.end(),
                                 [hook_id](const ShutdownHook& hook) { return hook.id == hook_id; });
    if (it == g_shutdown_hooks.end()) {
        return false;
    }

    eshkol_debug("Unregistered shutdown hook %u: %s", hook_id, it->name.c_str());
    g_shutdown_hooks.erase(it);
    return true;
}

int eshkol_runtime_init(void) {
    eshkol_runtime_state_t expected = ESHKOL_RUNTIME_INITIALIZING;
    if (!g_runtime_state.compare_exchange_strong(expected, ESHKOL_RUNTIME_RUNNING)) {
        if (expected == ESHKOL_RUNTIME_RUNNING) {
            return 0;
        }
        eshkol_warn("Runtime init called in unexpected state: %d", static_cast<int>(expected));
        return -1;
    }

    g_sig_runtime_state = static_cast<sig_atomic_t>(ESHKOL_RUNTIME_RUNNING);
    setvbuf(stdout, NULL, _IONBF, 0);
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

    g_sig_runtime_state = static_cast<sig_atomic_t>(ESHKOL_RUNTIME_SHUTTING_DOWN);
    g_shutdown_reason.store(reason, std::memory_order_release);
    g_sig_shutdown_reason = static_cast<sig_atomic_t>(reason);
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

    const uint32_t operation_count = eshkol_runtime_get_operation_count();
    if (operation_count > 0) {
        eshkol_info("Waiting for %u in-flight operations to complete...", operation_count);
        if (!eshkol_runtime_drain_operations(5000)) {
            eshkol_warn("Timeout waiting for operations, proceeding with shutdown");
        }
    }

    std::vector<ShutdownHook> hooks_copy;
    {
        std::lock_guard<std::mutex> lock(g_hooks_mutex);
        hooks_copy = g_shutdown_hooks;
    }

    std::reverse(hooks_copy.begin(), hooks_copy.end());
    for (const auto& hook : hooks_copy) {
        eshkol_debug("Calling shutdown hook: %s", hook.name.c_str());
        const int result = hook.callback(hook.context, reason);
        if (result != 0) {
            eshkol_warn("Shutdown hook '%s' returned error: %d", hook.name.c_str(), result);
        }
    }

    eshkol_runtime_restore_signals();
    g_runtime_state.store(ESHKOL_RUNTIME_TERMINATED, std::memory_order_release);
    g_sig_runtime_state = static_cast<sig_atomic_t>(ESHKOL_RUNTIME_TERMINATED);
    eshkol_info("Shutdown complete");
}

eshkol_runtime_state_t eshkol_runtime_get_state(void) {
    return g_runtime_state.load(std::memory_order_acquire);
}

uint32_t eshkol_runtime_begin_operation(const char* name) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);

    const uint32_t id = g_next_operation_id.fetch_add(1, std::memory_order_relaxed);
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
    const auto it = std::find_if(g_in_flight_operations.begin(), g_in_flight_operations.end(),
                                 [operation_id](const InFlightOperation& operation) {
                                     return operation.id == operation_id;
                                 });
    if (it != g_in_flight_operations.end()) {
        const auto duration = std::chrono::steady_clock::now() - it->start_time;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        eshkol_debug("Completed operation %u: %s (took %lld ms)",
                     operation_id, it->name.c_str(), static_cast<long long>(elapsed_ms));
        g_in_flight_operations.erase(it);
    }

    g_operations_cv.notify_all();
}

bool eshkol_runtime_drain_operations(int timeout_ms) {
    std::unique_lock<std::mutex> lock(g_operations_mutex);

    if (timeout_ms == 0) {
        return g_in_flight_operations.empty();
    }
    if (timeout_ms < 0) {
        g_operations_cv.wait(lock, []() { return g_in_flight_operations.empty(); });
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    return g_operations_cv.wait_until(lock, deadline, []() {
        return g_in_flight_operations.empty();
    });
}

uint32_t eshkol_runtime_get_operation_count(void) {
    std::lock_guard<std::mutex> lock(g_operations_mutex);
    return static_cast<uint32_t>(g_in_flight_operations.size());
}

} // extern "C"
