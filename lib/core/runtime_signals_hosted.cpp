/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted signal and exception handler installation for runtime shutdown.
 */

#include "runtime_hosted_internal.h"

#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>    // STDERR_FILENO, write(), _exit()
#endif

#include <atomic>
#include <cstddef>
#include <csignal>
#include <cstdio>
#include <cstring>

namespace {

#ifdef _WIN32
constexpr int kEshkolStderrFd = 2;
constexpr DWORD kEshkolStatusBadStack = 0xC0000028UL;
using eshkol_exception_filter_t = LPTOP_LEVEL_EXCEPTION_FILTER;
#endif

// Signal-safe shadow variables: std::atomic is NOT guaranteed async-signal-safe
// in C++. These volatile sig_atomic_t variables are the ONLY ones the signal
// handler reads/writes. Normal API functions update both atomics and shadows.
volatile sig_atomic_t g_sig_runtime_state = ESHKOL_RUNTIME_INITIALIZING;
volatile sig_atomic_t g_sig_shutdown_reason = ESHKOL_SHUTDOWN_NONE;

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
struct sigaction g_old_sigsegv_handler;
struct sigaction g_old_sigbus_handler;
struct sigaction g_old_sigabrt_handler;
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

/* Bug AA: handler for fatal synchronous signals (SIGSEGV/SIGBUS/SIGABRT).
 * The default action for these is to terminate without flushing any stdio
 * buffers, which means a `(display "step 1") (newline)` issued before a crash
 * never reaches the terminal.
 *
 * Strategy: write a minimal one-line diagnostic to stderr, restore the default
 * disposition for this signal, and re-raise it so the kernel produces the
 * normal coredump / 128+signum exit code.
 */
void eshkol_fatal_signal_handler(int signum) {
    const char* name = "unknown";
    switch (signum) {
        case SIGSEGV: name = "SIGSEGV (segmentation fault)"; break;
#ifdef SIGBUS
        case SIGBUS:  name = "SIGBUS (bus error)";           break;
#endif
        case SIGABRT: name = "SIGABRT (abort)";              break;
        default: break;
    }
    /* Best-effort flush of stdout/stderr. fflush() is technically not on
     * POSIX's async-signal-safe list; in practice it works on every libc we
     * run on when the process is otherwise quiescent. */
    std::fflush(stdout);
    std::fflush(stderr);

    static const char prefix[] = "\n[Eshkol] fatal signal: ";
    static const char suffix[] = " — terminating; output above is what made it to stdout before the crash\n";
    eshkol_signal_safe_write(prefix, sizeof(prefix) - 1);
    eshkol_signal_safe_write(name, std::strlen(name));
    eshkol_signal_safe_write(suffix, sizeof(suffix) - 1);

#ifndef _WIN32
    // Restore default disposition and re-raise so the process exits with
    // the canonical 128+signum code (and dumps core if so configured).
    struct sigaction sa_dfl;
    std::memset(&sa_dfl, 0, sizeof(sa_dfl));
    sa_dfl.sa_handler = SIG_DFL;
    sigemptyset(&sa_dfl.sa_mask);
    sigaction(signum, &sa_dfl, nullptr);
#endif
    std::raise(signum);
    // Unreachable, but defensive.
    eshkol_signal_safe_exit(128 + signum);
}

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
    // Read from volatile sig_atomic_t (NOT std::atomic - not async-signal-safe)
    if (g_sig_runtime_state == (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN) {
        const char* force_msg = "[Eshkol] Second interrupt, forcing exit!\n";
        eshkol_signal_safe_write(force_msg, 42);
        eshkol_signal_safe_exit(128 + signum);
    }

    // Mark as shutting down (signal-safe variable)
    g_sig_runtime_state = (sig_atomic_t)ESHKOL_RUNTIME_SHUTTING_DOWN;
}

}  // namespace

extern "C" {

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

    // Bug AA: install fatal-signal handlers that announce themselves on stderr
    // before terminating, so the user knows the program crashed.
    struct sigaction sa_fatal;
    std::memset(&sa_fatal, 0, sizeof(sa_fatal));
    sa_fatal.sa_handler = eshkol_fatal_signal_handler;
    sa_fatal.sa_flags = SA_RESETHAND;
    sigemptyset(&sa_fatal.sa_mask);
    if (sigaction(SIGSEGV, &sa_fatal, &g_old_sigsegv_handler) != 0) {
        eshkol_warn("Failed to install SIGSEGV handler");
    }
#ifdef SIGBUS
    if (sigaction(SIGBUS,  &sa_fatal, &g_old_sigbus_handler)  != 0) {
        eshkol_warn("Failed to install SIGBUS handler");
    }
#endif
    /* Note: we deliberately do NOT install a handler for SIGABRT. The Bug AA
     * fix is about not losing user output when an Eshkol program crashes with
     * SIGSEGV/SIGBUS. SIGABRT can be raised by libsystem during process
     * teardown; handling it here can turn clean shutdown into a spurious abort.
     */
    (void)g_old_sigabrt_handler;
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
    sigaction(SIGSEGV, &g_old_sigsegv_handler, nullptr);
#ifdef SIGBUS
    sigaction(SIGBUS,  &g_old_sigbus_handler,  nullptr);
#endif
    sigaction(SIGABRT, &g_old_sigabrt_handler, nullptr);
#endif

    g_signals_installed = false;
    eshkol_debug("Signal handlers restored");
}

}  // extern "C"

namespace eshkol::runtime_hosted {

void set_signal_runtime_state(eshkol_runtime_state_t state) {
    g_sig_runtime_state = (sig_atomic_t)state;
}

void set_signal_shutdown_reason(eshkol_shutdown_reason_t reason) {
    g_sig_shutdown_reason = (sig_atomic_t)reason;
}

}  // namespace eshkol::runtime_hosted
