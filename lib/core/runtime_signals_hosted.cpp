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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Detect compilation under a sanitizer. Sanitizers (ASan/TSan/MSan) install
// their own fatal-signal handlers and their own alternate signal stack; if we
// register on top of them we either fight their diagnostics or double-register.
// Under a sanitizer we leave the fault signals (SIGSEGV/SIGBUS/SIGILL/SIGFPE)
// to the sanitizer and only manage the graceful SIGINT/SIGTERM path.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
      __has_feature(memory_sanitizer)
#    define ESHKOL_UNDER_SANITIZER 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#  define ESHKOL_UNDER_SANITIZER 1
#endif

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
struct sigaction g_old_sigill_handler;
struct sigaction g_old_sigfpe_handler;

// Alternate signal stack so the fatal-signal handler can run even when the
// normal stack has been exhausted by deep recursion. Without SA_ONSTACK, a
// stack-overflow SIGSEGV/SIGILL cannot invoke a handler that itself needs
// stack, so the diagnostic never prints and the process dies silently.
void* g_altstack_mem = nullptr;
stack_t g_altstack_prev;          // previous altstack (for restoration)
bool g_altstack_installed = false;  // true only if WE installed the altstack
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

#ifndef _WIN32
// Async-signal-safe hex writer for a pointer/address (no printf, no malloc).
inline void eshkol_signal_safe_write_ptr(const void* p) {
    uintptr_t v = (uintptr_t)p;
    constexpr int kNibbles = (int)(sizeof(uintptr_t) * 2);
    char buf[2 + kNibbles];
    buf[0] = '0';
    buf[1] = 'x';
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < kNibbles; ++i) {
        buf[2 + i] = hexd[(v >> ((kNibbles - 1 - i) * 4)) & 0xF];
    }
    eshkol_signal_safe_write(buf, sizeof(buf));
}

/* Bug AA / ESH-0119: handler for fatal synchronous signals
 * (SIGSEGV/SIGBUS/SIGILL/SIGFPE). The default action for these is to terminate
 * without flushing any stdio buffers, which means a `(display "step 1")
 * (newline)` issued before a crash never reaches the terminal.
 *
 * ESH-0119: stack overflow from deep recursion frequently surfaces as SIGILL
 * (rc 132) — not just SIGSEGV/SIGBUS — so SIGILL must be caught too, otherwise
 * the depth limit is a SILENT crash instead of a diagnosable failure. This
 * handler is installed with SA_SIGINFO (for the fault address) and SA_ONSTACK
 * (so it can run on the alternate signal stack even when the normal stack is
 * exhausted — a stack-overflow handler that needs stack is itself unreliable).
 *
 * Strategy: write a minimal diagnostic to stderr (async-signal-safe: write(),
 * no malloc), flag likely stack-overflow for the fault signals, restore the
 * default disposition for this signal, and re-raise it so the kernel produces
 * the normal coredump / 128+signum exit code.
 */
void eshkol_fatal_signal_handler(int signum, siginfo_t* info, void* /*ucontext*/) {
    const char* name = "unknown";
    bool likely_overflow = false;
    switch (signum) {
        case SIGSEGV: name = "SIGSEGV (segmentation fault)"; likely_overflow = true; break;
#ifdef SIGBUS
        case SIGBUS:  name = "SIGBUS (bus error)";           likely_overflow = true; break;
#endif
        case SIGILL:  name = "SIGILL (illegal instruction)"; likely_overflow = true; break;
#ifdef SIGFPE
        case SIGFPE:  name = "SIGFPE (arithmetic exception)";                        break;
#endif
        case SIGABRT: name = "SIGABRT (abort)";                                      break;
        default: break;
    }
    /* Best-effort flush of stdout/stderr. fflush() is technically not on
     * POSIX's async-signal-safe list; in practice it works on every libc we
     * run on when the process is otherwise quiescent. */
    std::fflush(stdout);
    std::fflush(stderr);

    static const char prefix[] = "\n[Eshkol] fatal signal: ";
    eshkol_signal_safe_write(prefix, sizeof(prefix) - 1);
    eshkol_signal_safe_write(name, std::strlen(name));

    // For memory-access faults, report the faulting address.
    if (info != nullptr && (signum == SIGSEGV
#ifdef SIGBUS
        || signum == SIGBUS
#endif
        )) {
        static const char at[] = " at address ";
        eshkol_signal_safe_write(at, sizeof(at) - 1);
        eshkol_signal_safe_write_ptr(info->si_addr);
    }

    if (likely_overflow) {
        // SIGSEGV/SIGBUS/SIGILL from deep recursion are almost always the
        // native stack being exhausted. Name that explicitly so the failure is
        // diagnosable ("recursion too deep") rather than a bare crash.
        static const char hint[] =
            "\n[Eshkol] this is most likely a stack overflow (recursion too deep) — "
            "use tail recursion or reduce the recursion depth\n";
        eshkol_signal_safe_write(hint, sizeof(hint) - 1);
    } else {
        static const char nl[] = "\n";
        eshkol_signal_safe_write(nl, 1);
    }
    static const char suffix[] =
        "[Eshkol] terminating; output above is what made it to stdout before the crash\n";
    eshkol_signal_safe_write(suffix, sizeof(suffix) - 1);

    // Restore default disposition and re-raise so the process exits with
    // the canonical 128+signum code (and dumps core if so configured).
    struct sigaction sa_dfl;
    std::memset(&sa_dfl, 0, sizeof(sa_dfl));
    sa_dfl.sa_handler = SIG_DFL;
    sigemptyset(&sa_dfl.sa_mask);
    sigaction(signum, &sa_dfl, nullptr);
    std::raise(signum);
    // Unreachable, but defensive.
    eshkol_signal_safe_exit(128 + signum);
}
#endif  // !_WIN32

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

    // Bug AA / ESH-0119: install fatal-signal handlers that announce themselves
    // on stderr before terminating, so the user knows the program crashed and
    // that a deep-recursion overflow is a diagnosable failure, not a silent
    // rc132. Under a sanitizer we skip these entirely and let the sanitizer's
    // own handlers report the fault (avoids double-registration / fighting its
    // alternate stack and diagnostics).
#if defined(ESHKOL_UNDER_SANITIZER)
    (void)g_old_sigsegv_handler;
    (void)g_old_sigbus_handler;
    (void)g_old_sigill_handler;
    (void)g_old_sigfpe_handler;
    (void)g_old_sigabrt_handler;
    (void)g_altstack_installed;
    (void)g_altstack_mem;
#else
    // Install an alternate signal stack so the handler can run even when the
    // normal stack is exhausted (the whole point for stack-overflow SIGILL).
    // Only install our own if none already exists; otherwise reuse the existing
    // one via SA_ONSTACK without clobbering it.
    bool have_altstack = false;
    stack_t existing;
    std::memset(&existing, 0, sizeof(existing));
    if (sigaltstack(nullptr, &existing) == 0) {
        if (existing.ss_flags & SS_DISABLE) {
            size_t altsz = (size_t)SIGSTKSZ;
            if (altsz < (size_t)MINSIGSTKSZ) altsz = (size_t)MINSIGSTKSZ;
            if (altsz < (size_t)65536)       altsz = (size_t)65536;
            g_altstack_mem = std::malloc(altsz);
            if (g_altstack_mem != nullptr) {
                stack_t ss;
                std::memset(&ss, 0, sizeof(ss));
                ss.ss_sp = g_altstack_mem;
                ss.ss_size = altsz;
                ss.ss_flags = 0;
                if (sigaltstack(&ss, &g_altstack_prev) == 0) {
                    g_altstack_installed = true;
                    have_altstack = true;
                } else {
                    std::free(g_altstack_mem);
                    g_altstack_mem = nullptr;
                }
            }
        } else {
            // An alternate stack already exists; reuse it.
            have_altstack = true;
        }
    }

    struct sigaction sa_fatal;
    std::memset(&sa_fatal, 0, sizeof(sa_fatal));
    sa_fatal.sa_sigaction = eshkol_fatal_signal_handler;
    sa_fatal.sa_flags = SA_SIGINFO | SA_RESETHAND | (have_altstack ? SA_ONSTACK : 0);
    sigemptyset(&sa_fatal.sa_mask);
    if (sigaction(SIGSEGV, &sa_fatal, &g_old_sigsegv_handler) != 0) {
        eshkol_warn("Failed to install SIGSEGV handler");
    }
#ifdef SIGBUS
    if (sigaction(SIGBUS,  &sa_fatal, &g_old_sigbus_handler)  != 0) {
        eshkol_warn("Failed to install SIGBUS handler");
    }
#endif
    // ESH-0119: stack overflow from deep recursion frequently surfaces as
    // SIGILL, so catch it too — this is what turns silent rc132 deaths into a
    // clean "stack overflow (recursion too deep)" diagnostic.
    if (sigaction(SIGILL, &sa_fatal, &g_old_sigill_handler) != 0) {
        eshkol_warn("Failed to install SIGILL handler");
    }
#ifdef SIGFPE
    if (sigaction(SIGFPE, &sa_fatal, &g_old_sigfpe_handler) != 0) {
        eshkol_warn("Failed to install SIGFPE handler");
    }
#endif
    /* Note: we deliberately do NOT install a handler for SIGABRT. The Bug AA
     * fix is about not losing user output when an Eshkol program crashes with
     * SIGSEGV/SIGBUS/SIGILL/SIGFPE. SIGABRT can be raised by libsystem during
     * process teardown; handling it here can turn clean shutdown into a
     * spurious abort. It remains on its default disposition, which already
     * produces a nonzero (128+SIGABRT) exit.
     */
    (void)g_old_sigabrt_handler;
#endif  // ESHKOL_UNDER_SANITIZER
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
#if !defined(ESHKOL_UNDER_SANITIZER)
    sigaction(SIGSEGV, &g_old_sigsegv_handler, nullptr);
#ifdef SIGBUS
    sigaction(SIGBUS,  &g_old_sigbus_handler,  nullptr);
#endif
    sigaction(SIGILL,  &g_old_sigill_handler,  nullptr);
#ifdef SIGFPE
    sigaction(SIGFPE,  &g_old_sigfpe_handler,  nullptr);
#endif
    (void)g_old_sigabrt_handler;
    // Tear down our alternate signal stack if we installed one.
    if (g_altstack_installed) {
        sigaltstack(&g_altstack_prev, nullptr);
        g_altstack_installed = false;
    }
    if (g_altstack_mem != nullptr) {
        std::free(g_altstack_mem);
        g_altstack_mem = nullptr;
    }
#endif
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
