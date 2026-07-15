/*******************************************************************************
 * Signal Handling and Atexit for Eshkol Agent
 *
 * Provides: SIGINT/SIGTERM/SIGWINCH handling with flag-based notification,
 *           atexit cleanup registration.
 *
 * Design: Signal handlers set atomic flags. Eshkol code polls these flags
 * via eshkol_signal_check() at safe points (between tool calls, in event
 * loop iterations). This avoids calling complex Eshkol closures from
 * async signal context.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#elif !defined(_WIN32)
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#if defined(__linux__)
#  include <dirent.h>
#endif
#endif

/*******************************************************************************
 * Signal Flag Table
 *
 * We track the last received signal via a volatile flag. The agent polls
 * this between operations.
 ******************************************************************************/

#ifdef _WIN32

static volatile LONG g_last_signal = 0;
static volatile LONG g_signal_count = 0;
static volatile LONG g_installed_mask = 0;
static volatile LONG g_ignored_mask = 0;
static int g_atexit_registered = 0;

static LONG signal_bit(int signum) {
    if (signum == SIGINT) return 1L << 0;
    if (signum == SIGTERM) return 1L << 1;
#ifdef SIGBREAK
    if (signum == SIGBREAK) return 1L << 2;
#endif
    return 0;
}

static BOOL WINAPI console_control_handler(DWORD event) {
    int signum = 0;
    switch (event) {
        case CTRL_C_EVENT: signum = SIGINT; break;
        case CTRL_BREAK_EVENT:
#ifdef SIGBREAK
            signum = SIGBREAK;
#else
            signum = SIGTERM;
#endif
            break;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT: signum = SIGTERM; break;
        default: return FALSE;
    }
    LONG bit = signal_bit(signum);
    if (bit == 0 || !(InterlockedCompareExchange(&g_installed_mask, 0, 0) & bit)) {
        return FALSE;
    }
    if (!(InterlockedCompareExchange(&g_ignored_mask, 0, 0) & bit)) {
        InterlockedExchange(&g_last_signal, signum);
        InterlockedIncrement(&g_signal_count);
    }
    return TRUE;
}

int32_t eshkol_signal_handler_install(int32_t signum) {
    LONG bit = signal_bit(signum);
    if (bit == 0) return -1;
    if (!SetConsoleCtrlHandler(console_control_handler, TRUE)) return -1;
    InterlockedOr(&g_installed_mask, bit);
    InterlockedAnd(&g_ignored_mask, ~bit);
    return 0;
}

int32_t eshkol_signal_check(void) {
    return (int32_t)InterlockedExchange(&g_last_signal, 0);
}

int32_t eshkol_signal_total_count(void) {
    return (int32_t)InterlockedCompareExchange(&g_signal_count, 0, 0);
}

int32_t eshkol_signal_handler_reset(int32_t signum) {
    LONG bit = signal_bit(signum);
    if (bit == 0) return -1;
    InterlockedAnd(&g_installed_mask, ~bit);
    InterlockedAnd(&g_ignored_mask, ~bit);
    if (InterlockedCompareExchange(&g_installed_mask, 0, 0) == 0) {
        SetConsoleCtrlHandler(console_control_handler, FALSE);
    }
    return 0;
}

int32_t eshkol_signal_ignore(int32_t signum) {
    LONG bit = signal_bit(signum);
    if (bit == 0) return -1;
    if (!SetConsoleCtrlHandler(console_control_handler, TRUE)) return -1;
    InterlockedOr(&g_installed_mask, bit);
    InterlockedOr(&g_ignored_mask, bit);
    return 0;
}

static void eshkol_atexit_handler(void) {
    fflush(stderr);
    fflush(stdout);
}

int32_t eshkol_atexit_init(void) {
    if (!g_atexit_registered) {
        if (atexit(eshkol_atexit_handler) != 0) return -1;
        g_atexit_registered = 1;
    }
    return 0;
}

static int terminate_process_tree(DWORD root_pid, DWORD exit_code) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return -1;

    DWORD children[1024];
    size_t child_count = 0;
    PROCESSENTRY32 entry;
    memset(&entry, 0, sizeof(entry));
    entry.dwSize = sizeof(entry);
    if (Process32First(snapshot, &entry)) {
        do {
            if (entry.th32ParentProcessID == root_pid && child_count < 1024) {
                children[child_count++] = entry.th32ProcessID;
            }
        } while (Process32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);

    int result = 0;
    for (size_t i = 0; i < child_count; ++i) {
        if (terminate_process_tree(children[i], exit_code) != 0) result = -1;
    }

    HANDLE process = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, root_pid);
    if (!process) return -1;
    if (!TerminateProcess(process, exit_code)) {
        DWORD error = GetLastError();
        if (error != ERROR_ACCESS_DENIED || WaitForSingleObject(process, 0) != WAIT_OBJECT_0) {
            result = -1;
        }
    }
    CloseHandle(process);
    return result;
}

int32_t eshkol_process_kill_tree(int32_t pid, int32_t sig) {
    if (pid <= 0) return -1;
    DWORD exit_code = sig > 0 ? (DWORD)(128 + sig) : 1;
    return terminate_process_tree((DWORD)pid, exit_code);
}

#else

static volatile sig_atomic_t g_last_signal = 0;
static volatile sig_atomic_t g_signal_count = 0;

/**
 * @brief Async-signal-safe handler that records the last-received signal.
 *
 * Only touches sig_atomic_t flags, deferring all real work to
 * eshkol_signal_check() polled from normal (non-signal) context.
 *
 * @param sig Signal number delivered by the OS.
 */
static void signal_handler(int sig) {
    g_last_signal = sig;
    g_signal_count++;
}

/**
 * @brief Installs signal_handler() for @p signum via sigaction().
 *
 * Uses SA_RESTART so interrupted syscalls are automatically restarted.
 * Common signal numbers: 2 = SIGINT (Ctrl-C), 15 = SIGTERM (kill),
 * 28 = SIGWINCH (terminal resize).
 *
 * @param signum Signal number to install the handler for.
 * @return 0 on success, -1 on error.
 */
/*
 * Install signal handler for the given signal number.
 *
 * Common signals:
 *   2  = SIGINT  (Ctrl-C)
 *   15 = SIGTERM (kill)
 *   28 = SIGWINCH (terminal resize)
 *
 * Returns: 0 success, -1 error
 */
int32_t eshkol_signal_handler_install(int32_t signum) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESTART;  /* Restart interrupted syscalls */
    sigemptyset(&sa.sa_mask);
    return sigaction(signum, &sa, NULL) == 0 ? 0 : -1;
}

/**
 * @brief Checks and consumes the most recently received signal.
 *
 * Intended to be polled at safe points in the Eshkol event loop between
 * tool calls, avoiding the need to run Eshkol closures from async-signal
 * context. Calling this clears the flag (consume-once semantics), so a
 * signal is only reported to one caller.
 *
 * @return The signal number if one was pending, or 0 if none was pending.
 */
/*
 * Check if a signal has been received since last check.
 *
 * Returns: signal number if received, 0 if no signal pending
 *
 * Calling this clears the flag (consume-once semantics).
 */
int32_t eshkol_signal_check(void) {
    int32_t sig = (int32_t)g_last_signal;
    if (sig != 0) {
        g_last_signal = 0;
    }
    return sig;
}

/**
 * @brief Returns the cumulative count of signals received since process start.
 *
 * Unlike eshkol_signal_check(), this does not reset the counter; it's meant
 * for debugging/metrics rather than consume-once dispatch.
 *
 * @return Total number of signals delivered so far.
 */
/*
 * Get total signal count (for debugging/metrics).
 * Does not reset.
 */
int32_t eshkol_signal_total_count(void) {
    return (int32_t)g_signal_count;
}

/**
 * @brief Restores the default disposition (SIG_DFL) for the given signal.
 *
 * @param signum Signal number to reset.
 * @return 0 on success, -1 on error.
 */
/*
 * Reset signal handler to default for the given signal.
 * Returns: 0 success, -1 error
 */
int32_t eshkol_signal_handler_reset(int32_t signum) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    return sigaction(signum, &sa, NULL) == 0 ? 0 : -1;
}

/**
 * @brief Sets the given signal's disposition to SIG_IGN so it is ignored.
 *
 * @param signum Signal number to ignore.
 * @return 0 on success, -1 on error.
 */
/*
 * Ignore the given signal.
 * Returns: 0 success, -1 error
 */
int32_t eshkol_signal_ignore(int32_t signum) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SIG_IGN;
    sigemptyset(&sa.sa_mask);
    return sigaction(signum, &sa, NULL) == 0 ? 0 : -1;
}

/*******************************************************************************
 * Atexit Registration
 *
 * Register a C-level cleanup function. For Eshkol closure-based cleanup,
 * the agent should use dynamic-wind at the top level.
 *
 * This is primarily for restoring terminal state and flushing buffers.
 ******************************************************************************/

/* Standard atexit is sufficient — the agent's main.esk should register
 * cleanup via (at-exit (lambda () ...)) which wraps dynamic-wind.
 *
 * We provide a C-level stderr flush to ensure error output isn't lost. */

static int g_atexit_registered = 0;

/**
 * @brief atexit() callback that flushes stderr and stdout on process exit.
 *
 * Ensures buffered error/output text isn't lost if the process terminates
 * abruptly; Eshkol-level cleanup should instead be registered via
 * (at-exit ...) / dynamic-wind at the top level.
 */
static void eshkol_atexit_handler(void) {
    fflush(stderr);
    fflush(stdout);
}

/**
 * @brief Registers eshkol_atexit_handler() with atexit(), exactly once.
 *
 * Idempotent: safe to call multiple times, the handler is only registered
 * on the first call.
 *
 * @return Always 0.
 */
/*
 * Ensure atexit flush is registered. Idempotent.
 * Returns: 0
 */
int32_t eshkol_atexit_init(void) {
    if (!g_atexit_registered) {
        atexit(eshkol_atexit_handler);
        g_atexit_registered = 1;
    }
    return 0;
}

/*******************************************************************************
 * Process Tree Kill
 *
 * Kill a process and all its descendants. Walks the process tree bottom-up
 * to prevent orphaned children.
 *
 * On macOS: uses pgrep -P to find children
 * On Linux: reads /proc/PID/children
 ******************************************************************************/

/* Visit each direct child of `pid` via `cb`. Returns 0 on success.
 *
 * Linux: read /proc/<pid>/task/<tid>/children for every task in the
 * process. No shell, no PATH, no popen; closes the popen-pgrep
 * injection vector the previous implementation exposed.
 *
 * macOS/BSD: fork + execv("/usr/bin/pgrep", ["pgrep", "-P", "<pid>", NULL])
 * with stdout captured via a pipe. Still uses pgrep, but no shell or
 * PATH lookup is involved.
 */
typedef void (*eshkol_kt_visitor)(pid_t child, void* ctx);

/* Forward declaration so kt_recurse_cb can refer back. */
static void kill_tree_recursive(pid_t pid, int sig);

#if defined(__linux__)
/**
 * @brief Invokes @p cb for every direct child process of @p pid (Linux).
 *
 * Reads /proc/<pid>/task/<tid>/children for each task (thread) belonging to
 * the process, avoiding any shell or PATH lookup. This closes the popen-pgrep
 * command-injection vector the previous shell-based implementation exposed.
 *
 * @param pid Parent process ID whose children are enumerated.
 * @param cb Callback invoked once per direct child PID found.
 * @param ctx Opaque context pointer forwarded to @p cb.
 */
static void kt_visit_children(pid_t pid, eshkol_kt_visitor cb, void* ctx) {
    char task_dir[64];
    snprintf(task_dir, sizeof(task_dir), "/proc/%d/task", (int)pid);
    DIR* td = opendir(task_dir);
    if (!td) return;
    struct dirent* tent;
    while ((tent = readdir(td)) != NULL) {
        if (tent->d_name[0] < '0' || tent->d_name[0] > '9') continue;
        char child_file[128];
        snprintf(child_file, sizeof(child_file),
                 "/proc/%d/task/%s/children", (int)pid, tent->d_name);
        FILE* fp = fopen(child_file, "r");
        if (!fp) continue;
        char buf[1024];
        size_t got = fread(buf, 1, sizeof(buf) - 1, fp);
        fclose(fp);
        if (got == 0) continue;
        buf[got] = '\0';
        char* p = buf;
        while (*p) {
            while (*p == ' ' || *p == '\n' || *p == '\t') p++;
            if (!*p) break;
            pid_t child = (pid_t)atoi(p);
            if (child > 0) cb(child, ctx);
            while (*p && *p != ' ' && *p != '\n' && *p != '\t') p++;
        }
    }
    closedir(td);
}
#else
/**
 * @brief Invokes @p cb for every direct child process of @p pid (macOS/BSD).
 *
 * Forks and execv()s "/usr/bin/pgrep -P <pid>" directly (no shell, no PATH
 * lookup), captures its stdout via a pipe, waits for it to exit, and parses
 * the resulting whitespace-separated PID list.
 *
 * @param pid Parent process ID whose children are enumerated.
 * @param cb Callback invoked once per direct child PID found.
 * @param ctx Opaque context pointer forwarded to @p cb.
 */
static void kt_visit_children(pid_t pid, eshkol_kt_visitor cb, void* ctx) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return;
    pid_t worker = fork();
    if (worker < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return;
    }
    if (worker == 0) {
        /* child: redirect stdout to the pipe, exec pgrep with argv list */
        close(pipefd[0]);
        if (pipefd[1] != STDOUT_FILENO) {
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
        }
        /* silence stderr */
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        char pidbuf[16];
        snprintf(pidbuf, sizeof(pidbuf), "%d", (int)pid);
        char* const argv[] = {(char*)"pgrep", (char*)"-P", pidbuf, NULL};
        execv("/usr/bin/pgrep", argv);
        _exit(127);
    }
    /* parent */
    close(pipefd[1]);
    char buf[1024];
    size_t total = 0;
    ssize_t r;
    while (total < sizeof(buf) - 1 &&
           (r = read(pipefd[0], buf + total, sizeof(buf) - 1 - total)) > 0) {
        total += (size_t)r;
    }
    close(pipefd[0]);
    int status;
    while (waitpid(worker, &status, 0) < 0 && errno == EINTR) { /* retry */ }
    if (total == 0) return;
    buf[total] = '\0';
    char* p = buf;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\t') p++;
        if (!*p) break;
        pid_t child = (pid_t)atoi(p);
        if (child > 0) cb(child, ctx);
        while (*p && *p != ' ' && *p != '\n' && *p != '\t') p++;
    }
}
#endif

/**
 * @brief kt_visit_children() callback that recurses kill_tree_recursive() into a child.
 *
 * @param child Child PID reported by kt_visit_children().
 * @param ctx Pointer to the `int` signal number to deliver.
 */
static void kt_recurse_cb(pid_t child, void* ctx) {
    int sig = *(const int*)ctx;
    kill_tree_recursive(child, sig);
}

/**
 * @brief Recursively delivers @p sig to @p pid's entire descendant tree, then to @p pid itself.
 *
 * Visits children depth-first via kt_visit_children()/kt_recurse_cb() before
 * signaling the current process, so descendants are signaled before their
 * parent to avoid leaving orphans behind.
 *
 * @param pid Root process ID of the tree to signal.
 * @param sig Signal number to deliver to every process in the tree.
 */
static void kill_tree_recursive(pid_t pid, int sig) {
    /* Find children first, then kill this process. */
    kt_visit_children(pid, kt_recurse_cb, &sig);
    kill(pid, sig);
}

/**
 * @brief Signals a process and its entire descendant tree.
 *
 * @param pid Root process ID to kill (must be positive).
 * @param sig Signal number to deliver (e.g. 15 = SIGTERM, 9 = SIGKILL).
 * @return 0 on success, -1 if @p pid is not positive.
 */
/*
 * Kill process and all descendants.
 *
 * pid: process ID to kill
 * signal: signal number (15=SIGTERM, 9=SIGKILL)
 *
 * Returns: 0 success, -1 error
 */
int32_t eshkol_process_kill_tree(int32_t pid, int32_t sig) {
    if (pid <= 0) return -1;
    kill_tree_recursive((pid_t)pid, sig);
    return 0;
}

#endif
