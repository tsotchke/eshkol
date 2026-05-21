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

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <sys/types.h>
#if defined(__linux__)
#  include <dirent.h>
#endif

/*******************************************************************************
 * Signal Flag Table
 *
 * We track the last received signal via a volatile flag. The agent polls
 * this between operations.
 ******************************************************************************/

static volatile sig_atomic_t g_last_signal = 0;
static volatile sig_atomic_t g_signal_count = 0;

static void signal_handler(int sig) {
    g_last_signal = sig;
    g_signal_count++;
}

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

/*
 * Get total signal count (for debugging/metrics).
 * Does not reset.
 */
int32_t eshkol_signal_total_count(void) {
    return (int32_t)g_signal_count;
}

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

static void eshkol_atexit_handler(void) {
    fflush(stderr);
    fflush(stdout);
}

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

static void kt_recurse_cb(pid_t child, void* ctx) {
    int sig = *(const int*)ctx;
    kill_tree_recursive(child, sig);
}

static void kill_tree_recursive(pid_t pid, int sig) {
    /* Find children first, then kill this process. */
    kt_visit_children(pid, kt_recurse_cb, &sig);
    kill(pid, sig);
}

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
