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

static void kill_tree_recursive(pid_t pid, int sig) {
    /* Find children first */
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "pgrep -P %d 2>/dev/null", (int)pid);

    FILE* fp = popen(cmd, "r");
    if (fp) {
        char line[32];
        while (fgets(line, sizeof(line), fp)) {
            pid_t child = (pid_t)atoi(line);
            if (child > 0) {
                kill_tree_recursive(child, sig);
            }
        }
        pclose(fp);
    }

    /* Kill this process after children */
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
