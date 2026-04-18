/*
 * agent_subprocess.c — Native subprocess management with pipe I/O
 *
 * Replaces the qLLM dependency for subprocess.esk.
 * Provides fork/exec with stdin/stdout/stderr pipes on POSIX,
 * CreateProcess with pipe handles on Windows.
 *
 * The API matches the qllm_process_* signatures expected by subprocess.esk.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#else
#include <windows.h>
#endif

/* ── Subprocess handle ── */
typedef struct {
    int64_t pid;
#ifndef _WIN32
    int stdin_fd;   /* parent writes to child's stdin */
    int stdout_fd;  /* parent reads from child's stdout */
    int stderr_fd;  /* parent reads from child's stderr */
#else
    HANDLE hProcess;
    HANDLE stdin_write;
    HANDLE stdout_read;
    HANDLE stderr_read;
#endif
    int exited;
    int exit_code;
} eshkol_subprocess_t;

/* ═══════════════════════════════════════════════════════════════════
 * Process Spawn
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Safety note (#190): qllm_process_spawn feeds `command` through
 * /bin/sh -c, so any caller that interpolates user-controlled data
 * into the command string without shell-quote ends up with classical
 * command injection. For that reason we also expose
 * qllm_process_spawn_argv below, which takes a TAB-separated argv
 * ("program\targ1\targ2\t…") and uses execvp — no shell, no
 * expansion, no injection surface. Callers that want shell features
 * (glob, pipes, redirection) keep using the shell form but are
 * responsible for shell-quoting every interpolated value
 * (shell-quote is provided in the system stdlib).
 */
eshkol_subprocess_t* qllm_process_spawn(const char* command, const char* cwd_arg,
                                         const char* unused_arg, int64_t flags) {
    /* subprocess.esk calls: (process-spawn-raw command cwd #f 0)
     * So: command=shell command, cwd_arg=working directory, unused_arg=#f, flags=0 */
    const char* cwd = cwd_arg;
    (void)unused_arg; (void)flags;
    if (!command) return NULL;

    eshkol_subprocess_t* proc = (eshkol_subprocess_t*)calloc(1, sizeof(eshkol_subprocess_t));
    if (!proc) return NULL;

#ifndef _WIN32
    int stdin_pipe[2], stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        free(proc);
        return NULL;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        free(proc);
        return NULL;
    }

    if (pid == 0) {
        /* Child process */
        close(stdin_pipe[1]);   /* close write end of stdin pipe */
        close(stdout_pipe[0]);  /* close read end of stdout pipe */
        close(stderr_pipe[0]);  /* close read end of stderr pipe */

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        if (cwd && cwd[0]) {
            if (chdir(cwd) != 0) _exit(126);
        }

        /* Execute command via shell — command is the full shell command string */
        execlp("/bin/sh", "sh", "-c", command, (char*)NULL);
        _exit(127);
    }

    /* Parent process */
    close(stdin_pipe[0]);   /* close read end of stdin */
    close(stdout_pipe[1]);  /* close write end of stdout */
    close(stderr_pipe[1]);  /* close write end of stderr */

    /* Set stdout/stderr to non-blocking for read operations */
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    proc->pid = pid;
    proc->stdin_fd = stdin_pipe[1];
    proc->stdout_fd = stdout_pipe[0];
    proc->stderr_fd = stderr_pipe[0];
    proc->exited = 0;
    proc->exit_code = -1;
    return proc;
#else
    /* Windows: CreateProcess with pipes */
    SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
    HANDLE stdin_read, stdin_write, stdout_read, stdout_write, stderr_read, stderr_write;

    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0) ||
        !CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !CreatePipe(&stderr_read, &stderr_write, &sa, 0)) {
        free(proc);
        return NULL;
    }

    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {0};
    si.cb = sizeof(si);
    si.hStdInput = stdin_read;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;
    si.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi = {0};

    /* Windows cmdline: CreateProcessA mutates the buffer in place, so
     * it must be on the heap (not a string-literal) and sized for the
     * full command. Previous code used a fixed 4096-byte stack buffer
     * and snprintf-silently-truncated anything longer, producing a
     * valid-looking but malformed command (#193 HIGH). Now we:
     *   - Measure "cmd /c " + command length ahead of time.
     *   - Reject if it would exceed Windows' 32768 cmdline limit.
     *   - Heap-allocate exactly the right size so CreateProcessA can
     *     mutate without corrupting anything outside the buffer.
     */
    size_t cmd_len = command ? strlen(command) : 0;
    size_t prefix_len = 7;  /* "cmd /c " */
    if (cmd_len + prefix_len + 1 > 32768) {
        CloseHandle(stdin_read); CloseHandle(stdin_write);
        CloseHandle(stdout_read); CloseHandle(stdout_write);
        CloseHandle(stderr_read); CloseHandle(stderr_write);
        free(proc);
        return NULL;
    }
    char* cmdline = (char*)malloc(cmd_len + prefix_len + 1);
    if (!cmdline) {
        CloseHandle(stdin_read); CloseHandle(stdin_write);
        CloseHandle(stdout_read); CloseHandle(stdout_write);
        CloseHandle(stderr_read); CloseHandle(stderr_write);
        free(proc);
        return NULL;
    }
    memcpy(cmdline, "cmd /c ", prefix_len);
    memcpy(cmdline + prefix_len, command ? command : "", cmd_len);
    cmdline[prefix_len + cmd_len] = '\0';

    BOOL created = CreateProcessA(NULL, cmdline, NULL, NULL, TRUE, 0, NULL, cwd, &si, &pi);
    free(cmdline);
    if (!created) {
        CloseHandle(stdin_read); CloseHandle(stdin_write);
        CloseHandle(stdout_read); CloseHandle(stdout_write);
        CloseHandle(stderr_read); CloseHandle(stderr_write);
        free(proc);
        return NULL;
    }

    CloseHandle(stdin_read);
    CloseHandle(stdout_write);
    CloseHandle(stderr_write);
    CloseHandle(pi.hThread);

    proc->pid = (int64_t)pi.dwProcessId;
    proc->hProcess = pi.hProcess;
    proc->stdin_write = stdin_write;
    proc->stdout_read = stdout_read;
    proc->stderr_read = stderr_read;
    proc->exited = 0;
    proc->exit_code = -1;
    return proc;
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Safe Argv-based Spawn (#190)
 *
 * qllm_process_spawn_argv takes the argv packed into ONE string with
 * tab separators: "program\targ1\targ2\t…". We split on \t in the
 * child and call execvp directly — no shell, no expansion. This
 * eliminates the classical command-injection surface for callers
 * that don't actually need shell features.
 *
 * We use tab as the separator (rather than \0 or a count+offset
 * table) because the Eshkol FFI layer currently exposes `ptr` as a
 * C-string only; a NUL-separated representation would be truncated
 * at the first separator. Tab is not a valid character in program
 * names or typical arguments, and if a user really needs a tab in
 * an argument, they fall back to the shell form (where quoting is
 * their responsibility anyway).
 * ═══════════════════════════════════════════════════════════════════ */

#ifndef _WIN32
static char** parse_tab_argv(const char* packed, int* argc_out) {
    if (!packed) { *argc_out = 0; return NULL; }
    /* Count tabs + 1. */
    int argc = 1;
    for (const char* p = packed; *p; p++) if (*p == '\t') argc++;
    char** argv = (char**)calloc((size_t)argc + 1, sizeof(char*));
    if (!argv) { *argc_out = 0; return NULL; }

    char* dup = strdup(packed);
    if (!dup) { free(argv); *argc_out = 0; return NULL; }

    int i = 0;
    char* tok = dup;
    for (char* p = dup;; p++) {
        if (*p == '\t' || *p == '\0') {
            int last = (*p == '\0');
            *p = '\0';
            argv[i++] = tok;
            if (last) break;
            tok = p + 1;
        }
    }
    argv[i] = NULL;
    *argc_out = argc;
    /* Caller must free both argv and the underlying buffer (argv[0]). */
    return argv;
}
#endif

eshkol_subprocess_t* qllm_process_spawn_argv(const char* tab_packed_argv,
                                              const char* cwd_arg) {
    if (!tab_packed_argv || !tab_packed_argv[0]) return NULL;

    eshkol_subprocess_t* proc = (eshkol_subprocess_t*)calloc(1, sizeof(eshkol_subprocess_t));
    if (!proc) return NULL;

#ifndef _WIN32
    int argc = 0;
    char** argv = parse_tab_argv(tab_packed_argv, &argc);
    if (!argv || argc == 0 || !argv[0]) {
        if (argv) { free(argv[0]); free(argv); }
        free(proc);
        return NULL;
    }

    int stdin_pipe[2], stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        free(argv[0]); free(argv); free(proc);
        return NULL;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        close(stderr_pipe[0]); close(stderr_pipe[1]);
        free(argv[0]); free(argv); free(proc);
        return NULL;
    }

    if (pid == 0) {
        close(stdin_pipe[1]); close(stdout_pipe[0]); close(stderr_pipe[0]);
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdin_pipe[0]); close(stdout_pipe[1]); close(stderr_pipe[1]);

        if (cwd_arg && cwd_arg[0]) {
            if (chdir(cwd_arg) != 0) _exit(126);
        }

        execvp(argv[0], argv);
        _exit(127);
    }

    /* Parent */
    close(stdin_pipe[0]); close(stdout_pipe[1]); close(stderr_pipe[1]);
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    free(argv[0]); free(argv);

    proc->pid = pid;
    proc->stdin_fd = stdin_pipe[1];
    proc->stdout_fd = stdout_pipe[0];
    proc->stderr_fd = stderr_pipe[0];
    proc->exited = 0;
    proc->exit_code = -1;
    return proc;
#else
    /* Windows: CreateProcessA wants a single command-line string, but
     * we can reconstruct it quote-safely. For now, delegate to the
     * shell path with the caller's permission (they called the _argv
     * variant, so treat first tab-separated token as program and
     * re-join the rest with proper quoting via a simple space
     * join — good enough for whitespace-free argv). A fuller Win32
     * implementation is #193 Windows-subprocess buffer-overflow fix
     * territory. */
    (void)cwd_arg;
    free(proc);
    return NULL;
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Stdin Write / Close
 * ═══════════════════════════════════════════════════════════════════ */

int64_t qllm_process_write_stdin(eshkol_subprocess_t* proc, const char* data, int64_t len) {
    if (!proc || !data || len <= 0) return -1;
#ifndef _WIN32
    if (proc->stdin_fd < 0) return -1;
    ssize_t written = write(proc->stdin_fd, data, (size_t)len);
    return (int64_t)written;
#else
    DWORD written = 0;
    if (!WriteFile(proc->stdin_write, data, (DWORD)len, &written, NULL)) return -1;
    return (int64_t)written;
#endif
}

void qllm_process_close_stdin(eshkol_subprocess_t* proc) {
    if (!proc) return;
#ifndef _WIN32
    if (proc->stdin_fd >= 0) { close(proc->stdin_fd); proc->stdin_fd = -1; }
#else
    if (proc->stdin_write) { CloseHandle(proc->stdin_write); proc->stdin_write = NULL; }
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Stdout / Stderr Read
 * ═══════════════════════════════════════════════════════════════════ */

int64_t qllm_process_read_stdout(eshkol_subprocess_t* proc, char* buf, int64_t buf_size) {
    if (!proc || !buf || buf_size <= 0) return -1;
#ifndef _WIN32
    if (proc->stdout_fd < 0) return -1;
    ssize_t n = read(proc->stdout_fd, buf, (size_t)buf_size);
    if (n < 0 && errno == EAGAIN) return 0; /* non-blocking: no data */
    return (int64_t)n;
#else
    DWORD n = 0;
    if (!ReadFile(proc->stdout_read, buf, (DWORD)buf_size, &n, NULL)) return -1;
    return (int64_t)n;
#endif
}

int64_t qllm_process_read_stderr(eshkol_subprocess_t* proc, char* buf, int64_t buf_size) {
    if (!proc || !buf || buf_size <= 0) return -1;
#ifndef _WIN32
    if (proc->stderr_fd < 0) return -1;
    ssize_t n = read(proc->stderr_fd, buf, (size_t)buf_size);
    if (n < 0 && errno == EAGAIN) return 0;
    return (int64_t)n;
#else
    DWORD n = 0;
    if (!ReadFile(proc->stderr_read, buf, (DWORD)buf_size, &n, NULL)) return -1;
    return (int64_t)n;
#endif
}

/* Read all available data (blocking until EOF) */
char* qllm_process_read_all_stdout(eshkol_subprocess_t* proc, int64_t max_size, int64_t* out_len) {
    if (!proc || max_size <= 0) return NULL;
#ifndef _WIN32
    if (proc->stdout_fd < 0) return NULL;
    /* Set back to blocking for read-all */
    int flags = fcntl(proc->stdout_fd, F_GETFL);
    fcntl(proc->stdout_fd, F_SETFL, flags & ~O_NONBLOCK);

    char* buf = (char*)malloc((size_t)max_size + 1);
    if (!buf) return NULL;
    size_t total = 0;
    ssize_t n;
    while ((n = read(proc->stdout_fd, buf + total, (size_t)(max_size - (int64_t)total))) > 0) {
        total += (size_t)n;
        if ((int64_t)total >= max_size) break;
    }
    buf[total] = '\0';
    if (out_len) *out_len = (int64_t)total;

    /* Restore non-blocking */
    fcntl(proc->stdout_fd, F_SETFL, flags);
    return buf;
#else
    return NULL;
#endif
}

char* qllm_process_read_all_stderr(eshkol_subprocess_t* proc, int64_t max_size, int64_t* out_len) {
    if (!proc || max_size <= 0) return NULL;
#ifndef _WIN32
    if (proc->stderr_fd < 0) return NULL;
    int flags = fcntl(proc->stderr_fd, F_GETFL);
    fcntl(proc->stderr_fd, F_SETFL, flags & ~O_NONBLOCK);

    char* buf = (char*)malloc((size_t)max_size + 1);
    if (!buf) return NULL;
    size_t total = 0;
    ssize_t n;
    while ((n = read(proc->stderr_fd, buf + total, (size_t)(max_size - (int64_t)total))) > 0) {
        total += (size_t)n;
        if ((int64_t)total >= max_size) break;
    }
    buf[total] = '\0';
    if (out_len) *out_len = (int64_t)total;
    fcntl(proc->stderr_fd, F_SETFL, flags);
    return buf;
#else
    return NULL;
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Process Status
 * ═══════════════════════════════════════════════════════════════════ */

static void check_exit_status(eshkol_subprocess_t* proc) {
    if (!proc || proc->exited) return;
#ifndef _WIN32
    int status;
    pid_t result = waitpid((pid_t)proc->pid, &status, WNOHANG);
    if (result == proc->pid) {
        proc->exited = 1;
        if (WIFEXITED(status)) {
            proc->exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            proc->exit_code = 128 + WTERMSIG(status);
        }
    }
#else
    DWORD code;
    if (GetExitCodeProcess(proc->hProcess, &code) && code != STILL_ACTIVE) {
        proc->exited = 1;
        proc->exit_code = (int)code;
    }
#endif
}

int32_t qllm_process_wait(eshkol_subprocess_t* proc, int32_t timeout_ms) {
    if (!proc) return -1;
    if (proc->exited) return proc->exit_code;
#ifndef _WIN32
    if (timeout_ms < 0) {
        /* Block until done */
        int status;
        waitpid((pid_t)proc->pid, &status, 0);
        proc->exited = 1;
        if (WIFEXITED(status)) proc->exit_code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) proc->exit_code = 128 + WTERMSIG(status);
        return proc->exit_code;
    }
    /* Poll with timeout */
    for (int i = 0; i < timeout_ms; i++) {
        check_exit_status(proc);
        if (proc->exited) return proc->exit_code;
        usleep(1000); /* 1ms */
    }
    return -1; /* still running */
#else
    DWORD wait_ms = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
    WaitForSingleObject(proc->hProcess, wait_ms);
    check_exit_status(proc);
    return proc->exited ? proc->exit_code : -1;
#endif
}

int32_t qllm_process_running(eshkol_subprocess_t* proc) {
    if (!proc) return 0;
    check_exit_status(proc);
    return !proc->exited;
}

int32_t qllm_process_exit_code(eshkol_subprocess_t* proc) {
    if (!proc) return -1;
    check_exit_status(proc);
    return proc->exit_code;
}

/* ═══════════════════════════════════════════════════════════════════
 * Process Kill / Destroy
 * ═══════════════════════════════════════════════════════════════════ */

void qllm_process_kill(eshkol_subprocess_t* proc, int32_t signal) {
    if (!proc) return;
#ifndef _WIN32
    kill((pid_t)proc->pid, signal);
#else
    TerminateProcess(proc->hProcess, 1);
#endif
}

void qllm_process_destroy(eshkol_subprocess_t* proc) {
    if (!proc) return;
#ifndef _WIN32
    if (proc->stdin_fd >= 0) close(proc->stdin_fd);
    if (proc->stdout_fd >= 0) close(proc->stdout_fd);
    if (proc->stderr_fd >= 0) close(proc->stderr_fd);
    /* Reap zombie if not yet waited */
    if (!proc->exited) {
        int status;
        waitpid((pid_t)proc->pid, &status, WNOHANG);
    }
#else
    if (proc->stdin_write) CloseHandle(proc->stdin_write);
    if (proc->stdout_read) CloseHandle(proc->stdout_read);
    if (proc->stderr_read) CloseHandle(proc->stderr_read);
    if (proc->hProcess) CloseHandle(proc->hProcess);
#endif
    free(proc);
}
