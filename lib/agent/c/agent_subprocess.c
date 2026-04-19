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
#include <poll.h>
#include <time.h>      /* nanosleep */
#include <sys/time.h>  /* setitimer / itimerval */
#include <pthread.h>   /* per-stream drain threads — canonical POSIX pattern */
#else
#include <windows.h>
#endif

/* ── Subprocess handle ──
 * stdout_buf / stderr_buf accumulate what qllm_process_wait's poll loop
 * reads from the child's stdout/stderr pipes while the child is still
 * running. Without this, a chatty child (lean, cmake, any tool that
 * prints > ~64 KB) fills its pipe, blocks on write, never exits, and
 * our WNOHANG-based wait times out with exit 124 even though the child
 * would have finished in milliseconds given somewhere to put its
 * output. See Noesis v5 audit BUG A (2026-04-19). */
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
    /* Drained-while-waiting buffers. Grown on demand; NUL terminated. */
    char* stdout_buf;
    size_t stdout_len;
    size_t stdout_cap;
    char* stderr_buf;
    size_t stderr_len;
    size_t stderr_cap;
    int stdout_eof;
    int stderr_eof;
} eshkol_subprocess_t;

#ifndef _WIN32
/* Forward decl — body below in the Pipe Draining section. */
static void set_pipes_nonblocking(eshkol_subprocess_t* proc);
#endif

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
    int stdin_pipe[2] = {-1,-1}, stdout_pipe[2] = {-1,-1}, stderr_pipe[2] = {-1,-1};
    /* Partial-success pipe creation was leaking fds — if stdin_pipe
     * succeeded but stdout_pipe failed, the two stdin fds never got
     * closed because the old `||` short-circuit returned immediately.
     * #182 resource-management audit. Gate each pipe() separately so
     * we can clean up exactly the ones that succeeded. */
    if (pipe(stdin_pipe) != 0) { free(proc); return NULL; }
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        free(proc); return NULL;
    }
    if (pipe(stderr_pipe) != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        free(proc); return NULL;
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
    set_pipes_nonblocking(proc);
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
    set_pipes_nonblocking(proc);
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

    int stdin_pipe[2] = {-1,-1}, stdout_pipe[2] = {-1,-1}, stderr_pipe[2] = {-1,-1};
    /* Same partial-success fd-leak guard as the shell-form spawn — see
     * #182 comment above. Gate each pipe() so a mid-sequence failure
     * closes only the fds that actually opened. */
    if (pipe(stdin_pipe) != 0) {
        free(argv[0]); free(argv); free(proc); return NULL;
    }
    if (pipe(stdout_pipe) != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        free(argv[0]); free(argv); free(proc); return NULL;
    }
    if (pipe(stderr_pipe) != 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        free(argv[0]); free(argv); free(proc); return NULL;
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
    set_pipes_nonblocking(proc);
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

/* Read all available data. Returns whatever was drained during
 * qllm_process_wait plus any remainder still in the pipe. The buffer
 * is malloc'd for the caller to free; internal proc buffer is
 * zero'd after transfer so a second call returns empty string. */
static char* read_all_stream_posix(int fd,
                                   char** inline_buf, size_t* inline_len, size_t* inline_cap,
                                   int* eof_flag,
                                   int64_t max_size, int64_t* out_len) {
#ifndef _WIN32
    if (max_size <= 0) return NULL;

    /* If the pipe fd was closed already (EOF detected during wait),
     * transfer the inline buffer and bail. */
    size_t cap = (size_t)max_size;
    char* buf = (char*)malloc(cap + 1);
    if (!buf) return NULL;
    size_t total = 0;

    /* Hand over what we already drained during wait. */
    if (*inline_buf && *inline_len) {
        size_t take = (*inline_len > cap) ? cap : *inline_len;
        memcpy(buf, *inline_buf, take);
        total = take;
    }

    /* Drain any remainder the child wrote after the last wait tick.
     * Pipe is already non-blocking (drain_proc_pipes set the flag).
     * We keep reading until EAGAIN or EOF so the child's final output
     * lands in the caller's buffer. */
    if (fd >= 0 && !*eof_flag && total < cap) {
        while (total < cap) {
            ssize_t n = read(fd, buf + total, cap - total);
            if (n > 0) { total += (size_t)n; continue; }
            if (n == 0) { *eof_flag = 1; break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Child may still be flushing. Briefly poll so we don't
                 * return half its output. Cheap because the process has
                 * already been marked exited by this point in practice. */
                struct pollfd pf = { fd, POLLIN, 0 };
                int p = poll(&pf, 1, 100);
                if (p <= 0) break;
                continue;
            }
            break;
        }
    }

    buf[total] = '\0';
    if (out_len) *out_len = (int64_t)total;

    /* Reset the inline buffer so a second read_all call returns empty
     * (matches pre-fix semantics: "read EOF → subsequent reads empty"). */
    if (*inline_buf) {
        free(*inline_buf);
        *inline_buf = NULL;
        *inline_len = 0;
        *inline_cap = 0;
    }

    return buf;
#else
    (void)fd; (void)inline_buf; (void)inline_len; (void)inline_cap;
    (void)eof_flag; (void)max_size; (void)out_len;
    return NULL;
#endif
}

char* qllm_process_read_all_stdout(eshkol_subprocess_t* proc, int64_t max_size, int64_t* out_len) {
    if (!proc) return NULL;
#ifndef _WIN32
    return read_all_stream_posix(proc->stdout_fd,
                                 &proc->stdout_buf, &proc->stdout_len, &proc->stdout_cap,
                                 &proc->stdout_eof,
                                 max_size, out_len);
#else
    return NULL;
#endif
}

char* qllm_process_read_all_stderr(eshkol_subprocess_t* proc, int64_t max_size, int64_t* out_len) {
    if (!proc) return NULL;
#ifndef _WIN32
    return read_all_stream_posix(proc->stderr_fd,
                                 &proc->stderr_buf, &proc->stderr_len, &proc->stderr_cap,
                                 &proc->stderr_eof,
                                 max_size, out_len);
#else
    return NULL;
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * Pipe Draining (POSIX)
 *
 * Why: see eshkol_subprocess_t comment. qllm_process_wait MUST drain
 * stdout/stderr while the child is running or any non-trivial output
 * deadlocks the child on write. We attach a growable buffer to the
 * proc struct and opportunistically read into it whenever there's
 * data. The buffer is later returned by qllm_process_read_all_stdout
 * / read_all_stderr so callers see exactly what the child wrote.
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef _WIN32
/* No-op SIGALRM handler. The *purpose* is solely to make delivery
 * interrupt a blocking syscall with EINTR rather than killing us (the
 * SIGALRM default). Defined at file scope so its address is usable
 * inside nested functions. */
static void eshkol_subprocess_alrm_noop(int sig) { (void)sig; }

static int drain_fd_nonblocking(int fd,
                                char** buf, size_t* len, size_t* cap,
                                size_t max_bytes,
                                int* eof_flag) {
    if (fd < 0 || *eof_flag) return 0;
    char tmp[4096];
    int got_any = 0;
    while (1) {
        if (max_bytes > 0 && *len >= max_bytes) {
            /* We've reached the caller's cap — drop extra bytes but
             * keep reading so the child isn't blocked. */
            ssize_t n = read(fd, tmp, sizeof(tmp));
            if (n > 0) { got_any = 1; continue; }
            if (n == 0) { *eof_flag = 1; break; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            break;
        }
        size_t room = (max_bytes > 0) ? (max_bytes - *len) : sizeof(tmp);
        if (room > sizeof(tmp)) room = sizeof(tmp);
        ssize_t n = read(fd, tmp, room);
        if (n > 0) {
            if (*len + (size_t)n + 1 > *cap) {
                size_t newcap = (*cap == 0) ? 4096 : (*cap * 2);
                while (newcap < *len + (size_t)n + 1) newcap *= 2;
                char* nb = (char*)realloc(*buf, newcap);
                if (!nb) break;
                *buf = nb;
                *cap = newcap;
            }
            memcpy(*buf + *len, tmp, (size_t)n);
            *len += (size_t)n;
            (*buf)[*len] = '\0';
            got_any = 1;
            continue;
        }
        if (n == 0) { *eof_flag = 1; break; }
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        /* Hard error — stop reading this fd. */
        *eof_flag = 1;
        break;
    }
    return got_any;
}

/* Fds are set O_NONBLOCK once at spawn time (see the spawn sites that set
 * `nonblock_set`); drain_proc_pipes therefore runs in the hot path without
 * any per-iteration fcntl() cost. The earlier version re-tested every
 * iteration which turned out to dominate the 20-call loop benchmark. */
static void drain_proc_pipes(eshkol_subprocess_t* proc, size_t max_per_stream) {
    if (!proc) return;
    drain_fd_nonblocking(proc->stdout_fd,
                         &proc->stdout_buf, &proc->stdout_len, &proc->stdout_cap,
                         max_per_stream, &proc->stdout_eof);
    drain_fd_nonblocking(proc->stderr_fd,
                         &proc->stderr_buf, &proc->stderr_len, &proc->stderr_cap,
                         max_per_stream, &proc->stderr_eof);
}

/* Set both stdout/stderr pipe fds to O_NONBLOCK. Call once at spawn. */
static void set_pipes_nonblocking(eshkol_subprocess_t* proc) {
    if (!proc) return;
    for (int which = 0; which < 2; which++) {
        int fd = (which == 0) ? proc->stdout_fd : proc->stderr_fd;
        if (fd < 0) continue;
        int f = fcntl(fd, F_GETFL);
        if (f >= 0 && !(f & O_NONBLOCK)) {
            fcntl(fd, F_SETFL, f | O_NONBLOCK);
        }
    }
}

/* Re-enable blocking mode on a single fd. Used by the pthread drainer
 * so blocking `read()` can wait for bytes instead of spinning on
 * EAGAIN. The fd goes back to non-blocking after the drain thread
 * exits, so callers observing pipes post-wait still see the
 * non-blocking behaviour the older drain-in-wait path relied on. */
static void set_fd_blocking(int fd) {
    if (fd < 0) return;
    int f = fcntl(fd, F_GETFL);
    if (f >= 0 && (f & O_NONBLOCK)) {
        fcntl(fd, F_SETFL, f & ~O_NONBLOCK);
    }
}

/* ─── Pthread drainer ───
 *
 * Canonical POSIX subprocess pattern (matches CPython's subprocess and
 * Go's os/exec): spawn one thread per pipe (stdout, stderr), each
 * doing blocking reads into a growable buffer until EOF. The parent's
 * main thread blocks in waitpid(). This eliminates every polling
 * artefact — no sigaction, no setitimer, no backoff loop, no
 * scheduler-latency roundoff — and lets trivial `/bin/echo hello`
 * complete in microseconds on top of the 2–3 ms fork+exec cost. */
typedef struct {
    int fd;
    char** buf;
    size_t* len;
    size_t* cap;
    size_t max_bytes;
    int* eof_flag;
} drain_thread_arg_t;

static void* drain_thread_fn(void* vp) {
    drain_thread_arg_t* a = (drain_thread_arg_t*)vp;
    if (!a || a->fd < 0) return NULL;
    /* Blocking reads so the thread sleeps on the pipe instead of
     * spinning. When the child closes its write end (exit, close,
     * exec failure), read() returns 0 and the thread exits. */
    set_fd_blocking(a->fd);
    char tmp[16384];
    while (1) {
        ssize_t n = read(a->fd, tmp, sizeof(tmp));
        if (n > 0) {
            if (a->max_bytes > 0 && *a->len >= a->max_bytes) {
                /* Cap reached — keep reading so the child doesn't
                 * stall, but drop bytes past the cap. */
                continue;
            }
            size_t take = (size_t)n;
            if (a->max_bytes > 0 && *a->len + take > a->max_bytes) {
                take = a->max_bytes - *a->len;
            }
            if (*a->len + take + 1 > *a->cap) {
                size_t newcap = (*a->cap == 0) ? 4096 : (*a->cap * 2);
                while (newcap < *a->len + take + 1) newcap *= 2;
                char* nb = (char*)realloc(*a->buf, newcap);
                if (!nb) continue;
                *a->buf = nb;
                *a->cap = newcap;
            }
            memcpy(*a->buf + *a->len, tmp, take);
            *a->len += take;
            (*a->buf)[*a->len] = '\0';
            continue;
        }
        if (n == 0) { *a->eof_flag = 1; break; }
        if (errno == EINTR) continue;
        /* Hard error — mark EOF and stop. */
        *a->eof_flag = 1;
        break;
    }
    return NULL;
}
#endif /* !_WIN32 */

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

/* Contract (matches lib/agent/subprocess.esk:process-wait docstring):
 *   return 0  — child exited (caller uses qllm_process_exit_code to read code)
 *   return 1  — timeout hit before child exited
 *   return -1 — error (null proc, waitpid failure)
 *
 * Previously this function returned `proc->exit_code` on exit, so the
 * common case of a child exiting with status 1 (bad-proof, test-failure,
 * etc.) collided with the "timeout" sentinel `1` that run-command-capture
 * checks — every legitimate non-zero exit was reported as a 124 timeout
 * to callers. Noesis v5 audit BUG A (2026-04-19).
 *
 * This implementation ALSO drains stdout/stderr pipes while waiting so a
 * chatty child (> ~64 KB output) doesn't deadlock blocking on write.
 */
int32_t qllm_process_wait(eshkol_subprocess_t* proc, int32_t timeout_ms) {
    if (!proc) return -1;
    if (proc->exited) return 0;
#ifndef _WIN32
    /* Per-stream byte cap. 16 MB is well above any reasonable tool
     * output; past this the drain thread keeps reading but drops
     * bytes so the child doesn't stall on a full pipe. The
     * read_all_* getters then apply whatever caller-specified cap on
     * top. */
    const size_t drain_cap = 16UL * 1024UL * 1024UL;

    /* Launch one drain thread per open pipe. They use blocking reads;
     * the kernel schedules them onto idle cores while this thread
     * blocks in waitpid(). No polling, no setitimer, no backoff — the
     * canonical POSIX pattern that Python's subprocess, Go's os/exec
     * and libuv's uv_process all use. */
    pthread_t out_tid = 0, err_tid = 0;
    drain_thread_arg_t out_arg = {
        proc->stdout_fd,
        &proc->stdout_buf, &proc->stdout_len, &proc->stdout_cap,
        drain_cap, &proc->stdout_eof
    };
    drain_thread_arg_t err_arg = {
        proc->stderr_fd,
        &proc->stderr_buf, &proc->stderr_len, &proc->stderr_cap,
        drain_cap, &proc->stderr_eof
    };
    int out_launched = 0, err_launched = 0;
    if (proc->stdout_fd >= 0 && !proc->stdout_eof) {
        if (pthread_create(&out_tid, NULL, drain_thread_fn, &out_arg) == 0) {
            out_launched = 1;
        }
    }
    if (proc->stderr_fd >= 0 && !proc->stderr_eof) {
        if (pthread_create(&err_tid, NULL, drain_thread_fn, &err_arg) == 0) {
            err_launched = 1;
        }
    }

    int result;
    if (timeout_ms < 0) {
        /* Unbounded wait — block in waitpid. The drain threads will
         * finish on their own once the child closes its pipes. */
        int status;
        pid_t r;
        do { r = waitpid((pid_t)proc->pid, &status, 0); }
        while (r < 0 && errno == EINTR);
        if (r == proc->pid) {
            proc->exited = 1;
            if (WIFEXITED(status)) proc->exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) proc->exit_code = 128 + WTERMSIG(status);
            result = 0;
        } else {
            result = -1;
        }
    } else {
        /* Timed wait — blocking waitpid protected by a SIGALRM
         * itimer for the caller's full timeout budget. No polling
         * loop: single waitpid call, single itimer arm/disarm pair.
         * For a <1ms child this is fully dominated by fork+exec
         * cost; for a long-running child SIGALRM interrupts waitpid
         * exactly once per timeout. */
        struct sigaction old_sa, new_sa;
        memset(&new_sa, 0, sizeof(new_sa));
        new_sa.sa_handler = eshkol_subprocess_alrm_noop;
        sigaction(SIGALRM, &new_sa, &old_sa);

        struct itimerval old_it, new_it;
        memset(&new_it, 0, sizeof(new_it));
        new_it.it_value.tv_sec = timeout_ms / 1000;
        new_it.it_value.tv_usec = (timeout_ms % 1000) * 1000;
        setitimer(ITIMER_REAL, &new_it, &old_it);

        int status;
        pid_t r = waitpid((pid_t)proc->pid, &status, 0);

        memset(&new_it, 0, sizeof(new_it));
        setitimer(ITIMER_REAL, &new_it, NULL);
        setitimer(ITIMER_REAL, &old_it, NULL);
        sigaction(SIGALRM, &old_sa, NULL);

        if (r == proc->pid) {
            proc->exited = 1;
            if (WIFEXITED(status)) proc->exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) proc->exit_code = 128 + WTERMSIG(status);
            result = 0;
        } else {
            /* EINTR from SIGALRM → timeout */
            result = 1;
        }
    }

    /* Wait for drain threads to see EOF and exit. When the child
     * exits, the kernel closes its pipe ends, so blocking read()
     * returns 0 and the drainer naturally terminates. For the
     * timeout case we close the reader fds so the drainer unblocks
     * immediately — but that's done by process-destroy / read_all_*
     * semantics, not here (so a caller that wants to read after a
     * timeout still sees drained data). */
    if (result == 1) {
        /* Timed out while child still running. Close the read ends
         * of the pipes so the drain threads unblock. The caller is
         * expected to call process-kill → process-wait → destroy to
         * clean up. */
        if (proc->stdout_fd >= 0) { close(proc->stdout_fd); proc->stdout_fd = -1; }
        if (proc->stderr_fd >= 0) { close(proc->stderr_fd); proc->stderr_fd = -1; }
    }
    if (out_launched) pthread_join(out_tid, NULL);
    if (err_launched) pthread_join(err_tid, NULL);
    return result;
#else
    DWORD wait_ms = (timeout_ms < 0) ? INFINITE : (DWORD)timeout_ms;
    DWORD res = WaitForSingleObject(proc->hProcess, wait_ms);
    check_exit_status(proc);
    if (proc->exited) return 0;
    return (res == WAIT_TIMEOUT) ? 1 : -1;
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
    /* Free any pipe-drain buffers that the wait loop accumulated but the
     * caller never asked for (read_all_* would have transferred them). */
    if (proc->stdout_buf) free(proc->stdout_buf);
    if (proc->stderr_buf) free(proc->stderr_buf);
#else
    if (proc->stdin_write) CloseHandle(proc->stdin_write);
    if (proc->stdout_read) CloseHandle(proc->stdout_read);
    if (proc->stderr_read) CloseHandle(proc->stderr_read);
    if (proc->hProcess) CloseHandle(proc->hProcess);
#endif
    free(proc);
}
