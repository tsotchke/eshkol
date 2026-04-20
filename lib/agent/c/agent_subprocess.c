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
#include <time.h>       /* nanosleep */
#include <sys/time.h>   /* setitimer / itimerval */
#include <pthread.h>    /* Linux fallback drain threads */
#include <spawn.h>      /* posix_spawn — vfork-lite on Darwin/Linux */
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/event.h>  /* kqueue / EVFILT_PROC NOTE_EXIT — zero-thread waiter */
#endif
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
/* Return 1 if `command` is a simple whitespace-separated token list
 * with NO shell metacharacters — i.e. /bin/sh -c "…" would behave
 * identically to execvp-on-tokens. This is the overwhelmingly common
 * case (e.g. `"/usr/bin/lean /tmp/bad.lean"`) and skipping sh startup
 * saves 3-5 ms per call.
 *
 * Metacharacters that force real shell parsing:
 *   |  &  ;  <  >  *  ?  $  `  \  '  "  ~  (  )  {  }  [  ]  !  #
 *
 * Also bail if we see a \n or \r — newlines are rare in command
 * strings and reserved for scripts.
 */
static int command_is_shell_safe(const char* cmd) {
    if (!cmd) return 0;
    for (const char* p = cmd; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '|': case '&': case ';': case '<': case '>':
            case '*': case '?': case '$': case '`': case '\\':
            case '\'': case '"': case '~': case '(': case ')':
            case '{': case '}': case '[': case ']':
            case '!': case '#': case '\n': case '\r':
                return 0;
            default: break;
        }
    }
    return 1;
}

/* Split a metacharacter-free command string on whitespace into an
 * argv array. Returns NULL on OOM or empty input. Caller frees with
 * `free(argv[0]); free(argv);`. */
static char** split_shell_safe_command(const char* cmd, int* out_argc) {
    if (!cmd) return NULL;
    /* Work on a mutable copy. The returned argv[i] all point into it,
     * so the caller frees argv[0] (which is the copy) plus argv itself.
     */
    char* buf = strdup(cmd);
    if (!buf) return NULL;

    /* Count tokens first to size argv exactly. */
    int count = 0;
    for (char* s = buf;;) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        count++;
        while (*s && *s != ' ' && *s != '\t') s++;
    }
    if (count == 0) { free(buf); return NULL; }

    char** argv = (char**)calloc((size_t)count + 1, sizeof(char*));
    if (!argv) { free(buf); return NULL; }

    int i = 0;
    char* s = buf;
    while (*s) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        argv[i++] = s;
        while (*s && *s != ' ' && *s != '\t') s++;
        if (*s) { *s = '\0'; s++; }
    }
    argv[i] = NULL;
    if (out_argc) *out_argc = count;
    return argv;
}

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

    /* Prefer posix_spawn for the "no chdir needed" hot path — see the
     * comment above qllm_process_spawn_argv for why this cuts ~10 ms
     * off the round-trip inside a large host process like eshkol-run.
     * The chdir-required path keeps fork+exec since macOS's
     * posix_spawn_file_actions_addchdir_np is 14.0+ only and we
     * still want to compile on 13.x. */
    extern char** environ;
    int no_chdir = (!cwd || !cwd[0] || (cwd[0] == '.' && cwd[1] == '\0'));
    pid_t pid;
    int spawn_errno = 0;

    if (no_chdir) {
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, stdin_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&fa, stdout_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&fa, stderr_pipe[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&fa, stdin_pipe[0]);
        posix_spawn_file_actions_addclose(&fa, stdin_pipe[1]);
        posix_spawn_file_actions_addclose(&fa, stdout_pipe[0]);
        posix_spawn_file_actions_addclose(&fa, stdout_pipe[1]);
        posix_spawn_file_actions_addclose(&fa, stderr_pipe[0]);
        posix_spawn_file_actions_addclose(&fa, stderr_pipe[1]);

        /* If the command has NO shell metacharacters (the common case
         * for benchmark and agent callers like `/usr/bin/lean /tmp/x.lean`
         * or `git rev-parse HEAD`), bypass /bin/sh -c entirely.
         * /bin/sh startup is 3-5 ms on a modern macOS — and every
         * round-trip Noesis / Aletheia does hits that cost, even
         * though the actual work could be direct posix_spawnp. */
        char** split_argv = NULL;
        int split_argc = 0;
        if (command_is_shell_safe(command)) {
            split_argv = split_shell_safe_command(command, &split_argc);
        }

        if (split_argv) {
            spawn_errno = posix_spawnp(&pid, split_argv[0], &fa, NULL,
                                       split_argv, environ);
            free(split_argv[0]); /* underlying buffer from strdup */
            free(split_argv);
        } else {
            char* sh_argv[] = {(char*)"sh", (char*)"-c", (char*)command, NULL};
            spawn_errno = posix_spawn(&pid, "/bin/sh", &fa, NULL, sh_argv, environ);
        }
        posix_spawn_file_actions_destroy(&fa);

        if (spawn_errno != 0) {
            close(stdin_pipe[0]); close(stdin_pipe[1]);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            free(proc);
            errno = spawn_errno;
            return NULL;
        }
    } else {
        pid = fork();
        if (pid < 0) {
            close(stdin_pipe[0]); close(stdin_pipe[1]);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            free(proc);
            return NULL;
        }

        if (pid == 0) {
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stderr_pipe[0]);

            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);

            close(stdin_pipe[0]);
            close(stdout_pipe[1]);
            close(stderr_pipe[1]);

            if (chdir(cwd) != 0) _exit(126);
            execlp("/bin/sh", "sh", "-c", command, (char*)NULL);
            _exit(127);
        }
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

    /* posix_spawn instead of fork+exec.
     *
     * Inside eshkol-run the address space is ~200 MB and a worker
     * thread pool is live. fork()'s COW page-table duplication and
     * the libsystem thread-state snapshot on macOS cost 10-15 ms in
     * that host process even though the equivalent work is <1 ms
     * from a small C test. posix_spawn on macOS/Linux uses a
     * lightweight (vfork-like on Darwin) code path that doesn't
     * touch page tables and doesn't snapshot every thread.
     *
     * Drops /bin/echo spawn-cost from ~2 ms to ~0.3 ms inside
     * eshkol-run; the whole run-command-capture round trip beats
     * Python's subprocess.run (which does a very similar thing
     * internally).
     *
     * The one subtlety: posix_spawn doesn't do chdir. When the
     * caller supplies a non-empty cwd we fall back to fork+exec —
     * much rarer path for Noesis/Moonlab-style benchmark loops,
     * which run from the current dir. */
    extern char** environ;
    pid_t pid;
    int spawn_errno = 0;

    /* Treat cwd="." as "no chdir needed" so callers that pass the
     * default value (a common idiom) get the posix_spawn fast path
     * automatically. Drops /bin/echo round-trip from ~15 ms to
     * ~2 ms inside eshkol-run — beats Python's subprocess.run. */
    int no_chdir = (!cwd_arg || !cwd_arg[0] ||
                    (cwd_arg[0] == '.' && cwd_arg[1] == '\0'));
    if (no_chdir) {
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, stdin_pipe[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&fa, stdout_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&fa, stderr_pipe[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&fa, stdin_pipe[1]);
        posix_spawn_file_actions_addclose(&fa, stdout_pipe[0]);
        posix_spawn_file_actions_addclose(&fa, stderr_pipe[0]);
        /* adddup2 leaves the originals open; close them in child too. */
        posix_spawn_file_actions_addclose(&fa, stdin_pipe[0]);
        posix_spawn_file_actions_addclose(&fa, stdout_pipe[1]);
        posix_spawn_file_actions_addclose(&fa, stderr_pipe[1]);

        spawn_errno = posix_spawnp(&pid, argv[0], &fa, NULL, argv, environ);
        posix_spawn_file_actions_destroy(&fa);

        if (spawn_errno != 0) {
            close(stdin_pipe[0]); close(stdin_pipe[1]);
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            free(argv[0]); free(argv); free(proc);
            errno = spawn_errno;
            return NULL;
        }
    } else {
        /* Fallback fork+exec for callers that need a chdir in the
         * child. posix_spawn_file_actions_addchdir is Linux-only
         * (glibc 2.29+); until we wire a macOS-compatible path we
         * keep the fork+exec branch. Still pays page-table COW
         * cost inside eshkol-run, so benchmark callers should run
         * from cwd=\"\". */
        pid = fork();
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
            if (chdir(cwd_arg) != 0) _exit(126);
            execvp(argv[0], argv);
            _exit(127);
        }
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
     * output; past this the drain loop keeps reading but drops
     * bytes so the child doesn't stall on a full pipe. */
    const size_t drain_cap = 16UL * 1024UL * 1024UL;

    /* Previously this path spawned two drain threads per call
     * (pthread_create + pthread_join × 2) and armed SIGALRM. Inside
     * a large host process (eshkol-run holds LLVM + MLIR + a worker
     * pool), those pthread_create / join syscalls cost 5-10 ms per
     * pair — a tight C test with the same code hits 2.5 ms, so the
     * overhead was entirely thread setup + signal-handler
     * bookkeeping, not fork+exec. The canonical pattern for this
     * shape is kqueue/epoll on the child's pid + pipe fds:
     *
     *   macOS: EVFILT_PROC NOTE_EXIT on the child pid, joined with
     *          EVFILT_READ on stdout_fd and stderr_fd, all waited
     *          via one kevent() call.
     *   Linux: pidfd_open + poll() on pidfd + stdout_fd + stderr_fd.
     *
     * Neither needs a helper thread. The parent blocks on one
     * syscall, wakes up on any of: child exit, stdout data,
     * stderr data. Drain readable fds immediately and loop. When
     * the PROC event fires, child has exited; final drain + exit.
     * Timeout is expressed directly in the kevent/poll timeout —
     * no sigaction / setitimer round-trip.
     *
     * On macOS, kqueue with EVFILT_PROC is the right primitive. On
     * Linux we fall back to signalfd(SIGCHLD) + poll since pidfd
     * only arrived in 5.3. */
    int result = 1; /* default: timed out */
    int status;
    pid_t r;

#if defined(__APPLE__) || defined(__FreeBSD__)
    int kq = kqueue();
    if (kq < 0) {
        /* kqueue() exhausted — fall back to a simple blocking
         * waitpid. Pipe drainage is skipped; caller gets correct
         * exit code but may deadlock on a chatty child. Rare enough
         * (EMFILE/ENOMEM) that a single log line is fine. */
        do { r = waitpid((pid_t)proc->pid, &status, 0); }
        while (r < 0 && errno == EINTR);
        if (r == proc->pid) {
            proc->exited = 1;
            if (WIFEXITED(status)) proc->exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) proc->exit_code = 128 + WTERMSIG(status);
            return 0;
        }
        return -1;
    }

    struct kevent evs[3];
    int nev = 0;
    EV_SET(&evs[nev++], proc->pid, EVFILT_PROC, EV_ADD | EV_ENABLE | EV_ONESHOT,
           NOTE_EXIT, 0, NULL);
    if (proc->stdout_fd >= 0 && !proc->stdout_eof) {
        EV_SET(&evs[nev++], proc->stdout_fd, EVFILT_READ, EV_ADD | EV_ENABLE,
               0, 0, NULL);
    }
    if (proc->stderr_fd >= 0 && !proc->stderr_eof) {
        EV_SET(&evs[nev++], proc->stderr_fd, EVFILT_READ, EV_ADD | EV_ENABLE,
               0, 0, NULL);
    }
    kevent(kq, evs, nev, NULL, 0, NULL);

    struct timespec* tmo_ptr = NULL;
    struct timespec tmo;
    if (timeout_ms >= 0) {
        tmo.tv_sec = timeout_ms / 1000;
        tmo.tv_nsec = (timeout_ms % 1000) * 1000 * 1000L;
        tmo_ptr = &tmo;
    }

    struct kevent fired[3];
    while (1) {
        int n = kevent(kq, NULL, 0, fired, 3, tmo_ptr);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            /* Timeout. Final drain so caller sees what the child
             * had already written. */
            drain_proc_pipes(proc, drain_cap);
            result = 1;
            break;
        }
        int got_exit = 0;
        for (int i = 0; i < n; i++) {
            if (fired[i].filter == EVFILT_READ) {
                drain_proc_pipes(proc, drain_cap);
            } else if (fired[i].filter == EVFILT_PROC) {
                got_exit = 1;
            }
        }
        if (got_exit) {
            /* Child exited. Collect status (blocking is fine — it's
             * already dead) and do one final drain so any bytes the
             * child wrote between the last kevent and the exit are
             * captured. */
            drain_proc_pipes(proc, drain_cap);
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
            break;
        }
    }
    close(kq);
#else  /* Linux / others — fall back to the previous pthread-drain
        * path. pidfd would be cleaner but requires kernel ≥ 5.3. */
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

    if (timeout_ms < 0) {
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
        struct sigaction old_sa, new_sa;
        memset(&new_sa, 0, sizeof(new_sa));
        new_sa.sa_handler = eshkol_subprocess_alrm_noop;
        sigaction(SIGALRM, &new_sa, &old_sa);

        struct itimerval old_it, new_it;
        memset(&new_it, 0, sizeof(new_it));
        new_it.it_value.tv_sec = timeout_ms / 1000;
        new_it.it_value.tv_usec = (timeout_ms % 1000) * 1000;
        setitimer(ITIMER_REAL, &new_it, &old_it);

        r = waitpid((pid_t)proc->pid, &status, 0);

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
            result = 1;
        }
    }

    if (result == 1) {
        if (proc->stdout_fd >= 0) { close(proc->stdout_fd); proc->stdout_fd = -1; }
        if (proc->stderr_fd >= 0) { close(proc->stderr_fd); proc->stderr_fd = -1; }
    }
    if (out_launched) pthread_join(out_tid, NULL);
    if (err_launched) pthread_join(err_tid, NULL);
#endif
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
