/*******************************************************************************
 * IO Multiplexing for Eshkol Agent
 *
 * Provides: multi-fd poll, non-blocking I/O, pipe creation, line buffering.
 *
 * This eliminates busy-wait polling loops (50ms sleep) in MCP transport
 * and enables simultaneous API SSE + subprocess + keyboard handling.
 *
 * All functions use standard Eshkol agent FFI conventions.
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
#include <errno.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#endif

#ifdef _WIN32
static HANDLE handle_from_fd(int32_t fd) {
    intptr_t raw = _get_osfhandle(fd);
    return raw == -1 ? INVALID_HANDLE_VALUE : (HANDLE)raw;
}

/* Anonymous Windows pipe handles are not waitable objects. PeekNamedPipe is
 * the documented, non-consuming readiness primitive; callers still get the
 * same 0/1/2/4 result bits as poll(2). */
static int inspect_fd(int32_t fd, int32_t direction, int32_t* ready) {
    HANDLE handle = handle_from_fd(fd);
    if (handle == INVALID_HANDLE_VALUE || !ready) return -1;
    *ready = 0;

    DWORD type = GetFileType(handle);
    if (type == FILE_TYPE_UNKNOWN && GetLastError() != NO_ERROR) {
        *ready = 4;
        return 0;
    }

    if (direction & 1) {
        if (type == FILE_TYPE_PIPE) {
            DWORD available = 0;
            if (PeekNamedPipe(handle, NULL, 0, NULL, &available, NULL)) {
                if (available > 0) *ready |= 1;
            } else {
                DWORD error = GetLastError();
                if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
                    *ready |= 4;
                } else if (error != ERROR_NO_DATA) {
                    *ready |= 4;
                }
            }
        } else if (type == FILE_TYPE_CHAR) {
            DWORD events = 0;
            if (GetNumberOfConsoleInputEvents(handle, &events) && events > 0) {
                *ready |= 1;
            }
        } else {
            /* Disk files are always ready for synchronous reads. */
            *ready |= 1;
        }
    }

    if (direction & 2) {
        if (type == FILE_TYPE_PIPE) {
            DWORD flags = 0, outbound = 0;
            if (GetNamedPipeInfo(handle, &flags, &outbound, NULL, NULL)) {
                *ready |= 2;
            } else {
                DWORD error = GetLastError();
                if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
                    *ready |= 4;
                } else {
                    *ready |= 2;
                }
            }
        } else {
            *ready |= 2;
        }
    }
    return 0;
}
#endif

/*******************************************************************************
 * Multi-fd Poll
 *
 * Wait for any of N file descriptors to become ready for I/O.
 *
 * fds:        array of file descriptor integers
 * directions: array of direction flags (1=POLLIN, 2=POLLOUT, 3=both)
 * nfds:       number of file descriptors
 * timeout_ms: milliseconds to wait (-1 = infinite, 0 = immediate)
 * ready_out:  filled with ready status per fd (0=not ready, 1=readable,
 *             2=writable, 4=error/hangup)
 *
 * Returns: number of ready fds (0=timeout, -1=error)
 ******************************************************************************/

/**
 * @brief Waits for any of several file descriptors to become ready for I/O, batching a poll(2) call.
 *
 * @param fds Array of @p nfds file descriptors to watch.
 * @param directions Array of @p nfds direction flags per fd: bit 0 (1) = watch for readable (POLLIN), bit 1 (2) = watch for writable (POLLOUT); either or both may be set.
 * @param nfds Number of entries in @p fds / @p directions (must be in (0, 256]).
 * @param timeout_ms Milliseconds to wait; -1 blocks indefinitely, 0 polls without blocking.
 * @param ready_out Array of @p nfds ints filled with a status bitmask per fd: 1=readable, 2=writable, 4=error/hangup/invalid (bits may combine).
 * @return Number of ready fds, 0 on timeout (including an interrupting signal), or -1 on invalid arguments or a poll() error.
 */
int32_t eshkol_poll(const int32_t* fds, const int32_t* directions,
                     int32_t nfds, int32_t timeout_ms, int32_t* ready_out) {
    if (!fds || !directions || !ready_out || nfds <= 0 || nfds > 256) return -1;

#ifdef _WIN32
    ULONGLONG start = GetTickCount64();
    DWORD sleep_ms = 0;
    for (;;) {
        int32_t count = 0;
        for (int32_t i = 0; i < nfds; ++i) {
            if (inspect_fd(fds[i], directions[i], &ready_out[i]) != 0) {
                ready_out[i] = 4;
            }
            if (ready_out[i] != 0) ++count;
        }
        if (count > 0 || timeout_ms == 0) return count;
        if (timeout_ms > 0) {
            ULONGLONG elapsed = GetTickCount64() - start;
            if (elapsed >= (ULONGLONG)timeout_ms) return 0;
            DWORD remaining = (DWORD)((ULONGLONG)timeout_ms - elapsed);
            sleep_ms = sleep_ms == 0 ? 1 : (sleep_ms < 8 ? sleep_ms * 2 : 8);
            if (sleep_ms > remaining) sleep_ms = remaining;
        } else {
            sleep_ms = sleep_ms == 0 ? 1 : (sleep_ms < 8 ? sleep_ms * 2 : 8);
        }
        Sleep(sleep_ms);
    }
#else
    struct pollfd pfds[256];
    for (int i = 0; i < nfds; i++) {
        pfds[i].fd = fds[i];
        pfds[i].events = 0;
        if (directions[i] & 1) pfds[i].events |= POLLIN;
        if (directions[i] & 2) pfds[i].events |= POLLOUT;
        pfds[i].revents = 0;
    }

    int ret = poll(pfds, (nfds_t)nfds, timeout_ms);
    if (ret < 0) {
        if (errno == EINTR) return 0;  /* Interrupted, treat as timeout */
        return -1;
    }

    for (int i = 0; i < nfds; i++) {
        ready_out[i] = 0;
        if (pfds[i].revents & POLLIN)  ready_out[i] |= 1;
        if (pfds[i].revents & POLLOUT) ready_out[i] |= 2;
        if (pfds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) ready_out[i] |= 4;
    }

    return (int32_t)ret;
#endif
}

/*******************************************************************************
 * Single-fd convenience poll
 *
 * Check if a single fd is readable within timeout.
 * Returns: 1=readable, 0=timeout, -1=error/hangup
 ******************************************************************************/

/**
 * @brief Checks whether a single file descriptor becomes readable within a timeout.
 *
 * Convenience wrapper around poll() for the common single-fd case.
 *
 * @param fd File descriptor to watch for POLLIN.
 * @param timeout_ms Milliseconds to wait; -1 blocks indefinitely, 0 polls without blocking.
 * @return 1 if readable, 0 on timeout (including an interrupting signal), -1 on error, hangup, or invalid fd.
 */
int32_t eshkol_poll_read(int32_t fd, int32_t timeout_ms) {
#ifdef _WIN32
    int32_t direction = 1;
    int32_t ready = 0;
    int32_t ret = eshkol_poll(&fd, &direction, 1, timeout_ms, &ready);
    if (ret <= 0) return ret;
    if (ready & 4) return -1;
    return (ready & 1) ? 1 : 0;
#else
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret < 0) return (errno == EINTR) ? 0 : -1;
    if (ret == 0) return 0;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return -1;
    return 1;
#endif
}

/*******************************************************************************
 * Non-blocking fd operations
 ******************************************************************************/

/**
 * @brief Sets O_NONBLOCK on a file descriptor.
 *
 * @param fd File descriptor to modify.
 * @return 0 on success, -1 if the current flags could not be read or fcntl(F_SETFL) failed.
 */
int32_t eshkol_fd_set_nonblocking(int32_t fd) {
#ifdef _WIN32
    /* Nonblocking semantics are implemented by read_available using
       PeekNamedPipe. Validate the CRT descriptor here. */
    return handle_from_fd(fd) == INVALID_HANDLE_VALUE ? -1 : 0;
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 ? 0 : -1;
#endif
}

/**
 * @brief Clears O_NONBLOCK on a file descriptor, restoring blocking I/O.
 *
 * @param fd File descriptor to modify.
 * @return 0 on success, -1 if the current flags could not be read or fcntl(F_SETFL) failed.
 */
int32_t eshkol_fd_set_blocking(int32_t fd) {
#ifdef _WIN32
    return handle_from_fd(fd) == INVALID_HANDLE_VALUE ? -1 : 0;
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) == 0 ? 0 : -1;
#endif
}

/*******************************************************************************
 * Non-blocking read
 *
 * Read available data without blocking.
 * Returns: bytes read (>0), 0=nothing available, -1=EOF/error
 ******************************************************************************/

/**
 * @brief Reads whatever data is immediately available from a (typically non-blocking) fd.
 *
 * @param fd File descriptor to read from.
 * @param buf Destination buffer.
 * @param buf_size Size of @p buf in bytes.
 * @return Number of bytes read (> 0), 0 if no data is currently available (EAGAIN/EWOULDBLOCK), or -1 on EOF, error, or invalid arguments.
 */
int32_t eshkol_fd_read_available(int32_t fd, char* buf, int32_t buf_size) {
    if (!buf || buf_size <= 0) return -1;
#ifdef _WIN32
    HANDLE handle = handle_from_fd(fd);
    if (handle == INVALID_HANDLE_VALUE) return -1;
    if (GetFileType(handle) == FILE_TYPE_PIPE) {
        DWORD available = 0;
        if (!PeekNamedPipe(handle, NULL, 0, NULL, &available, NULL)) {
            DWORD error = GetLastError();
            return (error == ERROR_NO_DATA) ? 0 : -1;
        }
        if (available == 0) return 0;
        int32_t requested = available < (DWORD)buf_size ? (int32_t)available : buf_size;
        int n = _read(fd, buf, (unsigned int)requested);
        return n > 0 ? n : -1;
    }
    int n = _read(fd, buf, (unsigned int)buf_size);
    return n > 0 ? n : -1;
#else
    ssize_t n = read(fd, buf, (size_t)buf_size);
    if (n > 0) return (int32_t)n;
    if (n == 0) return -1;  /* EOF */
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
#endif
}

/*******************************************************************************
 * Non-blocking write
 *
 * Write data without blocking.
 * Returns: bytes written (>0), 0=would block, -1=error
 ******************************************************************************/

/**
 * @brief Writes data to a (typically non-blocking) fd without blocking if the write would stall.
 *
 * @param fd File descriptor to write to.
 * @param data Bytes to write.
 * @param len Number of bytes in @p data.
 * @return Number of bytes written (> 0, possibly less than @p len for a partial write), 0 if the write would block (EAGAIN/EWOULDBLOCK), or -1 on error or invalid arguments.
 */
int32_t eshkol_fd_write_available(int32_t fd, const char* data, int32_t len) {
    if (!data || len <= 0) return -1;
#ifdef _WIN32
    HANDLE handle = handle_from_fd(fd);
    if (handle == INVALID_HANDLE_VALUE) return -1;
    int n = _write(fd, data, (unsigned int)len);
    if (n >= 0) return n;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
#else
    ssize_t n = write(fd, data, (size_t)len);
    if (n >= 0) return (int32_t)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
#endif
}

/*******************************************************************************
 * Pipe Creation
 *
 * Create a unidirectional pipe.
 * Returns: 0 success (read_fd and write_fd filled), -1 error
 ******************************************************************************/

/**
 * @brief Creates a unidirectional OS pipe.
 *
 * @param read_fd Out-parameter set to the pipe's read-end fd.
 * @param write_fd Out-parameter set to the pipe's write-end fd.
 * @return 0 on success, -1 if either out-parameter is NULL or pipe() failed.
 */
int32_t eshkol_make_pipe(int32_t* read_fd, int32_t* write_fd) {
    if (!read_fd || !write_fd) return -1;
    int pipefd[2];
#ifdef _WIN32
    if (_pipe(pipefd, 65536, _O_BINARY | _O_NOINHERIT) != 0) return -1;
#else
    if (pipe(pipefd) != 0) return -1;
#endif
    *read_fd = pipefd[0];
    *write_fd = pipefd[1];
    return 0;
}

/*******************************************************************************
 * Line Reader — buffered line-oriented reading from a file descriptor
 *
 * Accumulates reads into an internal buffer and delivers complete lines
 * (terminated by \n). Partial lines are held until the next read completes
 * them or EOF is reached.
 *
 * This is the core primitive for MCP JSON-RPC (one JSON object per line)
 * and subprocess output processing.
 ******************************************************************************/

#define MAX_LINE_READERS 32
#define LINE_READER_BUF_SIZE 65536

typedef struct {
    int fd;
    char buf[LINE_READER_BUF_SIZE];
    int32_t len;       /* bytes in buffer */
    int32_t eof;       /* fd reached EOF */
} LineReader;

static LineReader* g_line_readers[MAX_LINE_READERS] = {0};
static int g_next_reader = 1;

/**
 * @brief Finds a free slot in the global line-reader table.
 *
 * Scans forward from the last allocated index (wrapping around) so
 * slot reuse spreads across the table rather than always restarting
 * at index 1.
 *
 * @return A free slot index in [1, MAX_LINE_READERS), or -1 if the table is full.
 */
static int alloc_reader(void) {
    for (int i = g_next_reader; i < MAX_LINE_READERS; i++) {
        if (!g_line_readers[i]) { g_next_reader = i + 1; return i; }
    }
    for (int i = 1; i < g_next_reader; i++) {
        if (!g_line_readers[i]) { g_next_reader = i + 1; return i; }
    }
    return -1;
}

/**
 * @brief Looks up a LineReader by its handle, validating the handle range.
 *
 * @param handle Handle previously returned by eshkol_line_reader_create().
 * @return The LineReader, or NULL if @p handle is out of range or the slot is unallocated.
 */
static LineReader* get_reader(int64_t handle) {
    if (handle < 1 || handle >= MAX_LINE_READERS) return NULL;
    return g_line_readers[handle];
}

/*
 * Create a line reader for the given file descriptor.
 * The fd should already be open and preferably non-blocking.
 * Returns: handle (>= 1), -1 on error
 */
/**
 * @brief Allocates and registers a buffered line reader for a file descriptor.
 *
 * @p fd should already be open, and preferably set to non-blocking mode
 * (see eshkol_fd_set_nonblocking()) so eshkol_line_reader_next() never stalls.
 *
 * @param fd File descriptor to read lines from; ownership stays with the caller (not closed by this module).
 * @return A handle (>= 1) usable with the other eshkol_line_reader_* functions, or -1 if the reader table is full or allocation failed.
 */
int64_t eshkol_line_reader_create(int32_t fd) {
    int slot = alloc_reader();
    if (slot < 0) return -1;

    LineReader* lr = (LineReader*)calloc(1, sizeof(LineReader));
    if (!lr) return -1;
    lr->fd = fd;
    lr->len = 0;
    lr->eof = 0;
    g_line_readers[slot] = lr;
    return (int64_t)slot;
}

/*
 * Read the next complete line from the reader.
 *
 * This does NOT block. It reads whatever is available on the fd,
 * appends to the internal buffer, and returns the first complete line
 * (including the \n terminator, which is stripped).
 *
 * Returns:
 *   > 0: line length (line is in buf, null-terminated, \n stripped)
 *   0:   no complete line available yet (not an error, call again later)
 *   -1:  EOF or error (no more lines will come)
 *   -2:  handle invalid
 */
/**
 * @brief Reads the next complete, newline-terminated line from a line reader, without blocking.
 *
 * Reads whatever is currently available on the underlying fd into an
 * internal 64KB buffer, then returns the first complete line found
 * (the trailing '\n' is stripped and not included in the output). If
 * EOF is reached with a trailing partial line (no '\n'), that
 * remaining data is delivered as a final line.
 *
 * @param handle Handle from eshkol_line_reader_create().
 * @param out_buf Destination buffer for the line text (NUL-terminated); truncated if longer than @p out_size - 1.
 * @param out_size Size of @p out_buf in bytes.
 * @return Line length in bytes if a complete line was delivered, 0 if
 *   no complete line is available yet (not an error, call again
 *   later), -1 on EOF/error with nothing left to deliver, or -2 if
 *   @p handle or the buffer arguments are invalid.
 */
int32_t eshkol_line_reader_next(int64_t handle, char* out_buf, int32_t out_size) {
    LineReader* lr = get_reader(handle);
    if (!lr) return -2;
    if (!out_buf || out_size <= 0) return -2;

    /* Try to read more data if not at EOF */
    if (!lr->eof && lr->len < LINE_READER_BUF_SIZE - 1) {
        int32_t n = eshkol_fd_read_available(
            lr->fd, lr->buf + lr->len, LINE_READER_BUF_SIZE - 1 - lr->len);
        if (n > 0) {
            lr->len += (int32_t)n;
        } else if (n == 0) {
            lr->eof = 1;
        } else if (n < 0) {
            lr->eof = 1;
        }
    }

    /* Scan for newline */
    char* newline = (char*)memchr(lr->buf, '\n', (size_t)lr->len);
    if (newline) {
        int32_t line_len = (int32_t)(newline - lr->buf);
        if (line_len >= out_size) line_len = out_size - 1;  /* Truncate if needed */

        memcpy(out_buf, lr->buf, (size_t)line_len);
        out_buf[line_len] = '\0';

        /* Remove consumed data (including the \n) from buffer */
        int32_t consumed = (int32_t)(newline - lr->buf) + 1;
        lr->len -= consumed;
        if (lr->len > 0) {
            memmove(lr->buf, newline + 1, (size_t)lr->len);
        }
        return line_len;
    }

    /* No newline found */
    if (lr->eof && lr->len > 0) {
        /* At EOF with remaining data — deliver as final line */
        int32_t line_len = lr->len;
        if (line_len >= out_size) line_len = out_size - 1;
        memcpy(out_buf, lr->buf, (size_t)line_len);
        out_buf[line_len] = '\0';
        lr->len = 0;
        return line_len;
    }

    if (lr->eof) return -1;  /* EOF, nothing left */

    return 0;  /* No complete line yet */
}

/*
 * Check if the reader has reached EOF.
 * Returns: 1 if EOF reached, 0 if more data may come
 */
/**
 * @brief Reports whether a line reader has hit EOF with no buffered data left.
 *
 * @param handle Handle from eshkol_line_reader_create().
 * @return 1 if the underlying fd reached EOF and all buffered bytes have been consumed (or @p handle is invalid), 0 if more data may still arrive.
 */
int32_t eshkol_line_reader_eof(int64_t handle) {
    LineReader* lr = get_reader(handle);
    if (!lr) return 1;
    return lr->eof && lr->len == 0 ? 1 : 0;
}

/*
 * Peek at buffered data without consuming it.
 * Returns: number of bytes currently buffered
 */
/**
 * @brief Returns the number of bytes currently buffered but not yet delivered as a line.
 *
 * @param handle Handle from eshkol_line_reader_create().
 * @return Buffered byte count, or 0 if @p handle is invalid.
 */
int32_t eshkol_line_reader_buffered(int64_t handle) {
    LineReader* lr = get_reader(handle);
    if (!lr) return 0;
    return lr->len;
}

/*
 * Close and free a line reader. Does NOT close the underlying fd.
 */
/**
 * @brief Frees a line reader and removes it from the reader table.
 *
 * Does not close the underlying file descriptor; the caller retains ownership of it.
 *
 * @param handle Handle from eshkol_line_reader_create(); invalid handles are a no-op.
 */
void eshkol_line_reader_close(int64_t handle) {
    LineReader* lr = get_reader(handle);
    if (!lr) return;
    free(lr);
    g_line_readers[handle] = NULL;
}

/*******************************************************************************
 * Process fd extraction
 *
 * These expose the stdout/stderr pipe fds from a qllm_process_t handle
 * so they can be used with poll(). The process struct layout is:
 *   struct qllm_process { pid_t pid; int stdin_fd, stdout_fd, stderr_fd; ... }
 *
 * We cast the opaque pointer and read the fd fields directly. This creates
 * a coupling to the qllm_process_t layout — if that struct changes, these
 * offsets must be updated.
 ******************************************************************************/

/* qllm_process_t layout: { pid_t pid; int stdin_fd; int stdout_fd; int stderr_fd; ... } */
/* On both macOS and Linux, pid_t is int (4 bytes), so offsets are:       */
/*   pid: offset 0 (4 bytes)                                              */
/*   stdin_fd: offset 4 (4 bytes)                                         */
/*   stdout_fd: offset 8 (4 bytes)                                        */
/*   stderr_fd: offset 12 (4 bytes)                                       */

/**
 * @brief Extracts the stdout pipe fd from an opaque qllm_process_t handle.
 *
 * Reinterprets @p proc as a flat array of ints per the qllm_process_t
 * layout `{ pid_t pid; int stdin_fd; int stdout_fd; int stderr_fd; ... }`
 * (pid_t is 4 bytes on both macOS and Linux), so stdout_fd is the 3rd
 * int field. This creates a coupling to that struct's layout: if it
 * changes, this offset must be updated too.
 *
 * @param proc Pointer to a qllm_process_t; NULL is rejected.
 * @return The stdout file descriptor, or -1 if @p proc is NULL.
 */
extern int32_t qllm_process_stdout_fd(void* proc);
extern int32_t qllm_process_stderr_fd(void* proc);
extern int64_t qllm_process_pid(void* proc);

int32_t eshkol_process_stdout_fd(void* proc) {
    return qllm_process_stdout_fd(proc);
}

/**
 * @brief Extracts the stderr pipe fd from an opaque qllm_process_t handle.
 *
 * See eshkol_process_stdout_fd() for the assumed struct layout and its caveats.
 *
 * @param proc Pointer to a qllm_process_t; NULL is rejected.
 * @return The stderr file descriptor, or -1 if @p proc is NULL.
 */
int32_t eshkol_process_stderr_fd(void* proc) {
    return qllm_process_stderr_fd(proc);
}

/**
 * @brief Extracts the process ID from an opaque qllm_process_t handle.
 *
 * See eshkol_process_stdout_fd() for the assumed struct layout and its caveats.
 *
 * @param proc Pointer to a qllm_process_t; NULL is rejected.
 * @return The process ID, or -1 if @p proc is NULL.
 */
int32_t eshkol_process_pid(void* proc) {
    int64_t pid = qllm_process_pid(proc);
    return pid > INT32_MAX ? -1 : (int32_t)pid;
}
