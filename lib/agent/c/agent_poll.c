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

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>

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

int32_t eshkol_poll(const int32_t* fds, const int32_t* directions,
                     int32_t nfds, int32_t timeout_ms, int32_t* ready_out) {
    if (!fds || !directions || !ready_out || nfds <= 0 || nfds > 256) return -1;

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
}

/*******************************************************************************
 * Single-fd convenience poll
 *
 * Check if a single fd is readable within timeout.
 * Returns: 1=readable, 0=timeout, -1=error/hangup
 ******************************************************************************/

int32_t eshkol_poll_read(int32_t fd, int32_t timeout_ms) {
    struct pollfd pfd = { .fd = fd, .events = POLLIN, .revents = 0 };
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret < 0) return (errno == EINTR) ? 0 : -1;
    if (ret == 0) return 0;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return -1;
    return 1;
}

/*******************************************************************************
 * Non-blocking fd operations
 ******************************************************************************/

int32_t eshkol_fd_set_nonblocking(int32_t fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 ? 0 : -1;
}

int32_t eshkol_fd_set_blocking(int32_t fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) == 0 ? 0 : -1;
}

/*******************************************************************************
 * Non-blocking read
 *
 * Read available data without blocking.
 * Returns: bytes read (>0), 0=nothing available, -1=EOF/error
 ******************************************************************************/

int32_t eshkol_fd_read_available(int32_t fd, char* buf, int32_t buf_size) {
    if (!buf || buf_size <= 0) return -1;
    ssize_t n = read(fd, buf, (size_t)buf_size);
    if (n > 0) return (int32_t)n;
    if (n == 0) return -1;  /* EOF */
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
}

/*******************************************************************************
 * Non-blocking write
 *
 * Write data without blocking.
 * Returns: bytes written (>0), 0=would block, -1=error
 ******************************************************************************/

int32_t eshkol_fd_write_available(int32_t fd, const char* data, int32_t len) {
    if (!data || len <= 0) return -1;
    ssize_t n = write(fd, data, (size_t)len);
    if (n >= 0) return (int32_t)n;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
    return -1;
}

/*******************************************************************************
 * Pipe Creation
 *
 * Create a unidirectional pipe.
 * Returns: 0 success (read_fd and write_fd filled), -1 error
 ******************************************************************************/

int32_t eshkol_make_pipe(int32_t* read_fd, int32_t* write_fd) {
    if (!read_fd || !write_fd) return -1;
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;
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

static int alloc_reader(void) {
    for (int i = g_next_reader; i < MAX_LINE_READERS; i++) {
        if (!g_line_readers[i]) { g_next_reader = i + 1; return i; }
    }
    for (int i = 1; i < g_next_reader; i++) {
        if (!g_line_readers[i]) { g_next_reader = i + 1; return i; }
    }
    return -1;
}

static LineReader* get_reader(int64_t handle) {
    if (handle < 1 || handle >= MAX_LINE_READERS) return NULL;
    return g_line_readers[handle];
}

/*
 * Create a line reader for the given file descriptor.
 * The fd should already be open and preferably non-blocking.
 * Returns: handle (>= 1), -1 on error
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
int32_t eshkol_line_reader_next(int64_t handle, char* out_buf, int32_t out_size) {
    LineReader* lr = get_reader(handle);
    if (!lr) return -2;
    if (!out_buf || out_size <= 0) return -2;

    /* Try to read more data if not at EOF */
    if (!lr->eof && lr->len < LINE_READER_BUF_SIZE - 1) {
        ssize_t n = read(lr->fd, lr->buf + lr->len,
                          (size_t)(LINE_READER_BUF_SIZE - 1 - lr->len));
        if (n > 0) {
            lr->len += (int32_t)n;
        } else if (n == 0) {
            lr->eof = 1;
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
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
int32_t eshkol_line_reader_eof(int64_t handle) {
    LineReader* lr = get_reader(handle);
    if (!lr) return 1;
    return lr->eof && lr->len == 0 ? 1 : 0;
}

/*
 * Peek at buffered data without consuming it.
 * Returns: number of bytes currently buffered
 */
int32_t eshkol_line_reader_buffered(int64_t handle) {
    LineReader* lr = get_reader(handle);
    if (!lr) return 0;
    return lr->len;
}

/*
 * Close and free a line reader. Does NOT close the underlying fd.
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

int32_t eshkol_process_stdout_fd(void* proc) {
    if (!proc) return -1;
    const int* fields = (const int*)proc;
    return fields[2];  /* stdout_fd is the 3rd int field */
}

int32_t eshkol_process_stderr_fd(void* proc) {
    if (!proc) return -1;
    const int* fields = (const int*)proc;
    return fields[3];  /* stderr_fd is the 4th int field */
}

int32_t eshkol_process_pid(void* proc) {
    if (!proc) return -1;
    const int* fields = (const int*)proc;
    return fields[0];  /* pid is the 1st int field */
}
