/* SPDX-License-Identifier: MIT */
#ifndef ESHKOL_CORE_UNIX_SOCKET_STREAM_H
#define ESHKOL_CORE_UNIX_SOCKET_STREAM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Unix-domain stream sockets on the existing event loop. This header is the
 * paired-build scaffold described in
 * Selene apprentice/gabe-kahen/notes/IPC_DESIGN-2026-11-02.md ("Later Unix
 * streams" row) and the week's socket design addendum. Every entry point
 * below validates the arguments it can check without touching a socket, and
 * otherwise returns ESHKOL_UNIX_STREAM_SOCKET_PLANNED. No socket is created, bound,
 * connected, read, written, or closed by this file.
 *
 * The core implementation (listen/accept/connect/read/write/cancel/close over
 * epoll/kqueue, with a SOCKET-kind lease in the existing handle registry,
 * following the same acquire/borrow/release/close rules as the PROCESS
 * session in inc/eshkol/core/subprocess_stream.h) is owned by Gabe Kahen and
 * lands in a follow-on PR stacked on this scaffold branch.
 */

enum {
    ESHKOL_UNIX_STREAM_SOCKET_EINVAL = -1,
    ESHKOL_UNIX_STREAM_SOCKET_WOULD_BLOCK = -2,
    ESHKOL_UNIX_STREAM_SOCKET_ENOTSUP = -3,
    ESHKOL_UNIX_STREAM_SOCKET_ESYSTEM = -4,
    ESHKOL_UNIX_STREAM_SOCKET_EBUSY = -5,
    ESHKOL_UNIX_STREAM_SOCKET_ETIMEDOUT = -6,
    ESHKOL_UNIX_STREAM_SOCKET_EDENIED = -7,
    ESHKOL_UNIX_STREAM_SOCKET_ECANCELED = -8,
    /* A listen path that already exists and is not confirmed stale. */
    ESHKOL_UNIX_STREAM_SOCKET_EEXIST = -9,
    /* Returned by every entry point below until the core implementation
     * replaces this scaffold. Distinct from every other negative result so
     * callers and tests can tell "planned" apart from a real failure. */
    ESHKOL_UNIX_STREAM_SOCKET_PLANNED = -100
};

/* Bind and listen on a filesystem path on the given event loop. A listener
 * owns the path: close unlinks it. A pre-existing path that nothing is
 * listening on (stale) must be refused, not silently unlinked and reused. */
int64_t eshkol_unix_stream_socket_listen(int64_t loop, const char *path);
/* Accept one connection within one bounded, monotonic deadline. Zero makes
 * one nonblocking pass; it never sleeps. */
int64_t eshkol_unix_stream_socket_accept(int64_t listener, int32_t deadline_ms);
/* Connect within one monotonic deadline shared across retries. */
int64_t eshkol_unix_stream_socket_connect(int64_t loop, const char *path, int32_t deadline_ms);
/* Byte-counted nonblocking read: a positive count, zero for EOF,
 * ESHKOL_UNIX_STREAM_SOCKET_WOULD_BLOCK, or another negative error. NUL bytes and
 * payloads over 16 MiB survive unmodified; reads may be partial. */
int64_t eshkol_unix_stream_socket_read(int64_t id, void *buffer, int64_t capacity);
/* Byte-counted nonblocking write, possibly partial. A closed peer is
 * reported as ESHKOL_UNIX_STREAM_SOCKET_ESYSTEM; SIGPIPE is suppressed by the
 * write path itself, never left for the caller's process-wide disposition. */
int64_t eshkol_unix_stream_socket_write(int64_t id, const void *buffer, int64_t count);
/* Request cancellation of an outstanding accept, connect, read, or write on
 * this identity. Idempotent; never restarts a deadline. */
int32_t eshkol_unix_stream_socket_cancel(int64_t id);
/* Close a listener, an accepted peer, or an outbound connection exactly
 * once. A listener close unlinks the path it bound. A repeated or invalid
 * close returns ESHKOL_UNIX_STREAM_SOCKET_EINVAL rather than touching memory again. */
int32_t eshkol_unix_stream_socket_close(int64_t id);

#ifdef __cplusplus
}
#endif
#endif
