/* SPDX-License-Identifier: MIT */
#include "eshkol/core/unix_socket_stream.h"

/*
 * Scaffold only. No socket is created, bound, connected, read, written, or
 * closed by this file. Every entry point validates the arguments it can
 * check without touching a socket, then reports ESHKOL_UNIX_STREAM_SOCKET_PLANNED
 * so a caller gets a stable, documented "Planned for v1.4.0" result instead
 * of silent success or a hang. Gabe Kahen's core implementation (socket
 * core, paired with this scaffold) replaces this file.
 */

int64_t eshkol_unix_stream_socket_listen(int64_t loop, const char *path) {
    (void)loop;
    if (!path || !path[0]) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int64_t eshkol_unix_stream_socket_accept(int64_t listener, int32_t deadline_ms) {
    if (listener <= 0 || deadline_ms < 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int64_t eshkol_unix_stream_socket_connect(int64_t loop, const char *path, int32_t deadline_ms) {
    (void)loop;
    if (!path || !path[0] || deadline_ms < 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int64_t eshkol_unix_stream_socket_read(int64_t id, void *buffer, int64_t capacity) {
    if (id <= 0 || !buffer || capacity <= 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int64_t eshkol_unix_stream_socket_write(int64_t id, const void *buffer, int64_t count) {
    if (id <= 0 || !buffer || count < 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int32_t eshkol_unix_stream_socket_cancel(int64_t id) {
    if (id <= 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}

int32_t eshkol_unix_stream_socket_close(int64_t id) {
    if (id <= 0) return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
