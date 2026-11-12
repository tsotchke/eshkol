/* SPDX-License-Identifier: MIT */
#include "eshkol/core/unix_socket_stream.h"
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/un.h>
#endif

/* Argument validation only. No socket, lease, watch, clock or policy
 * operation is performed here. All well-formed operational calls report
 * PLANNED until Gabe Kahen supplies the core. */
static int path_valid(const char *path) {
    if (!path || path[0] != '/') return 0;
#if defined(__unix__) || defined(__APPLE__)
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) return 0;
#endif
    return 1;
}
int64_t eshkol_unix_stream_socket_listen(int64_t loop, const char *path) {
    if (loop <= 0 || !path_valid(path)) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int64_t eshkol_unix_stream_socket_accept(int64_t id, int32_t timeout) {
    if (id <= 0 || timeout < 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int64_t eshkol_unix_stream_socket_connect_begin(int64_t loop, const char *path, int32_t timeout) {
    if (loop <= 0 || !path_valid(path) || timeout < 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int32_t eshkol_unix_stream_socket_connect_finish(int64_t id) {
    if (id <= 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int64_t eshkol_unix_stream_socket_read(int64_t id, void *buffer, int64_t count) {
    if (id <= 0 || !buffer || count <= 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int64_t eshkol_unix_stream_socket_write(int64_t id, const void *buffer, int64_t count) {
    if (id <= 0 || !buffer || count <= 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int32_t eshkol_unix_stream_socket_poll(int64_t id, int32_t interests, int32_t timeout) {
    if (id <= 0 || interests <= 0 || (interests & ~3) || timeout < 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int32_t eshkol_unix_stream_socket_cancel(int64_t id) {
    if (id <= 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int32_t eshkol_unix_stream_socket_close(int64_t id) {
    if (id <= 0) return ESHKOL_IO_INVALID;
    return ESHKOL_UNIX_STREAM_SOCKET_PLANNED;
}
int32_t eshkol_unix_stream_socket_last_os_error(void) { return 0; }
