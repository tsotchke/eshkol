/* SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <string.h>
#include "eshkol/eshkol.h"
#include "eshkol/core/unix_socket_stream.h"

extern void eshkol_unix_stream_socket_clear_os_error(void);
/* Scope-only recovery handoff; not part of the public resource ABI. */
#include "../../core/resource_handle_internal.h"

/* Align the private Scheme outcome names when shared native codes change. */
_Static_assert(ESHKOL_IO_TIMEOUT == -2, "update Scheme Unix timeout outcome");
_Static_assert(ESHKOL_IO_INVALID == -4, "update Scheme Unix invalid outcome");
_Static_assert(ESHKOL_IO_CANCELLED == -6, "update Scheme Unix cancellation outcome");
_Static_assert(ESHKOL_IO_RESET == -7, "update Scheme Unix reset outcome");

/* Eshkol bytevector payloads contain an i64 length followed by their bytes.
 * These bindings only translate the bytevector boundary; they never touch a
 * socket themselves (see lib/agent/c/agent_unix_socket_stream.c). */

int64_t eshkol_unix_stream_socket_read_bytevector(int64_t id, void *buffer,
                                            int64_t start, int64_t count) {
    if (!buffer || ESHKOL_GET_SUBTYPE(buffer) != HEAP_SUBTYPE_BYTEVECTOR ||
        start < 0 || count <= 0)
        return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    int64_t length;
    memcpy(&length, buffer, sizeof(length));
    if (length < 0 || start > length || count > length - start)
        return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return eshkol_unix_stream_socket_read(id, (unsigned char *)buffer + sizeof(length) + start, count);
}

int64_t eshkol_unix_stream_socket_write_bytevector(int64_t id, void *buffer,
                                             int64_t start, int64_t count) {
    if (!buffer || ESHKOL_GET_SUBTYPE(buffer) != HEAP_SUBTYPE_BYTEVECTOR ||
        start < 0 || count <= 0)
        return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    int64_t length;
    memcpy(&length, buffer, sizeof(length));
    if (length < 0 || start > length || count > length - start)
        return ESHKOL_UNIX_STREAM_SOCKET_EINVAL;
    return eshkol_unix_stream_socket_write(id, (const unsigned char *)buffer + sizeof(length) + start, count);
}

int64_t eshkol_unix_stream_socket_cleanup_result_packed(int64_t ticket) {
    eshkol_unix_stream_socket_clear_os_error();
    eshkol_resource_cleanup_result_t result;
    int status = eshkol_resource_cleanup_result(ticket, &result);
    if (status < 0) return status;
    if (result.state < ESHKOL_CLEANUP_PENDING || result.state > ESHKOL_CLEANUP_FAILED ||
        result.stage < 0 || result.stage > ESHKOL_CLEANUP_LOOP || result.os_error < 0)
        return ESHKOL_IO_SYSTEM;
    return ((int64_t)result.os_error << 6) | ((result.retryable != 0) << 5) |
           (result.stage << 2) | result.state;
}

int32_t eshkol_unix_stream_socket_cleanup_retry(int64_t ticket) {
    eshkol_unix_stream_socket_clear_os_error();
    return eshkol_resource_cleanup_retry(ticket);
}

int32_t eshkol_unix_stream_socket_cleanup_release(int64_t ticket) {
    eshkol_unix_stream_socket_clear_os_error();
    return eshkol_resource_cleanup_release(ticket);
}

int64_t eshkol_unix_stream_socket_cleanup_ticket(int64_t retired_id) {
    eshkol_unix_stream_socket_clear_os_error();
    return eshkol_resource_cleanup_ticket(retired_id);
}

int32_t eshkol_unix_stream_socket_cleanup_detach(int64_t ticket) {
    eshkol_unix_stream_socket_clear_os_error();
    return eshkol_resource_cleanup_detach(ticket);
}
