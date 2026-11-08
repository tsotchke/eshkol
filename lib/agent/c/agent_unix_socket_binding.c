/* SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <string.h>
#include "eshkol/eshkol.h"
#include "eshkol/core/unix_socket_stream.h"

/* Eshkol bytevector payloads contain an i64 length followed by their bytes.
 * These bindings only translate the bytevector boundary; they never touch a
 * socket themselves (see lib/core/unix_socket_stream_stub.c). */

int64_t eshkol_unix_socket_read_bytevector(int64_t id, void *buffer,
                                            int64_t start, int64_t count) {
    if (!buffer || ESHKOL_GET_SUBTYPE(buffer) != HEAP_SUBTYPE_BYTEVECTOR ||
        start < 0 || count <= 0)
        return ESHKOL_UNIX_SOCKET_EINVAL;
    int64_t length;
    memcpy(&length, buffer, sizeof(length));
    if (length < 0 || start > length || count > length - start)
        return ESHKOL_UNIX_SOCKET_EINVAL;
    return eshkol_unix_socket_read(id, (unsigned char *)buffer + sizeof(length) + start, count);
}

int64_t eshkol_unix_socket_write_bytevector(int64_t id, void *buffer,
                                             int64_t start, int64_t count) {
    if (!buffer || ESHKOL_GET_SUBTYPE(buffer) != HEAP_SUBTYPE_BYTEVECTOR ||
        start < 0 || count <= 0)
        return ESHKOL_UNIX_SOCKET_EINVAL;
    int64_t length;
    memcpy(&length, buffer, sizeof(length));
    if (length < 0 || start > length || count > length - start)
        return ESHKOL_UNIX_SOCKET_EINVAL;
    return eshkol_unix_socket_write(id, (const unsigned char *)buffer + sizeof(length) + start, count);
}
