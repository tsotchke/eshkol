/* SPDX-License-Identifier: MIT */
#include <stdint.h>
#include <string.h>
#include "eshkol/eshkol.h"
#include "eshkol/core/subprocess_stream.h"

/* Eshkol bytevector payloads contain an i64 length followed by their bytes. */
int64_t eshkol_process_stream_read_bytevector(int64_t id, int32_t which,
                                            void *buffer, int64_t start,
                                            int64_t count) {
    if (!buffer || ESHKOL_GET_SUBTYPE(buffer) != HEAP_SUBTYPE_BYTEVECTOR ||
        start < 0 || count <= 0)
        return ESHKOL_PROCESS_STREAM_EINVAL;
    int64_t length;
    memcpy(&length, buffer, sizeof(length));
    if (length < 0 || start > length || count > length - start)
        return ESHKOL_PROCESS_STREAM_EINVAL;
    return eshkol_process_stream_read(id, which,
        (unsigned char *)buffer + sizeof(length) + start, count);
}
