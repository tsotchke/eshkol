#include "agent_http_sse_stream.h"
#include "eshkol/agent_capabilities.h"
#include "eshkol/core/event_loop.h"
#include "eshkol/core/websocket_handshake.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int eshkol_capability_runtime_allows(const char* capability);

enum { CHUNK_PREFIX_MAX = 6 };

struct HttpSseStream {
    eshkol_http_sse_adapter_t* adapter;
    int64_t loop;
    int64_t lease;
    int64_t ticket;
    int64_t deadline;
    uint64_t cookie;
    size_t start;
    size_t length;
    size_t offset;
    int finishing;
    int retired;
    uint8_t pending[ESHKOL_SSE_MAX_CHUNK_BYTES];
};

static int on_owner_loop(const eshkol_http_sse_stream_t* stream) {
    if (!stream) return 0;
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(stream->loop);
    return loop && eshkol_event_loop_is_owner(loop);
}

static int retire(eshkol_http_sse_stream_t* stream) {
    stream->retired = 1;
    stream->deadline = -1;
    if (!stream->lease) return 0;
    int64_t ticket = eshkol_http_sse_adapter_request_close(stream->lease);
    if (ticket < 0) return (int)ticket;
    stream->lease = 0;
    stream->adapter = NULL;
    stream->ticket = ticket;
    return 0;
}

static int stop(eshkol_http_sse_stream_t* stream, int outcome) {
    int status = retire(stream);
    return status < 0 ? status : outcome;
}

static void admit(eshkol_http_sse_stream_t* stream, size_t start, size_t length,
                  int64_t deadline) {
    stream->start = start;
    stream->length = length;
    stream->offset = 0;
    stream->deadline = deadline;
}

int eshkol_http_sse_stream_begin(int64_t server, int64_t loop, uint64_t watch_cookie,
                                 eshkol_http_sse_stream_t** out) {
    static const char headers[] = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Transfer-Encoding: chunked\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: close\r\n\r\n";
    if (!out) return ESHKOL_IO_INVALID;
    *out = NULL;
    if (!eshkol_capability_runtime_allows("network")) {
        errno = EACCES;
        return ESHKOL_IO_SYSTEM;
    }
    int64_t deadline = eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS);
    if (deadline < 0) return ESHKOL_IO_SYSTEM;
    eshkol_http_sse_stream_t* stream = calloc(1, sizeof(*stream));
    if (!stream) return ESHKOL_IO_SYSTEM;
    int status = eshkol_http_sse_adapter_adopt(server, loop, &stream->lease, &stream->adapter);
    if (status < 0) {
        free(stream);
        return status;
    }
    stream->loop = loop;
    stream->cookie = watch_cookie;
    memcpy(stream->pending, headers, sizeof(headers) - 1);
    admit(stream, 0, sizeof(headers) - 1, deadline);
    *out = stream;
    return 0;
}

static int check_idle(eshkol_http_sse_stream_t* stream) {
    int32_t remaining = eshkol_resource_remaining(stream->deadline);
    if (remaining == 0) return stop(stream, ESHKOL_IO_TIMEOUT);
    if (remaining < 0) return stop(stream, ESHKOL_IO_SYSTEM);
    return 0;
}

static int check_available(eshkol_http_sse_stream_t* stream) {
    if (!on_owner_loop(stream) || stream->retired) return ESHKOL_IO_INVALID;
    if (stream->length) return ESHKOL_IO_BUSY;
    return check_idle(stream);
}

static int append(uint8_t* buffer, size_t* used, const char* bytes, size_t length) {
    if (length > ESHKOL_SSE_MAX_CHUNK_BYTES - *used) return ESHKOL_IO_TRUNCATED;
    memcpy(buffer + *used, bytes, length);
    *used += length;
    return 0;
}

static int64_t bounded_string_length(const char* text) {
    size_t length = 0;
    if (text) {
        while (length < ESHKOL_SSE_MAX_CHUNK_BYTES && text[length]) ++length;
    }
    return (int64_t)length;
}

static int check_text(const char* text, int64_t length) {
    if (length < 0 || (!text && length != 0)) return ESHKOL_IO_INVALID;
    /* A valid chunk always needs framing bytes in addition to its fields. */
    if (length >= (int64_t)ESHKOL_SSE_MAX_CHUNK_BYTES) return ESHKOL_IO_TRUNCATED;
    size_t size = (size_t)length;
    if (size && (memchr(text, 0, size) || !eshkol_ws_valid_utf8(text, size))) {
        return ESHKOL_IO_INVALID;
    }
    return 0;
}

static int append_field(uint8_t* buffer, size_t* used, const char* name, size_t name_len,
                        const char* value, size_t value_len) {
    int status = append(buffer, used, name, name_len);
    if (status == 0) status = append(buffer, used, value, value_len);
    if (status == 0) status = append(buffer, used, "\n", 1);
    return status;
}

static int append_metadata(uint8_t* buffer, size_t* used, const char* name, size_t name_len,
                           const char* value, size_t size) {
    if (memchr(value, '\r', size) || memchr(value, '\n', size)) return ESHKOL_IO_INVALID;
    return append_field(buffer, used, name, name_len, value, size);
}

static int append_data(uint8_t* buffer, size_t* used, const char* data, size_t size) {
    const char* const end = data + size;
    const char* line = data;
    for (const char* p = data; p < end; ++p) {
        if (*p != '\r' && *p != '\n') continue;
        int status = append_field(buffer, used, "data: ", 6, line, (size_t)(p - line));
        if (status < 0) return status;
        if (*p == '\r' && p + 1 < end && p[1] == '\n') ++p;
        line = p + 1;
    }
    return append_field(buffer, used, "data: ", 6, line, (size_t)(end - line));
}

static int admit_event(eshkol_http_sse_stream_t* stream, const char* data, int64_t data_len,
                       const char* event_type, int64_t event_type_len,
                       const char* id, int64_t id_len) {
    if (!data) return ESHKOL_IO_INVALID;
    int status = check_text(data, data_len);
    if (status == 0) status = check_text(event_type, event_type_len);
    if (status == 0) status = check_text(id, id_len);
    if (status < 0) return status;

    size_t used = CHUNK_PREFIX_MAX;
    if (event_type_len != 0) {
        status = append_metadata(stream->pending, &used, "event: ", 7,
                                 event_type, (size_t)event_type_len);
    }
    if (status == 0 && id) {
        status = append_metadata(stream->pending, &used, "id: ", 4, id, (size_t)id_len);
    }
    if (status == 0) status = append_data(stream->pending, &used, data, (size_t)data_len);
    if (status == 0) status = append(stream->pending, &used, "\n", 1);
    if (status < 0) return status;

    char prefix[CHUNK_PREFIX_MAX + 1];
    int size = snprintf(prefix, sizeof(prefix), "%zx\r\n", used - CHUNK_PREFIX_MAX);
    if (size <= 0 || size > CHUNK_PREFIX_MAX) return ESHKOL_IO_SYSTEM;
    status = append(stream->pending, &used, "\r\n", 2);
    if (status < 0) return status;

    status = check_idle(stream);
    if (status < 0) return status;
    int64_t deadline = eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS);
    if (deadline < 0) return stop(stream, ESHKOL_IO_SYSTEM);
    size_t start = CHUNK_PREFIX_MAX - (size_t)size;
    memcpy(stream->pending + start, prefix, (size_t)size);
    admit(stream, start, used - start, deadline);
    return 0;
}

int eshkol_http_sse_stream_send(eshkol_http_sse_stream_t* stream, const char* data,
                                const char* event_type, const char* id) {
    int status = check_available(stream);
    if (status < 0) return status;
    return admit_event(stream, data, bounded_string_length(data),
                       event_type, bounded_string_length(event_type),
                       id, bounded_string_length(id));
}

int eshkol_http_sse_stream_send_bytes(eshkol_http_sse_stream_t* stream,
                                      const char* data, int64_t data_len,
                                      const char* event_type, int64_t event_type_len,
                                      const char* id, int64_t id_len) {
    int status = check_available(stream);
    if (status < 0) return status;
    return admit_event(stream, data, data_len, event_type, event_type_len, id, id_len);
}

int eshkol_http_sse_stream_finish(eshkol_http_sse_stream_t* stream) {
    int status = check_available(stream);
    if (status < 0) return status;
    int64_t deadline = eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS);
    if (deadline < 0) return stop(stream, ESHKOL_IO_SYSTEM);
    memcpy(stream->pending, "0\r\n\r\n", 5);
    admit(stream, 0, 5, deadline);
    stream->finishing = 1;
    return 0;
}

int eshkol_http_sse_stream_progress(eshkol_http_sse_stream_t* stream) {
    if (!on_owner_loop(stream) || stream->retired) return ESHKOL_IO_INVALID;
    if (!eshkol_capability_runtime_allows("network")) {
        int status = stop(stream, ESHKOL_IO_SYSTEM);
        errno = EACCES;
        return status;
    }
    if (!stream->length) return check_idle(stream);

    int status = eshkol_http_sse_adapter_write_step(
        stream->lease, stream->adapter, stream->pending + stream->start,
        stream->length, &stream->offset, stream->deadline, stream->cookie, NULL);
    if (status < 0) return stop(stream, status);
    if (status == ESHKOL_SSE_WOULD_BLOCK) return status;
    if (stream->finishing) return stop(stream, ESHKOL_HTTP_SSE_FINISHED);

    stream->length = 0;
    stream->deadline = eshkol_resource_deadline(ESHKOL_SSE_IDLE_TIMEOUT_MS);
    return stream->deadline < 0 ? stop(stream, ESHKOL_IO_SYSTEM) : 0;
}

int64_t eshkol_http_sse_stream_deadline(const eshkol_http_sse_stream_t* stream) {
    return stream ? stream->deadline : -1;
}

int eshkol_http_sse_stream_end(eshkol_http_sse_stream_t* stream, int64_t* out_ticket) {
    if (!out_ticket) return ESHKOL_IO_INVALID;
    *out_ticket = 0;
    if (!on_owner_loop(stream)) return ESHKOL_IO_INVALID;
    int status = retire(stream);
    if (status < 0) return status;
    *out_ticket = stream->ticket;
    free(stream);
    return 0;
}

eshkol_http_sse_stream_t* eshkol_http_sse_scope_begin(int64_t server, int64_t loop,
                                                     int64_t cookie) {
    eshkol_http_sse_stream_t* stream = NULL;
    if (cookie < 0) return NULL;
    return eshkol_http_sse_stream_begin(server, loop, (uint64_t)cookie, &stream)
        == 0 ? stream : NULL;
}

int64_t eshkol_http_sse_scope_id(const eshkol_http_sse_stream_t* stream) {
    return stream ? stream->lease : 0;
}

int32_t eshkol_http_sse_scope_close(eshkol_http_sse_stream_t* stream) {
    if (!on_owner_loop(stream)) return ESHKOL_IO_INVALID;
    int status = retire(stream);
    if (status < 0) return status;

    /* Detached pending work stays rooted across a non-local exit. A terminal
     * teardown error remains available by its original lease identity. */
    status = eshkol_resource_cleanup_detach(stream->ticket);
    if (status == 0) free(stream);
    return status;
}

int64_t eshkol_http_sse_scope_cleanup_ticket(int64_t retired_id) {
    return eshkol_resource_cleanup_ticket(retired_id);
}
