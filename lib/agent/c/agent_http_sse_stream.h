#ifndef ESHKOL_AGENT_HTTP_SSE_STREAM_H
#define ESHKOL_AGENT_HTTP_SSE_STREAM_H

#include "eshkol/agent_http_sse_adapter.h"
#include "eshkol/core/resource_lease.h"

typedef struct HttpSseStream eshkol_http_sse_stream_t;

#define ESHKOL_HTTP_SSE_FINISHED 2

/*
 * Native HTTP/1.1 SSE core. Use on the loop owner thread only. Call begin
 * after accepting and routing a valid HTTP/1.1 GET. If begin fails, the
 * HTTP server retains the client. On success, progress sends the headers.
 */
int eshkol_http_sse_stream_begin(int64_t server, int64_t loop, uint64_t watch_cookie,
                                 eshkol_http_sse_stream_t** out);

/*
 * Admit one event. Strings are NUL-terminated UTF-8. NULL omits metadata;
 * an empty event type is also omitted, and an empty id resets the client id.
 * Busy leaves the existing frame and deadline untouched. Oversize returns
 * ESHKOL_IO_TRUNCATED. Admission itself performs no I/O.
 */
int eshkol_http_sse_stream_send(eshkol_http_sse_stream_t* stream, const char* data,
                                const char* event_type, const char* id);
/* Explicit byte counts preserve embedded NULs for validation. NULL
 * metadata must have zero length. Stream state is checked first. */
int eshkol_http_sse_stream_send_bytes(eshkol_http_sse_stream_t* stream,
                                      const char* data, int64_t data_len,
                                      const char* event_type, int64_t event_type_len,
                                      const char* id, int64_t id_len);
int eshkol_http_sse_stream_finish(eshkol_http_sse_stream_t* stream);

/*
 * Call for initial progress, the watch cookie when writable, or the next
 * deadline even when no fd is ready. Return zero when ready for an event,
 * ESHKOL_SSE_WOULD_BLOCK while pending, or ESHKOL_HTTP_SSE_FINISHED after
 * sending the final chunk and requesting cleanup. Errors from the write
 * step or idle expiration retire the connection.
 */
int eshkol_http_sse_stream_progress(eshkol_http_sse_stream_t* stream);
int64_t eshkol_http_sse_stream_deadline(const eshkol_http_sse_stream_t* stream);

/*
 * Dispose without sending anything. The caller owns the returned cleanup
 * ticket and must observe its result, retry any retryable failure, and
 * release it. If close is rejected, the stream remains owned by the caller.
 */
int eshkol_http_sse_stream_end(eshkol_http_sse_stream_t* stream, int64_t* out_ticket);

#endif
