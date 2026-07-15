#ifndef ESHKOL_AGENT_HTTP_H
#define ESHKOL_AGENT_HTTP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qllm_http_response qllm_http_response_t;
typedef struct eshkol_sse_event eshkol_sse_event_t;

/*
 * Stable Eshkol-owned HTTP ABI.
 *
 * header_lines is a UTF-8 string containing zero or more RFC 9110 field
 * lines separated by '\n'.  Embedded CR/LF/control bytes are rejected by the
 * Scheme wrapper before this boundary; native callers must follow the same
 * rule.  The bytes entry point is the canonical ABI and is binary-safe;
 * eshkol_http_request() is a source-compatible text convenience wrapper.
 */
qllm_http_response_t* eshkol_http_request_bytes(const char* method,
                                                const char* url,
                                                const char* header_lines,
                                                const char* body,
                                                int64_t body_len,
                                                int32_t timeout_ms);
qllm_http_response_t* eshkol_http_request(const char* method,
                                          const char* url,
                                          const char* header_lines,
                                          const char* body,
                                          int32_t timeout_ms);

/* Incremental Server-Sent Events transport.  The open call establishes the
 * HTTP response; next drives the connection until one complete SSE event,
 * end-of-stream, an error, or timeout.  A returned event is owned by the
 * caller and must be released with eshkol_sse_event_free(). */
void* eshkol_http_stream_open_bytes(const char* method,
                                    const char* url,
                                    const char* header_lines,
                                    const char* body,
                                    int64_t body_len,
                                    int32_t timeout_ms);
void* eshkol_http_stream_open(const char* method,
                              const char* url,
                              const char* header_lines,
                              const char* body,
                              int32_t timeout_ms);
eshkol_sse_event_t* eshkol_http_stream_next(void* stream,
                                            int32_t timeout_ms);
int32_t eshkol_http_stream_done(void* stream);
const char* eshkol_http_stream_error(void* stream);
void eshkol_http_stream_close(void* stream);

const char* eshkol_sse_event_type(const eshkol_sse_event_t* event);
const char* eshkol_sse_event_data(const eshkol_sse_event_t* event);
const char* eshkol_sse_event_id(const eshkol_sse_event_t* event);
int64_t eshkol_sse_event_retry_ms(const eshkol_sse_event_t* event);
void eshkol_sse_event_free(eshkol_sse_event_t* event);

/* Existing synchronous compatibility surface.  These functions use the same
 * implementation and ownership rules as eshkol_http_request(). */
int32_t qllm_http_init(void);
void qllm_http_shutdown(void);
int32_t qllm_http_has_ssl(void);
qllm_http_response_t* qllm_http_get(const char* url, int32_t timeout_ms);
qllm_http_response_t* qllm_http_post(const char* url,
                                     const char** headers,
                                     int64_t header_count,
                                     const char* body,
                                     int64_t body_len,
                                     int32_t timeout_ms);
qllm_http_response_t* qllm_http_post_json(const char* url,
                                          const char* body,
                                          const char* auth_header,
                                          int32_t timeout_ms);
int32_t qllm_http_response_status(qllm_http_response_t* response);
const char* qllm_http_response_body(qllm_http_response_t* response);
int64_t qllm_http_response_body_len(qllm_http_response_t* response);
const char* qllm_http_response_error(qllm_http_response_t* response);
void qllm_http_response_free(qllm_http_response_t* response);
const char* qllm_http_error_string(int32_t code);

#ifdef __cplusplus
}
#endif

#endif
