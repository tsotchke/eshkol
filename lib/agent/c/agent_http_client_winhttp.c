/*
 * agent_http_client_winhttp.c — native Windows HTTP/HTTPS client.
 *
 * Implements the qllm_http_* ABI used by lib/agent/http.esk without a
 * libcurl/MSYS dependency. WinHTTP supplies the Windows trust store, proxy
 * discovery, TLS verification, and redirect handling. Each request owns its
 * connection/request handles; the process-wide session is reference counted
 * and may be shared concurrently by WinHTTP.
 */

#if defined(_WIN32)

#include "eshkol/agent_http.h"
#include "agent_http_internal.h"
#include "agent_sse_internal.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ESHKOL_HTTP_DEFAULT_TIMEOUT_MS 30000
#define ESHKOL_HTTP_CONNECT_TIMEOUT_MS 10000
#define ESHKOL_HTTP_MAX_RESPONSE_BYTES ((size_t)256 * 1024 * 1024)

typedef struct qllm_http_response {
    int32_t status;
    char* body;
    int64_t body_len;
    char* error;
} qllm_http_response_t;

typedef struct body_buf {
    char* data;
    size_t len;
    size_t cap;
} body_buf_t;

static SRWLOCK g_http_lock = SRWLOCK_INIT;
static HINTERNET g_http_session = NULL;
static LONG g_http_init_refs = 0;
static LONG g_http_active_requests = 0;

static void* http_xmalloc(size_t n) {
    void* p = malloc(n == 0 ? 1 : n);
    if (!p) {
        fprintf(stderr, "agent_http_client_winhttp: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

static char* http_strdup_n(const char* src, size_t n) {
    char* out = (char*)http_xmalloc(n + 1);
    if (n > 0 && src) memcpy(out, src, n);
    out[n] = '\0';
    return out;
}

static wchar_t* utf8_to_wide(const char* text) {
    if (!text) return NULL;
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, NULL, 0);
    if (count <= 0) return NULL;
    wchar_t* out = (wchar_t*)http_xmalloc((size_t)count * sizeof(wchar_t));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, out, count) <= 0) {
        free(out);
        return NULL;
    }
    return out;
}

static wchar_t* wide_dup_n(const wchar_t* text, DWORD count) {
    wchar_t* out = (wchar_t*)http_xmalloc(((size_t)count + 1) * sizeof(wchar_t));
    if (count > 0) memcpy(out, text, (size_t)count * sizeof(wchar_t));
    out[count] = L'\0';
    return out;
}

typedef struct header_array {
    char** items;
    int64_t count;
} header_array_t;

static void free_header_array(header_array_t* headers) {
    if (!headers) return;
    for (int64_t i = 0; i < headers->count; ++i) free(headers->items[i]);
    free(headers->items);
    headers->items = NULL;
    headers->count = 0;
}

static int parse_header_lines(const char* lines, header_array_t* headers) {
    memset(headers, 0, sizeof(*headers));
    if (!lines || !*lines) return 1;
    const char* cursor = lines;
    while (*cursor) {
        const char* end = strchr(cursor, '\n');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length == 0) {
            cursor = end ? end + 1 : cursor + length;
            continue;
        }
        const char* colon = (const char*)memchr(cursor, ':', length);
        if (!colon) goto invalid;
        size_t name_length = (size_t)(colon - cursor);
        const char* value = colon + 1;
        size_t value_length = length - name_length - 1;
        while (value_length && *value == ' ') {
            value++;
            value_length--;
        }
        if (!eshkol_http_valid_token(cursor, name_length) ||
            !eshkol_http_valid_field_value(value, value_length)) goto invalid;
        char** grown = (char**)realloc(headers->items,
                                      (size_t)(headers->count + 1) * sizeof(char*));
        if (!grown) goto invalid;
        headers->items = grown;
        headers->items[headers->count] = http_strdup_n(cursor, length);
        headers->count++;
        cursor = end ? end + 1 : cursor + length;
    }
    return 1;
invalid:
    free_header_array(headers);
    return 0;
}

static char* format_windows_error(DWORD code) {
    wchar_t* wide = NULL;
    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                  FORMAT_MESSAGE_IGNORE_INSERTS;
    DWORD chars = FormatMessageW(flags, NULL, code, 0, (wchar_t*)&wide, 0, NULL);
    if (chars == 0 || !wide) {
        char fallback[64];
        int n = snprintf(fallback, sizeof(fallback), "WinHTTP error %lu", (unsigned long)code);
        return http_strdup_n(fallback, n > 0 ? (size_t)n : 0);
    }
    while (chars > 0 && (wide[chars - 1] == L'\r' || wide[chars - 1] == L'\n' ||
                         wide[chars - 1] == L' ')) {
        wide[--chars] = L'\0';
    }
    int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, (int)chars, NULL, 0, NULL, NULL);
    if (bytes <= 0) {
        LocalFree(wide);
        return http_strdup_n("WinHTTP error", 13);
    }
    char* out = (char*)http_xmalloc((size_t)bytes + 1);
    WideCharToMultiByte(CP_UTF8, 0, wide, (int)chars, out, bytes, NULL, NULL);
    out[bytes] = '\0';
    LocalFree(wide);
    return out;
}

static qllm_http_response_t* alloc_response(void) {
    qllm_http_response_t* resp = (qllm_http_response_t*)http_xmalloc(sizeof(*resp));
    memset(resp, 0, sizeof(*resp));
    return resp;
}

static qllm_http_response_t* error_response(DWORD code) {
    qllm_http_response_t* resp = alloc_response();
    resp->error = format_windows_error(code);
    resp->body = http_strdup_n("", 0);
    return resp;
}

static int body_append(body_buf_t* buf, const char* data, size_t count) {
    if (count > ESHKOL_HTTP_MAX_RESPONSE_BYTES - buf->len) return 0;
    size_t required = buf->len + count + 1;
    if (required > buf->cap) {
        size_t new_cap = buf->cap ? buf->cap : 4096;
        while (new_cap < required) {
            if (new_cap > ESHKOL_HTTP_MAX_RESPONSE_BYTES / 2) {
                new_cap = ESHKOL_HTTP_MAX_RESPONSE_BYTES + 1;
                break;
            }
            new_cap *= 2;
        }
        if (new_cap < required || new_cap > ESHKOL_HTTP_MAX_RESPONSE_BYTES + 1) return 0;
        char* grown = (char*)realloc(buf->data, new_cap);
        if (!grown) return 0;
        buf->data = grown;
        buf->cap = new_cap;
    }
    if (count > 0) memcpy(buf->data + buf->len, data, count);
    buf->len += count;
    buf->data[buf->len] = '\0';
    return 1;
}

static HINTERNET create_session(void) {
    return WinHttpOpen(L"eshkol-agent/1.3",
                       WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                       WINHTTP_NO_PROXY_NAME,
                       WINHTTP_NO_PROXY_BYPASS,
                       0);
}

static HINTERNET acquire_session(DWORD* failure) {
    HINTERNET session;
    AcquireSRWLockExclusive(&g_http_lock);
    if (!g_http_session) {
        g_http_session = create_session();
        if (!g_http_session && failure) *failure = GetLastError();
    }
    session = g_http_session;
    if (session) g_http_active_requests++;
    ReleaseSRWLockExclusive(&g_http_lock);
    return session;
}

static void release_session(void) {
    AcquireSRWLockExclusive(&g_http_lock);
    if (g_http_active_requests > 0) g_http_active_requests--;
    if (g_http_init_refs == 0 && g_http_active_requests == 0 && g_http_session) {
        WinHttpCloseHandle(g_http_session);
        g_http_session = NULL;
    }
    ReleaseSRWLockExclusive(&g_http_lock);
}

int32_t qllm_http_init(void) {
    int ok;
    AcquireSRWLockExclusive(&g_http_lock);
    if (!g_http_session) g_http_session = create_session();
    ok = g_http_session != NULL;
    if (ok) g_http_init_refs++;
    ReleaseSRWLockExclusive(&g_http_lock);
    return ok ? 1 : 0;
}

void qllm_http_shutdown(void) {
    AcquireSRWLockExclusive(&g_http_lock);
    if (g_http_init_refs > 0) g_http_init_refs--;
    if (g_http_init_refs == 0 && g_http_active_requests == 0 && g_http_session) {
        WinHttpCloseHandle(g_http_session);
        g_http_session = NULL;
    }
    ReleaseSRWLockExclusive(&g_http_lock);
}

int32_t qllm_http_has_ssl(void) { return 1; }

static qllm_http_response_t* perform_request(const wchar_t* method,
                                             const char* url,
                                             const char** headers,
                                             int64_t header_count,
                                             const char* body,
                                             int64_t body_len,
                                             int32_t timeout_ms) {
    if (!url || body_len < 0 || (uint64_t)body_len > UINT32_MAX ||
        (!body && body_len != 0)) return NULL;
    for (int64_t i = 0; headers && i < header_count; ++i) {
        if (headers[i] && !eshkol_http_valid_header_line(headers[i])) return NULL;
    }
    const int timeout = timeout_ms > 0 ? timeout_ms : ESHKOL_HTTP_DEFAULT_TIMEOUT_MS;
    ULONGLONG deadline = GetTickCount64() + (ULONGLONG)timeout;
    qllm_http_response_t* result = NULL;
    DWORD failure = ERROR_SUCCESS;
    HINTERNET session = acquire_session(&failure);
    HINTERNET connect = NULL;
    HINTERNET request = NULL;
    wchar_t* wide_url = NULL;
    wchar_t* host = NULL;
    wchar_t* path = NULL;
    body_buf_t response_body = {0};

    if (!session) return error_response(failure ? failure : ERROR_WINHTTP_INTERNAL_ERROR);
    wide_url = utf8_to_wide(url);
    if (!wide_url) {
        failure = ERROR_NO_UNICODE_TRANSLATION;
        goto done;
    }

    URL_COMPONENTS parts;
    memset(&parts, 0, sizeof(parts));
    parts.dwStructSize = sizeof(parts);
    parts.dwHostNameLength = (DWORD)-1;
    parts.dwUrlPathLength = (DWORD)-1;
    parts.dwExtraInfoLength = (DWORD)-1;
    parts.dwSchemeLength = (DWORD)-1;
    if (!WinHttpCrackUrl(wide_url, 0, 0, &parts) ||
        (parts.nScheme != INTERNET_SCHEME_HTTP && parts.nScheme != INTERNET_SCHEME_HTTPS) ||
        !parts.lpszHostName || parts.dwHostNameLength == 0) {
        failure = GetLastError();
        if (failure == ERROR_SUCCESS) failure = ERROR_WINHTTP_INVALID_URL;
        goto done;
    }
    host = wide_dup_n(parts.lpszHostName, parts.dwHostNameLength);
    DWORD path_chars = parts.dwUrlPathLength + parts.dwExtraInfoLength;
    if (path_chars == 0) {
        path = wide_dup_n(L"/", 1);
    } else {
        path = (wchar_t*)http_xmalloc(((size_t)path_chars + 1) * sizeof(wchar_t));
        DWORD offset = 0;
        if (parts.dwUrlPathLength > 0) {
            memcpy(path, parts.lpszUrlPath, (size_t)parts.dwUrlPathLength * sizeof(wchar_t));
            offset += parts.dwUrlPathLength;
        }
        if (parts.dwExtraInfoLength > 0) {
            memcpy(path + offset, parts.lpszExtraInfo,
                   (size_t)parts.dwExtraInfoLength * sizeof(wchar_t));
            offset += parts.dwExtraInfoLength;
        }
        path[offset] = L'\0';
    }

    connect = WinHttpConnect(session, host, parts.nPort, 0);
    if (!connect) {
        failure = GetLastError();
        goto done;
    }
    DWORD flags = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    request = WinHttpOpenRequest(connect, method, path, NULL,
                                 WINHTTP_NO_REFERER,
                                 WINHTTP_DEFAULT_ACCEPT_TYPES,
                                 flags);
    if (!request) {
        failure = GetLastError();
        goto done;
    }

    int connect_timeout = timeout < ESHKOL_HTTP_CONNECT_TIMEOUT_MS
                              ? timeout
                              : ESHKOL_HTTP_CONNECT_TIMEOUT_MS;
    if (!WinHttpSetTimeouts(request, connect_timeout, connect_timeout, timeout, timeout)) {
        failure = GetLastError();
        goto done;
    }
    DWORD max_redirects = 10;
    (void)WinHttpSetOption(request, WINHTTP_OPTION_MAX_HTTP_AUTOMATIC_REDIRECTS,
                           &max_redirects, sizeof(max_redirects));

    for (int64_t i = 0; headers && i < header_count; ++i) {
        if (!headers[i]) continue;
        wchar_t* header = utf8_to_wide(headers[i]);
        if (!header) {
            failure = ERROR_NO_UNICODE_TRANSLATION;
            goto done;
        }
        BOOL added = WinHttpAddRequestHeaders(request, header, (DWORD)-1L,
                                              WINHTTP_ADDREQ_FLAG_ADD |
                                              WINHTTP_ADDREQ_FLAG_REPLACE);
        free(header);
        if (!added) {
            failure = GetLastError();
            goto done;
        }
    }

    if (!WinHttpSendRequest(request,
                            WINHTTP_NO_ADDITIONAL_HEADERS,
                            0,
                            body_len > 0 ? (LPVOID)body : WINHTTP_NO_REQUEST_DATA,
                            (DWORD)body_len,
                            (DWORD)body_len,
                            0) ||
        !WinHttpReceiveResponse(request, NULL)) {
        failure = GetLastError();
        goto done;
    }

    result = alloc_response();
    DWORD status = 0;
    DWORD status_size = sizeof(status);
    if (!WinHttpQueryHeaders(request,
                             WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX,
                             &status,
                             &status_size,
                             WINHTTP_NO_HEADER_INDEX)) {
        failure = GetLastError();
        free(result);
        result = NULL;
        goto done;
    }
    result->status = (int32_t)status;

    for (;;) {
        ULONGLONG now = GetTickCount64();
        if (now >= deadline) {
            failure = ERROR_WINHTTP_TIMEOUT;
            free(result);
            result = NULL;
            goto done;
        }
        DWORD remaining = (DWORD)(deadline - now);
        (void)WinHttpSetTimeouts(request, connect_timeout, connect_timeout,
                                 (int)remaining, (int)remaining);
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available)) {
            failure = GetLastError();
            free(result);
            result = NULL;
            goto done;
        }
        if (available == 0) break;
        char chunk[64 * 1024];
        DWORD wanted = available < sizeof(chunk) ? available : (DWORD)sizeof(chunk);
        DWORD received = 0;
        if (!WinHttpReadData(request, chunk, wanted, &received)) {
            failure = GetLastError();
            free(result);
            result = NULL;
            goto done;
        }
        if (received == 0) break;
        if (!body_append(&response_body, chunk, (size_t)received)) {
            failure = ERROR_NOT_ENOUGH_MEMORY;
            free(result);
            result = NULL;
            goto done;
        }
    }
    if (!response_body.data) response_body.data = http_strdup_n("", 0);
    result->body = response_body.data;
    response_body.data = NULL;
    result->body_len = (int64_t)response_body.len;

done:
    free(response_body.data);
    if (request) WinHttpCloseHandle(request);
    if (connect) WinHttpCloseHandle(connect);
    free(path);
    free(host);
    free(wide_url);
    release_session();
    if (!result) result = error_response(failure == ERROR_SUCCESS ? ERROR_GEN_FAILURE : failure);
    return result;
}

qllm_http_response_t* qllm_http_get(const char* url, int32_t timeout_ms) {
    return perform_request(L"GET", url, NULL, 0, NULL, 0, timeout_ms);
}

qllm_http_response_t* qllm_http_post(const char* url,
                                     const char** headers,
                                     int64_t header_count,
                                     const char* body,
                                     int64_t body_len,
                                     int32_t timeout_ms) {
    return perform_request(L"POST", url, headers, header_count, body, body_len, timeout_ms);
}

qllm_http_response_t* qllm_http_post_json(const char* url,
                                          const char* body,
                                          const char* auth_header,
                                          int32_t timeout_ms) {
    const char* headers[3];
    int64_t count = 0;
    headers[count++] = "Content-Type: application/json";
    headers[count++] = "Accept: application/json";
    if (auth_header && auth_header[0]) headers[count++] = auth_header;
    return qllm_http_post(url, headers, count, body, body ? (int64_t)strlen(body) : 0,
                          timeout_ms);
}

qllm_http_response_t* eshkol_http_request(const char* method,
                                          const char* url,
                                          const char* header_lines,
                                          const char* body,
                                          int32_t timeout_ms) {
    return eshkol_http_request_bytes(method, url, header_lines, body,
                                     body ? (int64_t)strlen(body) : 0,
                                     timeout_ms);
}

qllm_http_response_t* eshkol_http_request_bytes(const char* method,
                                                const char* url,
                                                const char* header_lines,
                                                const char* body,
                                                int64_t body_len,
                                                int32_t timeout_ms) {
    if (!eshkol_http_valid_method(method) || !url) return NULL;
    wchar_t* wide_method = utf8_to_wide(method);
    if (!wide_method) return NULL;
    header_array_t headers;
    if (!parse_header_lines(header_lines, &headers)) {
        free(wide_method);
        return NULL;
    }
    qllm_http_response_t* response = perform_request(
        wide_method, url, (const char**)headers.items, headers.count,
        body, body_len, timeout_ms);
    free_header_array(&headers);
    free(wide_method);
    return response;
}

int32_t qllm_http_response_status(qllm_http_response_t* resp) {
    return resp ? resp->status : 0;
}

const char* qllm_http_response_body(qllm_http_response_t* resp) {
    return resp ? resp->body : NULL;
}

int64_t qllm_http_response_body_len(qllm_http_response_t* resp) {
    return resp ? resp->body_len : 0;
}

const char* qllm_http_response_error(qllm_http_response_t* resp) {
    return resp ? resp->error : NULL;
}

void qllm_http_response_free(qllm_http_response_t* resp) {
    if (!resp) return;
    free(resp->body);
    free(resp->error);
    free(resp);
}

const char* qllm_http_error_string(int32_t code) {
    static _Thread_local char buffer[512];
    if (code <= 0) return "no error";
    char* message = format_windows_error((DWORD)code);
    strncpy(buffer, message, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    free(message);
    return buffer;
}

typedef struct eshkol_winhttp_stream {
    HINTERNET connect;
    HINTERNET request;
    eshkol_sse_parser_t* parser;
    int session_acquired;
    int done;
    char error[512];
} eshkol_winhttp_stream_t;

static void set_stream_error(eshkol_winhttp_stream_t* stream, DWORD code) {
    char* message = format_windows_error(code);
    strncpy(stream->error, message, sizeof(stream->error) - 1);
    stream->error[sizeof(stream->error) - 1] = '\0';
    free(message);
    stream->done = 1;
}

void* eshkol_http_stream_open(const char* method,
                              const char* url,
                              const char* header_lines,
                              const char* body,
                              int32_t timeout_ms) {
    return eshkol_http_stream_open_bytes(method, url, header_lines, body,
                                         body ? (int64_t)strlen(body) : 0,
                                         timeout_ms);
}

void* eshkol_http_stream_open_bytes(const char* method,
                                    const char* url,
                                    const char* header_lines,
                                    const char* body,
                                    int64_t body_len,
                                    int32_t timeout_ms) {
    if (!eshkol_http_valid_method(method) || !url || body_len < 0 ||
        (uint64_t)body_len > UINT32_MAX || (!body && body_len != 0)) return NULL;
    const int timeout = timeout_ms > 0 ? timeout_ms : ESHKOL_HTTP_DEFAULT_TIMEOUT_MS;
    eshkol_winhttp_stream_t* stream =
        (eshkol_winhttp_stream_t*)calloc(1, sizeof(*stream));
    wchar_t* wide_url = NULL;
    wchar_t* wide_method = NULL;
    wchar_t* host = NULL;
    wchar_t* path = NULL;
    header_array_t headers;
    memset(&headers, 0, sizeof(headers));
    HINTERNET session = NULL;
    DWORD failure = ERROR_SUCCESS;
    if (!stream) return NULL;
    stream->parser = eshkol_sse_parser_create();
    if (!stream->parser) goto failure;
    if (!parse_header_lines(header_lines, &headers)) {
        failure = ERROR_INVALID_DATA;
        goto failure;
    }
    session = acquire_session(&failure);
    if (!session) {
        if (failure == ERROR_SUCCESS) failure = ERROR_WINHTTP_INTERNAL_ERROR;
        goto failure;
    }
    stream->session_acquired = 1;
    wide_url = utf8_to_wide(url);
    wide_method = utf8_to_wide(method);
    if (!wide_url || !wide_method) {
        failure = ERROR_NO_UNICODE_TRANSLATION;
        goto failure;
    }
    URL_COMPONENTS parts;
    memset(&parts, 0, sizeof(parts));
    parts.dwStructSize = sizeof(parts);
    parts.dwHostNameLength = (DWORD)-1;
    parts.dwUrlPathLength = (DWORD)-1;
    parts.dwExtraInfoLength = (DWORD)-1;
    parts.dwSchemeLength = (DWORD)-1;
    if (!WinHttpCrackUrl(wide_url, 0, 0, &parts) ||
        (parts.nScheme != INTERNET_SCHEME_HTTP && parts.nScheme != INTERNET_SCHEME_HTTPS) ||
        !parts.lpszHostName || parts.dwHostNameLength == 0) {
        failure = GetLastError();
        if (failure == ERROR_SUCCESS) failure = ERROR_WINHTTP_INVALID_URL;
        goto failure;
    }
    host = wide_dup_n(parts.lpszHostName, parts.dwHostNameLength);
    DWORD path_chars = parts.dwUrlPathLength + parts.dwExtraInfoLength;
    if (path_chars == 0) {
        path = wide_dup_n(L"/", 1);
    } else {
        path = (wchar_t*)http_xmalloc(((size_t)path_chars + 1) * sizeof(wchar_t));
        DWORD offset = 0;
        if (parts.dwUrlPathLength) {
            memcpy(path, parts.lpszUrlPath,
                   (size_t)parts.dwUrlPathLength * sizeof(wchar_t));
            offset += parts.dwUrlPathLength;
        }
        if (parts.dwExtraInfoLength) {
            memcpy(path + offset, parts.lpszExtraInfo,
                   (size_t)parts.dwExtraInfoLength * sizeof(wchar_t));
            offset += parts.dwExtraInfoLength;
        }
        path[offset] = L'\0';
    }
    stream->connect = WinHttpConnect(session, host, parts.nPort, 0);
    if (!stream->connect) {
        failure = GetLastError();
        goto failure;
    }
    DWORD flags = parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
    stream->request = WinHttpOpenRequest(stream->connect, wide_method, path, NULL,
                                         WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!stream->request) {
        failure = GetLastError();
        goto failure;
    }
    int connect_timeout = timeout < ESHKOL_HTTP_CONNECT_TIMEOUT_MS
                              ? timeout : ESHKOL_HTTP_CONNECT_TIMEOUT_MS;
    if (!WinHttpSetTimeouts(stream->request, connect_timeout, connect_timeout,
                            timeout, timeout)) {
        failure = GetLastError();
        goto failure;
    }
    DWORD max_redirects = 10;
    (void)WinHttpSetOption(stream->request, WINHTTP_OPTION_MAX_HTTP_AUTOMATIC_REDIRECTS,
                           &max_redirects, sizeof(max_redirects));
    for (int64_t i = 0; i < headers.count; ++i) {
        wchar_t* header = utf8_to_wide(headers.items[i]);
        if (!header) {
            failure = ERROR_NO_UNICODE_TRANSLATION;
            goto failure;
        }
        BOOL added = WinHttpAddRequestHeaders(stream->request, header, (DWORD)-1L,
                                              WINHTTP_ADDREQ_FLAG_ADD |
                                              WINHTTP_ADDREQ_FLAG_REPLACE);
        free(header);
        if (!added) {
            failure = GetLastError();
            goto failure;
        }
    }
    if (!WinHttpAddRequestHeaders(stream->request, L"Accept: text/event-stream", (DWORD)-1L,
                                  WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE) ||
        !WinHttpAddRequestHeaders(stream->request, L"Cache-Control: no-cache", (DWORD)-1L,
                                  WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
        failure = GetLastError();
        goto failure;
    }
    DWORD request_body_len = (DWORD)body_len;
    if (!WinHttpSendRequest(stream->request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            request_body_len ? (LPVOID)body : WINHTTP_NO_REQUEST_DATA,
                            request_body_len, request_body_len, 0) ||
        !WinHttpReceiveResponse(stream->request, NULL)) {
        failure = GetLastError();
        goto failure;
    }
    DWORD status = 0;
    DWORD status_size = sizeof(status);
    if (!WinHttpQueryHeaders(stream->request,
                             WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
                             WINHTTP_NO_HEADER_INDEX) || status < 200 || status >= 300) {
        failure = ERROR_WINHTTP_INVALID_SERVER_RESPONSE;
        goto failure;
    }
    free_header_array(&headers);
    free(path);
    free(host);
    free(wide_method);
    free(wide_url);
    return stream;

failure:
    free_header_array(&headers);
    free(path);
    free(host);
    free(wide_method);
    free(wide_url);
    if (failure != ERROR_SUCCESS) set_stream_error(stream, failure);
    eshkol_http_stream_close(stream);
    return NULL;
}

eshkol_sse_event_t* eshkol_http_stream_next(void* opaque, int32_t timeout_ms) {
    eshkol_winhttp_stream_t* stream = (eshkol_winhttp_stream_t*)opaque;
    if (!stream) return NULL;
    eshkol_sse_event_t* event = eshkol_sse_parser_next(stream->parser);
    if (event) return event;
    if (eshkol_sse_parser_failed(stream->parser)) {
        set_stream_error(stream, ERROR_INVALID_DATA);
        return NULL;
    }
    ULONGLONG deadline = GetTickCount64() +
                         (ULONGLONG)(timeout_ms > 0 ? timeout_ms : ESHKOL_HTTP_DEFAULT_TIMEOUT_MS);
    while (!stream->done) {
        ULONGLONG now = GetTickCount64();
        if (now >= deadline) return NULL;
        int remaining = (int)(deadline - now);
        (void)WinHttpSetTimeouts(stream->request, remaining, remaining, remaining, remaining);
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(stream->request, &available)) {
            DWORD failure = GetLastError();
            if (failure == ERROR_WINHTTP_TIMEOUT) return NULL;
            set_stream_error(stream, failure);
            break;
        }
        if (available == 0) {
            stream->done = 1;
            break;
        }
        char chunk[64 * 1024];
        while (available > 0) {
            DWORD wanted = available < sizeof(chunk) ? available : (DWORD)sizeof(chunk);
            DWORD received = 0;
            if (!WinHttpReadData(stream->request, chunk, wanted, &received)) {
                set_stream_error(stream, GetLastError());
                break;
            }
            if (received == 0) {
                stream->done = 1;
                break;
            }
            if (!eshkol_sse_parser_feed(stream->parser, chunk, received)) {
                set_stream_error(stream, ERROR_FILE_TOO_LARGE);
                break;
            }
            available -= received;
            event = eshkol_sse_parser_next(stream->parser);
            if (event) return event;
            if (eshkol_sse_parser_failed(stream->parser)) {
                set_stream_error(stream, ERROR_INVALID_DATA);
                break;
            }
        }
    }
    return eshkol_sse_parser_next(stream->parser);
}

int32_t eshkol_http_stream_done(void* opaque) {
    eshkol_winhttp_stream_t* stream = (eshkol_winhttp_stream_t*)opaque;
    if (!stream) return 1;
    return stream->done && !eshkol_sse_parser_has_complete_event(stream->parser);
}

const char* eshkol_http_stream_error(void* opaque) {
    eshkol_winhttp_stream_t* stream = (eshkol_winhttp_stream_t*)opaque;
    if (!stream) return "invalid stream";
    return stream->error[0] ? stream->error : NULL;
}

void eshkol_http_stream_close(void* opaque) {
    eshkol_winhttp_stream_t* stream = (eshkol_winhttp_stream_t*)opaque;
    if (!stream) return;
    if (stream->request) WinHttpCloseHandle(stream->request);
    if (stream->connect) WinHttpCloseHandle(stream->connect);
    if (stream->session_acquired) release_session();
    eshkol_sse_parser_destroy(stream->parser);
    free(stream);
}

#endif /* _WIN32 */
