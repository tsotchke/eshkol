#include "eshkol/agent_http.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE test_thread_t;
#else
#include <pthread.h>
typedef pthread_t test_thread_t;
#endif

extern int64_t eshkol_http_server_create(int32_t port);
extern int32_t eshkol_http_server_port(int64_t handle);
extern int32_t eshkol_http_server_accept(int64_t handle, char* buffer,
                                         int32_t buffer_size, int32_t timeout_ms);
extern void eshkol_http_server_respond(int64_t handle, int32_t status,
                                       const char* content_type, const char* body);
extern void eshkol_http_server_close(int64_t handle);

static int failures = 0;

static void check(int condition, const char* name) {
    if (condition) printf("PASS: %s\n", name);
    else {
        fprintf(stderr, "FAIL: %s\n", name);
        failures++;
    }
}

typedef struct server_context {
    int64_t server;
    const char* expected_a;
    const char* expected_b;
    const char* response_type;
    const char* response_body;
    const char* expected_body;
    size_t expected_body_len;
    int accepted;
} server_context_t;

static const char* find_bytes(const char* haystack, size_t haystack_len,
                              const char* needle, size_t needle_len) {
    if (needle_len == 0) return haystack;
    if (!haystack || !needle || haystack_len < needle_len) return NULL;
    for (size_t i = 0; i <= haystack_len - needle_len; ++i) {
        if (memcmp(haystack + i, needle, needle_len) == 0) return haystack + i;
    }
    return NULL;
}

static void serve_once(server_context_t* context) {
    char request[32768];
    int32_t length = eshkol_http_server_accept(context->server, request,
                                                (int32_t)sizeof(request), 10000);
    context->accepted = length > 0 &&
                        strstr(request, context->expected_a) != NULL &&
                        strstr(request, context->expected_b) != NULL;
    if (context->accepted && context->expected_body) {
        static const char separator[] = "\r\n\r\n";
        const char* body = find_bytes(request, (size_t)length, separator, 4);
        context->accepted = body &&
            (size_t)length - (size_t)(body + 4 - request) == context->expected_body_len &&
            memcmp(body + 4, context->expected_body, context->expected_body_len) == 0;
    }
    if (length > 0)
        eshkol_http_server_respond(context->server, 200,
                                   context->response_type, context->response_body);
}

#ifdef _WIN32
static DWORD WINAPI server_thread_entry(LPVOID opaque) {
    serve_once((server_context_t*)opaque);
    return 0;
}
static int start_server_thread(test_thread_t* thread, server_context_t* context) {
    *thread = CreateThread(NULL, 0, server_thread_entry, context, 0, NULL);
    return *thread != NULL;
}
static void join_server_thread(test_thread_t thread) {
    WaitForSingleObject(thread, 10000);
    CloseHandle(thread);
}
#else
static void* server_thread_entry(void* opaque) {
    serve_once((server_context_t*)opaque);
    return NULL;
}
static int start_server_thread(test_thread_t* thread, server_context_t* context) {
    return pthread_create(thread, NULL, server_thread_entry, context) == 0;
}
static void join_server_thread(test_thread_t thread) { pthread_join(thread, NULL); }
#endif

static void test_general_request(void) {
    static const char binary_body[] =
        {'c','o','n','t','r','a','c','t','\0','b','o','d','y'};
    int64_t server = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(server);
    check(server > 0 && port > 0, "loopback HTTP server starts");
    if (server <= 0 || port <= 0) return;
    server_context_t context = {
        server, "bREW /contract", "X.Eshkol-Contract: complete",
        "text/plain", "request-ok", binary_body, sizeof(binary_body), 0
    };
    test_thread_t thread;
    check(start_server_thread(&thread, &context), "request server thread starts");
    char url[256];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d/contract", port);
    qllm_http_response_t* response = eshkol_http_request_bytes(
        "bREW", url,
        "Content-Type: application/octet-stream\nX.Eshkol-Contract: complete\n",
        binary_body, sizeof(binary_body), 10000);
    check(response && qllm_http_response_status(response) == 200 &&
          strcmp(qllm_http_response_body(response), "request-ok") == 0,
          "RFC token method, headers, and binary body round-trip");
    qllm_http_response_free(response);
    join_server_thread(thread);
    check(context.accepted, "server observes method, custom header, and exact binary body");
    eshkol_http_server_close(server);

    check(eshkol_http_request("BAD METHOD", url, "X-Test: value\n", "", 1000) == NULL,
          "invalid method token is rejected before transport");
    check(eshkol_http_request("GET", url, "Bad Name: value\n", "", 1000) == NULL,
          "invalid header-name token is rejected before transport");
}

static void test_sse_stream(void) {
    static const char events[] =
        ": comment\r\n"
        "id: event-7\r\n"
        "event: update\r\n"
        "data: first line\r\n"
        "data: second line\r\n"
        "retry: 1500\r\n\r\n"
        "data: final\n\n";
    int64_t server = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(server);
    check(server > 0 && port > 0, "SSE loopback server starts");
    if (server <= 0 || port <= 0) return;
    server_context_t context = {
        server, "GET /events", "X-Eshkol-Stream: native",
        "text/event-stream", events, NULL, 0, 0
    };
    test_thread_t thread;
    check(start_server_thread(&thread, &context), "SSE server thread starts");
    char url[256];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d/events", port);
    void* stream = eshkol_http_stream_open(
        "GET", url, "X-Eshkol-Stream: native\n", NULL, 10000);
    check(stream != NULL, "native SSE stream opens");
    if (stream) {
        eshkol_sse_event_t* first = eshkol_http_stream_next(stream, 10000);
        check(first && strcmp(eshkol_sse_event_type(first), "update") == 0 &&
              strcmp(eshkol_sse_event_data(first), "first line\nsecond line") == 0 &&
              strcmp(eshkol_sse_event_id(first), "event-7") == 0 &&
              eshkol_sse_event_retry_ms(first) == 1500,
              "SSE parser preserves type, multiline data, id, and retry");
        eshkol_sse_event_free(first);
        eshkol_sse_event_t* second = eshkol_http_stream_next(stream, 10000);
        check(second && strcmp(eshkol_sse_event_type(second), "message") == 0 &&
              strcmp(eshkol_sse_event_data(second), "final") == 0,
              "SSE parser returns subsequent default message event");
        eshkol_sse_event_free(second);
        check(eshkol_http_stream_next(stream, 10000) == NULL &&
              eshkol_http_stream_done(stream) == 1 &&
              eshkol_http_stream_error(stream) == NULL,
              "SSE stream reaches clean end-of-stream");
        eshkol_http_stream_close(stream);
    }
    join_server_thread(thread);
    check(context.accepted, "SSE request carries complete headers");
    eshkol_http_server_close(server);
}

int main(void) {
    check(qllm_http_init() == 1 && qllm_http_has_ssl() == 1,
          "native HTTP backend initializes with TLS support");
    test_general_request();
    test_sse_stream();
    qllm_http_shutdown();
    printf("Agent HTTP contract checks: %s (%d failure%s)\n",
           failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
