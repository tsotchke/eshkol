#ifdef NDEBUG
#undef NDEBUG
#endif

#include "agent_http_sse_stream.h"
#include "agent_sse_internal.h"
#include "eshkol/agent_capabilities.h"
#include "eshkol/core/event_loop.h"

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

extern void eshkol_sse_test_set_network_allowed(int allowed);

static char oversized[ESHKOL_SSE_MAX_CHUNK_BYTES + 1];

struct fixture {
    int64_t server;
    int64_t loop;
    int peer;
    eshkol_http_sse_stream_t* stream;
};

static int accept_request(int64_t server) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    struct timeval receive_timeout = {2, 0};
    assert(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                      sizeof(receive_timeout)) == 0);
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t)eshkol_http_server_port(server));
    assert(connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0);

    const char request[] = "GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n";
    assert(send(fd, request, sizeof(request) - 1, 0) == (ssize_t)(sizeof(request) - 1));
    char accepted[512];
    assert(eshkol_http_server_accept(server, accepted, sizeof(accepted), 1000) ==
           (int32_t)(sizeof(request) - 1));
    assert(strcmp(accepted, request) == 0);
    return fd;
}

static struct fixture open_accepted_fixture(void) {
    struct fixture f = {0};
    f.server = eshkol_http_server_create(0);
    f.loop = eshkol_event_loop_open_handle(8);
    assert(f.server > 0 && f.loop > 0);
    f.peer = accept_request(f.server);
    return f;
}

static struct fixture open_fixture(void) {
    struct fixture f = open_accepted_fixture();
    assert(eshkol_http_sse_stream_begin(f.server, f.loop, 123u, &f.stream) == 0);
    return f;
}

static void receive_exact(int fd, char* result, size_t size) {
    while (size) {
        ssize_t count = recv(fd, result, size, 0);
        if (count < 0 && errno == EINTR) continue;
        assert(count > 0);
        result += count;
        size -= (size_t)count;
    }
}

static void receive_headers(int fd, char* result, size_t cap) {
    for (size_t size = 1; size < cap; ++size) {
        receive_exact(fd, result + size - 1, 1);
        if (size >= 4 && memcmp(result + size - 4, "\r\n\r\n", 4) == 0) {
            result[size] = 0;
            return;
        }
    }
    assert(!"expected the HTTP header terminator");
}

static void assert_peer_quiet(int fd) {
    struct pollfd peer = { .fd = fd, .events = POLLIN };
    int ready;
    do ready = poll(&peer, 1, 0); while (ready < 0 && errno == EINTR);
    assert(ready == 0);
}

static size_t receive_to_eof(int fd, char* result, size_t cap) {
    size_t size = 0;
    while (size + 1 < cap) {
        ssize_t count = recv(fd, result + size, cap - size - 1, 0);
        if (count < 0 && errno == EINTR) continue;
        assert(count >= 0);
        if (!count) {
            result[size] = 0;
            return size;
        }
        size += (size_t)count;
    }
    assert(!"expected EOF before the receive buffer filled");
    return 0;
}

static char* response_body(char* response) {
    assert(strncmp(response, "HTTP/1.1 200 OK\r\n", 17) == 0);
    char* body = strstr(response, "\r\n\r\n");
    assert(body);
    *body = 0;
    assert(strstr(response, "\r\nContent-Type: text/event-stream\r\n"));
    assert(strstr(response, "\r\nTransfer-Encoding: chunked\r\n"));
    return body + 4;
}

static void release_stream(struct fixture* f) {
    int64_t ticket;
    assert(eshkol_http_sse_stream_end(f->stream, &ticket) == 0);
    assert(ticket > 0);
    f->stream = NULL;
    eshkol_resource_cleanup_result_t result;
    assert(eshkol_resource_cleanup_result(ticket, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
}

static void close_fixture(struct fixture* f) {
    close(f->peer);
    eshkol_http_server_close(f->server);
    assert(eshkol_event_loop_close_handle(f->loop) == ESHKOL_EVENT_LOOP_OK);
}

static int progress(struct fixture* f) {
    int status = eshkol_http_sse_stream_progress(f->stream);
    for (int tries = 0; status == ESHKOL_SSE_WOULD_BLOCK && tries < 10; ++tries) {
        eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(f->loop);
        const eshkol_event_t* events;
        int count;
        assert(eshkol_event_loop_poll(loop, 100, &events, &count) == ESHKOL_EVENT_LOOP_OK);
        for (int i = 0; i < count; ++i) {
            if (events[i].user_data == 123u) status = eshkol_http_sse_stream_progress(f->stream);
        }
    }
    return status;
}

struct off_thread_send {
    eshkol_http_sse_stream_t* stream;
    int status;
};

static void* send_off_thread(void* context) {
    struct off_thread_send* send = context;
    send->status = eshkol_http_sse_stream_send_bytes(send->stream, oversized, sizeof(oversized),
                                                     NULL, 0, NULL, 0);
    return NULL;
}

static void test_http_and_receiver(void) {
    struct fixture f = open_fixture();
    assert(eshkol_http_sse_stream_send(f.stream, "early", NULL, NULL) == ESHKOL_IO_BUSY);
    struct off_thread_send send = { .stream = f.stream };
    pthread_t other;
    assert(pthread_create(&other, NULL, send_off_thread, &send) == 0);
    assert(pthread_join(other, NULL) == 0);
    assert(send.status == ESHKOL_IO_INVALID);
    assert(progress(&f) == 0);

    /* Taking over the SSE peer must leave the listener available. */
    int ordinary = accept_request(f.server);
    eshkol_http_server_respond(f.server, 200, "text/plain", "ordinary-ok");
    char response[512];
    receive_to_eof(ordinary, response, sizeof(response));
    assert(strstr(response, "\r\n\r\nordinary-ok"));
    close(ordinary);

    receive_headers(f.peer, response, sizeof(response));
    assert(*response_body(response) == 0);
    int64_t idle = eshkol_http_sse_stream_deadline(f.stream);
    assert(eshkol_http_sse_stream_send(f.stream, "x", "bad\r\ninjected", NULL) == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send(f.stream, "x", NULL, "bad\nid") == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send(f.stream, "x", "\xc0\x80", NULL) == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send(f.stream, "\xc0\x80", NULL, NULL) == ESHKOL_IO_INVALID);
    const char embedded_nul[] = { 'a', 0, 'b' };
    assert(eshkol_http_sse_stream_send_bytes(f.stream, "x", 1, NULL, 1, NULL, 0)
           == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send_bytes(f.stream, "x", 1, NULL, 0,
                                            embedded_nul, sizeof(embedded_nul))
           == ESHKOL_IO_INVALID);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_deadline(f.stream) == idle);
    assert_peer_quiet(f.peer);

    assert(eshkol_http_sse_stream_send(f.stream, "caf\xc3\xa9\r\nsecond line", "", NULL) == 0);
    int64_t admitted = eshkol_http_sse_stream_deadline(f.stream);
    assert(eshkol_http_sse_stream_send(f.stream, "replacement", NULL, NULL) == ESHKOL_IO_BUSY);
    assert(eshkol_http_sse_stream_send_bytes(f.stream, oversized, sizeof(oversized),
                                            NULL, 0, NULL, 0) == ESHKOL_IO_BUSY);
    assert(eshkol_http_sse_stream_finish(f.stream) == ESHKOL_IO_BUSY);
    assert(eshkol_http_sse_stream_deadline(f.stream) == admitted);
    assert_peer_quiet(f.peer);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_deadline(f.stream) > admitted);

    /* Bound the live TCP reads independently of kernel packet coalescing.
     * Decode the chunk-size line and trailer separately from the SSE payload. */
    const char canonical[] = "1f\r\ndata: caf\xc3\xa9\ndata: second line\n\n\r\n";
    char wire[sizeof(canonical) - 1];
    enum { payload_start = 4, utf8_split = 14, blank_line = 34, payload_end = 35 };
    eshkol_sse_parser_t* parser = eshkol_sse_parser_create();
    assert(parser);
    receive_exact(f.peer, wire, utf8_split);
    assert(memcmp(wire, "1f\r\n", payload_start) == 0);
    assert((unsigned char)wire[utf8_split - 1] == 0xc3);
    assert(eshkol_sse_parser_feed(parser, wire + payload_start, utf8_split - payload_start) == 1);
    assert(eshkol_sse_parser_next(parser) == NULL);

    receive_exact(f.peer, wire + utf8_split, blank_line - utf8_split);
    assert((unsigned char)wire[utf8_split] == 0xa9);
    assert(eshkol_sse_parser_feed(parser, wire + utf8_split, blank_line - utf8_split) == 1);
    assert(eshkol_sse_parser_next(parser) == NULL);

    receive_exact(f.peer, wire + blank_line, payload_end - blank_line);
    assert(eshkol_sse_parser_feed(parser, wire + blank_line, payload_end - blank_line) == 1);
    eshkol_sse_event_t* event = eshkol_sse_parser_next(parser);
    assert(event && strcmp(eshkol_sse_event_data(event), "caf\xc3\xa9\nsecond line") == 0);
    assert(eshkol_sse_parser_next(parser) == NULL);
    eshkol_sse_event_free(event);
    eshkol_sse_parser_destroy(parser);

    receive_exact(f.peer, wire + payload_end, sizeof(wire) - payload_end);
    assert(sizeof(wire) == 37 && memcmp(wire, canonical, sizeof(wire)) == 0);
    assert(eshkol_http_sse_stream_finish(f.stream) == 0);
    assert_peer_quiet(f.peer);
    assert(progress(&f) == ESHKOL_HTTP_SSE_FINISHED);
    assert(eshkol_http_sse_stream_progress(f.stream) == ESHKOL_IO_INVALID);
    release_stream(&f);
    assert(receive_to_eof(f.peer, response, sizeof(response)) == 5);
    assert(memcmp(response, "0\r\n\r\n", 5) == 0);
    close_fixture(&f);
}

static void test_capacity(void) {
    struct fixture f = open_fixture();
    assert(progress(&f) == 0);
    char data[16370];
    memset(data, 'x', sizeof(data));
    data[16369] = 0;
    assert(eshkol_http_sse_stream_send(f.stream, data, NULL, NULL) == ESHKOL_IO_TRUNCATED);
    data[16368] = 0;
    assert(eshkol_http_sse_stream_send(f.stream, data, NULL, NULL) == 0);
    assert(progress(&f) == 0);
    release_stream(&f);
    char response[17000];
    receive_to_eof(f.peer, response, sizeof(response));
    char* body = response_body(response);
    assert(strlen(body) == ESHKOL_SSE_MAX_CHUNK_BYTES);
    assert(strncmp(body, "3ff8\r\ndata: ", 12) == 0);
    assert(memcmp(body + 12, data, 16368) == 0);
    assert(strcmp(body + 16380, "\n\n\r\n") == 0);
    close_fixture(&f);
}

static void wait_until_expired(int64_t deadline) {
    int32_t remaining;
    while ((remaining = eshkol_resource_remaining(deadline)) > 0) {
        assert(poll(NULL, 0, remaining) == 0 || errno == EINTR);
    }
    assert(remaining == 0);
}

static void test_abandonment_and_timeout(void) {
    for (int wait_for_expiry = 0; wait_for_expiry < 2; ++wait_for_expiry) {
        struct fixture f = open_fixture();
        assert(progress(&f) == 0);
        assert(eshkol_http_sse_stream_send(f.stream, "must not appear", NULL, NULL) == 0);
        if (wait_for_expiry) {
            wait_until_expired(eshkol_http_sse_stream_deadline(f.stream));
            assert(progress(&f) == ESHKOL_IO_TIMEOUT);
        }
        release_stream(&f);
        char response[512];
        receive_to_eof(f.peer, response, sizeof(response));
        assert(*response_body(response) == 0);
        close_fixture(&f);
    }
}

static void test_network_capability(void) {
    struct fixture f = open_accepted_fixture();
    eshkol_sse_test_set_network_allowed(0);
    errno = 0;
    assert(eshkol_http_sse_stream_begin(f.server, f.loop, 123u, &f.stream) == ESHKOL_IO_SYSTEM);
    assert(errno == EACCES && f.stream == NULL);
    assert_peer_quiet(f.peer);

    /* An acquisition denied by policy must leave the HTTP client slot owned
     * by the listener, so the existing complete-response route can use it. */
    eshkol_sse_test_set_network_allowed(1);
    eshkol_http_server_respond(f.server, 200, "text/plain", "ordinary-after-denial");
    char response[512];
    receive_to_eof(f.peer, response, sizeof(response));
    assert(strstr(response, "\r\n\r\nordinary-after-denial"));
    close(f.peer);

    f.peer = accept_request(f.server);
    assert(eshkol_http_sse_stream_begin(f.server, f.loop, 123u, &f.stream) == 0);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_send(f.stream, "queued before revocation", NULL, NULL) == 0);
    eshkol_sse_test_set_network_allowed(0);
    errno = 0;
    assert(eshkol_http_sse_stream_progress(f.stream) == ESHKOL_IO_SYSTEM);
    assert(errno == EACCES);
    receive_to_eof(f.peer, response, sizeof(response));
    assert(*response_body(response) == 0);
    release_stream(&f);
    close_fixture(&f);
    eshkol_sse_test_set_network_allowed(1);
}

static void test_idle_timeout(void) {
    struct fixture f = open_fixture();
    assert(progress(&f) == 0);
    int64_t idle = eshkol_http_sse_stream_deadline(f.stream);
    int32_t remaining = eshkol_resource_remaining(idle);
    assert(remaining > 55000 && remaining <= ESHKOL_SSE_IDLE_TIMEOUT_MS);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_deadline(f.stream) == idle);
    wait_until_expired(idle);
    assert(eshkol_http_sse_stream_send_bytes(f.stream, oversized, sizeof(oversized),
                                            NULL, 0, NULL, 0) == ESHKOL_IO_TIMEOUT);
    char response[512];
    receive_to_eof(f.peer, response, sizeof(response));
    assert(*response_body(response) == 0);
    release_stream(&f);
    close_fixture(&f);
}

int main(int argc, char** argv) {
    memset(oversized, 'x', sizeof(oversized));
    if (argc == 2 && strcmp(argv[1], "--idle") == 0) {
        test_idle_timeout();
        puts("HTTP SSE idle timeout passed");
        return 0;
    }
    assert(argc == 1);
    test_http_and_receiver();
    test_capacity();
    test_abandonment_and_timeout();
    test_network_capability();
    puts("HTTP SSE stream: 4/4 suites passed");
    return 0;
}
