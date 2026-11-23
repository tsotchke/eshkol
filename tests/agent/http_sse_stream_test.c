#ifdef NDEBUG
#undef NDEBUG
#endif

#include "agent_http_sse_stream.h"
#include "agent_sse_internal.h"
#include "eshkol/core/event_loop.h"

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

extern int64_t eshkol_http_server_create(int32_t port);
extern int32_t eshkol_http_server_port(int64_t handle);
extern int32_t eshkol_http_server_accept(int64_t handle, char* buf, int32_t buf_size,
                                        int32_t timeout_ms);
extern void eshkol_http_server_respond(int64_t handle, int32_t status,
                                       const char* content_type, const char* body);
extern void eshkol_http_server_close(int64_t handle);

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

static struct fixture open_fixture(void) {
    struct fixture f = {0};
    f.server = eshkol_http_server_create(0);
    f.loop = eshkol_event_loop_open_handle(8);
    assert(f.server > 0 && f.loop > 0);
    f.peer = accept_request(f.server);
    assert(eshkol_http_sse_stream_begin(f.server, f.loop, 123u, &f.stream) == 0);
    return f;
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

static void test_http_and_receiver(void) {
    struct fixture f = open_fixture();
    assert(eshkol_http_sse_stream_send(f.stream, "early", NULL, NULL) == ESHKOL_IO_BUSY);
    assert(progress(&f) == 0);

    /* Taking over the SSE peer must leave the listener available. */
    int ordinary = accept_request(f.server);
    eshkol_http_server_respond(f.server, 200, "text/plain", "ordinary-ok");
    char response[512];
    receive_to_eof(ordinary, response, sizeof(response));
    assert(strstr(response, "\r\n\r\nordinary-ok"));
    close(ordinary);

    assert(eshkol_http_sse_stream_send(f.stream, "x", "bad\r\ninjected", NULL) == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send(f.stream, "\xc0\x80", NULL, NULL) == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_stream_send(f.stream, "caf\xc3\xa9\r\nsecond line", NULL, NULL) == 0);
    int64_t admitted = eshkol_http_sse_stream_deadline(f.stream);
    assert(eshkol_http_sse_stream_send(f.stream, "replacement", NULL, NULL) == ESHKOL_IO_BUSY);
    assert(eshkol_http_sse_stream_finish(f.stream) == ESHKOL_IO_BUSY);
    assert(eshkol_http_sse_stream_deadline(f.stream) == admitted);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_deadline(f.stream) > admitted);

    assert(eshkol_http_sse_stream_finish(f.stream) == 0);
    assert(progress(&f) == ESHKOL_HTTP_SSE_FINISHED);
    assert(eshkol_http_sse_stream_progress(f.stream) == ESHKOL_IO_INVALID);
    release_stream(&f);
    receive_to_eof(f.peer, response, sizeof(response));
    char* body = response_body(response);
    assert(strcmp(body, "1f\r\ndata: caf\xc3\xa9\ndata: second line\n\n\r\n0\r\n\r\n") == 0);

    eshkol_sse_parser_t* parser = eshkol_sse_parser_create();
    assert(parser);
    assert(eshkol_sse_parser_feed(parser, body + 4, 30) == 1);
    assert(eshkol_sse_parser_next(parser) == NULL);
    assert(eshkol_sse_parser_feed(parser, body + 34, 1) == 1);
    eshkol_sse_event_t* event = eshkol_sse_parser_next(parser);
    assert(event && strcmp(eshkol_sse_event_data(event), "caf\xc3\xa9\nsecond line") == 0);
    assert(eshkol_sse_parser_next(parser) == NULL);
    eshkol_sse_event_free(event);
    eshkol_sse_parser_destroy(parser);
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

static void test_idle_timeout(void) {
    struct fixture f = open_fixture();
    assert(progress(&f) == 0);
    int64_t idle = eshkol_http_sse_stream_deadline(f.stream);
    int32_t remaining = eshkol_resource_remaining(idle);
    assert(remaining > 55000 && remaining <= ESHKOL_SSE_IDLE_TIMEOUT_MS);
    assert(progress(&f) == 0);
    assert(eshkol_http_sse_stream_deadline(f.stream) == idle);
    wait_until_expired(idle);
    assert(progress(&f) == ESHKOL_IO_TIMEOUT);
    release_stream(&f);
    char response[512];
    receive_to_eof(f.peer, response, sizeof(response));
    assert(*response_body(response) == 0);
    close_fixture(&f);
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--idle") == 0) {
        test_idle_timeout();
        puts("HTTP SSE idle timeout passed");
        return 0;
    }
    assert(argc == 1);
    test_http_and_receiver();
    test_capacity();
    test_abandonment_and_timeout();
    puts("HTTP SSE stream: 3/3 suites passed");
    return 0;
}
