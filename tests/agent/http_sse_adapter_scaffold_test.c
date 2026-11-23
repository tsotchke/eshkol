#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "eshkol/agent_http_sse_adapter.h"
#include "eshkol/core/event_loop.h"
#include "eshkol/core/resource_lease.h"

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

extern int64_t eshkol_http_server_create(int32_t port);
extern int32_t eshkol_http_server_port(int64_t handle);
extern int32_t eshkol_http_server_accept(int64_t handle, char* buf, int32_t buf_size, int32_t timeout_ms);
extern void eshkol_http_server_respond(int64_t handle, int32_t status, const char* content_type, const char* body);
extern void eshkol_http_server_close(int64_t handle);

static int connect_loopback(int32_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    assert(connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
    const char req[] = "GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    assert(send(fd, req, sizeof(req) - 1, 0) == (ssize_t)(sizeof(req) - 1));
    return fd;
}

static size_t read_until_eof(int fd, uint8_t* out, size_t cap) {
    size_t total = 0;
    while (total < cap) {
        ssize_t n = recv(fd, out + total, cap - total, 0);
        if (n > 0) total += (size_t)n;
        else if (n == 0) break;
        else if (errno == EINTR) continue;
        else break;
    }
    return total;
}

static void test_atomic_handoff_and_complete_response(void) {
    int64_t srv = eshkol_http_server_create(0);
    assert(srv >= 1);
    int32_t port = eshkol_http_server_port(srv);
    assert(port > 0);

    int client_a = connect_loopback(port);
    char req_buf[1024];
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);

    int64_t bad_lease = 0;
    eshkol_http_sse_adapter_t* bad_adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, 999999, &bad_lease, &bad_adapter) == ESHKOL_IO_INVALID);
    assert(bad_lease == 0 && bad_adapter == NULL);

    eshkol_http_server_respond(srv, 200, "text/plain", "ordinary-ok");
    uint8_t resp_a[512];
    size_t got_a = read_until_eof(client_a, resp_a, sizeof(resp_a) - 1);
    resp_a[got_a] = '\0';
    assert(strstr((const char*)resp_a, "200 OK") != NULL);
    assert(strstr((const char*)resp_a, "ordinary-ok") != NULL);
    close(client_a);

    int64_t loop = eshkol_event_loop_open_handle(16);
    assert(loop > 0);
    int client_b = connect_loopback(port);
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);

    int64_t lease_id = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, loop, &lease_id, &adapter) == 0);
    assert(lease_id > 0 && adapter != NULL && eshkol_http_sse_adapter_fd(adapter) >= 0);

    eshkol_http_server_close(srv);

    const uint8_t ping[] = "0\r\n\r\n";
    size_t off = 0, written = 0;
    int64_t deadline = eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS);
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, ping, sizeof(ping) - 1, &off, deadline, 101u, &written) == 0);
    assert(off == sizeof(ping) - 1 && written == sizeof(ping) - 1);

    int64_t ticket = eshkol_http_sse_adapter_request_close(lease_id);
    assert(ticket > 0);
    eshkol_resource_cleanup_result_t res;
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0);
    assert(res.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_http_sse_adapter_close_count(adapter) == 1);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);

    uint8_t resp_b[64];
    size_t got_b = read_until_eof(client_b, resp_b, sizeof(resp_b));
    assert(got_b == 5 && memcmp(resp_b, "0\r\n\r\n", 5) == 0);
    close(client_b);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_OK);
}

typedef struct {
    int calls;
    size_t cap_first;
    int fail_second_eagain;
    uint8_t second_offered_byte;
    uint8_t third_offered_byte;
} SendProbe;

static int64_t probe_send(int fd, const uint8_t* data, size_t len, void* ctx) {
    SendProbe* p = (SendProbe*)ctx;
    p->calls += 1;
    if (p->calls == 1 && p->cap_first > 0) {
        size_t n = len < p->cap_first ? len : p->cap_first;
        return (int64_t)send(fd, data, n, 0);
    }
    if (p->calls == 2) {
        p->second_offered_byte = data[0];
        if (p->fail_second_eagain) {
            errno = EAGAIN;
            return -1;
        }
    }
    if (p->calls == 3) {
        p->third_offered_byte = data[0];
    }
    return (int64_t)send(fd, data, len, 0);
}

static void test_short_write_resume_and_presend_deadline(void) {
    static const uint8_t chunk[] =
        "1f\r\ndata: caf\xc3\xa9\ndata: second line\n\n\r\n";
    const size_t chunk_len = sizeof(chunk) - 1;
    assert(chunk_len == 37);
    assert(chunk[14] == 0xa9u);

    int64_t srv = eshkol_http_server_create(0);
    assert(srv >= 1);
    int32_t port = eshkol_http_server_port(srv);
    int64_t loop = eshkol_event_loop_open_handle(16);
    assert(loop > 0);
    int client = connect_loopback(port);
    char req_buf[1024];
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);

    int64_t lease_id = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, loop, &lease_id, &adapter) == 0);
    eshkol_http_server_close(srv);

    SendProbe probe = { .calls = 0, .cap_first = 14, .fail_second_eagain = 1 };
    eshkol_http_sse_adapter_hooks_t hooks = { .send_override = probe_send, .ctx = &probe };
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);

    int64_t deadline = eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS);
    size_t off = 0, written = 0;

    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, chunk, chunk_len, &off, deadline, 202u, &written) == ESHKOL_SSE_WOULD_BLOCK);
    assert(off == 14 && written == 14 && eshkol_http_sse_adapter_watch_armed(adapter) == 1);

    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, chunk, chunk_len, &off, deadline, 202u, &written) == ESHKOL_SSE_WOULD_BLOCK);
    assert(off == 14 && written == 0 && probe.second_offered_byte == 0xa9u);

    int64_t expired = eshkol_resource_now_ms() - 1;
    int calls_before_expired = probe.calls;
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, chunk, chunk_len, &off, expired, 202u, &written) == ESHKOL_IO_TIMEOUT);
    assert(probe.calls == calls_before_expired && off == 14);

    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, chunk, chunk_len, &off, deadline, 202u, &written) == 0);
    assert(off == 37 && written == 23 && probe.third_offered_byte == 0xa9u);
    assert(eshkol_http_sse_adapter_watch_armed(adapter) == 0);

    static const uint8_t term[] = "0\r\n\r\n";
    size_t toff = 0;
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, term, 5, &toff, deadline, 202u, &written) == 0);
    assert(toff == 5);

    int64_t ticket = eshkol_http_sse_adapter_request_close(lease_id);
    eshkol_resource_cleanup_result_t res;
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0 && res.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);

    uint8_t wire[128];
    size_t got = read_until_eof(client, wire, sizeof(wire));
    assert(got == 42);
    assert(memcmp(wire, chunk, 37) == 0);
    assert(memcmp(wire + 37, "0\r\n\r\n", 5) == 0);
    close(client);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_OK);
}

static void test_capacity_boundary_and_busy(void) {
    uint8_t* exact = (uint8_t*)malloc(ESHKOL_SSE_MAX_CHUNK_BYTES + 1);
    assert(exact != NULL);
    memset(exact, 'a', ESHKOL_SSE_MAX_CHUNK_BYTES + 1);
    assert(eshkol_http_sse_adapter_check_chunk_size(exact, ESHKOL_SSE_MAX_CHUNK_BYTES) == 0);
    assert(eshkol_http_sse_adapter_check_chunk_size(exact, ESHKOL_SSE_MAX_CHUNK_BYTES + 1) == ESHKOL_IO_TRUNCATED);
    assert(eshkol_http_sse_adapter_check_chunk_size(exact, 0) == ESHKOL_IO_INVALID);

    int64_t srv = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(srv);
    int64_t loop = eshkol_event_loop_open_handle(16);
    int client = connect_loopback(port);
    char req_buf[1024];
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);
    int64_t lease_id = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, loop, &lease_id, &adapter) == 0);
    eshkol_http_server_close(srv);

    eshkol_resource_lease_t active_borrow;
    assert(eshkol_resource_borrow(lease_id, ESHKOL_HANDLE_KIND_SOCKET, ESHKOL_HTTP_SSE_LEASE_MASK, &active_borrow) == 0);
    size_t off = 7, written = 99;
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, exact, 32, &off, eshkol_resource_deadline(5000), 303u, &written) == ESHKOL_IO_BUSY);
    assert(off == 7 && written == 0);
    eshkol_resource_release(&active_borrow);

    int64_t ticket = eshkol_http_sse_adapter_request_close(lease_id);
    eshkol_resource_cleanup_result_t res;
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0 && res.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);
    free(exact);
    close(client);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_OK);
}

typedef struct {
    int fail_once;
    int remove_calls;
} WatchFailCtx;

static int fail_watch_remove_once(int64_t loop_handle, int fd, void* ctx) {
    WatchFailCtx* wf = (WatchFailCtx*)ctx;
    wf->remove_calls += 1;
    if (wf->fail_once && wf->remove_calls == 1) {
        return EIO;
    }
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_handle);
    assert(loop != NULL);
    int rc = eshkol_event_loop_remove(loop, fd);
    return (rc == ESHKOL_EVENT_LOOP_OK || rc == ESHKOL_EVENT_LOOP_ENOENT) ? 0 : EIO;
}

static void assert_foreign_pipe_delivers(eshkol_event_loop_t* loop,
                                         int wr_fd,
                                         int rd_fd,
                                         uint64_t expected_cookie,
                                         uint8_t byte) {
    assert(write(wr_fd, &byte, 1) == 1);
    const eshkol_event_t* evs = NULL;
    int n = 0;
    int saw_foreign = 0;
    for (int tries = 0; tries < 5 && !saw_foreign; ++tries) {
        assert(eshkol_event_loop_poll(loop, 200, &evs, &n) == ESHKOL_EVENT_LOOP_OK);
        for (int i = 0; i < n; ++i) {
            if (evs[i].fd == rd_fd && evs[i].user_data == expected_cookie &&
                (evs[i].events & ESHKOL_EVENT_READ)) {
                uint8_t got = 0;
                assert(read(rd_fd, &got, 1) == 1 && got == byte);
                saw_foreign = 1;
                break;
            }
        }
    }
    assert(saw_foreign == 1);
}

static void test_mixed_loop_cancel_watch_failure_and_foreign_progress(void) {
    static const uint8_t chunk[] =
        "1f\r\ndata: caf\xc3\xa9\ndata: second line\n\n\r\n";
    const size_t chunk_len = sizeof(chunk) - 1;
    assert(chunk_len == 37);

    int64_t srv = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(srv);
    int64_t loop_handle = eshkol_event_loop_open_handle(16);
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_handle);
    assert(loop != NULL);

    int foreign_pipe[2];
    assert(pipe(foreign_pipe) == 0);
    assert(fcntl(foreign_pipe[0], F_SETFL, O_NONBLOCK) == 0);
    const uint64_t foreign_cookie = UINT64_C(0x50524f435f31);
    assert(eshkol_event_loop_add(loop, foreign_pipe[0], ESHKOL_EVENT_READ, foreign_cookie) == ESHKOL_EVENT_LOOP_OK);

    int client = connect_loopback(port);
    char req_buf[1024];
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);
    int64_t lease_id = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, loop_handle, &lease_id, &adapter) == 0);
    eshkol_http_server_close(srv);

    SendProbe sp = { .calls = 0, .cap_first = 14 };
    WatchFailCtx wf = { .fail_once = 1, .remove_calls = 0 };
    eshkol_http_sse_adapter_hooks_t hooks = {
        .send_override = probe_send,
        .watch_remove_override = fail_watch_remove_once,
        .ctx = &sp
    };
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);
    size_t off = 0, written = 0;
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, chunk, chunk_len, &off,
        eshkol_resource_deadline(5000), 404u, &written) == ESHKOL_SSE_WOULD_BLOCK);
    assert(off == 14 && eshkol_http_sse_adapter_watch_armed(adapter) == 1);

    hooks.send_override = NULL;
    hooks.ctx = &wf;
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);

    eshkol_resource_lease_t borrow;
    assert(eshkol_resource_borrow(lease_id, ESHKOL_HANDLE_KIND_SOCKET, ESHKOL_HTTP_SSE_LEASE_MASK, &borrow) == 0);
    assert(eshkol_http_sse_adapter_cancel(lease_id) == 0);
    int64_t ticket = eshkol_http_sse_adapter_request_close(lease_id);
    assert(ticket > 0);

    eshkol_resource_lease_t rejected;
    assert(eshkol_resource_borrow(lease_id, ESHKOL_HANDLE_KIND_SOCKET, ESHKOL_HTTP_SSE_LEASE_MASK, &rejected) == ESHKOL_IO_INVALID);
    assert(eshkol_http_sse_adapter_fd(adapter) >= 0);
    eshkol_resource_cleanup_result_t res;
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0 && res.state == ESHKOL_CLEANUP_PENDING);

    eshkol_resource_release(&borrow);
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0);
    assert(res.state == ESHKOL_CLEANUP_FAILED);
    assert(res.stage == ESHKOL_CLEANUP_STAGE_WATCH);
    assert(res.retryable == 1);
    assert(eshkol_http_sse_adapter_close_count(adapter) == 0);
    assert(eshkol_event_loop_close_handle(loop_handle) == ESHKOL_EVENT_LOOP_EBUSY);

    assert_foreign_pipe_delivers(loop, foreign_pipe[1], foreign_pipe[0], foreign_cookie, 0x41u);

    assert(eshkol_resource_cleanup_retry(ticket) == 0);
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0);
    assert(res.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_http_sse_adapter_close_count(adapter) == 1);
    assert(eshkol_http_sse_adapter_watch_armed(adapter) == 0);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);

    uint8_t wire[64];
    size_t got = read_until_eof(client, wire, sizeof(wire));
    assert(got == 14 && memcmp(wire, chunk, 14) == 0);
    close(client);

    assert_foreign_pipe_delivers(loop, foreign_pipe[1], foreign_pipe[0], foreign_cookie, 0x5au);

    assert(eshkol_event_loop_remove(loop, foreign_pipe[0]) == ESHKOL_EVENT_LOOP_OK);
    close(foreign_pipe[0]);
    close(foreign_pipe[1]);
    assert(eshkol_event_loop_close_handle(loop_handle) == ESHKOL_EVENT_LOOP_OK);
}

static void test_cancel_and_peer_disconnect(void) {
    int64_t srv = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(srv);
    int64_t loop = eshkol_event_loop_open_handle(16);
    int client = connect_loopback(port);
    char req_buf[1024];
    assert(eshkol_http_server_accept(srv, req_buf, (int32_t)sizeof(req_buf), 2000) > 0);
    int64_t lease_id = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(srv, loop, &lease_id, &adapter) == 0);
    eshkol_http_server_close(srv);

    /* Close peer cleanly and verify subsequent write observes ESHKOL_IO_RESET without SIGPIPE. */
    struct linger lg = { .l_onoff = 1, .l_linger = 0 };
    (void)setsockopt(client, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    close(client);

    uint8_t payload[64];
    memset(payload, 'x', sizeof(payload));
    int saw_reset = 0;
    for (int i = 0; i < 64 && !saw_reset; ++i) {
        size_t off = 0, written = 0;
        int rc = eshkol_http_sse_adapter_write_step(
            lease_id, adapter, payload, sizeof(payload), &off,
            eshkol_resource_deadline(5000), 505u, &written);
        if (rc == ESHKOL_IO_RESET) {
            saw_reset = 1;
            break;
        }
        usleep(2000);
    }
    assert(saw_reset == 1);

    /* Cancellation precedes further sends and returns ESHKOL_IO_CANCELLED. */
    assert(eshkol_http_sse_adapter_cancel(lease_id) == 0);
    size_t off = 0, written = 0;
    assert(eshkol_http_sse_adapter_write_step(
        lease_id, adapter, payload, sizeof(payload), &off,
        eshkol_resource_deadline(5000), 505u, &written) == ESHKOL_IO_CANCELLED);
    assert(off == 0 && written == 0);

    int64_t ticket = eshkol_http_sse_adapter_request_close(lease_id);
    eshkol_resource_cleanup_result_t res;
    assert(eshkol_resource_cleanup_result(ticket, &res) == 0 && res.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_http_sse_adapter_close_count(adapter) == 1);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_OK);
}

static void test_completed_send_reports_watch_failure(void) {
    int64_t server = eshkol_http_server_create(0);
    int64_t loop = eshkol_event_loop_open_handle(16);
    assert(server > 0 && loop > 0);
    int peer = connect_loopback(eshkol_http_server_port(server));
    char request[512];
    assert(eshkol_http_server_accept(server, request, sizeof(request), 1000) > 0);

    int64_t lease = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(server, loop, &lease, &adapter) == 0);
    WatchFailCtx state = {1, 0};
    eshkol_http_sse_adapter_hooks_t hooks = {
        .watch_remove_override = fail_watch_remove_once, .ctx = &state
    };
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);
    assert(eshkol_http_sse_adapter_arm_write_watch(adapter, 42u) == 0);
    const uint8_t sent[] = "abc";
    size_t offset = 0, written = 0;
    assert(eshkol_http_sse_adapter_write_step(
        lease, adapter, sent, sizeof(sent) - 1, &offset,
        eshkol_resource_deadline(5000), 42u, &written) == ESHKOL_IO_SYSTEM);
    assert(offset == 3 && written == 3 && state.remove_calls == 1);
    assert(eshkol_http_sse_adapter_watch_armed(adapter) == 1);

    int64_t ticket = eshkol_http_sse_adapter_request_close(lease);
    eshkol_resource_cleanup_result_t result;
    assert(ticket > 0 && eshkol_resource_cleanup_result(ticket, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_COMPLETE);
    assert(state.remove_calls == 2 && eshkol_http_sse_adapter_close_count(adapter) == 1);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    free(adapter);
    uint8_t received[16];
    assert(read_until_eof(peer, received, sizeof(received)) == 3);
    assert(memcmp(sent, received, 3) == 0);
    close(peer);
    eshkol_http_server_close(server);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_OK);
}

int main(void) {
    test_atomic_handoff_and_complete_response();
    test_short_write_resume_and_presend_deadline();
    test_capacity_boundary_and_busy();
    test_mixed_loop_cancel_watch_failure_and_foreign_progress();
    test_cancel_and_peer_disconnect();
    test_completed_send_reports_watch_failure();
    printf("http_sse_adapter_scaffold_test: 6/6 suites PASSED (backend=%s)\n",
           eshkol_event_loop_backend_name());
    return 0;
}
