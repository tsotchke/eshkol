/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "eshkol/agent_http_sse_adapter.h"
#include "eshkol/core/event_loop.h"
#include "eshkol/core/subprocess_stream.h"
#include "eshkol/core/unix_socket_stream.h"

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

extern int64_t eshkol_http_server_create(int32_t port);
extern int32_t eshkol_http_server_port(int64_t handle);
extern int32_t eshkol_http_server_accept(int64_t handle, char* buf,
                                        int32_t size, int32_t timeout_ms);
extern void eshkol_http_server_close(int64_t handle);

enum { SHORT_SEND = 14, READY_MS = 4000 };
static const uint64_t UNIX_PEER_COOKIE = UINT64_C(0x554e495850454552);

/* Standalone fixture policy for the production native capability checks. */
int eshkol_capability_runtime_allows(const char* capability) {
    return capability && (strcmp(capability, "network") == 0 ||
                          strcmp(capability, "subprocess") == 0);
}

void eshkol_capability_enforce(const char* capability, const char* operation) {
    if (eshkol_capability_runtime_allows(capability)) return;
    fprintf(stderr, "Standalone mixed-loop capability denied: %s (%s)\n",
            capability, operation);
    abort();
}

typedef struct {
    int64_t id;
    int prompt;
    int stdout_fd;
} ProcessPeer;

typedef struct {
    int64_t listener;
    int64_t socket;
    int peer_fd;
} UnixPeer;

static int child(const char* fifo) {
    int prompt = open(fifo, O_RDONLY | O_NONBLOCK);
    if (prompt < 0) return 1;
    int connected = 0;
    int result = 1;
    int64_t deadline = eshkol_resource_deadline(20000);
    while (eshkol_resource_remaining(deadline) > 0) {
        char byte;
        ssize_t n = read(prompt, &byte, 1);
        if (n == 1) {
            if (write(STDOUT_FILENO, &byte, 1) != 1) break;
            connected = 1;
            continue;
        }
        if (n == 0 && connected) {
            result = 0;
            break;
        }
        if (n < 0 && errno != EAGAIN && errno != EINTR) break;
        poll(NULL, 0, 5);
    }
    close(prompt);
    return result;
}

static ProcessPeer open_process(int64_t loop, const char* program, const char* fifo) {
    assert(mkfifo(fifo, 0600) == 0);
    char argv[1024];
    int length = snprintf(argv, sizeof(argv), "%s\t--emit\t%s", program, fifo);
    assert(length > 0 && (size_t)length < sizeof(argv));
    int64_t id = eshkol_process_stream_open(loop, argv, ".", NULL);
    assert(id > 0);
    int64_t deadline = eshkol_resource_deadline(READY_MS);
    do {
        int prompt = open(fifo, O_WRONLY | O_NONBLOCK);
        if (prompt >= 0) return (ProcessPeer){id, prompt, -1};
        assert(errno == ENXIO || errno == EINTR);
        poll(NULL, 0, 5);
    } while (eshkol_resource_remaining(deadline) > 0);
    assert(!"subprocess did not open the prompt FIFO");
    abort();
}

static UnixPeer open_unix_peer(int64_t loop_id, const char* path) {
    int64_t listener = eshkol_unix_stream_socket_listen(loop_id, path);
    assert(listener > 0);
    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;
    assert(strlen(path) < sizeof(address.sun_path));
    memcpy(address.sun_path, path, strlen(path) + 1);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(fd >= 0);
    assert(connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0);
    int64_t connected = eshkol_unix_stream_socket_accept(listener, READY_MS);
    assert(connected > 0);
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_id);
    assert(eshkol_event_loop_add(loop, fd, ESHKOL_EVENT_READ, UNIX_PEER_COOKIE) ==
           ESHKOL_EVENT_LOOP_OK);
    return (UnixPeer){listener, connected, fd};
}

/* Every phase prompts two new bytes before polling the original watches. */
static void verify_progress(eshkol_event_loop_t* loop, ProcessPeer* process,
                            const UnixPeer* unix_peer, char value) {
    assert(write(process->prompt, &value, 1) == 1);
    assert(eshkol_unix_stream_socket_write(unix_peer->socket, &value, 1) == 1);
    int saw_process = 0, saw_unix = 0;
    int64_t deadline = eshkol_resource_deadline(READY_MS);
    while ((!saw_process || !saw_unix) && eshkol_resource_remaining(deadline) > 0) {
        const eshkol_event_t* events;
        int count;
        assert(eshkol_event_loop_poll(loop, 25, &events, &count) == ESHKOL_EVENT_LOOP_OK);
        for (int i = 0; i < count; ++i) {
            const eshkol_event_t* event = &events[i];
            char actual;
            if (!(event->events & ESHKOL_EVENT_READ)) continue;
            if (!saw_process && event->user_data == (uint64_t)process->id) {
                assert(eshkol_process_stream_read(process->id, ESHKOL_PROCESS_STREAM_STDOUT,
                                                  &actual, 1) == 1);
                if (process->stdout_fd < 0) process->stdout_fd = event->fd;
                assert(event->fd == process->stdout_fd && actual == value);
                saw_process = 1;
            }
            if (!saw_unix && event->fd == unix_peer->peer_fd &&
                event->user_data == UNIX_PEER_COOKIE) {
                assert(recv(unix_peer->peer_fd, &actual, 1, MSG_DONTWAIT) == 1);
                assert(actual == value);
                saw_unix = 1;
            }
        }
    }
    assert(saw_process && saw_unix);
}

static int connect_http(int64_t server) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    struct timeval receive_timeout = {READY_MS / 1000, 0};
    assert(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                      sizeof(receive_timeout)) == 0);
    int32_t port = eshkol_http_server_port(server);
    assert(port > 0);
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t)port);
    assert(connect(fd, (struct sockaddr*)&address, sizeof(address)) == 0);
    char authority[INET_ADDRSTRLEN];
    assert(inet_ntop(AF_INET, &address.sin_addr, authority, sizeof(authority)));
    char request[128];
    int length = snprintf(request, sizeof(request),
                          "GET /events HTTP/1.1\r\nHost: %s:%d\r\n\r\n", authority, port);
    assert(length > 0 && (size_t)length < sizeof(request));
    assert(send(fd, request, (size_t)length, 0) == length);
    char accepted[512];
    assert(eshkol_http_server_accept(server, accepted, sizeof(accepted), READY_MS) > 0);
    return fd;
}

static int64_t send_prefix(int fd, const uint8_t* data, size_t len, void* context) {
    (void)context;
    assert(len > SHORT_SEND);
    return send(fd, data, SHORT_SEND, 0);
}

static int remove_watch(int64_t loop_id, int fd, void* context) {
    unsigned* attempts = context;
    if (++*attempts == 1) return EIO;
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_id);
    return eshkol_event_loop_remove(loop, fd) == ESHKOL_EVENT_LOOP_OK ? 0 : EIO;
}

static void assert_prefix_then_eof(int fd, const uint8_t* prefix) {
    uint8_t actual[SHORT_SEND];
    size_t received = 0;
    while (received < sizeof(actual)) {
        ssize_t n = recv(fd, actual + received, sizeof(actual) - received, 0);
        if (n < 0 && errno == EINTR) continue;
        assert(n > 0);
        received += (size_t)n;
    }
    assert(memcmp(prefix, actual, sizeof(actual)) == 0);
    uint8_t extra;
    ssize_t n;
    do { n = recv(fd, &extra, 1, 0); } while (n < 0 && errno == EINTR);
    assert(n == 0);
}

static void release_socket(int64_t id) {
    int64_t ticket = eshkol_unix_stream_socket_close_request(id);
    eshkol_resource_cleanup_result_t result;
    assert(ticket > 0 && eshkol_resource_cleanup_result(ticket, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
}

static void verify_sse_teardown(int64_t loop_id, ProcessPeer* process, const UnixPeer* unix_peer) {
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_id);
    verify_progress(loop, process, unix_peer, 'A');

    int64_t server = eshkol_http_server_create(0);
    assert(server > 0);
    int client = connect_http(server);
    int64_t lease = 0;
    eshkol_http_sse_adapter_t* adapter = NULL;
    assert(eshkol_http_sse_adapter_adopt(server, loop_id, &lease, &adapter) == 0);
    int sse_fd = eshkol_http_sse_adapter_fd(adapter);
    assert(sse_fd >= 0);
    eshkol_http_sse_adapter_hooks_t hooks = {.send_override = send_prefix};
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);
    const uint8_t chunk[] = "1f\r\ndata: caf\xc3\xa9\ndata: second line\n\n\r\n";
    size_t offset = 0, written = 0;
    assert(eshkol_http_sse_adapter_write_step(lease, adapter, chunk, sizeof(chunk) - 1,
        &offset, eshkol_resource_deadline(ESHKOL_SSE_WRITE_BUDGET_MS), 42u, &written) ==
        ESHKOL_SSE_WOULD_BLOCK);
    assert(offset == SHORT_SEND && written == SHORT_SEND);
    assert(eshkol_http_sse_adapter_watch_armed(adapter) == 1);

    unsigned removals = 0;
    hooks = (eshkol_http_sse_adapter_hooks_t){.watch_remove_override = remove_watch,
                                           .ctx = &removals};
    eshkol_http_sse_adapter_set_hooks(adapter, &hooks);
    eshkol_resource_lease_t borrow;
    assert(eshkol_resource_borrow(lease, ESHKOL_HANDLE_KIND_SOCKET,
                                  ESHKOL_HTTP_SSE_LEASE_MASK, &borrow) == 0);
    assert(eshkol_http_sse_adapter_cancel(lease) == 0);
    int64_t cleanup = eshkol_http_sse_adapter_request_close(lease);
    eshkol_resource_cleanup_result_t result;
    assert(cleanup > 0 && eshkol_resource_cleanup_result(cleanup, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_PENDING && removals == 0);
    assert(fcntl(sse_fd, F_GETFD) >= 0);
    verify_progress(loop, process, unix_peer, 'B');
    eshkol_resource_release(&borrow);

    assert(eshkol_resource_cleanup_result(cleanup, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_FAILED && result.retryable);
    assert(result.stage == ESHKOL_CLEANUP_STAGE_WATCH && result.os_error == EIO);
    assert(removals == 1 && eshkol_http_sse_adapter_close_count(adapter) == 0);
    assert(fcntl(sse_fd, F_GETFD) >= 0);
    verify_progress(loop, process, unix_peer, 'C');

    assert(eshkol_resource_cleanup_retry(cleanup) == 0);
    assert(eshkol_resource_cleanup_result(cleanup, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_COMPLETE);
    assert(removals == 2 && eshkol_http_sse_adapter_close_count(adapter) == 1);
    assert(fcntl(sse_fd, F_GETFD) == -1 && errno == EBADF);
    assert(eshkol_resource_cleanup_release(cleanup) == 0);
    free(adapter);
    assert_prefix_then_eof(client, chunk);
    verify_progress(loop, process, unix_peer, 'D');
    close(client);
    eshkol_http_server_close(server);
}

int main(int argc, char** argv) {
    if (argc == 3 && strcmp(argv[1], "--emit") == 0) return child(argv[2]);
    assert(argc == 2);
    char fifo[256], path[256];
    int size = snprintf(fifo, sizeof(fifo), "%s/p", argv[1]);
    assert(size > 0 && (size_t)size < sizeof(fifo));
    size = snprintf(path, sizeof(path), "%s/u", argv[1]);
    assert(size > 0 && (size_t)size < sizeof(path));
    int64_t loop_id = eshkol_event_loop_open_handle(16);
    eshkol_event_loop_t* loop = eshkol_event_loop_from_handle(loop_id);
    assert(loop);
    ProcessPeer process = open_process(loop_id, argv[0], fifo);
    UnixPeer unix_peer = open_unix_peer(loop_id, path);

    verify_sse_teardown(loop_id, &process, &unix_peer);

    assert(eshkol_event_loop_remove(loop, unix_peer.peer_fd) == ESHKOL_EVENT_LOOP_OK);
    close(unix_peer.peer_fd);
    release_socket(unix_peer.socket);
    release_socket(unix_peer.listener);
    assert(access(path, F_OK) < 0 && errno == ENOENT);
    close(process.prompt);
    assert(eshkol_process_stream_close(process.id, READY_MS, 10) == 0);
    assert(eshkol_process_cleanup_consume(process.id) == 0);
    assert(eshkol_process_stream_runtime_shutdown() == 0);
    assert(eshkol_event_loop_close_handle(loop_id) == ESHKOL_EVENT_LOOP_OK);
    assert(unlink(fifo) == 0);
    puts("PASS: PROCESS and Unix peer delivered fresh bytes on original watches during and after SSE teardown");
    return 0;
}
