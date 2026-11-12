#define _POSIX_C_SOURCE 200809L
#include "eshkol/core/resource_lease.h"
#include "eshkol/core/event_loop.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define TAG 12u
#define MASK (UINT32_C(1) << TAG)
typedef struct { int fd, closes, calls, retry_path, terminal_error; pthread_t owner; } Native;
static int dispose(uintptr_t value, void *data, eshkol_resource_cleanup_result_t *result) {
    Native *n = (Native *)value;
    (void)data;
    assert(pthread_equal(n->owner, pthread_self()));
    ++n->calls;
    if (n->fd >= 0) { int fd = n->fd; n->fd = -1; assert(close(fd) == 0); ++n->closes; }
    if (n->retry_path && n->calls == 1) {
        *result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_FAILED,
            ESHKOL_CLEANUP_PATH, EACCES, 1};
        return ESHKOL_IO_SYSTEM;
    }
    if (n->terminal_error) {
        *result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_FAILED,
            ESHKOL_CLEANUP_DESCRIPTOR, EIO, 0};
        return ESHKOL_IO_SYSTEM;
    }
    return 0;
}
static int64_t adopt(int64_t loop, Native *n, int fd) {
    *n = (Native){fd, 0, 0, 0, 0, pthread_self()};
    int64_t id = eshkol_resource_adopt_on_loop(ESHKOL_HANDLE_KIND_SOCKET, TAG,
        (uintptr_t)n, dispose, NULL, loop, 1);
    assert(id > 0);
    return id;
}
static void finish(int64_t ticket) {
    eshkol_resource_cleanup_result_t result;
    assert(ticket > 0);
    eshkol_resource_lease_t rejected;
    assert(eshkol_resource_borrow(ticket, ESHKOL_HANDLE_KIND_FILE, 0, &rejected) == ESHKOL_IO_INVALID);
    assert(eshkol_resource_borrow(ticket, ESHKOL_HANDLE_KIND_SOCKET, 0, &rejected) == ESHKOL_IO_INVALID);
    assert(eshkol_resource_cleanup_result(ticket, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_COMPLETE);
    assert(eshkol_resource_cleanup_release(ticket) == 0);
    assert(eshkol_resource_cleanup_result(ticket, &result) == ESHKOL_IO_INVALID);
}
typedef struct { int64_t id, ticket; int close, check_wrong_thread; } Request;
static void *request(void *value) {
    Request *r = value;
    if (r->check_wrong_thread) {
        eshkol_resource_lease_t lease;
        assert(eshkol_resource_borrow(r->id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == ESHKOL_IO_BUSY);
    }
    struct timespec delay = {0, 20000000};
    nanosleep(&delay, NULL);
    if (r->close == 2) assert(eshkol_resource_close(r->id, ESHKOL_HANDLE_KIND_SOCKET, MASK) == 0);
    else if (r->close) r->ticket = eshkol_resource_close_request(r->id, ESHKOL_HANDLE_KIND_SOCKET, MASK);
    else assert(eshkol_resource_cancel(r->id, ESHKOL_HANDLE_KIND_SOCKET, MASK) == 0);
    return NULL;
}
static void *close_loop_other_thread(void *value) {
    int64_t loop = *(int64_t *)value;
    assert(eshkol_resource_close(loop, ESHKOL_HANDLE_KIND_EVENT_LOOP, 0) == ESHKOL_IO_BUSY);
    return NULL;
}
static void identity_readiness(void) {
    int64_t loop = eshkol_event_loop_open_handle(8); assert(loop > 0);
    pthread_t wrong_thread;
    assert(pthread_create(&wrong_thread, NULL, close_loop_other_thread, &loop) == 0);
    assert(pthread_join(wrong_thread, NULL) == 0 && eshkol_event_loop_from_handle(loop));
    int pipefd[2], other[2]; assert(pipe(pipefd) == 0 && pipe(other) == 0);
    Native n; int64_t id = adopt(loop, &n, pipefd[0]);
    assert(id != loop && eshkol_event_loop_from_handle(id) == NULL);
    eshkol_resource_lease_t lease;
    assert(eshkol_resource_borrow(loop, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == ESHKOL_IO_INVALID);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_EBUSY);
    assert(eshkol_resource_close(loop, ESHKOL_HANDLE_KIND_EVENT_LOOP, 0) == ESHKOL_IO_BUSY);
    assert(eshkol_event_loop_from_handle(loop) != NULL);
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == 0);
    eshkol_event_loop_t *native_loop = eshkol_event_loop_from_handle(loop);
    assert(eshkol_event_loop_add(native_loop, other[0], ESHKOL_EVENT_READ, 777) == 0);
    assert(write(other[1], "x", 1) == 1 && write(pipefd[1], "y", 1) == 1);
    assert(eshkol_resource_wait_on_loop(&lease, pipefd[0], ESHKOL_RESOURCE_READ,
                                     eshkol_resource_deadline(100)) == ESHKOL_RESOURCE_READ);
    assert(eshkol_resource_managed_registration_count(loop) == 0);
    assert(eshkol_event_loop_registration_matches(native_loop, other[0], ESHKOL_EVENT_READ, 777));
    const eshkol_event_t *events; int count;
    assert(eshkol_event_loop_poll(native_loop, 0, &events, &count) == 0 && count == 1);
    assert(events[0].fd == other[0] && events[0].user_data == 777);
    char byte; assert(read(other[0], &byte, 1) == 1 && byte == 'x');
    assert(read(pipefd[0], &byte, 1) == 1 && byte == 'y');
    assert(eshkol_event_loop_remove(native_loop, other[0]) == 0);
    eshkol_resource_release(&lease);
    finish(eshkol_resource_close_request(id, ESHKOL_HANDLE_KIND_SOCKET, MASK));
    assert(n.closes == 1 && n.calls == 1);
    assert(eshkol_event_loop_close_handle(loop) == 0);
    close(pipefd[1]); close(other[0]); close(other[1]);
}
static void cancellation(void) {
    int64_t loop = eshkol_event_loop_open_handle(8); int fds[2]; assert(pipe(fds) == 0);
    Native n; int64_t id = adopt(loop, &n, fds[0]);
    eshkol_resource_lease_t lease;
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == 0);
    Request req = {id, 0, 0, 1}; pthread_t thread; assert(pthread_create(&thread, NULL, request, &req) == 0);
    int64_t begin = eshkol_resource_now_ms();
    assert(eshkol_resource_wait_on_loop(&lease, fds[0], ESHKOL_RESOURCE_READ,
                                     eshkol_resource_deadline(2000)) == ESHKOL_IO_CANCELLED);
    assert(eshkol_resource_now_ms() - begin < 1000);
    assert(pthread_join(thread, NULL) == 0);
    assert(eshkol_resource_check(&lease) == ESHKOL_IO_CANCELLED);
    eshkol_resource_release(&lease);
    assert(eshkol_resource_cancel(id, ESHKOL_HANDLE_KIND_SOCKET, MASK) == 0);
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == ESHKOL_IO_CANCELLED);
    finish(eshkol_resource_close_request(id, ESHKOL_HANDLE_KIND_SOCKET, MASK));
    assert(n.closes == 1); close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
}
static void deferred_close(void) {
    int64_t loop = eshkol_event_loop_open_handle(8); int fds[2]; assert(pipe(fds) == 0);
    Native n; int64_t id = adopt(loop, &n, fds[0]); eshkol_resource_lease_t lease;
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == 0);
    Request req = {id, 0, 1, 0}; pthread_t thread; assert(pthread_create(&thread, NULL, request, &req) == 0);
    assert(eshkol_resource_wait_on_loop(&lease, fds[0], ESHKOL_RESOURCE_READ,
                                     eshkol_resource_deadline(2000)) == ESHKOL_IO_CANCELLED);
    assert(pthread_join(thread, NULL) == 0 && req.ticket > 0 && n.closes == 0);
    eshkol_resource_cleanup_result_t result;
    assert(eshkol_resource_cleanup_result(req.ticket, &result) == 0 && result.state == ESHKOL_CLEANUP_PENDING);
    assert(eshkol_resource_cleanup_ticket(id) == ESHKOL_IO_INVALID);
    eshkol_resource_lease_t rejected;
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &rejected) == ESHKOL_IO_INVALID);
    eshkol_resource_release(&lease);
    assert(eshkol_resource_cleanup_ticket(id) == ESHKOL_IO_INVALID);
    finish(req.ticket); assert(n.closes == 1 && n.calls == 1);
    close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
}
static void queued_and_retry(void) {
    int64_t loop = eshkol_event_loop_open_handle(8); int fds[2]; assert(pipe(fds) == 0);
    Native n; int64_t id = adopt(loop, &n, fds[0]); n.retry_path = 1;
    Request req = {id, 0, 1, 1}; pthread_t thread; assert(pthread_create(&thread, NULL, request, &req) == 0);
    assert(pthread_join(thread, NULL) == 0 && req.ticket > 0 && n.closes == 0);
    eshkol_resource_cleanup_result_t result;
    assert(eshkol_resource_cleanup_result(req.ticket, &result) == 0);
    assert(result.state == ESHKOL_CLEANUP_FAILED && result.stage == ESHKOL_CLEANUP_PATH &&
           result.os_error == EACCES && result.retryable && n.closes == 1);
    assert(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_EBUSY);
    assert(eshkol_resource_cleanup_release(req.ticket) == ESHKOL_IO_BUSY);
    assert(eshkol_resource_cleanup_retry(req.ticket) == 0);
    finish(req.ticket); assert(n.closes == 1 && n.calls == 2);
    close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
    assert(pipe(fds) == 0); loop = eshkol_event_loop_open_handle(8); id = adopt(loop, &n, fds[0]); n.retry_path = 1;
    assert(eshkol_resource_close(id, ESHKOL_HANDLE_KIND_SOCKET, MASK) == 0);
    int64_t ticket = eshkol_resource_cleanup_ticket(id); assert(ticket > 0);
    assert(eshkol_resource_cleanup_retry(ticket) == 0); finish(ticket);
    assert(n.closes == 1); close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
    assert(pipe(fds) == 0); loop = eshkol_event_loop_open_handle(8); id = adopt(loop, &n, fds[0]); n.terminal_error = 1;
    ticket = eshkol_resource_close_request(id, ESHKOL_HANDLE_KIND_SOCKET, MASK);
    assert(eshkol_resource_cleanup_result(ticket, &result) == 0 && result.state == ESHKOL_CLEANUP_FAILED && !result.retryable);
    assert(eshkol_resource_cleanup_retry(ticket) == ESHKOL_IO_INVALID);
    assert(eshkol_resource_cleanup_release(ticket) == 0 && n.closes == 1 && n.calls == 1);
    close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
}
static void nonsticky_admission(void) {
    int fds[2]; assert(pipe(fds) == 0); int64_t loop = eshkol_event_loop_open_handle(8);
    Native n = {fds[0], 0, 0, 0, 0, pthread_self()};
    int64_t id = eshkol_resource_adopt_on_loop(ESHKOL_HANDLE_KIND_SOCKET, TAG,
        (uintptr_t)&n, dispose, NULL, loop, 0);
    assert(id > 0 && eshkol_resource_cancel(id, ESHKOL_HANDLE_KIND_SOCKET, MASK) == 0);
    eshkol_resource_lease_t lease;
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == ESHKOL_IO_CANCELLED);
    assert(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_SOCKET, MASK, &lease) == 0);
    eshkol_resource_release(&lease);
    finish(eshkol_resource_close_request(id, ESHKOL_HANDLE_KIND_SOCKET, MASK));
    close(fds[1]); assert(eshkol_event_loop_close_handle(loop) == 0);
}
static void idle_wakeup_and_capacity(void) {
    int fds[2]; assert(pipe(fds) == 0);
    int64_t loop = eshkol_event_loop_open_handle(1);
    Native n; int64_t id = adopt(loop, &n, fds[0]);
    eshkol_event_loop_t *native_loop = eshkol_event_loop_from_handle(loop);
    assert(eshkol_event_loop_count(native_loop) == 0);
    assert(eshkol_event_loop_add(native_loop, fds[0], ESHKOL_EVENT_READ, 123) == 0);
    assert(eshkol_event_loop_count(native_loop) == 1);
    assert(eshkol_event_loop_remove(native_loop, fds[0]) == 0);
    Request req = {id, 0, 2, 0}; pthread_t thread;
    assert(pthread_create(&thread, NULL, request, &req) == 0);
    const eshkol_event_t *events; int count;
    int64_t begin = eshkol_resource_now_ms();
    assert(eshkol_event_loop_poll(native_loop, 2000, &events, &count) == 0);
    assert(eshkol_resource_now_ms() - begin < 1000);
    assert(pthread_join(thread, NULL) == 0 && count == 0);
    /* Acceptance may wake poll before registry retirement has queued cleanup.
     * The queued callback signals again, so the next loop turn drains it. */
    if (!n.closes) assert(eshkol_event_loop_poll(native_loop, 1000, &events, &count) == 0);
    assert(n.closes == 1);
    assert(eshkol_resource_cleanup_ticket(id) == ESHKOL_IO_INVALID);
    assert(eshkol_event_loop_close_handle(loop) == 0); close(fds[1]);
    assert(pipe(fds) == 0); loop = eshkol_event_loop_open_handle(1); id = adopt(loop, &n, fds[0]);
    req = (Request){id, 0, 2, 0}; assert(pthread_create(&thread, NULL, request, &req) == 0);
    assert(pthread_join(thread, NULL) == 0 && n.closes == 0);
    assert(eshkol_event_loop_close_handle(loop) == 0 && n.closes == 1); close(fds[1]);
}
static void no_maintenance(void *data) { (void)data; }
static void adoption_rollback(void) {
    int fds[2]; assert(pipe(fds) == 0); int64_t loop = eshkol_event_loop_open_handle(1);
    eshkol_event_loop_t *native_loop = eshkol_event_loop_from_handle(loop);
    assert(eshkol_event_loop_set_maintenance(native_loop, no_maintenance, NULL) == 0);
    Native n = {fds[0], 0, 0, 0, 0, pthread_self()};
    assert(eshkol_resource_adopt_on_loop(ESHKOL_HANDLE_KIND_SOCKET, TAG, (uintptr_t)&n,
        dispose, NULL, loop, 1) == ESHKOL_IO_BUSY);
    assert(n.closes == 0 && write(fds[1], "z", 1) == 1);
    char byte; assert(read(fds[0], &byte, 1) == 1 && byte == 'z');
    assert(eshkol_event_loop_close_handle(loop) == 0); close(fds[0]); close(fds[1]);
#ifdef ESHKOL_RESOURCE_TESTING
    extern void eshkol_resource_test_fail_ticket_once(void);
    assert(pipe(fds) == 0); loop = eshkol_event_loop_open_handle(1);
    n = (Native){fds[0], 0, 0, 0, 0, pthread_self()};
    eshkol_resource_test_fail_ticket_once();
    assert(eshkol_resource_adopt_on_loop(ESHKOL_HANDLE_KIND_SOCKET, TAG, (uintptr_t)&n,
        dispose, NULL, loop, 1) == ESHKOL_IO_SYSTEM);
    assert(n.closes == 0 && write(fds[1], "q", 1) == 1);
    assert(read(fds[0], &byte, 1) == 1 && byte == 'q');
    assert(eshkol_event_loop_close_handle(loop) == 0); close(fds[0]); close(fds[1]);
#endif
}
int main(void) {
    identity_readiness(); cancellation(); deferred_close(); queued_and_retry(); nonsticky_admission();
    idle_wakeup_and_capacity(); adoption_rollback();
    puts("PASS: explicit-loop leases, mixed readiness, sticky cancellation, deferred and retryable cleanup");
    return 0;
}
