#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include "eshkol/core/net.h"
#include "eshkol/core/event_loop.h"

/* Instrument the event-loop wait after the operation holds its lease. */
static pthread_mutex_t poll_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t poll_cv;
static int poll_watch_armed;
static int poll_entered;
static int watched_poll(eshkol_event_loop_t *, int, const eshkol_event_t **, int *);
#define eshkol_event_loop_poll watched_poll
#include "../../lib/core/resource_lease.c"
#undef eshkol_event_loop_poll

static int watched_poll(eshkol_event_loop_t *loop, int timeout_ms,
                        const eshkol_event_t **events, int *count) {
    int rc = pthread_mutex_lock(&poll_mutex);
    if (rc) return ESHKOL_EVENT_LOOP_EINVAL;
    if (poll_watch_armed && !poll_entered) {
        poll_entered = 1;
        rc = pthread_cond_signal(&poll_cv);
    }
    int unlock_rc = pthread_mutex_unlock(&poll_mutex);
    if (rc || unlock_rc) return ESHKOL_EVENT_LOOP_EINVAL;
    return eshkol_event_loop_poll(loop, timeout_ms, events, count);
}

/* Far beyond the completion bound: a worker that returns only at its own
 * deadline fails the prompt-cancellation check. */
#define OP_TIMEOUT_MS 20000
#define CANCEL_BOUND_MS 2000
#define FD_HEADROOM 24

enum operation { OP_ACCEPT, OP_TCP_READ, OP_UDP_RECV };
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t done_cv;
    int done;
    enum operation op;
    int64_t handle;
    int64_t result;
} worker_case;

static _Atomic int failures;
static void check(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        atomic_fetch_add(&failures, 1);
    }
}
static void check_pthread(int rc, const char *what) {
    if (rc) {
        fprintf(stderr, "FAIL: %s: %s\n", what, strerror(rc));
        atomic_fetch_add(&failures, 1);
    }
}
static int deadline_after(struct timespec *t, long ms) {
    if (clock_gettime(CLOCK_MONOTONIC, t) != 0) return -1;
    t->tv_sec += ms / 1000;
    t->tv_nsec += (ms % 1000) * 1000000L;
    if (t->tv_nsec >= 1000000000L) { t->tv_sec++; t->tv_nsec -= 1000000000L; }
    return 0;
}
static int cond_init_monotonic(pthread_cond_t *cv) {
#ifdef __APPLE__
    /* Darwin provides relative timed waits rather than condattr_setclock. */
    return pthread_cond_init(cv, NULL);
#else
    pthread_condattr_t attr;
    int rc = pthread_condattr_init(&attr);
    if (rc) return rc;
    rc = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    if (!rc) rc = pthread_cond_init(cv, &attr);
    int destroy_rc = pthread_condattr_destroy(&attr);
    return rc ? rc : destroy_rc;
#endif
}
static int cond_wait_until(pthread_cond_t *cv, pthread_mutex_t *mutex,
                           const struct timespec *deadline) {
#ifdef __APPLE__
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return errno ? errno : EINVAL;
    struct timespec remaining = {deadline->tv_sec - now.tv_sec,
                                 deadline->tv_nsec - now.tv_nsec};
    if (remaining.tv_nsec < 0) { remaining.tv_sec--; remaining.tv_nsec += 1000000000L; }
    if (remaining.tv_sec < 0 || (remaining.tv_sec == 0 && remaining.tv_nsec == 0))
        return ETIMEDOUT;
    return pthread_cond_timedwait_relative_np(cv, mutex, &remaining);
#else
    return pthread_cond_timedwait(cv, mutex, deadline);
#endif
}
static int wait_for_poll(long timeout_ms) {
    struct timespec deadline;
    if (deadline_after(&deadline, timeout_ms) != 0) return 0;
    int rc = pthread_mutex_lock(&poll_mutex);
    if (rc) { check_pthread(rc, "lock poll watch"); return 0; }
    while (!poll_entered && rc == 0) rc = cond_wait_until(&poll_cv, &poll_mutex, &deadline);
    int entered = poll_entered;
    poll_watch_armed = 0;
    int unlock_rc = pthread_mutex_unlock(&poll_mutex);
    check_pthread(unlock_rc, "unlock poll watch");
    if (rc != 0 && rc != ETIMEDOUT) check_pthread(rc, "wait for poll entry");
    return entered;
}
static void arm_poll_watch(void) {
    int rc = pthread_mutex_lock(&poll_mutex);
    if (rc) { check_pthread(rc, "lock poll watch"); return; }
    poll_entered = 0;
    poll_watch_armed = 1;
    rc = pthread_mutex_unlock(&poll_mutex);
    check_pthread(rc, "unlock poll watch");
}
static void *blocked_operation(void *opaque) {
    worker_case *c = opaque;
    char buffer[32], peer[128];
    if (c->op == OP_ACCEPT) c->result = eshkol_net_accept(c->handle, OP_TIMEOUT_MS);
    else if (c->op == OP_TCP_READ) c->result = eshkol_net_read(c->handle, buffer, sizeof(buffer), OP_TIMEOUT_MS);
    else c->result = eshkol_net_recvfrom(c->handle, buffer, sizeof(buffer), peer, sizeof(peer), OP_TIMEOUT_MS);
    int rc = pthread_mutex_lock(&c->mutex);
    if (rc) { check_pthread(rc, "worker lock"); return NULL; }
    c->done = 1;
    rc = pthread_cond_signal(&c->done_cv);
    if (rc) check_pthread(rc, "worker signal");
    rc = pthread_mutex_unlock(&c->mutex);
    if (rc) check_pthread(rc, "worker unlock");
    return NULL;
}

/* The hook proves the operation holds its lease and is waiting. Close
 * invalidates the handle at once, wakes the waiter (which reports
 * cancellation), and closes the descriptor when the waiter releases. */
static void close_race(enum operation op, int64_t handle, const char *label) {
    worker_case c;
    memset(&c, 0, sizeof(c));
    c.op = op; c.handle = handle;
    int mutex_ready = pthread_mutex_init(&c.mutex, NULL) == 0;
    check(mutex_ready, "initialize worker mutex");
    if (!mutex_ready) { eshkol_net_close(handle); return; }
    int cond_ready = cond_init_monotonic(&c.done_cv) == 0;
    check(cond_ready, "initialize monotonic worker condition");
    if (!cond_ready) {
        pthread_mutex_destroy(&c.mutex);
        eshkol_net_close(handle);
        return;
    }

    pthread_t thread;
    /* Arm before creation. The worker cannot enter the hook until acquire()
     * has taken a ref; the condition signal gives an exact in-flight point. */
    arm_poll_watch();
    int rc = pthread_create(&thread, NULL, blocked_operation, &c);
    check_pthread(rc, "create operation worker");
    if (rc) {
        pthread_mutex_lock(&poll_mutex);
        poll_watch_armed = 0;
        pthread_mutex_unlock(&poll_mutex);
        eshkol_net_close(handle);
        goto cleanup;
    }
    int in_flight = wait_for_poll(OP_TIMEOUT_MS);
    check(in_flight, "operation holds its lease and reached the event-loop wait");
    check(eshkol_net_close(handle) == 0, "close succeeds during an in-flight operation");
    check(eshkol_net_close(handle) == -4, "closed handle is immediately stale");

    struct timespec deadline;
    int deadline_ok = deadline_after(&deadline, CANCEL_BOUND_MS) == 0;
    check(deadline_ok, "create monotonic worker deadline");
    int lock_rc = pthread_mutex_lock(&c.mutex);
    check_pthread(lock_rc, "lock worker completion");
    int rc_wait = deadline_ok && !lock_rc ? 0 : EINVAL;
    while (!lock_rc && !c.done && rc_wait == 0)
        rc_wait = cond_wait_until(&c.done_cv, &c.mutex, &deadline);
    int done = !lock_rc && c.done;
    if (!lock_rc) check_pthread(pthread_mutex_unlock(&c.mutex), "unlock worker completion");
    if (!done) {
        fprintf(stderr, "FAIL: %s worker was not woken by close\n", label);
        atomic_fetch_add(&failures, 1);
        /* Do not return with a live worker even on a failure path. */
        check_pthread(pthread_cancel(thread), "cancel timed-out worker");
    }
    check_pthread(pthread_join(thread, NULL), "join operation worker");
    if (done) check(c.result == ESHKOL_IO_CANCELLED, "in-flight operation reports cancellation");
cleanup:
    pthread_cond_destroy(&c.done_cv);
    pthread_mutex_destroy(&c.mutex);
}

static int64_t open_listener(void) { return eshkol_net_open("127.0.0.1", 0, 1, 1000); }
static int64_t open_udp(void) { return eshkol_net_open("127.0.0.1", 0, 3, 1000); }

static void close_blocked_accept(void) {
    int64_t listener = open_listener();
    check(listener > 0, "open listener for blocked accept");
    if (listener > 0) close_race(OP_ACCEPT, listener, "accept");
}
static void close_blocked_read(void) {
    int64_t listener = open_listener();
    int port = listener > 0 ? eshkol_net_port(listener, 0) : -1;
    int64_t client = port > 0 ? eshkol_net_open("127.0.0.1", port, 2, 1000) : -1;
    int64_t server = listener > 0 ? eshkol_net_accept(listener, 1000) : -1;
    check(listener > 0 && client > 0 && server > 0, "establish quiet TCP pair for blocked read");
    if (listener > 0) eshkol_net_close(listener);
    if (server > 0) close_race(OP_TCP_READ, server, "TCP read");
    if (client > 0) check(eshkol_net_close(client) == 0, "close TCP peer");
}
static void close_blocked_udp_recv(void) {
    int64_t udp = open_udp();
    check(udp > 0, "open UDP socket for blocked receive");
    if (udp > 0) close_race(OP_UDP_RECV, udp, "UDP receive");
}

static void stale_id_and_slot_control(void) {
    int64_t first = open_udp();
    check(first > 0 && eshkol_net_close(first) == 0, "close first UDP handle");
    int64_t second = open_udp();
    check(second > 0 && second != first, "opaque handle IDs are not reused");
    check(eshkol_net_close(first) == -4, "stale ID cannot close later socket");
    check(second > 0 && eshkol_net_port(second, 0) >= 0, "later socket remains live after stale close");
    if (second > 0) eshkol_net_close(second);

    /* No slot table: the descriptor limit bounds live sockets. Lower it,
     * exhaust it, and recover by closing one. */
    struct rlimit original;
    check(getrlimit(RLIMIT_NOFILE, &original) == 0, "read descriptor limit");
    int probe = dup(0);
    check(probe >= 0, "probe descriptor");
    if (probe >= 0) close(probe);
    struct rlimit lowered = original;
    lowered.rlim_cur = (rlim_t)(probe + FD_HEADROOM);
    if (probe < 0 || setrlimit(RLIMIT_NOFILE, &lowered) != 0) {
        check(0, "lower descriptor limit");
        return;
    }
    int64_t handles[FD_HEADROOM + 4] = {0};
    int opened = 0;
    for (; opened < FD_HEADROOM + 4; opened++) {
        handles[opened] = open_udp();
        if (handles[opened] <= 0) break;
    }
    check(opened > 0 && opened < FD_HEADROOM + 4, "descriptor exhaustion refuses another socket");
    check(opened < FD_HEADROOM + 4 && handles[opened] == ESHKOL_IO_SYSTEM,
          "exhaustion reports a system outcome");
    if (opened > 0) {
        check(eshkol_net_close(handles[0]) == 0, "free one descriptor");
        handles[0] = open_udp();
        check(handles[0] > 0, "socket creation recovers after close");
    }
    for (int i = 0; i < opened; i++) if (handles[i] > 0) eshkol_net_close(handles[i]);
    check(setrlimit(RLIMIT_NOFILE, &original) == 0, "restore descriptor limit");
}

int main(void) {
    int rc = cond_init_monotonic(&poll_cv);
    if (rc) { fprintf(stderr, "FAIL: initialize poll condition: %s\n", strerror(rc)); return 2; }

    close_blocked_accept();
    close_blocked_read();
    close_blocked_udp_recv();
    stale_id_and_slot_control();

    pthread_cond_destroy(&poll_cv);
    int count = atomic_load(&failures);
    if (count) {
        fprintf(stderr, "RESULT: FAIL (%d failures)\n", count);
        return 1;
    }
    puts("PASS: in-flight close cancels promptly, retired IDs, and descriptor exhaustion/recovery");
    puts("RESULT: PASS");
    return 0;
}
