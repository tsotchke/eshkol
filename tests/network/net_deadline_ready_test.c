#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
/* Persistent readiness must not extend a network deadline.
 *
 * The event loop can keep reporting a descriptor as ready while the next
 * nonblocking system call on it still answers EAGAIN (a datagram socket after
 * shutdown behaves that way). The wait inside the resource lease layer is
 * therefore bounded by its deadline alone: it consults the loop only while
 * budget remains, and an exhausted budget yields ESHKOL_IO_TIMEOUT.
 *
 * The clock and the loop poll are both injected, so the counts below are exact
 * rather than dependent on real timing. Loopback only. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(_WIN32) || defined(ESHKOL_VM_WASM) || defined(__EMSCRIPTEN__)
#include <stdio.h>
int main(void) {
    puts("RESULT: PASS (no supported socket backend on this platform)");
    return 0;
}
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "eshkol/core/event_loop.h"

/* Virtual monotonic clock: only an injected poll advances it, one millisecond
 * per call, so a wait's poll count is exactly its millisecond budget. */
static int64_t tick_ms;
static int clock_failed;
static int readiness_polls;
static int watched_fd = -1;
static int deliver_on_poll;     /* queue one datagram when readiness is first reported */
static int local_port;
static eshkol_event_t reported;

static int injected_clock(clockid_t which, struct timespec *out) {
    (void)which;
    if (clock_failed) { errno = EIO; return -1; }
    out->tv_sec = (time_t)(tick_ms / 1000);
    out->tv_nsec = (long)((tick_ms % 1000) * 1000000L);
    return 0;
}
static void deliver_datagram(void) {
    int sender = socket(AF_INET, SOCK_DGRAM, 0);
    if (sender < 0) return;
    struct sockaddr_in to;
    memset(&to, 0, sizeof(to));
    to.sin_family = AF_INET;
    to.sin_port = htons((uint16_t)local_port);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ssize_t sent = sendto(sender, "x", 1, 0, (struct sockaddr *)&to, sizeof(to));
    (void)sent;
    close(sender);
    /* Give the loopback datagram time to arrive before the next receive. */
    struct timespec pause = {0, 20 * 1000 * 1000L};
    nanosleep(&pause, NULL);
}
static int injected_add(eshkol_event_loop_t *loop, int fd, int events, uint64_t cookie) {
    if (cookie == 1) watched_fd = fd;
    return eshkol_event_loop_add(loop, fd, events, cookie);
}
static int injected_poll(eshkol_event_loop_t *loop, int timeout_ms,
                         const eshkol_event_t **out_events, int *n_events) {
    (void)loop; (void)timeout_ms;
    ++readiness_polls;
    ++tick_ms;
    if (deliver_on_poll) { deliver_on_poll = 0; deliver_datagram(); }
    reported.fd = watched_fd;
    reported.events = ESHKOL_EVENT_READ | ESHKOL_EVENT_WRITE;
    reported.user_data = 1;
    *out_events = &reported;
    *n_events = 1;
    return ESHKOL_EVENT_LOOP_OK;
}
#define clock_gettime injected_clock
#define eshkol_event_loop_add injected_add
#define eshkol_event_loop_poll injected_poll
#include "../../lib/core/resource_lease.c"
#undef eshkol_event_loop_poll
#undef eshkol_event_loop_add
#undef clock_gettime

#include "eshkol/core/net.h"

static int failures;
static void check(int ok, const char *label) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", label); ++failures; }
}
static void reset(void) { readiness_polls = 0; deliver_on_poll = 0; }

int main(void) {
    int64_t udp = eshkol_net_open("127.0.0.1", 0, 3, 0);
    if (udp <= 0) { puts("FAIL: open datagram socket"); return 1; }
    local_port = eshkol_net_port(udp, 0);
    if (local_port <= 0) { puts("FAIL: resolve the bound port"); return 1; }

    char buffer[8], peer[128];

    reset();
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 3) ==
                  ESHKOL_IO_TIMEOUT &&
              readiness_polls == 3,
          "persistent readiness cannot extend the receive deadline");

    reset();
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 0) ==
                  ESHKOL_IO_TIMEOUT &&
              readiness_polls == 0,
          "a zero wait budget reports would-block without consulting the loop");

    reset();
    deliver_datagram();
    memset(buffer, 0, sizeof(buffer));
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 0) == 1 &&
              buffer[0] == 'x' && readiness_polls == 0 &&
              strncmp(peer, "127.0.0.1:", 10) == 0,
          "an already available datagram succeeds with a zero wait budget");

    reset();
    deliver_on_poll = 1;
    memset(buffer, 0, sizeof(buffer));
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 5) == 1 &&
              buffer[0] == 'x' && readiness_polls == 1,
          "readiness before the deadline stays usable");

    reset();
    check(eshkol_net_sendto(udp, "127.0.0.1", local_port, "y", 1, 0) == 1 &&
              readiness_polls == 0,
          "a writable datagram socket sends without consulting the loop");
    struct timespec pause = {0, 20 * 1000 * 1000L};
    nanosleep(&pause, NULL);
    reset();
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 0) == 1,
          "the sent datagram is receivable");

    reset();
    clock_failed = 1;
    check(eshkol_net_recvfrom(udp, buffer, sizeof(buffer), peer, sizeof(peer), 3) ==
                  ESHKOL_IO_SYSTEM &&
              readiness_polls == 0,
          "a clock failure is a system failure, not a wait");
    clock_failed = 0;

    check(eshkol_net_close(udp) == 0, "close the datagram socket");
    if (failures) { printf("RESULT: FAIL (%d failures)\n", failures); return 1; }
    puts("RESULT: PASS (deadlines bound persistent readiness; zero-wait data still arrives)");
    return 0;
}
#endif
