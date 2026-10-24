#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>

/* Deterministic kernel model: readiness persists while recvmsg would block.
 * The advancing clock must stop retries, not a lucky real poll timeout. */
enum { READY, INTERRUPTED, INVALID };
static int mode, polls, queued, clock_failed;
static int64_t tick;
static int injected_clock(clockid_t which, struct timespec *out) {
    (void)which;
    if (clock_failed) { errno = EIO; return -1; }
    out->tv_sec = tick / 1000;
    out->tv_nsec = (tick % 1000) * 1000000L;
    return 0;
}
static int injected_poll(struct pollfd *fds, nfds_t count, int timeout) {
    (void)count; (void)timeout;
    ++polls; ++tick;
    /* Bound the pre-fix negative control instead of hanging the test runner. */
    if (polls >= 8) { errno = EIO; return -1; }
    if (mode == INTERRUPTED) { errno = EINTR; return -1; }
    fds[0].revents = mode == INVALID ? POLLNVAL : POLLIN;
    return 1;
}
static ssize_t injected_recvmsg(int fd, struct msghdr *msg, int flags) {
    (void)fd; (void)flags;
    if (!queued) { errno = EAGAIN; return -1; }
    struct sockaddr_in peer;
    memset(&peer, 0, sizeof(peer));
    peer.sin_family = AF_INET; peer.sin_port = htons(1234);
    peer.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    memcpy(msg->msg_name, &peer, sizeof(peer));
    ((char *)msg->msg_iov[0].iov_base)[0] = 'x';
    msg->msg_flags = 0;
    queued = 0;
    return 1;
}
#define clock_gettime injected_clock
#define poll injected_poll
#define recvmsg injected_recvmsg
#include "../../lib/core/net.c"
#undef recvmsg
#undef poll
#undef clock_gettime

static int failures;
static void check(int ok, const char *label) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", label); ++failures; }
}
static void reset(void) {
    tick = 0; polls = 0; mode = READY; queued = 0; clock_failed = 0;
    memset(slots, 0, sizeof(slots));
    slots[0] = (Net){-1, 3, 0, 0, 0, 1};
}
static int receive(int timeout) {
    char buffer[8], peer[128];
    int result = eshkol_net_recvfrom(1, buffer, sizeof(buffer), peer, sizeof(peer), timeout);
    if (result == 1) check(buffer[0] == 'x' && strcmp(peer, "127.0.0.1:1234") == 0,
                           "immediate datagram contents survive");
    return result;
}
int main(void) {
    reset();
    check(receive(3) == -2 && polls == 3 && tick == 3 && slots[0].refs == 0,
          "persistent readiness cannot extend the receive deadline or retain a reference");
    reset();
    check(receive(0) == -2 && polls == 0 && slots[0].refs == 0,
          "zero wait budget returns would-block without polling");
    reset(); tick = 5;
    check(wait_fd(-1, POLLIN, 4) == -2 && polls == 0, "expired deadline refuses readiness");
    reset();
    check(wait_fd(-1, POLLIN, 3) == 0 && polls == 1, "readiness before deadline remains usable");
    reset(); mode = INTERRUPTED;
    check(wait_fd(-1, POLLIN, 3) == -2 && polls == 3, "interruptions cannot extend the deadline");
    reset(); mode = INVALID;
    check(wait_fd(-1, POLLIN, 3) == -1, "invalid descriptor remains a system error");
    reset(); clock_failed = 1;
    check(receive(3) == -1 && polls == 0 && slots[0].refs == 0,
          "clock failure remains a system error without acquiring a reference");
    reset(); queued = 1;
    check(receive(0) == 1 && polls == 0 && slots[0].refs == 0,
          "already available datagram succeeds with zero wait budget");
    if (failures) { printf("RESULT: FAIL (%d failures)\n", failures); return 1; }
    puts("RESULT: PASS (persistent readiness, deadlines and zero-wait data)");
    return 0;
}
