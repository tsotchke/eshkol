/* Inject clock failure into the actual runtime; never read poisoned times. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(_WIN32) || defined(ESHKOL_VM_WASM) || defined(__EMSCRIPTEN__)
#include <stdio.h>
int main(void) { puts("PASS: clock failure test has no supported socket backend"); return 0; }
#else
#include <time.h>
#include <errno.h>
#include <stdio.h>
static int clock_fails;
static int injected_clock(clockid_t which, struct timespec *value) {
    if (!clock_fails) return clock_gettime(which, value);
    value->tv_sec = 100; value->tv_nsec = 0; errno = EIO; return -1;
}
#define clock_gettime injected_clock
#include "../../lib/core/resource_lease.c"
#undef clock_gettime
#include "eshkol/core/net.h"
#include <sys/socket.h>
int main(void) {
    clock_fails = 1;
    if (eshkol_resource_now_ms() != -1) { puts("FAIL: failed clock became a timestamp"); return 1; }
    if (eshkol_resource_deadline(1000) != -1 || eshkol_resource_remaining(5000) != -1) {
        puts("FAIL: failed clock became a deadline"); return 1;
    }
    if (eshkol_net_open("127.0.0.1", 0, 2, 20) != -1) {
        puts("FAIL: connect without a clock did not refuse"); return 1;
    }
    clock_fails = 0;
    int64_t udp = eshkol_net_open("127.0.0.1", 0, 3, 0);
    if (udp <= 0) { puts("FAIL: open datagram socket"); return 1; }
    clock_fails = 1;
    char buf[8], peer[64];
    if (eshkol_net_recvfrom(udp, buf, sizeof(buf), peer, sizeof(peer), 20) != -1) {
        puts("FAIL: receive did not propagate clock system failure"); return 1;
    }
    eshkol_handle_state_t state = ESHKOL_HANDLE_STATE_INVALID;
    if (eshkol_handle_get_state(eshkol_resource_registry(), (eshkol_handle_id_t)udp, &state) !=
            ESHKOL_HANDLE_OK || state != ESHKOL_HANDLE_STATE_OPEN) {
        puts("FAIL: clock refusal kept the lease"); return 1;
    }
    clock_fails = 0;
    if (eshkol_net_close(udp) != 0) { puts("FAIL: close after clock refusal"); return 1; }
    puts("PASS: networking clock failure refuses without retained leases"); return 0;
}
#endif
