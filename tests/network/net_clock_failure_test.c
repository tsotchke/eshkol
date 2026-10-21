/* Inject clock failure into the actual runtime; never read poisoned times. */
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
static int injected_clock(clockid_t which, struct timespec *value) {
    (void)which; value->tv_sec = 100; value->tv_nsec = 0; errno = EIO; return -1;
}
#define clock_gettime injected_clock
#include "../../lib/core/net.c"
#undef clock_gettime
int main(void) {
    if (now_ms() != -1) { puts("FAIL: failed clock became a timestamp"); return 1; }
    if (wait_fd(-1, POLLIN, 1000) != -1) { puts("FAIL: failed clock became a timeout"); return 1; }
    int64_t deadline = 0;
    /* A live synthetic slot reaches acquire's clock path without sockets. */
    slots[0] = (Net){-1, 2, 0, 0, 0, 1};
    if (acquire(1, 2, 20, &deadline) || deadline != -1 || slots[0].refs) {
        puts("FAIL: clock refusal acquired a reference or hid its system error"); return 1;
    }
    char buf[8];
    if (eshkol_net_read(1, buf, sizeof(buf), 20) != -1 || slots[0].refs) {
        puts("FAIL: read did not propagate clock system failure"); return 1;
    }
    memset(&slots[0], 0, sizeof(slots[0]));
    puts("PASS: networking clock failure refuses without leaked references"); return 0;
}
#endif
