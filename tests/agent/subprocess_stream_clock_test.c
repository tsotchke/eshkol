/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include <eshkol/core/event_loop.h>
#include <eshkol/core/subprocess_stream.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fail_clock;
int __real_clock_gettime(clockid_t, struct timespec *);

int __wrap_clock_gettime(clockid_t which, struct timespec *ts) {
    if (which == CLOCK_MONOTONIC && fail_clock) {
        errno = EIO;
        return -1;
    }
    return __real_clock_gettime(which, ts);
}

int eshkol_capability_runtime_allows(const char *name) {
    return name && strcmp(name, "subprocess") == 0;
}

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: subprocess_stream_clock_test:%d: %s\n", __LINE__, #expr); \
        return 1; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc > 1) {
        signal(SIGTERM, SIG_IGN);
        alarm(5); /* Bound the helper lifetime even if a parent assertion fails. */
        CHECK(write(STDOUT_FILENO, "!", 1) == 1);
        for (;;) pause();
    }

    char packed[4096];
    int n = snprintf(packed, sizeof packed, "%s\tchild", argv[0]);
    CHECK(n > 0 && (size_t)n < sizeof packed);
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = eshkol_process_stream_open(loop, packed, ".", NULL);
    CHECK(id > 0);
    CHECK(eshkol_process_stream_poll(id, 1000) > 0);
    char byte;
    CHECK(eshkol_process_stream_read(id, 1, &byte, 1) == 1 && byte == '!');

    fail_clock = 1;
    CHECK(eshkol_process_stream_close(id, 100, 20) == ESHKOL_PROCESS_STREAM_ESYSTEM);
    CHECK(eshkol_process_cleanup_next(0) == id);
    CHECK(eshkol_process_stream_read(id, 1, &byte, 1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    CHECK(eshkol_process_stream_runtime_shutdown() == ESHKOL_PROCESS_STREAM_EBUSY);

    fail_clock = 0;
    loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    CHECK(eshkol_process_cleanup_drain(loop, 1000) == 0);
    CHECK(eshkol_process_cleanup_status(id) == 0);
    CHECK(eshkol_process_cleanup_consume(id) == 0);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    CHECK(eshkol_process_stream_runtime_shutdown() == 0);
    puts("PASS: monotonic failure retains discoverable owner and cancellation intent");
    return 0;
}
