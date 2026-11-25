/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include <eshkol/core/event_loop.h>
#include <eshkol/core/resource_lease.h>
#include <eshkol/core/subprocess_stream.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int fail_remove;
int __real_eshkol_event_loop_remove(eshkol_event_loop_t *, int);
int __wrap_eshkol_event_loop_remove(eshkol_event_loop_t *loop, int fd) {
    if (fail_remove) { errno = EIO; return ESHKOL_EVENT_LOOP_EOSERR; }
    return __real_eshkol_event_loop_remove(loop, fd);
}
int eshkol_capability_runtime_allows(const char *name) {
    return name && !strcmp(name, "subprocess");
}
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL watch-removal line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static int finish(int64_t id, int64_t loop) {
    int rc = eshkol_process_stream_close(id, 1000, 0);
    CHECK(rc == 0 || rc == ESHKOL_PROCESS_STREAM_PENDING);
    CHECK(eshkol_process_cleanup_drain(loop, 1000) == 0);
    CHECK(eshkol_process_cleanup_status(id) == 0);
    CHECK(eshkol_process_cleanup_consume(id) == 0);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    return 0;
}
int main(int argc, char **argv) {
    if (argc > 1) {
        alarm(5);
        if (!strcmp(argv[1], "eof")) close(STDOUT_FILENO);
        CHECK(write(STDERR_FILENO, "!", 1) == 1);
        for (;;) pause();
    }
    char packed[4096];
    int64_t loop = eshkol_event_loop_open_handle(8); CHECK(loop > 0);
    CHECK(snprintf(packed, sizeof(packed), "%s\tidle", argv[0]) < (int)sizeof(packed));
    int64_t id = eshkol_process_stream_open(loop, packed, ".", NULL); CHECK(id > 0);
    CHECK(eshkol_process_stream_poll(id, 1000) > 0);
    char byte; CHECK(eshkol_process_stream_read(id, 2, &byte, 1) == 1);
    CHECK(eshkol_resource_close(id, ESHKOL_HANDLE_KIND_PROCESS, 0) == ESHKOL_IO_BUSY);
    fail_remove = 1;
    CHECK(eshkol_process_stream_close(id, 0, 0) == ESHKOL_PROCESS_STREAM_ESYSTEM);
    CHECK(eshkol_process_cleanup_next(0) == id);
    CHECK(eshkol_event_loop_count(eshkol_event_loop_from_handle(loop)) == 2);
    CHECK(eshkol_event_loop_close_handle(loop) == ESHKOL_EVENT_LOOP_EBUSY);
    CHECK(eshkol_process_cleanup_status(id) == ESHKOL_PROCESS_STREAM_ESYSTEM);
    eshkol_resource_lease_t lease;
    CHECK(eshkol_resource_borrow(id, ESHKOL_HANDLE_KIND_PROCESS, 0, &lease) == ESHKOL_IO_BUSY);
    fail_remove = 0;
    CHECK(finish(id, loop) == 0);

    loop = eshkol_event_loop_open_handle(8); CHECK(loop > 0);
    CHECK(snprintf(packed, sizeof(packed), "%s\teof", argv[0]) < (int)sizeof(packed));
    id = eshkol_process_stream_open(loop, packed, ".", NULL); CHECK(id > 0);
    CHECK(eshkol_process_stream_poll(id, 1000) > 0);
    const eshkol_event_t *events; int count;
    CHECK(eshkol_event_loop_poll(eshkol_event_loop_from_handle(loop), 1000, &events, &count) == 0);
    int watched[2], n = 0;
    for (int i = 0; i < count && n < 2; ++i) if (events[i].user_data == (uint64_t)id) watched[n++] = events[i].fd;
    CHECK(n > 0);
    fail_remove = 1;
    CHECK(eshkol_process_stream_read(id, 1, &byte, 1) == ESHKOL_PROCESS_STREAM_ESYSTEM);
    CHECK(eshkol_event_loop_count(eshkol_event_loop_from_handle(loop)) == 2);
    for (int i = 0; i < n; ++i) CHECK(fcntl(watched[i], F_GETFD) >= 0);
    fail_remove = 0;
    CHECK(eshkol_process_stream_read(id, 1, &byte, 1) == 0);
    CHECK(eshkol_event_loop_count(eshkol_event_loop_from_handle(loop)) == 1);
    CHECK(finish(id, loop) == 0);
    CHECK(eshkol_process_stream_runtime_shutdown() == 0);
    puts("PASS: failed watch removal retains PROCESS descriptors/loop/lease and retries close/EOF cleanup");
    return 0;
}
