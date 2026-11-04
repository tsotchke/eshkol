/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
/* ru_maxrss is a Darwin extension hidden by the strict POSIX profile. */
#define _DARWIN_C_SOURCE 1
#endif
#include <eshkol/core/event_loop.h>
#include <eshkol/core/subprocess_stream.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#define LARGE_BYTES (17u * 1024u * 1024u + 37u)
#define OUT ESHKOL_PROCESS_STREAM_STDOUT
#define ERR ESHKOL_PROCESS_STREAM_STDERR
#define BLOCK ESHKOL_PROCESS_STREAM_WOULD_BLOCK
#define PENDING ESHKOL_PROCESS_STREAM_PENDING

static char *self;
static int capability_allowed = 1;
static unsigned int checks, scenarios;
static int64_t tracked[64];
static pid_t descendant;

int eshkol_capability_runtime_allows(const char *name) {
    return capability_allowed && name && strcmp(name, "subprocess") == 0;
}

static int64_t now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) abort();
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void sleep_ms(long ms) {
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000000 };
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
}

static void check(int ok, const char *expr, int line) {
    ++checks;
    if (!ok) {
        fprintf(stderr, "FAIL: subprocess_stream_test:%d: %s\n", line, expr);
        exit(1);
    }
}
#define CHECK(expr) check(!!(expr), #expr, __LINE__)

static int write_all(int fd, const void *data, size_t count) {
    const unsigned char *p = data;
    while (count) {
        ssize_t n = write(fd, p, count);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += n;
        count -= (size_t)n;
    }
    return 0;
}

static unsigned char pattern(size_t offset, unsigned int stream) {
    return (unsigned char)((offset * 33u + stream * 71u) % 251u);
}

static int child(const char *mode) {
    if (strcmp(mode, "binary") == 0) {
        const unsigned char out[] = { 'a', 0, 'b', 255, '\n' };
        const unsigned char err[] = { 0, 'e', 0, 'r' };
        unsigned char input;
        if (read(STDIN_FILENO, &input, 1) != 0) return 90;
        if (write_all(STDOUT_FILENO, out, sizeof(out)) ||
            write_all(STDERR_FILENO, err, sizeof(err))) return 91;
        return 7;
    }
    if (strcmp(mode, "large") == 0) {
        unsigned char out[4096], err[4096];
        size_t offset = 0;
        while (offset < LARGE_BYTES) {
            size_t n = LARGE_BYTES - offset;
            if (n > sizeof(out)) n = sizeof(out);
            for (size_t i = 0; i < n; ++i) {
                out[i] = pattern(offset + i, 0);
                err[i] = pattern(offset + i, 1);
            }
            if (write_all(STDOUT_FILENO, out, n) ||
                write_all(STDERR_FILENO, err, n)) return 92;
            offset += n;
        }
        return 19;
    }
    if (strcmp(mode, "idle") == 0 || strcmp(mode, "ignore-term") == 0) {
        if (strcmp(mode, "ignore-term") == 0) signal(SIGTERM, SIG_IGN);
        if (write_all(STDOUT_FILENO, "R", 1)) return 93;
        sleep_ms(5000);
        return 0;
    }
    if (strcmp(mode, "early-close") == 0) {
        if (write_all(STDOUT_FILENO, "R", 1)) return 94;
        close(STDOUT_FILENO);
        sleep_ms(30);
        if (write_all(STDERR_FILENO, "after", 5)) return 95;
        return 23;
    }
    if (strcmp(mode, "inherited-writer") == 0) {
        pid_t pid = fork();
        if (pid < 0) return 96;
        if (pid == 0) {
            sleep_ms(5000);
            _exit(0);
        }
        if (write_all(STDOUT_FILENO, &pid, sizeof(pid))) return 97;
        return 31;
    }
    return 99;
}

static void track(int64_t id) {
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i) {
        if (!tracked[i]) { tracked[i] = id; return; }
    }
    CHECK(0);
}

static void untrack(int64_t id) {
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i)
        if (tracked[i] == id) tracked[i] = 0;
}

static void emergency_cleanup(void) {
    if (descendant > 0) {
        kill(descendant, SIGKILL);
#ifdef __linux__
        while (waitpid(descendant, NULL, 0) < 0 && errno == EINTR) {}
#endif
    }
    int any = 0;
    for (size_t i = 0; i < sizeof(tracked) / sizeof(*tracked); ++i) {
        if (tracked[i] > 0) {
            eshkol_process_stream_close(tracked[i], 50, 0);
            any = 1;
        }
    }
    if (any) {
        int64_t loop = eshkol_event_loop_open_handle(8);
        if (loop > 0) {
            eshkol_process_cleanup_drain(loop, 1000);
            for (int64_t id = eshkol_process_cleanup_next(0); id > 0;
                 id = eshkol_process_cleanup_next(id))
                eshkol_process_cleanup_consume(id);
            eshkol_event_loop_close_handle(loop);
        }
    }
}

static int64_t open_child(int64_t loop, const char *mode) {
    size_t cap = strlen(self) + strlen(mode) + 12;
    char *argv = malloc(cap);
    CHECK(argv != NULL);
    snprintf(argv, cap, "%s\t--child=%s", self, mode);
    int64_t id = eshkol_process_stream_open(loop, argv, ".", NULL);
    free(argv);
    CHECK(id > 0);
    track(id);
    return id;
}

static void finish(int64_t id, int64_t loop) {
    int status = eshkol_process_stream_close(id, 1000, 10);
    CHECK(status == 0 || status == PENDING);
    int64_t deadline = now_ms() + 1500;
    while (eshkol_process_cleanup_status(id) == PENDING && now_ms() < deadline)
        CHECK(eshkol_process_cleanup_drain(loop, 50) >= 0);
    CHECK(eshkol_process_cleanup_status(id) == 0);
    CHECK(eshkol_process_cleanup_consume(id) == 0);
    untrack(id);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
}

static void ready(int64_t id) {
    unsigned char c = 0;
    int64_t deadline = now_ms() + 2000;
    for (;;) {
        int64_t n = eshkol_process_stream_read(id, OUT, &c, 1);
        if (n == 1) break;
        CHECK(n == BLOCK);
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 100) >= 0);
    }
    CHECK(c == 'R');
}

static void pass(const char *name) {
    ++scenarios;
    printf("PASS: %s\n", name);
}

static void binary_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = open_child(loop, "binary");
    CHECK(eshkol_event_loop_close_handle(loop) < 0);
    unsigned char output[2][16] = {{0}};
    size_t used[2] = {0, 0};
    int eof[2] = {0, 0};
    int64_t deadline = now_ms() + 2000;
    while (!eof[0] || !eof[1] || eshkol_process_stream_exit_code(id) == BLOCK) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 100) >= 0);
        for (int i = 0; i < 2; ++i) {
            if (eof[i]) continue;
            int64_t n = eshkol_process_stream_read(id, i ? ERR : OUT,
                output[i] + used[i], (int64_t)(sizeof(output[i]) - used[i]));
            CHECK(n >= 0 || n == BLOCK);
            if (n == 0) eof[i] = 1;
            if (n > 0) used[i] += (size_t)n;
        }
    }
    const unsigned char expected_out[] = { 'a', 0, 'b', 255, '\n' };
    const unsigned char expected_err[] = { 0, 'e', 0, 'r' };
    CHECK(used[0] == sizeof(expected_out) && !memcmp(output[0], expected_out, used[0]));
    CHECK(used[1] == sizeof(expected_err) && !memcmp(output[1], expected_err, used[1]));
    CHECK(eshkol_process_stream_exit_code(id) == 7);
    finish(id, loop);
    pass("NUL-containing stdout/stderr, null stdin, exit status, loop ownership");
}

static long maxrss_kib(void) {
    struct rusage usage;
    CHECK(getrusage(RUSAGE_SELF, &usage) == 0);
#ifdef __APPLE__
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

static void large_output_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    long baseline = maxrss_kib();
    int64_t id = open_child(loop, "large");
    sleep_ms(100);
    CHECK(eshkol_process_stream_exit_code(id) == BLOCK);
    unsigned char data[4096];
    size_t used[2] = {0, 0}, maximum_skew = 0;
    int eof[2] = {0, 0};
    unsigned int rounds = 0;
    int64_t deadline = now_ms() + 12000;
    while (!eof[0] || !eof[1] || eshkol_process_stream_exit_code(id) == BLOCK) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 100) >= 0);
        for (int i = 0; i < 2; ++i) {
            if (eof[i]) continue;
            int64_t n = eshkol_process_stream_read(id, i ? ERR : OUT, data, sizeof(data));
            CHECK(n >= 0 || n == BLOCK);
            if (n == 0) eof[i] = 1;
            for (int64_t j = 0; j < (n > 0 ? n : 0); ++j)
                CHECK(data[j] == pattern(used[i] + (size_t)j, (unsigned int)i));
            if (n > 0) used[i] += (size_t)n;
        }
        size_t skew = used[0] > used[1] ? used[0] - used[1] : used[1] - used[0];
        if (skew > maximum_skew) maximum_skew = skew;
        if (++rounds % 64 == 0) sleep_ms(1);
    }
    CHECK(used[0] == LARGE_BYTES && used[1] == LARGE_BYTES);
    CHECK(maximum_skew <= 1024u * 1024u);
    CHECK(eshkol_process_stream_exit_code(id) == 19);
    CHECK(maxrss_kib() - baseline < 8 * 1024);
    printf("MEASURE: bytes_stdout=%zu bytes_stderr=%zu max_skew=%zu rss_growth_kib=%ld\n",
        used[0], used[1], maximum_skew, maxrss_kib() - baseline);
    finish(id, loop);
    pass("lossless >16 MiB per stream, slow-consumer backpressure, fair bounded memory");
}

static void deadlines_and_cancel_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = open_child(loop, "ignore-term");
    ready(id);
    int64_t start = now_ms();
    CHECK(eshkol_process_stream_poll(id, 0) == 0);
    CHECK(now_ms() - start < 100);
    CHECK(eshkol_process_stream_exit_code(id) == BLOCK);
    start = now_ms();
    CHECK(eshkol_process_stream_poll(id, 100) == 0);
    int64_t elapsed = now_ms() - start;
    CHECK(elapsed >= 60 && elapsed < 600);
    CHECK(eshkol_process_stream_exit_code(id) == BLOCK);
    CHECK(eshkol_process_stream_cancel(id, 40) == 0);
    CHECK(eshkol_process_stream_cancel(id, 4000) == 0);
    int64_t deadline = now_ms() + 1000;
    while (eshkol_process_stream_exit_code(id) == BLOCK) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 50) >= 0);
    }
    CHECK(eshkol_process_stream_exit_code(id) == 128 + SIGKILL);
    printf("MEASURE: poll_100_ms_elapsed=%lld\n", (long long)elapsed);
    finish(id, loop);
    pass("zero/positive deadlines, no implicit kill, immutable cancellation grace");
}

static void early_close_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = open_child(loop, "early-close");
    ready(id);
    char data[16] = {0};
    size_t used = 0;
    int eof_out = 0, eof_err = 0;
    int64_t deadline = now_ms() + 2000;
    while (!eof_out || !eof_err || eshkol_process_stream_exit_code(id) == BLOCK) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 100) >= 0);
        if (!eof_out) {
            int64_t n = eshkol_process_stream_read(id, OUT, data + used, 1);
            CHECK(n == BLOCK || n == 0);
            eof_out = n == 0;
        }
        if (!eof_err) {
            int64_t n = eshkol_process_stream_read(id, ERR, data + used, (int64_t)(sizeof(data) - used));
            CHECK(n >= 0 || n == BLOCK);
            eof_err = n == 0;
            if (n > 0) used += (size_t)n;
        }
    }
    CHECK(used == 5 && !memcmp(data, "after", 5));
    CHECK(eshkol_process_stream_exit_code(id) == 23);
    finish(id, loop);
    pass("early stdout EOF preserves stderr progress");
}

static void exit_before_eof_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = open_child(loop, "inherited-writer");
    size_t used = 0;
    int64_t deadline = now_ms() + 2000;
    while (used != sizeof(descendant) || eshkol_process_stream_exit_code(id) == BLOCK) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_stream_poll(id, 100) >= 0);
        if (used != sizeof(descendant)) {
            int64_t n = eshkol_process_stream_read(id, OUT,
                (unsigned char *)&descendant + used, (int64_t)(sizeof(descendant) - used));
            CHECK(n > 0 || n == BLOCK);
            if (n > 0) used += (size_t)n;
        }
    }
    CHECK(descendant > 0);
    CHECK(eshkol_process_stream_exit_code(id) == 31);
    unsigned char data;
    CHECK(eshkol_process_stream_read(id, OUT, &data, 1) == BLOCK);
    CHECK(eshkol_process_stream_read(id, ERR, &data, 1) == BLOCK);
    int64_t start = now_ms();
    CHECK(eshkol_process_stream_close(id, 100, 10) == 0);
    CHECK(now_ms() - start < 100);
    CHECK(eshkol_process_cleanup_consume(id) == 0);
    untrack(id);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    CHECK(kill(descendant, SIGKILL) == 0);
#ifdef __linux__
    CHECK(waitpid(descendant, NULL, 0) == descendant);
#endif
    descendant = 0;
    pass("direct-child reap and cleanup progress independently of inherited pipe writers");
}

static void pending_cleanup_test(void) {
    int64_t original = eshkol_event_loop_open_handle(8);
    CHECK(original > 0);
    int64_t id = open_child(original, "ignore-term");
    ready(id);
    int64_t start = now_ms();
    CHECK(eshkol_process_stream_close(id, 0, 1000) == PENDING);
    CHECK(now_ms() - start < 100);
    char data;
    CHECK(eshkol_process_stream_read(id, OUT, &data, 1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_cleanup_next(0) == id);
    CHECK(eshkol_process_cleanup_status(id) == PENDING);
    CHECK(eshkol_process_cleanup_consume(id) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_process_stream_runtime_shutdown() == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_close_handle(original) == 0);
    int64_t fresh = eshkol_event_loop_open_handle(8);
    CHECK(fresh > 0);
    CHECK(eshkol_process_cleanup_drain(fresh, 0) > 0);
    int64_t deadline = now_ms() + 2000;
    while (eshkol_process_cleanup_status(id) == PENDING) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_process_cleanup_drain(fresh, 100) >= 0);
    }
    CHECK(eshkol_process_cleanup_status(id) == 0);
    CHECK(eshkol_process_stream_close(id, 0, 0) == 0);
    CHECK(eshkol_process_cleanup_consume(id) == 0);
    untrack(id);
    CHECK(eshkol_process_cleanup_next(0) == 0);
    CHECK(eshkol_process_cleanup_status(id) == ESHKOL_PROCESS_STREAM_EINVAL);
    int64_t next = open_child(fresh, "idle");
    CHECK(next != id);
    CHECK(eshkol_process_stream_read(id, OUT, &data, 1) == ESHKOL_PROCESS_STREAM_EINVAL);
    finish(next, fresh);
    pass("bounded teardown, retained owner across loop closure, terminal consumption and stale IDs");
}

static void registration_rollback_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(1);
    CHECK(loop > 0);
    size_t cap = strlen(self) + 20;
    char *argv = malloc(cap);
    CHECK(argv != NULL);
    snprintf(argv, cap, "%s\t--child=idle", self);
    CHECK(eshkol_process_stream_open(loop, argv, ".", NULL) < 0);
    free(argv);
    eshkol_event_loop_t *native = eshkol_event_loop_from_handle(loop);
    CHECK(native != NULL && eshkol_event_loop_count(native) == 0);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    int64_t fresh = eshkol_event_loop_open_handle(8);
    CHECK(fresh > 0);
    CHECK(eshkol_process_cleanup_drain(fresh, 1500) == 0);
    for (int64_t id = eshkol_process_cleanup_next(0); id > 0;
         id = eshkol_process_cleanup_next(id)) {
        CHECK(eshkol_process_cleanup_status(id) == 0);
        CHECK(eshkol_process_cleanup_consume(id) == 0);
    }
    CHECK(eshkol_event_loop_close_handle(fresh) == 0);
    pass("partial watch registration rolls back without orphaning ownership");
}

static void watch_identity_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t id = open_child(loop, "idle");
    eshkol_event_loop_t *native = eshkol_event_loop_from_handle(loop);
    CHECK(native != NULL);
    const eshkol_event_t *events;
    int count = 0, watched = -1;
    int64_t deadline = now_ms() + 2000;
    while (watched < 0) {
        CHECK(now_ms() < deadline);
        CHECK(eshkol_event_loop_poll(native, 100, &events, &count) == 0);
        for (int i = 0; i < count; ++i)
            if (events[i].user_data == (uint64_t)id) watched = events[i].fd;
    }
    ready(id);
    CHECK(eshkol_event_loop_count(native) == 2);
    CHECK(eshkol_event_loop_add(native, watched, ESHKOL_EVENT_READ, (uint64_t)id + 1) == 0);
    CHECK(eshkol_event_loop_count(native) == 2);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_add(native, watched, ESHKOL_EVENT_READ, (uint64_t)id) == 0);
    CHECK(eshkol_process_stream_poll(id, 0) == 0);
    CHECK(eshkol_event_loop_add(native, watched, ESHKOL_EVENT_WRITE, (uint64_t)id) == 0);
    CHECK(eshkol_event_loop_count(native) == 2);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_add(native, watched, ESHKOL_EVENT_READ, (uint64_t)id) == 0);
    int unmanaged[2];
    CHECK(pipe(unmanaged) == 0);
    CHECK(eshkol_event_loop_add(native, unmanaged[0], ESHKOL_EVENT_READ, 0) == 0);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_remove(native, unmanaged[0]) == 0);
    CHECK(eshkol_event_loop_remove(native, watched) == 0);
    CHECK(eshkol_event_loop_add(native, unmanaged[0], ESHKOL_EVENT_READ, (uint64_t)id) == 0);
    CHECK(eshkol_event_loop_count(native) == 2);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_remove(native, unmanaged[0]) == 0);
    CHECK(eshkol_event_loop_add(native, watched, ESHKOL_EVENT_READ, (uint64_t)id) == 0);
    CHECK(close(unmanaged[0]) == 0 && close(unmanaged[1]) == 0);
    CHECK(eshkol_process_stream_poll(id, 0) == 0);
    finish(id, loop);
    pass("unmanaged watches and same-count descriptor/cookie/interest tampering are rejected");
}

static int thread_status;
static void *foreign_thread(void *arg) {
    int64_t id = *(int64_t *)arg;
    thread_status = eshkol_process_stream_poll(id, 0);
    return NULL;
}

static void validation_and_policy_test(void) {
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    capability_allowed = 0;
    CHECK(eshkol_process_stream_open(loop, self, ".", NULL) == ESHKOL_PROCESS_STREAM_EDENIED);
    capability_allowed = 1;
    CHECK(eshkol_process_stream_open(-1, self, ".", NULL) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_open(loop, "", ".", NULL) == ESHKOL_PROCESS_STREAM_EINVAL);
    int64_t id = open_child(loop, "idle");
    ready(id);
    char data;
    CHECK(eshkol_process_stream_read(id, 0, &data, 1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_read(id, OUT, NULL, 1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_read(id, OUT, &data, 0) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_read(id, OUT, &data, -1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_poll(id, -1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_cancel(id, -1) == ESHKOL_PROCESS_STREAM_EINVAL);
    CHECK(eshkol_process_stream_close(id, -1, 0) == ESHKOL_PROCESS_STREAM_EINVAL);
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, foreign_thread, &id) == 0);
    CHECK(pthread_join(thread, NULL) == 0);
    CHECK(thread_status == ESHKOL_PROCESS_STREAM_ETHREAD);
    capability_allowed = 0;
    CHECK(eshkol_process_stream_read(id, OUT, &data, 1) == ESHKOL_PROCESS_STREAM_EDENIED);
    CHECK(eshkol_process_stream_poll(id, 0) == ESHKOL_PROCESS_STREAM_EDENIED);
    CHECK(eshkol_process_stream_exit_code(id) == ESHKOL_PROCESS_STREAM_EDENIED);
    finish(id, loop);
    capability_allowed = 1;
    pass("argument validation, owner-thread confinement, capability denial and unconditional cleanup");
}

static void capacity_test(void) {
    CHECK(setenv("ESHKOL_SUBPROC_MAX_CONCURRENT", "1", 1) == 0);
    int64_t loop = eshkol_event_loop_open_handle(8);
    CHECK(loop > 0);
    int64_t first = open_child(loop, "ignore-term");
    ready(first);
    size_t cap = strlen(self) + 20;
    char *argv = malloc(cap);
    CHECK(argv != NULL);
    snprintf(argv, cap, "%s\t--child=idle", self);
    CHECK(eshkol_process_stream_open(loop, argv, ".", NULL) == ESHKOL_PROCESS_STREAM_ELIMIT);
    CHECK(eshkol_process_stream_close(first, 0, 1000) == PENDING);
    CHECK(eshkol_process_cleanup_consume(first) == ESHKOL_PROCESS_STREAM_EBUSY);
    CHECK(eshkol_event_loop_close_handle(loop) == 0);
    int64_t fresh = eshkol_event_loop_open_handle(8);
    CHECK(fresh > 0);
    CHECK(eshkol_process_cleanup_drain(fresh, 1500) == 0);
    CHECK(eshkol_process_stream_open(fresh, argv, ".", NULL) == ESHKOL_PROCESS_STREAM_ELIMIT);
    free(argv);
    CHECK(eshkol_process_cleanup_consume(first) == 0);
    untrack(first);
    int64_t next = open_child(fresh, "idle");
    finish(next, fresh);
    CHECK(unsetenv("ESHKOL_SUBPROC_MAX_CONCURRENT") == 0);
    pass("reserved cleanup records respect subprocess budget and release capacity");
}

int main(int argc, char **argv) {
    if (argc == 2 && !strncmp(argv[1], "--child=", 8)) return child(argv[1] + 8);
    self = realpath(argv[0], NULL);
    CHECK(self != NULL);
    CHECK(strchr(self, '\t') == NULL);
#ifdef __linux__
    CHECK(prctl(PR_SET_CHILD_SUBREAPER, 1) == 0);
#endif
    atexit(emergency_cleanup);
    binary_test();
    large_output_test();
    deadlines_and_cancel_test();
    early_close_test();
    exit_before_eof_test();
    pending_cleanup_test();
    validation_and_policy_test();
    capacity_test();
    registration_rollback_test();
    watch_identity_test();
    CHECK(eshkol_process_stream_runtime_shutdown() == 0);
    printf("PASS: subprocess_stream_contract (%u scenarios, %u checks)\n", scenarios, checks);
    free(self);
    return 0;
}
