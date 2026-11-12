/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#include <eshkol/core/unix_socket_stream.h>
#include <eshkol/core/event_loop.h>
#include "unix_socket_stream_fixture_support.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Pending core acceptance, not a second implementation. Raw sockets below
 * only create controlled external peers/backlogs. Every tested operation
 * uses the public Unix-stream API. Each invocation gets a fresh private
 * directory; failures retain it. No pre-existing pathname is removed. */
static unsigned checks;
static atomic_int network_allowed = 1;
static atomic_int policy_calls;
int eshkol_capability_runtime_allows(const char *capability) {
    atomic_fetch_add(&policy_calls, 1);
    return strcmp(capability, "network") == 0 && atomic_load(&network_allowed);
}
static void check(int ok, const char *what, int line) {
    ++checks;
    if (!ok) { fprintf(stderr, "FAIL: unix-stream:%d: %s (errno=%d %s)\n", line, what, errno, strerror(errno)); exit(1); }
}
#define CHECK(x) check(!!(x), #x, __LINE__)
static int64_t now_ms(void) {
    struct timespec t; CHECK(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static void nap(int ms) {
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    while (nanosleep(&t, &t) && errno == EINTR) {}
}
static char directory[80], path[104];
static int64_t loop_handle;
static void setup(void) {
    strcpy(directory, "/tmp/eshkol-unix-XXXXXX");
    CHECK(mkdtemp(directory) != NULL);
    CHECK(chmod(directory, 0700) == 0);
    CHECK(snprintf(path, sizeof(path), "%s/stream.sock", directory) < (int)sizeof(path));
    fprintf(stderr, "SCRATCH: %s\n", directory);
    loop_handle = eshkol_event_loop_open_handle(64); CHECK(loop_handle > 0);
}
static void finish_loop(void) { CHECK(eshkol_event_loop_close_handle(loop_handle) == 0); }
static void close_socket(int64_t id) { CHECK(eshkol_unix_stream_socket_close(id) == 0); }
typedef struct { int64_t listener, client, peer; } pair_t;
static pair_t pair_open(void) {
    pair_t p;
    p.listener = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(p.listener > 0);
    p.client = eshkol_unix_stream_socket_connect_begin(loop_handle, path, 1000); CHECK(p.client > 0);
    p.peer = eshkol_unix_stream_socket_accept(p.listener, 1000); CHECK(p.peer > 0);
    CHECK(eshkol_unix_stream_socket_connect_finish(p.client) == 0);
    return p;
}
static void pair_close(pair_t p) { close_socket(p.peer); close_socket(p.client); close_socket(p.listener); }
static void ready(int64_t id, int mask, int64_t deadline) {
    int64_t left = deadline - now_ms(); CHECK(left > 0);
    int n = eshkol_unix_stream_socket_poll(id, mask, (int32_t)(left > 50 ? 50 : left));
    CHECK(n == ESHKOL_IO_TIMEOUT || (n > 0 && (n & mask) && !(n & ~mask)));
}
static long peak_kib(void) {
    struct rusage r; CHECK(getrusage(RUSAGE_SELF, &r) == 0);
#if defined(__APPLE__)
    return r.ru_maxrss / 1024;
#else
    return r.ru_maxrss;
#endif
}
static void transfer(pair_t p, size_t size, int slow) {
    unsigned char *sent = malloc(size), *got = malloc(size); CHECK(sent && got);
    for (size_t i = 0; i < size; ++i) sent[i] = (unsigned char)(i % 7 ? i * 31 : 0);
    memset(got, 0xaa, size); /* fault in fixture buffers before core RSS measurement */
    long before = peak_kib();
    size_t written = 0, read_total = 0; unsigned short_writes = 0, blocked = 0;
    int64_t deadline = now_ms() + 20000;
    while (read_total < size) {
        CHECK(now_ms() < deadline);
        if (written < size) {
            size_t want = size - written; if (want > 65536) want = 65536;
            int64_t n = eshkol_unix_stream_socket_write(p.client, sent + written, (int64_t)want);
            if (n == ESHKOL_IO_TIMEOUT) {
                ++blocked;
                int mask = eshkol_unix_stream_socket_poll(p.client, 2, 0);
                CHECK(mask == ESHKOL_IO_TIMEOUT || mask == 2);
            }
            else { CHECK(n > 0 && (uint64_t)n <= want); short_writes += (size_t)n < want; written += (size_t)n; }
        }
        size_t want = size - read_total; if (want > 4096) want = 4096;
        int64_t n = eshkol_unix_stream_socket_read(p.peer, got + read_total, (int64_t)want);
        if (n == ESHKOL_IO_TIMEOUT) ready(p.peer, 1, deadline);
        else { CHECK(n > 0 && (uint64_t)n <= want); read_total += (size_t)n; }
        if (slow) nap(1);
    }
    CHECK(written == size && memcmp(sent, got, size) == 0);
    if (slow) CHECK(short_writes + blocked > 0);
    long growth = peak_kib() - before;
    CHECK(growth <= 8192); /* fixture's two touched buffers excluded above */
    fprintf(stderr, "MEASURE: bytes=%zu short_writes=%u would_block=%u core_peak_growth_kib=%ld\n", size, short_writes, blocked, growth);
    free(got); free(sent);
}
static int descriptor_count(void) {
    int count = eshkol_unix_socket_test_fd_count(); CHECK(count >= 0); return count;
}
static void scenario_connect_accept(void) { pair_t p = pair_open(); transfer(p, 257, 0); pair_close(p); }
static void scenario_partial_slow_peer(void) { pair_t p = pair_open(); transfer(p, 2u * 1024u * 1024u, 1); pair_close(p); }
static void scenario_nul_and_large_payload(void) { pair_t p = pair_open(); transfer(p, 16u * 1024u * 1024u + 37u, 1); pair_close(p); }
static void scenario_deadline_accept(void) {
    int64_t listener = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(listener > 0);
    int64_t start = now_ms();
    CHECK(eshkol_unix_stream_socket_accept(listener, 150) == ESHKOL_IO_TIMEOUT);
    int64_t elapsed = now_ms() - start; CHECK(elapsed >= 140 && elapsed < 1500);
    fprintf(stderr, "MEASURE: accept_150_ms_elapsed=%lld\n", (long long)elapsed); close_socket(listener);
}
/* A genuinely full OS listen queue, distinct from missing/refused paths.
 * The finite fill loop keeps all peer descriptors alive until after the
 * tested operation. Platforms that cannot establish it fail explicitly. */
typedef struct { int listener, peers[128], count; } backlog_t;
static backlog_t full_backlog(int probe_core) {
    /* With the scaffold, fail before constructing any raw socket fixture. */
    if (probe_core) { int64_t probe = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(probe > 0); close_socket(probe); }
    backlog_t b = {.listener = -1, .count = 0};
    b.listener = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(b.listener >= 0);
    struct sockaddr_un a; memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX;
    CHECK(strlen(path) < sizeof(a.sun_path)); strcpy(a.sun_path, path);
    CHECK(bind(b.listener, (struct sockaddr *)&a, sizeof(a)) == 0); CHECK(listen(b.listener, 1) == 0);
    CHECK(eshkol_unix_socket_test_pending_path(path) == 0);
#if defined(__APPLE__)
    fprintf(stderr, "MEASURE: pending_control=deterministic_EAGAIN_not_OS_pending\n");
#else
    fprintf(stderr, "MEASURE: pending_control=real_OS_backlog\n");
#endif
    int full = 0; int64_t limit = now_ms() + 1000;
    while (b.count < 128 && now_ms() < limit) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(fd >= 0);
        CHECK(fcntl(fd, F_SETFL, O_NONBLOCK) == 0); b.peers[b.count++] = fd;
        if (connect(fd, (struct sockaddr *)&a, sizeof(a)) < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { full = 1; break; }
            CHECK(errno == EINPROGRESS);
            struct sockaddr_un peer; socklen_t length = sizeof(peer);
            if (getpeername(fd, (struct sockaddr *)&peer, &length) != 0) { full = 1; break; }
        }
    }
    CHECK(full); return b;
}
static void backlog_close(backlog_t b) {
    for (int i = 0; i < b.count; ++i) CHECK(close(b.peers[i]) == 0);
    CHECK(close(b.listener) == 0); CHECK(unlink(path) == 0);
}
static void scenario_deadline_connect(void) {
    backlog_t b = full_backlog(1); int64_t start = now_ms();
    int64_t id = eshkol_unix_stream_socket_connect_begin(loop_handle, path, 250); CHECK(id > 0);
    CHECK(now_ms() - start < 150); nap(350);
    int64_t finish = now_ms(); CHECK(eshkol_unix_stream_socket_connect_finish(id) == ESHKOL_IO_TIMEOUT);
    CHECK(now_ms() - finish < 150); CHECK(now_ms() - start >= 350);
    close_socket(id); backlog_close(b);
}
static void scenario_destination_errors(void) {
    CHECK(eshkol_unix_stream_socket_connect_begin(loop_handle, path, 1000) == ESHKOL_IO_SYSTEM);
    CHECK(eshkol_unix_stream_socket_last_os_error() == ENOENT);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(fd >= 0);
    struct sockaddr_un a; memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX; strcpy(a.sun_path, path);
    CHECK(bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0); CHECK(close(fd) == 0);
    int64_t n = eshkol_unix_stream_socket_connect_begin(loop_handle, path, 1000);
    CHECK(n == ESHKOL_IO_REFUSED || (n == ESHKOL_IO_SYSTEM && eshkol_unix_stream_socket_last_os_error() == ECONNREFUSED));
    CHECK(unlink(path) == 0);
}
typedef struct { int64_t id; int first, second; } cancel_t;
static void *cancel_after(void *arg) {
    cancel_t *c = arg; nap(50);
    c->first = eshkol_unix_stream_socket_cancel(c->id); c->second = eshkol_unix_stream_socket_cancel(c->id); return NULL;
}
static void scenario_cancellation(void) {
    int64_t id = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(id > 0);
    cancel_t c = {.id = id}; pthread_t thread; CHECK(pthread_create(&thread, NULL, cancel_after, &c) == 0);
    int64_t start = now_ms(); CHECK(eshkol_unix_stream_socket_accept(id, 2500) == ESHKOL_IO_CANCELLED);
    CHECK(pthread_join(thread, NULL) == 0); CHECK(c.first == 0 && c.second == 0); CHECK(now_ms() - start < 1500);
    CHECK(eshkol_unix_stream_socket_accept(id, 0) == ESHKOL_IO_CANCELLED); close_socket(id);
}
static void scenario_connect_cancellation(void) {
    backlog_t b = full_backlog(1); int64_t id = eshkol_unix_stream_socket_connect_begin(loop_handle, path, 2500); CHECK(id > 0);
    cancel_t c = {.id = id}; pthread_t thread; CHECK(pthread_create(&thread, NULL, cancel_after, &c) == 0);
    int64_t start = now_ms(); CHECK(eshkol_unix_stream_socket_connect_finish(id) == ESHKOL_IO_CANCELLED);
    CHECK(pthread_join(thread, NULL) == 0); CHECK(c.first == 0 && c.second == 0); CHECK(now_ms() - start < 1500);
    CHECK(eshkol_unix_stream_socket_poll(id, 2, 0) == ESHKOL_IO_CANCELLED); close_socket(id); backlog_close(b);
}
static void peer_close_child(void) {
    CHECK(signal(SIGPIPE, SIG_DFL) != SIG_ERR);
    pair_t p = pair_open(); close_socket(p.peer);
    unsigned char bytes[4096] = {0}; int observed = 0; int64_t deadline = now_ms() + 1000;
    while (!observed && now_ms() < deadline) {
        int64_t n = eshkol_unix_stream_socket_write(p.client, bytes, sizeof(bytes));
        if (n == ESHKOL_IO_TIMEOUT) ready(p.client, 2, deadline);
        else if (n < 0) {
            int detail = eshkol_unix_stream_socket_last_os_error();
            CHECK(n == ESHKOL_IO_RESET || (n == ESHKOL_IO_SYSTEM && (detail == EPIPE || detail == ECONNRESET)));
            observed = 1;
        } else CHECK(n > 0 && n <= (int64_t)sizeof(bytes));
    }
    CHECK(observed); close_socket(p.client); close_socket(p.listener);
}
static void scenario_peer_close_epipe(void) {
    pid_t child = fork(); CHECK(child >= 0);
    if (!child) { loop_handle = eshkol_event_loop_open_handle(64); CHECK(loop_handle > 0); peer_close_child(); finish_loop(); _exit(0); }
    int status; CHECK(waitpid(child, &status, 0) == child); CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void scenario_descriptor_count_check(void) {
    pair_close(pair_open()); /* warm lazy common bookkeeping before baseline */
    int before = descriptor_count();
    for (int i = 0; i < 1000; ++i) pair_close(pair_open());
    int after = descriptor_count(); CHECK(after == before);
    fprintf(stderr, "MEASURE: fd_before=%d fd_after=%d cycles=1000\n", before, after);
}
static void scenario_capability_denial(void) {
    int before = descriptor_count(); atomic_store(&network_allowed, 0); atomic_store(&policy_calls, 0);
    CHECK(eshkol_unix_stream_socket_listen(loop_handle, path) == ESHKOL_IO_SYSTEM);
    int detail = eshkol_unix_stream_socket_last_os_error(); CHECK(detail == EACCES || detail == EPERM);
    CHECK(atomic_load(&policy_calls) > 0); CHECK(descriptor_count() == before);
    struct stat st; CHECK(lstat(path, &st) < 0 && errno == ENOENT); atomic_store(&network_allowed, 1);
}
static void scenario_capability_revocation(void) {
    pair_t p = pair_open(); atomic_store(&network_allowed, 0);
    char byte; CHECK(eshkol_unix_stream_socket_read(p.peer, &byte, 1) == ESHKOL_IO_SYSTEM);
    int detail = eshkol_unix_stream_socket_last_os_error(); CHECK(detail == EACCES || detail == EPERM);
    CHECK(eshkol_unix_stream_socket_cancel(p.client) == 0); CHECK(eshkol_unix_stream_socket_cancel(p.client) == 0);
    pair_close(p); CHECK(eshkol_unix_stream_socket_close(loop_handle) == ESHKOL_IO_INVALID);
    atomic_store(&network_allowed, 1);
}
static void scenario_path_lifecycle(void) {
    int64_t id = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(id > 0);
    struct stat st; CHECK(lstat(path, &st) == 0 && S_ISSOCK(st.st_mode) && (st.st_mode & 0777) == 0600);
    close_socket(id); CHECK(lstat(path, &st) < 0 && errno == ENOENT);
    id = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(id > 0); close_socket(id);
    int fd = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(fd >= 0);
    struct sockaddr_un a; memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX; strcpy(a.sun_path, path);
    CHECK(bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0); CHECK(close(fd) == 0);
    CHECK(lstat(path, &st) == 0); ino_t inode = st.st_ino;
    CHECK(eshkol_unix_stream_socket_listen(loop_handle, path) == ESHKOL_IO_SYSTEM);
    CHECK(eshkol_unix_stream_socket_last_os_error() == EEXIST);
    CHECK(lstat(path, &st) == 0 && st.st_ino == inode); CHECK(unlink(path) == 0);
    id = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(id > 0);
    CHECK(unlink(path) == 0); FILE *replacement = fopen(path, "wx"); CHECK(replacement != NULL); CHECK(fclose(replacement) == 0);
    CHECK(lstat(path, &st) == 0); inode = st.st_ino;
    int result = eshkol_unix_stream_socket_close(id);
    CHECK(result == 0 || result == ESHKOL_IO_SYSTEM);
    CHECK(lstat(path, &st) == 0 && S_ISREG(st.st_mode) && st.st_ino == inode);
    CHECK(eshkol_unix_stream_socket_close(id) == ESHKOL_IO_INVALID);
}
static void scenario_loop_lifetime(void) {
    pair_t p = pair_open();
    CHECK(eshkol_event_loop_close_handle(loop_handle) == ESHKOL_EVENT_LOOP_EBUSY);
    close_socket(p.listener); /* inherited peer retains remain */
    CHECK(eshkol_event_loop_close_handle(loop_handle) == ESHKOL_EVENT_LOOP_EBUSY);
    close_socket(p.client); close_socket(p.peer);
}
static void scenario_borrowed_close(void) {
    pair_t p = pair_open(); eshkol_resource_lease_t lease, other;
    CHECK(eshkol_resource_borrow(p.client, ESHKOL_HANDLE_KIND_SOCKET, 0, &lease) == 0);
    close_socket(p.client);
    CHECK(eshkol_resource_borrow(p.client, ESHKOL_HANDLE_KIND_SOCKET, 0, &other) == ESHKOL_IO_INVALID);
    eshkol_resource_release(&lease);
    CHECK(eshkol_unix_stream_socket_close(p.client) == ESHKOL_IO_INVALID);
    close_socket(p.peer); close_socket(p.listener);
}
static void scenario_zero_budget(void) {
    int64_t id = eshkol_unix_stream_socket_listen(loop_handle, path); CHECK(id > 0);
    int64_t start = now_ms(); CHECK(eshkol_unix_stream_socket_accept(id, 0) == ESHKOL_IO_TIMEOUT);
    CHECK(now_ms() - start < 150); close_socket(id);
    backlog_t b = full_backlog(1); int before = descriptor_count(); start = now_ms();
    CHECK(eshkol_unix_stream_socket_connect_begin(loop_handle, path, 0) == ESHKOL_IO_TIMEOUT);
    CHECK(now_ms() - start < 150); CHECK(descriptor_count() == before); backlog_close(b);
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s scenario [evidence-dir]\n", argv[0]); return 2; }
    if (!strcmp(argv[1], "--backlog-peer") || !strcmp(argv[1], "--leftover-peer")) {
        CHECK(argc == 3);
        CHECK(snprintf(path, sizeof(path), "%s/stream.sock", argv[2]) < (int)sizeof(path));
        struct stat st; CHECK(lstat(argv[2], &st) == 0 && S_ISDIR(st.st_mode));
        CHECK(st.st_uid == getuid() && (st.st_mode & 0777) == 0700);
        int leftover = !strcmp(argv[1], "--leftover-peer");
        backlog_t b = {.listener = -1, .count = 0};
        if (leftover) {
            int fd = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(fd >= 0);
            struct sockaddr_un a; memset(&a, 0, sizeof(a)); a.sun_family = AF_UNIX; strcpy(a.sun_path, path);
            CHECK(bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0); CHECK(close(fd) == 0);
        } else b = full_backlog(0);
        char ready_path[128], stop_path[128];
        CHECK(snprintf(ready_path, sizeof(ready_path), "%s/ready", argv[2]) < (int)sizeof(ready_path));
        CHECK(snprintf(stop_path, sizeof(stop_path), "%s/stop", argv[2]) < (int)sizeof(stop_path));
        FILE *ready_file = fopen(ready_path, "wx"); CHECK(ready_file != NULL); CHECK(fclose(ready_file) == 0);
        int64_t deadline = now_ms() + 30000;
        while (access(stop_path, F_OK) != 0 && now_ms() < deadline) nap(10);
        if (leftover) CHECK(unlink(path) == 0); else backlog_close(b);
        fprintf(stderr, "PASS: bounded raw peer\n"); return 0;
    }
    if (!strcmp(argv[1], "--peer-fixture-selftest")) {
        setup(); backlog_t b = full_backlog(0); backlog_close(b);
        /* The injected private path must not affect real connects elsewhere. */
        char real_path[104]; CHECK(snprintf(real_path, sizeof(real_path), "%s/real.sock", directory) < (int)sizeof(real_path));
        int server = socket(AF_UNIX, SOCK_STREAM, 0), client = socket(AF_UNIX, SOCK_STREAM, 0);
        CHECK(server >= 0 && client >= 0);
        struct sockaddr_un address; memset(&address, 0, sizeof(address)); address.sun_family = AF_UNIX; strcpy(address.sun_path, real_path);
        CHECK(bind(server, (struct sockaddr *)&address, sizeof(address)) == 0); CHECK(listen(server, 1) == 0);
        CHECK(connect(client, (struct sockaddr *)&address, sizeof(address)) == 0);
        int accepted = accept(server, NULL, NULL); CHECK(accepted >= 0);
        CHECK(close(accepted) == 0 && close(client) == 0 && close(server) == 0); CHECK(unlink(real_path) == 0);
        finish_loop();
        fprintf(stderr, "PASS: controlled pending-connect fixture (%u checks)\n", checks); return 0;
    }
    setup(); const char *name = argv[1];
#define CASE(n) if (!strcmp(name, #n)) scenario_##n(); else
    CASE(connect_accept) CASE(partial_slow_peer) CASE(peer_close_epipe)
    CASE(deadline_accept) CASE(deadline_connect) CASE(destination_errors)
    CASE(cancellation) CASE(connect_cancellation) CASE(nul_and_large_payload)
    CASE(descriptor_count_check) CASE(capability_denial) CASE(capability_revocation)
    CASE(path_lifecycle) CASE(loop_lifetime) CASE(borrowed_close) CASE(zero_budget)
    { fprintf(stderr, "unknown scenario: %s\n", name); return 2; }
#undef CASE
    finish_loop(); fprintf(stderr, "PASS: unix_socket_stream_test %s (%u checks)\n", name, checks); return 0;
}
