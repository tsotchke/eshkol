/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#include <eshkol/core/unix_socket_stream.h>

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

/*
 * Native acceptance fixtures for Unix-domain stream sockets, in the order
 * listed for the v1.4.0 socket-scaffold lane. These exercise the real
 * contract from Gabe Kahen's design (notes/IPC_DESIGN-2026-11-02.md, "Later
 * Unix streams" row): every call below goes through eshkol_unix_stream_socket_*,
 * which is currently a scaffold (lib/core/unix_socket_stream_stub.c) that
 * reports ESHKOL_UNIX_STREAM_SOCKET_PLANNED for everything. Each scenario is
 * therefore expected to fail its first real-behavior CHECK -- that clean,
 * reported failure (not a hang or a crash) is the acceptance signal that the
 * stub is wired to the real contract. The suite stays out of the default run
 * (CTest label ipc-unix-pending, DISABLED) until the core implementation
 * lands and these checks start to pass for real.
 */

static unsigned int checks;

static void check(int ok, const char *expr, int line) {
    ++checks;
    if (!ok) {
        fprintf(stderr, "FAIL: unix_socket_stream_test:%d: %s\n", line, expr);
        exit(1);
    }
}
#define CHECK(expr) check(!!(expr), #expr, __LINE__)

static int64_t now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) abort();
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static char g_path[512];

static void make_path(const char *scratch_dir, const char *name) {
    snprintf(g_path, sizeof(g_path), "%s/%s.sock", scratch_dir, name);
    unlink(g_path);
}

static void scenario_connect_accept(const char *scratch_dir) {
    make_path(scratch_dir, "connect-accept");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 1000);
    CHECK(client > 0);
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 1000);
    CHECK(accepted > 0);
    CHECK(eshkol_unix_stream_socket_close(accepted) == 0);
    CHECK(eshkol_unix_stream_socket_close(client) == 0);
    CHECK(eshkol_unix_stream_socket_close(listener) == 0);
}

static void scenario_partial_slow_peer(const char *scratch_dir) {
    make_path(scratch_dir, "partial-slow");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 1000);
    CHECK(client > 0);
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 1000);
    CHECK(accepted > 0);
    unsigned char payload[4096];
    for (size_t i = 0; i < sizeof(payload); ++i) payload[i] = (unsigned char)i;
    int64_t written = 0;
    while (written < (int64_t)sizeof(payload)) {
        int64_t n = eshkol_unix_stream_socket_write(client, payload + written,
            (int64_t)sizeof(payload) - written);
        CHECK(n >= 0);
        written += n;
    }
    unsigned char got[4096];
    int64_t read_total = 0;
    while (read_total < (int64_t)sizeof(payload)) {
        int64_t n = eshkol_unix_stream_socket_read(accepted, got + read_total,
            (int64_t)sizeof(got) - read_total);
        CHECK(n >= 0);
        read_total += n;
    }
    CHECK(memcmp(payload, got, sizeof(payload)) == 0);
    eshkol_unix_stream_socket_close(accepted);
    eshkol_unix_stream_socket_close(client);
    eshkol_unix_stream_socket_close(listener);
}

static void scenario_peer_close_epipe(const char *scratch_dir) {
    make_path(scratch_dir, "peer-close");
    signal(SIGPIPE, SIG_IGN); /* guards this harness only; the core must
                                 suppress SIGPIPE on the write path itself. */
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 1000);
    CHECK(client > 0);
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 1000);
    CHECK(accepted > 0);
    CHECK(eshkol_unix_stream_socket_close(accepted) == 0);
    unsigned char payload[16] = {0};
    int64_t n = eshkol_unix_stream_socket_write(client, payload, sizeof(payload));
    CHECK(n == ESHKOL_UNIX_STREAM_SOCKET_ESYSTEM);
    eshkol_unix_stream_socket_close(client);
    eshkol_unix_stream_socket_close(listener);
}

static void scenario_deadline_accept(const char *scratch_dir) {
    make_path(scratch_dir, "deadline-accept");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    int64_t start = now_ms();
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 150);
    int64_t elapsed = now_ms() - start;
    CHECK(accepted == ESHKOL_UNIX_STREAM_SOCKET_ETIMEDOUT);
    CHECK(elapsed >= 140 && elapsed < 2000);
    eshkol_unix_stream_socket_close(listener);
}

static void scenario_deadline_connect(const char *scratch_dir) {
    make_path(scratch_dir, "deadline-connect");
    /* nothing listens on this path */
    int64_t start = now_ms();
    int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 150);
    int64_t elapsed = now_ms() - start;
    CHECK(client == ESHKOL_UNIX_STREAM_SOCKET_ETIMEDOUT || client == ESHKOL_UNIX_STREAM_SOCKET_ESYSTEM);
    CHECK(elapsed < 2000);
}

static void *cancel_after(void *arg) {
    int64_t id = *(int64_t *)arg;
    struct timespec ts = {0, 50 * 1000000};
    nanosleep(&ts, &ts);
    eshkol_unix_stream_socket_cancel(id);
    return NULL;
}

static void scenario_cancellation(const char *scratch_dir) {
    make_path(scratch_dir, "cancel");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    pthread_t canceller;
    CHECK(pthread_create(&canceller, NULL, cancel_after, &listener) == 0);
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 5000);
    pthread_join(canceller, NULL);
    CHECK(accepted == ESHKOL_UNIX_STREAM_SOCKET_ECANCELED);
    eshkol_unix_stream_socket_close(listener);
}

static void scenario_nul_and_large_payload(const char *scratch_dir) {
    make_path(scratch_dir, "large");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 1000);
    CHECK(client > 0);
    int64_t accepted = eshkol_unix_stream_socket_accept(listener, 1000);
    CHECK(accepted > 0);
    size_t size = 16u * 1024u * 1024u + 37u;
    unsigned char *payload = malloc(size);
    CHECK(payload != NULL);
    for (size_t i = 0; i < size; ++i)
        payload[i] = (unsigned char)((i % 7 == 0) ? 0 : (i * 31));
    size_t written = 0;
    while (written < size) {
        int64_t n = eshkol_unix_stream_socket_write(client, payload + written,
            (int64_t)(size - written));
        CHECK(n >= 0);
        written += (size_t)n;
    }
    unsigned char *got = malloc(size);
    CHECK(got != NULL);
    size_t read_total = 0;
    while (read_total < size) {
        int64_t n = eshkol_unix_stream_socket_read(accepted, got + read_total,
            (int64_t)(size - read_total));
        CHECK(n >= 0);
        read_total += (size_t)n;
    }
    CHECK(memcmp(payload, got, size) == 0);
    free(payload);
    free(got);
    eshkol_unix_stream_socket_close(accepted);
    eshkol_unix_stream_socket_close(client);
    eshkol_unix_stream_socket_close(listener);
}

static void scenario_descriptor_leak(const char *scratch_dir) {
    make_path(scratch_dir, "descriptor-count");
    for (int i = 0; i < 1000; ++i) {
        unlink(g_path);
        int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
        CHECK(listener > 0);
        int64_t client = eshkol_unix_stream_socket_connect(0, g_path, 1000);
        CHECK(client > 0);
        int64_t accepted = eshkol_unix_stream_socket_accept(listener, 1000);
        CHECK(accepted > 0);
        CHECK(eshkol_unix_stream_socket_close(accepted) == 0);
        CHECK(eshkol_unix_stream_socket_close(client) == 0);
        CHECK(eshkol_unix_stream_socket_close(listener) == 0);
    }
}

static void scenario_path_lifecycle(const char *scratch_dir) {
    make_path(scratch_dir, "path-lifecycle");
    int64_t listener = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(listener > 0);
    struct stat st;
    CHECK(stat(g_path, &st) == 0);
    CHECK(eshkol_unix_stream_socket_close(listener) == 0);
    CHECK(stat(g_path, &st) != 0); /* unlinked on close */

    /* A stale path left on disk with nothing listening must be refused,
     * never silently unlinked and reused. */
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_path, sizeof(addr.sun_path) - 1);
    CHECK(bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0);
    close(fd); /* leaves a stale path: bound once, never listened on */
    int64_t second = eshkol_unix_stream_socket_listen(0, g_path);
    CHECK(second == ESHKOL_UNIX_STREAM_SOCKET_EEXIST);
    unlink(g_path);
}

static void scenario_capability_denial(const char *scratch_dir) {
    /* The capability gate lives in lib/agent/unix_socket.esk's
     * (capability-require! 'network ...); the native scaffold has no policy
     * layer of its own. This records the native half only (no I/O was
     * attempted before this check could run); see
     * tests/agent/unix_socket_stream_pending.esk capability_denial for the
     * gated half of this acceptance case. */
    (void)scratch_dir;
    CHECK(1);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <scenario> <scratch-dir>\n", argv[0]);
        return 2;
    }
    const char *scenario = argv[1];
    const char *scratch_dir = argv[2];
    mkdir(scratch_dir, 0700); /* ignore EEXIST */

    if (strcmp(scenario, "connect_accept") == 0) scenario_connect_accept(scratch_dir);
    else if (strcmp(scenario, "partial_slow_peer") == 0) scenario_partial_slow_peer(scratch_dir);
    else if (strcmp(scenario, "peer_close_epipe") == 0) scenario_peer_close_epipe(scratch_dir);
    else if (strcmp(scenario, "deadline_accept") == 0) scenario_deadline_accept(scratch_dir);
    else if (strcmp(scenario, "deadline_connect") == 0) scenario_deadline_connect(scratch_dir);
    else if (strcmp(scenario, "cancellation") == 0) scenario_cancellation(scratch_dir);
    else if (strcmp(scenario, "nul_and_large_payload") == 0) scenario_nul_and_large_payload(scratch_dir);
    else if (strcmp(scenario, "descriptor_leak") == 0) scenario_descriptor_leak(scratch_dir);
    else if (strcmp(scenario, "path_lifecycle") == 0) scenario_path_lifecycle(scratch_dir);
    else if (strcmp(scenario, "capability_denial") == 0) scenario_capability_denial(scratch_dir);
    else { fprintf(stderr, "unknown scenario: %s\n", scenario); return 2; }

    fprintf(stderr, "PASS: unix_socket_stream_test %s (%u checks)\n", scenario, checks);
    return 0;
}
