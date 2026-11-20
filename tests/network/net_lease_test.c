/* TCP/UDP on the shared resource lease layer: partial writes against a slow
 * peer, peer reset, deadline expiry, cancellation, busy leases, shutdown,
 * kind-checked close, datagram truncation, and a constant descriptor count
 * across 1,000 connection cycles. Loopback only. */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#if defined(_WIN32) || defined(__EMSCRIPTEN__)
#include <stdio.h>
#include "eshkol/core/net.h"
int main(void) {
    if (eshkol_net_open("127.0.0.1", 0, 1, 0) != -3) { puts("FAIL: unsupported refusal"); return 1; }
    puts("PASS: socket lease layer refuses on this platform");
    puts("RESULT: PASS");
    return 0;
}
#else
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "eshkol/core/net.h"

static _Atomic int failures;
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); atomic_fetch_add(&failures, 1); }
}
static int64_t now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static void sleep_ms(int ms) {
    struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L};
    while (nanosleep(&t, &t) != 0) {}
}
/* The Eshkol bytevector layout: an exact length word followed by its bytes. */
typedef struct { int64_t length; uint8_t bytes[]; } Bytes;
static Bytes *bytes_new(int64_t length) {
    Bytes *b = (Bytes *)calloc(1, sizeof(Bytes) + (size_t)length);
    if (b) b->length = length;
    return b;
}
static int count_descriptors(void) {
#ifdef __linux__
    const char *dir = "/proc/self/fd";
#else
    const char *dir = "/dev/fd";
#endif
    DIR *d = opendir(dir);
    if (!d) return -1;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) if (e->d_name[0] != '.') n++;
    closedir(d);
    return n;
}
typedef struct { int64_t listener, client, server; int port; } Pair;
static int make_pair(Pair *p) {
    p->listener = eshkol_net_open("127.0.0.1", 0, 1, 0);
    p->port = p->listener > 0 ? eshkol_net_port(p->listener, 0) : -1;
    p->client = p->port > 0 ? eshkol_net_open("127.0.0.1", p->port, 2, 1000) : -1;
    p->server = p->client > 0 ? eshkol_net_accept(p->listener, 1000) : -1;
    return p->listener > 0 && p->client > 0 && p->server > 0;
}
static void close_pair(Pair *p) {
    if (p->client > 0) eshkol_net_close_kind(p->client, 1);
    if (p->server > 0) eshkol_net_close_kind(p->server, 1);
    if (p->listener > 0) eshkol_net_close_kind(p->listener, 1);
}

typedef struct { int64_t handle; Bytes *data; int64_t result; } Writer;
static void *write_all_worker(void *opaque) {
    Writer *w = (Writer *)opaque;
    w->result = eshkol_net_write_all_bytes(w->handle, w->data, 0, w->data->length, 30000);
    return NULL;
}
static void partial_writes_with_slow_peer(void) {
    Pair p; check(make_pair(&p), "pair for partial writes");
    const int64_t total = 8 * 1024 * 1024;
    Bytes *out = bytes_new(total), *in = bytes_new(total);
    check(out && in, "allocate payloads");
    if (!out || !in) return;
    for (int64_t i = 0; i < total; i++) out->bytes[i] = (uint8_t)(i * 131 + (i >> 9));
    /* The peer reads nothing yet: one call accepts only what the kernel buffers. */
    int32_t first = eshkol_net_write_bytes(p.client, out, 0, total, 1000);
    check(first > 0 && first < total, "a large write to a slow peer is partial");
    /* Keep offering bytes with a zero deadline until both kernel buffers are
     * full; that pass must report would-block without waiting. */
    int64_t sent = first;
    int32_t step = 0;
    int64_t started = 0;
    for (int i = 0; i < 4096 && sent < total; i++) {
        started = now_ms();
        step = eshkol_net_write_bytes(p.client, out, sent, total - sent, 0);
        if (step < 0) break;
        sent += step;
    }
    check(step == -2, "a zero deadline on a full stream reports would-block");
    check(now_ms() - started < 200, "would-block returns without waiting");
    check(sent < total, "the slow peer holds back the remainder");
    first = (int32_t)sent;
    /* The peer now drains slowly while the remainder goes out under one deadline. */
    Writer w = {p.client, out, 0};
    Bytes *rest = bytes_new(total - first);
    memcpy(rest->bytes, out->bytes + first, (size_t)(total - first));
    w.data = rest;
    pthread_t thread;
    check(pthread_create(&thread, NULL, write_all_worker, &w) == 0, "start writer");
    int64_t got = 0;
    int reads = 0;
    while (got < total) {
        int32_t chunk = got < 256 * 1024 ? 4096 : 262144;
        if (chunk > total - got) chunk = (int32_t)(total - got);
        int32_t n = eshkol_net_read_bytes(p.server, in, got, chunk, 5000);
        if (n <= 0) { check(0, "slow reader receives every byte"); break; }
        got += n; reads++;
        if (reads < 16) sleep_ms(5);
    }
    pthread_join(thread, NULL);
    check(w.result == total - first, "write-all completes the remainder under one deadline");
    check(got == total && memcmp(in->bytes, out->bytes, (size_t)total) == 0,
          "slow-peer stream preserves every byte in order");
    free(out); free(in); free(rest);
    close_pair(&p);
}
static void peer_reset(void) {
    Pair p; check(make_pair(&p), "pair for reset");
    Bytes *b = bytes_new(16);
    memcpy(b->bytes, "unread", 6);
    check(eshkol_net_write_bytes(p.client, b, 0, 6, 1000) == 6, "send bytes the peer never reads");
    sleep_ms(50);
    /* Closing with unread received data sends a reset to the peer. */
    check(eshkol_net_close_kind(p.server, 1) == 0, "close peer with unread data");
    p.server = 0;
    int32_t r = eshkol_net_read_bytes(p.client, b, 0, 16, 2000);
    check(r == -7, "read after peer reset reports reset");
    int32_t w = 0;
    for (int i = 0; i < 4 && w >= 0; i++) w = eshkol_net_write_bytes(p.client, b, 0, 6, 1000);
    check(w == -7, "write after peer reset reports reset");
    free(b);
    close_pair(&p);
}
static void deadlines(void) {
    Pair p; check(make_pair(&p), "pair for deadlines");
    Bytes *b = bytes_new(8);
    int64_t started = now_ms();
    check(eshkol_net_read_bytes(p.server, b, 0, 8, 120) == -2, "idle read reaches its deadline");
    int64_t elapsed = now_ms() - started;
    check(elapsed >= 110 && elapsed < 1000, "read deadline is honoured, not extended");
    started = now_ms();
    check(eshkol_net_accept(p.listener, 80) == -2, "idle accept reaches its deadline");
    elapsed = now_ms() - started;
    check(elapsed >= 70 && elapsed < 1000, "accept deadline is honoured");
    free(b);
    close_pair(&p);

    /* Connect deadline: a raw listener whose accept queue is full drops SYNs. */
    int raw = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET; inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
    socklen_t len = sizeof(a);
    check(raw >= 0 && bind(raw, (struct sockaddr *)&a, sizeof(a)) == 0 && listen(raw, 1) == 0 &&
          getsockname(raw, (struct sockaddr *)&a, &len) == 0, "raw listener with a minimal queue");
    int port = ntohs(a.sin_port);
    int64_t opened[64]; int count = 0, timed_out = 0;
    for (; count < 64 && !timed_out; count++) {
        started = now_ms();
        opened[count] = eshkol_net_open("127.0.0.1", port, 2, 300);
        elapsed = now_ms() - started;
        if (opened[count] == -2) {
            timed_out = 1;
            check(elapsed >= 280 && elapsed < 2000, "connect deadline is honoured");
        }
    }
    check(timed_out, "connect to a saturated listener reaches its deadline");
    for (int i = 0; i < count; i++) if (opened[i] > 0) eshkol_net_close(opened[i]);
    close(raw);
}
typedef struct { int64_t handle; int op; int64_t result; int64_t elapsed; } Blocked;
static void *blocked_worker(void *opaque) {
    Blocked *b = (Blocked *)opaque;
    int64_t started = now_ms();
    Bytes *buf = bytes_new(8);
    if (b->op == 0) b->result = eshkol_net_accept(b->handle, 20000);
    else b->result = eshkol_net_read_bytes(b->handle, buf, 0, 8, 20000);
    b->elapsed = now_ms() - started;
    free(buf);
    return NULL;
}
static void cancellation_and_busy(void) {
    Pair p; check(make_pair(&p), "pair for cancellation");
    Blocked accepter = {p.listener, 0, 0, 0};
    pthread_t thread;
    check(pthread_create(&thread, NULL, blocked_worker, &accepter) == 0, "start blocked accept");
    sleep_ms(100);
    check(eshkol_net_accept(p.listener, 0) == -5, "a second operation on a leased handle is busy");
    check(eshkol_net_cancel(p.listener) == 0, "cancel from another thread");
    pthread_join(thread, NULL);
    check(accepter.result == -6, "cancelled accept reports cancellation");
    check(accepter.elapsed < 2000, "cancellation wakes the waiter promptly");
    check(eshkol_net_accept(p.listener, 10) == -2, "the handle remains usable after cancellation");

    Blocked reader = {p.server, 1, 0, 0};
    check(pthread_create(&thread, NULL, blocked_worker, &reader) == 0, "start blocked read");
    sleep_ms(100);
    check(eshkol_net_cancel(p.server) == 0, "cancel a blocked read");
    pthread_join(thread, NULL);
    check(reader.result == -6 && reader.elapsed < 2000, "cancelled read returns promptly");

    /* With nothing waiting, cancellation applies to the next wait only. */
    Bytes *b = bytes_new(8);
    check(eshkol_net_cancel(p.server) == 0, "cancel with no operation in flight");
    check(eshkol_net_read_bytes(p.server, b, 0, 8, 5000) == -6, "next wait observes the cancellation");
    check(eshkol_net_read_bytes(p.server, b, 0, 8, 20) == -2, "cancellation is consumed once");
    free(b);
    close_pair(&p);
    check(eshkol_net_cancel(p.server) == -4, "cancel of a closed handle is invalid");
}
static void shutdown_and_kinds(void) {
    Pair p; check(make_pair(&p), "pair for shutdown");
    Bytes *b = bytes_new(8);
    memcpy(b->bytes, "abc", 3);
    check(eshkol_net_shutdown(p.client, 1) == 0, "shut down the write direction");
    check(eshkol_net_read_bytes(p.server, b, 0, 8, 1000) == 0, "peer reads end of stream");
    check(eshkol_net_write_bytes(p.server, b, 0, 3, 1000) == 3, "peer still writes back");
    check(eshkol_net_read_bytes(p.client, b, 3, 3, 1000) == 3, "half-closed side still reads");
    check(eshkol_net_shutdown(p.listener, 2) == -4, "shutdown refuses a listener");
    check(eshkol_net_shutdown(p.client, 3) == -4, "shutdown refuses an unknown direction");
    check(eshkol_net_read_bytes(p.client, b, 6, 3, 0) == -4, "range past the end refuses");
    check(eshkol_net_read_bytes(p.client, b, -1, 2, 0) == -4, "negative start refuses");
    check(eshkol_net_read_bytes(p.client, b, 0, 0, 0) == -4, "an empty read refuses");
    int64_t udp = eshkol_net_open("127.0.0.1", 0, 3, 0);
    check(eshkol_net_close_kind(udp, 1) == -4, "TCP close refuses a UDP handle");
    check(eshkol_net_close_kind(p.client, 2) == -4, "UDP close refuses a TCP handle");
    check(eshkol_net_close_kind(udp, 2) == 0, "UDP close accepts its own kind");
    check(eshkol_net_close_kind(udp, 2) == -4, "double close refuses");
    free(b);
    close_pair(&p);
}
static void datagrams(void) {
    int64_t a = eshkol_net_open("127.0.0.1", 0, 3, 0), b = eshkol_net_open("127.0.0.1", 0, 3, 0);
    check(a > 0 && b > 0, "bind datagram sockets");
    int pa = eshkol_net_port(a, 0), pb = eshkol_net_port(b, 0);
    Bytes *out = bytes_new(1200), *in = bytes_new(1200);
    for (int i = 0; i < 1200; i++) out->bytes[i] = (uint8_t)(i ^ 0x5a);
    char peer[64], expected[64];
    snprintf(expected, sizeof(expected), "127.0.0.1:%d", pa);
    check(eshkol_net_sendto_bytes(a, "127.0.0.1", pb, out, 100, 1000, 1000) == 1000, "datagram send");
    check(eshkol_net_recvfrom_bytes(b, in, 0, 1200, peer, sizeof(peer), 1000) == 1000,
          "datagram receive");
    check(memcmp(in->bytes, out->bytes + 100, 1000) == 0 && strcmp(peer, expected) == 0,
          "datagram payload and sender");
    check(eshkol_net_sendto_bytes(a, "127.0.0.1", pb, out, 0, 1200, 1000) == 1200, "large datagram");
    memset(in->bytes, 0, 1200);
    check(eshkol_net_recvfrom_bytes(b, in, 0, 16, peer, sizeof(peer), 1000) == -8,
          "a datagram beyond capacity reports truncation");
    check(memcmp(in->bytes, out->bytes, 16) == 0 && in->bytes[16] == 0,
          "truncation stores exactly the requested prefix");
    check(eshkol_net_recvfrom_bytes(b, in, 0, 16, peer, sizeof(peer), 0) == -2,
          "the truncated datagram was consumed");
    eshkol_net_close(a); eshkol_net_close(b);
    free(out); free(in);
}
static void descriptor_cycles(void) {
    int64_t listener = eshkol_net_open("127.0.0.1", 0, 1, 0);
    int port = eshkol_net_port(listener, 0);
    Bytes *b = bytes_new(4);
    memcpy(b->bytes, "ping", 4);
    /* Warm the per-thread wait loop before taking the baseline. */
    check(eshkol_net_accept(listener, 0) == -2, "warm wait loop");
    int before = count_descriptors();
    int ok = 1;
    for (int i = 0; i < 1000 && ok; i++) {
        int64_t c = eshkol_net_open("127.0.0.1", port, 2, 1000);
        int64_t s = eshkol_net_accept(listener, 1000);
        ok = c > 0 && s > 0 && eshkol_net_write_bytes(c, b, 0, 4, 1000) == 4 &&
             eshkol_net_read_bytes(s, b, 0, 4, 1000) == 4 &&
             eshkol_net_close(c) == 0 && eshkol_net_close(s) == 0;
    }
    check(ok, "1,000 connect/accept/transfer/close cycles");
    int after = count_descriptors();
    check(before > 0 && before == after, "descriptor count is unchanged after 1,000 cycles");
    eshkol_net_close(listener);
    check(count_descriptors() == before - 1, "closing the listener releases its descriptor");
    free(b);
}
int main(void) {
    partial_writes_with_slow_peer();
    peer_reset();
    deadlines();
    cancellation_and_busy();
    shutdown_and_kinds();
    datagrams();
    descriptor_cycles();
    int n = atomic_load(&failures);
    if (n) { fprintf(stderr, "RESULT: FAIL (%d failures)\n", n); return 1; }
    puts("PASS: partial writes, reset, deadlines, cancellation, shutdown, truncation, descriptor cycles");
    puts("RESULT: PASS");
    return 0;
}
#endif
