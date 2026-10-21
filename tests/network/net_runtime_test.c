/* Independent POSIX peers exercise the public C runtime boundary. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "eshkol/core/net.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32) || defined(ESHKOL_VM_WASM) || defined(__EMSCRIPTEN__)
int main(void) {
    assert(eshkol_net_open("127.0.0.1", 0, 1, 0) == -3);
    assert(eshkol_net_accept(1, 0) == -3);
    assert(eshkol_net_close(1) == -3);
    puts("PASS: unsupported network backend refuses");
}
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>
static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
int main(void) {
    const char payload[] = {'a', 0, 'b', (char)255};
    char buf[128], peer[64];
    int64_t listener = eshkol_net_open("127.0.0.1", 0, 1, 0);
    assert(listener > 0);
    int port = eshkol_net_port(listener, 0); assert(port > 0);
    double start = seconds();
    assert(eshkol_net_accept(listener, 30) == -2);
    double elapsed = seconds() - start;
    assert(elapsed >= .015 && elapsed < 2);
    int raw = socket(AF_INET, SOCK_STREAM, 0); assert(raw >= 0);
    struct sockaddr_in a; memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET; assert(inet_pton(AF_INET, "127.0.0.1", &a.sin_addr) == 1);
    a.sin_port = htons((uint16_t)port);
    assert(connect(raw, (struct sockaddr *)&a, sizeof(a)) == 0);
    int64_t accepted = eshkol_net_accept(listener, 1000); assert(accepted > 0);
    assert(send(raw, payload, sizeof(payload), 0) == sizeof(payload));
    assert(eshkol_net_read(accepted, buf, sizeof(buf), 1000) == sizeof(payload));
    assert(memcmp(buf, payload, sizeof(payload)) == 0);
    assert(eshkol_net_write(accepted, "ok", 2, 1000) == 2);
    assert(recv(raw, buf, sizeof(buf), 0) == 2 && memcmp(buf, "ok", 2) == 0);
    assert(eshkol_net_nonblocking(accepted, 1) == 0);
    start = seconds();
    assert(eshkol_net_read(accepted, buf, sizeof(buf), 10000) == -2);
    assert(seconds() - start < 1);
    assert(eshkol_net_read(accepted, buf, 0, 0) == -4);
    assert(eshkol_net_nonblocking(accepted, 0) == 0);
    close(raw);
    assert(eshkol_net_read(accepted, buf, sizeof(buf), 1000) == 0);
    assert(eshkol_net_close(accepted) == 0);
    assert(eshkol_net_close(listener) == 0);
    int64_t replacement = eshkol_net_open("127.0.0.1", 0, 1, 0); assert(replacement > 0);
    assert(replacement != listener && eshkol_net_close(listener) == -4);
    assert(eshkol_net_port(replacement, 0) > 0);
    assert(eshkol_net_close(replacement) == 0);

    int64_t udp = eshkol_net_open("127.0.0.1", 0, 3, 0); assert(udp > 0);
    raw = socket(AF_INET, SOCK_DGRAM, 0); assert(raw >= 0);
    a.sin_port = 0; assert(bind(raw, (struct sockaddr *)&a, sizeof(a)) == 0);
    socklen_t len = sizeof(a); assert(getsockname(raw, (struct sockaddr *)&a, &len) == 0);
    int raw_port = ntohs(a.sin_port);
    a.sin_port = htons((uint16_t)eshkol_net_port(udp, 0));
    assert(sendto(raw, payload, sizeof(payload), 0, (struct sockaddr *)&a, sizeof(a)) == sizeof(payload));
    assert(eshkol_net_recvfrom(udp, buf, sizeof(buf), peer, sizeof(peer), 1000) == sizeof(payload));
    assert(memcmp(buf, payload, sizeof(payload)) == 0);
    char expected[64]; snprintf(expected, sizeof(expected), "127.0.0.1:%d", raw_port);
    assert(strcmp(peer, expected) == 0);
    assert(eshkol_net_sendto(udp, "127.0.0.1", raw_port, payload, sizeof(payload), 1000) == sizeof(payload));
    assert(recv(raw, buf, sizeof(buf), 0) == sizeof(payload));
    assert(memcmp(buf, payload, sizeof(payload)) == 0);
    assert(sendto(raw, payload, sizeof(payload), 0, (struct sockaddr *)&a, sizeof(a)) == sizeof(payload));
    assert(eshkol_net_recvfrom(udp, buf, 1, peer, sizeof(peer), 1000) == -4);
    assert(eshkol_net_recvfrom(udp, buf, sizeof(buf), peer, sizeof(peer), 0) == -2);
    close(raw); assert(eshkol_net_close(udp) == 0);
    for (int i = 0; i < 1000; i++) {
        int64_t h = eshkol_net_open("127.0.0.1", 0, 3, 0);
        assert(h > 0 && eshkol_net_close(h) == 0);
    }
    int64_t v6 = eshkol_net_open("::1", 0, 1, 0); assert(v6 > 0);
    int64_t v6client = eshkol_net_open("::1", eshkol_net_port(v6, 0), 2, 1000); assert(v6client > 0);
    int64_t v6server = eshkol_net_accept(v6, 1000); assert(v6server > 0);
    assert(eshkol_net_address(v6server, 1, buf, sizeof(buf)) == 3 && strcmp(buf, "::1") == 0);
    assert(eshkol_net_write(v6client, payload, sizeof(payload), 1000) == sizeof(payload));
    assert(eshkol_net_read(v6server, buf, sizeof(buf), 1000) == sizeof(payload));
    assert(memcmp(buf, payload, sizeof(payload)) == 0);
    assert(eshkol_net_close(v6client) == 0 && eshkol_net_close(v6server) == 0 && eshkol_net_close(v6) == 0);
    assert(eshkol_net_open("host.invalid", 0, 1, 0) == -4);
    assert(eshkol_net_open("127.0.0.1", 65536, 3, 0) == -4);
    puts("PASS: independent TCP/UDP runtime peers");
}
#endif
