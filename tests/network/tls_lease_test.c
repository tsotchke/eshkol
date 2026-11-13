/* TLS over leased TCP streams against a local test authority (tls_test_ca):
 * client and server handshakes started on owned TCP handles, verification on
 * by default, untrusted and mismatched certificates refused with distinct
 * outcomes, the explicit insecure option, a >16 MiB round trip, cancellation,
 * and the session's exclusive lease on its TCP stream.
 * Usage: tls_lease_test <pki-directory> */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "eshkol/core/net.h"
#include "eshkol/core/tls.h"

static _Atomic int failures;
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); atomic_fetch_add(&failures, 1); }
}
static int64_t now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
typedef struct { int64_t length; uint8_t bytes[]; } Bytes;
static Bytes *bytes_new(int64_t length) {
    Bytes *b = (Bytes *)calloc(1, sizeof(Bytes) + (size_t)length);
    if (b) b->length = length;
    return b;
}
static char dir[2048], ca[2300], cert[2300], key[2300], rogue_cert[2300], rogue_key[2300];

/* Server: accept one TCP stream, start TLS on it, echo `echo_bytes` bytes back. */
typedef struct {
    int64_t listener, context, session, tcp;
    int64_t echo_bytes, echoed;
    int32_t start_result;
} Server;
static void *serve(void *opaque) {
    Server *s = (Server *)opaque;
    s->tcp = eshkol_net_accept(s->listener, 5000);
    if (s->tcp <= 0) { s->start_result = (int32_t)s->tcp; return NULL; }
    s->session = eshkol_tls_server_start(s->context, s->tcp, 5000);
    s->start_result = s->session > 0 ? 0 : (int32_t)s->session;
    if (s->session <= 0) return NULL;
    if (s->echo_bytes > 0) {
        Bytes *buf = bytes_new(s->echo_bytes);
        int64_t got = 0;
        while (got < s->echo_bytes) {
            int64_t want = s->echo_bytes - got > 1048576 ? 1048576 : s->echo_bytes - got;
            int32_t n = eshkol_tls_read_bytes(s->session, buf, got, want, 10000);
            if (n <= 0) break;
            got += n;
        }
        if (got == s->echo_bytes &&
            eshkol_tls_write_all_bytes(s->session, buf, 0, got, 20000) == got)
            s->echoed = got;
        free(buf);
    }
    Bytes *tail = bytes_new(1);
    (void)eshkol_tls_read_bytes(s->session, tail, 0, 1, 5000); /* wait for close-notify */
    free(tail);
    eshkol_tls_session_close(s->session);
    return NULL;
}
static int start_server(Server *s, const char *c, const char *k, int64_t echo, int *port) {
    memset(s, 0, sizeof(*s));
    s->context = eshkol_tls_server_context(c, k);
    s->listener = eshkol_net_open("127.0.0.1", 0, 1, 0);
    s->echo_bytes = echo;
    *port = s->listener > 0 ? eshkol_net_port(s->listener, 0) : -1;
    return s->context > 0 && s->listener > 0 && *port > 0;
}
static void stop_server(Server *s, pthread_t thread) {
    pthread_join(thread, NULL);
    eshkol_net_close(s->listener);
    eshkol_tls_context_close(s->context);
}
static int64_t client_session(int64_t context, int port, const char *name, int64_t *tcp_out) {
    int64_t tcp = eshkol_net_open("127.0.0.1", port, 2, 2000);
    if (tcp_out) *tcp_out = tcp;
    if (tcp <= 0) return tcp;
    return eshkol_tls_client_start(context, tcp, name, 5000);
}

static void verified_roundtrip_and_lease(void) {
    Server s; int port; pthread_t thread;
    check(start_server(&s, cert, key, 17 * 1024 * 1024 + 123, &port), "start TLS server");
    pthread_create(&thread, NULL, serve, &s);
    int64_t context = eshkol_tls_client_context(ca);
    int64_t tcp = 0;
    int64_t session = client_session(context, port, "localhost", &tcp);
    check(session > 0, "client handshake on an owned TCP stream verifies the server");
    Bytes *probe = bytes_new(4);
    check(eshkol_net_read_bytes(tcp, probe, 0, 4, 0) == -5,
          "the session holds the TCP stream lease exclusively");
    const int64_t total = 17 * 1024 * 1024 + 123;
    Bytes *out = bytes_new(total), *in = bytes_new(total);
    for (int64_t i = 0; i < total; i++) out->bytes[i] = (uint8_t)((i * 2654435761u) >> 13);
    int64_t started = now_ms();
    check(eshkol_tls_write_all_bytes(session, out, 0, total, 30000) == total,
          "write a >16 MiB payload under one deadline");
    int64_t got = 0;
    while (got < total) {
        int64_t want = total - got > 1048576 ? 1048576 : total - got;
        int32_t n = eshkol_tls_read_bytes(session, in, got, want, 30000);
        if (n <= 0) break;
        got += n;
    }
    check(got == total && memcmp(in->bytes, out->bytes, (size_t)total) == 0,
          ">16 MiB TLS round trip preserves every byte");
    check(now_ms() - started < 60000, "round trip completes within its deadline");
    check(eshkol_tls_session_close(session) == 0, "close the client session");
    check(eshkol_net_close(tcp) == -4, "closing the session closed its TCP stream");
    stop_server(&s, thread);
    check(s.start_result == 0 && s.echoed == total, "server handshake on an owned TCP stream echoes");
    eshkol_tls_context_close(context);
    free(out); free(in); free(probe);
}
static void refusals(void) {
    Server s; int port; pthread_t thread;
    int64_t context = eshkol_tls_client_context(ca);
    /* A certificate from an unrelated authority is refused by default. */
    check(start_server(&s, rogue_cert, rogue_key, 0, &port), "start rogue server");
    pthread_create(&thread, NULL, serve, &s);
    int64_t tcp = 0;
    int64_t session = client_session(context, port, "localhost", &tcp);
    check(session == -10, "an untrusted certificate is refused as untrusted");
    check(eshkol_net_close(tcp) == -4, "a failed handshake closes the stream it consumed");
    stop_server(&s, thread);
    /* A trusted certificate for another name is refused as an identity mismatch. */
    check(start_server(&s, cert, key, 0, &port), "start server for mismatch");
    pthread_create(&thread, NULL, serve, &s);
    session = client_session(context, port, "other.example", NULL);
    check(session == -11, "a hostname mismatch is refused as an identity mismatch");
    stop_server(&s, thread);
    check(start_server(&s, cert, key, 0, &port), "start server for IP identity");
    pthread_create(&thread, NULL, serve, &s);
    session = client_session(context, port, "127.0.0.1", NULL);
    check(session > 0, "the IP address SAN verifies");
    if (session > 0) eshkol_tls_session_close(session);
    stop_server(&s, thread);
    /* Only the explicit option accepts an unverified peer. */
    int64_t insecure = eshkol_tls_client_context_options(ca, ESHKOL_TLS_INSECURE_NO_VERIFY);
    check(insecure > 0, "explicit insecure client context");
    check(eshkol_tls_client_context_options(ca, 0x8) == -4, "unknown options are refused");
    check(start_server(&s, rogue_cert, rogue_key, 0, &port), "start rogue server again");
    pthread_create(&thread, NULL, serve, &s);
    session = client_session(insecure, port, "localhost", NULL);
    check(session > 0, "the insecure option accepts an unverified peer");
    if (session > 0) eshkol_tls_session_close(session);
    stop_server(&s, thread);
    eshkol_tls_context_close(insecure);
    eshkol_tls_context_close(context);
}
typedef struct { int64_t session; int32_t result; int64_t elapsed; } Reader;
static void *blocked_read(void *opaque) {
    Reader *r = (Reader *)opaque;
    Bytes *b = bytes_new(8);
    int64_t started = now_ms();
    r->result = eshkol_tls_read_bytes(r->session, b, 0, 8, 20000);
    r->elapsed = now_ms() - started;
    free(b);
    return NULL;
}
static void cancellation(void) {
    Server s; int port; pthread_t thread, reader_thread;
    check(start_server(&s, cert, key, 0, &port), "start server for cancellation");
    pthread_create(&thread, NULL, serve, &s);
    int64_t context = eshkol_tls_client_context(ca);
    int64_t session = client_session(context, port, "localhost", NULL);
    check(session > 0, "session for cancellation");
    Reader r = {session, 0, 0};
    pthread_create(&reader_thread, NULL, blocked_read, &r);
    struct timespec pause = {0, 100000000L};
    nanosleep(&pause, NULL);
    check(eshkol_tls_cancel(session) == 0, "cancel a blocked TLS read");
    pthread_join(reader_thread, NULL);
    check(r.result == -6 && r.elapsed < 2000, "the TLS read reports cancellation promptly");
    check(eshkol_tls_session_close(session) == 0, "close after cancellation");
    stop_server(&s, thread);
    eshkol_tls_context_close(context);
}
int main(int argc, char **argv) {
    if (!eshkol_tls_available()) {
        check(eshkol_tls_client_context(NULL) == -3, "unavailable TLS refuses explicitly");
        puts(atomic_load(&failures) ? "RESULT: FAIL" : "RESULT: PASS");
        return atomic_load(&failures) != 0;
    }
    if (argc != 2) { fprintf(stderr, "usage: tls_lease_test <pki-directory>\n"); return 2; }
    snprintf(dir, sizeof(dir), "%s", argv[1]);
    snprintf(ca, sizeof(ca), "%s/ca.pem", dir);
    snprintf(cert, sizeof(cert), "%s/server-cert.pem", dir);
    snprintf(key, sizeof(key), "%s/server-key.pem", dir);
    snprintf(rogue_cert, sizeof(rogue_cert), "%s/rogue-cert.pem", dir);
    snprintf(rogue_key, sizeof(rogue_key), "%s/rogue-key.pem", dir);
    verified_roundtrip_and_lease();
    refusals();
    cancellation();
    int n = atomic_load(&failures);
    if (n) { fprintf(stderr, "RESULT: FAIL (%d failures)\n", n); return 1; }
    puts("PASS: leased TLS handshakes, verification refusals, >16 MiB round trip, cancellation");
    puts("RESULT: PASS");
    return 0;
}
