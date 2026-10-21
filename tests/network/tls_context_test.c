/* Focused public native TLS API checks. All peers bind loopback only. */
#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "eshkol/core/tls.h"
#include "eshkol/core/net.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if !ESHKOL_TLS_AVAILABLE
int main(void) {
    char payload = 'x';
    assert(eshkol_tls_available() == 0);
    assert(eshkol_tls_client_context(NULL) == -3);
    assert(eshkol_tls_server_context("cert.pem", "key.pem") == -3);
    assert(eshkol_tls_listen(1, "127.0.0.1", 0) == -3);
    assert(eshkol_tls_connect(1, "127.0.0.1", 443, "localhost", 10) == -3);
    assert(eshkol_tls_accept(1, 10) == -3);
    assert(eshkol_tls_read(1, &payload, 1, 10) == -3);
    assert(eshkol_tls_write(1, &payload, 1, 10) == -3);
    assert(eshkol_tls_session_close(1) == -3);
    assert(eshkol_tls_listener_close(1) == -3);
    puts("PASS: unavailable TLS fails explicitly");
    return 0;
}
#else

#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#ifndef ESHKOL_TEST_PYTHON
#define ESHKOL_TEST_PYTHON "python3"
#endif

typedef struct { int64_t context, listener, session; int32_t read_result; } Server;
typedef struct { int64_t context, listener, accepted; int delay_ms; } IdleServer;
typedef struct { int64_t session; int32_t result; int32_t timeout_ms; int started; } Reader;
static char tempdir[PATH_MAX];
static char cert_path[PATH_MAX + 64], key_path[PATH_MAX + 64];
static char wrong_cert_path[PATH_MAX + 64], wrong_key_path[PATH_MAX + 64];
static char expired_cert_path[PATH_MAX + 64], expired_key_path[PATH_MAX + 64];
static char encrypted_key_path[PATH_MAX + 64];

static EVP_PKEY *make_key(void) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
    EVP_PKEY *key = NULL;
    assert(ctx && EVP_PKEY_keygen_init(ctx) > 0);
    assert(EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1) > 0);
    assert(EVP_PKEY_keygen(ctx, &key) > 0);
    EVP_PKEY_CTX_free(ctx);
    return key;
}
static void make_cert_files(const char *cert_path, const char *key_path,
                            const char *common_name, int expired) {
    EVP_PKEY *key = make_key();
    X509 *cert = X509_new();
    assert(cert && X509_set_version(cert, 2) == 1);
    assert(ASN1_INTEGER_set(X509_get_serialNumber(cert), expired ? 3 : 1) == 1);
    assert(X509_gmtime_adj(X509_getm_notBefore(cert), -120));
    assert(X509_gmtime_adj(X509_getm_notAfter(cert), expired ? -60 : 3600));
    assert(X509_set_pubkey(cert, key) == 1);
    X509_NAME *name = X509_get_subject_name(cert);
    assert(X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
        (const unsigned char *)common_name, -1, -1, 0) == 1);
    assert(X509_set_issuer_name(cert, name) == 1);
    X509V3_CTX extension_context;
    X509V3_set_ctx(&extension_context, cert, cert, NULL, NULL, 0);
    X509_EXTENSION *extension = X509V3_EXT_conf_nid(NULL, &extension_context,
        NID_subject_alt_name, "DNS:localhost");
    assert(extension && X509_add_ext(cert, extension, -1) == 1);
    X509_EXTENSION_free(extension);
    extension = X509V3_EXT_conf_nid(NULL, &extension_context,
        NID_basic_constraints, "critical,CA:TRUE");
    assert(extension && X509_add_ext(cert, extension, -1) == 1);
    X509_EXTENSION_free(extension);
    assert(X509_sign(cert, key, EVP_sha256()) > 0);
    FILE *file = fopen(cert_path, "wb"); assert(file);
    assert(PEM_write_X509(file, cert) == 1); fclose(file);
    file = fopen(key_path, "wb"); assert(file);
    assert(PEM_write_PrivateKey(file, key, NULL, NULL, 0, NULL, NULL) == 1);
    fclose(file);
    X509_free(cert); EVP_PKEY_free(key);
}
static void *serve_echo(void *opaque) {
    Server *server = (Server *)opaque;
    server->session = eshkol_tls_accept(server->listener, 3000);
    assert(eshkol_tls_context_close(server->context) == 0);
    if (server->session < 0) return NULL;
    unsigned char input[32];
    server->read_result = eshkol_tls_read(server->session, input, sizeof(input), 2000);
    if (server->read_result > 0)
        (void)eshkol_tls_write(server->session, input, server->read_result, 2000);
    eshkol_tls_session_close(server->session);
    return NULL;
}
static int64_t monotonic_ms(void) {
    struct timespec ts; assert(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
static void delay_ms(int ms) {
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000};
    while (nanosleep(&ts, &ts) && errno == EINTR) {}
}
static void *serve_idle(void *opaque) {
    IdleServer *server = (IdleServer *)opaque;
    server->accepted = eshkol_tls_accept(server->listener, 3000);
    assert(eshkol_tls_context_close(server->context) == 0);
    if (server->accepted > 0) {
        delay_ms(server->delay_ms);
        eshkol_tls_session_close(server->accepted);
    }
    return NULL;
}
static void *stall_raw_peer(void *opaque) {
    IdleServer *server = (IdleServer *)opaque;
    server->accepted = eshkol_net_accept(server->listener, 1000);
    if (server->accepted > 0) {
        delay_ms(server->delay_ms);
        eshkol_net_close(server->accepted);
    }
    return NULL;
}
static void *read_blocked(void *opaque) {
    Reader *reader = (Reader *)opaque;
    unsigned char buffer[8];
    __atomic_store_n(&reader->started, 1, __ATOMIC_RELEASE);
    reader->result = eshkol_tls_read(reader->session, buffer, sizeof(buffer), reader->timeout_ms);
    return NULL;
}
static void raise_unrelated_openssl_error(void) {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
    ERR_raise(ERR_LIB_USER, ERR_R_INTERNAL_ERROR);
#else
    ERR_put_error(ERR_LIB_USER, 0, ERR_R_INTERNAL_ERROR, __FILE__, __LINE__);
#endif
}
static int64_t start_server(const char *cert, const char *key, Server *server,
                            pthread_t *thread) {
    int64_t context = eshkol_tls_server_context(cert, key);
    assert(context > 0);
    server->context = context;
    server->listener = eshkol_tls_listen(context, "127.0.0.1", 0);
    if (server->listener < 0) fprintf(stderr, "tls_listen failed: %lld\n", (long long)server->listener);
    assert(server->listener > 0);
    server->session = -1; server->read_result = -9;
    assert(pthread_create(thread, NULL, serve_echo, server) == 0);
    return eshkol_tls_listener_port(server->listener);
}
static void join_server(Server *server, pthread_t thread) {
    assert(pthread_join(thread, NULL) == 0);
    assert(eshkol_tls_listener_close(server->listener) == 0);
    assert(eshkol_tls_listener_close(server->listener) == -4);
}
static void trusted_binary_roundtrip(void) {
    int64_t client_context = eshkol_tls_client_context(cert_path);
    assert(client_context > 0);
    assert(eshkol_tls_listen(client_context, "127.0.0.1", 0) == -4);
    Server server; pthread_t thread;
    int64_t port = start_server(cert_path, key_path, &server, &thread);
    assert(eshkol_tls_connect(server.context, "127.0.0.1", (int32_t)port,
                              "localhost", 100) == -4);
    raise_unrelated_openssl_error();
    int64_t client = eshkol_tls_connect(client_context, "127.0.0.1", (int32_t)port,
                                        "localhost", 2000);
    if (client < 0) fprintf(stderr, "TLS connect failed: %lld\n", (long long)client);
    assert(client > 0);
    assert(eshkol_tls_context_close(client_context) == 0); /* session retains context */
    const unsigned char payload[] = {'a', 0, 'b', 255};
    raise_unrelated_openssl_error();
    assert(eshkol_tls_write(client, payload, sizeof(payload), 1000) == sizeof(payload));
    unsigned char received[16] = {0};
    raise_unrelated_openssl_error();
    assert(eshkol_tls_read(client, received, sizeof(received), 1000) == sizeof(payload));
    assert(memcmp(payload, received, sizeof(payload)) == 0);
    assert(eshkol_tls_read(client, received, sizeof(received), 1000) == 0);
    assert(eshkol_tls_write(client, payload, sizeof(payload), 1000) == -1);
    assert(eshkol_tls_session_close(client) == 0);
    assert(eshkol_tls_session_close(client) == -4);
    join_server(&server, thread);
    assert(server.read_result == sizeof(payload));
}
static void rejected_peer(const char *trust_cert, const char *server_cert,
                          const char *server_key, const char *identity) {
    int64_t client_context = eshkol_tls_client_context(trust_cert);
    assert(client_context > 0);
    Server server; pthread_t thread;
    int64_t port = start_server(server_cert, server_key, &server, &thread);
    int64_t client = eshkol_tls_connect(client_context, "127.0.0.1", (int32_t)port,
                                        identity, 1000);
    assert(client < 0);
    assert(eshkol_tls_context_close(client_context) == 0);
    join_server(&server, thread);
    assert(server.session == -1);
}
static void python_ssl_roundtrip(void) {
    Server server; pthread_t thread;
    int64_t port = start_server(cert_path, key_path, &server, &thread);
    pid_t child = fork(); assert(child >= 0);
    if (child == 0) {
        char port_text[24]; snprintf(port_text, sizeof(port_text), "%lld", (long long)port);
        const char *script =
            "import socket,ssl,sys\n"
            "ctx=ssl.create_default_context(cafile=sys.argv[1])\n"
            "with socket.create_connection(('127.0.0.1',int(sys.argv[2])),timeout=3) as raw:\n"
            "  with ctx.wrap_socket(raw,server_hostname='localhost') as s:\n"
            "    p=b'a\\x00b\\xff'; s.sendall(p); d=s.recv(32); assert d==p,(d,p)\n";
        execl(ESHKOL_TEST_PYTHON, ESHKOL_TEST_PYTHON, "-c", script,
              cert_path, port_text, (char *)NULL);
        _exit(127);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    join_server(&server, thread);
    assert(server.read_result == 4);
}
static void stalled_handshake_deadline(void) {
    int64_t listener = eshkol_net_open("127.0.0.1", 0, 1, 0);
    assert(listener > 0);
    int32_t port = eshkol_net_port(listener, 0); assert(port > 0);
    IdleServer peer = {0, listener, -1, 300}; pthread_t thread;
    assert(pthread_create(&thread, NULL, stall_raw_peer, &peer) == 0);
    int64_t context = eshkol_tls_client_context(cert_path); assert(context > 0);
    int64_t started = monotonic_ms();
    int64_t session = eshkol_tls_connect(context, "127.0.0.1", port,
                                         "localhost", 80);
    int64_t elapsed = monotonic_ms() - started;
    assert(session == -2 && elapsed >= 40 && elapsed < 500);
    assert(eshkol_tls_context_close(context) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(peer.accepted > 0);
    assert(eshkol_net_close(listener) == 0);
}
static int64_t start_idle_server(int delay, IdleServer *server, pthread_t *thread) {
    int64_t context = eshkol_tls_server_context(cert_path, key_path); assert(context > 0);
    server->listener = eshkol_tls_listen(context, "127.0.0.1", 0); assert(server->listener > 0);
    server->context = context; server->accepted = -1; server->delay_ms = delay;
    assert(pthread_create(thread, NULL, serve_idle, server) == 0);
    int64_t port = eshkol_tls_listener_port(server->listener);
    return port;
}
static int64_t connect_idle_client(int64_t port, int64_t *context) {
    *context = eshkol_tls_client_context(cert_path); assert(*context > 0);
    int64_t session = eshkol_tls_connect(*context, "127.0.0.1", (int32_t)port,
                                         "localhost", 1500);
    assert(session > 0);
    assert(eshkol_tls_context_close(*context) == 0); /* session retains context */
    return session;
}
static void stalled_read_deadline(void) {
    IdleServer server; pthread_t thread;
    int64_t port = start_idle_server(500, &server, &thread);
    int64_t context = -1;
    int64_t session = connect_idle_client(port, &context);
    unsigned char buffer[8]; int64_t started = monotonic_ms();
    assert(eshkol_tls_read(session, buffer, sizeof(buffer), 80) == -2);
    int64_t elapsed = monotonic_ms() - started;
    assert(elapsed >= 40 && elapsed < 400);
    assert(eshkol_tls_session_close(session) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(eshkol_tls_listener_close(server.listener) == 0);
}
static void stalled_write_deadline(void) {
    IdleServer server; pthread_t thread;
    int64_t port = start_idle_server(2500, &server, &thread);
    int64_t context = -1;
    int64_t session = connect_idle_client(port, &context);
    unsigned char *payload = (unsigned char *)malloc(1024 * 1024);
    assert(payload); memset(payload, 'w', 1024 * 1024);
    int timed_out = 0; int64_t started = monotonic_ms();
    for (int i = 0; i < 40; ++i) {
        int32_t result = eshkol_tls_write(session, payload, 1024 * 1024, 100);
        if (result == -2) { timed_out = 1; break; }
        assert(result == 1024 * 1024);
    }
    int64_t elapsed = monotonic_ms() - started;
    assert(timed_out && elapsed < 2500);
    assert(eshkol_tls_write(session, payload, 1, 100) == -1);
    free(payload);
    assert(eshkol_tls_session_close(session) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(eshkol_tls_listener_close(server.listener) == 0);
}
static void session_lock_wait_obeys_deadline(void) {
    IdleServer server; pthread_t server_thread;
    int64_t port = start_idle_server(900, &server, &server_thread);
    int64_t context = -1;
    int64_t session = connect_idle_client(port, &context);
    Reader first = {session, -9, 500, 0}; pthread_t reader_thread;
    assert(pthread_create(&reader_thread, NULL, read_blocked, &first) == 0);
    while (!__atomic_load_n(&first.started, __ATOMIC_ACQUIRE)) delay_ms(1);
    delay_ms(50);
    unsigned char buffer[8]; int64_t started = monotonic_ms();
    assert(eshkol_tls_read(session, buffer, sizeof(buffer), 20) == -2);
    int64_t elapsed = monotonic_ms() - started;
    assert(elapsed >= 10 && elapsed < 150);
    assert(pthread_join(reader_thread, NULL) == 0);
    assert(first.result == -2);
    assert(eshkol_tls_session_close(session) == 0);
    assert(pthread_join(server_thread, NULL) == 0);
    assert(eshkol_tls_listener_close(server.listener) == 0);
}
static void concurrent_close_invalidates_first(void) {
    IdleServer server; pthread_t server_thread;
    int64_t port = start_idle_server(1200, &server, &server_thread);
    int64_t context = -1;
    int64_t session = connect_idle_client(port, &context);
    Reader reader = {session, -9, 800, 0}; pthread_t reader_thread;
    assert(pthread_create(&reader_thread, NULL, read_blocked, &reader) == 0);
    delay_ms(40);
    int64_t started = monotonic_ms();
    assert(eshkol_tls_session_close(session) == 0);
    int64_t close_elapsed = monotonic_ms() - started;
    assert(close_elapsed < 250);
    assert(eshkol_tls_session_close(session) == -4);
    assert(pthread_join(reader_thread, NULL) == 0);
    assert(reader.result == -2 || reader.result == -1);
    assert(pthread_join(server_thread, NULL) == 0);
    assert(eshkol_tls_listener_close(server.listener) == 0);
}
static void encrypted_key_never_reads_terminal(const char *executable) {
    FILE *file = fopen(key_path, "rb"); assert(file);
    EVP_PKEY *key = PEM_read_PrivateKey(file, NULL, NULL, NULL);
    fclose(file); assert(key);
    file = fopen(encrypted_key_path, "wb"); assert(file);
    unsigned char password[] = "test-only";
    assert(PEM_write_PrivateKey(file, key, EVP_aes_256_cbc(), password,
                               sizeof(password) - 1, NULL, NULL) == 1);
    fclose(file); EVP_PKEY_free(key);

    int input[2], errors[2];
    assert(pipe(input) == 0 && pipe(errors) == 0);
    pid_t child = fork(); assert(child >= 0);
    if (!child) {
        if (setsid() < 0 || dup2(input[0], STDIN_FILENO) < 0 ||
            dup2(errors[1], STDERR_FILENO) < 0) _exit(126);
        int descriptors[] = {input[0], input[1], errors[0], errors[1]};
        for (size_t i = 0; i < sizeof(descriptors) / sizeof(descriptors[0]); ++i)
            if (descriptors[i] > STDERR_FILENO) close(descriptors[i]);
        execl(executable, executable, "--encrypted-key-probe", cert_path,
              encrypted_key_path, (char *)NULL);
        _exit(127);
    }
    close(input[0]); close(errors[1]);
    /* Keep stdin open without providing a password. A default interactive
     * PEM callback blocks here; the explicit rejection callback must finish. */
    int status = 0;
    pid_t completed = 0;
    int64_t deadline = monotonic_ms() + 3000;
    while (monotonic_ms() < deadline) {
        completed = waitpid(child, &status, WNOHANG);
        if (completed == child) break;
        if (completed < 0 && errno != EINTR) break;
        delay_ms(10);
    }
    if (completed != child) {
        kill(child, SIGKILL);
        pid_t reaped;
        do { reaped = waitpid(child, &status, 0); } while (reaped < 0 && errno == EINTR);
        assert(reaped == child);
    }
    close(input[1]);
    char diagnostic[1024];
    ssize_t n;
    do { n = read(errors[0], diagnostic, sizeof(diagnostic)); } while (n < 0 && errno == EINTR);
    close(errors[0]);
    assert(completed == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    assert(n == 0);  /* No password request or terminal diagnostics. */
}
int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--encrypted-key-probe") == 0) {
        int64_t result = eshkol_tls_server_context(argv[2], argv[3]);
        if (result > 0) eshkol_tls_context_close(result);
        return result == -1 ? 0 : 1;
    }
    assert(eshkol_tls_available() == 1);
    const char *temporary_root = getenv("TMPDIR");
    if (!temporary_root || !temporary_root[0]) temporary_root = "/tmp";
    int path_length = snprintf(tempdir, sizeof(tempdir), "%s/eshkol-tls-test-XXXXXX",
                               temporary_root);
    assert(path_length > 0 && (size_t)path_length < sizeof(tempdir));
    assert(mkdtemp(tempdir));
    snprintf(cert_path, sizeof(cert_path), "%s/trusted.pem", tempdir);
    snprintf(key_path, sizeof(key_path), "%s/trusted-key.pem", tempdir);
    snprintf(wrong_cert_path, sizeof(wrong_cert_path), "%s/wrong-ca.pem", tempdir);
    snprintf(wrong_key_path, sizeof(wrong_key_path), "%s/wrong-key.pem", tempdir);
    snprintf(expired_cert_path, sizeof(expired_cert_path), "%s/expired.pem", tempdir);
    snprintf(expired_key_path, sizeof(expired_key_path), "%s/expired-key.pem", tempdir);
    snprintf(encrypted_key_path, sizeof(encrypted_key_path), "%s/encrypted-key.pem", tempdir);
    make_cert_files(cert_path, key_path, "localhost", 0);
    make_cert_files(wrong_cert_path, wrong_key_path, "different.example", 0);
    make_cert_files(expired_cert_path, expired_key_path, "localhost", 1);

    trusted_binary_roundtrip();
    python_ssl_roundtrip();
    stalled_handshake_deadline();
    stalled_read_deadline();
    stalled_write_deadline();
    session_lock_wait_obeys_deadline();
    concurrent_close_invalidates_first();
    rejected_peer(wrong_cert_path, cert_path, key_path, "localhost"); /* wrong CA */
    rejected_peer(cert_path, cert_path, key_path, "wrong.example"); /* wrong name */
    rejected_peer(expired_cert_path, expired_cert_path, expired_key_path, "localhost");
    assert(eshkol_tls_server_context(cert_path, wrong_key_path) == -1); /* mismatch */
    assert(eshkol_tls_client_context("/definitely/missing/ca.pem") == -1);
    assert(eshkol_tls_server_context("/definitely/missing/cert.pem",
                                    "/definitely/missing/key.pem") == -1);
    encrypted_key_never_reads_terminal(argv[0]);

    unlink(cert_path); unlink(key_path); unlink(wrong_cert_path); unlink(wrong_key_path);
    unlink(expired_cert_path); unlink(expired_key_path);
    unlink(encrypted_key_path); rmdir(tempdir);
    puts("PASS: TLS handshake, binary roundtrip, peer verification, and cleanup");
    return 0;
}
#endif
