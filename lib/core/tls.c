#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
/* Optional native TLS contexts and record sessions over opaque core.net
 * handles. No socket descriptors cross this boundary. */
#include "eshkol/core/tls.h"
#include "eshkol/core/net.h"
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ESHKOL_TLS_AVAILABLE
#define ESHKOL_TLS_AVAILABLE 0
#endif

#define TLS_MAX_BYTES (1024 * 1024)
static int valid_tls_id(int64_t id) {
    return id > 0 && id <= INT64_C(9007199254740991);
}
static int valid_tls_path(const char *path, int optional) {
    if (!path) return optional;
    size_t n = 0;
    while (n <= 4096 && path[n]) n++;
    return (optional || n > 0) && n <= 4096;
}
static int valid_tls_address(const char *address) {
    if (!valid_tls_path(address, 0)) return 0;
    size_t n = 0; while (n <= 45 && address[n]) n++;
    return n <= 45;
}
static int valid_tls_identity(const char *name) {
    if (!valid_tls_path(name, 0)) return 0;
    size_t n = 0; while (n <= 253 && name[n]) n++;
    return n <= 253;
}

#if ESHKOL_TLS_AVAILABLE && !defined(_WIN32) && \
    !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

enum { TLS_CONTEXT_SLOTS = 64, TLS_LISTENER_SLOTS = 64, TLS_SESSION_SLOTS = 128 };
enum { TLS_ROLE_CLIENT = 1, TLS_ROLE_SERVER = 2 };
#define TLS_LAST_ID INT64_C(9007199254740991)

typedef struct {
    SSL_CTX *ctx;
    int refs;
    int closed;
    int role;
    int64_t id;
} TlsContext;
typedef struct {
    TlsContext *context;
    int64_t net_handle;
    int refs;
    int closed;
    int net_closed;
    int64_t id;
} TlsListener;
typedef struct TlsSession {
    SSL *ssl;
    TlsContext *context;
    int64_t net_handle;
    int refs;
    int closed;
    int net_closed;
    int slot;
    int io_error;
    int failed;
    int fatal_tls_error;
    int read_eof;
    int64_t deadline_ms;
    int64_t id;
    pthread_mutex_t operation_lock;
} TlsSession;
typedef struct { TlsSession *session; } TlsBioData;

static TlsContext contexts[TLS_CONTEXT_SLOTS];
static TlsListener listeners[TLS_LISTENER_SLOTS];
static TlsSession *sessions[TLS_SESSION_SLOTS];
static int64_t next_tls_id = 1;
static pthread_mutex_t registry_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t bio_once = PTHREAD_ONCE_INIT;
static BIO_METHOD *net_bio_method;
static int bio_method_ready;

static int64_t now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1;
    if (ts.tv_sec > INT64_MAX / 1000) return -1;
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
static int64_t deadline_after(int32_t timeout_ms) {
    int64_t now = now_ms();
    if (now < 0 || timeout_ms < 0 || now > INT64_MAX - timeout_ms) return -1;
    return now + timeout_ms;
}
static int32_t remaining_ms(int64_t deadline) {
    int64_t now = now_ms();
    if (now < 0) return -1;
    if (now >= deadline) return 0;
    int64_t left = deadline - now;
    return left > INT32_MAX ? INT32_MAX : (int32_t)left;
}
/* pthread_mutex_timedlock uses CLOCK_REALTIME on this target. Poll trylock
 * against the operation's existing monotonic deadline instead. */
static int lock_session_until(pthread_mutex_t *mutex, int64_t deadline) {
    int waited = 0;
    for (;;) {
        int rc = pthread_mutex_trylock(mutex);
        if (!rc) {
            if (waited) {
                int32_t left = remaining_ms(deadline);
                if (left < 0) { pthread_mutex_unlock(mutex); return -1; }
                if (!left) { pthread_mutex_unlock(mutex); return 0; }
            }
            return 1;
        }
        if (rc != EBUSY) return -1;
        waited = 1;
        int32_t left = remaining_ms(deadline);
        if (left < 0) return -1;
        if (!left) return 0;
        struct timespec pause = {0, 1000000L};
        while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {}
    }
}
static int64_t next_id_locked(void) {
    return next_tls_id > 0 && next_tls_id <= TLS_LAST_ID ? next_tls_id++ : 0;
}
static int64_t register_context_id(SSL_CTX *ctx, int role) {
    int64_t result = -1;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_CONTEXT_SLOTS; ++i) if (!contexts[i].ctx) {
        int64_t id = next_id_locked();
        if (id) { contexts[i] = (TlsContext){ctx, 0, 0, role, id}; result = id; }
        break;
    }
    pthread_mutex_unlock(&registry_lock);
    if (result < 0) SSL_CTX_free(ctx);
    return result;
}
static TlsContext *context_acquire(int64_t id) {
    TlsContext *found = NULL;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_CONTEXT_SLOTS; ++i)
        if (contexts[i].id == id && !contexts[i].closed) {
            contexts[i].refs++; found = &contexts[i]; break;
        }
    pthread_mutex_unlock(&registry_lock);
    return found;
}
static TlsContext *context_acquire_from_listener(TlsContext *context) {
    TlsContext *found = NULL;
    pthread_mutex_lock(&registry_lock);
    if (context && !context->closed) { context->refs++; found = context; }
    pthread_mutex_unlock(&registry_lock);
    return found;
}
static void context_release(TlsContext *context) {
    SSL_CTX *free_ctx = NULL;
    pthread_mutex_lock(&registry_lock);
    if (context && context->refs > 0) {
        context->refs--;
        if (context->closed && context->refs == 0) {
            free_ctx = context->ctx;
            memset(context, 0, sizeof(*context));
        }
    }
    pthread_mutex_unlock(&registry_lock);
    if (free_ctx) SSL_CTX_free(free_ctx);
}

static int bio_create(BIO *bio) { BIO_set_init(bio, 1); return 1; }
static int bio_destroy(BIO *bio) {
    if (!bio) return 0;
    free(BIO_get_data(bio));
    BIO_set_data(bio, NULL);
    BIO_set_init(bio, 0);
    return 1;
}
static int bio_read(BIO *bio, char *buffer, int length) {
    TlsBioData *data = (TlsBioData *)BIO_get_data(bio);
    if (!data || !data->session || !buffer || length <= 0) return 0;
    TlsSession *session = data->session;
    BIO_clear_retry_flags(bio);
    int32_t timeout = remaining_ms(session->deadline_ms);
    if (timeout < 0) { session->io_error = -1; return -1; }
    int32_t result = eshkol_net_read(session->net_handle, buffer, length, timeout);
    if (result >= 0) return result;
    session->io_error = result;
    if (result == -2) BIO_set_retry_read(bio);
    return -1;
}
static int bio_write(BIO *bio, const char *buffer, int length) {
    TlsBioData *data = (TlsBioData *)BIO_get_data(bio);
    if (!data || !data->session || !buffer || length <= 0) return 0;
    TlsSession *session = data->session;
    BIO_clear_retry_flags(bio);
    int32_t timeout = remaining_ms(session->deadline_ms);
    if (timeout < 0) { session->io_error = -1; return -1; }
    int32_t result = eshkol_net_write(session->net_handle, buffer, length, timeout);
    if (result > 0) return result;
    if (result == 0) { session->io_error = -1; return -1; }
    session->io_error = result;
    if (result == -2) BIO_set_retry_write(bio);
    return -1;
}
static int bio_puts(BIO *bio, const char *text) {
    size_t n = text ? strlen(text) : 0;
    return n > INT_MAX ? -1 : bio_write(bio, text, (int)n);
}
static long bio_ctrl(BIO *bio, int command, long value, void *pointer) {
    (void)bio; (void)value; (void)pointer;
    switch (command) {
        case BIO_CTRL_FLUSH: return 1;
        case BIO_CTRL_DUP: return 1;
        case BIO_CTRL_GET_CLOSE: return 0;
        case BIO_CTRL_SET_CLOSE: return 1;
        case BIO_CTRL_PENDING: case BIO_CTRL_WPENDING: case BIO_CTRL_EOF: return 0;
        default: return 0;
    }
}
static void create_bio_method(void) {
    BIO_METHOD *method = BIO_meth_new(BIO_TYPE_SOURCE_SINK | BIO_get_new_index(),
                                      "Eshkol opaque net handle");
    if (!method) return;
    if (BIO_meth_set_create(method, bio_create) != 1 ||
        BIO_meth_set_destroy(method, bio_destroy) != 1 ||
        BIO_meth_set_read(method, bio_read) != 1 ||
        BIO_meth_set_write(method, bio_write) != 1 ||
        BIO_meth_set_puts(method, bio_puts) != 1 ||
        BIO_meth_set_ctrl(method, bio_ctrl) != 1) {
        BIO_meth_free(method); return;
    }
    net_bio_method = method;
    bio_method_ready = 1;
}
static BIO *new_net_bio(TlsSession *session) {
    pthread_once(&bio_once, create_bio_method);
    if (!bio_method_ready) return NULL;
    BIO *bio = BIO_new(net_bio_method);
    TlsBioData *data = (TlsBioData *)malloc(sizeof(*data));
    if (!bio || !data) { BIO_free(bio); free(data); return NULL; }
    data->session = session;
    BIO_set_data(bio, data);
    return bio;
}
static int tls_attach_bios(TlsSession *session) {
    BIO *read_bio = new_net_bio(session);
    BIO *write_bio = new_net_bio(session);
    if (!read_bio || !write_bio) { BIO_free(read_bio); BIO_free(write_bio); return 0; }
    /* Each distinct BIO transfers exactly one owned reference to SSL. */
    SSL_set0_rbio(session->ssl, read_bio);
    SSL_set0_wbio(session->ssl, write_bio);
    return 1;
}
static int set_peer_identity(SSL *ssl, const char *name) {
    if (!valid_tls_identity(name)) return 0;
    X509_VERIFY_PARAM *param = SSL_get0_param(ssl);
    unsigned char ip[16];
    if (inet_pton(AF_INET, name, ip) == 1 || inet_pton(AF_INET6, name, ip) == 1)
        return X509_VERIFY_PARAM_set1_ip_asc(param, name) == 1;
    return X509_VERIFY_PARAM_set1_host(param, name, 0) == 1 &&
           SSL_set_tlsext_host_name(ssl, name) == 1;
}
static int handshake(TlsSession *session, int server) {
    if (server) SSL_set_accept_state(session->ssl);
    else SSL_set_connect_state(session->ssl);
    for (;;) {
        int32_t before = remaining_ms(session->deadline_ms);
        if (before < 0) return -1;
        if (!before) return -2;
        ERR_clear_error();
        session->io_error = 0;
        int result = server ? SSL_accept(session->ssl) : SSL_connect(session->ssl);
        if (result == 1) {
            int32_t after = remaining_ms(session->deadline_ms);
            return after < 0 ? -1 : (after == 0 ? -2 : 0);
        }
        int ssl_error = SSL_get_error(session->ssl, result);
        if (ssl_error != SSL_ERROR_WANT_READ && ssl_error != SSL_ERROR_WANT_WRITE) {
            if (ssl_error == SSL_ERROR_SSL || ssl_error == SSL_ERROR_SYSCALL)
                session->fatal_tls_error = 1;
            return session->io_error == -2 ? -2 : -1;
        }
        int32_t left = remaining_ms(session->deadline_ms);
        if (left < 0) return -1;
        if (!left) return -2;
    }
}
static TlsSession *session_new(TlsContext *context, int64_t net_handle,
                               const char *server_name, int64_t deadline,
                               int server, int32_t *error) {
    if (error) *error = -1;
    TlsSession *session = (TlsSession *)calloc(1, sizeof(*session));
    if (!session) { eshkol_net_close(net_handle); context_release(context); return NULL; }
    session->context = context;
    session->net_handle = net_handle;
    session->deadline_ms = deadline;
    if (deadline < 0 || pthread_mutex_init(&session->operation_lock, NULL)) {
        eshkol_net_close(net_handle); context_release(context); free(session); return NULL;
    }
    session->ssl = SSL_new(context->ctx);
    if (!session->ssl || (!server && !set_peer_identity(session->ssl, server_name)) ||
        !tls_attach_bios(session)) {
        SSL_free(session->ssl); eshkol_net_close(net_handle); context_release(context);
        pthread_mutex_destroy(&session->operation_lock); free(session); return NULL;
    }
    int result = handshake(session, server);
    session->deadline_ms = -1;
    if (result) {
        if (error) *error = result;
        SSL_free(session->ssl); eshkol_net_close(net_handle); context_release(context);
        pthread_mutex_destroy(&session->operation_lock); free(session); return NULL;
    }
    if (error) *error = 0;
    return session;
}
static int64_t register_session(TlsSession *session) {
    int64_t result = -1;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_SESSION_SLOTS; ++i) if (!sessions[i]) {
        int64_t id = next_id_locked();
        if (id) {
            session->slot = i; session->id = id; sessions[i] = session; result = id;
        }
        break;
    }
    pthread_mutex_unlock(&registry_lock);
    if (result < 0) {
        SSL_free(session->ssl); eshkol_net_close(session->net_handle);
        context_release(session->context); pthread_mutex_destroy(&session->operation_lock);
        free(session);
    }
    return result;
}
static TlsSession *session_acquire(int64_t id) {
    TlsSession *found = NULL;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_SESSION_SLOTS; ++i)
        if (sessions[i] && sessions[i]->id == id && !sessions[i]->closed) {
            sessions[i]->refs++; found = sessions[i]; break;
        }
    pthread_mutex_unlock(&registry_lock);
    return found;
}
static void session_destroy(TlsSession *session) {
    if (!session->net_closed) {
        if (!session->failed && !session->fatal_tls_error) {
            int64_t deadline = deadline_after(10);
            if (deadline >= 0) {
                session->deadline_ms = deadline;
                ERR_clear_error();
                (void)SSL_shutdown(session->ssl);
            }
        }
        eshkol_net_close(session->net_handle);
    }
    SSL_free(session->ssl);
    context_release(session->context);
    pthread_mutex_destroy(&session->operation_lock);
    free(session);
}
static void session_release(TlsSession *session) {
    int destroy = 0;
    pthread_mutex_lock(&registry_lock);
    if (session->refs > 0) session->refs--;
    if (session->closed && session->refs == 0) {
        sessions[session->slot] = NULL;
        destroy = 1;
    }
    pthread_mutex_unlock(&registry_lock);
    if (destroy) session_destroy(session);
}
static TlsListener *listener_acquire(int64_t id) {
    TlsListener *found = NULL;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_LISTENER_SLOTS; ++i)
        if (listeners[i].id == id && !listeners[i].closed) {
            listeners[i].refs++; found = &listeners[i]; break;
        }
    pthread_mutex_unlock(&registry_lock);
    return found;
}
static void listener_release(TlsListener *listener) {
    int64_t net_handle = 0;
    TlsContext *context = NULL;
    pthread_mutex_lock(&registry_lock);
    if (listener && listener->refs > 0) listener->refs--;
    if (listener && listener->closed && listener->refs == 0) {
        if (!listener->net_closed) net_handle = listener->net_handle;
        context = listener->context;
        memset(listener, 0, sizeof(*listener));
    }
    pthread_mutex_unlock(&registry_lock);
    if (net_handle) eshkol_net_close(net_handle);
    if (context) context_release(context);
}
static int32_t ssl_error_code(TlsSession *session, int ssl_error, int writing) {
    if (ssl_error == SSL_ERROR_SSL || ssl_error == SSL_ERROR_SYSCALL)
        session->fatal_tls_error = 1;
    if (ssl_error == SSL_ERROR_ZERO_RETURN) return writing ? -1 : 0;
    if (session->io_error == -2) return -2;
    return -1;
}
/* The API has no password argument. Never let PEM loading fall back to an
 * interactive terminal dialog in a context constructor. */
static int refuse_pem_password(char *buffer, int size, int writing, void *userdata) {
    (void)buffer; (void)size; (void)writing; (void)userdata;
    return -1;
}

int32_t eshkol_tls_available(void) { return 1; }
int64_t eshkol_tls_client_context(const char *ca_file) {
    if (!valid_tls_path(ca_file, 1)) return -4;
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return -1;
    int ok = SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 1;
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
    if (ok) ok = ca_file && ca_file[0]
        ? SSL_CTX_load_verify_locations(ctx, ca_file, NULL) == 1
        : SSL_CTX_set_default_verify_paths(ctx) == 1;
    if (!ok) { SSL_CTX_free(ctx); ERR_clear_error(); return -1; }
    return register_context_id(ctx, TLS_ROLE_CLIENT);
}
int64_t eshkol_tls_server_context(const char *certificate_file,
                                  const char *private_key_file) {
    if (!valid_tls_path(certificate_file, 0) || !valid_tls_path(private_key_file, 0) ||
        !certificate_file || !private_key_file) return -4;
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) return -1;
    SSL_CTX_set_default_passwd_cb(ctx, refuse_pem_password);
    int ok = SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 1 &&
        SSL_CTX_use_certificate_chain_file(ctx, certificate_file) == 1 &&
        SSL_CTX_use_PrivateKey_file(ctx, private_key_file, SSL_FILETYPE_PEM) == 1 &&
        SSL_CTX_check_private_key(ctx) == 1;
    if (!ok) { SSL_CTX_free(ctx); ERR_clear_error(); return -1; }
    return register_context_id(ctx, TLS_ROLE_SERVER);
}
int32_t eshkol_tls_context_close(int64_t id) {
    if (!valid_tls_id(id)) return -4;
    SSL_CTX *free_ctx = NULL;
    int found = 0;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_CONTEXT_SLOTS; ++i) if (contexts[i].id == id && !contexts[i].closed) {
        found = 1;
        contexts[i].closed = 1; contexts[i].id = 0;
        if (!contexts[i].refs) { free_ctx = contexts[i].ctx; memset(&contexts[i], 0, sizeof(contexts[i])); }
        break;
    }
    pthread_mutex_unlock(&registry_lock);
    if (!found) return -4;
    if (!free_ctx) return 0;
    SSL_CTX_free(free_ctx);
    return 0;
}
int64_t eshkol_tls_listen(int64_t id, const char *address, int32_t port) {
    if (!valid_tls_id(id) || !valid_tls_address(address) || port < 0 || port > 65535) return -4;
    TlsContext *context = context_acquire(id);
    if (!context) return -4;
    if (context->role != TLS_ROLE_SERVER) { context_release(context); return -4; }
    int64_t net_handle = eshkol_net_open(address, port, 1, 0);
    if (net_handle < 0) { context_release(context); return net_handle; }
    int64_t result = -1;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_LISTENER_SLOTS; ++i) if (!listeners[i].context) {
        int64_t listener_id = next_id_locked();
        if (listener_id) {
            listeners[i] = (TlsListener){context, net_handle, 0, 0, 0, listener_id};
            result = listener_id;
        }
        break;
    }
    pthread_mutex_unlock(&registry_lock);
    if (result < 0) { eshkol_net_close(net_handle); context_release(context); }
    return result;
}
int32_t eshkol_tls_listener_port(int64_t id) {
    TlsListener *listener = listener_acquire(id);
    if (!listener) return -4;
    int32_t port = eshkol_net_port(listener->net_handle, 0);
    listener_release(listener);
    return port;
}
int64_t eshkol_tls_accept(int64_t id, int32_t timeout_ms) {
    if (!valid_tls_id(id) || timeout_ms < 0) return -4;
    int64_t deadline = deadline_after(timeout_ms);
    if (deadline < 0) return -1;
    TlsListener *listener = listener_acquire(id);
    if (!listener) return -4;
    TlsContext *context = context_acquire_from_listener(listener->context);
    if (!context) { listener_release(listener); return -4; }
    int32_t left = remaining_ms(deadline);
    if (left < 0) { context_release(context); listener_release(listener); return -1; }
    int64_t net_handle = eshkol_net_accept(listener->net_handle, left);
    listener_release(listener);
    if (net_handle < 0) { context_release(context); return net_handle; }
    int32_t error = -1;
    TlsSession *session = session_new(context, net_handle, NULL, deadline, 1, &error);
    if (!session) return error;
    return register_session(session);
}
int64_t eshkol_tls_connect(int64_t id, const char *address, int32_t port,
                           const char *server_name, int32_t timeout_ms) {
    if (!valid_tls_id(id) || !valid_tls_address(address) || !valid_tls_identity(server_name) ||
        port < 0 || port > 65535 || timeout_ms < 0) return -4;
    int64_t deadline = deadline_after(timeout_ms);
    if (deadline < 0) return -1;
    TlsContext *context = context_acquire(id);
    if (!context) return -4;
    if (context->role != TLS_ROLE_CLIENT) { context_release(context); return -4; }
    int32_t left = remaining_ms(deadline);
    if (left < 0) { context_release(context); return -1; }
    int64_t net_handle = eshkol_net_open(address, port, 2, left);
    if (net_handle < 0) { context_release(context); return net_handle; }
    int32_t error = -1;
    TlsSession *session = session_new(context, net_handle, server_name, deadline, 0, &error);
    if (!session) return error;
    return register_session(session);
}
static int32_t transfer(int64_t id, void *buffer, int32_t length,
                        int32_t timeout_ms, int writing) {
    if (length < 0 || length > TLS_MAX_BYTES || (!writing && length == 0) ||
        (!buffer && length > 0) || timeout_ms < 0) return -4;
    int64_t deadline = deadline_after(timeout_ms);
    if (deadline < 0) return -1;
    TlsSession *session = session_acquire(id);
    if (!session) return -4;
    if (writing && length == 0) { session_release(session); return 0; }
    int locked = lock_session_until(&session->operation_lock, deadline);
    if (locked <= 0) {
        session_release(session);
        return locked == 0 ? -2 : -1;
    }
    if (session->failed || (writing && session->read_eof)) {
        pthread_mutex_unlock(&session->operation_lock);
        session_release(session);
        return -1;
    }
    session->deadline_ms = deadline;
    session->io_error = 0;
    int32_t result = -1;
    if (writing) {
        size_t written = 0;
        for (;;) {
            ERR_clear_error();
            session->io_error = 0;
            int ok = SSL_write_ex(session->ssl, buffer, (size_t)length, &written);
            if (ok == 1) {
                result = written > INT32_MAX ? -1 : (int32_t)written;
                break;
            }
            int ssl_error = SSL_get_error(session->ssl, ok);
            if (ssl_error != SSL_ERROR_WANT_READ && ssl_error != SSL_ERROR_WANT_WRITE) {
                result = ssl_error_code(session, ssl_error, 1);
                break;
            }
            int32_t left = remaining_ms(session->deadline_ms);
            if (left < 0) { result = -1; break; }
            if (!left) { result = -2; break; }
        }
    } else {
        size_t received = 0;
        for (;;) {
            ERR_clear_error();
            session->io_error = 0;
            int ok = SSL_read_ex(session->ssl, buffer, (size_t)length, &received);
            if (ok == 1) {
                result = received > INT32_MAX ? -1 : (int32_t)received;
                break;
            }
            int ssl_error = SSL_get_error(session->ssl, ok);
            if (ssl_error != SSL_ERROR_WANT_READ && ssl_error != SSL_ERROR_WANT_WRITE) {
                result = ssl_error_code(session, ssl_error, 0);
                break;
            }
            int32_t left = remaining_ms(session->deadline_ms);
            if (left < 0) { result = -1; break; }
            if (!left) { result = -2; break; }
        }
    }
    if (!writing && result == 0) session->read_eof = 1;
    if (writing && result < 0) session->failed = 1;
    if (!writing && result == -1) session->failed = 1;
    session->deadline_ms = -1;
    pthread_mutex_unlock(&session->operation_lock);
    session_release(session);
    return result;
}
int32_t eshkol_tls_read(int64_t id, void *buffer, int32_t capacity, int32_t timeout_ms) {
    return transfer(id, buffer, capacity, timeout_ms, 0);
}
int32_t eshkol_tls_write(int64_t id, const void *buffer, int32_t length, int32_t timeout_ms) {
    return transfer(id, (void *)buffer, length, timeout_ms, 1);
}
int32_t eshkol_tls_session_close(int64_t id) {
    if (!valid_tls_id(id)) return -4;
    TlsSession *session = NULL;
    int64_t net_handle = 0;
    int destroy = 0;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_SESSION_SLOTS; ++i)
        if (sessions[i] && sessions[i]->id == id && !sessions[i]->closed) {
            session = sessions[i]; session->closed = 1; session->id = 0;
            net_handle = session->net_handle;
            if (!session->refs) { sessions[i] = NULL; destroy = 1; }
            else session->net_closed = 1;
            break;
        }
    pthread_mutex_unlock(&registry_lock);
    if (!session) return -4;
    if (destroy) session_destroy(session);
    else eshkol_net_close(net_handle);
    return 0;
}
int32_t eshkol_tls_listener_close(int64_t id) {
    if (!valid_tls_id(id)) return -4;
    TlsListener *listener = NULL;
    int64_t net_handle = 0;
    TlsContext *context = NULL;
    pthread_mutex_lock(&registry_lock);
    for (int i = 0; i < TLS_LISTENER_SLOTS; ++i)
        if (listeners[i].id == id && !listeners[i].closed) {
            listener = &listeners[i]; listener->closed = 1; listener->id = 0;
            listener->net_closed = 1; net_handle = listener->net_handle;
            if (!listener->refs) { context = listener->context; memset(listener, 0, sizeof(*listener)); }
            break;
        }
    pthread_mutex_unlock(&registry_lock);
    if (!listener) return -4;
    eshkol_net_close(net_handle);
    if (context) context_release(context);
    return 0;
}
#else
int32_t eshkol_tls_available(void) { return 0; }
int64_t eshkol_tls_client_context(const char *ca_file) {
    return valid_tls_path(ca_file, 1) ? -3 : -4;
}
int64_t eshkol_tls_server_context(const char *certificate_file, const char *private_key_file) {
    return valid_tls_path(certificate_file, 0) && valid_tls_path(private_key_file, 0)
        ? -3 : -4;
}
int32_t eshkol_tls_context_close(int64_t context) { return valid_tls_id(context) ? -3 : -4; }
int64_t eshkol_tls_listen(int64_t context, const char *address, int32_t port) {
    return valid_tls_id(context) && valid_tls_address(address) && port >= 0 && port <= 65535
        ? -3 : -4;
}
int64_t eshkol_tls_accept(int64_t listener, int32_t timeout_ms) {
    return valid_tls_id(listener) && timeout_ms >= 0 ? -3 : -4;
}
int64_t eshkol_tls_connect(int64_t context, const char *address, int32_t port,
                           const char *server_name, int32_t timeout_ms) {
    return valid_tls_id(context) && valid_tls_address(address) && valid_tls_identity(server_name) &&
        port >= 0 && port <= 65535 && timeout_ms >= 0 ? -3 : -4;
}
int32_t eshkol_tls_listener_port(int64_t listener) { return valid_tls_id(listener) ? -3 : -4; }
int32_t eshkol_tls_read(int64_t session, void *buffer, int32_t capacity, int32_t timeout_ms) {
    return valid_tls_id(session) && buffer && capacity > 0 && capacity <= TLS_MAX_BYTES &&
        timeout_ms >= 0 ? -3 : -4;
}
int32_t eshkol_tls_write(int64_t session, const void *buffer, int32_t length, int32_t timeout_ms) {
    return valid_tls_id(session) && length >= 0 && length <= TLS_MAX_BYTES &&
        (buffer || length == 0) && timeout_ms >= 0 ? -3 : -4;
}
int32_t eshkol_tls_session_close(int64_t session) { return valid_tls_id(session) ? -3 : -4; }
int32_t eshkol_tls_listener_close(int64_t listener) { return valid_tls_id(listener) ? -3 : -4; }
#endif
