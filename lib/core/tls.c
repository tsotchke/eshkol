#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
/* Optional native OpenSSL context construction. Public session operations are
 * intentionally implemented in the same core boundary in a later slice. */
#include "eshkol/core/tls.h"
#include <pthread.h>
#include <stddef.h>
#include <string.h>

#if defined(ESHKOL_HAVE_OPENSSL_TLS) && !defined(_WIN32) && \
    !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
#include <openssl/ssl.h>
#include <openssl/err.h>

#define TLS_CONTEXT_SLOTS 64
#define TLS_LAST_ID INT64_C(9007199254740991)
typedef struct { SSL_CTX *ctx; int64_t id; } TlsContext;
static TlsContext contexts[TLS_CONTEXT_SLOTS];
static int64_t next_context_id = 1;
static pthread_mutex_t contexts_lock = PTHREAD_MUTEX_INITIALIZER;

int32_t eshkol_tls_available(void) { return 1; }

static int valid_path(const char *path, int optional) {
    if (!path) return 1;
    size_t n = strnlen(path, 4097);
    return (optional || n > 0) && n <= 4096;
}

static int64_t register_context(SSL_CTX *ctx) {
    int64_t result = -1;
    pthread_mutex_lock(&contexts_lock);
    if (next_context_id <= TLS_LAST_ID) {
        for (int i = 0; i < TLS_CONTEXT_SLOTS; ++i) {
            if (!contexts[i].id) {
                contexts[i] = (TlsContext){ctx, next_context_id++};
                result = contexts[i].id;
                break;
            }
        }
    }
    pthread_mutex_unlock(&contexts_lock);
    if (result < 0) SSL_CTX_free(ctx);
    return result;
}

int64_t eshkol_tls_client_context(const char *ca_file) {
    if (!valid_path(ca_file, 1)) return -4;
    SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) return -1;
    int ok = SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 1;
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
    if (ok) {
        ok = ca_file && ca_file[0]
                 ? SSL_CTX_load_verify_locations(ctx, ca_file, NULL) == 1
                 : SSL_CTX_set_default_verify_paths(ctx) == 1;
    }
    if (!ok) { SSL_CTX_free(ctx); ERR_clear_error(); return -1; }
    return register_context(ctx);
}

int64_t eshkol_tls_server_context(const char *certificate_file,
                                  const char *private_key_file) {
    if (!valid_path(certificate_file, 0) || !valid_path(private_key_file, 0) ||
        !certificate_file || !private_key_file) return -4;
    SSL_CTX *ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) return -1;
    int ok = SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 1 &&
             SSL_CTX_use_certificate_chain_file(ctx, certificate_file) == 1 &&
             SSL_CTX_use_PrivateKey_file(ctx, private_key_file, SSL_FILETYPE_PEM) == 1 &&
             SSL_CTX_check_private_key(ctx) == 1;
    if (!ok) { SSL_CTX_free(ctx); ERR_clear_error(); return -1; }
    return register_context(ctx);
}

int32_t eshkol_tls_context_close(int64_t id) {
    if (id <= 0 || id > TLS_LAST_ID) return -4;
    SSL_CTX *ctx = NULL;
    pthread_mutex_lock(&contexts_lock);
    for (int i = 0; i < TLS_CONTEXT_SLOTS; ++i) {
        if (contexts[i].id == id) {
            ctx = contexts[i].ctx;
            memset(&contexts[i], 0, sizeof(contexts[i]));
            break;
        }
    }
    pthread_mutex_unlock(&contexts_lock);
    if (!ctx) return -4;
    SSL_CTX_free(ctx);
    return 0;
}
#else
int32_t eshkol_tls_available(void) { return 0; }
int64_t eshkol_tls_client_context(const char *ca_file) { (void)ca_file; return -3; }
int64_t eshkol_tls_server_context(const char *certificate_file,
                                  const char *private_key_file) {
    (void)certificate_file; (void)private_key_file; return -3;
}
int32_t eshkol_tls_context_close(int64_t context) { (void)context; return -3; }
#endif
