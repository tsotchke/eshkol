#ifndef ESHKOL_CORE_TLS_H
#define ESHKOL_CORE_TLS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Optional native TLS context API. Context identifiers are positive, exact
 * binary64 integers and never reused. Results: -1 system/TLS failure,
 * -3 TLS unavailable on this build/platform, -4 invalid argument/handle. */
int32_t eshkol_tls_available(void);
/** Create a peer-verifying TLS client context. A null or empty CA file uses
 * system default trust paths; a non-empty file replaces that trust source. */
int64_t eshkol_tls_client_context(const char *ca_file);
/** Create a TLS server context from a PEM certificate chain and private key.
 * A missing file or non-matching key fails closed. */
int64_t eshkol_tls_server_context(const char *certificate_file,
                                 const char *private_key_file);
/** Close a context. Stale and double-closed identifiers return -4. */
int32_t eshkol_tls_context_close(int64_t context);
/** Create a TLS listener on a numeric IPv4/IPv6 address and port. */
int64_t eshkol_tls_listen(int64_t context, const char *address, int32_t port);
/** Accept and complete a server handshake within one total deadline. */
int64_t eshkol_tls_accept(int64_t listener, int32_t timeout_ms);
/** Connect to a numeric address and verify its certificate against server_name. */
int64_t eshkol_tls_connect(int64_t context, const char *address, int32_t port,
                           const char *server_name, int32_t timeout_ms);
/** Query a TLS listener's selected local port. */
int32_t eshkol_tls_listener_port(int64_t listener);
/** Read plaintext bytes; zero indicates authenticated TLS close-notify/EOF. */
int32_t eshkol_tls_read(int64_t session, void *buffer, int32_t capacity,
                        int32_t timeout_ms);
/** Write plaintext bytes, bounded by one total operation deadline. */
int32_t eshkol_tls_write(int64_t session, const void *buffer, int32_t length,
                         int32_t timeout_ms);
/** Close a TLS session or listener once; stale identifiers return -4. */
int32_t eshkol_tls_session_close(int64_t session);
int32_t eshkol_tls_listener_close(int64_t listener);
#ifdef __cplusplus
}
#endif
#endif
