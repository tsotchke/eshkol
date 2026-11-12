#ifndef ESHKOL_CORE_TLS_H
#define ESHKOL_CORE_TLS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Optional native TLS (OpenSSL) over core.net TCP streams. Identifiers are
 * positive, exact binary64 integers and never reused. Results use the shared
 * ESHKOL_IO_* codes (eshkol/core/resource_lease.h): -1 system/TLS failure,
 * -2 operation deadline, -3 TLS unavailable on this build/platform,
 * -4 invalid argument/handle, -5 busy, -6 cancelled, -7 peer reset,
 * -10 peer certificate not trusted, -11 peer certificate names another host.
 * A session holds an exclusive borrow of its TCP stream for its lifetime. */
/** Client-context flag: explicitly accept peers without certificate checks. */
#define ESHKOL_TLS_INSECURE_NO_VERIFY 0x1
int32_t eshkol_tls_available(void);
/** Create a peer-verifying TLS client context. A null or empty CA file uses
 * system default trust paths; a non-empty file replaces that trust source. */
int64_t eshkol_tls_client_context(const char *ca_file);
/** As eshkol_tls_client_context; flags may only be ESHKOL_TLS_INSECURE_NO_VERIFY. */
int64_t eshkol_tls_client_context_options(const char *ca_file, int32_t flags);
/** Create a TLS server context from a PEM chain and unencrypted private key.
 * Missing files, encrypted keys, and non-matching keys fail without prompting. */
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
/** Client handshake on an owned TCP stream; SNI and the verified identity are
 * server_name. The stream moves into the session (closed on handshake
 * failure); a stream that cannot be borrowed (-4/-5) stays with the caller. */
int64_t eshkol_tls_client_start(int64_t context, int64_t net_handle, const char *server_name,
                                int32_t timeout_ms);
/** Server handshake on an owned TCP stream, with the same ownership rule. */
int64_t eshkol_tls_server_start(int64_t context, int64_t net_handle, int32_t timeout_ms);
/** Query a TLS listener's selected local port. */
int32_t eshkol_tls_listener_port(int64_t listener);
/** Read plaintext bytes; zero indicates authenticated TLS close-notify/EOF. */
int32_t eshkol_tls_read(int64_t session, void *buffer, int32_t capacity,
                        int32_t timeout_ms);
/** Write plaintext bytes, bounded by one total operation deadline. */
int32_t eshkol_tls_write(int64_t session, const void *buffer, int32_t length,
                         int32_t timeout_ms);
/** Byte-counted plaintext I/O on bytevector ranges, at most 1 MiB per call. */
int32_t eshkol_tls_read_bytes(int64_t session, void *bytevector, int64_t start, int64_t count,
                              int32_t timeout_ms);
int32_t eshkol_tls_write_bytes(int64_t session, const void *bytevector, int64_t start,
                               int64_t count, int32_t timeout_ms);
/** Write the whole range under one deadline; returns count or an outcome code. */
int64_t eshkol_tls_write_all_bytes(int64_t session, const void *bytevector, int64_t start,
                                   int64_t count, int32_t timeout_ms);
/** Cancel the session's waiting operation (or its next one); any thread. */
int32_t eshkol_tls_cancel(int64_t session);
/** Close a TLS session or listener once; stale identifiers return -4. */
int32_t eshkol_tls_session_close(int64_t session);
int32_t eshkol_tls_listener_close(int64_t listener);
#ifdef __cplusplus
}
#endif
#endif
