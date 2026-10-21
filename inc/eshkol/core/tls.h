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
#ifdef __cplusplus
}
#endif
#endif
