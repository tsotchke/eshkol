/* Focused C API checks for optional native TLS context construction. */
#include "eshkol/core/tls.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    int32_t available = eshkol_tls_available();
    assert(available == 0 || available == 1);
    if (!available) {
        assert(eshkol_tls_client_context(NULL) == -3);
        assert(eshkol_tls_server_context("cert.pem", "key.pem") == -3);
        assert(eshkol_tls_context_close(1) == -3);
        puts("PASS: unavailable TLS fails explicitly");
        return 0;
    }

    int64_t client = eshkol_tls_client_context(NULL);
    assert(client > 0 && client <= INT64_C(9007199254740991));
    assert(eshkol_tls_context_close(client) == 0);
    assert(eshkol_tls_context_close(client) == -4);
    assert(eshkol_tls_client_context("/definitely/missing/eshkol-ca.pem") == -1);
    assert(eshkol_tls_server_context(NULL, NULL) == -4);
    assert(eshkol_tls_server_context("/definitely/missing/cert.pem",
                                     "/definitely/missing/key.pem") == -1);

    int64_t next = eshkol_tls_client_context(NULL);
    assert(next > client && next <= INT64_C(9007199254740991));
    assert(eshkol_tls_context_close(next) == 0);
    puts("PASS: TLS client/server context validation and stale handle refusal");
    return 0;
}
