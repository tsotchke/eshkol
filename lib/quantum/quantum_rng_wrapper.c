#include "quantum_rng_wrapper.h"
#include "quantum_rng.h"
#include <stdlib.h>
#include <string.h>

// Global quantum RNG context
static qrng_ctx* g_qrng_ctx = NULL;
static int g_qrng_initialized = 0;

int eshkol_qrng_init(void) {
    if (g_qrng_initialized) {
        return 0;  // Already initialized
    }

    qrng_error err = qrng_init(&g_qrng_ctx, NULL, 0);
    if (err != QRNG_SUCCESS) {
        return (int)err;
    }

    g_qrng_initialized = 1;
    return 0;
}

// Ensure initialization before any operation
static inline void ensure_init(void) {
    if (!g_qrng_initialized) {
        eshkol_qrng_init();
    }
}

double eshkol_qrng_double(void) {
    ensure_init();
    return qrng_double(g_qrng_ctx);
}

uint64_t eshkol_qrng_uint64(void) {
    ensure_init();
    return qrng_uint64(g_qrng_ctx);
}

int64_t eshkol_qrng_range(int64_t min, int64_t max) {
    ensure_init();
    if (min > max) {
        int64_t tmp = min;
        min = max;
        max = tmp;
    }
    // Use 64-bit range function
    uint64_t umin = (uint64_t)min;
    uint64_t umax = (uint64_t)max;
    uint64_t result = qrng_range64(g_qrng_ctx, umin, umax);
    return (int64_t)result;
}

int eshkol_qrng_bytes(uint8_t* buffer, size_t len) {
    ensure_init();
    return (int)qrng_bytes(g_qrng_ctx, buffer, len);
}

const char* eshkol_qrng_version(void) {
    return qrng_version();
}
