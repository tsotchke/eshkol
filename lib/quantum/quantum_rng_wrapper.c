/*
 * quantum_rng_wrapper.c - Eshkol runtime entry points behind `quantum-random`,
 * shared by BOTH native backends (the LLVM AOT/JIT codegen calls these
 * eshkol_qrng_* symbols directly; the VM interpreter's vm_native.c dispatch
 * for the same builtins also calls these, not its own generator - see
 * lib/backend/vm_native.c's honesty notice above vm_qrng_next_u64 for that
 * history). Routing both backends through this single file is what makes
 * "the VM and AOT/JIT backends agree on the source" true.
 *
 * HONESTY NOTICE / two build configurations:
 *
 *  - ESHKOL_MOONLAB_QRNG_ENABLED not defined (default, and always on
 *    WASM/browser builds): every function below is backed by the classical
 *    software PRNG in quantum_rng.c - NOT real quantum hardware, NOT
 *    Bell-verified. See quantum_rng.h's file comment.
 *  - ESHKOL_MOONLAB_QRNG_ENABLED defined (set by CMake only when built with
 *    -DESHKOL_QUANTUM_ENABLED=ON and Moonlab's `quantumsim` target linked):
 *    every function below draws its raw bytes from Moonlab's
 *    moonlab_qrng_bytes() (applications/moonlab_export.h), which "combines a
 *    hardware entropy pool (RDSEED / /dev/urandom / SecRandomCopyBytes) with
 *    a Bell-verified quantum simulation layer" (Moonlab's own documentation
 *    of that function). On the rare documented failure of that call, this
 *    file falls back to the classical generator for just that one draw and
 *    prints a one-time diagnostic to stderr - never silently gives a
 *    same-looking-but-different-source result without saying so.
 *
 * eshkol_qrng_source_label() reports which configuration is active.
 */
#include "quantum_rng_wrapper.h"
#include "quantum_rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ESHKOL_MOONLAB_QRNG_ENABLED
#include "applications/moonlab_export.h"
#endif

// Global classical-PRNG context (see file-level honesty notice above). Used
// unconditionally as the entropy source when ESHKOL_MOONLAB_QRNG_ENABLED is
// not defined, and as a same-call fallback in the Moonlab-enabled path if
// moonlab_qrng_bytes() itself fails.
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

/** @brief Lazily initializes the global classical-PRNG context on first use. */
static inline void ensure_init(void) {
    if (!g_qrng_initialized) {
        eshkol_qrng_init();
    }
}

#ifdef ESHKOL_MOONLAB_QRNG_ENABLED
/**
 * @brief Draws sizeof(uint64_t) raw bytes from Moonlab's Bell-verified v3
 *        QRNG. On the rare documented failure (moonlab_qrng_bytes returning
 *        nonzero: NULL buffer, engine-init failure, or a rejected
 *        Bell-verification epoch), falls back to the classical generator for
 *        just this one draw and prints a one-time stderr diagnostic so a
 *        persistent failure is discoverable rather than silently masked.
 */
static uint64_t moonlab_next_u64(void) {
    uint64_t result = 0;
    int rc = moonlab_qrng_bytes((uint8_t*)&result, sizeof(result));
    if (rc != 0) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr,
                    "eshkol: moonlab_qrng_bytes failed (rc=%d); falling back to the "
                    "classical PRNG for this draw only. quantum-random-source still "
                    "reports moonlab-qrng because that is this build's configured "
                    "source -- a persistent failure will keep printing this warning.\n",
                    rc);
            warned = 1;
        }
        ensure_init();
        qrng_bytes(g_qrng_ctx, (uint8_t*)&result, sizeof(result));
    }
    return result;
}
#endif /* ESHKOL_MOONLAB_QRNG_ENABLED */

uint64_t eshkol_qrng_uint64(void) {
#ifdef ESHKOL_MOONLAB_QRNG_ENABLED
    return moonlab_next_u64();
#else
    ensure_init();
    return qrng_uint64(g_qrng_ctx);
#endif
}

double eshkol_qrng_double(void) {
    /* Standard "top 53 bits -> [0,1)" conversion (same formula
     * quantum_rng.c's own qrng_double uses), applied uniformly to whichever
     * source eshkol_qrng_uint64() drew from this build. */
    return (double)(eshkol_qrng_uint64() >> 11) * (1.0 / 9007199254740992.0);
}

int64_t eshkol_qrng_range(int64_t min, int64_t max) {
    if (min > max) {
        int64_t tmp = min;
        min = max;
        max = tmp;
    }
    uint64_t umin = (uint64_t)min;
    uint64_t umax = (uint64_t)max;
    if (umin == umax) {
        return (int64_t)umin;
    }
    uint64_t range = umax - umin + 1; /* 0 only if [min,max] spans the full uint64 range */
    if (range == 0) {
        return (int64_t)eshkol_qrng_uint64();
    }
    /* Rejection sampling for an unbiased draw in [0, range) -- the same
     * technique quantum_rng.c's own qrng_range64 uses -- built on top of
     * eshkol_qrng_uint64() so this logic (and its bias-freedom) is identical
     * regardless of which source backs this build. */
    uint64_t threshold = (uint64_t)(-range) % range;
    uint64_t r;
    do {
        r = eshkol_qrng_uint64();
    } while (r < threshold);
    return (int64_t)(umin + (r % range));
}

int eshkol_qrng_bytes(uint8_t* buffer, size_t len) {
#ifdef ESHKOL_MOONLAB_QRNG_ENABLED
    int rc = moonlab_qrng_bytes(buffer, len);
    if (rc == 0) return 0;
    static int warned = 0;
    if (!warned) {
        fprintf(stderr,
                "eshkol: moonlab_qrng_bytes failed (rc=%d) for eshkol_qrng_bytes; "
                "falling back to the classical PRNG for this call only.\n", rc);
        warned = 1;
    }
    ensure_init();
    return (int)qrng_bytes(g_qrng_ctx, buffer, len);
#else
    ensure_init();
    return (int)qrng_bytes(g_qrng_ctx, buffer, len);
#endif
}

const char* eshkol_qrng_version(void) {
    return qrng_version();
}

const char* eshkol_qrng_source_label(void) {
#if defined(ESHKOL_MOONLAB_QRNG_ENABLED)
    return "moonlab-qrng";
#else
    return "classical-fallback";
#endif
}
