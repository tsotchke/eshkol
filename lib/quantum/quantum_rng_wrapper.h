#ifndef QUANTUM_RNG_WRAPPER_H
#define QUANTUM_RNG_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

/**
 * @file quantum_rng_wrapper.h
 * @brief Eshkol runtime entry points behind the `quantum-random` family of builtins.
 *
 * HONESTY NOTICE: as configured by default (ESHKOL_QUANTUM_ENABLED=OFF, and
 * always on WASM/browser builds), every function in this header is backed by
 * the classical software PRNG in quantum_rng.c/.h - NOT real hardware quantum
 * entropy, and NOT Bell-verified. See quantum_rng.h's file comment for the
 * full disclosure. Call eshkol_qrng_source_label() to find out, at runtime,
 * which source is actually wired up in the build you are running - do not
 * assume "quantum" from the function/builtin names alone.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the global quantum RNG context (called once automatically)
 * @return 0 on success, non-zero on error
 */
int eshkol_qrng_init(void);

/**
 * @brief Get a random double in [0, 1)
 * @return Random double between 0 (inclusive) and 1 (exclusive)
 */
double eshkol_qrng_double(void);

/**
 * @brief Get a random 64-bit unsigned integer
 * @return Random uint64_t value
 */
uint64_t eshkol_qrng_uint64(void);

/**
 * @brief Get a random integer in range [min, max]
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random integer in the specified range
 */
int64_t eshkol_qrng_range(int64_t min, int64_t max);

/**
 * @brief Get random bytes
 * @param buffer Output buffer
 * @param len Number of bytes to generate
 * @return 0 on success, non-zero on error
 */
int eshkol_qrng_bytes(uint8_t* buffer, size_t len);

/**
 * @brief Get the quantum RNG version string
 * @return Version string
 */
const char* eshkol_qrng_version(void);

/**
 * @brief Report which entropy source actually backs the eshkol_qrng_* /
 *        `quantum-random` family in this build, honestly.
 *
 * This is the machine-readable half of the honesty fix: rather than trusting
 * the "quantum" name of the builtins, callers (and test/CI gates) can check
 * this label at runtime.
 *
 * @return A static, NUL-terminated string, one of:
 *   - "classical-fallback" - the vendored quantum_rng.c software PRNG
 *     (wall-clock/pid/stack-address seeded, deterministic given a fixed
 *     seed). This is the value in every build with ESHKOL_QUANTUM_ENABLED=OFF
 *     (the default) and on every WASM/browser build. NOT real quantum
 *     entropy, NOT Bell-verified.
 *   - "moonlab-qrng" - Moonlab's Bell-verified `moonlab_qrng_bytes()` v3 QRNG
 *     engine, combining hardware entropy (RDSEED / /dev/urandom /
 *     SecRandomCopyBytes) with a verified quantum-simulation layer. Only
 *     possible when built with -DESHKOL_QUANTUM_ENABLED=ON and the runtime
 *     successfully routed to it (see Part D of docs/design/MOONLAB_INTEGRATION.md).
 */
const char* eshkol_qrng_source_label(void);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_RNG_WRAPPER_H */
