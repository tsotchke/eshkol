#ifndef QUANTUM_RNG_WRAPPER_H
#define QUANTUM_RNG_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

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

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_RNG_WRAPPER_H */
