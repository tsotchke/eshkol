#ifndef QUANTUM_RNG_H
#define QUANTUM_RNG_H

#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <sys/time.h>

/**
 * @file quantum_rng.h
 * @brief Quantum-inspired random number generator
 *
 * This library implements a high-performance random number generator that leverages
 * quantum computing principles on classical hardware. It uses quantum circuit simulation,
 * floating-point uncertainty, and quantum entanglement to achieve high-quality
 * non-deterministic random numbers.
 */

/**
 * @brief Core quantum simulation parameters
 */
#define QRNG_NUM_QUBITS 8              /**< Number of qubits in quantum circuit */
#define QRNG_STATE_MULTIPLIER 16       /**< State vector size multiplier */
#define QRNG_STATE_SIZE (QRNG_NUM_QUBITS * QRNG_STATE_MULTIPLIER)  /**< Total state size */
#define QRNG_BUFFER_SIZE QRNG_STATE_SIZE  /**< Internal buffer size */
#define QRNG_MIXING_ROUNDS 4           /**< Number of quantum mixing rounds */

/**
 * @brief Error codes returned by library functions
 */
typedef enum {
    QRNG_SUCCESS = 0,                  /**< Operation completed successfully */
    QRNG_ERROR_NULL_CONTEXT = -1,      /**< NULL context provided */
    QRNG_ERROR_NULL_BUFFER = -2,       /**< NULL buffer provided */
    QRNG_ERROR_INVALID_LENGTH = -3,    /**< Invalid length parameter */
    QRNG_ERROR_INSUFFICIENT_ENTROPY = -4, /**< Not enough entropy available */
    QRNG_ERROR_INVALID_RANGE = -5      /**< Invalid range parameters */
} qrng_error;

/**
 * @brief Context structure for the RNG state
 */
typedef struct qrng_ctx_t {
    uint64_t phase[QRNG_NUM_QUBITS];
    uint64_t entangle[QRNG_NUM_QUBITS];
    double quantum_state[QRNG_NUM_QUBITS];
    uint64_t last_measurement[QRNG_NUM_QUBITS];
    union {
        uint8_t bytes[QRNG_BUFFER_SIZE];
        uint64_t words[QRNG_BUFFER_SIZE / sizeof(uint64_t)];
    } buffer;
    size_t buffer_pos;
    uint64_t counter;
    double entropy_pool[16];
    uint64_t pool_mixer;
    uint8_t pool_index;
    struct timeval init_time;
    pid_t pid;
    uint64_t unique_id;
    uint64_t system_entropy;
    uint64_t runtime_entropy;
} qrng_ctx;

/**
 * @brief Initialize a new RNG context
 *
 * Creates and initializes a new quantum RNG context with the given seed.
 * The context must be freed with qrng_free() when no longer needed.
 *
 * @param ctx[out] Pointer to context pointer to initialize
 * @param seed[in] Seed data for initialization
 * @param seed_len Length of seed data in bytes
 * @return QRNG_SUCCESS on success, error code on failure
 */
qrng_error qrng_init(qrng_ctx **ctx, const uint8_t *seed, size_t seed_len);

/**
 * @brief Free an RNG context
 *
 * Frees all resources associated with the given context.
 *
 * @param ctx Context to free
 */
void qrng_free(qrng_ctx *ctx);

/**
 * @brief Reseed an existing RNG context
 *
 * Updates the internal state with new seed data.
 *
 * @param ctx Context to reseed
 * @param seed New seed data
 * @param seed_len Length of new seed data
 * @return QRNG_SUCCESS on success, error code on failure
 */
qrng_error qrng_reseed(qrng_ctx *ctx, const uint8_t *seed, size_t seed_len);

/**
 * @brief Generate random bytes
 *
 * Fills the output buffer with random bytes generated using quantum simulation.
 *
 * @param ctx RNG context
 * @param out Output buffer to fill
 * @param len Number of bytes to generate
 * @return QRNG_SUCCESS on success, error code on failure
 */
qrng_error qrng_bytes(qrng_ctx *ctx, uint8_t *out, size_t len);

/**
 * @brief Generate a random 64-bit unsigned integer
 *
 * @param ctx RNG context
 * @return Random uint64_t value
 */
uint64_t qrng_uint64(qrng_ctx *ctx);

/**
 * @brief Generate a random double in [0,1)
 *
 * @param ctx RNG context
 * @return Random double between 0 (inclusive) and 1 (exclusive)
 */
double qrng_double(qrng_ctx *ctx);

/**
 * @brief Generate a random integer in [min,max]
 *
 * @param ctx RNG context
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random integer in the specified range
 */
int32_t qrng_range32(qrng_ctx *ctx, int32_t min, int32_t max);

/**
 * @brief Generate a random unsigned 64-bit integer in [min,max]
 *
 * @param ctx RNG context
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @return Random integer in the specified range, or max+1 on error
 */
uint64_t qrng_range64(qrng_ctx *ctx, uint64_t min, uint64_t max);

/**
 * @brief Get entropy estimate
 *
 * Returns an estimate of the entropy per bit in the RNG output.
 *
 * @param ctx RNG context
 * @return Estimated entropy per bit (between 0 and 1)
 */
double qrng_get_entropy_estimate(qrng_ctx *ctx);

/**
 * @brief Entangle two quantum states
 *
 * Creates quantum entanglement between two state buffers, causing their
 * measurements to be correlated.
 *
 * @param ctx RNG context
 * @param state1 First state buffer
 * @param state2 Second state buffer
 * @param len Length of state buffers
 * @return QRNG_SUCCESS on success, error code on failure
 */
qrng_error qrng_entangle_states(qrng_ctx *ctx, uint8_t *state1, uint8_t *state2, size_t len);

/**
 * @brief Measure a quantum state
 *
 * Performs a quantum measurement on the given state buffer, collapsing
 * superpositions into definite values.
 *
 * @param ctx RNG context
 * @param state State buffer to measure
 * @param len Length of state buffer
 * @return QRNG_SUCCESS on success, error code on failure
 */
qrng_error qrng_measure_state(qrng_ctx *ctx, uint8_t *state, size_t len);

/**
 * @brief Version information
 */
#define QRNG_VERSION_MAJOR 1  /**< Major version number */
#define QRNG_VERSION_MINOR 1  /**< Minor version number */
#define QRNG_VERSION_PATCH 0  /**< Patch version number */

/**
 * @brief Get version string
 *
 * @return Version string in format "major.minor.patch"
 */
const char* qrng_version(void);

/**
 * @brief Get error string
 *
 * Returns a human-readable description of an error code.
 *
 * @param err Error code to describe
 * @return Error description string
 */
const char* qrng_error_string(qrng_error err);

#endif /* QUANTUM_RNG_H */
