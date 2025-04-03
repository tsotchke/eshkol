/**
 * @file simd.h
 * @brief SIMD detection and capability utilities for Eshkol
 * 
 * This file defines utilities for detecting SIMD capabilities and
 * selecting the most efficient implementation.
 */

#ifndef ESHKOL_SIMD_H
#define ESHKOL_SIMD_H

#include <stdbool.h>
#include <xmmintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SIMD instruction set flags
 */
typedef enum {
    SIMD_NONE     = 0,      /**< No SIMD instructions */
    SIMD_SSE      = 1 << 0, /**< SSE instructions */
    SIMD_SSE2     = 1 << 1, /**< SSE2 instructions */
    SIMD_SSE3     = 1 << 2, /**< SSE3 instructions */
    SIMD_SSSE3    = 1 << 3, /**< SSSE3 instructions */
    SIMD_SSE4_1   = 1 << 4, /**< SSE4.1 instructions */
    SIMD_SSE4_2   = 1 << 5, /**< SSE4.2 instructions */
    SIMD_AVX      = 1 << 6, /**< AVX instructions */
    SIMD_AVX2     = 1 << 7, /**< AVX2 instructions */
    SIMD_AVX512F  = 1 << 8, /**< AVX-512 Foundation instructions */
    SIMD_NEON     = 1 << 9, /**< ARM NEON instructions */
} SimdFlags;

/**
 * @brief SIMD capability information
 */
typedef struct {
    SimdFlags flags;        /**< SIMD instruction set flags */
    const char* name;       /**< CPU name */
    int vector_size;        /**< Vector size in bytes */
    bool has_fma;           /**< FMA instructions */
    bool has_popcnt;        /**< POPCNT instruction */
} SimdInfo;

/**
 * @brief Initialize SIMD detection
 * 
 * This function initializes SIMD detection and must be called before
 * any other SIMD functions.
 */
void simd_init(void);

/**
 * @brief Get SIMD capability information
 * 
 * @return SIMD capability information
 */
const SimdInfo* simd_get_info(void);

/**
 * @brief Check if a SIMD instruction set is supported
 * 
 * @param flags SIMD instruction set flags to check
 * @return true if all specified instruction sets are supported, false otherwise
 */
bool simd_is_supported(SimdFlags flags);

/**
 * @brief Get the best SIMD implementation for a function
 * 
 * This function returns the best SIMD implementation for a function
 * based on the available SIMD instruction sets.
 * 
 * @param generic Generic implementation (no SIMD)
 * @param sse SSE implementation
 * @param avx AVX implementation
 * @param avx2 AVX2 implementation
 * @param avx512 AVX-512 implementation
 * @param neon NEON implementation
 * @return The best implementation for the current CPU
 */
void* simd_get_best_impl(void* generic, void* sse, void* avx, void* avx2, void* avx512, void* neon);

/**
 * @brief Print SIMD capability information
 * 
 * This function prints SIMD capability information to stdout.
 */
void simd_print_info(void);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_SIMD_H */
