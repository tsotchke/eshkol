/**
 * @file memory_tracking.h
 * @brief Memory tracking utilities for Eshkol
 * 
 * This file defines the memory tracking interface for Eshkol,
 * which provides utilities for tracking memory allocations and detecting leaks.
 */

#ifndef ESHKOL_MEMORY_TRACKING_H
#define ESHKOL_MEMORY_TRACKING_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Memory tracking statistics
 */
typedef struct {
    size_t total_allocations;      /**< Total number of allocations */
    size_t active_allocations;     /**< Number of active allocations */
    size_t total_bytes_allocated;  /**< Total bytes allocated */
    size_t active_bytes_allocated; /**< Active bytes allocated */
    size_t peak_bytes_allocated;   /**< Peak bytes allocated */
} MemoryStats;

/**
 * @brief Initialize memory tracking
 * 
 * This function initializes the memory tracking system.
 * It should be called before any memory allocation.
 */
void memory_tracking_init(void);

/**
 * @brief Shutdown memory tracking
 * 
 * This function shuts down the memory tracking system.
 * It should be called after all memory has been freed.
 * 
 * @return true if no memory leaks were detected, false otherwise
 */
bool memory_tracking_shutdown(void);

/**
 * @brief Track a memory allocation
 * 
 * This function tracks a memory allocation.
 * 
 * @param ptr Pointer to the allocated memory
 * @param size Size of the allocation in bytes
 * @param file Source file where the allocation occurred
 * @param line Line number where the allocation occurred
 * @param func Function where the allocation occurred
 */
void memory_tracking_track_alloc(void* ptr, size_t size, const char* file, int line, const char* func);

/**
 * @brief Track a memory deallocation
 * 
 * This function tracks a memory deallocation.
 * 
 * @param ptr Pointer to the memory being freed
 * @param file Source file where the deallocation occurred
 * @param line Line number where the deallocation occurred
 * @param func Function where the deallocation occurred
 */
void memory_tracking_track_free(void* ptr, const char* file, int line, const char* func);

/**
 * @brief Get memory tracking statistics
 * 
 * @return Memory tracking statistics
 */
MemoryStats memory_tracking_get_stats(void);

/**
 * @brief Print memory tracking statistics
 */
void memory_tracking_print_stats(void);

/**
 * @brief Print memory leak report
 * 
 * This function prints a report of all memory leaks.
 * 
 * @return Number of memory leaks detected
 */
size_t memory_tracking_print_leaks(void);

/**
 * @brief Enable or disable memory tracking
 * 
 * @param enabled true to enable memory tracking, false to disable
 */
void memory_tracking_set_enabled(bool enabled);

/**
 * @brief Check if memory tracking is enabled
 * 
 * @return true if memory tracking is enabled, false otherwise
 */
bool memory_tracking_is_enabled(void);

/**
 * @brief Macro for tracking memory allocations
 * 
 * This macro should be used to track memory allocations.
 * It automatically captures the source file, line number, and function name.
 * 
 * @param ptr Pointer to the allocated memory
 * @param size Size of the allocation in bytes
 */
#define MEMORY_TRACK_ALLOC(ptr, size) \
    memory_tracking_track_alloc(ptr, size, __FILE__, __LINE__, __func__)

/**
 * @brief Macro for tracking memory deallocations
 * 
 * This macro should be used to track memory deallocations.
 * It automatically captures the source file, line number, and function name.
 * 
 * @param ptr Pointer to the memory being freed
 */
#define MEMORY_TRACK_FREE(ptr) \
    memory_tracking_track_free(ptr, __FILE__, __LINE__, __func__)

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_MEMORY_TRACKING_H */
