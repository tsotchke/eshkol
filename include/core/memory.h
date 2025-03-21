/**
 * @file memory.h
 * @brief Memory management system for Eshkol
 * 
 * This file defines the memory management interfaces for Eshkol,
 * including the arena allocator for efficient memory allocation.
 */

#ifndef ESHKOL_MEMORY_H
#define ESHKOL_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Arena allocator for efficient memory management
 * 
 * The arena allocator provides fast allocation with bulk deallocation.
 * It's designed for the allocation patterns common in compilers, where
 * many small objects are allocated during a phase and then freed together.
 */
typedef struct Arena Arena;

/**
 * @brief Create a new arena with the specified initial capacity
 * 
 * @param initial_capacity Initial capacity in bytes
 * @return A new arena, or NULL on failure
 */
Arena* arena_create(size_t initial_capacity);

/**
 * @brief Allocate memory from the arena
 * 
 * @param arena The arena to allocate from
 * @param size Size in bytes to allocate
 * @return Pointer to the allocated memory, or NULL on failure
 */
void* arena_alloc(Arena* arena, size_t size);

/**
 * @brief Allocate aligned memory from the arena
 * 
 * @param arena The arena to allocate from
 * @param size Size in bytes to allocate
 * @param alignment Alignment requirement (must be a power of 2)
 * @return Pointer to the allocated memory, or NULL on failure
 */
void* arena_alloc_aligned(Arena* arena, size_t size, size_t alignment);

/**
 * @brief Reset the arena, freeing all allocations at once
 * 
 * This operation is very fast as it simply resets the arena's internal
 * state without actually freeing any memory. The memory can be reused
 * for new allocations.
 * 
 * @param arena The arena to reset
 */
void arena_reset(Arena* arena);

/**
 * @brief Destroy the arena and free all memory
 * 
 * @param arena The arena to destroy
 */
void arena_destroy(Arena* arena);

/**
 * @brief Get the current memory usage of the arena
 * 
 * @param arena The arena to query
 * @return Current memory usage in bytes
 */
size_t arena_get_used_memory(const Arena* arena);

/**
 * @brief Get the number of allocations made from the arena
 * 
 * @param arena The arena to query
 * @return Number of allocations
 */
size_t arena_get_allocation_count(const Arena* arena);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_MEMORY_H */
