/**
 * @file object_pool.h
 * @brief Object pool for efficient allocation of fixed-size objects
 * 
 * This file defines the object pool interface for Eshkol,
 * which provides efficient allocation and deallocation of fixed-size objects.
 */

#ifndef ESHKOL_OBJECT_POOL_H
#define ESHKOL_OBJECT_POOL_H

#include "core/memory.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Object pool for efficient allocation of fixed-size objects
 * 
 * The object pool provides efficient allocation and deallocation of
 * fixed-size objects by reusing memory and reducing fragmentation.
 */
typedef struct ObjectPool ObjectPool;

/**
 * @brief Create a new object pool
 * 
 * @param arena Arena to allocate from
 * @param object_size Size of each object in bytes
 * @param objects_per_chunk Number of objects per chunk
 * @return A new object pool, or NULL on failure
 */
ObjectPool* object_pool_create(Arena* arena, size_t object_size, size_t objects_per_chunk);

/**
 * @brief Allocate an object from the pool
 * 
 * @param pool The object pool
 * @return Pointer to the allocated object, or NULL on failure
 */
void* object_pool_alloc(ObjectPool* pool);

/**
 * @brief Free an object back to the pool
 * 
 * @param pool The object pool
 * @param object Pointer to the object to free
 */
void object_pool_free(ObjectPool* pool, void* object);

/**
 * @brief Reset the object pool
 * 
 * Resets the object pool, making all objects available for allocation.
 * This does not free any memory allocated by the pool.
 * 
 * @param pool The object pool
 */
void object_pool_reset(ObjectPool* pool);

/**
 * @brief Get the number of allocated objects
 * 
 * @param pool The object pool
 * @return Number of allocated objects
 */
size_t object_pool_get_allocated_count(ObjectPool* pool);

/**
 * @brief Get the number of free objects
 * 
 * @param pool The object pool
 * @return Number of free objects
 */
size_t object_pool_get_free_count(ObjectPool* pool);

/**
 * @brief Get the total capacity of the pool
 * 
 * @param pool The object pool
 * @return Total capacity of the pool
 */
size_t object_pool_get_capacity(ObjectPool* pool);

/**
 * @brief Get the size of each object
 * 
 * @param pool The object pool
 * @return Size of each object in bytes
 */
size_t object_pool_get_object_size(ObjectPool* pool);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_OBJECT_POOL_H */
