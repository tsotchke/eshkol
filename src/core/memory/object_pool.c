/**
 * @file object_pool.c
 * @brief Implementation of the object pool for efficient allocation of fixed-size objects
 */

#include "core/object_pool.h"
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Chunk of objects in the pool
 */
typedef struct Chunk {
    struct Chunk* next;    /**< Next chunk in the list */
    size_t size;           /**< Size of the chunk in bytes */
    char data[];           /**< Flexible array member for object data */
} Chunk;

/**
 * @brief Free list entry
 * 
 * When an object is freed, it's added to the free list.
 * The first few bytes of the object are used to store a pointer
 * to the next free object.
 */
typedef struct FreeListEntry {
    struct FreeListEntry* next;
} FreeListEntry;

/**
 * @brief Object pool structure
 */
struct ObjectPool {
    Arena* arena;                /**< Arena for allocations */
    Chunk* chunks;               /**< List of chunks */
    FreeListEntry* free_list;    /**< List of free objects */
    size_t object_size;          /**< Size of each object in bytes */
    size_t objects_per_chunk;    /**< Number of objects per chunk */
    size_t allocated_count;      /**< Number of allocated objects */
    size_t free_count;           /**< Number of free objects */
    size_t total_capacity;       /**< Total capacity of the pool */
};

ObjectPool* object_pool_create(Arena* arena, size_t object_size, size_t objects_per_chunk) {
    assert(arena != NULL);
    assert(object_size >= sizeof(FreeListEntry));
    assert(objects_per_chunk > 0);
    
    // Allocate object pool
    ObjectPool* pool = arena_alloc(arena, sizeof(ObjectPool));
    if (!pool) return NULL;
    
    // Initialize pool
    pool->arena = arena;
    pool->chunks = NULL;
    pool->free_list = NULL;
    pool->object_size = object_size;
    pool->objects_per_chunk = objects_per_chunk;
    pool->allocated_count = 0;
    pool->free_count = 0;
    pool->total_capacity = 0;
    
    return pool;
}

/**
 * @brief Allocate a new chunk of objects
 * 
 * @param pool The object pool
 * @return true if successful, false otherwise
 */
static bool allocate_chunk(ObjectPool* pool) {
    assert(pool != NULL);
    
    // Calculate chunk size
    size_t data_size = pool->object_size * pool->objects_per_chunk;
    size_t chunk_size = sizeof(Chunk) + data_size;
    
    // Allocate chunk
    Chunk* chunk = arena_alloc(pool->arena, chunk_size);
    if (!chunk) return false;
    
    // Initialize chunk
    chunk->next = pool->chunks;
    chunk->size = data_size;
    
    // Add chunk to pool
    pool->chunks = chunk;
    
    // Add objects to free list
    char* data = chunk->data;
    for (size_t i = 0; i < pool->objects_per_chunk; i++) {
        FreeListEntry* entry = (FreeListEntry*)(data + i * pool->object_size);
        entry->next = pool->free_list;
        pool->free_list = entry;
    }
    
    // Update counts
    pool->free_count += pool->objects_per_chunk;
    pool->total_capacity += pool->objects_per_chunk;
    
    return true;
}

void* object_pool_alloc(ObjectPool* pool) {
    assert(pool != NULL);
    
    // If free list is empty, allocate a new chunk
    if (!pool->free_list) {
        if (!allocate_chunk(pool)) {
            return NULL;
        }
    }
    
    // Get object from free list
    FreeListEntry* entry = pool->free_list;
    pool->free_list = entry->next;
    
    // Update counts
    pool->allocated_count++;
    pool->free_count--;
    
    // Clear the memory
    memset(entry, 0, pool->object_size);
    
    return entry;
}

void object_pool_free(ObjectPool* pool, void* object) {
    assert(pool != NULL);
    assert(object != NULL);
    
    // Add object to free list
    FreeListEntry* entry = (FreeListEntry*)object;
    entry->next = pool->free_list;
    pool->free_list = entry;
    
    // Update counts
    pool->allocated_count--;
    pool->free_count++;
}

void object_pool_reset(ObjectPool* pool) {
    assert(pool != NULL);
    
    // Reset free list
    pool->free_list = NULL;
    
    // Add all objects to free list
    Chunk* chunk = pool->chunks;
    while (chunk) {
        char* data = chunk->data;
        for (size_t i = 0; i < pool->objects_per_chunk; i++) {
            FreeListEntry* entry = (FreeListEntry*)(data + i * pool->object_size);
            entry->next = pool->free_list;
            pool->free_list = entry;
        }
        chunk = chunk->next;
    }
    
    // Update counts
    pool->allocated_count = 0;
    pool->free_count = pool->total_capacity;
}

size_t object_pool_get_allocated_count(ObjectPool* pool) {
    assert(pool != NULL);
    return pool->allocated_count;
}

size_t object_pool_get_free_count(ObjectPool* pool) {
    assert(pool != NULL);
    return pool->free_count;
}

size_t object_pool_get_capacity(ObjectPool* pool) {
    assert(pool != NULL);
    return pool->total_capacity;
}

size_t object_pool_get_object_size(ObjectPool* pool) {
    assert(pool != NULL);
    return pool->object_size;
}
