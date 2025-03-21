/**
 * @file arena.c
 * @brief Implementation of the arena allocator
 */

#include "eshkol/core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Memory block in the arena
 * 
 * The arena is implemented as a linked list of blocks.
 * Each block has a header followed by the data.
 */
typedef struct Block {
    struct Block* next;  /**< Next block in the list */
    size_t size;         /**< Size of the data area in bytes */
    size_t used;         /**< Number of bytes used in the data area */
    uintptr_t data[];    /**< Flexible array member for the data */
} Block;

/**
 * @brief Arena allocator structure
 */
struct Arena {
    Block* current;          /**< Current block for allocations */
    Block* first;            /**< First block in the list */
    size_t total_used;       /**< Total bytes used across all blocks */
    size_t allocation_count; /**< Number of allocations made */
    size_t min_block_size;   /**< Minimum block size for new blocks */
};

Arena* arena_create(size_t initial_capacity) {
    size_t min_size = 1024;  // Minimum 1KB
    size_t block_size = initial_capacity > min_size ? initial_capacity : min_size;
    
    // Allocate the arena control structure
    Arena* arena = malloc(sizeof(Arena));
    if (!arena) return NULL;
    
    // Allocate the initial block
    size_t header_size = sizeof(Block);
    Block* block = malloc(header_size + block_size);
    if (!block) {
        free(arena);
        return NULL;
    }
    
    // Initialize the block
    block->next = NULL;
    block->size = block_size;
    block->used = 0;
    
    // Initialize the arena
    arena->current = block;
    arena->first = block;
    arena->total_used = 0;
    arena->allocation_count = 0;
    arena->min_block_size = min_size;
    
    return arena;
}

/**
 * @brief Allocate memory from a specific block
 * 
 * @param arena The arena
 * @param block The block to allocate from
 * @param size Size in bytes to allocate
 * @param alignment Alignment requirement (must be a power of 2)
 * @return Pointer to the allocated memory, or NULL if the block doesn't have enough space
 */
static void* arena_alloc_from_block(Arena* arena, Block* block, size_t size, size_t alignment) {
    assert(arena != NULL);
    assert(block != NULL);
    assert(size > 0);
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0);  // Power of 2
    
    // Calculate aligned address
    uintptr_t address = (uintptr_t)block->data + block->used;
    uintptr_t aligned = (address + alignment - 1) & ~(alignment - 1);
    size_t padding = aligned - address;
    
    // Check if enough space in block
    if (block->used + padding + size > block->size) {
        return NULL;
    }
    
    // Update block usage
    block->used += padding + size;
    arena->total_used += padding + size;
    arena->allocation_count++;
    
    return (void*)aligned;
}

/**
 * @brief Add a new block to the arena
 * 
 * @param arena The arena
 * @param min_size Minimum size needed for the new block
 * @return The new block, or NULL on failure
 */
static Block* arena_add_block(Arena* arena, size_t min_size) {
    assert(arena != NULL);
    assert(min_size > 0);
    
    // Double the block size for each new block, but at least min_size
    size_t new_size = arena->current->size * 2;
    if (new_size < min_size) new_size = min_size;
    
    // Allocate new block
    size_t header_size = sizeof(Block);
    Block* block = malloc(header_size + new_size);
    if (!block) return NULL;
    
    // Initialize the block
    block->next = NULL;
    block->size = new_size;
    block->used = 0;
    
    // Add to arena
    arena->current->next = block;
    arena->current = block;
    
    return block;
}

void* arena_alloc(Arena* arena, size_t size) {
    return arena_alloc_aligned(arena, size, sizeof(void*));
}

void* arena_alloc_aligned(Arena* arena, size_t size, size_t alignment) {
    assert(arena != NULL);
    assert(size > 0);
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0);  // Power of 2
    
    // Try to allocate from current block
    void* result = arena_alloc_from_block(arena, arena->current, size, alignment);
    if (result) return result;
    
    // Need a new block
    size_t min_size = size + alignment - 1;  // Ensure enough space for alignment
    Block* block = arena_add_block(arena, min_size);
    if (!block) return NULL;
    
    // Allocate from the new block
    result = arena_alloc_from_block(arena, block, size, alignment);
    assert(result != NULL);  // Should always succeed
    
    return result;
}

void arena_reset(Arena* arena) {
    assert(arena != NULL);
    
    // Reset all blocks
    Block* block = arena->first;
    while (block) {
        block->used = 0;
        block = block->next;
    }
    
    // Reset arena stats
    arena->total_used = 0;
    arena->allocation_count = 0;
}

void arena_destroy(Arena* arena) {
    assert(arena != NULL);
    
    // Free all blocks
    Block* block = arena->first;
    while (block) {
        Block* next = block->next;
        free(block);
        block = next;
    }
    
    // Free arena control structure
    free(arena);
}

size_t arena_get_used_memory(const Arena* arena) {
    assert(arena != NULL);
    return arena->total_used;
}

size_t arena_get_allocation_count(const Arena* arena) {
    assert(arena != NULL);
    return arena->allocation_count;
}
