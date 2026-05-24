/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Arena Memory Management Implementation
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/eshkol.h"
#include "../../inc/eshkol/core/bignum.h"
#include "../../inc/eshkol/core/logic.h"
#include "../../inc/eshkol/core/inference.h"
#include "../../inc/eshkol/core/workspace.h"
#include "../../inc/eshkol/core/rational.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
#include <new>      // for std::bad_alloc
#include <stdexcept>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>      // std::memset
#endif

void* eshkol_arena_mutex_create(void);
void eshkol_arena_mutex_destroy(void* mutex);
void eshkol_arena_mutex_lock(void* mutex);
void eshkol_arena_mutex_unlock(void* mutex);
void eshkol_arena_global_once(void (*init)(void));

// Default alignment for memory allocations
#define DEFAULT_ALIGNMENT 8

// Global shared arena for REPL mode (persistent across evaluations)
// Atomic to synchronize writes (REPL init) and reads (runtime exception handlers)
extern "C" std::atomic<arena_t*> __repl_shared_arena{nullptr};

// Global command-line arguments (for (command-line) procedure)
// In REPL mode, these remain zero/null since there's no real main()
// Use weak linkage so generated code can override in standalone mode
__attribute__((weak)) int32_t __eshkol_argc = 0;
__attribute__((weak)) char** __eshkol_argv = nullptr;

// Utility function to align size to boundary
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Create a new arena block
static arena_block_t* create_arena_block(size_t size) {
    arena_block_t* block = (arena_block_t*)malloc(sizeof(arena_block_t));
    if (!block) {
        eshkol_error("Failed to allocate arena block structure");
        return nullptr;
    }
    
    block->memory = (uint8_t*)malloc(size);
    if (!block->memory) {
        eshkol_error("Failed to allocate arena block memory of size %zu", size);
        free(block);
        return nullptr;
    }
    
    block->size = size;
    block->used = 0;
    block->next = nullptr;
    
    return block;
}

// Free an arena block
static void free_arena_block(arena_block_t* block) {
    if (block) {
        free(block->memory);
        free(block);
    }
}

// Arena creation and destruction
arena_t* arena_create(size_t default_block_size) {
    if (default_block_size < 1024) {
        default_block_size = 1024; // Minimum block size
    }

    arena_t* arena = (arena_t*)malloc(sizeof(arena_t));
    if (!arena) {
        eshkol_error("Failed to allocate arena structure");
        return nullptr;
    }

    arena->current_block = create_arena_block(default_block_size);
    if (!arena->current_block) {
        free(arena);
        return nullptr;
    }

    arena->current_scope = nullptr;
    arena->default_block_size = default_block_size;
    arena->total_allocated = default_block_size;
    arena->alignment = DEFAULT_ALIGNMENT;
    arena->mutex = nullptr;
    arena->thread_safe = false;

    eshkol_debug("Created arena with default block size %zu", default_block_size);
    return arena;
}

// Thread-safe arena creation
arena_t* arena_create_threadsafe(size_t default_block_size) {
    arena_t* arena = arena_create(default_block_size);
    if (!arena) {
        return nullptr;
    }

    void* mutex = eshkol_arena_mutex_create();
    if (!mutex) {
        eshkol_error("Failed to create mutex for thread-safe arena");
        arena_destroy(arena);
        return nullptr;
    }

    arena->mutex = mutex;
    arena->thread_safe = true;

    eshkol_debug("Created thread-safe arena with default block size %zu", default_block_size);
    return arena;
}

// Thread-safety control functions
void arena_lock(arena_t* arena) {
    if (arena && arena->thread_safe && arena->mutex) {
        eshkol_arena_mutex_lock(arena->mutex);
    }
}

void arena_unlock(arena_t* arena) {
    if (arena && arena->thread_safe && arena->mutex) {
        eshkol_arena_mutex_unlock(arena->mutex);
    }
}

void arena_destroy(arena_t* arena) {
    if (!arena) return;

    // Destroy mutex if thread-safe
    if (arena->thread_safe && arena->mutex) {
        eshkol_arena_mutex_destroy(arena->mutex);
    }

    // Free all blocks
    arena_block_t* block = arena->current_block;
    while (block) {
        arena_block_t* next = block->next;
        free_arena_block(block);
        block = next;
    }

    // Free all scopes
    arena_scope_t* scope = arena->current_scope;
    while (scope) {
        arena_scope_t* parent = scope->parent;
        free(scope);
        scope = parent;
    }

    eshkol_debug("Destroyed arena, freed %zu bytes", arena->total_allocated);
    free(arena);
}

// Core allocation function (thread-safe if arena was created with arena_create_threadsafe)
void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return nullptr;

    // Lock if thread-safe arena
    arena_lock(arena);

    if (alignment == 0) alignment = DEFAULT_ALIGNMENT;

    // Align the size
    size_t aligned_size = align_size(size, alignment);

    // Check if current block has enough space
    arena_block_t* block = arena->current_block;
    size_t current_used = align_size(block->used, alignment);

    if (current_used + aligned_size > block->size) {
        // Need a new block
        size_t new_block_size = (aligned_size > arena->default_block_size) ?
                               aligned_size : arena->default_block_size;

        arena_block_t* new_block = create_arena_block(new_block_size);
        if (!new_block) {
            eshkol_error("Failed to allocate new arena block of size %zu", new_block_size);
            arena_unlock(arena);
            return nullptr;
        }

        // Link the new block to the front
        new_block->next = arena->current_block;
        arena->current_block = new_block;
        arena->total_allocated += new_block_size;

        block = new_block;
        current_used = 0;
    }

    // Allocate from current block
    void* ptr = block->memory + current_used;
    block->used = current_used + aligned_size;

    // Unlock if thread-safe arena
    arena_unlock(arena);

    return ptr;
}

void* arena_allocate(arena_t* arena, size_t size) {
    return arena_allocate_aligned(arena, size, DEFAULT_ALIGNMENT);
}

void* arena_allocate_zeroed(arena_t* arena, size_t size) {
    void* ptr = arena_allocate(arena, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// HEADER-AWARE ALLOCATION FUNCTIONS
// These functions allocate objects with an eshkol_object_header_t prepended.
// The returned pointer points to the DATA, not the header.
// Use ESHKOL_GET_HEADER(ptr) to access the header from the data pointer.
// Memory layout: [header (8 bytes)][object data (variable)]
// ═══════════════════════════════════════════════════════════════════════════

// Allocate object with header prepended
// Returns pointer to data (after header), or nullptr on failure
void* arena_allocate_with_header(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags) {
    if (!arena || data_size == 0) {
        return nullptr;
    }

    // #192 CRITICAL: integer-overflow guard on total_size. If data_size
    // is close to SIZE_MAX, adding sizeof(header) + alignment rounding
    // wraps to a small value; arena_allocate_aligned then returns a
    // buffer much smaller than the caller expects. Callers often derive
    // data_size from file contents (kb-load) or tensor shapes
    // (user-controlled), so any wrap yields an immediate heap-overflow
    // primitive.
    if (data_size > SIZE_MAX - sizeof(eshkol_object_header_t) - 8) {
        eshkol_error("arena_allocate_with_header: data_size=%zu would overflow", data_size);
        return nullptr;
    }

    // #192 HIGH: header->size is uint32_t. Silently truncating a 4GB+
    // allocation to its low 32 bits makes downstream readers think the
    // object is tiny and under-copy it. Reject rather than truncate.
    if (data_size > UINT32_MAX) {
        eshkol_error("arena_allocate_with_header: data_size=%zu exceeds UINT32_MAX", data_size);
        return nullptr;
    }

    // Total size: header + data, aligned to 8 bytes
    size_t total_size = sizeof(eshkol_object_header_t) + data_size;
    total_size = (total_size + 7) & ~7;  // Round up to 8-byte alignment

    // Allocate the full block
    void* raw = arena_allocate_aligned(arena, total_size, 8);
    if (!raw) {
        eshkol_error("Failed to allocate object with header (size=%zu)", data_size);
        return nullptr;
    }

    // Initialize header
    eshkol_object_header_t* header = (eshkol_object_header_t*)raw;
    header->subtype = subtype;
    header->flags = flags;
    header->ref_count = 0;
    header->size = (uint32_t)data_size;

    // Return pointer to data (after header)
    return (void*)((uint8_t*)raw + sizeof(eshkol_object_header_t));
}

// Allocate zeroed object with header
void* arena_allocate_with_header_zeroed(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags) {
    void* ptr = arena_allocate_with_header(arena, data_size, subtype, flags);
    if (ptr) {
        memset(ptr, 0, data_size);
    }
    return ptr;
}

// ───────────────────────────────────────────────────────────────────────────
// TYPED ALLOCATION HELPERS FOR NEW CONSOLIDATED TYPES
// These will be used for new allocations during and after migration
// ───────────────────────────────────────────────────────────────────────────

// Allocate a multi-value container (for multiple return values)
// count: number of tagged values to hold
void* arena_allocate_multi_value(arena_t* arena, size_t count) {
    // Overflow check: count * sizeof(eshkol_tagged_value_t) + sizeof(size_t)
    if (count > (SIZE_MAX - sizeof(size_t)) / sizeof(eshkol_tagged_value_t)) {
        eshkol_error("integer overflow in multi-value allocation size (count=%zu)", count);
        return nullptr;
    }
    size_t data_size = sizeof(size_t) + count * sizeof(eshkol_tagged_value_t);
    void* ptr = arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_MULTI_VALUE, 0);
    if (ptr) {
        // Initialize count field
        *((size_t*)ptr) = count;
    }
    return ptr;
}

// Get the count from a multi-value container
static inline size_t multi_value_get_count(void* multi_value_ptr) {
    return *((size_t*)multi_value_ptr);
}

// Get pointer to the values array in a multi-value container
static inline eshkol_tagged_value_t* multi_value_get_values(void* multi_value_ptr) {
    return (eshkol_tagged_value_t*)((uint8_t*)multi_value_ptr + sizeof(size_t));
}

// ───────────────────────────────────────────────────────────────────────────
// CONS CELL WITH HEADER (for consolidated HEAP_PTR type)
// ───────────────────────────────────────────────────────────────────────────

// Allocate a cons cell with object header (new consolidated format)
// Returns pointer to cons cell data (header is at offset -8)
arena_tagged_cons_cell_t* arena_allocate_cons_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate cons with header: null arena");
        return nullptr;
    }

    // Allocate header + cons cell, aligned for tagged values
    size_t total = sizeof(eshkol_object_header_t) + sizeof(arena_tagged_cons_cell_t);
    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate cons cell with header");
        return nullptr;
    }

    // Initialize header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_CONS;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = sizeof(arena_tagged_cons_cell_t);

    // Initialize cons cell
    arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)(mem + sizeof(eshkol_object_header_t));
    cell->car.type = ESHKOL_VALUE_NULL;
    cell->car.flags = 0;
    cell->car.reserved = 0;
    cell->car.data.raw_val = 0;
    cell->cdr.type = ESHKOL_VALUE_NULL;
    cell->cdr.flags = 0;
    cell->cdr.reserved = 0;
    cell->cdr.data.raw_val = 0;

    return cell;
}

// ───────────────────────────────────────────────────────────────────────────
// STRING WITH HEADER (for consolidated HEAP_PTR type)
// ───────────────────────────────────────────────────────────────────────────

// Allocate a string with object header (new consolidated format).
//
// Contract:
//   `length` is the number of payload (character) bytes the caller
//    will store — NOT including the trailing NUL terminator. The
//    allocator reserves `length + 1` bytes of data and writes the NUL
//    at str[length]. `hdr->size` is set to `length + 1` so that
//    downstream consumers (`eshkol_string_byte_length`) can recover
//    the caller-visible length by subtracting 1.
//
//   Historical hazard: several early call sites passed `len + 1`
//    thinking they had to account for the NUL themselves. That
//    silently over-allocated by a byte and — since `string-length`
//    previously relied on C strlen semantics — was invisible. Making
//    the header authoritative for string length (R7RS §6.7) exposed
//    the off-by-one as `string-length` returning `N+1`. All such
//    callers have been corrected to pass the bare character count.
//
// Returns pointer to string data (header is at offset -8).
char* arena_allocate_string_with_header(arena_t* arena, size_t length) {
    if (!arena) {
        eshkol_error("Cannot allocate string with header: null arena");
        return nullptr;
    }

    // Overflow check: length + 1 + header must not wrap
    if (length >= SIZE_MAX - sizeof(eshkol_object_header_t) - 8) {
        eshkol_error("String length overflow (length=%zu)", length);
        return nullptr;
    }

    // Allocate header + string data + null terminator
    size_t data_size = length + 1;  // +1 for null terminator
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;  // Align to 8 bytes

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate string with header (length=%zu)", length);
        return nullptr;
    }

    // Initialize header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_STRING;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Return pointer to string data
    char* str = (char*)(mem + sizeof(eshkol_object_header_t));
    str[0] = '\0';  // Initialize as empty string

    return str;
}

// ───────────────────────────────────────────────────────────────────────────
// VECTOR WITH HEADER (for consolidated HEAP_PTR type)
// ───────────────────────────────────────────────────────────────────────────

// Allocate a vector with object header (new consolidated format)
// Uses simple layout compatible with existing codegen:
//   - offset -8: object header (8 bytes)
//   - offset 0:  length (int64_t, 8 bytes)
//   - offset 8:  elements (capacity * sizeof(tagged_value))
// Returns pointer to length field (header is at offset -8)
void* arena_allocate_vector_with_header(arena_t* arena, size_t capacity) {
    if (!arena) {
        eshkol_error("Cannot allocate vector with header: null arena");
        return nullptr;
    }

    // Overflow check: capacity * sizeof(eshkol_tagged_value_t) can wrap at SIZE_MAX
    if (capacity > (SIZE_MAX - 8 - sizeof(eshkol_object_header_t)) / sizeof(eshkol_tagged_value_t)) {
        eshkol_error("Vector capacity overflow (capacity=%zu)", capacity);
        return nullptr;
    }

    // Calculate size: header + length field + element storage
    size_t data_size = 8 + capacity * sizeof(eshkol_tagged_value_t);  // length + elements
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 15) & ~15;  // Align to 16 bytes for tagged values

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate vector with header (capacity=%zu)", capacity);
        return nullptr;
    }

    // Initialize header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_VECTOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Return pointer to data (length field at offset 0)
    // Note: Caller is responsible for setting length and initializing elements
    // This matches existing codegen behavior
    return mem + sizeof(eshkol_object_header_t);
}

// ───────────────────────────────────────────────────────────────────────────
// SYMBOL WITH HEADER (for symbol->string conversion)
// ───────────────────────────────────────────────────────────────────────────

// Allocate a symbol with object header (same structure as string, different subtype)
// Layout: [header(8)] + [length(8)] + [char data]
// Returns pointer to length field (header is at offset -8)
void* arena_allocate_symbol_with_header(arena_t* arena, size_t length) {
    if (!arena) {
        eshkol_error("Cannot allocate symbol with header: null arena");
        return nullptr;
    }

    // Calculate size: header + length field + character data + null terminator
    size_t data_size = 8 + length + 1;  // length field + chars + null
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;  // Align to 8 bytes

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate symbol with header (length=%zu)", length);
        return nullptr;
    }

    // Initialize header with HEAP_SUBTYPE_SYMBOL
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_SYMBOL;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Return pointer to data (length field at offset 0)
    return mem + sizeof(eshkol_object_header_t);
}

// Scope management
void arena_push_scope(arena_t* arena) {
    if (!arena) return;
    
    arena_scope_t* scope = (arena_scope_t*)malloc(sizeof(arena_scope_t));
    if (!scope) {
        eshkol_error("Failed to allocate arena scope");
        return;
    }
    
    scope->block = arena->current_block;
    scope->used = arena->current_block->used;
    scope->parent = arena->current_scope;
    arena->current_scope = scope;
    
    eshkol_debug("Pushed arena scope");
}

void arena_pop_scope(arena_t* arena) {
    if (!arena || !arena->current_scope) {
        eshkol_error("Attempted to pop arena scope with no matching push — "
                     "unbalanced scope operations risk memory corruption");
        return;  // Graceful: skip the pop rather than kill the process
    }

    arena_scope_t* scope = arena->current_scope;

    /* Bug-BB-class diagnostic: when ESHKOL_ARENA_POISON=1 is set in
     * the environment, fill the popped region with a recognisable
     * sentinel byte (0xCB) before releasing it.  Any later
     * dereference of a stale pointer into that region will crash
     * with an address that contains the byte 0xCB in obvious
     * positions — turning a silent SEGV at a random-looking
     * mangled address into a "this was an arena UAF" diagnosis.
     *
     * This is opt-in (default off) because it adds work to every
     * pop; turn it on when investigating a "second-call SEGV" or
     * any state-corruption-shaped bug, then bisect by which arena
     * scope's pop precedes the crash.
     *
     * The byte 0xCB picked because:
     *   - High-bit set so addresses look "obviously poisoned" in
     *     register dumps (e.g. 0xCB...CB...) without colliding with
     *     real heap addresses on macOS arm64.
     *   - Distinct from libc's MallocPreScribble (0xAA) and
     *     MallocScribble (0x55) so they don't shadow each other.
     *   - Distinct from glibc malloc's perturb_free (0x42).
     */
    static int poison_enabled = -1;
    if (poison_enabled < 0) {
        const char* env = std::getenv("ESHKOL_ARENA_POISON");
        poison_enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    if (poison_enabled) {
        // Poison anything between scope's saved-used and current block's
        // current-used, plus any blocks beyond scope->block.
        if (scope->block && scope->block == arena->current_block) {
            char* base = (char*)scope->block->memory + scope->used;
            size_t len = (arena->current_block->used > scope->used)
                       ? (arena->current_block->used - scope->used) : 0;
            std::memset(base, 0xCB, len);
        }
        // Walk extra blocks added after the scope and poison their used range.
        for (arena_block_t* b = arena->current_block; b && b != scope->block; b = b->next) {
            std::memset(b->memory, 0xCB, b->used);
        }
    }

    // Restore arena state to scope start
    // Free any blocks allocated after this scope
    arena_block_t* block = arena->current_block;
    while (block && block != scope->block) {
        arena_block_t* next = block->next;
        arena->total_allocated -= block->size;
        free_arena_block(block);
        block = next;
    }

    arena->current_block = scope->block;
    if (arena->current_block) {
        arena->current_block->used = scope->used;
    }

    arena->current_scope = scope->parent;
    free(scope);

    eshkol_debug("Popped arena scope");
}

void arena_reset(arena_t* arena) {
    if (!arena) return;
    
    // Reset all blocks except the first one
    arena_block_t* first_block = nullptr;
    arena_block_t* block = arena->current_block;
    
    // Find the last block (first allocated)
    while (block) {
        if (block->next == nullptr) {
            first_block = block;
            break;
        }
        block = block->next;
    }
    
    // Free all blocks except the first
    block = arena->current_block;
    while (block && block != first_block) {
        arena_block_t* next = block->next;
        arena->total_allocated -= block->size;
        free_arena_block(block);
        block = next;
    }
    
    // Reset first block
    if (first_block) {
        first_block->used = 0;
        first_block->next = nullptr;
        arena->current_block = first_block;
    }
    
    // Clear all scopes
    arena_scope_t* scope = arena->current_scope;
    while (scope) {
        arena_scope_t* parent = scope->parent;
        free(scope);
        scope = parent;
    }
    arena->current_scope = nullptr;
    
    eshkol_debug("Reset arena");
}

// Statistics
size_t arena_get_used_memory(const arena_t* arena) {
    if (!arena) return 0;
    
    size_t used = 0;
    arena_block_t* block = arena->current_block;
    while (block) {
        used += block->used;
        block = block->next;
    }
    return used;
}

size_t arena_get_total_memory(const arena_t* arena) {
    return arena ? arena->total_allocated : 0;
}

size_t arena_get_block_count(const arena_t* arena) {
    if (!arena) return 0;
    
    size_t count = 0;
    arena_block_t* block = arena->current_block;
    while (block) {
        count++;
        block = block->next;
    }
    return count;
}

// List-specific allocation functions
arena_cons_cell_t* arena_allocate_cons_cell(arena_t* arena) {
    return (arena_cons_cell_t*)arena_allocate_aligned(arena, sizeof(arena_cons_cell_t), 
                                                     alignof(arena_cons_cell_t));
}

void* arena_allocate_list_node(arena_t* arena, size_t element_size, size_t count) {
    return arena_allocate(arena, element_size * count);
}

// Tagged cons cell allocation implementation
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_cell(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate tagged cons cell: null arena");
        return nullptr;
    }
    
    arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, sizeof(arena_tagged_cons_cell_t), 16);
    
    if (!cell) {
        eshkol_error("Failed to allocate tagged cons cell");
        return nullptr;
    }
    
    // Initialize with null types and zero values (Phase 3B: new structure)
    cell->car.type = ESHKOL_VALUE_NULL;
    cell->car.flags = 0;
    cell->car.reserved = 0;
    cell->car.data.raw_val = 0;
    
    cell->cdr.type = ESHKOL_VALUE_NULL;
    cell->cdr.flags = 0;
    cell->cdr.reserved = 0;
    cell->cdr.data.raw_val = 0;
    
    return cell;
}

// Batch allocation for efficiency
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch tagged cons allocation");
        return nullptr;
    }
    
    // Overflow check: count * sizeof(arena_tagged_cons_cell_t) can wrap at SIZE_MAX
    if (count > SIZE_MAX / sizeof(arena_tagged_cons_cell_t)) {
        eshkol_error("Tagged cons batch allocation overflow (count=%zu)", count);
        return nullptr;
    }

    size_t total_size = sizeof(arena_tagged_cons_cell_t) * count;
    arena_tagged_cons_cell_t* cells = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, total_size, 16);
    
    if (!cells) {
        eshkol_error("Failed to allocate %zu tagged cons cells", count);
        return nullptr;
    }
    
    // Initialize all cells (Phase 3B: new structure)
    for (size_t i = 0; i < count; i++) {
        cells[i].car.type = ESHKOL_VALUE_NULL;
        cells[i].car.flags = 0;
        cells[i].car.reserved = 0;
        cells[i].car.data.raw_val = 0;
        
        cells[i].cdr.type = ESHKOL_VALUE_NULL;
        cells[i].cdr.flags = 0;
        cells[i].cdr.reserved = 0;
        cells[i].cdr.data.raw_val = 0;
    }
    
    return cells;
}

// Convenience constructor for int64 values
arena_tagged_cons_cell_t* arena_create_int64_cons(arena_t* arena,
                                                   int64_t car, uint8_t car_type,
                                                   int64_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;
    
    // Phase 3B: Access nested tagged_value structure
    cell->car.type = car_type;
    cell->car.data.int_val = car;
    cell->cdr.type = cdr_type;
    cell->cdr.data.int_val = cdr;
    
    return cell;
}

// Convenience constructor for mixed values
arena_tagged_cons_cell_t* arena_create_mixed_cons(arena_t* arena,
                                                   eshkol_tagged_data_t car, uint8_t car_type,
                                                   eshkol_tagged_data_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;
    
    // Phase 3B: Access nested tagged_value structure
    // Use raw_val for efficient union copy (all union members share the same memory)
    cell->car.type = car_type;
    cell->car.data.raw_val = car.raw_val;
    
    cell->cdr.type = cdr_type;
    cell->cdr.data.raw_val = cdr.raw_val;
    
    return cell;
}

// Type-safe data retrieval functions
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get int64 from null tagged cons cell");
        return 0;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_INT_STORAGE_TYPE(type)) {
        eshkol_error("Attempted to get int64 from non-int-storage cell (type=%d)", type);
        return 0;
    }

    return tv->data.int_val;
}

double arena_tagged_cons_get_double(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get double from null tagged cons cell");
        return 0.0;
    }
    
    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Attempted to get double from non-double cell (type=%d)", type);
        return 0.0;
    }
    
    return tv->data.double_val;
}

uint64_t arena_tagged_cons_get_ptr(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get pointer from null tagged cons cell");
        return 0;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    uint8_t base_type = ESHKOL_GET_BASE_TYPE(type);

    if (base_type == ESHKOL_VALUE_NULL) {
        return 0;
    }

    // HOMOICONIC FIX: Accept any pointer type (CONS_PTR, STRING_PTR, VECTOR_PTR, etc.)
    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Attempted to get pointer from non-pointer cell (type=%d)", type);
        return 0;
    }

    return tv->data.ptr_val;
}

// Type-safe data setting functions
void arena_tagged_cons_set_int64(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                  int64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set int64 on null tagged cons cell");
        return;
    }
    
    if (!ESHKOL_IS_INT_STORAGE_TYPE(type)) {
        eshkol_error("Invalid type for int64 storage value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.int_val = value;
}

void arena_tagged_cons_set_double(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set double on null tagged cons cell");
        return;
    }
    
    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Invalid type for double value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.double_val = value;
}

void arena_tagged_cons_set_ptr(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                uint64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set pointer on null tagged cons cell");
        return;
    }

    // Allow any pointer type (CONS_PTR, STRING_PTR, VECTOR_PTR, TENSOR_PTR, AD_NODE_PTR, LAMBDA_SEXPR)
    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Invalid type for pointer value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.ptr_val = value;
}

void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot set null on null tagged cons cell");
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = ESHKOL_VALUE_NULL;
    tv->data.raw_val = 0;
}

// Type query functions
uint8_t arena_tagged_cons_get_type(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get type from null tagged cons cell");
        return ESHKOL_VALUE_NULL;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->type;
}

// Get flags from tagged cons cell
uint8_t arena_tagged_cons_get_flags(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get flags from null tagged cons cell");
        return 0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->flags;
}

bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type) {
    if (!cell) return false;
    
    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t actual_type = tv->type;
    return ESHKOL_GET_BASE_TYPE(actual_type) == ESHKOL_GET_BASE_TYPE(type);
}

// Direct tagged value access functions (NEW in Phase 3B)
// These enable direct storage and retrieval of complete tagged_value structs
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value) {
    if (!cell || !value) {
        eshkol_error("Cannot set tagged value: null parameter");
        return;
    }

    // Direct struct copy - this is the key optimization of Phase 3B!
    if (is_cdr) {
        cell->cdr = *value;
    } else {
        cell->car = *value;
    }
}

eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(const arena_tagged_cons_cell_t* cell,
                                                          bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get tagged value from null cell");
        // Return null value
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }
    
    // Direct struct copy - this is the key optimization of Phase 3B!
    return is_cdr ? cell->cdr : cell->car;
}

// ===== DEEP EQUALITY COMPARISON =====
// Runtime helper for deep structural equality of tagged values
// Takes pointers to avoid struct-by-value ABI issues

bool eshkol_deep_equal(const eshkol_tagged_value_t* val1, const eshkol_tagged_value_t* val2) {
    if (!val1 || !val2) {
        return val1 == val2;  // Both null -> equal, one null -> not equal
    }

    // Compute base type correctly:
    // - Legacy types (>= 32): use directly
    // - Consolidated (8-31): use directly
    // - Immediate (< 8): mask to strip exactness flags
    auto get_base_type = [](uint8_t t) -> uint8_t {
        if (t >= 8) return t;  // Legacy, consolidated, or multimedia types
        return t & 0x0F;  // Immediate types: strip exactness flags
    };
    uint8_t type1 = get_base_type(val1->type);
    uint8_t type2 = get_base_type(val2->type);

    // Helper to check if a value is a cons cell (legacy or consolidated)
    auto is_cons = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_CONS_PTR) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_CONS;
        }
        return false;
    };

    // Helper to check if a value represents empty list
    auto is_empty_list = [&is_cons](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_NULL) return true;
        if (type == ESHKOL_VALUE_CONS_PTR && val->data.ptr_val == 0) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val == 0) return true;
        return false;
    };

    // Helper to check if a value is a string (legacy or consolidated)
    auto is_string = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_STRING_PTR) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_STRING;
        }
        return false;
    };

    // Helper to check if a value is a symbol (consolidated HEAP_PTR with HEAP_SUBTYPE_SYMBOL)
    auto is_symbol = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_SYMBOL;
        }
        return false;
    };

    bool empty1 = is_empty_list(type1, val1);
    bool empty2 = is_empty_list(type2, val2);

    // Both empty lists: equal (handles NULL vs CONS_PTR with null pointer)
    if (empty1 && empty2) {
        return true;
    }

    // One empty, one not: not equal
    if (empty1 || empty2) {
        return false;
    }

    // Both cons (legacy or consolidated): recursively compare cars and cdrs
    bool is_cons1 = is_cons(type1, val1);
    bool is_cons2 = is_cons(type2, val2);
    if (is_cons1 && is_cons2) {
        arena_tagged_cons_cell_t* cell1 = (arena_tagged_cons_cell_t*)val1->data.ptr_val;
        arena_tagged_cons_cell_t* cell2 = (arena_tagged_cons_cell_t*)val2->data.ptr_val;

        if (!cell1 || !cell2) {
            // If both null pointers, equal; if one null, not equal
            return cell1 == cell2;
        }

        // Compare cars recursively
        eshkol_tagged_value_t car1 = arena_tagged_cons_get_tagged_value(cell1, false);
        eshkol_tagged_value_t car2 = arena_tagged_cons_get_tagged_value(cell2, false);
        if (!eshkol_deep_equal(&car1, &car2)) {
            return false;
        }

        // Compare cdrs recursively
        eshkol_tagged_value_t cdr1 = arena_tagged_cons_get_tagged_value(cell1, true);
        eshkol_tagged_value_t cdr2 = arena_tagged_cons_get_tagged_value(cell2, true);
        return eshkol_deep_equal(&cdr1, &cdr2);
    }

    // Both strings (legacy or consolidated): compare content
    bool is_str1 = is_string(type1, val1);
    bool is_str2 = is_string(type2, val2);
    if (is_str1 && is_str2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        if (!val1->data.ptr_val || !val2->data.ptr_val) return false;
        return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;
    }

    // Both symbols (consolidated HEAP_PTR): compare content like strings
    bool is_sym1 = is_symbol(type1, val1);
    bool is_sym2 = is_symbol(type2, val2);
    if (is_sym1 && is_sym2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        if (!val1->data.ptr_val || !val2->data.ptr_val) return false;
        // Symbol data is a null-terminated string starting at the pointer
        return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;
    }

    // Special case: numeric comparison between INT64 and DOUBLE
    // This allows (equal? 130 130.0) to return #t when values are numerically equal
    if ((type1 == ESHKOL_VALUE_INT64 && type2 == ESHKOL_VALUE_DOUBLE) ||
        (type1 == ESHKOL_VALUE_DOUBLE && type2 == ESHKOL_VALUE_INT64)) {
        double d1 = (type1 == ESHKOL_VALUE_DOUBLE) ? val1->data.double_val : (double)val1->data.int_val;
        double d2 = (type2 == ESHKOL_VALUE_DOUBLE) ? val2->data.double_val : (double)val2->data.int_val;
        return d1 == d2;
    }

    // Bignum comparison: bignum vs bignum, bignum vs int64
    auto is_bignum = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_BIGNUM;
        }
        return false;
    };
    bool is_bn1 = is_bignum(type1, val1);
    bool is_bn2 = is_bignum(type2, val2);
    if (is_bn1 && is_bn2) {
        return eshkol_bignum_compare((const eshkol_bignum_t*)val1->data.ptr_val,
                                     (const eshkol_bignum_t*)val2->data.ptr_val) == 0;
    }
    if (is_bn1 && type2 == ESHKOL_VALUE_INT64) {
        return eshkol_bignum_compare_int64((const eshkol_bignum_t*)val1->data.ptr_val,
                                           val2->data.int_val) == 0;
    }
    if (type1 == ESHKOL_VALUE_INT64 && is_bn2) {
        return eshkol_bignum_compare_int64((const eshkol_bignum_t*)val2->data.ptr_val,
                                           val1->data.int_val) == 0;
    }

    // Tensor vs tensor: compare dimensions + element-wise doubles.
    // Two #(...) literals with identical contents must return #t under
    // R7RS-style equal? even though they are distinct allocations.
    auto is_tensor = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_TENSOR;
        }
        return false;
    };
    bool is_t1 = is_tensor(type1, val1);
    bool is_t2 = is_tensor(type2, val2);
    if (is_t1 && is_t2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        eshkol_tensor_t* t1 = (eshkol_tensor_t*)val1->data.ptr_val;
        eshkol_tensor_t* t2 = (eshkol_tensor_t*)val2->data.ptr_val;
        if (!t1 || !t2) return t1 == t2;
        if (t1->num_dimensions != t2->num_dimensions) return false;
        if (t1->total_elements != t2->total_elements) return false;
        for (uint64_t i = 0; i < t1->num_dimensions; i++) {
            if (t1->dimensions[i] != t2->dimensions[i]) return false;
        }
        // Elements are doubles stored as int64 bit patterns — compare as doubles
        // so NaN != NaN falls through correctly and -0.0 == 0.0.
        for (uint64_t i = 0; i < t1->total_elements; i++) {
            union { int64_t i; double d; } u1, u2;
            u1.i = t1->elements[i];
            u2.i = t2->elements[i];
            if (u1.d != u2.d) return false;
        }
        return true;
    }

    // Vector vs vector: heterogeneous tagged elements. Compare each pair
    // recursively via eshkol_deep_equal so nested structures match too.
    auto is_vector = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_VECTOR;
        }
        return false;
    };
    bool is_v1 = is_vector(type1, val1);
    bool is_v2 = is_vector(type2, val2);
    if (is_v1 && is_v2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        int64_t len1 = *(int64_t*)(uintptr_t)val1->data.ptr_val;
        int64_t len2 = *(int64_t*)(uintptr_t)val2->data.ptr_val;
        if (len1 != len2) return false;
        eshkol_tagged_value_t* elems1 =
            (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)val1->data.ptr_val + 8);
        eshkol_tagged_value_t* elems2 =
            (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)val2->data.ptr_val + 8);
        for (int64_t i = 0; i < len1; i++) {
            if (!eshkol_deep_equal(&elems1[i], &elems2[i])) return false;
        }
        return true;
    }

    // Different types: not equal
    if (type1 != type2) {
        return false;
    }

    // Same types, compare values based on type
    switch (type1) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
            return val1->data.int_val == val2->data.int_val;

        case ESHKOL_VALUE_DOUBLE:
            return val1->data.double_val == val2->data.double_val;

        case ESHKOL_VALUE_STRING_PTR:
            // Compare string content using strcmp
            // Dynamic strings (e.g., from string-append) may have same content but different addresses
            if (val1->data.ptr_val == val2->data.ptr_val) return true;  // Same pointer = equal
            if (!val1->data.ptr_val || !val2->data.ptr_val) return false;  // One null = not equal
            return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;

        case ESHKOL_VALUE_CLOSURE_PTR:
        case ESHKOL_VALUE_LAMBDA_SEXPR:
            // For closures and lambdas, compare pointer addresses
            return val1->data.ptr_val == val2->data.ptr_val;

        default:
            // Unknown types: compare raw int data
            return val1->data.int_val == val2->data.int_val;
    }
}

// ===== OALR (Ownership-Aware Lexical Regions) IMPLEMENTATION =====

// Thread-local region stack (safe for parallel-map + with-region)
thread_local eshkol_region_t* __region_stack[MAX_REGION_DEPTH] = {nullptr};
thread_local uint64_t __region_stack_depth = 0;

// Default global arena for allocations outside of any region
// Non-static to allow JIT code to access it directly
// Use weak linkage so generated code can override in standalone mode
__attribute__((weak)) arena_t* __global_arena = nullptr;

// Forward declaration — defined below in per-thread arena section
static thread_local arena_t* __thread_local_arena;

static void init_global_arena_internal() {
    // Create a thread-safe arena for global allocations
    __global_arena = arena_create_threadsafe(65536);  // 64KB default, thread-safe
    if (!__global_arena) {
        eshkol_error("Failed to create global arena");
    }
}

// Get the appropriate arena for the current thread.
// Worker threads get their own per-thread arena (no lock contention).
// Main thread gets the global thread-safe arena.
arena_t* get_global_arena() {
    // Use platform once semantics for thread-safe initialization.
    eshkol_arena_global_once(init_global_arena_internal);

    // On worker threads, use the per-thread arena if available.
    // This eliminates mutex contention in parallel-map/fold/filter.
    if (__thread_local_arena) return __thread_local_arena;

    return __global_arena;
}

// Get the global arena specifically (bypasses per-thread override).
// Used when allocation MUST go into the shared arena (e.g. result lists).
arena_t* get_global_arena_shared() {
    eshkol_arena_global_once(init_global_arena_internal);
    return __global_arena;
}

// ── Per-thread arena management (v1.2) ──

// __thread_local_arena is forward-declared above get_global_arena()

arena_t* arena_get_thread_local(void) {
    // Worker threads have their own TLS arena managed by thread_pool.cpp.
    // For non-worker threads, use the thread-local arena or global arena.
    if (__thread_local_arena) return __thread_local_arena;
    return get_global_arena();
}

arena_t* arena_create_thread_local(size_t size_hint) {
    if (__thread_local_arena) {
        // Already exists — return existing
        return __thread_local_arena;
    }
    size_t block_size = size_hint > 0 ? size_hint : (1024 * 1024); // 1 MB default
    __thread_local_arena = arena_create(block_size);
    return __thread_local_arena;
}

/* Initialise all thread-local runtime state for a worker thread.
 *
 * Architectural purpose: parallel-map workers can execute user lambdas that
 * call AD primitives (ad-tape-new, ad-var, ad-sin, ...) or OALR regions
 * (with-region). Those primitives read/write the following thread_local
 * variables declared above:
 *
 *   __ad_tape_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH]
 *   __ad_tape_depth
 *   __outer_ad_node_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH]
 *   __outer_ad_node_depth
 *   __region_stack[MAX_REGION_DEPTH]   (line 1606)
 *   __region_stack_depth                (line 1607)
 *   __thread_local_arena                (line 1622)
 *
 * On POSIX+glibc, thread_local is lazily initialised on first access; on
 * macOS Mach-O, it goes through __tlv_* wrappers. Explicitly touching each
 * TLS slot at worker startup forces initialisation before the first user
 * task runs, preventing any "silent zero" hazards where codegen reads a
 * TLS slot before it has been materialised.
 *
 * We also eagerly allocate the thread-local arena so the first allocation
 * call inside a task does not pay the creation cost (latency smoothing)
 * and so any arena_* call made during a signal handler or ctor path is
 * safe. */
void eshkol_thread_init_worker(size_t arena_size_hint) {
    /* AD tape TLS slots: explicit zero so readers get defined values even if
     * the TLS image is not zero-filled (conservative; on POSIX it is). */
    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;

    /* Double-backward storage */
    __outer_ad_node_storage = nullptr;
    __outer_ad_node_to_inner = nullptr;
    __outer_grad_accumulator = nullptr;
    __inner_var_node_ptr = nullptr;
    __gradient_x_degree = 0;

    /* OALR region TLS slots */
    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;

    /* Eagerly create the thread-local arena so the first arena_allocate
     * in a worker task finds an arena ready. If one already exists (e.g.
     * init called twice), arena_create_thread_local is a no-op. */
    (void)arena_create_thread_local(arena_size_hint);
}

/* Destroy the thread-local arena and clear TLS handles. Safe to call
 * multiple times (idempotent). */
void eshkol_thread_shutdown_worker(void) {
    if (__thread_local_arena) {
        arena_destroy(__thread_local_arena);
        __thread_local_arena = nullptr;
    }
    /* Clear tape + region stack pointers so stale tape pointers from a
     * previous worker lifetime are never observed after pthread re-use. */
    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;
    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;
}

void arena_merge_to_parent(arena_t* dest, arena_t* src) {
    if (!dest || !src || dest == src) return;

    if (dest->thread_safe) arena_lock(dest);

    /* Append src's blocks to the END of dest's chain.
     * dest->current_block stays unchanged — dest continues allocating
     * from its own newest block. Src's blocks become "old" blocks
     * containing finalized data that won't be allocated into.
     *
     * This preserves dest's allocation invariants:
     *   - dest->current_block->used reflects dest's allocation state
     *   - New allocations go into dest->current_block (or new blocks)
     *   - Src blocks are only read, never written to after merge */
    if (src->current_block) {
        if (dest->current_block) {
            /* Find the end of dest's block chain */
            arena_block_t* dest_tail = dest->current_block;
            while (dest_tail->next) dest_tail = dest_tail->next;
            /* Append src's entire chain after dest's tail */
            dest_tail->next = src->current_block;
        } else {
            /* dest has no blocks — take src's chain */
            dest->current_block = src->current_block;
        }

        dest->total_allocated += src->total_allocated;

        /* Clear src without freeing blocks (now owned by dest) */
        src->current_block = nullptr;
        src->total_allocated = 0;
    }

    if (dest->thread_safe) arena_unlock(dest);
}

/* Worker thread detection — delegates to thread_pool's TLS flag.
 * Weak symbol: returns 0 if thread_pool is not linked. */
extern "C" int eshkol_thread_pool_is_worker(void) __attribute__((weak));

int arena_is_worker_thread(void) {
    /* If thread_pool is linked, use its TLS flag. Otherwise return 0. */
    if (eshkol_thread_pool_is_worker) {
        return eshkol_thread_pool_is_worker();
    }
    return 0;
}

// Region creation
eshkol_region_t* region_create(const char* name, size_t size_hint) {
    // Use global arena to allocate the region structure itself
    arena_t* global = get_global_arena();

    eshkol_region_t* region = (eshkol_region_t*)
        arena_allocate_aligned(global, sizeof(eshkol_region_t), 8);

    if (!region) {
        eshkol_error("Failed to allocate region structure");
        return nullptr;
    }

    // Determine arena size
    size_t arena_size = (size_hint > 0) ? size_hint : 8192;
    if (arena_size < 1024) arena_size = 1024;

    // Create the region's arena
    region->arena = arena_create(arena_size);
    if (!region->arena) {
        eshkol_error("Failed to create region arena");
        return nullptr;
    }

    // Copy name if provided
    if (name) {
        size_t name_len = strlen(name) + 1;
        char* name_copy = (char*)arena_allocate(region->arena, name_len);
        if (name_copy) {
            memcpy(name_copy, name, name_len - 1);
            name_copy[name_len - 1] = '\0';
            region->name = name_copy;
        } else {
            region->name = nullptr;
        }
    } else {
        region->name = nullptr;
    }

    region->parent = nullptr;
    region->size_hint = size_hint;
    region->escape_count = 0;
    region->is_active = 0;

    eshkol_debug("Created region '%s' with size hint %zu",
                name ? name : "(anonymous)", size_hint);

    return region;
}

// Region destruction
void region_destroy(eshkol_region_t* region) {
    if (!region) return;

    if (region->is_active) {
        eshkol_warn("Destroying active region '%s' - popping from stack first",
                   region->name ? region->name : "(anonymous)");
        region_pop();
    }

    const char* name = region->name ? region->name : "(anonymous)";
    size_t used = region->arena ? arena_get_used_memory(region->arena) : 0;

    // Destroy the region's arena
    if (region->arena) {
        arena_destroy(region->arena);
        region->arena = nullptr;
    }

    eshkol_debug("Destroyed region '%s', freed %zu bytes", name, used);
}

// Push a region onto the stack
void region_push(eshkol_region_t* region) {
    if (!region) {
        eshkol_error("Cannot push null region");
        return;
    }

    if (__region_stack_depth >= MAX_REGION_DEPTH) {
        eshkol_error("Region stack overflow (max depth: %d)", MAX_REGION_DEPTH);
        return;
    }

    // Set parent to current top
    if (__region_stack_depth > 0) {
        region->parent = __region_stack[__region_stack_depth - 1];
    } else {
        region->parent = nullptr;
    }

    region->is_active = 1;
    __region_stack[__region_stack_depth++] = region;

    eshkol_debug("Pushed region '%s' (depth: %llu)",
                region->name ? region->name : "(anonymous)",
                __region_stack_depth);
}

// Pop the current region from the stack
void region_pop(void) {
    if (__region_stack_depth == 0) {
        eshkol_warn("Attempted to pop from empty region stack");
        return;
    }

    eshkol_region_t* region = __region_stack[--__region_stack_depth];
    region->is_active = 0;

    eshkol_debug("Popped region '%s' (depth: %llu)",
                region->name ? region->name : "(anonymous)",
                __region_stack_depth);

    // Destroy the region (this frees all its memory)
    region_destroy(region);
}

// Get the current active region
eshkol_region_t* region_current(void) {
    if (__region_stack_depth == 0) {
        return nullptr;
    }
    return __region_stack[__region_stack_depth - 1];
}

// Allocate in the current region (or global arena if no region)
void* region_allocate(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate(region->arena, size);
    }
    // Fallback to global arena
    return arena_allocate(get_global_arena(), size);
}

void* region_allocate_aligned(size_t size, size_t alignment) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_aligned(region->arena, size, alignment);
    }
    return arena_allocate_aligned(get_global_arena(), size, alignment);
}

void* region_allocate_zeroed(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_zeroed(region->arena, size);
    }
    return arena_allocate_zeroed(get_global_arena(), size);
}

// Region-aware tagged cons cell allocation
arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_tagged_cons_cell(region->arena);
    }
    return arena_allocate_tagged_cons_cell(get_global_arena());
}

// Region statistics
size_t region_get_used_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_used_memory(region->arena);
}

size_t region_get_total_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_total_memory(region->arena);
}

const char* region_get_name(const eshkol_region_t* region) {
    if (!region) return nullptr;
    return region->name;
}

uint64_t region_get_depth(void) {
    return __region_stack_depth;
}

// Region escape — copy a value from the current region to the parent region (or global arena)
void* region_escape(const void* ptr, size_t size) {
    if (!ptr || size == 0) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) {
        // Not in any region — nothing to escape from
        return (void*)ptr;
    }

    // Allocate in parent region, or global arena if no parent
    arena_t* target;
    if (current->parent && current->parent->arena) {
        target = current->parent->arena;
    } else {
        target = get_global_arena();
    }

    void* copy = arena_allocate_aligned(target, size, 8);
    if (!copy) {
        eshkol_error("region_escape: failed to allocate %zu bytes in target arena", size);
        return nullptr;
    }

    memcpy(copy, ptr, size);
    current->escape_count++;

    return copy;
}

void* region_escape_string(const char* str) {
    if (!str) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (void*)str;

    size_t len = strlen(str);

    arena_t* target;
    if (current->parent && current->parent->arena) {
        target = current->parent->arena;
    } else {
        target = get_global_arena();
    }

    // Check if the string has a header (8 bytes before pointer)
    // For heap-allocated strings, we need to copy the header too
    // Allocate string with header in target arena
    char* copy = (char*)arena_allocate_string_with_header(target, len);
    if (!copy) {
        eshkol_error("region_escape_string: failed to allocate string of length %zu", len);
        return nullptr;
    }

    memcpy(copy, str, len);
    copy[len] = '\0';
    current->escape_count++;

    return copy;
}

arena_tagged_cons_cell_t* region_escape_tagged_cons_cell(const arena_tagged_cons_cell_t* cell) {
    if (!cell) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (arena_tagged_cons_cell_t*)cell;

    arena_t* target;
    if (current->parent && current->parent->arena) {
        target = current->parent->arena;
    } else {
        target = get_global_arena();
    }

    arena_tagged_cons_cell_t* copy = arena_allocate_tagged_cons_cell(target);
    if (!copy) {
        eshkol_error("region_escape_tagged_cons_cell: failed to allocate");
        return nullptr;
    }

    copy->car = cell->car;
    copy->cdr = cell->cdr;
    current->escape_count++;

    return copy;
}

static eshkol_tagged_value_t region_escape_tagged_value_impl(eshkol_tagged_value_t val) {
    // For immediate types (int, double, bool, char, etc.) — no escape needed
    uint8_t type = val.type;

    // Ports are encoded as HEAP_PTR plus port flags; exact numbers also use 0x10,
    // so a raw port-bit mask would misclassify exact immediates as heap pointers.
    bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                   ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);

    // Check if this is a pointer-like type (might point into region memory)
    bool is_heap = ESHKOL_IS_ANY_PTR_TYPE(type) || is_port;

    if (!is_heap) {
        // Immediate value — safe to return as-is
        return val;
    }

    eshkol_region_t* current = region_current();
    if (!current) return val;

    // For heap pointers, we need to deep-copy the pointed-to data
    // This is a shallow escape — the caller is responsible for deep structures
    // For now, copy the pointed-to object using the header's size field
    void* ptr = (void*)(uintptr_t)val.data.ptr_val;
    if (!ptr) return val;

    // Read the object header (8 bytes before the data pointer)
    eshkol_object_header_t* header = (eshkol_object_header_t*)((uint8_t*)ptr - sizeof(eshkol_object_header_t));
    size_t obj_size = header->size;

    if (obj_size == 0) {
        // Can't determine size — return as-is (conservative)
        return val;
    }

    // Allocate in target arena (header + data)
    arena_t* target;
    if (current->parent && current->parent->arena) {
        target = current->parent->arena;
    } else {
        target = get_global_arena();
    }

    size_t total = sizeof(eshkol_object_header_t) + obj_size;
    void* raw = arena_allocate_aligned(target, total, 8);
    if (!raw) {
        eshkol_error("region_escape_tagged_value: failed to allocate %zu bytes", total);
        return val;
    }

    // Copy header + data
    memcpy(raw, header, total);

    // Update the tagged value to point to the new copy (skip header)
    void* new_data = (uint8_t*)raw + sizeof(eshkol_object_header_t);
    val.data.ptr_val = (uint64_t)(uintptr_t)new_data;
    current->escape_count++;

    return val;
}

extern "C" eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val) {
    return region_escape_tagged_value_impl(val);
}

extern "C" void region_escape_tagged_value_into(eshkol_tagged_value_t* out, const eshkol_tagged_value_t* val) {
    if (!out) return;
    if (!val) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = region_escape_tagged_value_impl(*val);
}

// ===== END OALR IMPLEMENTATION =====

#ifdef __cplusplus

// C++ Arena wrapper implementation
Arena::Arena(size_t default_block_size) : arena_(arena_create(default_block_size)) {
    if (!arena_) {
        throw std::bad_alloc();
    }
}

Arena::~Arena() {
    arena_destroy(arena_);
}

Arena::Arena(Arena&& other) noexcept : arena_(other.arena_) {
    other.arena_ = nullptr;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        arena_destroy(arena_);
        arena_ = other.arena_;
        other.arena_ = nullptr;
    }
    return *this;
}

void* Arena::allocate(size_t size) {
    return arena_allocate(arena_, size);
}

void* Arena::allocate_aligned(size_t size, size_t alignment) {
    return arena_allocate_aligned(arena_, size, alignment);
}

void* Arena::allocate_zeroed(size_t size) {
    return arena_allocate_zeroed(arena_, size);
}

size_t Arena::get_used_memory() const {
    return arena_get_used_memory(arena_);
}

size_t Arena::get_total_memory() const {
    return arena_get_total_memory(arena_);
}

size_t Arena::get_block_count() const {
    return arena_get_block_count(arena_);
}

void Arena::reset() {
    arena_reset(arena_);
}

// Arena::Scope implementation
Arena::Scope::Scope(Arena& arena) : arena_(&arena), active_(true) {
    arena_push_scope(arena.arena_);
}

Arena::Scope::~Scope() {
    if (active_) {
        arena_pop_scope(arena_->arena_);
    }
}


#endif // __cplusplus
