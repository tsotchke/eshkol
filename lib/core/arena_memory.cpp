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
#include <setjmp.h>

#ifdef __cplusplus
#include <new>      // for std::bad_alloc
#include <stdexcept>
#include <atomic>
#include <cmath>
#include <cstring>      // std::memset
#include <filesystem>   // Bug W ask 2: provider-file scan
#include <fstream>      // Bug W ask 2: read .esk files for provide/define lookup
#include <string>       // std::string for the provider scan
#include <cctype>       // std::isspace / std::isalnum
#endif

void* eshkol_arena_mutex_create(void);
void eshkol_arena_mutex_destroy(void* mutex);
void eshkol_arena_mutex_lock(void* mutex);
void eshkol_arena_mutex_unlock(void* mutex);
void eshkol_arena_global_once(void (*init)(void));
void eshkol_hash_table_lock(void);
void eshkol_hash_table_unlock(void);

// Default alignment for memory allocations
#define DEFAULT_ALIGNMENT 8

// Global tape pointer for AD operations.
// NOTE: Not thread_local — LLVM ExternalLinkage globals cannot link against
// C thread_local symbols portably (macOS ARM64 TLS ABI mismatch). Thread
// safety is ensured by the AD tape STACK (__ad_tape_stack) which IS thread_local.
// parallel-map workers don't perform AD internally; the main thread controls
// AD mode. If parallel AD is needed (v1.2+), use per-task tape allocation.
ad_tape_t* __current_ad_tape = nullptr;

// Global AD mode flag — same portability constraint as __current_ad_tape.
// CRITICAL: Must be shared so lambdas from one LLVM module can see AD mode
// set by another module (cross-module AD in REPL).
bool __ad_mode_active = false;

// Debug helper to print AD mode state
void debug_print_ad_mode(const char* context) {
    fprintf(stderr, "[AD_MODE_DEBUG] %s: __ad_mode_active = %s\n",
            context, __ad_mode_active ? "TRUE" : "FALSE");
}

// Debug helper to print pointer value
void debug_print_ptr(const char* context, void* ptr) {
    fprintf(stderr, "[PTR_DEBUG] %s: ptr = %p (0x%llx)\n",
            context, ptr, (unsigned long long)(uintptr_t)ptr);
}

// NESTED GRADIENT FIX: Tape stack for arbitrary-depth nested gradients
// MAX_TAPE_DEPTH must match the value in llvm_codegen.cpp
// thread_local: AD tape state is per-thread to prevent corruption under parallel autodiff
#define MAX_TAPE_DEPTH 32
thread_local ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __ad_tape_depth = 0;

// DOUBLE BACKWARD: Storage for outer AD node when in nested gradient
thread_local void* __outer_ad_node_storage = nullptr;
thread_local void* __outer_ad_node_to_inner = nullptr;
thread_local void* __outer_grad_accumulator = nullptr;
thread_local void* __inner_var_node_ptr = nullptr;
thread_local uint64_t __gradient_x_degree = 0;

// N-DIMENSIONAL DERIVATIVES: Stack of outer AD nodes for arbitrary depth nesting
// Uses same MAX_TAPE_DEPTH (32)
thread_local void* __outer_ad_node_stack[MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __outer_ad_node_depth = 0;

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

// ───────────────────────────────────────────────────────────────────────────
// CLOSURE WITH HEADER (for consolidated CALLABLE type)
// ───────────────────────────────────────────────────────────────────────────

// Allocate a closure with object header (new consolidated format)
// Returns pointer to closure data (header is at offset -8)
// This is for new allocations - existing arena_allocate_closure remains for compatibility
extern "C" eshkol_closure_t* arena_allocate_closure_with_header(arena_t* arena, uint64_t func_ptr,
                                                      size_t packed_info, uint64_t sexpr_ptr,
                                                      uint64_t return_type_info,
                                                      const char* name) {
    if (!arena) {
        eshkol_error("Cannot allocate closure with header: null arena");
        return nullptr;
    }

    // VARIADIC CLOSURE FIX: Unpack the num_captures from the packed info
    // The packed_info field contains:
    //   - Bits 0-15:  actual num_captures
    //   - Bits 16-31: fixed_param_count
    //   - Bit 63:     is_variadic flag
    size_t actual_num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(packed_info);

    // Calculate size: header + closure structure
    size_t data_size = sizeof(eshkol_closure_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;  // Align to 8 bytes

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate closure with header");
        return nullptr;
    }

    // Initialize header
    // Use LAMBDA_SEXPR subtype for closures without captures (semantically they're lambda s-expressions)
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = (actual_num_captures == 0) ? CALLABLE_SUBTYPE_LAMBDA_SEXPR : CALLABLE_SUBTYPE_CLOSURE;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Initialize closure
    eshkol_closure_t* closure = (eshkol_closure_t*)(mem + sizeof(eshkol_object_header_t));
    closure->func_ptr = func_ptr;
    closure->sexpr_ptr = sexpr_ptr;
    closure->name = name;  // Store procedure name (NULL for anonymous lambdas)
    closure->return_type = (uint8_t)(return_type_info & 0xFF);
    closure->input_arity = (uint8_t)((return_type_info >> 8) & 0xFF);

    // VARIADIC FIX: Store variadic flag in closure->flags so it's available even when env is NULL
    closure->flags = CLOSURE_ENV_IS_VARIADIC(packed_info) ? CLOSURE_FLAG_VARIADIC : 0;
    if (name) {
        closure->flags |= ESHKOL_CLOSURE_FLAG_NAMED;
    }
    closure->reserved = 0;
    closure->hott_type_id = (uint32_t)((return_type_info >> 16) & 0xFFFFFFFF);

    // Allocate environment if there are captures
    if (actual_num_captures > 0) {
        // Allocate env with actual capture count
        closure->env = arena_allocate_closure_env(arena, actual_num_captures);
        if (closure->env) {
            // Store the full packed_info (including variadic flag) in the env's num_captures field
            closure->env->num_captures = packed_info;
        }
    } else {
        closure->env = nullptr;
    }

    return closure;
}

// ═══════════════════════════════════════════════════════════════════════════
// END HEADER-AWARE ALLOCATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

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

// ===== AD MEMORY MANAGEMENT IMPLEMENTATION =====

// Dual number allocation
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate dual number: null arena");
        return nullptr;
    }
    
    eshkol_dual_number_t* dual = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dual_number_t), 8);
    
    if (dual) {
        dual->value = 0.0;
        dual->derivative = 0.0;
    }
    
    return dual;
}

eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch dual number allocation");
        return nullptr;
    }
    
    size_t total_size = sizeof(eshkol_dual_number_t) * count;
    eshkol_dual_number_t* duals = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, total_size, 8);
    
    if (duals) {
        for (size_t i = 0; i < count; i++) {
            duals[i].value = 0.0;
            duals[i].derivative = 0.0;
        }
    }
    
    return duals;
}

// AD node allocation for computational graphs
ad_node_t* arena_allocate_ad_node(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate AD node: null arena");
        return nullptr;
    }

    ad_node_t* node = (ad_node_t*)
        arena_allocate_aligned(arena, sizeof(ad_node_t), 8);

    if (node) {
        node->type = AD_NODE_CONSTANT;
        node->value = 0.0;
        node->gradient = 0.0;
        node->input1 = nullptr;
        node->input2 = nullptr;
        node->id = 0;
        // Zero-initialize extended tensor AD fields
        node->tensor_value = nullptr;
        node->tensor_gradient = nullptr;
        node->input3 = nullptr;
        node->input4 = nullptr;
        node->saved_tensors = nullptr;
        node->num_saved = 0;
        memset(&node->params, 0, sizeof(node->params));
        node->shape = nullptr;
        node->ndim = 0;
    }

    return node;
}

// AD node allocation with object header (for consolidated CALLABLE type)
ad_node_t* arena_allocate_ad_node_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate AD node with header: null arena");
        return nullptr;
    }

    // Calculate size: header + ad_node structure
    size_t data_size = sizeof(ad_node_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;  // Align to 8 bytes

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate AD node with header");
        return nullptr;
    }

    // Initialize header with AD_NODE subtype
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = CALLABLE_SUBTYPE_AD_NODE;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Initialize AD node (all 15 fields)
    ad_node_t* node = (ad_node_t*)(mem + sizeof(eshkol_object_header_t));
    node->type = AD_NODE_CONSTANT;
    node->value = 0.0;
    node->gradient = 0.0;
    node->input1 = nullptr;
    node->input2 = nullptr;
    node->id = 0;
    // Zero-initialize extended tensor AD fields
    node->tensor_value = nullptr;
    node->tensor_gradient = nullptr;
    node->input3 = nullptr;
    node->input4 = nullptr;
    node->saved_tensors = nullptr;
    node->num_saved = 0;
    memset(&node->params, 0, sizeof(node->params));
    node->shape = nullptr;
    node->ndim = 0;

    return node;
}

ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch AD node allocation");
        return nullptr;
    }
    
    size_t total_size = sizeof(ad_node_t) * count;
    ad_node_t* nodes = (ad_node_t*)
        arena_allocate_aligned(arena, total_size, 8);
    
    if (nodes) {
        for (size_t i = 0; i < count; i++) {
            nodes[i].type = AD_NODE_CONSTANT;
            nodes[i].value = 0.0;
            nodes[i].gradient = 0.0;
            nodes[i].input1 = nullptr;
            nodes[i].input2 = nullptr;
            nodes[i].id = 0;
            // Zero-initialize extended tensor AD fields
            nodes[i].tensor_value = nullptr;
            nodes[i].tensor_gradient = nullptr;
            nodes[i].input3 = nullptr;
            nodes[i].input4 = nullptr;
            nodes[i].saved_tensors = nullptr;
            nodes[i].num_saved = 0;
            memset(&nodes[i].params, 0, sizeof(nodes[i].params));
            nodes[i].shape = nullptr;
            nodes[i].ndim = 0;
        }
    }
    
    return nodes;
}

// Tape allocation and management
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity) {
    if (!arena) {
        eshkol_error("Cannot allocate tape: null arena");
        return nullptr;
    }
    
    if (initial_capacity == 0) {
        initial_capacity = 64; // Default capacity
    }
    
    ad_tape_t* tape = (ad_tape_t*)
        arena_allocate_aligned(arena, sizeof(ad_tape_t), 8);
    
    if (!tape) {
        eshkol_error("Failed to allocate tape structure");
        return nullptr;
    }
    
    // Allocate nodes array
    size_t nodes_size = sizeof(ad_node_t*) * initial_capacity;
    tape->nodes = (ad_node_t**)arena_allocate_aligned(arena, nodes_size, 8);
    
    if (!tape->nodes) {
        eshkol_error("Failed to allocate tape nodes array");
        return nullptr;
    }
    
    tape->num_nodes = 0;
    tape->capacity = initial_capacity;
    tape->variables = nullptr;
    tape->num_variables = 0;
    
    return tape;
}

void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node) {
    if (!tape || !node) {
        eshkol_error("Cannot add node to tape: null parameter");
        return;
    }

    if (tape->num_nodes >= tape->capacity) {
        // Dynamic growth: allocate 2x capacity from the arena, copy existing nodes
        arena_t* arena = __repl_shared_arena.load();
        if (!arena) {
            eshkol_error("Tape capacity exceeded and no arena available for growth: %zu/%zu",
                         tape->num_nodes, tape->capacity);
            return;
        }

        size_t new_capacity = tape->capacity * 2;
        if (new_capacity < 128) new_capacity = 128;  // minimum growth

        size_t new_size = sizeof(ad_node_t*) * new_capacity;
        ad_node_t** new_nodes = (ad_node_t**)arena_allocate_aligned(arena, new_size, 8);

        if (!new_nodes) {
            eshkol_error("Failed to grow tape from %zu to %zu nodes",
                         tape->capacity, new_capacity);
            return;
        }

        // Copy existing nodes to the new array
        memcpy(new_nodes, tape->nodes, sizeof(ad_node_t*) * tape->num_nodes);

        // Update tape to use the new array (old array is arena-allocated, freed with arena)
        tape->nodes = new_nodes;
        tape->capacity = new_capacity;
    }

    tape->nodes[tape->num_nodes++] = node;
}

void arena_tape_reset(ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot reset tape: null parameter");
        return;
    }
    
    // CRITICAL FIX: Reset gradients BEFORE resetting node count!
    // The loop was never executing because num_nodes was already 0
    size_t node_count = tape->num_nodes;
    
    // Reset all node gradients to 0
    for (size_t i = 0; i < node_count; i++) {
        if (tape->nodes[i]) {
            tape->nodes[i]->gradient = 0.0;
        }
    }
    
    // Now reset node count (for next forward pass)
    tape->num_nodes = 0;
}

// Additional tape query functions
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index) {
    if (!tape) {
        eshkol_error("Cannot get node from null tape");
        return nullptr;
    }
    
    if (index >= tape->num_nodes) {
        eshkol_error("Tape index out of bounds: %zu >= %zu", index, tape->num_nodes);
        return nullptr;
    }
    
    return tape->nodes[index];
}

size_t arena_tape_get_node_count(const ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot get node count from null tape");
        return 0;
    }
    
    return tape->num_nodes;
}

// ===== END AD MEMORY MANAGEMENT IMPLEMENTATION =====

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
 *   __ad_tape_stack[MAX_TAPE_DEPTH]   (line 105)
 *   __ad_tape_depth                    (line 106)
 *   __outer_ad_node_stack[MAX_TAPE_DEPTH]
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
    for (size_t i = 0; i < MAX_TAPE_DEPTH; ++i) {
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
    for (size_t i = 0; i < MAX_TAPE_DEPTH; ++i) {
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

// ===== SHARED (REFERENCE-COUNTED) MEMORY MANAGEMENT IMPLEMENTATION =====
// Reference-counted allocation for values with complex, dynamic lifetimes

// Get the shared header from a user pointer (header is before user data)
eshkol_shared_header_t* shared_get_header(void* ptr) {
    if (!ptr) return nullptr;
    // Header is stored before the user data
    return (eshkol_shared_header_t*)((uint8_t*)ptr - sizeof(eshkol_shared_header_t));
}

// Allocate a shared (reference-counted) value
void* shared_allocate(size_t size, void (*destructor)(void*)) {
    return shared_allocate_typed(size, ESHKOL_VALUE_NULL, destructor);
}

void* shared_allocate_typed(size_t size, uint8_t value_type, void (*destructor)(void*)) {
    // Allocate header + user data
    size_t total_size = sizeof(eshkol_shared_header_t) + size;

    // Use malloc for shared allocations (they outlive regions)
    uint8_t* memory = (uint8_t*)malloc(total_size);
    if (!memory) {
        eshkol_error("Failed to allocate shared memory of size %zu", size);
        return nullptr;
    }

    // Initialize header
    eshkol_shared_header_t* header = (eshkol_shared_header_t*)memory;
    header->destructor = destructor;
    header->ref_count = 1;      // Starts with ref count of 1
    header->weak_count = 0;
    header->flags = 0;
    header->value_type = value_type;
    header->reserved = 0;
    header->reserved2 = 0;

    // Return pointer to user data (after header)
    void* user_data = memory + sizeof(eshkol_shared_header_t);

    eshkol_debug("Allocated shared memory at %p (header at %p), size=%zu, type=%d",
                user_data, (void*)header, size, value_type);

    return user_data;
}

// Increment reference count
void shared_retain(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    header->ref_count++;

    eshkol_debug("Retained shared at %p, ref_count now %u", ptr, header->ref_count);
}

// Decrement reference count, deallocate if zero
void shared_release(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    if (header->ref_count == 0) {
        eshkol_warn("Releasing shared with zero ref count at %p", ptr);
        return;
    }

    header->ref_count--;

    eshkol_debug("Released shared at %p, ref_count now %u", ptr, header->ref_count);

    if (header->ref_count == 0) {
        // Call destructor if provided
        if (header->destructor) {
            eshkol_debug("Calling destructor for shared at %p", ptr);
            header->destructor(ptr);
        }

        // If there are no weak references, free immediately
        if (header->weak_count == 0) {
            eshkol_debug("Freeing shared memory at %p", ptr);
            free(header);
        } else {
            // Mark as deallocated but keep header for weak refs
            header->flags |= 0x01;  // DEALLOCATED flag
            eshkol_debug("Shared at %p deallocated but %u weak refs remain",
                        ptr, header->weak_count);
        }
    }
}

// Get current reference count (for debugging)
uint32_t shared_ref_count(void* ptr) {
    if (!ptr) return 0;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return 0;

    return header->ref_count;
}

// Create a weak reference to a shared value
eshkol_weak_ref_t* weak_ref_create(void* shared_ptr) {
    if (!shared_ptr) return nullptr;

    eshkol_shared_header_t* header = shared_get_header(shared_ptr);
    if (!header) return nullptr;

    // Allocate weak ref structure
    eshkol_weak_ref_t* weak = (eshkol_weak_ref_t*)malloc(sizeof(eshkol_weak_ref_t));
    if (!weak) {
        eshkol_error("Failed to allocate weak reference");
        return nullptr;
    }

    weak->header = header;
    weak->data = shared_ptr;

    // Increment weak count
    header->weak_count++;

    eshkol_debug("Created weak ref at %p to shared %p, weak_count now %u",
                (void*)weak, shared_ptr, header->weak_count);

    return weak;
}

// Upgrade weak reference to strong (returns NULL if value was freed)
void* weak_ref_upgrade(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return nullptr;

    // Check if the shared value has been deallocated
    if (weak->header->flags & 0x01) {  // DEALLOCATED flag
        eshkol_debug("Cannot upgrade weak ref - target deallocated");
        return nullptr;
    }

    // Check if ref count is zero (shouldn't happen if not deallocated)
    if (weak->header->ref_count == 0) {
        return nullptr;
    }

    // Increment strong ref count
    weak->header->ref_count++;

    eshkol_debug("Upgraded weak ref at %p to strong, ref_count now %u",
                (void*)weak, weak->header->ref_count);

    return weak->data;
}

// Release a weak reference
void weak_ref_release(eshkol_weak_ref_t* weak) {
    if (!weak) return;

    if (weak->header) {
        weak->header->weak_count--;

        eshkol_debug("Released weak ref at %p, weak_count now %u",
                    (void*)weak, weak->header->weak_count);

        // If shared value was deallocated and this was last weak ref, free header
        if ((weak->header->flags & 0x01) && weak->header->weak_count == 0) {
            eshkol_debug("Freeing shared header at %p (all refs gone)", (void*)weak->header);
            free(weak->header);
        }
    }

    free(weak);
}

// Check if weak reference target still exists
bool weak_ref_is_alive(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return false;

    // Check deallocated flag
    if (weak->header->flags & 0x01) return false;

    // Check ref count
    return weak->header->ref_count > 0;
}

// ===== END SHARED MEMORY MANAGEMENT IMPLEMENTATION =====

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


// ===== CLOSURE ENVIRONMENT MEMORY MANAGEMENT IMPLEMENTATION =====

eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures) {
    if (!arena) {
        eshkol_error("Cannot allocate closure environment: null arena");
        return nullptr;
    }
    
    if (num_captures == 0) {
        eshkol_warn("Allocating closure environment with zero captures");
    }
    
    // Calculate total size: header + captures array
    size_t size = sizeof(eshkol_closure_env_t) + 
                  (num_captures * sizeof(eshkol_tagged_value_t));
    
    eshkol_closure_env_t* env = (eshkol_closure_env_t*)
        arena_allocate_aligned(arena, size, 16);
    
    if (!env) {
        eshkol_error("Failed to allocate closure environment for %zu captures", num_captures);
        return nullptr;
    }
    
    // Initialize environment header
    env->num_captures = num_captures;
    
    // Initialize all captures to null
    for (size_t i = 0; i < num_captures; i++) {
        env->captures[i].type = ESHKOL_VALUE_NULL;
        env->captures[i].flags = 0;
        env->captures[i].reserved = 0;
        env->captures[i].data.raw_val = 0;
    }
    
    eshkol_debug("Allocated closure environment for %zu captures at %p",
                num_captures, (void*)env);

    return env;
}

extern "C" eshkol_closure_t* arena_allocate_closure(arena_t* arena, uint64_t func_ptr, size_t packed_info,
                                         uint64_t sexpr_ptr, uint64_t return_type_info,
                                         const char* name) {
    if (!arena) {
        eshkol_error("Cannot allocate closure: null arena");
        return nullptr;
    }

    // VARIADIC CLOSURE FIX: Unpack the num_captures from the packed info
    // The packed_info field contains:
    //   - Bits 0-15:  actual num_captures
    //   - Bits 16-31: fixed_param_count
    //   - Bit 63:     is_variadic flag
    size_t actual_num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(packed_info);

    // Allocate the closure structure
    eshkol_closure_t* closure = (eshkol_closure_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_closure_t), 16);

    if (!closure) {
        eshkol_error("Failed to allocate closure structure");
        return nullptr;
    }

    closure->func_ptr = func_ptr;
    closure->sexpr_ptr = sexpr_ptr;  // Store S-expression for homoiconicity
    closure->name = name;            // Store procedure name (NULL for anonymous lambdas)

    // Unpack return type metadata:
    //   - Bits 0-7:   return_type (CLOSURE_RETURN_*)
    //   - Bits 8-15:  input_arity
    //   - Bits 16-47: hott_type_id
    closure->return_type = (uint8_t)(return_type_info & 0xFF);
    closure->input_arity = (uint8_t)((return_type_info >> 8) & 0xFF);
    closure->flags = CLOSURE_ENV_IS_VARIADIC(packed_info) ? CLOSURE_FLAG_VARIADIC : 0;
    if (name) {
        closure->flags |= ESHKOL_CLOSURE_FLAG_NAMED;
    }
    closure->reserved = 0;
    closure->hott_type_id = (uint32_t)((return_type_info >> 16) & 0xFFFFFFFF);

    // Allocate environment if there are captures
    if (actual_num_captures > 0) {
        // Allocate env with actual capture count
        closure->env = arena_allocate_closure_env(arena, actual_num_captures);
        if (!closure->env) {
            eshkol_error("Failed to allocate closure environment");
            return nullptr;
        }
        // Store the full packed_info (including variadic flag) in the env's num_captures field
        closure->env->num_captures = packed_info;
    } else {
        closure->env = nullptr;
    }

    eshkol_debug("Allocated closure at %p with func_ptr=%p, env=%p (%zu captures), return_type=%d, arity=%d, name=%s",
                (void*)closure, (void*)func_ptr, (void*)closure->env, actual_num_captures,
                closure->return_type, closure->input_arity, name ? name : "(anonymous)");

    return closure;
}

// ===== END CLOSURE ENVIRONMENT MEMORY MANAGEMENT IMPLEMENTATION =====

// ===== HASH TABLE IMPLEMENTATION =====

// FNV-1a hash constants
static const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
static const uint64_t FNV_PRIME = 1099511628211ULL;

// Hash a string using FNV-1a
static uint64_t fnv1a_hash_string(const char* str) {
    uint64_t hash = FNV_OFFSET_BASIS;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= FNV_PRIME;
    }
    return hash;
}

// Hash a 64-bit value
static uint64_t fnv1a_hash_u64(uint64_t val) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (int i = 0; i < 8; i++) {
        hash ^= (val >> (i * 8)) & 0xFF;
        hash *= FNV_PRIME;
    }
    return hash;
}

// Hash a tagged value
uint64_t hash_tagged_value(const eshkol_tagged_value_t* value) {
    if (!value) return 0;

    // Compute base type: legacy/consolidated (>= 8) use directly, immediate (< 8) mask
    uint8_t full = value->type;
    uint8_t type = (full >= 8) ? full : (full & 0x0F);
    uint64_t hash = FNV_OFFSET_BASIS;

    // Mix in the type
    hash ^= type;
    hash *= FNV_PRIME;

    switch (type) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
        case ESHKOL_VALUE_CHAR:
            hash ^= fnv1a_hash_u64(value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE: {
            // Hash double as its bit pattern
            uint64_t bits;
            memcpy(&bits, &value->data.double_val, sizeof(double));
            hash ^= fnv1a_hash_u64(bits);
            break;
        }

        case ESHKOL_VALUE_STRING_PTR:
            if (value->data.ptr_val) {
                hash ^= fnv1a_hash_string((const char*)value->data.ptr_val);
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            // Symbols are interned, so we can hash the pointer directly
            hash ^= fnv1a_hash_u64(value->data.ptr_val);
            break;

        case ESHKOL_VALUE_HEAP_PTR:
            // M1 CONSOLIDATION: Check subtype in header to determine how to hash
            if (value->data.ptr_val) {
                uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)value->data.ptr_val);
                if (subtype == HEAP_SUBTYPE_STRING) {
                    // Hash string content, not pointer address
                    hash ^= fnv1a_hash_string((const char*)value->data.ptr_val);
                } else if (subtype == HEAP_SUBTYPE_BIGNUM) {
                    // Hash bignum by sign + limbs for value-based equality
                    eshkol_bignum_t* bn = (eshkol_bignum_t*)value->data.ptr_val;
                    hash ^= (uint64_t)bn->sign;
                    hash *= FNV_PRIME;
                    uint64_t* limbs = BIGNUM_LIMBS(bn);
                    for (uint32_t i = 0; i < bn->num_limbs; i++) {
                        hash ^= fnv1a_hash_u64(limbs[i]);
                        hash *= FNV_PRIME;
                    }
                } else {
                    // For other heap types (cons, vector, tensor, hash), hash pointer
                    hash ^= fnv1a_hash_u64(value->data.ptr_val);
                }
            }
            break;

        default:
            // For other types, hash the raw pointer/value
            hash ^= fnv1a_hash_u64(value->data.raw_val);
            break;
    }

    return hash;
}

// Compare two tagged values for equality (as hash keys)
bool hash_keys_equal(const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b) {
    if (!a || !b) return a == b;

    // Compute base type: legacy/consolidated (>= 8) use directly, immediate (< 8) mask
    auto get_base_type = [](uint8_t t) -> uint8_t {
        return (t >= 8) ? t : (t & 0x0F);
    };
    uint8_t type_a = get_base_type(a->type);
    uint8_t type_b = get_base_type(b->type);

    // Must be same type to be equal (for hash key purposes)
    if (type_a != type_b) return false;

    switch (type_a) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
        case ESHKOL_VALUE_CHAR:
            return a->data.int_val == b->data.int_val;

        case ESHKOL_VALUE_DOUBLE:
            return a->data.double_val == b->data.double_val;

        case ESHKOL_VALUE_STRING_PTR:
            if (a->data.ptr_val == b->data.ptr_val) return true;
            if (!a->data.ptr_val || !b->data.ptr_val) return false;
            return strcmp((const char*)a->data.ptr_val, (const char*)b->data.ptr_val) == 0;

        case ESHKOL_VALUE_SYMBOL:
            // Symbols are interned - pointer equality is sufficient
            return a->data.ptr_val == b->data.ptr_val;

        case ESHKOL_VALUE_NULL:
            return true;  // All nulls are equal

        case ESHKOL_VALUE_HEAP_PTR: {
            // M1 CONSOLIDATION: Check subtype in header to determine comparison method
            if (a->data.ptr_val == b->data.ptr_val) return true;
            if (!a->data.ptr_val || !b->data.ptr_val) return false;

            uint8_t subtype_a = ESHKOL_GET_SUBTYPE((void*)a->data.ptr_val);
            uint8_t subtype_b = ESHKOL_GET_SUBTYPE((void*)b->data.ptr_val);

            // Different subtypes are not equal
            if (subtype_a != subtype_b) return false;

            if (subtype_a == HEAP_SUBTYPE_STRING) {
                // Compare string content, not pointer addresses
                return strcmp((const char*)a->data.ptr_val, (const char*)b->data.ptr_val) == 0;
            }

            if (subtype_a == HEAP_SUBTYPE_BIGNUM) {
                // Compare bignum values
                return eshkol_bignum_compare((const eshkol_bignum_t*)a->data.ptr_val,
                                             (const eshkol_bignum_t*)b->data.ptr_val) == 0;
            }

            // For other heap types (cons, vector, tensor, hash), compare pointers
            return a->data.ptr_val == b->data.ptr_val;
        }

        default:
            // For other types, compare raw values
            return a->data.raw_val == b->data.raw_val;
    }
}

// Allocate a hash table with specified initial capacity
eshkol_hash_table_t* arena_allocate_hash_table(arena_t* arena, size_t initial_capacity) {
    if (!arena || initial_capacity == 0) {
        eshkol_error("Invalid parameters for hash table allocation");
        return nullptr;
    }

    // Allocate the hash table structure
    eshkol_hash_table_t* table = (eshkol_hash_table_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_hash_table_t), 8);
    if (!table) return nullptr;

    // Allocate arrays for keys, values, and status
    table->keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * initial_capacity);

    if (!table->keys || !table->values || !table->status) {
        eshkol_error("Failed to allocate hash table arrays");
        return nullptr;
    }

    table->capacity = initial_capacity;
    table->size = 0;
    table->tombstones = 0;

    return table;
}

// Create a hash table with default initial capacity
eshkol_hash_table_t* arena_hash_table_create(arena_t* arena) {
    return arena_allocate_hash_table(arena, HASH_TABLE_INITIAL_CAPACITY);
}

// Create a hash table with object header (for consolidated HEAP_PTR type)
eshkol_hash_table_t* arena_hash_table_create_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Invalid arena for hash table allocation");
        return nullptr;
    }

    // Calculate total size: header + hash table structure
    size_t data_size = sizeof(eshkol_hash_table_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;  // 8-byte alignment

    // Allocate header + hash table as one block
    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate hash table with header");
        return nullptr;
    }

    // Initialize the object header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_HASH;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Get pointer to hash table structure (after header)
    eshkol_hash_table_t* table = (eshkol_hash_table_t*)(mem + sizeof(eshkol_object_header_t));

    // Allocate arrays for keys, values, and status
    size_t initial_capacity = HASH_TABLE_INITIAL_CAPACITY;
    table->keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * initial_capacity);

    if (!table->keys || !table->values || !table->status) {
        eshkol_error("Failed to allocate hash table arrays");
        return nullptr;
    }

    table->capacity = initial_capacity;
    table->size = 0;
    table->tombstones = 0;

    return table;
}

// ===== TENSOR ALLOCATION FUNCTIONS =====

// Allocate tensor with object header (for consolidated HEAP_PTR type)
// Returns pointer to tensor data (header is at offset -8)
// Does NOT allocate dimensions or elements arrays - caller must allocate separately
eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    // Calculate total size: header + tensor structure
    size_t data_size = sizeof(eshkol_tensor_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 63) & ~63;  // 64-byte alignment for AVX-512

    // Allocate header + tensor as one block (64-byte aligned for SIMD)
    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 64);
    if (!mem) {
        eshkol_error("Failed to allocate tensor with header");
        return nullptr;
    }

    // Initialize the object header
    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_TENSOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    // Get pointer to tensor structure (after header)
    eshkol_tensor_t* tensor = (eshkol_tensor_t*)(mem + sizeof(eshkol_object_header_t));

    // Initialize to zero (caller will set dimensions, elements, etc.)
    tensor->dimensions = nullptr;
    tensor->num_dimensions = 0;
    tensor->elements = nullptr;
    tensor->total_elements = 0;

    return tensor;
}

// Allocate tensor with dimensions and elements arrays in one call
// Returns fully initialized tensor with dimensions and elements arrays allocated
eshkol_tensor_t* arena_allocate_tensor_full(arena_t* arena, uint64_t num_dims, uint64_t total_elements) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    // First allocate the tensor struct with header
    eshkol_tensor_t* tensor = arena_allocate_tensor_with_header(arena);
    if (!tensor) {
        return nullptr;
    }

    // Allocate dimensions array
    if (num_dims > 0) {
        if (num_dims > SIZE_MAX / sizeof(uint64_t)) {
            eshkol_error("Tensor dimensions allocation overflow (num_dims=%zu)", num_dims);
            return nullptr;
        }
        tensor->dimensions = (uint64_t*)arena_allocate_aligned(arena, num_dims * sizeof(uint64_t), 64);
        if (!tensor->dimensions) {
            eshkol_error("Failed to allocate tensor dimensions array");
            return nullptr;
        }
    }

    // Allocate elements array (int64_t for storing double bit patterns)
    if (total_elements > 0) {
        // Overflow check: total_elements * sizeof(int64_t) can wrap at SIZE_MAX
        if (total_elements > SIZE_MAX / sizeof(int64_t)) {
            eshkol_error("Tensor elements allocation overflow (total_elements=%zu)", total_elements);
            return nullptr;
        }
        size_t elem_size = total_elements * sizeof(int64_t);
        tensor->elements = (int64_t*)arena_allocate_aligned(arena, elem_size, 64);
        if (tensor->elements) memset(tensor->elements, 0, elem_size);
        if (!tensor->elements) {
            eshkol_error("Failed to allocate tensor elements array");
            return nullptr;
        }
    }

    tensor->num_dimensions = num_dims;
    tensor->total_elements = total_elements;

    return tensor;
}

// ===== END TENSOR ALLOCATION FUNCTIONS =====

// Find slot for a key (returns -1 if not found, otherwise returns index)
// Also returns first tombstone slot via tombstone_slot parameter
static int64_t find_slot(const eshkol_hash_table_t* table, const eshkol_tagged_value_t* key,
                         int64_t* tombstone_slot) {
    uint64_t hash = hash_tagged_value(key);
    size_t index = hash % table->capacity;
    int64_t first_tombstone = -1;

    for (size_t i = 0; i < table->capacity; i++) {
        size_t probe_index = (index + i) % table->capacity;
        uint8_t status = table->status[probe_index];

        if (status == HASH_ENTRY_EMPTY) {
            // Empty slot - key not in table
            if (tombstone_slot) *tombstone_slot = first_tombstone;
            return -1;
        }

        if (status == HASH_ENTRY_DELETED) {
            // Track first tombstone for potential insertion
            if (first_tombstone == -1) {
                first_tombstone = (int64_t)probe_index;
            }
            continue;
        }

        // status == HASH_ENTRY_OCCUPIED
        if (hash_keys_equal(&table->keys[probe_index], key)) {
            if (tombstone_slot) *tombstone_slot = -1;
            return (int64_t)probe_index;
        }
    }

    // Table is full (shouldn't happen with proper load factor management)
    if (tombstone_slot) *tombstone_slot = first_tombstone;
    return -1;
}

struct hash_table_lock_guard {
    hash_table_lock_guard() { eshkol_hash_table_lock(); }
    ~hash_table_lock_guard() { eshkol_hash_table_unlock(); }
    hash_table_lock_guard(const hash_table_lock_guard&) = delete;
    hash_table_lock_guard& operator=(const hash_table_lock_guard&) = delete;
};

// Resize the hash table when load factor exceeds threshold.
// Caller must hold hash_table_lock_guard.
static bool hash_table_resize(arena_t* arena, eshkol_hash_table_t* table, size_t new_capacity) {
    // Allocate new arrays
    eshkol_tagged_value_t* new_keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * new_capacity);
    eshkol_tagged_value_t* new_values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * new_capacity);
    uint8_t* new_status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * new_capacity);

    if (!new_keys || !new_values || !new_status) {
        return false;
    }

    // Rehash all existing entries
    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            uint64_t hash = hash_tagged_value(&table->keys[i]);
            size_t index = hash % new_capacity;

            // Linear probing to find empty slot
            while (new_status[index] != HASH_ENTRY_EMPTY) {
                index = (index + 1) % new_capacity;
            }

            new_keys[index] = table->keys[i];
            new_values[index] = table->values[i];
            new_status[index] = HASH_ENTRY_OCCUPIED;
        }
    }

    // Update table to use new arrays
    table->keys = new_keys;
    table->values = new_values;
    table->status = new_status;
    table->capacity = new_capacity;
    table->tombstones = 0;  // No tombstones after rehash

    return true;
}

// Set a key-value pair in the hash table
bool hash_table_set(arena_t* arena, eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, const eshkol_tagged_value_t* value) {
    if (!arena || !table || !key || !value) return false;
    hash_table_lock_guard lock;

    // Check if we need to resize (load factor > 0.75)
    double load = (double)(table->size + table->tombstones) / table->capacity;
    if (load > HASH_TABLE_LOAD_FACTOR) {
        if (!hash_table_resize(arena, table, table->capacity * 2)) {
            return false;
        }
    }

    int64_t tombstone_slot;
    int64_t slot = find_slot(table, key, &tombstone_slot);

    if (slot >= 0) {
        // Key exists - update value
        table->values[slot] = *value;
        return true;
    }

    // Key doesn't exist - insert at tombstone slot or find new slot
    size_t insert_index;
    if (tombstone_slot >= 0) {
        insert_index = (size_t)tombstone_slot;
        table->tombstones--;  // Reusing tombstone
    } else {
        // Find empty slot
        uint64_t hash = hash_tagged_value(key);
        insert_index = hash % table->capacity;
        while (table->status[insert_index] != HASH_ENTRY_EMPTY) {
            insert_index = (insert_index + 1) % table->capacity;
        }
    }

    table->keys[insert_index] = *key;
    table->values[insert_index] = *value;
    table->status[insert_index] = HASH_ENTRY_OCCUPIED;
    table->size++;

    return true;
}

// Get a value from the hash table
bool hash_table_get(const eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, eshkol_tagged_value_t* out_value) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;

    int64_t slot = find_slot(table, key, nullptr);
    if (slot < 0) return false;

    if (out_value) {
        *out_value = table->values[slot];
    }
    return true;
}

// Check if a key exists in the hash table
bool hash_table_has_key(const eshkol_hash_table_t* table, const eshkol_tagged_value_t* key) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;
    return find_slot(table, key, nullptr) >= 0;
}

// Remove a key from the hash table
bool hash_table_remove(eshkol_hash_table_t* table, const eshkol_tagged_value_t* key) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;

    int64_t slot = find_slot(table, key, nullptr);
    if (slot < 0) return false;

    // Mark as deleted (tombstone)
    table->status[slot] = HASH_ENTRY_DELETED;
    table->size--;
    table->tombstones++;

    return true;
}

// Clear all entries from the hash table
void hash_table_clear(eshkol_hash_table_t* table) {
    if (!table) return;
    hash_table_lock_guard lock;

    memset(table->status, HASH_ENTRY_EMPTY, table->capacity);
    table->size = 0;
    table->tombstones = 0;
}

// Get the number of entries in the hash table
size_t hash_table_count(const eshkol_hash_table_t* table) {
    if (!table) return 0;
    hash_table_lock_guard lock;
    return table->size;
}

// Get all keys as a list
arena_tagged_cons_cell_t* hash_table_keys(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table) return nullptr;
    hash_table_lock_guard lock;
    if (table->size == 0) return nullptr;

    arena_tagged_cons_cell_t* head = nullptr;
    arena_tagged_cons_cell_t* tail = nullptr;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            // M1 CONSOLIDATION: Use arena_allocate_cons_with_header for proper HEAP_PTR support
            arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
            if (!cell) return head;  // Return what we have so far

            // Set car to the key
            arena_tagged_cons_set_tagged_value(cell, false, &table->keys[i]);
            // Set cdr to null initially
            arena_tagged_cons_set_null(cell, true);

            if (!head) {
                head = tail = cell;
            } else {
                // Set previous tail's cdr to this cell
                // M1 CONSOLIDATION: Use HEAP_PTR (8) not legacy CONS_PTR (32)
                eshkol_tagged_value_t cell_val;
                cell_val.type = ESHKOL_VALUE_HEAP_PTR;
                cell_val.flags = 0;
                cell_val.reserved = 0;
                cell_val.data.ptr_val = (uint64_t)cell;
                arena_tagged_cons_set_tagged_value(tail, true, &cell_val);
                tail = cell;
            }
        }
    }

    return head;
}

// Reverse a tagged-cons list. Walks the list, allocates a new cons cell per
// element, writes the resulting tagged_value (NULL for empty) into *out.
// Stops at any non-CONS terminator (NULL, dotted-pair tail, etc.) — the
// function is robust to dotted lists but the dotted tail is dropped
// (matching R7RS reverse on improper lists). Used by string-split codegen.
//
// Output-pointer rather than struct-return-by-value sidesteps the ABI
// coupling between LLVM's IR-level struct and the platform's calling
// convention for ≤16-byte structs.
extern "C" void eshkol_list_reverse_tagged(
    arena_t* arena, const eshkol_tagged_value_t* head_tv,
    eshkol_tagged_value_t* out) {
    if (!out) return;
    out->type = ESHKOL_VALUE_NULL;
    out->flags = 0;
    out->reserved = 0;
    out->data.ptr_val = 0;
    if (!arena || !head_tv) return;

    eshkol_tagged_value_t cur = *head_tv;
    while (cur.type == ESHKOL_VALUE_HEAP_PTR) {
        // Validate it's actually a cons cell via the object header.
        eshkol_object_header_t* hdr = (eshkol_object_header_t*)
            ((uint8_t*)cur.data.ptr_val - sizeof(eshkol_object_header_t));
        if (hdr->subtype != HEAP_SUBTYPE_CONS) break;
        arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)cur.data.ptr_val;

        arena_tagged_cons_cell_t* new_cell = arena_allocate_cons_with_header(arena);
        if (!new_cell) return;
        arena_tagged_cons_set_tagged_value(new_cell, false, &cell->car);
        arena_tagged_cons_set_tagged_value(new_cell, true, out);

        out->type = ESHKOL_VALUE_HEAP_PTR;
        out->flags = 0;
        out->reserved = 0;
        out->data.ptr_val = (uint64_t)new_cell;

        cur = cell->cdr;
    }
}

// Get all values as a list
arena_tagged_cons_cell_t* hash_table_values(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table) return nullptr;
    hash_table_lock_guard lock;
    if (table->size == 0) return nullptr;

    arena_tagged_cons_cell_t* head = nullptr;
    arena_tagged_cons_cell_t* tail = nullptr;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            // M1 CONSOLIDATION: Use arena_allocate_cons_with_header for proper HEAP_PTR support
            arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
            if (!cell) return head;

            // Set car to the value
            arena_tagged_cons_set_tagged_value(cell, false, &table->values[i]);
            // Set cdr to null initially
            arena_tagged_cons_set_null(cell, true);

            if (!head) {
                head = tail = cell;
            } else {
                // M1 CONSOLIDATION: Use HEAP_PTR (8) not legacy CONS_PTR (32)
                eshkol_tagged_value_t cell_val;
                cell_val.type = ESHKOL_VALUE_HEAP_PTR;
                cell_val.flags = 0;
                cell_val.reserved = 0;
                cell_val.data.ptr_val = (uint64_t)cell;
                arena_tagged_cons_set_tagged_value(tail, true, &cell_val);
                tail = cell;
            }
        }
    }

    return head;
}

// ===== END HASH TABLE IMPLEMENTATION =====

// ===== EXCEPTION HANDLING IMPLEMENTATION =====
// Runtime support for R7RS-compatible exception handling

// Global exception state
eshkol_exception_t* g_current_exception = nullptr;
eshkol_exception_handler_t* g_exception_handler_stack = nullptr;

// R7RS: stores the original raised tagged_value for with-exception-handler
eshkol_tagged_value_t g_raised_tagged_value = {0, 0, 0, {0}};
static bool g_raised_value_set_by_user = false;

// Store a tagged value before raising (called from codegen for user `raise`)
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t* value) {
    g_raised_tagged_value = *value;
    g_raised_value_set_by_user = true;
}

// Retrieve the raised tagged value (called from with-exception-handler and guard)
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t* out) {
    *out = g_raised_tagged_value;
}

// Create a new exception object with object header (for consolidated HEAP_PTR type)
extern "C" eshkol_exception_t* eshkol_make_exception_with_header(eshkol_exception_type_t type, const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        eshkol_error("No arena available for exception allocation");
        return nullptr;
    }

    size_t data_size = sizeof(eshkol_exception_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate exception with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_EXCEPTION;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    eshkol_exception_t* exc = (eshkol_exception_t*)(mem + sizeof(eshkol_object_header_t));

    exc->type = type;
    if (message) {
        size_t len = strlen(message) + 1;
        exc->message = (char*)arena_allocate(arena, len);
        if (exc->message) {
            memcpy(exc->message, message, len - 1);
            exc->message[len - 1] = '\0';
        }
    } else {
        exc->message = nullptr;
    }
    exc->irritants = nullptr;
    exc->num_irritants = 0;
    exc->line = 0;
    exc->column = 0;
    exc->filename = nullptr;

    return exc;
}

// Create a new exception object (legacy - no header)
extern "C" eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type, const char* message) {
    arena_t* arena = __repl_shared_arena.load();
    if (!arena) {
        // Allocate from heap if no arena available
        eshkol_exception_t* exc = (eshkol_exception_t*)malloc(sizeof(eshkol_exception_t));
        if (!exc) return nullptr;

        exc->type = type;
        exc->message = message ? strdup(message) : nullptr;
        exc->irritants = nullptr;
        exc->num_irritants = 0;
        exc->line = 0;
        exc->column = 0;
        exc->filename = nullptr;
        return exc;
    }

    // Allocate from arena
    eshkol_exception_t* exc = (eshkol_exception_t*)arena_allocate(arena, sizeof(eshkol_exception_t));
    if (!exc) return nullptr;

    exc->type = type;
    if (message) {
        size_t len = strlen(message) + 1;
        exc->message = (char*)arena_allocate(arena, len);
        if (exc->message) {
            memcpy(exc->message, message, len - 1);
            exc->message[len - 1] = '\0';
        }
    } else {
        exc->message = nullptr;
    }
    exc->irritants = nullptr;
    exc->num_irritants = 0;
    exc->line = 0;
    exc->column = 0;
    exc->filename = nullptr;

    return exc;
}

// Add an irritant to an exception
extern "C" void eshkol_exception_add_irritant(eshkol_exception_t* exc, eshkol_tagged_value_t irritant) {
    if (!exc) return;

    // Grow irritants array
    uint32_t new_count = exc->num_irritants + 1;
    eshkol_tagged_value_t* new_irritants;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        new_irritants = (eshkol_tagged_value_t*)arena_allocate(arena, new_count * sizeof(eshkol_tagged_value_t));
    } else {
        new_irritants = (eshkol_tagged_value_t*)malloc(new_count * sizeof(eshkol_tagged_value_t));
    }

    if (!new_irritants) return;

    // Copy existing irritants
    if (exc->irritants && exc->num_irritants > 0) {
        memcpy(new_irritants, exc->irritants, exc->num_irritants * sizeof(eshkol_tagged_value_t));
    }

    // Add new irritant
    new_irritants[exc->num_irritants] = irritant;
    exc->irritants = new_irritants;
    exc->num_irritants = new_count;
}

// Set source location on exception
extern "C" void eshkol_exception_set_location(eshkol_exception_t* exc, uint32_t line, uint32_t column, const char* filename) {
    if (!exc) return;

    exc->line = line;
    exc->column = column;

    if (filename) {
        arena_t* arena = __repl_shared_arena.load();
        if (arena) {
            size_t len = strlen(filename) + 1;
            exc->filename = (char*)arena_allocate(arena, len);
            if (exc->filename) {
                memcpy(exc->filename, filename, len - 1);
                exc->filename[len - 1] = '\0';
            }
        } else {
            exc->filename = strdup(filename);
        }
    }
}

// Raise an exception - jumps to nearest handler
/* Bug W: REPL forward-ref call where the named function was never
 * defined. The codegen call site emits this BEFORE the indirect call,
 * passing the function name as a literal. The codegen then calls this
 * helper with the loaded fn_ptr and the stub-detection sentinel. If
 * the slot still points at the unresolved stub, raise with a clear
 * message that names the function. Otherwise return the resolved ptr.
 *
 * The sentinel comparison is necessary because every unresolved slot
 * shares the same stub address — codegen can't tell at emit time
 * whether the slot will be resolved by JIT load order.
 */
/* Bug W ask 2: scan project .esk files for `(provide …)` or `(define …)`
 * lists that mention `name` and suggest the most likely file to (load …).
 *
 * Heuristics — text-based, not a full parse — so this never blocks the
 * error from being raised.  We bound the work: max 4 dirs deep, max 800
 * files, max 256 KB read per file.  Result is an arena-allocated
 * malloc'd buffer (caller-owned), or NULL if nothing found.
 *
 * Search order:
 *   1. CWD recursively
 *   2. ./src recursively
 *   3. ./lib recursively
 *   4. $ESHKOL_PROJECT_ROOT recursively (if set)
 *
 * Match scoring (highest wins):
 *   - 100  exact `(provide name)` form found
 *   -  50  exact `(define name …)` or `(define (name …) …)` found
 *   -  10  name appears in some other position (weak hint)
 */
static char* eshkol_find_provider_file(const char* name);

// Derive an Eshkol module name from a file path under a `lib/` (or
// equivalent) directory. e.g.
//   /Users/.../lib/agent/http.esk  → "agent.http"
//   /Users/.../lib/core/json.esk   → "core.json"
//   /tmp/myfile.esk                → ""  (no module-style location)
// Returns malloc'd string the caller frees, or NULL if no module name
// can be derived.
static char* derive_module_name_from_path(const char* file_path) {
    if (!file_path || !file_path[0]) return nullptr;
    std::string path(file_path);

    // Look for "/lib/" anywhere in the path; the module path starts
    // after that. Falls back to "src/" since some projects use that.
    size_t lib_pos = path.find("/lib/");
    size_t skip = 5;  // "/lib/" length
    if (lib_pos == std::string::npos) {
        lib_pos = path.find("/src/");
        skip = 5;
    }
    if (lib_pos == std::string::npos) return nullptr;

    std::string mod = path.substr(lib_pos + skip);
    // Drop ".esk" suffix
    if (mod.size() > 4 && mod.compare(mod.size() - 4, 4, ".esk") == 0) {
        mod.resize(mod.size() - 4);
    }
    // Replace "/" with "." for module syntax. Reject paths containing
    // any character that wouldn't be valid in a Scheme symbol —
    // unusual filenames mean we can't safely synthesise a module name.
    for (size_t i = 0; i < mod.size(); ++i) {
        char c = mod[i];
        if (c == '/') { mod[i] = '.'; continue; }
        if (!(std::isalnum(static_cast<unsigned char>(c)) ||
              c == '-' || c == '_' || c == '.' || c == '!' || c == '?')) {
            return nullptr;
        }
    }
    if (mod.empty()) return nullptr;
    return strdup(mod.c_str());
}

extern "C" void* eshkol_check_forward_ref(void* loaded_fn_ptr,
                                          void* stub_sentinel,
                                          const char* func_name) {
    if (loaded_fn_ptr == stub_sentinel) {
        char buf[1024];
        char* hint = nullptr;
        if (func_name && func_name[0]) {
            hint = eshkol_find_provider_file(func_name);
            if (hint) {
                // If the matched file looks like a module (has a `lib/`
                // ancestor), suggest `(require module-name)` since that's
                // the modern entry point. Falls back to `(load …)` for
                // ad-hoc files that aren't part of the module tree.
                // Bug W "ask 2" (eshkol-agent 2026-05-06): the previous
                // hint always suggested (load …), but agents normally
                // wire imports through (require …) and provide forms.
                char* module_name = derive_module_name_from_path(hint);
                if (module_name) {
                    snprintf(buf, sizeof(buf),
                             "called undefined function '%s' "
                             "(forward-referenced but never defined). "
                             "Likely missing: (require %s) — that module's "
                             "file %s appears to define '%s'.",
                             func_name, module_name, hint, func_name);
                    std::free(module_name);
                } else {
                    snprintf(buf, sizeof(buf),
                             "called undefined function '%s' "
                             "(forward-referenced but never defined). "
                             "Likely missing: (load \"%s\") — that file "
                             "appears to define '%s'.",
                             func_name, hint, func_name);
                }
                std::free(hint);
            } else {
                snprintf(buf, sizeof(buf),
                         "called undefined function '%s' "
                         "(forward-referenced but never defined; "
                         "check that the file containing its `define` is `(load …)`ed "
                         "or `(require …)`d before the call site)",
                         func_name);
            }
        } else {
            snprintf(buf, sizeof(buf),
                     "called a forward-referenced function that was never defined");
        }
        eshkol_exception_t* exc = eshkol_make_exception(
            ESHKOL_EXCEPTION_ERROR, buf);
        if (exc) {
            eshkol_raise(exc);
        }
        // eshkol_raise exits if no handler. If it returns (handler was
        // installed), we still must not proceed with a stub call —
        // return null and let the caller's null-handling raise an
        // exception of its own. This keeps the "fatal under no handler"
        // invariant while preserving guard-clause behaviour.
        return nullptr;
    }
    return loaded_fn_ptr;
}

// Bug W ask 2 — implementation.
// Walk a few project-relative directories looking for `.esk` files that
// either `(provide … name …)` or `(define name …)` / `(define (name …) …)`.
// Returns a malloc'd string the caller frees, or NULL.  Bounded work:
// max ~800 files, max 256 KB per file, max 4 dirs deep.
namespace {
constexpr size_t kMaxFilesScanned   = 800;
constexpr size_t kMaxFileBytes      = 256 * 1024;
constexpr int    kMaxDepth          = 4;
constexpr int    kScoreProvide      = 100;
constexpr int    kScoreDefineHead   =  50;
constexpr int    kScoreDefineParen  =  50;
constexpr int    kScoreWeakMention  =  10;

struct ScanResult {
    std::string best_path;
    int         best_score = 0;
    size_t      files_scanned = 0;
};

bool starts_word(const std::string& s, size_t pos, const std::string& word) {
    if (pos + word.size() > s.size()) return false;
    return s.compare(pos, word.size(), word) == 0;
}

bool name_at(const std::string& haystack, size_t pos, const std::string& name) {
    if (pos + name.size() > haystack.size()) return false;
    if (haystack.compare(pos, name.size(), name) != 0) return false;
    // Word boundary on both sides — Eshkol identifiers can contain alnum,
    // -, _, !, ?, *, +, /, <, >, =, .  Anything else is a separator.
    auto is_id = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) ||
               c == '-' || c == '_' || c == '!' || c == '?' ||
               c == '*' || c == '+' || c == '/' || c == '<' ||
               c == '>' || c == '=' || c == '.';
    };
    if (pos > 0 && is_id(haystack[pos - 1])) return false;
    size_t end = pos + name.size();
    if (end < haystack.size() && is_id(haystack[end])) return false;
    return true;
}

int score_file_text(const std::string& text, const std::string& name) {
    int best = 0;
    size_t pos = 0;
    while ((pos = text.find(name, pos)) != std::string::npos) {
        if (!name_at(text, pos, name)) { pos += name.size(); continue; }

        // Look back to find the most recent `(`
        size_t back = pos;
        while (back > 0 && text[back - 1] != '(' && text[back - 1] != '\n') --back;
        if (back == 0 || text[back - 1] != '(') {
            // No `(` on this line before the name → weak mention only
            if (best < kScoreWeakMention) best = kScoreWeakMention;
            pos += name.size();
            continue;
        }
        // Identify the head form just after `(`
        size_t head_start = back;
        while (head_start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[head_start]))) ++head_start;
        if (starts_word(text, head_start, "provide")) {
            return kScoreProvide;  // can't do better
        }
        if (starts_word(text, head_start, "define")) {
            // (define name …)  OR  (define (name …) …)
            // Distinguish: head is "define", then whitespace, then either
            //   - the name itself
            //   - `(` then the name
            size_t after = head_start + 6;  // past "define"
            while (after < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[after]))) ++after;
            if (after < text.size() && text[after] == '(') {
                ++after;
                while (after < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[after]))) ++after;
                if (after == pos) {
                    if (best < kScoreDefineParen) best = kScoreDefineParen;
                }
            } else if (after == pos) {
                if (best < kScoreDefineHead) best = kScoreDefineHead;
            }
        }
        pos += name.size();
    }
    return best;
}

void scan_dir(const std::filesystem::path& dir, const std::string& name,
              int depth, ScanResult& out) {
    if (depth > kMaxDepth || out.files_scanned >= kMaxFilesScanned) return;
    if (out.best_score >= kScoreProvide) return;  // can't beat that

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
        return;
    }

    for (auto it = std::filesystem::directory_iterator(dir, std::filesystem::directory_options::skip_permission_denied, ec);
         !ec && it != std::filesystem::directory_iterator{} && out.files_scanned < kMaxFilesScanned;
         it.increment(ec)) {
        if (ec) break;
        const auto& entry = *it;
        std::error_code ec2;
        if (entry.is_directory(ec2)) {
            // Skip well-known build / vendor / hidden dirs
            std::string dn = entry.path().filename().string();
            if (dn.empty() || dn[0] == '.' ||
                dn == "build" || dn == "build-x64" || dn == "build-asan" ||
                dn == "build-xla" || dn == "build-cuda" || dn == "build-debug" ||
                dn == "node_modules" || dn == "deps" || dn == "dist" ||
                dn == "artifacts" || dn == "Testing") continue;
            scan_dir(entry.path(), name, depth + 1, out);
            continue;
        }
        if (!entry.is_regular_file(ec2)) continue;
        if (entry.path().extension() != ".esk") continue;

        ++out.files_scanned;
        std::ifstream f(entry.path(), std::ios::binary);
        if (!f) continue;
        // Read up to kMaxFileBytes
        std::string text;
        text.resize(kMaxFileBytes);
        f.read(text.data(), kMaxFileBytes);
        text.resize(f.gcount());

        int sc = score_file_text(text, name);
        if (sc > out.best_score) {
            out.best_score = sc;
            out.best_path  = entry.path().string();
            if (out.best_score >= kScoreProvide) return;
        }
    }
}
}  // anonymous namespace

static char* eshkol_find_provider_file(const char* name) {
    if (!name || !name[0]) return nullptr;
    std::string sname(name);

    ScanResult res;
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        scan_dir(cwd, sname, 0, res);
    }
    if (res.best_score < kScoreProvide) {
        if (const char* root = std::getenv("ESHKOL_PROJECT_ROOT")) {
            scan_dir(fs::path(root), sname, 0, res);
        }
    }
    if (res.best_score == 0) return nullptr;
    return strdup(res.best_path.c_str());
}

extern "C" void eshkol_raise(eshkol_exception_t* exception) {
    g_current_exception = exception;

    // If user didn't set a raised value via eshkol_set_raised_value,
    // use the exception pointer as a fallback HEAP_PTR tagged value
    if (!g_raised_value_set_by_user) {
        g_raised_tagged_value.type = ESHKOL_VALUE_HEAP_PTR;
        g_raised_tagged_value.flags = 0;
        g_raised_tagged_value.reserved = 0;
        g_raised_tagged_value.data.ptr_val = (uint64_t)exception;
    }
    g_raised_value_set_by_user = false;  // Reset for next raise

    if (g_exception_handler_stack && g_exception_handler_stack->jmp_buf_ptr) {
        // Jump to the handler
        longjmp(*(jmp_buf*)g_exception_handler_stack->jmp_buf_ptr, 1);
    } else {
        // No handler - print error and exit gracefully (not abort)
        fprintf(stderr, "Unhandled exception: ");
        if (exception && exception->message) {
            fprintf(stderr, "%s", exception->message);
        } else {
            fprintf(stderr, "(unknown error)");
        }
        if (exception && exception->line > 0) {
            fprintf(stderr, " at line %u", exception->line);
            if (exception->column > 0) {
                fprintf(stderr, ", column %u", exception->column);
            }
            if (exception->filename) {
                fprintf(stderr, " in %s", exception->filename);
            }
        }
        fprintf(stderr, "\n");
        exit(1);
    }
}

// Push exception handler onto stack
extern "C" void eshkol_push_exception_handler(void* jmp_buf_ptr) {
    eshkol_exception_handler_t* handler;

    arena_t* arena = __repl_shared_arena.load();
    if (arena) {
        handler = (eshkol_exception_handler_t*)arena_allocate(arena, sizeof(eshkol_exception_handler_t));
    } else {
        handler = (eshkol_exception_handler_t*)malloc(sizeof(eshkol_exception_handler_t));
    }

    if (!handler) {
        eshkol_error("Failed to allocate exception handler");
        return;
    }

    handler->jmp_buf_ptr = jmp_buf_ptr;
    handler->prev = g_exception_handler_stack;
    g_exception_handler_stack = handler;
}

// Pop exception handler from stack
extern "C" void eshkol_pop_exception_handler(void) {
    if (g_exception_handler_stack) {
        eshkol_exception_handler_t* popped = g_exception_handler_stack;
        g_exception_handler_stack = popped->prev;
        // Note: If allocated from arena, memory is automatically freed with arena
        // If from heap, we leak here - but exception handlers should be short-lived
    }
}

// Check if exception matches a specific type
extern "C" int eshkol_exception_type_matches(eshkol_exception_t* exc, eshkol_exception_type_t type) {
    if (!exc) return 0;
    return exc->type == type;
}

// Get current exception (for handlers)
extern "C" eshkol_exception_t* eshkol_get_current_exception(void) {
    return g_current_exception;
}

// Clear current exception
extern "C" void eshkol_clear_current_exception(void) {
    g_current_exception = nullptr;
}

// Display exception for debugging
extern "C" void eshkol_display_exception(eshkol_exception_t* exc) {
    if (!exc) {
        printf("#<exception:null>");
        return;
    }

    const char* type_name;
    switch (exc->type) {
        case ESHKOL_EXCEPTION_ERROR: type_name = "error"; break;
        case ESHKOL_EXCEPTION_TYPE_ERROR: type_name = "type-error"; break;
        case ESHKOL_EXCEPTION_FILE_ERROR: type_name = "file-error"; break;
        case ESHKOL_EXCEPTION_READ_ERROR: type_name = "read-error"; break;
        case ESHKOL_EXCEPTION_SYNTAX_ERROR: type_name = "syntax-error"; break;
        case ESHKOL_EXCEPTION_RANGE_ERROR: type_name = "range-error"; break;
        case ESHKOL_EXCEPTION_ARITY_ERROR: type_name = "arity-error"; break;
        case ESHKOL_EXCEPTION_DIVIDE_BY_ZERO: type_name = "divide-by-zero"; break;
        case ESHKOL_EXCEPTION_USER_DEFINED: type_name = "user-exception"; break;
        default: type_name = "unknown"; break;
    }

    printf("#<%s: %s>", type_name, exc->message ? exc->message : "");
}

// ===== END EXCEPTION HANDLING IMPLEMENTATION =====

// ===== FIRST-CLASS CONTINUATIONS RUNTIME =====

// Global dynamic-wind handler stack
eshkol_dynamic_wind_entry_t* g_dynamic_wind_stack = nullptr;

extern "C" eshkol_continuation_state_t* eshkol_make_continuation_state(void* arena_void, void* jmp_buf_ptr) {
    arena_t* arena = (arena_t*)arena_void;
    eshkol_continuation_state_t* state = (eshkol_continuation_state_t*)arena_allocate_aligned(arena, sizeof(eshkol_continuation_state_t), 8);
    if (!state) {
        eshkol_error("Failed to allocate continuation state");
        return nullptr;
    }
    state->jmp_buf_ptr = jmp_buf_ptr;
    memset(&state->value, 0, sizeof(eshkol_tagged_value_t));
    state->value.type = ESHKOL_VALUE_NULL;
    state->wind_mark = (void*)g_dynamic_wind_stack;
    return state;
}

extern "C" void* eshkol_make_continuation_closure(void* arena_void, void* state_ptr) {
    arena_t* arena = (arena_t*)arena_void;

    // Allocate closure with 1 capture (the state pointer)
    // packed_info: 1 capture in bits 0-15, 1 fixed param in bits 16-31
    size_t packed_info = 1 | (1ULL << 16);  // 1 capture, 1 param (arity=1)
    eshkol_closure_t* closure = arena_allocate_closure_with_header(
        arena, 0, packed_info, 0, 0, "<continuation>");

    if (!closure) {
        eshkol_error("Failed to allocate continuation closure");
        return nullptr;
    }

    // Override the header subtype to CALLABLE_SUBTYPE_CONTINUATION
    uint8_t* closure_bytes = (uint8_t*)closure;
    eshkol_object_header_t* header = (eshkol_object_header_t*)(closure_bytes - sizeof(eshkol_object_header_t));
    header->subtype = CALLABLE_SUBTYPE_CONTINUATION;

    // Store state pointer as a tagged value in captures[0]
    if (closure->env) {
        closure->env->captures[0].type = ESHKOL_VALUE_HEAP_PTR;
        closure->env->captures[0].flags = 0;
        closure->env->captures[0].reserved = 0;
        closure->env->captures[0].data.int_val = (uint64_t)(uintptr_t)state_ptr;
    }

    return (void*)closure;
}

// Call a 0-arg Eshkol closure from C runtime (for dynamic-wind thunks)
// Handles closures with 0-4 captures by matching LLVM calling convention
static eshkol_tagged_value_t call_thunk_closure(eshkol_closure_t* closure) {
    if (!closure || !closure->func_ptr) {
        eshkol_tagged_value_t null_val;
        memset(&null_val, 0, sizeof(null_val));
        null_val.type = ESHKOL_VALUE_NULL;
        return null_val;
    }

    size_t num_captures = 0;
    if (closure->env) {
        num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(closure->env->num_captures);
    }

    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));
    result.type = ESHKOL_VALUE_NULL;

#if defined(__aarch64__) || defined(_M_ARM64)
    // AArch64 returns this 16-byte aggregate in registers, so the thunk bridge
    // must match the direct return ABI instead of passing a hidden result slot.
    typedef eshkol_tagged_value_t (*fn0_t)(void);
    typedef eshkol_tagged_value_t (*fn1_t)(void*);
    typedef eshkol_tagged_value_t (*fn2_t)(void*, void*);
    typedef eshkol_tagged_value_t (*fn3_t)(void*, void*, void*);
    typedef eshkol_tagged_value_t (*fn4_t)(void*, void*, void*, void*);

    switch (num_captures) {
        case 0:
            result = ((fn0_t)(uintptr_t)closure->func_ptr)();
            break;
        case 1:
            result = ((fn1_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0]);
            break;
        case 2:
            result = ((fn2_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1]);
            break;
        case 3:
            result = ((fn3_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2]);
            break;
        case 4:
            result = ((fn4_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2], &closure->env->captures[3]);
            break;
        default:
            result = ((fn0_t)(uintptr_t)closure->func_ptr)();
            break;
    }
#else
    // The currently-supported x86/Windows thunk ABI uses a hidden return buffer,
    // so the runtime bridge must pass the result slot first.
    typedef void (*fn0_t)(eshkol_tagged_value_t*);
    typedef void (*fn1_t)(eshkol_tagged_value_t*, void*);
    typedef void (*fn2_t)(eshkol_tagged_value_t*, void*, void*);
    typedef void (*fn3_t)(eshkol_tagged_value_t*, void*, void*, void*);
    typedef void (*fn4_t)(eshkol_tagged_value_t*, void*, void*, void*, void*);

    switch (num_captures) {
        case 0:
            ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
            break;
        case 1:
            ((fn1_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0]);
            break;
        case 2:
            ((fn2_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1]);
            break;
        case 3:
            ((fn3_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2]);
            break;
        case 4:
            ((fn4_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2], &closure->env->captures[3]);
            break;
        default:
            ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
            break;
    }
#endif

    return result;
}

// Call a thunk stored as a tagged value (CALLABLE type)
static void call_thunk_from_tagged(const eshkol_tagged_value_t* thunk) {
    if (!thunk || thunk->type != ESHKOL_VALUE_CALLABLE) return;
    eshkol_closure_t* closure = (eshkol_closure_t*)(uintptr_t)thunk->data.int_val;
    call_thunk_closure(closure);
}

// Push a dynamic-wind entry onto the global stack
extern "C" void eshkol_push_dynamic_wind(void* arena_void,
    const eshkol_tagged_value_t* before, const eshkol_tagged_value_t* after) {
    arena_t* arena = (arena_t*)arena_void;
    eshkol_dynamic_wind_entry_t* entry = (eshkol_dynamic_wind_entry_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dynamic_wind_entry_t), 8);
    if (!entry) return;
    entry->before = *before;
    entry->after = *after;
    entry->prev = g_dynamic_wind_stack;
    g_dynamic_wind_stack = entry;
}

// Pop the top dynamic-wind entry
extern "C" void eshkol_pop_dynamic_wind(void) {
    if (g_dynamic_wind_stack) {
        g_dynamic_wind_stack = g_dynamic_wind_stack->prev;
    }
}

// Unwind dynamic-wind stack down to a saved mark, calling after thunks
extern "C" void eshkol_unwind_dynamic_wind(void* saved_wind_mark) {
    eshkol_dynamic_wind_entry_t* mark = (eshkol_dynamic_wind_entry_t*)saved_wind_mark;
    while (g_dynamic_wind_stack != NULL && g_dynamic_wind_stack != mark) {
        eshkol_dynamic_wind_entry_t* entry = g_dynamic_wind_stack;
        g_dynamic_wind_stack = entry->prev;
        call_thunk_from_tagged(&entry->after);
    }
}

// ===== END FIRST-CLASS CONTINUATIONS RUNTIME =====

// ===== LINEAR ALGEBRA RUNTIME FUNCTIONS =====
// These are called from LLVM-generated code via extern "C" linkage.
// They operate on raw double arrays in row-major order.

// LU decomposition with partial pivoting (in-place)
// A is n×n row-major, piv[i] stores the row swapped with row i.
// Returns the sign of the permutation (+1 or -1), or 0 if singular.
extern "C" int64_t eshkol_lu_decompose(double* A, int64_t* piv, int64_t n) {
    int64_t sign = 1;
    for (int64_t i = 0; i < n; i++) piv[i] = i;

    for (int64_t k = 0; k < n; k++) {
        // Find pivot: largest absolute value in column k, rows k..n-1
        double max_val = 0.0;
        int64_t max_row = k;
        for (int64_t i = k; i < n; i++) {
            double v = fabs(A[i * n + k]);
            if (v > max_val) { max_val = v; max_row = i; }
        }
        if (max_val < 1e-15) return 0; // Singular

        // Swap rows k and max_row
        if (max_row != k) {
            sign = -sign;
            int64_t tmp_piv = piv[k]; piv[k] = piv[max_row]; piv[max_row] = tmp_piv;
            for (int64_t j = 0; j < n; j++) {
                double tmp = A[k * n + j];
                A[k * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
        }

        // Eliminate below pivot
        double pivot = A[k * n + k];
        for (int64_t i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / pivot;
            A[i * n + k] = factor; // Store L factor in-place
            for (int64_t j = k + 1; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }
    return sign;
}

// Determinant from LU-decomposed matrix: product of diagonal * sign
extern "C" double eshkol_det_from_lu(const double* LU, int64_t n, int64_t sign) {
    double det = (double)sign;
    for (int64_t i = 0; i < n; i++) {
        det *= LU[i * n + i];
    }
    return det;
}

// Solve Ax=b using LU decomposition (LU and piv from eshkol_lu_decompose)
// b is overwritten with the solution x.
extern "C" void eshkol_lu_solve(const double* LU, const int64_t* piv, double* b, int64_t n) {
    // Apply permutation: create permuted b
    double* pb = (double*)malloc(n * sizeof(double));
    if (!pb) return;
    for (int64_t i = 0; i < n; i++) pb[i] = b[piv[i]];

    // Forward substitution: Ly = pb (L has 1s on diagonal, factors below)
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i; j++) {
            pb[i] -= LU[i * n + j] * pb[j];
        }
    }

    // Back substitution: Ux = y
    for (int64_t i = n - 1; i >= 0; i--) {
        for (int64_t j = i + 1; j < n; j++) {
            pb[i] -= LU[i * n + j] * pb[j];
        }
        pb[i] /= LU[i * n + i];
    }

    memcpy(b, pb, n * sizeof(double));
    free(pb);
}

// Matrix inverse via LU decomposition
// inv is n×n output, LU and piv from eshkol_lu_decompose
extern "C" void eshkol_lu_inverse(const double* LU, const int64_t* piv, double* inv, int64_t n) {
    // Solve LU * x_col = e_col for each column of identity
    for (int64_t col = 0; col < n; col++) {
        // Set up identity column
        double* b = (double*)malloc(n * sizeof(double));
        if (!b) return;
        memset(b, 0, n * sizeof(double));
        b[col] = 1.0;

        eshkol_lu_solve(LU, piv, b, n);

        // Copy solution to inv column
        for (int64_t row = 0; row < n; row++) {
            inv[row * n + col] = b[row];
        }
        free(b);
    }
}

// Cholesky decomposition: A = L @ L^T (A must be SPD)
// L is output (lower triangular), n×n. Returns 0 on success, -1 if not SPD.
extern "C" int64_t eshkol_cholesky(const double* A, double* L, int64_t n) {
    memset(L, 0, n * n * sizeof(double));

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                // Diagonal element
                for (int64_t k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                double val = A[j * n + j] - sum;
                if (val <= 0.0) return -1; // Not SPD
                L[j * n + j] = sqrt(val);
            } else {
                // Off-diagonal element
                for (int64_t k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return 0;
}

// QR decomposition via Householder reflections
// A is m×n input, Q is m×m output, R is m×n output
extern "C" void eshkol_qr_decompose(const double* A, double* Q, double* R, int64_t m, int64_t n) {
    // Copy A to R (we modify R in place)
    memcpy(R, A, m * n * sizeof(double));

    // Initialize Q to identity
    memset(Q, 0, m * m * sizeof(double));
    for (int64_t i = 0; i < m; i++) Q[i * m + i] = 1.0;

    int64_t min_mn = (m < n) ? m : n;

    for (int64_t k = 0; k < min_mn; k++) {
        // Extract column k of R below diagonal
        double* v = (double*)malloc(m * sizeof(double));
        if (!v) return;
        memset(v, 0, m * sizeof(double));

        double norm_sq = 0.0;
        for (int64_t i = k; i < m; i++) {
            v[i] = R[i * n + k];
            norm_sq += v[i] * v[i];
        }
        double norm = sqrt(norm_sq);
        if (norm < 1e-15) { free(v); continue; }

        // Compute Householder vector
        double sign = (v[k] >= 0.0) ? 1.0 : -1.0;
        v[k] += sign * norm;

        // Recompute norm of v for normalization
        double v_norm_sq = 0.0;
        for (int64_t i = k; i < m; i++) v_norm_sq += v[i] * v[i];
        if (v_norm_sq < 1e-30) { free(v); continue; }

        double scale = 2.0 / v_norm_sq;

        // Apply H = I - scale * v * v^T to R: R = H * R
        for (int64_t j = k; j < n; j++) {
            double dot = 0.0;
            for (int64_t i = k; i < m; i++) dot += v[i] * R[i * n + j];
            for (int64_t i = k; i < m; i++) R[i * n + j] -= scale * v[i] * dot;
        }

        // Apply H to Q: Q = Q * H (accumulate from right)
        for (int64_t i = 0; i < m; i++) {
            double dot = 0.0;
            for (int64_t j2 = k; j2 < m; j2++) dot += Q[i * m + j2] * v[j2];
            for (int64_t j2 = k; j2 < m; j2++) Q[i * m + j2] -= scale * dot * v[j2];
        }

        free(v);
    }
}

// ===== SVD VIA ONE-SIDED JACOBI ROTATION =====
// A is m×n input (row-major), U is m×n output, S is min(m,n) output, V is n×n output
// Computes A = U @ diag(S) @ V^T
extern "C" void eshkol_tensor_svd(
    const double* A, int64_t m, int64_t n,
    double* U, double* S, double* V)
{
    int64_t k = (m < n) ? m : n; // min(m, n)

    // B = working copy of A (m × n), will become U * diag(S) during iteration
    double* B = (double*)malloc(m * n * sizeof(double));
    if (!B) return;
    memcpy(B, A, m * n * sizeof(double));

    // Initialize V to n×n identity
    memset(V, 0, n * n * sizeof(double));
    for (int64_t i = 0; i < n; i++) V[i * n + i] = 1.0;

    const double eps = 1e-15;
    const int max_sweeps = 100;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_norm = 0.0;

        // For each column pair (p, q) where p < q < n
        for (int64_t p = 0; p < n - 1; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                // Compute alpha = B_p^T B_p, beta = B_q^T B_q, gamma = B_p^T B_q
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int64_t i = 0; i < m; i++) {
                    double bp = B[i * n + p];
                    double bq = B[i * n + q];
                    alpha += bp * bp;
                    beta  += bq * bq;
                    gamma += bp * bq;
                }

                off_norm += gamma * gamma;

                // Skip if columns are already orthogonal
                double threshold = eps * sqrt(alpha * beta);
                if (threshold < 1e-300) threshold = 1e-300; // avoid 0
                if (fabs(gamma) < threshold) continue;

                // Compute Jacobi rotation angle
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t;
                if (zeta >= 0.0) {
                    t = 1.0 / (zeta + sqrt(zeta * zeta + 1.0));
                } else {
                    t = -1.0 / (-zeta + sqrt(zeta * zeta + 1.0));
                }
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                // Apply rotation to columns of B: B[:, p] and B[:, q]
                for (int64_t i = 0; i < m; i++) {
                    double bp = B[i * n + p];
                    double bq = B[i * n + q];
                    B[i * n + p] =  c * bp + s * bq;
                    B[i * n + q] = -s * bp + c * bq;
                }

                // Apply rotation to columns of V: V[:, p] and V[:, q]
                for (int64_t i = 0; i < n; i++) {
                    double vp = V[i * n + p];
                    double vq = V[i * n + q];
                    V[i * n + p] =  c * vp + s * vq;
                    V[i * n + q] = -s * vp + c * vq;
                }
            }
        }

        // Check convergence
        if (off_norm < eps * eps) break;
    }

    // Extract singular values and U columns from B
    // S[j] = ||B[:, j]||, U[:, j] = B[:, j] / S[j]
    for (int64_t j = 0; j < k; j++) {
        double norm = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double v = B[i * n + j];
            norm += v * v;
        }
        norm = sqrt(norm);
        S[j] = norm;

        if (norm > eps) {
            for (int64_t i = 0; i < m; i++) {
                U[i * k + j] = B[i * n + j] / norm;
            }
        } else {
            // Zero singular value: set U column to zero
            for (int64_t i = 0; i < m; i++) {
                U[i * k + j] = 0.0;
            }
        }
    }

    // Sort singular values in descending order (bubble sort - small k)
    for (int64_t i = 0; i < k - 1; i++) {
        for (int64_t j = 0; j < k - 1 - i; j++) {
            if (S[j] < S[j + 1]) {
                // Swap S[j] and S[j+1]
                double tmp = S[j]; S[j] = S[j + 1]; S[j + 1] = tmp;
                // Swap U columns j and j+1
                for (int64_t r = 0; r < m; r++) {
                    double t2 = U[r * k + j]; U[r * k + j] = U[r * k + (j + 1)]; U[r * k + (j + 1)] = t2;
                }
                // Swap V columns j and j+1
                for (int64_t r = 0; r < n; r++) {
                    double t2 = V[r * n + j]; V[r * n + j] = V[r * n + (j + 1)]; V[r * n + (j + 1)] = t2;
                }
            }
        }
    }

    free(B);
}

// ===== ND BROADCAST RUNTIME FUNCTION =====

// Broadcast a source tensor to a target shape.
// src_data: source elements (doubles), src_dims: source shape, src_ndim: source rank
// dst_data: output elements (doubles), dst_dims: target shape, dst_ndim: target rank
// Returns 0 on success, -1 on incompatible shapes.
// NumPy broadcasting rules: dimensions are compared from trailing end.
// dim is compatible if equal, or one of them is 1.
extern "C" int64_t eshkol_broadcast_copy(
    const double* src_data, const int64_t* src_dims, int64_t src_ndim,
    double* dst_data, const int64_t* dst_dims, int64_t dst_ndim)
{
    // Compute total destination elements
    int64_t dst_total = 1;
    for (int64_t d = 0; d < dst_ndim; d++) dst_total *= dst_dims[d];

    /* Checked-multiply stride computation (audit C5).
     * The previous loop multiplied `strides[d+1] * dims[d+1]` as plain
     * signed int64 with no overflow detection. A dim of -1 produced
     * negative strides that later `offset = idx * stride` walks turn
     * into OOB heap reads/writes. Reject negative / zero dims and
     * trap multiply overflow up-front. */
    /* Reject negative dims (zero dims are legal — empty tensors). */
    int64_t src_strides[16]; // max 16 dimensions
    if (src_ndim > 16) return -1;
    for (int64_t d = 0; d < src_ndim; d++) {
        if (src_dims[d] < 0) return -1;
    }
    if (src_ndim > 0) {
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; d--) {
            int64_t a = src_strides[d + 1];
            int64_t b = src_dims[d + 1];
            if (a > 0 && b > INT64_MAX / a) return -1;  // overflow
            src_strides[d] = a * b;
        }
    }

    int64_t dst_strides[16];
    if (dst_ndim > 16) return -1;
    for (int64_t d = 0; d < dst_ndim; d++) {
        if (dst_dims[d] < 0) return -1;
    }
    if (dst_ndim > 0) {
        dst_strides[dst_ndim - 1] = 1;
        for (int64_t d = dst_ndim - 2; d >= 0; d--) {
            int64_t a = dst_strides[d + 1];
            int64_t b = dst_dims[d + 1];
            if (a > 0 && b > INT64_MAX / a) return -1;
            dst_strides[d] = a * b;
        }
    }

    // Validate broadcast compatibility and build mapping
    int64_t offset = dst_ndim - src_ndim;
    for (int64_t d = 0; d < src_ndim; d++) {
        int64_t dd = d + offset;
        if (src_dims[d] != 1 && src_dims[d] != dst_dims[dd]) {
            return -1; // Incompatible shapes
        }
    }

    // For each destination element, compute source index
    for (int64_t flat = 0; flat < dst_total; flat++) {
        // Decompose flat index into dst multi-dimensional indices
        int64_t remaining = flat;
        int64_t src_flat = 0;

        for (int64_t d = 0; d < dst_ndim; d++) {
            int64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];

            // Map to source dimension
            int64_t src_d = d - offset;
            if (src_d >= 0 && src_d < src_ndim) {
                if (src_dims[src_d] != 1) {
                    src_flat += idx * src_strides[src_d];
                }
                // If src_dims[src_d] == 1, index is 0 (broadcast)
            }
        }

        dst_data[flat] = src_data[src_flat];
    }
    return 0;
}

// Walk a Scheme cons list and extract int64 dimension values into an array.
// Returns the number of dimensions extracted.
extern "C" int64_t eshkol_cons_list_to_dims(
    const void* cons_ptr, int64_t* dims_out, int64_t max_dims)
{
    int64_t count = 0;
    const arena_tagged_cons_cell_t* current =
        (const arena_tagged_cons_cell_t*)cons_ptr;

    while (current != NULL && count < max_dims) {
        // Extract car as int64 dimension value
        dims_out[count] = arena_tagged_cons_get_int64(current, false);
        count++;

        // Check cdr type — if null, we've reached end of list
        uint8_t cdr_type = arena_tagged_cons_get_type(current, true);
        uint8_t cdr_base = ESHKOL_GET_BASE_TYPE(cdr_type);
        if (cdr_base == ESHKOL_VALUE_NULL) break;

        // Get cdr pointer to next cons cell
        uint64_t cdr_ptr = arena_tagged_cons_get_ptr(current, true);
        if (cdr_ptr == 0) break;
        current = (const arena_tagged_cons_cell_t*)(uintptr_t)cdr_ptr;
    }

    return count;
}

// Compute the product of an int64 dimensions array.
extern "C" int64_t eshkol_compute_dims_total(
    const int64_t* dims, int64_t ndim)
{
    int64_t total = 1;
    for (int64_t i = 0; i < ndim; i++) {
        total *= dims[i];
    }
    return total;
}

// Extract tensor elements (double bitpatterns) as int64 dimension values.
// Returns the number of dimensions extracted.
extern "C" int64_t eshkol_tensor_to_dims(
    const void* tensor_ptr, int64_t* dims_out, int64_t max_dims)
{
    const eshkol_tensor_t* t = (const eshkol_tensor_t*)tensor_ptr;
    if (!t || !dims_out) return 0;
    int64_t count = (int64_t)t->total_elements;
    if (count > max_dims) count = max_dims;
    for (int64_t i = 0; i < count; i++) {
        double dval;
        memcpy(&dval, &t->elements[i], sizeof(double));
        dims_out[i] = (int64_t)dval;
    }
    return count;
}

// Check if two tensor shapes are identical.
// Returns 1 if shapes match, 0 otherwise.
extern "C" int64_t eshkol_shapes_equal(
    const int64_t* dims1, int64_t ndim1,
    const int64_t* dims2, int64_t ndim2)
{
    if (ndim1 != ndim2) return 0;
    for (int64_t i = 0; i < ndim1; i++) {
        if (dims1[i] != dims2[i]) return 0;
    }
    return 1;
}

// Compute broadcast output shape from two input shapes.
// Returns output ndim (<=16), or -1 if shapes are incompatible.
// out_dims must have space for at least 16 int64s.
static int64_t compute_broadcast_shape(
    const int64_t* a_dims, int64_t a_ndim,
    const int64_t* b_dims, int64_t b_ndim,
    int64_t* out_dims)
{
    int64_t out_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    if (out_ndim > 16) return -1;

    for (int64_t i = 0; i < out_ndim; i++) {
        int64_t ai = (i < a_ndim) ? a_dims[a_ndim - 1 - i] : 1;
        int64_t bi = (i < b_ndim) ? b_dims[b_ndim - 1 - i] : 1;

        if (ai == bi) {
            out_dims[out_ndim - 1 - i] = ai;
        } else if (ai == 1) {
            out_dims[out_ndim - 1 - i] = bi;
        } else if (bi == 1) {
            out_dims[out_ndim - 1 - i] = ai;
        } else {
            return -1; // Incompatible
        }
    }
    return out_ndim;
}

// Perform broadcast elementwise operation on two tensors.
// op: 0=add, 1=sub, 2=mul, 3=div
// Writes result into out_data (caller must allocate out_total doubles).
// out_dims/out_ndim/out_total are written by this function.
// Returns 0 on success, -1 on incompatible shapes.
extern "C" int64_t eshkol_broadcast_elementwise_f64(
    int64_t op,
    const double* a_data, const int64_t* a_dims, int64_t a_ndim,
    const double* b_data, const int64_t* b_dims, int64_t b_ndim,
    double* out_data, int64_t* out_dims, int64_t* out_ndim_out,
    int64_t* out_total_out)
{
    int64_t bcast_dims[16];
    int64_t out_ndim = compute_broadcast_shape(a_dims, a_ndim, b_dims, b_ndim, bcast_dims);
    if (out_ndim < 0) return -1;

    // Copy broadcast dims to output
    for (int64_t i = 0; i < out_ndim; i++) out_dims[i] = bcast_dims[i];
    *out_ndim_out = out_ndim;

    int64_t out_total = 1;
    for (int64_t d = 0; d < out_ndim; d++) out_total *= bcast_dims[d];
    *out_total_out = out_total;

    // Precompute strides for output, a, and b
    int64_t out_strides[16], a_strides[16], b_strides[16];
    if (out_ndim > 0) {
        out_strides[out_ndim - 1] = 1;
        for (int64_t d = out_ndim - 2; d >= 0; d--)
            out_strides[d] = out_strides[d + 1] * bcast_dims[d + 1];
    }
    if (a_ndim > 0) {
        a_strides[a_ndim - 1] = 1;
        for (int64_t d = a_ndim - 2; d >= 0; d--)
            a_strides[d] = a_strides[d + 1] * a_dims[d + 1];
    }
    if (b_ndim > 0) {
        b_strides[b_ndim - 1] = 1;
        for (int64_t d = b_ndim - 2; d >= 0; d--)
            b_strides[d] = b_strides[d + 1] * b_dims[d + 1];
    }

    int64_t a_offset = out_ndim - a_ndim;
    int64_t b_offset = out_ndim - b_ndim;

    for (int64_t flat = 0; flat < out_total; flat++) {
        int64_t remaining = flat;
        int64_t a_flat = 0, b_flat = 0;

        for (int64_t d = 0; d < out_ndim; d++) {
            int64_t idx = remaining / out_strides[d];
            remaining %= out_strides[d];

            int64_t ad = d - a_offset;
            if (ad >= 0 && ad < a_ndim && a_dims[ad] != 1)
                a_flat += idx * a_strides[ad];

            int64_t bd = d - b_offset;
            if (bd >= 0 && bd < b_ndim && b_dims[bd] != 1)
                b_flat += idx * b_strides[bd];
        }

        double a_val = a_data[a_flat];
        double b_val = b_data[b_flat];
        double result;

        switch (op) {
            case 0: result = a_val + b_val; break;
            case 1: result = a_val - b_val; break;
            case 2: result = a_val * b_val; break;
            case 3: result = (b_val != 0.0) ? a_val / b_val : 0.0; break;
            default: result = 0.0; break;
        }

        out_data[flat] = result;
    }
    return 0;
}

// Stride-aware tensor concatenation along an arbitrary axis.
// result_data: pre-allocated output buffer
// num_tensors: number of input tensors
// src_datas: array of pointers to each input tensor's elements
// src_axis_dims: each tensor's size along the concat axis
// stride_after: product of dims after the concat axis
// outer_count: product of dims before the concat axis
extern "C" void eshkol_concat_strided(
    double* result_data,
    int64_t num_tensors,
    const double** src_datas,
    const int64_t* src_axis_dims,
    int64_t stride_after,
    int64_t outer_count)
{
    double* dst = result_data;
    for (int64_t outer = 0; outer < outer_count; outer++) {
        for (int64_t t = 0; t < num_tensors; t++) {
            int64_t chunk = src_axis_dims[t] * stride_after;
            int64_t src_offset = outer * chunk;
            memcpy(dst, src_datas[t] + src_offset, (size_t)(chunk * (int64_t)sizeof(double)));
            dst += chunk;
        }
    }
}

// ===== END LINEAR ALGEBRA RUNTIME FUNCTIONS =====

// ===== BATCH MATRIX MULTIPLICATION RUNTIME =====

/**
 * eshkol_batch_matmul_f64 — batched GEMM with plain C loops.
 *
 * A: [batch, M, K]   B: [batch, K, N]   C: [batch, M, N]
 *
 * Each slice is: C[b,:,:] = A[b,:,:] @ B[b,:,:]
 * Inner loop uses a simple triple-loop (i,j,k) that the compiler
 * can auto-vectorise.
 */
extern "C" void eshkol_batch_matmul_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double*       __restrict__ c,
    int64_t batch, int64_t M, int64_t K, int64_t N)
{
    for (int64_t bs = 0; bs < batch; bs++) {
        const double* A = a + bs * M * K;
        const double* B = b + bs * K * N;
        double*       C = c + bs * M * N;

        // Initialise output slice to zero
        for (int64_t idx = 0; idx < M * N; idx++) C[idx] = 0.0;

        // Standard triple loop, cache-friendly for row-major B iteration
        for (int64_t i = 0; i < M; i++) {
            for (int64_t kk = 0; kk < K; kk++) {
                double a_ik = A[i * K + kk];
                for (int64_t j = 0; j < N; j++) {
                    C[i * N + j] += a_ik * B[kk * N + j];
                }
            }
        }
    }
}

// ===== END BATCH MATRIX MULTIPLICATION RUNTIME =====

// ===== STACK OVERFLOW PROTECTION =====

// Per-thread recursion depth counter
// thread_local is correct: recursion depth tracks the call stack, which is per-thread
static thread_local int64_t __eshkol_recursion_depth = 0;
static const int64_t ESHKOL_MAX_RECURSION_DEPTH = 100000;  // 100K frames (512MB stack)

extern "C" int64_t eshkol_check_recursion_depth(void) {
    __eshkol_recursion_depth++;
    if (__eshkol_recursion_depth > ESHKOL_MAX_RECURSION_DEPTH) {
        __eshkol_recursion_depth = 0;  // Reset so handler can run
        eshkol_exception_t* exc = eshkol_make_exception(
            ESHKOL_EXCEPTION_ERROR,
            "maximum recursion depth exceeded");
        if (exc) {
            eshkol_raise(exc);
        }
        // If raise returns (no handler), exit gracefully instead of aborting
        fprintf(stderr, "Error: maximum recursion depth (%lld) exceeded\n",
                (long long)ESHKOL_MAX_RECURSION_DEPTH);
        exit(1);
    }
    return __eshkol_recursion_depth;
}

extern "C" void eshkol_decrement_recursion_depth(void) {
    if (__eshkol_recursion_depth > 0) {
        __eshkol_recursion_depth--;
    }
}

/* Safety guards emitted inline by the codegen for car / cdr / list-ref when
 * the input argument's static type cannot be proven to be a pair. Previously
 * these sites dereferenced whatever integer lived in the tagged value's data
 * field, turning e.g. `(car #t)` (pair_int_safe == 1) into a load of address
 * 0x1 and a SIGSEGV. Now the codegen calls into these helpers on the error
 * branch and the runtime raises a typed exception that `guard` can catch. */
extern "C" void eshkol_raise_not_pair(const char* op_name) {
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_TYPE_ERROR,
        op_name ? op_name : "car/cdr: argument is not a pair");
    if (exc) eshkol_raise(exc);
    /* If no handler is installed, eshkol_raise calls exit(1) itself. */
    fprintf(stderr, "Eshkol: %s\n", op_name ? op_name : "not a pair");
    exit(1);
}

/* Append `lhs` (expected to be a proper list) to `rhs`, returning the
 * concatenated list. Used by the quasiquote codegen to implement
 * `,@splice-list` — the parser emits a list-constructor call where a splicing
 * element's runtime value must be appended to the rest of the list, not just
 * consed. Allocates fresh cons cells in the global arena so the original
 * lhs is not mutated. If lhs is not a proper cons list, it is treated as a
 * single-element list (falls back to cons'ing onto rhs). */
extern "C" void eshkol_append_tagged_sret(eshkol_tagged_value_t* out,
                                          const eshkol_tagged_value_t* lhs,
                                          const eshkol_tagged_value_t* rhs) {
    if (!out) return;
    if (!rhs) {
        *out = (eshkol_tagged_value_t){};
        out->type = ESHKOL_VALUE_NULL;
        return;
    }
    if (!lhs || lhs->type == ESHKOL_VALUE_NULL) {
        *out = *rhs;
        return;
    }

    /* Is lhs a cons cell? Accept both legacy CONS_PTR and consolidated
     * HEAP_PTR with HEAP_SUBTYPE_CONS. */
    bool is_cons = false;
    if (lhs->type == ESHKOL_VALUE_CONS_PTR) is_cons = true;
    else if (lhs->type == ESHKOL_VALUE_HEAP_PTR && lhs->data.ptr_val) {
        eshkol_object_header_t* h =
            ESHKOL_GET_HEADER((void*)(uintptr_t)lhs->data.ptr_val);
        if (h && h->subtype == HEAP_SUBTYPE_CONS) is_cons = true;
    }

    if (!is_cons) {
        /* Non-list splice value — treat as a single element cons onto rhs. */
        arena_t* arena = get_global_arena();
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!cell) { *out = *rhs; return; }
        cell->car = *lhs;
        cell->cdr = *rhs;
        out->type = ESHKOL_VALUE_HEAP_PTR;
        out->flags = 0; out->reserved = 0;
        out->data.ptr_val = (uint64_t)(uintptr_t)cell;
        return;
    }

    /* Collect lhs elements front-to-back so we can rebuild onto rhs. */
    std::vector<eshkol_tagged_value_t> elems;
    eshkol_tagged_value_t cur = *lhs;
    while (cur.type != ESHKOL_VALUE_NULL && cur.data.ptr_val != 0) {
        bool cur_is_cons = false;
        if (cur.type == ESHKOL_VALUE_CONS_PTR) cur_is_cons = true;
        else if (cur.type == ESHKOL_VALUE_HEAP_PTR) {
            eshkol_object_header_t* h =
                ESHKOL_GET_HEADER((void*)(uintptr_t)cur.data.ptr_val);
            if (h && h->subtype == HEAP_SUBTYPE_CONS) cur_is_cons = true;
        }
        if (!cur_is_cons) {
            /* Improper list — store the tail atom as the last element. */
            elems.push_back(cur);
            break;
        }
        arena_tagged_cons_cell_t* cell =
            (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
        elems.push_back(cell->car);
        cur = cell->cdr;
    }

    /* Build acc = rhs, then prepend elems in reverse. */
    eshkol_tagged_value_t acc = *rhs;
    arena_t* arena = get_global_arena();
    for (auto it = elems.rbegin(); it != elems.rend(); ++it) {
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!cell) { *out = acc; return; }
        cell->car = *it;
        cell->cdr = acc;
        acc.type = ESHKOL_VALUE_HEAP_PTR;
        acc.flags = 0; acc.reserved = 0;
        acc.data.ptr_val = (uint64_t)(uintptr_t)cell;
    }
    *out = acc;
}

extern "C" void eshkol_raise_index_oob(const char* op_name, int64_t idx,
                                        int64_t length) {
    char buf[160];
    snprintf(buf, sizeof(buf),
             "%s: index %lld out of bounds (length=%lld)",
             op_name ? op_name : "list-ref/vector-ref",
             (long long)idx, (long long)length);
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_ERROR, buf);
    if (exc) eshkol_raise(exc);
    fprintf(stderr, "Eshkol: %s\n", buf);
    exit(1);
}

/* Raise an "improper list" error from codegen-generated walkers
 * (audit M7). Used by list->vector and similar tail-traversals
 * that encounter a non-pair, non-() tail. */
extern "C" void eshkol_raise_improper_list(const char* msg) {
    eshkol_exception_t* exc = eshkol_make_exception(
        ESHKOL_EXCEPTION_ERROR, msg ? msg : "improper list");
    if (exc) eshkol_raise(exc);
    fprintf(stderr, "Eshkol: %s\n", msg ? msg : "improper list");
    exit(1);
}

extern "C" void eshkol_reset_recursion_depth(void) {
    __eshkol_recursion_depth = 0;
}

// ===== END STACK OVERFLOW PROTECTION =====

#endif // __cplusplus
