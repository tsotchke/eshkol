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
#include <pthread.h>
#include <sys/resource.h>

#ifdef __cplusplus
#include <new>      // for std::bad_alloc
#include <stdexcept>
#include <atomic>
#endif

// Default alignment for memory allocations
#define DEFAULT_ALIGNMENT 8

// Global tape pointer for AD operations (shared across JIT modules in REPL)
// NOTE: Not thread_local because REPL JIT resolves these via dlsym/ADD_DATA_SYMBOL.
// For parallel-map with AD, the codegen should use per-task tape allocation.
ad_tape_t* __current_ad_tape = nullptr;

// Global AD mode flag (shared across JIT modules in REPL)
// CRITICAL: This must be shared so lambdas from one module can see AD mode set by another
bool __ad_mode_active = false;

// ============================================================================
// Stack Size Initialization
// ============================================================================
// Increases process stack size for deep recursion support.
// On macOS, the main thread stack is set at link time (-Wl,-stack_size),
// but setrlimit still helps for spawned threads and as a Linux fallback.
// Env var ESHKOL_STACK_SIZE overrides the default (in bytes).
extern "C" void eshkol_init_stack_size() {
    const rlim_t default_stack = 512ULL * 1024 * 1024;  // 512MB
    rlim_t target = default_stack;

    const char* env_val = getenv("ESHKOL_STACK_SIZE");
    if (env_val) {
        char* end = nullptr;
        unsigned long long parsed = strtoull(env_val, &end, 0);
        if (end != env_val && parsed >= 1024 * 1024) {  // min 1MB
            target = (rlim_t)parsed;
        }
    }

    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
        if (rl.rlim_cur < target) {
            rl.rlim_cur = target;
            if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < target) {
                rl.rlim_cur = rl.rlim_max;  // cannot exceed hard limit
            }
            setrlimit(RLIMIT_STACK, &rl);
        }
    }
}

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
std::atomic<arena_t*> __repl_shared_arena{nullptr};

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

    // Allocate and initialize mutex
    pthread_mutex_t* mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (!mutex) {
        eshkol_error("Failed to allocate mutex for thread-safe arena");
        arena_destroy(arena);
        return nullptr;
    }

    if (pthread_mutex_init(mutex, nullptr) != 0) {
        eshkol_error("Failed to initialize mutex for thread-safe arena");
        free(mutex);
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
        pthread_mutex_lock((pthread_mutex_t*)arena->mutex);
    }
}

void arena_unlock(arena_t* arena) {
    if (arena && arena->thread_safe && arena->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)arena->mutex);
    }
}

void arena_destroy(arena_t* arena) {
    if (!arena) return;

    // Destroy mutex if thread-safe
    if (arena->thread_safe && arena->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)arena->mutex);
        free(arena->mutex);
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

// Allocate a string with object header (new consolidated format)
// Returns pointer to string data (header is at offset -8)
// Includes space for null terminator
char* arena_allocate_string_with_header(arena_t* arena, size_t length) {
    if (!arena) {
        eshkol_error("Cannot allocate string with header: null arena");
        return nullptr;
    }

    // Overflow check: length + 1 can wrap at SIZE_MAX
    if (length >= SIZE_MAX) {
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
        abort();  // Fatal: continuing with corrupted scope stack causes silent memory corruption
    }
    
    arena_scope_t* scope = arena->current_scope;
    
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

// Thread-safe global arena initialization
static pthread_once_t global_arena_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t global_arena_mutex = PTHREAD_MUTEX_INITIALIZER;

static void init_global_arena_internal() {
    // Create a thread-safe arena for global allocations
    __global_arena = arena_create_threadsafe(65536);  // 64KB default, thread-safe
    if (!__global_arena) {
        eshkol_error("Failed to create global arena");
    }
}

// Get or create the global arena (thread-safe)
arena_t* get_global_arena() {
    // Use pthread_once for thread-safe initialization
    pthread_once(&global_arena_once, init_global_arena_internal);
    return __global_arena;
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

eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val) {
    // For immediate types (int, double, bool, char, etc.) — no escape needed
    uint8_t type = val.type;

    // Check if this is a heap pointer type (might point into region memory)
    bool is_heap = (type == ESHKOL_VALUE_HEAP_PTR) ||
                   (type == ESHKOL_VALUE_CALLABLE) ||
                   ((type & ESHKOL_PORT_ANY_FLAG) != 0);  // ports

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

// ===== LAMBDA REGISTRY IMPLEMENTATION =====
// Runtime table for mapping function pointers to S-expressions (homoiconicity)

eshkol_lambda_registry_t* g_lambda_registry = nullptr;

void eshkol_lambda_registry_init(void) {
    if (g_lambda_registry) {
        return;  // Already initialized
    }

    g_lambda_registry = (eshkol_lambda_registry_t*)malloc(sizeof(eshkol_lambda_registry_t));
    if (!g_lambda_registry) {
        eshkol_error("Failed to allocate lambda registry");
        return;
    }

    g_lambda_registry->capacity = 64;  // Initial capacity
    g_lambda_registry->count = 0;
    g_lambda_registry->entries = (eshkol_lambda_entry_t*)malloc(
        sizeof(eshkol_lambda_entry_t) * g_lambda_registry->capacity);

    if (!g_lambda_registry->entries) {
        eshkol_error("Failed to allocate lambda registry entries");
        free(g_lambda_registry);
        g_lambda_registry = nullptr;
        return;
    }

    eshkol_debug("Lambda registry initialized with capacity %zu", g_lambda_registry->capacity);
}

void eshkol_lambda_registry_destroy(void) {
    if (!g_lambda_registry) {
        return;
    }

    if (g_lambda_registry->entries) {
        free(g_lambda_registry->entries);
    }
    free(g_lambda_registry);
    g_lambda_registry = nullptr;
}

void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name) {
    if (!g_lambda_registry) {
        eshkol_lambda_registry_init();
    }

    if (!g_lambda_registry) {
        return;  // Init failed
    }

    // Check if already registered (update if so)
    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            g_lambda_registry->entries[i].sexpr_ptr = sexpr_ptr;
            g_lambda_registry->entries[i].name = name;
            eshkol_debug("Updated lambda registry entry for %s at %p -> sexpr %p",
                        name ? name : "(anonymous)", (void*)func_ptr, (void*)sexpr_ptr);
            return;
        }
    }

    // Grow if needed
    if (g_lambda_registry->count >= g_lambda_registry->capacity) {
        size_t new_capacity = g_lambda_registry->capacity * 2;
        eshkol_lambda_entry_t* new_entries = (eshkol_lambda_entry_t*)realloc(
            g_lambda_registry->entries,
            sizeof(eshkol_lambda_entry_t) * new_capacity);

        if (!new_entries) {
            eshkol_error("Failed to grow lambda registry");
            return;
        }

        g_lambda_registry->entries = new_entries;
        g_lambda_registry->capacity = new_capacity;
    }

    // Add new entry
    size_t idx = g_lambda_registry->count++;
    g_lambda_registry->entries[idx].func_ptr = func_ptr;
    g_lambda_registry->entries[idx].sexpr_ptr = sexpr_ptr;
    g_lambda_registry->entries[idx].name = name;

    eshkol_debug("Lambda registry: added %s func=%p sexpr=%p",
                name ? name : "(anon)", (void*)func_ptr, (void*)sexpr_ptr);
}

uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr) {
    if (!g_lambda_registry) {
        return 0;
    }

    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            return g_lambda_registry->entries[i].sexpr_ptr;
        }
    }

    return 0;  // Not found
}

// ===== END LAMBDA REGISTRY IMPLEMENTATION =====

// ===== UNIFIED DISPLAY IMPLEMENTATION =====
// Single source of truth for displaying all Eshkol values

// Forward declarations for internal helpers
static void display_atom(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts);
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts);
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts);
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts);

// Get output stream (defaults to stdout)
static FILE* get_output(eshkol_display_opts_t* opts) {
    if (opts && opts->output) {
        return (FILE*)opts->output;
    }
    return stdout;
}

// Display a single tagged value
void eshkol_display_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    eshkol_display_value_opts(value, &opts);
}

void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts) {
    if (!value) {
        fprintf(get_output(opts), "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(get_output(opts), "...");
        return;
    }

    uint8_t full_type = value->type;
    // Compute base type correctly:
    // - Legacy types (>= 32): use full_type directly, no masking
    // - Consolidated (8-15) and multimedia (16-31): use full_type directly
    // - Immediate types (< 8): might have exactness flags, mask with 0x0F
    uint8_t base_type;
    if (full_type >= 32) {
        base_type = full_type;  // Legacy types: CONS_PTR=32, STRING_PTR=33, etc.
    } else if (full_type >= 8) {
        base_type = full_type;  // Consolidated/multimedia: HEAP_PTR=8, HANDLE=16, etc.
    } else {
        base_type = full_type & 0x0F;  // Immediate types: strip exactness flags
    }

    // Check for port types BEFORE the switch (they use special flag encoding)
    // Input port: CONS_PTR | 0x10 = 48
    // Output port: CONS_PTR | 0x40 = 96 (NOT 0x20! CONS_PTR=32=0x20)
    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x10)) {
        FILE* fp = (FILE*)value->data.ptr_val;
        int fd = fp ? fileno(fp) : -1;
        fprintf(get_output(opts), "#<input-port fd:%d>", fd);
        return;
    }
    if (full_type == (ESHKOL_VALUE_CONS_PTR | 0x40)) {
        FILE* fp = (FILE*)value->data.ptr_val;
        int fd = fp ? fileno(fp) : -1;
        fprintf(get_output(opts), "#<output-port fd:%d>", fd);
        return;
    }

    switch (base_type) {
        case ESHKOL_VALUE_NULL:
            fprintf(get_output(opts), "()");
            break;

        case ESHKOL_VALUE_HEAP_PTR: {
            // Consolidated heap pointer - read subtype from object header
            void* data_ptr = (void*)value->data.ptr_val;
            if (!data_ptr) {
                fprintf(get_output(opts), "()");
                break;
            }
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
            switch (header->subtype) {
                case HEAP_SUBTYPE_CONS:
                    eshkol_display_list(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_STRING:
                    if (opts->quote_strings) {
                        fprintf(get_output(opts), "\"%s\"", (const char*)data_ptr);
                    } else {
                        fprintf(get_output(opts), "%s", (const char*)data_ptr);
                    }
                    break;
                case HEAP_SUBTYPE_SYMBOL:
                    // Display symbol name without quotes (homoiconic representation)
                    fprintf(get_output(opts), "%s", (const char*)data_ptr);
                    break;
                case HEAP_SUBTYPE_VECTOR:
                    display_vector(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_TENSOR:
                    display_tensor(value->data.ptr_val, opts);
                    break;
                case HEAP_SUBTYPE_HASH:
                    fprintf(get_output(opts), "#<hash>");
                    break;
                case HEAP_SUBTYPE_EXCEPTION:
                    fprintf(get_output(opts), "#<exception>");
                    break;
                case HEAP_SUBTYPE_MULTI_VALUE:
                    fprintf(get_output(opts), "#<values>");
                    break;
                case HEAP_SUBTYPE_BIGNUM:
                    eshkol_bignum_display((const eshkol_bignum_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_SUBSTITUTION:
                    eshkol_display_substitution((const eshkol_substitution_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACT:
                    eshkol_display_fact((const eshkol_fact_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_KNOWLEDGE_BASE:
                    eshkol_display_kb((const eshkol_knowledge_base_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_FACTOR_GRAPH:
                    eshkol_display_factor_graph((const eshkol_factor_graph_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_WORKSPACE:
                    eshkol_display_workspace((const eshkol_workspace_t*)data_ptr, get_output(opts));
                    break;
                case HEAP_SUBTYPE_PROMISE:
                    fprintf(get_output(opts), "#<promise>");
                    break;
                case HEAP_SUBTYPE_RATIONAL: {
                    eshkol_rational_t* rat = (eshkol_rational_t*)data_ptr;
                    fprintf(get_output(opts), "%lld/%lld",
                        (long long)rat->numerator, (long long)rat->denominator);
                    break;
                }
                default:
                    fprintf(get_output(opts), "#<heap:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_CALLABLE: {
            // Consolidated callable - read subtype from object header
            void* data_ptr = (void*)value->data.ptr_val;
            if (!data_ptr) {
                fprintf(get_output(opts), "#<procedure>");
                break;
            }
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
            switch (header->subtype) {
                case CALLABLE_SUBTYPE_CLOSURE:
                    eshkol_display_closure(value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
                    eshkol_display_lambda(value->data.ptr_val, opts);
                    break;
                case CALLABLE_SUBTYPE_AD_NODE:
                    fprintf(get_output(opts), "#<ad-node>");
                    break;
                case CALLABLE_SUBTYPE_PRIMITIVE:
                    fprintf(get_output(opts), "#<primitive>");
                    break;
                case CALLABLE_SUBTYPE_CONTINUATION:
                    fprintf(get_output(opts), "#<continuation>");
                    break;
                default:
                    fprintf(get_output(opts), "#<callable:%d>", header->subtype);
                    break;
            }
            break;
        }

        case ESHKOL_VALUE_LOGIC_VAR:
            eshkol_display_logic_var(value->data.int_val, get_output(opts));
            break;

        case ESHKOL_VALUE_INT64:
            fprintf(get_output(opts), "%lld", (long long)value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE:
            fprintf(get_output(opts), "%g", value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            fprintf(get_output(opts), "%s", value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_CHAR:
            display_char((uint32_t)value->data.int_val, opts);
            break;

        case ESHKOL_VALUE_STRING_PTR:
            if (opts->quote_strings) {
                fprintf(get_output(opts), "\"%s\"", (const char*)value->data.ptr_val);
            } else {
                fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            break;

        case ESHKOL_VALUE_CONS_PTR:
            eshkol_display_list(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_LAMBDA_SEXPR:
            eshkol_display_lambda(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_CLOSURE_PTR:
            eshkol_display_closure(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_TENSOR_PTR:
            display_tensor(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_VECTOR_PTR:
            display_vector(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_DUAL_NUMBER: {
            eshkol_dual_number_t* dual = (eshkol_dual_number_t*)value->data.ptr_val;
            if (dual) {
                fprintf(get_output(opts), "(dual %g %g)", dual->value, dual->derivative);
            } else {
                fprintf(get_output(opts), "(dual 0 0)");
            }
            break;
        }

        case ESHKOL_VALUE_COMPLEX: {
            // Complex number: data.ptr_val points to {double real, double imag}
            double* parts = (double*)value->data.ptr_val;
            if (parts) {
                double re = parts[0];
                double im = parts[1];
                if (im == 0.0) {
                    fprintf(get_output(opts), "%g", re);
                } else if (re == 0.0) {
                    fprintf(get_output(opts), "%gi", im);
                } else if (im < 0.0) {
                    fprintf(get_output(opts), "%g%gi", re, im);
                } else {
                    fprintf(get_output(opts), "%g+%gi", re, im);
                }
            } else {
                fprintf(get_output(opts), "0");
            }
            break;
        }

        case ESHKOL_VALUE_AD_NODE_PTR:
            fprintf(get_output(opts), "#<ad-node>");
            break;

        case ESHKOL_VALUE_HASH_PTR: {
            eshkol_hash_table_t* ht = (eshkol_hash_table_t*)value->data.ptr_val;
            if (!ht) {
                fprintf(get_output(opts), "#<hash:0>");
            } else {
                fprintf(get_output(opts), "#<hash:%zu>", ht->size);
            }
            break;
        }

        default:
            if (opts->show_types) {
                fprintf(get_output(opts), "#<unknown-type-%d:0x%llx>",
                       base_type, (unsigned long long)value->data.ptr_val);
            } else {
                fprintf(get_output(opts), "#<unknown>");
            }
            break;
    }
}

// Scheme 'write' semantics - quotes strings
void eshkol_write_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    eshkol_display_value_opts(value, &opts);
}

void eshkol_write_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

void eshkol_display_value_to_port(const eshkol_tagged_value_t* value, void* port) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.output = port;
    eshkol_display_value_opts(value, &opts);
}

// Display a list (cons cell chain)
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (cons_ptr == 0) {
        fprintf(out, "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(out, "(...)");
        return;
    }

    fprintf(out, "(");
    opts->current_depth++;

    uint64_t current = cons_ptr;
    bool first = true;

    while (current != 0) {
        arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)current;

        if (!first) {
            fprintf(out, " ");
        }
        first = false;

        // Display car
        eshkol_display_value_opts(&cell->car, opts);

        // Check cdr type - handle both legacy and consolidated formats
        uint8_t cdr_full = cell->cdr.type;
        uint8_t cdr_type = (cdr_full >= 32) ? cdr_full : (cdr_full >= 8) ? cdr_full : (cdr_full & 0x0F);

        if (cdr_type == ESHKOL_VALUE_NULL) {
            // Proper list end
            break;
        } else if (cdr_type == ESHKOL_VALUE_CONS_PTR) {
            // Legacy cons - continue to next cell
            current = cell->cdr.data.ptr_val;
        } else if (cdr_type == ESHKOL_VALUE_HEAP_PTR) {
            // Consolidated heap pointer - check if it's a cons cell
            void* cdr_ptr = (void*)cell->cdr.data.ptr_val;
            if (cdr_ptr) {
                eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(cdr_ptr);
                if (hdr->subtype == HEAP_SUBTYPE_CONS) {
                    // Continue to next cons cell
                    current = cell->cdr.data.ptr_val;
                } else {
                    // Not a cons - dotted pair
                    fprintf(out, " . ");
                    eshkol_display_value_opts(&cell->cdr, opts);
                    break;
                }
            } else {
                break;
            }
        } else {
            // Dotted pair - display cdr and break
            fprintf(out, " . ");
            eshkol_display_value_opts(&cell->cdr, opts);
            break;
        }
    }

    opts->current_depth--;
    fprintf(out, ")");
}

// Display a lambda by extracting embedded S-expression or looking up in registry
void eshkol_display_lambda(uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        fprintf(out, "#<procedure>");
        return;
    }

    // For LAMBDA_SEXPR subtype, the closure_ptr IS a closure struct with sexpr_ptr
    // (same structure as CLOSURE subtype, just no environment)
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        // Display the embedded S-expression
        eshkol_display_list(sexpr, opts);
    } else {
        // No embedded S-expression - try the registry as fallback
        uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (registry_sexpr != 0) {
            eshkol_display_list(registry_sexpr, opts);
        } else {
            fprintf(out, "#<procedure>");
        }
    }
}

// Display a closure by extracting its embedded S-expression
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        fprintf(out, "#<closure>");
        return;
    }

    // Closure struct: { func_ptr (8), env (8), sexpr_ptr (8) }
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        // Display the embedded S-expression
        eshkol_display_list(sexpr, opts);
    } else {
        // No S-expression - try the registry as fallback
        uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (registry_sexpr != 0) {
            eshkol_display_list(registry_sexpr, opts);
        } else {
            fprintf(out, "#<closure>");
        }
    }
}

// Display a character
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (!opts->quote_strings) {
        // display mode: output raw character
        if (codepoint < 128) {
            fputc((char)codepoint, out);
        } else {
            // UTF-8 encode
            if (codepoint < 0x80) {
                fputc(codepoint, out);
            } else if (codepoint < 0x800) {
                fputc(0xC0 | (codepoint >> 6), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            } else if (codepoint < 0x10000) {
                fputc(0xE0 | (codepoint >> 12), out);
                fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            } else {
                fputc(0xF0 | (codepoint >> 18), out);
                fputc(0x80 | ((codepoint >> 12) & 0x3F), out);
                fputc(0x80 | ((codepoint >> 6) & 0x3F), out);
                fputc(0x80 | (codepoint & 0x3F), out);
            }
        }
        return;
    }

    // write mode: output #\ notation
    switch (codepoint) {
        case ' ':  fprintf(out, "#\\space"); break;
        case '\n': fprintf(out, "#\\newline"); break;
        case '\t': fprintf(out, "#\\tab"); break;
        case '\r': fprintf(out, "#\\return"); break;
        case 0:    fprintf(out, "#\\null"); break;
        default:
            if (codepoint < 128 && codepoint >= 32) {
                fprintf(out, "#\\%c", (char)codepoint);
            } else {
                fprintf(out, "#\\x%X", codepoint);
            }
            break;
    }
}

// Tensor struct layout is defined in arena_memory.h (eshkol_tensor_t)
// Must match LLVM TypeSystem tensor_type:
// struct Tensor {
//     uint64_t* dimensions;      // idx 0: array of dimension sizes
//     uint64_t  num_dimensions;  // idx 1: number of dimensions
//     int64_t*  elements;        // idx 2: element data (doubles stored as int64 bits)
//     uint64_t  total_elements;  // idx 3: total number of elements
// };

// Recursive helper for displaying N-dimensional tensors
static void display_tensor_recursive(FILE* out, const eshkol_tensor_t* tensor,
                                      uint64_t current_dim, uint64_t offset) {
    if (tensor->num_dimensions == 0) {
        fprintf(out, "#()");
        return;
    }

    uint64_t dim_size = tensor->dimensions[current_dim];

    // Base case: innermost dimension - print actual elements
    if (current_dim == tensor->num_dimensions - 1) {
        fprintf(out, "(");
        for (uint64_t i = 0; i < dim_size; i++) {
            if (i > 0) fprintf(out, " ");
            // Elements stored as int64 bit pattern of double
            int64_t bits = tensor->elements[offset + i];
            double value;
            memcpy(&value, &bits, sizeof(double));
            fprintf(out, "%g", value);
        }
        fprintf(out, ")");
        return;
    }

    // Recursive case: compute stride and iterate over slices
    uint64_t stride = 1;
    for (uint64_t k = current_dim + 1; k < tensor->num_dimensions; k++) {
        stride *= tensor->dimensions[k];
    }

    fprintf(out, "(");
    for (uint64_t i = 0; i < dim_size; i++) {
        if (i > 0) fprintf(out, " ");
        display_tensor_recursive(out, tensor, current_dim + 1, offset + i * stride);
    }
    fprintf(out, ")");
}

// Display a tensor with proper N-dimensional structure
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (tensor_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    const eshkol_tensor_t* tensor = (const eshkol_tensor_t*)tensor_ptr;

    // Validate tensor structure
    if (tensor->num_dimensions == 0 || tensor->total_elements == 0) {
        fprintf(out, "#()");
        return;
    }

    if (tensor->dimensions == NULL || tensor->elements == NULL) {
        fprintf(out, "#<invalid-tensor>");
        return;
    }

    // Print tensor prefix then contents
    fprintf(out, "#");
    display_tensor_recursive(out, tensor, 0, 0);
}

// Scheme vector structure:
// [length: i64] at offset 0 (8 bytes)
// [elem0: tagged_value] at offset 8 (16 bytes each)
// [elem1: tagged_value] at offset 24
// etc.

// Display a Scheme vector with proper element formatting
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (vector_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    // Read length from start of vector
    uint64_t* len_ptr = (uint64_t*)vector_ptr;
    uint64_t length = *len_ptr;

    // Validate length (sanity check)
    if (length > 10000) {
        fprintf(out, "#<invalid-vector>");
        return;
    }

    fprintf(out, "#(");

    // Elements start after the 8-byte length field
    // Each element is a tagged_value (16 bytes)
    uint8_t* elem_base = (uint8_t*)vector_ptr + 8;

    for (uint64_t i = 0; i < length; i++) {
        if (i > 0) fprintf(out, " ");

        // Get pointer to i-th tagged_value element
        eshkol_tagged_value_t* elem = (eshkol_tagged_value_t*)(elem_base + i * sizeof(eshkol_tagged_value_t));

        // Recursively display the element
        eshkol_display_value_opts(elem, opts);
    }

    fprintf(out, ")");
}

// ===== END UNIFIED DISPLAY IMPLEMENTATION =====

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

// Resize the hash table when load factor exceeds threshold
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
    return find_slot(table, key, nullptr) >= 0;
}

// Remove a key from the hash table
bool hash_table_remove(eshkol_hash_table_t* table, const eshkol_tagged_value_t* key) {
    if (!table || !key) return false;

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

    memset(table->status, HASH_ENTRY_EMPTY, table->capacity);
    table->size = 0;
    table->tombstones = 0;
}

// Get the number of entries in the hash table
size_t hash_table_count(const eshkol_hash_table_t* table) {
    return table ? table->size : 0;
}

// Get all keys as a list
arena_tagged_cons_cell_t* hash_table_keys(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table || table->size == 0) return nullptr;

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

// Get all values as a list
arena_tagged_cons_cell_t* hash_table_values(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table || table->size == 0) return nullptr;

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

    typedef eshkol_tagged_value_t (*fn0_t)(void);
    typedef eshkol_tagged_value_t (*fn1_t)(void*);
    typedef eshkol_tagged_value_t (*fn2_t)(void*, void*);
    typedef eshkol_tagged_value_t (*fn3_t)(void*, void*, void*);
    typedef eshkol_tagged_value_t (*fn4_t)(void*, void*, void*, void*);

    switch (num_captures) {
        case 0: return ((fn0_t)(uintptr_t)closure->func_ptr)();
        case 1: return ((fn1_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0]);
        case 2: return ((fn2_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1]);
        case 3: return ((fn3_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2]);
        case 4: return ((fn4_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2], &closure->env->captures[3]);
        default: return ((fn0_t)(uintptr_t)closure->func_ptr)();
    }
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

// ===== STRING PORT RUNTIME FUNCTIONS =====
// String ports use fmemopen (input) / open_memstream (output) to create
// FILE*-backed ports from in-memory strings. All existing I/O operations
// work unchanged since they operate on FILE* pointers.

#include <cmath>
#include <cstring>
#include <cstdlib>

// Side table tracking open_memstream buffers for get-output-string
#define MAX_STRING_OUTPUT_PORTS 256
static struct {
    FILE* fp;
    char* buf;
    size_t size;
} string_output_ports[MAX_STRING_OUTPUT_PORTS];
static int num_string_output_ports = 0;

extern "C" void* eshkol_open_input_string(void* arena_void, const char* str, int64_t len) {
    if (len == 0) {
        // fmemopen with size 0 returns NULL on macOS.
        // tmpfile() is portable and returns EOF immediately on first read.
        return tmpfile();
    }
    arena_t* arena = (arena_t*)arena_void;
    // Copy string to arena so fmemopen has a stable buffer
    char* copy = (char*)arena_allocate(arena, len + 1);
    memcpy(copy, str, len);
    copy[len] = '\0';
    FILE* fp = fmemopen(copy, len, "r");
    return fp;
}

extern "C" void* eshkol_open_output_string(void) {
    if (num_string_output_ports >= MAX_STRING_OUTPUT_PORTS) return NULL;
    int idx = num_string_output_ports++;
    string_output_ports[idx].buf = NULL;
    string_output_ports[idx].size = 0;
    FILE* fp = open_memstream(&string_output_ports[idx].buf,
                              &string_output_ports[idx].size);
    string_output_ports[idx].fp = fp;
    return fp;
}

extern "C" void* eshkol_get_output_string(void* arena_void, void* fp_void) {
    arena_t* arena = (arena_t*)arena_void;
    FILE* fp = (FILE*)fp_void;
    fflush(fp);
    for (int i = 0; i < num_string_output_ports; i++) {
        if (string_output_ports[i].fp == fp) {
            size_t len = string_output_ports[i].size;
            char* result = (char*)arena_allocate_with_header(
                arena, len + 1, HEAP_SUBTYPE_STRING, 0);
            memcpy(result, string_output_ports[i].buf, len);
            result[len] = '\0';
            return result;
        }
    }
    // Not found - return empty string
    char* result = (char*)arena_allocate_with_header(arena, 1, HEAP_SUBTYPE_STRING, 0);
    result[0] = '\0';
    return result;
}

// ===== S-EXPRESSION READER (R7RS `read`) =====
// Lightweight runtime S-expression reader: tokenizer + recursive descent parser
// Produces tagged values directly (not compiler AST nodes)

static int read_skip_whitespace(FILE* fp) {
    int ch;
    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) return EOF;
        if (ch == ';') {
            // Line comment: skip to end of line
            while ((ch = fgetc(fp)) != EOF && ch != '\n') {}
            if (ch == EOF) return EOF;
            continue;
        }
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') return ch;
    }
}

// Forward declaration
static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char);

static eshkol_tagged_value_t make_eof_tagged(void) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = 0xFF; // EOF object type
    return val;
}

static eshkol_tagged_value_t make_null_tagged(void) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_NULL;
    return val;
}

static eshkol_tagged_value_t make_int_tagged(int64_t n) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_INT64;
    val.data.int_val = n;
    return val;
}

static eshkol_tagged_value_t make_double_tagged(double d) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_DOUBLE;
    union { double d; int64_t i; } conv;
    conv.d = d;
    val.data.int_val = conv.i;
    return val;
}

static eshkol_tagged_value_t make_bool_tagged(int b) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_BOOL;
    val.data.int_val = b ? 1 : 0;
    return val;
}

static eshkol_tagged_value_t make_char_tagged(int32_t ch) {
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_CHAR;
    val.data.int_val = ch;
    return val;
}

static eshkol_tagged_value_t make_string_tagged(arena_t* arena, const char* str, size_t len) {
    char* s = (char*)arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_STRING, 0);
    memcpy(s, str, len);
    s[len] = '\0';
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)s;
    return val;
}

static eshkol_tagged_value_t make_symbol_tagged(arena_t* arena, const char* name, size_t len) {
    char* s = (char*)arena_allocate_with_header(arena, len + 1, HEAP_SUBTYPE_SYMBOL, 0);
    memcpy(s, name, len);
    s[len] = '\0';
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)s;
    return val;
}

static eshkol_tagged_value_t make_cons_tagged(arena_t* arena,
    eshkol_tagged_value_t car, eshkol_tagged_value_t cdr) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    cell->car = car;
    cell->cdr = cdr;
    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)cell;
    // Need header for cons cells - use allocate_with_header approach
    // Actually cons cells don't have headers in the current system,
    // they use CONS_PTR type directly
    val.type = ESHKOL_VALUE_CONS_PTR;
    return val;
}

// Read a list: ( datum ... ) or ( datum ... . datum )
static eshkol_tagged_value_t read_list(arena_t* arena, FILE* fp) {
    int ch = read_skip_whitespace(fp);
    if (ch == EOF) return make_eof_tagged();
    if (ch == ')') return make_null_tagged(); // empty after elements

    // Read first element
    eshkol_tagged_value_t car = read_datum(arena, fp, ch);
    if (car.type == 0xFF) return car; // propagate EOF

    // Check for dot notation
    ch = read_skip_whitespace(fp);
    if (ch == '.') {
        // Improper list: (a . b)
        int next = fgetc(fp);
        if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
            int ch2 = read_skip_whitespace(fp);
            eshkol_tagged_value_t cdr = read_datum(arena, fp, ch2);
            // Consume closing paren
            read_skip_whitespace(fp); // should be ')'
            return make_cons_tagged(arena, car, cdr);
        }
        // Not a dot pair, it's a symbol starting with .
        ungetc(next, fp);
        ungetc('.', fp);
    } else {
        ungetc(ch, fp);
    }

    // Read rest of list
    eshkol_tagged_value_t cdr = read_list(arena, fp);
    return make_cons_tagged(arena, car, cdr);
}

// Read a vector: #( datum ... )
static eshkol_tagged_value_t read_vector(arena_t* arena, FILE* fp) {
    // Collect elements into a temporary list first, then convert to vector
    // Use a bounded stack to avoid recursion issues
    eshkol_tagged_value_t elems[1024];
    int count = 0;

    for (;;) {
        int ch = read_skip_whitespace(fp);
        if (ch == EOF) return make_eof_tagged();
        if (ch == ')') break;
        if (count >= 1024) break; // safety limit
        elems[count++] = read_datum(arena, fp, ch);
    }

    // Allocate vector: [header | length(i64) | elements(tagged_value * count)]
    size_t data_size = 8 + count * sizeof(eshkol_tagged_value_t);
    char* vec = (char*)arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_VECTOR, 0);
    *(int64_t*)vec = count; // length
    eshkol_tagged_value_t* vec_elems = (eshkol_tagged_value_t*)(vec + 8);
    for (int i = 0; i < count; i++) {
        vec_elems[i] = elems[i];
    }

    eshkol_tagged_value_t val;
    memset(&val, 0, sizeof(val));
    val.type = ESHKOL_VALUE_HEAP_PTR;
    val.data.int_val = (int64_t)(uintptr_t)vec;
    return val;
}

// Read an atom (number, symbol, string, #t, #f, #\char)
static eshkol_tagged_value_t read_atom(arena_t* arena, FILE* fp, int first_char) {
    // String literal
    if (first_char == '"') {
        char buf[4096];
        int len = 0;
        int ch;
        while ((ch = fgetc(fp)) != EOF && ch != '"') {
            if (ch == '\\') {
                ch = fgetc(fp);
                if (ch == EOF) break;
                switch (ch) {
                    case 'n': buf[len++] = '\n'; break;
                    case 't': buf[len++] = '\t'; break;
                    case 'r': buf[len++] = '\r'; break;
                    case '\\': buf[len++] = '\\'; break;
                    case '"': buf[len++] = '"'; break;
                    default: buf[len++] = ch; break;
                }
            } else {
                if (len < 4095) buf[len++] = ch;
            }
        }
        return make_string_tagged(arena, buf, len);
    }

    // Hash prefix: #t, #f, #\char, #(vector
    if (first_char == '#') {
        int ch = fgetc(fp);
        if (ch == 't') {
            int next = fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) ungetc(next, fp);
                return make_bool_tagged(1);
            }
            ungetc(next, fp);
            // Could be #true
            char rest[16];
            int rlen = 0;
            rest[rlen++] = 'r';
            while (rlen < 15) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
                rest[rlen++] = c;
            }
            rest[rlen] = '\0';
            if (strcmp(rest, "rue") == 0) return make_bool_tagged(1);
            return make_bool_tagged(1); // fallback
        }
        if (ch == 'f') {
            int next = fgetc(fp);
            if (next == EOF || next == ' ' || next == '\n' || next == '\r' ||
                next == '\t' || next == ')' || next == '(') {
                if (next != EOF) ungetc(next, fp);
                return make_bool_tagged(0);
            }
            ungetc(next, fp);
            // Could be #false — consume rest and return false
            while (1) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
            }
            return make_bool_tagged(0);
        }
        if (ch == '\\') {
            // Character literal
            int c1 = fgetc(fp);
            if (c1 == EOF) return make_eof_tagged();
            int c2 = fgetc(fp);
            if (c2 == EOF || c2 == ' ' || c2 == '\n' || c2 == '\r' ||
                c2 == '\t' || c2 == ')' || c2 == '(') {
                if (c2 != EOF) ungetc(c2, fp);
                return make_char_tagged(c1);
            }
            // Multi-char name: space, newline, tab, etc.
            char name[32];
            name[0] = c1;
            name[1] = c2;
            int nlen = 2;
            while (nlen < 31) {
                int c = fgetc(fp);
                if (c == EOF || c == ' ' || c == '\n' || c == ')' || c == '(') {
                    if (c != EOF) ungetc(c, fp);
                    break;
                }
                name[nlen++] = c;
            }
            name[nlen] = '\0';
            if (strcmp(name, "space") == 0) return make_char_tagged(' ');
            if (strcmp(name, "newline") == 0) return make_char_tagged('\n');
            if (strcmp(name, "tab") == 0) return make_char_tagged('\t');
            if (strcmp(name, "return") == 0) return make_char_tagged('\r');
            if (strcmp(name, "null") == 0) return make_char_tagged(0);
            if (name[0] == 'x') {
                // Hex character: #\x41
                int codepoint = (int)strtol(name + 1, NULL, 16);
                return make_char_tagged(codepoint);
            }
            return make_char_tagged(c1); // fallback: first char
        }
        if (ch == '(') {
            return read_vector(arena, fp);
        }
        // Unknown # form — treat as symbol
        char buf[256];
        buf[0] = '#';
        buf[1] = ch;
        int blen = 2;
        while (blen < 255) {
            int c = fgetc(fp);
            if (c == EOF || c == ' ' || c == '\n' || c == '\r' ||
                c == '\t' || c == ')' || c == '(' || c == '"') {
                if (c != EOF) ungetc(c, fp);
                break;
            }
            buf[blen++] = c;
        }
        return make_symbol_tagged(arena, buf, blen);
    }

    // Number or symbol
    char buf[256];
    buf[0] = first_char;
    int blen = 1;
    while (blen < 255) {
        int ch = fgetc(fp);
        if (ch == EOF || ch == ' ' || ch == '\n' || ch == '\r' ||
            ch == '\t' || ch == ')' || ch == '(' || ch == '"' || ch == ';') {
            if (ch != EOF) ungetc(ch, fp);
            break;
        }
        buf[blen++] = ch;
    }
    buf[blen] = '\0';

    // Try to parse as number
    char* endp;
    long long ival = strtoll(buf, &endp, 10);
    if (endp == buf + blen && blen > 0) {
        return make_int_tagged(ival);
    }
    // Try as double
    double dval = strtod(buf, &endp);
    if (endp == buf + blen && blen > 0) {
        return make_double_tagged(dval);
    }
    // Try as rational: num/denom
    char* slash = strchr(buf, '/');
    if (slash && slash != buf && slash != buf + blen - 1) {
        *slash = '\0';
        char *ep1, *ep2;
        long long num = strtoll(buf, &ep1, 10);
        long long den = strtoll(slash + 1, &ep2, 10);
        if (*ep1 == '\0' && *ep2 == '\0' && den != 0) {
            void* rat = eshkol_rational_create(arena, num, den);
            eshkol_tagged_value_t val;
            memset(&val, 0, sizeof(val));
            if (((eshkol_rational_t*)rat)->denominator == 1) {
                val.type = ESHKOL_VALUE_INT64;
                val.data.int_val = ((eshkol_rational_t*)rat)->numerator;
            } else {
                val.type = ESHKOL_VALUE_HEAP_PTR;
                val.data.int_val = (int64_t)(uintptr_t)rat;
            }
            return val;
        }
        *slash = '/'; // restore
    }

    // Symbol
    return make_symbol_tagged(arena, buf, blen);
}

// Read a single S-expression datum from a FILE*
static eshkol_tagged_value_t read_datum(arena_t* arena, FILE* fp, int first_char) {
    if (first_char == EOF) return make_eof_tagged();

    // Quote shorthand: 'x -> (quote x)
    if (first_char == '\'') {
        int ch = read_skip_whitespace(fp);
        eshkol_tagged_value_t quoted = read_datum(arena, fp, ch);
        eshkol_tagged_value_t quote_sym = make_symbol_tagged(arena, "quote", 5);
        eshkol_tagged_value_t inner = make_cons_tagged(arena, quoted, make_null_tagged());
        return make_cons_tagged(arena, quote_sym, inner);
    }

    // List
    if (first_char == '(') {
        return read_list(arena, fp);
    }

    // Everything else: atoms
    return read_atom(arena, fp, first_char);
}

// Main entry point: read one S-expression from a FILE*
extern "C" void eshkol_read_sexpr(void* arena_void, void* fp_void,
                                   eshkol_tagged_value_t* result) {
    arena_t* arena = (arena_t*)arena_void;
    FILE* fp = (FILE*)fp_void;
    if (!fp) fp = stdin;

    int ch = read_skip_whitespace(fp);
    *result = read_datum(arena, fp, ch);
}

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

    // Precompute source strides (row-major)
    int64_t src_strides[16]; // max 16 dimensions
    if (src_ndim > 16) return -1;
    if (src_ndim > 0) {
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; d--) {
            src_strides[d] = src_strides[d + 1] * src_dims[d + 1];
        }
    }

    // Precompute destination strides
    int64_t dst_strides[16];
    if (dst_ndim > 16) return -1;
    if (dst_ndim > 0) {
        dst_strides[dst_ndim - 1] = 1;
        for (int64_t d = dst_ndim - 2; d >= 0; d--) {
            dst_strides[d] = dst_strides[d + 1] * dst_dims[d + 1];
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

extern "C" void eshkol_reset_recursion_depth(void) {
    __eshkol_recursion_depth = 0;
}

// ===== END STACK OVERFLOW PROTECTION =====

#endif // __cplusplus
