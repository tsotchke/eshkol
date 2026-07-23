/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Raw arena block, scope, statistics, and legacy list allocation runtime.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <atomic>
#include <cstring>
#endif

void* eshkol_arena_mutex_create(void);
void eshkol_arena_mutex_destroy(void* mutex);
void eshkol_arena_mutex_lock(void* mutex);
void eshkol_arena_mutex_unlock(void* mutex);

extern "C" int eshkol_arena_poison_enabled(void);

// Default alignment for memory allocations
#define DEFAULT_ALIGNMENT 8

// Global shared arena for REPL mode (persistent across evaluations)
// Atomic to synchronize writes (REPL init) and reads (runtime exception handlers)
extern "C" std::atomic<arena_t*> __repl_shared_arena{nullptr};

// Global command-line arguments (for (command-line) procedure)
// In REPL mode, these remain zero/null since there's no real main()
// Use weak linkage where object format support lets standalone code override.
ESHKOL_RUNTIME_WEAK int32_t __eshkol_argc = 0;
ESHKOL_RUNTIME_WEAK char** __eshkol_argv = nullptr;

// Utility function to align size to boundary
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

static size_t align_block_offset(const arena_block_t* block, size_t used, size_t alignment) {
    uintptr_t raw_addr = (uintptr_t)(block->memory + used);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    return (size_t)(aligned_addr - (uintptr_t)block->memory);
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
    arena->bounded = false;

    eshkol_debug("Created arena with default block size %zu", default_block_size);
    return arena;
}

// ESH-0039 / v1.8: bounded, no-grow arena.
// Single fixed-capacity block; allocation never grows the arena. When the
// capacity is exhausted, arena_allocate* returns NULL (see the `bounded` guard
// in arena_allocate_aligned) rather than malloc'ing a new block. This is the
// embedded / hard-ceiling seam for v1.8; the hosted impl simply reuses
// arena_create for the one-shot backing block.
arena_t* arena_create_bounded(size_t capacity) {
    if (capacity < 1024) capacity = 1024;
    arena_t* arena = arena_create(capacity);
    if (!arena) return nullptr;
    arena->bounded = true;
    eshkol_debug("Created bounded arena with capacity %zu", capacity);
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
    arena->bounded = false;

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

    // Free all blocks. When arena poisoning is enabled (ESHKOL_ARENA_POISON),
    // fill each block's live bytes with the 0xCB sentinel BEFORE releasing it.
    // arena_destroy() is the teardown path for region arenas (region_pop ->
    // region_destroy -> arena_destroy), which — unlike scope pops — otherwise
    // hand memory back to the allocator untouched, so a stale interior pointer
    // into a popped region would silently read valid-looking data. Poisoning
    // here turns any such region use-after-free into an immediate crash at an
    // obvious 0xCB.. address, which is what the region-escape evacuator gates
    // rely on to distinguish a real fix from working by luck.
    const bool poison = eshkol_arena_poison_enabled() != 0;
    arena_block_t* block = arena->current_block;
    while (block) {
        arena_block_t* next = block->next;
        if (poison && block->memory && block->used > 0) {
            std::memset(block->memory, 0xCB, block->used);
        }
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
    if ((alignment & (alignment - 1)) != 0) {
        eshkol_error("Invalid arena alignment %zu: alignment must be a power of two", alignment);
        arena_unlock(arena);
        return nullptr;
    }
    if (size > SIZE_MAX - (alignment - 1)) {
        eshkol_error("Arena allocation size overflow: size=%zu alignment=%zu", size, alignment);
        arena_unlock(arena);
        return nullptr;
    }

    // Align the requested size and the absolute returned pointer address.
    size_t aligned_size = align_size(size, alignment);
    if (aligned_size > SIZE_MAX - (alignment - 1)) {
        eshkol_error("Arena allocation block size overflow: size=%zu alignment=%zu", size, alignment);
        arena_unlock(arena);
        return nullptr;
    }

    // Check if current block has enough space
    arena_block_t* block = arena->current_block;
    size_t current_used = align_block_offset(block, block->used, alignment);

    if (current_used + aligned_size > block->size) {
        // ESH-0039 / v1.8: bounded arenas never grow — a request that overflows
        // the fixed capacity fails instead of malloc'ing a new block.
        if (arena->bounded) {
            eshkol_warn("Bounded arena exhausted: request %zu bytes exceeds remaining capacity",
                        aligned_size);
            arena_unlock(arena);
            return nullptr;
        }
        // Need a new block
        size_t min_block_size = aligned_size + alignment - 1;
        size_t new_block_size = (min_block_size > arena->default_block_size) ?
                               min_block_size : arena->default_block_size;

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
        current_used = align_block_offset(block, 0, alignment);
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

// ─────────────────────────────────────────────────────────────────────────
// Concurrency invariant for scope-based reclamation
// ─────────────────────────────────────────────────────────────────────────
// A bump-allocator arena's scope stack (push mark / pop-rewind / commit) is
// intrinsically single-threaded: it is one per-arena LIFO of (block, used)
// marks, and a pop rewinds the arena's shared bump pointer and frees/poisons
// every block allocated since the matching push.
//
// While a work-stealing construct is active (parallel-map/-fold/-filter/
// -execute and async futures) every pool worker is pinned to the SAME
// thread-safe process arena (the #217 parallel-scope pin), so concurrent
// workers would push/pop/rewind that single scope stack simultaneously. That
// is doubly broken: (a) a data race on current_block / current_block->used /
// current_scope, and (b) a cross-thread LIFO violation — worker A's pop
// rewinds (and poisons/frees) memory worker B is still using, handing B a
// dangling / overlapping cons cell. Symptom: nondeterministic "car/cdr:
// argument is not a pair", SIGSEGV/SIGBUS, or a hang, but only once the input
// crosses the parallel threshold and the closure body uses scope-based
// reclamation (an internal named-let loop's per-iteration scope, or a builtin
// such as memv that brackets scratch in a push/pop pair). Per-op locking would
// fix (a) but not (b); the scope stack simply cannot be shared.
//
// Fix: on a pool worker operating on a thread-safe (shared) arena, scope
// operations degrade to COMMIT-ONLY — allocations are retained and the shared
// scope stack is left untouched. This is exactly ESH-0214b's documented safe
// fallback ("commit = correctness over reclamation"): per-iteration reclamation
// is deferred for the duration of parallel execution (the shared arena keeps
// the memory, released at its normal lifetime), which is correct and bounded by
// the parallel construct. Single-threaded arenas (main thread; the flat-RSS
// loop path) and non-thread-safe per-worker/region arenas are unaffected, so
// ESH-0214b reclamation is preserved everywhere it is actually safe.
static inline bool arena_scope_ops_are_commit_only(const arena_t* arena) {
    return arena && arena->thread_safe && arena_is_worker_thread();
}

// Scope management
void arena_push_scope(arena_t* arena) {
    if (!arena) return;
    // Concurrent pool worker on the shared arena: do not touch the shared scope
    // stack (see arena_scope_ops_are_commit_only). The matching pop/commit is
    // likewise a no-op, so push/pop stay balanced.
    if (arena_scope_ops_are_commit_only(arena)) return;

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
    // Concurrent pool worker on the shared arena: commit-only. Retain the
    // iteration's allocations and leave the shared scope stack/bump pointer
    // untouched — rewinding here would free/poison memory a sibling worker is
    // still using (see arena_scope_ops_are_commit_only). Balances the no-op
    // push above.
    if (arena_scope_ops_are_commit_only(arena)) return;

    if (!arena || !arena->current_scope) {
        eshkol_error("Attempted to pop arena scope with no matching push - "
                     "unbalanced scope operations risk memory corruption");
        return;  // Graceful: skip the pop rather than kill the process
    }

    arena_scope_t* scope = arena->current_scope;

    /* Bug-BB-class diagnostic: when the hosted poison hook is enabled, fill
     * the popped region with a recognisable sentinel byte (0xCB) before
     * releasing it. Any later dereference of a stale pointer into that region
     * will crash with an address that contains the byte 0xCB in obvious
     * positions, turning a silent SEGV at a random-looking mangled address
     * into a clear "this was an arena UAF" diagnosis.
     *
     * The hook is profile-provided: hosted builds currently back it with
     * ESHKOL_ARENA_POISON, while freestanding profiles can provide a target
     * policy without making runtime-core read process environment variables.
     */
    if (eshkol_arena_poison_enabled()) {
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

// ═════════════════════════════════════════════════════════════════════════
// ESH-0214b: automatic per-iteration loop scope reclamation primitives
// ═════════════════════════════════════════════════════════════════════════

/* Discard the innermost scope record WITHOUT rewinding the arena: every
 * allocation made since the matching arena_push_scope is kept ("committed"
 * to the enclosing scope). This is the safe fallback the per-iteration loop
 * scoping uses when a loop-carried value was allocated inside the iteration
 * and therefore must survive it: the memory is retained (same behavior as
 * before this feature existed), but the scope stack stays balanced so
 * enclosing push/pop pairs keep their LIFO discipline. */
void arena_commit_scope(arena_t* arena) {
    // Concurrent pool worker on the shared arena: the no-op push left nothing on
    // the shared scope stack, and commit already means "retain allocations", so
    // there is nothing to do (see arena_scope_ops_are_commit_only).
    if (arena_scope_ops_are_commit_only(arena)) return;

    if (!arena || !arena->current_scope) {
        eshkol_error("Attempted to commit arena scope with no matching push");
        return;
    }
    arena_scope_t* scope = arena->current_scope;
    arena->current_scope = scope->parent;
    free(scope);
    eshkol_debug("Committed arena scope (allocations kept)");
}

/* Does ptr point into memory allocated AFTER the innermost scope mark?
 * Blocks are head-inserted (arena->current_block is the newest), so the
 * post-mark span is: every block from current_block down to (but excluding)
 * scope->block, plus [scope->used, used) within scope->block itself. */
int arena_top_scope_contains(const arena_t* arena, const void* ptr) {
    if (!arena || !arena->current_scope || !ptr) return 0;
    const arena_scope_t* scope = arena->current_scope;
    const uint8_t* p = (const uint8_t*)ptr;
    for (const arena_block_t* b = arena->current_block; b; b = b->next) {
        if (b == scope->block) {
            return (p >= b->memory + scope->used) && (p < b->memory + b->used);
        }
        if (p >= b->memory && p < b->memory + b->used) return 1;
    }
    return 0;
}

/* End a per-iteration loop scope (ESH-0214b automatic reclamation).
 *
 * vals/n are the values that flow OUT of the ending iteration: the freshly
 * evaluated loop arguments at a tail-call back edge, or the loop's result
 * value at loop exit. If none of them can point into this iteration's scope
 * span, everything allocated during the iteration is garbage and the scope
 * is popped (rewound -- bounded RSS). If any of them might, the scope is
 * committed instead: memory is retained exactly as it was before this
 * feature (correctness over reclamation), and the scope stack stays
 * balanced either way.
 *
 * The escape test is deliberately conservative in the safe direction:
 * only provably pointer-free immediates (null / int64 / double / bool /
 * char, plus the pointer-free eof-object) skip the pointer check; every
 * other type tag is treated as potentially pointer-carrying. Pre-existing
 * structures cannot point INTO the iteration span (mutation is excluded
 * statically by the codegen-side analysis that gates this whole mechanism),
 * so a shallow per-value span test is sufficient -- no transitive walk. */
void eshkol_arena_iter_scope_end(arena_t* arena, const eshkol_tagged_value_t* vals, uint64_t n) {
    if (!arena) return;
    // Concurrent pool worker on the shared arena: commit-only. The loop-entry
    // arena_push_scope was a no-op, so there is no per-iteration scope to end;
    // retain this iteration's allocations and never rewind the shared arena
    // (see arena_scope_ops_are_commit_only). Skips the escape test entirely,
    // which is the conservative (always-commit) direction anyway.
    if (arena_scope_ops_are_commit_only(arena)) return;
    if (!arena->current_scope) {
        eshkol_error("iter_scope_end with no active arena scope - unbalanced loop scoping");
        return;
    }

    int escapes = 0;
    for (uint64_t i = 0; i < n && !escapes; ++i) {
        const uint8_t t = vals[i].type;
        /* Immediate (pointer-free) tags: NULL(0)/INT64(1)/DOUBLE(2)/BOOL(3)/
         * CHAR(4). Exactness lives in the separate flags byte, so these tag
         * values are exact matches. 0xFF is the eof-object (data always 0).
         * Anything else (heap, callable, symbol, dual, complex, ports with
         * flag bits OR'd into the tag, logic vars, multimedia, legacy tags)
         * is treated as potentially pointer-carrying -- misclassifying a
         * non-pointer as a pointer can only cause a spurious commit (a
         * missed reclamation), never a use-after-free. */
        if (t <= ESHKOL_VALUE_CHAR || t == 0xFF) continue;
        const void* p = (const void*)(uintptr_t)vals[i].data.ptr_val;
        if (p && arena_top_scope_contains(arena, p)) escapes = 1;
    }

    if (escapes) {
        arena_commit_scope(arena);
    } else {
        arena_pop_scope(arena);
    }
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
    if (count != 0 && element_size > SIZE_MAX / count) {
        eshkol_error("Arena list-node allocation overflow: element_size=%zu count=%zu",
                     element_size, count);
        return nullptr;
    }
    return arena_allocate(arena, element_size * count);
}
