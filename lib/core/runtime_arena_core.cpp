/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Raw arena block, scope, statistics, and legacy list allocation runtime.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include <eshkol/core/resource_limits.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <atomic>
#include <cstring>
#include <vector>
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
    // SW-10: the process heap ceiling (ESHKOL_MAX_HEAP) is enforced here
    // because this is the ONE place the arena asks the OS for memory — every
    // other allocation is a bump of a pointer inside a block that already
    // exists. Checking here therefore costs one comparison per block (a
    // megabyte at a time by default), never one per object, and leaves the
    // bump-pointer fast path in arena_allocate_aligned() byte-for-byte as it
    // was. It cannot perturb any computation: it either admits the block or
    // ends the process.
    // The heap ceiling only binds a run that asked for one (ESHKOL_MAX_HEAP).
    // eshkol_track_allocation() still accounts every block either way, so
    // eshkol_get_heap_usage()/peak stay accurate for diagnostics.
    if (!eshkol_track_allocation(size) &&
        eshkol_limit_is_active(ESHKOL_LIMIT_ACTIVE_HEAP)) {
        // Terminates under ESHKOL_ENFORCE_LIMITS=true. If it returns, limits
        // are advisory: record the breach and hand the block over anyway. The
        // arena's callers treat a null block as a fatal allocation failure, so
        // refusing here would make "limits are not enforced" the more violent
        // of the two settings.
        eshkol_limit_enforce(ESHKOL_LIMIT_HEAP_HARD, "arena block");
    }

    arena_block_t* block = (arena_block_t*)malloc(sizeof(arena_block_t));
    if (!block) {
        eshkol_error("Failed to allocate arena block structure");
        eshkol_track_deallocation(size);
        return nullptr;
    }

    block->memory = (uint8_t*)malloc(size);
    if (!block->memory) {
        eshkol_error("Failed to allocate arena block memory of size %zu", size);
        free(block);
        eshkol_track_deallocation(size);
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
        // Mirror of the create_arena_block() accounting, so a program that
        // allocates and releases in a loop is measured by what it HOLDS rather
        // than by everything it has ever touched. Without this pairing the
        // heap ceiling would degrade into a cap on cumulative allocation.
        eshkol_track_deallocation(block->size);
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

    arena->adopted_blocks = nullptr;   // SW-74: filled only by arena_adopt_blocks()
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

    // SW-74: blocks promoted into this arena from a pinned region
    // (arena_adopt_blocks) are owned here and die here. They are poisoned on
    // the same terms as the allocation chain, which is what makes a stale
    // continuation-held pointer into a promoted region read as 0xCB rather
    // than as plausible data once the enclosing scope has ended.
    block = arena->adopted_blocks;
    while (block) {
        arena_block_t* next = block->next;
        if (poison && block->memory && block->used > 0) {
            std::memset(block->memory, 0xCB, block->used);
        }
        free_arena_block(block);
        block = next;
    }
    arena->adopted_blocks = nullptr;

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

/* SW-164: end the innermost scope, retaining the named header-prefixed objects.
 *
 * See arena_memory.h for the contract. The implementation is a two-phase copy
 * because a bump allocator cannot allocate BELOW its own high-water mark: the
 * survivors are staged out of the dying span into a reusable thread-local
 * buffer, the scope is popped (which rewinds the bump pointer and frees every
 * block added since the mark), and the survivors are then re-allocated at the
 * rewound mark and copied back. Cost is O(size of the RESULT), paid once per
 * bracketed operation — against a scratch footprint that is O(the operation's
 * intermediate work), which is the whole point.
 *
 * The staging buffer is thread-local and grows monotonically to the largest
 * retained set a thread has seen, so the steady state does no malloc at all. */
namespace {

struct RetainStagingBuffer {
    std::vector<uint8_t> bytes;
};

// One reusable staging buffer per thread. Retention brackets are strictly
// LIFO and never nest across a retention (the staged copy is consumed before
// the primitive returns), so a single buffer per thread is sufficient.
static RetainStagingBuffer& retain_staging() {
    static thread_local RetainStagingBuffer buf;
    return buf;
}

} // namespace

int arena_scope_end_retaining(arena_t* arena, void** objects, size_t n) {
    if (!arena) return 0;

    // Concurrent pool worker on the shared arena: the matching push was a
    // no-op and rewinding would free memory a sibling worker may still be
    // using, so this degenerates to a commit — retain everything, move no
    // pointers (see arena_scope_ops_are_commit_only).
    if (arena_scope_ops_are_commit_only(arena)) return 0;

    if (!arena->current_scope) {
        eshkol_error("arena_scope_end_retaining with no active arena scope - "
                     "unbalanced scope operations risk memory corruption");
        return 0;
    }

    // Phase 1 — stage. Read every survivor's header BEFORE the pop, because
    // the pop is what makes those bytes reusable. Objects that already live
    // below the scope mark are not staged: they outlive the rewind untouched,
    // and copying them would needlessly break pointer identity.
    RetainStagingBuffer& staging = retain_staging();
    staging.bytes.clear();

    // offsets[i] is the staged copy's offset into staging.bytes, or SIZE_MAX
    // for an entry that is not being moved (NULL, or already below the mark).
    size_t stack_offsets[8];
    void*  stack_old_payloads[8];
    std::vector<size_t> heap_offsets;
    std::vector<void*>  heap_old_payloads;
    size_t* offsets = stack_offsets;
    void** old_payloads = stack_old_payloads;
    if (n > (sizeof(stack_offsets) / sizeof(stack_offsets[0]))) {
        heap_offsets.resize(n);
        heap_old_payloads.resize(n);
        offsets = heap_offsets.data();
        old_payloads = heap_old_payloads.data();
    }

    bool any_staged = false;
    for (size_t i = 0; i < n; ++i) {
        offsets[i] = SIZE_MAX;
        old_payloads[i] = nullptr;
        void* payload = objects ? objects[i] : nullptr;
        if (!payload) continue;
        if (!arena_top_scope_contains(arena, payload)) continue;

        // Layout is read through the ESHKOL_GET_* accessor family, never
        // recomputed here: the object header is scheduled to change, and a
        // site that derives the offset itself is a site the migration can
        // miss (see scripts/abi_header_inventory.py).
        const uint8_t* base = (const uint8_t*)ESHKOL_GET_OBJECT_BASE(payload);
        const size_t total = ESHKOL_GET_OBJECT_TOTAL_SIZE(payload);
        const size_t off = staging.bytes.size();
        staging.bytes.insert(staging.bytes.end(), base, base + total);
        offsets[i] = off;
        old_payloads[i] = payload;
        any_staged = true;
    }

    if (!any_staged) {
        // Nothing to move: the plain rewind is the whole operation.
        arena_pop_scope(arena);
        return 1;
    }

    // Phase 2 — rewind. Everything the bracketed operation allocated, other
    // than the staged survivors, is reclaimed here.
    arena_pop_scope(arena);

    // Phase 3 — re-allocate the survivors at the rewound mark and copy back.
    // Allocation goes through arena_allocate_aligned with the SAME 8-byte
    // alignment arena_allocate_with_header uses, and the staged bytes already
    // carry the object header, so the reconstructed object is byte-identical
    // apart from its address.
    for (size_t i = 0; i < n; ++i) {
        if (offsets[i] == SIZE_MAX) continue;
        uint8_t* staged = staging.bytes.data() + offsets[i];
        // The staged bytes begin with the object header, so the staged
        // PAYLOAD is what the accessor family expects to be handed.
        void* staged_payload = ESHKOL_GET_DATA_PTR(staged);
        const size_t total = ESHKOL_GET_OBJECT_TOTAL_SIZE(staged_payload);

        void* raw = arena_allocate_aligned(arena, total, 8);
        if (!raw) {
            // Out of memory mid-restore. The survivor cannot be reconstructed
            // and its old address is gone, so say so loudly rather than hand
            // back a dangling pointer: a silent wrong answer here would be an
            // arena use-after-free with no diagnostic.
            eshkol_error("arena_scope_end_retaining: failed to re-allocate a "
                         "retained object of %zu bytes after rewind", total);
            objects[i] = nullptr;
            continue;
        }
        std::memcpy(raw, staged, total);
        objects[i] = ESHKOL_GET_DATA_PTR(raw);
    }

    // Interior references BETWEEN retained objects are the CALLER's to
    // repair, and deliberately so: this primitive will not scan a retained
    // payload for word-sized values that look like old addresses. A bignum's
    // limb array is arbitrary 64-bit arithmetic data, and rewriting a limb
    // that merely collided with a stale address would corrupt a number with
    // no diagnostic — exactly the silent-wrong outcome the memory model must
    // not produce. Callers declare the whole retained set in one call, so
    // objects[] gives them every new address they need to re-point.

    return 1;
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

/* SW-164: generalization of arena_top_scope_contains to an arbitrary scope
 * mark. Same block walk — blocks are head-inserted, so the post-mark span is
 * every block from current_block down to (but excluding) scope->block, plus
 * [scope->used, used) inside scope->block itself — but starting from a scope
 * the caller names rather than the innermost one. That is what lets a loop ask
 * "would popping the LOOP's scope reclaim this?" while an iteration scope is
 * still open inside it. A NULL scope degenerates to whole-arena containment. */
int arena_scope_span_contains(const arena_t* arena, const arena_scope_t* scope,
                              const void* ptr) {
    if (!arena || !ptr) return 0;
    if (!scope) return arena_contains(arena, ptr);
    const uint8_t* p = (const uint8_t*)ptr;
    for (const arena_block_t* b = arena->current_block; b; b = b->next) {
        if (b == scope->block) {
            return (p >= b->memory + scope->used) && (p < b->memory + b->used);
        }
        if (p >= b->memory && p < b->memory + b->used) return 1;
    }
    return 0;
}

/* Does ptr point into ANY live allocation of this arena?
 *
 * Scope-independent companion to arena_top_scope_contains: walks every block in
 * the chain and asks whether ptr lies below that block's high-water mark. This
 * is the residency test a caller needs BEFORE dereferencing an integer that
 * might be an arena pointer — see eshkol_ad_node_probe(), where the integer in
 * question may equally well be the bit pattern of a subnormal double. */
int arena_contains(const arena_t* arena, const void* ptr) {
    if (!arena || !ptr) return 0;
    const uint8_t* p = (const uint8_t*)ptr;
    for (const arena_block_t* b = arena->current_block; b; b = b->next) {
        if (p >= b->memory && p < b->memory + b->used) return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SW-164: the per-iteration loop scope PROMOTES; it no longer gives up.
 *
 * ESH-0214b gave a self-tail loop a per-iteration arena scope so its transient
 * garbage is reclaimed at every back edge. It reclaimed only when NOTHING
 * flowing into the next iteration pointed into the ending iteration's span; if
 * anything did, it committed — retained the whole iteration, exactly as if the
 * feature were switched off — because there was no way to move a survivor out
 * of a span that was about to be rewound.
 *
 * That fallback is the common case, not the rare one: any loop that ACCUMULATES
 * builds its accumulator inside the iteration, so the accumulator always points
 * into the span, so the scope always commits and the loop always retains every
 * iteration. It happened to be invisible for loops whose accumulator is a
 * machine word or a small bignum (a few hundred bytes an iteration), and very
 * visible for one whose accumulator is an exact rational.
 *
 * The fix is to promote instead of to give up. A loop now opens a LOOP scope
 * once at entry (eshkol_arena_loop_scope_begin), outside the per-iteration
 * scope, and an escaping back edge:
 *
 *   1. deep-evacuates the loop-carried out-values out of everything above the
 *      LOOP mark — which is this iteration's garbage AND the previous
 *      iteration's promoted, now-dead accumulator — into a scratch arena;
 *   2. pops the iteration scope and the loop scope, rewinding the arena all the
 *      way back to the loop's entry mark;
 *   3. reopens the loop scope and copies the survivors back in.
 *
 * The arena and the scratch arena are the two halves of a semispace: the live
 * loop-carried set ping-pongs between them and everything else dies each pass,
 * so resident size is bounded by (live set × 2) instead of growing with the
 * iteration count. Cost is one deep copy of the LIVE set per escaping back
 * edge — proportional to what the iteration produced, never to what it
 * allocated to get there.
 *
 * The no-escape fast path is untouched: a plain pop, no copy, no scratch arena.
 * Soundness rests on the same codegen-side precondition the whole feature has
 * always rested on (loopBodyIterScopeSafe): a loop body admitted here cannot
 * leak a value into pre-existing structure, so the loop-carried out-values are
 * the ONLY roots into the span and moving them moves everything reachable.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Deep evacuation lives with the region evacuator (lib/core/runtime_regions.cpp)
 * so there is exactly one copier, one forwarding map and one per-subtype
 * interior walk in the runtime. */
extern "C" int eshkol_arena_span_evacuate(eshkol_tagged_value_t* vals, uint64_t n,
                                          arena_t* target,
                                          const arena_t* span_arena,
                                          const arena_scope_t* span_scope);

namespace {

/* The registered LOOP scope for each arena with a loop currently running on
 * this thread. A stack, because loops nest. Thread-local because arena scopes
 * are: a worker thread's loop must never see the main thread's mark. */
struct LoopScopeRecord {
    arena_t* arena;
    arena_scope_t* scope;
    /* Bytes the last promotion had to copy — this loop's live set as most
     * recently measured. Drives the amortization rule below. */
    size_t last_live_bytes;
};

static std::vector<LoopScopeRecord>& loop_scopes() {
    static thread_local std::vector<LoopScopeRecord> stack;
    return stack;
}

/* The other half of the semispace: one scratch arena per thread, created on
 * the first escaping back edge and reused (arena_reset keeps its first block,
 * so the steady state does no malloc) until the thread ends. */
struct IterStagingArena {
    arena_t* arena = nullptr;
    ~IterStagingArena() { if (arena) arena_destroy(arena); }
};

static arena_t* iter_staging_arena() {
    static thread_local IterStagingArena holder;
    if (!holder.arena) holder.arena = arena_create(64 * 1024);
    return holder.arena;
}

/* Could any out-value point into the ending iteration's span?
 *
 * Deliberately conservative in the safe direction: only provably pointer-free
 * immediates (null / int64 / double / bool / char, plus the pointer-free
 * eof-object) skip the pointer check; every other type tag is treated as
 * potentially pointer-carrying. Misclassifying a non-pointer as a pointer costs
 * a promotion that was not needed, never a use-after-free. */
static bool iter_values_escape(const arena_t* arena,
                               const eshkol_tagged_value_t* vals, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        const uint8_t t = vals[i].type;
        if (t <= ESHKOL_VALUE_CHAR || t == 0xFF) continue;
        const void* p = (const void*)(uintptr_t)vals[i].data.ptr_val;
        if (p && arena_top_scope_contains(arena, p)) return true;
    }
    return false;
}

/* Bytes @p arena has allocated above @p scope's mark: this loop's whole
 * footprint since it started, live and dead together. Same block walk as
 * arena_scope_span_contains. */
static size_t arena_span_bytes(const arena_t* arena, const arena_scope_t* scope) {
    if (!arena || !scope) return 0;
    size_t total = 0;
    for (const arena_block_t* b = arena->current_block; b; b = b->next) {
        if (b == scope->block) {
            return total + (b->used > scope->used ? b->used - scope->used : 0);
        }
        total += b->used;
    }
    return total;
}

/* Bytes currently held in @p arena's blocks. Used to measure a promotion's
 * live set after the fact: what landed in the scratch arena IS the live set. */
static size_t arena_used_bytes(const arena_t* arena) {
    if (!arena) return 0;
    size_t total = 0;
    for (const arena_block_t* b = arena->current_block; b; b = b->next) total += b->used;
    return total;
}

/* ─── When promotion is worth its cost ──────────────────────────────────────
 *
 * Promotion buys reclamation with a copy of the LIVE set, so it pays only when
 * the span it reclaims is bigger than the live set it copies. For a loop whose
 * accumulator is a value — an exact rational, a bignum, a string — that is
 * true almost every iteration: the span holds every superseded accumulator and
 * the live set is one of them.
 *
 * For a loop whose accumulator GROWS BY ACCRETION — consing onto a list,
 * extending a vector — it is not. There the span IS the live set: nothing in it
 * is garbage, and promoting on every back edge would copy the whole structure
 * every iteration and turn a linear loop into a quadratic one. That is a far
 * worse failure than the retention it would be curing.
 *
 * So promotion is gated the way a copying collector gates a minor collection:
 * only when the span has grown to a multiple of the last measured live set,
 * which makes the copying amortize to O(total allocation) — each promotion
 * copies at most half of what it reclaims. An accretion loop measures span
 * ≈ live on its first promotion and then never qualifies again, falling back
 * to exactly the pre-SW-164 retention. A value-accumulator loop qualifies every
 * few iterations and stays flat. The floor keeps very small loops from paying
 * for any of this. */
static const size_t kIterPromoteMinSpanBytes = 64u * 1024u;
static const size_t kIterPromoteSpanOverLive = 2u;

static bool iter_promotion_is_worthwhile(const arena_t* arena,
                                         const LoopScopeRecord& rec) {
    const size_t span = arena_span_bytes(arena, rec.scope);
    if (span < kIterPromoteMinSpanBytes) return false;
    if (rec.last_live_bytes == 0) return true;
    if (rec.last_live_bytes > span / kIterPromoteSpanOverLive) return false;
    return true;
}

/* The LOOP scope registered for @p arena, if the innermost one belongs to it
 * AND is exactly the parent of the scope now ending. Anything else means the
 * codegen did not open a loop scope for this loop (or the scope stack is not in
 * the shape this expects), in which case the caller falls back to the old
 * commit — retaining memory, which is always the safe direction. */
/* Drop registrations whose scope is no longer live on the arena.
 *
 * eshkol_arena_loop_scope_begin() runs once per loop activation and
 * eshkol_arena_iter_scope_finish() balances it on every exit the codegen
 * emits — but a NON-LOCAL exit (a raise unwinding through the loop, a guard
 * transferring out) reaches neither. Without this the abandoned record stays on
 * top of the stack forever, so every later loop on the thread compares its own
 * scope against a stale one, mismatches, and silently falls back to retention:
 * the reclamation would switch itself off for the rest of the run and say
 * nothing. Walking the arena's live scope chain and dropping records that are
 * no longer on it makes the registry self-healing. A freed scope record whose
 * address malloc has since reused reads as live here; that only costs the
 * mismatch check below, which then retains rather than rewinds. */
static void prune_dead_loop_scopes(const arena_t* arena) {
    std::vector<LoopScopeRecord>& stack = loop_scopes();
    while (!stack.empty()) {
        const LoopScopeRecord& top = stack.back();
        if (top.arena != arena) break;   /* another arena's loop: leave it */
        bool live = false;
        for (const arena_scope_t* sc = arena->current_scope; sc; sc = sc->parent) {
            if (sc == top.scope) { live = true; break; }
        }
        if (live) break;
        stack.pop_back();
    }
}

static arena_scope_t* registered_loop_scope(const arena_t* arena) {
    std::vector<LoopScopeRecord>& stack = loop_scopes();
    if (stack.empty()) return nullptr;
    const LoopScopeRecord& top = stack.back();
    if (top.arena != arena) return nullptr;
    return top.scope;
}

/* Steps 1-3 of the block comment above. @p reopen distinguishes a back edge
 * (reopen the loop scope for the next iteration) from the loop's exit (do not;
 * the survivors land in the scope that encloses the whole loop). */
static void iter_retain(arena_t* arena, bool reopen) {
    /* Give up on reclaiming this iteration, keeping the scope stack balanced —
     * the pre-SW-164 behavior. At a loop exit the loop scope is committed too
     * and unregistered, so the survivors keep their addresses in the scope that
     * encloses the whole loop. */
    arena_commit_scope(arena);
    if (!reopen) {
        arena_commit_scope(arena);
        if (!loop_scopes().empty()) loop_scopes().pop_back();
    }
}

static void iter_promote_and_rewind(arena_t* arena, arena_scope_t* loop_scope,
                                    eshkol_tagged_value_t* vals, uint64_t n,
                                    bool reopen) {
    arena_t* staging = iter_staging_arena();
    if (!staging) {
        iter_retain(arena, reopen);
        return;
    }

    /* Stage into a COPY of the out-values. Nothing in the arena has moved yet,
     * so when the evacuator refuses (see EvacState::refuse — something in the
     * span must not be moved, a live AD tape node above all) the staged bytes
     * are simply dropped and the caller's values are still the originals. */
    std::vector<eshkol_tagged_value_t> staged(vals, vals + n);
    const int ok = eshkol_arena_span_evacuate(staged.data(), n, staging,
                                              arena, loop_scope);
    if (!ok) {
        arena_reset(staging);
        iter_retain(arena, reopen);
        return;
    }

    /* What landed in the scratch arena IS this loop's live set; remember it so
     * the amortization gate above can compare the next span against it. */
    if (!loop_scopes().empty()) {
        loop_scopes().back().last_live_bytes = arena_used_bytes(staging);
    }

    arena_pop_scope(arena);   /* the ending iteration */
    arena_pop_scope(arena);   /* the loop scope: back to the loop's entry mark */

    if (reopen) {
        arena_push_scope(arena);
        loop_scopes().back().scope = arena->current_scope;
    } else {
        loop_scopes().pop_back();
    }

    /* Past the point of no return: the span is gone, so this copy back cannot
     * be refused — and cannot need to be. Everything now being copied was
     * produced by the copy OUT, which already accepted every object in it. */
    eshkol_arena_span_evacuate(staged.data(), n, arena, staging, nullptr);
    for (uint64_t i = 0; i < n; ++i) vals[i] = staged[i];
    arena_reset(staging);
}

} // namespace

/* Open a loop's LOOP scope. Called once per loop activation, in the setup block
 * that dominates the loop header, for a loop lowered onto the per-iteration
 * arena scope. Balanced by eshkol_arena_iter_scope_finish on every exit path. */
void eshkol_arena_loop_scope_begin(arena_t* arena) {
    if (!arena) return;
    /* Concurrent pool worker on the shared arena: scope ops are commit-only,
     * so there is no mark to register and nothing will ever be rewound. */
    if (arena_scope_ops_are_commit_only(arena)) return;
    /* A loop abandoned by a non-local exit never reached its finish; clear any
     * such record before registering this one (see prune_dead_loop_scopes). */
    prune_dead_loop_scopes(arena);
    arena_push_scope(arena);
    LoopScopeRecord rec;
    rec.arena = arena;
    rec.scope = arena->current_scope;
    rec.last_live_bytes = 0;
    loop_scopes().push_back(rec);
}

/* End a per-iteration loop scope at a TAIL-CALL BACK EDGE.
 *
 * vals/n are the values that flow OUT of the ending iteration: the freshly
 * evaluated loop arguments. They are rewritten IN PLACE when a promotion moves
 * them, so the caller must store the values back into the loop's parameter
 * slots after this returns — the originals no longer exist. */
void eshkol_arena_iter_scope_end(arena_t* arena, eshkol_tagged_value_t* vals, uint64_t n) {
    if (!arena) return;
    /* Concurrent pool worker on the shared arena: commit-only. The loop-entry
     * arena_push_scope was a no-op, so there is no per-iteration scope to end;
     * retain this iteration's allocations and never rewind the shared arena
     * (see arena_scope_ops_are_commit_only). Skips the escape test entirely,
     * which is the conservative (always-commit) direction anyway. */
    if (arena_scope_ops_are_commit_only(arena)) return;
    if (!arena->current_scope) {
        eshkol_error("iter_scope_end with no active arena scope - unbalanced loop scoping");
        return;
    }

    if (!iter_values_escape(arena, vals, n)) {
        arena_pop_scope(arena);
        return;
    }

    arena_scope_t* loop_scope = registered_loop_scope(arena);
    if (!loop_scope || loop_scope != arena->current_scope->parent) {
        /* No loop scope in the expected shape: fall back to the pre-SW-164
         * behavior — retain this iteration, keep the scope stack balanced. */
        arena_commit_scope(arena);
        return;
    }

    if (!iter_promotion_is_worthwhile(arena, loop_scopes().back())) {
        /* Not yet worth a copy of the live set (see the note above). Retain
         * this iteration; a later back edge, with more garbage behind it, will
         * reclaim these bytes too. */
        arena_commit_scope(arena);
        return;
    }

    iter_promote_and_rewind(arena, loop_scope, vals, n, /*reopen=*/true);
}

/* End the FINAL iteration and close the loop: pops both the iteration scope and
 * the loop scope, promoting the loop's result out of them first. vals/n are the
 * values flowing out of the loop (its result); they are rewritten in place, so
 * the caller must use the values this leaves behind. */
void eshkol_arena_iter_scope_finish(arena_t* arena, eshkol_tagged_value_t* vals, uint64_t n) {
    if (!arena) return;
    if (arena_scope_ops_are_commit_only(arena)) return;
    if (!arena->current_scope) {
        eshkol_error("iter_scope_finish with no active arena scope - unbalanced loop scoping");
        return;
    }

    arena_scope_t* loop_scope = registered_loop_scope(arena);
    if (!loop_scope || loop_scope != arena->current_scope->parent) {
        /* Unexpected shape: end the iteration the old way. This loop is
         * leaving, so drop its registration if it is the one on top —
         * otherwise the abandoned record would disable promotion for every
         * later loop on this thread. */
        if (iter_values_escape(arena, vals, n)) arena_commit_scope(arena);
        else arena_pop_scope(arena);
        std::vector<LoopScopeRecord>& stack = loop_scopes();
        if (!stack.empty() && stack.back().arena == arena &&
            stack.back().scope == loop_scope) {
            stack.pop_back();
        }
        return;
    }

    /* Unlike a back edge there is no fast path here: even when nothing points
     * into the FINAL iteration, the loop scope still holds whatever earlier
     * back edges promoted into it, and the result is very often one of those.
     * It has to be staged out before the loop scope is rewound. */
    iter_promote_and_rewind(arena, loop_scope, vals, n, /*reopen=*/false);
}

void arena_reset(arena_t* arena) {
    if (!arena) return;

    // SW-74: a reset discards everything this arena holds, and adopted blocks
    // (promoted from a pinned region) are held by this arena, so they go too.
    // Keeping them would make reset a partial rewind and leave the promoted
    // bytes unreachable but still charged.
    {
        arena_block_t* adopted = arena->adopted_blocks;
        while (adopted) {
            arena_block_t* next = adopted->next;
            arena->total_allocated -= adopted->size;
            free_arena_block(adopted);
            adopted = next;
        }
        arena->adopted_blocks = nullptr;
    }

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

// SW-74: zero-copy promotion of a pinned region's arena into the arena that
// encloses it. See arena_memory.h for the contract and the VM analogue
// (vm_evac_promote_all_blocks, lib/backend/vm_region_evac.c).
//
// The moved chain is spliced onto dst->adopted_blocks rather than onto
// dst->current_block for one reason that matters: dst is usually still being
// allocated into, and both allocation (arena_allocate_aligned) and rewind
// (arena_pop_scope, arena_reset) walk the current_block chain. Splicing there
// would either hand a later allocation a pointer INTO memory a live
// continuation still reads, or let a scope pop free it early. The adopted list
// is walked by nothing but teardown.
size_t arena_adopt_blocks(arena_t* dst, arena_t* src) {
    if (!dst || !src || dst == src) return 0;

    arena_lock(src);
    arena_block_t* chain = src->current_block;
    arena_block_t* also = src->adopted_blocks;
    src->current_block = nullptr;
    src->adopted_blocks = nullptr;
    arena_unlock(src);

    if (!chain && !also) return 0;

    // A region that was itself a promotion target (nested pinned regions pop
    // innermost-first) carries adopted blocks of its own; they promote onward
    // in the same move, so an N-deep pinned nest costs one splice per level and
    // never a copy.
    arena_block_t* tail = chain;
    if (tail) {
        while (tail->next) tail = tail->next;
        tail->next = also;
    } else {
        chain = also;
    }

    size_t moved = 0;
    arena_block_t* last = chain;
    for (arena_block_t* b = chain; b; b = b->next) {
        moved += b->size;
        last = b;
    }

    arena_lock(dst);
    last->next = dst->adopted_blocks;
    dst->adopted_blocks = chain;
    dst->total_allocated += moved;
    arena_unlock(dst);

    // src->total_allocated is deliberately left alone: src is on its way to
    // arena_destroy(), and decrementing it would only make the debug line
    // printed there disagree with what the arena actually held.
    return moved;
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
