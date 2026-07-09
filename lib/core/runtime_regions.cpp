/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime region and thread-local arena helpers.
 *
 * This unit owns the process/global arena selection ABI, per-worker
 * thread-local arena lifecycle, and ownership-aware lexical region stack.
 * Raw arena block/scope mechanics remain in arena_memory.cpp.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <utility>
#include <atomic>
#include <mutex>

void eshkol_arena_global_once(void (*init)(void));

// Thread-local region stack (safe for parallel-map + with-region).
thread_local eshkol_region_t* __region_stack[MAX_REGION_DEPTH] = {nullptr};
thread_local uint64_t __region_stack_depth = 0;

// Default global arena for allocations outside any region.
// Non-static to allow JIT code to access it directly. Weak where object format
// support lets generated standalone code override it.
ESHKOL_RUNTIME_WEAK arena_t* __global_arena = nullptr;

static thread_local arena_t* __thread_local_arena = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// THREAD-SAFE REGION ARENA ROUTING (parallel-map/fold/execute/future × with-region)
//
// Generated code funnels *every* allocation through the single process-global
// __global_arena pointer (the "current allocation arena" slot). `with-region`
// makes body allocations land in the region's arena by temporarily overwriting
// that slot with the region arena and restoring it on exit. That save/hijack/
// restore is correct for a single thread, but __global_arena is ONE shared
// location: when work-stealing workers run a body that uses with-region, several
// threads hijack/restore the same pointer concurrently — a worker allocates into
// another worker's region arena, whose region_pop then frees it out from under
// the first — SIGSEGV / heap corruption.
//
// Making __global_arena itself thread-local is not viable: generated code reads
// it as a plain (non-TLS) global across the LLVM↔C / JIT boundary, and this
// codebase deliberately avoids TLS symbols there (see the __current_ad_tape note
// in arena_memory.h). Instead the hijack is moved out of codegen into this
// runtime, where it is performed ONLY when it is genuinely safe — a
// single-threaded, non-parallel context — and a parallel-scope guard keeps the
// shared slot pointed at the true thread-safe process arena for the duration of
// any work-stealing construct, so concurrent workers never observe a hijacked
// (non-thread-safe) region arena. In a parallel/worker context body allocations
// stay in the thread-safe shared arena; escape promotion (region_escape*/write
// barrier) still copies any region-arena-resident value out, so results are
// correct. The only tradeoff is weaker region reclamation while parallel work is
// in flight (transient region allocations land in the shared arena instead of
// being freed at region_pop) — a bounded, documented cost, never a correctness
// hazard. Single-threaded programs are entirely unaffected (identical hijack).
static arena_t* s_shared_root_arena = nullptr;         // true process-wide thread-safe arena
static std::atomic<int> s_parallel_depth{0};           // >0 while a work-stealing construct may run
static std::mutex s_parallel_arena_mtx;                // guards the 0<->1 transition swap
static arena_t* s_saved_arena_at_parallel = nullptr;   // __global_arena captured at the 0->1 edge

// Sentinel returned by eshkol_region_enter when it did NOT hijack the shared
// slot (parallel/worker context). No real arena lives at address 0x1.
static arena_t* const REGION_NO_HIJACK = reinterpret_cast<arena_t*>(0x1);

/**
 * @brief One-time initializer that creates the process-wide global arena.
 *
 * Invoked exactly once (via eshkol_arena_global_once) to create a
 * thread-safe arena and store it in __global_arena; logs an error if
 * creation fails. Also records it as the immutable thread-safe "root" arena
 * (s_shared_root_arena) that the parallel-scope guard points workers at while a
 * work-stealing construct is active — captured here, before any with-region can
 * hijack the __global_arena slot.
 */
static void init_global_arena_internal() {
    __global_arena = arena_create_threadsafe(65536);
    if (!__global_arena) {
        eshkol_error("Failed to create global arena");
    }
    s_shared_root_arena = __global_arena;
}

/**
 * @brief Return the arena that allocations outside any lexical region should use.
 *
 * Lazily creates the process-wide global arena on first call (thread-safe,
 * via eshkol_arena_global_once). Prefers the calling thread's dedicated
 * thread-local arena, if one has been created via
 * arena_create_thread_local(); otherwise falls back to the shared global
 * arena.
 *
 * @return The current thread's thread-local arena if set, else the shared
 *         global arena.
 */
arena_t* get_global_arena() {
    eshkol_arena_global_once(init_global_arena_internal);
    if (__thread_local_arena) return __thread_local_arena;
    return __global_arena;
}

/**
 * @brief Return the shared (non-thread-local) process-wide global arena, creating it if needed.
 *
 * Unlike get_global_arena(), always returns __global_arena directly, even if
 * the calling thread has its own thread-local arena.
 *
 * @return The shared global arena.
 */
arena_t* get_global_arena_shared() {
    eshkol_arena_global_once(init_global_arena_internal);
    return __global_arena;
}

/**
 * @brief Return the calling thread's thread-local arena, falling back to the global arena.
 *
 * @return This thread's arena if one was created via
 *         arena_create_thread_local(), else the result of
 *         get_global_arena().
 */
arena_t* arena_get_thread_local(void) {
    if (__thread_local_arena) return __thread_local_arena;
    return get_global_arena();
}

/**
 * @brief Create (once) this thread's dedicated thread-local arena.
 *
 * If the calling thread already has a thread-local arena, returns it
 * unchanged (idempotent). Otherwise creates a new arena with a block size
 * of @p size_hint bytes, or 1 MiB if @p size_hint is 0.
 *
 * @param size_hint  Initial block size in bytes, or 0 to use the default
 *                    (1 MiB).
 * @return            The thread-local arena (existing or newly created), or
 *                     NULL if creation failed.
 */
arena_t* arena_create_thread_local(size_t size_hint) {
    if (__thread_local_arena) return __thread_local_arena;

    const size_t block_size = size_hint > 0 ? size_hint : (1024 * 1024);
    __thread_local_arena = arena_create(block_size);
    return __thread_local_arena;
}

/**
 * @brief Initialize all per-thread runtime state for a newly started worker thread.
 *
 * Resets the thread-local AD (automatic differentiation) tape stack, outer
 * AD node stack, and their associated depth/pointer bookkeeping, resets the
 * region stack to empty, and creates the thread's dedicated thread-local
 * arena via arena_create_thread_local(). Must be called before a worker
 * thread (e.g. a parallel-map worker) runs any Eshkol code, so its AD/region
 * state starts clean rather than inheriting the spawning thread's values.
 *
 * @param arena_size_hint Initial block size hint (bytes) forwarded to
 *                        arena_create_thread_local(); 0 uses the default.
 */
void eshkol_thread_init_worker(size_t arena_size_hint) {
    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;
    __ad_pert_level = 0;

    __outer_ad_node_storage = nullptr;
    __outer_ad_node_to_inner = nullptr;
    __outer_grad_accumulator = nullptr;
    __inner_var_node_ptr = nullptr;
    __gradient_x_degree = 0;

    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;

    (void)arena_create_thread_local(arena_size_hint);
}

/**
 * @brief Tear down all per-thread runtime state for a worker thread that is exiting.
 *
 * Destroys and clears the thread's thread-local arena (if any), then resets
 * the same AD tape/outer-node stacks and region stack that
 * eshkol_thread_init_worker() initializes, so no stale pointers survive the
 * worker's exit.
 */
void eshkol_thread_shutdown_worker(void) {
    if (__thread_local_arena) {
        arena_destroy(__thread_local_arena);
        __thread_local_arena = nullptr;
    }

    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;
    __ad_pert_level = 0;

    __outer_ad_node_storage = nullptr;
    __outer_ad_node_to_inner = nullptr;
    __outer_grad_accumulator = nullptr;
    __inner_var_node_ptr = nullptr;
    __gradient_x_degree = 0;

    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;
}

/**
 * @brief Splice @p src's allocated blocks onto the end of @p dest's block list, emptying @p src.
 *
 * Appends src's block chain (if any) to the tail of dest's block chain,
 * adds src's total_allocated to dest's, and clears src's block pointer and
 * total_allocated so src is left empty but still valid. Takes dest's lock
 * for the duration if dest is thread-safe. No-op if either arena is NULL or
 * they are the same arena.
 *
 * @param dest Arena that receives @p src's blocks (mutated in place).
 * @param src  Arena whose blocks are moved out (left empty on return).
 */
void arena_merge_to_parent(arena_t* dest, arena_t* src) {
    if (!dest || !src || dest == src) return;

    if (dest->thread_safe) arena_lock(dest);

    if (src->current_block) {
        if (dest->current_block) {
            arena_block_t* dest_tail = dest->current_block;
            while (dest_tail->next) dest_tail = dest_tail->next;
            dest_tail->next = src->current_block;
        } else {
            dest->current_block = src->current_block;
        }

        dest->total_allocated += src->total_allocated;
        src->current_block = nullptr;
        src->total_allocated = 0;
    }

    if (dest->thread_safe) arena_unlock(dest);
}

extern "C" int eshkol_thread_pool_is_worker(void) __attribute__((weak));

/**
 * @brief Report whether the calling thread is a thread-pool worker (e.g. a parallel-map worker).
 *
 * Delegates to the weakly-linked eshkol_thread_pool_is_worker() if the
 * thread-pool module is linked in; if it isn't (weak symbol resolves to
 * null), conservatively returns 0 (not a worker).
 *
 * @return Non-zero if running on a thread-pool worker thread, 0 otherwise.
 */
int arena_is_worker_thread(void) {
    if (eshkol_thread_pool_is_worker) {
        return eshkol_thread_pool_is_worker();
    }
    return 0;
}

eshkol_region_t* region_create(const char* name, size_t size_hint) {
    // ESH-0214: the eshkol_region_t control block has a fully deterministic,
    // single-owner lifetime -- created here, freed in region_destroy(), always
    // exactly once (region_pop -> region_destroy is the only caller graph).
    // It must NOT be arena-allocated: with-region is meant to be usable inside
    // a hot loop (the per-iteration-scratch-region idiom is the documented
    // workaround for unbounded interpreter-loop growth, see ESH-0214), and an
    // arena allocation here would land in whatever arena is *currently*
    // active -- typically the persistent global/REPL arena when with-region
    // is used at the top of a loop body -- permanently leaking one struct's
    // worth of bytes on every iteration for the life of the process. A plain
    // malloc/free pairs exactly with this struct's real lifetime and keeps
    // with-region's steady-state footprint at O(1) regardless of how many
    // times it is entered.
    auto* region = (eshkol_region_t*)std::malloc(sizeof(eshkol_region_t));
    if (!region) {
        eshkol_error("Failed to allocate region structure");
        return nullptr;
    }

    size_t arena_size = (size_hint > 0) ? size_hint : 8192;
    if (arena_size < 1024) arena_size = 1024;

    region->arena = arena_create(arena_size);
    if (!region->arena) {
        eshkol_error("Failed to create region arena");
        std::free(region);
        return nullptr;
    }

    if (name) {
        const size_t name_len = std::strlen(name) + 1;
        auto* name_copy = (char*)std::malloc(name_len);
        if (name_copy) {
            std::memcpy(name_copy, name, name_len);
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
    region->escape_base = nullptr;  // captured at region_push (see ESH-0214c)
    region->fwd_map = nullptr;      // lazily created at first deep escape
    region->fwd_target = nullptr;

    eshkol_debug("Created region '%s' with size hint %zu",
                 name ? name : "(anonymous)", size_hint);

    return region;
}

/**
 * @brief Free a region's arena and its malloc'd control block (name + struct).
 *
 * If the region is still marked active (still on the region stack), pops it
 * first via region_pop() (which itself calls back into this function once
 * the region is inactive) rather than freeing it out from under the stack.
 * Otherwise destroys the region's arena (releasing all memory allocated
 * within the region) and frees the malloc'd name string and the
 * eshkol_region_t struct itself, per the single-owner lifetime documented at
 * region_create() (ESH-0214).
 *
 * @param region Region to destroy (no-op if NULL).
 */
// Defined below (after the evacuator's map type) -- frees region->fwd_map.
static void region_free_fwd_map(eshkol_region_t* region);

void region_destroy(eshkol_region_t* region) {
    if (!region) return;

    if (region->is_active) {
        eshkol_warn("Destroying active region '%s' - popping from stack first",
                    region->name ? region->name : "(anonymous)");
        region_pop();
        return;
    }

    const char* name = region->name ? region->name : "(anonymous)";
    const size_t used = region->arena ? arena_get_used_memory(region->arena) : 0;
    eshkol_debug("Destroying region '%s', freeing %zu bytes", name, used);

    if (region->arena) {
        arena_destroy(region->arena);
        region->arena = nullptr;
    }

    // ESH-0214c: the deep-escape forwarding map's keys reference this region's
    // (now freed) arena; drop it with the region.
    region_free_fwd_map(region);

    // ESH-0214: region->name and the region struct itself are malloc'd (see
    // region_create) precisely so this call is a complete, bounded release --
    // no bytes belonging to this with-region activation survive it.
    std::free((void*)region->name);
    region->name = nullptr;
    std::free(region);
}

/**
 * @brief Push a region onto the calling thread's lexical region stack, activating it.
 *
 * Links @p region's parent to whatever region currently sits atop the
 * (thread-local) stack, marks it active, and pushes it. Used to implement
 * entry into `(with-region ...)`. Fails with a logged error (leaving the
 * stack unchanged) if @p region is NULL or the stack is already at
 * MAX_REGION_DEPTH.
 *
 * @param region Region to push and activate.
 */
void region_push(eshkol_region_t* region) {
    if (!region) {
        eshkol_error("Cannot push null region");
        return;
    }

    if (__region_stack_depth >= MAX_REGION_DEPTH) {
        eshkol_error("Region stack overflow (max depth: %d)", MAX_REGION_DEPTH);
        return;
    }

    region->parent = (__region_stack_depth > 0) ?
        __region_stack[__region_stack_depth - 1] : nullptr;

    // ESH-0214c: capture the arena that outlives this region NOW, before the
    // with-region codegen overwrites the __global_arena allocation slot with
    // this region's arena. At this instant that slot still holds the true
    // enclosing arena (the parent region's arena when nested, or the real
    // process/global arena at top level), which is exactly where escaping values
    // must be promoted. Prefer the parent region's own arena when nested (robust
    // even if callers push without going through with-region codegen).
    if (region->parent && region->parent->arena) {
        region->escape_base = region->parent->arena;
    } else {
        region->escape_base = get_global_arena();
    }

    region->is_active = 1;
    __region_stack[__region_stack_depth++] = region;

    eshkol_debug("Pushed region '%s' (depth: %llu)",
                 region->name ? region->name : "(anonymous)",
                 (unsigned long long)__region_stack_depth);
}

/**
 * @brief Pop the innermost active region off the calling thread's region stack and destroy it.
 *
 * Marks the popped region inactive and calls region_destroy() on it, which
 * frees its arena and control block. Used to implement exit from
 * `(with-region ...)`, including the early-exit path from region_destroy()
 * when asked to destroy a still-active region. Logs a warning and is a no-op
 * if the stack is already empty.
 */
void region_pop(void) {
    if (__region_stack_depth == 0) {
        eshkol_warn("Attempted to pop from empty region stack");
        return;
    }

    eshkol_region_t* region = __region_stack[--__region_stack_depth];
    __region_stack[__region_stack_depth] = nullptr;
    region->is_active = 0;

    eshkol_debug("Popped region '%s' (depth: %llu)",
                 region->name ? region->name : "(anonymous)",
                 (unsigned long long)__region_stack_depth);

    region_destroy(region);
}

/** @brief Return the innermost active region on the calling thread's stack, or NULL if none. */
eshkol_region_t* region_current(void) {
    if (__region_stack_depth == 0) return nullptr;
    return __region_stack[__region_stack_depth - 1];
}

/**
 * @brief Enter a region's allocation scope: redirect the shared current-arena
 *        slot to @p region's arena, but ONLY when that is thread-safe.
 *
 * Called by with-region codegen immediately after region_push. Replaces the
 * old inline codegen hijack of the __global_arena GlobalVariable so the decision
 * of whether it is safe to mutate that process-shared pointer is made here, at
 * runtime, with thread context in view:
 *
 *   • Single-threaded, non-parallel context — the common case, and every
 *     sequential program — hijacks exactly as before: body allocations land in
 *     the region arena and are freed at region_pop (full reclamation).
 *   • Worker thread, or any thread while a work-stealing parallel construct is
 *     active — does NOT touch the shared slot (returns REGION_NO_HIJACK). Body
 *     allocations stay in the thread-safe shared arena; escape promotion still
 *     copies any value that DID land in a region arena out at region_pop, so
 *     results are correct. This is what makes parallel-map + with-region safe.
 *
 * @param region Region being entered (its arena is the hijack target).
 * @return The arena that was displaced (to be restored by eshkol_region_leave),
 *         or the REGION_NO_HIJACK sentinel if the slot was left untouched.
 */
extern "C" arena_t* eshkol_region_enter(eshkol_region_t* region) {
    if (!region || !region->arena) return REGION_NO_HIJACK;

    // Unsafe to mutate the process-shared __global_arena when other threads may
    // be reading it concurrently: on a pool worker, or while any work-stealing
    // construct is in flight (which also covers the main thread running the
    // parallel-map JIT-warmup item while workers spin up).
    if (s_parallel_depth.load(std::memory_order_acquire) != 0 ||
        arena_is_worker_thread()) {
        return REGION_NO_HIJACK;
    }

    arena_t* saved = __global_arena;
    __global_arena = region->arena;
    return saved;
}

/**
 * @brief Leave a region's allocation scope: undo the redirect performed by
 *        eshkol_region_enter (no-op if it declined to hijack).
 *
 * Called by with-region codegen after region_pop.
 *
 * @param saved The value returned by the matching eshkol_region_enter.
 */
extern "C" void eshkol_region_leave(arena_t* saved) {
    if (saved == REGION_NO_HIJACK) return;
    __global_arena = saved;
}

/**
 * @brief Open a parallel scope around a work-stealing construct
 *        (parallel-map/fold/execute/filter/for-each and async futures).
 *
 * While the scope is open, region hijacks are suppressed (eshkol_region_enter
 * returns without touching the shared slot) and the shared current-arena slot is
 * forced to the true thread-safe process arena (s_shared_root_arena) so every
 * worker that reads __global_arena allocates into a lockable arena rather than
 * into whatever region arena a with-region on the spawning thread may have
 * hijacked it to. The displaced value (e.g. an enclosing region's arena) is
 * captured on the 0->1 edge and restored on the 1->0 edge; nested/overlapping
 * scopes just bump the depth. Idempotent w.r.t. the arena value because every
 * scope forces the same root.
 */
extern "C" void eshkol_parallel_scope_begin(void) {
    std::lock_guard<std::mutex> lk(s_parallel_arena_mtx);
    if (s_parallel_depth.fetch_add(1, std::memory_order_acq_rel) == 0) {
        arena_t* root = s_shared_root_arena;
        if (!root) root = get_global_arena_shared();  // ensures init + captures root
        s_saved_arena_at_parallel = __global_arena;
        if (root) __global_arena = root;
    }
}

/**
 * @brief Close a parallel scope opened by eshkol_parallel_scope_begin(),
 *        restoring the current-arena slot displaced on the 0->1 edge once the
 *        last overlapping scope closes.
 */
extern "C" void eshkol_parallel_scope_end(void) {
    std::lock_guard<std::mutex> lk(s_parallel_arena_mtx);
    if (s_parallel_depth.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        __global_arena = s_saved_arena_at_parallel;
        s_saved_arena_at_parallel = nullptr;
    }
}

/**
 * @brief Allocate @p size bytes from the innermost active region, or the global arena if none.
 *
 * @param size Number of bytes to allocate.
 * @return     Pointer into the current region's arena if a region is
 *             active, else into the result of get_global_arena().
 */
void* region_allocate(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate(region->arena, size);
    }
    return arena_allocate(get_global_arena(), size);
}

/**
 * @brief Allocate @p size bytes at @p alignment from the innermost active region, or the global arena if none.
 *
 * @param size      Number of bytes to allocate.
 * @param alignment Required alignment in bytes.
 * @return          Aligned pointer into the current region's arena if a
 *                  region is active, else into get_global_arena().
 */
void* region_allocate_aligned(size_t size, size_t alignment) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_aligned(region->arena, size, alignment);
    }
    return arena_allocate_aligned(get_global_arena(), size, alignment);
}

/**
 * @brief Allocate @p size zero-initialized bytes from the innermost active region, or the global arena if none.
 *
 * @param size Number of bytes to allocate.
 * @return     Zeroed pointer into the current region's arena if a region is
 *             active, else into get_global_arena().
 */
void* region_allocate_zeroed(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_zeroed(region->arena, size);
    }
    return arena_allocate_zeroed(get_global_arena(), size);
}

/**
 * @brief Allocate a single tagged cons cell from the innermost active region, or the global arena if none.
 *
 * @return New cons cell (car/cdr initialized to NULL) allocated in the
 *         current region's arena if a region is active, else in
 *         get_global_arena().
 */
arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_tagged_cons_cell(region->arena);
    }
    return arena_allocate_tagged_cons_cell(get_global_arena());
}

/** @brief Return the number of bytes currently allocated (used) in @p region's arena, or 0 if none. */
size_t region_get_used_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_used_memory(region->arena);
}

/** @brief Return the total number of bytes reserved (allocated from the OS/backing store) in @p region's arena, or 0 if none. */
size_t region_get_total_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_total_memory(region->arena);
}

/** @brief Return @p region's name (as given to region_create()), or NULL if @p region is NULL or unnamed. */
const char* region_get_name(const eshkol_region_t* region) {
    if (!region) return nullptr;
    return region->name;
}

/** @brief Return the number of regions currently active on the calling thread's region stack. */
uint64_t region_get_depth(void) {
    return __region_stack_depth;
}

/**
 * @brief Determine which arena an escaping value from @p current should be copied into.
 *
 * Escaping a value means copying it out of the current (about-to-be-freed)
 * region's arena into a longer-lived arena. That target is the immediately
 * enclosing region's arena, if there is a parent region, or otherwise the
 * process/thread global arena (so a value escaping the outermost region
 * survives as long as ordinary heap allocations).
 *
 * @param current Region the value is escaping from (must be non-NULL).
 * @return        Arena the escaped copy should be allocated in.
 */
static arena_t* region_escape_target(eshkol_region_t* current) {
    // ESH-0214c: escape_base is captured at region_push and is the arena that
    // genuinely outlives this region. It must be used instead of
    // get_global_arena(), which during a with-region body has been hijacked by
    // codegen to point at THIS region's arena (so escaping through it would copy
    // the value straight back into the arena about to be freed).
    if (current->escape_base) {
        return current->escape_base;
    }
    if (current->parent && current->parent->arena) {
        return current->parent->arena;
    }
    return get_global_arena();
}

/**
 * @brief Copy @p size bytes at @p ptr out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p ptr unchanged (nothing to
 * escape). Otherwise allocates @p size bytes (8-byte aligned) in
 * region_escape_target()'s arena, memcpy's the bytes over, increments the
 * current region's escape_count, and returns the new copy. Used so a value
 * created inside a `(with-region ...)` block can safely be returned/escaped
 * before the region's arena is destroyed.
 *
 * @param ptr  Source bytes to copy (must be non-NULL).
 * @param size Number of bytes to copy (must be > 0).
 * @return     Pointer to the escaped copy, @p ptr unchanged if no region is
 *             active, or NULL on allocation failure or invalid arguments.
 */
void* region_escape(const void* ptr, size_t size) {
    if (!ptr || size == 0) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (void*)ptr;

    void* copy = arena_allocate_aligned(region_escape_target(current), size, 8);
    if (!copy) {
        eshkol_error("region_escape: failed to allocate %zu bytes in target arena", size);
        return nullptr;
    }

    std::memcpy(copy, ptr, size);
    current->escape_count++;
    return copy;
}

/**
 * @brief Copy a NUL-terminated string out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p str unchanged. Otherwise
 * allocates a header-prefixed string object of the right length in
 * region_escape_target()'s arena, copies the bytes plus a NUL terminator,
 * increments the current region's escape_count, and returns the new string.
 *
 * @param str Source NUL-terminated string (must be non-NULL).
 * @return    Pointer to the escaped copy, @p str unchanged if no region is
 *            active, or NULL on allocation failure.
 */
void* region_escape_string(const char* str) {
    if (!str) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (void*)str;

    const size_t len = std::strlen(str);
    auto* copy = (char*)arena_allocate_string_with_header(region_escape_target(current), len);
    if (!copy) {
        eshkol_error("region_escape_string: failed to allocate string of length %zu", len);
        return nullptr;
    }

    std::memcpy(copy, str, len);
    copy[len] = '\0';
    current->escape_count++;
    return copy;
}

/**
 * @brief Copy a single tagged cons cell out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p cell unchanged. Otherwise
 * allocates a new cons cell in region_escape_target()'s arena, shallow-copies
 * car and cdr (tagged values, so this does not recursively escape anything
 * the cell's car/cdr point to), increments the current region's
 * escape_count, and returns the new cell.
 *
 * @param cell Source cons cell (must be non-NULL).
 * @return     Pointer to the escaped copy, @p cell unchanged if no region is
 *             active, or NULL on allocation failure.
 */
arena_tagged_cons_cell_t* region_escape_tagged_cons_cell(const arena_tagged_cons_cell_t* cell) {
    if (!cell) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (arena_tagged_cons_cell_t*)cell;

    arena_tagged_cons_cell_t* copy =
        arena_allocate_tagged_cons_cell(region_escape_target(current));
    if (!copy) {
        eshkol_error("region_escape_tagged_cons_cell: failed to allocate");
        return nullptr;
    }

    copy->car = cell->car;
    copy->cdr = cell->cdr;
    current->escape_count++;
    return copy;
}

// ───────────────────────────────────────────────────────────────────────────
// DEEP TRANSITIVE ESCAPE PROMOTION (evacuation) — ESH-0214c
//
// A region's arena is torn down at region_pop. A value that escapes the region
// (returned as the with-region body result, or stored by mutation into an outer
// container) must have its ENTIRE reachable subgraph that lives in the dying
// region copied out first, otherwise the escaped object keeps interior pointers
// (car/cdr, vector slots, captured env, hash key/value arrays, tensor buffers)
// aimed into freed memory -> "car/cdr: argument is not a pair" corruption.
//
// The evacuator is a Cheney-style copying collector restricted to the escaping
// subgraph: a forwarding map (old data-ptr -> new data-ptr) preserves shared
// structure and terminates on cycles, and a worklist drives a breadth-first
// walk so arbitrarily deep lists do not overflow the native stack.
//
// A node is COPIED iff it lives in an active region strictly INNER than the
// destination's owning region (`boundary_idx`); anything at-or-outside the
// boundary (an ancestor region, the global arena, static/stack storage) already
// outlives the destination and is left in place (forwarding identity). This is
// correct for nested regions: escaping to the global arena copies everything in
// any active region; escaping to region D copies only what lives in D's
// descendants.
// ───────────────────────────────────────────────────────────────────────────

// Walk an arena's block chain and test raw containment. Region arenas are
// non-thread-safe single-owner arenas (region_create uses arena_create), so no
// locking is required here.
static bool arena_contains_ptr(const arena_t* a, const void* p) {
    if (!a || !p) return false;
    const uint8_t* q = (const uint8_t*)p;
    for (const arena_block_t* b = a->current_block; b; b = b->next) {
        if (q >= b->memory && q < b->memory + b->used) return true;
    }
    return false;
}

// Index (into __region_stack) of the innermost active region whose arena
// contains @p p, or -1 if @p p is not inside any active region arena (i.e. it
// lives in the global/thread arena or in static/stack storage).
static int region_index_owning(const void* p) {
    for (int i = (int)__region_stack_depth - 1; i >= 0; --i) {
        eshkol_region_t* r = __region_stack[i];
        if (r && r->arena && arena_contains_ptr(r->arena, p)) return i;
    }
    return -1;
}

namespace {

// How to traverse a copied object's interior after the contiguous header+payload
// copy. EVAC_LEAF objects are self-contained (no interior region pointers) and
// need no further work.
enum EvacKind : uint8_t {
    EVAC_LEAF = 0,
    EVAC_CONS,        // arena_tagged_cons_cell_t: car + cdr tagged values
    EVAC_VECTOR,      // [i64 length][length tagged values]  (also records)
    EVAC_MULTIVALUE,  // [size_t count][count tagged values]
    EVAC_HASH,        // eshkol_hash_table_t: keys/values/status arrays + home_arena
    EVAC_TENSOR,      // eshkol_tensor_t: dimensions + elements raw buffers
    EVAC_EXCEPTION,   // eshkol_exception_t: message/filename/irritants
    EVAC_CLOSURE,     // eshkol_closure_t: captured environment
};

using EvacFwdMap = std::unordered_map<const void*, void*>;

struct EvacState {
    arena_t* target;
    int boundary_idx;
    EvacFwdMap* fwd;   // persistent per-region map (see eshkol_region_t::fwd_map)
    std::vector<std::pair<void*, EvacKind>> worklist;
    size_t copies = 0;
};

} // namespace

// Free a region's persistent deep-escape forwarding map (declared above
// region_destroy; the map type is only visible from here down).
static void region_free_fwd_map(eshkol_region_t* region) {
    if (region && region->fwd_map) {
        delete (EvacFwdMap*)region->fwd_map;
        region->fwd_map = nullptr;
        region->fwd_target = nullptr;
    }
}

// Classify an object (given its live original data pointer and the tagged value
// referencing it) into an EvacKind. Ports are never deep-traversed (they wrap OS
// resources / fds); they are leaf-copied with care so the escaped port struct is
// stable, but the underlying handle is intentionally shared, not duplicated.
static EvacKind evac_kind_for(const eshkol_tagged_value_t& v, const void* old_data) {
    const uint8_t type = v.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    if (is_port) return EVAC_LEAF;

    const auto* h = (const eshkol_object_header_t*)
        ((const uint8_t*)old_data - sizeof(eshkol_object_header_t));
    const uint8_t sub = h->subtype;

    if (ESHKOL_IS_ANY_CALLABLE_TYPE(type)) {
        if (sub == CALLABLE_SUBTYPE_CLOSURE) return EVAC_CLOSURE;
        // LAMBDA_SEXPR / AD_NODE / PRIMITIVE / CONTINUATION: their interior
        // reference graph is not confidently traversable here and they almost
        // never escape a region via mutation. Kept shallow (documented).
        return EVAC_LEAF;
    }

    switch (sub) {
        case HEAP_SUBTYPE_CONS:        return EVAC_CONS;
        case HEAP_SUBTYPE_VECTOR:      return EVAC_VECTOR;   // records are vectors too
        case HEAP_SUBTYPE_MULTI_VALUE: return EVAC_MULTIVALUE;
        case HEAP_SUBTYPE_HASH:        return EVAC_HASH;
        case HEAP_SUBTYPE_TENSOR:      return EVAC_TENSOR;
        case HEAP_SUBTYPE_EXCEPTION:   return EVAC_EXCEPTION;
        // STRING / SYMBOL / BIGNUM / RATIONAL / BYTEVECTOR: self-contained
        // payloads -> a contiguous leaf copy fully preserves them.
        //
        // SUBSTITUTION / FACT / KNOWLEDGE_BASE / FACTOR_GRAPH / WORKSPACE /
        // PROMISE / DNC / SDNC / TAYLOR(rational-coeff): may carry interior
        // tagged values, but (a) they are exceedingly rare to escape a region by
        // mutation and (b) their internal layouts are not traversed here. They
        // fall through to a shallow leaf copy — a documented limitation, not a
        // regression (this matches the pre-ESH-0214c behavior for every subtype).
        default:                       return EVAC_LEAF;
    }
}

// Copy a headerless raw buffer (closure env, hash arrays, tensor buffers,
// C strings) into the target arena, with forwarding so shared/aliased buffers
// are copied once.
static void* evac_raw(EvacState& st, const void* old, size_t size) {
    if (!old || size == 0) return (void*)old;
    auto it = st.fwd->find(old);
    if (it != st.fwd->end()) return it->second;
    void* raw = arena_allocate_aligned(st.target, size, 16);
    if (!raw) return (void*)old;
    std::memcpy(raw, old, size);
    (*st.fwd)[old] = raw;
    st.copies++;
    return raw;
}

// Copy a header-prefixed object into the target arena (contiguous header+payload),
// register forwarding, and enqueue it for interior traversal if non-leaf.
static void* evac_object(EvacState& st, void* old_data, const eshkol_tagged_value_t& v) {
    auto it = st.fwd->find(old_data);
    if (it != st.fwd->end()) return it->second;

    auto* h = (eshkol_object_header_t*)((uint8_t*)old_data - sizeof(eshkol_object_header_t));
    const size_t total = sizeof(eshkol_object_header_t) + h->size;
    if (h->size == 0) return old_data;  // nothing to copy; leave in place
    // Plausibility guard: every live producer of tagged heap values allocates
    // through the *_with_header paths, but a handful of legacy helpers
    // (arena_allocate_tagged_cons_cell, arena_create_*_cons) emit HEADERLESS
    // cells; if one ever leaked into a traversed graph, the 8 bytes before it
    // are arbitrary. A garbage size would make the memcpy below read/allocate
    // wildly; cap it at the region's own footprint upper bound. Skipping the
    // copy degrades to the pre-ESH-0214c shallow behavior for that node only.
    if (h->size > (uint32_t)0x10000000u) {  // 256MB: far above any real object
        eshkol_warn("region evacuate: implausible object size %u at %p; "
                    "leaving in place (headerless allocation?)",
                    (unsigned)h->size, old_data);
        return old_data;
    }

    void* raw = arena_allocate_aligned(st.target, total, 16);
    if (!raw) {
        eshkol_error("region evacuate: failed to allocate %zu bytes", total);
        return old_data;
    }
    std::memcpy(raw, h, total);
    void* new_data = (uint8_t*)raw + sizeof(eshkol_object_header_t);
    (*st.fwd)[old_data] = new_data;
    st.copies++;

    EvacKind k = evac_kind_for(v, old_data);
    if (k != EVAC_LEAF) st.worklist.push_back({new_data, k});
    return new_data;
}

// Rewrite one tagged value: if it points into a dying region (strictly inner
// than the boundary), evacuate the pointed-to object and repoint; otherwise
// leave it untouched.
static eshkol_tagged_value_t evac_value(EvacState& st, eshkol_tagged_value_t v) {
    const uint8_t type = v.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    const bool is_heap = ESHKOL_IS_ANY_PTR_TYPE(type) || is_port;
    if (!is_heap) return v;

    void* p = (void*)(uintptr_t)v.data.ptr_val;
    if (!p) return v;
    if (region_index_owning(p) <= st.boundary_idx) return v;  // stable relative to dst

    void* np = evac_object(st, p, v);
    v.data.ptr_val = (uint64_t)(uintptr_t)np;
    return v;
}

// Drive the deep evacuation of @p val into @p target, copying everything
// reachable that lives in an active region strictly inner than @p boundary_idx.
static eshkol_tagged_value_t region_evacuate_value(eshkol_tagged_value_t val,
                                                   arena_t* target,
                                                   int boundary_idx) {
    if (!target) return val;
    EvacState st;
    st.target = target;
    st.boundary_idx = boundary_idx;

    // Use the innermost active region's PERSISTENT forwarding map so shared
    // structure is preserved across SEPARATE escapes within one region body
    // (two vector-set! calls storing lists that share a tail must keep sharing
    // it after promotion). The map's keys reference memory in this region or an
    // enclosing one -- the innermost region is the first of those to die, and
    // region_destroy frees the map with it, so no entry can ever go stale. The
    // map is only valid for one promotion target; if a nested escape targets a
    // different arena (rare: barrier into an intermediate region), reset it.
    EvacFwdMap local_fwd;
    eshkol_region_t* owner = region_current();
    if (owner) {
        if (!owner->fwd_map) {
            owner->fwd_map = new (std::nothrow) EvacFwdMap();
            owner->fwd_target = target;
        } else if (owner->fwd_target != target) {
            ((EvacFwdMap*)owner->fwd_map)->clear();
            owner->fwd_target = target;
        }
        st.fwd = owner->fwd_map ? (EvacFwdMap*)owner->fwd_map : &local_fwd;
    } else {
        st.fwd = &local_fwd;
    }

    eshkol_tagged_value_t root = evac_value(st, val);

    while (!st.worklist.empty()) {
        std::pair<void*, EvacKind> item = st.worklist.back();
        st.worklist.pop_back();
        void* nd = item.first;
        switch (item.second) {
            case EVAC_CONS: {
                auto* c = (arena_tagged_cons_cell_t*)nd;
                c->car = evac_value(st, c->car);
                c->cdr = evac_value(st, c->cdr);
                break;
            }
            case EVAC_VECTOR: {
                int64_t len = *(int64_t*)nd;
                if (len < 0) len = 0;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(int64_t));
                for (int64_t i = 0; i < len; ++i) elems[i] = evac_value(st, elems[i]);
                break;
            }
            case EVAC_MULTIVALUE: {
                size_t count = *(size_t*)nd;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(size_t));
                for (size_t i = 0; i < count; ++i) elems[i] = evac_value(st, elems[i]);
                break;
            }
            case EVAC_EXCEPTION: {
                auto* ex = (eshkol_exception_t*)nd;
                if (ex->message && region_index_owning(ex->message) > st.boundary_idx)
                    ex->message = (char*)evac_raw(st, ex->message, std::strlen(ex->message) + 1);
                if (ex->filename && region_index_owning(ex->filename) > st.boundary_idx)
                    ex->filename = (char*)evac_raw(st, ex->filename, std::strlen(ex->filename) + 1);
                if (ex->irritants && ex->num_irritants &&
                    region_index_owning(ex->irritants) > st.boundary_idx) {
                    ex->irritants = (eshkol_tagged_value_t*)evac_raw(
                        st, ex->irritants, (size_t)ex->num_irritants * sizeof(eshkol_tagged_value_t));
                }
                if (ex->irritants) {
                    for (uint32_t i = 0; i < ex->num_irritants; ++i)
                        ex->irritants[i] = evac_value(st, ex->irritants[i]);
                }
                break;
            }
            case EVAC_TENSOR: {
                auto* t = (eshkol_tensor_t*)nd;
                if (t->dimensions && region_index_owning(t->dimensions) > st.boundary_idx)
                    t->dimensions = (uint64_t*)evac_raw(
                        st, t->dimensions, (size_t)t->num_dimensions * sizeof(uint64_t));
                if (t->elements && region_index_owning(t->elements) > st.boundary_idx)
                    t->elements = (int64_t*)evac_raw(
                        st, t->elements, (size_t)t->total_elements * sizeof(int64_t));
                break;
            }
            case EVAC_HASH: {
                auto* tbl = (eshkol_hash_table_t*)nd;
                const size_t cap = tbl->capacity;
                if (tbl->keys && region_index_owning(tbl->keys) > st.boundary_idx)
                    tbl->keys = (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->keys, cap * sizeof(eshkol_tagged_value_t));
                if (tbl->values && region_index_owning(tbl->values) > st.boundary_idx)
                    tbl->values = (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->values, cap * sizeof(eshkol_tagged_value_t));
                if (tbl->status && region_index_owning(tbl->status) > st.boundary_idx)
                    tbl->status = (uint8_t*)evac_raw(st, tbl->status, cap * sizeof(uint8_t));
                if (tbl->keys && tbl->values && tbl->status) {
                    for (size_t i = 0; i < cap; ++i) {
                        if (tbl->status[i] == HASH_ENTRY_OCCUPIED) {
                            tbl->keys[i]   = evac_value(st, tbl->keys[i]);
                            tbl->values[i] = evac_value(st, tbl->values[i]);
                        }
                    }
                }
                // Future resizes must grow the arrays in the surviving arena,
                // never the dying region.
                tbl->home_arena = st.target;
                break;
            }
            case EVAC_CLOSURE: {
                auto* c = (eshkol_closure_t*)nd;
                if (c->env && region_index_owning(c->env) > st.boundary_idx) {
                    const size_t ncap = CLOSURE_ENV_GET_NUM_CAPTURES(c->env->num_captures);
                    const size_t env_size =
                        sizeof(eshkol_closure_env_t) + ncap * sizeof(eshkol_tagged_value_t);
                    auto* ne = (eshkol_closure_env_t*)evac_raw(st, c->env, env_size);
                    c->env = ne;
                    for (size_t i = 0; i < ncap; ++i) {
                        eshkol_tagged_value_t& cap = ne->captures[i];
                        // MUTABLE-CAPTURE CELLS: lambda codegen moves a set!-able
                        // captured local into a 16-byte headerless arena cell
                        // (one eshkol_tagged_value_t) and stores the CELL ADDRESS
                        // in the capture slot packed as an exact INT64
                        // (packInt64ToTaggedValue(PtrToInt(cell))) — see
                        // "CLOSURE ESCAPE FIX" / "MUTABLE CAPTURE FIX" in
                        // llvm_codegen.cpp. During a with-region body that cell
                        // lives in the region arena and would dangle after
                        // region_pop. There is no type tag distinguishing this
                        // packed pointer from a genuine integer, so we test
                        // conservatively: an INT64 capture whose value, read as
                        // a pointer, falls INSIDE a dying region arena is treated
                        // as a capture cell — copied (forwarded, so cells shared
                        // between closures stay shared), its contained tagged
                        // value evacuated, and the slot repointed. A genuine
                        // captured integer can only misfire if it exactly equals
                        // a live interior address of the few-KB region arena
                        // active at escape time (Boehm-style conservatism;
                        // astronomically unlikely, documented in the PR). The
                        // other producers of INT64-packed pointers in capture
                        // slots are GlobalVariable addresses and JIT/AOT code
                        // addresses, neither of which is ever inside a region
                        // arena, so they always take the identity path here.
                        if (cap.type == ESHKOL_VALUE_INT64 && cap.data.int_val != 0) {
                            void* cell = (void*)(uintptr_t)cap.data.int_val;
                            if (region_index_owning(cell) > st.boundary_idx) {
                                auto* nc =
                                    (eshkol_tagged_value_t*)evac_raw(st, cell, 16);
                                if (nc != cell) {
                                    *nc = evac_value(st, *nc);
                                    cap.data.int_val = (int64_t)(intptr_t)nc;
                                }
                                continue;
                            }
                        }
                        cap = evac_value(st, cap);
                    }
                }
                if (c->name && region_index_owning(c->name) > st.boundary_idx)
                    c->name = (const char*)evac_raw(st, c->name, std::strlen(c->name) + 1);
                break;
            }
            case EVAC_LEAF:
            default:
                break;
        }
    }

    eshkol_region_t* cur = region_current();
    if (cur) cur->escape_count += st.copies;
    return root;
}

/**
 * @brief Copy a tagged value's heap payload (and its whole in-region subgraph)
 *        out of the current region.
 *
 * Non-heap values (ints, doubles, etc.) are returned unchanged, as are heap
 * values that are NULL. Otherwise the value's reachable subgraph that lives in
 * the current (about-to-be-destroyed) region's arena is deep-copied into
 * region_escape_target()'s arena via region_evacuate_value(): shared structure
 * and cycles are preserved by a forwarding map, and objects already living in an
 * enclosing region or the global arena are left in place. If no region is
 * currently active, the value is returned unchanged. Shared implementation
 * behind the two `region_escape_tagged_value*` extern "C" entry points.
 *
 * @param val Tagged value to escape (passed and returned by value).
 * @return    Escaped value (with an updated heap pointer if a copy was
 *            made), or @p val unchanged if escaping wasn't needed/possible.
 */
static eshkol_tagged_value_t region_escape_tagged_value_impl(eshkol_tagged_value_t val) {
    const uint8_t type = val.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    const bool is_heap = ESHKOL_IS_ANY_PTR_TYPE(type) || is_port;

    if (!is_heap) return val;

    eshkol_region_t* current = region_current();
    if (!current) return val;

    void* ptr = (void*)(uintptr_t)val.data.ptr_val;
    if (!ptr) return val;

    // Escaping FROM `current` (stack index depth-1) into its parent (or the
    // global arena). Everything living in `current`'s arena must be copied;
    // anything already in an ancestor region or the global arena is stable.
    // boundary = index of current's parent = depth-2.
    arena_t* target = region_escape_target(current);
    const int boundary = (int)__region_stack_depth - 2;
    return region_evacuate_value(val, target, boundary);
}

/**
 * @brief Codegen-facing entry point: escape a tagged value's heap payload out of the current region, returned by value.
 *
 * @param val Tagged value to escape.
 * @return    See region_escape_tagged_value_impl().
 */
extern "C" eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val) {
    return region_escape_tagged_value_impl(val);
}

/**
 * @brief Codegen-facing entry point: escape a tagged value's heap payload out of the current region, returned via out-pointer.
 *
 * Output-pointer form of region_escape_tagged_value(), used where an sret
 * ABI is preferred over a returned struct. If @p val is NULL, writes an
 * all-zero (NULL-typed) tagged value to *out.
 *
 * @param out Destination for the escaped value (no-op if NULL).
 * @param val Tagged value to escape (may be NULL).
 */
extern "C" void region_escape_tagged_value_into(eshkol_tagged_value_t* out,
                                                const eshkol_tagged_value_t* val) {
    if (!out) return;
    if (!val) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    *out = region_escape_tagged_value_impl(*val);
}

/**
 * @brief Region write barrier: promote @p value's in-region subgraph when it is
 *        stored into a longer-lived destination (ESH-0214c).
 *
 * Called by codegen at every mutation channel that can store a value into a
 * location that outlives the value's region (set-car!/set-cdr!, vector-set!,
 * hash-table-set!, set! of a global). The FAST PATH — no active region — is a
 * single thread-local load and branch, so the barrier is essentially free
 * outside `(with-region ...)`.
 *
 * When a region is active, the barrier evacuates @p value's reachable subgraph
 * out of any region strictly inner than @p dst's owning region iff @p value
 * actually points into such a region. If @p value already lives at least as long
 * as @p dst (same or an enclosing region, or the global arena), it is stored
 * unchanged. The evacuation preserves shared structure and cycles via a
 * forwarding map, so this is safe to apply on every store.
 *
 * @param out   Destination for the (possibly promoted) value to actually store.
 * @param dst   Address of the container/slot being written (used only to locate
 *              which region owns the destination); NULL is treated as
 *              global/outer.
 * @param value The value about to be stored.
 */
extern "C" void eshkol_region_write_barrier_into(eshkol_tagged_value_t* out,
                                                 const void* dst,
                                                 const eshkol_tagged_value_t* value) {
    if (!out) return;
    if (!value) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    eshkol_tagged_value_t v = *value;

    // FAST PATH: no active region -> nothing can dangle.
    if (__region_stack_depth == 0) { *out = v; return; }

    const uint8_t type = v.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    const bool is_heap = ESHKOL_IS_ANY_PTR_TYPE(type) || is_port;
    if (!is_heap) { *out = v; return; }

    void* vptr = (void*)(uintptr_t)v.data.ptr_val;
    if (!vptr) { *out = v; return; }

    const int val_idx = region_index_owning(vptr);
    if (val_idx < 0) { *out = v; return; }          // value already outer/global

    const int dst_idx = region_index_owning(dst);   // -1 when dst is outer/global
    if (val_idx <= dst_idx) { *out = v; return; }    // value outlives-or-coeval with dst

    // Where the promoted subgraph lands: the destination's own region arena
    // when dst lives in an active region, otherwise the TRUE global arena. The
    // outermost region's escape_base is that true global (captured before the
    // __global_arena slot was hijacked by with-region codegen), so use it rather
    // than get_global_arena(), which is hijacked during a region body.
    arena_t* target = (dst_idx >= 0) ? __region_stack[dst_idx]->arena
                                     : __region_stack[0]->escape_base;
    if (!target) target = get_global_arena();
    *out = region_evacuate_value(v, target, dst_idx);
}

/**
 * @brief Range form of the region write barrier: fix up @p n tagged slots at
 *        @p slots after a bulk copy (vector-copy!) into destination @p dst.
 *
 * Called by codegen after the memmove of `(vector-copy! to at from ...)`. The
 * fast path (no active region) is a single thread-local load + branch. When a
 * region is active, each copied slot that points into a region strictly inner
 * than @p dst's owning region is deep-promoted in place (same semantics as
 * eshkol_region_write_barrier_into, applied slot-by-slot; the per-region
 * forwarding map keeps shared structure shared across the whole range).
 *
 * @param dst   Address of the destination container (region-ownership probe).
 * @param slots First copied slot (already holding the copied values).
 * @param n     Number of copied slots.
 */
extern "C" void eshkol_region_write_barrier_range(const void* dst,
                                                  eshkol_tagged_value_t* slots,
                                                  uint64_t n) {
    if (__region_stack_depth == 0 || !slots) return;
    for (uint64_t i = 0; i < n; ++i) {
        eshkol_region_write_barrier_into(&slots[i], dst, &slots[i]);
    }
}
