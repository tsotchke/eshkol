/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime region and thread-local arena helpers: lifecycle.
 *
 * This unit owns the process/global arena selection ABI, per-worker
 * thread-local arena lifecycle, and ownership-aware lexical region stack.
 * Raw arena block/scope mechanics remain in arena_memory.cpp.  Escape and
 * evacuation live in runtime_regions_evacuation.cpp, user region handles in
 * runtime_regions_handles.cpp and resident owners in
 * runtime_regions_resident.cpp; runtime_regions_internal.h is the interface
 * the four units share.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/exhaustive_dispatch.h"
#include "../../inc/eshkol/core/logic.h"       // eshkol_substitution_t / _fact_t / _knowledge_base_t
#include "../../inc/eshkol/core/inference.h"   // eshkol_factor_graph_t / _factor_t
#include "../../inc/eshkol/core/workspace.h"   // eshkol_workspace_t / _workspace_module_t
#include "../../inc/eshkol/core/rational.h"    // eshkol_rational_t (bignum-backed exact rationals)
#include "../../inc/eshkol/core/coroutine.h"   // eshkol_exec_context_t

#include <cstring>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>
#include <utility>
#include <atomic>
#include <mutex>
#include <array>

#include "runtime_regions_internal.h"

using namespace eshkol::regions_internal;

void eshkol_arena_global_once(void (*init)(void));

// Defined in lib/core/runtime_arena_diagnostics_hosted.cpp (ESHKOL_RUNTIME_HOSTED_SRC).
// Forward-declared rather than called through a header because this
// translation unit is ESHKOL_RUNTIME_CORE_SRC and must not itself read the
// environment or touch stdio -- the same reason eshkol_arena_poison_enabled()
// (also defined in that hosted file) is forward-declared locally further down
// in this file instead of included from a header.
extern "C" void eshkol_region_pin_notice(void);

// Thread-local region stack (safe for parallel-map + with-region).
thread_local eshkol_region_t* __region_stack[MAX_REGION_DEPTH] = {nullptr};
thread_local uint64_t __region_stack_depth = 0;

// Default global arena for allocations outside any region.
// Non-static to allow JIT code to access it directly. Weak where object format
// support lets generated standalone code override it.
ESHKOL_RUNTIME_WEAK arena_t* __global_arena = nullptr;

static thread_local arena_t* __thread_local_arena = nullptr;

// ─────────────────────────────────────────────────────────────────────────────
// OALR Phase A: per-thread memory context (ADR-0001).
//
// The memctx makes the "current allocation arena" a thread-local property reached
// through eshkol_current_arena(), instead of a direct read of the shared
// __global_arena global. `with-region` now updates this thread-local domain
// (mirrored into __global_arena in the single-threaded safe case only, see
// eshkol_region_enter) so allocation routing is thread-local. The full ABI-v2
// memctx (region_top/residence/resident txn) and the object-header change are
// deferred to later phases.
namespace eshkol::regions_internal {
thread_local eshkol_memctx_t t_memctx = {
    ESHKOL_MEMORY_ABI_PHASE_A, 0u, 0ull, nullptr, nullptr
};
} // namespace eshkol::regions_internal
static std::atomic<uint64_t> s_next_thread_id{1};

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
namespace eshkol::regions_internal {
std::atomic<int> s_parallel_depth{0};                  // >0 while a work-stealing construct may run
} // namespace eshkol::regions_internal
static std::mutex s_parallel_arena_mtx;                // guards the 0<->1 transition swap

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
 * @brief Return the calling thread's memory context (OALR Phase A).
 *
 * Never NULL. The context is a thread_local with static storage duration, so no
 * allocation occurs; the thread_id is assigned lazily (for diagnostics) on first
 * access. See ADR-0001 §1.
 */
extern "C" eshkol_memctx_t* eshkol_memctx_current(void) {
    if (t_memctx.thread_id == 0) {
        t_memctx.thread_id = s_next_thread_id.fetch_add(1, std::memory_order_relaxed);
        t_memctx.abi_version = ESHKOL_MEMORY_ABI_PHASE_A;
    }
    return &t_memctx;
}

/**
 * @brief Return the arena the calling thread's allocations should currently
 *        target — the OALR Phase A allocation accessor.
 *
 * Generated code and runtime helpers call this instead of loading __global_arena
 * directly, so that `with-region` can redirect allocation by updating the
 * thread-local memory context rather than writing the shared global.
 *
 * Ordering / #217 preservation:
 *   • While any work-stealing construct is active (s_parallel_depth != 0) this
 *     returns __global_arena directly, exactly matching the parallel-scope guard
 *     that pins the shared slot to the thread-safe process arena. This keeps the
 *     accessor in agreement with any not-yet-migrated direct __global_arena read
 *     and stops the spawning thread from allocating into a region arena while
 *     workers run.
 *   • Otherwise it returns the thread-local allocation domain (set by
 *     eshkol_region_enter for a `with-region` body), or __global_arena when no
 *     region is active.
 */
extern "C" arena_t* eshkol_current_arena(void) {
    eshkol_arena_global_once(init_global_arena_internal);
    if (s_parallel_depth.load(std::memory_order_acquire) != 0) {
        return __global_arena;
    }
    arena_t* domain = t_memctx.allocation_domain;
    if (domain) return domain;
    // Outside a region the accessor must resolve to EXACTLY the arena a direct
    // read of the codegen "current arena" slot (the __global_arena GlobalVariable)
    // would. getArenaPtr() sites (which call this accessor) are paired with the
    // many codegen sites that still load __global_arena inline — e.g. a loop's
    // per-iteration arena_push_scope reads this accessor while the same loop's
    // body allocations read __global_arena directly. Returning anything else here
    // (e.g. a thread-local arena) would desynchronize those paired reads and
    // defeat loop reclamation (ESH-0214b). Mirror the raw slot exactly.
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

#if defined(_WIN32)
// The native Windows runtime always links ESHKOL_RUNTIME_HOSTED_SRC, which
// contains thread_pool.cpp. PE/COFF has no ELF-style weak function reference,
// so use the required hosted-runtime symbol directly on Windows.
extern "C" int eshkol_thread_pool_is_worker(void);
#else
extern "C" int eshkol_thread_pool_is_worker(void) __attribute__((weak));
#endif

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
#if defined(_WIN32)
    return eshkol_thread_pool_is_worker();
#else
    if (eshkol_thread_pool_is_worker) {
        return eshkol_thread_pool_is_worker();
    }
    return 0;
#endif
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
    // #341: no allocation-slot hijack recorded yet. REGION_NO_HIJACK (rather
    // than nullptr) so a teardown that runs before eshkol_region_enter treats
    // the slot as untouched instead of restoring a null arena.
    region->entry_saved_arena = REGION_NO_HIJACK;
    region->pinned = 0;  // SW-59: set only by eshkol_region_pin_all()
    region->handle_owned = 0;  // SW-74: set by eshkol_region_handle_open()

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
// Defined in runtime_regions_evacuation.cpp (owner of the evacuator's map
// type) -- frees region->fwd_map.

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

    // SW-59/SW-74: a pinned region — one a continuation that may outlive it was
    // captured inside, see eshkol_region_pin_all() — must NOT have its arena
    // RECLAIMED here. The continuation's raw C-stack snapshot
    // (eshkol_continuation_capture_stack(), runtime_continuations.cpp) may hold
    // interior pointers into this arena that this call site cannot see and
    // therefore cannot walk, unlike the with-region result value, which IS
    // deep-promoted before this runs (see eshkol_region_unwind_to()). Reclaiming
    // region->arena here means the resumed continuation dereferences 0xCB (or
    // worse, silently reused memory) the instant it touches anything the
    // with-region body built.
    //
    // SW-74: "not reclaimed" used to mean the block chain was simply dropped —
    // arena_destroy() was skipped and nothing ever freed the blocks, an unowned
    // leak for the rest of the process that no arena counter could even see. It
    // now means the chain is PROMOTED into the arena that encloses this region,
    // which is exactly what the bytecode VM does (vm_evac_promote_all_blocks(),
    // lib/backend/vm_region_evac.c, splices a pinned region's blocks behind the
    // parent's bump block). Ownership moves outward with the data:
    //
    //   • the promoted blocks are never allocated from again, so nothing the
    //     continuation still reads can be overwritten;
    //   • they are freed when the enclosing scope ends — at the outermost level
    //     that is the process-global arena, i.e. at exit rather than never;
    //   • their bytes are charged to the enclosing arena, so ESHKOL_ARENA_REPORT=1
    //     and the resident-memory gates measure the retention instead of losing
    //     it, which is what let SW-74 hide in the first place.
    //
    // escape_base is the promotion target because it is the one arena captured
    // at region_push that is GUARANTEED to outlive this region (the parent
    // region's arena when nested, the real process arena at top level).
    // get_global_arena() cannot be used here: inside a with-region body codegen
    // has hijacked that slot to point at this very region.
    if (region->pinned) {
        arena_t* target = region->escape_base;
        if (!target && region->parent) target = region->parent->arena;
        if (!target) target = get_global_arena();

        // A pin is taken by eshkol_region_pin_all(), which pins EVERY open
        // region, so an enclosing region that receives this promotion is
        // already pinned and will promote onward when it pops. Re-asserting it
        // costs nothing and makes the invariant local to this call site: the
        // arena we are handing memory to must not be one that gets reclaimed
        // while the continuation is still able to read it.
        if (region->parent && region->parent->arena == target) {
            region->parent->pinned = 1;
        }

        size_t promoted = 0;
        if (region->arena && target && target != region->arena) {
            promoted = arena_adopt_blocks(target, region->arena);
        }
        eshkol_debug("Region '%s' is pinned (a continuation captured inside it may "
                     "outlive it); promoting %zu bytes (%zu block bytes) into the "
                     "enclosing arena instead of reclaiming", name, used, promoted);
        eshkol_region_pin_notice();
        // arena_adopt_blocks left the region arena with no blocks, so this
        // releases only its scope records, mutex and control struct — the
        // region's own bookkeeping, which nothing outside it can reference.
        if (region->arena) {
            arena_destroy(region->arena);
        }
    } else {
        eshkol_debug("Destroying region '%s', freeing %zu bytes", name, used);
        if (region->arena) {
            arena_destroy(region->arena);
        }
    }
    region->arena = nullptr;

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

    // #713: a promotion out of this region can fail only while a region is
    // open, so the thread's allocation-failure condition is reserved now,
    // while memory is available (a no-op after the first region).
    eshkol_reserve_allocation_failure_condition();

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
 * @brief Pin every region currently open on the calling thread's region stack
 *        (SW-59 — the native analogue of the bytecode VM's
 *        heap_region_pin_all(), lib/backend/vm_core.c).
 *
 * Called from eshkol_make_continuation_state_flags() (runtime_continuations.cpp)
 * whenever a first-class continuation THAT MAY OUTLIVE ITS FRAME is captured
 * with __region_stack_depth > 0 — the guard the VM applies
 * (`vm->heap.regions.depth > 0`) before its own pin_all, narrowed by SW-74
 * with the compiler's escape classification. A capture the compiler proves is
 * escape-only (ESHKOL_CONT_FLAG_ESCAPE_ONLY) cannot be invoked once the
 * capturing frame has returned, and the region's dynamic extent strictly
 * contains that frame's, so no such continuation can ever observe the region
 * after it closes and no pin is needed. That is the early-return idiom, and
 * before SW-74 it pinned — and therefore permanently leaked — a region per
 * capture.
 *
 * A pinned region's region_destroy() promotes its arena into the enclosing
 * arena instead of reclaiming it (see region_destroy() above), because the
 * continuation's stack snapshot may hold interior pointers into ANY
 * currently-open region's arena, not only the innermost one — a `call/cc`
 * nested two `with-region`s deep can be re-entered after both have exited,
 * and either frame's locals may need either arena. That is why this pins the
 * whole stack rather than region_current().
 *
 * A pin, once taken, is never lifted (matching the VM, which has no unpin path
 * either): deciding that a first-class continuation can no longer be invoked
 * needs a tracing collector, which this runtime does not have. What SW-74
 * changed is the cost of a pin — promotion rather than an unowned leak — and
 * how often one is taken at all. See
 * docs/reference/language/continuations.md for the tradeoff as documented
 * for users, and .icc/silent-wrong-ledger.yaml SW-59/SW-74 for the defects
 * this closes.
 */
// A continuation keeps an opaque native stack image, so a pinned region cannot
// be reclaimed by the typed evacuator. Bound the otherwise process-lifetime
// retention and fail closed once the bound is reached.
static constexpr size_t ESHKOL_CONTINUATION_PIN_BUDGET = 64u * 1024u * 1024u;
static size_t g_continuation_pinned_bytes = 0;
static std::mutex g_continuation_pin_mutex;

int eshkol_region_pin_all(void) {
    std::lock_guard<std::mutex> lock(g_continuation_pin_mutex);
    size_t additional = 0;
    for (uint64_t i = 0; i < __region_stack_depth; ++i) {
        eshkol_region_t* r = __region_stack[i];
        if (r && !r->pinned && r->arena)
            additional += arena_get_used_memory(r->arena);
    }
    if (additional > ESHKOL_CONTINUATION_PIN_BUDGET -
                    (g_continuation_pinned_bytes < ESHKOL_CONTINUATION_PIN_BUDGET
                         ? g_continuation_pinned_bytes : ESHKOL_CONTINUATION_PIN_BUDGET)) {
        // "capture rejected", not "resume rejected": both engines refuse the
        // capture itself (eshkol_make_continuation_state_flags on native,
        // vm_capture_continuation_dynamic_state on the VM). The old wording
        // described native's pre-fix behaviour, where the capture was accepted
        // and the process only died later, at the invocation, via abort().
        eshkol_error("continuation region-pin budget exceeded (%zu bytes); "
                     "capture rejected to prevent an unbounded pinned-region leak",
                     ESHKOL_CONTINUATION_PIN_BUDGET);
        return 0;
    }
    for (uint64_t i = 0; i < __region_stack_depth; ++i) {
        eshkol_region_t* r = __region_stack[i];
        if (r && !r->pinned) {
            r->pinned = 1;
            if (r->arena) g_continuation_pinned_bytes += arena_get_used_memory(r->arena);
        }
    }
    return 1;
}

/**
 * @brief SW-74: is any open region closable out-of-line (handle-owned)?
 *
 * The escape-only pin skip rests on "no region open at capture time can be
 * torn down before the capturing frame returns". That holds for every lexical
 * `with-region`, whose exit is downstream of the `call/cc` it encloses, and
 * fails for a region opened through the handle API, because `(region-close h)`
 * is an ordinary call that can run inside the `call/cc` procedure itself.
 * Consulted once per capture; the region stack is at most MAX_REGION_DEPTH (64)
 * deep and is one in the overwhelming majority of programs.
 *
 * @return Non-zero if at least one open region was opened via a region handle.
 */
int eshkol_region_any_handle_owned_open(void) {
    for (uint64_t i = 0; i < __region_stack_depth; ++i) {
        const eshkol_region_t* r = __region_stack[i];
        if (r && r->handle_owned) return 1;
    }
    return 0;
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

    // Workers keep using the thread-safe process arena while parallel work is
    // active; the lexical domain is thread-local everywhere else.
    if (s_parallel_depth.load(std::memory_order_acquire) != 0 ||
        arena_is_worker_thread()) {
        region->entry_saved_arena = REGION_NO_HIJACK;
        return REGION_NO_HIJACK;
    }

    arena_t* saved = eshkol_memctx_current()->allocation_domain;
    region->entry_saved_arena = saved ? saved : REGION_NO_HIJACK;
    eshkol_memctx_current()->allocation_domain = region->arena;
    return region->entry_saved_arena;
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
    if (saved == REGION_NO_HIJACK) {
        eshkol_memctx_current()->allocation_domain = nullptr;
        return;
    }
    eshkol_memctx_current()->allocation_domain = saved;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-execution-context region stack (inc/eshkol/core/coroutine.h).
//
// The lexical region stack is LIFO by construction, and one OS thread running
// several coroutines interleaves their region extents: coroutine A opens R1 and
// suspends, B opens R2 and suspends, and A's exit would pop R2. So the open
// regions and the allocation domain they select belong to the execution
// context, not the thread: a coroutine switch saves the outgoing context's
// stack and installs the incoming one's. A coroutine starts with no region
// open, allocating into the default domain, because it can outlive every
// region that was open where it was created. This is ADR-0001's per-computation
// memory context at execution-context granularity.

extern "C" int eshkol_region_context_save(eshkol_exec_context_t* ctx) {
    const uint64_t depth = __region_stack_depth;
    if (depth > ctx->region_capacity) {
        uint64_t want = ctx->region_capacity ? ctx->region_capacity * 2 : 4;
        if (want < depth) want = depth;
        void* grown = realloc(ctx->region_slots, (size_t)want * sizeof(void*));
        if (!grown) return 0;
        ctx->region_slots = (void**)grown;
        ctx->region_capacity = want;
    }
    for (uint64_t i = 0; i < depth; ++i) ctx->region_slots[i] = __region_stack[i];
    ctx->region_depth = depth;
    ctx->allocation_domain = t_memctx.allocation_domain;
    return 1;
}

extern "C" void eshkol_region_context_load(const eshkol_exec_context_t* ctx) {
    const uint64_t depth = ctx->region_depth <= MAX_REGION_DEPTH
        ? ctx->region_depth : (uint64_t)MAX_REGION_DEPTH;
    for (uint64_t i = 0; i < depth; ++i) {
        __region_stack[i] = (eshkol_region_t*)ctx->region_slots[i];
    }
    for (uint64_t i = depth; i < __region_stack_depth; ++i) __region_stack[i] = nullptr;
    __region_stack_depth = depth;
    t_memctx.allocation_domain = (arena_t*)ctx->allocation_domain;
}

extern "C" void eshkol_exec_context_dispose(eshkol_exec_context_t* ctx) {
    if (!ctx) return;
    free(ctx->region_slots);
    ctx->region_slots = nullptr;
    free(ctx->param_bindings);
    ctx->param_bindings = nullptr;
    ctx->param_capacity = 0;
    ctx->param_count = 0;
    ctx->region_capacity = 0;
    ctx->region_depth = 0;
}

/**
 * @brief SW-164: the home for values CACHED across iterations.
 *
 * Picking the right ARENA is not enough for such a value; it also has to be
 * outside every scope SPAN. `__global_arena` is only the current-arena slot,
 * which `with-region` hijacks. But even the true process arena is not safe on
 * its own: a loop's per-iteration scope marks a point in whichever arena is
 * current and rewinds to it, so a constant materialized lazily on the first
 * iteration is allocated ABOVE that mark and the first rewind reclaims it,
 * leaving the cache pointing at memory the next allocation hands out again.
 * (Observed during this work as an exact rational literal whose numerator
 * later read back as an unrelated bignum: right arena, wrong side of a mark.)
 *
 * So constants get an arena of their own. Nothing else allocates from it,
 * nothing ever pushes a scope on it, and it is never reset — the only way to
 * be certain no rewind can reach what it holds. Thread-safe because a literal
 * may first be evaluated on any thread; small, and it grows only by the
 * distinct constants a program actually evaluates.
 */
static arena_t* s_literal_arena = nullptr;

extern "C" arena_t* eshkol_literal_arena(void) {
    eshkol_arena_global_once(init_global_arena_internal);
    if (!s_literal_arena) s_literal_arena = arena_create_threadsafe(4096);
    return s_literal_arena ? s_literal_arena : get_global_arena_shared();
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
    s_parallel_depth.fetch_add(1, std::memory_order_acq_rel);
}

/**
 * @brief Close a parallel scope opened by eshkol_parallel_scope_begin(),
 *        restoring the current-arena slot displaced on the 0->1 edge once the
 *        last overlapping scope closes.
 */
extern "C" void eshkol_parallel_scope_end(void) {
    std::lock_guard<std::mutex> lk(s_parallel_arena_mtx);
    s_parallel_depth.fetch_sub(1, std::memory_order_acq_rel);
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
