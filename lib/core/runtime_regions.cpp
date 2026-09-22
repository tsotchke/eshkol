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
#include "runtime_region_promotion_internal.h"
#include "runtime_region_leaf_layouts.h"
#include "../../inc/eshkol/core/bignum.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/exhaustive_dispatch.h"
#include "../../inc/eshkol/core/logic.h"       // eshkol_substitution_t / _fact_t / _knowledge_base_t
#include "../../inc/eshkol/core/inference.h"   // eshkol_factor_graph_t / _factor_t
#include "../../inc/eshkol/core/workspace.h"   // eshkol_workspace_t / _workspace_module_t
#include "../../inc/eshkol/core/rational.h"    // eshkol_rational_t (bignum-backed exact rationals)

#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <utility>
#include <atomic>
#include <mutex>
#include <limits>
#include <new>
#include <stdexcept>
#include <memory>

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
// OALR Phase A: per-thread memory context (ADR-0001).
//
// The memctx makes the "current allocation arena" a thread-local property reached
// through eshkol_current_arena(), instead of a direct read of the shared
// __global_arena global. `with-region` now updates this thread-local domain
// (mirrored into __global_arena in the single-threaded safe case only, see
// eshkol_region_enter) so allocation routing is thread-local. The full ABI-v2
// memctx (region_top/residence/resident txn) and the object-header change are
// deferred to later phases.
static thread_local eshkol_memctx_t t_memctx = {
    ESHKOL_MEMORY_ABI_PHASE_A, 0u, 0ull, nullptr, nullptr
};
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

// Compiler-private allocation owner for escaping continuation state. Unlike
// the mutable shared allocation slot, this identity survives region routing.
extern "C" arena_t* eshkol_root_arena_v1() {
    eshkol_arena_global_once(init_global_arena_internal);
    if (!s_shared_root_arena) eshkol_runtime_emergency_raise_v1(4);
    return s_shared_root_arena;
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

    // SW-59: a pinned region — one a continuation was captured inside, see
    // eshkol_region_pin_all() — must NOT have its arena freed here. The
    // continuation's raw C-stack snapshot (eshkol_continuation_capture_stack(),
    // runtime_continuations.cpp) may hold interior pointers into this arena
    // that this call site cannot see and therefore cannot promote, unlike the
    // with-region result value, which IS deep-promoted before this runs (see
    // eshkol_region_unwind_to()). Freeing region->arena here means the
    // resumed continuation dereferences 0xCB (or worse, silently reused
    // memory) the instant it touches anything the with-region body built.
    // The deliberate failure direction — matching the VM's own
    // heap_region_pin_all()/vm_region_evac.c promote-wholesale path exactly —
    // is to leak the arena instead: it is never freed for the rest of the
    // process, but nothing captured inside it can ever dangle.
    if (region->pinned) {
        eshkol_debug("Region '%s' is pinned (continuation captured inside it); "
                     "leaking %zu bytes of arena instead of freeing", name, used);
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
 * Called from eshkol_make_continuation_state() (runtime_continuations.cpp)
 * whenever a first-class continuation is captured with __region_stack_depth
 * > 0 — exactly the guard the VM applies (`vm->heap.regions.depth > 0`)
 * before its own pin_all. A pinned region's region_destroy() leaks its arena
 * instead of freeing it (see region_destroy() above), because the
 * continuation's stack snapshot may hold interior pointers into ANY
 * currently-open region's arena, not only the innermost one — a `call/cc`
 * nested two `with-region`s deep can be re-entered after both have exited,
 * and either frame's locals may need either arena.
 *
 * Deliberately never unpinned (matching the VM, which has no unpin path
 * either: a pin is a Stage-1 policy that trades "this region's memory is
 * never reclaimed" for "no continuation can ever observe a freed region",
 * for the remainder of the process). See
 * docs/reference/language/continuations.md for the tradeoff as documented
 * for users, and .icc/silent-wrong-ledger.yaml SW-59 for the defect this
 * closes.
 */
void eshkol_region_pin_all(void) {
    for (uint64_t i = 0; i < __region_stack_depth; ++i) {
        eshkol_region_t* r = __region_stack[i];
        if (r) r->pinned = 1;
    }
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
        region->entry_saved_arena = REGION_NO_HIJACK;
        return REGION_NO_HIJACK;
    }

    arena_t* saved = __global_arena;
    // #341: record the displaced arena ON THE REGION as well as returning it.
    // with-region codegen keeps the returned token in an SSA register, but a
    // non-lexical teardown (region-close, or an unwind crossing this region)
    // reaches the region only through the region stack and has no register to
    // read — so the restore token has to be recoverable from the region itself.
    region->entry_saved_arena = saved;
    __global_arena = region->arena;
    // OALR Phase A: mirror the redirect into the thread-local memory context so
    // eshkol_current_arena() (the accessor generated code now routes through)
    // resolves body allocations to the region arena WITHOUT reading the shared
    // slot. Kept in lockstep with the __global_arena write above (single-threaded,
    // non-parallel — this branch is unreachable on a worker or during a parallel
    // scope) so migrated (accessor) and not-yet-migrated (direct __global_arena)
    // allocation sites always resolve to the same arena.
    eshkol_memctx_current()->allocation_domain = region->arena;
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
    // OALR Phase A: restore the thread-local allocation domain in lockstep with
    // the shared slot (see eshkol_region_enter).
    eshkol_memctx_current()->allocation_domain = saved;
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
    // ESH-0214d: neuro-symbolic / workspace subtypes that carry interior
    // tagged values and/or raw arena buffers (previously dropped to EVAC_LEAF,
    // which silently left those interior pointers aimed into the dying region).
    EVAC_SUBSTITUTION,   // eshkol_substitution_t: inline terms[] tagged values
    EVAC_FACT,           // eshkol_fact_t: inline args[] tagged values (+ predicate str)
    EVAC_KNOWLEDGE_BASE, // eshkol_knowledge_base_t: facts[] raw array of fact ptrs
    EVAC_FACTOR_GRAPH,   // eshkol_factor_graph_t: nested raw numeric buffers
    EVAC_WORKSPACE,      // eshkol_workspace_t: content buffer + per-module name/process_fn
    EVAC_PROMISE,        // [forced:i64][thunk:tagged @8][cached:tagged @24] (delay/force)
    EVAC_PRIMITIVE,      // native function address + potentially regional name
    EVAC_RATIONAL,       // eshkol_rational_t: big_num/big_den raw bignum pointers (is_big==1 only)
    // SW-66: an EXACT-COEFFICIENT (COEFF_RATIONAL) Taylor tower's c[] is an
    // array of eshkol_tagged_value_t that can hold HEAP_PTRs to arena-
    // resident bignum/rational coefficients (see runtime_taylor.c). A
    // COEFF_F64 tower's c[] is raw doubles -- nothing to walk, same shape as
    // EVAC_RATIONAL's is_big==0 fast path being a cheap no-op.
    EVAC_TAYLOR,         // esh_taylor_t: c[] tagged-value array (COEFF_RATIONAL only)
};

struct PromotionFailure { int32_t status; };
[[noreturn]] static void promotion_fail(int32_t status) { throw PromotionFailure{status}; }

#ifdef ESHKOL_PROMOTION_TESTING
static thread_local eshkol_promotion_test_stats promotion_stats{};
static thread_local int promotion_fail_site = -2;
static thread_local int64_t promotion_fail_after = -1;
static thread_local uint64_t promotion_fail_attempt = 0;
#endif
static void promotion_allocation_attempt(int site) {
#ifdef ESHKOL_PROMOTION_TESTING
    ++promotion_stats.attempts[site];
    if (promotion_fail_after >= 0 &&
        (promotion_fail_site == site || (promotion_fail_site == -1 && site < 4)) &&
        promotion_fail_attempt++ >= static_cast<uint64_t>(promotion_fail_after))
        throw std::bad_alloc();
#else
    (void)site;
#endif
}

template<class T, int Site> struct PromotionAllocator {
    using value_type = T;
    template<class U> struct rebind { using other = PromotionAllocator<U, Site>; };
    PromotionAllocator() noexcept = default;
    template<class U> PromotionAllocator(const PromotionAllocator<U, Site>&) noexcept {}
    T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) promotion_fail(3);
        promotion_allocation_attempt(Site);
        T* p = static_cast<T*>(::operator new(n * sizeof(T)));
#ifdef ESHKOL_PROMOTION_TESTING
        auto& live = promotion_stats.live_bytes[Site];
        live += n * sizeof(T);
        if (live > promotion_stats.peak_bytes[Site]) promotion_stats.peak_bytes[Site] = live;
#endif
        return p;
    }
    void deallocate(T* p, size_t n) noexcept {
#ifdef ESHKOL_PROMOTION_TESTING
        promotion_stats.live_bytes[Site] -= n * sizeof(T);
#else
        (void)n;
#endif
        ::operator delete(p);
    }
    template<class U> bool operator==(const PromotionAllocator<U, Site>&) const noexcept { return true; }
    template<class U> bool operator!=(const PromotionAllocator<U, Site>&) const noexcept { return false; }
};
struct EvacForward { void* data; size_t size; bool object; };
using EvacFwdMap = std::unordered_map<const void*, EvacForward, std::hash<const void*>,
    std::equal_to<const void*>, PromotionAllocator<std::pair<const void* const, EvacForward>, 0>>;
static void evac_delete_map(EvacFwdMap* p) noexcept {
    if (!p) return;
    p->~EvacFwdMap();
    PromotionAllocator<EvacFwdMap, 0>{}.deallocate(p, 1);
}
static EvacFwdMap* evac_new_map(const EvacFwdMap* original) {
    auto allocator = PromotionAllocator<EvacFwdMap, 0>{};
    EvacFwdMap* p = allocator.allocate(1);
    try {
        if (original) new(p) EvacFwdMap(*original);
        else new(p) EvacFwdMap();
    } catch (...) { allocator.deallocate(p, 1); throw; }
    return p;
}
struct EvacSpan { void* data; size_t size; };
struct EvacState {
    arena_t* target;
    int boundary_idx;
    EvacFwdMap* fwd = nullptr; // private candidate until the only commit
    std::vector<std::pair<void*, EvacKind>, PromotionAllocator<std::pair<void*, EvacKind>, 1>> worklist;
    std::vector<EvacSpan, PromotionAllocator<EvacSpan, 2>> spans;
    size_t copies = 0;
    bool committed = false;
    EvacState(arena_t* arena, int boundary) : target(arena), boundary_idx(boundary) {}
    ~EvacState() {
        if (!committed) for (const auto& span : spans) std::memset(span.data, 0, span.size);
        evac_delete_map(fwd);
    }
};

static size_t evac_size(size_t count, size_t width, size_t prefix = 0) {
    if (count > (std::numeric_limits<size_t>::max() - prefix) / width) promotion_fail(3);
    return prefix + count * width;
}
// A live runtime pointer is required. This check bounds regional source spans
// against used bytes before reading headers, lengths, or raw buffers.
static size_t evac_source_available(const void* p) {
    const auto addr = reinterpret_cast<uintptr_t>(p);
    const int index = region_index_owning(p);
    if (index < 0) return std::numeric_limits<size_t>::max();
    for (auto* block = __region_stack[index]->arena->current_block; block; block = block->next) {
        const auto base = reinterpret_cast<uintptr_t>(block->memory);
        if (addr >= base && addr - base < block->used) return block->used - (addr - base);
    }
    promotion_fail(2);
}
static void evac_require_span(const void* p, size_t n) {
    if ((!p && n) || (p && n > evac_source_available(p))) promotion_fail(2);
}
static size_t evac_string_size(const char* p) {
    const size_t available = evac_source_available(p);
    const void* end = std::memchr(p, 0, available);
    if (!end) promotion_fail(2);
    return static_cast<const char*>(end) - p + 1;
}
static void* evac_allocate(EvacState& st, size_t size) {
    if (size > std::numeric_limits<size_t>::max() - 16) promotion_fail(3);
    st.spans.reserve(evac_size(st.spans.size() + 1, 1));
    promotion_allocation_attempt(4);
    void* p = eshkol_region_allocate_quiet(st.target, size, 16);
    if (!p) promotion_fail(1);
    st.spans.push_back({p, size}); // reserved BEFORE the target allocation
#ifdef ESHKOL_PROMOTION_TESTING
    promotion_stats.target_bytes += size;
#endif
    return p;
}
// Stable/previously committed buffers may be inspected, never rewritten. A
// changed interior in such a buffer means the original lifetime invariant was
// already violated; fail rather than mutate the live source/old map graph.
template<class T> static void evac_assign(EvacState& st, T& slot, T value) {
    if (std::memcmp(&slot, &value, sizeof(T)) == 0) return;
    const auto addr = reinterpret_cast<uintptr_t>(&slot);
    for (const auto& span : st.spans) {
        const auto base = reinterpret_cast<uintptr_t>(span.data);
        if (addr >= base && addr - base <= span.size && sizeof(T) <= span.size - (addr - base)) {
            slot = value;
            return;
        }
    }
    promotion_fail(2);
}

} // namespace

// Free a region's persistent deep-escape forwarding map (declared above
// region_destroy; the map type is only visible from here down).
static void region_free_fwd_map(eshkol_region_t* region) {
    if (region && region->fwd_map) {
        evac_delete_map((EvacFwdMap*)region->fwd_map);
        region->fwd_map = nullptr;
        region->fwd_target = nullptr;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// WHICH TAGS CARRY AN ARENA POINTER
//
// Every region escape path — the write barrier, the with-region result escape,
// and the ESH-0214e nursery recycle — has to answer one question first: does
// this tagged value hold a pointer into the region being reclaimed? It asked
// ESHKOL_IS_ANY_PTR_TYPE, which enumerates the two CONSOLIDATED pointer tags
// (HEAP_PTR / CALLABLE) plus ports. That set is NOT the set of pointer-carrying
// tags.
//
// ESHKOL_VALUE_DUAL_NUMBER and ESHKOL_VALUE_COMPLEX sit in the 0-7 block that
// eshkol_value_type_t labels "IMMEDIATE VALUES — data stored directly in tagged
// value", but neither is an immediate: both put a POINTER in data.ptr_val,
// aimed at a headerless 16-byte pair arena-allocated at the operation that
// produced it (arena_allocate_dual_number; ComplexCodegen::packComplexToTagged).
// Reading them as immediates made every escape path a silent no-op for them, so
// a forward-mode dual or a complex number produced inside a region and stored
// into a structure outside it kept a pointer into the arena about to be
// recycled — and the next iteration reallocated the same address. The values
// stayed dereferenceable and plausible, which is why this corrupted answers
// instead of crashing.
//
// The arena-scope half of the same feature (eshkol_arena_iter_scope_end) got
// this right by inverting the test: only PROVABLY pointer-free immediates
// (NULL/INT64/DOUBLE/BOOL/CHAR, plus the eof-object) skip the pointer check.
// These helpers bring the region half into line with that rule.
//
// Deliberately NOT included:
//   SYMBOL (5)   — data.ptr_val is an interned symbol's stable text, owned by
//                  the interning table rather than a region arena. Copying it
//                  per escape would both be unnecessary and break the pointer
//                  identity interning exists to provide.
//   LOGIC_VAR(10)— data is a var_id integer, not a pointer.
//   HANDLE/BUFFER/STREAM/EVENT (16-19) — linear resources whose payloads are
//                  externally owned; duplicating one would duplicate the
//                  resource. A region-resident one is reported below rather
//                  than silently copied or silently dropped.
// ───────────────────────────────────────────────────────────────────────────

/* Byte size of the headerless, self-contained payload a pointer-carrying
 * IMMEDIATE tag addresses, or 0 when the tag is not one of those. These cannot
 * go through evac_object: it sizes and classifies an object from the
 * eshkol_object_header_t 8 bytes below the payload, and these carry no header —
 * those 8 bytes belong to whatever was allocated before them. A flat copy is
 * complete for both (two doubles, no interior pointers), and neither has
 * observable pointer identity, so copying cannot break eq?-style sharing the
 * way copying an interned symbol would. */
static size_t region_headerless_payload_size(uint8_t type) {
    /* Exactness flags may be OR'd into a numeric tag; strip them before
     * matching. The port flags share those bits but only ever ride on
     * HEAP_PTR, which is not one of the values matched here. */
    switch (type & (uint8_t)~(ESHKOL_VALUE_EXACT_FLAG | ESHKOL_VALUE_INEXACT_FLAG)) {
        case ESHKOL_VALUE_DUAL_NUMBER: return sizeof(eshkol_dual_number_t);
        case ESHKOL_VALUE_COMPLEX:     return 2 * sizeof(double);
        default:                       return 0;
    }
}

/* Does a value with this tag hold a pointer that a region reclaim could
 * invalidate? The single predicate every escape path shares. */
static bool region_linear_resource(uint8_t type) {
    return type >= ESHKOL_VALUE_HANDLE && type <= ESHKOL_VALUE_EVENT;
}
static bool region_value_carries_pointer(uint8_t type) {
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    return ESHKOL_IS_ANY_PTR_TYPE(type) || is_port ||
           region_headerless_payload_size(type) != 0 || region_linear_resource(type);
}

// Classify an object (given its live original data pointer and the tagged value
// referencing it) into an EvacKind. Regional ports have no admitted layout;
// stable external port identities take the no-promotion path before this point.
static EvacKind evac_kind_for(const eshkol_tagged_value_t& v, const void* old_data) {
    const uint8_t type = v.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    if (is_port) promotion_fail(2); // native ports are registry-owned FILE*, not regional headers

    const auto* h = (const eshkol_object_header_t*)
        ((const uint8_t*)old_data - sizeof(eshkol_object_header_t));
    const uint8_t sub = h->subtype;

    if (ESHKOL_IS_ANY_CALLABLE_TYPE(type)) {
        // An exhaustive switch rather than an if-chain, for the same reason as
        // the heap-subtype switch below: an if-chain's fallthrough is a default
        // wearing different syntax, and a new callable subtype would inherit
        // EVAC_LEAF from it in silence.
        ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
        switch ((callable_subtype_t)sub) {
        case CALLABLE_SUBTYPE_CLOSURE:
        case CALLABLE_SUBTYPE_LAMBDA_SEXPR: // zero-capture closure producer uses this tag
            return EVAC_CLOSURE;
        case CALLABLE_SUBTYPE_PRIMITIVE:
            return EVAC_PRIMITIVE;
        case CALLABLE_SUBTYPE_CONTINUATION:
        case CALLABLE_SUBTYPE_AD_NODE:
            break;  // handled below
        }
        ESHKOL_EXHAUSTIVE_SWITCH_END
        if (sub == CALLABLE_SUBTYPE_AD_NODE) {
            if (h->size < sizeof(ad_node_t)) promotion_fail(2);
            evac_require_span(old_data, sizeof(ad_node_t));
            const auto* n = static_cast<const ad_node_t*>(old_data);
            if (!n->input1 && !n->input2 && !n->input3 && !n->input4 &&
                !n->tensor_value && !n->tensor_gradient && !n->saved_tensors && !n->shape) return EVAC_LEAF;
        }
        promotion_fail(2); // no proven traversal for these callable layouts

    }

    switch ((heap_subtype_t)sub) {
        case HEAP_SUBTYPE_CONS:        return EVAC_CONS;
        case HEAP_SUBTYPE_VECTOR:      return EVAC_VECTOR;   // records are vectors too
        case HEAP_SUBTYPE_RECORD:      return EVAC_VECTOR;   // records allocate as vectors;
                                                             // belt-and-suspenders so a future
                                                             // record allocator stamping subtype 7
                                                             // cannot silently regress to leaf.
        case HEAP_SUBTYPE_MULTI_VALUE: return EVAC_MULTIVALUE;
        case HEAP_SUBTYPE_HASH:        return EVAC_HASH;
        case HEAP_SUBTYPE_TENSOR:      return EVAC_TENSOR;
        case HEAP_SUBTYPE_EXCEPTION:   return EVAC_EXCEPTION;
        // ESH-0214d: neuro-symbolic / workspace subtypes now deep-walked. Each
        // carries interior tagged values and/or raw arena buffers that a shallow
        // leaf copy left dangling into the popped region arena.
        case HEAP_SUBTYPE_SUBSTITUTION:   return EVAC_SUBSTITUTION;
        case HEAP_SUBTYPE_FACT:           return EVAC_FACT;
        case HEAP_SUBTYPE_KNOWLEDGE_BASE: return EVAC_KNOWLEDGE_BASE;
        case HEAP_SUBTYPE_FACTOR_GRAPH:   return EVAC_FACTOR_GRAPH;
        case HEAP_SUBTYPE_WORKSPACE:      return EVAC_WORKSPACE;
        case HEAP_SUBTYPE_PROMISE:        return EVAC_PROMISE;
        // RATIONAL: the int64 fast path (is_big == 0) is fully self-contained
        // (numerator/denominator are inline scalars), but the bignum path
        // (is_big == 1) carries two RAW eshkol_bignum_t* pointers (big_num /
        // big_den) that can themselves be region-resident. A shallow leaf copy
        // of a big rational left those two pointers aimed into the popped
        // region arena -- the same class of gap ESH-0214d closed for the
        // logic/workspace subtypes. Always dispatched through EVAC_RATIONAL;
        // the is_big==0 case is a cheap no-op there (nothing to walk).
        case HEAP_SUBTYPE_RATIONAL:       return EVAC_RATIONAL;
        // SW-66 (2026-08-26 libclang architecture-verify admission): only the
        // EXACT-COEFFICIENT (COEFF_RATIONAL) representation of a Taylor tower
        // carries interior region pointers -- its c[] is reinterpreted as an
        // eshkol_tagged_value_t array, and any entry can be a HEAP_PTR to a
        // bignum/rational coefficient allocated in the same (possibly dying)
        // region (see the exact-tower path in runtime_taylor.c). A COEFF_F64
        // tower's c[] is raw doubles. Always dispatched through EVAC_TAYLOR;
        // the F64 case is a cheap no-op there, mirroring EVAC_RATIONAL's
        // is_big==0 fast path just above.
        case HEAP_SUBTYPE_TAYLOR:         return EVAC_TAYLOR;
        // ── EVAC_LEAF, one subtype at a time ────────────────────────────
        //
        // THERE IS NO `default:` HERE, AND THAT IS THE POINT. A default in
        // this switch would mean "any heap subtype I have not thought about
        // is self-contained" — which is the ESH-0214d bug stated as a policy.
        // KB / FACT / SUBSTITUTION / WORKSPACE each carried interior tagged
        // values, each fell to that default, and each left pointers aimed into
        // an arena that had already been reclaimed. Listing every member means
        // a NEW subtype cannot inherit that answer by omission: adding one to
        // heap_subtype_t without deciding its evacuation is a compile error
        // (this file is built with -Werror=switch-enum; see
        // eshkol_require_exhaustive_dispatch in CMakeLists.txt).
        //
        // STRING / SYMBOL / BIGNUM / BYTEVECTOR / I128: self-contained payloads
        // -> a contiguous leaf copy fully preserves them. I128 in particular is
        // a flat 16-byte {lo,hi} POD with no interior pointers, so the leaf
        // path (a straight memcpy of header+payload) is correct and complete —
        // no deep walk is needed (confirmed against ESH-0214d).
        case HEAP_SUBTYPE_STRING:
        case HEAP_SUBTYPE_SYMBOL:
        case HEAP_SUBTYPE_BIGNUM:
        case HEAP_SUBTYPE_BYTEVECTOR:
        case HEAP_SUBTYPE_I128:
        // Deliberately kept EVAC_LEAF (no interior region pointers, or their
        // interior graph is not confidently/safely traversable here, AND they
        // are not observed to escape a region by mutation):
        //   PORT      - wraps an OS fd/FILE*; handle intentionally shared, not copied.
        //   PRNG      - self-contained state words, no interior pointers.
        //   PARAMETER - malloc-owned stack slots and converter are root-promoted
        //               at publication; regional control copies hold no young edges.
        //   DNC/SDNC  - VERIFIED SW-66 by reading both handle layouts:
        //               DncHandle.{mem,usage} (lib/core/dnc_api.c) and
        //               SdncHandle.w (lib/core/sdnc_api.c) are calloc'd on
        //               the plain C heap, never arena-allocated
        //               (SdncHandle.pe[][] is inline scalar data with no
        //               pointer at all). A shallow copy preserves those raw
        //               pointer VALUES exactly; they cannot dangle when the
        //               region's arena is freed, because they never pointed
        //               into it in the first place. This is a confirmed
        //               property of the current handle layouts, not an
        //               assumption -- see the debug guard just below.
        //   TAYLOR is NOT in this leaf list -- see its explicit EVAC_TAYLOR
        //   case above (SW-66's exact-coefficient deep-walk fix).
        // The checked leaf validator below admits producer-owned layouts and
        // rejects regional ports explicitly in every build mode.
        case HEAP_SUBTYPE_PORT:
        case HEAP_SUBTYPE_PRNG:
        case HEAP_SUBTYPE_PARAMETER:
        case HEAP_SUBTYPE_DNC:
        case HEAP_SUBTYPE_SDNC:
            return EVAC_LEAF;
    }

    promotion_fail(2); // undeclared subtype is not a leaf
}

// Raw allocation boundaries are not tracked by the arena. Preserve an already
// copied same-base prefix; reject any other known source overlap instead of
// producing independently copied aliases or a truncated larger view.
static void evac_check_overlap(const EvacState& st, const void* source, size_t size, bool object) {
    uintptr_t start = reinterpret_cast<uintptr_t>(source);
    if (object) {
        if (start < sizeof(eshkol_object_header_t)) promotion_fail(3);
        start -= sizeof(eshkol_object_header_t);
        size = evac_size(size, 1, sizeof(eshkol_object_header_t));
    }
    if (size > UINTPTR_MAX - start) promotion_fail(3);
    for (const auto& entry : *st.fwd) {
        uintptr_t other = reinterpret_cast<uintptr_t>(entry.first);
        size_t extent = entry.second.size;
        if (entry.second.object) {
            other -= sizeof(eshkol_object_header_t);
            extent = evac_size(extent, 1, sizeof(eshkol_object_header_t));
        }
        if (extent > UINTPTR_MAX - other) promotion_fail(3);
        if ((start < other + extent && other < start + size) ||
            (!size && start >= other && start < other + extent) ||
            (!extent && other >= start && other < start + size)) promotion_fail(2);
    }
}

// Copy a headerless raw buffer (closure env, hash arrays, tensor buffers,
// C strings) into the target arena, with forwarding so shared/aliased buffers
// are copied once.
static void* evac_raw(EvacState& st, const void* old, size_t size) {
    if (!old) return nullptr;
    auto it = st.fwd->find(old);
    if (it != st.fwd->end()) {
        if (it->second.object || size > it->second.size) promotion_fail(2);
        return it->second.data;
    }
    evac_require_span(old, size);
    evac_check_overlap(st, old, size, false);
    void* raw = evac_allocate(st, size ? size : 1);
    if (size) std::memcpy(raw, old, size);
    else std::memset(raw, 0, 1);
    st.fwd->emplace(old, EvacForward{raw, size, false});
    st.copies++;
    return raw;
}

static void evac_validate_object(const void* data, size_t size, EvacKind kind, const eshkol_tagged_value_t& value) {
    auto need = [size](size_t n) { if (n > size) promotion_fail(2); };
    switch (kind) {
    case EVAC_CONS: need(sizeof(arena_tagged_cons_cell_t)); break;
    case EVAC_VECTOR: {
        need(8); const int64_t n = *static_cast<const int64_t*>(data);
        if (n < 0) promotion_fail(2);
        need(evac_size(static_cast<size_t>(n), sizeof(eshkol_tagged_value_t), 8)); break;
    }
    case EVAC_MULTIVALUE:
        need(sizeof(size_t));
        need(evac_size(*static_cast<const size_t*>(data), sizeof(eshkol_tagged_value_t), sizeof(size_t))); break;
    case EVAC_EXCEPTION: need(sizeof(eshkol_exception_t)); break;
    case EVAC_TENSOR: {
        need(offsetof(eshkol_tensor_t, dtype));
        if (size != offsetof(eshkol_tensor_t, dtype) && size != sizeof(eshkol_tensor_t)) promotion_fail(2);
        if (size == sizeof(eshkol_tensor_t)) {
            const auto dtype = static_cast<const eshkol_tensor_t*>(data)->dtype;
            if (dtype > ESHKOL_TENSOR_DTYPE_I8 && dtype != ESHKOL_TENSOR_DTYPE_DUAL) promotion_fail(2);
        }
        break; // legacy numeric producers have four fields
    }
    case EVAC_HASH: need(sizeof(eshkol_hash_table_t)); break;
    case EVAC_CLOSURE: need(sizeof(eshkol_closure_t)); break;
    case EVAC_PRIMITIVE: need(sizeof(eshkol_primitive_t)); break;
    case EVAC_SUBSTITUTION: {
        need(sizeof(eshkol_substitution_t)); auto* s = static_cast<const eshkol_substitution_t*>(data);
        if (s->num_bindings > s->capacity) promotion_fail(2);
        need(evac_size(s->capacity, sizeof(uint64_t) + sizeof(eshkol_tagged_value_t), sizeof(*s))); break;
    }
    case EVAC_FACT: {
        need(sizeof(eshkol_fact_t)); auto* f = static_cast<const eshkol_fact_t*>(data);
        need(evac_size(f->arity, sizeof(eshkol_tagged_value_t), sizeof(*f))); break;
    }
    case EVAC_KNOWLEDGE_BASE: {
        need(sizeof(eshkol_knowledge_base_t)); auto* k = static_cast<const eshkol_knowledge_base_t*>(data);
        if (k->num_facts > k->capacity) promotion_fail(2); break;
    }
    case EVAC_FACTOR_GRAPH: {
        need(sizeof(eshkol_factor_graph_t)); auto* f = static_cast<const eshkol_factor_graph_t*>(data);
        if (f->num_factors > f->max_factors) promotion_fail(2); break;
    }
    case EVAC_WORKSPACE: {
        need(sizeof(eshkol_workspace_t)); auto* w = static_cast<const eshkol_workspace_t*>(data);
        if (w->num_modules > w->max_modules) promotion_fail(2);
        need(evac_size(w->max_modules, sizeof(eshkol_workspace_module_t), sizeof(*w))); break;
    }
    case EVAC_PROMISE: need(40); break;
    case EVAC_RATIONAL: need(sizeof(eshkol_rational_t)); break;
    case EVAC_TAYLOR: {
        need(sizeof(esh_taylor_t)); auto* t = static_cast<const esh_taylor_t*>(data);
        const uint32_t type = t->flags & ESH_TAYLOR_COEFF_MASK;
        if (type != ESH_TAYLOR_COEFF_RATIONAL && type != ESH_TAYLOR_COEFF_F64) promotion_fail(2);
        need(evac_size(static_cast<size_t>(t->order_k) + 1,
            type == ESH_TAYLOR_COEFF_RATIONAL ? sizeof(eshkol_tagged_value_t) : sizeof(double), sizeof(*t))); break;
    }
    case EVAC_LEAF: {
        if (ESHKOL_IS_ANY_CALLABLE_TYPE(value.type)) { need(sizeof(ad_node_t)); break; }
        const uint8_t subtype = ESHKOL_GET_HEADER(data)->subtype;
        // Deep kinds cannot reach this arm; the exhaustive classifier above is
        // the authority for new kinds. Opaque layouts stay owned by producers.
        if (subtype == HEAP_SUBTYPE_STRING || subtype == HEAP_SUBTYPE_SYMBOL) {
            if (!std::memchr(data, 0, size)) promotion_fail(2);
        } else if (subtype == HEAP_SUBTYPE_BIGNUM) {
            need(sizeof(eshkol_bignum_t));
            const auto* n = static_cast<const eshkol_bignum_t*>(data);
            need(evac_size(n->num_limbs, sizeof(uint64_t), sizeof(*n)));
        } else if (subtype == HEAP_SUBTYPE_BYTEVECTOR) {
            need(sizeof(int64_t)); const auto n = *static_cast<const int64_t*>(data);
            if (n < 0) promotion_fail(2);
            need(evac_size(static_cast<size_t>(n), 1, sizeof(int64_t)));
        } else if (subtype == HEAP_SUBTYPE_I128) need(16);
        else if (subtype == HEAP_SUBTYPE_PRNG) need(sizeof(uint64_t));
        else if (subtype == HEAP_SUBTYPE_PARAMETER) need(eshkol_parameter_promotion_size());
        else if (subtype == HEAP_SUBTYPE_DNC) need(eshkol_dnc_promotion_size());
        else if (subtype == HEAP_SUBTYPE_SDNC) need(eshkol_sdnc_promotion_size());
        else promotion_fail(2); // unproved regional port/leaf layout
        break;
    }
    }
}

// Copy a header-prefixed object into the target arena (contiguous header+payload),
// register forwarding, and enqueue it for interior traversal if non-leaf.
static void* evac_object(EvacState& st, void* old_data, const eshkol_tagged_value_t& v) {
    auto it = st.fwd->find(old_data);
    if (it != st.fwd->end()) {
        if (!it->second.object) promotion_fail(2);
        return it->second.data;
    }

    auto* h = (eshkol_object_header_t*)((uint8_t*)old_data - sizeof(eshkol_object_header_t));
    if (region_index_owning(h) != region_index_owning(old_data)) promotion_fail(2);
    evac_require_span(h, sizeof(*h));
    const size_t total = evac_size(h->size, 1, sizeof(*h));
    if (!h->size) promotion_fail(2);
    evac_require_span(h, total);
    EvacKind k = evac_kind_for(v, old_data);
    evac_validate_object(old_data, h->size, k, v);
    evac_check_overlap(st, old_data, h->size, true);
    void* raw = evac_allocate(st, total);
    std::memcpy(raw, h, total);
    void* new_data = (uint8_t*)raw + sizeof(*h);
    st.fwd->emplace(old_data, EvacForward{new_data, h->size, true});
    ++st.copies;
    if (k != EVAC_LEAF) st.worklist.push_back({new_data, k});
    return new_data;
}

// Rewrite one tagged value: if it points into a dying region (strictly inner
// than the boundary), evacuate the pointed-to object and repoint; otherwise
// leave it untouched.
static eshkol_tagged_value_t evac_value(EvacState& st, eshkol_tagged_value_t v) {
    const uint8_t type = v.type;
    if (!region_value_carries_pointer(type)) return v;

    void* p = (void*)(uintptr_t)v.data.ptr_val;
    if (!p) return v;
    if (region_index_owning(p) <= st.boundary_idx) return v;  // stable relative to dst
    if (region_linear_resource(type)) promotion_fail(2); // cannot duplicate linear ownership

    // Headerless fixed-size payload (dual number / complex): a flat copy is the
    // whole object. evac_raw forwards, so two tagged values sharing one payload
    // still share it after promotion.
    if (const size_t raw_size = region_headerless_payload_size(type)) {
        v.data.ptr_val = (uint64_t)(uintptr_t)evac_raw(st, p, raw_size);
        return v;
    }

    void* np = evac_object(st, p, v);
    v.data.ptr_val = (uint64_t)(uintptr_t)np;
    return v;
}

// Evacuate a header-prefixed object referenced only by a RAW data pointer (no
// enclosing tagged value), e.g. the fact pointers held in a knowledge base's
// facts[] array. Synthesizes a plain HEAP_PTR tagged value so evac_object can
// classify it by its object header and drive the normal deep walk (forwarding,
// cycles, shared structure all preserved). Objects already at/outside the
// boundary are returned unchanged.
static void* evac_object_ptr(EvacState& st, void* data_ptr) {
    if (!data_ptr) return data_ptr;
    if (region_index_owning(data_ptr) <= st.boundary_idx) return data_ptr;
    eshkol_tagged_value_t synth;
    std::memset(&synth, 0, sizeof(synth));
    synth.type = ESHKOL_VALUE_HEAP_PTR;
    synth.data.ptr_val = (uint64_t)(uintptr_t)data_ptr;
    return evac_object(st, data_ptr, synth);
}

static void evac_walk(EvacState& st) {
    while (!st.worklist.empty()) {
        std::pair<void*, EvacKind> item = st.worklist.back();
        st.worklist.pop_back();
        void* nd = item.first;
        switch (item.second) {
            case EVAC_CONS: {
                auto* c = (arena_tagged_cons_cell_t*)nd;
                evac_assign(st, c->car, evac_value(st, c->car));
                evac_assign(st, c->cdr, evac_value(st, c->cdr));
                break;
            }
            case EVAC_VECTOR: {
                int64_t len = *(int64_t*)nd;
                if (len < 0) len = 0;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(int64_t));
                for (int64_t i = 0; i < len; ++i) evac_assign(st, elems[i], evac_value(st, elems[i]));
                break;
            }
            case EVAC_MULTIVALUE: {
                size_t count = *(size_t*)nd;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(size_t));
                for (size_t i = 0; i < count; ++i) evac_assign(st, elems[i], evac_value(st, elems[i]));
                break;
            }
            case EVAC_EXCEPTION: {
                auto* ex = (eshkol_exception_t*)nd;
                if (ex->message && region_index_owning(ex->message) > st.boundary_idx)
                    evac_assign(st, ex->message, (char*)evac_raw(st, ex->message, evac_string_size(ex->message)));
                if (ex->filename && region_index_owning(ex->filename) > st.boundary_idx)
                    evac_assign(st, ex->filename, (char*)evac_raw(st, ex->filename, evac_string_size(ex->filename)));
                if (ex->irritants &&
                    region_index_owning(ex->irritants) > st.boundary_idx) {
                    evac_assign(st, ex->irritants, (eshkol_tagged_value_t*)evac_raw(
                        st, ex->irritants, evac_size(ex->num_irritants, sizeof(eshkol_tagged_value_t))));
                }
                if (ex->irritants) {
                    evac_require_span(ex->irritants, evac_size(ex->num_irritants, sizeof(eshkol_tagged_value_t)));
                    for (uint32_t i = 0; i < ex->num_irritants; ++i)
                        evac_assign(st, ex->irritants[i], evac_value(st, ex->irritants[i]));
                }
                break;
            }
            case EVAC_TENSOR: {
                auto* t = (eshkol_tensor_t*)nd;
                const bool dual = ESHKOL_GET_HEADER(nd)->size >= sizeof(*t) && t->dtype == ESHKOL_TENSOR_DTYPE_DUAL;
                if (t->dimensions && region_index_owning(t->dimensions) > st.boundary_idx)
                    evac_assign(st, t->dimensions, (uint64_t*)evac_raw(
                        st, t->dimensions, evac_size(t->num_dimensions, sizeof(uint64_t))));
                if (t->elements && region_index_owning(t->elements) > st.boundary_idx)
                    evac_assign(st, t->elements, (int64_t*)evac_raw(
                        st, t->elements, evac_size(t->total_elements, dual ? sizeof(eshkol_tagged_value_t) : sizeof(int64_t))));
                if (dual && t->elements) {
                    auto* values = reinterpret_cast<eshkol_tagged_value_t*>(t->elements);
                    evac_require_span(values, evac_size(t->total_elements, sizeof(*values)));
                    for (uint64_t i = 0; i < t->total_elements; ++i)
                        evac_assign(st, values[i], evac_value(st, values[i]));
                }
                break;
            }
            case EVAC_HASH: {
                auto* tbl = (eshkol_hash_table_t*)nd;
                const size_t cap = tbl->capacity;
                if (tbl->keys && region_index_owning(tbl->keys) > st.boundary_idx)
                    evac_assign(st, tbl->keys, (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->keys, evac_size(cap, sizeof(eshkol_tagged_value_t))));
                if (tbl->values && region_index_owning(tbl->values) > st.boundary_idx)
                    evac_assign(st, tbl->values, (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->values, evac_size(cap, sizeof(eshkol_tagged_value_t))));
                if (tbl->status && region_index_owning(tbl->status) > st.boundary_idx)
                    evac_assign(st, tbl->status, (uint8_t*)evac_raw(st, tbl->status, evac_size(cap, sizeof(uint8_t))));
                if (tbl->keys && tbl->values && tbl->status) {
                    evac_require_span(tbl->keys, evac_size(cap, sizeof(eshkol_tagged_value_t)));
                    evac_require_span(tbl->values, evac_size(cap, sizeof(eshkol_tagged_value_t)));
                    evac_require_span(tbl->status, cap);
                    for (size_t i = 0; i < cap; ++i) {
                        if (tbl->status[i] == HASH_ENTRY_OCCUPIED) {
                            evac_assign(st, tbl->keys[i], evac_value(st, tbl->keys[i]));
                            evac_assign(st, tbl->values[i], evac_value(st, tbl->values[i]));
                        }
                    }
                }
                // Future resizes must grow the arrays in the surviving arena,
                // never the dying region.
                evac_assign(st, tbl->home_arena, st.target);
                break;
            }
            case EVAC_PRIMITIVE: {
                auto* p = static_cast<eshkol_primitive_t*>(nd);
                if (p->name && region_index_owning(p->name) > st.boundary_idx)
                    evac_assign(st, p->name, static_cast<const char*>(evac_raw(st, p->name, evac_string_size(p->name))));
                break;
            }
            case EVAC_CLOSURE: {
                auto* c = (eshkol_closure_t*)nd;
                if (c->sexpr_ptr)
                    evac_assign(st, c->sexpr_ptr, (uint64_t)(uintptr_t)evac_object_ptr(
                        st, (void*)(uintptr_t)c->sexpr_ptr));
                if (c->env) {
                    evac_require_span(c->env, sizeof(eshkol_closure_env_t));
                    const size_t ncap = CLOSURE_ENV_GET_NUM_CAPTURES(c->env->num_captures);
                    const size_t env_size = evac_size(ncap, sizeof(eshkol_tagged_value_t), sizeof(eshkol_closure_env_t));
                    evac_require_span(c->env, env_size);
                    auto* ne = c->env;
                    if (region_index_owning(ne) > st.boundary_idx)
                        ne = (eshkol_closure_env_t*)evac_raw(st, ne, env_size);
                    evac_assign(st, c->env, ne);
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
                                    evac_assign(st, *nc, evac_value(st, *nc));
                                    evac_assign(st, cap.data.int_val, (int64_t)(intptr_t)nc);
                                }
                                continue;
                            }
                        }
                        evac_assign(st, cap, evac_value(st, cap));
                    }
                }
                if (c->name && region_index_owning(c->name) > st.boundary_idx)
                    evac_assign(st, c->name, (const char*)evac_raw(st, c->name, evac_string_size(c->name)));
                break;
            }
            case EVAC_SUBSTITUTION: {
                // Layout [hdr][struct][var_ids u64[cap]][terms tagged[cap]] — all
                // inline (already copied contiguously). var_ids are plain ints;
                // only the bound terms carry heap pointers. Unused capacity slots
                // beyond num_bindings are uninitialized and must not be walked.
                auto* s = (eshkol_substitution_t*)nd;
                eshkol_tagged_value_t* terms = SUBST_TERMS(s);
                for (uint32_t i = 0; i < s->num_bindings; ++i)
                    evac_assign(st, terms[i], evac_value(st, terms[i]));
                break;
            }
            case EVAC_FACT: {
                // Layout [hdr][struct][args tagged[arity]] — args inline.
                auto* f = (eshkol_fact_t*)nd;
                eshkol_tagged_value_t* args = FACT_ARGS(f);
                for (uint32_t i = 0; i < f->arity; ++i)
                    evac_assign(st, args[i], evac_value(st, args[i]));
                // predicate is normally a pointer into the immortal interned
                // predicate pool (static storage -> region_index_owning == -1 ->
                // left in place, preserving pointer-equality matching). Only a
                // non-interned predicate string allocated inside the dying region
                // is copied (defensive; unify() has a string-compare fallback).
                if (f->predicate) {
                    void* pred = (void*)(uintptr_t)f->predicate;
                    if (region_index_owning(pred) > st.boundary_idx)
                        evac_assign(st, f->predicate, (uint64_t)(uintptr_t)evac_raw(
                            st, pred, evac_string_size((const char*)pred)));
                }
                break;
            }
            case EVAC_KNOWLEDGE_BASE: {
                // Layout [hdr][struct]; facts[] is a separate raw arena array of
                // FACT data pointers (capacity slots, num_facts used).
                auto* kb = (eshkol_knowledge_base_t*)nd;
                if (kb->facts && region_index_owning(kb->facts) > st.boundary_idx)
                    evac_assign(st, kb->facts, (eshkol_fact_t**)evac_raw(
                        st, kb->facts, evac_size(kb->capacity, sizeof(eshkol_fact_t*))));
                if (kb->facts) {
                    evac_require_span(kb->facts, evac_size(kb->capacity, sizeof(eshkol_fact_t*)));
                    for (uint32_t i = 0; i < kb->num_facts; ++i)
                        evac_assign(st, kb->facts[i], (eshkol_fact_t*)evac_object_ptr(st, kb->facts[i]));
                }
                break;
            }
            case EVAC_WORKSPACE: {
                // Layout [hdr][struct][modules module_t[max_modules]] — modules
                // inline (already copied). content is a separate raw double buffer;
                // each used module carries an arena name string + process_fn closure.
                auto* ws = (eshkol_workspace_t*)nd;
                if (ws->content && region_index_owning(ws->content) > st.boundary_idx)
                    evac_assign(st, ws->content, (double*)evac_raw(
                        st, ws->content, evac_size(ws->dim, sizeof(double))));
                eshkol_workspace_module_t* mods = WS_MODULES(ws);
                for (uint32_t i = 0; i < ws->num_modules; ++i) {
                    if (mods[i].name &&
                        region_index_owning(mods[i].name) > st.boundary_idx)
                        evac_assign(st, mods[i].name, (char*)evac_raw(
                            st, mods[i].name, evac_string_size(mods[i].name)));
                    evac_assign(st, mods[i].process_fn, evac_value(st, mods[i].process_fn));
                }
                break;
            }
            case EVAC_PROMISE: {
                // [forced:i64 @0][thunk:tagged @8][cached:tagged @24] (40 bytes,
                // llvm_codegen.cpp %make-lazy-promise / make-promise). Both the
                // thunk closure (unforced) and the memoized cached value (forced)
                // can be region-allocated; a promise that escapes its region via
                // delay/force/make-promise must have both slots evacuated or they
                // dangle into the freed arena after region_pop (ESH-0214d gap:
                // subtype 18 was previously leaf-copied). ESH-0214e.
                auto* thunk  = (eshkol_tagged_value_t*)((uint8_t*)nd + 8);
                auto* cached = (eshkol_tagged_value_t*)((uint8_t*)nd + 24);
                evac_assign(st, *thunk, evac_value(st, *thunk));
                evac_assign(st, *cached, evac_value(st, *cached));
                break;
            }
            case EVAC_RATIONAL: {
                // is_big == 0 (fast path): numerator/denominator are inline
                // int64 scalars -- already copied verbatim by the contiguous
                // header+payload copy, nothing further to walk.
                //
                // is_big == 1 (bignum path): big_num/big_den are RAW
                // eshkol_bignum_t* pointers (not tagged values -- same shape
                // as a knowledge base's facts[] array), each an independently
                // header-prefixed HEAP_SUBTYPE_BIGNUM object that can itself
                // be region-resident (e.g. the result of exact arithmetic
                // performed inside a `with-region` body). Left un-walked, a
                // rational whose reduced numerator/denominator overflow
                // int64 -- promoted out of its region via vector-set!/
                // set-car!/etc. -- would carry two pointers straight into the
                // arena region_pop is about to free.
                auto* r = (eshkol_rational_t*)nd;
                if (r->is_big) {
                    if (r->big_num) evac_assign(st, r->big_num, (eshkol_bignum_t*)evac_object_ptr(st, r->big_num));
                    if (r->big_den) evac_assign(st, r->big_den, (eshkol_bignum_t*)evac_object_ptr(st, r->big_den));
                }
                break;
            }
            case EVAC_TAYLOR: {
                // SW-66: only a COEFF_RATIONAL (exact) tower's c[] holds
                // interior tagged values -- reinterpret it exactly the way
                // runtime_taylor.c's taylor_exact_c()/taylor_exact_c_const()
                // do, and evac_value() each of the order_k+1 entries (most
                // are plain int64 scalars needing no work; any that are
                // HEAP_PTRs to a bignum/rational allocated in this same
                // region get walked and repointed, exactly like a vector's
                // element array). A COEFF_F64 tower's c[] is raw doubles,
                // already fully preserved by the contiguous header+payload
                // copy above -- nothing to do.
                //
                // The seed-tangent half (ESH_TAYLOR_TANGENT_FLAG) is never
                // combined with COEFF_RATIONAL by any producer in
                // runtime_taylor.c (the tangent series is only ever built
                // alongside COEFF_F64 towers), so it is deliberately not
                // considered here; if that combination is ever introduced,
                // its storage doubling would need its own case.
                auto* t = (esh_taylor_t*)nd;
                if ((t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_RATIONAL) {
                    auto* c = (eshkol_tagged_value_t*)(void*)t->c;
                    const size_t ncoeff = (size_t)t->order_k + 1;
                    for (size_t i = 0; i < ncoeff; ++i) evac_assign(st, c[i], evac_value(st, c[i]));
                }
                break;
            }
            case EVAC_FACTOR_GRAPH: {
                // A factor graph carries NO interior tagged values — only nested
                // raw numeric buffers (arrays and arrays-of-arrays). Sizes are
                // reconstructed from the graph's own counts, mirroring the
                // allocation layout in inference.cpp.
                auto* fg = (eshkol_factor_graph_t*)nd;
                const uint32_t nvars = fg->num_vars;
                if (fg->var_dims && region_index_owning(fg->var_dims) > st.boundary_idx)
                    evac_assign(st, fg->var_dims, (uint32_t*)evac_raw(
                        st, fg->var_dims, evac_size(nvars, sizeof(uint32_t))));
                if (fg->beliefs && region_index_owning(fg->beliefs) > st.boundary_idx)
                    evac_assign(st, fg->beliefs, (double**)evac_raw(
                        st, fg->beliefs, evac_size(nvars, sizeof(double*))));
                if (fg->beliefs && fg->var_dims) {
                    evac_require_span(fg->beliefs, evac_size(nvars, sizeof(double*)));
                    evac_require_span(fg->var_dims, evac_size(nvars, sizeof(uint32_t)));
                    for (uint32_t i = 0; i < nvars; ++i) {
                        if (fg->beliefs[i] &&
                            region_index_owning(fg->beliefs[i]) > st.boundary_idx)
                            evac_assign(st, fg->beliefs[i], (double*)evac_raw(
                                st, fg->beliefs[i], evac_size(fg->var_dims[i], sizeof(double))));
                    }
                }
                if (fg->observed && region_index_owning(fg->observed) > st.boundary_idx)
                    evac_assign(st, fg->observed, (bool*)evac_raw(
                        st, fg->observed, evac_size(nvars, sizeof(bool))));
                // factors[] holds max_factors ptrs (num_factors used). Each factor
                // is a HEADERLESS raw buffer [eshkol_factor_t][var_indices...] with
                // interior cpt/dims raw buffers.
                if (fg->factors && region_index_owning(fg->factors) > st.boundary_idx)
                    evac_assign(st, fg->factors, (eshkol_factor_t**)evac_raw(
                        st, fg->factors, evac_size(fg->max_factors, sizeof(eshkol_factor_t*))));
                if (fg->factors) {
                    evac_require_span(fg->factors, evac_size(fg->max_factors, sizeof(eshkol_factor_t*)));
                    for (uint32_t fi = 0; fi < fg->num_factors; ++fi) {
                        eshkol_factor_t* f = fg->factors[fi];
                        if (!f) continue;
                        evac_require_span(f, sizeof(*f));
                        if (region_index_owning(f) > st.boundary_idx) {
                            const size_t fsz = evac_size(f->num_vars, sizeof(uint32_t), sizeof(eshkol_factor_t));
                            f = (eshkol_factor_t*)evac_raw(st, f, fsz);
                            evac_assign(st, fg->factors[fi], f);
                        }
                        if (f->cpt && region_index_owning(f->cpt) > st.boundary_idx)
                            evac_assign(st, f->cpt, (double*)evac_raw(
                                st, f->cpt, evac_size(f->cpt_size, sizeof(double))));
                        if (f->dims && region_index_owning(f->dims) > st.boundary_idx)
                            evac_assign(st, f->dims, (uint32_t*)evac_raw(
                                st, f->dims, evac_size(f->num_vars, sizeof(uint32_t))));
                    }
                }
                // msg_fv / msg_vf: parallel arrays of total_messages double* (NULL
                // until eshkol_fg_infer allocates them). Edge k -> var_dims[var]
                // doubles, where the (factor, var-slot) -> edge order mirrors
                // ensure_messages() in inference.cpp (intentional coupling).
                auto evac_msg_array = [&](double**& arr) {
                    if (!arr) return;
                    if (region_index_owning(arr) > st.boundary_idx)
                        evac_assign(st, arr, (double**)evac_raw(
                            st, arr, evac_size(fg->total_messages, sizeof(double*))));
                    evac_require_span(arr, evac_size(fg->total_messages, sizeof(double*)));
                    uint32_t k = 0;
                    for (uint32_t fi = 0;
                         fi < fg->num_factors && k < fg->total_messages; ++fi) {
                        eshkol_factor_t* f = fg->factors ? fg->factors[fi] : nullptr;
                        if (!f) continue;
                        evac_require_span(f, sizeof(*f));
                        evac_require_span(f, evac_size(f->num_vars, sizeof(uint32_t), sizeof(*f)));
                        const uint32_t* vidx = FACTOR_VAR_INDICES(f);
                        for (uint32_t vi = 0;
                             vi < f->num_vars && k < fg->total_messages; ++vi, ++k) {
                            const uint32_t var_id = vidx[vi];
                            if (!fg->var_dims || var_id >= nvars) promotion_fail(2);
                            const uint32_t dim = fg->var_dims[var_id];
                            if (arr[k] &&
                                region_index_owning(arr[k]) > st.boundary_idx)
                                evac_assign(st, arr[k], (double*)evac_raw(
                                    st, arr[k], evac_size(dim, sizeof(double))));
                        }
                    }
                };
                evac_msg_array(fg->msg_fv);
                evac_msg_array(fg->msg_vf);
                break;
            }
            case EVAC_LEAF:
                // Never enqueued (region_evacuate only pushes non-leaf kinds),
                // but named rather than defaulted: this switch has no default,
                // so a new EvacKind cannot be added without deciding here how
                // its interior is walked. A default would let one be added and
                // silently walk nothing — which, for a kind added precisely
                // because it HAS interior pointers, is the ESH-0214d bug back
                // at the other end of the same pipeline.
                break;
        }
    }

}

// Root snapshot + all graph work + staged map precede the one publication.
static int32_t region_evacuate_batch(eshkol_tagged_value_t* out,
    const eshkol_tagged_value_t* values, size_t count, arena_t* target,
    int boundary_idx) noexcept {
    try {
        const size_t bytes = evac_size(count, sizeof(*values));
        if (!count) return 0;
        if (!out || !values) return 4;
        bool needed = false;
        for (size_t i = 0; i < count; ++i) {
            if (region_value_carries_pointer(values[i].type) && values[i].data.ptr_val &&
                region_index_owning((void*)(uintptr_t)values[i].data.ptr_val) > boundary_idx) {
                if (region_linear_resource(values[i].type)) return 2;
                needed = true;
                break;
            }
        }
        if (!needed) { std::memmove(out, values, bytes); return 0; }
        auto* owner = region_current();
        if (!owner || !target) return 4;
        auto* previous = owner->fwd_target == target ? static_cast<EvacFwdMap*>(owner->fwd_map) : nullptr;
        // A fully forwarded root is a no-allocation scalar path.
        if (count == 1 && previous) {
            auto found = previous->find((void*)(uintptr_t)values[0].data.ptr_val);
            if (found != previous->end()) {
                const size_t raw_size = region_headerless_payload_size(values[0].type);
                if ((raw_size && (found->second.object || raw_size > found->second.size)) ||
                    (!raw_size && !found->second.object)) return 2;
                auto value = values[0];
                value.data.ptr_val = (uint64_t)(uintptr_t)found->second.data;
                *out = value;
                return 0;
            }
        }
        EvacState st{target, boundary_idx};
        st.fwd = evac_new_map(previous);
        std::vector<eshkol_tagged_value_t, PromotionAllocator<eshkol_tagged_value_t, 3>> roots(values, values + count);
        for (auto& root : roots) root = evac_value(st, root);
        evac_walk(st);
        if (st.copies > std::numeric_limits<decltype(owner->escape_count)>::max() - owner->escape_count)
            promotion_fail(3);
        auto* retired = static_cast<EvacFwdMap*>(owner->fwd_map);
        owner->fwd_map = st.fwd;
        owner->fwd_target = target;
        owner->escape_count += st.copies;
        st.fwd = nullptr;
        st.committed = true;
        evac_delete_map(retired);
        std::memmove(out, roots.data(), bytes);
        return 0;
    } catch (const PromotionFailure& failure) { return failure.status; }
      catch (const std::bad_alloc&) { return 1; }
      catch (const std::length_error&) { return 3; }
}

#ifdef ESHKOL_PROMOTION_TESTING
void eshkol_promotion_test_reset() noexcept {
    promotion_fail_site = -2;
    promotion_fail_after = -1;
    promotion_fail_attempt = 0;
    for (auto& attempt : promotion_stats.attempts) attempt = 0;
    for (int i = 0; i < 4; ++i) promotion_stats.peak_bytes[i] = promotion_stats.live_bytes[i];
    promotion_stats.target_bytes = 0;
}
void eshkol_promotion_test_arm(int site, int64_t fail_after) noexcept {
    eshkol_promotion_test_reset();
    promotion_fail_site = site;
    promotion_fail_after = fail_after;
}
eshkol_promotion_test_stats eshkol_promotion_test_snapshot() noexcept { return promotion_stats; }
#endif

static void region_escape_batch_or_raise(eshkol_tagged_value_t* values, uint64_t count) {
    if (!values || !count) return; // preserved optional kept-values convention
    auto* current = region_current();
    if (!current) return;
    const int32_t status = region_evacuate_batch(values, values, count,
        region_escape_target(current), static_cast<int>(__region_stack_depth) - 2);
    if (status) eshkol_runtime_emergency_raise_v1(status);
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
 *            made), or @p val unchanged if no promotion is needed. Failure
 *            transfers a fixed emergency after transaction cleanup.
 */
static eshkol_tagged_value_t region_escape_tagged_value_impl(eshkol_tagged_value_t val) {
    if (!region_value_carries_pointer(val.type)) return val;

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
    eshkol_tagged_value_t staged;
    const int32_t status = region_evacuate_batch(&staged, &val, 1, target, boundary);
    if (status) eshkol_runtime_emergency_raise_v1(status);
    return staged;
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
 * hash-table-set!, set! of a global). The no-promotion path validates the call
 * and copies the tagged value without allocating transaction storage.
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
int32_t eshkol_region_copy_tagged_checked(eshkol_tagged_value_t* out,
    const void* dst, const eshkol_tagged_value_t* values, uint64_t count) noexcept {
    if (count > std::numeric_limits<size_t>::max() / sizeof(*values)) return 3;
    if (count && (!out || !values)) return 4;
    if (!count) return 0;
    const int boundary = region_index_owning(dst);
    arena_t* target = nullptr;
    if (__region_stack_depth) {
        target = boundary >= 0 ? __region_stack[boundary]->arena : __region_stack[0]->escape_base;
    }
    return region_evacuate_batch(out, values, static_cast<size_t>(count), target, boundary);
}
extern "C" int32_t eshkol_region_write_barrier_checked_v1(eshkol_tagged_value_t* out,
    const void* dst, const eshkol_tagged_value_t* value) {
    return eshkol_region_copy_tagged_checked(out, dst, value, 1);
}
extern "C" void eshkol_region_write_barrier_into(eshkol_tagged_value_t* out,
    const void* dst, const eshkol_tagged_value_t* value) {
    if (!out) return;
    if (!value) { std::memset(out, 0, sizeof(*out)); return; }
    eshkol_tagged_value_t staged;
    const int32_t status = eshkol_region_write_barrier_checked_v1(&staged, dst, value);
    if (status) eshkol_runtime_emergency_raise_v1(status);
    *out = staged;
}

// ───────────────────────────────────────────────────────────────────────────
// ESH-0214e: iter-scope PARTIAL RECLAMATION for mutating tick-loops.
//
// ESH-0214b's automatic per-iteration reclamation (arena_push_scope /
// eshkol_arena_iter_scope_end, runtime_arena_core.cpp) is all-or-nothing: the
// static gate (iterScopeSafeExpr) rejects a loop body outright the moment it
// contains any persistent mutation, because a value the iteration allocates and
// then stores into outer/persistent state would dangle when the scope is
// rewound, and the shallow out-value span test at the back edge cannot see that
// store. A resident tick loop that mutates persistent state EVERY tick
// therefore got NO reclamation and leaked one iteration's transient garbage per
// tick, unbounded (~3 KB/tick, linear).
//
// The fix does not fork a second evacuator or a second reclamation policy. It
// reuses, verbatim, the with-region machinery whose deep-transitive escape
// promotion has been validated over a 48-h resident run (ESH-0214c/d):
//
//   * The loop runs inside a NURSERY REGION (region_create/region_push/
//     eshkol_region_enter, opened once per activation), so every iteration
//     allocation lands in the nursery arena and region_index_owning() can
//     classify it. Each of the six mutation channels' EXISTING write barriers
//     (eshkol_region_write_barrier_into) then deep-promotes any barrier-recorded
//     escapee — a nursery value stored into persistent/outer state — out of the
//     nursery into the enclosing arena at the store, exactly as inside a
//     with-region body. The barrier IS the escape-root record: it evacuates
//     each escapee out of the tick arena the instant it becomes reachable from
//     persistent state.
//
//   * eshkol_iter_nursery_recycle (below) runs at each TCO back edge. It
//     promotes the loop-carried out-values (the freshly evaluated tail-call
//     args, which the barrier never sees) out of the nursery via the SAME
//     evacuator (region_escape_tagged_value_impl), then arena_reset()s the
//     nursery — reclaiming the whole iteration's transient garbage — as if no
//     mutation had happened. Its forwarding map (whose keys reference the arena
//     just reset) is dropped so no stale entry survives into the next
//     iteration's promotions.
//
// The reclamation is sound by the SAME invariant as with-region's region_pop:
// after promotion no surviving object holds a pointer into the region being
// freed (here, into the span being reset). This is textbook generational minor
// collection (Appel) — the nursery is the young generation, the write barrier
// is the remembered set, and recycle is the minor collection — but fully
// deterministic (bounded per-iteration work, no tracing pause), which preserves
// Eshkol's no-GC determinism.
//
// FALLBACK (correctness over reclamation): if the region stack is not in the
// expected shape at a back edge (the nursery is not the innermost active
// region — an unbalanced region op in the body, which well-formed codegen never
// emits), recycle RETAINS this iteration (skips promotion AND reset) rather than
// evacuate from the wrong region. Retention is bounded: the next well-formed
// back edge resets the whole arena, reclaiming the retained bytes too.
// ───────────────────────────────────────────────────────────────────────────

extern "C" int eshkol_arena_poison_enabled(void);

/**
 * @brief End-of-iteration recycle for an ESH-0214e nursery loop (see block above).
 *
 * @param region  The loop's nursery region (opened once at loop entry).
 * @param vals    The loop-carried out-values (tail-call args) about to flow into
 *                the next iteration; each is promoted out of the nursery IN
 *                PLACE so the caller stores the promoted (surviving) value into
 *                the loop's parameter slot.
 * @param n       Number of out-values.
 */
extern "C" void eshkol_iter_nursery_recycle(eshkol_region_t* region,
                                            eshkol_tagged_value_t* vals,
                                            uint64_t n) {
    if (!region) return;

    // Fallback guard: only recycle when this nursery is the innermost active
    // region, i.e. the iteration body left the region stack exactly as it found
    // it. Otherwise retain (skip promotion + reset) — never evacuate from the
    // wrong region. Bounded: a later well-formed recycle reclaims these bytes.
    if (region_current() != region) return;

    // 1. Promote every loop-carried out-value out of the nursery into the
    //    enclosing arena (region->escape_base). This is the identical evacuator
    //    and code path as with-region's result escape
    //    (region_escape_tagged_value); it shares the nursery region's forwarding
    //    map, so any structure an out-value shares with a barrier-promoted
    //    escapee from the same iteration stays shared (eq?-preserving). Values
    //    already living outside the nursery (immediates, or pointers into the
    //    enclosing/global arena) are returned unchanged and cost only a region
    //    ownership probe.
    region_escape_batch_or_raise(vals, n);

    // 2. Under the poison allocator, stamp the bytes about to be recycled with
    //    0xCB so any interior pointer we FAILED to promote out (a missed escape
    //    root) dereferences an obvious 0xCB.. address and crashes loudly instead
    //    of silently reading recycled memory — the dangling-pointer tripwire the
    //    ESH-0214 memory-model tests rely on.
    if (eshkol_arena_poison_enabled() && region->arena) {
        for (arena_block_t* b = region->arena->current_block; b; b = b->next) {
            if (b->used) std::memset(b->memory, 0xCB, b->used);
        }
    }

    // 3. Reset the nursery arena: reclaim ALL of this iteration's transient
    //    garbage (bump pointer back to zero, free any spilled blocks, keep the
    //    first block). O(1) on the steady-state single-block case — no per-
    //    iteration malloc/free churn (the reason a persistent nursery + reset is
    //    used instead of region_create/destroy per iteration).
    arena_reset(region->arena);

    // 4. Drop the forwarding map: its keys reference the arena just reset and
    //    MUST NOT alias a fresh object that reuses the same address next
    //    iteration. It is lazily recreated on the next escape/barrier promotion.
    region_free_fwd_map(region);
}

// ───────────────────────────────────────────────────────────────────────────
// #341: USER-REACHABLE REGION HANDLES — non-lexical scoped reclamation.
//
// See the block comment over the declarations in arena_memory.h for the surface
// and the safety contract. This half is the mechanism.
//
// WHY A GENERATION-TAGGED INTEGER AND NOT A HEAP HANDLE OBJECT. The whole point
// of the API is a hot loop — a training step — so the handle must cost zero
// allocation: a heap handle object would land in the ENCLOSING arena (it has to,
// or closing the region would free the handle naming it) and leak one object per
// iteration for the life of the process, reintroducing in miniature exactly the
// unbounded growth #341 is about. An immediate integer allocates nothing. The
// generation counter then buys the safety property a raw pointer cannot have:
// every stale token is *detectably* stale, so double-close / use-after-close /
// a fabricated integer all fail validation and raise instead of dereferencing
// freed memory. This is the standard slot+generation handle, chosen here for the
// memory-safety property rather than for indirection.
//
// WHY ONE TEARDOWN PATH. eshkol_region_unwind_to() below is the ONLY code that
// takes a region down. Explicit close, out-of-order close, `with-region` exit, a
// raise crossing an open region, and a continuation escape all funnel through
// it, so the promotion-then-restore-then-pop ordering — and the evacuator that
// performs the promotion — cannot drift between the structured and unstructured
// surfaces. That ordering matters: the kept values are promoted while the region
// is still current (the evacuator resolves ownership off the live region stack),
// the allocation slot is restored BEFORE the arena is destroyed (so no
// allocation can ever target a freed arena), and only then is the region popped.
// ───────────────────────────────────────────────────────────────────────────

extern "C" void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

namespace {

/** @brief One user-reachable region-handle slot (thread-local, see t_region_handles). */
struct RegionHandleSlot {
    eshkol_region_t* region;  ///< The open region; NULL for a bookkeeping-only handle.
    uint64_t generation;      ///< Bumped on every close; a token carrying a stale
                              ///< generation can never validate again. Never 0 once used.
    uint64_t region_depth;    ///< __region_stack_depth immediately after region_push (1-based).
    uint64_t open_seq;        ///< Monotonic open order; the unwind mark on substrates
                              ///< with no region stack (see eshkol_region_handle_seq_mark).
    uint8_t reclaim;          ///< 1 = owns a real region; 0 = bookkeeping-only (VM contract).
    uint8_t in_use;           ///< 1 while the slot names an open handle.
};

}  // namespace

// Per-thread, matching the region stack's thread-locality: a handle is only
// meaningful on the thread that opened it (a token carried to another thread
// finds an empty slot table there and fails validation cleanly, which is the
// right answer — the other thread's region stack does not contain that region).
static thread_local RegionHandleSlot t_region_handles[ESHKOL_MAX_REGION_HANDLES];

// Monotonic open counter. Used as the unwind mark on substrates that have no
// region stack to measure depth against (see eshkol_region_handle_seq_mark).
static thread_local uint64_t t_region_handle_seq = 0;

// Token layout: low RH_SLOT_BITS bits hold slot+1 (so 0 is never a valid token
// and a zeroed/uninitialised integer cannot masquerade as a handle), the rest
// hold the slot's generation. ESHKOL_MAX_REGION_HANDLES is 64, so slot+1 is
// 1..64 and fits comfortably; the generation gets the remaining 55 bits.
#define RH_SLOT_BITS 8
#define RH_SLOT_MASK ((int64_t)((1 << RH_SLOT_BITS) - 1))

/**
 * @brief Validate @p token and resolve it to a slot index.
 * @return true iff the token names a currently-open handle on this thread.
 */
static bool rh_decode(int64_t token, int* slot_out) {
    if (token <= 0) return false;
    const int slot = (int)(token & RH_SLOT_MASK) - 1;
    if (slot < 0 || slot >= ESHKOL_MAX_REGION_HANDLES) return false;
    const RegionHandleSlot& s = t_region_handles[slot];
    if (!s.in_use) return false;
    if ((uint64_t)(token >> RH_SLOT_BITS) != s.generation) return false;
    if (slot_out) *slot_out = slot;
    return true;
}

/**
 * @brief Retire a handle slot, invalidating every token that named it.
 *
 * Bumping the generation is what makes double-close and use-after-close safe:
 * the caller's now-stale token no longer matches, so a second close raises a
 * clean error instead of tearing down whatever region happens to occupy that
 * stack position later.
 */
static void rh_retire(int slot) {
    RegionHandleSlot& s = t_region_handles[slot];
    s.in_use = 0;
    s.region = nullptr;
    s.region_depth = 0;
    s.open_seq = 0;
    s.reclaim = 0;
    s.generation++;
    if (s.generation == 0) s.generation = 1;  // 0 is the never-used sentinel
}

/** @brief Retire handles only at the current, successfully staged region depth. */
static void rh_retire_at(uint64_t depth) {
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        const RegionHandleSlot& s = t_region_handles[i];
        if (s.in_use && s.reclaim && s.region_depth == depth) rh_retire(i);
    }
}

/** @brief Region-stack depth — the mark form recorded by exception handlers and continuations. */
extern "C" uint64_t eshkol_region_mark(void) {
    return __region_stack_depth;
}

extern "C" uint64_t eshkol_region_handle_seq_mark(void) {
    return t_region_handle_seq;
}

extern "C" void eshkol_region_handle_seq_unwind_to(uint64_t mark) {
    // Only bookkeeping-only handles are retired here. A reclaiming handle's
    // teardown must go through the region stack (eshkol_region_unwind_to) so its
    // arena is actually freed and the allocation slot restored — retiring it by
    // sequence number would drop the token while leaving the region open.
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        const RegionHandleSlot& s = t_region_handles[i];
        if (s.in_use && !s.reclaim && s.open_seq > mark) rh_retire(i);
    }
}

extern "C" void eshkol_region_unwind_to(uint64_t mark,
                                        eshkol_tagged_value_t* vals,
                                        uint64_t n) {
    if (__region_stack_depth <= mark) return;

    while (__region_stack_depth > mark) {
        eshkol_region_t* region = __region_stack[__region_stack_depth - 1];

        // 1. Promote the kept / in-flight values ONE LEVEL out, using the same
        //    deep-transitive evacuator (interior-pointer walk, forwarding map,
        //    ESH-0214c/d subtype coverage) that with-region's result escape
        //    uses. Level-by-level rather than straight to the target arena so a
        //    multi-level cascade lands each object in the arena that genuinely
        //    outlives it, and so shared structure stays shared via each
        //    region's own forwarding map.
        region_escape_batch_or_raise(vals, n);
        // Promotion succeeded for this level only; no live handle is retired
        // on a failed attempt. Static emergency retry takes the noalloc path.
        rh_retire_at(__region_stack_depth);

        // 2. Restore the allocation slot BEFORE the arena dies. If this is
        //    skipped, the slot keeps pointing at freed memory and the very next
        //    allocation writes into it. No-op when enter declined to hijack.
        if (region) eshkol_region_leave(region->entry_saved_arena);

        // 3. Pop and destroy: frees the region arena (poisoning it with 0xCB
        //    first when ESHKOL_ARENA_POISON is set, see arena_destroy), so any
        //    value we failed to promote out reads as an obvious sentinel rather
        //    than as plausible stale data.
        region_pop();
    }
}

extern "C" const char* eshkol_region_handle_status_message(int status) {
    switch (status) {
        case ESHKOL_RH_OK:
            return "region handle ok";
        case ESHKOL_RH_ERR_STALE:
            return "region-close: invalid or already-closed region handle";
        case ESHKOL_RH_ERR_NOT_LIVE:
            return "region-close: region handle no longer names a live region";
        case ESHKOL_RH_ERR_TOO_MANY:
            return "region-open: too many open region handles — a region-close is missing";
        case ESHKOL_RH_ERR_DEPTH:
            return "region-open: region stack overflow — a region-close is missing";
        case ESHKOL_RH_ERR_CREATE:
            return "region-open: failed to create region";
        default:
            return "region handle: unknown error";
    }
}

extern "C" int64_t eshkol_region_handle_open(const char* name, uint64_t size_hint,
                                             int reclaim, int* status) {
    if (status) *status = ESHKOL_RH_OK;

    int slot = -1;
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        if (!t_region_handles[i].in_use) { slot = i; break; }
    }
    if (slot < 0) {
        // The bounded outcome for "never closed": a loop that opens without
        // closing hits this after ESHKOL_MAX_REGION_HANDLES iterations and gets a
        // clean catchable error naming the cause, rather than growing without
        // bound or corrupting anything.
        if (status) *status = ESHKOL_RH_ERR_TOO_MANY;
        return 0;
    }

    RegionHandleSlot& s = t_region_handles[slot];
    if (reclaim) {
        if (__region_stack_depth >= MAX_REGION_DEPTH) {
            if (status) *status = ESHKOL_RH_ERR_DEPTH;
            return 0;
        }
        eshkol_region_t* region = region_create(name, size_hint);
        if (!region) {
            if (status) *status = ESHKOL_RH_ERR_CREATE;
            return 0;
        }
        region_push(region);
        if (region_current() != region) {
            // region_push declined (it logs and leaves the stack untouched).
            // Drop the orphan rather than handing back a handle to a region that
            // is not on the stack.
            region_destroy(region);
            if (status) *status = ESHKOL_RH_ERR_CREATE;
            return 0;
        }
        // Redirect allocation into the new region and record the displaced arena
        // on the region (eshkol_region_enter stores it) so close/unwind can
        // restore it without a lexical register.
        (void)eshkol_region_enter(region);
        s.region = region;
        s.region_depth = __region_stack_depth;
        s.reclaim = 1;
    } else {
        // Bookkeeping-only contract (bytecode VM): the handle protocol, its
        // validation and its error text are identical, but no region is created
        // and close reclaims nothing. See tests/vm_parity/PARITY.tsv.
        s.region = nullptr;
        s.region_depth = 0;
        s.reclaim = 0;
    }

    if (s.generation == 0) s.generation = 1;
    s.open_seq = ++t_region_handle_seq;
    s.in_use = 1;
    return (int64_t)((s.generation << RH_SLOT_BITS) | (uint64_t)(slot + 1));
}

extern "C" int eshkol_region_handle_close(int64_t token,
                                          eshkol_tagged_value_t* vals,
                                          uint64_t n) {
    int slot = -1;
    if (!rh_decode(token, &slot)) {
        // Covers double-close, use-after-close, a token from another thread, and
        // a fabricated integer — all the same clean, catchable failure.
        return ESHKOL_RH_ERR_STALE;
    }

    RegionHandleSlot& s = t_region_handles[slot];

    if (!s.reclaim) {
        // Bookkeeping-only handle: values already live in the substrate heap and
        // need no promotion. But the observable token protocol must still match
        // the reclaiming substrate exactly, which means an out-of-order close has
        // to CASCADE here too: closing an outer handle retires every handle
        // opened after it. The reclaiming path gets that ordering from the region
        // stack; with no region stack to consult, the monotonic open sequence is
        // the same ordering. (Caught by tests/vm_parity/corpus/
        // region_handle_contract.esk, which observed the inner handle still live
        // on the VM after the outer one was closed.)
        const uint64_t seq = s.open_seq;
        for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
            const RegionHandleSlot& o = t_region_handles[i];
            if (o.in_use && !o.reclaim && o.open_seq >= seq) rh_retire(i);
        }
        return ESHKOL_RH_OK;
    }

    // The region must still occupy the stack position it was opened at. If it
    // does not, an earlier teardown already removed it and this token is stale
    // in a way rh_decode could not see; refuse rather than unwind a stack shape
    // we no longer recognise.
    if (s.region_depth == 0 || s.region_depth > __region_stack_depth ||
        __region_stack[s.region_depth - 1] != s.region) {
        rh_retire(slot);
        return ESHKOL_RH_ERR_NOT_LIVE;
    }

    // Closing an OUTER handle while inner regions are still open is a defined
    // cascade, not an error: unwind_to closes every region above this one,
    // innermost first, promoting the kept values out at every level and retiring
    // the inner handles' tokens. This is deliberately the same operation a
    // non-local exit performs, so there is one teardown path and no ordering
    // rule for callers to get wrong.
    eshkol_region_unwind_to(s.region_depth - 1, vals, n);
    return ESHKOL_RH_OK;
}

extern "C" int eshkol_region_handle_live(int64_t token) {
    return rh_decode(token, nullptr) ? 1 : 0;
}

// ── Surface entry points: (region-open …) / (region-close …) / (region-open? …)
//
// One C function per builtin, shared VERBATIM by both substrates so the handle
// protocol, the argument coercions and — critically — the error message text
// cannot diverge between native and VM. Optional arguments arrive as null
// pointers rather than as sentinel values, which keeps every arity of
// `region-open` on a single implementation. This mirrors the uniform
// cross-substrate contract established for the bytevector bounds checks.

/** @brief Debug label from a tagged symbol or string, else NULL. Never copies. */
static const char* rh_label_of(const eshkol_tagged_value_t* v) {
    if (!v) return nullptr;
    const uint8_t t = v->type;
    if (t == ESHKOL_VALUE_SYMBOL || t == ESHKOL_VALUE_STRING_PTR) {
        return (const char*)(uintptr_t)v->data.ptr_val;
    }
    if (t == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val) {
        const auto* hdr = (const eshkol_object_header_t*)
            ((const uint8_t*)(uintptr_t)v->data.ptr_val - sizeof(eshkol_object_header_t));
        if (hdr->subtype == HEAP_SUBTYPE_STRING) {
            return (const char*)(uintptr_t)v->data.ptr_val;
        }
    }
    return nullptr;
}

/** @brief True when @p v is a number (so a lone `region-open` argument reads as a size hint). */
static bool rh_is_number(const eshkol_tagged_value_t* v) {
    if (!v) return false;
    return v->type == ESHKOL_VALUE_INT64 || v->type == ESHKOL_VALUE_DOUBLE;
}

/** @brief Non-negative byte count from a tagged int/double, else 0 (= default size). */
static uint64_t rh_size_of(const eshkol_tagged_value_t* v) {
    if (!v) return 0;
    if (v->type == ESHKOL_VALUE_INT64) {
        return v->data.int_val > 0 ? (uint64_t)v->data.int_val : 0;
    }
    if (v->type == ESHKOL_VALUE_DOUBLE) {
        return v->data.double_val > 0 ? (uint64_t)v->data.double_val : 0;
    }
    return 0;
}

/**
 * @brief `(region-open)` / `(region-open name-or-size)` / `(region-open name size)`.
 *
 * A lone NUMERIC argument is the size hint; a lone non-numeric argument is the
 * debug name. @p reclaim selects the substrate contract (see
 * eshkol_region_handle_open).
 *
 * @param out     Receives the handle token as an exact integer.
 * @param a,b     Optional arguments; NULL for an argument that was not supplied.
 * @param reclaim Non-zero on a substrate that can actually reclaim (native).
 */
extern "C" void eshkol_region_open_builtin(eshkol_tagged_value_t* out,
                                           const eshkol_tagged_value_t* a,
                                           const eshkol_tagged_value_t* b,
                                           int reclaim) {
    if (!out) return;
    const char* name = nullptr;
    uint64_t size_hint = 0;
    if (a && b) {
        name = rh_label_of(a);
        size_hint = rh_size_of(b);
    } else if (a) {
        if (rh_is_number(a)) size_hint = rh_size_of(a);
        else name = rh_label_of(a);
    }
    int status = ESHKOL_RH_OK;
    const int64_t token = eshkol_region_handle_open(name, size_hint, reclaim, &status);
    if (status != ESHKOL_RH_OK) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(status));
        return;
    }
    *out = eshkol_make_int64(token, true);
}

/**
 * @brief `(region-close handle v ...)` — close the region, hand back the kept values.
 *
 * Returns the single promoted value for one keep, a freshly consed list of the
 * promoted values for several, and an empty list for none. The result list is
 * built AFTER the unwind, so it is allocated in the surviving (enclosing) arena
 * rather than in the region being torn down.
 *
 * @param out    Receives the result.
 * @param handle The handle token.
 * @param vals   The keep list, promoted IN PLACE by the unwind (may be NULL when n == 0).
 * @param n      Number of keeps.
 */
extern "C" void eshkol_region_close_builtin(eshkol_tagged_value_t* out,
                                            const eshkol_tagged_value_t* handle,
                                            eshkol_tagged_value_t* vals,
                                            uint64_t n) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_NULL;

    if (!handle || handle->type != ESHKOL_VALUE_INT64) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(ESHKOL_RH_ERR_STALE));
        return;
    }

    const int status = eshkol_region_handle_close(handle->data.int_val, vals, n);
    if (status != ESHKOL_RH_OK) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(status));
        return;
    }

    if (n == 1 && vals) {
        *out = vals[0];
    } else if (n > 1 && vals) {
        // Build the result list back-to-front in the arena that survived the
        // close (the slot has already been restored by the unwind).
        eshkol_tagged_value_t acc;
        std::memset(&acc, 0, sizeof(acc));
        acc.type = ESHKOL_VALUE_NULL;
        for (uint64_t i = n; i > 0; --i) {
            arena_tagged_cons_cell_t* cell =
                arena_allocate_tagged_cons_cell(get_global_arena());
            if (!cell) {
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                                     "region-close: out of memory building result list");
                return;
            }
            cell->car = vals[i - 1];
            cell->cdr = acc;
            std::memset(&acc, 0, sizeof(acc));
            acc.type = ESHKOL_VALUE_CONS_PTR;
            acc.data.ptr_val = (uint64_t)(uintptr_t)cell;
        }
        *out = acc;
    }
}

/**
 * @brief `(region-open? handle)` — #t while @p handle names a live open region.
 *
 * Never raises: this is the probe that lets cleanup code be written idempotently
 * (and the one the safety tests use to observe that an unwind closed a handle).
 */
extern "C" void eshkol_region_open_p_builtin(eshkol_tagged_value_t* out,
                                             const eshkol_tagged_value_t* handle) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_BOOL;
    const bool live = handle && handle->type == ESHKOL_VALUE_INT64 &&
                      eshkol_region_handle_live(handle->data.int_val) != 0;
    out->data.int_val = live ? 1 : 0;
}
