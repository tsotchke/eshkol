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

void eshkol_arena_global_once(void (*init)(void));

// Thread-local region stack (safe for parallel-map + with-region).
thread_local eshkol_region_t* __region_stack[MAX_REGION_DEPTH] = {nullptr};
thread_local uint64_t __region_stack_depth = 0;

// Default global arena for allocations outside any region.
// Non-static to allow JIT code to access it directly. Weak where object format
// support lets generated standalone code override it.
ESHKOL_RUNTIME_WEAK arena_t* __global_arena = nullptr;

static thread_local arena_t* __thread_local_arena = nullptr;

/**
 * @brief One-time initializer that creates the process-wide global arena.
 *
 * Invoked exactly once (via eshkol_arena_global_once) to create a
 * thread-safe arena and store it in __global_arena; logs an error if
 * creation fails.
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

/**
 * @brief Copy a tagged value's heap payload (if any) out of the current region.
 *
 * Non-heap values (ints, doubles, etc.) and non-pointer/port heap flags are
 * returned unchanged, as are heap values that are NULL pointers or whose
 * object header reports zero size. Otherwise reads the object header
 * immediately preceding the value's payload, copies the header plus payload
 * as one contiguous block into region_escape_target()'s arena, rewrites the
 * returned value's pointer to the copy's payload (past the copied header),
 * and increments the current region's escape_count. If no region is
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

    auto* header = (eshkol_object_header_t*)((uint8_t*)ptr - sizeof(eshkol_object_header_t));
    const size_t obj_size = header->size;
    if (obj_size == 0) return val;

    const size_t total = sizeof(eshkol_object_header_t) + obj_size;
    void* raw = arena_allocate_aligned(region_escape_target(current), total, 8);
    if (!raw) {
        eshkol_error("region_escape_tagged_value: failed to allocate %zu bytes", total);
        return val;
    }

    std::memcpy(raw, header, total);
    val.data.ptr_val = (uint64_t)(uintptr_t)((uint8_t*)raw + sizeof(eshkol_object_header_t));
    current->escape_count++;
    return val;
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
