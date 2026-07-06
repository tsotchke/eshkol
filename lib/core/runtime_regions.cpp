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

static void init_global_arena_internal() {
    __global_arena = arena_create_threadsafe(65536);
    if (!__global_arena) {
        eshkol_error("Failed to create global arena");
    }
}

arena_t* get_global_arena() {
    eshkol_arena_global_once(init_global_arena_internal);
    if (__thread_local_arena) return __thread_local_arena;
    return __global_arena;
}

arena_t* get_global_arena_shared() {
    eshkol_arena_global_once(init_global_arena_internal);
    return __global_arena;
}

arena_t* arena_get_thread_local(void) {
    if (__thread_local_arena) return __thread_local_arena;
    return get_global_arena();
}

arena_t* arena_create_thread_local(size_t size_hint) {
    if (__thread_local_arena) return __thread_local_arena;

    const size_t block_size = size_hint > 0 ? size_hint : (1024 * 1024);
    __thread_local_arena = arena_create(block_size);
    return __thread_local_arena;
}

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

eshkol_region_t* region_current(void) {
    if (__region_stack_depth == 0) return nullptr;
    return __region_stack[__region_stack_depth - 1];
}

void* region_allocate(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate(region->arena, size);
    }
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

arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_tagged_cons_cell(region->arena);
    }
    return arena_allocate_tagged_cons_cell(get_global_arena());
}

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

static arena_t* region_escape_target(eshkol_region_t* current) {
    if (current->parent && current->parent->arena) {
        return current->parent->arena;
    }
    return get_global_arena();
}

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

extern "C" eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val) {
    return region_escape_tagged_value_impl(val);
}

extern "C" void region_escape_tagged_value_into(eshkol_tagged_value_t* out,
                                                const eshkol_tagged_value_t* val) {
    if (!out) return;
    if (!val) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    *out = region_escape_tagged_value_impl(*val);
}
