/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef ESHKOL_CORE_ARENA_H
#define ESHKOL_CORE_ARENA_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arena eshkol_arena_t;

/*
 * Create a heap-backed arena that may grow by allocating additional blocks.
 * This is the hosted/embedding path and is not suitable as the only bootstrap
 * story for freestanding startup code.
 */
eshkol_arena_t* eshkol_arena_create_heap(size_t default_block_size);

/*
 * Return the minimum buffer size needed to bootstrap an embedded arena with at
 * least the requested initial block capacity. The implementation may clamp the
 * block capacity upward to its runtime minimum.
 */
size_t eshkol_arena_embedded_bytes(size_t initial_block_size);

/*
 * Bootstrap an arena inside caller-owned storage. Embedded arenas do not grow
 * via heap allocation; once the initial block is exhausted, allocations fail.
 */
eshkol_arena_t* eshkol_arena_init_embedded(void* buffer, size_t buffer_size);

void eshkol_arena_destroy(eshkol_arena_t* arena);
void eshkol_arena_reset(eshkol_arena_t* arena);

void* eshkol_arena_allocate(eshkol_arena_t* arena, size_t size);
void* eshkol_arena_allocate_aligned(eshkol_arena_t* arena, size_t size,
                                    size_t alignment);
void* eshkol_arena_allocate_zeroed(eshkol_arena_t* arena, size_t size);

size_t eshkol_arena_used_bytes(const eshkol_arena_t* arena);
size_t eshkol_arena_total_bytes(const eshkol_arena_t* arena);
bool eshkol_arena_supports_heap_growth(const eshkol_arena_t* arena);

/*
 * Bind the process-global runtime arena used by generated code and runtime
 * metadata. Rebinding to a different arena is rejected. Freestanding source
 * init hooks do this internally before running global initialization.
 */
bool eshkol_arena_bind_runtime_global(eshkol_arena_t* arena);

#ifdef __cplusplus
}
#endif

#endif  // ESHKOL_CORE_ARENA_H
