// Single-threaded observation only. The enclosing driver judges recorded snapshots.
#include "lib/core/arena_memory.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

static_assert(sizeof(eshkol_object_header_t) == 8 && sizeof(arena_tagged_cons_cell_t) == 32);
static_assert(MAX_REGION_DEPTH == 64);
static int phase = -1;
static uintptr_t address(const void* p) { return reinterpret_cast<uintptr_t>(p); }

static void emit(int event, const void* object, const arena_block_t* block) {
    auto* route = eshkol_current_arena();
    std::printf("REGION %d %" PRIuPTR " %" PRIuPTR " %" PRIuPTR " %" PRIu64
                " %" PRIuPTR " %" PRIuPTR " %zu %zu",
                event, address(__global_arena), address(eshkol_memctx_current()->allocation_domain),
                address(route), __region_stack_depth, address(object),
                block ? address(block->memory) : 0, block ? block->used : 0, block ? block->size : 0);
    for (auto* slot : __region_stack) std::printf(" %" PRIuPTR, address(slot));
    std::puts("");
    phase = event;
}

extern "C" void nested_boundary(int64_t event) {
    emit(event, nullptr, nullptr);
}

extern "C" void nested_pair(int64_t event, const void* pair) {
    auto* arena = eshkol_current_arena();
    if (!arena || !arena->current_block) std::abort();
    emit(event, pair, arena->current_block);
}

extern "C" void __real_arena_destroy(arena_t*);
extern "C" void __wrap_arena_destroy(arena_t* arena) {
    if (phase != 2 && phase != 6) {
        __real_arena_destroy(arena);
        return;
    }
    const int entry = phase + 1;
    if (!arena || !arena->current_block || arena->current_block->next || arena->adopted_blocks)
        std::abort();
    emit(entry, arena, arena->current_block);
    __real_arena_destroy(arena);
    emit(entry+1, nullptr, nullptr);
}
