#include "../../lib/backend/vm_arena.h"

static int rejected_without_mutation(VmArena* arena, size_t bytes) {
    VmArenaBlock* block = arena->current;
    size_t used = block->used;
    size_t total = arena->total_used;
    size_t allocated = arena->total_allocated;
    int blocks = arena->n_blocks;
    if (vm_arena_alloc(arena, bytes) != NULL || arena->current != block ||
        block->used != used || arena->total_used != total ||
        arena->total_allocated != allocated || arena->n_blocks != blocks) {
        fprintf(stderr, "FAIL: overflowing arena request %zu was accepted or changed arena state\n", bytes);
        return 1;
    }
    return 0;
}

int main(void) {
    VmArena arena;
    vm_arena_init(&arena, 128);
    if (!arena.current) return 1;
    if (rejected_without_mutation(&arena, SIZE_MAX) ||
        rejected_without_mutation(&arena, SIZE_MAX - (VM_ARENA_ALIGNMENT - 2)) ||
        rejected_without_mutation(&arena, SIZE_MAX / 2 + VM_ARENA_ALIGNMENT)) {
        vm_arena_destroy(&arena);
        return 1;
    }
    for (size_t bytes = 1; bytes <= 33; ++bytes) {
        unsigned char* p = vm_arena_alloc(&arena, bytes);
        if (!p || (uintptr_t)p % VM_ARENA_ALIGNMENT != 0 ||
            arena.current->used > arena.current->size) {
            fprintf(stderr, "FAIL: ordinary aligned allocation %zu\n", bytes);
            vm_arena_destroy(&arena);
            return 1;
        }
        memset(p, (int)bytes, bytes);
    }
    vm_arena_destroy(&arena);
    puts("PASS: checked arena requests reject overflow without mutation; 33 valid sizes allocate");
    return 0;
}
