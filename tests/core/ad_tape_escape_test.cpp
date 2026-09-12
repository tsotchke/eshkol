#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstdio>

int main() {
    arena_t* arena = get_global_arena();
    if (!arena) return 1;
    __repl_shared_arena.store(arena);

    ad_tape_t* tape = arena_allocate_tape(arena, 4);
    if (!tape) return 1;
    uint64_t* escaped = (uint64_t*)arena_allocate_aligned(arena, 2 * sizeof(uint64_t), 8);
    if (!escaped) return 1;
    escaped[0] = 111;
    escaped[1] = 222;

    arena_tape_release(tape);
    for (int i = 0; i < 32; ++i) {
        uint64_t* pressure = (uint64_t*)arena_allocate_aligned(arena, 2 * sizeof(uint64_t), 8);
        if (!pressure) return 1;
        pressure[0] = 9000 + (uint64_t)i;
        pressure[1] = 9001 + (uint64_t)i;
    }
    if (escaped[0] != 111 || escaped[1] != 222) {
        std::fprintf(stderr, "ad_tape_escape_test: FAIL: escaped allocation was reclaimed\n");
        return 1;
    }

    std::printf("ad_tape_escape_test: PASS: escaped allocation survives tape release\n");
    return 0;
}
