#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

}  // namespace

int main() {
    if (arena_get_used_memory(nullptr) != 0) return fail("null used-memory query mismatch");
    if (arena_get_total_memory(nullptr) != 0) return fail("null total-memory query mismatch");
    if (arena_get_block_count(nullptr) != 0) return fail("null block-count query mismatch");

    arena_t* arena = arena_create(8);
    if (!arena) return fail("arena_create returned null");
    if (arena_get_total_memory(arena) != 1024) return fail("minimum block size mismatch");
    if (arena_get_block_count(arena) != 1) return fail("initial block count mismatch");
    if (arena_get_used_memory(arena) != 0) return fail("initial used memory mismatch");

    if (arena_allocate(nullptr, 8) != nullptr) return fail("null arena allocation succeeded");
    if (arena_allocate(arena, 0) != nullptr) return fail("zero-size allocation succeeded");

    void* first = arena_allocate(arena, 7);
    if (!first) return fail("basic arena allocation returned null");
    if (!is_aligned(first, 8)) return fail("default allocation was not 8-byte aligned");
    if (arena_get_used_memory(arena) < 7) return fail("used memory did not increase");

    const size_t used_before_invalid = arena_get_used_memory(arena);
    if (arena_allocate_aligned(arena, 8, 24) != nullptr) {
        return fail("non-power-of-two alignment allocation succeeded");
    }
    if (arena_get_used_memory(arena) != used_before_invalid) {
        return fail("invalid alignment changed arena state");
    }

    void* aligned = arena_allocate_aligned(arena, 1, 32);
    if (!aligned) return fail("over-aligned allocation returned null");
    if (!is_aligned(aligned, 32)) return fail("over-aligned allocation returned misaligned pointer");

    auto* zeroed = static_cast<unsigned char*>(arena_allocate_zeroed(arena, 32));
    if (!zeroed) return fail("zeroed allocation returned null");
    for (size_t i = 0; i < 32; ++i) {
        if (zeroed[i] != 0) return fail("zeroed allocation contained non-zero byte");
    }

    arena_cons_cell_t* cons = arena_allocate_cons_cell(arena);
    if (!cons) return fail("legacy cons allocation returned null");
    cons->car = 11;
    cons->cdr = 22;
    if (cons->car != 11 || cons->cdr != 22) return fail("legacy cons write/read mismatch");

    auto* nodes = static_cast<int64_t*>(arena_allocate_list_node(arena, sizeof(int64_t), 3));
    if (!nodes) return fail("list-node allocation returned null");
    nodes[0] = 1;
    nodes[1] = 2;
    nodes[2] = 3;
    if (nodes[2] != 3) return fail("list-node write/read mismatch");

    const size_t used_before_scope = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 2048)) return fail("scoped large allocation returned null");
    if (arena_get_block_count(arena) <= 1) return fail("large scoped allocation did not add block");
    arena_pop_scope(arena);
    if (arena_get_used_memory(arena) != used_before_scope) {
        return fail("scope pop did not restore used memory");
    }
    if (arena_get_block_count(arena) != 1) return fail("scope pop did not release extra block");

    if (!arena_allocate(arena, 4096)) return fail("large allocation returned null");
    if (arena_get_block_count(arena) <= 1) return fail("large allocation did not add block");
    arena_reset(arena);
    if (arena_get_used_memory(arena) != 0) return fail("reset did not clear used memory");
    if (arena_get_block_count(arena) != 1) return fail("reset did not restore one block");
    if (arena_get_total_memory(arena) != 1024) return fail("reset did not restore total memory");

    const size_t max = std::numeric_limits<size_t>::max();
    if (arena_allocate_aligned(arena, max, 8) != nullptr) {
        return fail("overflowing aligned allocation succeeded");
    }
    if (arena_allocate_list_node(arena, max / 2 + 1, 3) != nullptr) {
        return fail("overflowing list-node allocation succeeded");
    }

    arena_destroy(arena);

    std::cout << "PASS\n";
    return 0;
}
