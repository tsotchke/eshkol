#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <iostream>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

struct alignas(32) WideValue {
    int64_t lanes[4];
};

bool is_aligned(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

}  // namespace

int main() {
    Arena arena(64);
    if (!arena.get_arena()) return fail("Arena wrapper did not create arena");
    if (arena.get_total_memory() != 1024) return fail("Arena wrapper minimum block mismatch");

    int* value = arena.allocate<int>();
    if (!value) return fail("typed allocation returned null");
    *value = 42;
    if (*value != 42) return fail("typed allocation write/read mismatch");

    WideValue* wide = arena.allocate<WideValue>();
    if (!wide) return fail("over-aligned typed allocation returned null");
    if (!is_aligned(wide, alignof(WideValue))) {
        return fail("over-aligned typed allocation returned misaligned pointer");
    }

    auto* zeroed = static_cast<uint8_t*>(arena.allocate_zeroed(16));
    if (!zeroed) return fail("zeroed wrapper allocation returned null");
    for (size_t i = 0; i < 16; ++i) {
        if (zeroed[i] != 0) return fail("zeroed wrapper allocation contained non-zero byte");
    }

    const size_t used_before_scope = arena.get_used_memory();
    {
        Arena::Scope scope(arena);
        if (!arena.allocate(512)) return fail("scoped wrapper allocation returned null");
        if (arena.get_used_memory() <= used_before_scope) {
            return fail("scoped wrapper allocation did not increase used memory");
        }
    }
    if (arena.get_used_memory() != used_before_scope) {
        return fail("Arena::Scope did not restore used memory");
    }

    int64_t* array = arena.allocate_array<int64_t>(4);
    if (!array) return fail("array allocation returned null");
    array[3] = 99;
    if (array[3] != 99) return fail("array allocation write/read mismatch");

    Arena moved(std::move(arena));
    if (!moved.get_arena()) return fail("move constructor lost arena");
    if (arena.get_arena() != nullptr) return fail("move constructor did not clear source");
    if (!moved.allocate(8)) return fail("moved arena allocation returned null");

    Arena assigned(1024);
    assigned = std::move(moved);
    if (!assigned.get_arena()) return fail("move assignment lost arena");
    if (moved.get_arena() != nullptr) return fail("move assignment did not clear source");
    if (!assigned.allocate(8)) return fail("assigned arena allocation returned null");

    assigned.reset();
    if (assigned.get_used_memory() != 0) return fail("wrapper reset did not clear used memory");
    if (assigned.get_block_count() != 1) return fail("wrapper reset did not restore one block");

    std::cout << "PASS\n";
    return 0;
}
