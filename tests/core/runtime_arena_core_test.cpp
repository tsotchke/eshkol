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

    // ── ESH-0214b: per-iteration loop scope primitives ──

    // arena_commit_scope keeps allocations but drops the scope record.
    const size_t used_before_commit = arena_get_used_memory(arena);
    arena_push_scope(arena);
    void* committed_alloc = arena_allocate(arena, 64);
    if (!committed_alloc) return fail("commit-scope test allocation returned null");
    arena_commit_scope(arena);
    if (arena_get_used_memory(arena) <= used_before_commit) {
        return fail("commit released memory it should have kept");
    }
    std::memset(committed_alloc, 0x5A, 64);  // must still be writable (ASAN lane)

    // arena_top_scope_contains: in-span vs pre-span vs foreign pointers.
    void* before_scope = arena_allocate(arena, 16);
    arena_push_scope(arena);
    void* in_scope = arena_allocate(arena, 16);
    int on_stack = 0;
    if (!arena_top_scope_contains(arena, in_scope)) {
        return fail("in-scope pointer not detected");
    }
    if (arena_top_scope_contains(arena, before_scope)) {
        return fail("pre-scope pointer misdetected as in-scope");
    }
    if (arena_top_scope_contains(arena, &on_stack)) {
        return fail("foreign (stack) pointer misdetected as in-scope");
    }
    // ...including across a block boundary added inside the scope.
    void* in_scope_big = arena_allocate(arena, 4096);
    if (!arena_top_scope_contains(arena, in_scope_big)) {
        return fail("in-scope pointer in overflow block not detected");
    }
    arena_pop_scope(arena);

    // eshkol_arena_iter_scope_end: POP path (no out-flowing heap values).
    const size_t used_before_iter = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 512)) return fail("iter-pop test allocation returned null");
    eshkol_tagged_value_t ints[2];
    std::memset(ints, 0, sizeof(ints));
    ints[0].type = ESHKOL_VALUE_INT64;  ints[0].data.int_val = 41;
    ints[1].type = ESHKOL_VALUE_DOUBLE; ints[1].data.double_val = 2.5;
    eshkol_arena_iter_scope_end(arena, ints, 2);
    if (arena_get_used_memory(arena) != used_before_iter) {
        return fail("iter-scope-end with immediates did not pop (reclaim)");
    }

    // POP path with a heap value that lies OUTSIDE the scope span (the
    // carried-port shape): reclamation must still happen.
    void* pre_alloc = arena_allocate(arena, 32);
    const size_t used_before_iter2 = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 256)) return fail("iter-pop2 test allocation returned null");
    eshkol_tagged_value_t carried;
    std::memset(&carried, 0, sizeof(carried));
    carried.type = ESHKOL_VALUE_HEAP_PTR;
    carried.data.ptr_val = (uint64_t)(uintptr_t)pre_alloc;
    eshkol_arena_iter_scope_end(arena, &carried, 1);
    if (arena_get_used_memory(arena) != used_before_iter2) {
        return fail("iter-scope-end with pre-scope heap value did not pop");
    }

    // COMMIT path: an out-flowing heap value allocated INSIDE the scope.
    arena_push_scope(arena);
    void* escaping = arena_allocate(arena, 128);
    if (!escaping) return fail("iter-commit test allocation returned null");
    const size_t used_at_commit = arena_get_used_memory(arena);
    eshkol_tagged_value_t esc;
    std::memset(&esc, 0, sizeof(esc));
    esc.type = ESHKOL_VALUE_HEAP_PTR;
    esc.data.ptr_val = (uint64_t)(uintptr_t)escaping;
    eshkol_arena_iter_scope_end(arena, &esc, 1);
    if (arena_get_used_memory(arena) != used_at_commit) {
        return fail("iter-scope-end with escaping heap value did not commit (keep memory)");
    }
    std::memset(escaping, 0x7E, 128);  // must still be writable (ASAN lane)

    // Conservative typing: an unknown/exotic type tag with a null pointer
    // must not block reclamation (eof-object shape), and the scope stack
    // must stay balanced through mixed pop/commit sequences.
    const size_t used_mixed = arena_get_used_memory(arena);
    arena_push_scope(arena);
    if (!arena_allocate(arena, 64)) return fail("mixed test allocation returned null");
    eshkol_tagged_value_t eof;
    std::memset(&eof, 0, sizeof(eof));
    eof.type = 0xFF;  // eof-object: data always 0
    eshkol_arena_iter_scope_end(arena, &eof, 1);
    if (arena_get_used_memory(arena) != used_mixed) {
        return fail("iter-scope-end with eof-object did not pop");
    }

    arena_destroy(arena);

    std::cout << "PASS\n";
    return 0;
}
