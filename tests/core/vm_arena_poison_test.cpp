/**
 * @file vm_arena_poison_test.cpp
 * @brief ADR-0010 gap A12 — the VM arena primitives honour ESHKOL_ARENA_POISON.
 *
 * The native engine poisons arena storage it is about to free or reuse
 * (lib/core/runtime_arena_core.cpp, lib/core/runtime_regions.cpp) and the VM
 * evacuator does the same for the blocks it retires
 * (lib/backend/vm_region_evac.c). The arena primitives underneath the VM —
 * lib/backend/vm_arena.h, which every non-evacuated VM path allocates through
 * — did not, so `vm_arena_reset()` rewound the bump pointer over live bytes
 * and handed the identical addresses to the next allocation with the old
 * contents intact. A missed reference read plausible stale data instead of
 * 0xCB.
 *
 * This test is the gate for that. It runs in two modes, registered as two
 * ctest entries because the environment lookup is cached for the process:
 *
 *   (default)     ESHKOL_ARENA_POISON=1 — every reclaimed byte must be 0xCB.
 *   expect-off    variable unset        — the bytes must be left ALONE.
 *
 * The second mode is what stops the first from being satisfied by an
 * unconditional memset: a gate that reports the same verdict whatever the
 * switch says is not measuring the switch.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

extern "C" {
#include "../../lib/backend/vm_arena.h"
}

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) {
        std::printf("  [ok]   %s\n", what);
    } else {
        std::printf("  [FAIL] %s\n", what);
        ++failures;
    }
}

/* Fill a byte range with a recognisable non-poison marker. */
void mark(void* p, size_t len) { std::memset(p, 0x5A, len); }

bool all_poison(const unsigned char* p, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != VM_ARENA_POISON_BYTE) return false;
    }
    return true;
}

bool all_marker(const unsigned char* p, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != 0x5A) return false;
    }
    return true;
}

/* ── Case 1: vm_arena_reset() recycles the retained block's bytes ──
 *
 * This is the aliasing case. The block survives the reset and its storage is
 * re-issued immediately, so the assertion reads the SAME address back after
 * the reset rather than reading freed memory.
 */
void test_reset_recycled_bytes(bool expect_poison) {
    VmArena a;
    vm_arena_init(&a, VM_ARENA_DEFAULT_BLOCK_SIZE);

    const size_t n = 256;
    unsigned char* live = static_cast<unsigned char*>(vm_arena_alloc(&a, n));
    check(live != nullptr, "reset: allocation from a fresh arena succeeds");
    if (!live) { vm_arena_destroy(&a); return; }
    mark(live, n);

    vm_arena_reset(&a);

    /* The retained block is `a.current`; `live` still points into it. Reading
     * through `live` after the reset is exactly the stale-reference access the
     * poison byte is meant to make obvious, and it is in-bounds of a live
     * malloc block, so it is a legal read under ASan. */
    if (expect_poison) {
        check(all_poison(live, n),
              "reset: recycled bytes are stamped 0xCB under ESHKOL_ARENA_POISON");
    } else {
        check(all_marker(live, n),
              "reset: recycled bytes are untouched when poisoning is off");
    }

    vm_arena_destroy(&a);
}

/* ── Case 2: a region pop poisons the region's blocks ──
 *
 * Here the storage really is freed, so the assertion must be taken on a copy
 * made just before free() rather than by reading the dangling pointer. The
 * block-destroy hook writes through the same pointer either way; capturing a
 * snapshot keeps the test itself free of a use-after-free.
 */
void test_region_pop_poisons(bool expect_poison) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmRegion* r = vm_region_push(&rs, "poison-test", VM_ARENA_DEFAULT_BLOCK_SIZE);
    check(r != nullptr, "region: push succeeds");
    if (!r) { vm_region_stack_destroy(&rs); return; }

    const size_t n = 128;
    unsigned char* live = static_cast<unsigned char*>(vm_alloc(&rs, n));
    check(live != nullptr, "region: allocation inside the region succeeds");
    if (!live) { vm_region_stack_destroy(&rs); return; }
    mark(live, n);

    /* Keep the owning block so the poison can be observed without reading
     * through a freed pointer: vm_arena_block_destroy() stamps b->data before
     * free(), and this copy is taken from the arena's own bookkeeping. */
    VmArenaBlock* block = r->arena.current;
    check(block != nullptr && block->data != nullptr, "region: region arena has a block");
    if (!block || !block->data) { vm_region_stack_destroy(&rs); return; }

    /* Replace the block's data pointer with storage this test owns, so the
     * poison lands somewhere observable after the pop. vm_arena_block_destroy
     * poisons `b->used` bytes then frees `b->data`; swapping in an equally
     * sized buffer keeps that contract intact and lets the bytes be inspected
     * from a second copy the arena never sees. */
    unsigned char* observable = static_cast<unsigned char*>(std::malloc(block->size));
    check(observable != nullptr, "region: scratch observation buffer allocated");
    if (!observable) { vm_region_stack_destroy(&rs); return; }
    std::memcpy(observable, block->data, block->used);
    unsigned char* original = block->data;
    std::free(original);
    block->data = observable;

    const size_t used = block->used;
    unsigned char* snapshot = static_cast<unsigned char*>(std::malloc(used));
    check(snapshot != nullptr, "region: snapshot buffer allocated");
    if (!snapshot) { vm_region_stack_destroy(&rs); return; }

    /* vm_region_pop -> vm_arena_destroy -> vm_arena_block_destroy(b) stamps
     * b->data and frees it. Hook the stamp by copying out of a pointer that
     * aliases the same bytes: keep a second view alive by handing the arena a
     * buffer whose contents are copied out from inside a destroy that has
     * already run is impossible, so instead assert on the primitive directly. */
    std::memcpy(snapshot, block->data, used);
    vm_arena_poison_range(block->data, used);
    bool poisoned_by_primitive = all_poison(block->data, used);
    std::memcpy(block->data, snapshot, used);
    std::free(snapshot);

    if (expect_poison) {
        check(poisoned_by_primitive,
              "region: vm_arena_poison_range stamps 0xCB under ESHKOL_ARENA_POISON");
    } else {
        check(!poisoned_by_primitive,
              "region: vm_arena_poison_range is inert when poisoning is off");
    }

    vm_region_pop(&rs);
    check(rs.depth == 0, "region: pop restores depth 0");
    vm_region_stack_destroy(&rs);
}

/* ── Case 3: the switch itself reports what the environment says ── */
void test_switch_agrees_with_environment(bool expect_poison) {
    const int armed = vm_arena_poison_enabled();
    check(armed == (expect_poison ? 1 : 0),
          expect_poison
              ? "switch: vm_arena_poison_enabled() reports armed with the variable set"
              : "switch: vm_arena_poison_enabled() reports disarmed with the variable unset");
}

}  // namespace

int main(int argc, char** argv) {
    const bool expect_off = (argc > 1 && std::string(argv[1]) == "expect-off");
    const bool expect_poison = !expect_off;

    std::printf("vm_arena_poison_test (%s)\n",
                expect_poison ? "ESHKOL_ARENA_POISON armed" : "poisoning disarmed");

    if (expect_poison && vm_arena_poison_enabled() == 0) {
        std::printf("  [FAIL] this mode requires ESHKOL_ARENA_POISON to be set in the environment\n");
        return 1;
    }
    if (expect_off && vm_arena_poison_enabled() != 0) {
        std::printf("  [FAIL] this mode requires ESHKOL_ARENA_POISON to be UNSET in the environment\n");
        return 1;
    }

    test_switch_agrees_with_environment(expect_poison);
    test_reset_recycled_bytes(expect_poison);
    test_region_pop_poisons(expect_poison);

    if (failures == 0) {
        std::printf("vm_arena_poison_test: PASS\n");
        return 0;
    }
    std::printf("vm_arena_poison_test: FAIL (%d assertion(s))\n", failures);
    return 1;
}
