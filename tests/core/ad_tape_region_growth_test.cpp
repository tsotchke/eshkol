/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * tests/core/ad_tape_region_growth_test.cpp — #341 dangling-pointer trap gate.
 *
 * The AD tape's node-pointer array grows (doubling) from an arena. Before #341
 * that growth always drew from the pinned __repl_shared_arena, which leaked the
 * array whenever the tape lived in a `(with-region ...)` (the array outlived the
 * region-reclaimed header). The fix grows from the arena the tape HEADER lives
 * in (ad_tape_t::owner_arena, captured at creation).
 *
 * The obvious-but-wrong alternative — "grow from the CURRENT arena" — is a
 * use-after-free: a tape created OUTSIDE a region but grown INSIDE one would put
 * its grown pointer array in the region arena, which region_pop frees while the
 * tape header (in the global arena) survives. This test exercises exactly that
 * lifetime and asserts the owning-arena rule keeps the array valid AFTER the
 * inner region is torn down.
 *
 * It MUST be run with ESHKOL_ARENA_POISON=1: region_pop -> region_destroy ->
 * arena_destroy stamps the freed region arena with 0xCB before releasing it, so
 * a grown array that (wrongly) lived there reads back as 0xCB.. garbage and this
 * test fails loudly instead of passing by luck on not-yet-reused memory.
 */

#include "../../lib/core/arena_memory.h"  // ad_tape_t / ad_node_t (via eshkol.h) + arena/region/tape API

#include <cstdio>
#include <cstdlib>

// Declared in runtime_arena_core.cpp; not exported in a public header.
extern "C" int eshkol_arena_poison_enabled(void);

namespace {
int fail(const char* msg) {
    std::fprintf(stderr, "ad_tape_region_growth_test: FAIL: %s\n", msg);
    return 1;
}
}  // namespace

int main() {
    arena_t* global = get_global_arena();
    if (!global) return fail("no global arena");

    // Mirror the AOT/REPL init that pins __repl_shared_arena to the process
    // arena, so the legacy fallback path (owner == null) is also exercised sanely.
    __repl_shared_arena.store(global);

    if (!eshkol_arena_poison_enabled()) {
        return fail("ESHKOL_ARENA_POISON not enabled — run this test with ESHKOL_ARENA_POISON=1 "
                    "so the freed region arena is 0xCB-stamped (the dangling-pointer trap)");
    }

    // 1. Create the tape OUTSIDE any region, from the global arena, with a tiny
    //    initial capacity so a few appends already force a grow.
    const size_t init_cap = 4;
    ad_tape_t* tape = arena_allocate_tape(global, init_cap);
    if (!tape) return fail("tape allocation returned null");
    if (tape->owner_arena != global)
        return fail("owner_arena was not recorded as the creation (global) arena");

    // Dummy nodes allocated from the GLOBAL arena: they must outlive the region,
    // and each carries a sentinel id so a post-region readback can be checked.
    const size_t N = 300;  // 4 -> 128 -> 256 -> 512: multiple grows, all inside the region
    ad_node_t** owned = (ad_node_t**)std::malloc(sizeof(ad_node_t*) * N);
    if (!owned) return fail("host bookkeeping malloc failed");
    for (size_t i = 0; i < N; ++i) {
        ad_node_t* n = (ad_node_t*)arena_allocate_aligned(global, sizeof(ad_node_t), 8);
        if (!n) { std::free(owned); return fail("global dummy-node allocation failed"); }
        n->id = i + 1;               // sentinel (1-based so 0/0xCB.. are both distinguishable)
        n->value = (double)i;
        owned[i] = n;
    }

    // A few appends while still outside the region (array still the original one).
    for (size_t i = 0; i < 3; ++i) arena_tape_add_node(tape, owned[i]);

    // 2. Enter a per-step region — swaps the current allocation arena to the
    //    region arena, exactly as `(with-region ...)` does.
    eshkol_region_t* region = region_create("tape-grow-inside", (size_t)1 << 20);
    if (!region) { std::free(owned); return fail("region_create failed"); }
    region_push(region);
    arena_t* saved = eshkol_region_enter(region);

    // 3. Grow the tape INSIDE the region. The array is reallocated (several times)
    //    here — the fix routes those reallocations to tape->owner_arena (global),
    //    NOT to the now-current region arena.
    for (size_t i = 3; i < N; ++i) arena_tape_add_node(tape, owned[i]);
    if (tape->capacity <= init_cap) { std::free(owned); return fail("tape never grew — test would not exercise the trap"); }
    if (tape->num_nodes != N)       { std::free(owned); return fail("not all nodes were appended"); }
    if (tape->owner_arena != global) { std::free(owned); return fail("owner_arena drifted while inside the region"); }

    // 4. Tear the region down: region_pop -> region_destroy poisons (0xCB) and
    //    frees the region arena. If the grown array lived there, it is now gone.
    region_pop();
    eshkol_region_leave(saved);

    // 5. Use the tape AFTER the region is gone. Every node pointer must read back
    //    intact; a grown array that had dangled into the freed region arena would
    //    yield 0xCB.. pointers (crash) or wrong sentinels here.
    if (tape->num_nodes != N) { std::free(owned); return fail("num_nodes changed across region_pop"); }
    for (size_t i = 0; i < N; ++i) {
        ad_node_t* n = arena_tape_get_node(tape, i);
        if (!n) { std::free(owned); return fail("null node after region_pop (grown array dangled)"); }
        if (n->id != i + 1) {
            std::free(owned);
            std::fprintf(stderr,
                "ad_tape_region_growth_test: node %zu read back id=%zu (expected %zu) — the grown "
                "pointer array dangled into the freed/poisoned region arena\n", i, n->id, i + 1);
            return 1;
        }
    }

    std::free(owned);

    // ── Part 2: the leak this bug was reported for (#341). A tape created INSIDE
    //    a per-step region, grown there, must be FULLY reclaimed at region_pop:
    //    its pointer array must NOT accrete in the pinned process/global arena.
    //    Before the fix the array grew from __repl_shared_arena, so each step
    //    leaked ~ (sum of the abandoned doublings) into the global arena — the
    //    ~8 MB/step residual the field report measured on a large reverse-mode
    //    training loop. Here we drive many region-scoped tape grows and assert
    //    the global arena stays flat (deterministic, unlike process RSS).
    {
        const int    steps         = 200;
        const size_t nodes_per_pass = 500;  // 4 -> 128 -> 256 -> 512: several grows/pass
        const size_t before = arena_get_used_memory(global);

        for (int s = 0; s < steps; ++s) {
            eshkol_region_t* r = region_create("ad-step", (size_t)1 << 20);
            if (!r) return fail("Part 2: region_create failed");
            region_push(r);
            arena_t* rsaved = eshkol_region_enter(r);

            // Tape created from the CURRENT (region) arena — exactly what
            // createTape()'s getArenaPtr()==__global_arena resolves to inside a
            // with-region body. owner_arena is therefore the region arena.
            arena_t* cur = eshkol_current_arena();
            ad_tape_t* t = arena_allocate_tape(cur, 4);
            if (!t) return fail("Part 2: tape alloc failed");
            for (size_t i = 0; i < nodes_per_pass; ++i) {
                ad_node_t* n = (ad_node_t*)arena_allocate_aligned(cur, sizeof(ad_node_t), 8);
                if (!n) return fail("Part 2: node alloc failed");
                arena_tape_add_node(t, n);
            }

            region_pop();               // frees the region arena (header + nodes + grown array)
            eshkol_region_leave(rsaved);
        }

        const size_t after  = arena_get_used_memory(global);
        const size_t grew    = (after > before) ? (after - before) : 0;
        // The fix leaves the global arena flat. Allow a small slack for any
        // incidental global-arena bookkeeping; the pre-fix leak here is ~1.4 MB
        // (7 KB/step * 200), far above this bound, and scales without limit with
        // tape size — so the gate has wide separation in both directions.
        const size_t slack = 64 * 1024;  // 64 KB
        if (grew > slack) {
            std::fprintf(stderr,
                "ad_tape_region_growth_test: FAIL: global arena grew %zu bytes (%.1f KB/step) across "
                "%d region-scoped tape grows — the tape pointer array is leaking into the pinned "
                "arena again instead of the region arena (#341)\n",
                grew, grew / 1024.0 / steps, steps);
            return 1;
        }
        std::printf("ad_tape_region_growth_test: reclamation OK "
                    "(global arena flat: +%zu bytes over %d region-scoped tape grows)\n", grew, steps);
    }

    std::printf("ad_tape_region_growth_test: PASS "
                "(tape grown inside a region from its owning global arena survives region_pop; "
                "%zu nodes intact; per-step region grows fully reclaimed)\n", N);
    return 0;
}
