/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Pins the heap object ABI: the header's size, every field's offset and width,
 * the payload's position and alignment, and the link-time guard that keeps two
 * halves of a toolchain from disagreeing about any of it.
 *
 * Why a test and not only static assertions: the assertions in
 * inc/eshkol/abi_fingerprint.h check what the *compiler* was told. This checks
 * what the *runtime archive* actually does — that a real allocation puts the
 * subtype byte where the compiler will emit code to read it, and that the guard
 * symbol the runtime defines is the one the caller requires. Those are the two
 * things that can be true separately and still not be true together, which is
 * the entire failure mode this suite exists for.
 *
 * The object model gives a wrong layout no way to announce itself. A payload
 * pointer carries no discriminator: `subtype` is inside the header, so reading
 * it already assumes the answer. A migration that misses a site does not crash
 * — it returns wrong data. Every assertion below is therefore a tripwire for a
 * silent failure, and each is written to name what would have gone wrong.
 */

#include <eshkol/eshkol.h>
#include <eshkol/abi_fingerprint.h>

#include "../../lib/core/arena_memory.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what, const std::string& consequence) {
    if (ok) {
        std::cout << "  ok    " << what << '\n';
        return;
    }
    std::cout << "  FAIL  " << what << '\n';
    std::cout << "        if this is deliberate: " << consequence << '\n';
    ++failures;
}

template <typename T>
void check_eq(T actual, T expected, const std::string& what,
              const std::string& consequence) {
    if (actual == expected) {
        std::cout << "  ok    " << what << " = " << +expected << '\n';
        return;
    }
    std::cout << "  FAIL  " << what << ": expected " << +expected
              << ", got " << +actual << '\n';
    std::cout << "        if this is deliberate: " << consequence << '\n';
    ++failures;
}

const char* const kBumpAbi =
    "bump ESHKOL_OBJECT_ABI_VERSION in inc/eshkol/abi_fingerprint.h so every "
    "artifact built against the old layout fails to link, then re-run "
    "scripts/abi_header_inventory.py baseline";

}  // namespace

int main() {
    std::cout << "object header layout pin\n";

    // ── 1. The header's shape ────────────────────────────────────────────────
    // Size first: it is the quantity ESHKOL_GET_HEADER subtracts, and the one
    // the code generator bakes into every emitted GEP as a literal.
    check_eq(sizeof(eshkol_object_header_t), static_cast<size_t>(8),
             "sizeof(eshkol_object_header_t)", kBumpAbi);
    check_eq(alignof(eshkol_object_header_t), static_cast<size_t>(4),
             "alignof(eshkol_object_header_t)", kBumpAbi);

    // Field offsets. These are not redundant with the size: the code generator
    // emits -8 for subtype, -7 for flags, -6 for ref_count and -4 for size as
    // separate literals, so a field that moves within a header of unchanged
    // size breaks generated code while every size assertion still passes.
    check_eq(offsetof(eshkol_object_header_t, subtype), static_cast<size_t>(0),
             "offsetof(header, subtype)", kBumpAbi);
    check_eq(offsetof(eshkol_object_header_t, flags), static_cast<size_t>(1),
             "offsetof(header, flags)", kBumpAbi);
    check_eq(offsetof(eshkol_object_header_t, ref_count), static_cast<size_t>(2),
             "offsetof(header, ref_count)", kBumpAbi);
    check_eq(offsetof(eshkol_object_header_t, size), static_cast<size_t>(4),
             "offsetof(header, size)", kBumpAbi);

    // Field widths. A subtype widened from 8 to 16 bits keeps its offset and
    // changes the meaning of every emitted i8 load.
    check_eq(sizeof(eshkol_object_header_t::subtype), static_cast<size_t>(1),
             "sizeof(header.subtype)", kBumpAbi);
    check_eq(sizeof(eshkol_object_header_t::flags), static_cast<size_t>(1),
             "sizeof(header.flags)", kBumpAbi);
    check_eq(sizeof(eshkol_object_header_t::ref_count), static_cast<size_t>(2),
             "sizeof(header.ref_count)", kBumpAbi);
    check_eq(sizeof(eshkol_object_header_t::size), static_cast<size_t>(4),
             "sizeof(header.size)", kBumpAbi);

    // ── 2. The fingerprint describes the layout it claims to ─────────────────
    // Against the ACTIVE header, so these follow ESHKOL_MEMORY_ABI_V2 rather
    // than needing to be edited when it flips. Under v1 the active header is
    // the struct above; under v2 it is the 32-byte successor, and the guard
    // symbol renames itself accordingly.
    check_eq(static_cast<size_t>(ESHKOL_OBJECT_ABI_HEADER_SIZE),
             ESHKOL_OBJECT_HEADER_SIZE,
             "ESHKOL_OBJECT_ABI_HEADER_SIZE matches the active header",
             "the fingerprint is stale and would let an incompatible link succeed");
    check_eq(static_cast<size_t>(ESHKOL_OBJECT_ABI_SUBTYPE_OFF),
             offsetof(eshkol_object_header_active_t, subtype),
             "ESHKOL_OBJECT_ABI_SUBTYPE_OFF matches the active header",
             "the fingerprint is stale and would let an incompatible link succeed");
    check_eq(static_cast<uint32_t>(ESHKOL_OBJECT_ABI_VERSION),
             eshkol_memory_abi_active(),
             "fingerprint ABI version matches the runtime's reported ABI",
             "the runtime and this test were built against different layouts");

    // ── 3. The runtime agrees with this caller ───────────────────────────────
    // These two are compiled separately. Equality here is what a static link
    // guarantees and what a dlopen() does not — which is why the accessors
    // exist at all.
    check_eq(eshkol_abi_runtime_header_size(), ESHKOL_OBJECT_HEADER_SIZE,
             "runtime archive's header size matches this translation unit's",
             "the runtime and this test were built against different layouts");
    check(std::strcmp(eshkol_abi_fingerprint_name(), ESHKOL_ABI_FINGERPRINT_NAME) == 0,
          std::string("guard symbol name agrees: ") + ESHKOL_ABI_FINGERPRINT_NAME,
          "the runtime and this test were built against different layouts");
    check_eq(ESHKOL_ABI_FINGERPRINT_SYMBOL,
             static_cast<size_t>(ESHKOL_OBJECT_ABI_HEADER_SIZE),
             "guard symbol resolves and carries the header size",
             "the guard symbol was not linked in");

    // ── 4. A real allocation is laid out where the compiler will look ────────
    // The assertions above are about types. This is about the arena: it is the
    // runtime, not the header, that decides where a payload actually starts.
    arena_t* arena = arena_create(4096);
    if (!arena) {
        std::cout << "  FAIL  arena_create returned null\n";
        return 1;
    }

    const uint32_t payload_size = 24;
    void* data = arena_allocate_with_header(arena, payload_size,
                                            HEAP_SUBTYPE_BYTEVECTOR,
                                            ESHKOL_OBJ_FLAG_PINNED);
    if (!data) {
        std::cout << "  FAIL  arena_allocate_with_header returned null\n";
        arena_destroy(arena);
        return 1;
    }

    // The relationship the entire ABI rests on: the header is exactly
    // sizeof(header) bytes below the pointer everyone passes around.
    auto* via_macro = ESHKOL_GET_HEADER(data);
    auto* via_arithmetic = reinterpret_cast<eshkol_object_header_t*>(
        static_cast<uint8_t*>(data) - sizeof(eshkol_object_header_t));
    check(via_macro == via_arithmetic,
          "ESHKOL_GET_HEADER(data) == data - sizeof(header)",
          "the accessor and the raw arithmetic sites in the tree disagree");
    check(ESHKOL_GET_DATA_PTR(via_macro) == data,
          "ESHKOL_GET_DATA_PTR round-trips back to the payload",
          kBumpAbi);

    // Read the fields the way generated code does: as raw bytes at a negative
    // offset from the payload. If this and the struct ever disagree, generated
    // code and the runtime are reading different objects.
    const auto* bytes = static_cast<const uint8_t*>(data);
    check_eq(static_cast<uint32_t>(*(bytes - 8)),
             static_cast<uint32_t>(HEAP_SUBTYPE_BYTEVECTOR),
             "subtype byte read raw at payload-8", kBumpAbi);
    check_eq(static_cast<uint32_t>(*(bytes - 7)),
             static_cast<uint32_t>(ESHKOL_OBJ_FLAG_PINNED),
             "flags byte read raw at payload-7", kBumpAbi);
    uint16_t raw_refcount = 0;
    std::memcpy(&raw_refcount, bytes - 6, sizeof(raw_refcount));
    check_eq(raw_refcount, static_cast<uint16_t>(0),
             "ref_count read raw at payload-6", kBumpAbi);
    uint32_t raw_size = 0;
    std::memcpy(&raw_size, bytes - 4, sizeof(raw_size));
    check_eq(raw_size, payload_size, "size read raw at payload-4", kBumpAbi);

    // And the same values through the accessor, so a divergence between the two
    // routes is caught rather than averaged.
    check_eq(static_cast<uint32_t>(ESHKOL_GET_SUBTYPE(data)),
             static_cast<uint32_t>(HEAP_SUBTYPE_BYTEVECTOR),
             "ESHKOL_GET_SUBTYPE agrees with the raw read", kBumpAbi);
    check_eq(static_cast<uint32_t>(ESHKOL_GET_FLAGS(data)),
             static_cast<uint32_t>(ESHKOL_OBJ_FLAG_PINNED),
             "ESHKOL_GET_FLAGS agrees with the raw read", kBumpAbi);

    // Payload alignment. The GPU zero-copy paths and every aligned load in
    // generated code depend on this, and it is a function of the header size —
    // change the header and the payload's alignment class moves with it.
    //
    // Checked against the allocator's own guarantee of 8 rather than against
    // ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN, because those are two different claims
    // and only one of them is about this code path. Selecting ABI v2 today
    // changes what the *types* say; the arena has not been migrated (ADR-0012
    // Stage 6) and still prepends the v1 header. The stricter comparison is made
    // only when the declared ABI and the allocator are the same generation, so
    // this test reports the allocator's actual behaviour in both flag states
    // instead of asserting a property the allocator has not been asked for yet.
    check_eq(reinterpret_cast<uintptr_t>(data) % 8,
             static_cast<uintptr_t>(0),
             "payload is 8-byte aligned (the arena's guarantee)", kBumpAbi);
    if (ESHKOL_OBJECT_ABI_VERSION == 1) {
        check_eq(reinterpret_cast<uintptr_t>(data) % ESHKOL_OBJECT_ABI_PAYLOAD_ALIGN,
                 static_cast<uintptr_t>(0),
                 "payload satisfies the declared ABI's payload alignment", kBumpAbi);
    } else {
        std::cout << "  note  declared ABI is v" << ESHKOL_OBJECT_ABI_VERSION
                  << " while the arena still allocates the v1 header;"
                     " payload alignment is checked against the arena\n";
    }

    arena_destroy(arena);

    std::cout << '\n';
    if (failures != 0) {
        std::cout << "abi_layout_pin_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "abi_layout_pin_test: PASS\n";
    return 0;
}
