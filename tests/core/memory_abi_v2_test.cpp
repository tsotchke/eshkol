// memory_abi_v2_test.cpp — pins the OALR object ABI layouts (ADR-0001 §3).
//
// The static assertions in inc/eshkol/memory_abi_v2.h already fail the BUILD if
// either layout moves, so this executable exists for the two things a static
// assertion cannot check:
//
//   1. that the *runtime archive* reports the same ABI as the translation unit
//      asking — the mismatch that would otherwise be a silent miscompile,
//      because the object header prefix appears in no symbol signature; and
//   2. that the ABI actually selected is the one the build asked for, so a
//      default-off flag cannot quietly become default-on (or vice versa).
//
// It runs and must pass in BOTH flag states. With ESHKOL_MEMORY_ABI_V2=OFF it
// asserts v1 is live; with it ON it asserts v2 is live.

#include "eshkol/eshkol.h"

#include <cstdint>
#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) {
        std::cout << "ok   " << what << '\n';
    } else {
        std::cout << "FAIL " << what << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    // ---- both layouts are defined and sized, whichever one is selected ----
    check(sizeof(eshkol_object_header_t) == 8,
          "ABI v1 header is 8 bytes");
    check(sizeof(eshkol_object_header_v2_t) == 32,
          "ABI v2 header is 32 bytes");
    check(alignof(eshkol_object_header_v2_t) == 16,
          "ABI v2 header is 16-byte aligned so payloads stay 16-byte aligned");

    // ---- v2 field offsets are the ones the evacuator/codegen will encode ----
    check(offsetof(eshkol_object_header_v2_t, payload_size) == 0, "v2 payload_size @0");
    check(offsetof(eshkol_object_header_v2_t, layout_id) == 4, "v2 layout_id @4");
    check(offsetof(eshkol_object_header_v2_t, subtype) == 6, "v2 subtype @6");
    check(offsetof(eshkol_object_header_v2_t, flags) == 7, "v2 flags @7");
    check(offsetof(eshkol_object_header_v2_t, object_id) == 8, "v2 object_id @8");
    check(offsetof(eshkol_object_header_v2_t, home) == 16, "v2 home @16");
    check(offsetof(eshkol_object_header_v2_t, aux) == 24, "v2 aux @24");

    // ---- the layout descriptor record ADR-0001 §3 specifies ----
    eshkol_layout_desc_t desc;
    desc.layout_id = 1;
    desc.flags = ESHKOL_LAYOUT_FLAG_LEAF;
    desc.min_size = 0;
    desc.trace = nullptr;
    desc.finalize = nullptr;
    check(desc.layout_id == 1 && desc.flags == ESHKOL_LAYOUT_FLAG_LEAF,
          "layout descriptor is a usable aggregate");
    check(ESHKOL_LAYOUT_ID_INVALID == 0,
          "layout id 0 is reserved as 'unassigned'");

    // ---- selection: which ABI is live, and does the runtime agree? ----
    const uint32_t compiled_against = (uint32_t)ESHKOL_MEMORY_ABI_ACTIVE;
    const uint32_t runtime_reports = eshkol_memory_abi_active();

    check(compiled_against == runtime_reports,
          "runtime archive and this translation unit agree on the object ABI");

#if defined(ESHKOL_MEMORY_ABI_V2_ENABLED) && (ESHKOL_MEMORY_ABI_V2_ENABLED)
    check(compiled_against == ESHKOL_MEMORY_ABI_V2,
          "ESHKOL_MEMORY_ABI_V2=ON selects ABI v2");
    check(ESHKOL_OBJECT_HEADER_SIZE == 32, "v2 selected: header prefix is 32 bytes");
    check(ESHKOL_OBJECT_PAYLOAD_ALIGN == 16, "v2 selected: payload alignment is 16");
#else
    check(compiled_against == ESHKOL_MEMORY_ABI_V1,
          "ESHKOL_MEMORY_ABI_V2 defaults to OFF, so ABI v1 stays live");
    check(ESHKOL_OBJECT_HEADER_SIZE == 8, "v1 selected: header prefix is 8 bytes");
    check(ESHKOL_OBJECT_PAYLOAD_ALIGN == 8, "v1 selected: payload alignment is 8");
    // The allocator must still be producing v1 objects: nothing in Phase A
    // migrated an allocation site, so this is the guard that says so out loud.
    check(sizeof(eshkol_object_header_active_t) == sizeof(eshkol_object_header_t),
          "v1 selected: the active header IS the v1 header");
#endif

    if (failures != 0) {
        std::cout << "memory_abi_v2_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "memory_abi_v2_test: all checks passed (ABI v"
              << runtime_reports << " active)\n";
    return 0;
}
