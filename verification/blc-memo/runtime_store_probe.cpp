// Linux/ELF probe. The linker redirects references to the tagged setter here.
// Keep allocations and unrelated calls out of the publication-store count.
#include "lib/core/arena_memory.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static_assert(sizeof(eshkol_tagged_value_t) == 16);
static_assert(offsetof(arena_tagged_cons_cell_t, car) == 0);
static_assert(offsetof(arena_tagged_cons_cell_t, cdr) == 16);

extern "C" void __real_arena_tagged_cons_set_tagged_value(
    arena_tagged_cons_cell_t*, bool, const eshkol_tagged_value_t*);

static bool same_value(const eshkol_tagged_value_t& a, const eshkol_tagged_value_t& b) {
    return a.type == b.type && a.flags == b.flags && a.reserved == b.reserved &&
           std::memcmp(&a.data, &b.data, sizeof(a.data)) == 0;
}

// A legal idempotent store on a live test-owned tag byte. Keeping this
// function out of line gives the independent instruction counter a boundary.
extern "C" __attribute__((noinline)) void blc_probe_direct_tag(volatile uint8_t* tag, uint8_t value) {
    *tag = value;
}

extern "C" void __wrap_arena_tagged_cons_set_tagged_value(
    arena_tagged_cons_cell_t* cell, bool cdr, const eshkol_tagged_value_t* input) {
    Dl_info caller{};
    bool named = dladdr(__builtin_return_address(0), &caller) && caller.dli_sname;
    bool payload = named && std::strcmp(caller.dli_sname, "__core_blc-fast__audit-raw-cdr!") == 0;
    bool tag = named && std::strcmp(caller.dli_sname, "__core_blc-fast__audit-raw-car!") == 0;
    if (!payload && !tag) {
        __real_arena_tagged_cons_set_tagged_value(cell, cdr, input);
        return;
    }
    static arena_tagged_cons_cell_t* publishing = nullptr;
    if (cdr != payload || (cdr ? publishing != nullptr : publishing != cell)) std::abort();
    publishing = cdr ? cell : nullptr;
    const auto& selected = cdr ? cell->cdr : cell->car;
    const auto& peer = cdr ? cell->car : cell->cdr;
    const auto original_peer = peer;
    const auto expected = *input;
    const int copies = payload && std::getenv("BLC_PROBE_DUPLICATE") ? 2 : 1;
    for (int copy = 0; copy < copies; ++copy) {
        __real_arena_tagged_cons_set_tagged_value(cell, cdr, input);
        if (payload && std::getenv("BLC_PROBE_DIRECT"))
            blc_probe_direct_tag(&cell->cdr.type, expected.type);
        if (!same_value(selected, expected) || !same_value(peer, original_peer)) std::abort();
        std::puts(cdr ? "RUNTIME CDR" : "RUNTIME CAR");
    }
}
