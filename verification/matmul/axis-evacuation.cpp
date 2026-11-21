// Exercise supported outward stores. Inspect destination ownership before popping
// the young region. No pointer into a popped region is read.
#include "arena_memory.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>

static void require(bool condition, const char* reason) {
    if (!condition) {
        std::fprintf(stderr, "axis evacuation: %s\n", reason);
        std::exit(1);
    }
}

static eshkol_tagged_value_t tag(eshkol_tensor_t* t) {
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.data.ptr_val = reinterpret_cast<uintptr_t>(t);
    return v;
}

enum class Axes { sharedYoung, sharedOld, distinctYoung };

static void observe(Axes axes, bool batch) {
    auto* target = get_global_arena_shared();
    auto* region = region_create("axis-probe", 4096);
    require(target && region, "missing arena");
    region_push(region);
    auto* first = arena_allocate_tensor_full(region->arena, 2, 6);
    auto* second = arena_allocate_tensor_full(region->arena, 2, 6);
    require(first && second, "allocation failed");
    if (axes == Axes::sharedOld) {
        auto* older = arena_allocate_tensor_full(target, 2, 6);
        require(older, "older tensor allocation failed");
        first->dimensions = older->dimensions;
    }
    if (axes != Axes::distinctYoung) second->dimensions = first->dimensions;
    for (auto* t : {first, second}) {
        t->dimensions[0] = 3;
        t->dimensions[1] = 2;
    }
    first->elements[5] = 41;
    second->elements[5] = 42;

    std::array<eshkol_tagged_value_t, 2> roots{tag(first), tag(second)};
    if (batch) {
        eshkol_region_write_barrier_range(nullptr, roots.data(), roots.size());
    } else {
        for (auto& root : roots) eshkol_region_write_barrier_into(&root, nullptr, &root);
    }
    std::array<const eshkol_tensor_t*, 2> live{};
    for (size_t i = 0; i < live.size(); ++i) {
        require(roots[i].type == ESHKOL_VALUE_HEAP_PTR, "missing tensor root");
        live[i] = reinterpret_cast<const eshkol_tensor_t*>(roots[i].data.ptr_val);
        require(arena_contains(target, live[i]), "header outside destination");
        require(arena_contains(target, live[i]->dimensions), "axes outside destination");
        require(arena_contains(target, live[i]->elements), "elements outside destination");
    }
    require(live[0] != live[1], "headers coalesced");
    require(live[0]->elements != live[1]->elements, "element buffers coalesced");
    require((live[0]->dimensions == live[1]->dimensions) == (axes != Axes::distinctYoung),
            "axis sharing changed");
    const bool unmoved = live[0]->dimensions == first->dimensions;
    require(unmoved == (axes == Axes::sharedOld),
            "incorrect handling of axis ownership");
    region_pop();
    for (size_t i = 0; i < live.size(); ++i) {
        auto* t = live[i];
        require(t->num_dimensions == 2 && t->total_elements == 6 &&
                t->dimensions[0] == 3 && t->dimensions[1] == 2, "shape changed");
        require(t->elements[5] == static_cast<int64_t>(41 + i), "element value changed");
    }
    const char* name = axes == Axes::sharedOld ? "shared-old" :
                       axes == Axes::sharedYoung ? "shared-young" : "distinct-young";
    std::printf("%s %s %s %s %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                batch ? "batch" : "separate", name,
                live[0]->dimensions == live[1]->dimensions ? "true" : "false",
                unmoved ? "true" : "false", live[0]->dimensions[0], live[0]->dimensions[1],
                live[1]->dimensions[0], live[1]->dimensions[1]);
}

int main() {
    for (bool batch : {true, false}) {
        for (Axes axes : {Axes::sharedYoung, Axes::sharedOld, Axes::distinctYoung})
            observe(axes, batch);
    }
}
