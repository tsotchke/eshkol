#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t*);
static uintptr_t identity;
static int catches;
static uintptr_t unsupported_identity;
static int unsupported_catches;
static eshkol_region_t* unsupported_source;
static void* unsupported_map;
static arena_t* unsupported_target;
static size_t unsupported_escapes;
static eshkol_tensor_t *unsupported_full, *unsupported_small;
static eshkol_tensor_t original_full, original_small;
static int64_t* unsupported_elements;
static eshkol_promotion_test_stats unsupported_baseline;
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL AOT barrier: %s\n", message); std::abort(); }
}
extern "C" int64_t checked_barrier_test_arm() {
    eshkol_promotion_test_emergency_reset();
    // One successful speculative copy followed by failure exposes premature
    // partial-graph output publication, not just the first allocation branch.
    eshkol_promotion_test_arm(4, 1);
    return 0;
}
extern "C" int64_t checked_barrier_test_caught(int64_t ordinal) {
    eshkol_tagged_value_t value{}; eshkol_get_raised_value(&value);
    check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "canonical emergency tag");
    auto* exception = reinterpret_cast<eshkol_exception_t*>(value.data.ptr_val);
    check(exception == g_current_exception && exception &&
          !std::strcmp(exception->message, "region promotion allocation failed"),
          "promotion condition one");
    if (ordinal == 1) identity = value.data.ptr_val;
    else check(value.data.ptr_val == identity, "explicit raise preserves exact identity");
    check(++catches == ordinal && eshkol_promotion_test_emergency_transfers() == ordinal,
          "one transfer per original raise and explicit rethrow");
    const auto stats = eshkol_promotion_test_snapshot();
    check(stats.attempts[4] == 2, "failure after one speculative allocation");
    for (auto live : stats.live_bytes) check(live == 0, "transaction scratch/map rollback");
    if (ordinal == 2) eshkol_promotion_test_reset();
    return 0;
}
extern "C" int64_t checked_barrier_test_unsupported_graph(void* graph) {
    // Tests-only factory: fill a genuine Scheme vector in its current region.
    // The LIFO walker sees the small same-base raw span before the full span.
    unsupported_source = region_current();
    check(unsupported_source && graph && *static_cast<uint64_t*>(graph) == 2,
          "unsupported fixture root");
    auto* arena = unsupported_source->arena;
    unsupported_elements = static_cast<int64_t*>(arena_allocate(arena, 3 * sizeof(double)));
    unsupported_full = arena_allocate_tensor_with_header(arena);
    unsupported_small = arena_allocate_tensor_with_header(arena);
    auto* full_dims = static_cast<uint64_t*>(arena_allocate(arena, sizeof(uint64_t)));
    auto* small_dims = static_cast<uint64_t*>(arena_allocate(arena, sizeof(uint64_t)));
    check(unsupported_elements && unsupported_full && unsupported_small && full_dims && small_dims,
          "unsupported fixture allocations");
    const double elements[] = {1.5, 2.5, 3.5};
    std::memcpy(unsupported_elements, elements, sizeof elements);
    *full_dims = 3; *small_dims = 1;
    unsupported_full->dimensions = full_dims; unsupported_full->num_dimensions = 1;
    unsupported_full->elements = unsupported_elements; unsupported_full->total_elements = 3;
    unsupported_full->dtype = ESHKOL_TENSOR_DTYPE_F64;
    unsupported_small->dimensions = small_dims; unsupported_small->num_dimensions = 1;
    unsupported_small->elements = unsupported_elements; unsupported_small->total_elements = 1;
    unsupported_small->dtype = ESHKOL_TENSOR_DTYPE_F64;
    std::memcpy(&original_full, unsupported_full, sizeof original_full);
    std::memcpy(&original_small, unsupported_small, sizeof original_small);
    auto* values = reinterpret_cast<eshkol_tagged_value_t*>(static_cast<char*>(graph) + 8);
    values[0] = {}; values[0].type = ESHKOL_VALUE_HEAP_PTR;
    values[0].data.ptr_val = reinterpret_cast<uintptr_t>(unsupported_full);
    values[1] = {}; values[1].type = ESHKOL_VALUE_HEAP_PTR;
    values[1].data.ptr_val = reinterpret_cast<uintptr_t>(unsupported_small);
    unsupported_map = unsupported_source->fwd_map;
    unsupported_target = unsupported_source->fwd_target;
    unsupported_escapes = unsupported_source->escape_count;
    unsupported_baseline = eshkol_promotion_test_snapshot();
    eshkol_promotion_test_emergency_reset();
    return 0;
}
extern "C" int64_t checked_barrier_test_unsupported_caught(int64_t ordinal) {
    eshkol_tagged_value_t value{}; eshkol_get_raised_value(&value);
    check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "unsupported canonical emergency tag");
    auto* exception = reinterpret_cast<eshkol_exception_t*>(value.data.ptr_val);
    check(exception == g_current_exception && exception &&
          !std::strcmp(exception->message, "region promotion layout is unsupported"),
          "promotion condition two");
    if (ordinal == 1) unsupported_identity = value.data.ptr_val;
    else check(value.data.ptr_val == unsupported_identity, "unsupported exact rethrow identity");
    check(value.data.ptr_val != identity, "condition two distinct from condition one");
    check(++unsupported_catches == ordinal &&
          eshkol_promotion_test_emergency_transfers() == ordinal,
          "unsupported original and rethrow transfer counts");
    check(region_current() == unsupported_source && unsupported_source->fwd_map == unsupported_map &&
          unsupported_source->fwd_target == unsupported_target &&
          unsupported_source->escape_count == unsupported_escapes,
          "unsupported publication leaves source level and committed map unchanged");
    check(!std::memcmp(unsupported_full, &original_full, sizeof original_full) &&
          !std::memcmp(unsupported_small, &original_small, sizeof original_small),
          "unsupported source headers unchanged");
    const auto stats = eshkol_promotion_test_snapshot();
    for (int site = 0; site < 4; ++site)
        check(stats.live_bytes[site] == unsupported_baseline.live_bytes[site],
              "unsupported scratch released before guard transfer");
    return 0;
}
extern "C" int64_t checked_barrier_test_unsupported_survived(void* graph) {
    check(unsupported_catches == 2 && graph && *static_cast<uint64_t*>(graph) == 2,
          "unsupported retry published after both guards");
    auto* values = reinterpret_cast<eshkol_tagged_value_t*>(static_cast<char*>(graph) + 8);
    auto* small = reinterpret_cast<eshkol_tensor_t*>(values[0].data.ptr_val);
    auto* full = reinterpret_cast<eshkol_tensor_t*>(values[1].data.ptr_val);
    check(small != unsupported_small && full != unsupported_full &&
          full->elements != unsupported_elements && small->elements == full->elements,
          "valid retry preserves canonical raw-buffer alias after poisoned exit");
    double last = 0; std::memcpy(&last, full->elements + 2, sizeof last);
    check(full->total_elements == 3 && small->total_elements == 1 && last == 3.5,
          "valid retry graph survives poisoned exit");
    return 0;
}
extern "C" int64_t checked_barrier_test_finish() {
    check(catches == 2 && unsupported_catches == 2, "both conditions traversed both established guards");
    std::puts("PASS AOT conditions 1/2 unchanged holders, marker restoration, exact rethrow and poisoned retry");
    return 0;
}
