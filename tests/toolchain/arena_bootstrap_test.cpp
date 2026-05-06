#include <eshkol/core/arena.h>
#include <eshkol/eshkol.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
eshkol_arena_t* get_global_arena(void);
void arena_push_scope(eshkol_arena_t* arena);
void arena_pop_scope(eshkol_arena_t* arena);
void* region_create(const char* name, size_t size_hint);
void region_push(void* region);
void region_pop(void);
void* region_allocate(size_t size);
void* region_escape(const void* ptr, size_t size);
}

namespace {

template <typename T>
bool expect_equal(const T& actual, const T& expected, const char* label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << "FAIL: " << label << std::endl;
    return false;
}

bool expect_true(bool value, const char* label) {
    return expect_equal(value, true, label);
}

bool expect_false(bool value, const char* label) {
    return expect_equal(value, false, label);
}

bool test_embedded_arena_surface() {
    const size_t storage_bytes = eshkol_arena_embedded_bytes(2048);
    std::vector<unsigned char> storage(storage_bytes);

    eshkol_arena_t* arena =
        eshkol_arena_init_embedded(storage.data(), storage.size());
    if (!expect_true(arena != nullptr, "embedded arena initializes")) {
        return false;
    }

    void* raw = eshkol_arena_allocate(arena, 128);
    void* aligned = eshkol_arena_allocate_aligned(arena, 64, 16);
    void* zeroed = eshkol_arena_allocate_zeroed(arena, 32);

    const bool ok =
        expect_true(raw != nullptr, "embedded arena allocates raw storage") &&
        expect_true(aligned != nullptr, "embedded arena allocates aligned storage") &&
        expect_true(zeroed != nullptr, "embedded arena allocates zeroed storage") &&
        expect_equal(reinterpret_cast<uintptr_t>(aligned) % 16, uintptr_t{0},
                     "embedded arena preserves requested alignment") &&
        expect_equal(std::memcmp(zeroed, std::vector<unsigned char>(32, 0).data(), 32), 0,
                     "embedded arena zeroed allocation starts cleared") &&
        expect_false(eshkol_arena_supports_heap_growth(arena),
                     "embedded arena disables heap growth") &&
        expect_true(eshkol_arena_used_bytes(arena) >= 224,
                    "embedded arena tracks used bytes") &&
        expect_true(eshkol_arena_total_bytes(arena) >= 2048,
                    "embedded arena exposes total bootstrap capacity");

    eshkol_arena_destroy(arena);
    return ok;
}

bool test_embedded_arena_exhaustion_and_reset() {
    const size_t storage_bytes = eshkol_arena_embedded_bytes(1024);
    std::vector<unsigned char> storage(storage_bytes);

    eshkol_arena_t* arena =
        eshkol_arena_init_embedded(storage.data(), storage.size());
    if (!expect_true(arena != nullptr, "embedded exhaustion arena initializes")) {
        return false;
    }

    const size_t total = eshkol_arena_total_bytes(arena);
    void* large = eshkol_arena_allocate(arena, total > 128 ? total - 128 : total);
    void* overflow = eshkol_arena_allocate(arena, 256);

    const bool ok =
        expect_true(large != nullptr, "embedded arena accepts near-capacity allocation") &&
        expect_true(overflow == nullptr,
                    "embedded arena fails instead of silently heap-growing") &&
        expect_true(eshkol_arena_used_bytes(arena) > 0,
                    "embedded arena reports non-zero usage before reset");

    eshkol_arena_reset(arena);
    const bool reset_ok =
        expect_equal(eshkol_arena_used_bytes(arena), size_t{0},
                     "embedded arena reset rewinds usage");

    eshkol_arena_destroy(arena);
    return ok && reset_ok;
}

bool test_heap_arena_growth_surface() {
    eshkol_arena_t* arena = eshkol_arena_create_heap(1024);
    if (!expect_true(arena != nullptr, "heap arena initializes")) {
        return false;
    }

    void* first = eshkol_arena_allocate(arena, 900);
    void* second = eshkol_arena_allocate(arena, 900);

    const bool ok =
        expect_true(first != nullptr, "heap arena handles first allocation") &&
        expect_true(second != nullptr, "heap arena grows for later allocations") &&
        expect_true(eshkol_arena_supports_heap_growth(arena),
                    "heap arena reports heap growth support") &&
        expect_true(eshkol_arena_total_bytes(arena) > 1024,
                    "heap arena total bytes increase after growth");

    eshkol_arena_destroy(arena);
    return ok;
}

bool test_runtime_global_binding_and_embedded_runtime_surface() {
    const size_t primary_bytes = eshkol_arena_embedded_bytes(8192);
    const size_t secondary_bytes = eshkol_arena_embedded_bytes(2048);
    std::vector<unsigned char> primary_storage(primary_bytes);
    std::vector<unsigned char> secondary_storage(secondary_bytes);

    eshkol_arena_t* primary =
        eshkol_arena_init_embedded(primary_storage.data(), primary_storage.size());
    eshkol_arena_t* secondary =
        eshkol_arena_init_embedded(secondary_storage.data(), secondary_storage.size());
    if (!expect_true(primary != nullptr, "primary runtime arena initializes") ||
        !expect_true(secondary != nullptr, "secondary runtime arena initializes")) {
        return false;
    }

    eshkol_lambda_registry_destroy();

    bool ok =
        expect_true(eshkol_arena_bind_runtime_global(primary),
                    "runtime global arena binds on first use") &&
        expect_true(eshkol_arena_bind_runtime_global(primary),
                    "runtime global arena accepts idempotent rebind") &&
        expect_true(get_global_arena() == primary,
                    "hosted global arena lookup respects the prebound runtime arena") &&
        expect_false(eshkol_arena_bind_runtime_global(secondary),
                     "runtime global arena rejects a different arena");

    const size_t before_init = eshkol_arena_used_bytes(primary);
    eshkol_lambda_registry_init();
    const size_t after_init = eshkol_arena_used_bytes(primary);

    for (uint64_t i = 0; i < 80; ++i) {
        eshkol_lambda_registry_add(0x1000 + i, 0x2000 + i, "runtime-lambda");
    }
    const size_t after_growth = eshkol_arena_used_bytes(primary);

    ok = ok &&
         expect_true(g_lambda_registry != nullptr,
                     "lambda registry initializes after runtime arena binding") &&
         expect_true(after_init > before_init,
                     "lambda registry initialization consumes bound arena bytes") &&
         expect_true(after_growth > after_init,
                     "lambda registry growth stays within the bound arena") &&
         expect_equal(eshkol_lambda_registry_lookup(0x104f), uint64_t{0x204f},
                      "lambda registry lookup returns stored S-expression pointer") &&
         expect_equal(eshkol_lambda_registry_lookup(0x9999), uint64_t{0},
                      "lambda registry lookup misses cleanly");

    const size_t before_scope = eshkol_arena_used_bytes(primary);
    arena_push_scope(primary);
    const size_t after_scope_push = eshkol_arena_used_bytes(primary);
    void* scoped_value = eshkol_arena_allocate(primary, 96);
    const size_t after_scope_alloc = eshkol_arena_used_bytes(primary);
    arena_pop_scope(primary);
    const size_t after_scope_pop = eshkol_arena_used_bytes(primary);

    ok = ok &&
         expect_true(after_scope_push > before_scope,
                     "arena scope metadata is allocated inside the arena") &&
         expect_true(scoped_value != nullptr,
                     "embedded runtime arena still allocates inside a scope") &&
         expect_true(after_scope_alloc > after_scope_push,
                     "scoped allocations advance arena usage") &&
         expect_equal(after_scope_pop, before_scope,
                      "popping a scope rewinds both scope metadata and scoped allocations");

    const size_t before_region_create = eshkol_arena_used_bytes(primary);
    void* parent_region = region_create("embedded-parent", 1024);
    const size_t after_parent_region_create = eshkol_arena_used_bytes(primary);

    region_push(parent_region);
    void* parent_payload = region_allocate(48);
    void* child_region = region_create("embedded-child", 1024);
    const size_t after_child_region_create = eshkol_arena_used_bytes(primary);
    region_push(child_region);
    unsigned char* child_payload =
        reinterpret_cast<unsigned char*>(region_allocate(32));
    std::vector<unsigned char> expected_escape(32);
    for (size_t i = 0; i < 32; ++i) {
        expected_escape[i] = static_cast<unsigned char>(i + 1);
        child_payload[i] = expected_escape[i];
    }
    void* escaped_payload = region_escape(child_payload, 32);
    region_pop();  // child

    const bool escape_ok =
        escaped_payload != nullptr &&
        std::memcmp(escaped_payload, expected_escape.data(), expected_escape.size()) == 0;

    region_pop();  // parent

    void* reused_region = region_create("embedded-reuse", 1024);
    const size_t after_reuse_region_create = eshkol_arena_used_bytes(primary);
    region_push(reused_region);
    void* reused_payload = region_allocate(24);
    region_pop();

    ok = ok &&
         expect_true(parent_region != nullptr,
                     "embedded no-heap runtime creates parent regions") &&
         expect_true(parent_payload != nullptr,
                     "embedded parent region allocates inside its subarena") &&
         expect_true(child_region != nullptr,
                     "embedded no-heap runtime creates nested child regions") &&
         expect_true(child_payload != nullptr,
                     "embedded child region allocates inside its subarena") &&
         expect_true(escape_ok,
                     "embedded child-to-parent region_escape preserves escaped bytes") &&
         expect_true(after_parent_region_create > before_region_create,
                     "first embedded region creation consumes global runtime arena bytes") &&
         expect_true(after_child_region_create > after_parent_region_create,
                     "nested embedded region creation consumes additional global arena bytes") &&
         expect_true(reused_region != nullptr,
                     "destroyed embedded region chunks can be reused") &&
         expect_true(reused_payload != nullptr,
                     "reused embedded regions remain allocatable") &&
         expect_equal(after_reuse_region_create, after_child_region_create,
                      "reused embedded regions do not consume more global arena bytes") &&
         expect_true(get_global_arena() == primary,
                     "embedded region lifecycle does not clobber the runtime-global arena");

    eshkol_lambda_registry_destroy();
    eshkol_arena_destroy(secondary);
    // Keep the bound runtime arena alive until process exit because the binding
    // is intentionally singleton and cannot be redirected to another arena.
    return ok;
}

}  // namespace

int main() {
    if (!test_embedded_arena_surface()) {
        return 1;
    }
    if (!test_embedded_arena_exhaustion_and_reset()) {
        return 1;
    }
    if (!test_heap_arena_growth_surface()) {
        return 1;
    }
    if (!test_runtime_global_binding_and_embedded_runtime_surface()) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
