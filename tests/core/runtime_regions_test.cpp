#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

void set_int(eshkol_tagged_value_t& value, int64_t n) {
    value.type = ESHKOL_VALUE_INT64;
    value.flags = 0;
    value.reserved = 0;
    value.data.int_val = n;
}

}  // namespace

int main() {
    arena_t* shared = get_global_arena_shared();
    if (!shared) return fail("shared global arena is null");
    if (get_global_arena_shared() != shared) return fail("shared global arena changed");

    eshkol_thread_init_worker(2048);
    arena_t* local = arena_get_thread_local();
    if (!local) return fail("thread-local arena is null after worker init");
    if (local == shared) return fail("thread-local arena did not override shared global arena");
    if (get_global_arena() != local) return fail("get_global_arena did not return worker arena");

    void* fallback_alloc = region_allocate(24);
    if (!fallback_alloc) return fail("region_allocate fallback returned null");

    eshkol_region_t* outer = region_create("outer", 4096);
    if (!outer) return fail("outer region create failed");
    region_push(outer);
    if (region_get_depth() != 1) return fail("outer region depth mismatch");
    if (region_current() != outer) return fail("outer is not current region");
    if (!region_get_name(outer) || std::strcmp(region_get_name(outer), "outer") != 0) {
        return fail("outer region name mismatch");
    }

    auto* bytes = static_cast<unsigned char*>(region_allocate_zeroed(16));
    if (!bytes) return fail("region zeroed allocation failed");
    for (int i = 0; i < 16; ++i) {
        if (bytes[i] != 0) return fail("region zeroed allocation was not zero-filled");
    }
    std::memcpy(bytes, "region", 7);

    auto* escaped_outer = static_cast<unsigned char*>(region_escape(bytes, 7));
    if (!escaped_outer) return fail("region_escape to global failed");
    if (escaped_outer == bytes) return fail("region_escape did not copy while inside region");
    if (std::memcmp(escaped_outer, "region", 7) != 0) return fail("region_escape copied wrong bytes");
    if (outer->escape_count != 1) return fail("outer escape count after raw escape mismatch");

    const char* escaped_string = static_cast<const char*>(region_escape_string("hello"));
    if (!escaped_string) return fail("region_escape_string returned null");
    if (std::strcmp(escaped_string, "hello") != 0) return fail("escaped string content mismatch");
    if (ESHKOL_GET_HEADER(escaped_string)->subtype != HEAP_SUBTYPE_STRING) {
        return fail("escaped string is not header-backed");
    }
    if (outer->escape_count != 2) return fail("outer escape count after string escape mismatch");

    arena_tagged_cons_cell_t* cell = region_allocate_tagged_cons_cell();
    if (!cell) return fail("region cons allocation failed");
    set_int(cell->car, 11);
    set_int(cell->cdr, 22);
    arena_tagged_cons_cell_t* escaped_cell = region_escape_tagged_cons_cell(cell);
    if (!escaped_cell) return fail("region_escape_tagged_cons_cell returned null");
    if (escaped_cell == cell) return fail("region_escape_tagged_cons_cell did not copy");
    if (escaped_cell->car.data.int_val != 11 || escaped_cell->cdr.data.int_val != 22) {
        return fail("escaped cons cell contents mismatch");
    }
    if (outer->escape_count != 3) return fail("outer escape count after cons escape mismatch");

    char* local_string = static_cast<char*>(arena_allocate_string_with_header(outer->arena, 3));
    if (!local_string) return fail("local string allocation failed");
    std::memcpy(local_string, "abc", 4);
    eshkol_tagged_value_t tagged{};
    tagged.type = ESHKOL_VALUE_HEAP_PTR;
    tagged.data.ptr_val = reinterpret_cast<uint64_t>(local_string);
    eshkol_tagged_value_t escaped_tagged = region_escape_tagged_value(tagged);
    if (escaped_tagged.data.ptr_val == tagged.data.ptr_val) {
        return fail("region_escape_tagged_value did not copy heap value");
    }
    const char* escaped_tagged_string =
        reinterpret_cast<const char*>(escaped_tagged.data.ptr_val);
    if (std::strcmp(escaped_tagged_string, "abc") != 0) {
        return fail("escaped tagged string content mismatch");
    }
    if (outer->escape_count != 4) return fail("outer escape count after tagged escape mismatch");

    eshkol_region_t* inner = region_create("inner", 2048);
    if (!inner) return fail("inner region create failed");
    region_push(inner);
    if (region_get_depth() != 2) return fail("inner region depth mismatch");
    if (inner->parent != outer) return fail("inner parent mismatch");

    char* inner_data = static_cast<char*>(region_allocate(6));
    if (!inner_data) return fail("inner allocation failed");
    std::memcpy(inner_data, "inner", 6);
    char* escaped_inner = static_cast<char*>(region_escape(inner_data, 6));
    if (!escaped_inner) return fail("inner escape failed");
    if (std::strcmp(escaped_inner, "inner") != 0) return fail("inner escape content mismatch");
    if (inner->escape_count != 1) return fail("inner escape count mismatch");
    region_pop();
    if (region_get_depth() != 1) return fail("depth after inner pop mismatch");
    if (std::strcmp(escaped_inner, "inner") != 0) {
        return fail("inner escaped data did not survive inner pop");
    }

    eshkol_region_t* active = region_create("active", 1024);
    if (!active) return fail("active region create failed");
    region_push(active);
    if (region_get_depth() != 2) return fail("active region depth mismatch");
    region_destroy(active);
    if (region_get_depth() != 1) return fail("active region destroy did not pop once");

    region_pop();
    if (region_get_depth() != 0) return fail("depth after outer pop mismatch");

    arena_t* dest = arena_create(1024);
    arena_t* src = arena_create(2048);
    if (!dest || !src) return fail("merge test arena create failed");
    (void)arena_allocate(dest, 32);
    void* src_value = arena_allocate(src, 64);
    if (!src_value) return fail("merge source allocation failed");
    const size_t src_total = src->total_allocated;
    arena_merge_to_parent(dest, src);
    if (src->current_block != nullptr) return fail("merge did not transfer source blocks");
    if (src->total_allocated != 0) return fail("merge did not clear source total");
    if (dest->total_allocated < src_total) return fail("merge did not account destination total");
    arena_destroy(src);
    arena_destroy(dest);

    eshkol_thread_shutdown_worker();
    if (region_get_depth() != 0) return fail("worker shutdown did not clear region stack");
    if (arena_get_thread_local() != shared) {
        return fail("thread-local arena still overrides global after shutdown");
    }

    std::cout << "PASS\n";
    return 0;
}
