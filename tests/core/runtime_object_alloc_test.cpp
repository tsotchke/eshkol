#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool is_null_tagged(const eshkol_tagged_value_t& value) {
    return value.type == ESHKOL_VALUE_NULL &&
           value.flags == 0 &&
           value.reserved == 0 &&
           value.data.raw_val == 0;
}

int require_header(void* data, uint8_t subtype, uint8_t flags, uint32_t size) {
    if (!data) return fail("allocation returned null");

    eshkol_object_header_t* header = ESHKOL_GET_HEADER(data);
    if (header->subtype != subtype) return fail("header subtype mismatch");
    if (header->flags != flags) return fail("header flags mismatch");
    if (header->ref_count != 0) return fail("header ref_count was not initialized");
    if (header->size != size) return fail("header size mismatch");
    return 0;
}

}  // namespace

int main() {
    arena_t* arena = arena_create(1024);
    if (!arena) return fail("arena_create returned null");

    if (arena_allocate_with_header(nullptr, 16, HEAP_SUBTYPE_BYTEVECTOR, 0) != nullptr) {
        return fail("null arena header allocation did not fail");
    }
    if (arena_allocate_with_header(arena, 0, HEAP_SUBTYPE_BYTEVECTOR, 0) != nullptr) {
        return fail("zero-size header allocation did not fail");
    }

    void* raw = arena_allocate_with_header(arena, 24, HEAP_SUBTYPE_BYTEVECTOR, 3);
    if (int rc = require_header(raw, HEAP_SUBTYPE_BYTEVECTOR, 3, 24)) return rc;

    auto* zeroed = static_cast<uint8_t*>(
        arena_allocate_with_header_zeroed(arena, 16, HEAP_SUBTYPE_VECTOR, 0));
    if (int rc = require_header(zeroed, HEAP_SUBTYPE_VECTOR, 0, 16)) return rc;
    for (size_t i = 0; i < 16; ++i) {
        if (zeroed[i] != 0) return fail("zeroed header allocation contained non-zero data");
    }

    void* multi = arena_allocate_multi_value(arena, 2);
    const uint32_t multi_size =
        static_cast<uint32_t>(sizeof(size_t) + 2 * sizeof(eshkol_tagged_value_t));
    if (int rc = require_header(multi, HEAP_SUBTYPE_MULTI_VALUE, 0, multi_size)) return rc;
    if (*static_cast<size_t*>(multi) != 2) return fail("multi-value count mismatch");

    arena_tagged_cons_cell_t* cons = arena_allocate_cons_with_header(arena);
    if (int rc = require_header(cons, HEAP_SUBTYPE_CONS, 0, sizeof(arena_tagged_cons_cell_t))) {
        return rc;
    }
    if (!is_null_tagged(cons->car)) return fail("cons car was not null-initialized");
    if (!is_null_tagged(cons->cdr)) return fail("cons cdr was not null-initialized");

    char* str = arena_allocate_string_with_header(arena, 3);
    if (int rc = require_header(str, HEAP_SUBTYPE_STRING, 0, 4)) return rc;
    if (str[0] != '\0') return fail("string was not initialized with leading NUL");
    std::memcpy(str, "abc", 4);
    if (std::strcmp(str, "abc") != 0) return fail("string payload write/read failed");

    auto* vector_data = static_cast<uint8_t*>(arena_allocate_vector_with_header(arena, 2));
    const uint32_t vector_size =
        static_cast<uint32_t>(8 + 2 * sizeof(eshkol_tagged_value_t));
    if (int rc = require_header(vector_data, HEAP_SUBTYPE_VECTOR, 0, vector_size)) return rc;
    *reinterpret_cast<int64_t*>(vector_data) = 2;

    void* symbol = arena_allocate_symbol_with_header(arena, 5);
    if (int rc = require_header(symbol, HEAP_SUBTYPE_SYMBOL, 0, 14)) return rc;

    if (arena_allocate_with_header(arena, UINT64_MAX, HEAP_SUBTYPE_VECTOR, 0) != nullptr) {
        return fail("overflowing header allocation did not fail");
    }

    arena_destroy(arena);

    std::cout << "PASS\n";
    return 0;
}
