#include "../../lib/core/arena_memory.h"

#include <cmath>
#include <cstdint>
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

int require_null_cell(const arena_tagged_cons_cell_t& cell) {
    if (!is_null_tagged(cell.car)) return fail("car was not null-initialized");
    if (!is_null_tagged(cell.cdr)) return fail("cdr was not null-initialized");
    return 0;
}

}  // namespace

int main() {
    arena_t* arena = arena_create(1024);
    if (!arena) return fail("arena_create returned null");

    if (arena_allocate_tagged_cons_cell(nullptr) != nullptr) {
        return fail("null arena tagged cons allocation did not fail");
    }

    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return fail("tagged cons allocation returned null");
    if (int rc = require_null_cell(*cell)) return rc;

    arena_tagged_cons_set_int64(cell, false, 42, ESHKOL_VALUE_INT64);
    if (arena_tagged_cons_get_int64(cell, false) != 42) return fail("int64 getter mismatch");
    if (arena_tagged_cons_get_type(cell, false) != ESHKOL_VALUE_INT64) {
        return fail("int64 type mismatch");
    }
    if (!arena_tagged_cons_is_type(cell, false, ESHKOL_VALUE_INT64)) {
        return fail("base int64 type predicate mismatch");
    }

    arena_tagged_cons_set_double(cell, true, 3.5, ESHKOL_VALUE_DOUBLE);
    if (std::fabs(arena_tagged_cons_get_double(cell, true) - 3.5) > 0.000001) {
        return fail("double getter mismatch");
    }
    if (arena_tagged_cons_get_type(cell, true) != ESHKOL_VALUE_DOUBLE) {
        return fail("double type mismatch");
    }

    arena_tagged_cons_set_ptr(cell, true, 0x12345678ULL, ESHKOL_VALUE_HEAP_PTR);
    if (arena_tagged_cons_get_ptr(cell, true) != 0x12345678ULL) {
        return fail("pointer getter mismatch");
    }

    arena_tagged_cons_set_null(cell, true);
    if (arena_tagged_cons_get_ptr(cell, true) != 0) return fail("null pointer getter mismatch");
    if (arena_tagged_cons_get_type(cell, true) != ESHKOL_VALUE_NULL) {
        return fail("set_null type mismatch");
    }

    eshkol_tagged_value_t tagged;
    tagged.type = ESHKOL_VALUE_BOOL;
    tagged.flags = 7;
    tagged.reserved = 9;
    tagged.data.int_val = 1;
    arena_tagged_cons_set_tagged_value(cell, false, &tagged);
    eshkol_tagged_value_t copied = arena_tagged_cons_get_tagged_value(cell, false);
    if (copied.type != tagged.type ||
        copied.flags != tagged.flags ||
        copied.reserved != tagged.reserved ||
        copied.data.int_val != tagged.data.int_val) {
        return fail("tagged value copy mismatch");
    }
    if (arena_tagged_cons_get_flags(cell, false) != 7) return fail("flags getter mismatch");

    arena_tagged_cons_cell_t* batch = arena_allocate_tagged_cons_batch(arena, 3);
    if (!batch) return fail("tagged cons batch allocation returned null");
    for (size_t i = 0; i < 3; ++i) {
        if (int rc = require_null_cell(batch[i])) return rc;
    }
    if (arena_allocate_tagged_cons_batch(arena, 0) != nullptr) {
        return fail("zero-count batch allocation did not fail");
    }

    arena_tagged_cons_cell_t* ints = arena_create_int64_cons(
        arena, -2, ESHKOL_VALUE_INT64, 4, ESHKOL_VALUE_CHAR);
    if (!ints) return fail("int64 cons constructor returned null");
    if (arena_tagged_cons_get_int64(ints, false) != -2) return fail("constructor car mismatch");
    if (arena_tagged_cons_get_int64(ints, true) != 4) return fail("constructor cdr mismatch");

    eshkol_tagged_data_t car_data;
    car_data.double_val = 9.25;
    eshkol_tagged_data_t cdr_data;
    cdr_data.ptr_val = 0xABCDEFULL;
    arena_tagged_cons_cell_t* mixed = arena_create_mixed_cons(
        arena, car_data, ESHKOL_VALUE_DOUBLE, cdr_data, ESHKOL_VALUE_CALLABLE);
    if (!mixed) return fail("mixed cons constructor returned null");
    if (std::fabs(arena_tagged_cons_get_double(mixed, false) - 9.25) > 0.000001) {
        return fail("mixed car mismatch");
    }
    if (arena_tagged_cons_get_ptr(mixed, true) != 0xABCDEFULL) {
        return fail("mixed cdr pointer mismatch");
    }

    arena_destroy(arena);

    std::cout << "PASS\n";
    return 0;
}
