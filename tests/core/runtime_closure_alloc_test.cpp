#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool capture_is_null(const eshkol_tagged_value_t& value) {
    return value.type == ESHKOL_VALUE_NULL &&
           value.flags == 0 &&
           value.reserved == 0 &&
           value.data.raw_val == 0;
}

}  // namespace

int main() {
    arena_t* arena = arena_create(4096);
    if (!arena) return fail("arena_create returned null");

    eshkol_closure_env_t* env = arena_allocate_closure_env(arena, 2);
    if (!env) return fail("closure environment allocation returned null");
    if (env->num_captures != 2) return fail("closure environment count mismatch");
    if (!capture_is_null(env->captures[0])) return fail("first capture was not null-initialized");
    if (!capture_is_null(env->captures[1])) return fail("second capture was not null-initialized");

    const uint64_t func_ptr = 0x1020304050607080ULL;
    const uint64_t sexpr_ptr = 0x0102030405060708ULL;
    const uint64_t hott_type_id = 0x00ABCDEFULL;
    const uint64_t return_info =
        CLOSURE_RETURN_VECTOR | (uint64_t(3) << 8) | (hott_type_id << 16);
    const size_t packed_info = CLOSURE_ENV_PACK(2, 3, 1);

    eshkol_closure_t* closure =
        arena_allocate_closure(arena, func_ptr, packed_info, sexpr_ptr, return_info, "worker");
    if (!closure) return fail("closure allocation returned null");
    if (closure->func_ptr != func_ptr) return fail("closure func pointer mismatch");
    if (closure->sexpr_ptr != sexpr_ptr) return fail("closure sexpr pointer mismatch");
    if (closure->return_type != CLOSURE_RETURN_VECTOR) return fail("closure return type mismatch");
    if (closure->input_arity != 3) return fail("closure input arity mismatch");
    if (closure->hott_type_id != hott_type_id) return fail("closure HoTT type id mismatch");
    if ((closure->flags & CLOSURE_FLAG_VARIADIC) == 0) return fail("variadic flag missing");
    if ((closure->flags & ESHKOL_CLOSURE_FLAG_NAMED) == 0) return fail("named flag missing");
    if (!closure->env) return fail("closure environment missing");
    if (closure->env->num_captures != packed_info) return fail("packed capture info not preserved");

    const uint64_t scalar_return_info =
        CLOSURE_RETURN_SCALAR | (uint64_t(1) << 8) | (uint64_t(17) << 16);
    eshkol_closure_t* header_lambda = arena_allocate_closure_with_header(
        arena, func_ptr + 1, CLOSURE_ENV_PACK(0, 1, 0), sexpr_ptr + 1,
        scalar_return_info, nullptr);
    if (!header_lambda) return fail("header lambda allocation returned null");
    eshkol_object_header_t* lambda_header = ESHKOL_GET_HEADER(header_lambda);
    if (lambda_header->subtype != CALLABLE_SUBTYPE_LAMBDA_SEXPR) {
        return fail("zero-capture header closure did not use lambda-sexpr subtype");
    }
    if (lambda_header->size != sizeof(eshkol_closure_t)) return fail("header lambda size mismatch");
    if (header_lambda->env != nullptr) return fail("zero-capture header lambda has env");
    if (header_lambda->flags != 0) return fail("anonymous non-variadic header lambda flags mismatch");

    eshkol_closure_t* header_closure = arena_allocate_closure_with_header(
        arena, func_ptr + 2, CLOSURE_ENV_PACK(1, 1, 0), sexpr_ptr + 2,
        scalar_return_info, "capturing");
    if (!header_closure) return fail("header closure allocation returned null");
    eshkol_object_header_t* closure_header = ESHKOL_GET_HEADER(header_closure);
    if (closure_header->subtype != CALLABLE_SUBTYPE_CLOSURE) {
        return fail("capturing header closure did not use closure subtype");
    }
    if (!header_closure->env) return fail("capturing header closure env missing");
    if (CLOSURE_ENV_GET_NUM_CAPTURES(header_closure->env->num_captures) != 1) {
        return fail("capturing header closure capture count mismatch");
    }
    if (!capture_is_null(header_closure->env->captures[0])) {
        return fail("capturing header closure capture was not null-initialized");
    }
    if ((header_closure->flags & ESHKOL_CLOSURE_FLAG_NAMED) == 0) {
        return fail("capturing header closure named flag missing");
    }

    arena_destroy(arena);

    std::cout << "PASS\n";
    return 0;
}
