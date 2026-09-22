// Test-only link wrappers: --wrap=arena_allocate_vector_with_header,
// --wrap=arena_allocate_cons_with_header, --wrap=malloc.
#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t*);
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_arena_allocate_vector_with_header(arena_t*, size_t);
extern "C" arena_tagged_cons_cell_t* __real_arena_allocate_cons_with_header(arena_t*);
static int armed, steps, expected, caught, cases, reused;
static void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL constructor: %s (case=%d caught=%d armed=%d steps=%d)\n",
                     message, cases, caught, armed, steps);
        std::abort();
    }
}
extern "C" int64_t constructor_test_arm(int64_t kind, int64_t before) {
    check(!armed, "previous failure not consumed");
    armed = static_cast<int>(kind); steps = 0; expected = static_cast<int>(before);
    ++cases;
    return 0;
}
extern "C" int64_t constructor_test_step(int64_t value) { ++steps; return value; }
static bool fail(int kind) {
    if (armed != kind) return false;
    check(steps == expected, "operand evaluation order at allocation");
    armed = 0; return true;
}
extern "C" void* __wrap_arena_allocate_vector_with_header(arena_t* arena, size_t n) {
    return fail(1) ? nullptr : __real_arena_allocate_vector_with_header(arena, n);
}
extern "C" arena_tagged_cons_cell_t* __wrap_arena_allocate_cons_with_header(arena_t* arena) {
    return fail(2) ? nullptr : __real_arena_allocate_cons_with_header(arena);
}
extern "C" void* __wrap_malloc(size_t n) {
    if (n == sizeof(eshkol_exception_handler_t) && fail(3)) return nullptr;
    return __real_malloc(n);
}
extern "C" int64_t constructor_test_caught() {
    check(!armed && steps == expected, "failure consumed without later operand evaluation");
    eshkol_tagged_value_t value{}; eshkol_get_raised_value(&value);
    check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "canonical condition tag");
    auto* exception = reinterpret_cast<eshkol_exception_t*>(value.data.ptr_val);
    check(exception == g_current_exception && exception &&
          !std::strcmp(exception->message, "object or exception-handler allocation failed"),
          "distinct allocation condition 5 reached established handler");
    ++caught; return 0;
}
extern "C" int64_t constructor_test_reused_list() {
    check(armed == 2 && steps == 1 && caught == 7, "apply reused existing rest list");
    armed = 0; ++reused; return 0;
}
extern "C" int64_t constructor_test_finish() {
    check(cases == 8 && caught == 7 && reused == 1 && !armed,
          "seven constructor failures caught and existing-list apply validated");
    std::puts("PASS AOT seven constructor null failures/order and existing-list apply"); return 0;
}
