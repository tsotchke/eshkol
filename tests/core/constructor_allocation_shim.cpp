// Test-only link wrappers for constructor allocators and handler malloc.
#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t*);
extern "C" void* __real_malloc(size_t);
extern "C" void* __real_arena_allocate_aligned(arena_t*, size_t, size_t);
extern "C" void* __real_arena_allocate_vector_with_header(arena_t*, size_t);
extern "C" arena_tagged_cons_cell_t* __real_arena_allocate_cons_with_header(arena_t*);
extern "C" eshkol_closure_t* __real_arena_allocate_closure_with_header(
    arena_t*, uint64_t, uint64_t, uint64_t, uint64_t, const char*);
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
extern "C" eshkol_closure_t* __wrap_arena_allocate_closure_with_header(
    arena_t* arena, uint64_t function, uint64_t captures, uint64_t sexpr,
    uint64_t return_type, const char* name) {
    return fail(5) ? nullptr : __real_arena_allocate_closure_with_header(
        arena, function, captures, sexpr, return_type, name);
}
extern "C" void* __wrap_malloc(size_t n) {
    if (n == sizeof(eshkol_exception_handler_t) && fail(3)) return nullptr;
    return __real_malloc(n);
}
extern "C" void* __wrap_arena_allocate_aligned(arena_t* arena, size_t bytes, size_t alignment) {
    if (bytes == sizeof(eshkol_closure_env_t) + sizeof(eshkol_tagged_value_t) && fail(4))
        return nullptr;
    return __real_arena_allocate_aligned(arena, bytes, alignment);
}
extern "C" int64_t constructor_test_caught() {
    check(!armed && steps == expected, "failure consumed without later operand evaluation");
    eshkol_tagged_value_t value{}; eshkol_get_raised_value(&value);
    check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "allocation condition tag");
    auto* exception = reinterpret_cast<eshkol_exception_t*>(value.data.ptr_val);
    check(exception == g_current_exception && exception &&
          std::strstr(exception->message, "out of memory") != nullptr,
          "allocation failure reached established handler");
    ++caught; return 0;
}
extern "C" int64_t constructor_test_reused_list() {
    check(armed == 2 && steps == 1 && caught == 7, "apply reused existing rest list");
    armed = 0; ++reused; return 0;
}
extern "C" int64_t constructor_test_finish() {
    check(cases == 10 && caught == 9 && reused == 1 && !armed,
          "nine constructor failures caught and existing-list apply validated");
    std::puts("PASS AOT nine constructor null failures/order and existing-list apply"); return 0;
}
