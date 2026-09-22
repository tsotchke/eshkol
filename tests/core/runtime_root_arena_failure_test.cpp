// Fresh-process root initialization refusal must transfer before returning a
// null allocation owner. The existing once initializer does not retry failure.
#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>

static unsigned root_attempts;
extern "C" arena_t* __wrap_arena_create_threadsafe(size_t) {
    ++root_attempts;
    return nullptr;
}
static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
static eshkol_exception_t* catch_invalid_root(bool getter) {
    jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    if (setjmp(handler) == 0) {
        if (getter) (void)eshkol_root_arena_v1();
        else eshkol_runtime_emergency_raise_v1(4);
        check(false, "root failure must not return");
    }
    auto* caught = g_current_exception;
    check(caught, "fixed exception delivered");
    eshkol_pop_exception_handler();
    return caught;
}
int main() {
    auto* identity = catch_invalid_root(false);
    check(root_attempts == 0, "preinstalled handler and identity need no root");
    check(catch_invalid_root(true) == identity, "failed root has condition 4 identity");
    check(root_attempts == 1, "root creation refusal consumed");
    check(catch_invalid_root(true) == identity, "repeated getter remains condition 4");
    check(root_attempts == 1, "existing once failure is not retried");
    std::puts("PASS root initialization refusal and repeated fixed transfer");
}
