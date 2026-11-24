// Test-only sensitivity control. The old child is live for the entire denied execution.
#include "lib/core/arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static eshkol_region_t* child;
static eshkol_region_t* parent;
static unsigned entries;

extern "C" void __real_nested_boundary(int64_t);
extern "C" void __wrap_nested_boundary(int64_t event) {
    __real_nested_boundary(event);
    if (event != 2) return;
    if (__region_stack_depth != 2 || __region_stack[MAX_REGION_DEPTH - 1])
        std::_Exit(91);
    child = __region_stack[1];
    parent = __region_stack[0];
    entries = 0;
    if (std::getenv("NESTED_GATE_CONTROL"))
        __region_stack[MAX_REGION_DEPTH - 1] = child;
}

// Only the runtime translation unit is built with function-entry instrumentation.
extern "C" void __cyg_profile_func_enter(void* fn, void*) {
    if (fn == reinterpret_cast<void*>(region_destroy) && child) {
        ++entries;
        std::fputs("nested destructor entered\n", stderr);
        child = nullptr;
    }
}
extern "C" void __cyg_profile_func_exit(void*, void*) {}

extern "C" void __real_abort();
extern "C" void __wrap_abort() {
    if (child && entries == 0 && __region_stack_depth == 1 &&
        __region_stack[0] == parent && __region_stack[MAX_REGION_DEPTH-1] == child &&
        __region_stack[1] == nullptr && eshkol_current_arena() == parent->arena &&
        eshkol_memctx_current()->allocation_domain == parent->arena && !child->is_active) {
        std::fputs("restored parent; inactive slot retains live child; destructor entries 0\n",stderr);
    }
    __real_abort();
}
