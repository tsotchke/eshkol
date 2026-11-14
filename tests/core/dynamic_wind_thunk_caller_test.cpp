// The thunk caller of a dynamic-wind entry (lib/core/runtime_continuations.cpp).
//
// A dynamic-wind entry stores its before/after procedures together with the
// caller that enters them. When a transfer of control leaves the extent the
// runtime hands the after procedure to that caller; it never calls a
// procedure's code address itself. An entry that holds a procedure therefore
// always holds a caller: eshkol_push_dynamic_wind refuses, by aborting the
// process with a diagnostic, to create one that does not, so no extent can be
// left with an exit thunk that would not run.

#include "../../lib/core/arena_memory.h"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_failures = 0;
int g_calls = 0;
const eshkol_tagged_value_t* g_last_thunk = nullptr;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void recording_caller(const eshkol_tagged_value_t* thunk, eshkol_tagged_value_t* result) {
    ++g_calls;
    g_last_thunk = thunk;
    std::memset(result, 0, sizeof(*result));
}

eshkol_tagged_value_t callable(uintptr_t marker) {
    eshkol_tagged_value_t v;
    std::memset(&v, 0, sizeof(v));
    v.type = ESHKOL_VALUE_CALLABLE;
    v.data.int_val = (uint64_t)marker;
    return v;
}

void cleanup_counter(void* arg) { ++*static_cast<int*>(arg); }

}  // namespace

int main() {
    arena_t* arena = arena_create(1 << 16);
    const eshkol_tagged_value_t before = callable(0x1000);
    const eshkol_tagged_value_t after = callable(0x2000);

    // Leaving an extent enters the after procedure once, through the caller.
    eshkol_push_dynamic_wind(arena, &before, &after, &recording_caller);
    eshkol_unwind_dynamic_wind(nullptr);
    check(g_calls == 1, "unwinding enters the after procedure once");
    check(g_last_thunk && g_last_thunk->data.int_val == 0x2000,
          "the caller receives the after procedure");

    // A normal pop runs nothing; generated code calls the after thunk itself.
    g_calls = 0;
    eshkol_push_dynamic_wind(arena, &before, &after, &recording_caller);
    eshkol_pop_dynamic_wind();
    eshkol_unwind_dynamic_wind(nullptr);
    check(g_calls == 0, "a popped entry is not entered");

    // Nested extents are left innermost first, each through its own caller.
    g_calls = 0;
    const eshkol_tagged_value_t inner_after = callable(0x3000);
    eshkol_push_dynamic_wind(arena, &before, &after, &recording_caller);
    eshkol_push_dynamic_wind(arena, &before, &inner_after, &recording_caller);
    eshkol_unwind_dynamic_wind(nullptr);
    check(g_calls == 2, "both nested extents are left");
    check(g_last_thunk && g_last_thunk->data.int_val == 0x2000,
          "the outer extent is left last");

    // A native-cleanup entry needs no caller.
    int cleanups = 0;
    g_calls = 0;
    eshkol_push_dynamic_wind_cleanup(arena, &cleanup_counter, &cleanups);
    eshkol_unwind_dynamic_wind(nullptr);
    check(cleanups == 1 && g_calls == 0, "a native cleanup entry runs its cleanup");

    // An entry without procedures needs no caller either.
    eshkol_tagged_value_t none;
    std::memset(&none, 0, sizeof(none));
    eshkol_push_dynamic_wind(arena, &none, &none, nullptr);
    eshkol_unwind_dynamic_wind(nullptr);
    check(g_calls == 0, "an entry without procedures is not entered");

    // A procedure without a caller is refused when the entry is created.
    std::fflush(nullptr);
    const pid_t child = fork();
    if (child == 0) {
        std::freopen("/dev/null", "w", stderr);
        eshkol_push_dynamic_wind(arena, &before, &after, nullptr);
        _exit(0);  // not reached
    }
    int status = 0;
    check(child > 0 && waitpid(child, &status, 0) == child,
          "the refusing process can be observed");
    check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
          "a procedure without a thunk caller aborts the process (SIGABRT) at push time");

    arena_destroy(arena);
    if (g_failures == 0) {
        std::printf("PASS: dynamic-wind entries enter their thunks through the thunk caller\n");
        return 0;
    }
    return 1;
}
