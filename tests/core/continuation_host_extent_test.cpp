// Host evaluation extents for native continuations
// (lib/core/runtime_continuations.cpp).
//
// An interactive host brackets each top-level evaluation with
// eshkol_continuation_extent_enter/leave. A continuation whose stack image was
// captured inside a bracketed evaluation is admitted by
// eshkol_continuation_check_extent only while that evaluation is live; once it
// has returned, invoking the continuation raises a catchable condition. A
// capture outside every bracketed evaluation (a batch host or an AOT
// executable) is always admitted.
//
// The record of live evaluations has no fixed capacity, so the contract must
// hold at every nesting depth. This test nests real recursive host frames,
// each one bracketing an evaluation, and captures at the innermost level for
// depths on both sides of every power of two the storage grows through.

#include "../../lib/core/arena_memory.h"

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;
uintptr_t g_stack_base = 0;

void check(bool ok, const char* what, int depth) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (depth %d)\n", what, depth);
        ++g_failures;
    }
}

uintptr_t test_stack_base(void) { return g_stack_base; }

jmp_buf g_landing;

// Run the admission check under a handler. Returns true when it raised.
bool check_extent_raises(eshkol_continuation_state_t* state) {
    eshkol_push_exception_handler(&g_landing);
    if (setjmp(g_landing) == 0) {
        eshkol_continuation_check_extent(state);
        eshkol_pop_exception_handler();
        return false;
    }
    eshkol_pop_exception_handler();
    return true;
}

bool raised_finished_evaluation() {
    return g_current_exception && g_current_exception->message &&
           std::strstr(g_current_exception->message, "continuation cannot be resumed") != nullptr;
}

struct Capture {
    arena_t* arena;
    jmp_buf capture_point;      // never jumped to; the state only records it
    eshkol_continuation_state_t* state = nullptr;
    bool admitted_inside = false;
};

eshkol_continuation_state_t* capture_here(Capture& c) {
    eshkol_continuation_state_t* state =
        eshkol_make_continuation_state_flags(c.arena, &c.capture_point, 0);
    if (state) eshkol_continuation_capture_stack(c.arena, state);
    return state;
}

// One host frame per level, each bracketing its own evaluation, as the REPL
// brackets each top-level form. The innermost level captures.
__attribute__((noinline))
void nest(int remaining, int depth, Capture& c) {
    volatile char frame_marker[64];
    frame_marker[0] = (char)remaining;
    const uint64_t extent =
        eshkol_continuation_extent_enter(__builtin_frame_address(0));
    if (remaining > 1) {
        nest(remaining - 1, depth, c);
    } else {
        c.state = capture_here(c);
        check(c.state && c.state->saved_stack, "capture took a stack image", depth);
        if (c.state) c.admitted_inside = !check_extent_raises(c.state);
    }
    eshkol_continuation_extent_leave(extent);
    __asm__ __volatile__("" :: "r"(&frame_marker[0]) : "memory");
}

void run_depth(int depth) {
    Capture c;
    c.arena = arena_create(1 << 16);
    nest(depth, depth, c);
    check(c.admitted_inside, "admitted while its evaluation is live", depth);
    if (c.state) {
        const bool raised = check_extent_raises(c.state);
        check(raised, "refused after its evaluation returned", depth);
        check(raised && raised_finished_evaluation(),
              "refusal names the finished evaluation", depth);
    }
    arena_destroy(c.arena);
}

// A capture outside every bracketed evaluation is the batch-host case.
void run_unbracketed() {
    Capture c;
    c.arena = arena_create(1 << 16);
    c.state = capture_here(c);
    check(c.state && c.state->saved_stack, "unbracketed capture took a stack image", 0);
    if (c.state) check(!check_extent_raises(c.state), "unbracketed capture is admitted", 0);
    arena_destroy(c.arena);
}

// A capture in an outer evaluation stays admissible while a nested one runs
// and after the nested one returns.
__attribute__((noinline))
void run_outer_inner() {
    Capture c;
    c.arena = arena_create(1 << 16);
    const uint64_t outer = eshkol_continuation_extent_enter(__builtin_frame_address(0));
    c.state = capture_here(c);
    Capture inner;
    inner.arena = c.arena;
    nest(3, 3, inner);
    if (c.state) check(!check_extent_raises(c.state), "outer capture admitted after inner returns", 1);
    eshkol_continuation_extent_leave(outer);
    if (c.state) check(check_extent_raises(c.state), "outer capture refused after outer returns", 1);
    arena_destroy(c.arena);
}

}  // namespace

int main() {
    g_stack_base = (uintptr_t)__builtin_frame_address(0);
    eshkol_set_stack_base_hook(&test_stack_base);

    run_unbracketed();
    run_outer_inner();
    const int depths[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000};
    for (int depth : depths) run_depth(depth);
    run_unbracketed();

    if (g_failures) {
        std::fprintf(stderr, "continuation host extents: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: continuation host extents hold at every nesting depth\n");
    return 0;
}
