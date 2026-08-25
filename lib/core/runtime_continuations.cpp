/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * First-class continuation and dynamic-wind runtime helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/eshkol.h"

#include <cstdint>
#include <cstring>
#include <csetjmp>
#include <cstdlib>

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ESHKOL_CONT_ASAN 1
#  endif
#endif
#if !defined(ESHKOL_CONT_ASAN) && defined(__SANITIZE_ADDRESS__)
#  define ESHKOL_CONT_ASAN 1
#endif
#if defined(ESHKOL_CONT_ASAN)
extern "C" void __asan_unpoison_memory_region(void const volatile* addr, size_t size);
extern "C" void __asan_handle_no_return(void);
#endif

// Global dynamic-wind handler stack
eshkol_dynamic_wind_entry_t* g_dynamic_wind_stack = nullptr;

/* ── Stack-copying re-entrant continuations ──────────────────────────────────
 *
 * `call/cc` records a setjmp point, but a jmp_buf only names a stack address:
 * once the capturing frame returns and its memory is reused, longjmp'ing back
 * to it resumes on top of whatever now occupies those bytes. That is what made
 * every generator / `amb` / coroutine shape crash on native (SW-51).
 *
 * The fix is to give the continuation a durable copy of the frames it needs.
 * At capture we memcpy the live stack — from just below the capture point up
 * to the thread's stack base — into the arena. At resume we copy those bytes
 * back to the SAME addresses and then longjmp. Restoring in place is what
 * makes this safe without any pointer relocation: frame pointers, saved
 * registers spilled to the stack, addresses of locals that closures captured,
 * and the jmp_buf itself all point where they always did.
 *
 * The one ordering constraint is that the restoring helper must not be running
 * inside the region it is about to overwrite. resume_trampoline() recurses to
 * push its own frame below stack_lo before copying.
 *
 * Multi-shot falls out for free: saved_stack is written once and never
 * mutated, so each invocation restores the same pristine image.
 */

/* Slack kept between the restoring frame and the region being restored. */
#define ESHKOL_RESUME_MARGIN 4096u
/* Per-recursion stack consumed while pushing the trampoline frame down. */
#define ESHKOL_RESUME_PAD 2048u

/* Stack geometry is a platform question, and this file is freestanding core
 * (see tests/toolchain/runtime_core_boundary_test.cpp — no pthread/OS calls
 * here). The hosted runtime installs a probe at startup via
 * eshkol_set_stack_base_hook(); a freestanding target that never installs one
 * simply keeps escape-only continuations, which is the right answer for a
 * target with no thread-stack notion to interrogate. */
static eshkol_stack_base_fn g_stack_base_hook = nullptr;

extern "C" void eshkol_set_stack_base_hook(eshkol_stack_base_fn fn) {
    g_stack_base_hook = fn;
}

/** @brief Highest address of the current thread's stack, or 0 if unknown. */
static uintptr_t eshkol_stack_base(void) {
    return g_stack_base_hook ? (uintptr_t)g_stack_base_hook() : (uintptr_t)0;
}

/**
 * @brief Snapshot the live C stack above the capturing `call/cc` frame.
 *
 * Called from the call/cc normal path *after* setjmp has written the jmp_buf,
 * so the captured image contains a jmp_buf that is valid to jump through.
 * Failure is non-fatal and simply leaves saved_stack null, in which case
 * resume falls back to the historical escape-only longjmp.
 */
extern "C" void eshkol_continuation_capture_stack(void* arena_void, void* state_void) {
    auto* state = (eshkol_continuation_state_t*)state_void;
    if (!state) return;
    state->stack_lo = nullptr;
    state->stack_hi = nullptr;
    state->saved_stack = nullptr;
    state->saved_len = 0;

    uintptr_t base = eshkol_stack_base();
    if (!base) return;                     /* unknown stack geometry: escape-only */

    /* Our own frame is the deepest thing that must survive the copy. */
    volatile char here = 0;
    uintptr_t lo = (uintptr_t)&here;
    if (lo >= base) return;                /* not the stack we think it is */

    size_t len = (size_t)(base - lo);
    void* copy = arena_allocate_aligned((arena_t*)arena_void, len, 16);
    if (!copy) return;                     /* escape-only rather than half-captured */

    memcpy(copy, (const void*)lo, len);
    state->stack_lo = (void*)lo;
    state->stack_hi = (void*)base;
    state->saved_stack = copy;
    state->saved_len = (uint64_t)len;
}

/**
 * @brief Push this frame below the region about to be restored, then restore.
 *
 * Recurses (never tail-calls: the volatile pad and the asm barrier keep the
 * frame real) until its own locals sit below stack_lo, so the memcpy cannot
 * overwrite the frame performing it.
 */
static void resume_trampoline(eshkol_continuation_state_t* state) {
    volatile char pad[ESHKOL_RESUME_PAD];
    pad[0] = 0;

    if ((uintptr_t)&pad[0] >= (uintptr_t)state->stack_lo - ESHKOL_RESUME_MARGIN) {
        resume_trampoline(state);
        __asm__ __volatile__("" :: "r"(&pad[0]) : "memory");
        return;                            /* unreachable: the callee longjmps */
    }

#if defined(ESHKOL_CONT_ASAN)
    /* The restored bytes are ordinary stack the sanitizer has poisoned as dead
     * frames; writing them is intentional, and the longjmp leaves the shadow
     * for the abandoned chain behind. */
    __asan_unpoison_memory_region(state->stack_lo, (size_t)state->saved_len);
    __asan_handle_no_return();
#endif

    memcpy(state->stack_lo, state->saved_stack, (size_t)state->saved_len);
    longjmp(*(jmp_buf*)state->jmp_buf_ptr, 1);
}

/**
 * @brief Resume a captured continuation. Does not return.
 *
 * Restores the continuation's stack image when it has one, then longjmps to
 * the capture point. Continuations captured where the stack geometry was
 * unknown keep the historical escape-only behaviour.
 */
extern "C" void eshkol_continuation_resume(void* state_void) {
    auto* state = (eshkol_continuation_state_t*)state_void;
    if (!state || !state->jmp_buf_ptr) {
        eshkol_error("Invoked a continuation with no capture point");
        abort();
    }
    if (state->saved_stack && state->saved_len) {
        resume_trampoline(state);
    }
    longjmp(*(jmp_buf*)state->jmp_buf_ptr, 1);
}

/**
 * @brief Allocate and initialize the state captured by a `call/cc` invocation.
 *
 * Allocates an eshkol_continuation_state_t in the given arena, records the
 * caller-supplied setjmp buffer pointer to jump back to when the
 * continuation is invoked, zero-initializes the carried value to null, and
 * snapshots the current top of the global dynamic-wind stack
 * (g_dynamic_wind_stack) as `wind_mark` so eshkol_unwind_dynamic_wind can
 * later run the correct `after` thunks if the continuation escapes its
 * dynamic extent. The returned state is arena-owned.
 *
 * @param arena_void   Arena to allocate from, passed as void* across the ABI.
 * @param jmp_buf_ptr  Pointer to the jmp_buf to longjmp back into on invocation.
 * @return             Newly allocated continuation state, or nullptr on failure.
 */
extern "C" eshkol_continuation_state_t* eshkol_make_continuation_state(void* arena_void, void* jmp_buf_ptr) {
    arena_t* arena = (arena_t*)arena_void;
    eshkol_continuation_state_t* state = (eshkol_continuation_state_t*)arena_allocate_aligned(arena, sizeof(eshkol_continuation_state_t), 8);
    if (!state) {
        eshkol_error("Failed to allocate continuation state");
        return nullptr;
    }
    state->jmp_buf_ptr = jmp_buf_ptr;
    memset(&state->value, 0, sizeof(eshkol_tagged_value_t));
    state->value.type = ESHKOL_VALUE_NULL;
    state->wind_mark = (void*)g_dynamic_wind_stack;
    state->promise_mark = eshkol_promise_eval_mark();
    state->region_mark = eshkol_region_mark();  // #341
    // Filled in by eshkol_continuation_capture_stack() once setjmp has run.
    state->stack_lo = nullptr;
    state->stack_hi = nullptr;
    state->saved_stack = nullptr;
    state->saved_len = 0;
    return state;
}

/**
 * @brief Close every region entered since this continuation was captured,
 *        promoting the delivered value out of them first (#341).
 *
 * The third member of the non-local-exit unwind trio, alongside
 * eshkol_unwind_dynamic_wind and eshkol_promise_eval_unwind_to, and called from
 * the same place in the continuation-invoke path. `state->value` (already
 * written by the invoke site) is passed as the in-flight value so a value
 * allocated inside a region being torn down is deep-promoted to an arena that
 * outlives the jump, instead of being delivered as a pointer into a freed arena.
 *
 * @param state_void The continuation state being invoked (no-op if NULL).
 */
extern "C" void eshkol_region_unwind_for_continuation(void* state_void) {
    auto* state = (eshkol_continuation_state_t*)state_void;
    if (!state) return;
    eshkol_region_unwind_to(state->region_mark, &state->value, 1);
}

/**
 * @brief Wrap a continuation state in a callable closure object.
 *
 * Builds a 1-arity, 1-capture closure (via arena_allocate_closure_with_header)
 * named "<continuation>" whose single capture slot stores `state_ptr` (the
 * eshkol_continuation_state_t from eshkol_make_continuation_state) as a
 * HEAP_PTR-tagged value, then overwrites the allocated object's header
 * subtype from the default closure subtype to CALLABLE_SUBTYPE_CONTINUATION
 * so generated call sites and introspection code can distinguish a
 * continuation from an ordinary closure. The closure's `func_ptr` is left 0
 * — invoking a continuation is handled specially by the codegen'd call path,
 * not through this func_ptr. Returned value is arena-owned.
 *
 * @param arena_void  Arena to allocate from, passed as void* across the ABI.
 * @param state_ptr   Continuation state to capture (see eshkol_make_continuation_state).
 * @return            The continuation closure as an opaque void*, or nullptr on failure.
 */
extern "C" void* eshkol_make_continuation_closure(void* arena_void, void* state_ptr) {
    arena_t* arena = (arena_t*)arena_void;

    // Allocate closure with 1 capture (the state pointer)
    // packed_info: 1 capture in bits 0-15, 1 fixed param in bits 16-31
    size_t packed_info = 1 | (1ULL << 16);  // 1 capture, 1 param (arity=1)
    eshkol_closure_t* closure = arena_allocate_closure_with_header(
        arena, 0, packed_info, 0, 0, "<continuation>");

    if (!closure) {
        eshkol_error("Failed to allocate continuation closure");
        return nullptr;
    }

    // Override the header subtype to CALLABLE_SUBTYPE_CONTINUATION
    uint8_t* closure_bytes = (uint8_t*)closure;
    eshkol_object_header_t* header = (eshkol_object_header_t*)(closure_bytes - sizeof(eshkol_object_header_t));
    header->subtype = CALLABLE_SUBTYPE_CONTINUATION;

    // Store state pointer as a tagged value in captures[0]
    if (closure->env) {
        closure->env->captures[0].type = ESHKOL_VALUE_HEAP_PTR;
        closure->env->captures[0].flags = 0;
        closure->env->captures[0].reserved = 0;
        closure->env->captures[0].data.int_val = (uint64_t)(uintptr_t)state_ptr;
    }

    return (void*)closure;
}

// Call a 0-arg Eshkol closure from C runtime (for dynamic-wind thunks)
// Handles closures with 0-4 captures by matching LLVM calling convention
static eshkol_tagged_value_t call_thunk_closure(eshkol_closure_t* closure) {
    if (!closure || !closure->func_ptr) {
        eshkol_tagged_value_t null_val;
        memset(&null_val, 0, sizeof(null_val));
        null_val.type = ESHKOL_VALUE_NULL;
        return null_val;
    }

    size_t num_captures = 0;
    if (closure->env) {
        num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(closure->env->num_captures);
    }

    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));
    result.type = ESHKOL_VALUE_NULL;

#if defined(__aarch64__) || defined(_M_ARM64)
    // AArch64 returns this 16-byte aggregate in registers, so the thunk bridge
    // must match the direct return ABI instead of passing a hidden result slot.
    typedef eshkol_tagged_value_t (*fn0_t)(void);
    typedef eshkol_tagged_value_t (*fn1_t)(void*);
    typedef eshkol_tagged_value_t (*fn2_t)(void*, void*);
    typedef eshkol_tagged_value_t (*fn3_t)(void*, void*, void*);
    typedef eshkol_tagged_value_t (*fn4_t)(void*, void*, void*, void*);

    switch (num_captures) {
        case 0:
            result = ((fn0_t)(uintptr_t)closure->func_ptr)();
            break;
        case 1:
            result = ((fn1_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0]);
            break;
        case 2:
            result = ((fn2_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1]);
            break;
        case 3:
            result = ((fn3_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2]);
            break;
        case 4:
            result = ((fn4_t)(uintptr_t)closure->func_ptr)(&closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2], &closure->env->captures[3]);
            break;
        default:
            result = ((fn0_t)(uintptr_t)closure->func_ptr)();
            break;
    }
#else
    // The currently-supported x86/Windows thunk ABI uses a hidden return buffer,
    // so the runtime bridge must pass the result slot first.
    typedef void (*fn0_t)(eshkol_tagged_value_t*);
    typedef void (*fn1_t)(eshkol_tagged_value_t*, void*);
    typedef void (*fn2_t)(eshkol_tagged_value_t*, void*, void*);
    typedef void (*fn3_t)(eshkol_tagged_value_t*, void*, void*, void*);
    typedef void (*fn4_t)(eshkol_tagged_value_t*, void*, void*, void*, void*);

    switch (num_captures) {
        case 0:
            ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
            break;
        case 1:
            ((fn1_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0]);
            break;
        case 2:
            ((fn2_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1]);
            break;
        case 3:
            ((fn3_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2]);
            break;
        case 4:
            ((fn4_t)(uintptr_t)closure->func_ptr)(&result, &closure->env->captures[0], &closure->env->captures[1], &closure->env->captures[2], &closure->env->captures[3]);
            break;
        default:
            ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
            break;
    }
#endif

    return result;
}

// Call a thunk stored as a tagged value (CALLABLE type)
static void call_thunk_from_tagged(const eshkol_tagged_value_t* thunk) {
    if (!thunk || thunk->type != ESHKOL_VALUE_CALLABLE) return;
    eshkol_closure_t* closure = (eshkol_closure_t*)(uintptr_t)thunk->data.int_val;
    call_thunk_closure(closure);
}

// Push a dynamic-wind entry onto the global stack
extern "C" void eshkol_push_dynamic_wind(void* arena_void,
    const eshkol_tagged_value_t* before, const eshkol_tagged_value_t* after) {
    arena_t* arena = (arena_t*)arena_void;
    eshkol_dynamic_wind_entry_t* entry = (eshkol_dynamic_wind_entry_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dynamic_wind_entry_t), 8);
    if (!entry) return;
    entry->before = *before;
    entry->after = *after;
    entry->prev = g_dynamic_wind_stack;
    g_dynamic_wind_stack = entry;
}

// Pop the top dynamic-wind entry
extern "C" void eshkol_pop_dynamic_wind(void) {
    if (g_dynamic_wind_stack) {
        g_dynamic_wind_stack = g_dynamic_wind_stack->prev;
    }
}

// Unwind dynamic-wind stack down to a saved mark, calling after thunks
extern "C" void eshkol_unwind_dynamic_wind(void* saved_wind_mark) {
    eshkol_dynamic_wind_entry_t* mark = (eshkol_dynamic_wind_entry_t*)saved_wind_mark;
    while (g_dynamic_wind_stack != nullptr && g_dynamic_wind_stack != mark) {
        eshkol_dynamic_wind_entry_t* entry = g_dynamic_wind_stack;
        g_dynamic_wind_stack = entry->prev;
        call_thunk_from_tagged(&entry->after);
    }
}

/** @brief Number of entries from @p e down to the root. */
static int eshkol_wind_depth(const eshkol_dynamic_wind_entry_t* e) {
    int d = 0;
    while (e) { d++; e = e->prev; }
    return d;
}

/**
 * @brief Run `before` thunks from @p common outward to @p target, in order.
 *
 * Recurses first so the OUTERMOST extent being entered runs its `before`
 * first, which is the order R7RS requires.
 */
static void eshkol_wind_enter(eshkol_dynamic_wind_entry_t* target,
                              eshkol_dynamic_wind_entry_t* common) {
    if (!target || target == common) return;
    eshkol_wind_enter(target->prev, common);
    call_thunk_from_tagged(&target->before);
    g_dynamic_wind_stack = target;
}

/**
 * @brief Move the dynamic-wind stack to @p target_void, running the thunks
 *        the transfer crosses (R7RS 6.10 "rerooting").
 *
 * Unwinding alone is only correct when the target is an ancestor of the
 * current extent — the escape case. Re-entering a continuation captured
 * *inside* a `dynamic-wind` whose extent has since been left has to run that
 * extent's `before` thunk again on the way back in, or the body resumes with
 * its setup undone: exactly the silent state corruption naive implementations
 * exhibit. Splitting at the common ancestor handles both directions and the
 * general case where the jump leaves some extents and enters others.
 *
 * @param target_void The wind-stack mark saved at capture time.
 */
extern "C" void eshkol_reroot_dynamic_wind(void* target_void) {
    auto* target = (eshkol_dynamic_wind_entry_t*)target_void;
    if (g_dynamic_wind_stack == target) return;

    /* Find the deepest entry common to both chains. */
    const eshkol_dynamic_wind_entry_t* c = g_dynamic_wind_stack;
    const eshkol_dynamic_wind_entry_t* t = target;
    int dc = eshkol_wind_depth(c);
    int dt = eshkol_wind_depth(t);
    while (dc > dt) { c = c->prev; dc--; }
    while (dt > dc) { t = t->prev; dt--; }
    while (c != t) { c = c->prev; t = t->prev; }
    auto* common = (eshkol_dynamic_wind_entry_t*)(uintptr_t)c;

    /* Leave: innermost `after` first. */
    while (g_dynamic_wind_stack != common) {
        eshkol_dynamic_wind_entry_t* entry = g_dynamic_wind_stack;
        g_dynamic_wind_stack = entry->prev;
        call_thunk_from_tagged(&entry->after);
    }
    /* Enter: outermost `before` first. */
    eshkol_wind_enter(target, common);
    g_dynamic_wind_stack = target;
}
