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

// Global dynamic-wind handler stack
eshkol_dynamic_wind_entry_t* g_dynamic_wind_stack = nullptr;

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
    return state;
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
