/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted dynamic parameter helpers.
 *
 * This owns the current make-parameter / parameterize storage path. The
 * parameter object itself is arena allocated, while the dynamic binding stack
 * grows through malloc/realloc and logs hosted warnings on growth failure.
 */

#include "runtime_region_promotion_internal.h"
#include <eshkol/logger.h>

#include <cstdint>
#include <cstdlib>

// HEAP_SUBTYPE_PARAMETER is defined canonically in inc/eshkol/eshkol.h
// (heap_subtype_t). It used to be locally #define'd here as 20, which
// numerically collided with HEAP_SUBTYPE_PRNG (also 20) — since both flow
// through the same ESHKOL_VALUE_HEAP_PTR dispatch (isHeapSubtype checks,
// eshkol_format_value_type_tag, display), a parameter object could be
// misidentified as a PRNG object (e.g. `(prng? a-parameter)` => #t). Fixed
// by giving HEAP_SUBTYPE_PARAMETER its own id (24) in the canonical table.

extern "C" {

typedef struct {
    eshkol_tagged_value_t* stack;
    int top;
    int capacity;
    // The converter is Scheme code and is therefore invoked by generated
    // code, not by this C ABI.  Keeping it alongside the dynamic stack makes
    // the parameter object self-contained: call dispatch and parameterize
    // can retrieve the exact converter associated with this handle.
    eshkol_tagged_value_t converter;
} eshkol_param_t;

static eshkol_tagged_value_t eshkol_parameter_null_value() {
    eshkol_tagged_value_t null_val{};
    null_val.type = ESHKOL_VALUE_NULL;
    return null_val;
}

/**
 * @brief Create an R7RS parameter object seeded with `default_val` (`make-parameter` support).
 *
 * The `eshkol_param_t` control block is arena-allocated (freed only when the
 * arena is reset/destroyed), while its value stack — the dynamic binding
 * stack `parameterize` pushes/pops through — is a separately malloc'd array
 * that can grow independently via realloc. If the initial stack allocation
 * fails, the parameter is still returned but left in an empty state
 * (`top = -1`, `capacity = 0`) so later ref/push calls fail safely instead of
 * dereferencing null.
 *
 * @param arena        Arena to allocate the parameter control block from.
 * @param default_val  Initial (bottom-of-stack) value for the parameter.
 * @return              Opaque parameter handle, or null if the arena allocation fails.
 */
void* eshkol_make_parameter(void* arena, eshkol_tagged_value_t default_val) {
    eshkol_param_t* param = (eshkol_param_t*)arena_allocate_with_header(
        static_cast<arena_t*>(arena), sizeof(eshkol_param_t), HEAP_SUBTYPE_PARAMETER, 0);
    if (!param) {
        return nullptr;
    }

    param->converter = eshkol_parameter_null_value();
    param->stack = nullptr;
    param->top = -1;
    param->capacity = 0;

    const int initial_capacity = 8;
    auto* stack = (eshkol_tagged_value_t*)std::malloc(
        initial_capacity * sizeof(eshkol_tagged_value_t));
    if (!stack) return (void*)param;

    // The value stack is a plain malloc'd buffer that lives independently of
    // any region arena (it is never itself region-allocated and outlives any
    // region that may be active when the value is bound). If `default_val` is
    // a heap/pointer-tagged value currently living inside an active region's
    // arena, storing it here without promotion would leave a dangling pointer
    // once that region pops. Route it through the region write barrier
    // (ESH-0214c) so its reachable subgraph is evacuated out to the global
    // arena first -- the same treatment every other cross-region store
    // (global set!, vector-set!, ...) already gets. `dst` is the malloc
    // storage owner; it is never inside a region arena, so the barrier
    // promotes all the way out, exactly as intended.
    eshkol_tagged_value_t promoted;
    const int32_t status = eshkol_region_write_barrier_checked_v1(
        &promoted, stack, &default_val);
    if (status != 0) {
        // The arena control stays inert; no malloc storage survives a failed
        // unpublished construction, including the emergency longjmp.
        std::free(stack);
        eshkol_runtime_emergency_raise_v1(status);
    }
    stack[0] = promoted;
    param->stack = stack;
    param->capacity = initial_capacity;
    param->top = 0;
    return (void*)param;
}

/**
 * @brief Push a new dynamic binding onto a parameter's value stack (entering a
 * `parameterize` body).
 *
 * Doubles the malloc'd stack capacity via realloc when full. If the realloc
 * fails, the new binding is silently dropped (a warning is logged) and the
 * previous top-of-stack value remains active, rather than corrupting the
 * existing binding or crashing.
 *
 * @param param_ptr  Parameter handle from eshkol_make_parameter (no-op if null).
 * @param val        Value to bind for the dynamic extent being entered.
 */
void eshkol_parameter_push(void* param_ptr, eshkol_tagged_value_t val) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    // Stage before realloc or top publication. A promotion failure leaves the
    // existing stack pointer, capacity, top and bindings unchanged. The stack
    // is malloc-owned, so a null or existing stack pointer denotes root lifetime.
    eshkol_tagged_value_t promoted;
    const int32_t status = eshkol_region_write_barrier_checked_v1(
        &promoted, param->stack, &val);
    if (status != 0) eshkol_runtime_emergency_raise_v1(status);

    if (param->top + 1 >= param->capacity) {
        int new_capacity = param->capacity * 2;
        if (new_capacity < 8) new_capacity = 8;
        eshkol_tagged_value_t* new_stack = (eshkol_tagged_value_t*)std::realloc(
            param->stack, new_capacity * sizeof(eshkol_tagged_value_t));
        if (!new_stack) {
            eshkol_warn("parameter-push: realloc(%d -> %d entries) failed; "
                        "new binding dropped, previous value remains",
                        param->capacity, new_capacity);
            return;
        }
        param->stack = new_stack;
        param->capacity = new_capacity;
    }

    param->stack[param->top + 1] = promoted;
    param->top++;
}

/**
 * @brief Pop the most recent dynamic binding off a parameter's value stack
 * (leaving a `parameterize` body), restoring the previous value.
 *
 * The bottom-most (index 0, the constructor default) binding is never popped.
 * @param param_ptr  Parameter handle from eshkol_make_parameter (no-op if null).
 */
void eshkol_parameter_pop(void* param_ptr) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top > 0) {
        param->top--;
    }
}

/**
 * @brief Replace the current binding of a parameter object.
 *
 * This is the one-argument procedure-call path, distinct from
 * `parameterize`'s push/pop dynamic extent.  It updates the top-most slot so
 * a write inside a parameterize body remains local to that binding.  As with
 * construction and push, route the value through the region barrier before it
 * reaches the long-lived malloc stack.
 */
void eshkol_parameter_set(void* param_ptr, eshkol_tagged_value_t val) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;
    if (param->top < 0 || !param->stack) return;
    eshkol_tagged_value_t promoted;
    const int32_t status = eshkol_region_write_barrier_checked_v1(
        &promoted, param->stack, &val);
    if (status != 0) eshkol_runtime_emergency_raise_v1(status);
    param->stack[param->top] = promoted;
}

/** Store the optional Scheme converter associated with a parameter object. */
void eshkol_parameter_set_converter(void* param_ptr,
                                    eshkol_tagged_value_t converter) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;
    eshkol_tagged_value_t promoted;
    // Parameter controls are leaf-copied during escape. Like their malloc
    // value stacks, converters must therefore outlive every region even when
    // the control itself currently belongs to a region.
    const int32_t status = eshkol_region_write_barrier_checked_v1(
        &promoted, nullptr, &converter);
    if (status != 0) eshkol_runtime_emergency_raise_v1(status);
    param->converter = promoted;
}

/** Return the optional Scheme converter, or #<null> when none was supplied. */
eshkol_tagged_value_t eshkol_parameter_converter_ref(void* param_ptr) {
    if (!param_ptr) return eshkol_parameter_null_value();
    return ((eshkol_param_t*)param_ptr)->converter;
}

/**
 * @brief Read the current (top-of-stack) value bound to a parameter.
 *
 * Returns a zeroed ESHKOL_VALUE_NULL tagged value if the handle is null or the
 * parameter's stack is empty/unallocated (e.g. the initial malloc failed),
 * rather than dereferencing invalid memory.
 *
 * @param param_ptr  Parameter handle from eshkol_make_parameter.
 * @return           The currently bound value.
 */
eshkol_tagged_value_t eshkol_parameter_ref(void* param_ptr) {
    if (!param_ptr) {
        return eshkol_parameter_null_value();
    }
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top < 0 || !param->stack) {
        return eshkol_parameter_null_value();
    }

    return param->stack[param->top];
}

/** @brief Pointer-argument wrapper around eshkol_make_parameter for ABI sites that pass tagged values by pointer; returns null if `default_val` is null. */
void* eshkol_make_parameter_ptr(void* arena, const eshkol_tagged_value_t* default_val) {
    if (!default_val) return nullptr;
    return eshkol_make_parameter(arena, *default_val);
}

/** @brief Pointer-argument wrapper around eshkol_parameter_push; no-op if `val` is null. */
void eshkol_parameter_push_ptr(void* param, const eshkol_tagged_value_t* val) {
    if (!val) return;
    eshkol_parameter_push(param, *val);
}

/** Pointer-argument wrapper around eshkol_parameter_set. */
void eshkol_parameter_set_ptr(void* param, const eshkol_tagged_value_t* val) {
    if (!val) return;
    eshkol_parameter_set(param, *val);
}

/** Pointer-argument wrapper around eshkol_parameter_set_converter. */
void eshkol_parameter_set_converter_ptr(void* param,
                                        const eshkol_tagged_value_t* converter) {
    if (!converter) return;
    eshkol_parameter_set_converter(param, *converter);
}

/** @brief Pointer-argument wrapper around eshkol_parameter_ref; writes the current value through `result` (no-op if `result` is null). */
void eshkol_parameter_ref_ptr(void* param, eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_ref(param);
}

/** Pointer-result wrapper around eshkol_parameter_converter_ref. */
void eshkol_parameter_converter_ref_ptr(void* param,
                                        eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_converter_ref(param);
}

}  // extern "C"

// Runtime-private layout query; keeps evacuation validation with the owner.
size_t eshkol_parameter_promotion_size() noexcept {
    return sizeof(eshkol_param_t);
}
