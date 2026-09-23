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

#include <eshkol/eshkol.h>
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

extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

// Region write barrier (ESH-0214c, lib/core/runtime_regions.cpp): promotes a
// tagged value's in-region subgraph out of any active region strictly inner
// than the region owning `dst` before it is stored there. The fast path (no
// active region) is a single thread-local load + branch, so this is safe to
// call unconditionally on every store -- the same convention codegen uses at
// every other mutation channel (set-car!/set-cdr!, vector-set!,
// hash-table-set!, global set!). Immediate (non-heap) tagged values pass
// through untouched; only HEAP_PTR/CALLABLE/port-tagged values are evacuated.
extern void eshkol_region_write_barrier_into(eshkol_tagged_value_t* out,
                                             const void* dst,
                                             const eshkol_tagged_value_t* value);

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
    // Promote the default FIRST (SW-232): the barrier raises when the
    // promotion cannot be completed, and it must do so before anything --
    // the control block, the malloc'd stack, a published top -- exists to be
    // left half-built. The value stack is a plain malloc'd buffer that lives
    // independently of any region arena, so the barrier promotes all the way
    // out (NULL destination = outside every region), the same treatment every
    // other cross-region store (global set!, vector-set!, ...) gets.
    eshkol_tagged_value_t promoted;
    eshkol_region_write_barrier_into(&promoted, nullptr, &default_val);

    eshkol_param_t* param = (eshkol_param_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_param_t), HEAP_SUBTYPE_PARAMETER, 0);
    if (!param) {
        return nullptr;
    }

    param->converter = eshkol_parameter_null_value();

    const int initial_capacity = 8;
    param->stack = (eshkol_tagged_value_t*)std::malloc(
        initial_capacity * sizeof(eshkol_tagged_value_t));
    if (!param->stack) {
        param->top = -1;
        param->capacity = 0;
        return (void*)param;
    }

    param->capacity = initial_capacity;
    param->stack[0] = promoted;
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

    // Same region write barrier treatment as the constructor default (see
    // eshkol_make_parameter above): `val` may point into a region that is
    // still active on entry to this `parameterize` binding but pops before
    // the binding is popped/read again, so it must be promoted out of any
    // active region before landing in the malloc'd (never region-owned)
    // value stack. Promote, store, THEN publish the new top (SW-232): if the
    // promotion raises, top still names the previous binding, not a slot
    // that was never written.
    eshkol_tagged_value_t promoted;
    eshkol_region_write_barrier_into(&promoted, &param->stack[param->top + 1], &val);
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
    eshkol_region_write_barrier_into(&param->stack[param->top],
                                     &param->stack[param->top], &val);
}

/** Store the optional Scheme converter associated with a parameter object. */
void eshkol_parameter_set_converter(void* param_ptr,
                                    eshkol_tagged_value_t converter) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;
    eshkol_region_write_barrier_into(&param->converter, &param->converter,
                                     &converter);
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

/**
 * Visit every tagged value owned by a parameter object. The native region
 * evacuator uses this to preserve the parameter's current value, converter,
 * and dynamic-binding stack when the parameter object itself escapes a
 * region. The visitor mutates each slot in place, so the helper does not
 * expose the private control-block layout to another translation unit.
 */
typedef void (*eshkol_parameter_value_visitor)(eshkol_tagged_value_t* value,
                                               void* context);

void eshkol_parameter_visit_values(void* param_ptr,
                                   eshkol_parameter_value_visitor visitor,
                                   void* context) {
    if (!param_ptr || !visitor) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;
    visitor(&param->converter, context);
    if (!param->stack || param->top < 0) return;
    for (int i = 0; i <= param->top; ++i)
        visitor(&param->stack[i], context);
}

/** Pointer-result wrapper around eshkol_parameter_converter_ref. */
void eshkol_parameter_converter_ref_ptr(void* param,
                                        eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_converter_ref(param);
}

}  // extern "C"
