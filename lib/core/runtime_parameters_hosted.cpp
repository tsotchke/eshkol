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
} eshkol_param_t;

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
        arena, sizeof(eshkol_param_t), HEAP_SUBTYPE_PARAMETER, 0);
    if (!param) {
        return nullptr;
    }

    const int initial_capacity = 8;
    param->stack = (eshkol_tagged_value_t*)std::malloc(
        initial_capacity * sizeof(eshkol_tagged_value_t));
    if (!param->stack) {
        param->top = -1;
        param->capacity = 0;
        return (void*)param;
    }

    param->capacity = initial_capacity;
    param->top = 0;
    // The value stack is a plain malloc'd buffer that lives independently of
    // any region arena (it is never itself region-allocated and outlives any
    // region that may be active when the value is bound). If `default_val` is
    // a heap/pointer-tagged value currently living inside an active region's
    // arena, storing it here without promotion would leave a dangling pointer
    // once that region pops. Route it through the region write barrier
    // (ESH-0214c) so its reachable subgraph is evacuated out to the global
    // arena first -- the same treatment every other cross-region store
    // (global set!, vector-set!, ...) already gets. `dst` is the actual
    // storage slot; it is never inside a region arena, so the barrier always
    // promotes all the way out, exactly as intended.
    eshkol_region_write_barrier_into(&param->stack[0], &param->stack[0], &default_val);
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

    param->top++;
    // Same region write barrier treatment as the constructor default (see
    // eshkol_make_parameter above): `val` may point into a region that is
    // still active on entry to this `parameterize` binding but pops before
    // the binding is popped/read again, so it must be promoted out of any
    // active region before landing in the malloc'd (never region-owned)
    // value stack.
    eshkol_region_write_barrier_into(&param->stack[param->top], &param->stack[param->top], &val);
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
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top < 0 || !param->stack) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
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

/** @brief Pointer-argument wrapper around eshkol_parameter_ref; writes the current value through `result` (no-op if `result` is null). */
void eshkol_parameter_ref_ptr(void* param, eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_ref(param);
}

}  // extern "C"
