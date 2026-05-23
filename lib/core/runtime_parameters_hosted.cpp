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

#define HEAP_SUBTYPE_PARAMETER 20

extern "C" {

extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);

typedef struct {
    eshkol_tagged_value_t* stack;
    int top;
    int capacity;
} eshkol_param_t;

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
    param->stack[0] = default_val;
    return (void*)param;
}

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
    param->stack[param->top] = val;
}

void eshkol_parameter_pop(void* param_ptr) {
    if (!param_ptr) return;
    eshkol_param_t* param = (eshkol_param_t*)param_ptr;

    if (param->top > 0) {
        param->top--;
    }
}

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

void* eshkol_make_parameter_ptr(void* arena, const eshkol_tagged_value_t* default_val) {
    if (!default_val) return nullptr;
    return eshkol_make_parameter(arena, *default_val);
}

void eshkol_parameter_push_ptr(void* param, const eshkol_tagged_value_t* val) {
    if (!val) return;
    eshkol_parameter_push(param, *val);
}

void eshkol_parameter_ref_ptr(void* param, eshkol_tagged_value_t* result) {
    if (!result) return;
    *result = eshkol_parameter_ref(param);
}

}  // extern "C"
