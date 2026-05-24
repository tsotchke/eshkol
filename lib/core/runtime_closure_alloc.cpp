/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime closure allocation helpers.
 *
 * These helpers implement the closure object/environment allocation ABI used
 * by generated closure code, higher-order functions, and continuation setup.
 * They sit above raw arena block allocation but do not depend on hosted
 * process, filesystem, or thread APIs.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

extern "C" {

eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures) {
    if (!arena) {
        eshkol_error("Cannot allocate closure environment: null arena");
        return nullptr;
    }

    if (num_captures == 0) {
        eshkol_warn("Allocating closure environment with zero captures");
    }

    const size_t size = sizeof(eshkol_closure_env_t) +
                        (num_captures * sizeof(eshkol_tagged_value_t));

    auto* env = (eshkol_closure_env_t*)arena_allocate_aligned(arena, size, 16);
    if (!env) {
        eshkol_error("Failed to allocate closure environment for %zu captures", num_captures);
        return nullptr;
    }

    env->num_captures = num_captures;

    for (size_t i = 0; i < num_captures; i++) {
        env->captures[i].type = ESHKOL_VALUE_NULL;
        env->captures[i].flags = 0;
        env->captures[i].reserved = 0;
        env->captures[i].data.raw_val = 0;
    }

    eshkol_debug("Allocated closure environment for %zu captures at %p",
                 num_captures, (void*)env);

    return env;
}

eshkol_closure_t* arena_allocate_closure(arena_t* arena, uint64_t func_ptr, size_t packed_info,
                                         uint64_t sexpr_ptr, uint64_t return_type_info,
                                         const char* name) {
    if (!arena) {
        eshkol_error("Cannot allocate closure: null arena");
        return nullptr;
    }

    const size_t actual_num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(packed_info);

    auto* closure = (eshkol_closure_t*)arena_allocate_aligned(
        arena, sizeof(eshkol_closure_t), 16);
    if (!closure) {
        eshkol_error("Failed to allocate closure structure");
        return nullptr;
    }

    closure->func_ptr = func_ptr;
    closure->sexpr_ptr = sexpr_ptr;
    closure->name = name;
    closure->return_type = (uint8_t)(return_type_info & 0xFF);
    closure->input_arity = (uint8_t)((return_type_info >> 8) & 0xFF);
    closure->flags = CLOSURE_ENV_IS_VARIADIC(packed_info) ? CLOSURE_FLAG_VARIADIC : 0;
    if (name) {
        closure->flags |= ESHKOL_CLOSURE_FLAG_NAMED;
    }
    closure->reserved = 0;
    closure->hott_type_id = (uint32_t)((return_type_info >> 16) & 0xFFFFFFFF);

    if (actual_num_captures > 0) {
        closure->env = arena_allocate_closure_env(arena, actual_num_captures);
        if (!closure->env) {
            eshkol_error("Failed to allocate closure environment");
            return nullptr;
        }
        closure->env->num_captures = packed_info;
    } else {
        closure->env = nullptr;
    }

    eshkol_debug("Allocated closure at %p with func_ptr=%p, env=%p (%zu captures), return_type=%d, arity=%d, name=%s",
                 (void*)closure, (void*)func_ptr, (void*)closure->env, actual_num_captures,
                 closure->return_type, closure->input_arity, name ? name : "(anonymous)");

    return closure;
}

eshkol_closure_t* arena_allocate_closure_with_header(arena_t* arena, uint64_t func_ptr,
                                                     size_t packed_info, uint64_t sexpr_ptr,
                                                     uint64_t return_type_info,
                                                     const char* name) {
    if (!arena) {
        eshkol_error("Cannot allocate closure with header: null arena");
        return nullptr;
    }

    const size_t actual_num_captures = CLOSURE_ENV_GET_NUM_CAPTURES(packed_info);

    const size_t data_size = sizeof(eshkol_closure_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~((size_t)7);

    auto* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate closure with header");
        return nullptr;
    }

    auto* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = (actual_num_captures == 0) ?
        CALLABLE_SUBTYPE_LAMBDA_SEXPR : CALLABLE_SUBTYPE_CLOSURE;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    auto* closure = (eshkol_closure_t*)(mem + sizeof(eshkol_object_header_t));
    closure->func_ptr = func_ptr;
    closure->sexpr_ptr = sexpr_ptr;
    closure->name = name;
    closure->return_type = (uint8_t)(return_type_info & 0xFF);
    closure->input_arity = (uint8_t)((return_type_info >> 8) & 0xFF);
    closure->flags = CLOSURE_ENV_IS_VARIADIC(packed_info) ? CLOSURE_FLAG_VARIADIC : 0;
    if (name) {
        closure->flags |= ESHKOL_CLOSURE_FLAG_NAMED;
    }
    closure->reserved = 0;
    closure->hott_type_id = (uint32_t)((return_type_info >> 16) & 0xFFFFFFFF);

    if (actual_num_captures > 0) {
        closure->env = arena_allocate_closure_env(arena, actual_num_captures);
        if (closure->env) {
            closure->env->num_captures = packed_info;
        }
    } else {
        closure->env = nullptr;
    }

    return closure;
}

}  // extern "C"
