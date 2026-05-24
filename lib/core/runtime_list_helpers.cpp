/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * List helper ABI used by generated code.
 *
 * These helpers operate on tagged cons cells and arena allocation only. Error
 * paths delegate to the shared runtime fatal sink until the freestanding
 * panic/error hook ABI replaces the hosted implementation.
 */

#include "arena_memory.h"

#include <cstdint>

extern "C" {

extern void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

static bool tagged_is_cons(const eshkol_tagged_value_t* value) {
    if (!value) return false;
    if (value->type == ESHKOL_VALUE_CONS_PTR) return value->data.ptr_val != 0;
    if (value->type != ESHKOL_VALUE_HEAP_PTR || value->data.ptr_val == 0) return false;

    const auto* header =
        ESHKOL_GET_HEADER((void*)(uintptr_t)value->data.ptr_val);
    return header && header->subtype == HEAP_SUBTYPE_CONS;
}

static void set_null_tagged(eshkol_tagged_value_t* out) {
    out->type = ESHKOL_VALUE_NULL;
    out->flags = 0;
    out->reserved = 0;
    out->data.ptr_val = 0;
}

static void set_cons_tagged(eshkol_tagged_value_t* out,
                            arena_tagged_cons_cell_t* cell) {
    out->type = ESHKOL_VALUE_HEAP_PTR;
    out->flags = 0;
    out->reserved = 0;
    out->data.ptr_val = (uint64_t)(uintptr_t)cell;
}

// Reverse a tagged-cons list. Walks the list, allocates a new cons cell per
// element, writes the resulting tagged_value (NULL for empty) into *out.
// Stops at any non-CONS terminator (NULL, dotted-pair tail, etc.) -- the
// function is robust to dotted lists but the dotted tail is dropped
// (matching R7RS reverse on improper lists). Used by string-split codegen.
//
// Output-pointer rather than struct-return-by-value sidesteps the ABI
// coupling between LLVM's IR-level struct and the platform's calling
// convention for <=16-byte structs.
void eshkol_list_reverse_tagged(arena_t* arena,
                                const eshkol_tagged_value_t* head_tv,
                                eshkol_tagged_value_t* out) {
    if (!out) return;
    set_null_tagged(out);
    if (!arena || !head_tv) return;

    eshkol_tagged_value_t cur = *head_tv;
    while (tagged_is_cons(&cur)) {
        auto* cell = (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;

        arena_tagged_cons_cell_t* new_cell = arena_allocate_cons_with_header(arena);
        if (!new_cell) return;
        arena_tagged_cons_set_tagged_value(new_cell, false, &cell->car);
        arena_tagged_cons_set_tagged_value(new_cell, true, out);

        set_cons_tagged(out, new_cell);
        cur = cell->cdr;
    }
}

// ===== STACK OVERFLOW PROTECTION =====

// Per-thread recursion depth counter.
// thread_local is correct: recursion depth tracks the call stack, per thread.
static thread_local int64_t __eshkol_recursion_depth = 0;
static const int64_t ESHKOL_MAX_RECURSION_DEPTH = 100000;  // 100K frames

int64_t eshkol_check_recursion_depth(void) {
    __eshkol_recursion_depth++;
    if (__eshkol_recursion_depth > ESHKOL_MAX_RECURSION_DEPTH) {
        __eshkol_recursion_depth = 0;
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "maximum recursion depth (%lld) exceeded",
                             (long long)ESHKOL_MAX_RECURSION_DEPTH);
    }
    return __eshkol_recursion_depth;
}

void eshkol_decrement_recursion_depth(void) {
    if (__eshkol_recursion_depth > 0) {
        __eshkol_recursion_depth--;
    }
}

/* Safety guards emitted inline by the codegen for car / cdr / list-ref when
 * the input argument's static type cannot be proven to be a pair. */
void eshkol_raise_not_pair(const char* op_name) {
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "%s",
                         op_name ? op_name : "car/cdr: argument is not a pair");
}

/* Append `lhs` (expected to be a proper list) to `rhs`, returning the
 * concatenated list. Used by quasiquote codegen for `,@splice-list`.
 *
 * The old implementation buffered lhs in a dynamic array and rebuilt it backward.
 * This version preserves the same observable behavior while allocating only
 * arena cons cells as it walks lhs, keeping the helper suitable for the
 * runtime-core source set. */
void eshkol_append_tagged_sret(eshkol_tagged_value_t* out,
                               const eshkol_tagged_value_t* lhs,
                               const eshkol_tagged_value_t* rhs) {
    if (!out) return;
    if (!rhs) {
        set_null_tagged(out);
        return;
    }
    if (!lhs || lhs->type == ESHKOL_VALUE_NULL) {
        *out = *rhs;
        return;
    }

    arena_t* arena = get_global_arena();
    if (!tagged_is_cons(lhs)) {
        arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
        if (!cell) {
            *out = *rhs;
            return;
        }
        cell->car = *lhs;
        cell->cdr = *rhs;
        set_cons_tagged(out, cell);
        return;
    }

    eshkol_tagged_value_t result;
    set_null_tagged(&result);
    arena_tagged_cons_cell_t* tail = nullptr;
    eshkol_tagged_value_t cur = *lhs;

    while (cur.type != ESHKOL_VALUE_NULL && cur.data.ptr_val != 0) {
        eshkol_tagged_value_t next_car;
        if (tagged_is_cons(&cur)) {
            auto* src = (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            next_car = src->car;
            cur = src->cdr;
        } else {
            // Preserve prior dotted-list behavior: append the tail atom as the
            // last copied element before rhs.
            next_car = cur;
            set_null_tagged(&cur);
        }

        arena_tagged_cons_cell_t* node = arena_allocate_cons_with_header(arena);
        if (!node) {
            *out = result.type == ESHKOL_VALUE_NULL ? *rhs : result;
            return;
        }
        node->car = next_car;
        node->cdr = *rhs;

        eshkol_tagged_value_t node_tv;
        set_cons_tagged(&node_tv, node);
        if (!tail) {
            result = node_tv;
        } else {
            tail->cdr = node_tv;
        }
        tail = node;
    }

    if (!tail) {
        *out = *rhs;
        return;
    }
    tail->cdr = *rhs;
    *out = result;
}

void eshkol_raise_index_oob(const char* op_name, int64_t idx, int64_t length) {
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                         "%s: index %lld out of bounds (length=%lld)",
                         op_name ? op_name : "list-ref/vector-ref",
                         (long long)idx,
                         (long long)length);
}

/* Raise an "improper list" error from codegen-generated walkers
 * (audit M7). Used by list->vector and similar tail-traversals
 * that encounter a non-pair, non-() tail. */
void eshkol_raise_improper_list(const char* msg) {
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                         "%s",
                         msg ? msg : "improper list");
}

void eshkol_reset_recursion_depth(void) {
    __eshkol_recursion_depth = 0;
}

}  // extern "C"
