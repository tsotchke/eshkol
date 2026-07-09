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
#include <cstring>

extern "C" {

extern void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/**
 * @brief Test whether a tagged value denotes a live (non-null) cons cell.
 *
 * Accepts either the dedicated ESHKOL_VALUE_CONS_PTR tag (with a non-zero
 * pointer) or a generic ESHKOL_VALUE_HEAP_PTR whose object header reports
 * HEAP_SUBTYPE_CONS.
 *
 * @param value Value to test (may be NULL).
 * @return      true if @p value is a non-null pointer to a cons cell.
 */
static bool tagged_is_cons(const eshkol_tagged_value_t* value) {
    if (!value) return false;
    if (value->type == ESHKOL_VALUE_CONS_PTR) return value->data.ptr_val != 0;
    if (value->type != ESHKOL_VALUE_HEAP_PTR || value->data.ptr_val == 0) return false;

    const auto* header =
        ESHKOL_GET_HEADER((void*)(uintptr_t)value->data.ptr_val);
    return header && header->subtype == HEAP_SUBTYPE_CONS;
}

/** @brief Overwrite *out in place with the tagged representation of the empty list ('()). */
static void set_null_tagged(eshkol_tagged_value_t* out) {
    out->type = ESHKOL_VALUE_NULL;
    out->flags = 0;
    out->reserved = 0;
    out->data.ptr_val = 0;
}

/** @brief Overwrite *out in place with a HEAP_PTR tagged value pointing at @p cell. */
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

// Convert a tagged cons-list to a Scheme vector (HEAP_SUBTYPE_VECTOR) so the
// gradient/jacobian dispatch can treat a (list …) input identically to a
// (vector …) input. Multi-parameter reverse/forward-mode gradient previously
// SIGSEGV'd on a list input because the cons cell fell through to the vector
// path and was misread as [length][elements]. Returns the vector DATA pointer
// (length at offset 0, 16-byte tagged elements at offset 8; header with
// HEAP_SUBTYPE_VECTOR at -8), or nullptr on failure. The list head is passed
// by pointer (ABI-safe: avoids 16-byte struct-by-value across the C boundary).
void* eshkol_list_to_svec(arena_t* arena, const eshkol_tagged_value_t* head_tv) {
    if (!arena || !head_tv) return nullptr;

    int64_t n = 0;
    eshkol_tagged_value_t cur = *head_tv;
    while (tagged_is_cons(&cur)) {
        n++;
        cur = ((arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->cdr;
    }

    void* vec = arena_allocate_vector_with_header(arena, (size_t)n);
    if (!vec) return nullptr;
    *(int64_t*)vec = n;  // length at offset 0
    eshkol_tagged_value_t* elems =
        (eshkol_tagged_value_t*)((char*)vec + sizeof(int64_t));

    cur = *head_tv;
    int64_t i = 0;
    while (tagged_is_cons(&cur) && i < n) {
        arena_tagged_cons_cell_t* cell =
            (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
        eshkol_tagged_value_t e = cell->car;
        // The AD svec path reads each element as a double; promote exact
        // integers so (gradient f (list 1 2)) behaves like (list 1.0 2.0).
        if (e.type != ESHKOL_VALUE_DOUBLE) {
            // P2: an integer promotes to its value; a HEAP_PTR (bignum/rational/
            // other) must NOT have its pointer bits reinterpreted as a double —
            // use 0.0 rather than garbage.
            double d = (e.type == ESHKOL_VALUE_HEAP_PTR) ? 0.0 : (double)e.data.int_val;
            e.type = ESHKOL_VALUE_DOUBLE;
            e.data.double_val = d;
        }
        elems[i++] = e;
        cur = cell->cdr;
    }
    return vec;
}

// Build a 1-D tensor from a single (tensor X) argument: a list or Scheme vector
// is unpacked element-by-element (numpy-like), an existing tensor is returned
// as-is, and any scalar becomes a 1-element tensor. Without this, (tensor (list
// 1 2 3)) made a 1-element tensor whose sole element was the list pointer's bits
// reinterpreted as a double (garbage). Returns the tensor pointer or nullptr.
void* eshkol_tensor_from_collection(arena_t* arena, const eshkol_tagged_value_t* input) {
    if (!arena || !input) return nullptr;

    if (input->type == ESHKOL_VALUE_HEAP_PTR && input->data.ptr_val) {
        const auto* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)input->data.ptr_val);
        if (hdr && hdr->subtype == HEAP_SUBTYPE_TENSOR) {
            return (void*)(uintptr_t)input->data.ptr_val;  // already a tensor
        }
        if (hdr && hdr->subtype == HEAP_SUBTYPE_VECTOR) {
            char* v = (char*)(uintptr_t)input->data.ptr_val;
            int64_t len = *(int64_t*)v;
            if (len < 0) len = 0;
            const eshkol_tagged_value_t* elems =
                (const eshkol_tagged_value_t*)(v + sizeof(int64_t));
            eshkol_tensor_t* t = arena_allocate_tensor_full(arena, 1, (uint64_t)len);
            if (!t) return nullptr;
            if (t->dimensions) t->dimensions[0] = (uint64_t)len;
            for (int64_t i = 0; i < len; i++) {
                const eshkol_tagged_value_t* e = &elems[i];
                double d = (e->type == ESHKOL_VALUE_DOUBLE) ? e->data.double_val
                         : (e->type == ESHKOL_VALUE_HEAP_PTR) ? 0.0
                         : (double)e->data.int_val;
                std::memcpy(&t->elements[i], &d, sizeof(double));
            }
            return t;
        }
    }

    if (tagged_is_cons(input)) {
        int64_t n = 0;
        eshkol_tagged_value_t cur = *input;
        while (tagged_is_cons(&cur)) {
            n++;
            cur = ((arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->cdr;
        }
        eshkol_tensor_t* t = arena_allocate_tensor_full(arena, 1, (uint64_t)n);
        if (!t) return nullptr;
        if (t->dimensions) t->dimensions[0] = (uint64_t)n;
        cur = *input;
        int64_t i = 0;
        while (tagged_is_cons(&cur) && i < n) {
            arena_tagged_cons_cell_t* cell =
                (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            const eshkol_tagged_value_t* e = &cell->car;
            double d = (e->type == ESHKOL_VALUE_DOUBLE) ? e->data.double_val
                     : (e->type == ESHKOL_VALUE_HEAP_PTR) ? 0.0
                     : (double)e->data.int_val;
            std::memcpy(&t->elements[i++], &d, sizeof(double));
            cur = cell->cdr;
        }
        return t;
    }

    // Scalar -> 1-element tensor.
    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, 1, 1);
    if (!t) return nullptr;
    if (t->dimensions) t->dimensions[0] = 1;
    double d = (input->type == ESHKOL_VALUE_DOUBLE) ? input->data.double_val
             : (input->type == ESHKOL_VALUE_HEAP_PTR) ? 0.0
             : (double)input->data.int_val;
    std::memcpy(&t->elements[0], &d, sizeof(double));
    return t;
}

// Extract up to `max_n` scalar doubles from an AD operator input that may be a
// Scheme vector (HEAP_SUBTYPE_VECTOR, 16-byte tagged elements), a cons list, or
// a tensor (HEAP_SUBTYPE_TENSOR, 8-byte double bit patterns). Writes them into
// `out` and returns the count. Used by the multi-parameter finite-difference
// hessian/laplacian/directional-derivative paths, which need the point as a
// plain double array to call an N-ary function without constructing AD nodes
// (reverse-mode AD nodes passed as separate args crash function dispatch).
int64_t eshkol_ad_extract_doubles(const eshkol_tagged_value_t* input,
                                  double* out, int64_t max_n) {
    if (!input || !out || max_n <= 0) return 0;

    // Tensor: header subtype TENSOR, elements are int64 bit patterns of doubles.
    if (input->type == ESHKOL_VALUE_HEAP_PTR && input->data.ptr_val) {
        const auto* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)input->data.ptr_val);
        if (hdr && hdr->subtype == HEAP_SUBTYPE_TENSOR) {
            // eshkol_tensor_t: dimensions(0) num_dims(8) elements(16) total(24)
            char* t = (char*)(uintptr_t)input->data.ptr_val;
            int64_t total = *(int64_t*)(t + 24);
            const int64_t* elems = *(const int64_t**)(t + 16);
            int64_t n = total < max_n ? total : max_n;
            for (int64_t i = 0; i < n && elems; i++) {
                double d; std::memcpy(&d, &elems[i], sizeof(double));
                out[i] = d;
            }
            return elems ? n : 0;
        }
        if (hdr && hdr->subtype == HEAP_SUBTYPE_VECTOR) {
            // Scheme vector: [length:8][16-byte tagged elements].
            char* v = (char*)(uintptr_t)input->data.ptr_val;
            int64_t len = *(int64_t*)v;
            const eshkol_tagged_value_t* elems =
                (const eshkol_tagged_value_t*)(v + sizeof(int64_t));
            int64_t n = len < max_n ? len : max_n;
            for (int64_t i = 0; i < n; i++) {
                const eshkol_tagged_value_t* e = &elems[i];
                if (e->type == ESHKOL_VALUE_DOUBLE) { double d; std::memcpy(&d, &e->data, sizeof(double)); out[i] = d; }
                else out[i] = (e->type == ESHKOL_VALUE_HEAP_PTR) ? 0.0 : (double)e->data.int_val;  // P2: no pointer-bits-as-double
            }
            return n;
        }
    }
    // Cons list.
    eshkol_tagged_value_t cur = *input;
    int64_t i = 0;
    while (tagged_is_cons(&cur) && i < max_n) {
        arena_tagged_cons_cell_t* cell =
            (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
        const eshkol_tagged_value_t* e = &cell->car;
        if (e->type == ESHKOL_VALUE_DOUBLE) { double d; std::memcpy(&d, &e->data, sizeof(double)); out[i] = d; }
        else out[i] = (double)e->data.int_val;
        i++;
        cur = cell->cdr;
    }
    return i;
}

// ===== STACK OVERFLOW PROTECTION =====

// Per-thread recursion depth counter.
// thread_local is correct: recursion depth tracks the call stack, per thread.
static thread_local int64_t __eshkol_recursion_depth = 0;
static const int64_t ESHKOL_MAX_RECURSION_DEPTH = 100000;  // 100K frames

/**
 * @brief Increment and check the calling thread's recursion-depth counter.
 *
 * Emitted inline by codegen at the entry of recursive-call sites to guard
 * against native stack overflow. If the counter exceeds
 * ESHKOL_MAX_RECURSION_DEPTH (100,000), resets it to 0 and raises a fatal
 * ESHKOL_EXCEPTION_ERROR rather than letting the recursion continue.
 *
 * @return The thread-local recursion depth after incrementing (only
 *         reachable if under the limit; otherwise this call does not
 *         return normally).
 */
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

/** @brief Decrement the calling thread's recursion-depth counter on return from a guarded call (no-op if already 0). */
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

/* Convert a proper tagged-cons list into a heterogeneous Scheme vector
 * (HEAP_SUBTYPE_VECTOR). Layout matches the codegen `list->vector` / vector
 * literal path: an 8-byte length header at offset 0 followed by N 16-byte
 * tagged elements. Used by quasiquote-vector codegen (`#(1 ,x 3)`), where the
 * template is first materialised as a list (so unquote/unquote-splicing reuse
 * the existing list machinery) and then vectorised here. Result is written to
 * *out; an empty/absent list yields a zero-length vector. */
void eshkol_list_to_vector_sret(eshkol_tagged_value_t* out,
                                const eshkol_tagged_value_t* list_tv) {
    if (!out) return;
    set_null_tagged(out);

    int64_t n = 0;
    if (list_tv) {
        eshkol_tagged_value_t cur = *list_tv;
        while (tagged_is_cons(&cur)) {
            n++;
            auto* src = (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            cur = src->cdr;
        }
    }

    arena_t* arena = get_global_arena();
    size_t alloc_size = (size_t)n * sizeof(eshkol_tagged_value_t) + 8;
    void* vec = arena_allocate_with_header(arena, alloc_size, HEAP_SUBTYPE_VECTOR, 0);
    if (!vec) return;

    *(int64_t*)vec = n;
    eshkol_tagged_value_t* elems =
        (eshkol_tagged_value_t*)((char*)vec + 8);

    if (list_tv) {
        eshkol_tagged_value_t cur = *list_tv;
        int64_t i = 0;
        while (tagged_is_cons(&cur) && i < n) {
            auto* src = (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            elems[i++] = src->car;
            cur = src->cdr;
        }
    }

    out->type = ESHKOL_VALUE_HEAP_PTR;
    out->flags = 0;
    out->reserved = 0;
    out->data.ptr_val = (uint64_t)(uintptr_t)vec;
}

/**
 * @brief Raise a fatal out-of-bounds-index error from codegen-generated index checks.
 *
 * @param op_name Name of the offending operation (e.g. "list-ref",
 *                "vector-ref"), or a default message if NULL.
 * @param idx     Index that was out of bounds.
 * @param length  Length of the collection being indexed.
 */
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

/** @brief Reset the calling thread's recursion-depth counter to 0 (e.g. after recovering from a caught exception). */
void eshkol_reset_recursion_depth(void) {
    __eshkol_recursion_depth = 0;
}

}  // extern "C"
