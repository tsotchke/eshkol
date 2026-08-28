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
#include <eshkol/core/resource_limits.h>

#include <cstdint>
#include <cstring>

extern "C" {

extern void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/* ESH-0393: the AD seed/point coercion authority (lib/core/runtime_taylor.c).
 * Converts any numeric tagged value — int64, double, bignum, rational, jet,
 * Taylor tower — to the double it denotes, reporting via *ok whether the value
 * was a number at all. Used below so a (list …) AD point keeps exact elements. */
extern double eshkol_ad_seed_to_double(const eshkol_tagged_value_t* v, int32_t* ok);

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
            // ESH-0393: P2 correctly refused to reinterpret a HEAP_PTR element's
            // pointer bits as a double, but substituted 0.0 — which fabricates a
            // number just as surely, only quietly: (gradient f (list 1/3)) came
            // back #(0). An exact rational/bignum element HAS a double it
            // denotes, so ask the AD coercion authority for it and only fall
            // back for a genuinely non-numeric element.
            int32_t ok = 0;
            double d = eshkol_ad_seed_to_double(&e, &ok);
            if (!ok) d = 0.0;
            e.type = ESHKOL_VALUE_DOUBLE;
            e.data.double_val = d;
        }
        elems[i++] = e;
        cur = cell->cdr;
    }
    return vec;
}

// Convert a tagged cons-list to a Scheme vector (HEAP_SUBTYPE_VECTOR) WITHOUT
// coercing any element — each car is copied verbatim, tag and all.
//
// This is the OUTPUT-side counterpart of eshkol_list_to_svec above, and the two
// must not be interchanged. eshkol_list_to_svec exists for the AD *point*: it
// deliberately forces every element to ESHKOL_VALUE_DOUBLE so an exact
// rational/bignum seed becomes the double it denotes. Applying that to a
// vector field's RESULT would be catastrophic in the quiet direction: under AD
// mode each output component is an AD node (a HEAP_PTR/CALLABLE into the tape),
// eshkol_ad_seed_to_double would refuse it, `ok` would come back 0, and the
// component would be replaced by the fabricated 0.0 — every partial derivative
// would read zero and (curl F pt) would return #(0 0 0) for EVERY field, which
// is the correct answer for a gradient field and so would pass the very tests
// most likely to be written for it. Preserving the tag verbatim is what keeps
// the AD node reachable by the backpropagation that follows.
//
// LE-12: used by the jacobian/divergence/curl output path so a field written
// `(lambda (v) (list …))` — the shape docs/reference/ad/architecture.md already
// promises is accepted — is read as a vector instead of being misread as an
// eshkol_tensor_t. Returns the vector DATA pointer (length at offset 0,
// 16-byte tagged elements at offset 8; header with HEAP_SUBTYPE_VECTOR at -8),
// or nullptr on failure. The head is passed by pointer (ABI-safe: avoids
// 16-byte struct-by-value across the C boundary).
void* eshkol_list_to_svec_raw(arena_t* arena, const eshkol_tagged_value_t* head_tv) {
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
        elems[i++] = cell->car;   // verbatim — no numeric coercion
        cur = cell->cdr;
    }
    return vec;
}

// ── (tensor X) collection unpacking ──────────────────────────────────────────
//
// A single-argument `(tensor X)` classifies X by its RUNTIME VALUE, not by the
// form that produced it: a scalar becomes a 1-element tensor, an existing
// tensor passes through, and any nest of lists / Scheme vectors becomes the
// N-dimensional tensor its shape describes. `(tensor #(#(1 2) #(3 4)))` already
// built a 2x2 because the PARSER flattens nested `#(...)` literals at compile
// time; the runtime path below is what makes the same true when the nesting is
// only known dynamically — `(tensor (list (list 1 2) (list 3 4)))`.
//
// Before this, the runtime walked exactly ONE level: every element of the outer
// list was coerced with "HEAP_PTR -> 0.0", so a list of lists silently became a
// rank-1 tensor of zeros (displayed `#(0 0)`) and the shape was lost. A
// subsequent rank-2 `(tensor-ref t 0 0)` then indexed a rank-1 tensor and read
// past its 1-entry dimensions array. Ragged or otherwise non-rectangular nests
// cannot be a tensor at all, so those raise a clean catchable error instead of
// fabricating a wrong shape.

#define ESHKOL_TENSOR_COLLECTION_MAX_DIMS 8

/** @brief What role a tagged value plays inside a `(tensor X)` nest. */
typedef enum {
    COLL_LEAF = 0,  /**< a scalar element (or a non-collection heap object) */
    COLL_LIST,      /**< a cons list — one tensor dimension */
    COLL_VECTOR,    /**< a Scheme vector — one tensor dimension */
    COLL_TENSOR     /**< a tensor — contributes ALL of its own dimensions */
} coll_kind_t;

static coll_kind_t coll_classify(const eshkol_tagged_value_t* v) {
    if (!v) return COLL_LEAF;
    if (tagged_is_cons(v)) return COLL_LIST;
    if (v->type == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val) {
        const auto* hdr = ESHKOL_GET_HEADER((void*)(uintptr_t)v->data.ptr_val);
        if (hdr && hdr->subtype == HEAP_SUBTYPE_VECTOR) return COLL_VECTOR;
        if (hdr && hdr->subtype == HEAP_SUBTYPE_TENSOR) return COLL_TENSOR;
    }
    return COLL_LEAF;
}

/** @brief Number of elements in a cons list (stops at any non-cons tail). */
static int64_t coll_list_length(const eshkol_tagged_value_t* v) {
    int64_t n = 0;
    eshkol_tagged_value_t cur = *v;
    while (tagged_is_cons(&cur)) {
        n++;
        cur = ((arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val)->cdr;
    }
    return n;
}

/** @brief Scheme vector layout is [length:8][16-byte tagged elements]. */
static int64_t coll_vector_length(const eshkol_tagged_value_t* v) {
    int64_t len = *(const int64_t*)(uintptr_t)v->data.ptr_val;
    return len < 0 ? 0 : len;
}

static const eshkol_tagged_value_t* coll_vector_elems(const eshkol_tagged_value_t* v) {
    return (const eshkol_tagged_value_t*)((const char*)(uintptr_t)v->data.ptr_val
                                          + sizeof(int64_t));
}

static const eshkol_tensor_t* coll_as_tensor(const eshkol_tagged_value_t* v) {
    return (const eshkol_tensor_t*)(uintptr_t)v->data.ptr_val;
}

/** @brief Coerce a leaf to the double the tensor stores.
 *
 * A non-collection HEAP_PTR (bignum, rational, string, …) yields 0.0 rather
 * than its pointer bits reinterpreted as a double, matching the pre-existing
 * convention on every other tensor ingest path (P2). */
static double coll_leaf_double(const eshkol_tagged_value_t* e) {
    if (e->type == ESHKOL_VALUE_DOUBLE) return e->data.double_val;
    if (e->type == ESHKOL_VALUE_HEAP_PTR) return 0.0;
    return (double)e->data.int_val;
}

static void coll_raise(const char* message) {
    eshkol_raise(eshkol_make_exception_with_header(ESHKOL_EXCEPTION_ERROR, message));
}

/**
 * @brief Read the tensor shape a nest describes by descending its FIRST child.
 *
 * Each list/vector level contributes one dimension; a nested tensor contributes
 * all of its own. Siblings are NOT inspected here — coll_fill() validates that
 * every sibling matches this shape, which is what turns a ragged nest into a
 * clean error rather than a silently wrong tensor.
 *
 * @return false if the nest is deeper than ESHKOL_TENSOR_COLLECTION_MAX_DIMS.
 */
static bool coll_discover_shape(const eshkol_tagged_value_t* input,
                                uint64_t* dims, int* ndim, int max_dims) {
    eshkol_tagged_value_t node = *input;
    for (;;) {
        const coll_kind_t k = coll_classify(&node);
        if (k == COLL_LEAF) return true;

        if (k == COLL_TENSOR) {
            const eshkol_tensor_t* t = coll_as_tensor(&node);
            for (uint64_t d = 0; t && d < t->num_dimensions; d++) {
                if (*ndim >= max_dims) return false;
                dims[(*ndim)++] = t->dimensions ? t->dimensions[d] : 0;
            }
            return true;
        }

        const int64_t len = (k == COLL_LIST) ? coll_list_length(&node)
                                             : coll_vector_length(&node);
        if (*ndim >= max_dims) return false;
        dims[(*ndim)++] = (uint64_t)len;
        if (len == 0) return true;  // empty level: nothing deeper to measure

        // Descend into child 0. Copy through a temporary first: `node` is both
        // the source we read the child out of and the destination.
        eshkol_tagged_value_t child;
        if (k == COLL_LIST) {
            child = ((arena_tagged_cons_cell_t*)(uintptr_t)node.data.ptr_val)->car;
        } else {
            child = coll_vector_elems(&node)[0];
        }
        node = child;
    }
}

/**
 * @brief Flatten a nest into @p elements in row-major order, validating shape.
 *
 * @param level Depth of @p node in the nest; `level == ndim` means this node
 *              must be a scalar leaf.
 * @return false if the nest is ragged / mixes scalars with sub-collections at
 *         the same depth (an error has already been raised).
 */
static bool coll_fill(const eshkol_tagged_value_t* node,
                      const uint64_t* dims, int ndim, int level,
                      int64_t* elements, uint64_t* pos, uint64_t total) {
    const coll_kind_t k = coll_classify(node);

    if (level == ndim) {
        if (k != COLL_LEAF) {
            coll_raise("tensor: nested list/vector is not rectangular — a "
                       "sub-collection appears where a number was expected");
            return false;
        }
        if (*pos >= total) return true;  // defensive: never write past the buffer
        const double d = coll_leaf_double(node);
        std::memcpy(&elements[(*pos)++], &d, sizeof(double));
        return true;
    }

    if (k == COLL_TENSOR) {
        // A nested tensor supplies every remaining dimension at once.
        const eshkol_tensor_t* t = coll_as_tensor(node);
        const int remaining = ndim - level;
        if (!t || (int)t->num_dimensions != remaining || !t->dimensions || !t->elements) {
            coll_raise("tensor: nested tensor element does not have the same "
                       "shape as its siblings");
            return false;
        }
        for (int d = 0; d < remaining; d++) {
            if (t->dimensions[d] != dims[level + d]) {
                coll_raise("tensor: nested tensor element does not have the "
                           "same shape as its siblings");
                return false;
            }
        }
        for (uint64_t i = 0; i < t->total_elements && *pos < total; i++) {
            elements[(*pos)++] = t->elements[i];  // already double bit patterns
        }
        return true;
    }

    if (k == COLL_LEAF) {
        coll_raise("tensor: nested list/vector is not rectangular — a number "
                   "appears where a sub-collection was expected");
        return false;
    }

    const int64_t len = (k == COLL_LIST) ? coll_list_length(node)
                                         : coll_vector_length(node);
    if ((uint64_t)len != dims[level]) {
        coll_raise("tensor: nested list/vector is ragged — every "
                   "sub-collection at the same depth must have the same length");
        return false;
    }

    if (k == COLL_LIST) {
        eshkol_tagged_value_t cur = *node;
        for (int64_t i = 0; i < len && tagged_is_cons(&cur); i++) {
            arena_tagged_cons_cell_t* cell =
                (arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            const eshkol_tagged_value_t car = cell->car;
            if (!coll_fill(&car, dims, ndim, level + 1, elements, pos, total)) return false;
            cur = cell->cdr;
        }
        return true;
    }

    const eshkol_tagged_value_t* elems = coll_vector_elems(node);
    for (int64_t i = 0; i < len; i++) {
        if (!coll_fill(&elems[i], dims, ndim, level + 1, elements, pos, total)) return false;
    }
    return true;
}

// Build a tensor from a single (tensor X) argument: a nest of lists and/or
// Scheme vectors is unpacked into the N-dimensional tensor its shape describes
// (numpy-like), an existing tensor is returned as-is, and any scalar becomes a
// 1-element tensor. Without this, (tensor (list 1 2 3)) made a 1-element tensor
// whose sole element was the list pointer's bits reinterpreted as a double
// (garbage), and (tensor (list (list 1 2) (list 3 4))) made a rank-1 tensor of
// zeros. Returns the tensor pointer, or nullptr on allocation failure; a
// non-rectangular nest raises a catchable error and does not return.
void* eshkol_tensor_from_collection(arena_t* arena, const eshkol_tagged_value_t* input) {
    if (!arena || !input) return nullptr;

    const coll_kind_t kind = coll_classify(input);

    if (kind == COLL_TENSOR) {
        return (void*)(uintptr_t)input->data.ptr_val;  // already a tensor
    }

    if (kind == COLL_LEAF) {
        // Scalar -> 1-element tensor.
        eshkol_tensor_t* t = arena_allocate_tensor_full(arena, 1, 1);
        if (!t) return nullptr;
        if (t->dimensions) t->dimensions[0] = 1;
        const double d = coll_leaf_double(input);
        std::memcpy(&t->elements[0], &d, sizeof(double));
        return t;
    }

    uint64_t dims[ESHKOL_TENSOR_COLLECTION_MAX_DIMS];
    int ndim = 0;
    if (!coll_discover_shape(input, dims, &ndim,
                             ESHKOL_TENSOR_COLLECTION_MAX_DIMS)) {
        coll_raise("tensor: nested list/vector nests deeper than 8 dimensions");
        return nullptr;
    }
    if (ndim == 0) {
        // Unreachable: the leaf and tensor cases returned above, so the input
        // is a list or vector and contributed at least one dimension.
        coll_raise("tensor: cannot determine a shape for the given collection");
        return nullptr;
    }

    uint64_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] != 0 && total > UINT64_MAX / dims[i]) {
            coll_raise("tensor: nested list/vector shape overflows");
            return nullptr;
        }
        total *= dims[i];
    }

    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, (uint64_t)ndim, total);
    if (!t) return nullptr;
    if (t->dimensions) {
        for (int i = 0; i < ndim; i++) t->dimensions[i] = dims[i];
    }

    uint64_t pos = 0;
    if (total > 0 && !coll_fill(input, dims, ndim, 0, t->elements, &pos, total)) {
        return nullptr;  // coll_fill already raised
    }
    return t;
}

/**
 * @brief True when @p input is a list or vector at least one of whose elements
 *        is itself a collection — a nest that describes rank >= 2.
 *
 * Companion to eshkol_tensor_from_collection(), and deliberately in the same
 * translation unit so both answers come from one classifier (coll_classify).
 * Every tensor operation routes its operand through
 * eshkol_tensor_operand_checked(), which coerces a FLAT numeric vector to a
 * rank-1 tensor; this predicate is what tells it that the operand is instead a
 * nest and belongs to the rank-N walker. Without it, the two spellings of the
 * same value disagreed:
 *
 *     (tensor-shape #(#(1.0 2.0) #(3.0 4.0)))                  => (2 2)
 *     (tensor-shape (vector (vector 1.0 2.0) (vector 3.0 4.0))) => type error
 *
 * because the literal is flattened to a rank-2 tensor by the parser while the
 * runtime-built value reached the operand check as a vector whose elements are
 * not numbers.
 *
 * Only the top level is inspected: a deeper nest is validated (and a ragged one
 * rejected) by the walker itself, which is the single place raggedness is
 * decided. A nest is reported for a nested tensor element too, since
 * `(vector (tensor 1.0 2.0) (tensor 3.0 4.0))` is the same rank-2 value.
 */
bool eshkol_tensor_collection_is_nested(const eshkol_tagged_value_t* input) {
    if (!input) return false;

    const coll_kind_t kind = coll_classify(input);
    if (kind == COLL_LIST) {
        eshkol_tagged_value_t cur = *input;
        while (tagged_is_cons(&cur)) {
            const arena_tagged_cons_cell_t* cell =
                (const arena_tagged_cons_cell_t*)(uintptr_t)cur.data.ptr_val;
            if (!cell) return false;
            if (coll_classify(&cell->car) != COLL_LEAF) return true;
            cur = cell->cdr;
        }
        return false;
    }
    if (kind == COLL_VECTOR) {
        const int64_t len = coll_vector_length(input);
        const eshkol_tagged_value_t* elems = coll_vector_elems(input);
        for (int64_t i = 0; i < len; i++) {
            if (coll_classify(&elems[i]) != COLL_LEAF) return true;
        }
        return false;
    }
    return false;
}

// Extract up to `max_n` scalar doubles from an AD operator input that may be a
// Scheme vector (HEAP_SUBTYPE_VECTOR, 16-byte tagged elements), a cons list, or
// a tensor (HEAP_SUBTYPE_TENSOR, 8-byte double bit patterns). Writes them into
// `out` and returns the count. Used by the multi-parameter
// hessian/laplacian/directional-derivative paths, which need the point as a
// plain double array to call an N-ary function without constructing AD nodes
// (reverse-mode AD nodes passed as separate args crash function dispatch).
//
// Those paths were finite differences when this helper was written and are not
// any more — they seed forward-over-forward jets and read the mixed second-order
// component, exactly (autodiff_codegen.cpp, `evalDual2`). The stale word
// "finite-difference" was removed here rather than left to imply an FD fallback
// that no longer exists; `(ad-finite-difference-evals)` is the authority on
// whether any FD ran, and it stays 0 through these paths.
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

/**
 * @brief Increment and check the calling thread's recursion-depth counter.
 *
 * Emitted inline by codegen at the entry of recursive-call sites to guard
 * against native stack overflow. Exceeding the ceiling resets the counter and
 * terminates with the documented ESHKOL_EXIT_LIMIT_STACK status rather than
 * letting the recursion run the native stack into the ground.
 *
 * SW-10: the ceiling is `ESHKOL_MAX_STACK` (via eshkol_get_limits()), not the
 * hard-coded 100000 this used to compare against. There were two stack-depth
 * mechanisms in the tree with the same default and no connection between them
 * — this one, which codegen actually calls, and the configurable
 * `max_stack_depth` the environment variable fed, which nothing consulted.
 * They are now the same mechanism, which is what makes the documented variable
 * take effect. Reading the limit costs a load from an already-hot global, and
 * the depth counter itself is unchanged, so a program that stays under the
 * ceiling behaves exactly as before.
 *
 * Since this already runs at every guarded function entry, it is also the
 * natural place to notice a pending execution-timeout interrupt: the watchdog
 * thread can only request one, and something running user code has to act on
 * it. Recursive and call-heavy programs are covered here; tail-call loops,
 * which enter no new frames, are covered by the poll codegen emits at the loop
 * back-edge.
 *
 * @return The thread-local recursion depth after incrementing (only
 *         reachable if under the limit; otherwise this call does not
 *         return normally).
 */
int64_t eshkol_check_recursion_depth(void) {
    __eshkol_recursion_depth++;

    const eshkol_resource_limits_t* limits = eshkol_get_limits();
    const int64_t max_depth = (int64_t)limits->max_stack_depth;
    if (max_depth > 0 && __eshkol_recursion_depth > max_depth) {
        __eshkol_recursion_depth = 0;
        eshkol_limit_enforce(ESHKOL_LIMIT_STACK_OVERFLOW, nullptr);
        // Advisory mode (ESHKOL_ENFORCE_LIMITS=false) still must not let the
        // recursion continue into a real stack overflow, so it stays fatal —
        // just via the catchable runtime condition it has always used.
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "maximum recursion depth (%lld) exceeded",
                             (long long)max_depth);
    }

    eshkol_limit_poll_interrupt();
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
