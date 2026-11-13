/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime region helpers: escape and evacuation.
 *
 * This unit owns value escape out of a closing region: the Cheney-style
 * evacuator over the escaping subgraph (classification, copy, forwarding,
 * transactional rollback), the arena-span evacuator, the region write
 * barrier and the iteration-nursery recycle.  The region lifecycle is in
 * runtime_regions.cpp; runtime_regions_internal.h declares what the units share.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/exhaustive_dispatch.h"
#include "../../inc/eshkol/core/logic.h"       // eshkol_substitution_t / _fact_t / _knowledge_base_t
#include "../../inc/eshkol/core/inference.h"   // eshkol_factor_graph_t / _factor_t
#include "../../inc/eshkol/core/workspace.h"   // eshkol_workspace_t / _workspace_module_t
#include "../../inc/eshkol/core/rational.h"    // eshkol_rational_t (bignum-backed exact rationals)

#include <cstring>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>
#include <utility>
#include <atomic>
#include <mutex>
#include <array>

#include "runtime_regions_internal.h"

using namespace eshkol::regions_internal;

/**
 * @brief Determine which arena an escaping value from @p current should be copied into.
 *
 * Escaping a value means copying it out of the current (about-to-be-freed)
 * region's arena into a longer-lived arena. That target is the immediately
 * enclosing region's arena, if there is a parent region, or otherwise the
 * process/thread global arena (so a value escaping the outermost region
 * survives as long as ordinary heap allocations).
 *
 * @param current Region the value is escaping from (must be non-NULL).
 * @return        Arena the escaped copy should be allocated in.
 */
static arena_t* region_escape_target(eshkol_region_t* current) {
    // ESH-0214c: escape_base is captured at region_push and is the arena that
    // genuinely outlives this region. It must be used instead of
    // get_global_arena(), which during a with-region body has been hijacked by
    // codegen to point at THIS region's arena (so escaping through it would copy
    // the value straight back into the arena about to be freed).
    if (current->escape_base) {
        return current->escape_base;
    }
    if (current->parent && current->parent->arena) {
        return current->parent->arena;
    }
    return get_global_arena();
}

/**
 * @brief Copy @p size bytes at @p ptr out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p ptr unchanged (nothing to
 * escape). Otherwise allocates @p size bytes (8-byte aligned) in
 * region_escape_target()'s arena, memcpy's the bytes over, increments the
 * current region's escape_count, and returns the new copy. Used so a value
 * created inside a `(with-region ...)` block can safely be returned/escaped
 * before the region's arena is destroyed.
 *
 * @param ptr  Source bytes to copy (must be non-NULL).
 * @param size Number of bytes to copy (must be > 0).
 * @return     Pointer to the escaped copy, @p ptr unchanged if no region is
 *             active, or NULL on allocation failure or invalid arguments.
 */
void* region_escape(const void* ptr, size_t size) {
    if (!ptr || size == 0) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (void*)ptr;

    void* copy = arena_allocate_aligned(region_escape_target(current), size, 8);
    if (!copy) {
        eshkol_error("region_escape: failed to allocate %zu bytes in target arena", size);
        return nullptr;
    }

    std::memcpy(copy, ptr, size);
    current->escape_count++;
    return copy;
}

/**
 * @brief Copy a NUL-terminated string out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p str unchanged. Otherwise
 * allocates a header-prefixed string object of the right length in
 * region_escape_target()'s arena, copies the bytes plus a NUL terminator,
 * increments the current region's escape_count, and returns the new string.
 *
 * @param str Source NUL-terminated string (must be non-NULL).
 * @return    Pointer to the escaped copy, @p str unchanged if no region is
 *            active, or NULL on allocation failure.
 */
void* region_escape_string(const char* str) {
    if (!str) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (void*)str;

    const size_t len = std::strlen(str);
    auto* copy = (char*)arena_allocate_string_with_header(region_escape_target(current), len);
    if (!copy) {
        eshkol_error("region_escape_string: failed to allocate string of length %zu", len);
        return nullptr;
    }

    std::memcpy(copy, str, len);
    copy[len] = '\0';
    current->escape_count++;
    return copy;
}

/**
 * @brief Copy a single tagged cons cell out of the current region into the enclosing arena.
 *
 * If no region is currently active, returns @p cell unchanged. Otherwise
 * allocates a new cons cell in region_escape_target()'s arena, shallow-copies
 * car and cdr (tagged values, so this does not recursively escape anything
 * the cell's car/cdr point to), increments the current region's
 * escape_count, and returns the new cell.
 *
 * @param cell Source cons cell (must be non-NULL).
 * @return     Pointer to the escaped copy, @p cell unchanged if no region is
 *             active, or NULL on allocation failure.
 */
arena_tagged_cons_cell_t* region_escape_tagged_cons_cell(const arena_tagged_cons_cell_t* cell) {
    if (!cell) return nullptr;

    eshkol_region_t* current = region_current();
    if (!current) return (arena_tagged_cons_cell_t*)cell;

    arena_tagged_cons_cell_t* copy =
        arena_allocate_tagged_cons_cell(region_escape_target(current));
    if (!copy) {
        eshkol_error("region_escape_tagged_cons_cell: failed to allocate");
        return nullptr;
    }

    copy->car = cell->car;
    copy->cdr = cell->cdr;
    current->escape_count++;
    return copy;
}

// ───────────────────────────────────────────────────────────────────────────
// DEEP TRANSITIVE ESCAPE PROMOTION (evacuation) — ESH-0214c
//
// A region's arena is torn down at region_pop. A value that escapes the region
// (returned as the with-region body result, or stored by mutation into an outer
// container) must have its ENTIRE reachable subgraph that lives in the dying
// region copied out first, otherwise the escaped object keeps interior pointers
// (car/cdr, vector slots, captured env, hash key/value arrays, tensor buffers)
// aimed into freed memory -> "car/cdr: argument is not a pair" corruption.
//
// The evacuator is a Cheney-style copying collector restricted to the escaping
// subgraph: a forwarding map (old data-ptr -> new data-ptr) preserves shared
// structure and terminates on cycles, and a worklist drives a breadth-first
// walk so arbitrarily deep lists do not overflow the native stack.
//
// A node is COPIED iff it lives in an active region strictly INNER than the
// destination's owning region (`boundary_idx`); anything at-or-outside the
// boundary (an ancestor region, the global arena, static/stack storage) already
// outlives the destination and is left in place (forwarding identity). This is
// correct for nested regions: escaping to the global arena copies everything in
// any active region; escaping to region D copies only what lives in D's
// descendants.
// ───────────────────────────────────────────────────────────────────────────

// Walk an arena's block chain and test raw containment. Region arenas are
// non-thread-safe single-owner arenas (region_create uses arena_create), so no
// locking is required here.
static bool arena_contains_ptr(const arena_t* a, const void* p) {
    if (!a || !p) return false;
    const uint8_t* q = (const uint8_t*)p;
    for (const arena_block_t* b = a->current_block; b; b = b->next) {
        if (q >= b->memory && q < b->memory + b->used) return true;
    }
    return false;
}

// Index (into __region_stack) of the innermost active region whose arena
// contains @p p, or -1 if @p p is not inside any active region arena (i.e. it
// lives in the global/thread arena or in static/stack storage).
namespace eshkol::regions_internal {
int region_index_owning(const void* p) {
    for (int i = (int)__region_stack_depth - 1; i >= 0; --i) {
        eshkol_region_t* r = __region_stack[i];
        if (r && r->arena && arena_contains_ptr(r->arena, p)) return i;
    }
    return -1;
}
} // namespace eshkol::regions_internal

typedef void (*eshkol_parameter_value_visitor)(eshkol_tagged_value_t* value,
                                               void* context);
extern "C" void eshkol_parameter_visit_values(
    void* param, eshkol_parameter_value_visitor visitor, void* context);

// EvacKind, EvacFwdMap and EvacState are declared in runtime_regions_internal.h:
// the resident owner (runtime_regions_resident.cpp) runs the same transaction.

// Free a region's persistent deep-escape forwarding map (declared in
// runtime_regions_internal.h; region_destroy in runtime_regions.cpp calls it).
extern "C" size_t eshkol_region_forwarding_size(const eshkol_region_t* region) {
    return (region && region->fwd_map) ? ((const EvacFwdMap*)region->fwd_map)->size() : 0;
}

namespace eshkol::regions_internal {
void region_free_fwd_map(eshkol_region_t* region) {
    if (region && region->fwd_map) {
        delete (EvacFwdMap*)region->fwd_map;
        region->fwd_map = nullptr;
        region->fwd_target = nullptr;
    }
}
} // namespace eshkol::regions_internal

// ───────────────────────────────────────────────────────────────────────────
// WHICH TAGS CARRY AN ARENA POINTER
//
// Every region escape path — the write barrier, the with-region result escape,
// and the ESH-0214e nursery recycle — has to answer one question first: does
// this tagged value hold a pointer into the region being reclaimed? It asked
// ESHKOL_IS_ANY_PTR_TYPE, which enumerates the two CONSOLIDATED pointer tags
// (HEAP_PTR / CALLABLE) plus ports. That set is NOT the set of pointer-carrying
// tags.
//
// ESHKOL_VALUE_DUAL_NUMBER and ESHKOL_VALUE_COMPLEX sit in the 0-7 block that
// eshkol_value_type_t labels "IMMEDIATE VALUES — data stored directly in tagged
// value", but neither is an immediate: both put a POINTER in data.ptr_val,
// aimed at a headerless 16-byte pair arena-allocated at the operation that
// produced it (arena_allocate_dual_number; ComplexCodegen::packComplexToTagged).
// Reading them as immediates made every escape path a silent no-op for them, so
// a forward-mode dual or a complex number produced inside a region and stored
// into a structure outside it kept a pointer into the arena about to be
// recycled — and the next iteration reallocated the same address. The values
// stayed dereferenceable and plausible, which is why this corrupted answers
// instead of crashing.
//
// The arena-scope half of the same feature (eshkol_arena_iter_scope_end) got
// this right by inverting the test: only PROVABLY pointer-free immediates
// (NULL/INT64/DOUBLE/BOOL/CHAR, plus the eof-object) skip the pointer check.
// These helpers bring the region half into line with that rule.
//
// Deliberately NOT included:
//   SYMBOL (5)   — data.ptr_val is an interned symbol's stable text, owned by
//                  the interning table rather than a region arena. Copying it
//                  per escape would both be unnecessary and break the pointer
//                  identity interning exists to provide.
//   LOGIC_VAR(10)— data is a var_id integer, not a pointer.
//   HANDLE/BUFFER/STREAM/EVENT (16-19) — linear resources whose payloads are
//                  externally owned; duplicating one would duplicate the
//                  resource. A region-resident one is reported below rather
//                  than silently copied or silently dropped.
// ───────────────────────────────────────────────────────────────────────────

/* Byte size of the headerless, self-contained payload a pointer-carrying
 * IMMEDIATE tag addresses, or 0 when the tag is not one of those. These cannot
 * go through evac_object: it sizes and classifies an object from the
 * eshkol_object_header_t 8 bytes below the payload, and these carry no header —
 * those 8 bytes belong to whatever was allocated before them. A flat copy is
 * complete for both (the native dual is the eight-double JET8 payload; complex
 * is two doubles, with no interior pointers), and neither has
 * observable pointer identity, so copying cannot break eq?-style sharing the
 * way copying an interned symbol would. */
static size_t region_headerless_payload_size(uint8_t type) {
    /* Exactness flags may be OR'd into a numeric tag; strip them before
     * matching. The port flags share those bits but only ever ride on
     * HEAP_PTR, which is not one of the values matched here. */
    switch (type & (uint8_t)~(ESHKOL_VALUE_EXACT_FLAG | ESHKOL_VALUE_INEXACT_FLAG)) {
        case ESHKOL_VALUE_DUAL_NUMBER:
            return eshkol_ad_payload_size(ESHKOL_AD_PAYLOAD_DUAL_JET);
        case ESHKOL_VALUE_COMPLEX:
            return eshkol_ad_payload_size(ESHKOL_AD_PAYLOAD_USER_NUMBER);
        default:                       return 0;
    }
}

/* Does a value with this tag hold a pointer that a region reclaim could
 * invalidate? The single predicate every escape path shares. */
static bool region_value_carries_pointer(uint8_t type) {
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    return ESHKOL_IS_ANY_PTR_TYPE(type) || is_port ||
           region_headerless_payload_size(type) != 0;
}

// Classify an object (given its live original data pointer and the tagged value
// referencing it) into an EvacKind. Ports are never deep-traversed (they wrap OS
// resources / fds); they are leaf-copied with care so the escaped port struct is
// stable, but the underlying handle is intentionally shared, not duplicated.
static EvacKind evac_kind_for(EvacState& st, const eshkol_tagged_value_t& v, const void* old_data) {
    const uint8_t type = v.type;
    const bool is_port = ((type & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((type & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    if (is_port) { if (st.abort_on_refusal) st.refuse(); return EVAC_LEAF; }

    const auto* h = (const eshkol_object_header_t*)
        ((const uint8_t*)old_data - sizeof(eshkol_object_header_t));
    const uint8_t sub = h->subtype;

    if (ESHKOL_IS_ANY_CALLABLE_TYPE(type)) {
        const bool zero_capture_closure = sub == CALLABLE_SUBTYPE_LAMBDA_SEXPR &&
            h->size == sizeof(eshkol_closure_t) &&
            static_cast<const eshkol_closure_t*>(old_data)->env == nullptr;
        if (st.abort_on_refusal && sub != CALLABLE_SUBTYPE_CLOSURE && !zero_capture_closure) {
            st.refuse();
            eshkol_warn("resident evacuation refuses callable subtype %u", (unsigned)sub);
        }
        // An exhaustive switch rather than an if-chain, for the same reason as
        // the heap-subtype switch below: an if-chain's fallthrough is a default
        // wearing different syntax, and a new callable subtype would inherit
        // EVAC_LEAF from it in silence.
        ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
        switch ((callable_subtype_t)sub) {
        case CALLABLE_SUBTYPE_CLOSURE:
            return EVAC_CLOSURE;
        case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
            // The current allocator uses this subtype for a compiled closure
            // with no captures, not only legacy lambda-as-data. Residence
            // admits exactly that declared layout and walks name/source fields.
            if (st.abort_on_refusal && zero_capture_closure) return EVAC_CLOSURE;
            break;
        case CALLABLE_SUBTYPE_PRIMITIVE:
        case CALLABLE_SUBTYPE_CONTINUATION:
            break;  // handled below
        case CALLABLE_SUBTYPE_AD_NODE:
            // SW-164: in ARENA-SPAN mode an AD node is a REFUSAL, interior
            // graph or not. The tape's nodes[] array holds the ORIGINAL
            // pointer, so moving the node breaks the recorded evaluation order
            // however completely the node itself is copied -- only the span's
            // owner can decide not to reclaim. Outside span mode the ESH-0214d
            // deep walk (case EVAC_AD_NODE below) rebases the node's own graph
            // links and its self-described payloads, which is the right answer
            // for a headered standalone node crossing a region boundary.
            if (st.span_arena) {
                st.refuse();
                return EVAC_LEAF;
            }
            return EVAC_AD_NODE;
        }
        ESHKOL_EXHAUSTIVE_SWITCH_END
        // LAMBDA_SEXPR / PRIMITIVE / CONTINUATION have no safe generic
        // interior layout here and remain shallow. AD_NODE returned above has
        // its own explicitly sized deep-walk below.
        return EVAC_LEAF;
    }

    // Explicit residence admits value graphs and ordinary immutable closures,
    // not handles whose runtime-side aliases cannot be rewritten by this walk.
    if (st.abort_on_refusal &&
        !(sub == HEAP_SUBTYPE_CONS || sub == HEAP_SUBTYPE_VECTOR || sub == HEAP_SUBTYPE_RECORD ||
          sub == HEAP_SUBTYPE_MULTI_VALUE || sub == HEAP_SUBTYPE_TENSOR || sub == HEAP_SUBTYPE_EXCEPTION ||
          sub == HEAP_SUBTYPE_RATIONAL || sub == HEAP_SUBTYPE_TAYLOR || sub == HEAP_SUBTYPE_STRING ||
          sub == HEAP_SUBTYPE_SYMBOL || sub == HEAP_SUBTYPE_BIGNUM || sub == HEAP_SUBTYPE_BYTEVECTOR ||
          sub == HEAP_SUBTYPE_I128)) {
        st.refuse();
        eshkol_warn("resident evacuation refuses heap subtype %u", (unsigned)sub);
    }
    switch ((heap_subtype_t)sub) {
        case HEAP_SUBTYPE_CONS:        return EVAC_CONS;
        case HEAP_SUBTYPE_VECTOR:      return EVAC_VECTOR;   // records are vectors too
        case HEAP_SUBTYPE_RECORD:      return EVAC_VECTOR;   // records allocate as vectors;
                                                             // belt-and-suspenders so a future
                                                             // record allocator stamping subtype 7
                                                             // cannot silently regress to leaf.
        case HEAP_SUBTYPE_MULTI_VALUE: return EVAC_MULTIVALUE;
        case HEAP_SUBTYPE_HASH:        return EVAC_HASH;
        case HEAP_SUBTYPE_TENSOR:      return EVAC_TENSOR;
        case HEAP_SUBTYPE_EXCEPTION:   return EVAC_EXCEPTION;
        // ESH-0214d: neuro-symbolic / workspace subtypes now deep-walked. Each
        // carries interior tagged values and/or raw arena buffers that a shallow
        // leaf copy left dangling into the popped region arena.
        case HEAP_SUBTYPE_SUBSTITUTION:   return EVAC_SUBSTITUTION;
        case HEAP_SUBTYPE_FACT:           return EVAC_FACT;
        case HEAP_SUBTYPE_KNOWLEDGE_BASE: return EVAC_KNOWLEDGE_BASE;
        case HEAP_SUBTYPE_FACTOR_GRAPH:   return EVAC_FACTOR_GRAPH;
        case HEAP_SUBTYPE_WORKSPACE:      return EVAC_WORKSPACE;
        case HEAP_SUBTYPE_PROMISE:        return EVAC_PROMISE;
        // RATIONAL: the int64 fast path (is_big == 0) is fully self-contained
        // (numerator/denominator are inline scalars), but the bignum path
        // (is_big == 1) carries two RAW eshkol_bignum_t* pointers (big_num /
        // big_den) that can themselves be region-resident. A shallow leaf copy
        // of a big rational left those two pointers aimed into the popped
        // region arena -- the same class of gap ESH-0214d closed for the
        // logic/workspace subtypes. Always dispatched through EVAC_RATIONAL;
        // the is_big==0 case is a cheap no-op there (nothing to walk).
        case HEAP_SUBTYPE_RATIONAL:       return EVAC_RATIONAL;
        // SW-66 (2026-08-26 libclang architecture-verify admission): only the
        // EXACT-COEFFICIENT (COEFF_RATIONAL) representation of a Taylor tower
        // carries interior region pointers -- its c[] is reinterpreted as an
        // eshkol_tagged_value_t array, and any entry can be a HEAP_PTR to a
        // bignum/rational coefficient allocated in the same (possibly dying)
        // region (see the exact-tower path in runtime_taylor.c). A COEFF_F64
        // tower's c[] is raw doubles. Always dispatched through EVAC_TAYLOR;
        // the F64 case is a cheap no-op there, mirroring EVAC_RATIONAL's
        // is_big==0 fast path just above.
        case HEAP_SUBTYPE_TAYLOR:         return EVAC_TAYLOR;
        case HEAP_SUBTYPE_PARAMETER:      return EVAC_PARAMETER;
        // ── EVAC_LEAF, one subtype at a time ────────────────────────────
        //
        // THERE IS NO `default:` HERE, AND THAT IS THE POINT. A default in
        // this switch would mean "any heap subtype I have not thought about
        // is self-contained" — which is the ESH-0214d bug stated as a policy.
        // KB / FACT / SUBSTITUTION / WORKSPACE each carried interior tagged
        // values, each fell to that default, and each left pointers aimed into
        // an arena that had already been reclaimed. Listing every member means
        // a NEW subtype cannot inherit that answer by omission: adding one to
        // heap_subtype_t without deciding its evacuation is a compile error
        // (this file is built with -Werror=switch-enum; see
        // eshkol_require_exhaustive_dispatch in CMakeLists.txt).
        //
        // STRING / SYMBOL / BIGNUM / BYTEVECTOR / I128: self-contained payloads
        // -> a contiguous leaf copy fully preserves them. I128 in particular is
        // a flat 16-byte {lo,hi} POD with no interior pointers, so the leaf
        // path (a straight memcpy of header+payload) is correct and complete —
        // no deep walk is needed (confirmed against ESH-0214d).
        case HEAP_SUBTYPE_STRING:
        case HEAP_SUBTYPE_SYMBOL:
        case HEAP_SUBTYPE_BIGNUM:
        case HEAP_SUBTYPE_BYTEVECTOR:
        case HEAP_SUBTYPE_I128:
        // Deliberately kept EVAC_LEAF (no interior region pointers, or their
        // interior graph is not confidently/safely traversable here, AND they
        // are not observed to escape a region by mutation):
        //   PORT      - wraps an OS fd/FILE*; handle intentionally shared, not copied.
        //   PRNG      - self-contained state words, no interior pointers.
        //   PARAMETER is not listed here: its converter and dynamic-binding
        //               stack are walked by EVAC_PARAMETER below.
        //   DNC/SDNC  - VERIFIED SW-66 by reading both handle layouts:
        //               DncHandle.{mem,usage} (lib/core/dnc_api.c) and
        //               SdncHandle.w (lib/core/sdnc_api.c) are calloc'd on
        //               the plain C heap, never arena-allocated
        //               (SdncHandle.pe[][] is inline scalar data with no
        //               pointer at all). A shallow copy preserves those raw
        //               pointer VALUES exactly; they cannot dangle when the
        //               region's arena is freed, because they never pointed
        //               into it in the first place. This is a confirmed
        //               property of the current handle layouts, not an
        //               assumption -- see the debug guard just below.
        //   TAYLOR is NOT in this leaf list -- see its explicit EVAC_TAYLOR
        //   case above (SW-66's exact-coefficient deep-walk fix).
        // A guarded assert below (region_evacuate leaf handler) fires under a debug
        // build if any such subtype is ever seen with a real arena pointer or
        // actually escaping, so a real regression is discovered instead of
        // silently corrupting.
        case HEAP_SUBTYPE_PORT:
        case HEAP_SUBTYPE_PRNG:
        case HEAP_SUBTYPE_DNC:
        case HEAP_SUBTYPE_SDNC:
            return EVAC_LEAF;
    }

    // Unreachable for any DECLARED subtype: the switch above names every
    // member of heap_subtype_t and the compiler enforces that. This arm exists
    // for the value that is not a declared subtype at all — a header byte read
    // out of a corrupted or foreign object. Leaf-copying it would be the same
    // silent shallow copy this switch exists to prevent, so it is reported.
    st.refuse();
    eshkol_warn("region evacuate: undeclared heap subtype %u; leaf-copying, "
                "interior pointers (if any) will dangle", (unsigned)sub);
    return EVAC_LEAF;
}

// Copy a headerless raw buffer (closure env, hash arrays, tensor buffers,
// C strings) into the target arena, with forwarding so shared/aliased buffers
// are copied once.
static void* evac_raw(EvacState& st, const void* old, size_t size) {
    if (!old || size == 0 || st.failed) return (void*)old;
    if (void* fwd = st.lookup(old)) return fwd;
    void* raw = eshkol_alloc_failpoint_fire(ESHKOL_ALLOC_FAILPOINT_EVAC_COPY)
                    ? nullptr : arena_allocate_aligned(st.target, size, 16);
    if (!raw) {
        // #713: the transaction is now doomed; the caller aborts it. The old
        // pointer is returned only so the rest of the walk is the identity --
        // it is never published.
        st.fail(size);
        return (void*)old;
    }
    std::memcpy(raw, old, size);
    st.record(old, raw);
    st.copies++;
    return raw;
}

// Copy a header-prefixed object into the target arena (contiguous header+payload),
// register forwarding, and enqueue it for interior traversal if non-leaf.
static void* evac_object(EvacState& st, void* old_data, const eshkol_tagged_value_t& v) {
    if (st.failed) return old_data;
    if (void* fwd = st.lookup(old_data)) return fwd;

    auto* h = (eshkol_object_header_t*)((uint8_t*)old_data - sizeof(eshkol_object_header_t));
    const size_t total = sizeof(eshkol_object_header_t) + h->size;
    if (h->size == 0) { if (st.abort_on_refusal) st.refuse(); return old_data; }
    // Plausibility guard: every live producer of tagged heap values allocates
    // through the *_with_header paths, but a handful of legacy helpers
    // (arena_allocate_tagged_cons_cell, arena_create_*_cons) emit HEADERLESS
    // cells; if one ever leaked into a traversed graph, the 8 bytes before it
    // are arbitrary. A garbage size would make the memcpy below read/allocate
    // wildly; cap it at the region's own footprint upper bound. Skipping the
    // copy degrades to the pre-ESH-0214c shallow behavior for that node only.
    if (h->size > (uint32_t)0x10000000u) {  // 256MB: far above any real object
        st.refuse();
        eshkol_warn("region evacuate: implausible object size %u at %p; "
                    "leaving in place (headerless allocation?)",
                    (unsigned)h->size, old_data);
        return old_data;
    }

    void* raw = eshkol_alloc_failpoint_fire(ESHKOL_ALLOC_FAILPOINT_EVAC_COPY)
                    ? nullptr : arena_allocate_aligned(st.target, total, 16);
    if (!raw) {
        st.fail(total);   // #713: abort the whole transaction, see EvacState
        return old_data;
    }
    std::memcpy(raw, h, total);
    void* new_data = (uint8_t*)raw + sizeof(eshkol_object_header_t);
    st.record(old_data, new_data);
    st.copies++;

    EvacKind k = evac_kind_for(st, v, old_data);
#ifndef NDEBUG
    // SW-66 (formerly the ESH-0214d watchlist): DNC/SDNC are VERIFIED leaf --
    // their only pointer-shaped fields are calloc'd C-heap buffers, never
    // arena-resident (see the case-arm comment in evac_kind_for above) -- but
    // that verification is a property of TODAY's handle layout, not a law.
    // If a future change ever makes DncHandle/SdncHandle hold a real arena
    // pointer without updating evac_kind_for to match, this guard is what
    // catches it: warn loudly instead of silently corrupting interior state.
    // TAYLOR is NOT listed here any more -- it is no longer ever classified
    // EVAC_LEAF (see EVAC_TAYLOR above), so this branch is structurally
    // unreachable for it now.
    if (k == EVAC_LEAF) {
        const auto* dh = (const eshkol_object_header_t*)
            ((const uint8_t*)new_data - sizeof(eshkol_object_header_t));
        // No default: the watchlist is a decision about every subtype, so a
        // NEW subtype must be placed on it or off it explicitly. A default
        // would put every future subtype silently in the "not watched" half —
        // which is how a subtype gets shallow-copied for a release without
        // anyone hearing about it.
        switch ((heap_subtype_t)dh->subtype) {
            case HEAP_SUBTYPE_DNC:
            case HEAP_SUBTYPE_SDNC:
                st.refuse();
                eshkol_warn("region evacuate: subtype %u escaped a region as a "
                            "SHALLOW leaf copy (SW-66 verified-leaf guard); if "
                            "this fires, the handle's layout changed and this "
                            "subtype needs a real evac_kind_for case.",
                            (unsigned)dh->subtype);
                break;
            // Not watched: self-contained payloads, or objects whose interior
            // is reached by a path other than the evacuator (PORT wraps an fd,
            // PARAMETER is read through the dynamic environment).
            case HEAP_SUBTYPE_STRING:
            case HEAP_SUBTYPE_SYMBOL:
            case HEAP_SUBTYPE_BIGNUM:
            case HEAP_SUBTYPE_BYTEVECTOR:
            case HEAP_SUBTYPE_I128:
            case HEAP_SUBTYPE_PORT:
            case HEAP_SUBTYPE_PRNG:
            case HEAP_SUBTYPE_PARAMETER:
            // Deep-walked above, so reaching the leaf handler at all would mean
            // evac_kind_for and this watchlist disagree; nothing to warn about
            // here, because k == EVAC_LEAF already excluded them.
            case HEAP_SUBTYPE_CONS:
            case HEAP_SUBTYPE_VECTOR:
            case HEAP_SUBTYPE_RECORD:
            case HEAP_SUBTYPE_MULTI_VALUE:
            case HEAP_SUBTYPE_HASH:
            case HEAP_SUBTYPE_TENSOR:
            case HEAP_SUBTYPE_EXCEPTION:
            case HEAP_SUBTYPE_SUBSTITUTION:
            case HEAP_SUBTYPE_FACT:
            case HEAP_SUBTYPE_KNOWLEDGE_BASE:
            case HEAP_SUBTYPE_FACTOR_GRAPH:
            case HEAP_SUBTYPE_WORKSPACE:
            case HEAP_SUBTYPE_PROMISE:
            case HEAP_SUBTYPE_RATIONAL:
            case HEAP_SUBTYPE_TAYLOR:
                break;
        }
    }
#endif
    if (k != EVAC_LEAF) st.worklist.push_back({new_data, k});
    return new_data;
}

// Rewrite one tagged value: if it points into a dying region (strictly inner
// than the boundary), evacuate the pointed-to object and repoint; otherwise
// leave it untouched.
static eshkol_tagged_value_t evac_value(EvacState& st, eshkol_tagged_value_t v) {
    if (st.failed) return v;   // #713: an aborted transaction rewrites nothing
    const uint8_t type = v.type;
    if (!region_value_carries_pointer(type)) return v;

    void* p = (void*)(uintptr_t)v.data.ptr_val;
    if (!p) return v;
    bool copy_literal = false;
    if (st.resident_copy_mutable && type == ESHKOL_VALUE_HEAP_PTR &&
        !(st.resident_preserved_arena && arena_contains(st.resident_preserved_arena, p)) &&
        !(st.resident_destination_mark && arena_scope_span_contains(st.target, st.resident_destination_mark, p))) {
        // Mutable language literals can occupy LLVM read-only globals rather
        // than any arena. Imports/observations still need independent storage.
        // Existing owner nodes remain shared during a tick; compaction retains
        // its ordinary ownership predicate and moves all declared roots once.
        const uint8_t sub = ESHKOL_GET_SUBTYPE(p);
        copy_literal = sub == HEAP_SUBTYPE_CONS || sub == HEAP_SUBTYPE_VECTOR ||
            sub == HEAP_SUBTYPE_RECORD || sub == HEAP_SUBTYPE_TENSOR ||
            sub == HEAP_SUBTYPE_STRING || sub == HEAP_SUBTYPE_BYTEVECTOR;
    }
    if (!st.owns(p) && !copy_literal) return v;

    // Headerless fixed-size payload (dual number / complex): a flat copy is the
    // whole object. evac_raw forwards, so two tagged values sharing one payload
    // still share it after promotion.
    if (const size_t raw_size = region_headerless_payload_size(type)) {
        // A carrier complex (ADR-0025) is the plain pair plus two tagged
        // components, and those components are pointers (jets, tape nodes), so
        // it is copied at its own size and its components are escaped with it.
        const bool carrier_complex =
            (type & (uint8_t)~(ESHKOL_VALUE_EXACT_FLAG | ESHKOL_VALUE_INEXACT_FLAG)) == ESHKOL_VALUE_COMPLEX &&
            (v.flags & ESHKOL_COMPLEX_CARRIER_FLAG) != 0;
        if (carrier_complex) {
            auto* copy = (eshkol_complex_carrier_t*)evac_raw(
                st, p, eshkol_ad_payload_size(ESHKOL_AD_PAYLOAD_COMPLEX_CARRIER));
            if (copy) {
                copy->real = evac_value(st, copy->real);
                copy->imag = evac_value(st, copy->imag);
            }
            v.data.ptr_val = (uint64_t)(uintptr_t)copy;
            return v;
        }
        v.data.ptr_val = (uint64_t)(uintptr_t)evac_raw(st, p, raw_size);
        return v;
    }

    void* np = evac_object(st, p, v);
    v.data.ptr_val = (uint64_t)(uintptr_t)np;
    return v;
}

static void evac_parameter_value(eshkol_tagged_value_t* value, void* context) {
    if (!value || !context) return;
    EvacState* st = (EvacState*)context;
    // The binding stack is malloc-owned and shared with the original
    // parameter object, so it is not one of this transaction's copies.
    const eshkol_tagged_value_t next = evac_value(*st, *value);
    if (std::memcmp(&next, value, sizeof(next)) != 0) {
        st->preserve(value, sizeof(*value));
        *value = next;
    }
}

// Evacuate a header-prefixed object referenced only by a RAW data pointer (no
// enclosing tagged value), e.g. the fact pointers held in a knowledge base's
// facts[] array. Synthesizes a plain HEAP_PTR tagged value so evac_object can
// classify it by its object header and drive the normal deep walk (forwarding,
// cycles, shared structure all preserved). Objects already at/outside the
// boundary are returned unchanged.
static void* evac_object_ptr(EvacState& st, void* data_ptr) {
    if (!data_ptr || st.failed) return data_ptr;
    if (!st.owns(data_ptr)) return data_ptr;
    eshkol_tagged_value_t synth;
    std::memset(&synth, 0, sizeof(synth));
    synth.type = ESHKOL_VALUE_HEAP_PTR;
    synth.data.ptr_val = (uint64_t)(uintptr_t)data_ptr;
    return evac_object(st, data_ptr, synth);
}

static void evac_drain(EvacState& st);

// ───────────────────────────────────────────────────────────────────────────
// #713: THE PROMOTION TRANSACTION
//
// Every driver (write barrier, with-region result escape, region unwind, nursery
// recycle, arena-span loop reclamation) promotes through evac_transaction(). It
// evacuates a batch of roots into one destination with one forwarding relation,
// and it is all-or-nothing (ADR-0001, "All-or-nothing promotion"):
//
//   COMMIT  every root and everything it reaches that must move has a complete
//           copy in the destination; the roots are rewritten; the forwarding
//           entries become part of the region's persistent relation.
//   ABORT   an allocation failed (a bounded destination at capacity, the OS
//           refusing a new arena block, or the evacuator's own bookkeeping
//           failing to grow). The roots are left exactly as passed in, every
//           shared byte the walk rewrote is restored, the forwarding keys this
//           transaction added are removed, and the destination's bump pointer is
//           rewound when this thread may rewind it. Nothing is published.
//
// The caller decides what an abort means. A region-mode caller raises a
// catchable allocation error (eshkol_raise_allocation_failure) -- after this
// function has returned, so no C++ frame is ever unwound by longjmp. The
// arena-span caller treats it as a refusal and retains the span.
// ───────────────────────────────────────────────────────────────────────────

namespace {

// Promotion failures are reported by allocation size so the raised condition
// names what could not be allocated. Thread-local: set by the transaction that
// aborted, read by the raise that follows it on the same thread.
static thread_local size_t t_promotion_failed_bytes = 0;

} // namespace

/* Run one transaction over @p n roots in place. @p st must be configured
 * (target, ownership predicate, forwarding map) and otherwise empty. */
namespace eshkol::regions_internal {
EvacOutcome evac_transaction(EvacState& st, eshkol_tagged_value_t* roots,
                             uint64_t n) {
    EvacOutcome out;
    arena_t* target = st.target;

    // Mark the destination so an abort can rewind it. arena_push_scope is a
    // no-op where scope operations are commit-only (a concurrent pool worker
    // on the shared arena) and when its own record cannot be allocated; in
    // both cases no mark is taken and an abort leaves the aborted copies as
    // unreachable bytes in the destination -- never published, never reused.
    arena_scope_t* const before = target->current_scope;
    arena_push_scope(target);
    arena_scope_t* const mark =
        (target->current_scope != before) ? target->current_scope : nullptr;
    if (st.abort_on_refusal) {
        if (!mark) { out.ok = false; return out; }
        st.resident_destination_mark = mark;
    }

    // Results are staged so the roots themselves change only on commit.
    eshkol_tagged_value_t one;
    eshkol_tagged_value_t* staged = &one;
    std::vector<eshkol_tagged_value_t> many;
    try {
        if (n > 1) {
            many.resize((size_t)n);
            staged = many.data();
        }
        for (uint64_t i = 0; i < n && !st.failed; ++i) staged[i] = evac_value(st, roots[i]);
        evac_drain(st);
    } catch (const std::bad_alloc&) {
        // The worklist, the forwarding map or a preserve record could not grow.
        st.fail(0);
    }

    if (st.abort_on_refusal && st.refused) st.fail(0);
    if (!st.failed) {
        if (mark) arena_commit_scope(target);
        for (uint64_t i = 0; i < n; ++i) roots[i] = staged[i];
        return out;
    }

    // ABORT, newest first so overlapping saved ranges restore to the oldest
    // contents.
    for (size_t i = st.saved.size(); i-- > 0;) {
        const EvacState::SavedRange& r = st.saved[i];
        std::memcpy(r.addr, st.saved_bytes.data() + r.offset, r.size);
    }
    for (const void* key : st.inserted) st.fwd->erase(key);
    if (mark) arena_pop_scope(target);
    out.ok = false;
    out.failed_bytes = st.failed_bytes;
    return out;
}
} // namespace eshkol::regions_internal

// Deep-evacuate @p n roots in place into @p target, copying everything reachable
// that lives in an active region strictly inner than @p boundary_idx. Returns
// false -- with the roots, the region and its forwarding relation unchanged --
// when the promotion could not be completed; the caller must then raise rather
// than store anything.
static bool region_evacuate_values(eshkol_tagged_value_t* roots, uint64_t n,
                                   arena_t* target, int boundary_idx) {
    if (!target || n == 0) return true;
    EvacState st;
    st.target = target;
    st.boundary_idx = boundary_idx;

    // Use the innermost active region's PERSISTENT forwarding map so shared
    // structure is preserved across SEPARATE escapes within one region body
    // (two vector-set! calls storing lists that share a tail must keep sharing
    // it after promotion). The map's keys reference memory in this region or an
    // enclosing one -- the innermost region is the first of those to die, and
    // region_destroy frees the map with it, so no entry can ever go stale. The
    // map is only valid for one promotion target; when this escape targets a
    // different arena (rare: barrier into an intermediate region) it runs
    // against a fresh relation that REPLACES the old one on commit, so an
    // aborted retarget leaves the old relation intact.
    EvacFwdMap local_fwd;
    eshkol_region_t* owner = region_current();
    EvacFwdMap* persistent = nullptr;
    bool retarget = false;
    if (owner) {
        if (!owner->fwd_map) {
            owner->fwd_map = new (std::nothrow) EvacFwdMap();
            owner->fwd_target = target;
        }
        persistent = (EvacFwdMap*)owner->fwd_map;
        retarget = persistent && owner->fwd_target != target;
    }
    st.fwd = (persistent && !retarget) ? persistent : &local_fwd;

    const EvacOutcome res = evac_transaction(st, roots, n);
    if (!res.ok) {
        t_promotion_failed_bytes = res.failed_bytes;
        return false;
    }
    if (retarget) {
        persistent->swap(local_fwd);
        owner->fwd_target = target;
    }
    if (owner) owner->escape_count += st.copies;
    return true;
}

// SW-164: the breadth-first interior walk, shared by every driver. Split out of
// region_evacuate_value so the arena-span driver below runs the IDENTICAL
// subtype coverage rather than a second copy of it that could drift — the
// ESH-0214d lesson (a subtype dropped to a leaf in one evacuator and not the
// other) applies with double force once there are two entry points.
static void evac_drain(EvacState& st) {
    while (!st.worklist.empty() && !st.failed) {
        std::pair<void*, EvacKind> item = st.worklist.back();
        st.worklist.pop_back();
        void* nd = item.first;
        switch (item.second) {
            case EVAC_CONS: {
                auto* c = (arena_tagged_cons_cell_t*)nd;
                c->car = evac_value(st, c->car);
                c->cdr = evac_value(st, c->cdr);
                break;
            }
            case EVAC_VECTOR: {
                int64_t len = *(int64_t*)nd;
                if (len < 0) len = 0;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(int64_t));
                for (int64_t i = 0; i < len; ++i) elems[i] = evac_value(st, elems[i]);
                break;
            }
            case EVAC_MULTIVALUE: {
                size_t count = *(size_t*)nd;
                auto* elems = (eshkol_tagged_value_t*)((uint8_t*)nd + sizeof(size_t));
                for (size_t i = 0; i < count; ++i) elems[i] = evac_value(st, elems[i]);
                break;
            }
            case EVAC_EXCEPTION: {
                auto* ex = (eshkol_exception_t*)nd;
                if (ex->message && st.owns(ex->message))
                    ex->message = (char*)evac_raw(st, ex->message, std::strlen(ex->message) + 1);
                if (ex->filename && st.owns(ex->filename))
                    ex->filename = (char*)evac_raw(st, ex->filename, std::strlen(ex->filename) + 1);
                if (ex->irritants && ex->num_irritants &&
                    st.owns(ex->irritants)) {
                    ex->irritants = (eshkol_tagged_value_t*)evac_raw(
                        st, ex->irritants, (size_t)ex->num_irritants * sizeof(eshkol_tagged_value_t));
                } else if (ex->irritants && ex->num_irritants) {
                    // Shared with the original exception: rewritten in place.
                    st.preserve(ex->irritants,
                                (size_t)ex->num_irritants * sizeof(eshkol_tagged_value_t));
                }
                if (ex->irritants) {
                    for (uint32_t i = 0; i < ex->num_irritants; ++i)
                        ex->irritants[i] = evac_value(st, ex->irritants[i]);
                }
                break;
            }
            case EVAC_TENSOR: {
                auto* t = (eshkol_tensor_t*)nd;
                if (t->dimensions && (st.resident_copy_mutable || st.owns(t->dimensions)))
                    t->dimensions = (uint64_t*)evac_raw(
                        st, t->dimensions, (size_t)t->num_dimensions * sizeof(uint64_t));
                // A tagged-element carrier (a dual tensor, or a vector
                // promoted by a non-numeric store -- ADR-0020) holds 16-byte
                // tagged values that can point into the dying region, so its
                // buffer is wider and each slot is walked.
                if (eshkol_tensor_dtype_is_tagged(t->dtype)) {
                    if (t->elements && (st.resident_copy_mutable || st.owns(t->elements)))
                        t->elements = (int64_t*)evac_raw(
                            st, t->elements,
                            (size_t)t->total_elements * sizeof(eshkol_tagged_value_t));
                    else if (t->elements)
                        st.preserve(t->elements,
                                    (size_t)t->total_elements * sizeof(eshkol_tagged_value_t));
                    if (t->elements) {
                        auto* slots = (eshkol_tagged_value_t*)t->elements;
                        for (uint64_t i = 0; i < t->total_elements; ++i)
                            slots[i] = evac_value(st, slots[i]);
                    }
                    break;
                }
                if (t->elements && (st.resident_copy_mutable || st.owns(t->elements)))
                    t->elements = (int64_t*)evac_raw(
                        st, t->elements, (size_t)t->total_elements * sizeof(int64_t));
                break;
            }
            case EVAC_HASH: {
                auto* tbl = (eshkol_hash_table_t*)nd;
                const size_t cap = tbl->capacity;
                if (tbl->keys && st.owns(tbl->keys))
                    tbl->keys = (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->keys, cap * sizeof(eshkol_tagged_value_t));
                else if (tbl->keys)
                    st.preserve(tbl->keys, cap * sizeof(eshkol_tagged_value_t));
                if (tbl->values && st.owns(tbl->values))
                    tbl->values = (eshkol_tagged_value_t*)evac_raw(
                        st, tbl->values, cap * sizeof(eshkol_tagged_value_t));
                else if (tbl->values)
                    st.preserve(tbl->values, cap * sizeof(eshkol_tagged_value_t));
                if (tbl->status && st.owns(tbl->status))
                    tbl->status = (uint8_t*)evac_raw(st, tbl->status, cap * sizeof(uint8_t));
                if (tbl->keys && tbl->values && tbl->status) {
                    for (size_t i = 0; i < cap; ++i) {
                        if (tbl->status[i] == HASH_ENTRY_OCCUPIED) {
                            tbl->keys[i]   = evac_value(st, tbl->keys[i]);
                            tbl->values[i] = evac_value(st, tbl->values[i]);
                        }
                    }
                }
                // Future resizes must grow the arrays in the surviving arena,
                // never the dying region.
                tbl->home_arena = st.target;
                break;
            }
            case EVAC_CLOSURE: {
                auto* c = (eshkol_closure_t*)nd;
                if (c->env && st.owns(c->env)) {
                    const size_t ncap = CLOSURE_ENV_GET_NUM_CAPTURES(c->env->num_captures);
                    const size_t env_size =
                        sizeof(eshkol_closure_env_t) + ncap * sizeof(eshkol_tagged_value_t);
                    auto* ne = (eshkol_closure_env_t*)evac_raw(st, c->env, env_size);
                    c->env = ne;
                    for (size_t i = 0; i < ncap; ++i) {
                        eshkol_tagged_value_t& cap = ne->captures[i];
                        // MUTABLE-CAPTURE CELLS: lambda codegen moves a set!-able
                        // captured local into a 16-byte headerless arena cell
                        // (one eshkol_tagged_value_t) and stores the CELL ADDRESS
                        // in the capture slot packed as an exact INT64
                        // (packInt64ToTaggedValue(PtrToInt(cell))) — see
                        // "CLOSURE ESCAPE FIX" / "MUTABLE CAPTURE FIX" in
                        // llvm_codegen.cpp. During a with-region body that cell
                        // lives in the region arena and would dangle after
                        // region_pop. There is no type tag distinguishing this
                        // packed pointer from a genuine integer, so we test
                        // conservatively: an INT64 capture whose value, read as
                        // a pointer, falls INSIDE a dying region arena is treated
                        // as a capture cell — copied (forwarded, so cells shared
                        // between closures stay shared), its contained tagged
                        // value evacuated, and the slot repointed. A genuine
                        // captured integer can only misfire if it exactly equals
                        // a live interior address of the few-KB region arena
                        // active at escape time (Boehm-style conservatism;
                        // astronomically unlikely, documented in the PR). The
                        // other producers of INT64-packed pointers in capture
                        // slots are GlobalVariable addresses and JIT/AOT code
                        // addresses, neither of which is ever inside a region
                        // arena, so they always take the identity path here.
                        if (cap.type == ESHKOL_VALUE_INT64 && cap.data.int_val != 0) {
                            void* cell = (void*)(uintptr_t)cap.data.int_val;
                            if (st.owns(cell)) {
                                auto* nc =
                                    (eshkol_tagged_value_t*)evac_raw(st, cell, 16);
                                if (nc != cell) {
                                    *nc = evac_value(st, *nc);
                                    cap.data.int_val = (int64_t)(intptr_t)nc;
                                }
                                continue;
                            }
                        }
                        cap = evac_value(st, cap);
                    }
                }
                if (c->name && st.owns(c->name))
                    c->name = (const char*)evac_raw(st, c->name, std::strlen(c->name) + 1);
                if (st.abort_on_refusal && c->sexpr_ptr && st.owns((void*)(uintptr_t)c->sexpr_ptr) &&
                    !arena_contains(eshkol_literal_arena(), (void*)(uintptr_t)c->sexpr_ptr)) {
                    eshkol_tagged_value_t source{};
                    source.type = ESHKOL_VALUE_HEAP_PTR;
                    source.data.ptr_val = c->sexpr_ptr;
                    c->sexpr_ptr = evac_value(st, source).data.ptr_val;
                }
                break;
            }
            case EVAC_AD_NODE: {
                // Tape-retained nodes normally live in the tape owner's arena,
                // but a headered standalone node can still cross a region
                // boundary. Walk graph links and payloads whose sizes are
                // described by the node itself; never guess a saved payload's
                // size from an operation-specific constant.
                auto* n = (ad_node_t*)nd;
                if (n->exact_value) {
                    if (region_index_owning(n->exact_value) > st.boundary_idx)
                        n->exact_value = (eshkol_tagged_value_t*)evac_raw(
                            st, n->exact_value, sizeof(eshkol_tagged_value_t));
                    else
                        st.preserve(n->exact_value, sizeof(eshkol_tagged_value_t));
                    /* The sidecar slot may live in the outer/tape arena while
                     * its tagged bignum/rational payload lives in the region
                     * being popped. Walk the payload regardless of whether the
                     * slot itself needed rebasing. */
                    *n->exact_value = evac_value(st, *n->exact_value);
                }
                if (n->exact_gradient) {
                    if (region_index_owning(n->exact_gradient) > st.boundary_idx)
                        n->exact_gradient = (eshkol_tagged_value_t*)evac_raw(
                            st, n->exact_gradient, sizeof(eshkol_tagged_value_t));
                    else
                        st.preserve(n->exact_gradient, sizeof(eshkol_tagged_value_t));
                    *n->exact_gradient = evac_value(st, *n->exact_gradient);
                }
                auto evacuate_input = [&](ad_node_t*& input) {
                    if (input) input = (ad_node_t*)evac_object_ptr(st, input);
                };
                evacuate_input(n->input1);
                evacuate_input(n->input2);
                evacuate_input(n->input3);
                evacuate_input(n->input4);

                if (n->shape && n->ndim &&
                    n->ndim <= SIZE_MAX / sizeof(int64_t) &&
                    region_index_owning(n->shape) > st.boundary_idx) {
                    n->shape = (int64_t*)evac_raw(
                        st, n->shape, n->ndim * sizeof(int64_t));
                }

                size_t tensor_count = 1;
                bool valid_shape = n->ndim != 0 && n->shape;
                if (valid_shape) {
                    for (size_t i = 0; i < n->ndim; ++i) {
                        if (n->shape[i] <= 0 ||
                            tensor_count > SIZE_MAX / (size_t)n->shape[i]) {
                            valid_shape = false;
                            break;
                        }
                        tensor_count *= (size_t)n->shape[i];
                    }
                    valid_shape = valid_shape &&
                                  tensor_count <= SIZE_MAX / sizeof(double);
                }
                if (valid_shape) {
                    if (n->tensor_value &&
                        region_index_owning(n->tensor_value) > st.boundary_idx)
                        n->tensor_value = evac_raw(
                            st, n->tensor_value, tensor_count * sizeof(double));
                    if (n->tensor_gradient &&
                        region_index_owning(n->tensor_gradient) > st.boundary_idx)
                        n->tensor_gradient = evac_raw(
                            st, n->tensor_gradient, tensor_count * sizeof(double));
                }
                if (n->saved_tensors && n->num_saved &&
                    n->num_saved <= SIZE_MAX / sizeof(void*) &&
                    region_index_owning(n->saved_tensors) > st.boundary_idx) {
                    // The array length is part of the node layout. Its
                    // producer-specific payloads remain tape-owned; this
                    // copier does not invent a size for opaque void* entries.
                    n->saved_tensors = (void**)evac_raw(
                        st, n->saved_tensors, n->num_saved * sizeof(void*));
                }
                break;
            }
            case EVAC_SUBSTITUTION: {
                // Layout [hdr][struct][var_ids u64[cap]][terms tagged[cap]] — all
                // inline (already copied contiguously). var_ids are plain ints;
                // only the bound terms carry heap pointers. Unused capacity slots
                // beyond num_bindings are uninitialized and must not be walked.
                auto* s = (eshkol_substitution_t*)nd;
                eshkol_tagged_value_t* terms = SUBST_TERMS(s);
                for (uint32_t i = 0; i < s->num_bindings; ++i)
                    terms[i] = evac_value(st, terms[i]);
                break;
            }
            case EVAC_FACT: {
                // Layout [hdr][struct][args tagged[arity]] — args inline.
                auto* f = (eshkol_fact_t*)nd;
                eshkol_tagged_value_t* args = FACT_ARGS(f);
                for (uint32_t i = 0; i < f->arity; ++i)
                    args[i] = evac_value(st, args[i]);
                // predicate is normally a pointer into the immortal interned
                // predicate pool (static storage -> region_index_owning == -1 ->
                // left in place, preserving pointer-equality matching). Only a
                // non-interned predicate string allocated inside the dying region
                // is copied (defensive; unify() has a string-compare fallback).
                if (f->predicate) {
                    void* pred = (void*)(uintptr_t)f->predicate;
                    if (st.owns(pred))
                        f->predicate = (uint64_t)(uintptr_t)evac_raw(
                            st, pred, std::strlen((const char*)pred) + 1);
                }
                break;
            }
            case EVAC_KNOWLEDGE_BASE: {
                // Layout [hdr][struct]; facts[] is a separate raw arena array of
                // FACT data pointers (capacity slots, num_facts used).
                auto* kb = (eshkol_knowledge_base_t*)nd;
                if (kb->facts && st.owns(kb->facts))
                    kb->facts = (eshkol_fact_t**)evac_raw(
                        st, kb->facts, (size_t)kb->capacity * sizeof(eshkol_fact_t*));
                else if (kb->facts)
                    st.preserve(kb->facts, (size_t)kb->num_facts * sizeof(eshkol_fact_t*));
                if (kb->facts) {
                    for (uint32_t i = 0; i < kb->num_facts; ++i)
                        kb->facts[i] = (eshkol_fact_t*)evac_object_ptr(st, kb->facts[i]);
                }
                break;
            }
            case EVAC_WORKSPACE: {
                // Layout [hdr][struct][modules module_t[max_modules]] — modules
                // inline (already copied). content is a separate raw double buffer;
                // each used module carries an arena name string + process_fn closure.
                auto* ws = (eshkol_workspace_t*)nd;
                if (ws->content && st.owns(ws->content))
                    ws->content = (double*)evac_raw(
                        st, ws->content, (size_t)ws->dim * sizeof(double));
                eshkol_workspace_module_t* mods = WS_MODULES(ws);
                for (uint32_t i = 0; i < ws->num_modules; ++i) {
                    if (mods[i].name &&
                        st.owns(mods[i].name))
                        mods[i].name = (char*)evac_raw(
                            st, mods[i].name, std::strlen(mods[i].name) + 1);
                    mods[i].process_fn = evac_value(st, mods[i].process_fn);
                }
                break;
            }
            case EVAC_PROMISE: {
                // [forced:i64 @0][thunk:tagged @8][cached:tagged @24] (40 bytes,
                // llvm_codegen.cpp %make-lazy-promise / make-promise). Both the
                // thunk closure (unforced) and the memoized cached value (forced)
                // can be region-allocated; a promise that escapes its region via
                // delay/force/make-promise must have both slots evacuated or they
                // dangle into the freed arena after region_pop (ESH-0214d gap:
                // subtype 18 was previously leaf-copied). ESH-0214e.
                auto* thunk  = (eshkol_tagged_value_t*)((uint8_t*)nd + 8);
                auto* cached = (eshkol_tagged_value_t*)((uint8_t*)nd + 24);
                *thunk  = evac_value(st, *thunk);
                *cached = evac_value(st, *cached);
                break;
            }
            case EVAC_RATIONAL: {
                // is_big == 0 (fast path): numerator/denominator are inline
                // int64 scalars -- already copied verbatim by the contiguous
                // header+payload copy, nothing further to walk.
                //
                // is_big == 1 (bignum path): big_num/big_den are RAW
                // eshkol_bignum_t* pointers (not tagged values -- same shape
                // as a knowledge base's facts[] array), each an independently
                // header-prefixed HEAP_SUBTYPE_BIGNUM object that can itself
                // be region-resident (e.g. the result of exact arithmetic
                // performed inside a `with-region` body). Left un-walked, a
                // rational whose reduced numerator/denominator overflow
                // int64 -- promoted out of its region via vector-set!/
                // set-car!/etc. -- would carry two pointers straight into the
                // arena region_pop is about to free.
                auto* r = (eshkol_rational_t*)nd;
                if (r->is_big) {
                    if (r->big_num) r->big_num = (eshkol_bignum_t*)evac_object_ptr(st, r->big_num);
                    if (r->big_den) r->big_den = (eshkol_bignum_t*)evac_object_ptr(st, r->big_den);
                }
                break;
            }
            case EVAC_PARAMETER: {
                /* The control block's stack is malloc-owned and therefore
                 * survives region destruction, but every tagged value stored
                 * in it can still point into the dying arena. Walk the
                 * private layout through the runtime-owned visitor. */
                eshkol_parameter_visit_values(nd, evac_parameter_value, &st);
                break;
            }
            case EVAC_TAYLOR: {
                // SW-66: only a COEFF_RATIONAL (exact) tower's c[] holds
                // interior tagged values -- reinterpret it exactly the way
                // runtime_taylor.c's taylor_exact_c()/taylor_exact_c_const()
                // do, and evac_value() each of the order_k+1 entries (most
                // are plain int64 scalars needing no work; any that are
                // HEAP_PTRs to a bignum/rational allocated in this same
                // region get walked and repointed, exactly like a vector's
                // element array). A COEFF_F64 tower's c[] is raw doubles,
                // already fully preserved by the contiguous header+payload
                // copy above -- nothing to do.
                //
                // A dual-epoch tower may have exact value and tangent
                // sidecars after the two raw-double halves. Rebase both from
                // the copied object's layout before walking their tagged
                // entries; the stored exact_c pointer still names the source
                // region and must never be followed here.
                auto* t = (esh_taylor_t*)nd;
                const size_t ncoeff = (size_t)t->order_k + 1;
                if (ESH_TAYLOR_HAS_TANGENT(t->flags) &&
                    ESH_TAYLOR_TANGENT_IS_EXACT(t->flags)) {
                    t->exact_c = (eshkol_tagged_value_t*)(void*)(t->c + 2u * ncoeff);
                    for (size_t i = 0; i < 2u * ncoeff; ++i)
                        t->exact_c[i] = evac_value(st, t->exact_c[i]);
                } else if ((t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_RATIONAL ||
                           (t->flags & ESH_TAYLOR_COEFF_MASK) == ESH_TAYLOR_COEFF_CARRIER) {
                    /* A level carrier (ADR-0027) stores tagged coefficients
                     * the same way; each may itself be a jet, a bignum or
                     * rational, or an enclosing level's carrier. */
                    t->exact_c = (eshkol_tagged_value_t*)(void*)t->c;
                    auto* c = t->exact_c;
                    for (size_t i = 0; i < ncoeff; ++i) c[i] = evac_value(st, c[i]);
                }
                break;
            }
            case EVAC_FACTOR_GRAPH: {
                // A factor graph carries NO interior tagged values — only nested
                // raw numeric buffers (arrays and arrays-of-arrays). Sizes are
                // reconstructed from the graph's own counts, mirroring the
                // allocation layout in inference.cpp.
                auto* fg = (eshkol_factor_graph_t*)nd;
                const uint32_t nvars = fg->num_vars;
                if (fg->var_dims && st.owns(fg->var_dims))
                    fg->var_dims = (uint32_t*)evac_raw(
                        st, fg->var_dims, (size_t)nvars * sizeof(uint32_t));
                if (fg->beliefs && st.owns(fg->beliefs))
                    fg->beliefs = (double**)evac_raw(
                        st, fg->beliefs, (size_t)nvars * sizeof(double*));
                else if (fg->beliefs)
                    st.preserve(fg->beliefs, (size_t)nvars * sizeof(double*));
                if (fg->beliefs && fg->var_dims) {
                    for (uint32_t i = 0; i < nvars; ++i) {
                        if (fg->beliefs[i] &&
                            st.owns(fg->beliefs[i]))
                            fg->beliefs[i] = (double*)evac_raw(
                                st, fg->beliefs[i], (size_t)fg->var_dims[i] * sizeof(double));
                    }
                }
                if (fg->observed && st.owns(fg->observed))
                    fg->observed = (bool*)evac_raw(
                        st, fg->observed, (size_t)nvars * sizeof(bool));
                // factors[] holds max_factors ptrs (num_factors used). Each factor
                // is a HEADERLESS raw buffer [eshkol_factor_t][var_indices...] with
                // interior cpt/dims raw buffers.
                if (fg->factors && st.owns(fg->factors))
                    fg->factors = (eshkol_factor_t**)evac_raw(
                        st, fg->factors, (size_t)fg->max_factors * sizeof(eshkol_factor_t*));
                else if (fg->factors)
                    st.preserve(fg->factors, (size_t)fg->num_factors * sizeof(eshkol_factor_t*));
                if (fg->factors) {
                    for (uint32_t fi = 0; fi < fg->num_factors; ++fi) {
                        eshkol_factor_t* f = fg->factors[fi];
                        if (!f) continue;
                        if (st.owns(f)) {
                            const size_t fsz = sizeof(eshkol_factor_t) +
                                               (size_t)f->num_vars * sizeof(uint32_t);
                            f = (eshkol_factor_t*)evac_raw(st, f, fsz);
                            fg->factors[fi] = f;
                        } else {
                            st.preserve(f, sizeof(eshkol_factor_t));
                        }
                        if (f->cpt && st.owns(f->cpt))
                            f->cpt = (double*)evac_raw(
                                st, f->cpt, (size_t)f->cpt_size * sizeof(double));
                        if (f->dims && st.owns(f->dims))
                            f->dims = (uint32_t*)evac_raw(
                                st, f->dims, (size_t)f->num_vars * sizeof(uint32_t));
                    }
                }
                // msg_fv / msg_vf: parallel arrays of total_messages double* (NULL
                // until eshkol_fg_infer allocates them). Edge k -> var_dims[var]
                // doubles, where the (factor, var-slot) -> edge order mirrors
                // ensure_messages() in inference.cpp (intentional coupling).
                auto evac_msg_array = [&](double**& arr) {
                    if (!arr) return;
                    if (st.owns(arr))
                        arr = (double**)evac_raw(
                            st, arr, (size_t)fg->total_messages * sizeof(double*));
                    else
                        st.preserve(arr, (size_t)fg->total_messages * sizeof(double*));
                    uint32_t k = 0;
                    for (uint32_t fi = 0;
                         fi < fg->num_factors && k < fg->total_messages; ++fi) {
                        eshkol_factor_t* f = fg->factors ? fg->factors[fi] : nullptr;
                        if (!f) continue;
                        const uint32_t* vidx = FACTOR_VAR_INDICES(f);
                        for (uint32_t vi = 0;
                             vi < f->num_vars && k < fg->total_messages; ++vi, ++k) {
                            const uint32_t var_id = vidx[vi];
                            const uint32_t dim =
                                (fg->var_dims && var_id < nvars) ? fg->var_dims[var_id] : 0;
                            if (arr[k] && dim &&
                                st.owns(arr[k]))
                                arr[k] = (double*)evac_raw(
                                    st, arr[k], (size_t)dim * sizeof(double));
                        }
                    }
                };
                evac_msg_array(fg->msg_fv);
                evac_msg_array(fg->msg_vf);
                break;
            }
            case EVAC_LEAF:
                // Never enqueued (region_evacuate only pushes non-leaf kinds),
                // but named rather than defaulted: this switch has no default,
                // so a new EvacKind cannot be added without deciding here how
                // its interior is walked. A default would let one be added and
                // silently walk nothing — which, for a kind added precisely
                // because it HAS interior pointers, is the ESH-0214d bug back
                // at the other end of the same pipeline.
                break;
        }
    }
}

/* SW-164: drive a deep evacuation out of an ARENA SPAN — everything @p val
 * reaches that @p span_arena allocated after @p span_scope's mark is copied
 * into @p target, and @p val is rewritten to the surviving copy.
 *
 * This is the region evacuator with its ownership question rephrased (see
 * EvacState::owns): same Cheney copy, same forwarding map, same per-subtype
 * interior walk. The forwarding map is LOCAL to the call rather than the
 * innermost region's persistent one, because the span being evacuated is not a
 * region and has no place to hang a map — and because each of these calls is
 * self-contained: the caller stages a value out, rewinds, and stages it back,
 * so nothing shared needs to survive between calls. Sharing WITHIN one call is
 * preserved exactly as it is for a region escape. */
static eshkol_tagged_value_t arena_span_evacuate_value(eshkol_tagged_value_t val,
                                                       arena_t* target,
                                                       const arena_t* span_arena,
                                                       const arena_scope_t* span_scope,
                                                       EvacFwdMap* fwd,
                                                       bool* refused,
                                                       bool* failed) {
    if (!target || !span_arena) return val;
    EvacState st;
    st.target = target;
    st.boundary_idx = 0;          // unused in span mode
    st.span_arena = span_arena;
    st.span_scope = span_scope;
    st.fwd = fwd;
    st.refused = false;
    // #713: an aborted promotion is a refusal. The value comes back unchanged,
    // so the caller retains the span rather than rewinding it -- reclamation
    // is given up, correctness is not.
    const EvacOutcome res = evac_transaction(st, &val, 1);
    if (!res.ok) t_promotion_failed_bytes = res.failed_bytes;
    if (refused) *refused = st.refused || !res.ok;
    if (failed && !res.ok) *failed = true;
    return val;
}

/**
 * @brief SW-164: evacuate @p n tagged values out of an arena span, in place.
 *
 * The entry point the per-iteration loop scope uses (eshkol_arena_iter_scope_end,
 * lib/core/runtime_arena_core.cpp). Everything vals[0..n) reaches that lives
 * above @p span_scope's mark on @p span_arena is deep-copied into @p target and
 * the values are rewritten to the copies.
 *
 * All @p n values share ONE forwarding map, so structure shared between two
 * loop-carried values stays shared after the move — a loop carrying both a list
 * and a pointer into its tail must not silently acquire two copies of the tail.
 *
 * @param vals        Values to rewrite in place.
 * @param n           How many.
 * @param target      Arena the survivors are copied into.
 * @param span_arena  Arena whose span is being reclaimed.
 * @param span_scope  Mark: memory allocated after it is what gets copied.
 *                    NULL means the whole of @p span_arena.
 */
/* SW-164: may this value be MOVED out of a span that is about to be rewound?
 *
 * The evacuator can copy an arbitrary object graph, but copying is only half of
 * moving: every OTHER reference to the original has to be rewritten too, and
 * the evacuator can only rewrite the references it can reach from the roots it
 * was given. The codegen-side analysis (loopBodyIterScopeSafe) proves a loop
 * body cannot store a value into pre-existing Eshkol structure, which rules out
 * one class of hidden reference — but it says nothing about the C runtime's own
 * roots. The AD tape is the clearest example: its nodes[] array holds pointers
 * the evacuator neither sees nor rewrites, so moving a node leaves the recorded
 * evaluation order aimed at reclaimed memory and returns a plausible wrong
 * gradient rather than an error.
 *
 * Rather than enumerate every runtime-side root — a list that would silently go
 * stale the first time one was added — this admits only values that CANNOT be
 * held that way: immediates, and the exact numeric tower's heap payloads
 * (bignum, rational). Those are pure values. Nothing in the runtime retains a
 * pointer to a freshly computed intermediate number, they are immutable, they
 * carry no observable identity (R7RS does not promise `eq?` on two exact
 * rationals of equal value), and a rational's only interior pointers are to its
 * own two bignums, which travel with it.
 *
 * That is exactly the case this reclamation exists for — an exact accumulator
 * rebuilt every iteration — and everything else keeps the previous behavior:
 * the span is retained, not rewound. The trade is deliberate. Memory is what a
 * conservative answer costs here; a wrong answer is what an optimistic one
 * costs, and only one of those is recoverable. */
static bool span_movable_value(const eshkol_tagged_value_t& v) {
    const uint8_t t = v.type;
    /* Provably pointer-free immediates: null / int64 / double / bool / char,
     * plus the pointer-free eof-object. */
    if (t <= ESHKOL_VALUE_CHAR || t == 0xFF) return true;

    /* Ports OR their flag bits into the tag; they are never movable. */
    const bool is_port = ((t & ESHKOL_PORT_ANY_FLAG) != 0) &&
                         ((t & ESHKOL_VALUE_HEAP_PTR) == ESHKOL_VALUE_HEAP_PTR);
    if (is_port) return false;

    const uint8_t base = (uint8_t)(t & (uint8_t)~(ESHKOL_VALUE_EXACT_FLAG |
                                                  ESHKOL_VALUE_INEXACT_FLAG));
    if (base != ESHKOL_VALUE_HEAP_PTR) return false;

    void* p = (void*)(uintptr_t)v.data.ptr_val;
    if (!p) return true;
    const uint8_t sub = ESHKOL_GET_SUBTYPE(p);
    return sub == HEAP_SUBTYPE_BIGNUM || sub == HEAP_SUBTYPE_RATIONAL;
}

extern "C" int eshkol_arena_span_evacuate(eshkol_tagged_value_t* vals, uint64_t n,
                                          arena_t* target,
                                          const arena_t* span_arena,
                                          const arena_scope_t* span_scope) {
    if (!vals || n == 0 || !target || !span_arena) return 1;

    // A tape recording ANYWHERE on this thread is a blanket refusal, checked
    // before a byte is copied. The per-object refusal below catches a tape node
    // the values themselves reach; this catches the ones they do not — a node
    // the tape holds but nothing in vals[] references is invisible to a
    // reachability walk, and rewinding the span would reclaim it just the same.
    if (__ad_tape_depth != 0 || __current_ad_tape != nullptr) return 0;

    // Every out-value must be of a kind that can be moved at all (see above).
    for (uint64_t i = 0; i < n; ++i) {
        if (!span_movable_value(vals[i])) return 0;
    }

    EvacFwdMap fwd;
    bool refused = false;
    for (uint64_t i = 0; i < n; ++i) {
        bool one = false;
        bool failed = false;
        vals[i] = arena_span_evacuate_value(vals[i], target, span_arena,
                                            span_scope, &fwd, &one, &failed);
        // #713: an allocation failure leaves vals[i] as it was; earlier values
        // are complete copies. -1 lets a caller past its point of no return
        // tell "could not allocate" from "must not move".
        if (failed) return -1;
        if (one) refused = true;
    }
    return refused ? 0 : 1;
}

/* #713: raise the catchable allocation error for a promotion that aborted.
 * Called only after the aborted transaction has returned, so the longjmp inside
 * eshkol_raise never unwinds a C++ frame. The condition object is preallocated
 * (see eshkol_raise_allocation_failure): building it cannot fail under the same
 * exhaustion that caused the abort. */
namespace eshkol::regions_internal {
void region_raise_promotion_failure(void) {
    eshkol_raise_allocation_failure("region promotion", t_promotion_failed_bytes);
}
} // namespace eshkol::regions_internal

/* The arena a value escaping `current` is promoted into, and the boundary
 * index for that escape: everything in a region strictly inner than
 * current's parent (stack index depth-2) moves. */
namespace eshkol::regions_internal {
bool region_escape_values_in_place(eshkol_tagged_value_t* vals, uint64_t n) {
    eshkol_region_t* current = region_current();
    if (!current || !vals || n == 0) return true;
    bool any = false;
    for (uint64_t i = 0; i < n && !any; ++i) {
        any = region_value_carries_pointer(vals[i].type) && vals[i].data.ptr_val != 0;
    }
    if (!any) return true;
    // Escaping FROM `current` (stack index depth-1) into its parent (or the
    // global arena). Everything living in `current`'s arena must be copied;
    // anything already in an ancestor region or the global arena is stable.
    arena_t* target = region_escape_target(current);
    const int boundary = (int)__region_stack_depth - 2;
    return region_evacuate_values(vals, n, target, boundary);
}
} // namespace eshkol::regions_internal

/**
 * @brief Copy a tagged value's heap payload (and its whole in-region subgraph)
 *        out of the current region.
 *
 * Non-heap values (ints, doubles, etc.) are returned unchanged, as are heap
 * values that are NULL. Otherwise the value's reachable subgraph that lives in
 * the current (about-to-be-destroyed) region's arena is deep-copied into
 * region_escape_target()'s arena in one promotion transaction: shared structure
 * and cycles are preserved by a forwarding map, and objects already living in an
 * enclosing region or the global arena are left in place. If no region is
 * currently active, the value is returned unchanged. When the promotion cannot
 * be completed nothing is published: a catchable allocation error is raised with
 * the region unchanged (#713). Shared implementation behind the two
 * `region_escape_tagged_value*` extern "C" entry points.
 *
 * @param val Tagged value to escape (passed and returned by value).
 * @return    Escaped value (with an updated heap pointer if a copy was
 *            made), or @p val unchanged if escaping wasn't needed.
 */
static eshkol_tagged_value_t region_escape_tagged_value_impl(eshkol_tagged_value_t val) {
    if (!region_escape_values_in_place(&val, 1)) region_raise_promotion_failure();
    return val;
}

/**
 * @brief Codegen-facing entry point: escape a tagged value's heap payload out of the current region, returned by value.
 *
 * @param val Tagged value to escape.
 * @return    See region_escape_tagged_value_impl().
 */
extern "C" eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val) {
    return region_escape_tagged_value_impl(val);
}

/**
 * @brief Codegen-facing entry point: escape a tagged value's heap payload out of the current region, returned via out-pointer.
 *
 * Output-pointer form of region_escape_tagged_value(), used where an sret
 * ABI is preferred over a returned struct. If @p val is NULL, writes an
 * all-zero (NULL-typed) tagged value to *out. On a failed promotion *out is
 * not written (the call raises).
 *
 * @param out Destination for the escaped value (no-op if NULL).
 * @param val Tagged value to escape (may be NULL).
 */
extern "C" void region_escape_tagged_value_into(eshkol_tagged_value_t* out,
                                                const eshkol_tagged_value_t* val) {
    if (!out) return;
    if (!val) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    *out = region_escape_tagged_value_impl(*val);
}

/* #713: promote @p n staged values in place for a store into @p dst. The one
 * rule every store channel shares: a value is promoted BEFORE it is stored, all
 * values of one store are promoted in one transaction, and a store whose
 * promotion fails stores nothing. Returns false (values unchanged) on failure. */
static bool region_promote_for_store(const void* dst, eshkol_tagged_value_t* vals,
                                     uint64_t n) {
    // FAST PATH: no active region -> nothing can dangle.
    if (__region_stack_depth == 0 || !vals || n == 0) return true;

    // Locate the values first and the destination only if some value lives in
    // a region. Both probes walk region arenas block by block, and most stores
    // (every integer or float written into a vector inside a loop's nursery)
    // carry no pointer at all: probing dst unconditionally made each of them
    // pay a full arena walk (a 25x slowdown of a vector-heavy example).
    int youngest = -1;   // innermost region owning any stored value
    for (uint64_t i = 0; i < n; ++i) {
        const eshkol_tagged_value_t& v = vals[i];
        if (!region_value_carries_pointer(v.type) || v.data.ptr_val == 0) continue;
        const int idx = region_index_owning((const void*)(uintptr_t)v.data.ptr_val);
        if (idx > youngest) youngest = idx;
    }
    if (youngest < 0) return true;                  // every value outer/global
    const int dst_idx = region_index_owning(dst);   // -1 when dst is outer/global
    // A value that lives at least as long as dst (same or an enclosing region,
    // or outside every region) is stored unchanged.
    if (youngest <= dst_idx) return true;

    // Where the promoted subgraph lands: the destination's own region arena
    // when dst lives in an active region, otherwise the TRUE global arena. The
    // outermost region's escape_base is that true global (captured before the
    // __global_arena slot was hijacked by with-region codegen), so use it rather
    // than get_global_arena(), which is hijacked during a region body.
    arena_t* target = (dst_idx >= 0) ? __region_stack[dst_idx]->arena
                                     : __region_stack[0]->escape_base;
    if (!target) target = get_global_arena();
    return region_evacuate_values(vals, n, target, dst_idx);
}

/**
 * @brief Region write barrier: promote @p value's in-region subgraph when it is
 *        stored into a longer-lived destination (ESH-0214c).
 *
 * Called by codegen at every mutation channel that can store a value into a
 * location that outlives the value's region (set-car!/set-cdr!, vector-set!,
 * hash-table-set!, set! of a global). The FAST PATH — no active region — is a
 * single thread-local load and branch, so the barrier is essentially free
 * outside `(with-region ...)`.
 *
 * When a region is active, the barrier evacuates @p value's reachable subgraph
 * out of any region strictly inner than @p dst's owning region iff @p value
 * actually points into such a region. If @p value already lives at least as long
 * as @p dst (same or an enclosing region, or the global arena), it is stored
 * unchanged. The evacuation preserves shared structure and cycles via a
 * forwarding map, so this is safe to apply on every store.
 *
 * The promotion is all-or-nothing (#713, ADR-0001): if any object the value
 * reaches cannot be copied, *out is left untouched and a catchable allocation
 * error is raised, so the caller's store never runs.
 *
 * @param out   Destination for the (possibly promoted) value to actually store.
 * @param dst   Address of the container/slot being written (used only to locate
 *              which region owns the destination); NULL is treated as
 *              global/outer.
 * @param value The value about to be stored.
 */
extern "C" void eshkol_region_write_barrier_into(eshkol_tagged_value_t* out,
                                                 const void* dst,
                                                 const eshkol_tagged_value_t* value) {
    if (!out) return;
    if (!value) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    eshkol_tagged_value_t v = *value;
    if (!region_promote_for_store(dst, &v, 1)) region_raise_promotion_failure();
    *out = v;
}

/**
 * @brief Range form of the region write barrier: promote @p n STAGED values in
 *        place before they are stored into destination @p dst.
 *
 * Used by the runtime's bulk stores (vector-copy!, vector-fill!, tensor slot
 * stores). The caller stages the values, calls this, and only then copies them
 * into the destination: a store must never publish a value before its promotion
 * has completed, because a failed promotion would leave a young pointer in the
 * destination (#713). All @p n values are promoted in ONE transaction with one
 * forwarding relation, so structure shared between them stays shared. On
 * failure @p slots is unchanged and a catchable allocation error is raised.
 * The fast path (no region) is a single thread-local load + branch.
 *
 * @param dst   Address of the destination container (region-ownership probe).
 * @param slots The staged values; rewritten in place to the promoted values.
 * @param n     Number of values.
 */
extern "C" void eshkol_region_write_barrier_range(const void* dst,
                                                  eshkol_tagged_value_t* slots,
                                                  uint64_t n) {
    if (!region_promote_for_store(dst, slots, n)) region_raise_promotion_failure();
}

// ───────────────────────────────────────────────────────────────────────────
// ESH-0214e: iter-scope PARTIAL RECLAMATION for mutating tick-loops.
//
// ESH-0214b's automatic per-iteration reclamation (arena_push_scope /
// eshkol_arena_iter_scope_end, runtime_arena_core.cpp) is all-or-nothing: the
// static gate (iterScopeSafeExpr) rejects a loop body outright the moment it
// contains any persistent mutation, because a value the iteration allocates and
// then stores into outer/persistent state would dangle when the scope is
// rewound, and the shallow out-value span test at the back edge cannot see that
// store. A resident tick loop that mutates persistent state EVERY tick
// therefore got NO reclamation and leaked one iteration's transient garbage per
// tick, unbounded (~3 KB/tick, linear).
//
// The fix does not fork a second evacuator or a second reclamation policy. It
// reuses, verbatim, the with-region machinery whose deep-transitive escape
// promotion has been validated over a 48-h resident run (ESH-0214c/d):
//
//   * The loop runs inside a NURSERY REGION (region_create/region_push/
//     eshkol_region_enter, opened once per activation), so every iteration
//     allocation lands in the nursery arena and region_index_owning() can
//     classify it. Each of the six mutation channels' EXISTING write barriers
//     (eshkol_region_write_barrier_into) then deep-promotes any barrier-recorded
//     escapee — a nursery value stored into persistent/outer state — out of the
//     nursery into the enclosing arena at the store, exactly as inside a
//     with-region body. The barrier IS the escape-root record: it evacuates
//     each escapee out of the tick arena the instant it becomes reachable from
//     persistent state.
//
//   * eshkol_iter_nursery_recycle (below) runs at each TCO back edge. It
//     promotes the loop-carried out-values (the freshly evaluated tail-call
//     args, which the barrier never sees) out of the nursery via the SAME
//     evacuator (region_escape_tagged_value_impl), then arena_reset()s the
//     nursery — reclaiming the whole iteration's transient garbage — as if no
//     mutation had happened. Its forwarding map (whose keys reference the arena
//     just reset) is dropped so no stale entry survives into the next
//     iteration's promotions.
//
// The reclamation is sound by the SAME invariant as with-region's region_pop:
// after promotion no surviving object holds a pointer into the region being
// freed (here, into the span being reset). This is textbook generational minor
// collection (Appel) — the nursery is the young generation, the write barrier
// is the remembered set, and recycle is the minor collection — but fully
// deterministic (bounded per-iteration work, no tracing pause), which preserves
// Eshkol's no-GC determinism.
//
// FALLBACK (correctness over reclamation): if the region stack is not in the
// expected shape at a back edge (the nursery is not the innermost active
// region — an unbalanced region op in the body, which well-formed codegen never
// emits), recycle RETAINS this iteration (skips promotion AND reset) rather than
// evacuate from the wrong region. Retention is bounded: the next well-formed
// back edge resets the whole arena, reclaiming the retained bytes too.
// ───────────────────────────────────────────────────────────────────────────

extern "C" int eshkol_arena_poison_enabled(void);

/**
 * @brief End-of-iteration recycle for an ESH-0214e nursery loop (see block above).
 *
 * @param region  The loop's nursery region (opened once at loop entry).
 * @param vals    The loop-carried out-values (tail-call args) about to flow into
 *                the next iteration; each is promoted out of the nursery IN
 *                PLACE so the caller stores the promoted (surviving) value into
 *                the loop's parameter slot.
 * @param n       Number of out-values.
 */
extern "C" void eshkol_iter_nursery_recycle(eshkol_region_t* region,
                                            eshkol_tagged_value_t* vals,
                                            uint64_t n) {
    if (!region) return;

    // Fallback guard: only recycle when this nursery is the innermost active
    // region, i.e. the iteration body left the region stack exactly as it found
    // it. Otherwise retain (skip promotion + reset) — never evacuate from the
    // wrong region. Bounded: a later well-formed recycle reclaims these bytes.
    if (region_current() != region) return;

    // 1. Promote every loop-carried out-value out of the nursery into the
    //    enclosing arena (region->escape_base). This is the identical evacuator
    //    and code path as with-region's result escape
    //    (region_escape_tagged_value); it shares the nursery region's forwarding
    //    map, so any structure an out-value shares with a barrier-promoted
    //    escapee from the same iteration stays shared (eq?-preserving). Values
    //    already living outside the nursery (immediates, or pointers into the
    //    enclosing/global arena) are returned unchanged and cost only a region
    //    ownership probe.
    //    All out-values are promoted in one transaction. If it aborts, nothing
    //    has moved and the nursery is NOT reset: the raise below unwinds the
    //    nursery region like any other region a raise crosses (#713).
    if (vals && !region_escape_values_in_place(vals, n)) region_raise_promotion_failure();

    // 2. Under the poison allocator, stamp the bytes about to be recycled with
    //    0xCB so any interior pointer we FAILED to promote out (a missed escape
    //    root) dereferences an obvious 0xCB.. address and crashes loudly instead
    //    of silently reading recycled memory — the dangling-pointer tripwire the
    //    ESH-0214 memory-model tests rely on.
    if (eshkol_arena_poison_enabled() && region->arena) {
        for (arena_block_t* b = region->arena->current_block; b; b = b->next) {
            if (b->used) std::memset(b->memory, 0xCB, b->used);
        }
    }

    // 3. Reset the nursery arena: reclaim ALL of this iteration's transient
    //    garbage (bump pointer back to zero, free any spilled blocks, keep the
    //    first block). O(1) on the steady-state single-block case — no per-
    //    iteration malloc/free churn (the reason a persistent nursery + reset is
    //    used instead of region_create/destroy per iteration).
    arena_reset(region->arena);

    // 4. Drop the forwarding map: its keys reference the arena just reset and
    //    MUST NOT alias a fresh object that reuses the same address next
    //    iteration. It is lazily recreated on the next escape/barrier promotion.
    region_free_fwd_map(region);
}
