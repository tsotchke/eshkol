/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Internal interface shared by the runtime region translation units:
 *   runtime_regions.cpp             lifecycle (arenas, memctx, region stack)
 *   runtime_regions_evacuation.cpp  escape, evacuation, write barrier
 *   runtime_regions_handles.cpp     user region handles and the unwind path
 *   runtime_regions_resident.cpp    resident owners
 *
 * Everything here is runtime-private: it lives in eshkol::regions_internal
 * and has hidden visibility on ELF and Mach-O (COFF exports nothing without
 * dllexport), so the split adds no exported symbol.  The public region ABI
 * remains in arena_memory.h.
 */
#ifndef ESHKOL_CORE_RUNTIME_REGIONS_INTERNAL_H
#define ESHKOL_CORE_RUNTIME_REGIONS_INTERNAL_H

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

// Public runtime entry points one unit defines and another calls. Declared
// before the hidden-visibility region so they keep default visibility: both
// are resolved by name from generated and JIT code.
extern "C" eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val);
extern "C" void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#pragma GCC visibility push(hidden)
#endif

namespace eshkol::regions_internal {

/* ---- lifecycle state (runtime_regions.cpp) ---------------------------- */

// Memory-context access across units goes through eshkol_memctx_current();
// the defining unit retains its private thread-local storage.
// >0 while a work-stealing construct may run.
extern std::atomic<int> s_parallel_depth;

/* ---- evacuation (runtime_regions_evacuation.cpp) ----------------------- */

// Index of the innermost active region whose arena contains @p p, or -1.
int region_index_owning(const void* p);


// How to traverse a copied object's interior after the contiguous header+payload
// copy. EVAC_LEAF objects are self-contained (no interior region pointers) and
// need no further work.
enum EvacKind : uint8_t {
    EVAC_LEAF = 0,
    EVAC_CONS,        // arena_tagged_cons_cell_t: car + cdr tagged values
    EVAC_VECTOR,      // [i64 length][length tagged values]  (also records)
    EVAC_MULTIVALUE,  // [size_t count][count tagged values]
    EVAC_HASH,        // eshkol_hash_table_t: keys/values/status arrays + home_arena
    EVAC_TENSOR,      // eshkol_tensor_t: dimensions + elements raw buffers
    EVAC_EXCEPTION,   // eshkol_exception_t: message/filename/irritants
    EVAC_CLOSURE,     // eshkol_closure_t: captured environment
    // ESH-0214d: neuro-symbolic / workspace subtypes that carry interior
    // tagged values and/or raw arena buffers (previously dropped to EVAC_LEAF,
    // which silently left those interior pointers aimed into the dying region).
    EVAC_SUBSTITUTION,   // eshkol_substitution_t: inline terms[] tagged values
    EVAC_FACT,           // eshkol_fact_t: inline args[] tagged values (+ predicate str)
    EVAC_KNOWLEDGE_BASE, // eshkol_knowledge_base_t: facts[] raw array of fact ptrs
    EVAC_FACTOR_GRAPH,   // eshkol_factor_graph_t: nested raw numeric buffers
    EVAC_WORKSPACE,      // eshkol_workspace_t: content buffer + per-module name/process_fn
    EVAC_PROMISE,        // [forced:i64][thunk:tagged @8][cached:tagged @24] (delay/force)
    EVAC_RATIONAL,       // eshkol_rational_t: big_num/big_den raw bignum pointers (is_big==1 only)
    EVAC_PARAMETER,      // eshkol_param_t: converter + current/dynamic values
    // SW-66: an EXACT-COEFFICIENT (COEFF_RATIONAL) Taylor tower's c[] is an
    // array of eshkol_tagged_value_t that can hold HEAP_PTRs to arena-
    // resident bignum/rational coefficients (see runtime_taylor.c). A
    // COEFF_F64 tower's c[] is raw doubles -- nothing to walk, same shape as
    // EVAC_RATIONAL's is_big==0 fast path being a cheap no-op.
    EVAC_TAYLOR,         // esh_taylor_t: c[] tagged-value array (COEFF_RATIONAL only)
    EVAC_AD_NODE,        // headered ad_node_t: graph links and sized tensor payloads
};

using EvacFwdMap = std::unordered_map<const void*, void*>;

struct EvacState {
    arena_t* target;
    int boundary_idx;
    EvacFwdMap* fwd;   // persistent per-region map (see eshkol_region_t::fwd_map)
    std::vector<std::pair<void*, EvacKind>> worklist;
    size_t copies = 0;

    // SW-164: the evacuator answers exactly one question about every pointer
    // it meets — "is this about to be reclaimed, so must I copy it?" — and
    // there are now two kinds of caller asking. A with-region escape or a
    // nursery recycle asks it about REGIONS (copy anything living in a region
    // strictly inner than the destination's). A loop's per-iteration scope
    // asks it about an ARENA SPAN (copy anything allocated above a scope mark
    // on one arena). Same Cheney copier, same forwarding map, same subtype
    // coverage — only the ownership predicate differs, so it is abstracted
    // here rather than forked into a second evacuator that would drift.
    const arena_t* span_arena = nullptr;   // non-null => arena-span mode
    const arena_scope_t* span_scope = nullptr;  // mark; null => the whole arena
    // Explicit resident exports can reach both persistent owner storage and
    // the current tick nursery. Existing region/span callers leave this null.
    const arena_t* second_span_arena = nullptr;
    std::vector<const arena_t*> resident_sources;
    const arena_scope_t* resident_destination_mark = nullptr;
    const arena_t* resident_preserved_arena = nullptr;
    bool resident_copy_mutable = false;
    bool abort_on_refusal = false;

    // SW-164: set when this evacuation met something it must not MOVE.
    //
    // A region escape may leaf-copy an object whose interior it cannot walk and
    // still warn rather than stop, because what it copies OUT of the region is
    // a copy: the original stays put until the region dies, and the invariants
    // that keep such objects out of regions hold. An ARENA-SPAN evacuation has
    // neither property. The span it clears is the ordinary allocation arena, so
    // objects that are ordinarily never region-resident CAN be sitting in it,
    // and the rewind that follows reclaims the originals.
    //
    // The AD tape is the case that matters. A tape node is reachable from the
    // tape's own nodes[] array — a root in the C runtime that this evacuator
    // cannot see and does not rewrite — so MOVING a node leaves the recorded
    // evaluation order pointing at reclaimed memory, and the gradient that
    // comes back is plausible and wrong. That is true of a leaf node with no
    // inputs just as much as of one with a live interior graph.
    //
    // In span mode those cases refuse. The caller throws the staged copy away,
    // retains the span instead of rewinding it, and the loop behaves exactly as
    // it did before this feature existed. Reclamation is what is given up;
    // correctness is not.
    bool refused = false;

    void refuse() { if (span_arena) refused = true; }

    // #713: an evacuation is a TRANSACTION (ADR-0001 required semantics 8,
    // "all-or-nothing promotion"). It either promotes every object the roots
    // reach, or it leaves no trace: nothing is published, the forwarding
    // relation is exactly what it was, every byte outside the fresh copies is
    // exactly what it was, and the destination arena's bump pointer is rewound
    // when this thread owns it. A failed allocation used to log and hand back
    // the ORIGINAL pointer, which the barrier then stored -- so a root, or a
    // parent copied before its child failed, carried an edge into the region
    // about to be freed.
    //
    // Three records make the rollback exact:
    //   * `inserted` -- every forwarding key this transaction added. The map is
    //     the region's persistent one, so a stale entry would make a LATER
    //     escape reuse a half-built copy.
    //   * `saved`/`saved_bytes` -- the prior contents of every byte range the
    //     walk rewrites that is NOT one of this transaction's own copies
    //     (buffers a copied object shares with its original, a parameter's
    //     malloc-owned binding stack). Fresh copies need no record: they are
    //     unreachable once the transaction aborts.
    //   * `failed`/`failed_bytes` -- set by the first allocation that fails.
    //     From then on every copy primitive is the identity, so the rest of
    //     the walk writes back exactly the values it read and cannot publish
    //     anything new.
    std::vector<const void*> inserted;
    struct SavedRange { void* addr; size_t size; size_t offset; };
    std::vector<SavedRange> saved;
    std::vector<uint8_t> saved_bytes;
    bool failed = false;
    size_t failed_bytes = 0;

    void fail(size_t bytes) {
        if (!failed) {
            failed = true;
            failed_bytes = bytes;
        }
    }

    // Remember @p n bytes at @p p before the walk rewrites them in place.
    // Only for memory that existed before this transaction began.
    void preserve(void* p, size_t n) {
        if (failed || !p || n == 0) return;
        if (eshkol_alloc_failpoint_fire(ESHKOL_ALLOC_FAILPOINT_EVAC_SAVED)) throw std::bad_alloc();
        const size_t off = saved_bytes.size();
        saved_bytes.insert(saved_bytes.end(), (const uint8_t*)p, (const uint8_t*)p + n);
        saved.push_back({p, n, off});
    }

    void* lookup(const void* old) const {
        auto it = fwd->find(old);
        return it == fwd->end() ? nullptr : it->second;
    }

    // The key is recorded BEFORE it enters the map: if the record cannot
    // grow, the map is untouched; if the map cannot grow, erasing a recorded
    // key that never went in is harmless. The other order left a map entry
    // pointing into the rewound destination with nothing to remove it.
    void record(const void* old, void* copy) {
        if (eshkol_alloc_failpoint_fire(ESHKOL_ALLOC_FAILPOINT_EVAC_INSERTED)) throw std::bad_alloc();
        inserted.push_back(old);
        if (eshkol_alloc_failpoint_fire(ESHKOL_ALLOC_FAILPOINT_EVAC_FORWARD)) throw std::bad_alloc();
        (*fwd)[old] = copy;
    }

    bool owns(const void* p) const {
        // Visible caller arenas can also be the export destination. Only their
        // pre-transaction objects are sources; never recursively recopy a copy.
        if (resident_destination_mark && arena_scope_span_contains(target, resident_destination_mark, p))
            return false;
        if (resident_preserved_arena && arena_contains(resident_preserved_arena, p)) return false;
        for (const auto* source : resident_sources)
            if (arena_scope_span_contains(source, nullptr, p)) return true;
        if (span_arena) return arena_scope_span_contains(span_arena, span_scope, p) != 0 ||
            (second_span_arena && arena_scope_span_contains(second_span_arena, nullptr, p) != 0);
        return region_index_owning(p) > boundary_idx;
    }
};


struct EvacOutcome {
    bool ok = true;
    size_t failed_bytes = 0;
};

// Free a region's persistent deep-escape forwarding map.
void region_free_fwd_map(eshkol_region_t* region);
// Run one promotion transaction over @p n roots in place.
EvacOutcome evac_transaction(EvacState& st, eshkol_tagged_value_t* roots,
                             uint64_t n);
// Raise the catchable allocation error for a promotion that aborted.
void region_raise_promotion_failure(void);
// Promote @p vals out of the current region in place; false on abort.
bool region_escape_values_in_place(eshkol_tagged_value_t* vals, uint64_t n);

} // namespace eshkol::regions_internal

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#pragma GCC visibility pop
#endif

#endif /* ESHKOL_CORE_RUNTIME_REGIONS_INTERNAL_H */
