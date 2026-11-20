/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime region helpers: user region handles.
 *
 * This unit owns the generation-tagged region-handle protocol (open, close,
 * liveness, the builtins) and eshkol_region_unwind_to(), the single teardown
 * path shared by lexical exit, explicit close, raise and continuation escape.
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

// ───────────────────────────────────────────────────────────────────────────
// #341: USER-REACHABLE REGION HANDLES — non-lexical scoped reclamation.
//
// See the block comment over the declarations in arena_memory.h for the surface
// and the safety contract. This half is the mechanism.
//
// WHY A GENERATION-TAGGED INTEGER AND NOT A HEAP HANDLE OBJECT. The whole point
// of the API is a hot loop — a training step — so the handle must cost zero
// allocation: a heap handle object would land in the ENCLOSING arena (it has to,
// or closing the region would free the handle naming it) and leak one object per
// iteration for the life of the process, reintroducing in miniature exactly the
// unbounded growth #341 is about. An immediate integer allocates nothing. The
// generation counter then buys the safety property a raw pointer cannot have:
// every stale token is *detectably* stale, so double-close / use-after-close /
// a fabricated integer all fail validation and raise instead of dereferencing
// freed memory. This is the standard slot+generation handle, chosen here for the
// memory-safety property rather than for indirection.
//
// WHY ONE TEARDOWN PATH. eshkol_region_unwind_to() below is the ONLY code that
// takes a region down. Explicit close, out-of-order close, `with-region` exit, a
// raise crossing an open region, and a continuation escape all funnel through
// it, so the promotion-then-restore-then-pop ordering — and the evacuator that
// performs the promotion — cannot drift between the structured and unstructured
// surfaces. That ordering matters: the kept values are promoted while the region
// is still current (the evacuator resolves ownership off the live region stack),
// the allocation slot is restored BEFORE the arena is destroyed (so no
// allocation can ever target a freed arena), and only then is the region popped.
// ───────────────────────────────────────────────────────────────────────────

extern "C" void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

namespace {

/** @brief One user-reachable region-handle slot (thread-local, see t_region_handles). */
struct RegionHandleSlot {
    eshkol_region_t* region;  ///< The open region; NULL for a bookkeeping-only handle.
    uint64_t generation;      ///< Bumped on every close; a token carrying a stale
                              ///< generation can never validate again. Never 0 once used.
    uint64_t region_depth;    ///< __region_stack_depth immediately after region_push (1-based).
    uint64_t open_seq;        ///< Monotonic open order; the unwind mark on substrates
                              ///< with no region stack (see eshkol_region_handle_seq_mark).
    uint8_t reclaim;          ///< 1 = owns a real region; 0 = bookkeeping-only (VM contract).
    uint8_t in_use;           ///< 1 while the slot names an open handle.
};

}  // namespace

// Per-thread, matching the region stack's thread-locality: a handle is only
// meaningful on the thread that opened it (a token carried to another thread
// finds an empty slot table there and fails validation cleanly, which is the
// right answer — the other thread's region stack does not contain that region).
static thread_local RegionHandleSlot t_region_handles[ESHKOL_MAX_REGION_HANDLES];

// Monotonic open counter. Used as the unwind mark on substrates that have no
// region stack to measure depth against (see eshkol_region_handle_seq_mark).
static thread_local uint64_t t_region_handle_seq = 0;

// Token layout: low RH_SLOT_BITS bits hold slot+1 (so 0 is never a valid token
// and a zeroed/uninitialised integer cannot masquerade as a handle), the rest
// hold the slot's generation. ESHKOL_MAX_REGION_HANDLES is 64, so slot+1 is
// 1..64 and fits comfortably; the generation gets the remaining 55 bits.
#define RH_SLOT_BITS 8
#define RH_SLOT_MASK ((int64_t)((1 << RH_SLOT_BITS) - 1))

/**
 * @brief Validate @p token and resolve it to a slot index.
 * @return true iff the token names a currently-open handle on this thread.
 */
static bool rh_decode(int64_t token, int* slot_out) {
    if (token <= 0) return false;
    const int slot = (int)(token & RH_SLOT_MASK) - 1;
    if (slot < 0 || slot >= ESHKOL_MAX_REGION_HANDLES) return false;
    const RegionHandleSlot& s = t_region_handles[slot];
    if (!s.in_use) return false;
    if ((uint64_t)(token >> RH_SLOT_BITS) != s.generation) return false;
    if (slot_out) *slot_out = slot;
    return true;
}

/**
 * @brief Retire a handle slot, invalidating every token that named it.
 *
 * Bumping the generation is what makes double-close and use-after-close safe:
 * the caller's now-stale token no longer matches, so a second close raises a
 * clean error instead of tearing down whatever region happens to occupy that
 * stack position later.
 */
static void rh_retire(int slot) {
    RegionHandleSlot& s = t_region_handles[slot];
    s.in_use = 0;
    s.region = nullptr;
    s.region_depth = 0;
    s.open_seq = 0;
    s.reclaim = 0;
    s.generation++;
    if (s.generation == 0) s.generation = 1;  // 0 is the never-used sentinel
}

/** @brief Retire every reclaiming handle whose region sits above @p mark (i.e. is being closed). */
static void rh_retire_above(uint64_t mark) {
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        const RegionHandleSlot& s = t_region_handles[i];
        if (s.in_use && s.reclaim && s.region_depth > mark) rh_retire(i);
    }
}

/** @brief Region-stack depth — the mark form recorded by exception handlers and continuations. */
extern "C" uint64_t eshkol_region_mark(void) {
    return __region_stack_depth;
}

extern "C" uint64_t eshkol_region_handle_seq_mark(void) {
    return t_region_handle_seq;
}

extern "C" void eshkol_region_handle_seq_unwind_to(uint64_t mark) {
    // Only bookkeeping-only handles are retired here. A reclaiming handle's
    // teardown must go through the region stack (eshkol_region_unwind_to) so its
    // arena is actually freed and the allocation slot restored — retiring it by
    // sequence number would drop the token while leaving the region open.
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        const RegionHandleSlot& s = t_region_handles[i];
        if (s.in_use && !s.reclaim && s.open_seq > mark) rh_retire(i);
    }
}

extern "C" void eshkol_region_unwind_to(uint64_t mark,
                                        eshkol_tagged_value_t* vals,
                                        uint64_t n) {
    if (__region_stack_depth <= mark) return;

    // Retire the affected handles FIRST. Their regions are about to stop
    // existing, so their tokens must stop validating even if a promotion below
    // raises part-way through — a token that outlived its region is exactly the
    // dangling handle this design exists to make impossible.
    rh_retire_above(mark);

    while (__region_stack_depth > mark) {
        eshkol_region_t* region = __region_stack[__region_stack_depth - 1];

        // 1. Promote the kept / in-flight values ONE LEVEL out, using the same
        //    deep-transitive evacuator (interior-pointer walk, forwarding map,
        //    ESH-0214c/d subtype coverage) that with-region's result escape
        //    uses. Level-by-level rather than straight to the target arena so a
        //    multi-level cascade lands each object in the arena that genuinely
        //    outlives it, and so shared structure stays shared via each
        //    region's own forwarding map.
        //    All kept values move in one transaction (one forwarding relation).
        //    If it aborts, this region and every one below it are still intact
        //    and the kept values unchanged; the raise closes them through this
        //    same function, carrying the preallocated allocation condition,
        //    which needs no promotion (#713).
        if (vals && !region_escape_values_in_place(vals, n)) {
            region_raise_promotion_failure();
        }

        // 2. Restore the allocation slot BEFORE the arena dies. If this is
        //    skipped, the slot keeps pointing at freed memory and the very next
        //    allocation writes into it. No-op when enter declined to hijack.
        if (region) eshkol_region_leave(region->entry_saved_arena);

        // 3. Pop and destroy: frees the region arena (poisoning it with 0xCB
        //    first when ESHKOL_ARENA_POISON is set, see arena_destroy), so any
        //    value we failed to promote out reads as an obvious sentinel rather
        //    than as plausible stale data. A PINNED region is the exception:
        //    region_destroy() promotes its blocks into the enclosing arena
        //    instead (SW-74), because a continuation captured inside it may
        //    still read them.
        region_pop();
    }
}

extern "C" const char* eshkol_region_handle_status_message(int status) {
    switch (status) {
        case ESHKOL_RH_OK:
            return "region handle ok";
        case ESHKOL_RH_ERR_STALE:
            return "region-close: invalid or already-closed region handle";
        case ESHKOL_RH_ERR_NOT_LIVE:
            return "region-close: region handle no longer names a live region";
        case ESHKOL_RH_ERR_TOO_MANY:
            return "region-open: too many open region handles — a region-close is missing";
        case ESHKOL_RH_ERR_DEPTH:
            return "region-open: region stack overflow — a region-close is missing";
        case ESHKOL_RH_ERR_CREATE:
            return "region-open: failed to create region";
        default:
            return "region handle: unknown error";
    }
}

extern "C" int64_t eshkol_region_handle_open(const char* name, uint64_t size_hint,
                                             int reclaim, int* status) {
    if (status) *status = ESHKOL_RH_OK;

    int slot = -1;
    for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
        if (!t_region_handles[i].in_use) { slot = i; break; }
    }
    if (slot < 0) {
        // The bounded outcome for "never closed": a loop that opens without
        // closing hits this after ESHKOL_MAX_REGION_HANDLES iterations and gets a
        // clean catchable error naming the cause, rather than growing without
        // bound or corrupting anything.
        if (status) *status = ESHKOL_RH_ERR_TOO_MANY;
        return 0;
    }

    RegionHandleSlot& s = t_region_handles[slot];
    if (reclaim) {
        if (__region_stack_depth >= MAX_REGION_DEPTH) {
            if (status) *status = ESHKOL_RH_ERR_DEPTH;
            return 0;
        }
        eshkol_region_t* region = region_create(name, size_hint);
        if (!region) {
            if (status) *status = ESHKOL_RH_ERR_CREATE;
            return 0;
        }
        region_push(region);
        if (region_current() != region) {
            // region_push declined (it logs and leaves the stack untouched).
            // Drop the orphan rather than handing back a handle to a region that
            // is not on the stack.
            region_destroy(region);
            if (status) *status = ESHKOL_RH_ERR_CREATE;
            return 0;
        }
        // Redirect allocation into the new region and record the displaced arena
        // on the region (eshkol_region_enter stores it) so close/unwind can
        // restore it without a lexical register.
        (void)eshkol_region_enter(region);
        // SW-74: mark the region as closable out-of-line. `region-close` is an
        // ordinary call, so unlike a lexical `with-region` exit it can run from
        // inside a `call/cc` procedure and tear this region down while an
        // escape-only continuation captured in it is still invocable. That is
        // the one case in which an escape-only capture must still pin; see
        // eshkol_region_t::handle_owned.
        region->handle_owned = 1;
        s.region = region;
        s.region_depth = __region_stack_depth;
        s.reclaim = 1;
    } else {
        // Bookkeeping-only contract (bytecode VM): the handle protocol, its
        // validation and its error text are identical, but no region is created
        // and close reclaims nothing. See tests/vm_parity/PARITY.tsv.
        s.region = nullptr;
        s.region_depth = 0;
        s.reclaim = 0;
    }

    if (s.generation == 0) s.generation = 1;
    s.open_seq = ++t_region_handle_seq;
    s.in_use = 1;
    return (int64_t)((s.generation << RH_SLOT_BITS) | (uint64_t)(slot + 1));
}

extern "C" int eshkol_region_handle_close(int64_t token,
                                          eshkol_tagged_value_t* vals,
                                          uint64_t n) {
    int slot = -1;
    if (!rh_decode(token, &slot)) {
        // Covers double-close, use-after-close, a token from another thread, and
        // a fabricated integer — all the same clean, catchable failure.
        return ESHKOL_RH_ERR_STALE;
    }

    RegionHandleSlot& s = t_region_handles[slot];

    if (!s.reclaim) {
        // Bookkeeping-only handle: values already live in the substrate heap and
        // need no promotion. But the observable token protocol must still match
        // the reclaiming substrate exactly, which means an out-of-order close has
        // to CASCADE here too: closing an outer handle retires every handle
        // opened after it. The reclaiming path gets that ordering from the region
        // stack; with no region stack to consult, the monotonic open sequence is
        // the same ordering. (Caught by tests/vm_parity/corpus/
        // region_handle_contract.esk, which observed the inner handle still live
        // on the VM after the outer one was closed.)
        const uint64_t seq = s.open_seq;
        for (int i = 0; i < ESHKOL_MAX_REGION_HANDLES; ++i) {
            const RegionHandleSlot& o = t_region_handles[i];
            if (o.in_use && !o.reclaim && o.open_seq >= seq) rh_retire(i);
        }
        return ESHKOL_RH_OK;
    }

    // The region must still occupy the stack position it was opened at. If it
    // does not, an earlier teardown already removed it and this token is stale
    // in a way rh_decode could not see; refuse rather than unwind a stack shape
    // we no longer recognise.
    if (s.region_depth == 0 || s.region_depth > __region_stack_depth ||
        __region_stack[s.region_depth - 1] != s.region) {
        rh_retire(slot);
        return ESHKOL_RH_ERR_NOT_LIVE;
    }

    // Closing an OUTER handle while inner regions are still open is a defined
    // cascade, not an error: unwind_to closes every region above this one,
    // innermost first, promoting the kept values out at every level and retiring
    // the inner handles' tokens. This is deliberately the same operation a
    // non-local exit performs, so there is one teardown path and no ordering
    // rule for callers to get wrong.
    eshkol_region_unwind_to(s.region_depth - 1, vals, n);
    return ESHKOL_RH_OK;
}

extern "C" int eshkol_region_handle_live(int64_t token) {
    return rh_decode(token, nullptr) ? 1 : 0;
}

// ── Surface entry points: (region-open …) / (region-close …) / (region-open? …)
//
// One C function per builtin, shared VERBATIM by both substrates so the handle
// protocol, the argument coercions and — critically — the error message text
// cannot diverge between native and VM. Optional arguments arrive as null
// pointers rather than as sentinel values, which keeps every arity of
// `region-open` on a single implementation. This mirrors the uniform
// cross-substrate contract established for the bytevector bounds checks.

/** @brief Debug label from a tagged symbol or string, else NULL. Never copies. */
static const char* rh_label_of(const eshkol_tagged_value_t* v) {
    if (!v) return nullptr;
    const uint8_t t = v->type;
    if (t == ESHKOL_VALUE_SYMBOL || t == ESHKOL_VALUE_STRING_PTR) {
        return (const char*)(uintptr_t)v->data.ptr_val;
    }
    if (t == ESHKOL_VALUE_HEAP_PTR && v->data.ptr_val) {
        const auto* hdr = (const eshkol_object_header_t*)
            ((const uint8_t*)(uintptr_t)v->data.ptr_val - sizeof(eshkol_object_header_t));
        if (hdr->subtype == HEAP_SUBTYPE_STRING) {
            return (const char*)(uintptr_t)v->data.ptr_val;
        }
    }
    return nullptr;
}

/** @brief True when @p v is a number (so a lone `region-open` argument reads as a size hint). */
static bool rh_is_number(const eshkol_tagged_value_t* v) {
    if (!v) return false;
    return v->type == ESHKOL_VALUE_INT64 || v->type == ESHKOL_VALUE_DOUBLE;
}

/** @brief Non-negative byte count from a tagged int/double, else 0 (= default size). */
static uint64_t rh_size_of(const eshkol_tagged_value_t* v) {
    if (!v) return 0;
    if (v->type == ESHKOL_VALUE_INT64) {
        return v->data.int_val > 0 ? (uint64_t)v->data.int_val : 0;
    }
    if (v->type == ESHKOL_VALUE_DOUBLE) {
        return v->data.double_val > 0 ? (uint64_t)v->data.double_val : 0;
    }
    return 0;
}

/**
 * @brief `(region-open)` / `(region-open name-or-size)` / `(region-open name size)`.
 *
 * A lone NUMERIC argument is the size hint; a lone non-numeric argument is the
 * debug name. @p reclaim selects the substrate contract (see
 * eshkol_region_handle_open).
 *
 * @param out     Receives the handle token as an exact integer.
 * @param a,b     Optional arguments; NULL for an argument that was not supplied.
 * @param reclaim Non-zero on a substrate that can actually reclaim (native).
 */
extern "C" void eshkol_region_open_builtin(eshkol_tagged_value_t* out,
                                           const eshkol_tagged_value_t* a,
                                           const eshkol_tagged_value_t* b,
                                           int reclaim) {
    if (!out) return;
    const char* name = nullptr;
    uint64_t size_hint = 0;
    if (a && b) {
        name = rh_label_of(a);
        size_hint = rh_size_of(b);
    } else if (a) {
        if (rh_is_number(a)) size_hint = rh_size_of(a);
        else name = rh_label_of(a);
    }
    int status = ESHKOL_RH_OK;
    const int64_t token = eshkol_region_handle_open(name, size_hint, reclaim, &status);
    if (status != ESHKOL_RH_OK) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(status));
        return;
    }
    *out = eshkol_make_int64(token, true);
}

/**
 * @brief `(region-close handle v ...)` — close the region, hand back the kept values.
 *
 * Returns the single promoted value for one keep, a freshly consed list of the
 * promoted values for several, and an empty list for none. The result list is
 * built AFTER the unwind, so it is allocated in the surviving (enclosing) arena
 * rather than in the region being torn down.
 *
 * @param out    Receives the result.
 * @param handle The handle token.
 * @param vals   The keep list, promoted IN PLACE by the unwind (may be NULL when n == 0).
 * @param n      Number of keeps.
 */
extern "C" void eshkol_region_close_builtin(eshkol_tagged_value_t* out,
                                            const eshkol_tagged_value_t* handle,
                                            eshkol_tagged_value_t* vals,
                                            uint64_t n) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_NULL;

    if (!handle || handle->type != ESHKOL_VALUE_INT64) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(ESHKOL_RH_ERR_STALE));
        return;
    }

    const int status = eshkol_region_handle_close(handle->data.int_val, vals, n);
    if (status != ESHKOL_RH_OK) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s",
                             eshkol_region_handle_status_message(status));
        return;
    }

    if (n == 1 && vals) {
        *out = vals[0];
    } else if (n > 1 && vals) {
        // Build the result list back-to-front in the arena that survived the
        // close (the slot has already been restored by the unwind).
        eshkol_tagged_value_t acc;
        std::memset(&acc, 0, sizeof(acc));
        acc.type = ESHKOL_VALUE_NULL;
        for (uint64_t i = n; i > 0; --i) {
            arena_tagged_cons_cell_t* cell =
                arena_allocate_tagged_cons_cell(get_global_arena());
            if (!cell) {
                eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                                     "region-close: out of memory building result list");
                return;
            }
            cell->car = vals[i - 1];
            cell->cdr = acc;
            std::memset(&acc, 0, sizeof(acc));
            acc.type = ESHKOL_VALUE_CONS_PTR;
            acc.data.ptr_val = (uint64_t)(uintptr_t)cell;
        }
        *out = acc;
    }
}

/**
 * @brief `(region-open? handle)` — #t while @p handle names a live open region.
 *
 * Never raises: this is the probe that lets cleanup code be written idempotently
 * (and the one the safety tests use to observe that an unwind closed a handle).
 */
extern "C" void eshkol_region_open_p_builtin(eshkol_tagged_value_t* out,
                                             const eshkol_tagged_value_t* handle) {
    if (!out) return;
    std::memset(out, 0, sizeof(*out));
    out->type = ESHKOL_VALUE_BOOL;
    const bool live = handle && handle->type == ESHKOL_VALUE_INT64 &&
                      eshkol_region_handle_live(handle->data.int_val) != 0;
    out->data.int_val = live ? 1 : 0;
}
