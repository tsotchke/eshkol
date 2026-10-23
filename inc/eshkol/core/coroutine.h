/**
 * @file coroutine.h
 * @brief Stackful asymmetric coroutines and the per-execution-context state
 *        that every coroutine switch saves and restores.
 *
 * @section model The model
 *
 * A coroutine runs a nullary procedure on a stack of its own. The
 * procedure starts on the first `coroutine-resume`; `coroutine-yield` hands a
 * value back to whoever resumed it and suspends it, and the next resume
 * continues from that point with the resume's value as the yield's result.
 * A coroutine that returns is `done`; one whose body raised is `failed`, and
 * the raise is re-delivered to the resumer. This is the asymmetric
 * (resume/yield) shape Lua and Python generators use. The fiber scheduler in
 * lib/core/fibers.esk is built on it: fibers are coroutines resumed by a
 * scheduler that runs in the resuming context.
 *
 * @section why_stacks Why separate stacks rather than call/cc
 *
 * The language already has re-entrant first-class continuations. They were
 * the first candidate and they were rejected for this job, for three reasons
 * that are properties of the existing design rather than of an
 * implementation detail:
 *
 *   1. A native continuation is a copy of the C stack from the capture point
 *      to the stack base, allocated in the arena (runtime_continuations.cpp).
 *      A scheduler built on it pays a stack copy and an arena allocation on
 *      EVERY switch, and arena memory is reclaimed only by region exit. Ten
 *      thousand fibers switching in a loop grow memory without bound.
 *   2. Invoking a continuation reroots dynamic-wind: it runs the `after`
 *      thunks of the extent it leaves and the `before` thunks of the extent it
 *      enters. A fiber switch must do neither. A fiber parked inside a
 *      `dynamic-wind` has not left that extent; its `after` thunk must run
 *      exactly once, when the fiber's body actually leaves it (normally, by a
 *      raise, or by cancellation), never because another fiber was scheduled.
 *   3. The REPL refuses continuations whose image belongs to a finished
 *      top-level evaluation. A fiber spawned in one evaluation and resumed in
 *      a later one would be refused, although nothing about it is stale.
 *
 * A coroutine instead owns a stack and an execution context (below). A switch
 * saves the outgoing context's callee-saved registers and dynamic state and
 * loads the incoming one's: constant time, no allocation, no thunks run.
 * Continuations keep working INSIDE a coroutine: a capture copies only the
 * coroutine's own stack (the stack-base probe reports the coroutine's stack),
 * and a continuation is re-entered only on the coroutine that captured it.
 *
 * @section context The execution context
 *
 * Several runtime subsystems keep "the dynamic state of the running
 * computation" in thread-local storage: the dynamic-wind chain, the
 * exception-handler chain and the in-flight raise, the promise-evaluation
 * chain, the `parameterize` bindings, the parallel unwind-boundary chain, the lexical region stack and the
 * allocation domain it selects, the recursion-depth counter, and the stack
 * geometry the overflow guard and continuation capture read. With coroutines,
 * one OS thread runs several computations, so that state belongs to the
 * computation, not the thread. eshkol_exec_context_t is the one record of it;
 * each owning subsystem saves and loads its own fields through the functions
 * declared below, and the coroutine runtime calls them all on every switch.
 * A fresh coroutine starts from an empty context: no winds, no handlers, no
 * open regions. This is the per-computation memory context ADR-0001 calls
 * for, at the granularity of an execution context rather than an OS thread.
 *
 * Reverse-mode AD state is the one dynamic state that is NOT switched: a
 * switch while a differentiation pass is active is refused with a catchable
 * condition, because a tape that interleaves two computations has no
 * meaning.
 *
 * @section memory Memory
 *
 * Stacks are mmap'd (VirtualAlloc / CreateFiberEx on Windows) with a guard
 * page, never arena-allocated, and recycled through a bounded per-thread
 * pool. The Scheme-visible coroutine is a generation-tagged integer handle,
 * the convention of every kernel-resource holder in the runtime (see
 * inc/eshkol/core/event_loop.h). Values a coroutine stores across a switch
 * (its procedure, transferred values) pass the region write barrier first, so
 * nothing a coroutine holds can point into a region that closes under it.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#ifndef ESHKOL_CORE_COROUTINE_H
#define ESHKOL_CORE_COROUTINE_H

#include <stddef.h>
#include <stdint.h>

#include "eshkol/eshkol.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The dynamic state of one execution context.
 *
 * Zero-initialised means "a fresh computation on the thread's own stack":
 * empty chains, no open region, default allocation domain, depth 0.
 */
typedef struct eshkol_exec_context {
    /* runtime_continuations.cpp */
    void* wind_stack;                    /**< g_dynamic_wind_stack */
    uint64_t context_id;                 /**< owner id stamped into continuation images; 0 = thread stack */
    /* runtime_exceptions_hosted.cpp */
    void* handler_stack;                 /**< g_exception_handler_stack */
    int64_t handler_depth;
    void* current_exception;
    eshkol_tagged_value_t raised_value;
    int32_t raised_value_set_by_user;
    int32_t reserved0;
    void* promise_eval_head;
    /* lib/backend/parallel_codegen.cpp */
    void* parallel_boundary;             /**< innermost unwind boundary */
    /* runtime_regions.cpp */
    void** region_slots;                 /**< malloc'd; owned by this record */
    uint64_t region_depth;
    uint64_t region_capacity;
    void* allocation_domain;             /**< memctx allocation domain */
    /* runtime_parameters_hosted.cpp: the `parameterize` bindings this
     * context made, oldest first, lifted off their parameters while the
     * context is switched out. Elements are eshkol_parameter_binding_t. */
    void* param_bindings;
    uint64_t param_count;
    uint64_t param_capacity;
    /* resource_limits.cpp */
    uint64_t limit_stack_depth;
    /* runtime_stack_hosted.cpp: [stack_low, stack_high) of a coroutine
     * stack, or 0/0 for the thread's own stack. */
    uintptr_t stack_low;
    uintptr_t stack_high;
} eshkol_exec_context_t;

/* ── Per-subsystem save/load. Each moves only the fields its file owns. ── */

void eshkol_dynamic_wind_context_save(eshkol_exec_context_t* ctx);
void eshkol_dynamic_wind_context_load(const eshkol_exec_context_t* ctx);
void eshkol_exception_context_save(eshkol_exec_context_t* ctx);
void eshkol_exception_context_load(const eshkol_exec_context_t* ctx);
void eshkol_parallel_boundary_context_save(eshkol_exec_context_t* ctx);
void eshkol_parallel_boundary_context_load(const eshkol_exec_context_t* ctx);
/** @return 0 when the region slots could not be grown; nothing was changed. */
int  eshkol_region_context_save(eshkol_exec_context_t* ctx);
void eshkol_region_context_load(const eshkol_exec_context_t* ctx);
void eshkol_resource_limits_context_save(eshkol_exec_context_t* ctx);
void eshkol_resource_limits_context_load(const eshkol_exec_context_t* ctx);
void eshkol_stack_context_load(const eshkol_exec_context_t* ctx);

/** @return 0 when the binding record could not grow; nothing was changed. */
int  eshkol_parameter_context_save(eshkol_exec_context_t* ctx);
void eshkol_parameter_context_load(eshkol_exec_context_t* ctx);

/** Release storage an execution context record owns (region slots, parameter bindings). */
void eshkol_exec_context_dispose(eshkol_exec_context_t* ctx);

/**
 * @brief Id of the execution context running on this thread; 0 for the
 *        thread's own stack. Maintained by the coroutine runtime.
 */
uint64_t eshkol_exec_context_current_id(void);

/* ── Unwind boundary shared with the parallel schedulers ──────────────────
 *
 * A coroutine body runs under the same boundary a parallel-map callback runs
 * under (lib/backend/parallel_codegen.cpp): a raise inside it lands at the
 * boundary with the body's dynamic-wind, region and handler state already
 * unwound, and is recorded; a continuation captured outside it and invoked
 * inside it is recorded as a transfer. The resumer re-delivers either one.
 */
typedef struct eshkol_boundary_raise {
    int32_t pending;
    int32_t reserved;
    eshkol_tagged_value_t value;   /**< raised object, or the continuation's value */
    void* exception;               /**< eshkol_exception_t* of the raise */
    void* continuation;            /**< non-null for a recorded continuation transfer */
} eshkol_boundary_raise_t;

/**
 * @brief Call the nullary procedure @p thunk under an unwind boundary.
 *
 * @param thunk  A callable tagged value.
 * @param result Receives the procedure's value when it returns normally.
 * @param raised Receives the recorded raise or transfer; `pending` is 0 when
 *               the procedure returned.
 * @return 1 when the call was made; 0 when no thunk worker is registered (the
 *         compiled program has no procedures to call).
 */
int eshkol_boundary_call_thunk(const eshkol_tagged_value_t* thunk,
                               eshkol_tagged_value_t* result,
                               eshkol_boundary_raise_t* raised);

/** @return 1 when eshkol_boundary_call_thunk can call procedures (the
 *  compiled program registered its thunk worker). */
int eshkol_boundary_thunks_available(void);

/** Re-deliver a recorded raise or transfer in the calling context. Does not
 *  return when @p raised is pending. */
void eshkol_boundary_redeliver(const eshkol_boundary_raise_t* raised);

/* ── Coroutine C API ───────────────────────────────────────────────────── */

/** Coroutine states, as reported by eshkol_coroutine_status(). */
typedef enum {
    ESHKOL_COROUTINE_SUSPENDED = 0,  /**< created or yielded; resumable    */
    ESHKOL_COROUTINE_RUNNING   = 1,  /**< the coroutine executing now      */
    ESHKOL_COROUTINE_NORMAL    = 2,  /**< it resumed another coroutine     */
    ESHKOL_COROUTINE_DONE      = 3,  /**< its procedure returned           */
    ESHKOL_COROUTINE_FAILED    = 4   /**< its procedure raised              */
} eshkol_coroutine_status_t;

/** Counters reported by eshkol_coroutine_stats(). */
typedef struct {
    uint64_t live;            /**< coroutines holding a stack now          */
    uint64_t peak_live;       /**< high-water mark of `live`               */
    uint64_t created;         /**< coroutines ever created on this thread  */
    uint64_t stacks_mapped;   /**< stacks obtained from the OS             */
    uint64_t stacks_pooled;   /**< stacks parked in the reuse pool now     */
    uint64_t stack_bytes;     /**< reserved bytes per stack                */
    uint64_t handles_open;    /**< handles not yet released                */
} eshkol_coroutine_stats_t;

/** @return 1 when this build can run coroutines (0 on WASM and unknown targets). */
int eshkol_coroutines_available(void);

void eshkol_coroutine_stats(eshkol_coroutine_stats_t* out);

/* ── Builtin entry points (tagged-value ABI, see system_codegen.cpp) ────── */

void eshkol_builtin_make_coroutine(eshkol_tagged_value_t* out,
                                   const eshkol_tagged_value_t* thunk);
void eshkol_builtin_coroutine_resume(eshkol_tagged_value_t* out,
                                     const eshkol_tagged_value_t* co,
                                     const eshkol_tagged_value_t* value);
void eshkol_builtin_coroutine_raise(eshkol_tagged_value_t* out,
                                    const eshkol_tagged_value_t* co,
                                    const eshkol_tagged_value_t* obj);
void eshkol_builtin_coroutine_yield(eshkol_tagged_value_t* out,
                                    const eshkol_tagged_value_t* value);
void eshkol_builtin_coroutine_status(eshkol_tagged_value_t* out,
                                     const eshkol_tagged_value_t* co);
void eshkol_builtin_coroutine_release(eshkol_tagged_value_t* out,
                                      const eshkol_tagged_value_t* co);
void eshkol_builtin_current_coroutine(eshkol_tagged_value_t* out);
void eshkol_builtin_coroutine_p(eshkol_tagged_value_t* out,
                                const eshkol_tagged_value_t* value);
void eshkol_builtin_coroutine_stats(eshkol_tagged_value_t* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_COROUTINE_H */
