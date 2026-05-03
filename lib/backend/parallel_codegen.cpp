/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Parallel Primitives for Eshkol - C Runtime Implementation
 *
 * This module implements the parallel-map, parallel-fold, parallel-filter,
 * and parallel-for-each operations using the thread pool.
 *
 * ARCHITECTURE:
 * ============
 * The complexity of Eshkol's closure calling convention (dynamic capture counts)
 * is handled by LLVM-generated dispatcher functions. The C runtime simply calls
 * these dispatchers with a uniform interface:
 *
 *   __eshkol_call_unary_closure(item, closure) -> result
 *   __eshkol_call_binary_closure(arg1, arg2, closure) -> result
 *
 * These dispatchers are generated in parallel_llvm_codegen.cpp and handle:
 * - Extracting capture count from closure environment
 * - Dispatching to the correct function signature (0-32 captures)
 * - Passing capture pointers in the correct order
 *
 * This separation ensures:
 * - C runtime remains simple and portable
 * - All ABI complexity is in LLVM IR generation
 * - Works seamlessly with both LLVM and XLA backends
 */

#include "../../inc/eshkol/backend/thread_pool.h"
#include "../core/arena_memory.h"
#include "../../inc/eshkol/eshkol.h"
#include "../../inc/eshkol/logger.h"

#include <vector>
#include <future>
#include <atomic>
#include <algorithm>
#include <cstring>
#ifndef _WIN32
#include <dlfcn.h>   /* dlsym(RTLD_DEFAULT, …) for lazy worker resolution */
#endif

// ============================================================================
// LLVM Worker Function Pointers (Runtime Registration)
// ============================================================================
// These function pointers are set when stdlib.o is loaded (via __eshkol_register_parallel_workers).
// This breaks the circular build dependency: eshkol-run can be built without stdlib.o,
// then stdlib.o is compiled and registers its workers when loaded.
//
// ARCHITECTURE (Pure LLVM):
// The workers take void* and handle tagged values entirely in LLVM IR.
// This eliminates struct-by-value ABI issues at the C/LLVM boundary.
//
// Flow: C runtime → thread pool → LLVM worker → LLVM dispatcher → closure
//       (only void* crosses C/LLVM boundary, all tagged values stay in LLVM)

// Worker function type
typedef void* (*parallel_worker_fn)(void*);

// Function pointers for LLVM-generated workers (set by __eshkol_register_parallel_workers)
static parallel_worker_fn g_parallel_map_worker = nullptr;
static parallel_worker_fn g_parallel_fold_worker = nullptr;
static parallel_worker_fn g_parallel_filter_worker = nullptr;
static parallel_worker_fn g_parallel_execute_worker = nullptr;

// Function pointer types for closure dispatchers
typedef eshkol_tagged_value_t (*unary_closure_fn)(
    eshkol_tagged_value_t item,
    eshkol_tagged_value_t closure
);
typedef eshkol_tagged_value_t (*binary_closure_fn)(
    eshkol_tagged_value_t arg1,
    eshkol_tagged_value_t arg2,
    eshkol_tagged_value_t closure
);

// Function pointers for LLVM-generated dispatchers
static unary_closure_fn g_call_unary_closure = nullptr;
static binary_closure_fn g_call_binary_closure = nullptr;

extern "C" {

/**
 * __eshkol_register_parallel_workers - Register LLVM-generated workers
 *
 * Called automatically when stdlib.o is loaded. The LLVM codegen emits
 * a call to this function from the module initializer.
 */
void __eshkol_register_parallel_workers(
    void* map_worker,
    void* fold_worker,
    void* filter_worker,
    void* unary_dispatcher,
    void* binary_dispatcher,
    void* execute_worker)
{
    g_parallel_map_worker = reinterpret_cast<parallel_worker_fn>(map_worker);
    g_parallel_fold_worker = reinterpret_cast<parallel_worker_fn>(fold_worker);
    g_parallel_filter_worker = reinterpret_cast<parallel_worker_fn>(filter_worker);
    g_call_unary_closure = reinterpret_cast<unary_closure_fn>(unary_dispatcher);
    g_call_binary_closure = reinterpret_cast<binary_closure_fn>(binary_dispatcher);
    g_parallel_execute_worker = reinterpret_cast<parallel_worker_fn>(execute_worker);

    eshkol_debug("Parallel workers registered: map=%p, fold=%p, filter=%p, unary=%p, binary=%p, execute=%p",
                 map_worker, fold_worker, filter_worker, unary_dispatcher, binary_dispatcher, execute_worker);
}

/**
 * Check if parallel workers have been registered
 */
bool eshkol_parallel_workers_registered() {
    return g_parallel_map_worker != nullptr;
}

/**
 * Lazy worker resolution via dlsym fallback.
 *
 * Background.  The LLVM-generated workers
 * (`__parallel_map_worker`, `__parallel_fold_worker`, etc.) are
 * registered with the C runtime by a global constructor
 * (`__eshkol_init_parallel_workers`) emitted from
 * parallel_llvm_codegen.cpp into stdlib.o.  On most platforms the
 * dynamic loader runs `.init_array` entries from every linked object
 * before main(), so by the time any user code calls
 * `(parallel-execute …)` the worker pointers are non-NULL.
 *
 * On Linux AArch64 with lld, however, the constructor's `.ctors`
 * placement (rather than `.init_array`) is silently skipped by glibc
 * — the user binary contains the function and the entry, but the
 * loader never invokes it, so every parallel primitive errors with
 * `worker not registered (stdlib not loaded?)` and silently returns
 * the empty list.  Forcing the linkage / placement to fix lld is
 * fragile and target-dependent.
 *
 * Instead: at runtime, on the first call into any parallel
 * primitive, look up the worker symbols via `dlsym(RTLD_DEFAULT,
 * …)`.  RTLD_DEFAULT searches the main executable plus loaded
 * shared objects in load order, so for our compile-and-link path
 * (where stdlib.o is statically linked into the user binary) the
 * symbols are visible.  No reliance on `.init_array` ordering, no
 * reliance on linker section placement.
 *
 * The lookup is idempotent and runs once; subsequent calls take the
 * fast path through the already-set globals.
 */
static void eshkol_parallel_workers_lazy_resolve() {
    if (g_parallel_map_worker != nullptr) return;  // already set
#ifndef _WIN32
    /* dlsym/RTLD_DEFAULT: search the main executable plus all
     * already-loaded DSOs.  -Wl,--export-dynamic on the user link is
     * required so the static-linked symbols end up in the dynamic
     * symbol table — eshkol-run already passes that flag (see the
     * AArch64-Linux block in eshkol_compile_llvm_ir_to_executable). */
    void* map_w     = dlsym(RTLD_DEFAULT, "__parallel_map_worker");
    void* fold_w    = dlsym(RTLD_DEFAULT, "__parallel_fold_worker");
    void* filter_w  = dlsym(RTLD_DEFAULT, "__parallel_filter_worker");
    void* exec_w    = dlsym(RTLD_DEFAULT, "__parallel_execute_worker");
    void* unary_d   = dlsym(RTLD_DEFAULT, "__eshkol_call_unary_closure");
    void* binary_d  = dlsym(RTLD_DEFAULT, "__eshkol_call_binary_closure");
    if (map_w && filter_w && unary_d && binary_d && exec_w) {
        __eshkol_register_parallel_workers(map_w, fold_w, filter_w,
                                            unary_d, binary_d, exec_w);
    }
#endif
}

} // extern "C"

// ============================================================================
// C-Side Task Structs (must match LLVM layout in parallel_llvm_codegen.cpp)
// ============================================================================

// Task data for parallel map - fields are i64 to avoid struct-by-value ABI
struct llvm_parallel_map_task {
    uint64_t closure_ptr;   // offset 0: pointer to closure struct (from fn.data.ptr_val)
    uint64_t item_type;     // offset 8: type field (i64-extended from i8)
    uint64_t item_data;     // offset 16: data field (raw_val)
    uint64_t result_ptr;    // offset 24: pointer to eshkol_tagged_value_t for result
};

// Task data for parallel fold (binary closure)
struct llvm_parallel_fold_task {
    uint64_t closure_ptr;   // offset 0: pointer to closure struct
    uint64_t arg1_type;     // offset 8: first arg type
    uint64_t arg1_data;     // offset 16: first arg data
    uint64_t arg2_type;     // offset 24: second arg type
    uint64_t arg2_data;     // offset 32: second arg data
    uint64_t result_ptr;    // offset 40: pointer for result
};

// Task data for parallel execute (nullary closure / thunk)
struct llvm_parallel_execute_task {
    uint64_t closure_ptr;   // offset 0: pointer to closure struct (from fn.data.ptr_val)
    uint64_t closure_type;  // offset 8: type field (i64-extended from i8)
    uint64_t closure_flags; // offset 16: flags field (i64-extended from i8)
    uint64_t result_ptr;    // offset 24: pointer to eshkol_tagged_value_t for result
};

// ============================================================================
// Eager async future submission
// ============================================================================
//
// Builds on the existing thread-pool execute worker (g_parallel_execute_worker)
// to give `(future thunk)` true concurrent kickoff: at future-creation time
// we allocate a llvm_parallel_execute_task pointing at a result slot stored
// in the lazy_future, submit it to the global pool, and stash the resulting
// eshkol_future_t in the lazy_future's `pool_future` field.  `force` then
// checks `is_async`; if set, it calls `join_async` which future_get's the
// pool future (blocking until the worker completes) and copies the worker's
// written result into the lazy_future's stored fields.
//
// Layout of lazy_future async fields (lib/backend/thread_pool.cpp):
//   pool_future          → eshkol_future_t* (NULL = lazy)
//   async_result_slot    → eshkol_tagged_value_t* the worker writes
//   async_task           → llvm_parallel_execute_task* owned by lazy_future

// Mirror of the lazy_future struct in lib/backend/thread_pool.cpp.
// Kept here as a shadow because parallel_codegen.cpp doesn't have a
// public header for the struct.  Field offsets MUST match.
namespace {
struct eshkol_lazy_future_layout {
    uint64_t thunk_ptr;
    uint8_t  thunk_type;
    uint8_t  thunk_flags;
    uint8_t  forced;
    uint8_t  pad[5];
    uint64_t result_ptr;
    uint8_t  result_type;
    uint8_t  result_flags;
    uint8_t  pad2[6];
    void*    pool_future;
    void*    async_result_slot;
    void*    async_task;
};
} // anonymous namespace

extern "C" uint8_t eshkol_lazy_future_submit_async(
    void* lf_void, uint64_t closure_ptr, uint8_t closure_type, uint8_t closure_flags) {
    if (!lf_void) return 0;
    auto* lf = reinterpret_cast<eshkol_lazy_future_layout*>(lf_void);

    if (!g_parallel_execute_worker) eshkol_parallel_workers_lazy_resolve();
    if (!g_parallel_execute_worker) return 0;     // workers not registered

    eshkol_thread_pool_t* pool = thread_pool_global();
    if (!pool) return 0;

    // Allocate result slot + task on the heap; lazy_future owns them.
    auto* slot = new eshkol_tagged_value_t{};
    slot->type = ESHKOL_VALUE_NULL;
    auto* task = new llvm_parallel_execute_task{
        closure_ptr,
        static_cast<uint64_t>(closure_type),
        static_cast<uint64_t>(closure_flags),
        reinterpret_cast<uint64_t>(slot),
    };
    eshkol_future_t* fut = thread_pool_submit(pool, g_parallel_execute_worker, task);
    if (!fut) { delete slot; delete task; return 0; }

    lf->pool_future = fut;
    lf->async_result_slot = slot;
    lf->async_task = task;
    return 1;
}

extern "C" void eshkol_lazy_future_join_async(void* lf_void) {
    if (!lf_void) return;
    auto* lf = reinterpret_cast<eshkol_lazy_future_layout*>(lf_void);

    if (lf->forced) return;            // already done
    if (!lf->pool_future) return;      // not async

    eshkol_future_t* fut = static_cast<eshkol_future_t*>(lf->pool_future);
    (void)future_get(fut);             // blocks until worker completes
    auto* slot = static_cast<eshkol_tagged_value_t*>(lf->async_result_slot);
    if (slot) {
        lf->result_ptr   = slot->data.ptr_val;
        lf->result_type  = slot->type;
        lf->result_flags = slot->flags;
    }
    lf->forced = 1;
    future_release(fut);
    delete static_cast<llvm_parallel_execute_task*>(lf->async_task);
    delete slot;
    lf->pool_future = nullptr;
    lf->async_task = nullptr;
    lf->async_result_slot = nullptr;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Helper to check if a tagged value is "truthy"
static inline bool is_truthy(const eshkol_tagged_value_t& val) {
    // NULL is falsy
    if (val.type == ESHKOL_VALUE_NULL) {
        return false;
    }
    // Boolean false is falsy
    if (val.type == ESHKOL_VALUE_BOOL && val.data.int_val == 0) {
        return false;
    }
    // Everything else is truthy
    return true;
}

// Convert Scheme list to vector of tagged values
static std::vector<eshkol_tagged_value_t> list_to_vector(eshkol_tagged_value_t list) {
    std::vector<eshkol_tagged_value_t> result;

    eshkol_tagged_value_t current = list;
    while (current.type != ESHKOL_VALUE_NULL) {
        if (current.type != ESHKOL_VALUE_HEAP_PTR) {
            break;  // Not a proper list
        }

        // Get the cons cell
        auto* cell = reinterpret_cast<arena_tagged_cons_cell_t*>(current.data.ptr_val);
        if (!cell) break;

        result.push_back(cell->car);
        current = cell->cdr;
    }

    return result;
}

// Convert vector to Scheme list (in reverse to maintain order)
static eshkol_tagged_value_t vector_to_list(
    const std::vector<eshkol_tagged_value_t>& vec,
    arena_t* arena)
{
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_NULL;
    result.flags = 0;
    result.reserved = 0;
    result.data.raw_val = 0;

    // Build list from end to beginning
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        auto* cell = arena_allocate_cons_with_header(arena);
        if (!cell) {
            eshkol_error("Failed to allocate cons cell in vector_to_list");
            return result;
        }

        cell->car = *it;
        cell->cdr = result;

        result.type = ESHKOL_VALUE_HEAP_PTR;
        result.flags = HEAP_SUBTYPE_CONS;
        result.data.ptr_val = reinterpret_cast<uint64_t>(cell);
    }

    return result;
}

// ============================================================================
// Parallel Map Implementation (Pure LLVM Workers)
// ============================================================================
//
// The parallel map uses LLVM-generated workers to avoid ABI issues.
// The worker function __parallel_map_worker is generated in parallel_llvm_codegen.cpp
// and handles all tagged value operations in pure LLVM IR.

extern "C" {

// ============================================================================
// C API: Parallel Map
// ============================================================================

/**
 * eshkol_parallel_map - Apply function to each element in parallel
 *
 * @param fn        Closure to apply to each element (as tagged value)
 * @param list      Input list of elements
 * @param arena     Global arena for building result list
 * @return          New list with mapped values
 *
 * Architecture: Uses LLVM-generated __parallel_map_worker to avoid ABI issues.
 * Task data is decomposed to i64 fields (no struct-by-value across C/LLVM boundary).
 * The worker reconstructs tagged values in LLVM IR and calls the dispatcher directly.
 *
 * Usage in Scheme:
 *   (parallel-map fn list)
 *   (parallel-map (lambda (x) (* x x)) '(1 2 3 4 5))
 */
/* SRET-style wrapper for the pointer-all ABI used by LLVM codegen in this
 * codebase. Passes fn/list by pointer to sidestep struct-by-value ABI
 * differences between LLVM and the C calling convention. The original
 * by-value entry point is retained for callers outside LLVM. */
extern "C" void eshkol_parallel_map_sret(
    eshkol_tagged_value_t* out_result,
    const eshkol_tagged_value_t* fn_ptr,
    const eshkol_tagged_value_t* list_ptr,
    arena_t* arena);

void eshkol_parallel_map(
    eshkol_tagged_value_t fn,
    eshkol_tagged_value_t list,
    arena_t* arena,
    eshkol_tagged_value_t* out_result)
{
    eshkol_debug("parallel-map: fn.type=%d, list.type=%d", fn.type, list.type);

    // Handle empty list
    if (list.type == ESHKOL_VALUE_NULL) {
        *out_result = list; return;
    }

    // Validate closure
    if (fn.type != ESHKOL_VALUE_CALLABLE) {
        eshkol_error("parallel-map: expected callable, got type %d", fn.type);
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    // Convert list to vector for parallel processing
    auto items = list_to_vector(list);
    eshkol_debug("parallel-map: items.size()=%zu", items.size());
    if (items.empty()) {
        *out_result = list; return;
    }

    size_t n = items.size();

    // Allocate result storage (workers write results via pointer)
    std::vector<eshkol_tagged_value_t> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i].type = ESHKOL_VALUE_NULL;
        results[i].flags = 0;
        results[i].reserved = 0;
        results[i].data.raw_val = 0;
    }

    // Check if workers are registered, falling back to dlsym on platforms
    // (notably AArch64 Linux + lld) where the stdlib's global ctor that
    // would normally have done this is silently skipped by the loader.
    if (!g_parallel_map_worker) eshkol_parallel_workers_lazy_resolve();
    if (!g_parallel_map_worker) {
        eshkol_error("parallel-map: workers not registered (stdlib not loaded?)");
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    // For small lists, use LLVM worker directly (still benefits from LLVM→LLVM calls)
    if (n < 4) {
        eshkol_debug("parallel-map: sequential path for small list");
        for (size_t i = 0; i < n; ++i) {
            // Pack task data (decomposed to i64 fields)
            llvm_parallel_map_task task;
            task.closure_ptr = fn.data.ptr_val;
            task.item_type = static_cast<uint64_t>(items[i].type);
            task.item_data = items[i].data.raw_val;
            task.result_ptr = reinterpret_cast<uint64_t>(&results[i]);

            // Call LLVM worker via function pointer
            g_parallel_map_worker(&task);
        }
        *out_result = vector_to_list(results, arena); return;
    }

    // Get global thread pool
    eshkol_thread_pool_t* pool = thread_pool_global();
    if (!pool) {
        eshkol_error("parallel-map: failed to get thread pool");
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    // Create task data (decomposed to i64 fields - no struct-by-value ABI!)
    std::vector<llvm_parallel_map_task> tasks(n);
    std::vector<eshkol_future_t*> futures(n);

    for (size_t i = 0; i < n; ++i) {
        // Pack task: closure pointer, type as i64, data as i64, result pointer
        tasks[i].closure_ptr = fn.data.ptr_val;
        tasks[i].item_type = static_cast<uint64_t>(items[i].type);
        tasks[i].item_data = items[i].data.raw_val;
        tasks[i].result_ptr = reinterpret_cast<uint64_t>(&results[i]);
    }

    // Submit LLVM worker to thread pool via function pointer
    // Key: void* arg crosses C/LLVM boundary (safe), tagged values stay in LLVM
    eshkol_debug("parallel-map: submitting %zu tasks to thread pool", n);
    if (getenv("ESHKOL_DEBUG_PAR")) {
        eshkol_thread_pool_metrics_t m;
        thread_pool_get_metrics(pool, &m);
        fprintf(stderr, "[par-map] pool=%p threads=%zu  active=%zu  submitting=%zu tasks\n",
            (void*)pool, m.num_threads, m.active_workers, n);
    }
    for (size_t i = 0; i < n; ++i) {
        futures[i] = thread_pool_submit(pool, g_parallel_map_worker, &tasks[i]);
        if (!futures[i]) {
            eshkol_error("parallel-map: failed to submit task %zu", i);
        }
    }

    // Wait for all tasks to complete
    for (size_t i = 0; i < n; ++i) {
        if (futures[i]) {
            future_get(futures[i]);
            future_release(futures[i]);
        }
    }

    if (getenv("ESHKOL_DEBUG_PAR")) {
        eshkol_thread_pool_metrics_t m;
        thread_pool_get_metrics(pool, &m);
        fprintf(stderr, "[par-map] done.  submitted=%zu  completed=%zu  peak_q=%zu  avg_task=%lluus\n",
            m.tasks_submitted, m.tasks_completed, m.peak_queue_depth,
            (unsigned long long)m.avg_task_time_us);
    }

    eshkol_debug("parallel-map: all tasks completed, building result list");
    *out_result = vector_to_list(results, arena); return;
}

/* Pointer-argument entry point for LLVM codegen. */
extern "C" void eshkol_parallel_map_sret(
    eshkol_tagged_value_t* out_result,
    const eshkol_tagged_value_t* fn_ptr,
    const eshkol_tagged_value_t* list_ptr,
    arena_t* arena)
{
    if (!out_result) return;
    if (!fn_ptr || !list_ptr) {
        memset(out_result, 0, sizeof(*out_result));
        out_result->type = ESHKOL_VALUE_NULL;
        return;
    }
    eshkol_parallel_map(*fn_ptr, *list_ptr, arena, out_result);
}

// ============================================================================
// C API: Parallel Fold (Reduce)
// ============================================================================

/**
 * eshkol_parallel_fold - Reduction over a list
 *
 * @param fn        Binary function (fn acc item) -> new_acc
 * @param init      Initial accumulator value
 * @param list      Input list
 * @param arena     Arena for allocations (unused, kept for API consistency)
 * @return          Final accumulated value
 *
 * Note: Fold is inherently sequential for non-associative operations.
 * For parallelizable reductions, use parallel-map followed by fold.
 */
void eshkol_parallel_fold(
    eshkol_tagged_value_t fn,
    eshkol_tagged_value_t init,
    eshkol_tagged_value_t list,
    arena_t* arena,
    eshkol_tagged_value_t* out_result)
{
    (void)arena;  // Unused - closure dispatcher handles memory

    // Handle empty list
    if (list.type == ESHKOL_VALUE_NULL) {
        *out_result = init; return;
    }

    // Validate closure
    if (fn.type != ESHKOL_VALUE_CALLABLE) {
        eshkol_error("parallel-fold: expected callable, got type %d", fn.type);
        *out_result = init; return;
    }

    // Check if dispatcher is registered (with dlsym fallback)
    if (!g_call_binary_closure) eshkol_parallel_workers_lazy_resolve();
    if (!g_call_binary_closure) {
        eshkol_error("parallel-fold: binary closure dispatcher not registered (stdlib not loaded?)");
        *out_result = init; return;
    }

    // Sequential fold using LLVM-generated dispatcher
    eshkol_tagged_value_t acc = init;
    eshkol_tagged_value_t current = list;

    while (current.type != ESHKOL_VALUE_NULL) {
        if (current.type != ESHKOL_VALUE_HEAP_PTR) {
            break;
        }

        auto* cell = reinterpret_cast<arena_tagged_cons_cell_t*>(current.data.ptr_val);
        if (!cell) break;

        // Call fold function via function pointer: (fn acc item) -> new_acc
        acc = g_call_binary_closure(acc, cell->car, fn);
        current = cell->cdr;
    }

    *out_result = acc; return;
}

// ============================================================================
// C API: Parallel For-Each
// ============================================================================

/**
 * eshkol_parallel_for_each - Apply side-effecting function in parallel
 *
 * @param fn        Function to apply (for side effects)
 * @param list      Input list
 * @param arena     Arena for allocations
 *
 * Unlike parallel-map, this doesn't collect results.
 */
void eshkol_parallel_for_each(
    eshkol_tagged_value_t fn,
    eshkol_tagged_value_t list,
    arena_t* arena)
{
    // Use parallel-map and ignore results
    eshkol_tagged_value_t discard;
    eshkol_parallel_map(fn, list, arena, &discard);
}

// ============================================================================
// C API: Parallel Filter (Pure LLVM Workers)
// ============================================================================

/**
 * eshkol_parallel_filter - Filter list in parallel
 *
 * @param pred      Predicate function
 * @param list      Input list
 * @param arena     Arena for allocations
 * @return          New list with elements where pred returned true
 *
 * Architecture: Uses LLVM-generated __parallel_filter_worker (same struct as map).
 * The worker calls the predicate and stores the result (a boolean).
 * We then filter based on whether the result is truthy.
 */
void eshkol_parallel_filter(
    eshkol_tagged_value_t pred,
    eshkol_tagged_value_t list,
    arena_t* arena,
    eshkol_tagged_value_t* out_result)
{
    eshkol_debug("parallel-filter: pred.type=%d, list.type=%d", pred.type, list.type);

    if (list.type == ESHKOL_VALUE_NULL) {
        *out_result = list; return;
    }

    // Validate closure
    if (pred.type != ESHKOL_VALUE_CALLABLE) {
        eshkol_error("parallel-filter: expected callable, got type %d", pred.type);
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    auto items = list_to_vector(list);
    eshkol_debug("parallel-filter: items.size()=%zu", items.size());
    if (items.empty()) {
        *out_result = list; return;
    }

    size_t n = items.size();

    // Check if workers are registered (with dlsym fallback)
    if (!g_parallel_filter_worker) eshkol_parallel_workers_lazy_resolve();
    if (!g_parallel_filter_worker) {
        eshkol_error("parallel-filter: workers not registered (stdlib not loaded?)");
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    // Allocate predicate results (workers write results via pointer)
    std::vector<eshkol_tagged_value_t> pred_results(n);
    for (size_t i = 0; i < n; ++i) {
        pred_results[i].type = ESHKOL_VALUE_NULL;
        pred_results[i].flags = 0;
        pred_results[i].reserved = 0;
        pred_results[i].data.raw_val = 0;
    }

    // For small lists, use LLVM worker directly
    if (n < 4) {
        eshkol_debug("parallel-filter: sequential path for small list");
        for (size_t i = 0; i < n; ++i) {
            llvm_parallel_map_task task;
            task.closure_ptr = pred.data.ptr_val;
            task.item_type = static_cast<uint64_t>(items[i].type);
            task.item_data = items[i].data.raw_val;
            task.result_ptr = reinterpret_cast<uint64_t>(&pred_results[i]);

            g_parallel_filter_worker(&task);
        }

        // Collect items where predicate returned truthy
        std::vector<eshkol_tagged_value_t> filtered;
        for (size_t i = 0; i < n; ++i) {
            if (is_truthy(pred_results[i])) {
                filtered.push_back(items[i]);
            }
        }
        *out_result = vector_to_list(filtered, arena); return;
    }

    eshkol_thread_pool_t* pool = thread_pool_global();
    if (!pool) {
        eshkol_error("parallel-filter: failed to get thread pool");
        eshkol_tagged_value_t null_val = {};
        null_val.type = ESHKOL_VALUE_NULL;
        *out_result = null_val; return;
    }

    // Create task data (same struct as map - predicate is a unary function)
    std::vector<llvm_parallel_map_task> tasks(n);
    std::vector<eshkol_future_t*> futures(n);

    for (size_t i = 0; i < n; ++i) {
        tasks[i].closure_ptr = pred.data.ptr_val;
        tasks[i].item_type = static_cast<uint64_t>(items[i].type);
        tasks[i].item_data = items[i].data.raw_val;
        tasks[i].result_ptr = reinterpret_cast<uint64_t>(&pred_results[i]);
    }

    // Submit LLVM filter worker to thread pool via function pointer
    eshkol_debug("parallel-filter: submitting %zu tasks to thread pool", n);
    for (size_t i = 0; i < n; ++i) {
        futures[i] = thread_pool_submit(pool, g_parallel_filter_worker, &tasks[i]);
        if (!futures[i]) {
            eshkol_error("parallel-filter: failed to submit task %zu", i);
        }
    }

    // Wait for all tasks to complete
    for (size_t i = 0; i < n; ++i) {
        if (futures[i]) {
            future_get(futures[i]);
            future_release(futures[i]);
        }
    }

    // Collect items that passed the predicate (truthy results)
    std::vector<eshkol_tagged_value_t> filtered;
    for (size_t i = 0; i < n; ++i) {
        if (is_truthy(pred_results[i])) {
            filtered.push_back(items[i]);
        }
    }

    eshkol_debug("parallel-filter: filtered %zu items to %zu", n, filtered.size());
    *out_result = vector_to_list(filtered, arena); return;
}

// ============================================================================
// C API: Parallel Execute
// ============================================================================

/**
 * eshkol_parallel_execute - Execute N thunks in parallel, collect results
 *
 * @param thunks_ptr  Pointer to array of tagged values (each a zero-arg closure)
 * @param num_thunks  Number of thunks in the array
 * @param arena       Arena for building result list
 * @return            List of results in order matching input thunks
 *
 * Each thunk is a zero-argument closure. They are submitted to the thread pool
 * for parallel execution. Results are collected in order and returned as a list.
 */
void eshkol_parallel_execute(
    eshkol_tagged_value_t* thunks_ptr,
    int64_t num_thunks,
    arena_t* arena,
    eshkol_tagged_value_t* out_result)
{
    eshkol_debug("parallel-execute: num_thunks=%lld", (long long)num_thunks);

    eshkol_tagged_value_t null_val = {};
    null_val.type = ESHKOL_VALUE_NULL;
    null_val.flags = 0;
    null_val.reserved = 0;
    null_val.data.raw_val = 0;

    if (num_thunks <= 0) {
        *out_result = null_val; return;
    }

    if (!thunks_ptr) {
        eshkol_error("parallel-execute: null thunks array");
        *out_result = null_val; return;
    }

    // Check if execute worker is registered (with dlsym fallback)
    if (!g_parallel_execute_worker) eshkol_parallel_workers_lazy_resolve();
    if (!g_parallel_execute_worker) {
        eshkol_error("parallel-execute: execute worker not registered (stdlib not loaded?)");
        *out_result = null_val; return;
    }

    size_t n = static_cast<size_t>(num_thunks);

    // Allocate result storage (workers write results via pointer)
    std::vector<eshkol_tagged_value_t> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = null_val;
    }

    // Validate all thunks are callable
    for (size_t i = 0; i < n; ++i) {
        if (thunks_ptr[i].type != ESHKOL_VALUE_CALLABLE) {
            eshkol_error("parallel-execute: argument %zu is not callable (type=%d)", i, thunks_ptr[i].type);
            *out_result = null_val; return;
        }
    }

    // For a single thunk, execute sequentially (no parallelism benefit)
    if (n == 1) {
        eshkol_debug("parallel-execute: sequential path for single thunk");
        llvm_parallel_execute_task task;
        task.closure_ptr = thunks_ptr[0].data.ptr_val;
        task.closure_type = static_cast<uint64_t>(thunks_ptr[0].type);
        task.closure_flags = static_cast<uint64_t>(thunks_ptr[0].flags);
        task.result_ptr = reinterpret_cast<uint64_t>(&results[0]);
        g_parallel_execute_worker(&task);
        *out_result = vector_to_list(results, arena); return;
    }

    // Get global thread pool
    eshkol_thread_pool_t* pool = thread_pool_global();
    if (!pool) {
        // Fallback to sequential execution
        eshkol_warn("parallel-execute: no thread pool, falling back to sequential");
        for (size_t i = 0; i < n; ++i) {
            llvm_parallel_execute_task task;
            task.closure_ptr = thunks_ptr[i].data.ptr_val;
            task.closure_type = static_cast<uint64_t>(thunks_ptr[i].type);
            task.closure_flags = static_cast<uint64_t>(thunks_ptr[i].flags);
            task.result_ptr = reinterpret_cast<uint64_t>(&results[i]);
            g_parallel_execute_worker(&task);
        }
        *out_result = vector_to_list(results, arena); return;
    }

    // Create task data and submit to thread pool
    std::vector<llvm_parallel_execute_task> tasks(n);
    std::vector<eshkol_future_t*> futures(n);

    for (size_t i = 0; i < n; ++i) {
        tasks[i].closure_ptr = thunks_ptr[i].data.ptr_val;
        tasks[i].closure_type = static_cast<uint64_t>(thunks_ptr[i].type);
        tasks[i].closure_flags = static_cast<uint64_t>(thunks_ptr[i].flags);
        tasks[i].result_ptr = reinterpret_cast<uint64_t>(&results[i]);
    }

    eshkol_debug("parallel-execute: submitting %zu thunks to thread pool", n);
    for (size_t i = 0; i < n; ++i) {
        futures[i] = thread_pool_submit(pool, g_parallel_execute_worker, &tasks[i]);
        if (!futures[i]) {
            eshkol_error("parallel-execute: failed to submit thunk %zu to thread pool", i);
        }
    }

    // Wait for all thunks to complete
    for (size_t i = 0; i < n; ++i) {
        if (futures[i]) {
            future_get(futures[i]);
            future_release(futures[i]);
        }
    }

    eshkol_debug("parallel-execute: all %zu thunks completed, building result list", n);
    *out_result = vector_to_list(results, arena);
}

// ============================================================================
// C API: Thread Pool Info
// ============================================================================

/**
 * eshkol_thread_pool_num_threads - Get number of worker threads
 */
int64_t eshkol_thread_pool_num_threads(void) {
    eshkol_thread_pool_t* pool = thread_pool_global();
    return pool ? static_cast<int64_t>(thread_pool_num_threads(pool)) : 0;
}

/**
 * eshkol_thread_pool_print_stats - Print thread pool statistics
 */
void eshkol_thread_pool_print_stats(void) {
    eshkol_thread_pool_t* pool = thread_pool_global();
    if (pool) {
        thread_pool_print_metrics(pool);
    }
}

} // extern "C"
