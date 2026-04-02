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

    // Check if workers are registered
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

    eshkol_debug("parallel-map: all tasks completed, building result list");
    *out_result = vector_to_list(results, arena); return;
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

    // Check if dispatcher is registered
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

    // Check if workers are registered
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

    // Check if execute worker is registered
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
