# Parallel Computing in Eshkol

## Overview

Eshkol provides a structured concurrency model built on a work-stealing thread pool, per-thread arena allocation, and LLVM-generated closure dispatchers. The design separates concerns across three layers: a C runtime that manages threads and task scheduling (`parallel_codegen.cpp`), an LLVM code generation layer that handles closure calling conventions and tagged value marshalling (`parallel_llvm_codegen.cpp`), and a lock-free work-stealing deque based on the Chase-Lev algorithm (`work_stealing_deque.h`). This document describes the architecture, correctness properties, and performance characteristics of each layer.

---

## 1. Architecture

### 1.1 Work-Stealing Deque

The scheduler uses a Chase-Lev work-stealing deque as described in "Dynamic Circular Work-Stealing Deque" (Chase and Lev, 2005). Each worker thread owns a local deque with LIFO semantics for push and pop operations performed by the owner, and FIFO steal semantics for thief threads. The implementation resides in `inc/eshkol/backend/work_stealing_deque.h`.

The `WorkStealingDeque` class (line 199) maintains three cache-line-aligned atomic fields: `top_`, `bottom_`, and `array_`. The owner pushes items to the bottom and pops from the bottom (LIFO), preserving temporal locality. Thief threads steal from the top (FIFO), consuming the oldest tasks first. When the owner and a thief contend over the last item, the conflict is resolved via a compare-and-swap on `top_` (line 260). This yields O(1) amortized cost per push, pop, and steal.

The underlying `CircularArray` (line 149) uses power-of-two sizing with a bitmask for modular indexing. When the deque is full, the owner grows the array by doubling its capacity (line 180) and retiring the old array for epoch-based reclamation. Initial capacity is 4096 entries (log_initial_size = 12, per line 202).

### 1.2 Epoch-Based Memory Reclamation

Safe deque array reclamation uses an epoch manager (`EpochManager`, line 43). Each worker thread registers an epoch slot. Before accessing a deque, a worker enters a critical section by writing the current global epoch to its slot. The manager advances the global epoch periodically (every 256 completed tasks, line 254 in `thread_pool.cpp`) and determines the safe-to-reclaim epoch as two behind the minimum active epoch. Retired arrays whose epoch falls below this threshold are freed during `reclaimRetired()` (line 328).

### 1.3 Thread Pool Sizing

The global thread pool is lazily initialized as a singleton (`thread_pool_global`, line 530 in `thread_pool.cpp`). Thread count defaults to `std::thread::hardware_concurrency()`, with a fallback of 4 if the runtime cannot determine the value. The environment variable `ESHKOL_NUM_THREADS` may override this at pool creation time. Each thread receives a 1 MB local arena by default (`thread_arena_size`, line 49 in `thread_pool.h`).

Work-stealing mode is enabled by default. Setting the environment variable `ESHKOL_DISABLE_WORK_STEALING=1` falls back to a legacy shared-queue mode with mutex protection (line 422 in `thread_pool.cpp`).

### 1.4 Idle Backoff Strategy

When a worker finds no local work and fails to steal, it enters a three-stage backoff sequence (line 258 in `thread_pool.cpp`):

1. **Spin phase** (0--32 iterations): Architecture-specific pause instruction (`yield` on ARM64, `pause` on x86-64).
2. **Yield phase** (32--64 iterations): `std::this_thread::yield()`.
3. **Sleep phase** (beyond 64 iterations): `condition_variable::wait_for` with a 1 ms timeout, waking on new task submission or shutdown.

---

## 2. Memory Safety

### 2.1 Per-Thread Arena Isolation

Each worker thread maintains a thread-local arena (`tls_arena`, line 103 in `thread_pool.cpp`), lazily allocated on first use. Arena allocations during parallel task execution are confined to the executing thread's arena, eliminating cross-thread contention on the global allocator. When a worker shuts down, its arena is destroyed (line 291).

The global arena is used only for building the final result list after all parallel tasks have completed and their results have been collected. This ensures that result lists are allocated in the caller's memory region and remain valid after worker arenas are reset.

### 2.2 Void-Pointer ABI Boundary

A critical design constraint is that tagged values (`{i8, i8, i16, i32, i64}`) must never be passed by value across the C/LLVM boundary. Struct-by-value ABI divergences between optimization levels (documented in MEMORY.md under the REPL 3+ arg stdlib bug) cause silent data corruption on ARM64.

The solution is to decompose task data into `i64` fields in C-side structs (line 121 in `parallel_codegen.cpp`). The LLVM-generated worker functions receive a `void*` to this flat struct, reconstruct tagged values in pure LLVM IR, and call the closure dispatcher entirely within the LLVM domain. The call chain is:

```
C runtime --> thread pool --> LLVM worker (void*) --> LLVM dispatcher --> closure
```

Only `void*` crosses the C/LLVM boundary. All tagged value construction and destruction occurs within LLVM IR.

### 2.3 Copy-by-Value Captures

Closures passed to parallel primitives capture their environment by value at closure creation time. The parallel dispatcher extracts captures from the closure environment using pointer arithmetic within LLVM IR (line 299 in `parallel_llvm_codegen.cpp`). Because captures are read-only pointers into the environment, and each task operates on a distinct list element, there is no shared mutable state between concurrent tasks.

---

## 3. Data-Parallel Primitives

### 3.1 parallel-map

```scheme
(parallel-map f lst)
```

Applies a unary function `f` to each element of `lst` in parallel. Returns a new list preserving element order.

**Implementation.** The C runtime entry point is `eshkol_parallel_map` (line 245 in `parallel_codegen.cpp`). It converts the input list to a `std::vector`, allocates a result vector of the same size, and submits one task per element to the thread pool via the LLVM-generated `__parallel_map_worker` (line 775 in `parallel_llvm_codegen.cpp`). Each worker reconstructs the tagged value for its element, invokes `__eshkol_call_unary_closure`, and writes the result to a pre-allocated slot via pointer. After all futures complete, results are assembled into a cons list in the original order.

For lists with fewer than 4 elements, the sequential path is used (line 292 in `parallel_codegen.cpp`), calling the LLVM worker directly without thread pool submission.

The LLVM codegen path (`parallelMap`, line 1391 in `parallel_llvm_codegen.cpp`) generates an inline loop that builds the result list in reverse order, then calls `generateListReversal` (line 1283) to restore the correct order.

### 3.2 parallel-fold

```scheme
(parallel-fold f init lst)
```

Applies a binary function `f` as a left fold: `f(f(f(init, e1), e2), e3)`.

**Implementation.** The C runtime entry point is `eshkol_parallel_fold` (line 367 in `parallel_codegen.cpp`). The fold is sequential -- it walks the list from head to tail, calling `g_call_binary_closure(acc, item, fn)` at each step. The naming "parallel-fold" reflects its membership in the parallel primitives module rather than its execution model. Fold is inherently sequential for non-associative operations.

The LLVM codegen path (`parallelFold`, line 1550) generates an inline loop with PHI nodes for the current list element and accumulator, calling `__eshkol_call_binary_closure` at each iteration.

### 3.3 parallel-filter

```scheme
(parallel-filter pred lst)
```

Evaluates `pred` on each element concurrently and returns a list of elements for which `pred` returned a truthy value, preserving order.

**Implementation.** The C runtime entry point is `eshkol_parallel_filter` (line 439 in `parallel_codegen.cpp`). It follows the same pattern as parallel-map: submit one task per element using `__parallel_filter_worker` (line 1004 in `parallel_llvm_codegen.cpp`), collect boolean results, then filter the original elements. A value is truthy if it is neither `ESHKOL_VALUE_NULL` nor boolean false (line 151 in `parallel_codegen.cpp`).

The LLVM codegen path (`parallelFilter`, line 1667) generates an inline loop that tests each predicate result and conditionally conses the element onto the result list. As with parallel-map, the result is reversed after construction.

### 3.4 parallel-for-each

```scheme
(parallel-for-each f lst)
```

Applies `f` to each element for side effects. Return value is unspecified.

**Implementation.** The C runtime entry point is `eshkol_parallel_for_each` (line 425 in `parallel_codegen.cpp`), which delegates to `eshkol_parallel_map` and discards the result list.

---

## 4. Concurrent Execution and Futures

### 4.1 parallel-execute

```scheme
(parallel-execute thunk1 thunk2 ...)
```

Evaluates zero-argument closures (thunks) concurrently on the thread pool and returns a list of their results in argument order.

**Implementation.** The entry point is `eshkol_parallel_execute` (line 580 in `parallel_codegen.cpp`). Thunks are passed as an array of tagged values. Each is submitted to the thread pool via `__parallel_execute_worker` (line 1108 in `parallel_llvm_codegen.cpp`), which calls `__eshkol_call_nullary_closure` (line 210). Results are collected into a list. For a single thunk, the sequential path avoids thread pool overhead (line 625).

### 4.2 future / force / future-ready?

```scheme
(define f (future (lambda () (expensive-computation))))
(future-ready? f)  ; => #f (non-blocking check)
(force f)          ; => result (blocks until computed, then caches)
(force f)          ; => result (returns cached value immediately)
```

**Implementation.** Futures use a lazy evaluation model implemented with the `eshkol_lazy_future` struct (line 835 in `thread_pool.cpp`). The `future` codegen (line 2103 in `parallel_llvm_codegen.cpp`) checks whether its argument is callable. If callable, it stores the thunk pointer for deferred evaluation. If the argument is a plain value (not a closure), the future is marked as immediately resolved.

The `force` codegen (line 2188) checks the `forced` flag. If unforced, it reconstructs the thunk as a tagged value, calls it via `__eshkol_call_nullary_closure`, stores the result components, and marks the future as forced. Subsequent `force` calls return the cached result directly. The `force` operation is idempotent.

The `future-ready?` codegen (line 2343) calls `eshkol_lazy_future_is_ready` (line 861 in `thread_pool.cpp`) and returns a tagged boolean. This is a non-blocking check.

---

## 5. Correctness

### 5.1 Associativity Requirement

The `parallel-fold` primitive applies its combining function sequentially from left to right. If future implementations introduce tree-structured parallel reduction, the combining function must be associative to guarantee identical results. Commutativity is not required -- the fold respects list order regardless of scheduling.

### 5.2 Determinism Guarantees

- **parallel-map** and **parallel-filter**: Deterministic output. Although tasks may execute in arbitrary order, results are collected into pre-indexed slots and assembled in the original list order.
- **parallel-fold**: Deterministic. Sequential execution preserves left-to-right evaluation order.
- **parallel-execute**: Deterministic result list order (matches argument order). However, side effects within thunks may interleave non-deterministically.
- **parallel-for-each**: Side effects may execute in any order. No ordering guarantees.

### 5.3 Exception Safety

Worker threads catch both `std::exception` and unknown exceptions (line 214 in `thread_pool.cpp`). A task exception is logged but does not crash the pool. The future for a failed task receives a null result. The thread pool continues processing subsequent tasks.

---

## 6. Performance

### 6.1 Chunk Sizing

The C runtime submits one task per list element. For large lists, this creates fine-grained tasks that are individually cheap relative to thread pool overhead. The work-stealing deque amortizes this cost: the submitting thread pushes all tasks to its local deque in O(n) total, and idle workers steal in batches. Batch submission via `thread_pool_submit_batch` (line 613 in `thread_pool.cpp`) distributes tasks round-robin across deques for better initial load balance.

For small lists (fewer than 4 elements), the sequential path avoids thread pool submission entirely (line 292 in `parallel_codegen.cpp`), calling the LLVM worker function directly.

### 6.2 Nested Parallelism

Parallel primitives called from within a parallel task reuse the same global thread pool. Tasks from inner parallel calls are pushed to the calling worker's local deque and may be stolen by other workers. This provides natural work distribution for nested parallelism without requiring a separate thread pool. However, deeply nested parallel calls can saturate the deque and cause contention.

### 6.3 Work-Stealing Metrics

The thread pool tracks performance counters atomically (line 81 in `thread_pool.cpp`):

- `tasks_submitted`, `tasks_completed`: Total task throughput.
- `local_executions`: Tasks executed by the thread that created them.
- `tasks_stolen`: Tasks executed by a different thread via stealing.
- `steal_attempts`: Total steal operations attempted.

The steal ratio (`tasks_stolen / (local_executions + tasks_stolen)`) indicates load balance quality. A ratio near zero suggests well-distributed work; a high ratio suggests one thread is producing most of the work. Metrics are printed at pool shutdown or on demand via `(thread-pool-stats)`.

---

## 7. LLVM Code Generation

### 7.1 Closure Dispatchers

The core challenge in parallel execution is calling Eshkol closures from worker threads. Closures have a dynamic number of captures (0 to 32), and each capture count requires a distinct function signature. The LLVM codegen generates three dispatcher functions that handle this polymorphism:

- `__eshkol_call_nullary_closure(closure)` -- for thunks (line 210 in `parallel_llvm_codegen.cpp`).
- `__eshkol_call_unary_closure(item, closure)` -- for map and filter (line 376).
- `__eshkol_call_binary_closure(arg1, arg2, closure)` -- for fold (line 546).

Each dispatcher extracts the capture count from the closure environment's `packed_info` field (lower 16 bits, line 288), then uses a switch statement over 0 to `MAX_CAPTURES` (32) to dispatch to the correct calling convention. For each case, the dispatcher constructs the argument list as `(args..., &cap[0], &cap[1], ..., &cap[N-1])` and calls the function pointer stored in the closure struct.

### 7.2 Worker Function Compilation

Four LLVM worker functions are generated with `LinkOnceODRLinkage`:

- `__parallel_map_worker` (line 775): Unpacks `parallel_map_task`, reconstructs item and closure tagged values, calls unary dispatcher, stores result.
- `__parallel_fold_worker` (line 888): Same pattern with `parallel_fold_task` and binary dispatcher.
- `__parallel_filter_worker` (line 1004): Identical to map worker (predicates are unary functions).
- `__parallel_execute_worker` (line 1108): Unpacks execute task, calls nullary dispatcher for thunks.

`LinkOnceODRLinkage` ensures that when both stdlib.o and user code emit these symbols, the linker keeps exactly one definition. This avoids duplicate symbol errors while allowing both modules to reference the workers.

### 7.3 Worker Registration

Workers are registered with the C runtime at module load time via `llvm.global_ctors` (line 1997 in `parallel_llvm_codegen.cpp`). The generated initializer `__eshkol_init_parallel_workers` calls `__eshkol_register_parallel_workers` (line 88 in `parallel_codegen.cpp`) with pointers to all generated functions. This breaks the circular dependency between the C runtime (which needs to call workers) and the LLVM module (which needs to call C runtime functions).

The registration function stores worker pointers in static globals (`g_parallel_map_worker`, etc., line 60 in `parallel_codegen.cpp`). The C runtime checks `eshkol_parallel_workers_registered()` (line 110) before dispatching, returning an error if workers have not been registered (typically indicating that stdlib was not loaded).

---

## 8. Code Examples

### Basic parallel-map

```scheme
;; Square each element in parallel
(define result (parallel-map (lambda (x) (* x x)) '(1 2 3 4 5 6 7 8)))
(display result)  ; => (1 4 9 16 25 36 49 64)
```

### Parallel filtering

```scheme
;; Keep only even numbers, predicate evaluation is concurrent
(define evens (parallel-filter even? '(1 2 3 4 5 6 7 8 9 10)))
(display evens)  ; => (2 4 6 8 10)
```

### Parallel fold (sequential reduction)

```scheme
;; Sum a list -- fold is sequential, combining function is (+)
(define total (parallel-fold + 0 '(1 2 3 4 5)))
(display total)  ; => 15
```

### Concurrent thunk execution

```scheme
;; Execute three independent computations concurrently
(define results
  (parallel-execute
    (lambda () (fib 35))
    (lambda () (fib 36))
    (lambda () (fib 37))))
(display results)  ; => (9227465 14930352 24157817)
```

### Lazy futures

```scheme
;; Create a deferred computation
(define f (future (lambda () (heavy-computation))))

;; Check without blocking
(display (future-ready? f))  ; => #f

;; Force evaluation (blocks, caches result)
(define val (force f))

;; Subsequent forces return cached result
(assert (= val (force f)))
```

### Thread pool introspection

```scheme
;; Query available parallelism
(display (thread-pool-info))  ; => 8 (on an 8-core machine)

;; Print detailed metrics
(thread-pool-stats)
;; Output:
;;   Mode: work-stealing
;;   Threads: 8
;;   Tasks: submitted=1000, completed=1000, pending=0
;;   Local executions: 800
;;   Tasks stolen: 200
;;   Steal ratio: 20.0%
```

---

## 9. Per-Thread Arena Isolation

### 9.1 Thread-Local Storage Layout

Each worker thread maintains five `thread_local` variables declared at file scope in `thread_pool.cpp` (lines 103--107):

```cpp
thread_local arena_t* tls_arena = nullptr;
thread_local bool tls_is_worker_thread = false;
thread_local size_t tls_arena_size = 0;
thread_local size_t tls_worker_id = SIZE_MAX;
thread_local size_t tls_epoch_slot = SIZE_MAX;
```

When a worker thread starts in `work_stealing_worker_func` (line 140), it initializes `tls_is_worker_thread = true`, `tls_arena_size = pool->thread_arena_size`, and `tls_worker_id = worker_id`. The arena itself (`tls_arena`) is **not** created at this point -- it is lazily allocated on first use.

### 9.2 Lazy Arena Creation

The function `get_thread_local_arena()` (line 109 in `thread_pool.cpp`) creates the per-thread arena on demand:

```cpp
arena_t* get_thread_local_arena(void) {
    if (!tls_arena) {
        size_t size = tls_arena_size > 0 ? tls_arena_size : (1024 * 1024);
        tls_arena = arena_create(size);
    }
    return tls_arena;
}
```

The arena is created via `arena_create()`, **not** `arena_create_threadsafe()`. This is intentional: each thread-local arena is accessed exclusively by its owning thread, so the mutex overhead of `arena_create_threadsafe()` (which wraps `arena_create` and adds a pthread mutex, per line 181 in `arena_memory.cpp`) is unnecessary. Only arenas shared between threads -- such as the global arena `__global_arena`, which is created via `arena_create_threadsafe()` (line 1526 in `arena_memory.cpp`) -- require the thread-safe variant.

A second accessor, `thread_pool_get_thread_arena()` (line 123), guards against non-worker callers by checking `tls_is_worker_thread` and returning `nullptr` if the caller is not a pool worker.

### 9.3 Default Arena Size and Configuration

The default per-thread arena size is 1 MB (1,048,576 bytes), specified in `ESHKOL_THREAD_POOL_DEFAULT_CONFIG` (line 46 in `thread_pool.h`):

```c
#define ESHKOL_THREAD_POOL_DEFAULT_CONFIG { \
    .num_threads = 0,                       \
    .task_queue_capacity = 0,               \
    .thread_arena_size = 1024 * 1024,       \
    .enable_metrics = true,                 \
    .name = "default"                       \
}
```

This value is propagated to each worker via `pool->thread_arena_size` (line 412 in `thread_pool.cpp`). If the config's `thread_arena_size` is zero, the pool creation code defaults to 1 MB (line 414). Custom sizes can be passed through the `eshkol_thread_pool_config_t` struct when constructing a pool via `thread_pool_create()`.

### 9.4 Arena Reset and Destruction

Workers can reset their arena mid-flight via `thread_pool_reset_thread_arena()` (line 130 in `thread_pool.cpp`), which calls `arena_reset(tls_arena)`. This releases all allocations within the arena without destroying the arena itself, allowing it to be reused for the next task. On worker shutdown (line 291), the arena is fully destroyed via `arena_destroy(tls_arena)` and the pointer is nulled.

### 9.5 Global Arena for Result Marshaling

An important distinction: the current parallel primitives (`eshkol_parallel_map`, `eshkol_parallel_filter`, etc.) use the **global arena** for constructing the final result list, not thread-local arenas. The `arena` parameter passed to `eshkol_parallel_map` (line 245 in `parallel_codegen.cpp`) is the caller's global arena. The function `vector_to_list` (line 186 in `parallel_codegen.cpp`) calls `arena_allocate_tagged_cons_cell(arena)` on this global arena to build cons cells for the result list.

Workers write their per-element results into a pre-allocated `std::vector<eshkol_tagged_value_t>` (line 275 in `parallel_codegen.cpp`), which is stack-allocated in the calling thread. The result vector slots are filled by workers via pointer (each task's `result_ptr` field), avoiding any arena allocation during the parallel phase. Arena allocation occurs only during the sequential result-assembly phase after all futures have completed.

This means that thread-local arenas are currently available for use by closure bodies executing within workers (if those closures allocate via `get_thread_local_arena()`), but the parallel primitives' own marshaling infrastructure does not depend on them.

---

## 10. Problem Size Thresholds

### 10.1 Small List Optimization

Lists with fewer than 4 elements bypass the thread pool entirely. This threshold appears in all parallel primitives that use the pool:

- `eshkol_parallel_map` (line 292 in `parallel_codegen.cpp`): `if (n < 4)`, calls `g_parallel_map_worker` directly in a sequential loop.
- `eshkol_parallel_filter` (line 495 in `parallel_codegen.cpp`): `if (n < 4)`, calls `g_parallel_filter_worker` directly.
- `eshkol_parallel_execute` (line 625 in `parallel_codegen.cpp`): `if (n == 1)`, calls `g_parallel_execute_worker` directly for single-thunk invocations.

The threshold of 4 reflects the minimum list size at which thread pool submission overhead is amortized. Each task submission involves: allocating an `eshkol_task` and `eshkol_future` (two `new` calls), a deque push (atomic operations on `top_`/`bottom_`), a condition variable notify, and future completion synchronization. For closures that execute in microseconds (e.g., arithmetic), this overhead dominates unless there are enough tasks for work-stealing to amortize it.

### 10.2 Task Granularity

The C runtime submits **one task per list element** (line 332 in `parallel_codegen.cpp`). There is no chunk-size partitioning -- each element gets its own future. For large lists, this generates many fine-grained tasks, but the work-stealing deque handles this efficiently: the submitting thread pushes all tasks to a single deque in O(n), and idle workers steal in bulk. Batch submission via `thread_pool_submit_batch` (line 613 in `thread_pool.cpp`) distributes tasks round-robin across deques for better initial load balance.

### 10.3 Sequential Fallback for Fold

`eshkol_parallel_fold` (line 367 in `parallel_codegen.cpp`) is always sequential. It walks the list from head to tail, calling `g_call_binary_closure(acc, cell->car, fn)` at each step. The fold does not submit any tasks to the thread pool. This is inherently correct for non-associative combining functions, and avoids the complexity of tree-structured parallel reduction. The "parallel" in the name reflects the function's membership in the parallel primitives module, not its execution model.

### 10.4 Thread Count Tuning

The environment variable `ESHKOL_NUM_THREADS` overrides the thread count at pool creation time. The pool's default thread count is `std::thread::hardware_concurrency()` (line 406 in `thread_pool.cpp`), with a fallback of 4 if the runtime cannot detect CPU count (line 408). The environment variable is checked indirectly -- the default config sets `num_threads = 0`, which triggers auto-detection. Users can also disable work-stealing entirely via `ESHKOL_DISABLE_WORK_STEALING=1` (line 422), which falls back to the legacy shared-queue mode with mutex protection.

---

## 11. Exception Propagation in Parallel Workers

### 11.1 Worker Exception Handling

Worker threads wrap task execution in a `try`/`catch` block (lines 214--222 in `thread_pool.cpp`):

```cpp
void* result = nullptr;
try {
    result = task->fn(task->arg);
} catch (const std::exception& e) {
    eshkol_error("[%s] Worker %zu: Task exception: %s",
                pool->name.c_str(), worker_id, e.what());
} catch (...) {
    eshkol_error("[%s] Worker %zu: Unknown exception",
                pool->name.c_str(), worker_id);
}
```

If a worker task throws, the exception is caught, an error is logged via `eshkol_error`, and execution continues. The `result` variable remains `nullptr`. The legacy queue worker (line 350) has identical exception handling.

### 11.2 Future Result on Failure

After the catch block, the worker unconditionally completes the future (lines 237--243):

```cpp
if (task->future) {
    std::lock_guard<std::mutex> lock(task->future->mutex);
    task->future->result = result;
    task->future->completed = true;
    task->future->cv.notify_all();
}
```

When a task throws, `result` is `nullptr`, so `future->result` is set to `nullptr`. The calling thread's `future_get()` (line 752) unblocks and returns `nullptr`. For `eshkol_parallel_map`, this manifests as a `ESHKOL_VALUE_NULL` element in the result list (since the result slot was pre-initialized to null at line 277 in `parallel_codegen.cpp`).

There is no mechanism to propagate the original exception object to the caller. The exception is consumed within the worker thread. The caller sees a null result but no exception type or message.

### 11.3 Pending Task Behavior

A task failure does not cancel or abort other pending tasks. The thread pool continues executing remaining tasks in its deques. Each task is independent -- there is no task graph or dependency tracking. In `eshkol_parallel_map`, all futures are waited on sequentially regardless of individual failures (lines 340--345 in `parallel_codegen.cpp`).

### 11.4 Future Synchronization

The `future_get` function (line 752 in `thread_pool.cpp`) blocks indefinitely until the task completes:

```cpp
void* future_get(eshkol_future_t* future) {
    if (!future) return nullptr;
    std::unique_lock<std::mutex> lock(future->mutex);
    future->cv.wait(lock, [future]() { return future->completed; });
    return future->result;
}
```

There is no timeout on `future_get`. For timed waits, `future_wait` (line 760) accepts a timeout in milliseconds and returns `false` if the deadline expires. The `future_wait_all` function (line 808) distributes a global timeout across multiple futures, computing remaining time for each successive wait. `future_wait_any` (line 790) polls all futures in a loop with 1 ms sleep intervals.

### 11.5 Eshkol Runtime Errors

Eshkol's runtime `raise` function (used for Scheme-level errors like division by zero, type mismatches, or bounds violations) calls `longjmp` or `exit` depending on context. If a parallel worker triggers such an error, the behavior depends on the error path:

- `eshkol_error` (logging): Writes to stderr and returns. The worker continues; the task result is whatever was computed before the error.
- `exit()` or `abort()`: Terminates the entire process, including all worker threads.
- `longjmp` to an error handler: Undefined behavior if the jump target is on a different thread's stack. The parallel primitives do not install per-worker error handlers.

This means that Scheme-level errors within parallel-mapped closures that call `error` or `raise` may terminate the process rather than gracefully reporting per-task failures.

---

## 12. AD Tape Isolation

### 12.1 Per-Thread Tape Stack

The AD (automatic differentiation) tape state uses `thread_local` storage in `arena_memory.cpp` (lines 88--104):

```c
#define MAX_TAPE_DEPTH 32
thread_local ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __ad_tape_depth = 0;

// Double backward storage
thread_local void* __outer_ad_node_storage = nullptr;
thread_local void* __outer_ad_node_to_inner = nullptr;
thread_local void* __outer_grad_accumulator = nullptr;
thread_local void* __inner_var_node_ptr = nullptr;
thread_local uint64_t __gradient_x_degree = 0;

// N-dimensional derivatives
thread_local void* __outer_ad_node_stack[MAX_TAPE_DEPTH] = {nullptr};
thread_local uint64_t __outer_ad_node_depth = 0;
```

`MAX_TAPE_DEPTH` is 32 (matching `CodegenContext::MAX_TAPE_DEPTH` in `codegen_context.h`, line 54). Each thread in the thread pool gets its own copy of these variables, ensuring that parallel workers performing gradient computation do not corrupt each other's tape state. The LLVM codegen emits overflow checks: if `__ad_tape_depth >= MAX_TAPE_DEPTH`, the program aborts with a "Tape stack overflow" error (line 3157 in `autodiff_codegen.cpp`).

### 12.2 Gradient Computation in Parallel Closures

When a closure passed to `parallel-map` contains a `gradient` call, each worker thread records operations on its own per-thread tape. The LLVM-generated code for `gradient` pushes a new tape context onto `__ad_tape_stack[__ad_tape_depth]` (line 3183 in `autodiff_codegen.cpp`), performs the forward pass with tape recording, then backpropagates and pops the tape. Since `__ad_tape_stack` and `__ad_tape_depth` are `thread_local`, workers operating concurrently on different list elements maintain fully independent tape contexts.

### 12.3 The `__ad_mode_active` Global Flag Issue

The tape stack is thread-local, but two critical AD state variables are **not** thread-local -- they are plain global variables in `arena_memory.cpp` (lines 37, 41):

```c
ad_tape_t* __current_ad_tape = nullptr;   // line 37
bool __ad_mode_active = false;            // line 41
```

The comment at line 35 explains why: "Not thread_local because REPL JIT resolves these via dlsym/ADD_DATA_SYMBOL." In REPL mode, `__ad_mode_active` must be visible across multiple JIT-compiled modules so that lambdas from one module can observe AD mode set by another (line 400 in `repl_jit.cpp`).

This creates a race condition when `parallel-map` maps a gradient-computing closure. If worker thread A enters a gradient context and sets `__ad_mode_active = true`, worker thread B may see this flag and begin recording operations to a tape it did not create, or conversely, worker A may exit the gradient context and set `__ad_mode_active = false` while worker B is mid-computation.

### 12.4 Practical Implications

In the LLVM codegen layer, `__ad_mode_active` is accessed as a global variable via `CreateLoad` (line 19994 in `llvm_codegen.cpp`). Arithmetic operations check this flag at runtime to decide whether to create AD tape nodes. The race on `__ad_mode_active` means:

1. **Spurious tape recording**: A worker not in AD mode may see `__ad_mode_active == true` from another worker's gradient context, causing it to record operations to a potentially null or stale `__current_ad_tape`.
2. **Missed tape recording**: A worker in AD mode may see `__ad_mode_active == false` from another worker's exit, causing it to skip recording operations.
3. **Practical mitigation**: The tape stack (`__ad_tape_stack`, `__ad_tape_depth`) is thread-local and correctly isolates tape data. The flag issue primarily affects whether arithmetic operations attempt to create AD nodes. For correct parallel autodiff, each worker should be independently in AD mode with its own tape -- which the tape stack supports, but the global flag does not.

For production use, parallel autodiff is safe when gradient computations are confined to closures that manage their own tape push/pop lifecycle (which the `gradient` operator does), and where the `__ad_mode_active` flag race does not cause null pointer dereferences (the codegen checks `__current_ad_tape` before recording). The comment at line 36 in `arena_memory.cpp` acknowledges this: "For parallel-map with AD, the codegen should use per-task tape allocation."

---

## Source File Reference

| File | Lines | Role |
|------|-------|------|
| `lib/backend/parallel_codegen.cpp` | 705 | C runtime: task dispatch, list conversion, parallel-map/-fold/-filter/-execute |
| `lib/backend/parallel_llvm_codegen.cpp` | 2,401 | LLVM codegen: closure dispatchers, workers, inline loops, futures |
| `lib/backend/thread_pool.cpp` | 1,132 | Thread pool: work-stealing scheduler, futures, metrics, lazy future helpers |
| `inc/eshkol/backend/thread_pool.h` | 371 | Thread pool API: C and C++ interfaces, configuration, metrics struct |
| `inc/eshkol/backend/work_stealing_deque.h` | 667 | Chase-Lev deque, epoch-based reclamation, work-stealing scheduler |
