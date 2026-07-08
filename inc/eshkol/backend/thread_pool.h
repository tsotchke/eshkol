/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Thread Pool for Eshkol Parallel Execution
 *
 * Provides a work-queue based thread pool with support for:
 * - Task submission with futures
 * - Thread-local arena allocation
 * - Performance metrics and debugging
 */

#ifndef ESHKOL_THREAD_POOL_H
#define ESHKOL_THREAD_POOL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque handle to a thread pool instance. */
typedef struct eshkol_thread_pool eshkol_thread_pool_t;
/** @brief Opaque handle to a pending/completed task's result. */
typedef struct eshkol_future eshkol_future_t;
/** @brief Opaque handle to a queued task (internal to the pool implementation). */
typedef struct eshkol_task eshkol_task_t;

/** @brief Task function signature: takes user data, returns a result pointer. */
typedef void* (*eshkol_task_fn)(void* arg);

// ============================================================================
// Thread Pool Configuration
// ============================================================================

/** @brief Configuration used to construct a thread pool. */
typedef struct eshkol_thread_pool_config {
    size_t num_threads;           // Number of worker threads (0 = hardware_concurrency)
    size_t task_queue_capacity;   // Max pending tasks (0 = unlimited)
    size_t thread_arena_size;     // Size of each thread's local arena (default 1MB)
    bool enable_metrics;          // Track performance metrics
    const char* name;             // Pool name for debugging (optional)
} eshkol_thread_pool_config_t;

// Default configuration
#define ESHKOL_THREAD_POOL_DEFAULT_CONFIG { \
    .num_threads = 0,                       \
    .task_queue_capacity = 0,               \
    .thread_arena_size = 1024 * 1024,       \
    .enable_metrics = true,                 \
    .name = "default"                       \
}

// ============================================================================
// Thread Pool Lifecycle
// ============================================================================

/** @brief Create a thread pool with an explicit configuration. */
eshkol_thread_pool_t* thread_pool_create(const eshkol_thread_pool_config_t* config);

/** @brief Create a thread pool using ESHKOL_THREAD_POOL_DEFAULT_CONFIG. */
eshkol_thread_pool_t* thread_pool_create_default(void);

/** @brief Destroy a thread pool. Waits for all queued/running tasks to complete first. */
void thread_pool_destroy(eshkol_thread_pool_t* pool);

/** @brief Get the process-wide thread pool, lazily created on first call. */
eshkol_thread_pool_t* thread_pool_global(void);

/** @brief Shut down the global thread pool. Call at program exit. */
void thread_pool_global_shutdown(void);

// ============================================================================
// Task Submission
// ============================================================================

/**
 * @brief Submit a task and get a future for its result.
 * @param pool Thread pool to submit to
 * @param fn Task function to run
 * @param arg Argument passed to fn
 * @return A future handle, or NULL on failure (queue full, pool shutting down)
 */
eshkol_future_t* thread_pool_submit(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg
);

/**
 * @brief Submit multiple tasks in one batch call (more efficient than repeated thread_pool_submit).
 * @param pool Thread pool to submit to
 * @param fns Array of task functions, one per task
 * @param args Array of arguments, one per task
 * @param futures Output array of future handles, one per task
 * @param count Number of tasks in the batch
 * @return Number of tasks successfully submitted
 */
size_t thread_pool_submit_batch(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn* fns,
    void** args,
    eshkol_future_t** futures,
    size_t count
);

/**
 * @brief Submit a task without creating a future (fire-and-forget).
 * @param pool Thread pool to submit to
 * @param fn Task function to run
 * @param arg Argument passed to fn
 * @return true if the task was queued, false on failure
 */
bool thread_pool_submit_detached(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg
);

// ============================================================================
// Future Operations
// ============================================================================

/**
 * @brief Block until a future's task completes and return its result.
 * @param future The future to wait on
 * @return The task's return value
 */
void* future_get(eshkol_future_t* future);

/**
 * @brief Wait for a future with a timeout.
 * @param future The future to wait on
 * @param timeout_ms Maximum time to wait, in milliseconds
 * @return true if the task completed, false if the wait timed out
 */
bool future_wait(eshkol_future_t* future, uint64_t timeout_ms);

/**
 * @brief Non-blocking check of whether a future's task has completed.
 * @param future The future to check
 * @return true if the task has completed
 */
bool future_is_ready(const eshkol_future_t* future);

/**
 * @brief Release a future's resources.
 * Must be called after future_get() or when abandoning the future.
 * @param future The future to release
 */
void future_release(eshkol_future_t* future);

/**
 * @brief Wait for the first of several futures to complete.
 * @param futures Array of future handles
 * @param count Number of futures in the array
 * @param timeout_ms Maximum time to wait, in milliseconds
 * @return Index of the first completed future, or -1 on timeout
 */
int future_wait_any(eshkol_future_t** futures, size_t count, uint64_t timeout_ms);

/**
 * @brief Wait for all of several futures to complete.
 * @param futures Array of future handles
 * @param count Number of futures in the array
 * @param timeout_ms Maximum time to wait, in milliseconds
 * @return true if all futures completed, false if the wait timed out
 */
bool future_wait_all(eshkol_future_t** futures, size_t count, uint64_t timeout_ms);

// ============================================================================
// Thread Pool Metrics
// ============================================================================

/** @brief Snapshot of a thread pool's performance counters. */
typedef struct eshkol_thread_pool_metrics {
    size_t tasks_submitted;       // Total tasks submitted
    size_t tasks_completed;       // Total tasks completed
    size_t tasks_pending;         // Currently pending tasks
    size_t active_workers;        // Currently active workers
    size_t peak_queue_depth;      // Maximum queue depth observed
    uint64_t total_task_time_us;  // Total time spent executing tasks
    uint64_t total_wait_time_us;  // Total time tasks spent waiting
    uint64_t avg_task_time_us;    // Average task execution time
    uint64_t avg_wait_time_us;    // Average wait time in queue
    size_t num_threads;           // Number of worker threads
} eshkol_thread_pool_metrics_t;

/**
 * @brief Read a thread pool's current performance metrics.
 * @param pool Thread pool to query
 * @param metrics Output metrics struct to fill
 */
void thread_pool_get_metrics(
    const eshkol_thread_pool_t* pool,
    eshkol_thread_pool_metrics_t* metrics
);

/** @brief Reset a thread pool's metrics counters to zero (useful for benchmarking). */
void thread_pool_reset_metrics(eshkol_thread_pool_t* pool);

/** @brief Print a thread pool's current metrics to stderr. */
void thread_pool_print_metrics(const eshkol_thread_pool_t* pool);

// ============================================================================
// Thread Pool Control
// ============================================================================

/** @brief Get the number of worker threads in a pool. */
size_t thread_pool_num_threads(const eshkol_thread_pool_t* pool);

/** @brief Pause all workers. Tasks already running are allowed to complete. */
void thread_pool_pause(eshkol_thread_pool_t* pool);

/** @brief Resume workers previously paused with thread_pool_pause. */
void thread_pool_resume(eshkol_thread_pool_t* pool);

/** @brief Block until all currently queued/running tasks complete. */
void thread_pool_wait_idle(eshkol_thread_pool_t* pool);

// ============================================================================
// Thread-Local Arena Access
// ============================================================================

/**
 * @brief Get the calling worker thread's thread-local arena.
 * @return The arena, or NULL if called from a non-worker thread
 */
struct arena* thread_pool_get_thread_arena(void);

/**
 * @brief Get or lazily create a thread-local arena for the calling thread.
 * Unlike thread_pool_get_thread_arena, this works from any thread, not just workers.
 */
struct arena* get_thread_local_arena(void);

/** @brief Reset the calling thread's thread-local arena. Call after completing independent work. */
void thread_pool_reset_thread_arena(void);

#ifdef __cplusplus
} // extern "C"

// ============================================================================
// C++ Interface
// ============================================================================

#include <functional>
#include <future>
#include <memory>
#include <vector>

namespace eshkol {

/**
 * @brief RAII / template-friendly C++ wrapper around the eshkol_thread_pool_t C API.
 *
 * Provides std::future-based task submission and STL-style parallel algorithms
 * (parallelFor, parallelMap, parallelReduce) on top of the same underlying pool.
 */
class ThreadPool {
public:
    // Constructors

    /** @brief Construct a pool sized to hardware_concurrency with default settings. */
    ThreadPool();
    /** @brief Construct a pool with an explicit worker-thread count. */
    explicit ThreadPool(size_t num_threads);
    /** @brief Construct a pool from an explicit C configuration struct. */
    explicit ThreadPool(const eshkol_thread_pool_config_t& config);
    /** @brief Destroy the pool, waiting for outstanding tasks to complete (if owned). */
    ~ThreadPool();

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Movable
    ThreadPool(ThreadPool&& other) noexcept;
    ThreadPool& operator=(ThreadPool&& other) noexcept;

    /**
     * @brief Submit a callable and get a std::future for its result.
     * @param f Callable to invoke on a worker thread
     * @param args Arguments forwarded to f
     * @return A std::future resolving to f's return value
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    /**
     * @brief Submit a callable without tracking its result (fire-and-forget).
     * @param f Callable to invoke on a worker thread
     * @param args Arguments forwarded to f
     */
    template<typename F, typename... Args>
    void submitDetached(F&& f, Args&&... args);

    /**
     * @brief Apply f to every element in [begin, end) in parallel, blocking until all complete.
     * @param begin Iterator to the first element
     * @param end Iterator past the last element
     * @param f Function applied to each dereferenced element
     */
    template<typename Iterator, typename Func>
    void parallelFor(Iterator begin, Iterator end, Func&& f);

    /**
     * @brief Apply f to each element of input in parallel and collect the results.
     * @param input Elements to map over
     * @param f Mapping function applied to each element
     * @return Vector of results in the same order as input
     */
    template<typename T, typename Func>
    std::vector<T> parallelMap(const std::vector<T>& input, Func&& f);

    /**
     * @brief Map input with map_fn in parallel, then sequentially fold the results with reduce_fn.
     * @param input Elements to map over
     * @param init Initial accumulator value
     * @param map_fn Mapping function applied to each element (in parallel)
     * @param reduce_fn Binary reduction function folding (accumulator, mapped value) -> accumulator
     * @return The final reduced value
     */
    template<typename T, typename Func, typename Reducer>
    T parallelReduce(const std::vector<T>& input, T init, Func&& map_fn, Reducer&& reduce_fn);

    // Control

    /** @brief Pause all workers. Tasks already running are allowed to complete. */
    void pause();
    /** @brief Resume workers previously paused with pause(). */
    void resume();
    /** @brief Block until all currently queued/running tasks complete. */
    void waitIdle();

    // Metrics

    /** @brief Read this pool's current performance metrics. */
    eshkol_thread_pool_metrics_t getMetrics() const;
    /** @brief Reset this pool's metrics counters to zero. */
    void resetMetrics();
    /** @brief Print this pool's current metrics to stderr. */
    void printMetrics() const;

    // Properties

    /** @brief Number of worker threads in this pool. */
    size_t numThreads() const;

    /** @brief Get the underlying C API handle (for interop with the C thread_pool_* functions). */
    eshkol_thread_pool_t* handle() const { return pool_; }

    /** @brief Get the process-wide global ThreadPool instance, lazily created on first call. */
    static ThreadPool& global();

private:
    eshkol_thread_pool_t* pool_;
    bool owns_pool_;
};

// Template implementations
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    // Wrap in C-compatible function
    auto wrapper = new std::function<void()>([task]() { (*task)(); });

    bool submitted = thread_pool_submit_detached(pool_,
        [](void* arg) -> void* {
            auto* fn = static_cast<std::function<void()>*>(arg);
            (*fn)();
            delete fn;
            return nullptr;
        },
        wrapper
    );

    if (!submitted) {
        delete wrapper;
        throw std::runtime_error("Failed to submit task to thread pool");
    }

    return result;
}

template<typename F, typename... Args>
void ThreadPool::submitDetached(F&& f, Args&&... args)
{
    auto task = new std::function<void()>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    bool submitted = thread_pool_submit_detached(pool_,
        [](void* arg) -> void* {
            auto* fn = static_cast<std::function<void()>*>(arg);
            (*fn)();
            delete fn;
            return nullptr;
        },
        task
    );

    if (!submitted) {
        delete task;
        throw std::runtime_error("Failed to submit task to thread pool");
    }
}

template<typename Iterator, typename Func>
void ThreadPool::parallelFor(Iterator begin, Iterator end, Func&& f)
{
    std::vector<std::future<void>> futures;
    for (auto it = begin; it != end; ++it) {
        futures.push_back(submit([&f, it]() { f(*it); }));
    }
    for (auto& fut : futures) {
        fut.get();
    }
}

template<typename T, typename Func>
std::vector<T> ThreadPool::parallelMap(const std::vector<T>& input, Func&& f)
{
    std::vector<std::future<T>> futures;
    futures.reserve(input.size());

    for (const auto& item : input) {
        futures.push_back(submit([&f, &item]() { return f(item); }));
    }

    std::vector<T> results;
    results.reserve(input.size());

    for (auto& fut : futures) {
        results.push_back(fut.get());
    }

    return results;
}

template<typename T, typename Func, typename Reducer>
T ThreadPool::parallelReduce(const std::vector<T>& input, T init, Func&& map_fn, Reducer&& reduce_fn)
{
    if (input.empty()) return init;

    // Map phase
    auto mapped = parallelMap(input, std::forward<Func>(map_fn));

    // Reduce phase (could be parallelized for large inputs)
    T result = init;
    for (const auto& item : mapped) {
        result = reduce_fn(result, item);
    }

    return result;
}

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_THREAD_POOL_H
