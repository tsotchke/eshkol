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

// Forward declarations
typedef struct eshkol_thread_pool eshkol_thread_pool_t;
typedef struct eshkol_future eshkol_future_t;
typedef struct eshkol_task eshkol_task_t;

// Task function signature: takes user data, returns result
typedef void* (*eshkol_task_fn)(void* arg);

// ============================================================================
// Thread Pool Configuration
// ============================================================================

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

// Create thread pool with configuration
eshkol_thread_pool_t* thread_pool_create(const eshkol_thread_pool_config_t* config);

// Create thread pool with default configuration
eshkol_thread_pool_t* thread_pool_create_default(void);

// Destroy thread pool (waits for all tasks to complete)
void thread_pool_destroy(eshkol_thread_pool_t* pool);

// Get global thread pool (lazily initialized)
eshkol_thread_pool_t* thread_pool_global(void);

// Shutdown global thread pool (call at program exit)
void thread_pool_global_shutdown(void);

// ============================================================================
// Task Submission
// ============================================================================

// Submit a task and get a future
// Returns NULL on failure (queue full, pool shutting down)
eshkol_future_t* thread_pool_submit(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg
);

// Submit multiple tasks in batch (more efficient)
// Returns number of successfully submitted tasks
size_t thread_pool_submit_batch(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn* fns,
    void** args,
    eshkol_future_t** futures,
    size_t count
);

// Submit task without future (fire-and-forget)
bool thread_pool_submit_detached(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg
);

// ============================================================================
// Future Operations
// ============================================================================

// Wait for future to complete and get result
// Blocks until the task finishes
void* future_get(eshkol_future_t* future);

// Wait for future with timeout (milliseconds)
// Returns true if completed, false if timed out
bool future_wait(eshkol_future_t* future, uint64_t timeout_ms);

// Check if future is ready (non-blocking)
bool future_is_ready(const eshkol_future_t* future);

// Release future resources
// Must be called after future_get() or when abandoning the future
void future_release(eshkol_future_t* future);

// Wait for multiple futures
// Returns index of first completed future, or -1 on timeout
int future_wait_any(eshkol_future_t** futures, size_t count, uint64_t timeout_ms);

// Wait for all futures to complete
bool future_wait_all(eshkol_future_t** futures, size_t count, uint64_t timeout_ms);

// ============================================================================
// Thread Pool Metrics
// ============================================================================

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

// Get current metrics
void thread_pool_get_metrics(
    const eshkol_thread_pool_t* pool,
    eshkol_thread_pool_metrics_t* metrics
);

// Reset metrics (useful for benchmarking)
void thread_pool_reset_metrics(eshkol_thread_pool_t* pool);

// Print metrics to stderr
void thread_pool_print_metrics(const eshkol_thread_pool_t* pool);

// ============================================================================
// Thread Pool Control
// ============================================================================

// Get number of worker threads
size_t thread_pool_num_threads(const eshkol_thread_pool_t* pool);

// Pause all workers (tasks already running will complete)
void thread_pool_pause(eshkol_thread_pool_t* pool);

// Resume paused workers
void thread_pool_resume(eshkol_thread_pool_t* pool);

// Wait for all pending tasks to complete
void thread_pool_wait_idle(eshkol_thread_pool_t* pool);

// ============================================================================
// Thread-Local Arena Access
// ============================================================================

// Get thread-local arena for current worker thread
// Returns NULL if called from non-worker thread
struct arena* thread_pool_get_thread_arena(void);

// Get or create thread-local arena (works from any thread)
struct arena* get_thread_local_arena(void);

// Reset thread-local arena (call after completing independent work)
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

class ThreadPool {
public:
    // Constructors
    ThreadPool();
    explicit ThreadPool(size_t num_threads);
    explicit ThreadPool(const eshkol_thread_pool_config_t& config);
    ~ThreadPool();

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Movable
    ThreadPool(ThreadPool&& other) noexcept;
    ThreadPool& operator=(ThreadPool&& other) noexcept;

    // Submit with std::function and std::future
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

    // Submit without return value
    template<typename F, typename... Args>
    void submitDetached(F&& f, Args&&... args);

    // Parallel algorithms
    template<typename Iterator, typename Func>
    void parallelFor(Iterator begin, Iterator end, Func&& f);

    template<typename T, typename Func>
    std::vector<T> parallelMap(const std::vector<T>& input, Func&& f);

    template<typename T, typename Func, typename Reducer>
    T parallelReduce(const std::vector<T>& input, T init, Func&& map_fn, Reducer&& reduce_fn);

    // Control
    void pause();
    void resume();
    void waitIdle();

    // Metrics
    eshkol_thread_pool_metrics_t getMetrics() const;
    void resetMetrics();
    void printMetrics() const;

    // Properties
    size_t numThreads() const;

    // Get underlying C handle
    eshkol_thread_pool_t* handle() const { return pool_; }

    // Global pool access
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
