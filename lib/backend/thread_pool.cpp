/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Thread Pool Implementation for Eshkol Parallel Execution
 *
 * This implementation supports two modes:
 * 1. Work-Stealing Mode (default): Uses Chase-Lev deques for optimal load balancing
 * 2. Queue Mode (legacy): Uses a shared queue with mutex protection
 *
 * Work-stealing provides better performance for irregular workloads by allowing
 * idle workers to steal tasks from busy workers' local deques.
 */

#include "../../inc/eshkol/backend/thread_pool.h"
#include "../../inc/eshkol/backend/work_stealing_deque.h"
#include "../core/arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include <vector>
#include <functional>
#include <cstring>
#include <random>

// ============================================================================
// Internal Structures
// ============================================================================

// Task wrapper that bridges C API to internal scheduler
struct eshkol_task {
    eshkol_task_fn fn;
    void* arg;
    eshkol_future_t* future;
    std::chrono::steady_clock::time_point submit_time;
};

struct eshkol_future {
    std::mutex mutex;
    std::condition_variable cv;
    void* result;
    bool completed;
    bool released;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
};

// Work-stealing thread pool implementation
struct eshkol_thread_pool {
    // Configuration
    std::string name;
    size_t num_threads;
    size_t thread_arena_size;
    bool enable_metrics;
    bool use_work_stealing;

    // Work-stealing infrastructure
    std::unique_ptr<eshkol::EpochManager> epoch_mgr;
    std::vector<std::unique_ptr<eshkol::WorkStealingDeque>> deques;
    std::vector<std::mt19937> worker_rngs;

    // Thread management
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};

    // Legacy queue mode (when work_stealing is disabled)
    std::queue<eshkol_task*> legacy_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::condition_variable pause_cv;
    size_t max_queue_size;

    // Metrics (atomic for lock-free updates)
    std::atomic<size_t> tasks_submitted{0};
    std::atomic<size_t> tasks_completed{0};
    std::atomic<size_t> active_workers{0};
    std::atomic<size_t> peak_queue_depth{0};
    std::atomic<uint64_t> total_task_time_us{0};
    std::atomic<uint64_t> total_wait_time_us{0};

    // Work-stealing specific metrics
    std::atomic<uint64_t> tasks_stolen{0};
    std::atomic<uint64_t> steal_attempts{0};
    std::atomic<uint64_t> local_executions{0};

    // Idle waiting
    std::mutex idle_mutex;
    std::condition_variable idle_cv;
};

// ============================================================================
// Thread-Local Storage
// ============================================================================

// Thread-local arena for worker threads
thread_local arena_t* tls_arena = nullptr;
thread_local bool tls_is_worker_thread = false;
thread_local size_t tls_arena_size = 0;
thread_local size_t tls_worker_id = SIZE_MAX;
thread_local size_t tls_epoch_slot = SIZE_MAX;

arena_t* get_thread_local_arena(void) {
    if (!tls_arena) {
        // Use reasonable default size
        size_t size = tls_arena_size > 0 ? tls_arena_size : (1024 * 1024);
        tls_arena = arena_create(size);
        if (tls_arena) {
            eshkol_debug("Created thread-local arena (size: %zu)", size);
        } else {
            eshkol_error("Failed to create thread-local arena");
        }
    }
    return tls_arena;
}

arena_t* thread_pool_get_thread_arena(void) {
    if (!tls_is_worker_thread) {
        return nullptr;
    }
    return get_thread_local_arena();
}

void thread_pool_reset_thread_arena(void) {
    if (tls_arena) {
        arena_reset(tls_arena);
    }
}

// ============================================================================
// Work-Stealing Worker Implementation
// ============================================================================

static void work_stealing_worker_func(eshkol_thread_pool_t* pool, size_t worker_id) {
    tls_is_worker_thread = true;
    tls_arena_size = pool->thread_arena_size;
    tls_worker_id = worker_id;

    // Register with epoch manager for safe memory reclamation
    tls_epoch_slot = pool->epoch_mgr->registerThread();

    eshkol_debug("[%s] Work-stealing worker %zu started (epoch slot %zu)",
                pool->name.c_str(), worker_id, tls_epoch_slot);

    eshkol::WorkStealingDeque& my_deque = *pool->deques[worker_id];
    std::mt19937& rng = pool->worker_rngs[worker_id];

    size_t idle_spins = 0;
    constexpr size_t MAX_IDLE_SPINS = 32;
    constexpr size_t STEAL_ATTEMPTS_BEFORE_SLEEP = 64;

    while (!pool->stop.load(std::memory_order_acquire)) {
        // Handle pause
        if (pool->paused.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(pool->queue_mutex);
            pool->pause_cv.wait(lock, [pool]() {
                return !pool->paused.load() || pool->stop.load();
            });
            if (pool->stop.load()) break;
        }

        eshkol_task* task = nullptr;

        // Enter epoch critical section for safe deque access
        {
            eshkol::EpochGuard guard(*pool->epoch_mgr, tls_epoch_slot);

            // 1. Try local deque first (LIFO - good cache locality)
            task = static_cast<eshkol_task*>(my_deque.pop());

            if (task) {
                pool->local_executions.fetch_add(1, std::memory_order_relaxed);
                idle_spins = 0;
            } else {
                // 2. Try to steal from other workers
                std::uniform_int_distribution<size_t> dist(0, pool->num_threads - 1);

                for (size_t attempt = 0; attempt < pool->num_threads && !task; ++attempt) {
                    size_t victim = dist(rng);
                    if (victim == worker_id) continue;

                    pool->steal_attempts.fetch_add(1, std::memory_order_relaxed);
                    task = static_cast<eshkol_task*>(pool->deques[victim]->steal());

                    if (task) {
                        pool->tasks_stolen.fetch_add(1, std::memory_order_relaxed);
                        idle_spins = 0;
                    }
                }
            }
        }

        if (task) {
            // Execute the task
            pool->active_workers.fetch_add(1, std::memory_order_relaxed);

            auto start_time = std::chrono::steady_clock::now();

            // Measure wait time
            if (pool->enable_metrics) {
                auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    start_time - task->submit_time).count();
                pool->total_wait_time_us.fetch_add(wait_time, std::memory_order_relaxed);
            }

            // Execute task function
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

            auto end_time = std::chrono::steady_clock::now();

            // Update metrics
            if (pool->enable_metrics) {
                auto task_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time).count();
                pool->total_task_time_us.fetch_add(task_time, std::memory_order_relaxed);
            }

            pool->tasks_completed.fetch_add(1, std::memory_order_relaxed);
            pool->active_workers.fetch_sub(1, std::memory_order_relaxed);

            // Complete future if present
            if (task->future) {
                std::lock_guard<std::mutex> lock(task->future->mutex);
                task->future->result = result;
                task->future->completed = true;
                task->future->end_time = end_time;
                task->future->cv.notify_all();
            }

            delete task;

            // Notify idle waiters
            {
                std::lock_guard<std::mutex> lock(pool->idle_mutex);
                pool->idle_cv.notify_all();
            }

            // Periodically advance epoch and reclaim memory
            if ((pool->tasks_completed.load(std::memory_order_relaxed) & 0xFF) == 0) {
                pool->epoch_mgr->tryAdvanceEpoch();
                my_deque.reclaimRetired();
            }
        } else {
            // No work found - backoff strategy
            ++idle_spins;

            if (idle_spins < MAX_IDLE_SPINS) {
                // Spin briefly with pause instruction
                #if defined(__x86_64__) || defined(_M_X64)
                    __builtin_ia32_pause();
                #elif defined(__aarch64__)
                    asm volatile("yield" ::: "memory");
                #else
                    std::this_thread::yield();
                #endif
            } else if (idle_spins < STEAL_ATTEMPTS_BEFORE_SLEEP) {
                std::this_thread::yield();
            } else {
                // Sleep until notified
                std::unique_lock<std::mutex> lock(pool->queue_mutex);
                pool->queue_cv.wait_for(lock, std::chrono::milliseconds(1), [pool]() {
                    if (pool->stop.load(std::memory_order_relaxed)) return true;
                    for (const auto& d : pool->deques) {
                        if (!d->empty()) return true;
                    }
                    return false;
                });
                idle_spins = 0;
            }
        }
    }

    // Cleanup
    pool->epoch_mgr->unregisterThread(tls_epoch_slot);

    if (tls_arena) {
        arena_destroy(tls_arena);
        tls_arena = nullptr;
    }

    eshkol_debug("[%s] Work-stealing worker %zu stopped", pool->name.c_str(), worker_id);
}

// ============================================================================
// Legacy Queue Worker Implementation (for compatibility)
// ============================================================================

static void legacy_worker_func(eshkol_thread_pool_t* pool, size_t worker_id) {
    tls_is_worker_thread = true;
    tls_arena_size = pool->thread_arena_size;
    tls_worker_id = worker_id;

    eshkol_debug("[%s] Legacy worker %zu started", pool->name.c_str(), worker_id);

    while (true) {
        eshkol_task* task = nullptr;

        // Wait for task
        {
            std::unique_lock<std::mutex> lock(pool->queue_mutex);

            pool->queue_cv.wait(lock, [pool]() {
                return pool->stop.load() ||
                       (!pool->paused.load() && !pool->legacy_queue.empty());
            });

            // Handle pause
            while (pool->paused.load() && !pool->stop.load()) {
                pool->pause_cv.wait(lock);
            }

            if (pool->stop.load() && pool->legacy_queue.empty()) {
                break;
            }

            if (!pool->legacy_queue.empty()) {
                task = pool->legacy_queue.front();
                pool->legacy_queue.pop();
            }
        }

        if (!task) continue;

        // Execute task
        pool->active_workers.fetch_add(1);

        auto start_time = std::chrono::steady_clock::now();

        if (pool->enable_metrics) {
            auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(
                start_time - task->submit_time).count();
            pool->total_wait_time_us.fetch_add(wait_time);
        }

        void* result = nullptr;
        try {
            result = task->fn(task->arg);
        } catch (const std::exception& e) {
            eshkol_error("[%s] Worker %zu: Task threw exception: %s",
                        pool->name.c_str(), worker_id, e.what());
        } catch (...) {
            eshkol_error("[%s] Worker %zu: Task threw unknown exception",
                        pool->name.c_str(), worker_id);
        }

        auto end_time = std::chrono::steady_clock::now();

        if (pool->enable_metrics) {
            auto task_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count();
            pool->total_task_time_us.fetch_add(task_time);
        }

        pool->tasks_completed.fetch_add(1);
        pool->active_workers.fetch_sub(1);

        if (task->future) {
            std::lock_guard<std::mutex> lock(task->future->mutex);
            task->future->result = result;
            task->future->completed = true;
            task->future->end_time = end_time;
            task->future->cv.notify_all();
        }

        delete task;

        {
            std::lock_guard<std::mutex> lock(pool->idle_mutex);
            pool->idle_cv.notify_all();
        }
    }

    if (tls_arena) {
        arena_destroy(tls_arena);
        tls_arena = nullptr;
    }

    eshkol_debug("[%s] Legacy worker %zu stopped", pool->name.c_str(), worker_id);
}

// ============================================================================
// Thread Pool Lifecycle
// ============================================================================

eshkol_thread_pool_t* thread_pool_create(const eshkol_thread_pool_config_t* config) {
    auto pool = new eshkol_thread_pool_t();

    // Apply configuration
    pool->num_threads = config->num_threads;
    if (pool->num_threads == 0) {
        pool->num_threads = std::thread::hardware_concurrency();
        if (pool->num_threads == 0) {
            pool->num_threads = 4;  // Fallback
        }
    }

    pool->thread_arena_size = config->thread_arena_size;
    if (pool->thread_arena_size == 0) {
        pool->thread_arena_size = 1024 * 1024;  // 1MB default
    }

    pool->max_queue_size = config->task_queue_capacity;
    pool->enable_metrics = config->enable_metrics;
    pool->name = config->name ? config->name : "unnamed";

    // Enable work-stealing by default (can be disabled via environment variable)
    const char* disable_ws = std::getenv("ESHKOL_DISABLE_WORK_STEALING");
    pool->use_work_stealing = (disable_ws == nullptr || disable_ws[0] == '0');

    if (pool->use_work_stealing) {
        eshkol_info("[%s] Creating work-stealing thread pool with %zu workers",
                   pool->name.c_str(), pool->num_threads);

        // Initialize epoch-based memory reclamation
        pool->epoch_mgr = std::make_unique<eshkol::EpochManager>();

        // Create per-worker deques
        pool->deques.reserve(pool->num_threads);
        for (size_t i = 0; i < pool->num_threads; ++i) {
            pool->deques.push_back(
                std::make_unique<eshkol::WorkStealingDeque>(*pool->epoch_mgr, 12));  // 4096 initial capacity
        }

        // Initialize per-worker RNGs for random victim selection
        std::random_device rd;
        pool->worker_rngs.reserve(pool->num_threads);
        for (size_t i = 0; i < pool->num_threads; ++i) {
            pool->worker_rngs.push_back(std::mt19937(rd() + i));
        }

        // Create work-stealing workers
        pool->workers.reserve(pool->num_threads);
        for (size_t i = 0; i < pool->num_threads; ++i) {
            pool->workers.emplace_back(work_stealing_worker_func, pool, i);
        }
    } else {
        eshkol_info("[%s] Creating legacy thread pool with %zu threads",
                   pool->name.c_str(), pool->num_threads);

        // Create legacy workers
        pool->workers.reserve(pool->num_threads);
        for (size_t i = 0; i < pool->num_threads; ++i) {
            pool->workers.emplace_back(legacy_worker_func, pool, i);
        }
    }

    return pool;
}

eshkol_thread_pool_t* thread_pool_create_default(void) {
    eshkol_thread_pool_config_t config = ESHKOL_THREAD_POOL_DEFAULT_CONFIG;
    return thread_pool_create(&config);
}

void thread_pool_destroy(eshkol_thread_pool_t* pool) {
    if (!pool) return;

    eshkol_info("[%s] Shutting down thread pool", pool->name.c_str());

    // Signal workers to stop
    pool->stop.store(true);
    pool->queue_cv.notify_all();
    pool->pause_cv.notify_all();

    // Wait for all workers to finish
    for (auto& worker : pool->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    // Cleanup remaining tasks
    if (pool->use_work_stealing) {
        // Drain all deques
        for (auto& deque : pool->deques) {
            while (!deque->empty()) {
                auto* task = static_cast<eshkol_task*>(deque->pop());
                if (task) {
                    if (task->future) {
                        std::lock_guard<std::mutex> lock(task->future->mutex);
                        task->future->completed = true;
                        task->future->cv.notify_all();
                    }
                    delete task;
                }
            }
        }
    } else {
        while (!pool->legacy_queue.empty()) {
            auto* task = pool->legacy_queue.front();
            pool->legacy_queue.pop();

            if (task->future) {
                std::lock_guard<std::mutex> lock(task->future->mutex);
                task->future->completed = true;
                task->future->cv.notify_all();
            }

            delete task;
        }
    }

    // Print final metrics if enabled
    if (pool->enable_metrics) {
        thread_pool_print_metrics(pool);
    }

    delete pool;
}

// Global thread pool singleton
static std::mutex g_global_pool_mutex;
static eshkol_thread_pool_t* g_global_pool = nullptr;

eshkol_thread_pool_t* thread_pool_global(void) {
    std::lock_guard<std::mutex> lock(g_global_pool_mutex);
    if (!g_global_pool) {
        g_global_pool = thread_pool_create_default();
    }
    return g_global_pool;
}

void thread_pool_global_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_global_pool_mutex);
    if (g_global_pool) {
        thread_pool_destroy(g_global_pool);
        g_global_pool = nullptr;
    }
}

// ============================================================================
// Task Submission
// ============================================================================

eshkol_future_t* thread_pool_submit(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg)
{
    if (!pool || pool->stop.load()) {
        return nullptr;
    }

    auto* future = new eshkol_future_t();
    future->result = nullptr;
    future->completed = false;
    future->released = false;
    future->start_time = std::chrono::steady_clock::now();

    auto* task = new eshkol_task();
    task->fn = fn;
    task->arg = arg;
    task->future = future;
    task->submit_time = std::chrono::steady_clock::now();

    if (pool->use_work_stealing) {
        // Determine target deque
        size_t target;
        if (tls_worker_id < pool->num_threads) {
            // Called from worker thread - push to local deque
            target = tls_worker_id;
        } else {
            // Called from external thread - distribute based on thread ID hash
            target = std::hash<std::thread::id>{}(std::this_thread::get_id()) % pool->num_threads;
        }

        pool->deques[target]->push(task);
        pool->tasks_submitted.fetch_add(1, std::memory_order_relaxed);

        // Wake up a worker
        pool->queue_cv.notify_one();
    } else {
        // Legacy queue mode
        std::lock_guard<std::mutex> lock(pool->queue_mutex);

        if (pool->max_queue_size > 0 && pool->legacy_queue.size() >= pool->max_queue_size) {
            delete task;
            delete future;
            return nullptr;
        }

        pool->legacy_queue.push(task);
        pool->tasks_submitted.fetch_add(1);

        // Update peak queue depth
        size_t current_depth = pool->legacy_queue.size();
        size_t peak = pool->peak_queue_depth.load();
        while (current_depth > peak &&
               !pool->peak_queue_depth.compare_exchange_weak(peak, current_depth)) {
        }

        pool->queue_cv.notify_one();
    }

    return future;
}

size_t thread_pool_submit_batch(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn* fns,
    void** args,
    eshkol_future_t** futures,
    size_t count)
{
    if (!pool || pool->stop.load() || count == 0) {
        return 0;
    }

    size_t submitted = 0;

    if (pool->use_work_stealing) {
        // Determine target deque
        size_t target;
        if (tls_worker_id < pool->num_threads) {
            target = tls_worker_id;
        } else {
            target = std::hash<std::thread::id>{}(std::this_thread::get_id()) % pool->num_threads;
        }

        for (size_t i = 0; i < count; ++i) {
            auto* future = new eshkol_future_t();
            future->result = nullptr;
            future->completed = false;
            future->released = false;
            future->start_time = std::chrono::steady_clock::now();

            auto* task = new eshkol_task();
            task->fn = fns[i];
            task->arg = args[i];
            task->future = future;
            task->submit_time = std::chrono::steady_clock::now();

            // Round-robin distribution across deques for batch submissions
            pool->deques[(target + i) % pool->num_threads]->push(task);
            futures[i] = future;
            submitted++;
        }

        pool->tasks_submitted.fetch_add(submitted, std::memory_order_relaxed);

        // Wake up workers
        pool->queue_cv.notify_all();
    } else {
        std::lock_guard<std::mutex> lock(pool->queue_mutex);

        for (size_t i = 0; i < count; ++i) {
            if (pool->max_queue_size > 0 && pool->legacy_queue.size() >= pool->max_queue_size) {
                break;
            }

            auto* future = new eshkol_future_t();
            future->result = nullptr;
            future->completed = false;
            future->released = false;
            future->start_time = std::chrono::steady_clock::now();

            auto* task = new eshkol_task();
            task->fn = fns[i];
            task->arg = args[i];
            task->future = future;
            task->submit_time = std::chrono::steady_clock::now();

            pool->legacy_queue.push(task);
            futures[i] = future;
            submitted++;
        }

        pool->tasks_submitted.fetch_add(submitted);

        size_t current_depth = pool->legacy_queue.size();
        size_t peak = pool->peak_queue_depth.load();
        while (current_depth > peak &&
               !pool->peak_queue_depth.compare_exchange_weak(peak, current_depth)) {
        }

        for (size_t i = 0; i < submitted; ++i) {
            pool->queue_cv.notify_one();
        }
    }

    return submitted;
}

bool thread_pool_submit_detached(
    eshkol_thread_pool_t* pool,
    eshkol_task_fn fn,
    void* arg)
{
    if (!pool || pool->stop.load()) {
        return false;
    }

    auto* task = new eshkol_task();
    task->fn = fn;
    task->arg = arg;
    task->future = nullptr;
    task->submit_time = std::chrono::steady_clock::now();

    if (pool->use_work_stealing) {
        size_t target;
        if (tls_worker_id < pool->num_threads) {
            target = tls_worker_id;
        } else {
            target = std::hash<std::thread::id>{}(std::this_thread::get_id()) % pool->num_threads;
        }

        pool->deques[target]->push(task);
        pool->tasks_submitted.fetch_add(1, std::memory_order_relaxed);
        pool->queue_cv.notify_one();
    } else {
        std::lock_guard<std::mutex> lock(pool->queue_mutex);

        if (pool->max_queue_size > 0 && pool->legacy_queue.size() >= pool->max_queue_size) {
            delete task;
            return false;
        }

        pool->legacy_queue.push(task);
        pool->tasks_submitted.fetch_add(1);

        size_t current_depth = pool->legacy_queue.size();
        size_t peak = pool->peak_queue_depth.load();
        while (current_depth > peak &&
               !pool->peak_queue_depth.compare_exchange_weak(peak, current_depth)) {
        }

        pool->queue_cv.notify_one();
    }

    return true;
}

// ============================================================================
// Future Operations
// ============================================================================

void* future_get(eshkol_future_t* future) {
    if (!future) return nullptr;

    std::unique_lock<std::mutex> lock(future->mutex);
    future->cv.wait(lock, [future]() { return future->completed; });
    return future->result;
}

bool future_wait(eshkol_future_t* future, uint64_t timeout_ms) {
    if (!future) return false;

    std::unique_lock<std::mutex> lock(future->mutex);
    if (future->completed) return true;

    return future->cv.wait_for(lock,
        std::chrono::milliseconds(timeout_ms),
        [future]() { return future->completed; });
}

bool future_is_ready(const eshkol_future_t* future) {
    if (!future) return false;
    auto* f = const_cast<eshkol_future_t*>(future);
    std::lock_guard<std::mutex> lock(f->mutex);
    return f->completed;
}

void future_release(eshkol_future_t* future) {
    if (!future) return;

    {
        std::lock_guard<std::mutex> lock(future->mutex);
        if (future->released) return;
        future->released = true;
    }

    delete future;
}

int future_wait_any(eshkol_future_t** futures, size_t count, uint64_t timeout_ms) {
    if (!futures || count == 0) return -1;

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        for (size_t i = 0; i < count; ++i) {
            if (futures[i] && future_is_ready(futures[i])) {
                return static_cast<int>(i);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return -1;
}

bool future_wait_all(eshkol_future_t** futures, size_t count, uint64_t timeout_ms) {
    if (!futures || count == 0) return true;

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    for (size_t i = 0; i < count; ++i) {
        if (!futures[i]) continue;

        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();

        if (remaining <= 0) return false;

        if (!future_wait(futures[i], static_cast<uint64_t>(remaining))) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Eshkol Lazy Future Helpers for LLVM-Generated Code
// ============================================================================

// Lazy future structure - simple pointer storage
struct eshkol_lazy_future {
    uint64_t thunk_ptr;
    uint8_t thunk_type;
    uint8_t thunk_flags;
    uint8_t forced;
    uint8_t padding[5];
    uint64_t result_ptr;
    uint8_t result_type;
    uint8_t result_flags;
    uint8_t padding2[6];
};

extern "C" eshkol_lazy_future* eshkol_lazy_future_create_ptr(
    uint64_t thunk_ptr, uint8_t thunk_type, uint8_t thunk_flags)
{
    auto* lf = new eshkol_lazy_future();
    lf->thunk_ptr = thunk_ptr;
    lf->thunk_type = thunk_type;
    lf->thunk_flags = thunk_flags;
    lf->forced = 0;
    lf->result_ptr = 0;
    lf->result_type = ESHKOL_VALUE_NULL;
    lf->result_flags = 0;
    return lf;
}

extern "C" uint8_t eshkol_lazy_future_is_ready(eshkol_lazy_future* lf) {
    return lf ? lf->forced : 1;
}

extern "C" uint64_t eshkol_lazy_future_get_thunk_ptr(eshkol_lazy_future* lf) {
    return lf ? lf->thunk_ptr : 0;
}

extern "C" uint8_t eshkol_lazy_future_get_thunk_type(eshkol_lazy_future* lf) {
    return lf ? lf->thunk_type : ESHKOL_VALUE_NULL;
}

extern "C" uint8_t eshkol_lazy_future_get_thunk_flags(eshkol_lazy_future* lf) {
    return lf ? lf->thunk_flags : 0;
}

extern "C" uint64_t eshkol_lazy_future_get_result_ptr(eshkol_lazy_future* lf) {
    return (lf && lf->forced) ? lf->result_ptr : 0;
}

extern "C" uint8_t eshkol_lazy_future_get_result_type(eshkol_lazy_future* lf) {
    return (lf && lf->forced) ? lf->result_type : ESHKOL_VALUE_NULL;
}

extern "C" uint8_t eshkol_lazy_future_get_result_flags(eshkol_lazy_future* lf) {
    return (lf && lf->forced) ? lf->result_flags : 0;
}

extern "C" void eshkol_lazy_future_set_result_ptr(
    eshkol_lazy_future* lf, uint64_t result_ptr, uint8_t result_type, uint8_t result_flags)
{
    if (lf) {
        lf->result_ptr = result_ptr;
        lf->result_type = result_type;
        lf->result_flags = result_flags;
        lf->forced = 1;
    }
}

// ============================================================================
// Thread Pool Metrics
// ============================================================================

void thread_pool_get_metrics(
    const eshkol_thread_pool_t* pool,
    eshkol_thread_pool_metrics_t* metrics)
{
    if (!pool || !metrics) return;

    metrics->num_threads = pool->num_threads;
    metrics->tasks_submitted = pool->tasks_submitted.load();
    metrics->tasks_completed = pool->tasks_completed.load();
    metrics->active_workers = pool->active_workers.load();
    metrics->peak_queue_depth = pool->peak_queue_depth.load();
    metrics->total_task_time_us = pool->total_task_time_us.load();
    metrics->total_wait_time_us = pool->total_wait_time_us.load();

    // Calculate pending tasks
    if (pool->use_work_stealing) {
        size_t pending = 0;
        for (const auto& deque : pool->deques) {
            pending += deque->size();
        }
        metrics->tasks_pending = pending;
    } else {
        auto* p = const_cast<eshkol_thread_pool_t*>(pool);
        std::lock_guard<std::mutex> lock(p->queue_mutex);
        metrics->tasks_pending = p->legacy_queue.size();
    }

    // Calculate averages
    size_t completed = metrics->tasks_completed;
    if (completed > 0) {
        metrics->avg_task_time_us = metrics->total_task_time_us / completed;
        metrics->avg_wait_time_us = metrics->total_wait_time_us / completed;
    } else {
        metrics->avg_task_time_us = 0;
        metrics->avg_wait_time_us = 0;
    }
}

void thread_pool_reset_metrics(eshkol_thread_pool_t* pool) {
    if (!pool) return;

    pool->tasks_submitted.store(0);
    pool->tasks_completed.store(0);
    pool->peak_queue_depth.store(0);
    pool->total_task_time_us.store(0);
    pool->total_wait_time_us.store(0);
    pool->tasks_stolen.store(0);
    pool->steal_attempts.store(0);
    pool->local_executions.store(0);
}

void thread_pool_print_metrics(const eshkol_thread_pool_t* pool) {
    if (!pool) return;

    eshkol_thread_pool_metrics_t metrics;
    thread_pool_get_metrics(pool, &metrics);

    eshkol_info("[%s] Thread Pool Metrics:", pool->name.c_str());
    eshkol_info("  Mode: %s", pool->use_work_stealing ? "work-stealing" : "legacy");
    eshkol_info("  Threads: %zu", metrics.num_threads);
    eshkol_info("  Tasks: submitted=%zu, completed=%zu, pending=%zu",
               metrics.tasks_submitted, metrics.tasks_completed, metrics.tasks_pending);
    eshkol_info("  Active workers: %zu", metrics.active_workers);

    if (pool->use_work_stealing) {
        eshkol_info("  Work-stealing stats:");
        eshkol_info("    Local executions: %llu",
                   (unsigned long long)pool->local_executions.load());
        eshkol_info("    Tasks stolen: %llu",
                   (unsigned long long)pool->tasks_stolen.load());
        eshkol_info("    Steal attempts: %llu",
                   (unsigned long long)pool->steal_attempts.load());

        uint64_t total_exec = pool->local_executions.load() + pool->tasks_stolen.load();
        if (total_exec > 0) {
            double steal_ratio = 100.0 * pool->tasks_stolen.load() / total_exec;
            eshkol_info("    Steal ratio: %.1f%%", steal_ratio);
        }
    } else {
        eshkol_info("  Peak queue depth: %zu", metrics.peak_queue_depth);
    }

    eshkol_info("  Avg task time: %llu us", (unsigned long long)metrics.avg_task_time_us);
    eshkol_info("  Avg wait time: %llu us", (unsigned long long)metrics.avg_wait_time_us);
}

// ============================================================================
// Thread Pool Control
// ============================================================================

size_t thread_pool_num_threads(const eshkol_thread_pool_t* pool) {
    return pool ? pool->num_threads : 0;
}

void thread_pool_pause(eshkol_thread_pool_t* pool) {
    if (!pool) return;
    pool->paused.store(true);
}

void thread_pool_resume(eshkol_thread_pool_t* pool) {
    if (!pool) return;
    pool->paused.store(false);
    pool->pause_cv.notify_all();
}

void thread_pool_wait_idle(eshkol_thread_pool_t* pool) {
    if (!pool) return;

    std::unique_lock<std::mutex> lock(pool->idle_mutex);

    while (true) {
        size_t pending = 0;

        if (pool->use_work_stealing) {
            for (const auto& deque : pool->deques) {
                pending += deque->size();
            }
        } else {
            std::lock_guard<std::mutex> qlock(pool->queue_mutex);
            pending = pool->legacy_queue.size();
        }

        if (pending == 0 && pool->active_workers.load() == 0) {
            break;
        }

        pool->idle_cv.wait_for(lock, std::chrono::milliseconds(10));
    }
}

// ============================================================================
// C++ Wrapper Implementation
// ============================================================================

#ifdef __cplusplus

namespace eshkol {

ThreadPool::ThreadPool()
    : pool_(thread_pool_create_default()), owns_pool_(true) {}

ThreadPool::ThreadPool(size_t num_threads)
    : owns_pool_(true)
{
    eshkol_thread_pool_config_t config = ESHKOL_THREAD_POOL_DEFAULT_CONFIG;
    config.num_threads = num_threads;
    pool_ = thread_pool_create(&config);
}

ThreadPool::ThreadPool(const eshkol_thread_pool_config_t& config)
    : pool_(thread_pool_create(&config)), owns_pool_(true) {}

ThreadPool::~ThreadPool() {
    if (owns_pool_ && pool_) {
        thread_pool_destroy(pool_);
    }
}

ThreadPool::ThreadPool(ThreadPool&& other) noexcept
    : pool_(other.pool_), owns_pool_(other.owns_pool_)
{
    other.pool_ = nullptr;
    other.owns_pool_ = false;
}

ThreadPool& ThreadPool::operator=(ThreadPool&& other) noexcept {
    if (this != &other) {
        if (owns_pool_ && pool_) {
            thread_pool_destroy(pool_);
        }
        pool_ = other.pool_;
        owns_pool_ = other.owns_pool_;
        other.pool_ = nullptr;
        other.owns_pool_ = false;
    }
    return *this;
}

void ThreadPool::pause() {
    thread_pool_pause(pool_);
}

void ThreadPool::resume() {
    thread_pool_resume(pool_);
}

void ThreadPool::waitIdle() {
    thread_pool_wait_idle(pool_);
}

eshkol_thread_pool_metrics_t ThreadPool::getMetrics() const {
    eshkol_thread_pool_metrics_t metrics{};
    thread_pool_get_metrics(pool_, &metrics);
    return metrics;
}

void ThreadPool::resetMetrics() {
    thread_pool_reset_metrics(pool_);
}

void ThreadPool::printMetrics() const {
    thread_pool_print_metrics(pool_);
}

size_t ThreadPool::numThreads() const {
    return thread_pool_num_threads(pool_);
}

// Global pool singleton
static std::once_flag g_cpp_global_pool_flag;
static ThreadPool* g_cpp_global_pool = nullptr;

ThreadPool& ThreadPool::global() {
    std::call_once(g_cpp_global_pool_flag, []() {
        auto* c_pool = thread_pool_global();
        g_cpp_global_pool = new ThreadPool();
        if (g_cpp_global_pool->pool_) {
            thread_pool_destroy(g_cpp_global_pool->pool_);
        }
        g_cpp_global_pool->pool_ = c_pool;
        g_cpp_global_pool->owns_pool_ = false;
    });
    return *g_cpp_global_pool;
}

} // namespace eshkol

#endif // __cplusplus
