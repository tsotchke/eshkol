/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Production Work-Stealing Deque (Chase-Lev Algorithm)
 *
 * Based on "Dynamic Circular Work-Stealing Deque" by Chase and Lev (2005)
 * with epoch-based memory reclamation for safe array resizing.
 *
 * Properties:
 * - Owner thread: push/pop (LIFO, single-producer)
 * - Thief threads: steal (FIFO, multiple-consumer)
 * - Lock-free with bounded wait-free stealing
 * - Safe memory reclamation via epoch-based garbage collection
 *
 * Memory Model:
 * - Uses C++11 atomics with appropriate memory orderings
 * - Cache-line aligned to prevent false sharing
 * - Tested on x86-64 and ARM64
 */

#ifndef ESHKOL_WORK_STEALING_DEQUE_H
#define ESHKOL_WORK_STEALING_DEQUE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <thread>

namespace eshkol {

// Cache line size for alignment (64 bytes on most modern CPUs)
constexpr size_t CACHE_LINE_SIZE = 64;

// ============================================================================
// Epoch-Based Memory Reclamation
// ============================================================================

/**
 * @brief Epoch-based memory reclamation manager.
 *
 * Tracks, per registered thread, the global epoch observed the last time
 * that thread entered a critical section. Once every active thread has
 * observed an epoch, any memory retired before that epoch is safe to
 * reclaim (no thread can still hold a reference to it). Used by
 * WorkStealingDeque to safely free old CircularArray buffers after a
 * resize, without requiring readers (thieves) to take a lock.
 */
class EpochManager {
public:
    /** @brief Maximum number of threads that can be registered concurrently. */
    static constexpr size_t MAX_THREADS = 256;
    /** @brief Sentinel epoch value meaning "this thread slot is not in a critical section". */
    static constexpr uint64_t EPOCH_INACTIVE = UINT64_MAX;

    /** @brief Construct an EpochManager with all thread slots inactive. */
    EpochManager() : global_epoch_(0), retire_epoch_(0) {
        for (size_t i = 0; i < MAX_THREADS; ++i) {
            thread_epochs_[i].epoch.store(EPOCH_INACTIVE, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Register the calling thread with the epoch manager.
     * @return The assigned thread slot id, or MAX_THREADS if no slot was free
     *         (should not happen in practice).
     */
    size_t registerThread() {
        for (size_t i = 0; i < MAX_THREADS; ++i) {
            uint64_t expected = EPOCH_INACTIVE;
            if (thread_epochs_[i].epoch.compare_exchange_strong(
                    expected, global_epoch_.load(std::memory_order_relaxed),
                    std::memory_order_acq_rel)) {
                return i;
            }
        }
        // Should not happen in practice
        return MAX_THREADS;
    }

    /** @brief Unregister a thread slot previously obtained from registerThread(). */
    void unregisterThread(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(EPOCH_INACTIVE, std::memory_order_release);
        }
    }

    /** @brief Mark thread slot as entering a critical section at the current global epoch. */
    void enterCritical(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(
                global_epoch_.load(std::memory_order_relaxed),
                std::memory_order_release);
        }
    }

    /** @brief Mark thread slot as having exited its critical section. */
    void exitCritical(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(EPOCH_INACTIVE, std::memory_order_release);
        }
    }

    /**
     * @brief Advance the global epoch and update the safe-to-reclaim epoch.
     *
     * Scans all registered thread slots for the minimum observed epoch among
     * active threads, and if enough progress has been made, raises the
     * safe-to-reclaim epoch so retired memory older than it can be freed.
     */
    void tryAdvanceEpoch() {
        uint64_t current = global_epoch_.load(std::memory_order_relaxed);
        uint64_t min_epoch = current;

        for (size_t i = 0; i < MAX_THREADS; ++i) {
            uint64_t te = thread_epochs_[i].epoch.load(std::memory_order_acquire);
            if (te != EPOCH_INACTIVE && te < min_epoch) {
                min_epoch = te;
            }
        }

        // All threads have observed at least min_epoch
        // Safe to reclaim items retired before min_epoch - 2
        if (min_epoch > retire_epoch_.load(std::memory_order_relaxed) + 2) {
            retire_epoch_.store(min_epoch - 2, std::memory_order_release);
        }

        // Advance global epoch periodically
        global_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }

    /** @brief Get the current global epoch counter. */
    uint64_t currentEpoch() const {
        return global_epoch_.load(std::memory_order_acquire);
    }

    /** @brief Get the epoch below which retired memory is safe to reclaim. */
    uint64_t safeToReclaimEpoch() const {
        return retire_epoch_.load(std::memory_order_acquire);
    }

private:
    struct alignas(CACHE_LINE_SIZE) AlignedEpoch {
        std::atomic<uint64_t> epoch;
        char padding[CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>)];
    };

    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> global_epoch_;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> retire_epoch_;
    AlignedEpoch thread_epochs_[MAX_THREADS];
};

/**
 * @brief RAII guard that brackets an epoch-based critical section.
 *
 * Calls EpochManager::enterCritical() on construction and
 * EpochManager::exitCritical() on destruction for the given thread slot.
 */
class EpochGuard {
public:
    /** @brief Enter the critical section for thread slot on the given epoch manager. */
    EpochGuard(EpochManager& mgr, size_t slot) : mgr_(mgr), slot_(slot) {
        mgr_.enterCritical(slot_);
    }
    /** @brief Exit the critical section entered by the constructor. */
    ~EpochGuard() {
        mgr_.exitCritical(slot_);
    }
private:
    EpochManager& mgr_;
    size_t slot_;
};

// ============================================================================
// Circular Array with Safe Reclamation
// ============================================================================

/**
 * @brief Fixed-capacity circular buffer of `void*` slots backing a WorkStealingDeque.
 *
 * Capacity is always a power of two (2^log_size). When a deque outgrows its
 * current array, a larger CircularArray is created via grow() and the old
 * one is retained (retired) until epoch-based reclamation confirms no thief
 * can still be reading from it.
 */
class CircularArray {
public:
    /**
     * @brief Construct a circular array of size 2^log_size, all slots initialised to nullptr.
     * @param log_size Log2 of the desired capacity
     * @param retire_epoch Epoch at which this array itself was created/retired (for reclamation bookkeeping)
     */
    explicit CircularArray(size_t log_size, uint64_t retire_epoch = 0)
        : log_size_(log_size)
        , mask_((1ULL << log_size) - 1)
        , retire_epoch_(retire_epoch)
        , buffer_(new std::atomic<void*>[1ULL << log_size])
    {
        size_t size = 1ULL << log_size;
        for (size_t i = 0; i < size; ++i) {
            buffer_[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    ~CircularArray() {
        delete[] buffer_;
    }

    /** @brief Get the array's capacity (2^log_size). */
    size_t capacity() const { return mask_ + 1; }
    /** @brief Get log2 of the array's capacity. */
    size_t logSize() const { return log_size_; }
    /** @brief Get the epoch this array was retired at (for reclamation bookkeeping). */
    uint64_t retireEpoch() const { return retire_epoch_; }

    /** @brief Load the slot at index i (wrapped modulo capacity). */
    void* get(int64_t i) const {
        return buffer_[static_cast<size_t>(i) & mask_].load(std::memory_order_relaxed);
    }

    /** @brief Store item into the slot at index i (wrapped modulo capacity). */
    void put(int64_t i, void* item) {
        buffer_[static_cast<size_t>(i) & mask_].store(item, std::memory_order_relaxed);
    }

    /**
     * @brief Create a new CircularArray with double the capacity and copy elements [top, bottom) into it.
     * @param bottom Current bottom index of the deque (exclusive upper bound of live elements)
     * @param top Current top index of the deque (inclusive lower bound of live elements)
     * @param epoch Epoch to stamp on the new array's retire_epoch
     * @return A newly allocated, larger CircularArray (caller takes ownership)
     */
    CircularArray* grow(int64_t bottom, int64_t top, uint64_t epoch) const {
        auto* new_array = new CircularArray(log_size_ + 1, epoch);
        for (int64_t i = top; i < bottom; ++i) {
            new_array->put(i, get(i));
        }
        return new_array;
    }

private:
    size_t log_size_;
    size_t mask_;
    uint64_t retire_epoch_;
    std::atomic<void*>* buffer_;
};

// ============================================================================
// Work-Stealing Deque
// ============================================================================

/**
 * @brief Chase-Lev lock-free work-stealing deque.
 *
 * The owning thread pushes and pops from the bottom (LIFO), while other
 * ("thief") threads steal from the top (FIFO). Backed by a resizable
 * CircularArray whose old buffers are reclaimed via epoch-based memory
 * reclamation once no thief can still be reading them. See the file-level
 * comment for the full algorithm reference.
 */
class WorkStealingDeque {
public:
    /**
     * @brief Construct a deque with an initial capacity of 2^log_initial_size items (default 4096).
     * @param epoch_mgr Epoch manager used to safely reclaim old backing arrays after a resize
     * @param log_initial_size Log2 of the initial capacity
     */
    explicit WorkStealingDeque(EpochManager& epoch_mgr, size_t log_initial_size = 12)
        : epoch_mgr_(epoch_mgr)
        , top_(0)
        , bottom_(0)
        , array_(new CircularArray(log_initial_size))
    {
        retired_arrays_.reserve(16);
    }

    ~WorkStealingDeque() {
        // Clean up current array
        delete array_.load(std::memory_order_relaxed);

        // Clean up any retired arrays
        for (auto* arr : retired_arrays_) {
            delete arr;
        }
    }

    /**
     * @brief Push an item onto the bottom of the deque.
     *
     * Owner-thread only; not safe to call concurrently with another push() or
     * pop() from a different thread. Grows the backing array (and retires the
     * old one) if the deque is at capacity.
     * @param item Opaque task pointer to enqueue
     */
    void push(void* item) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_acquire);
        CircularArray* a = array_.load(std::memory_order_relaxed);

        int64_t size = b - t;
        if (size >= static_cast<int64_t>(a->capacity()) - 1) {
            // Need to grow - create new array at current epoch
            CircularArray* old = a;
            a = a->grow(b, t, epoch_mgr_.currentEpoch());
            array_.store(a, std::memory_order_release);

            // Retire old array for later cleanup
            retireArray(old);
        }

        a->put(b, item);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    /**
     * @brief Pop an item from the bottom of the deque.
     *
     * Owner-thread only; not safe to call concurrently with another pop() from
     * a different thread. Competes with concurrent steal() calls when only one
     * item remains.
     * @return The popped item, or nullptr if the deque is empty (or lost the race for the last item)
     */
    void* pop() {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        CircularArray* a = array_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_seq_cst);

        int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty
            void* item = a->get(b);

            if (t == b) {
                // Last item - compete with thieves
                if (!top_.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // Lost to thief
                    item = nullptr;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return item;
        } else {
            // Empty
            bottom_.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    /**
     * @brief Steal an item from the top of the deque.
     *
     * Thread-safe; may be called concurrently by any number of thief threads.
     * @return The stolen item, or nullptr if the deque is empty or the steal lost a race
     */
    void* steal() {
        int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t < b) {
            // Non-empty
            CircularArray* a = array_.load(std::memory_order_acquire);
            void* item = a->get(t);

            if (!top_.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                // Lost race - return failure
                return nullptr;
            }
            return item;
        }

        return nullptr;  // Empty
    }

    /** @brief Approximate emptiness check (heuristic only, not authoritative under concurrency). */
    bool empty() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b <= t;
    }

    /** @brief Approximate item count (heuristic only, for load-balancing decisions). */
    size_t size() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        int64_t s = b - t;
        return s > 0 ? static_cast<size_t>(s) : 0;
    }

    /** @brief Delete any retired backing arrays that the epoch manager has confirmed are safe to free. */
    void reclaimRetired() {
        uint64_t safe_epoch = epoch_mgr_.safeToReclaimEpoch();

        auto it = retired_arrays_.begin();
        while (it != retired_arrays_.end()) {
            if ((*it)->retireEpoch() < safe_epoch) {
                delete *it;
                it = retired_arrays_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    void retireArray(CircularArray* arr) {
        retired_arrays_.push_back(arr);

        // Periodically clean up
        if (retired_arrays_.size() > 8) {
            reclaimRetired();
        }
    }

    // Prevent copying
    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    EpochManager& epoch_mgr_;

    // Cache-line aligned to prevent false sharing between top/bottom
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> top_;
    alignas(CACHE_LINE_SIZE) std::atomic<int64_t> bottom_;
    alignas(CACHE_LINE_SIZE) std::atomic<CircularArray*> array_;

    // Retired arrays waiting for reclamation (accessed only by owner)
    std::vector<CircularArray*> retired_arrays_;
};

// ============================================================================
// Work-Stealing Scheduler
// ============================================================================

/** @brief Atomic performance counters tracked by a WorkStealingScheduler. */
struct WorkStealingStats {
    std::atomic<uint64_t> tasks_executed{0};
    std::atomic<uint64_t> tasks_stolen{0};
    std::atomic<uint64_t> steal_attempts{0};
    std::atomic<uint64_t> steal_failures{0};
    std::atomic<uint64_t> local_pops{0};
    std::atomic<uint64_t> local_pushes{0};
};

/**
 * @brief Multi-worker task scheduler built on per-worker WorkStealingDeque instances.
 *
 * Each worker thread owns a deque; it pops its own work first (LIFO, cache-
 * friendly) and, when empty, steals from a randomly chosen peer's deque
 * (FIFO). Workers sleep on a condition variable when no work is found after
 * a bounded number of spin/yield attempts.
 */
class WorkStealingScheduler {
public:
    /** @brief A unit of scheduled work along with its completion state, for use with wait()/isDone(). */
    struct Task {
        void* (*fn)(void*);
        void* arg;
        void* result;
        std::atomic<bool> completed{false};
        std::atomic<bool> has_waiter{false};
        std::mutex waiter_mutex;
        std::condition_variable waiter_cv;
    };

    /**
     * @brief Construct a scheduler with the given number of worker deques (workers are not started yet).
     * @param num_workers Number of workers; 0 uses std::thread::hardware_concurrency() (falling back to 4)
     */
    explicit WorkStealingScheduler(size_t num_workers = 0)
        : num_workers_(num_workers == 0 ? std::thread::hardware_concurrency() : num_workers)
        , running_(false)
        , epoch_mgr_()
        , stats_()
    {
        if (num_workers_ == 0) num_workers_ = 4;  // Fallback

        // Create per-worker deques
        deques_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            deques_.push_back(std::make_unique<WorkStealingDeque>(epoch_mgr_));
        }

        // Initialize per-worker RNGs for random victim selection
        worker_rngs_.reserve(num_workers_);
        std::random_device rd;
        for (size_t i = 0; i < num_workers_; ++i) {
            worker_rngs_.push_back(std::mt19937(rd() + i));
        }
    }

    ~WorkStealingScheduler() {
        stop();
    }

    /** @brief Start the worker threads. No-op if already running. */
    void start() {
        if (running_.exchange(true)) return;  // Already running

        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&WorkStealingScheduler::workerLoop, this, i);
        }
    }

    /** @brief Stop and join all worker threads. No-op if not running. */
    void stop() {
        if (!running_.exchange(false)) return;  // Already stopped

        // Wake up all workers
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_cv_.notify_all();
        }

        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    /**
     * @brief Submit a task for execution. Callable from any thread (worker or external).
     *
     * If called from a worker thread, the task is pushed to that worker's own
     * deque; otherwise it is pushed to a randomly chosen worker's deque.
     * @param fn Task function to run
     * @param arg Argument passed to fn
     * @return A newly allocated Task*; pass it to wait() or isDone() to retrieve its result.
     *         Ownership transfers to the caller of wait(), which deletes it.
     */
    Task* submit(void* (*fn)(void*), void* arg) {
        auto* task = new Task();
        task->fn = fn;
        task->arg = arg;
        task->result = nullptr;

        // Determine which deque to push to
        size_t worker_id = getCurrentWorkerId();
        if (worker_id < num_workers_) {
            // Called from worker - push to local deque
            deques_[worker_id]->push(task);
            stats_.local_pushes.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Called from external thread - push to random deque
            size_t target = std::hash<std::thread::id>{}(std::this_thread::get_id()) % num_workers_;
            deques_[target]->push(task);
        }

        // Wake up a worker
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_cv_.notify_one();
        }

        return task;
    }

    /**
     * @brief Block until a submitted task completes, then return its result and delete the Task.
     * @param task Task returned by submit()
     * @return The value returned by the task's function, or nullptr if task is null
     */
    void* wait(Task* task) {
        if (!task) return nullptr;

        if (!task->completed.load(std::memory_order_acquire)) {
            task->has_waiter.store(true, std::memory_order_release);
            std::unique_lock<std::mutex> lock(task->waiter_mutex);
            task->waiter_cv.wait(lock, [task]() {
                return task->completed.load(std::memory_order_acquire);
            });
        }

        void* result = task->result;
        delete task;
        return result;
    }

    /** @brief Non-blocking check of whether a submitted task has completed. */
    bool isDone(Task* task) const {
        return task && task->completed.load(std::memory_order_acquire);
    }

    /** @brief Get the number of worker threads configured for this scheduler. */
    size_t numWorkers() const { return num_workers_; }

    /** @brief Get the scheduler's live performance counters. */
    const WorkStealingStats& stats() const { return stats_; }

    /** @brief Reset all performance counters to zero. */
    void resetStats() {
        stats_.tasks_executed.store(0, std::memory_order_relaxed);
        stats_.tasks_stolen.store(0, std::memory_order_relaxed);
        stats_.steal_attempts.store(0, std::memory_order_relaxed);
        stats_.steal_failures.store(0, std::memory_order_relaxed);
        stats_.local_pops.store(0, std::memory_order_relaxed);
        stats_.local_pushes.store(0, std::memory_order_relaxed);
    }

private:
    void workerLoop(size_t worker_id) {
        // Register with epoch manager
        size_t epoch_slot = epoch_mgr_.registerThread();
        setCurrentWorkerId(worker_id);

        WorkStealingDeque& my_deque = *deques_[worker_id];
        std::mt19937& rng = worker_rngs_[worker_id];

        size_t idle_spins = 0;
        constexpr size_t MAX_IDLE_SPINS = 32;
        constexpr size_t STEAL_ATTEMPTS_BEFORE_SLEEP = 64;

        while (running_.load(std::memory_order_acquire)) {
            Task* task = nullptr;

            // Enter epoch critical section for safe array access
            {
                EpochGuard guard(epoch_mgr_, epoch_slot);

                // 1. Try local deque first (LIFO - good cache locality)
                task = static_cast<Task*>(my_deque.pop());

                if (task) {
                    stats_.local_pops.fetch_add(1, std::memory_order_relaxed);
                    idle_spins = 0;
                } else {
                    // 2. Try to steal from others
                    task = trySteal(worker_id, rng);

                    if (task) {
                        stats_.tasks_stolen.fetch_add(1, std::memory_order_relaxed);
                        idle_spins = 0;
                    }
                }
            }

            if (task) {
                executeTask(task);

                // Periodically advance epoch and reclaim memory
                if ((stats_.tasks_executed.load(std::memory_order_relaxed) & 0xFF) == 0) {
                    epoch_mgr_.tryAdvanceEpoch();
                    my_deque.reclaimRetired();
                }
            } else {
                // No work found
                ++idle_spins;

                if (idle_spins < MAX_IDLE_SPINS) {
                    // Spin briefly
                    std::this_thread::yield();
                } else if (idle_spins < STEAL_ATTEMPTS_BEFORE_SLEEP) {
                    // Yield then try stealing again
                    std::this_thread::yield();
                } else {
                    // Sleep until notified
                    std::unique_lock<std::mutex> lock(global_mutex_);
                    global_cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() {
                        return !running_.load(std::memory_order_relaxed) || hasWork();
                    });
                    idle_spins = 0;
                }
            }
        }

        epoch_mgr_.unregisterThread(epoch_slot);
    }

    Task* trySteal(size_t worker_id, std::mt19937& rng) {
        // Random victim selection (better load balancing than round-robin)
        std::uniform_int_distribution<size_t> dist(0, num_workers_ - 1);

        // Try multiple victims before giving up
        for (size_t attempt = 0; attempt < num_workers_; ++attempt) {
            size_t victim = dist(rng);
            if (victim == worker_id) continue;  // Don't steal from self

            stats_.steal_attempts.fetch_add(1, std::memory_order_relaxed);

            Task* task = static_cast<Task*>(deques_[victim]->steal());
            if (task) {
                return task;
            }

            stats_.steal_failures.fetch_add(1, std::memory_order_relaxed);
        }

        return nullptr;
    }

    void executeTask(Task* task) {
        // Execute
        void* result = task->fn(task->arg);

        // Complete
        task->result = result;
        task->completed.store(true, std::memory_order_release);

        // Wake waiter if any
        if (task->has_waiter.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(task->waiter_mutex);
            task->waiter_cv.notify_one();
        }

        stats_.tasks_executed.fetch_add(1, std::memory_order_relaxed);
    }

    bool hasWork() const {
        for (const auto& deque : deques_) {
            if (!deque->empty()) return true;
        }
        return false;
    }

    // Thread-local worker ID
    static thread_local size_t tl_worker_id_;
    static constexpr size_t INVALID_WORKER_ID = SIZE_MAX;

    static size_t getCurrentWorkerId() { return tl_worker_id_; }
    static void setCurrentWorkerId(size_t id) { tl_worker_id_ = id; }

    size_t num_workers_;
    std::atomic<bool> running_;
    EpochManager epoch_mgr_;

    std::vector<std::unique_ptr<WorkStealingDeque>> deques_;
    std::vector<std::mt19937> worker_rngs_;
    std::vector<std::thread> workers_;

    // For sleeping workers
    std::mutex global_mutex_;
    std::condition_variable global_cv_;

    WorkStealingStats stats_;
};

// Define thread-local storage
inline thread_local size_t WorkStealingScheduler::tl_worker_id_ = WorkStealingScheduler::INVALID_WORKER_ID;

} // namespace eshkol

#endif // ESHKOL_WORK_STEALING_DEQUE_H
