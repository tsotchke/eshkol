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

class EpochManager {
public:
    static constexpr size_t MAX_THREADS = 256;
    static constexpr uint64_t EPOCH_INACTIVE = UINT64_MAX;

    EpochManager() : global_epoch_(0), retire_epoch_(0) {
        for (size_t i = 0; i < MAX_THREADS; ++i) {
            thread_epochs_[i].epoch.store(EPOCH_INACTIVE, std::memory_order_relaxed);
        }
    }

    // Register thread and return slot ID
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

    void unregisterThread(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(EPOCH_INACTIVE, std::memory_order_release);
        }
    }

    // Enter critical section
    void enterCritical(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(
                global_epoch_.load(std::memory_order_relaxed),
                std::memory_order_release);
        }
    }

    // Exit critical section
    void exitCritical(size_t slot) {
        if (slot < MAX_THREADS) {
            thread_epochs_[slot].epoch.store(EPOCH_INACTIVE, std::memory_order_release);
        }
    }

    // Try to advance global epoch
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

    uint64_t currentEpoch() const {
        return global_epoch_.load(std::memory_order_acquire);
    }

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

// RAII guard for epoch critical section
class EpochGuard {
public:
    EpochGuard(EpochManager& mgr, size_t slot) : mgr_(mgr), slot_(slot) {
        mgr_.enterCritical(slot_);
    }
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

class CircularArray {
public:
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

    size_t capacity() const { return mask_ + 1; }
    size_t logSize() const { return log_size_; }
    uint64_t retireEpoch() const { return retire_epoch_; }

    void* get(int64_t i) const {
        return buffer_[static_cast<size_t>(i) & mask_].load(std::memory_order_relaxed);
    }

    void put(int64_t i, void* item) {
        buffer_[static_cast<size_t>(i) & mask_].store(item, std::memory_order_relaxed);
    }

    // Create larger array and copy elements
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

class WorkStealingDeque {
public:
    // Initial capacity: 2^log_initial_size (default 4096 items)
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

    // Push item to bottom (owner only, not thread-safe with other pushes)
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

    // Pop item from bottom (owner only, not thread-safe with other pops)
    // Returns nullptr if empty
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

    // Steal item from top (thread-safe, called by thieves)
    // Returns nullptr if empty or contended
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

    // Bulk steal - try to steal multiple items at once
    // Returns number of items stolen
    size_t stealBatch(void** items, size_t max_items) {
        size_t stolen = 0;

        while (stolen < max_items) {
            void* item = steal();
            if (!item) break;
            items[stolen++] = item;
        }

        return stolen;
    }

    // Check if approximately empty (for heuristics, not authoritative)
    bool empty() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return b <= t;
    }

    // Approximate size (for heuristics)
    size_t size() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        int64_t s = b - t;
        return s > 0 ? static_cast<size_t>(s) : 0;
    }

    // Clean up retired arrays that are safe to delete
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

struct WorkStealingStats {
    std::atomic<uint64_t> tasks_executed{0};
    std::atomic<uint64_t> tasks_stolen{0};
    std::atomic<uint64_t> steal_attempts{0};
    std::atomic<uint64_t> steal_failures{0};
    std::atomic<uint64_t> local_pops{0};
    std::atomic<uint64_t> local_pushes{0};
};

class WorkStealingScheduler {
public:
    struct Task {
        void* (*fn)(void*);
        void* arg;
        void* result;
        std::atomic<bool> completed{false};
        std::atomic<bool> has_waiter{false};
        std::mutex waiter_mutex;
        std::condition_variable waiter_cv;
    };

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

    void start() {
        if (running_.exchange(true)) return;  // Already running

        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&WorkStealingScheduler::workerLoop, this, i);
        }
    }

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

    // Submit task (can be called from any thread)
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

    // Submit batch of tasks
    void submitBatch(Task** tasks, size_t count) {
        if (count == 0) return;

        size_t worker_id = getCurrentWorkerId();
        size_t target = (worker_id < num_workers_) ? worker_id :
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % num_workers_;

        for (size_t i = 0; i < count; ++i) {
            deques_[target]->push(tasks[i]);
        }
        stats_.local_pushes.fetch_add(count, std::memory_order_relaxed);

        // Wake up workers
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_cv_.notify_all();
        }
    }

    // Wait for task completion
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

    // Check if task is done (non-blocking)
    bool isDone(Task* task) const {
        return task && task->completed.load(std::memory_order_acquire);
    }

    size_t numWorkers() const { return num_workers_; }

    const WorkStealingStats& stats() const { return stats_; }

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
