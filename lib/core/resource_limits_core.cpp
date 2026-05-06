/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Resource Limits Implementation
 *
 * Provides memory tracking, stack depth monitoring, execution timeout,
 * and configurable limits for production deployments.
 */

#include <eshkol/core/resource_limits.h>
#include <eshkol/core/runtime.h>

namespace {

// ============================================================================
// Global State
// ============================================================================

// Active resource limits
eshkol_resource_limits_t g_limits;

// Memory tracking
size_t g_heap_usage = 0;
size_t g_peak_heap_usage = 0;
bool g_soft_limit_warned = false;

#if defined(__STDC_HOSTED__) && (__STDC_HOSTED__ == 1)
thread_local size_t t_stack_depth = 0;
size_t& current_stack_depth() { return t_stack_depth; }
#else
size_t g_stack_depth = 0;
size_t& current_stack_depth() { return g_stack_depth; }
#endif

unsigned char g_limits_lock = 0;

// Timer state
bool g_timer_active = false;
uint64_t g_timer_start_ns = 0;
uint64_t g_timer_timeout_ms = 0;
bool g_timer_timebase_available = false;
bool g_timer_timebase_failure_warned = false;

// Last error
eshkol_limit_error_t g_last_error = ESHKOL_LIMIT_OK;

template <typename T>
T atomic_load_relaxed(const T* value) {
    return __atomic_load_n(value, __ATOMIC_RELAXED);
}

template <typename T>
T atomic_load_acquire(const T* value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

template <typename T>
void atomic_store_relaxed(T* target, T value) {
    __atomic_store_n(target, value, __ATOMIC_RELAXED);
}

template <typename T>
void atomic_store_release(T* target, T value) {
    __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

template <typename T>
T atomic_fetch_add_relaxed(T* target, T value) {
    return __atomic_fetch_add(target, value, __ATOMIC_RELAXED);
}

template <typename T>
T atomic_fetch_sub_relaxed(T* target, T value) {
    return __atomic_fetch_sub(target, value, __ATOMIC_RELAXED);
}

template <typename T>
T atomic_exchange_relaxed(T* target, T value) {
    return __atomic_exchange_n(target, value, __ATOMIC_RELAXED);
}

template <typename T>
bool atomic_compare_exchange(T* target, T* expected, T desired, int success_order,
                             int failure_order) {
    return __atomic_compare_exchange_n(target, expected, desired, false,
                                       success_order, failure_order);
}

struct LimitsLockGuard {
    LimitsLockGuard() {
        while (__atomic_test_and_set(&g_limits_lock, __ATOMIC_ACQUIRE)) {
        }
    }

    ~LimitsLockGuard() { __atomic_clear(&g_limits_lock, __ATOMIC_RELEASE); }
};

// ============================================================================
// Helper Functions
// ============================================================================

// Update peak usage
void update_peak(size_t current) {
    size_t peak = atomic_load_relaxed(&g_peak_heap_usage);
    while (current > peak) {
        if (atomic_compare_exchange(&g_peak_heap_usage, &peak, current,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            break;
        }
    }
}

void disable_timer_due_to_timebase_failure(const char* phase) {
    atomic_store_release(&g_timer_active, false);
    atomic_store_release(&g_timer_timebase_available, false);

    bool expected = false;
    if (atomic_compare_exchange(&g_timer_timebase_failure_warned, &expected, true,
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        eshkol_runtime_warnf(
            "Execution timer disabled: runtime monotonic time source unavailable during %s",
            phase ? phase : "timer operation");
    }
}

bool get_elapsed_time_ms(uint64_t* elapsed_ms) {
    if (!elapsed_ms) {
        return false;
    }

    const uint64_t start_ns = atomic_load_acquire(&g_timer_start_ns);
    uint64_t now_ns = 0;
    if (!eshkol_runtime_get_monotonic_time_ns(&now_ns)) {
        disable_timer_due_to_timebase_failure("timer read");
        return false;
    }

    if (now_ns <= start_ns) {
        *elapsed_ms = 0;
        return true;
    }

    *elapsed_ms = (now_ns - start_ns) / 1000000ULL;
    return true;
}

} // anonymous namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

eshkol_resource_limits_t eshkol_get_default_limits(void) {
    return {
        .max_heap_bytes = ESHKOL_DEFAULT_MAX_HEAP_BYTES,
        .heap_soft_limit_bytes = (ESHKOL_DEFAULT_MAX_HEAP_BYTES * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100,
        .max_execution_time_ms = ESHKOL_DEFAULT_TIMEOUT_MS,
        .max_stack_depth = ESHKOL_DEFAULT_MAX_STACK_DEPTH,
        .max_tensor_elements = ESHKOL_DEFAULT_MAX_TENSOR_ELEMENTS,
        .max_string_length = ESHKOL_DEFAULT_MAX_STRING_LENGTH,
        .enforce_hard_limits = true,
        .enable_warnings = true
    };
}

void eshkol_set_limits(const eshkol_resource_limits_t* limits) {
    if (!limits) return;

    LimitsLockGuard lock;
    g_limits = *limits;

    eshkol_runtime_infof(
        "Resource limits configured: heap=%zuMB, timeout=%llums, stack=%zu",
        g_limits.max_heap_bytes / (1024 * 1024),
        (unsigned long long)g_limits.max_execution_time_ms,
        g_limits.max_stack_depth);
}

const eshkol_resource_limits_t* eshkol_get_limits(void) {
    return &g_limits;
}

// ----------------------------------------------------------------------------
// Memory Tracking
// ----------------------------------------------------------------------------

bool eshkol_track_allocation(size_t bytes) {
    if (bytes == 0) return true;

    size_t current = atomic_fetch_add_relaxed(&g_heap_usage, bytes) + bytes;
    update_peak(current);

    // Check soft limit
    if (g_limits.enable_warnings &&
        current >= g_limits.heap_soft_limit_bytes &&
        !atomic_exchange_relaxed(&g_soft_limit_warned, true)) {
        eshkol_runtime_warnf("Heap usage at %zu%% of limit (%zuMB / %zuMB)",
                             (current * 100) / g_limits.max_heap_bytes,
                             current / (1024 * 1024),
                             g_limits.max_heap_bytes / (1024 * 1024));
        atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_HEAP_SOFT);
    }

    // Check hard limit
    if (current > g_limits.max_heap_bytes) {
        atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_HEAP_HARD);

        if (g_limits.enforce_hard_limits) {
            eshkol_runtime_errorf("Heap limit exceeded: %zuMB > %zuMB",
                                  current / (1024 * 1024),
                                  g_limits.max_heap_bytes / (1024 * 1024));
            eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_MEMORY);
            return false;
        }
    }

    return true;
}

void eshkol_track_deallocation(size_t bytes) {
    if (bytes == 0) return;
    atomic_fetch_sub_relaxed(&g_heap_usage, bytes);
}

size_t eshkol_get_heap_usage(void) {
    return atomic_load_relaxed(&g_heap_usage);
}

size_t eshkol_get_peak_heap_usage(void) {
    return atomic_load_relaxed(&g_peak_heap_usage);
}

bool eshkol_is_near_memory_limit(void) {
    size_t current = atomic_load_relaxed(&g_heap_usage);
    size_t threshold = (g_limits.max_heap_bytes * 90) / 100;  // 90% of limit
    return current >= threshold;
}

// ----------------------------------------------------------------------------
// Stack Tracking
// ----------------------------------------------------------------------------

bool eshkol_stack_push(void) {
    size_t& stack_depth = current_stack_depth();
    stack_depth++;

    if (stack_depth > g_limits.max_stack_depth) {
        atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_STACK_OVERFLOW);

        if (g_limits.enforce_hard_limits) {
            eshkol_runtime_errorf("Stack overflow: depth %zu > limit %zu",
                                  stack_depth, g_limits.max_stack_depth);
            stack_depth--;
            return false;
        }
    }

    return true;
}

void eshkol_stack_pop(void) {
    size_t& stack_depth = current_stack_depth();
    if (stack_depth > 0) {
        stack_depth--;
    }
}

size_t eshkol_get_stack_depth(void) {
    return current_stack_depth();
}

// ----------------------------------------------------------------------------
// Timeout Watchdog
// ----------------------------------------------------------------------------

void eshkol_start_timer(uint64_t timeout_ms) {
    const uint64_t effective_timeout =
        timeout_ms > 0 ? timeout_ms : g_limits.max_execution_time_ms;
    atomic_store_release(&g_timer_timeout_ms, effective_timeout);
    atomic_store_release(&g_timer_timebase_failure_warned, false);

    if (effective_timeout == 0) {
        atomic_store_release(&g_timer_start_ns, static_cast<uint64_t>(0));
        atomic_store_release(&g_timer_timebase_available, false);
        atomic_store_release(&g_timer_active, true);
        eshkol_runtime_debugf("Execution timer started: unlimited");
        return;
    }

    uint64_t start_ns = 0;
    if (!eshkol_runtime_get_monotonic_time_ns(&start_ns)) {
        atomic_store_release(&g_timer_start_ns, static_cast<uint64_t>(0));
        atomic_store_release(&g_timer_timebase_available, false);
        atomic_store_release(&g_timer_active, false);
        atomic_store_release(&g_timer_timebase_failure_warned, true);
        eshkol_runtime_warnf(
            "Execution timer could not start: runtime monotonic time source unavailable");
        return;
    }

    atomic_store_release(&g_timer_start_ns, start_ns);
    atomic_store_release(&g_timer_timebase_available, true);
    atomic_store_release(&g_timer_active, true);
    eshkol_runtime_debugf("Execution timer started: %llu ms",
                          (unsigned long long)atomic_load_relaxed(&g_timer_timeout_ms));
}

void eshkol_stop_timer(void) {
    atomic_store_release(&g_timer_active, false);
    atomic_store_release(&g_timer_timebase_available, false);
    atomic_store_release(&g_timer_timebase_failure_warned, false);
    eshkol_runtime_debugf("Execution timer stopped");
}

bool eshkol_is_timed_out(void) {
    if (!atomic_load_acquire(&g_timer_active)) {
        return false;
    }

    uint64_t timeout = atomic_load_acquire(&g_timer_timeout_ms);
    if (timeout == 0) {
        return false;  // No timeout set
    }

    if (!atomic_load_acquire(&g_timer_timebase_available)) {
        return false;
    }

    uint64_t elapsed = 0;
    if (!get_elapsed_time_ms(&elapsed)) {
        return false;
    }

    if (elapsed >= timeout) {
        atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_TIMEOUT);

        if (g_limits.enforce_hard_limits) {
            eshkol_runtime_errorf("Execution timeout: %lld ms >= %llu ms limit",
                                  (long long)elapsed,
                                  (unsigned long long)timeout);
            eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_TIMEOUT);
        }
        return true;
    }

    return false;
}

uint64_t eshkol_get_remaining_time_ms(void) {
    if (!atomic_load_acquire(&g_timer_active)) {
        return 0;
    }

    uint64_t timeout = atomic_load_acquire(&g_timer_timeout_ms);
    if (timeout == 0) {
        return UINT64_MAX;  // No timeout = unlimited
    }

    if (!atomic_load_acquire(&g_timer_timebase_available)) {
        return 0;
    }

    uint64_t elapsed = 0;
    if (!get_elapsed_time_ms(&elapsed)) {
        return 0;
    }

    if (elapsed >= timeout) {
        return 0;
    }

    return timeout - elapsed;
}

// ----------------------------------------------------------------------------
// Validation Functions
// ----------------------------------------------------------------------------

bool eshkol_check_tensor_size(size_t num_elements) {
    if (num_elements <= g_limits.max_tensor_elements) {
        return true;
    }

    atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_TENSOR_SIZE);

    if (g_limits.enforce_hard_limits) {
        eshkol_runtime_errorf("Tensor size limit exceeded: %zu > %zu elements",
                              num_elements, g_limits.max_tensor_elements);
    }

    return false;
}

bool eshkol_check_string_length(size_t length) {
    if (length <= g_limits.max_string_length) {
        return true;
    }

    atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_STRING_LENGTH);

    if (g_limits.enforce_hard_limits) {
        eshkol_runtime_errorf("String length limit exceeded: %zu > %zu bytes",
                              length, g_limits.max_string_length);
    }

    return false;
}

// ----------------------------------------------------------------------------
// Error Reporting
// ----------------------------------------------------------------------------

eshkol_limit_error_t eshkol_get_last_limit_error(void) {
    return atomic_load_relaxed(&g_last_error);
}

const char* eshkol_limit_error_message(eshkol_limit_error_t error) {
    switch (error) {
        case ESHKOL_LIMIT_OK:
            return "No error";
        case ESHKOL_LIMIT_HEAP_SOFT:
            return "Heap soft limit reached (warning)";
        case ESHKOL_LIMIT_HEAP_HARD:
            return "Heap hard limit exceeded";
        case ESHKOL_LIMIT_TIMEOUT:
            return "Execution timeout exceeded";
        case ESHKOL_LIMIT_STACK_OVERFLOW:
            return "Stack overflow (recursion too deep)";
        case ESHKOL_LIMIT_TENSOR_SIZE:
            return "Tensor size limit exceeded";
        case ESHKOL_LIMIT_STRING_LENGTH:
            return "String length limit exceeded";
        default:
            return "Unknown limit error";
    }
}

// ----------------------------------------------------------------------------
// Diagnostics
// ----------------------------------------------------------------------------

void eshkol_print_resource_stats(void) {
    eshkol_runtime_infof("=== Resource Usage Statistics ===");
    eshkol_runtime_infof("Heap: current=%zuMB, peak=%zuMB, limit=%zuMB",
                         atomic_load_relaxed(&g_heap_usage) / (1024 * 1024),
                         atomic_load_relaxed(&g_peak_heap_usage) / (1024 * 1024),
                         g_limits.max_heap_bytes / (1024 * 1024));
    eshkol_runtime_infof("Stack: current=%zu, limit=%zu", current_stack_depth(),
                         g_limits.max_stack_depth);

    if (atomic_load_relaxed(&g_timer_active)) {
        uint64_t remaining = eshkol_get_remaining_time_ms();
        eshkol_runtime_infof("Timer: %llums remaining of %llums",
                             (unsigned long long)remaining,
                             (unsigned long long)atomic_load_relaxed(&g_timer_timeout_ms));
    } else {
        eshkol_runtime_infof("Timer: inactive");
    }

    eshkol_limit_error_t last_error = atomic_load_relaxed(&g_last_error);
    if (last_error != ESHKOL_LIMIT_OK) {
        eshkol_runtime_infof("Last limit error: %s",
                             eshkol_limit_error_message(last_error));
    }
}

void eshkol_reset_resource_tracking(void) {
    atomic_store_relaxed(&g_heap_usage, static_cast<size_t>(0));
    atomic_store_relaxed(&g_peak_heap_usage, static_cast<size_t>(0));
    atomic_store_relaxed(&g_soft_limit_warned, false);
    current_stack_depth() = 0;
    atomic_store_relaxed(&g_timer_active, false);
    atomic_store_relaxed(&g_timer_start_ns, static_cast<uint64_t>(0));
    atomic_store_relaxed(&g_timer_timebase_available, false);
    atomic_store_relaxed(&g_timer_timebase_failure_warned, false);
    atomic_store_relaxed(&g_last_error, ESHKOL_LIMIT_OK);
    eshkol_runtime_debugf("Resource tracking reset");
}

} // extern "C"
