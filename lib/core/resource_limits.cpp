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
#include <eshkol/logger.h>

#include <atomic>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

// ============================================================================
// Global State
// ============================================================================

// Active resource limits
eshkol_resource_limits_t g_limits;
std::mutex g_limits_mutex;

// Memory tracking
std::atomic<size_t> g_heap_usage{0};
std::atomic<size_t> g_peak_heap_usage{0};
std::atomic<bool> g_soft_limit_warned{false};

// Stack tracking (thread-local)
thread_local size_t t_stack_depth = 0;

// Timer state
std::atomic<bool> g_timer_active{false};
std::chrono::steady_clock::time_point g_timer_start;
std::atomic<uint64_t> g_timer_timeout_ms{0};

// Last error
std::atomic<eshkol_limit_error_t> g_last_error{ESHKOL_LIMIT_OK};

// ============================================================================
// Helper Functions
// ============================================================================

// Parse size with K/M/G suffix
size_t parse_size(const char* str) {
    if (!str || !*str) return 0;

    char* end = nullptr;
    double value = strtod(str, &end);
    if (end == str) return 0;

    // Check for suffix
    if (end && *end) {
        switch (*end) {
            case 'K': case 'k':
                value *= 1024;
                break;
            case 'M': case 'm':
                value *= 1024 * 1024;
                break;
            case 'G': case 'g':
                value *= 1024 * 1024 * 1024;
                break;
        }
    }

    return static_cast<size_t>(value);
}

// Parse boolean
bool parse_bool(const char* str) {
    if (!str) return false;
    return (strcmp(str, "true") == 0 ||
            strcmp(str, "TRUE") == 0 ||
            strcmp(str, "1") == 0 ||
            strcmp(str, "yes") == 0 ||
            strcmp(str, "YES") == 0);
}

// Update peak usage
void update_peak(size_t current) {
    size_t peak = g_peak_heap_usage.load(std::memory_order_relaxed);
    while (current > peak) {
        if (g_peak_heap_usage.compare_exchange_weak(peak, current,
                                                     std::memory_order_relaxed)) {
            break;
        }
    }
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

eshkol_resource_limits_t eshkol_init_limits_from_env(void) {
    eshkol_resource_limits_t limits = eshkol_get_default_limits();

    // ESHKOL_MAX_HEAP
    const char* max_heap = std::getenv("ESHKOL_MAX_HEAP");
    if (max_heap) {
        limits.max_heap_bytes = parse_size(max_heap);
        limits.heap_soft_limit_bytes = (limits.max_heap_bytes * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100;
        eshkol_debug("Max heap from env: %zu bytes", limits.max_heap_bytes);
    }

    // ESHKOL_TIMEOUT_MS
    const char* timeout = std::getenv("ESHKOL_TIMEOUT_MS");
    if (timeout) {
        limits.max_execution_time_ms = static_cast<uint64_t>(atoll(timeout));
        eshkol_debug("Timeout from env: %llu ms", (unsigned long long)limits.max_execution_time_ms);
    }

    // ESHKOL_MAX_STACK
    const char* max_stack = std::getenv("ESHKOL_MAX_STACK");
    if (max_stack) {
        limits.max_stack_depth = static_cast<size_t>(atoll(max_stack));
        eshkol_debug("Max stack from env: %zu", limits.max_stack_depth);
    }

    // ESHKOL_MAX_TENSOR_ELEMS
    const char* max_tensor = std::getenv("ESHKOL_MAX_TENSOR_ELEMS");
    if (max_tensor) {
        limits.max_tensor_elements = parse_size(max_tensor);
        eshkol_debug("Max tensor elements from env: %zu", limits.max_tensor_elements);
    }

    // ESHKOL_MAX_STRING_LEN
    const char* max_string = std::getenv("ESHKOL_MAX_STRING_LEN");
    if (max_string) {
        limits.max_string_length = parse_size(max_string);
        eshkol_debug("Max string length from env: %zu", limits.max_string_length);
    }

    // ESHKOL_ENFORCE_LIMITS
    const char* enforce = std::getenv("ESHKOL_ENFORCE_LIMITS");
    if (enforce) {
        limits.enforce_hard_limits = parse_bool(enforce);
        eshkol_debug("Enforce limits from env: %s", limits.enforce_hard_limits ? "true" : "false");
    }

    // ESHKOL_LIMIT_WARNINGS
    const char* warnings = std::getenv("ESHKOL_LIMIT_WARNINGS");
    if (warnings) {
        limits.enable_warnings = parse_bool(warnings);
        eshkol_debug("Limit warnings from env: %s", limits.enable_warnings ? "true" : "false");
    }

    eshkol_set_limits(&limits);
    return limits;
}

void eshkol_set_limits(const eshkol_resource_limits_t* limits) {
    if (!limits) return;

    std::lock_guard<std::mutex> lock(g_limits_mutex);
    g_limits = *limits;

    eshkol_info("Resource limits configured: heap=%zuMB, timeout=%llums, stack=%zu",
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

    size_t current = g_heap_usage.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    update_peak(current);

    // Check soft limit
    if (g_limits.enable_warnings &&
        current >= g_limits.heap_soft_limit_bytes &&
        !g_soft_limit_warned.exchange(true, std::memory_order_relaxed)) {
        eshkol_warn("Heap usage at %zu%% of limit (%zuMB / %zuMB)",
                    (current * 100) / g_limits.max_heap_bytes,
                    current / (1024 * 1024),
                    g_limits.max_heap_bytes / (1024 * 1024));
        g_last_error.store(ESHKOL_LIMIT_HEAP_SOFT, std::memory_order_relaxed);
    }

    // Check hard limit
    if (current > g_limits.max_heap_bytes) {
        g_last_error.store(ESHKOL_LIMIT_HEAP_HARD, std::memory_order_relaxed);

        if (g_limits.enforce_hard_limits) {
            eshkol_error("Heap limit exceeded: %zuMB > %zuMB",
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
    g_heap_usage.fetch_sub(bytes, std::memory_order_relaxed);
}

size_t eshkol_get_heap_usage(void) {
    return g_heap_usage.load(std::memory_order_relaxed);
}

size_t eshkol_get_peak_heap_usage(void) {
    return g_peak_heap_usage.load(std::memory_order_relaxed);
}

bool eshkol_is_near_memory_limit(void) {
    size_t current = g_heap_usage.load(std::memory_order_relaxed);
    size_t threshold = (g_limits.max_heap_bytes * 90) / 100;  // 90% of limit
    return current >= threshold;
}

// ----------------------------------------------------------------------------
// Stack Tracking
// ----------------------------------------------------------------------------

bool eshkol_stack_push(void) {
    t_stack_depth++;

    if (t_stack_depth > g_limits.max_stack_depth) {
        g_last_error.store(ESHKOL_LIMIT_STACK_OVERFLOW, std::memory_order_relaxed);

        if (g_limits.enforce_hard_limits) {
            eshkol_error("Stack overflow: depth %zu > limit %zu",
                         t_stack_depth, g_limits.max_stack_depth);
            t_stack_depth--;
            return false;
        }
    }

    return true;
}

void eshkol_stack_pop(void) {
    if (t_stack_depth > 0) {
        t_stack_depth--;
    }
}

size_t eshkol_get_stack_depth(void) {
    return t_stack_depth;
}

// ----------------------------------------------------------------------------
// Timeout Watchdog
// ----------------------------------------------------------------------------

void eshkol_start_timer(uint64_t timeout_ms) {
    g_timer_start = std::chrono::steady_clock::now();
    g_timer_timeout_ms.store(timeout_ms > 0 ? timeout_ms : g_limits.max_execution_time_ms,
                              std::memory_order_release);
    g_timer_active.store(true, std::memory_order_release);
    eshkol_debug("Execution timer started: %llu ms",
                 (unsigned long long)g_timer_timeout_ms.load());
}

void eshkol_stop_timer(void) {
    g_timer_active.store(false, std::memory_order_release);
    eshkol_debug("Execution timer stopped");
}

bool eshkol_is_timed_out(void) {
    if (!g_timer_active.load(std::memory_order_acquire)) {
        return false;
    }

    uint64_t timeout = g_timer_timeout_ms.load(std::memory_order_acquire);
    if (timeout == 0) {
        return false;  // No timeout set
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_timer_start).count();

    if (static_cast<uint64_t>(elapsed) >= timeout) {
        g_last_error.store(ESHKOL_LIMIT_TIMEOUT, std::memory_order_relaxed);

        if (g_limits.enforce_hard_limits) {
            eshkol_error("Execution timeout: %lld ms >= %llu ms limit",
                         (long long)elapsed, (unsigned long long)timeout);
            eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_TIMEOUT);
        }
        return true;
    }

    return false;
}

uint64_t eshkol_get_remaining_time_ms(void) {
    if (!g_timer_active.load(std::memory_order_acquire)) {
        return 0;
    }

    uint64_t timeout = g_timer_timeout_ms.load(std::memory_order_acquire);
    if (timeout == 0) {
        return UINT64_MAX;  // No timeout = unlimited
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_timer_start).count();

    if (static_cast<uint64_t>(elapsed) >= timeout) {
        return 0;
    }

    return timeout - static_cast<uint64_t>(elapsed);
}

// ----------------------------------------------------------------------------
// Validation Functions
// ----------------------------------------------------------------------------

bool eshkol_check_tensor_size(size_t num_elements) {
    if (num_elements <= g_limits.max_tensor_elements) {
        return true;
    }

    g_last_error.store(ESHKOL_LIMIT_TENSOR_SIZE, std::memory_order_relaxed);

    if (g_limits.enforce_hard_limits) {
        eshkol_error("Tensor size limit exceeded: %zu > %zu elements",
                     num_elements, g_limits.max_tensor_elements);
    }

    return false;
}

bool eshkol_check_string_length(size_t length) {
    if (length <= g_limits.max_string_length) {
        return true;
    }

    g_last_error.store(ESHKOL_LIMIT_STRING_LENGTH, std::memory_order_relaxed);

    if (g_limits.enforce_hard_limits) {
        eshkol_error("String length limit exceeded: %zu > %zu bytes",
                     length, g_limits.max_string_length);
    }

    return false;
}

// ----------------------------------------------------------------------------
// Error Reporting
// ----------------------------------------------------------------------------

eshkol_limit_error_t eshkol_get_last_limit_error(void) {
    return g_last_error.load(std::memory_order_relaxed);
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
    eshkol_info("=== Resource Usage Statistics ===");
    eshkol_info("Heap: current=%zuMB, peak=%zuMB, limit=%zuMB",
                g_heap_usage.load() / (1024 * 1024),
                g_peak_heap_usage.load() / (1024 * 1024),
                g_limits.max_heap_bytes / (1024 * 1024));
    eshkol_info("Stack: current=%zu, limit=%zu", t_stack_depth, g_limits.max_stack_depth);

    if (g_timer_active.load()) {
        uint64_t remaining = eshkol_get_remaining_time_ms();
        eshkol_info("Timer: %llums remaining of %llums",
                    (unsigned long long)remaining,
                    (unsigned long long)g_timer_timeout_ms.load());
    } else {
        eshkol_info("Timer: inactive");
    }

    eshkol_limit_error_t last_error = g_last_error.load();
    if (last_error != ESHKOL_LIMIT_OK) {
        eshkol_info("Last limit error: %s", eshkol_limit_error_message(last_error));
    }
}

void eshkol_reset_resource_tracking(void) {
    g_heap_usage.store(0, std::memory_order_relaxed);
    g_peak_heap_usage.store(0, std::memory_order_relaxed);
    g_soft_limit_warned.store(false, std::memory_order_relaxed);
    t_stack_depth = 0;
    g_timer_active.store(false, std::memory_order_relaxed);
    g_last_error.store(ESHKOL_LIMIT_OK, std::memory_order_relaxed);
    eshkol_debug("Resource tracking reset");
}

} // extern "C"
