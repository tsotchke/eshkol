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
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <thread>

namespace {

// ============================================================================
// Global State
// ============================================================================

/** Build an @ref eshkol_resource_limits_t populated with the compiled-in
 *  ESHKOL_DEFAULT_* constants (heap, timeout, stack, tensor, string caps),
 *  with hard-limit enforcement and warnings enabled. */
eshkol_resource_limits_t make_default_limits() {
    eshkol_resource_limits_t limits{};
    limits.max_heap_bytes = ESHKOL_DEFAULT_MAX_HEAP_BYTES;
    limits.heap_soft_limit_bytes =
        (ESHKOL_DEFAULT_MAX_HEAP_BYTES * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100;
    limits.max_execution_time_ms = ESHKOL_DEFAULT_TIMEOUT_MS;
    limits.max_stack_depth = ESHKOL_DEFAULT_MAX_STACK_DEPTH;
    limits.max_tensor_elements = ESHKOL_DEFAULT_MAX_TENSOR_ELEMENTS;
    limits.max_string_length = ESHKOL_DEFAULT_MAX_STRING_LENGTH;
    limits.enforce_hard_limits = true;
    limits.enable_warnings = true;
    return limits;
}

// Active resource limits
eshkol_resource_limits_t g_limits = make_default_limits();
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
std::atomic<uint64_t> g_timer_generation{0};

// Last error
std::atomic<eshkol_limit_error_t> g_last_error{ESHKOL_LIMIT_OK};

// ============================================================================
// Helper Functions
// ============================================================================

/** Advance @p cursor past any leading whitespace characters.
 *  @return Pointer to the first non-space character, or to the terminating
 *          NUL; null if @p cursor is null. */
const char* skip_space(const char* cursor) {
    while (cursor && *cursor &&
           std::isspace(static_cast<unsigned char>(*cursor))) {
        ++cursor;
    }
    return cursor;
}

// Parse size with optional K/M/G suffix. Invalid values keep the caller's
// fallback so malformed hosted env vars cannot silently disable limits.
size_t parse_size_or_default(const char* str, size_t fallback) {
    if (!str) return fallback;

    const char* start = skip_space(str);
    if (!*start) return fallback;

    errno = 0;
    char* end = nullptr;
    double value = strtod(start, &end);
    if (end == start || errno == ERANGE || !std::isfinite(value) || value < 0.0) {
        return fallback;
    }

    end = const_cast<char*>(skip_space(end));

    double multiplier = 1.0;
    if (*end) {
        switch (*end) {
            case 'K': case 'k':
                multiplier = 1024.0;
                break;
            case 'M': case 'm':
                multiplier = 1024.0 * 1024.0;
                break;
            case 'G': case 'g':
                multiplier = 1024.0 * 1024.0 * 1024.0;
                break;
            default:
                return fallback;
        }
        ++end;
        if (*end == 'B' || *end == 'b') {
            ++end;
        }
        end = const_cast<char*>(skip_space(end));
    }

    if (*end) {
        return fallback;
    }

    const double bytes = value * multiplier;
    if (!std::isfinite(bytes) ||
        bytes > static_cast<double>(std::numeric_limits<size_t>::max())) {
        return fallback;
    }

    return static_cast<size_t>(bytes);
}

/** Parse @p str as an unsigned 64-bit decimal integer.
 *  @param str Candidate string (may be null).
 *  @param fallback Value returned if @p str is null, empty, negative,
 *         out of range, or has trailing non-space characters.
 *  @return Parsed value, or @p fallback on any parse failure. */
uint64_t parse_u64_or_default(const char* str, uint64_t fallback) {
    if (!str) return fallback;

    const char* start = skip_space(str);
    if (!*start || *start == '-') return fallback;

    errno = 0;
    char* end = nullptr;
    unsigned long long value = strtoull(start, &end, 10);
    if (end == start || errno == ERANGE) {
        return fallback;
    }

    end = const_cast<char*>(skip_space(end));
    if (*end) {
        return fallback;
    }

    return static_cast<uint64_t>(value);
}

/** Parse @p str as a boolean flag, accepting "true"/"TRUE"/"1"/"yes"/"YES"
 *  and "false"/"FALSE"/"0"/"no"/"NO" (case as shown); any other value,
 *  including null, yields @p fallback. */
bool parse_bool_or_default(const char* str, bool fallback) {
    if (!str) return fallback;
    if (strcmp(str, "true") == 0 ||
        strcmp(str, "TRUE") == 0 ||
        strcmp(str, "1") == 0 ||
        strcmp(str, "yes") == 0 ||
        strcmp(str, "YES") == 0) {
        return true;
    }
    if (strcmp(str, "false") == 0 ||
        strcmp(str, "FALSE") == 0 ||
        strcmp(str, "0") == 0 ||
        strcmp(str, "no") == 0 ||
        strcmp(str, "NO") == 0) {
        return false;
    }
    return fallback;
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

/** Return the compiled-in default resource limits. */
eshkol_resource_limits_t eshkol_get_default_limits(void) {
    return make_default_limits();
}

/** @brief Build resource limits from defaults, overridden by environment
 *  variables where present.
 *
 * Reads ESHKOL_MAX_HEAP, ESHKOL_TIMEOUT_MS, ESHKOL_MAX_STACK,
 * ESHKOL_MAX_TENSOR_ELEMS, ESHKOL_MAX_STRING_LEN, ESHKOL_ENFORCE_LIMITS,
 * and ESHKOL_LIMIT_WARNINGS, applies the parsed values on top of
 * eshkol_get_default_limits(), installs the result as the active limits
 * via eshkol_set_limits(), and returns it.
 * @return The resulting active resource limits. */
eshkol_resource_limits_t eshkol_init_limits_from_env(void) {
    eshkol_resource_limits_t limits = eshkol_get_default_limits();

    // ESHKOL_MAX_HEAP
    const char* max_heap = std::getenv("ESHKOL_MAX_HEAP");
    if (max_heap) {
        limits.max_heap_bytes = parse_size_or_default(max_heap, limits.max_heap_bytes);
        limits.heap_soft_limit_bytes = (limits.max_heap_bytes * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100;
        eshkol_debug("Max heap from env: %zu bytes", limits.max_heap_bytes);
    }

    // ESHKOL_TIMEOUT_MS
    const char* timeout = std::getenv("ESHKOL_TIMEOUT_MS");
    if (timeout) {
        limits.max_execution_time_ms = parse_u64_or_default(timeout, limits.max_execution_time_ms);
        eshkol_debug("Timeout from env: %llu ms", (unsigned long long)limits.max_execution_time_ms);
    }

    // ESHKOL_MAX_STACK
    const char* max_stack = std::getenv("ESHKOL_MAX_STACK");
    if (max_stack) {
        limits.max_stack_depth = parse_size_or_default(max_stack, limits.max_stack_depth);
        eshkol_debug("Max stack from env: %zu", limits.max_stack_depth);
    }

    // ESHKOL_MAX_TENSOR_ELEMS
    const char* max_tensor = std::getenv("ESHKOL_MAX_TENSOR_ELEMS");
    if (max_tensor) {
        limits.max_tensor_elements = parse_size_or_default(max_tensor, limits.max_tensor_elements);
        eshkol_debug("Max tensor elements from env: %zu", limits.max_tensor_elements);
    }

    // ESHKOL_MAX_STRING_LEN
    const char* max_string = std::getenv("ESHKOL_MAX_STRING_LEN");
    if (max_string) {
        limits.max_string_length = parse_size_or_default(max_string, limits.max_string_length);
        eshkol_debug("Max string length from env: %zu", limits.max_string_length);
    }

    // ESHKOL_ENFORCE_LIMITS
    const char* enforce = std::getenv("ESHKOL_ENFORCE_LIMITS");
    if (enforce) {
        limits.enforce_hard_limits = parse_bool_or_default(enforce, limits.enforce_hard_limits);
        eshkol_debug("Enforce limits from env: %s", limits.enforce_hard_limits ? "true" : "false");
    }

    // ESHKOL_LIMIT_WARNINGS
    const char* warnings = std::getenv("ESHKOL_LIMIT_WARNINGS");
    if (warnings) {
        limits.enable_warnings = parse_bool_or_default(warnings, limits.enable_warnings);
        eshkol_debug("Limit warnings from env: %s", limits.enable_warnings ? "true" : "false");
    }

    eshkol_set_limits(&limits);
    return limits;
}

/** @brief Install @p limits as the active global resource limits.
 *
 * Ignored if @p limits is null. Derives heap_soft_limit_bytes from
 * max_heap_bytes when the caller left it at zero, resets the
 * soft-limit-warned flag, and logs the new configuration.
 * @param limits New limits to copy into the global state. */
void eshkol_set_limits(const eshkol_resource_limits_t* limits) {
    if (!limits) return;

    std::lock_guard<std::mutex> lock(g_limits_mutex);
    g_limits = *limits;
    if (g_limits.max_heap_bytes > 0 && g_limits.heap_soft_limit_bytes == 0) {
        g_limits.heap_soft_limit_bytes =
            (g_limits.max_heap_bytes * ESHKOL_HEAP_SOFT_LIMIT_PERCENT) / 100;
    }
    g_soft_limit_warned.store(false, std::memory_order_relaxed);

    eshkol_info("Resource limits configured: heap=%zuMB, timeout=%llums, stack=%zu",
                g_limits.max_heap_bytes / (1024 * 1024),
                (unsigned long long)g_limits.max_execution_time_ms,
                g_limits.max_stack_depth);
}

/** Return a pointer to the currently active resource limits. */
const eshkol_resource_limits_t* eshkol_get_limits(void) {
    return &g_limits;
}

// ----------------------------------------------------------------------------
// Memory Tracking
// ----------------------------------------------------------------------------

/** @brief Record a heap allocation of @p bytes against the tracked usage.
 *
 * Atomically adds @p bytes to the global heap usage counter, guarding
 * against overflow and against exceeding max_heap_bytes. On overflow or
 * hard-limit breach, records ESHKOL_LIMIT_HEAP_HARD and, if hard limits
 * are enforced, requests a runtime interrupt (ESHKOL_SHUTDOWN_MEMORY).
 * Also updates peak usage and, if usage crosses the soft-limit threshold
 * for the first time, logs a one-shot warning.
 * @param bytes Number of bytes being allocated (0 is a no-op success).
 * @return true if the allocation is within limits (or limits aren't
 *         enforced conceptually for tracking purposes), false if it would
 *         exceed the configured heap limit or overflow the counter. */
bool eshkol_track_allocation(size_t bytes) {
    if (bytes == 0) return true;

    size_t previous = g_heap_usage.load(std::memory_order_relaxed);
    size_t current = 0;

    for (;;) {
        if (bytes > SIZE_MAX - previous) {
            g_last_error.store(ESHKOL_LIMIT_HEAP_HARD, std::memory_order_relaxed);
            if (g_limits.enforce_hard_limits) {
                eshkol_error("Heap accounting overflow: current=%zu, requested=%zu",
                             previous, bytes);
                eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_MEMORY);
            }
            return false;
        }

        current = previous + bytes;
        if (g_limits.max_heap_bytes > 0 && current > g_limits.max_heap_bytes) {
            g_last_error.store(ESHKOL_LIMIT_HEAP_HARD, std::memory_order_relaxed);
            if (g_limits.enforce_hard_limits) {
                eshkol_error("Heap limit exceeded: %zuMB > %zuMB",
                             current / (1024 * 1024),
                             g_limits.max_heap_bytes / (1024 * 1024));
                eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_MEMORY);
            }
            return false;
        }

        if (g_heap_usage.compare_exchange_weak(previous, current,
                                               std::memory_order_relaxed)) {
            break;
        }
    }

    update_peak(current);

    // Check soft limit
    if (g_limits.enable_warnings &&
        g_limits.max_heap_bytes > 0 &&
        current >= g_limits.heap_soft_limit_bytes &&
        !g_soft_limit_warned.exchange(true, std::memory_order_relaxed)) {
        eshkol_warn("Heap usage at %zu%% of limit (%zuMB / %zuMB)",
                    (current * 100) / g_limits.max_heap_bytes,
                    current / (1024 * 1024),
                    g_limits.max_heap_bytes / (1024 * 1024));
        g_last_error.store(ESHKOL_LIMIT_HEAP_SOFT, std::memory_order_relaxed);
    }

    return true;
}

/** @brief Record a heap deallocation of @p bytes, decrementing tracked usage.
 *
 * Clamps at zero rather than underflowing if @p bytes exceeds the
 * currently tracked usage. @param bytes Number of bytes being freed. */
void eshkol_track_deallocation(size_t bytes) {
    if (bytes == 0) return;

    size_t current = g_heap_usage.load(std::memory_order_relaxed);
    for (;;) {
        size_t next = (bytes >= current) ? 0 : (current - bytes);
        if (g_heap_usage.compare_exchange_weak(current, next,
                                               std::memory_order_relaxed)) {
            return;
        }
    }
}

/** Return the current tracked heap usage in bytes. */
size_t eshkol_get_heap_usage(void) {
    return g_heap_usage.load(std::memory_order_relaxed);
}

/** Return the highest tracked heap usage observed in bytes. */
size_t eshkol_get_peak_heap_usage(void) {
    return g_peak_heap_usage.load(std::memory_order_relaxed);
}

/** @brief Check whether heap usage is at or above 90% of max_heap_bytes.
 *  @return false if no heap limit is configured (max_heap_bytes == 0),
 *          otherwise whether current usage has crossed the 90% threshold. */
bool eshkol_is_near_memory_limit(void) {
    size_t current = g_heap_usage.load(std::memory_order_relaxed);
    if (g_limits.max_heap_bytes == 0) {
        return false;
    }
    size_t threshold = (g_limits.max_heap_bytes * 90) / 100;  // 90% of limit
    return current >= threshold;
}

// ----------------------------------------------------------------------------
// Stack Tracking
// ----------------------------------------------------------------------------

/** @brief Increment the thread-local recursion-depth counter and check it
 *  against max_stack_depth.
 *
 * On exceeding the limit, records ESHKOL_LIMIT_STACK_OVERFLOW, logs an
 * error if hard limits are enforced, and rolls back the increment.
 * @return true if the push is within the stack depth limit, false if it
 *         would exceed it (in which case the depth is not incremented). */
bool eshkol_stack_push(void) {
    t_stack_depth++;

    if (t_stack_depth > g_limits.max_stack_depth) {
        g_last_error.store(ESHKOL_LIMIT_STACK_OVERFLOW, std::memory_order_relaxed);

        if (g_limits.enforce_hard_limits) {
            eshkol_error("Stack overflow: depth %zu > limit %zu",
                         t_stack_depth, g_limits.max_stack_depth);
        }
        t_stack_depth--;
        return false;
    }

    return true;
}

/** Decrement the thread-local recursion-depth counter, if non-zero. */
void eshkol_stack_pop(void) {
    if (t_stack_depth > 0) {
        t_stack_depth--;
    }
}

/** Return the current thread's tracked recursion depth. */
size_t eshkol_get_stack_depth(void) {
    return t_stack_depth;
}

// ----------------------------------------------------------------------------
// Timeout Watchdog
// ----------------------------------------------------------------------------

/** @brief Start (or restart) the execution timeout watchdog.
 *
 * Bumps the timer generation, records the start time and effective
 * timeout (using @p timeout_ms if non-zero, else max_execution_time_ms),
 * and marks the timer active. If hard-limit enforcement is on and the
 * effective timeout is non-zero, spawns a detached watchdog thread that
 * sleeps for the timeout and then, if the timer generation is unchanged
 * and still active, records ESHKOL_LIMIT_TIMEOUT and requests a runtime
 * interrupt (ESHKOL_SHUTDOWN_TIMEOUT). The generation counter lets a
 * subsequent eshkol_start_timer()/eshkol_stop_timer() call invalidate a
 * stale watchdog thread.
 * @param timeout_ms Timeout in milliseconds, or 0 to use the configured
 *        default (max_execution_time_ms). */
void eshkol_start_timer(uint64_t timeout_ms) {
    const uint64_t generation =
        g_timer_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    const uint64_t effective_timeout =
        timeout_ms > 0 ? timeout_ms : g_limits.max_execution_time_ms;

    g_timer_start = std::chrono::steady_clock::now();
    g_timer_timeout_ms.store(effective_timeout, std::memory_order_release);
    g_timer_active.store(true, std::memory_order_release);
    eshkol_debug("Execution timer started: %llu ms",
                 (unsigned long long)g_timer_timeout_ms.load());

    if (effective_timeout > 0 && g_limits.enforce_hard_limits) {
        std::thread([generation, effective_timeout]() {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(effective_timeout));

            if (g_timer_generation.load(std::memory_order_acquire) != generation ||
                !g_timer_active.load(std::memory_order_acquire)) {
                return;
            }

            g_last_error.store(ESHKOL_LIMIT_TIMEOUT, std::memory_order_relaxed);
            g_timer_active.store(false, std::memory_order_release);
            eshkol_error("Execution timeout: %llums limit exceeded",
                         (unsigned long long)effective_timeout);
            eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_TIMEOUT);
        }).detach();
    }
}

/** Deactivate the execution timeout watchdog and bump its generation so
 *  any in-flight watchdog thread becomes a no-op when it wakes. */
void eshkol_stop_timer(void) {
    g_timer_active.store(false, std::memory_order_release);
    g_timer_generation.fetch_add(1, std::memory_order_acq_rel);
    eshkol_debug("Execution timer stopped");
}

/** @brief Poll whether the active execution timer has expired.
 *
 * Returns false if no timer is active or no timeout is configured.
 * Otherwise compares elapsed time since eshkol_start_timer() against the
 * configured timeout; on expiry records ESHKOL_LIMIT_TIMEOUT and, if hard
 * limits are enforced, deactivates the timer and requests a runtime
 * interrupt (ESHKOL_SHUTDOWN_TIMEOUT).
 * @return true if the timeout has been reached. */
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
            g_timer_active.store(false, std::memory_order_release);
            eshkol_runtime_request_interrupt(ESHKOL_SHUTDOWN_TIMEOUT);
        }
        return true;
    }

    return false;
}

/** @brief Compute time remaining before the active timer expires.
 *  @return 0 if the timer is inactive or has already expired; UINT64_MAX
 *          if the timer is active but has no configured timeout;
 *          otherwise the remaining milliseconds. */
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

/** @brief Validate a tensor's element count against max_tensor_elements.
 *
 * Records ESHKOL_LIMIT_TENSOR_SIZE and logs an error (if hard limits are
 * enforced) when @p num_elements exceeds the configured limit.
 * @return true if within limits, false otherwise. */
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

/** @brief Validate a string's byte length against max_string_length.
 *
 * Records ESHKOL_LIMIT_STRING_LENGTH and logs an error (if hard limits are
 * enforced) when @p length exceeds the configured limit.
 * @return true if within limits, false otherwise. */
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

/** Return the most recently recorded resource-limit error code. */
eshkol_limit_error_t eshkol_get_last_limit_error(void) {
    return g_last_error.load(std::memory_order_relaxed);
}

/** Return a human-readable description for a resource-limit error code. */
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

/** Log a snapshot of current resource usage (heap, stack, timer, and the
 *  last recorded limit error) via the eshkol_info logger. */
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

/** Reset all resource tracking state to its initial values: heap usage,
 *  peak usage, soft-limit-warned flag, stack depth, timer, and last
 *  error. Does not change the configured limits themselves. */
void eshkol_reset_resource_tracking(void) {
    g_heap_usage.store(0, std::memory_order_relaxed);
    g_peak_heap_usage.store(0, std::memory_order_relaxed);
    g_soft_limit_warned.store(false, std::memory_order_relaxed);
    t_stack_depth = 0;
    g_timer_active.store(false, std::memory_order_relaxed);
    g_timer_generation.fetch_add(1, std::memory_order_acq_rel);
    g_last_error.store(ESHKOL_LIMIT_OK, std::memory_order_relaxed);
    eshkol_debug("Resource tracking reset");
}

} // extern "C"
