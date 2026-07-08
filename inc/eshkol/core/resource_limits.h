/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Resource Limits
 *
 * Provides:
 * - Heap memory limits with soft/hard thresholds
 * - Execution timeout with watchdog
 * - Stack depth tracking to prevent overflow
 * - Tensor element count limits
 * - String length limits
 * - Configurable via environment variables or programmatically
 */
#ifndef ESHKOL_CORE_RESOURCE_LIMITS_H
#define ESHKOL_CORE_RESOURCE_LIMITS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Default Limits
// ============================================================================

/**
 * @brief Compile-time default resource limits.
 *
 * These seed eshkol_get_default_limits() / make_default_limits() and
 * are used unless overridden by eshkol_set_limits() or the
 * corresponding `ESHKOL_*` environment variables read by
 * eshkol_init_limits_from_env().
 */

// Default: 1GB heap
#define ESHKOL_DEFAULT_MAX_HEAP_BYTES      (1ULL * 1024 * 1024 * 1024)

// Soft limit triggers warning at 80% of max_heap_bytes (see
// eshkol_resource_limits_t::heap_soft_limit_bytes)
#define ESHKOL_HEAP_SOFT_LIMIT_PERCENT     80

// Default: 30 second timeout
#define ESHKOL_DEFAULT_TIMEOUT_MS          30000

// Default: 100,000 stack frames (512MB OS stack supports ~80K+ frames)
#define ESHKOL_DEFAULT_MAX_STACK_DEPTH     100000

// Default: 1 billion tensor elements
#define ESHKOL_DEFAULT_MAX_TENSOR_ELEMENTS (1ULL * 1000 * 1000 * 1000)

// Default: 100MB string
#define ESHKOL_DEFAULT_MAX_STRING_LENGTH   (100ULL * 1024 * 1024)

// ============================================================================
// Resource Limit Configuration
// ============================================================================

/**
 * @brief Active resource-limit configuration for the process.
 *
 * Obtain a default-populated instance with eshkol_get_default_limits()
 * or eshkol_init_limits_from_env(), then activate it with
 * eshkol_set_limits(). The currently active configuration is readable
 * via eshkol_get_limits(). When `heap_soft_limit_bytes` is left 0 on a
 * call to eshkol_set_limits(), it is auto-derived as
 * `max_heap_bytes * ESHKOL_HEAP_SOFT_LIMIT_PERCENT / 100`.
 */
typedef struct eshkol_resource_limits {
    // Memory limits
    size_t max_heap_bytes;           // Maximum heap allocation
    size_t heap_soft_limit_bytes;    // Soft limit (triggers warning)

    // Time limits
    uint64_t max_execution_time_ms;  // Maximum execution time (0 = unlimited)

    // Stack limits
    size_t max_stack_depth;          // Maximum recursion depth

    // Data structure limits
    size_t max_tensor_elements;      // Maximum elements in a tensor
    size_t max_string_length;        // Maximum string length

    // Behavior flags
    bool enforce_hard_limits;        // Kill on hard limit (vs return error)
    bool enable_warnings;            // Log soft limit warnings
} eshkol_resource_limits_t;

// ============================================================================
// Environment Variables
// ============================================================================

// Numeric env vars are parsed strictly; malformed or negative values keep the
// default for that field.
// ESHKOL_MAX_HEAP          - Max heap in bytes (supports K/M/G/KB/MB/GB suffix)
// ESHKOL_TIMEOUT_MS        - Execution timeout in milliseconds
// ESHKOL_MAX_STACK         - Max stack depth
// ESHKOL_MAX_TENSOR_ELEMS  - Max tensor elements
// ESHKOL_MAX_STRING_LEN    - Max string length
// ESHKOL_ENFORCE_LIMITS    - "true" or "false"
// ESHKOL_LIMIT_WARNINGS    - "true" or "false"

// ============================================================================
// Initialization
// ============================================================================

/**
 * @brief Get the compile-time default resource limits.
 *
 * Does not read the environment or mutate the active configuration;
 * pure construction from the `ESHKOL_DEFAULT_*` macros.
 *
 * @return A fully populated eshkol_resource_limits_t with default values.
 */
eshkol_resource_limits_t eshkol_get_default_limits(void);

/**
 * @brief Build limits from defaults overridden by environment variables,
 * and activate them.
 *
 * Starts from eshkol_get_default_limits(), then overrides each field
 * whose corresponding `ESHKOL_*` environment variable (see the
 * "Environment Variables" section above) is set and parses validly;
 * malformed or negative values silently keep the default for that
 * field. The resulting configuration is passed to eshkol_set_limits()
 * before returning, so it also becomes the active limits.
 *
 * @return The resolved, now-active resource limits.
 */
eshkol_resource_limits_t eshkol_init_limits_from_env(void);

/**
 * @brief Set the active resource limits (thread-safe).
 *
 * Copies `*limits` into the process-wide active configuration. If
 * `heap_soft_limit_bytes` is 0 while `max_heap_bytes` is non-zero, it is
 * auto-derived from ESHKOL_HEAP_SOFT_LIMIT_PERCENT. Resets the
 * soft-limit warning latch so a new configuration can warn again. A
 * no-op if `limits` is NULL.
 *
 * @param limits Configuration to activate; not retained by pointer (copied).
 */
void eshkol_set_limits(const eshkol_resource_limits_t* limits);

/**
 * @brief Get a pointer to the current active resource limits.
 *
 * @return Pointer to the process-wide active configuration. Valid for
 *         the life of the process; contents may change on a subsequent
 *         eshkol_set_limits() call from another thread.
 */
const eshkol_resource_limits_t* eshkol_get_limits(void);

// ============================================================================
// Memory Tracking
// ============================================================================

/**
 * @brief Register a heap allocation of `bytes` against the tracked usage
 * total (call from the arena allocator before/as it allocates).
 *
 * Thread-safe (lock-free CAS loop). If the new total would exceed
 * `max_heap_bytes` (when non-zero) or overflow accounting, records
 * ESHKOL_LIMIT_HEAP_HARD as the last limit error; if
 * `enforce_hard_limits` is set this additionally logs an error and
 * requests a runtime interrupt with ESHKOL_SHUTDOWN_MEMORY, and the call
 * returns false. Also updates peak usage and, once usage crosses the
 * soft-limit threshold, logs a one-time warning (if `enable_warnings`
 * is set) and records ESHKOL_LIMIT_HEAP_SOFT.
 *
 * @param bytes Number of bytes being allocated (0 is always allowed).
 * @return true if the allocation is allowed to proceed, false if it
 *         would exceed the hard limit.
 */
bool eshkol_track_allocation(size_t bytes);

/**
 * @brief Register a heap deallocation of `bytes` against the tracked
 * usage total.
 *
 * Thread-safe. Usage is clamped to 0 rather than underflowing if
 * `bytes` exceeds the currently tracked total.
 *
 * @param bytes Number of bytes being freed (0 is a no-op).
 */
void eshkol_track_deallocation(size_t bytes);

/**
 * @brief Get the current tracked heap usage.
 *
 * @return Current heap usage in bytes, as tracked by
 *         eshkol_track_allocation() / eshkol_track_deallocation().
 */
size_t eshkol_get_heap_usage(void);

/**
 * @brief Get the peak tracked heap usage observed so far.
 *
 * @return High-water mark of heap usage in bytes since the last
 *         eshkol_reset_resource_tracking() call.
 */
size_t eshkol_get_peak_heap_usage(void);

/**
 * @brief Check whether current heap usage is near the configured hard limit.
 *
 * @return true if current usage is at or above 90% of `max_heap_bytes`
 *         (when a non-zero max is configured), false otherwise
 *         (including when no heap limit is configured).
 */
bool eshkol_is_near_memory_limit(void);

// ============================================================================
// Stack Tracking
// ============================================================================

/**
 * @brief Enter a new tracked stack frame (call at function entry).
 *
 * Increments a thread-local depth counter. If the new depth exceeds
 * `max_stack_depth`, records ESHKOL_LIMIT_STACK_OVERFLOW as the last
 * limit error, decrements the counter back down (the disallowed frame
 * is not counted), and — if `enforce_hard_limits` is set — logs an
 * error. Pair every successful or unsuccessful call with a matching
 * eshkol_stack_pop() only when it returned true.
 *
 * @return true if the frame is allowed, false on stack overflow.
 */
bool eshkol_stack_push(void);

/**
 * @brief Exit a tracked stack frame (call at function exit).
 *
 * Decrements the thread-local depth counter; a no-op if already at 0.
 */
void eshkol_stack_pop(void);

/**
 * @brief Get the current thread's tracked stack depth.
 *
 * @return Number of currently active frames registered via
 *         eshkol_stack_push()/eshkol_stack_pop() on this thread.
 */
size_t eshkol_get_stack_depth(void);

// ============================================================================
// Timeout Watchdog
// ============================================================================

/**
 * @brief Start (or restart) the execution timer.
 *
 * Records the current time as the timer's start point and, if
 * `enforce_hard_limits` is set and the effective timeout is non-zero,
 * detaches a background thread that sleeps for the timeout duration
 * and then — unless the timer was stopped or restarted in the
 * meantime (tracked via a generation counter) — records
 * ESHKOL_LIMIT_TIMEOUT and requests a runtime interrupt with
 * ESHKOL_SHUTDOWN_TIMEOUT. eshkol_is_timed_out() additionally performs
 * its own polling-based check, so timeout enforcement does not rely on
 * the watchdog thread alone.
 *
 * @param timeout_ms Timeout override in milliseconds; 0 uses the
 *        currently configured `max_execution_time_ms` limit instead.
 */
void eshkol_start_timer(uint64_t timeout_ms);

/**
 * @brief Stop the execution timer.
 *
 * Marks the timer inactive and advances its generation counter so any
 * in-flight watchdog thread from eshkol_start_timer() observes the
 * change and exits without firing a timeout.
 */
void eshkol_stop_timer(void);

/**
 * @brief Poll whether the execution timeout has been exceeded.
 *
 * Should be called periodically in long-running operations as a
 * cooperative check (in addition to the background watchdog thread
 * started by eshkol_start_timer()). If elapsed time meets or exceeds
 * the configured timeout, records ESHKOL_LIMIT_TIMEOUT and — if
 * `enforce_hard_limits` is set — stops the timer and requests a runtime
 * interrupt with ESHKOL_SHUTDOWN_TIMEOUT.
 *
 * @return true if the timer is active and has timed out, false otherwise
 *         (including when no timer is active or no timeout is configured).
 */
bool eshkol_is_timed_out(void);

/**
 * @brief Get the remaining time before the execution timeout fires.
 *
 * @return Milliseconds remaining, 0 if the timer is inactive or has
 *         already timed out, or UINT64_MAX if the timer is active but
 *         no timeout (0) is configured (unlimited).
 */
uint64_t eshkol_get_remaining_time_ms(void);

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * @brief Check whether a tensor of `num_elements` is within the configured limit.
 *
 * On failure, records ESHKOL_LIMIT_TENSOR_SIZE as the last limit error
 * and, if `enforce_hard_limits` is set, logs an error.
 *
 * @param num_elements Element count of the tensor being validated.
 * @return true if allowed, false if it exceeds `max_tensor_elements`.
 */
bool eshkol_check_tensor_size(size_t num_elements);

/**
 * @brief Check whether a string of `length` bytes is within the configured limit.
 *
 * On failure, records ESHKOL_LIMIT_STRING_LENGTH as the last limit error
 * and, if `enforce_hard_limits` is set, logs an error.
 *
 * @param length Length in bytes of the string being validated.
 * @return true if allowed, false if it exceeds `max_string_length`.
 */
bool eshkol_check_string_length(size_t length);

// ============================================================================
// Error Reporting
// ============================================================================

/**
 * @brief Which resource-limit condition was most recently observed.
 *
 * Set by the tracking/validation functions above (eshkol_track_allocation(),
 * eshkol_stack_push(), eshkol_is_timed_out(), eshkol_check_tensor_size(),
 * eshkol_check_string_length()) and readable via
 * eshkol_get_last_limit_error(). This is a process-wide "last error"
 * value, not per-call-site — read it immediately after the call whose
 * outcome you need.
 */
typedef enum {
    ESHKOL_LIMIT_OK = 0,          // No limit condition recorded
    ESHKOL_LIMIT_HEAP_SOFT,       // Soft heap limit reached (warning)
    ESHKOL_LIMIT_HEAP_HARD,       // Hard heap limit exceeded
    ESHKOL_LIMIT_TIMEOUT,         // Execution timeout
    ESHKOL_LIMIT_STACK_OVERFLOW,  // Stack depth exceeded
    ESHKOL_LIMIT_TENSOR_SIZE,     // Tensor too large
    ESHKOL_LIMIT_STRING_LENGTH    // String too long
} eshkol_limit_error_t;

/**
 * @brief Get the last recorded resource-limit error.
 *
 * @return The most recently recorded eshkol_limit_error_t value
 *         (ESHKOL_LIMIT_OK if none has been recorded, or after
 *         eshkol_reset_resource_tracking()).
 */
eshkol_limit_error_t eshkol_get_last_limit_error(void);

/**
 * @brief Get a human-readable message for a limit error code.
 *
 * @param error Error code to describe.
 * @return Static, never-NULL string describing `error`; an unrecognized
 *         value returns "Unknown limit error".
 */
const char* eshkol_limit_error_message(eshkol_limit_error_t error);

// ============================================================================
// Diagnostics
// ============================================================================

/**
 * @brief Log a snapshot of current resource usage.
 *
 * Writes heap usage (current/peak/limit), stack depth, timer status
 * (remaining time if active), and the last recorded limit error (if
 * any) via the runtime's info logger.
 */
void eshkol_print_resource_stats(void);

/**
 * @brief Reset all resource-tracking counters and state to their
 * initial values.
 *
 * Clears heap usage/peak, the soft-limit warning latch, the calling
 * thread's stack depth, deactivates the timer, and resets the last
 * limit error to ESHKOL_LIMIT_OK. Does not change the configured
 * limits themselves (see eshkol_set_limits()).
 */
void eshkol_reset_resource_tracking(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ RAII Helpers
// ============================================================================

#ifdef __cplusplus

namespace eshkol {

/**
 * @brief RAII guard for stack-depth tracking.
 *
 * Calls eshkol_stack_push() on construction and, only if that
 * succeeded, eshkol_stack_pop() on destruction. Convert to `bool` (or
 * call isValid()) immediately after construction to detect stack
 * overflow; see ESHKOL_STACK_GUARD() / ESHKOL_STACK_GUARD_WITH_VALUE()
 * for the common early-return idiom. Non-copyable.
 */
class StackFrameGuard {
public:
    /** @brief Register a new stack frame for the current thread. */
    StackFrameGuard() : valid_(eshkol_stack_push()) {}
    ~StackFrameGuard() { if (valid_) eshkol_stack_pop(); }

    /** @brief Whether the frame was allowed (false means stack overflow). */
    bool isValid() const { return valid_; }
    /** @brief Equivalent to isValid(). */
    operator bool() const { return valid_; }

    // Non-copyable
    StackFrameGuard(const StackFrameGuard&) = delete;
    StackFrameGuard& operator=(const StackFrameGuard&) = delete;

private:
    bool valid_;
};

/**
 * @brief RAII guard for the execution-timeout watchdog.
 *
 * Starts the timer (eshkol_start_timer()) on construction and stops it
 * (eshkol_stop_timer()) on destruction, scoping the timeout to the
 * guard's lifetime. Non-copyable.
 */
class TimerGuard {
public:
    /**
     * @brief Start the execution timer.
     * @param timeout_ms Timeout override in milliseconds; 0 uses the
     *        configured limit.
     */
    explicit TimerGuard(uint64_t timeout_ms = 0) {
        eshkol_start_timer(timeout_ms);
    }
    ~TimerGuard() {
        eshkol_stop_timer();
    }

    /** @brief Whether the guarded timer has timed out; see eshkol_is_timed_out(). */
    bool isTimedOut() const { return eshkol_is_timed_out(); }
    /** @brief Milliseconds remaining before timeout; see eshkol_get_remaining_time_ms(). */
    uint64_t remainingMs() const { return eshkol_get_remaining_time_ms(); }

    // Non-copyable
    TimerGuard(const TimerGuard&) = delete;
    TimerGuard& operator=(const TimerGuard&) = delete;
};

/**
 * @brief Declare a StackFrameGuard and return `void` early on stack overflow.
 *
 * Place at the top of a `void`-returning function body to get automatic
 * stack-depth tracking and graceful bail-out without hand-writing the
 * eshkol_stack_push() check at every recursive entry point.
 */
#define ESHKOL_STACK_GUARD() \
    eshkol::StackFrameGuard _stack_guard; \
    if (!_stack_guard) { \
        return; \
    }

/**
 * @brief Declare a StackFrameGuard and return `val` early on stack overflow.
 *
 * Like ESHKOL_STACK_GUARD() but for functions that return a value.
 *
 * @param val Expression to return if the stack-depth limit is exceeded.
 */
#define ESHKOL_STACK_GUARD_WITH_VALUE(val) \
    eshkol::StackFrameGuard _stack_guard; \
    if (!_stack_guard) { \
        return (val); \
    }

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_RESOURCE_LIMITS_H
