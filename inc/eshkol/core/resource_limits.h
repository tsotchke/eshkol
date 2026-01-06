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

// Default: 1GB heap
#define ESHKOL_DEFAULT_MAX_HEAP_BYTES      (1ULL * 1024 * 1024 * 1024)

// Soft limit triggers warning at 80%
#define ESHKOL_HEAP_SOFT_LIMIT_PERCENT     80

// Default: 30 second timeout
#define ESHKOL_DEFAULT_TIMEOUT_MS          30000

// Default: 10,000 stack frames
#define ESHKOL_DEFAULT_MAX_STACK_DEPTH     10000

// Default: 1 billion tensor elements
#define ESHKOL_DEFAULT_MAX_TENSOR_ELEMENTS (1ULL * 1000 * 1000 * 1000)

// Default: 100MB string
#define ESHKOL_DEFAULT_MAX_STRING_LENGTH   (100ULL * 1024 * 1024)

// ============================================================================
// Resource Limit Configuration
// ============================================================================

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

// ESHKOL_MAX_HEAP          - Max heap in bytes (supports K/M/G suffix)
// ESHKOL_TIMEOUT_MS        - Execution timeout in milliseconds
// ESHKOL_MAX_STACK         - Max stack depth
// ESHKOL_MAX_TENSOR_ELEMS  - Max tensor elements
// ESHKOL_MAX_STRING_LEN    - Max string length
// ESHKOL_ENFORCE_LIMITS    - "true" or "false"
// ESHKOL_LIMIT_WARNINGS    - "true" or "false"

// ============================================================================
// Initialization
// ============================================================================

// Get default resource limits
eshkol_resource_limits_t eshkol_get_default_limits(void);

// Initialize limits from environment variables (overrides defaults)
// Returns configured limits
eshkol_resource_limits_t eshkol_init_limits_from_env(void);

// Set active resource limits
void eshkol_set_limits(const eshkol_resource_limits_t* limits);

// Get current active limits
const eshkol_resource_limits_t* eshkol_get_limits(void);

// ============================================================================
// Memory Tracking
// ============================================================================

// Register a heap allocation (call from arena allocator)
// Returns: true if allocation is allowed, false if limit exceeded
bool eshkol_track_allocation(size_t bytes);

// Register a heap deallocation
void eshkol_track_deallocation(size_t bytes);

// Get current heap usage
size_t eshkol_get_heap_usage(void);

// Get peak heap usage
size_t eshkol_get_peak_heap_usage(void);

// Check if near memory limit (within 10% of hard limit)
bool eshkol_is_near_memory_limit(void);

// ============================================================================
// Stack Tracking
// ============================================================================

// Enter a new stack frame (call at function entry)
// Returns: true if allowed, false if stack overflow
bool eshkol_stack_push(void);

// Exit a stack frame
void eshkol_stack_pop(void);

// Get current stack depth
size_t eshkol_get_stack_depth(void);

// ============================================================================
// Timeout Watchdog
// ============================================================================

// Start the execution timer
// - timeout_ms: Override timeout (0 = use configured limit)
void eshkol_start_timer(uint64_t timeout_ms);

// Stop the execution timer
void eshkol_stop_timer(void);

// Check if timeout has been exceeded
// Should be called periodically in long-running operations
bool eshkol_is_timed_out(void);

// Get remaining time in ms (0 if timed out or no timeout set)
uint64_t eshkol_get_remaining_time_ms(void);

// ============================================================================
// Validation Functions
// ============================================================================

// Check if tensor size is allowed
// Returns: true if allowed, false if exceeds limit
bool eshkol_check_tensor_size(size_t num_elements);

// Check if string length is allowed
// Returns: true if allowed, false if exceeds limit
bool eshkol_check_string_length(size_t length);

// ============================================================================
// Error Reporting
// ============================================================================

typedef enum {
    ESHKOL_LIMIT_OK = 0,
    ESHKOL_LIMIT_HEAP_SOFT,       // Soft heap limit reached (warning)
    ESHKOL_LIMIT_HEAP_HARD,       // Hard heap limit exceeded
    ESHKOL_LIMIT_TIMEOUT,         // Execution timeout
    ESHKOL_LIMIT_STACK_OVERFLOW,  // Stack depth exceeded
    ESHKOL_LIMIT_TENSOR_SIZE,     // Tensor too large
    ESHKOL_LIMIT_STRING_LENGTH    // String too long
} eshkol_limit_error_t;

// Get last limit error
eshkol_limit_error_t eshkol_get_last_limit_error(void);

// Get human-readable error message
const char* eshkol_limit_error_message(eshkol_limit_error_t error);

// ============================================================================
// Diagnostics
// ============================================================================

// Print resource usage statistics
void eshkol_print_resource_stats(void);

// Reset all counters and state
void eshkol_reset_resource_tracking(void);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ RAII Helpers
// ============================================================================

#ifdef __cplusplus

namespace eshkol {

// RAII guard for stack frame tracking
class StackFrameGuard {
public:
    StackFrameGuard() : valid_(eshkol_stack_push()) {}
    ~StackFrameGuard() { if (valid_) eshkol_stack_pop(); }

    bool isValid() const { return valid_; }
    operator bool() const { return valid_; }

    // Non-copyable
    StackFrameGuard(const StackFrameGuard&) = delete;
    StackFrameGuard& operator=(const StackFrameGuard&) = delete;

private:
    bool valid_;
};

// RAII guard for execution timer
class TimerGuard {
public:
    explicit TimerGuard(uint64_t timeout_ms = 0) {
        eshkol_start_timer(timeout_ms);
    }
    ~TimerGuard() {
        eshkol_stop_timer();
    }

    bool isTimedOut() const { return eshkol_is_timed_out(); }
    uint64_t remainingMs() const { return eshkol_get_remaining_time_ms(); }

    // Non-copyable
    TimerGuard(const TimerGuard&) = delete;
    TimerGuard& operator=(const TimerGuard&) = delete;
};

// Macro for automatic stack tracking with early return on overflow
#define ESHKOL_STACK_GUARD() \
    eshkol::StackFrameGuard _stack_guard; \
    if (!_stack_guard) { \
        return; \
    }

#define ESHKOL_STACK_GUARD_WITH_VALUE(val) \
    eshkol::StackFrameGuard _stack_guard; \
    if (!_stack_guard) { \
        return (val); \
    }

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_RESOURCE_LIMITS_H
