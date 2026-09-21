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

// Default: 10 million bytecode instructions before the VM calls a program
// runaway. 0 means unlimited.
#define ESHKOL_DEFAULT_MAX_VM_INSTRUCTIONS (10ULL * 1000 * 1000)

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

    // Execution limits
    // Bytecode-VM runaway-instruction guard (0 = unlimited). Lives here rather
    // than being read from the environment by the VM itself: lib/backend/vm_*.c
    // are freestanding-safe sources that may not call getenv(), so every
    // environment variable they obey has to arrive through this struct.
    uint64_t max_vm_instructions;

    // Behavior flags
    bool enforce_hard_limits;        // Kill on hard limit (vs return error)
    bool enable_warnings;            // Log soft limit warnings

    // Which ceilings are ACTIVE, as a bitmask of ESHKOL_LIMIT_ACTIVE_*.
    //
    // The values above are the documented defaults *for a limit you turn on* —
    // they are not ceilings every program is silently held to. A limit becomes
    // active when you ask for it: its environment variable is present, or you
    // set the bit yourself before eshkol_set_limits(). An inactive ceiling is
    // never checked and cannot terminate anything.
    //
    // This matters because the defaults are not idle documentation: 1 GiB of
    // heap is a real number that real programs pass. Turning them all on at
    // once would not be "enforcing what the docs say", it would be imposing a
    // ceiling on every existing program that never had one — this repository's
    // own tests/features/blc_test.esk allocates past 1 GiB, and the bytecode
    // VM's computed-goto dispatch never had an instruction guard at all.
    // Whether the defaults should also bind an unconfigured run is a release
    // decision, recorded in docs/reference/runtime/environment-variables.md.
    uint32_t active_limits;
} eshkol_resource_limits_t;

// Bits for eshkol_resource_limits_t::active_limits.
#define ESHKOL_LIMIT_ACTIVE_HEAP    (1u << 0)  // ESHKOL_MAX_HEAP
#define ESHKOL_LIMIT_ACTIVE_STACK   (1u << 1)  // ESHKOL_MAX_STACK
#define ESHKOL_LIMIT_ACTIVE_TENSOR  (1u << 2)  // ESHKOL_MAX_TENSOR_ELEMS
#define ESHKOL_LIMIT_ACTIVE_STRING  (1u << 3)  // ESHKOL_MAX_STRING_LEN
#define ESHKOL_LIMIT_ACTIVE_TIMEOUT (1u << 4)  // ESHKOL_TIMEOUT_MS
#define ESHKOL_LIMIT_ACTIVE_VM_INSN (1u << 5)  // ESHKOL_VM_MAX_INSN
#define ESHKOL_LIMIT_ACTIVE_ALL     0x3Fu

/**
 * @brief Whether a given ceiling is active for this process.
 *
 * @param which One of the `ESHKOL_LIMIT_ACTIVE_*` bits.
 * @return true if that limit was asked for and should be enforced.
 */
bool eshkol_limit_is_active(uint32_t which);

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
// ESHKOL_VM_MAX_INSN       - Bytecode-VM instruction ceiling (0 = unlimited)
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

/**
 * @brief Parse a byte-size string with an optional binary-multiple suffix.
 *
 * Accepts optional leading/trailing whitespace around a non-negative decimal
 * value, optionally followed by a size suffix: `K`/`k`, `M`/`m`, or `G`/`g`
 * (binary: 1024, 1024^2, 1024^3), each optionally followed by `i`/`I` and/or
 * `B`/`b` — so `512M`, `512MB`, `512MiB`, and `512 MiB` are all accepted and
 * equivalent. This is the parser every `ESHKOL_*` size environment variable
 * (`ESHKOL_MAX_HEAP`, `ESHKOL_MAX_STRING_LEN`, `ESHKOL_STACK_SIZE`, ...) is
 * documented to accept.
 *
 * Any content left over after the (optional) suffix — including an
 * unrecognized suffix letter — is trailing garbage and fails the parse.
 *
 * @param str Candidate string; NULL fails.
 * @param out_bytes Receives the parsed byte count on success; left
 *        unmodified on failure.
 * @return true on success, false on any parse failure (NULL/empty input,
 *         non-numeric value, negative value, unrecognized suffix, trailing
 *         garbage, or overflow of `size_t`).
 * content fails the parse.
 */
bool eshkol_parse_size(const char* str, size_t* out_bytes);

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
// Enforcement
// ============================================================================

/**
 * @brief Process exit statuses used when a hard resource limit terminates a run.
 *
 * `ESHKOL_ENFORCE_LIMITS=true` (the default) is documented to mean that a
 * hard-limit violation *terminates the process*; these are the statuses it
 * terminates with, one per limit, so a supervising process can tell which
 * ceiling was hit without parsing the diagnostic. `ESHKOL_EXIT_LIMIT_TIMEOUT`
 * is 124 to match GNU coreutils `timeout(1)` — the convention this repository
 * already uses for the subprocess timeouts in `run-command` / `run-argv`. The
 * rest occupy the adjacent band below it, and the set deliberately avoids
 * 126/127, which POSIX shells reserve.
 */
#define ESHKOL_EXIT_LIMIT_HEAP    120  // ESHKOL_MAX_HEAP exceeded
#define ESHKOL_EXIT_LIMIT_STACK   121  // ESHKOL_MAX_STACK exceeded
#define ESHKOL_EXIT_LIMIT_TENSOR  122  // ESHKOL_MAX_TENSOR_ELEMS exceeded
#define ESHKOL_EXIT_LIMIT_STRING  123  // ESHKOL_MAX_STRING_LEN exceeded
#define ESHKOL_EXIT_LIMIT_TIMEOUT 124  // ESHKOL_TIMEOUT_MS exceeded
#define ESHKOL_EXIT_LIMIT_VM_INSN 125  // ESHKOL_VM_MAX_INSN exceeded

/**
 * @brief The documented process exit status for a given limit condition.
 *
 * @param error Limit condition to map.
 * @return One of the `ESHKOL_EXIT_LIMIT_*` statuses; `ESHKOL_EXIT_LIMIT_HEAP`
 *         for the soft-heap warning (which never terminates on its own) and
 *         for any unrecognized value.
 */
int eshkol_limit_exit_code(eshkol_limit_error_t error);

/**
 * @brief Act on a detected hard-limit violation according to
 *        `enforce_hard_limits`.
 *
 * The single decision point behind every enforced limit, so that "what a
 * violation does" is defined once rather than re-derived at each call site.
 *
 * - When `enforce_hard_limits` is set (the default, `ESHKOL_ENFORCE_LIMITS=true`):
 *   writes a one-line `eshkol: fatal: ...` diagnostic naming the limit, the
 *   configured ceiling and the environment variable that sets it, to stderr;
 *   flushes stdout and stderr so nothing the program already printed is lost;
 *   requests a runtime interrupt so other threads observe the shutdown; and
 *   terminates the process with the matching `ESHKOL_EXIT_LIMIT_*` status.
 *   **Does not return.**
 * - When it is clear (`ESHKOL_ENFORCE_LIMITS=false`): the limit is advisory.
 *   Emits a warning if `enable_warnings` is set, leaves the condition readable
 *   via eshkol_get_last_limit_error(), and returns so the caller can fail just
 *   that operation — the documented "errors are returned" behaviour.
 *
 * @param error  Which limit was breached.
 * @param detail Optional extra context appended to the diagnostic (may be NULL).
 */
void eshkol_limit_enforce(eshkol_limit_error_t error, const char* detail);

/**
 * @brief Cooperative poll for a pending resource-limit interrupt.
 *
 * The execution-timeout watchdog runs on its own thread and can only *request*
 * an interrupt; something running the user's program has to notice. This is
 * that noticer: cheap enough to sit on a loop back-edge (it reads one global
 * and returns when no interrupt is pending) and, when an interrupt IS pending,
 * routes it through eshkol_limit_enforce() so a timeout terminates with the
 * documented status instead of being printed and ignored.
 *
 * Called from generated code at tail-call loop back-edges, from the recursion
 * guard at function entry, and from the bytecode VM's dispatch loop.
 */
void eshkol_limit_poll_interrupt(void);

/**
 * @brief Apply `ESHKOL_MAX_TENSOR_ELEMS` to a tensor about to be built.
 *
 * Native codegen does not construct tensors through
 * arena_allocate_tensor_full(); it emits the header, dimension and element
 * allocations inline and only knows the element count as an SSA value. This is
 * the entry point generated code calls with that count, so a compiled tensor
 * is bound by the same ceiling as a runtime-constructed one. Combines
 * eshkol_check_tensor_size() with eshkol_limit_enforce(); does not return when
 * the limit is exceeded and enforcement is on.
 *
 * @param num_elements Total element count of the tensor being created.
 */
void eshkol_enforce_tensor_elements(int64_t num_elements);

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
