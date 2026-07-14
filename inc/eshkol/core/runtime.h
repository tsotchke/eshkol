/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Runtime State Management
 *
 * Provides:
 * - Signal handling for graceful shutdown (SIGINT, SIGTERM)
 * - Interrupt checking for long-running operations
 * - Runtime state tracking for cleanup coordination
 * - Shutdown hooks for resource cleanup
 */
#ifndef ESHKOL_CORE_RUNTIME_H
#define ESHKOL_CORE_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration for tagged-value helpers that need it.
 * Full definition lives in inc/eshkol/eshkol.h. */
typedef struct eshkol_tagged_value eshkol_tagged_value_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Runtime State
// ============================================================================

/**
 * @brief Lifecycle state of the Eshkol runtime.
 *
 * Tracked in a process-wide atomic and advanced monotonically by
 * eshkol_runtime_init() / eshkol_runtime_shutdown(): INITIALIZING ->
 * RUNNING -> SHUTTING_DOWN -> TERMINATED. Query with
 * eshkol_runtime_get_state().
 */
typedef enum {
    ESHKOL_RUNTIME_INITIALIZING = 0,  // Before eshkol_runtime_init() has completed
    ESHKOL_RUNTIME_RUNNING,           // Normal operation
    ESHKOL_RUNTIME_SHUTTING_DOWN,     // eshkol_runtime_shutdown() in progress
    ESHKOL_RUNTIME_TERMINATED         // Shutdown complete
} eshkol_runtime_state_t;

/**
 * @brief Why the runtime is shutting down or was interrupted.
 *
 * Set by eshkol_runtime_request_interrupt() / eshkol_runtime_shutdown()
 * and readable via eshkol_runtime_get_shutdown_reason(); also passed to
 * every registered eshkol_shutdown_hook_t so hooks can react differently
 * (e.g. flush vs. abandon) depending on cause.
 */
typedef enum {
    ESHKOL_SHUTDOWN_NONE = 0,       // No shutdown in progress
    ESHKOL_SHUTDOWN_REQUESTED,      // User requested shutdown (Ctrl+C)
    ESHKOL_SHUTDOWN_TIMEOUT,        // Execution timeout reached
    ESHKOL_SHUTDOWN_MEMORY,         // Memory limit exceeded
    ESHKOL_SHUTDOWN_ERROR           // Unrecoverable error
} eshkol_shutdown_reason_t;

// ============================================================================
// Shutdown Hook System
// ============================================================================

/**
 * @brief Callback signature for a registered shutdown hook.
 *
 * @param context User-provided context pointer, passed through unchanged.
 * @param reason Why shutdown is happening.
 * @return 0 on success; non-zero is logged as a warning but does not
 *         prevent the remaining hooks from running or shutdown from
 *         completing.
 */
typedef int (*eshkol_shutdown_hook_t)(void* context, eshkol_shutdown_reason_t reason);

/**
 * @brief Register a callback to run during runtime shutdown.
 *
 * Hooks are invoked by eshkol_runtime_shutdown() in reverse registration
 * order (most-recently-registered first), each under its own copy of the
 * hook table so a hook may safely register or unregister other hooks.
 * Thread-safe.
 *
 * @param hook Function to call during shutdown; registration is ignored
 *             (returns 0) if NULL.
 * @param context User context passed to hook unchanged.
 * @param name Human-readable name for logging/debugging; NULL becomes
 *             "(unnamed)".
 * @return Hook ID (>0) on success, 0 on failure.
 */
uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                        void* context,
                                        const char* name);

/**
 * @brief Remove a previously registered shutdown hook.
 *
 * @param hook_id ID returned by eshkol_register_shutdown_hook().
 * @return true if a hook with this ID was found and removed, false
 *         otherwise (including hook_id == 0).
 */
bool eshkol_unregister_shutdown_hook(uint32_t hook_id);

// ============================================================================
// Signal Handling
// ============================================================================

/**
 * @brief Install the runtime's signal handlers.
 *
 * Call once at program start. Installs handlers for SIGINT and SIGTERM
 * (requesting a graceful interrupt) and ignores SIGPIPE.
 */
void eshkol_runtime_init_signals(void);

/**
 * @brief Restore the platform's default signal handlers.
 *
 * Call before exec() or fork() so a child process does not inherit
 * Eshkol's handlers.
 */
void eshkol_runtime_restore_signals(void);

// ============================================================================
// Interrupt Checking
// ============================================================================

/**
 * @brief Check whether an interrupt has been requested.
 *
 * Reads the process-wide `g_eshkol_interrupt_flag` (a `volatile
 * sig_atomic_t`, safe to poll from a signal handler's perspective).
 * Call this in hot loops to enable graceful cancellation on Ctrl+C,
 * timeout, or memory-limit exhaustion. Defined inline near the bottom
 * of this header for zero call overhead on the hot path.
 *
 * @return true if an interrupt is pending, false otherwise.
 */
static inline bool eshkol_runtime_interrupt_requested(void);

/**
 * @brief Non-inline accessor for the interrupt flag.
 *
 * Equivalent to eshkol_runtime_interrupt_requested() but exported as a
 * real symbol for FFI and tooling surfaces that cannot inline the
 * header's static function (e.g. across a dynamic-library boundary).
 *
 * @return true if an interrupt is pending, false otherwise.
 */
bool eshkol_runtime_interrupt_flag_is_set(void);

/**
 * @brief Request that the runtime interrupt/shut down (thread-safe).
 *
 * Sets the interrupt flag, records `reason` as the current shutdown
 * reason, and transitions the runtime state to
 * ESHKOL_RUNTIME_SHUTTING_DOWN. Safe to call from a signal handler or
 * any thread.
 *
 * @param reason Why the interrupt is being requested.
 */
void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason);

/**
 * @brief Clear the interrupt flag and reset the shutdown reason to NONE.
 *
 * Intended for restart scenarios (e.g. a REPL recovering after a
 * previous cancellation) where execution should resume normally.
 */
void eshkol_runtime_clear_interrupt(void);

/**
 * @brief Get the reason recorded for the current/most recent interrupt.
 *
 * @return The shutdown reason last set by eshkol_runtime_request_interrupt()
 *         or eshkol_runtime_shutdown(), or ESHKOL_SHUTDOWN_NONE if none.
 */
eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void);

// ============================================================================
// Runtime Lifecycle
// ============================================================================

/**
 * @brief Initialize the runtime system. Call once at program start.
 *
 * Transitions state from ESHKOL_RUNTIME_INITIALIZING to
 * ESHKOL_RUNTIME_RUNNING (a no-op success if already RUNNING), makes
 * stdout unbuffered so `display` output is never lost on a crash,
 * loads resource limits from the environment (starting the execution
 * timer if `ESHKOL_TIMEOUT_MS` is set), and installs signal handlers
 * via eshkol_runtime_init_signals().
 *
 * @return 0 on success, non-zero if called from an unexpected state.
 */
int eshkol_runtime_init(void);

/**
 * @brief Shut down the runtime system.
 *
 * Idempotent: a no-op if already SHUTTING_DOWN or TERMINATED. Transitions
 * to ESHKOL_RUNTIME_SHUTTING_DOWN, records `reason`, sets the interrupt
 * flag, waits (up to 5s) for in-flight operations to drain via
 * eshkol_runtime_drain_operations(), stops runtime-owned thread pools,
 * and then runs all registered shutdown hooks in reverse registration
 * order.
 *
 * @param reason Why shutdown is happening.
 */
void eshkol_runtime_shutdown(eshkol_shutdown_reason_t reason);

/**
 * @brief Get the current runtime lifecycle state.
 *
 * @return The current eshkol_runtime_state_t value.
 */
eshkol_runtime_state_t eshkol_runtime_get_state(void);

// ============================================================================
// In-flight Operation Tracking
// ============================================================================

// Track operations for graceful draining during shutdown
// Register when starting a long operation, unregister when done

/**
 * @brief Register the start of a long-running operation.
 *
 * Thread-safe; the returned ID must later be passed to
 * eshkol_runtime_end_operation(). eshkol_runtime_shutdown() waits for
 * all such operations to finish (up to a timeout) before proceeding.
 *
 * @param name Human-readable operation name for logging (NULL becomes
 *             "(unnamed)").
 * @return Operation ID (>0) on success, 0 on failure.
 */
uint32_t eshkol_runtime_begin_operation(const char* name);

/**
 * @brief Mark a previously registered operation as complete.
 *
 * Wakes any thread blocked in eshkol_runtime_drain_operations(). A
 * no-op if `operation_id` is 0 or unknown.
 *
 * @param operation_id ID returned by eshkol_runtime_begin_operation().
 */
void eshkol_runtime_end_operation(uint32_t operation_id);

/**
 * @brief Block until all in-flight operations complete or a timeout elapses.
 *
 * @param timeout_ms Maximum time to wait: 0 returns immediately (checking
 *                    the current count without waiting), a negative value
 *                    waits indefinitely, and a positive value bounds the
 *                    wait in milliseconds.
 * @return true if all operations completed (or none were in flight),
 *         false if the timeout elapsed first.
 */
bool eshkol_runtime_drain_operations(int timeout_ms);

/**
 * @brief Get the number of currently in-flight operations.
 *
 * @return Count of operations registered via eshkol_runtime_begin_operation()
 *         that have not yet been ended.
 */
uint32_t eshkol_runtime_get_operation_count(void);

// ============================================================================
// Type Errors (R7RS Compliance)
// ============================================================================

/**
 * @brief Report a type error and abort.
 *
 * Per R7RS, type errors are fatal: this formats "Type error in
 * <proc_name>: expected <expected_type>" and raises it as an
 * ESHKOL_EXCEPTION_TYPE_ERROR runtime fatal, which terminates execution.
 * Does not return.
 *
 * @param proc_name Name of the procedure that encountered the error.
 * @param expected_type The type that was expected (e.g., "number", "pair").
 */
void eshkol_type_error(const char* proc_name, const char* expected_type);

/**
 * @brief Map a tagged value's runtime type to a human-readable type name.
 *
 * Single source of truth for "what does the user see?" type-name
 * reporting in error messages. Returns a static string; never NULL.
 *
 * Type names use the canonical Eshkol/Scheme spellings:
 *   "integer", "rational", "bignum", "double", "complex", "dual-number",
 *   "symbol", "string", "character", "boolean",
 *   "pair", "vector", "tensor", "procedure", "continuation", "ad-node",
 *   "environment", "hash-table", "port", "promise", "prng", "null",
 *   "logic-var", "substitution", "fact", "knowledge-base",
 *   "factor-graph", "workspace", "bytevector", "record", "exception",
 *   "heap-object", "unknown-type".
 *
 * Implemented in lib/core/runtime_errors_hosted.cpp.
 *
 * @param v Tagged value to classify.
 * @return Static, never-NULL string naming the value's runtime type.
 */
const char* eshkol_format_value_type_tag(eshkol_tagged_value_t v);

/**
 * @brief Raise a type error reporting the operand's actual runtime type
 * alongside the expected one.
 *
 * Produces: "Type error in <proc>: expected <expected>, got <actual>".
 * Generated codegen sites that have the operand in scope should call
 * this instead of eshkol_type_error() so users see the concrete operand
 * type ("got string") rather than guessing which operand was wrong.
 * Does not return.
 *
 * @param proc_name Name of the procedure that encountered the error.
 * @param expected_type The type that was expected.
 * @param actual Pointer to the offending tagged value. `actual` is passed
 *        by pointer rather than by value: a 16-byte tagged-value struct
 *        passed by value does not survive the LLVM-IR -> C ABI boundary
 *        reliably (it arrives zeroed, formatting as "null"). Codegen
 *        stores the operand to an alloca and passes its address. NULL is
 *        treated as an ESHKOL_VALUE_NULL operand.
 */
void eshkol_type_error_with_operand(const char* proc_name,
                                    const char* expected_type,
                                    const eshkol_tagged_value_t* actual);

/**
 * @brief Set the source location to prefix onto the *next* runtime error
 * (v1.3 source-span errors).
 *
 * Codegen emits a call to this on the type-error branch immediately
 * before raising, so the error formatter can prepend a
 * "file:line:col:" prefix to the message. `file` must be a stable C
 * string (codegen emits a global; the pointer is held, not copied).
 * Thread-local, so concurrent workers don't clobber one another.
 * Clearing is not required between errors — the location is overwritten
 * at the next error site.
 *
 * @param file Stable, static source file path string.
 * @param line 1-based source line number.
 * @param column 1-based source column number.
 */
void eshkol_set_error_location(const char* file, uint32_t line, uint32_t column);

/**
 * @brief Clear the current thread's pending error source location.
 *
 * Resets the state set by eshkol_set_error_location() so a subsequent
 * error is reported without a "file:line:col:" prefix.
 */
void eshkol_clear_error_location(void);

/* ============================================================================
 * Source-Span Stack Traces (v1.3)
 *
 * Codegen pushes a frame at every Eshkol function entry; pops at exit.
 * The exception machinery walks the stack and renders it before the
 * runtime error message, so users see Eshkol-level frames instead of
 * raw C unwinds.
 *
 *   Traceback (most recent call last):
 *     File "scripts/agent.esk", line 142, in (agent-handle-request)
 *     File "scripts/agent.esk", line 89, in (route-message)
 *   Type error in <: expected number, got string
 *
 * All names + file paths are static C strings emitted as globals by
 * codegen, so push/pop is trivial (no allocation).  Thread-local; cap
 * at 256 frames.  Overflow records a single ".. truncated .." marker.
 * ============================================================================
 */

/**
 * @brief Push a new frame onto the current thread's Eshkol-level call stack.
 *
 * Codegen emits a call to this at every Eshkol function entry (paired
 * with eshkol_frame_pop() at exit) so runtime errors can render an
 * Eshkol-level traceback instead of a raw C unwind. `function_name` and
 * `source_file` must be stable C strings (codegen emits them as
 * globals); no copy is made. Thread-local; capped at 256 frames — once
 * the cap is reached, further pushes are dropped and recorded as a
 * single overflow marker rendered by eshkol_frame_print_trace().
 *
 * @param function_name Name of the entered function (NULL becomes "<anonymous>").
 * @param source_file Source file containing the call site, or NULL if unknown.
 * @param source_line Source line of the call site.
 * @param source_column Source column of the call site (0 if unknown/unused).
 */
void eshkol_frame_push(const char* function_name,
                       const char* source_file,
                       uint32_t source_line,
                       uint32_t source_column);

/**
 * @brief Pop the most recently pushed frame from the current thread's
 * Eshkol-level call stack.
 *
 * A no-op if the stack is already empty (beyond clearing the overflow
 * marker once depth has unwound below the cap).
 */
void eshkol_frame_pop(void);

/**
 * @brief Get the current thread's Eshkol-level call stack depth.
 *
 * @return Number of frames currently pushed (not counting truncated
 *         overflow frames beyond the 256-frame cap).
 */
uint32_t eshkol_frame_stack_depth(void);

/**
 * @brief Print the current thread's Eshkol-level frame stack as a
 * traceback, most-recent call last.
 *
 * Output format mirrors Python/Ruby/Julia tracebacks:
 *   Traceback (most recent call last):
 *     File "<file>", line <N>[:<C>], in (<function-name>)
 *     ...
 * Prints no trailing newline; the caller's error message is expected to
 * follow immediately. A no-op if the stack is empty and has not
 * overflowed.
 *
 * @param fp Destination stream, cast from `FILE*`; pass NULL to write to stderr.
 */
void eshkol_frame_print_trace(void* fp);          /* FILE*; pass NULL → stderr */

/**
 * @brief Convenience wrapper: print the current traceback to stderr.
 *
 * Equivalent to `eshkol_frame_print_trace(NULL)`. Used by the exception
 * machinery immediately before the runtime error message.
 */
void eshkol_frame_print_trace_stderr(void);

/**
 * @brief Reset the current thread's Eshkol-level call stack.
 *
 * Clears all pushed frames and the overflow marker. Call at REPL prompt
 * boundaries so a new top-level expression doesn't inherit the previous
 * expression's stack.
 */
void eshkol_frame_stack_reset(void);              /* call at REPL prompt boundary */

/**
 * @brief Report a type error with both expected and actual type names, and abort.
 *
 * Formats and raises a fatal type error including the concrete type the
 * caller observed, prefixed by any pending source location. Does not
 * return.
 *
 * @param proc_name Name of the procedure that encountered the error.
 * @param expected_type The type that was expected.
 * @param actual_type The type that was actually received.
 */
void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                   const char* actual_type);

// ============================================================================
// Opt-in executable language-surface coverage
// ============================================================================

/**
 * Record that the parser actually dispatched a source form.  These hooks are
 * inert unless ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR names an output directory.
 * The parser event is paired with a runtime operation/call event by source
 * location, so an expression in an untaken branch earns no execution credit.
 */
void eshkol_language_coverage_parse(const char* source,
                                    uint32_t line,
                                    uint32_t column,
                                    uint32_t operation,
                                    const char* name);

/** Record a fully accepted top-level parser result. */
void eshkol_language_coverage_accept(const char* source,
                                     uint32_t line,
                                     uint32_t column,
                                     uint32_t operation);

/** Record an intentionally rejected negative compile-time form. */
void eshkol_language_coverage_reject(const char* source,
                                     uint32_t line,
                                     uint32_t column,
                                     uint32_t operation,
                                     const char* name);

/** Record that code generation reached an AST operation. */
void eshkol_language_coverage_codegen(const char* source,
                                      uint32_t line,
                                      uint32_t column,
                                      uint32_t operation);

/** Record execution of an AST operation at its original source location. */
void eshkol_language_coverage_exec_op(const char* source,
                                      uint32_t line,
                                      uint32_t column,
                                      uint32_t operation);

/** Record execution of a direct Scheme call at its original source location. */
void eshkol_language_coverage_exec_call(const char* source,
                                        uint32_t line,
                                        uint32_t column,
                                        const char* name);

/**
 * Record an exact bytecode-VM native dispatch. The VM emits this only when
 * an opt-in coverage marker immediately precedes a matching native call, so
 * aliases sharing one native ID remain distinguishable after ESKB
 * serialization without granting credit for merely compiled bytecode.
 */
void eshkol_language_coverage_vm_dispatch(const char* name,
                                          uint32_t native_id);
/** Record a validated direct Scheme closure call from serialized VM bytecode.
 * The stable hash is resolved against the manifest with collision rejection. */
void eshkol_language_coverage_vm_call_hash(uint32_t name_hash);
/** Flush a pending opt-in language-coverage batch before exec/early exit. */
void eshkol_language_coverage_flush(void);

// ============================================================================
// Implementation Details (defined in runtime.cpp)
// ============================================================================

/**
 * @brief Process-wide interrupt flag, set by the signal handler.
 *
 * `volatile sig_atomic_t` so it is safe to write from an async-signal
 * context and safe to poll from ordinary code without additional
 * synchronization. Non-zero means an interrupt is pending. Prefer the
 * accessor functions (eshkol_runtime_interrupt_requested(),
 * eshkol_runtime_interrupt_flag_is_set()) over reading this directly.
 */
extern volatile sig_atomic_t g_eshkol_interrupt_flag;

// Inline implementation for hot-path interrupt checking
static inline bool eshkol_runtime_interrupt_requested(void) {
    return g_eshkol_interrupt_flag != 0;
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ RAII Helpers
// ============================================================================

#ifdef __cplusplus

namespace eshkol {

/**
 * @brief RAII guard registering an in-flight operation for the lifetime
 * of the guard.
 *
 * Calls eshkol_runtime_begin_operation() on construction and
 * eshkol_runtime_end_operation() on destruction (if the begin call
 * succeeded), so eshkol_runtime_shutdown() will wait for the guarded
 * scope to exit before proceeding. Movable but not copyable, so
 * ownership of the underlying operation ID transfers cleanly.
 */
class OperationGuard {
public:
    /**
     * @brief Register the start of an operation named `name`.
     * @param name Human-readable operation name for logging.
     */
    explicit OperationGuard(const char* name)
        : operation_id_(eshkol_runtime_begin_operation(name)) {}

    ~OperationGuard() {
        if (operation_id_ != 0) {
            eshkol_runtime_end_operation(operation_id_);
        }
    }

    // Non-copyable
    OperationGuard(const OperationGuard&) = delete;
    OperationGuard& operator=(const OperationGuard&) = delete;

    // Movable
    OperationGuard(OperationGuard&& other) noexcept
        : operation_id_(other.operation_id_) {
        other.operation_id_ = 0;
    }

    OperationGuard& operator=(OperationGuard&& other) noexcept {
        if (this != &other) {
            if (operation_id_ != 0) {
                eshkol_runtime_end_operation(operation_id_);
            }
            operation_id_ = other.operation_id_;
            other.operation_id_ = 0;
        }
        return *this;
    }

    /** @brief Whether registration succeeded (operation ID is non-zero). */
    bool isValid() const { return operation_id_ != 0; }
    /** @brief The underlying operation ID (0 if registration failed). */
    uint32_t id() const { return operation_id_; }

private:
    uint32_t operation_id_;
};

/**
 * @brief RAII guard registering a shutdown hook for the lifetime of the
 * guard.
 *
 * Calls eshkol_register_shutdown_hook() on construction and
 * eshkol_unregister_shutdown_hook() on destruction (if registration
 * succeeded). Neither copyable nor movable, since a hook's context
 * pointer and lifetime must stay tied to a single, stable guard
 * instance.
 */
class ShutdownHookGuard {
public:
    /**
     * @brief Register `hook` to be called during shutdown.
     * @param hook Callback to invoke during shutdown.
     * @param context User context passed to `hook` unchanged.
     * @param name Human-readable name for logging.
     */
    ShutdownHookGuard(eshkol_shutdown_hook_t hook, void* context, const char* name)
        : hook_id_(eshkol_register_shutdown_hook(hook, context, name)) {}

    ~ShutdownHookGuard() {
        if (hook_id_ != 0) {
            eshkol_unregister_shutdown_hook(hook_id_);
        }
    }

    // Non-copyable, non-movable (hooks should have stable lifetime)
    ShutdownHookGuard(const ShutdownHookGuard&) = delete;
    ShutdownHookGuard& operator=(const ShutdownHookGuard&) = delete;
    ShutdownHookGuard(ShutdownHookGuard&&) = delete;
    ShutdownHookGuard& operator=(ShutdownHookGuard&&) = delete;

    /** @brief Whether registration succeeded (hook ID is non-zero). */
    bool isRegistered() const { return hook_id_ != 0; }
    /** @brief The underlying hook ID (0 if registration failed). */
    uint32_t id() const { return hook_id_; }

private:
    uint32_t hook_id_;
};

/**
 * @brief Check for a pending interrupt and return void early if one is set.
 *
 * Expands to a `do { ... } while (0)` block; use in a `void`-returning
 * hot loop body to enable graceful cancellation without hand-writing
 * the interrupt check at every call site.
 */
#define ESHKOL_CHECK_INTERRUPT() \
    do { \
        if (eshkol_runtime_interrupt_requested()) { \
            return; \
        } \
    } while (0)

/**
 * @brief Check for a pending interrupt and return `val` early if one is set.
 *
 * Like ESHKOL_CHECK_INTERRUPT() but for functions that return a value;
 * `val` is returned on interrupt.
 *
 * @param val Expression to return if an interrupt is pending.
 */
#define ESHKOL_CHECK_INTERRUPT_WITH_VALUE(val) \
    do { \
        if (eshkol_runtime_interrupt_requested()) { \
            return (val); \
        } \
    } while (0)

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_RUNTIME_H
