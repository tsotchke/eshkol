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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#  if __has_include(<signal.h>)
#    include <signal.h>
#  else
typedef int sig_atomic_t;
#  endif
#else
#  include <signal.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Runtime State
// ============================================================================

typedef enum {
    ESHKOL_RUNTIME_INITIALIZING = 0,
    ESHKOL_RUNTIME_RUNNING,
    ESHKOL_RUNTIME_SHUTTING_DOWN,
    ESHKOL_RUNTIME_TERMINATED
} eshkol_runtime_state_t;

typedef enum {
    ESHKOL_SHUTDOWN_NONE = 0,
    ESHKOL_SHUTDOWN_REQUESTED,      // User requested shutdown (Ctrl+C)
    ESHKOL_SHUTDOWN_TIMEOUT,        // Execution timeout reached
    ESHKOL_SHUTDOWN_MEMORY,         // Memory limit exceeded
    ESHKOL_SHUTDOWN_ERROR           // Unrecoverable error
} eshkol_shutdown_reason_t;

// ============================================================================
// Shutdown Hook System
// ============================================================================

// Shutdown hook callback type
// - context: User-provided context pointer
// - reason: Why shutdown is happening
// Returns: 0 on success, non-zero on error (logged but doesn't prevent shutdown)
typedef int (*eshkol_shutdown_hook_t)(void* context, eshkol_shutdown_reason_t reason);

// Register a shutdown hook (called in reverse registration order)
// - hook: Function to call during shutdown
// - context: User context passed to hook
// - name: Human-readable name for debugging (optional, can be NULL)
// Returns: hook ID (>0) on success, 0 on failure
uint32_t eshkol_register_shutdown_hook(eshkol_shutdown_hook_t hook,
                                        void* context,
                                        const char* name);

// Unregister a shutdown hook by ID
// Returns: true if hook was found and removed
bool eshkol_unregister_shutdown_hook(uint32_t hook_id);

// ============================================================================
// Signal Handling
// ============================================================================

// Initialize signal handlers (call once at program start)
// Installs handlers for: SIGINT, SIGTERM, SIGPIPE (ignored)
void eshkol_runtime_init_signals(void);

// Restore default signal handlers (call before exec, fork)
void eshkol_runtime_restore_signals(void);

// ============================================================================
// Interrupt Checking
// ============================================================================

// Check if an interrupt has been requested
// Call this in hot loops to enable graceful cancellation
// Returns: true if interrupt requested, false otherwise
static inline bool eshkol_runtime_interrupt_requested(void);

// Request an interrupt (thread-safe)
// - reason: Why interrupt is requested
void eshkol_runtime_request_interrupt(eshkol_shutdown_reason_t reason);

// Clear interrupt flag (for restart scenarios)
void eshkol_runtime_clear_interrupt(void);

// Get the current shutdown reason
eshkol_shutdown_reason_t eshkol_runtime_get_shutdown_reason(void);

// ============================================================================
// Runtime Lifecycle
// ============================================================================

// Initialize the runtime system (call once at program start)
// - Sets up signal handlers
// - Initializes shutdown hook registry
// Returns: 0 on success, non-zero on error
int eshkol_runtime_init(void);

// Shutdown the runtime system
// - Calls all registered shutdown hooks
// - Cleans up runtime resources
// - reason: Why shutdown is happening
void eshkol_runtime_shutdown(eshkol_shutdown_reason_t reason);

// Get current runtime state
eshkol_runtime_state_t eshkol_runtime_get_state(void);

// ============================================================================
// In-flight Operation Tracking
// ============================================================================

// Track operations for graceful draining during shutdown
// Register when starting a long operation, unregister when done

// Register an in-flight operation
// - name: Human-readable operation name
// Returns: operation ID (>0) on success, 0 on failure
uint32_t eshkol_runtime_begin_operation(const char* name);

// Mark an operation as complete
void eshkol_runtime_end_operation(uint32_t operation_id);

// Wait for all in-flight operations to complete
// - timeout_ms: Maximum time to wait (0 = no wait, -1 = wait forever)
// Returns: true if all operations completed, false if timeout
bool eshkol_runtime_drain_operations(int timeout_ms);

// Get count of in-flight operations
uint32_t eshkol_runtime_get_operation_count(void);

// Install or clear a BSP-owned drain hook for runtimes that cannot block on
// in-flight operations internally.
//
// The hook is primarily for freestanding targets. It is invoked by
// `eshkol_runtime_drain_operations()` when operations are still active and the
// caller requested a positive timeout or an infinite wait. The hook may poll or
// sleep cooperatively, and may consult `eshkol_runtime_get_operation_count()`
// while servicing the drain request.
//
// Returns from the hook:
// - true: the hook serviced the drain request and the runtime should re-check
//   whether operations are now complete
// - false: the hook could not complete the wait (for example timeout or no
//   platform wait primitive available)
typedef bool (*eshkol_runtime_operation_drain_hook_t)(int timeout_ms,
                                                      void* context);
void eshkol_runtime_set_operation_drain_hook(
    eshkol_runtime_operation_drain_hook_t hook,
    void* context);
void eshkol_runtime_clear_operation_drain_hook(void);

// ============================================================================
// Runtime Timebase And Delay
// ============================================================================

// Install or clear profile/BSP-owned monotonic time and delay hooks.
//
// These hooks exist so shared runtime code can depend on an explicit monotonic
// timebase and bounded delay surface without pulling hosted clocks or sleep
// primitives directly into `runtime-core`.
//
// Hosted runtime provides default implementations backed by the host steady
// clock and sleep facilities. Freestanding runtime remains hook-driven: without
// an installed BSP hook, monotonic time queries and non-zero delays report
// failure instead of silently inventing policy.
//
// Time values and delay durations are expressed in nanoseconds.
typedef bool (*eshkol_runtime_monotonic_clock_hook_t)(uint64_t* out_time_ns,
                                                      void* context);
typedef bool (*eshkol_runtime_delay_hook_t)(uint64_t duration_ns,
                                            void* context);

void eshkol_runtime_set_monotonic_clock_hook(
    eshkol_runtime_monotonic_clock_hook_t hook,
    void* context);
void eshkol_runtime_clear_monotonic_clock_hook(void);

void eshkol_runtime_set_delay_hook(eshkol_runtime_delay_hook_t hook,
                                   void* context);
void eshkol_runtime_clear_delay_hook(void);

// Read the current monotonic time in nanoseconds.
// Returns true on success, false when the profile has no available timebase.
bool eshkol_runtime_get_monotonic_time_ns(uint64_t* out_time_ns);

// Delay for at least the requested duration in nanoseconds.
// A zero-duration delay is always treated as a successful no-op.
// Returns true on success, false when the profile has no available delay
// primitive for non-zero durations.
bool eshkol_runtime_delay_ns(uint64_t duration_ns);

// ============================================================================
// Runtime Capability Registry
// ============================================================================

// The runtime hook surface is the first explicit substrate capability family.
// Each capability describes one installable runtime-owned or profile-owned
// service that higher layers may inspect before binding policy around it.
typedef enum {
    ESHKOL_RUNTIME_CAPABILITY_DIAGNOSTIC_SINK = 0,
    ESHKOL_RUNTIME_CAPABILITY_FATAL_SINK = 1,
    ESHKOL_RUNTIME_CAPABILITY_MONOTONIC_CLOCK = 2,
    ESHKOL_RUNTIME_CAPABILITY_DELAY = 3,
    ESHKOL_RUNTIME_CAPABILITY_OPERATION_DRAIN = 4
} eshkol_runtime_capability_kind_t;

typedef enum {
    ESHKOL_RUNTIME_CAPABILITY_FLAG_DEFAULT_AVAILABLE = 1u << 0,
    ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLABLE = 1u << 1,
    ESHKOL_RUNTIME_CAPABILITY_FLAG_HOOK_INSTALLED = 1u << 2
} eshkol_runtime_capability_flag_t;

typedef struct {
    eshkol_runtime_capability_kind_t kind;
    const char* name;
    uint32_t flags;
} eshkol_runtime_capability_descriptor_t;

// Enumerate the bounded runtime capability family currently exposed by the
// shared runtime surface.
size_t eshkol_runtime_get_capability_count(void);

// Describe one capability by stable enumeration index or stable kind.
// Returns false on invalid index/kind or null output storage.
bool eshkol_runtime_describe_capability_at(
    size_t index,
    eshkol_runtime_capability_descriptor_t* out_descriptor);
bool eshkol_runtime_describe_capability(
    eshkol_runtime_capability_kind_t kind,
    eshkol_runtime_capability_descriptor_t* out_descriptor);

// ============================================================================
// Runtime Diagnostics
// ============================================================================

typedef enum {
    ESHKOL_RUNTIME_DIAGNOSTIC_DEBUG = 0,
    ESHKOL_RUNTIME_DIAGNOSTIC_INFO = 1,
    ESHKOL_RUNTIME_DIAGNOSTIC_WARNING = 2,
    ESHKOL_RUNTIME_DIAGNOSTIC_ERROR = 3
} eshkol_runtime_diagnostic_level_t;

typedef void (*eshkol_runtime_diagnostic_hook_t)(
    eshkol_runtime_diagnostic_level_t level,
    const char* message,
    void* context);

// Install or clear a runtime diagnostic hook.
// When set, shared runtime code routes debug/info/warn/error diagnostics to
// this hook instead of the hosted logger sink.
void eshkol_runtime_set_diagnostic_hook(eshkol_runtime_diagnostic_hook_t hook,
                                        void* context);
void eshkol_runtime_clear_diagnostic_hook(void);

// Profile-owned runtime diagnostic sinks for shared runtime code.
void eshkol_runtime_debugf(const char* format, ...);
void eshkol_runtime_infof(const char* format, ...);
void eshkol_runtime_warnf(const char* format, ...);
void eshkol_runtime_errorf(const char* format, ...);

// ============================================================================
// Fatal Runtime Reporting
// ============================================================================

typedef void (*eshkol_runtime_fatal_hook_t)(const char* message, void* context);

// Install or clear a fatal runtime hook.
// The hook is invoked before the profile's default fatal sink.
// Hooks that want to own termination should not return.
void eshkol_runtime_set_fatal_hook(eshkol_runtime_fatal_hook_t hook,
                                   void* context);
void eshkol_runtime_clear_fatal_hook(void);

// Report an unrecoverable runtime error and terminate.
// Profile implementations own the reporting sink and termination behavior.
void eshkol_runtime_fatalf(const char* format, ...);

// ============================================================================
// Type Errors (R7RS Compliance)
// ============================================================================

// Report a type error and abort (R7RS: type errors are fatal)
// - proc_name: Name of the procedure that encountered the error
// - expected_type: The type that was expected (e.g., "number", "pair")
void eshkol_type_error(const char* proc_name, const char* expected_type);

// Report a type error with actual type and abort
// - proc_name: Name of the procedure that encountered the error
// - expected_type: The type that was expected
// - actual_type: The type that was actually received
void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                   const char* actual_type);

// ============================================================================
// Implementation Details (defined in runtime.cpp)
// ============================================================================

// Global volatile flag for fast interrupt checking (set by signal handler)
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

// RAII guard for in-flight operations
class OperationGuard {
public:
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

    bool isValid() const { return operation_id_ != 0; }
    uint32_t id() const { return operation_id_; }

private:
    uint32_t operation_id_;
};

// RAII guard for shutdown hooks
class ShutdownHookGuard {
public:
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

    bool isRegistered() const { return hook_id_ != 0; }
    uint32_t id() const { return hook_id_; }

private:
    uint32_t hook_id_;
};

// Check for interrupt in loops with automatic early return
#define ESHKOL_CHECK_INTERRUPT() \
    do { \
        if (eshkol_runtime_interrupt_requested()) { \
            return; \
        } \
    } while (0)

#define ESHKOL_CHECK_INTERRUPT_WITH_VALUE(val) \
    do { \
        if (eshkol_runtime_interrupt_requested()) { \
            return (val); \
        } \
    } while (0)

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_RUNTIME_H
