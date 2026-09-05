/*
 * XLA Runtime Execution for Eshkol
 *
 * Executes compiled XLA computations and manages runtime state.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_XLA_RUNTIME_H
#define ESHKOL_XLA_RUNTIME_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eshkol {
namespace xla {

// Forward declarations
enum class Target;

/**
 * Element type of the bytes a BufferDescriptor points at.
 *
 * This exists because `element_size` alone cannot tell a device what it is
 * being handed. PJRT's bufferFromHost takes a TYPE, not a width, and the
 * execute() path below used to pass kF64 unconditionally — so an f32 buffer
 * would have been described to the plugin as f64, which is not a rejected
 * transfer but a silently misread one (see XLARuntime::execute).
 *
 * F64 is the default so that every buffer built before this enum existed keeps
 * exactly the meaning it had.
 */
enum class BufferElementType {
    F64,   // 64-bit float — every Eshkol tensor
    F32    // 32-bit float — the device element type where f64 is unavailable (TPU)
};

/**
 * Buffer descriptor for XLA execution
 */
struct BufferDescriptor {
    void* data;                          // Pointer to data
    std::vector<int64_t> shape;          // Tensor shape
    size_t element_size;                 // Size of each element (e.g., 8 for double)
    bool on_device;                      // True if buffer is on GPU
    BufferElementType elem = BufferElementType::F64;  // Type of the bytes at `data`
};

/**
 * Execution result
 */
struct ExecutionResult {
    bool success;                    // True if execution succeeded
    std::string error_message;       // Diagnostic message on failure (empty on success)
    int64_t execution_time_ns;       // Wall-clock execution time in nanoseconds
};

/**
 * XLARuntime - Executes XLA computations
 *
 * Manages execution of compiled XLA computations, including
 * memory allocation, data transfer, and synchronization.
 */
class XLARuntime {
public:
    /**
     * Construct an uninitialized runtime. Call initialize() before use.
     */
    XLARuntime();

    /**
     * Destroy the runtime, releasing any device resources.
     */
    ~XLARuntime();

    // Non-copyable
    XLARuntime(const XLARuntime&) = delete;
    XLARuntime& operator=(const XLARuntime&) = delete;

    // ===== Initialization =====

    /**
     * Initialize runtime for a target.
     *
     * Independently of `target`, if the environment variable ESHKOL_XLA_PJRT
     * is set to "1" (and this build has the optional PJRT client compiled
     * in), also attempts to select real device execution via PJRT — see
     * pjrt_client.h and ESHKOL_PJRT_PLUGIN_PATH. This never changes what this
     * function returns; whether PJRT ended up active is reported through
     * getDescription(), not through the return value.
     *
     * @param target Target backend
     * @return true on success
     */
    bool initialize(Target target);

    /**
     * Check if runtime is initialized.
     * @return true if ready
     */
    bool isInitialized() const;

    /**
     * Get the active target.
     * @return Current target
     */
    Target getTarget() const;

    // ===== Execution =====

    /**
     * Execute a compiled computation.
     *
     * What `executable` must be depends on how this runtime was initialized:
     * a raw `void(const void* const*, void* const*)` function pointer on the
     * LLVM-direct path (the default), or a `PJRT_LoadedExecutable*` from
     * PjrtClient::compile() when PJRT device execution is active (see
     * initialize()). There is no runtime tag distinguishing the two — pass
     * whichever this runtime was initialized for.
     *
     * @param executable Compiled executable
     * @param inputs Input buffers
     * @param outputs Output buffers (pre-allocated)
     * @return Execution result
     */
    ExecutionResult execute(void* executable,
                            const std::vector<BufferDescriptor>& inputs,
                            std::vector<BufferDescriptor>& outputs);

    /**
     * Compile a StableHLO module for the active PJRT device.
     *
     * This is the entry point the device lowering path uses (see
     * device_lowering.h): StableHLOEmitter builds the module text, this
     * compiles it, and execute() above runs the result. It exists here rather
     * than on PjrtClient's caller because the plugin, the client and the
     * chosen device already live inside this runtime, and duplicating that
     * selection elsewhere would make it possible for two parts of the process
     * to disagree about which device they are on.
     *
     * @param module_text StableHLO in textual (MLIR) form
     * @param error Set to a diagnostic when this returns nullptr
     * @return A `PJRT_LoadedExecutable*` as an opaque handle, suitable as
     *         execute()'s `executable`, or nullptr on failure — including
     *         when PJRT device execution is not active, which is reported
     *         rather than silently substituting the LLVM-direct path
     */
    void* compileStableHLO(const std::string& module_text, std::string* error);

    /**
     * Release an executable returned by compileStableHLO().
     * Safe to call with nullptr, and a no-op when PJRT is not active.
     */
    void releaseExecutable(void* executable);

    /**
     * Whether execute() will actually run on a PJRT device.
     *
     * getDescription() has always carried this information in prose; this is
     * the same fact as a predicate, so a caller can branch on it without
     * parsing a human-readable string.
     */
    bool isDeviceExecutionActive() const;

    /**
     * The PJRT status line: why the device is or is not active.
     * Empty when device execution was never requested.
     */
    std::string deviceStatus() const;

    /**
     * Execute asynchronously.
     * @param executable Compiled executable
     * @param inputs Input buffers
     * @param outputs Output buffers (pre-allocated)
     * @return Execution handle (use wait() to synchronize)
     */
    void* executeAsync(void* executable,
                       const std::vector<BufferDescriptor>& inputs,
                       std::vector<BufferDescriptor>& outputs);

    /**
     * Wait for async execution to complete.
     * @param handle Execution handle from executeAsync
     * @return Execution result
     */
    ExecutionResult wait(void* handle);

    // ===== Buffer Management =====

    /**
     * Allocate buffer on device.
     * @param shape Tensor shape
     * @param element_size Size of each element
     * @return Buffer descriptor
     */
    BufferDescriptor allocateDevice(const std::vector<int64_t>& shape,
                                     size_t element_size);

    /**
     * Transfer host buffer to device.
     * @param host_data Host data pointer
     * @param shape Tensor shape
     * @param element_size Size of each element
     * @return Device buffer descriptor
     */
    BufferDescriptor toDevice(void* host_data,
                               const std::vector<int64_t>& shape,
                               size_t element_size);

    /**
     * Transfer device buffer to host.
     * @param device_buffer Device buffer
     * @param host_data Host destination (must be pre-allocated)
     */
    void toHost(const BufferDescriptor& device_buffer, void* host_data);

    /**
     * Free a device buffer.
     * @param buffer Buffer to free
     */
    void freeBuffer(BufferDescriptor& buffer);

    // ===== Synchronization =====

    /**
     * Synchronize with device.
     * Waits for all pending operations to complete.
     */
    void synchronize();

    // ===== Diagnostics =====

    /**
     * Get memory usage statistics.
     * @param allocated_bytes Output: bytes currently allocated
     * @param peak_bytes Output: peak bytes allocated
     */
    void getMemoryStats(size_t& allocated_bytes, size_t& peak_bytes);

    /**
     * Get runtime description.
     *
     * This is how to tell whether execution is actually happening on a PJRT
     * device or still on the LLVM-direct CPU path: when PJRT is active the
     * string names the live platform and addressable device count instead of
     * just the compile target.
     *
     * @return Human-readable description
     */
    std::string getDescription() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Global runtime singleton for convenience.
 * Use for simple cases; create XLARuntime instances for advanced use.
 */
XLARuntime& getDefaultRuntime();

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_RUNTIME_H
