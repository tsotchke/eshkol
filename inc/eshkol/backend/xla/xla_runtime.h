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
 * Buffer descriptor for XLA execution
 */
struct BufferDescriptor {
    void* data;                          // Pointer to data
    std::vector<int64_t> shape;          // Tensor shape
    size_t element_size;                 // Size of each element (e.g., 8 for double)
    bool on_device;                      // True if buffer is on GPU
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
