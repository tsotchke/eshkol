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

#include <cstddef>
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

/** Semantic type for dense host buffers passed to typed PJRT execution. */
enum class BufferElementType : uint8_t { kF64, kF32, kBF16 };

/** Explicitly typed dense row-major host buffer. Type is never inferred from
 * element_size, so same-width integer buffers cannot be mistaken for floats. */
struct TypedBufferDescriptor {
    void* data;
    std::vector<int64_t> shape;
    BufferElementType element_type;
    bool on_device;
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
     * in), also attempts to select a PJRT device for executeStableHLO() — see
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
     * `executable` is a raw `void(const void* const*, void* const*)` function
     * pointer produced by the LLVM-direct compiler. This entry point always
     * keeps that contract, regardless of PJRT environment settings.
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
     * Compile and execute StableHLO on the active PJRT device. Input and
     * output descriptors must name host-resident, dense row-major f64 buffers
     * (`on_device == false`, `element_size == sizeof(double)`). The PJRT
     * executable stays inside this call, so it cannot be confused with the
     * raw function pointer accepted by execute(). Returns a diagnostic and
     * performs no host fallback when PJRT is inactive or compilation fails.
     * Use the typed overload below for f32 and bf16 buffers.
     */
    ExecutionResult executeStableHLO(const std::string& module_text,
                                     const std::vector<BufferDescriptor>& inputs,
                                     std::vector<BufferDescriptor>& outputs);

    /** Typed host-buffer overload; currently supports f64, f32 and bf16. */
    ExecutionResult executeStableHLO(const std::string& module_text,
                                     const std::vector<TypedBufferDescriptor>& inputs,
                                     std::vector<TypedBufferDescriptor>& outputs);

    /** Compile one explicitly reusable StableHLO program for this runtime's
     * active PJRT session. On failure returns nullptr and stores a diagnostic.
     * The program retains its session's client and plugin after this runtime
     * is destroyed, until the program itself is released. Calls that overlap
     * initialize(), or concurrent use of one runtime/program, require caller
     * serialization. */
    std::unique_ptr<class StableHLOProgram> compileStableHLO(
        const std::string& module_text, std::string* diagnostic = nullptr);

    /** Execute a compiled program only through the exact session that compiled
     * it. Calls overlapping initialize() require caller serialization. */
    ExecutionResult executeCompiledStableHLO(
        const class StableHLOProgram& program,
        const std::vector<BufferDescriptor>& inputs,
        std::vector<BufferDescriptor>& outputs);

    /** Typed host-buffer overload preserving the same program/session rules. */
    ExecutionResult executeCompiledStableHLO(
        const class StableHLOProgram& program,
        const std::vector<TypedBufferDescriptor>& inputs,
        std::vector<TypedBufferDescriptor>& outputs);

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
     * Get runtime description, including whether executeStableHLO() has an
     * active PJRT device and its reported platform.
     *
     * @return Human-readable description
     */
    std::string getDescription() const;

private:
    class Impl;
    ExecutionResult executePjrt(void* executable, const void* session,
                                const std::vector<BufferDescriptor>& inputs,
                                std::vector<BufferDescriptor>& outputs);
    ExecutionResult executePjrt(void* executable, const void* session,
                                const std::vector<TypedBufferDescriptor>& inputs,
                                std::vector<TypedBufferDescriptor>& outputs);
    std::unique_ptr<Impl> impl_;
};

/** Opaque, noncopyable owner of one PJRT executable and its originating session. */
class StableHLOProgram {
public:
    ~StableHLOProgram();
    StableHLOProgram(const StableHLOProgram&) = delete;
    StableHLOProgram& operator=(const StableHLOProgram&) = delete;
    StableHLOProgram(StableHLOProgram&&) noexcept;
    StableHLOProgram& operator=(StableHLOProgram&&) noexcept;
private:
    class Impl;
    explicit StableHLOProgram(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class XLARuntime;
};

/**
 * Global runtime singleton for convenience.
 * Use for simple cases; create XLARuntime instances for advanced use.
 */
XLARuntime& getDefaultRuntime();

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_RUNTIME_H
