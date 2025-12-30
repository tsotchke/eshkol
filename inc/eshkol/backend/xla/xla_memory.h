/*
 * XLA Memory Integration for Eshkol
 *
 * Integrates Eshkol's OALR (Ownership-Aware Lexical Regions) memory
 * model with XLA's buffer management.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_XLA_MEMORY_H
#define ESHKOL_XLA_MEMORY_H

#include <cstdint>
#include <memory>
#include <vector>

// Forward declarations
namespace llvm {
    class Value;
}

namespace eshkol {

class CodegenContext;

namespace xla {

// Forward declarations
enum class Target;
struct TensorType;

/**
 * Memory strategy for XLA buffers
 */
enum class MemoryStrategy {
    ZERO_COPY,      // Use arena memory directly (CPU only)
    PINNED,         // Pin arena memory for efficient GPU transfer
    DEVICE_ALLOC,   // Allocate on device, explicit transfer
    UNIFIED         // Unified memory (CPU/GPU share address space)
};

/**
 * Buffer lifetime scope
 */
enum class BufferScope {
    REGION,         // Lives until region exit (OALR managed)
    GLOBAL,         // Lives until program exit
    TEMPORARY       // May be reused after operation completes
};

/**
 * XLAMemoryIntegration - Bridges OALR and XLA memory models
 *
 * This class provides seamless integration between Eshkol's arena-based
 * OALR memory management and XLA's buffer management. Key features:
 *
 * - Zero-copy buffer sharing for CPU target
 * - Efficient GPU transfer via pinned memory
 * - Scope-based deallocation aligned with OALR regions
 * - Memory reuse optimization
 */
class XLAMemoryIntegration {
public:
    explicit XLAMemoryIntegration(CodegenContext& ctx);
    ~XLAMemoryIntegration();

    // Non-copyable
    XLAMemoryIntegration(const XLAMemoryIntegration&) = delete;
    XLAMemoryIntegration& operator=(const XLAMemoryIntegration&) = delete;

    // ===== Buffer Wrapping =====

    /**
     * Wrap an existing arena tensor for XLA use.
     * For CPU: Zero-copy, uses tensor's memory directly.
     * For GPU: Creates pinned buffer or transfers to device.
     *
     * @param arena_ptr LLVM Value for arena pointer
     * @param tensor_ptr LLVM Value for tensor struct pointer
     * @param target Target backend
     * @return LLVM Value for XLA buffer
     */
    llvm::Value* wrapArenaTensor(llvm::Value* arena_ptr,
                                  llvm::Value* tensor_ptr,
                                  Target target);

    /**
     * Create XLA buffer from raw data pointer.
     * @param data_ptr LLVM Value for data pointer
     * @param shape Tensor shape
     * @param element_size Size of each element
     * @param target Target backend
     * @return LLVM Value for XLA buffer
     */
    llvm::Value* wrapRawBuffer(llvm::Value* data_ptr,
                                const std::vector<int64_t>& shape,
                                size_t element_size,
                                Target target);

    // ===== Buffer Allocation =====

    /**
     * Allocate XLA buffer in arena with proper alignment.
     * @param arena_ptr LLVM Value for arena pointer
     * @param type Tensor type specification
     * @param scope Buffer lifetime scope
     * @return LLVM Value for tensor struct
     */
    llvm::Value* allocateXLABuffer(llvm::Value* arena_ptr,
                                    const TensorType& type,
                                    BufferScope scope = BufferScope::REGION);

    /**
     * Allocate aligned buffer with custom alignment.
     * @param arena_ptr Arena pointer
     * @param size_bytes Size in bytes
     * @param alignment Alignment (default 64 for AVX-512/XLA)
     * @return Aligned buffer pointer
     */
    llvm::Value* allocateAligned(llvm::Value* arena_ptr,
                                  size_t size_bytes,
                                  size_t alignment = 64);

    // ===== Device Transfer =====

    /**
     * Ensure tensor is on the specified device.
     * No-op if already on target, transfers if needed.
     * @param tensor_ptr Tensor pointer
     * @param target Target device
     * @return Device tensor value (may be same as input for CPU)
     */
    llvm::Value* ensureDevice(llvm::Value* tensor_ptr, Target target);

    /**
     * Synchronize tensor from device to host.
     * Updates arena buffer with device contents.
     * @param tensor_ptr Tensor pointer
     */
    void syncToHost(llvm::Value* tensor_ptr);

    // ===== Region Integration =====

    /**
     * Register buffer with OALR region for cleanup.
     * GPU buffers get cleanup callbacks, CPU buffers rely on arena.
     * @param buffer Buffer to register
     * @param region_ptr Region pointer
     * @param target Target backend
     */
    void registerWithRegion(llvm::Value* buffer,
                             llvm::Value* region_ptr,
                             Target target);

    /**
     * Begin XLA computation scope.
     * Enables memory reuse optimization within scope.
     * @param region_ptr Current OALR region
     */
    void beginComputationScope(llvm::Value* region_ptr);

    /**
     * End XLA computation scope.
     * Releases temporary buffers.
     */
    void endComputationScope();

    // ===== Memory Strategy =====

    /**
     * Set memory strategy for target.
     * @param target Target backend
     * @param strategy Memory strategy
     */
    void setStrategy(Target target, MemoryStrategy strategy);

    /**
     * Get current memory strategy.
     * @param target Target backend
     * @return Current strategy
     */
    MemoryStrategy getStrategy(Target target) const;

    /**
     * Get recommended strategy for target on this system.
     * @param target Target backend
     * @return Recommended strategy
     */
    static MemoryStrategy recommendedStrategy(Target target);

    // ===== Diagnostics =====

    /**
     * Get memory statistics.
     * @param arena_allocated Bytes allocated in arena for XLA
     * @param device_allocated Bytes allocated on device
     * @param transfers Number of host-device transfers
     */
    void getStats(size_t& arena_allocated,
                   size_t& device_allocated,
                   size_t& transfers) const;

    /**
     * Reset statistics counters.
     */
    void resetStats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_MEMORY_H
