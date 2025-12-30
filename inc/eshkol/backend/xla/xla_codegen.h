/*
 * XLA Backend Codegen for Eshkol
 *
 * Provides accelerated tensor operations via XLA/StableHLO for large tensors.
 * Falls back to SIMD implementation for small tensors (below threshold).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_XLA_CODEGEN_H
#define ESHKOL_XLA_CODEGEN_H

#include <cstddef>
#include <memory>
#include <vector>
#include <string>

// Forward declarations
namespace llvm {
    class Value;
}

namespace eshkol {

class CodegenContext;

namespace xla {

// Runtime-configurable threshold (default: 1000)
// Override via ESHKOL_XLA_THRESHOLD environment variable
extern size_t g_xla_threshold;

// API for runtime configuration
void xla_set_threshold(size_t threshold);
size_t xla_get_threshold();

// Element-wise operation types
enum class ElementwiseOp {
    ADD,
    SUB,
    MUL,
    DIV,
    EXP,
    LOG,
    SIN,
    COS,
    TANH,
    RELU,
    SIGMOID
};

// Reduce operation types
enum class ReduceOp {
    SUM,
    MEAN,
    MAX,
    MIN,
    PROD
};

// Target backend for compilation
enum class Target {
    CPU,      // XLA CPU backend
    CUDA,     // NVIDIA GPU via CUDA
    Metal,    // Apple GPU via Metal
    Vulkan    // Cross-platform GPU via Vulkan
};

/**
 * XLACodegen - Main XLA backend class
 *
 * Integrates with the existing TensorCodegen to provide XLA-accelerated
 * tensor operations for large tensors while maintaining compatibility
 * with the SIMD path for small tensors.
 *
 * Thread-safe: Can be used from multiple threads concurrently.
 */
class XLACodegen {
public:
    explicit XLACodegen(CodegenContext& ctx);
    ~XLACodegen();

    // Non-copyable
    XLACodegen(const XLACodegen&) = delete;
    XLACodegen& operator=(const XLACodegen&) = delete;

    // Movable
    XLACodegen(XLACodegen&&) noexcept;
    XLACodegen& operator=(XLACodegen&&) noexcept;

    // ===== Backend Selection =====

    /**
     * Set the threshold for using XLA vs SIMD.
     * Tensors with fewer elements than this will use SIMD.
     * @param min_elements Minimum elements to use XLA (default: 1000)
     */
    void setThreshold(size_t min_elements);

    /**
     * Check if XLA should be used for an operation.
     * @param num_elements Number of elements in the operation
     * @return true if XLA should be used, false for SIMD
     */
    bool shouldUseXLA(size_t num_elements) const;

    // ===== Tensor Operations =====

    /**
     * Emit XLA-accelerated matrix multiplication.
     * @param a Left operand tensor
     * @param b Right operand tensor
     * @return Result tensor value
     */
    llvm::Value* emitMatmul(llvm::Value* a, llvm::Value* b);

    /**
     * Emit XLA-accelerated element-wise operation.
     * @param a First operand tensor
     * @param b Second operand tensor (nullptr for unary ops)
     * @param op Element-wise operation type
     * @return Result tensor value
     */
    llvm::Value* emitElementwise(llvm::Value* a, llvm::Value* b, ElementwiseOp op);

    /**
     * Emit XLA-accelerated reduction.
     * @param input Input tensor
     * @param axis Axis to reduce along (-1 for all axes)
     * @param op Reduction operation type
     * @return Result tensor value
     */
    llvm::Value* emitReduce(llvm::Value* input, int64_t axis, ReduceOp op);

    // ===== Autodiff Integration =====

    /**
     * Emit gradient computation using XLA's autodiff.
     * @param output_node Output value to differentiate
     * @param wrt_vars Variables to compute gradients for
     * @return Gradient tensor value
     */
    llvm::Value* emitGradient(llvm::Value* output_node,
                               const std::vector<llvm::Value*>& wrt_vars);

    // ===== Compilation =====

    /**
     * Compile accumulated XLA computations to target.
     * @param target Target backend (CPU, CUDA, Metal, Vulkan)
     */
    void compile(Target target);

    /**
     * Get compiled executable.
     * @return Pointer to executable (type depends on target)
     */
    void* getExecutable() const;

    // ===== Memory Integration =====

    /**
     * Wrap an arena buffer for XLA use.
     * Zero-copy for CPU target, pinned for GPU.
     * @param arena_ptr Arena pointer
     * @param tensor_ptr Tensor pointer
     * @return XLA buffer value
     */
    llvm::Value* wrapArenaBuffer(llvm::Value* arena_ptr, llvm::Value* tensor_ptr);

    /**
     * Ensure tensor is on the specified device.
     * No-op for CPU, transfers for GPU.
     * @param tensor_ptr Tensor pointer
     * @param target Target device
     * @return Device tensor value
     */
    llvm::Value* ensureDevice(llvm::Value* tensor_ptr, Target target);

    // ===== Status =====

    /**
     * Check if XLA backend is available.
     * @return true if XLA is initialized and ready
     */
    bool isAvailable() const;

    /**
     * Get description of XLA configuration.
     * @return Human-readable configuration string
     */
    std::string getDescription() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_CODEGEN_H
