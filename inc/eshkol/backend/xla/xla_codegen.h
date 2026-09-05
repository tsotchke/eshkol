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

#include "eshkol/backend/xla/stablehlo_emitter.h"

// Forward declarations
namespace llvm {
    class Value;
}

namespace eshkol {

class CodegenContext;

namespace xla {

/**
 * Runtime-configurable threshold (default: 100000 elements = ~320x320 matrix).
 *
 * XLA is reserved for massive tensors only to amortize compilation overhead.
 * Dispatch hierarchy: XLA (>=100K) -> cBLAS (>=4K) -> SIMD (>=64) -> scalar.
 * Override via the ESHKOL_XLA_THRESHOLD environment variable.
 */
extern size_t g_xla_threshold;

/**
 * Set the global XLA dispatch threshold (in elements).
 * @param threshold Minimum element count for XLA to be used
 */
void xla_set_threshold(size_t threshold);

/**
 * Get the current global XLA dispatch threshold (in elements).
 * @return Minimum element count for XLA to be used
 */
size_t xla_get_threshold();

/**
 * Element-wise operation types supported by the XLA backend.
 */
enum class ElementwiseOp {
    ADD,     // Element-wise addition
    SUB,     // Element-wise subtraction
    MUL,     // Element-wise multiplication
    DIV,     // Element-wise division
    EXP,     // Element-wise exponential
    LOG,     // Element-wise natural logarithm
    SIN,     // Element-wise sine
    COS,     // Element-wise cosine
    TANH,    // Element-wise hyperbolic tangent
    RELU,    // Rectified linear unit
    SIGMOID  // Logistic sigmoid
};

/**
 * Reduction operation types supported by the XLA backend.
 */
enum class ReduceOp {
    SUM,   // Sum-reduce
    MEAN,  // Mean-reduce
    MAX,   // Max-reduce
    MIN,   // Min-reduce
    PROD   // Product-reduce
};

/**
 * Target backend for XLA compilation.
 */
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
    /**
     * Construct an XLA codegen instance bound to a codegen context.
     * @param ctx Codegen context to integrate with (owns LLVM module/builder state)
     */
    explicit XLACodegen(CodegenContext& ctx);

    /**
     * Destroy the codegen instance and release any owned XLA resources.
     */
    ~XLACodegen();

    // Non-copyable
    XLACodegen(const XLACodegen&) = delete;
    XLACodegen& operator=(const XLACodegen&) = delete;

    // Movable
    /**
     * Move-construct, transferring ownership of the underlying XLA state.
     */
    XLACodegen(XLACodegen&&) noexcept;

    /**
     * Move-assign, transferring ownership of the underlying XLA state.
     */
    XLACodegen& operator=(XLACodegen&&) noexcept;

    // ===== Backend Selection =====

    /**
     * Set the threshold for using XLA vs BLAS/SIMD.
     * Tensors with fewer elements than this will use BLAS/SIMD.
     * @param min_elements Minimum elements to use XLA (default: 100000)
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

    /**
     * Emit XLA-accelerated tensor transpose.
     * For 2D tensors, swaps rows and columns (permutation [1,0]).
     * @param input Input tensor (struct pointer)
     * @return Transposed tensor (struct pointer)
     */
    llvm::Value* emitTranspose(llvm::Value* input);

    /**
     * Emit XLA broadcast from src_shape to tgt_shape.
     * @param input Input tensor
     * @param tgt_shape Target shape values (LLVM i64 constants)
     * @param tgt_rank Target rank
     * @return Broadcasted tensor
     */
    llvm::Value* emitBroadcast(llvm::Value* input,
                                const std::vector<llvm::Value*>& tgt_shape,
                                int64_t tgt_rank);

    /**
     * Emit XLA tensor slice.
     * @param input Input tensor
     * @param starts Start indices per dimension
     * @param limits End indices per dimension
     * @param strides Step sizes per dimension (nullptr for all-1s)
     * @return Sliced tensor
     */
    llvm::Value* emitSlice(llvm::Value* input,
                            const std::vector<llvm::Value*>& starts,
                            const std::vector<llvm::Value*>& limits,
                            const std::vector<llvm::Value*>& strides);

    // ===== Autodiff Integration =====

    /**
     * Emit the HOST-side matmul backward pass: dC/dA = grad @ B^T,
     * dC/dB = A^T @ grad.
     *
     * This emits LLVM IR calling the `eshkol_xla_*` C runtime, so the
     * arithmetic runs on the host (BLAS/SIMD/GPU dispatch inside the
     * runtime). No StableHLO is produced and XLA never sees this
     * computation. The device path is emitDeviceGradient() below.
     *
     * Two limitations a caller must respect:
     *  - the forward op is assumed to be a matmul, and `output_node` is the
     *    upstream cotangent, not the forward output;
     *  - emitTranspose() reverses ALL dimensions, so these two transposes are
     *    the correct adjoints only for rank-2 operands. Batched (rank > 2)
     *    matmul is NOT handled correctly here.
     *
     * @param output_node Upstream gradient (cotangent) tensor
     * @param wrt_vars [A, B] forward operands
     * @return Pointer to 2-element array [grad_A, grad_B], or nullptr if the
     *         backend is unavailable, `wrt_vars` is not size 2, or any
     *         intermediate op could not be emitted
     */
    llvm::Value* emitGradient(llvm::Value* output_node,
                               const std::vector<llvm::Value*>& wrt_vars);

    /**
     * Emit the HOST-side elementwise backward pass via the chain rule.
     *
     * Like emitGradient(), this emits LLVM IR calling the `eshkol_xla_*` C
     * runtime; it is not a device computation.
     *
     * Every sign the derivative calls for is applied here — the caller is
     * never expected to fix one up afterwards. RELU is deliberately NOT
     * supported: its adjoint needs the mask (a > 0), and the elementwise
     * runtime op set has no comparison or select, so this returns nullptr
     * for RELU rather than an expression that is the right shape and the
     * wrong value.
     *
     * @param grad Upstream gradient (cotangent)
     * @param a Forward left operand
     * @param b Forward right operand (nullptr for unary)
     * @param result Forward output
     * @param op The elementwise operation
     * @return Pointer to a 2-element array [grad_a, grad_b] (grad_b is a null
     *         pointer for unary ops), or nullptr if the backend is
     *         unavailable, the op has no exact rule here (RELU), or any
     *         required gradient could not be emitted
     */
    llvm::Value* emitElementwiseGradient(llvm::Value* grad,
                                          llvm::Value* a,
                                          llvm::Value* b,
                                          llvm::Value* result,
                                          ElementwiseOp op);

    /**
     * Emit reduce gradient (broadcast upstream grad back to input shape).
     * @param grad Upstream gradient
     * @param input Original input tensor (for shape)
     * @param axis Reduction axis (-1 for all)
     * @param op Reduce operation
     * @return Gradient tensor with input shape
     */
    llvm::Value* emitReduceGradient(llvm::Value* grad,
                                     llvm::Value* input,
                                     int64_t axis,
                                     ReduceOp op);

    /**
     * Emit transpose gradient (transpose with inverse permutation).
     * @param grad Upstream gradient
     * @return Transposed gradient
     */
    llvm::Value* emitTransposeGradient(llvm::Value* grad);

    // ===== Device-Side Gradients (StableHLO reverse mode) =====

    /**
     * Check whether this build can emit device-side gradients at all.
     * @return true if the StableHLO emitter behind this codegen has a real
     *         MLIR/StableHLO backend; false in an LLVM-only build, where
     *         every device-gradient entry point below refuses with a
     *         diagnostic instead of emitting anything
     */
    bool hasDeviceGradientSupport() const;

    /**
     * Access the StableHLO emitter that owns this codegen's device graph.
     *
     * The forward pass and its gradient MUST be emitted through this one
     * instance. emitVJP() differentiates by walking the use-def chains of ops
     * that already exist in the emitter's module, so a forward graph built on
     * some other emitter is invisible to the backward sweep and would yield
     * shape-correct ZERO gradients rather than an error. Owning a single
     * emitter here, and handing that same one out, is what turns "forward and
     * backward live in the same module" from a convention into something the
     * type system can enforce.
     *
     * @return The shared StableHLO emitter (never null; may be unavailable,
     *         see isAvailable() on it)
     */
    StableHLOEmitter& stablehloEmitter();

    /**
     * Emit the reverse-mode gradient of a device graph built through
     * stablehloEmitter().
     *
     * This is the device training path: unlike emitGradient(), which emits
     * host runtime calls, this routes to StableHLOEmitter::emitVJP() and so
     * the gradient is StableHLO IR sitting in the same module as the forward
     * pass, ready to be lowered by compileDeviceGradient().
     *
     * `output`, `wrt` and `seed` are opaque emitter value handles as returned
     * by the stablehloEmitter() emit* methods — NOT llvm::Value*. That is the
     * type boundary: the LLVM domain (host tensor structs, runtime calls) and
     * the MLIR domain (a device graph over statically shaped tensors) do not
     * meet inside this function, and nothing here casts one to the other.
     *
     * Fails closed, exactly as emitVJP() does: on any failure the returned
     * VJPResult has `complete == false`, an EMPTY gradient vector, and a
     * non-empty diagnostic. The post-conditions of emitVJP are re-checked
     * here (one gradient per `wrt` entry, no null handles) so that a caller
     * cannot be handed a partial gradient set by a future emitter change.
     *
     * @param output Emitter handle for the value to differentiate (typically
     *               a scalar loss)
     * @param wrt Emitter handles for the parameters to differentiate against
     * @param seed Emitter handle for the cotangent of `output`; nullptr means
     *             ones_like(output), correct for a scalar loss
     * @return VJPResult; check `complete` before touching `gradients`
     */
    VJPResult emitDeviceGradient(void* output,
                                 const std::vector<void*>& wrt,
                                 void* seed = nullptr);

    /**
     * Outcome of lowering the device graph (forward pass plus the gradients
     * emitted into it) for a target.
     */
    struct DeviceGradientModule {
        bool ready = false;          // True only if a lowered executable came back
        void* executable = nullptr;  // llvm::Module* from XLACompiler; nullptr unless ready
        std::string entry_symbol;    // Name of the entry function inside it; empty unless ready
        std::string diagnostic;      // Why it is not ready; empty on success
    };

    /**
     * Lower the StableHLO module held by stablehloEmitter() — forward graph
     * and the gradients emitted into it by emitDeviceGradient() — to `target`.
     *
     * REFUSES, rather than compiling something plausible, when the module is
     * not a lowerable program: today the emitter appends its ops directly to
     * the module body and offers no way to create graph parameters or an
     * entry function, so there is nothing to bind runtime buffers to. That
     * case returns ready == false with a diagnostic naming the offending op.
     *
     * Note this is NOT compile(): compile() lowers this codegen's own MLIR
     * module (the one built by the internal buildXFunc helpers), which is a
     * different module from the emitter's.
     *
     * The MLIR pass pipeline rewrites the module in place, so the emitter is
     * reset() after any lowering attempt, successful or not: every value
     * handle previously returned by the emitter is invalid afterwards.
     *
     * @param target Target backend to lower for
     * @return DeviceGradientModule; check `ready` before using `executable`
     */
    DeviceGradientModule compileDeviceGradient(Target target);

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
