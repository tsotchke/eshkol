/*
 * StableHLO Operation Emitter for Eshkol
 *
 * Translates Eshkol tensor operations to StableHLO operations
 * for compilation via XLA.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_STABLEHLO_EMITTER_H
#define ESHKOL_STABLEHLO_EMITTER_H

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace eshkol {
namespace xla {

// Forward declarations for MLIR types (will be typedef'd when MLIR is available)
struct MLIRValue;
struct MLIRType;
struct MLIRContext;

/**
 * StableHLO operation types supported by the emitter
 */
enum class StableHLOOp {
    // Arithmetic
    ADD,       // Element-wise addition
    SUBTRACT,  // Element-wise subtraction
    MULTIPLY,  // Element-wise multiplication
    DIVIDE,    // Element-wise division
    NEGATE,    // Element-wise negation
    ABS,       // Element-wise absolute value

    // Transcendental
    EXP,       // Element-wise exponential
    LOG,       // Element-wise natural logarithm
    SIN,       // Element-wise sine
    COS,       // Element-wise cosine
    TANH,      // Element-wise hyperbolic tangent

    // Matrix
    DOT_GENERAL, // General matrix/batched contraction
    TRANSPOSE,   // Permute tensor dimensions

    // Shape
    RESHAPE,     // Change tensor shape without changing data
    BROADCAST,   // Expand a tensor along new/size-1 dimensions
    SLICE,       // Extract a sub-tensor
    CONCATENATE, // Join tensors along an axis

    // Reduction
    REDUCE_SUM,  // Sum-reduce along axes
    REDUCE_MAX,  // Max-reduce along axes
    REDUCE_MIN,  // Min-reduce along axes
    REDUCE_PROD, // Product-reduce along axes

    // Activation
    RELU,        // Rectified linear unit
    SIGMOID,     // Logistic sigmoid
    SOFTMAX      // Softmax normalization
};

/**
 * Dot dimension specification for DOT_GENERAL
 */
struct DotDimensionNumbers {
    std::vector<int64_t> lhs_batching_dims;      // Batch dimensions of the LHS operand
    std::vector<int64_t> rhs_batching_dims;      // Batch dimensions of the RHS operand
    std::vector<int64_t> lhs_contracting_dims;   // Contracted (summed-over) dimensions of the LHS operand
    std::vector<int64_t> rhs_contracting_dims;   // Contracted (summed-over) dimensions of the RHS operand
};

/**
 * StableHLOEmitter - Emits StableHLO operations for XLA compilation
 *
 * This class is responsible for translating Eshkol's tensor operations
 * into StableHLO IR that can be compiled by XLA.
 */
class StableHLOEmitter {
public:
    /**
     * Construct an emitter with an empty StableHLO module.
     */
    StableHLOEmitter();

    /**
     * Destroy the emitter and release any owned MLIR resources.
     */
    ~StableHLOEmitter();

    // Non-copyable
    StableHLOEmitter(const StableHLOEmitter&) = delete;
    StableHLOEmitter& operator=(const StableHLOEmitter&) = delete;

    /**
     * Check if the StableHLO emitter has a real MLIR backend.
     * @return true if MLIR+StableHLO dialects are loaded and ready
     */
    bool isAvailable() const;

    // ===== Arithmetic Operations =====

    /**
     * Emit element-wise addition.
     * @param lhs Left operand
     * @param rhs Right operand
     * @return Result value
     */
    void* emitAdd(void* lhs, void* rhs);

    /**
     * Emit element-wise subtraction.
     */
    void* emitSubtract(void* lhs, void* rhs);

    /**
     * Emit element-wise multiplication.
     */
    void* emitMultiply(void* lhs, void* rhs);

    /**
     * Emit element-wise division.
     */
    void* emitDivide(void* lhs, void* rhs);

    // ===== Matrix Operations =====

    /**
     * Emit matrix multiplication (DOT_GENERAL).
     * @param lhs Left matrix
     * @param rhs Right matrix
     * @param dims Dimension specification
     * @return Result matrix
     */
    void* emitMatmul(void* lhs, void* rhs, const DotDimensionNumbers& dims);

    /**
     * Emit matrix transpose.
     * @param input Input matrix
     * @param permutation Dimension permutation
     * @return Transposed matrix
     */
    void* emitTranspose(void* input, const std::vector<int64_t>& permutation);

    // ===== Transcendental Operations =====

    /**
     * Emit element-wise exponential.
     * @param input Input value
     * @return Result value
     */
    void* emitExp(void* input);

    /**
     * Emit element-wise natural logarithm.
     * @param input Input value
     * @return Result value
     */
    void* emitLog(void* input);

    /**
     * Emit element-wise sine.
     * @param input Input value
     * @return Result value
     */
    void* emitSin(void* input);

    /**
     * Emit element-wise cosine.
     * @param input Input value
     * @return Result value
     */
    void* emitCos(void* input);

    /**
     * Emit element-wise hyperbolic tangent.
     * @param input Input value
     * @return Result value
     */
    void* emitTanh(void* input);

    // ===== Reduction Operations =====

    /**
     * Emit reduction along axes.
     * @param input Input tensor
     * @param axes Axes to reduce
     * @param op Reduction operation
     * @return Reduced tensor
     */
    void* emitReduce(void* input, const std::vector<int64_t>& axes, StableHLOOp op);

    // ===== Shape Operations =====

    /**
     * Emit a reshape to a new shape (same element count).
     * @param input Input tensor
     * @param new_shape Target shape
     * @return Reshaped tensor
     */
    void* emitReshape(void* input, const std::vector<int64_t>& new_shape);

    /**
     * Emit a broadcast along the given dimensions.
     * @param input Input tensor
     * @param broadcast_dims Mapping of input dimensions to output dimensions
     * @return Broadcasted tensor
     */
    void* emitBroadcast(void* input, const std::vector<int64_t>& broadcast_dims);

    /**
     * Emit a strided slice of a tensor.
     * @param input Input tensor
     * @param start Start indices per dimension
     * @param limit End indices per dimension
     * @param strides Step sizes per dimension
     * @return Sliced tensor
     */
    void* emitSlice(void* input, const std::vector<int64_t>& start,
                    const std::vector<int64_t>& limit,
                    const std::vector<int64_t>& strides);

    // ===== Module Management =====

    /**
     * Get the MLIR module containing all emitted operations.
     * @return MLIR module pointer
     */
    void* getModule() const;

    /**
     * Serialize module to string for debugging.
     * @return StableHLO IR as string
     */
    std::string serializeToString() const;

    /**
     * Reset emitter for a new computation.
     */
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_STABLEHLO_EMITTER_H
