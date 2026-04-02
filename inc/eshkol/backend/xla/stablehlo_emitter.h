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
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    NEGATE,
    ABS,

    // Transcendental
    EXP,
    LOG,
    SIN,
    COS,
    TANH,

    // Matrix
    DOT_GENERAL,
    TRANSPOSE,

    // Shape
    RESHAPE,
    BROADCAST,
    SLICE,
    CONCATENATE,

    // Reduction
    REDUCE_SUM,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_PROD,

    // Activation
    RELU,
    SIGMOID,
    SOFTMAX
};

/**
 * Dot dimension specification for DOT_GENERAL
 */
struct DotDimensionNumbers {
    std::vector<int64_t> lhs_batching_dims;
    std::vector<int64_t> rhs_batching_dims;
    std::vector<int64_t> lhs_contracting_dims;
    std::vector<int64_t> rhs_contracting_dims;
};

/**
 * StableHLOEmitter - Emits StableHLO operations for XLA compilation
 *
 * This class is responsible for translating Eshkol's tensor operations
 * into StableHLO IR that can be compiled by XLA.
 */
class StableHLOEmitter {
public:
    StableHLOEmitter();
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

    void* emitExp(void* input);
    void* emitLog(void* input);
    void* emitSin(void* input);
    void* emitCos(void* input);
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

    void* emitReshape(void* input, const std::vector<int64_t>& new_shape);
    void* emitBroadcast(void* input, const std::vector<int64_t>& broadcast_dims);
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
