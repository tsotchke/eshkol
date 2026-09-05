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

#include "eshkol/backend/xla/xla_types.h"

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
    /** Element-wise division. */
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
    /** Expand a tensor along new/size-1 dimensions. */
    BROADCAST,   // Expand a tensor along new/size-1 dimensions
    SLICE,       // Extract a sub-tensor
    /** Join tensors along an axis. */
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
 * Gather dimension specification for GATHER (embedding lookup).
 * Note: StableHLO's on-the-wire GatherDimensionNumbers attribute also
 * carries operand_batching_dims/start_indices_batching_dims for the
 * batched-gather extension; Eshkol never emits batched gather, so
 * emitGather() always passes those two as empty.
 */
struct GatherDimensionNumbers {
    std::vector<int64_t> offset_dims;          // Output dims that index into the gathered slice
    std::vector<int64_t> collapsed_slice_dims; // Operand dims sliced away (size 1) and dropped from the result
    std::vector<int64_t> start_index_map;      // Maps each index-vector component to an operand dimension
    int64_t index_vector_dim = 0;              // Which start_indices dimension holds the index vector
};

/**
 * Scatter dimension specification for SCATTER (embedding gradient / scatter-add).
 * Note: as with GatherDimensionNumbers, the batched-scatter fields
 * (input_batching_dims/scatter_indices_batching_dims) are always empty.
 */
struct ScatterDimensionNumbers {
    std::vector<int64_t> update_window_dims;          // Update dims that index into the update window
    std::vector<int64_t> inserted_window_dims;        // Operand dims not present in the update window
    std::vector<int64_t> scatter_dims_to_operand_dims; // Maps each index-vector component to an operand dimension
    int64_t index_vector_dim = 0;                      // Which scatter_indices dimension holds the index vector
};

/**
 * Comparison direction for COMPARE.
 */
enum class ComparisonDirection {
    EQ,  // Equal
    NE,  // Not equal
    GE,  // Greater than or equal
    GT,  // Greater than
    LE,  // Less than or equal
    LT   // Less than
};

/**
 * Comparison element-type hint for COMPARE (mirrors StableHLO's
 * comparison_type attribute; NOTYPE lets StableHLO infer it from the
 * operand type).
 */
enum class ComparisonType {
    NOTYPE,      // Infer from operand type
    FLOAT,       // IEEE-754 float comparison
    TOTALORDER,  // Float comparison respecting a total order (NaN/signed-zero handling)
    SIGNED,      // Signed integer comparison
    UNSIGNED     // Unsigned integer comparison
};

/**
 * Result of one reverse-mode gradient (VJP) request.
 *
 * FAIL-CLOSED CONTRACT: `gradients` is populated ONLY when `complete` is
 * true. If any operation on the backward path has no VJP rule, or a shape
 * could not be reconciled, `gradients` is left EMPTY and `diagnostic` names
 * the failure. A partially-populated gradient vector is never returned: a
 * wrong gradient does not crash, it trains the model to garbage silently,
 * so there is no result a caller could misuse.
 */
struct VJPResult {
    std::vector<void*> gradients;  // One cotangent per `wrt` entry, in order; empty unless complete
    bool complete = false;         // True only if every op on the path had a VJP rule
    std::string diagnostic;        // Failure reason (op name / shape mismatch); empty on success
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

    // ===== Indexing Operations =====

    /**
     * Emit an embedding-table gather (GATHER).
     * @param operand Table tensor to gather rows/slices from
     * @param start_indices Index tensor selecting which slices to extract
     * @param dims Gather dimension numbers
     * @param slice_sizes Size of the slice extracted at each set of indices, one per operand dimension
     * @return Gathered tensor
     */
    void* emitGather(void* operand, void* start_indices, const GatherDimensionNumbers& dims,
                     const std::vector<int64_t>& slice_sizes);

    /**
     * Emit an embedding-gradient scatter-add (SCATTER). The update_computation
     * region is always a single add (accumulate), matching the
     * gradient-accumulation use case.
     * @param operand Tensor being scattered into
     * @param scatter_indices Index tensor selecting update positions
     * @param updates Values to accumulate into operand
     * @param dims Scatter dimension numbers
     * @return Updated tensor (same shape as operand)
     */
    void* emitScatter(void* operand, void* scatter_indices, void* updates,
                      const ScatterDimensionNumbers& dims);

    /**
     * Emit a dynamic (runtime-indexed) slice (DYNAMIC_SLICE).
     * @param operand Tensor to slice from
     * @param start_indices One scalar index tensor per operand dimension
     * @param slice_sizes Size of the extracted slice per dimension
     * @return Sliced tensor of shape slice_sizes
     */
    void* emitDynamicSlice(void* operand, const std::vector<void*>& start_indices,
                           const std::vector<int64_t>& slice_sizes);

    /**
     * Emit a dynamic (runtime-indexed) update-in-place (DYNAMIC_UPDATE_SLICE).
     * @param operand Tensor to write into
     * @param update Values to write
     * @param start_indices One scalar index tensor per operand dimension
     * @return Updated tensor (same shape as operand)
     */
    void* emitDynamicUpdateSlice(void* operand, void* update,
                                 const std::vector<void*>& start_indices);

    // ===== Elementwise Selection Operations =====

    /**
     * Emit an elementwise select (SELECT), e.g. for causal masking.
     * @param pred Boolean predicate tensor
     * @param on_true Values where pred is true
     * @param on_false Values where pred is false
     * @return Selected tensor (same type as on_true)
     */
    void* emitSelect(void* pred, void* on_true, void* on_false);

    /**
     * Emit an elementwise comparison (COMPARE), producing a boolean tensor.
     * @param lhs Left operand
     * @param rhs Right operand
     * @param direction Comparison direction
     * @param compare_type Comparison element-type hint (NOTYPE to infer)
     * @return Boolean (i1) result tensor
     */
    void* emitCompare(void* lhs, void* rhs, ComparisonDirection direction,
                      ComparisonType compare_type = ComparisonType::NOTYPE);

    // ===== Shape Construction Operations =====

    /**
     * Emit a concatenation along an axis (CONCATENATE).
     * @param inputs Tensors to join, in order
     * @param dimension Axis to concatenate along
     * @return Concatenated tensor
     */
    void* emitConcatenate(const std::vector<void*>& inputs, int64_t dimension);

    /**
     * Emit a pad (PAD): grows a tensor by constant amounts on each side of
     * each dimension, with optional interior (between-element) padding.
     * @param operand Tensor to pad
     * @param padding_value Scalar fill value
     * @param edge_padding_low Padding added before dimension 0..N per axis
     * @param edge_padding_high Padding added after dimension 0..N per axis
     * @param interior_padding Padding inserted between elements per axis
     * @return Padded tensor
     */
    void* emitPad(void* operand, void* padding_value,
                 const std::vector<int64_t>& edge_padding_low,
                 const std::vector<int64_t>& edge_padding_high,
                 const std::vector<int64_t>& interior_padding);

    /**
     * Emit an iota (IOTA): fills a tensor with increasing values along one
     * dimension, starting from zero.
     * @param shape Output tensor shape
     * @param iota_dimension Dimension along which values increase
     * @param elem Output element type (F16/F32/F64/BF16 only)
     * @return Iota tensor
     */
    void* emitIota(const std::vector<int64_t>& shape, int64_t iota_dimension, ElementType elem);

    // ===== Type Conversion Operations =====

    /**
     * Emit a dtype conversion (CONVERT).
     * @param input Tensor to convert
     * @param target Target element type (F16/F32/F64/BF16 only)
     * @return Converted tensor (same shape as input)
     */
    void* emitConvert(void* input, ElementType target);

    // ===== Constant / Shape Helpers =====

    /**
     * Emit a `stablehlo.broadcast_in_dim` with an explicit result shape.
     * Unlike emitBroadcast(), which can only produce an identity broadcast
     * (it reuses the input type as the result type), this one actually
     * changes shape, which is what a gradient needs when it un-reduces a
     * sum back over the axes it was reduced along.
     * @param input Tensor to broadcast
     * @param result_shape Shape of the broadcast result
     * @param broadcast_dims Result dimension each input dimension maps to
     *                       (one entry per input dimension; empty for a
     *                       rank-0 input, i.e. a splat)
     * @return Broadcast tensor, or nullptr if MLIR support isn't available
     */
    void* emitBroadcastInDim(void* input, const std::vector<int64_t>& result_shape,
                             const std::vector<int64_t>& broadcast_dims);

    /**
     * Emit a constant tensor of zeros with the same type as `value`.
     * @param value Tensor whose type is copied
     * @return Zero tensor, or nullptr on an unsupported element type
     */
    void* emitZerosLike(void* value);

    /**
     * Emit a constant tensor of ones with the same type as `value`.
     * Used as the default seed cotangent for a scalar loss.
     * @param value Tensor whose type is copied
     * @return Ones tensor, or nullptr on an unsupported element type
     */
    void* emitOnesLike(void* value);

    // ===== Reverse-Mode Gradients (VJP) =====

    /**
     * Emit the reverse-mode vector-Jacobian product of `output` with respect
     * to `wrt`, as StableHLO operations in the same module as the forward
     * pass.
     *
     * This is the training path. The forward emitters above already build an
     * SSA DAG, so no separate tape is recorded: the backward pass walks the
     * use-def chains of the emitted ops in reverse topological order and
     * emits a StableHLO op for each VJP rule. Contributions from multiple
     * consumers of the same value are summed with `stablehlo.add`.
     *
     * Broadcasting is handled where it actually occurs: the VJP of
     * `stablehlo.broadcast_in_dim` reduces the cotangent over exactly the
     * dimensions the operand did not span (and over the dimensions where the
     * operand had extent 1), so a [3] bias broadcast against a [2,3]
     * activation receives a [3] gradient, not a [2,3] one.
     *
     * @param output Value to differentiate (typically a scalar loss)
     * @param wrt Values to take the gradient with respect to (parameters);
     *            each must be a float-element tensor
     * @param seed Cotangent for `output`; nullptr means ones_like(output),
     *             which is the correct seed for a scalar loss
     * @return VJPResult; check `complete` before reading `gradients`
     */
    VJPResult emitVJP(void* output, const std::vector<void*>& wrt, void* seed = nullptr);

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
