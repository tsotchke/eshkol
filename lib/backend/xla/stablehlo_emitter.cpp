/*
 * StableHLO Operation Emitter Implementation for Eshkol
 *
 * Emits StableHLO operations when MLIR+StableHLO are available via the
 * ESHKOL_XLA_FULL_MLIR compilation path. All emit methods build real MLIR
 * StableHLO dialect operations (AddOp, DotGeneralOp, ReduceOp, etc.).
 *
 * When MLIR is not linked, all emit functions return nullptr, signaling
 * the LLVM-direct fallback path in xla_codegen.cpp should be used instead.
 *
 * void* convention: All parameters and return values are mlir::Value* cast
 * through void* to keep MLIR headers out of the public API. A value pool
 * inside Impl owns all returned Values until reset() is called.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/stablehlo_emitter.h"

// MLIR includes (conditional compilation)
#if defined(ESHKOL_MLIR_AVAILABLE) && defined(ESHKOL_STABLEHLO_AVAILABLE)
#define ESHKOL_XLA_FULL_MLIR 1
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>
#endif

namespace eshkol {
namespace xla {

// ===== StableHLOEmitter Implementation =====

class StableHLOEmitter::Impl {
public:
#ifdef ESHKOL_XLA_FULL_MLIR
    std::unique_ptr<mlir::MLIRContext> ctx_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;
    std::deque<mlir::Value> value_pool_;
    bool available_ = false;

    Impl() {
        ctx_ = std::make_unique<mlir::MLIRContext>();
        ctx_->loadDialect<mlir::func::FuncDialect>();
        ctx_->loadDialect<mlir::arith::ArithDialect>();
        ctx_->loadDialect<mlir::stablehlo::StablehloDialect>();
        builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
        module_ = mlir::ModuleOp::create(builder_->getUnknownLoc());
        // An OpBuilder constructed from a context alone has NO insertion
        // block, and OpBuilder::insert() silently does nothing in that state:
        // every op the emitters below create would be built, detached, and
        // leaked, so serializeToString() would print an empty module and the
        // backward pass would have no block to walk. Anchoring the builder in
        // the module body is the minimum that makes emitted ops reachable.
        // This is deliberately interim: StableHLO ops belong in a
        // func::FuncOp whose block arguments are the graph inputs, and
        // building that entry function is the job of the layer that knows the
        // computation signature (xla_codegen), not of this emitter.
        builder_->setInsertionPointToEnd(module_->getBody());
        available_ = true;
    }

    /// Store an mlir::Value in the pool, return void* to it.
    void* storeValue(mlir::Value v) {
        value_pool_.push_back(v);
        return static_cast<void*>(&value_pool_.back());
    }

    /// Dereference a void* back to mlir::Value.
    mlir::Value toValue(void* v) {
        return *static_cast<mlir::Value*>(v);
    }

    mlir::Location loc() { return builder_->getUnknownLoc(); }

    // ----- Reverse-mode gradient support -----
    // Declared here, defined in the "Reverse-Mode Gradients (VJP)" section
    // near the bottom of this file, where the design notes live.

    /// Forward value -> accumulated cotangent for one backward pass.
    using GradMap = llvm::DenseMap<mlir::Value, mlir::Value>;

    /// First failure encountered during a VJP walk; empty when the walk
    /// succeeded. Held on Impl rather than threaded through every rule so a
    /// rule can bail out with one `return false`.
    std::string vjp_diag_;

    // Constant / shape building blocks.
    mlir::Value constantFromAttr(mlir::Type elem_type, mlir::Attribute elem_attr);
    mlir::Value constantScalar(mlir::Type elem_type, double value);
    mlir::Value constantSplat(mlir::RankedTensorType type, double value);
    mlir::Value zerosLike(mlir::Value v);
    mlir::Value onesLike(mlir::Value v);

    // Elementwise building blocks (all assume equal operand shapes).
    mlir::Value addV(mlir::Value a, mlir::Value b);
    mlir::Value subV(mlir::Value a, mlir::Value b);
    mlir::Value mulV(mlir::Value a, mlir::Value b);
    mlir::Value divV(mlir::Value a, mlir::Value b);
    mlir::Value negate(mlir::Value v);

    // Shape building blocks.
    mlir::Value broadcastInDim(mlir::Value v, llvm::ArrayRef<int64_t> out_shape,
                               llvm::ArrayRef<int64_t> bcast_dims);
    mlir::Value reshapeTo(mlir::Value v, llvm::ArrayRef<int64_t> shape);
    mlir::Value transposeBy(mlir::Value v, llvm::ArrayRef<int64_t> perm);
    mlir::Value convertElem(mlir::Value v, mlir::Type elem_type);
    mlir::Value sliceOf(mlir::Value v, llvm::ArrayRef<int64_t> start,
                        llvm::ArrayRef<int64_t> limit, llvm::ArrayRef<int64_t> strides);

    // Structured ops shared by the forward emitters and the VJP rules.
    mlir::Value dotGeneral(mlir::Value lhs, mlir::Value rhs,
                           llvm::ArrayRef<int64_t> lhs_batch, llvm::ArrayRef<int64_t> rhs_batch,
                           llvm::ArrayRef<int64_t> lhs_contract, llvm::ArrayRef<int64_t> rhs_contract);
    mlir::Value reduceWithBody(mlir::Value v, llvm::ArrayRef<int64_t> axes, StableHLOOp op);
    mlir::Value scatterAdd(mlir::Value operand, mlir::Value indices, mlir::Value updates,
                           mlir::stablehlo::ScatterDimensionNumbersAttr dims);

    // Broadcast bookkeeping.
    mlir::Value reduceGradToShape(mlir::Value grad, mlir::RankedTensorType target);

    // Backward walk.
    static bool isFloatTensor(mlir::Value v);
    static bool isDifferentiableOperand(mlir::Operation* op, unsigned index);
    bool accumulateGrad(GradMap& grads, mlir::Value v, mlir::Value contrib);
    bool vjpForOp(mlir::Operation* op, mlir::Value g, GradMap& grads);
    bool vjpDotGeneral(mlir::stablehlo::DotGeneralOp op, mlir::Value g, GradMap& grads);
    bool vjpBroadcastInDim(mlir::stablehlo::BroadcastInDimOp op, mlir::Value g, GradMap& grads);
    bool vjpReduce(mlir::stablehlo::ReduceOp op, mlir::Value g, GradMap& grads);
    bool runVJP(mlir::Value output, llvm::ArrayRef<mlir::Value> wrt, mlir::Value seed,
                llvm::SmallVectorImpl<mlir::Value>& out);
#else
    bool available_ = false;
    Impl() = default;
#endif
};

/** @brief Construct the emitter, eagerly creating an MLIR context with the
 *         func/arith/stablehlo dialects loaded and an empty module when
 *         MLIR+StableHLO are compiled in; otherwise leaves it unavailable. */
StableHLOEmitter::StableHLOEmitter()
    : impl_(std::make_unique<Impl>()) {}

StableHLOEmitter::~StableHLOEmitter() = default;

/** @brief True if this emitter was built with MLIR+StableHLO support and can
 *         actually emit ops (all emit* methods are no-op nullptr returns
 *         otherwise). */
bool StableHLOEmitter::isAvailable() const {
    return impl_->available_;
}

// ===== Arithmetic Operations =====

/** @brief Emit a StableHLO `stablehlo.add` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitAdd(void* lhs, void* rhs) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);
    auto result = b.create<mlir::stablehlo::AddOp>(
        impl_->loc(), lhsVal.getType(), lhsVal, rhsVal);
    return impl_->storeValue(result.getResult());
#else
    (void)lhs; (void)rhs;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.subtract` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitSubtract(void* lhs, void* rhs) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);
    auto result = b.create<mlir::stablehlo::SubtractOp>(
        impl_->loc(), lhsVal.getType(), lhsVal, rhsVal);
    return impl_->storeValue(result.getResult());
#else
    (void)lhs; (void)rhs;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.multiply` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitMultiply(void* lhs, void* rhs) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);
    auto result = b.create<mlir::stablehlo::MulOp>(
        impl_->loc(), lhsVal.getType(), lhsVal, rhsVal);
    return impl_->storeValue(result.getResult());
#else
    (void)lhs; (void)rhs;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.divide` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitDivide(void* lhs, void* rhs) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);
    auto result = b.create<mlir::stablehlo::DivOp>(
        impl_->loc(), lhsVal.getType(), lhsVal, rhsVal);
    return impl_->storeValue(result.getResult());
#else
    (void)lhs; (void)rhs;
    return nullptr;
#endif
}

// ===== Matrix Operations =====

/** @brief Emit a StableHLO `stablehlo.dot_general` op, inferring the output
 *         shape from `dims`' batching/contracting dimensions. Delegates to
 *         Impl::dotGeneral(), which is also what the DOT_GENERAL VJP rule
 *         uses, so the forward and backward passes cannot disagree about
 *         the result layout.
 *
 *         NOTE (behaviour change): this previously carried a fast path that
 *         returned `{lhs[0], rhs[1]}` for any 2-D x 2-D dot. That is only
 *         the correct shape when the contraction is the canonical
 *         lhs_contracting={1}, rhs_contracting={0}; for e.g.
 *         lhs_contracting={0} (which is exactly what the DOT_GENERAL VJP
 *         emits when it contracts a gradient's batch-major dimension) it
 *         produced a result type that does not match the values the op
 *         actually computes. The general rule below reduces to the same
 *         `{M, N}` in the canonical case, so the fast path bought nothing
 *         and cost correctness. Returns nullptr if MLIR support isn't
 *         available. */
void* StableHLOEmitter::emitMatmul(void* lhs, void* rhs, const DotDimensionNumbers& dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto result = impl_->dotGeneral(
        impl_->toValue(lhs), impl_->toValue(rhs),
        dims.lhs_batching_dims, dims.rhs_batching_dims,
        dims.lhs_contracting_dims, dims.rhs_contracting_dims);
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)lhs; (void)rhs; (void)dims;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.transpose` op, computing the output
 *         shape by permuting `input`'s dimensions per `permutation`. Returns
 *         nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitTranspose(void* input, const std::vector<int64_t>& permutation) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());
    auto inputShape = inputType.getShape();

    // Compute output shape by permuting dimensions
    std::vector<int64_t> outShape(permutation.size());
    for (size_t i = 0; i < permutation.size(); i++) {
        outShape[i] = inputShape[permutation[i]];
    }
    auto outType = mlir::RankedTensorType::get(outShape, inputType.getElementType());

    auto permAttr = b.getDenseI64ArrayAttr(permutation);
    auto transposeOp = b.create<mlir::stablehlo::TransposeOp>(
        impl_->loc(), outType, inputVal, permAttr);
    return impl_->storeValue(transposeOp.getResult());
#else
    (void)input; (void)permutation;
    return nullptr;
#endif
}

// ===== Transcendental Operations =====

/** @brief Emit a StableHLO `stablehlo.exponential` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitExp(void* input) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto result = b.create<mlir::stablehlo::ExpOp>(
        impl_->loc(), inputVal.getType(), inputVal);
    return impl_->storeValue(result.getResult());
#else
    (void)input;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.log` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitLog(void* input) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto result = b.create<mlir::stablehlo::LogOp>(
        impl_->loc(), inputVal.getType(), inputVal);
    return impl_->storeValue(result.getResult());
#else
    (void)input;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.sine` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitSin(void* input) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto result = b.create<mlir::stablehlo::SineOp>(
        impl_->loc(), inputVal.getType(), inputVal);
    return impl_->storeValue(result.getResult());
#else
    (void)input;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.cosine` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitCos(void* input) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto result = b.create<mlir::stablehlo::CosineOp>(
        impl_->loc(), inputVal.getType(), inputVal);
    return impl_->storeValue(result.getResult());
#else
    (void)input;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.tanh` op. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitTanh(void* input) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto result = b.create<mlir::stablehlo::TanhOp>(
        impl_->loc(), inputVal.getType(), inputVal);
    return impl_->storeValue(result.getResult());
#else
    (void)input;
    return nullptr;
#endif
}

// ===== Reduction Operations =====

/** @brief Emit a StableHLO `stablehlo.reduce` op over `axes` with `op`
 *         (sum/prod/max/min). Delegates to Impl::reduceWithBody(), which
 *         builds the identity-element constant and the single-binary-op
 *         reduction body; the gradient rules need the same construction
 *         (a sum-reduce is how a cotangent is un-broadcast), so it lives in
 *         one place. Returns nullptr for an unsupported element type/op or
 *         if MLIR support isn't available. */
void* StableHLOEmitter::emitReduce(void* input, const std::vector<int64_t>& axes, StableHLOOp op) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto result = impl_->reduceWithBody(impl_->toValue(input), axes, op);
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)input; (void)axes; (void)op;
    return nullptr;
#endif
}

// ===== Shape Operations =====

/** @brief Emit a StableHLO `stablehlo.reshape` op to `new_shape`. Returns
 *         nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitReshape(void* input, const std::vector<int64_t>& new_shape) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());
    auto outType = mlir::RankedTensorType::get(new_shape, inputType.getElementType());
    auto reshapeOp = b.create<mlir::stablehlo::ReshapeOp>(
        impl_->loc(), outType, inputVal);
    return impl_->storeValue(reshapeOp.getResult());
#else
    (void)input; (void)new_shape;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.broadcast_in_dim` op. Note: currently
 *         reuses the input type as the result type (identity broadcast) —
 *         callers needing an actual shape change must set up the true
 *         output type via the XLA codegen layer. Returns nullptr if MLIR
 *         support isn't available. */
void* StableHLOEmitter::emitBroadcast(void* input, const std::vector<int64_t>& broadcast_dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());

    // broadcast_dims maps input dimensions to output dimensions.
    // The output type must be provided by the caller in a real pipeline;
    // for now, we use BroadcastInDimOp which requires an explicit result type.
    // The caller should set up the output type via the XLA codegen layer.
    // For self-contained usage, broadcast to same shape (identity).
    auto broadcastDimsAttr = b.getDenseI64ArrayAttr(broadcast_dims);
    auto broadcastOp = b.create<mlir::stablehlo::BroadcastInDimOp>(
        impl_->loc(), inputType, inputVal, broadcastDimsAttr);
    return impl_->storeValue(broadcastOp.getResult());
#else
    (void)input; (void)broadcast_dims;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.slice` op with the given per-dimension
 *         `[start, limit)` bounds and `strides`, computing the resulting
 *         shape. Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitSlice(void* input, const std::vector<int64_t>& start,
                                   const std::vector<int64_t>& limit,
                                   const std::vector<int64_t>& strides) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());

    // Compute output shape from slice parameters
    std::vector<int64_t> outShape(start.size());
    for (size_t i = 0; i < start.size(); i++) {
        outShape[i] = (limit[i] - start[i] + strides[i] - 1) / strides[i];
    }
    auto outType = mlir::RankedTensorType::get(outShape, inputType.getElementType());

    auto startAttr = b.getDenseI64ArrayAttr(start);
    auto limitAttr = b.getDenseI64ArrayAttr(limit);
    auto stridesAttr = b.getDenseI64ArrayAttr(strides);

    auto sliceOp = b.create<mlir::stablehlo::SliceOp>(
        impl_->loc(), outType, inputVal, startAttr, limitAttr, stridesAttr);
    return impl_->storeValue(sliceOp.getResult());
#else
    (void)input; (void)start; (void)limit; (void)strides;
    return nullptr;
#endif
}

// ===== Indexing Operations =====

/** @brief Emit a StableHLO `stablehlo.gather` op: embedding lookup; without
 *         it no language model can run — there is no other way to select
 *         rows of an embedding table by a dynamic (runtime) row index.
 *         Computes the output shape itself from `dims.offset_dims` /
 *         `collapsed_slice_dims` and `slice_sizes` per the StableHLO gather
 *         semantics (offset positions come from the non-collapsed slice
 *         sizes in operand-dimension order, batch positions come from
 *         `start_indices`' shape with `index_vector_dim` removed). Returns
 *         nullptr on malformed dimension numbers or if MLIR support isn't
 *         available. */
void* StableHLOEmitter::emitGather(void* operand, void* start_indices,
                                    const GatherDimensionNumbers& dims,
                                    const std::vector<int64_t>& slice_sizes) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto operandVal = impl_->toValue(operand);
    auto startIndicesVal = impl_->toValue(start_indices);
    auto operandType = mlir::cast<mlir::RankedTensorType>(operandVal.getType());
    auto startIndicesType = mlir::cast<mlir::RankedTensorType>(startIndicesVal.getType());
    auto operandShape = operandType.getShape();
    auto startIndicesShape = startIndicesType.getShape();
    int64_t startIndicesRank = (int64_t)startIndicesShape.size();

    if (slice_sizes.size() != operandShape.size()) return nullptr;  // malformed: one size per operand dim

    // Non-collapsed slice sizes, in operand-dimension order — these fill the
    // offset_dims positions of the output, in order.
    std::vector<int64_t> adjustedSliceSizes;
    for (int64_t d = 0; d < (int64_t)slice_sizes.size(); d++) {
        bool collapsed = false;
        for (auto c : dims.collapsed_slice_dims) if (c == d) { collapsed = true; break; }
        if (!collapsed) adjustedSliceSizes.push_back(slice_sizes[d]);
    }
    if (adjustedSliceSizes.size() != dims.offset_dims.size()) return nullptr;  // malformed

    // start_indices shape with index_vector_dim removed — these fill the
    // remaining (batch) positions of the output, in order.
    std::vector<int64_t> batchDimSizes;
    for (int64_t d = 0; d < startIndicesRank; d++) {
        if (d == dims.index_vector_dim) continue;
        batchDimSizes.push_back(startIndicesShape[d]);
    }

    int64_t outRank = (int64_t)dims.offset_dims.size() + (int64_t)batchDimSizes.size();
    std::vector<int64_t> outShape(outRank, 0);
    std::vector<bool> isOffsetDim(outRank, false);
    for (auto od : dims.offset_dims) {
        if (od < 0 || od >= outRank) return nullptr;  // malformed
        isOffsetDim[od] = true;
    }
    size_t offsetIdx = 0, batchIdx = 0;
    for (int64_t i = 0; i < outRank; i++) {
        if (isOffsetDim[i]) outShape[i] = adjustedSliceSizes[offsetIdx++];
        else outShape[i] = batchDimSizes[batchIdx++];
    }
    auto outType = mlir::RankedTensorType::get(outShape, operandType.getElementType());

    // operand_batching_dims/start_indices_batching_dims are the newer
    // batched-gather fields on StableHLO's GatherDimensionNumbers attribute;
    // Eshkol never emits batched gather, so both are passed empty here.
    auto gatherDimNumbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(
        impl_->ctx_.get(),
        dims.offset_dims,
        dims.collapsed_slice_dims,
        /*operandBatchingDims=*/{},
        /*startIndicesBatchingDims=*/{},
        dims.start_index_map,
        dims.index_vector_dim);

    // UNCERTAIN: GatherOp has no hand-declared `let builders` in
    // StablehloOps.td, so this relies on the ODS default builder taking
    // every declared operand then every declared attribute in argument
    // order (operand, start_indices, dimension_numbers, slice_sizes,
    // indices_are_sorted). The generated StablehloOps.h.inc is not present
    // in this worktree (nothing has been built yet), so I could not confirm
    // whether the trailing DefaultValuedOptionalAttr<BoolAttr> for
    // indices_are_sorted is (a) required exactly as passed here via
    // b.getBoolAttr(false), (b) elided by a shorter overload when omitted,
    // or (c) expected as a raw `bool` rather than a `BoolAttr`. Verify
    // against the built header before compiling.
    auto gatherOp = b.create<mlir::stablehlo::GatherOp>(
        impl_->loc(), outType, operandVal, startIndicesVal,
        gatherDimNumbers, b.getDenseI64ArrayAttr(slice_sizes),
        b.getBoolAttr(false));
    return impl_->storeValue(gatherOp.getResult());
#else
    (void)operand; (void)start_indices; (void)dims; (void)slice_sizes;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.scatter` op with an add-combiner body:
 *         embedding-gradient accumulation; without it, backprop through an
 *         embedding table has no way to accumulate gradients for rows that
 *         were gathered more than once. Delegates to Impl::scatterAdd(),
 *         which is the same routine the GATHER VJP rule calls — the
 *         embedding gradient IS a scatter-add, so there is exactly one
 *         implementation of it. Only the additive combiner is implemented
 *         (a replace-style scatter would need a different body). Returns
 *         nullptr for an unsupported element type or if MLIR support isn't
 *         available. */
void* StableHLOEmitter::emitScatter(void* operand, void* scatter_indices, void* updates,
                                     const ScatterDimensionNumbers& dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;

    // input_batching_dims/scatter_indices_batching_dims are the newer
    // batched-scatter fields on StableHLO's ScatterDimensionNumbers
    // attribute; Eshkol never emits batched scatter, so both are empty here.
    auto scatterDimNumbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        impl_->ctx_.get(),
        dims.update_window_dims,
        dims.inserted_window_dims,
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        dims.scatter_dims_to_operand_dims,
        dims.index_vector_dim);

    auto result = impl_->scatterAdd(
        impl_->toValue(operand), impl_->toValue(scatter_indices),
        impl_->toValue(updates), scatterDimNumbers);
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)operand; (void)scatter_indices; (void)updates; (void)dims;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.dynamic_slice` op: KV cache read —
 *         pulls the current-position slice out of the cache using a
 *         runtime index, which `stablehlo.slice`'s compile-time-constant
 *         bounds cannot express. Output shape is exactly `slice_sizes`
 *         (dynamic_slice never changes rank). Returns nullptr on a
 *         start_indices/slice_sizes/operand rank mismatch or if MLIR
 *         support isn't available. */
void* StableHLOEmitter::emitDynamicSlice(void* input, const std::vector<void*>& start_indices,
                                          const std::vector<int64_t>& slice_sizes) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());

    if (start_indices.size() != slice_sizes.size()) return nullptr;         // malformed
    if (slice_sizes.size() != inputType.getShape().size()) return nullptr; // malformed: one per operand dim

    std::vector<mlir::Value> startVals;
    startVals.reserve(start_indices.size());
    for (auto* idx : start_indices) startVals.push_back(impl_->toValue(idx));

    auto outType = mlir::RankedTensorType::get(slice_sizes, inputType.getElementType());
    auto sliceOp = b.create<mlir::stablehlo::DynamicSliceOp>(
        impl_->loc(), outType, inputVal, mlir::ValueRange(startVals),
        b.getDenseI64ArrayAttr(slice_sizes));
    return impl_->storeValue(sliceOp.getResult());
#else
    (void)input; (void)start_indices; (void)slice_sizes;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.dynamic_update_slice` op: KV cache
 *         write — writes the newest key/value slice into the cache at a
 *         runtime position without recompiling the graph on every decode
 *         step (the position advances every step; a static `stablehlo.pad`
 *         + concatenate cannot target a runtime offset). Result has the
 *         same shape as `operand` (the cache never changes shape on
 *         write). Returns nullptr on a start_indices/operand rank mismatch
 *         or if MLIR support isn't available. */
void* StableHLOEmitter::emitDynamicUpdateSlice(void* input, void* update,
                                                const std::vector<void*>& start_indices) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto updateVal = impl_->toValue(update);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());

    if (start_indices.size() != inputType.getShape().size()) return nullptr;  // malformed

    std::vector<mlir::Value> startVals;
    startVals.reserve(start_indices.size());
    for (auto* idx : start_indices) startVals.push_back(impl_->toValue(idx));

    auto updateOp = b.create<mlir::stablehlo::DynamicUpdateSliceOp>(
        impl_->loc(), inputType, inputVal, updateVal, mlir::ValueRange(startVals));
    return impl_->storeValue(updateOp.getResult());
#else
    (void)input; (void)update; (void)start_indices;
    return nullptr;
#endif
}

// ===== Elementwise Selection Operations =====

/** @brief Emit a StableHLO `stablehlo.select` op: causal masking — every
 *         autoregressive attention layer needs to replace future-position
 *         scores (or activations) elementwise based on a boolean mask
 *         tensor, and select is the only op that picks between two whole
 *         tensors by a predicate tensor rather than a single scalar
 *         condition. Result type matches `on_true`, consistent with every
 *         other emitter in this file always supplying an explicit result
 *         type rather than relying on StableHLO's own type inference.
 *         Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitSelect(void* pred, void* on_true, void* on_false) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto predVal = impl_->toValue(pred);
    auto onTrueVal = impl_->toValue(on_true);
    auto onFalseVal = impl_->toValue(on_false);
    auto selectOp = b.create<mlir::stablehlo::SelectOp>(
        impl_->loc(), onTrueVal.getType(), predVal, onTrueVal, onFalseVal);
    return impl_->storeValue(selectOp.getResult());
#else
    (void)pred; (void)on_true; (void)on_false;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.compare` op: causal masking (and
 *         greedy/argmax decode) both need an elementwise boolean
 *         comparison to produce the predicate tensor that `select`/`reduce`
 *         then act on — there is no other way to turn two tensors into a
 *         boolean mask. Uses the hand-declared convenience builder from
 *         StablehloOps.td (`Value lhs, Value rhs, ComparisonDirection,
 *         ComparisonType`), which is the one signature here I read
 *         verbatim off an explicit `let builders` clause rather than
 *         inferring from ODS defaults, and which infers the i1 result type
 *         itself. Returns nullptr for an unrecognized direction/compare_type
 *         or if MLIR support isn't available. */
void* StableHLOEmitter::emitCompare(void* lhs, void* rhs, ComparisonDirection direction,
                                     ComparisonType compare_type) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);

    mlir::stablehlo::ComparisonDirection dir;
    switch (direction) {
        case ComparisonDirection::EQ: dir = mlir::stablehlo::ComparisonDirection::EQ; break;
        case ComparisonDirection::NE: dir = mlir::stablehlo::ComparisonDirection::NE; break;
        case ComparisonDirection::GE: dir = mlir::stablehlo::ComparisonDirection::GE; break;
        case ComparisonDirection::GT: dir = mlir::stablehlo::ComparisonDirection::GT; break;
        case ComparisonDirection::LE: dir = mlir::stablehlo::ComparisonDirection::LE; break;
        case ComparisonDirection::LT: dir = mlir::stablehlo::ComparisonDirection::LT; break;
        default: return nullptr;  // Unrecognized comparison direction
    }
    mlir::stablehlo::ComparisonType type;
    switch (compare_type) {
        case ComparisonType::NOTYPE:     type = mlir::stablehlo::ComparisonType::NOTYPE; break;
        case ComparisonType::FLOAT:      type = mlir::stablehlo::ComparisonType::FLOAT; break;
        case ComparisonType::TOTALORDER: type = mlir::stablehlo::ComparisonType::TOTALORDER; break;
        case ComparisonType::SIGNED:     type = mlir::stablehlo::ComparisonType::SIGNED; break;
        case ComparisonType::UNSIGNED:   type = mlir::stablehlo::ComparisonType::UNSIGNED; break;
        default: return nullptr;  // Unrecognized comparison type
    }

    auto compareOp = b.create<mlir::stablehlo::CompareOp>(
        impl_->loc(), lhsVal, rhsVal, dir, type);
    return impl_->storeValue(compareOp.getResult());
#else
    (void)lhs; (void)rhs; (void)direction; (void)compare_type;
    return nullptr;
#endif
}

// ===== Shape Construction Operations =====

/** @brief Emit a StableHLO `stablehlo.concatenate` op: assembling the KV
 *         cache (old cache ++ newest slice) and rejoining split attention
 *         heads both require joining tensors along an axis, which nothing
 *         else in this file does. Computes the output shape by summing
 *         `dimension` across `inputs` and taking every other dimension
 *         from the first input. Returns nullptr if `inputs` is empty,
 *         `dimension` is out of range, or if MLIR support isn't available. */
void* StableHLOEmitter::emitConcatenate(const std::vector<void*>& inputs, int64_t dimension) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    if (inputs.empty()) return nullptr;  // Nothing to concatenate
    auto& b = *impl_->builder_;

    std::vector<mlir::Value> inputVals;
    inputVals.reserve(inputs.size());
    for (auto* in : inputs) inputVals.push_back(impl_->toValue(in));

    auto firstType = mlir::cast<mlir::RankedTensorType>(inputVals[0].getType());
    if (dimension < 0 || dimension >= (int64_t)firstType.getShape().size()) return nullptr;

    std::vector<int64_t> outShape(firstType.getShape().begin(), firstType.getShape().end());
    for (size_t i = 1; i < inputVals.size(); i++) {
        auto t = mlir::cast<mlir::RankedTensorType>(inputVals[i].getType());
        outShape[dimension] += t.getShape()[dimension];
    }
    auto outType = mlir::RankedTensorType::get(outShape, firstType.getElementType());

    auto concatOp = b.create<mlir::stablehlo::ConcatenateOp>(
        impl_->loc(), outType, mlir::ValueRange(inputVals),
        b.getI64IntegerAttr(dimension));
    return impl_->storeValue(concatOp.getResult());
#else
    (void)inputs; (void)dimension;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.pad` op: pre-growing a tensor by a
 *         constant amount on each axis (e.g. reserving KV-cache capacity,
 *         or padding a ragged batch to a fixed shape) before any value is
 *         written into the new region. Computes the output shape per
 *         StableHLO's pad spec: `low + high + size + max(size-1,0) *
 *         interior` per dimension. Returns nullptr on a
 *         low/high/interior-padding length mismatch against `operand`'s
 *         rank, or if MLIR support isn't available. */
void* StableHLOEmitter::emitPad(void* input, void* padding_value,
                                 const std::vector<int64_t>& edge_padding_low,
                                 const std::vector<int64_t>& edge_padding_high,
                                 const std::vector<int64_t>& interior_padding) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);
    auto paddingValueVal = impl_->toValue(padding_value);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());
    auto inputShape = inputType.getShape();

    if (edge_padding_low.size() != inputShape.size() ||
        edge_padding_high.size() != inputShape.size() ||
        interior_padding.size() != inputShape.size()) {
        return nullptr;  // malformed
    }

    std::vector<int64_t> outShape(inputShape.size());
    for (size_t i = 0; i < inputShape.size(); i++) {
        int64_t size = inputShape[i];
        int64_t interiorContribution = (size > 1 ? (size - 1) : 0) * interior_padding[i];
        outShape[i] = edge_padding_low[i] + edge_padding_high[i] + size + interiorContribution;
    }
    auto outType = mlir::RankedTensorType::get(outShape, inputType.getElementType());

    auto padOp = b.create<mlir::stablehlo::PadOp>(
        impl_->loc(), outType, inputVal, paddingValueVal,
        b.getDenseI64ArrayAttr(edge_padding_low),
        b.getDenseI64ArrayAttr(edge_padding_high),
        b.getDenseI64ArrayAttr(interior_padding));
    return impl_->storeValue(padOp.getResult());
#else
    (void)input; (void)padding_value; (void)edge_padding_low;
    (void)edge_padding_high; (void)interior_padding;
    return nullptr;
#endif
}

/** @brief Emit a StableHLO `stablehlo.iota` op: causal masking needs a
 *         position-index tensor (iota, compared against its own transpose)
 *         to build the lower-triangular mask in the first place — nothing
 *         else in this file can manufacture position indices out of
 *         nothing. `elem` is restricted to the F16/F32/F64/BF16 set
 *         ElementType currently maps to an MLIR type for (see
 *         xla_types.cpp's getMLIRElementType); integer iota (the more
 *         common case for position-index tensors in practice) is not wired
 *         up here. Returns nullptr for any other element type,
 *         `iota_dimension` out of range, or if MLIR support isn't
 *         available. */
void* StableHLOEmitter::emitIota(const std::vector<int64_t>& shape, int64_t iota_dimension,
                                  ElementType elem) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    if (iota_dimension < 0 || iota_dimension >= (int64_t)shape.size()) return nullptr;
    auto& b = *impl_->builder_;

    mlir::Type elemType;
    switch (elem) {
        case ElementType::F16:  elemType = b.getF16Type(); break;
        case ElementType::F32:  elemType = b.getF32Type(); break;
        case ElementType::F64:  elemType = b.getF64Type(); break;
        case ElementType::BF16: elemType = b.getBF16Type(); break;
        default: return nullptr;  // Unsupported element type
    }
    auto outType = mlir::RankedTensorType::get(shape, elemType);

    auto iotaOp = b.create<mlir::stablehlo::IotaOp>(
        impl_->loc(), outType, b.getI64IntegerAttr(iota_dimension));
    return impl_->storeValue(iotaOp.getResult());
#else
    (void)shape; (void)iota_dimension; (void)elem;
    return nullptr;
#endif
}

// ===== Type Conversion Operations =====

/** @brief Emit a StableHLO `stablehlo.convert` op: dtype conversion between
 *         the precisions ElementType currently maps to MLIR
 *         (F64/F32/F16/BF16) — needed wherever mixed-precision compute
 *         crosses a dtype boundary (e.g. weights kept in BF16, accumulation
 *         in F32). Uses the hand-declared convenience builder from
 *         StablehloOps.td (`Value operand, Type result_element_ty`), which
 *         builds the result tensor type itself from `operand`'s shape.
 *         Returns nullptr for an unsupported target element type or if
 *         MLIR support isn't available. */
void* StableHLOEmitter::emitConvert(void* input, ElementType target) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto inputVal = impl_->toValue(input);

    mlir::Type targetElemType;
    switch (target) {
        case ElementType::F16:  targetElemType = b.getF16Type(); break;
        case ElementType::F32:  targetElemType = b.getF32Type(); break;
        case ElementType::F64:  targetElemType = b.getF64Type(); break;
        case ElementType::BF16: targetElemType = b.getBF16Type(); break;
        default: return nullptr;  // Unsupported element type
    }

    auto convertOp = b.create<mlir::stablehlo::ConvertOp>(
        impl_->loc(), inputVal, targetElemType);
    return impl_->storeValue(convertOp.getResult());
#else
    (void)input; (void)target;
    return nullptr;
#endif
}

#ifdef ESHKOL_XLA_FULL_MLIR

// ===== Reverse-Mode Gradients (VJP) =====
//
// DESIGN NOTES — read these before changing anything below.
//
// THE TAPE IS THE IR. StableHLO is already SSA: every op emitted above
// records its operands and its result, and the graph is acyclic by
// construction. A separate shadow tape (op, inputs, output) would be
// duplicate state that can silently desync from the ops actually emitted —
// and a desynced tape produces a gradient that is wrong without being
// invalid — so the backward pass walks the use-def chains directly:
// `mlir::Value::getDefiningOp()` gives a value's producer and
// `op->getOperand(i)` gives that producer's inputs. `runVJP()` does an
// iterative post-order DFS from the output value, which yields the reachable
// sub-DAG in topological order (producers before consumers), and then walks
// that list in REVERSE. The DFS is iterative rather than recursive on
// purpose: a transformer graph is thousands of ops deep and a recursive walk
// would overflow the C stack on a real model.
//
// ACCUMULATION. Reverse mode requires SUMMING the contributions of every
// consumer of a value — a weight matrix read by two layers receives two
// cotangents, and taking only one of them silently halves the gradient.
// `GradMap` maps a forward value to its accumulated cotangent;
// `accumulateGrad()` stores the first contribution and emits a
// `stablehlo.add` for each later one, so N consumers leave an (N-1)-deep
// chain of adds that XLA's algebraic simplifier flattens. Correctness
// depends on ORDER: a value's cotangent is final only once every consumer
// has been processed, which the reverse topological walk guarantees —
// nothing reads a cotangent before the op that produced the value is popped.
//
// BROADCASTING. This is where naive VJPs are silently wrong, so it is
// handled in two clearly separated places:
//
//   1. `vjpBroadcastInDim()` is THE mechanism. Nothing broadcasts implicitly
//      in valid StableHLO: an Eshkol program that adds a [3] bias to a [2,3]
//      activation must first materialise a `stablehlo.broadcast_in_dim`, and
//      the gradient of THAT op is a sum-reduce over exactly (a) the result
//      dimensions the operand never spanned — here axis 0, the [2] — and
//      (b) the result dimensions the operand spanned with extent 1 while the
//      result is wider, followed by a transpose (broadcast_dimensions is a
//      mapping and is not required to be increasing) and a reshape that puts
//      the summed-away size-1 dimensions back. Omit (a) and the shape is
//      wrong and the compiler catches it; omit (b) and the shape is RIGHT
//      and the values are wrong, which is the failure that reaches
//      production.
//
//   2. `reduceGradToShape()` is a guard, not the mechanism. Eshkol's forward
//      emitters take an elementwise op's result type from its LHS operand and
//      never check that the operands agree, so a graph handed to us can
//      contain a mismatched elementwise op. Every elementwise rule routes its
//      contribution through `reduceGradToShape()`, which is the identity when
//      the shapes already match (the normal case, so it costs nothing) and
//      otherwise performs the NumPy-style un-broadcast: sum the leading extra
//      axes, sum any axis where the target has extent 1, reshape to the
//      target. It returns null rather than guessing when the shapes are not
//      broadcast-compatible.
//
// FAIL CLOSED. A wrong gradient does not crash; it trains a model to
// garbage over hours and the failure is silent. So an op with no VJP rule, a
// reduce whose combiner is not one we recognise, or a shape that cannot be
// reconciled aborts the ENTIRE request: `emitVJP()` returns
// `complete == false`, an EMPTY gradient vector, and a diagnostic naming the
// op. Nothing partial is ever returned, because a caller that forgets to
// check a flag and uses a half-populated vector gets a model that trains.

/** @brief Build a rank-0 (scalar) StableHLO constant tensor holding
 *         `elem_attr`. All constant creation in the gradient path funnels
 *         through here so that exactly one place needs checking against the
 *         built MLIR headers: `DenseElementsAttr::get(ShapedType,
 *         ArrayRef<Attribute>)` with a single element is unambiguous for a
 *         rank-0 (one-element) type, which is why the splat case goes
 *         through broadcastInDim() from a rank-0 constant instead of relying
 *         on the one-element-means-splat shorthand for a wider type. Note
 *         that the ArrayRef<double> form used by the pre-existing reduce code
 *         is only valid for an f64 element type — it asserts on bit width for
 *         f32 — which is a further reason the gradient path does not use it. */
mlir::Value StableHLOEmitter::Impl::constantFromAttr(mlir::Type elem_type, mlir::Attribute elem_attr) {
    if (!elem_attr) return nullptr;
    auto scalarType = mlir::RankedTensorType::get({}, elem_type);
    auto dense = mlir::DenseElementsAttr::get(
        scalarType, llvm::ArrayRef<mlir::Attribute>{elem_attr});
    return builder_->create<mlir::stablehlo::ConstantOp>(loc(), dense).getResult();
}

/** @brief Build a rank-0 constant of `elem_type` holding `value`. Floats go
 *         through FloatAttr, which converts the double into the element
 *         type's own APFloat semantics (so -INFINITY reaches an f32 as an f32
 *         infinity); integers go through an APInt of the type's exact width.
 *         Returns nullptr for any other element type. */
mlir::Value StableHLOEmitter::Impl::constantScalar(mlir::Type elem_type, double value) {
    auto& b = *builder_;
    if (mlir::isa<mlir::FloatType>(elem_type)) {
        return constantFromAttr(elem_type, b.getFloatAttr(elem_type, value));
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
        llvm::APInt raw(intType.getWidth(), static_cast<uint64_t>(static_cast<int64_t>(value)),
                        /*isSigned=*/true);
        return constantFromAttr(elem_type, mlir::IntegerAttr::get(elem_type, raw));
    }
    return nullptr;  // Unsupported element type for a constant
}

/** @brief Build a constant tensor of `type`'s shape with every element set to
 *         `value`, as a rank-0 constant broadcast out. The extra
 *         `broadcast_in_dim` is folded away by XLA before anything runs and
 *         buys certainty: a rank-0 constant is the one shape for which the
 *         DenseElementsAttr element-count contract is unambiguous. */
mlir::Value StableHLOEmitter::Impl::constantSplat(mlir::RankedTensorType type, double value) {
    auto scalar = constantScalar(type.getElementType(), value);
    if (!scalar) return nullptr;
    if (type.getRank() == 0) return scalar;
    return broadcastInDim(scalar, type.getShape(), /*bcast_dims=*/{});
}

/** @brief Zero tensor with the same type as `v` — the identity for gradient
 *         accumulation, and what an unreachable parameter's gradient is. */
mlir::Value StableHLOEmitter::Impl::zerosLike(mlir::Value v) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    return constantSplat(t, 0.0);
}

/** @brief Ones tensor with the same type as `v` — the default seed cotangent,
 *         which is the correct seed when the output is a scalar loss (and,
 *         for a non-scalar output, seeds the gradient of the SUM of its
 *         elements). */
mlir::Value StableHLOEmitter::Impl::onesLike(mlir::Value v) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    return constantSplat(t, 1.0);
}

/** @brief Emit `stablehlo.add`; result type taken from `a`. */
mlir::Value StableHLOEmitter::Impl::addV(mlir::Value a, mlir::Value b) {
    if (!a || !b) return nullptr;
    return builder_->create<mlir::stablehlo::AddOp>(loc(), a.getType(), a, b).getResult();
}

/** @brief Emit `stablehlo.subtract`; result type taken from `a`. */
mlir::Value StableHLOEmitter::Impl::subV(mlir::Value a, mlir::Value b) {
    if (!a || !b) return nullptr;
    return builder_->create<mlir::stablehlo::SubtractOp>(loc(), a.getType(), a, b).getResult();
}

/** @brief Emit `stablehlo.multiply`; result type taken from `a`. */
mlir::Value StableHLOEmitter::Impl::mulV(mlir::Value a, mlir::Value b) {
    if (!a || !b) return nullptr;
    return builder_->create<mlir::stablehlo::MulOp>(loc(), a.getType(), a, b).getResult();
}

/** @brief Emit `stablehlo.divide`; result type taken from `a`. */
mlir::Value StableHLOEmitter::Impl::divV(mlir::Value a, mlir::Value b) {
    if (!a || !b) return nullptr;
    return builder_->create<mlir::stablehlo::DivOp>(loc(), a.getType(), a, b).getResult();
}

/** @brief Negate `v` as a multiply by a -1 splat rather than by StableHLO's
 *         negate op. Deliberate: this file is being written without a built
 *         copy of the generated StablehloOps headers, and the C++ class name
 *         for `stablehlo.negate` (NegOp vs NegateOp) is exactly the kind of
 *         detail worth not guessing at, whereas MulOp is used a dozen lines
 *         above and is certain. XLA folds `x * -1` into a negate before any
 *         hardware sees it, so the choice costs nothing at runtime. */
mlir::Value StableHLOEmitter::Impl::negate(mlir::Value v) {
    if (!v) return nullptr;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    auto minusOne = constantSplat(t, -1.0);
    if (!minusOne) return nullptr;
    return mulV(v, minusOne);
}

/** @brief Emit `stablehlo.broadcast_in_dim` with an explicit result shape.
 *         `bcast_dims` gives, for each operand dimension in order, the result
 *         dimension it maps onto; it is empty for a rank-0 operand (a splat). */
mlir::Value StableHLOEmitter::Impl::broadcastInDim(mlir::Value v, llvm::ArrayRef<int64_t> out_shape,
                                                    llvm::ArrayRef<int64_t> bcast_dims) {
    if (!v) return nullptr;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    if (bcast_dims.size() != (size_t)t.getRank()) return nullptr;  // malformed
    auto outType = mlir::RankedTensorType::get(out_shape, t.getElementType());
    return builder_->create<mlir::stablehlo::BroadcastInDimOp>(
        loc(), outType, v, builder_->getDenseI64ArrayAttr(bcast_dims)).getResult();
}

/** @brief Emit `stablehlo.reshape` to `shape` (element count must match). */
mlir::Value StableHLOEmitter::Impl::reshapeTo(mlir::Value v, llvm::ArrayRef<int64_t> shape) {
    if (!v) return nullptr;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    auto outType = mlir::RankedTensorType::get(shape, t.getElementType());
    return builder_->create<mlir::stablehlo::ReshapeOp>(loc(), outType, v).getResult();
}

/** @brief Emit `stablehlo.transpose` permuting `v` by `perm`. */
mlir::Value StableHLOEmitter::Impl::transposeBy(mlir::Value v, llvm::ArrayRef<int64_t> perm) {
    if (!v) return nullptr;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t || perm.size() != (size_t)t.getRank()) return nullptr;
    auto inShape = t.getShape();
    std::vector<int64_t> outShape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        if (perm[i] < 0 || perm[i] >= t.getRank()) return nullptr;  // malformed
        outShape[i] = inShape[perm[i]];
    }
    auto outType = mlir::RankedTensorType::get(outShape, t.getElementType());
    return builder_->create<mlir::stablehlo::TransposeOp>(
        loc(), outType, v, builder_->getDenseI64ArrayAttr(perm)).getResult();
}

/** @brief Emit `stablehlo.convert` to `elem_type`, keeping the shape. Takes a
 *         raw mlir::Type rather than Eshkol's ElementType because the
 *         gradient path also has to convert an i1 mask to a float, which
 *         ElementType's F16/F32/F64/BF16-only mapping cannot express. */
mlir::Value StableHLOEmitter::Impl::convertElem(mlir::Value v, mlir::Type elem_type) {
    if (!v || !elem_type) return nullptr;
    return builder_->create<mlir::stablehlo::ConvertOp>(loc(), v, elem_type).getResult();
}

/** @brief Emit `stablehlo.slice` with explicit bounds, computing the result
 *         shape the same way emitSlice() does. */
mlir::Value StableHLOEmitter::Impl::sliceOf(mlir::Value v, llvm::ArrayRef<int64_t> start,
                                             llvm::ArrayRef<int64_t> limit,
                                             llvm::ArrayRef<int64_t> strides) {
    if (!v) return nullptr;
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!t) return nullptr;
    if (start.size() != (size_t)t.getRank() || limit.size() != start.size() ||
        strides.size() != start.size()) {
        return nullptr;  // malformed
    }
    std::vector<int64_t> outShape(start.size());
    for (size_t i = 0; i < start.size(); i++) {
        if (strides[i] <= 0 || limit[i] < start[i]) return nullptr;  // malformed
        outShape[i] = (limit[i] - start[i] + strides[i] - 1) / strides[i];
    }
    auto outType = mlir::RankedTensorType::get(outShape, t.getElementType());
    return builder_->create<mlir::stablehlo::SliceOp>(
        loc(), outType, v,
        builder_->getDenseI64ArrayAttr(start),
        builder_->getDenseI64ArrayAttr(limit),
        builder_->getDenseI64ArrayAttr(strides)).getResult();
}

/** @brief Emit `stablehlo.dot_general`, computing the result shape from the
 *         batching/contracting dimensions: the result is laid out as
 *         [batch dims in lhs_batching order][lhs free dims ascending]
 *         [rhs free dims ascending]. Shared by emitMatmul() and by the
 *         DOT_GENERAL VJP rule, which is what keeps the forward and backward
 *         passes from disagreeing about that layout — the VJP's transposes
 *         are derived from exactly this ordering. Returns nullptr on
 *         mismatched batching/contracting dimension counts or an
 *         out-of-range dimension. */
mlir::Value StableHLOEmitter::Impl::dotGeneral(mlir::Value lhs, mlir::Value rhs,
                                                llvm::ArrayRef<int64_t> lhs_batch,
                                                llvm::ArrayRef<int64_t> rhs_batch,
                                                llvm::ArrayRef<int64_t> lhs_contract,
                                                llvm::ArrayRef<int64_t> rhs_contract) {
    if (!lhs || !rhs) return nullptr;
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) return nullptr;
    if (lhs_batch.size() != rhs_batch.size()) return nullptr;        // malformed: batch dims pair up
    if (lhs_contract.size() != rhs_contract.size()) return nullptr;  // malformed: contractions pair up

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto member = [](llvm::ArrayRef<int64_t> v, int64_t x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };
    auto inRange = [](llvm::ArrayRef<int64_t> v, int64_t rank) {
        for (auto d : v) if (d < 0 || d >= rank) return false;
        return true;
    };
    if (!inRange(lhs_batch, lhsType.getRank()) || !inRange(lhs_contract, lhsType.getRank()) ||
        !inRange(rhs_batch, rhsType.getRank()) || !inRange(rhs_contract, rhsType.getRank())) {
        return nullptr;  // malformed
    }

    std::vector<int64_t> outShape;
    for (auto d : lhs_batch) outShape.push_back(lhsShape[d]);
    for (int64_t i = 0; i < lhsType.getRank(); i++)
        if (!member(lhs_batch, i) && !member(lhs_contract, i)) outShape.push_back(lhsShape[i]);
    for (int64_t i = 0; i < rhsType.getRank(); i++)
        if (!member(rhs_batch, i) && !member(rhs_contract, i)) outShape.push_back(rhsShape[i]);

    auto dotDimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        ctx_.get(), lhs_batch, rhs_batch, lhs_contract, rhs_contract);
    auto outType = mlir::RankedTensorType::get(outShape, lhsType.getElementType());
    return builder_->create<mlir::stablehlo::DotGeneralOp>(
        loc(), outType, lhs, rhs, dotDimNumbers,
        /*precision_config=*/nullptr,
        /*algorithm=*/nullptr).getResult();
}

/** @brief Emit `stablehlo.reduce` over `axes` with the identity element and
 *         single-binary-op body for `op` (SUM/PROD/MAX/MIN).
 *
 *         NOTE (bug fix relative to the previous inline version): the
 *         insertion point is saved BEFORE `createBlock()`, not after.
 *         `OpBuilder::createBlock()` moves the insertion point into the block
 *         it creates, so saving afterwards captured a point *inside the
 *         reduction body* and "restoring" it left the builder emitting into
 *         that region — every op created after a reduce would have landed in
 *         the reduce's body, after its return. That is invisible in a forward
 *         pass that emits one reduce and stops, and fatal for a backward pass
 *         that emits reduces in the middle of a long op stream. */
mlir::Value StableHLOEmitter::Impl::reduceWithBody(mlir::Value v, llvm::ArrayRef<int64_t> axes,
                                                    StableHLOOp op) {
    if (!v) return nullptr;
    auto& b = *builder_;
    auto l = loc();
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!inputType) return nullptr;
    auto elemType = inputType.getElementType();
    auto scalarType = mlir::RankedTensorType::get({}, elemType);

    // Identity element for the reduction.
    mlir::Value initValue;
    if (mlir::isa<mlir::FloatType>(elemType)) {
        double identity;
        switch (op) {
            case StableHLOOp::REDUCE_SUM:  identity = 0.0; break;
            case StableHLOOp::REDUCE_PROD: identity = 1.0; break;
            case StableHLOOp::REDUCE_MAX:  identity = -INFINITY; break;
            case StableHLOOp::REDUCE_MIN:  identity = INFINITY; break;
            default: return nullptr;
        }
        initValue = constantScalar(elemType, identity);
    } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        // Width-aware: a literal INT64_MIN truncated into an i8 identity is 0,
        // which is not the identity for an i8 max-reduce.
        unsigned w = intType.getWidth();
        llvm::APInt identity(w, 0);
        switch (op) {
            case StableHLOOp::REDUCE_SUM:  identity = llvm::APInt(w, 0); break;
            case StableHLOOp::REDUCE_PROD: identity = llvm::APInt(w, 1); break;
            case StableHLOOp::REDUCE_MAX:  identity = llvm::APInt::getSignedMinValue(w); break;
            case StableHLOOp::REDUCE_MIN:  identity = llvm::APInt::getSignedMaxValue(w); break;
            default: return nullptr;
        }
        initValue = constantFromAttr(elemType, mlir::IntegerAttr::get(elemType, identity));
    } else {
        return nullptr;  // Unsupported element type for reduction
    }
    if (!initValue) return nullptr;

    // Output shape: the input shape with the reduced dimensions removed.
    auto inputShape = inputType.getShape();
    std::vector<int64_t> outShape;
    for (int64_t i = 0; i < (int64_t)inputShape.size(); i++) {
        bool reduced = false;
        for (auto ax : axes) {
            if (ax == i) { reduced = true; break; }
        }
        if (!reduced) outShape.push_back(inputShape[i]);
    }
    auto outType = mlir::RankedTensorType::get(
        outShape.empty() ? llvm::ArrayRef<int64_t>{} : llvm::ArrayRef<int64_t>(outShape),
        elemType);

    auto reduceOp = b.create<mlir::stablehlo::ReduceOp>(
        l, mlir::TypeRange{outType}, mlir::ValueRange{v},
        mlir::ValueRange{initValue}, b.getDenseI64ArrayAttr(axes));

    // Save the insertion point BEFORE createBlock() moves it into the body.
    auto savedInsertionPoint = b.saveInsertionPoint();
    auto& body = reduceOp.getBody();
    auto* bodyBlock = b.createBlock(&body);
    bodyBlock->addArgument(scalarType, l);
    bodyBlock->addArgument(scalarType, l);
    b.setInsertionPointToStart(bodyBlock);
    auto arg0 = bodyBlock->getArgument(0);
    auto arg1 = bodyBlock->getArgument(1);

    mlir::Value bodyResult;
    switch (op) {
        case StableHLOOp::REDUCE_SUM:
            bodyResult = b.create<mlir::stablehlo::AddOp>(l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_PROD:
            bodyResult = b.create<mlir::stablehlo::MulOp>(l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_MAX:
            bodyResult = b.create<mlir::stablehlo::MaxOp>(l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_MIN:
            bodyResult = b.create<mlir::stablehlo::MinOp>(l, scalarType, arg0, arg1).getResult();
            break;
        default:
            b.restoreInsertionPoint(savedInsertionPoint);
            return nullptr;
    }
    b.create<mlir::stablehlo::ReturnOp>(l, mlir::ValueRange{bodyResult});
    b.restoreInsertionPoint(savedInsertionPoint);

    return reduceOp.getResult(0);
}

/** @brief Emit `stablehlo.scatter` with an add combiner: `operand` with
 *         `updates` ACCUMULATED (not overwritten) at the positions named by
 *         `indices`. This is the embedding gradient. Addition is the whole
 *         point of the combiner: a token that appears twice in a batch is
 *         gathered twice, so its row must receive the sum of both cotangents,
 *         and a replace-style scatter would keep only the last one.
 *
 *         Carries the same insertion-point fix as reduceWithBody(). */
mlir::Value StableHLOEmitter::Impl::scatterAdd(mlir::Value operand, mlir::Value indices,
                                                mlir::Value updates,
                                                mlir::stablehlo::ScatterDimensionNumbersAttr dims) {
    if (!operand || !indices || !updates) return nullptr;
    auto& b = *builder_;
    auto l = loc();
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (!operandType) return nullptr;
    auto elemType = operandType.getElementType();
    if (!mlir::isa<mlir::FloatType>(elemType) && !mlir::isa<mlir::IntegerType>(elemType)) {
        return nullptr;  // Unsupported element type for add-combine
    }
    auto scalarType = mlir::RankedTensorType::get({}, elemType);

    // Scatter is functional: the result has the operand's shape, not a new one.
    // UNCERTAIN (inherited, unchanged): ScatterOp has no hand-declared
    // `let builders` in StablehloOps.td, so this uses the ODS default builder
    // in declared-argument order (results, inputs, scatter_indices, updates,
    // scatter_dimension_numbers, indices_are_sorted, unique_indices). The
    // generated StablehloOps.h.inc is not present in this worktree, so I could
    // not confirm the two trailing DefaultValuedOptionalAttr<BoolAttr>
    // parameters are passed this way (vs. an elided overload, vs. raw `bool`).
    // Verify against the built header before compiling.
    auto scatterOp = b.create<mlir::stablehlo::ScatterOp>(
        l, mlir::TypeRange{operandType}, mlir::ValueRange{operand},
        indices, mlir::ValueRange{updates}, dims,
        /*indices_are_sorted=*/b.getBoolAttr(false),
        /*unique_indices=*/b.getBoolAttr(false));

    // Save the insertion point BEFORE createBlock() moves it into the region.
    auto savedInsertionPoint = b.saveInsertionPoint();
    auto& region = scatterOp.getUpdateComputation();
    auto* bodyBlock = b.createBlock(&region);
    bodyBlock->addArgument(scalarType, l);
    bodyBlock->addArgument(scalarType, l);
    b.setInsertionPointToStart(bodyBlock);
    auto sum = b.create<mlir::stablehlo::AddOp>(
        l, scalarType, bodyBlock->getArgument(0), bodyBlock->getArgument(1)).getResult();
    b.create<mlir::stablehlo::ReturnOp>(l, mlir::ValueRange{sum});
    b.restoreInsertionPoint(savedInsertionPoint);

    return scatterOp.getResult(0);
}

/** @brief Un-broadcast guard for the elementwise VJP rules: reduce `grad`
 *         down to `target`'s shape, NumPy-style.
 *
 *         In valid StableHLO this is the identity, because elementwise ops
 *         require equal operand and result shapes — the real broadcasting in
 *         an Eshkol program is materialised as a `broadcast_in_dim` whose own
 *         VJP does the reduction (see vjpBroadcastInDim). This exists because
 *         the forward emitters in this file take an elementwise op's result
 *         type from its LHS operand without checking the RHS, so a graph
 *         handed to us can legitimately contain `add([2,3], [3])`. In that
 *         case the gradient w.r.t. the [3] operand MUST be summed over axis
 *         0; returning the [2,3] cotangent unchanged is the classic silent
 *         wrong-gradient bug.
 *
 *         Returns nullptr (never a guess) if the shapes are not
 *         broadcast-compatible — a lower rank on the gradient side, or a
 *         target extent that is neither equal to the gradient's nor 1. */
mlir::Value StableHLOEmitter::Impl::reduceGradToShape(mlir::Value grad,
                                                       mlir::RankedTensorType target) {
    if (!grad || !target) return nullptr;
    auto gType = mlir::dyn_cast<mlir::RankedTensorType>(grad.getType());
    if (!gType) return nullptr;
    if (gType.getShape() == target.getShape()) return grad;  // the normal case: no-op

    const int64_t gr = gType.getRank();
    const int64_t tr = target.getRank();
    if (gr < tr) return nullptr;  // cannot un-broadcast to a HIGHER rank

    auto gShape = gType.getShape();
    auto tShape = target.getShape();
    std::vector<int64_t> axes;
    for (int64_t i = 0; i < gr - tr; i++) axes.push_back(i);   // leading dims the target lacks
    for (int64_t i = 0; i < tr; i++) {
        const int64_t g = gShape[gr - tr + i];
        const int64_t t = tShape[i];
        if (t == g) continue;
        if (t == 1) { axes.push_back(gr - tr + i); continue; }  // target was stretched here
        return nullptr;  // incompatible: not a broadcast of the target
    }

    mlir::Value reduced = axes.empty() ? grad : reduceWithBody(grad, axes, StableHLOOp::REDUCE_SUM);
    if (!reduced) return nullptr;
    auto rType = mlir::cast<mlir::RankedTensorType>(reduced.getType());
    if (rType.getShape() == tShape) return reduced;
    // The reduce DROPPED the summed axes; reshape puts the target's size-1
    // dimensions back and restores its rank.
    return reshapeTo(reduced, tShape);
}

/** @brief True if `v` is a ranked tensor of a floating-point element type,
 *         i.e. a value that can carry a gradient at all. This single check is
 *         what keeps integer index tensors, boolean masks and iota position
 *         tensors out of the backward pass without needing a special case in
 *         every rule. */
bool StableHLOEmitter::Impl::isFloatTensor(mlir::Value v) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    return t && mlir::isa<mlir::FloatType>(t.getElementType());
}

/** @brief True if operand `index` of `op` is on a differentiable path.
 *         Used by BOTH the forward DFS (so the cone never walks into an index
 *         or predicate computation) and the backward pass (so no cotangent is
 *         ever accumulated onto one). The float-tensor test already excludes
 *         integer and boolean operands; the explicit per-op cases below state
 *         the intent so a future float-typed index tensor cannot quietly
 *         start receiving a gradient. */
bool StableHLOEmitter::Impl::isDifferentiableOperand(mlir::Operation* op, unsigned index) {
    if (index >= op->getNumOperands()) return false;
    if (!isFloatTensor(op->getOperand(index))) return false;
    if (mlir::isa<mlir::stablehlo::CompareOp>(op)) return false;              // result is boolean
    if (mlir::isa<mlir::stablehlo::SelectOp>(op)) return index != 0;          // 0 = predicate
    if (mlir::isa<mlir::stablehlo::GatherOp>(op)) return index == 0;          // 1 = start_indices
    if (mlir::isa<mlir::stablehlo::DynamicSliceOp>(op)) return index == 0;    // 1.. = start indices
    if (mlir::isa<mlir::stablehlo::DynamicUpdateSliceOp>(op)) return index < 2; // 2.. = start indices
    if (mlir::isa<mlir::stablehlo::ScatterOp>(op)) return index != 1;         // 1 = scatter_indices
    return true;
}

/** @brief Add `contrib` into the accumulated cotangent of `v`. First
 *         contribution is stored as-is; every later one becomes a
 *         `stablehlo.add`, which is the summation reverse mode requires when
 *         a value has more than one consumer. Refuses (and records a
 *         diagnostic) if a contribution's shape does not match the value's —
 *         that means a VJP rule computed the wrong shape, and emitting the
 *         add anyway would produce invalid IR at best and a mis-broadcast
 *         gradient at worst. */
bool StableHLOEmitter::Impl::accumulateGrad(GradMap& grads, mlir::Value v, mlir::Value contrib) {
    if (!contrib) {
        vjp_diag_ = "gradient rule produced a null contribution";
        return false;
    }
    auto vType = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    auto cType = mlir::dyn_cast<mlir::RankedTensorType>(contrib.getType());
    if (!vType || !cType || vType.getShape() != cType.getShape()) {
        vjp_diag_ = "gradient contribution shape does not match the value it is accumulated onto";
        return false;
    }
    auto it = grads.find(v);
    if (it == grads.end()) {
        grads[v] = contrib;
        return true;
    }
    auto summed = addV(it->second, contrib);
    if (!summed) {
        vjp_diag_ = "failed to emit the accumulating add for a multi-consumer value";
        return false;
    }
    grads[v] = summed;
    return true;
}

/** @brief VJP of `stablehlo.dot_general`, generalised to arbitrary
 *         dot_dimension_numbers rather than the 2-D `dA = g @ B^T`,
 *         `dB = A^T @ g` special case.
 *
 *         The forward result is laid out as
 *           [batch dims][lhs free dims ascending][rhs free dims ascending]
 *         so the cotangent `g` has that same layout, and each gradient is
 *         another dot_general that contracts away the OTHER operand's free
 *         dims:
 *
 *           dLhs = dot(g, rhs)  contracting g's rhs-free block with rhs's
 *                               free dims, batching over the batch dims
 *           dRhs = dot(g, lhs)  contracting g's lhs-free block with lhs's
 *                               free dims, batching over the batch dims
 *
 *         For the plain 2-D case (Lc={1}, Rc={0}, no batch) the first is
 *         literally `g @ B^T` and the second `A^T @ g`; the machinery below is
 *         what makes it also correct for a batched attention contraction.
 *
 *         The subtle part is the LAYOUT of each product. dot_general emits its
 *         result in [batch][lhs free][rhs free] order, so `dot(g, rhs)` comes
 *         back as [batch][lhs free dims][the lhs dims paired with rhs's
 *         contracting dims, in ASCENDING RHS-DIM order] — which is generally
 *         NOT the lhs's own dimension order. `producedToLhs` records which lhs
 *         dimension each produced position corresponds to and the transpose
 *         inverts it. Skipping that transpose is a gradient that has the right
 *         shape whenever the contraction happens to be symmetric and the wrong
 *         one otherwise, which is exactly the kind of bug that only shows up
 *         as a model that will not converge. */
bool StableHLOEmitter::Impl::vjpDotGeneral(mlir::stablehlo::DotGeneralOp op, mlir::Value g,
                                            GradMap& grads) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    auto gType = mlir::dyn_cast<mlir::RankedTensorType>(g.getType());
    if (!lhsType || !rhsType || !gType) {
        vjp_diag_ = "dot_general: operands must be ranked tensors";
        return false;
    }

    auto dn = op.getDotDimensionNumbers();
    // UNCERTAIN (low risk): these accessors are assumed to return
    // ArrayRef<int64_t>, matching the DenseI64ArrayAttr-era StableHLO that
    // this file's forward emitters already assume (emitTranspose builds its
    // permutation with getDenseI64ArrayAttr). If this build pins an older
    // StableHLO where the dimension numbers are DenseIntElementsAttr, these
    // four lines need `llvm::to_vector(attr.getValues<int64_t>())` instead.
    std::vector<int64_t> lb(dn.getLhsBatchingDimensions().begin(),
                            dn.getLhsBatchingDimensions().end());
    std::vector<int64_t> rb(dn.getRhsBatchingDimensions().begin(),
                            dn.getRhsBatchingDimensions().end());
    std::vector<int64_t> lc(dn.getLhsContractingDimensions().begin(),
                            dn.getLhsContractingDimensions().end());
    std::vector<int64_t> rc(dn.getRhsContractingDimensions().begin(),
                            dn.getRhsContractingDimensions().end());
    if (lb.size() != rb.size() || lc.size() != rc.size()) {
        vjp_diag_ = "dot_general: batching/contracting dimensions do not pair up";
        return false;
    }

    const int64_t lhsRank = lhsType.getRank();
    const int64_t rhsRank = rhsType.getRank();
    auto member = [](const std::vector<int64_t>& v, int64_t x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };

    std::vector<int64_t> lf, rf;  // free (neither batched nor contracted), ascending
    for (int64_t i = 0; i < lhsRank; i++) if (!member(lb, i) && !member(lc, i)) lf.push_back(i);
    for (int64_t i = 0; i < rhsRank; i++) if (!member(rb, i) && !member(rc, i)) rf.push_back(i);

    const int64_t B = (int64_t)lb.size();
    const int64_t nlf = (int64_t)lf.size();
    const int64_t nrf = (int64_t)rf.size();
    if (gType.getRank() != B + nlf + nrf) {
        vjp_diag_ = "dot_general: cotangent rank does not match the dot's result layout";
        return false;
    }

    // Contraction pairs, sorted by each side's dimension index. dot_general
    // emits the surviving dims of an operand in ASCENDING order, so these are
    // the orders in which the contracted dims reappear in each product.
    std::vector<std::pair<int64_t, int64_t>> byRhs, byLhs;  // (that side's dim, the paired dim)
    for (size_t j = 0; j < lc.size(); j++) {
        byRhs.emplace_back(rc[j], lc[j]);
        byLhs.emplace_back(lc[j], rc[j]);
    }
    std::sort(byRhs.begin(), byRhs.end());
    std::sort(byLhs.begin(), byLhs.end());

    // Positions of each block inside the cotangent's layout.
    std::vector<int64_t> gBatchPos, gLhsFreePos, gRhsFreePos;
    for (int64_t k = 0; k < B; k++) gBatchPos.push_back(k);
    for (int64_t k = 0; k < nlf; k++) gLhsFreePos.push_back(B + k);
    for (int64_t k = 0; k < nrf; k++) gRhsFreePos.push_back(B + nlf + k);

    // ---- gradient w.r.t. the LHS ----
    if (isDifferentiableOperand(op, 0)) {
        auto prod = dotGeneral(g, rhs,
                               /*lhs_batch=*/gBatchPos, /*rhs_batch=*/rb,
                               /*lhs_contract=*/gRhsFreePos, /*rhs_contract=*/rf);
        if (!prod) {
            vjp_diag_ = "dot_general: could not emit the lhs-gradient contraction";
            return false;
        }
        // Which lhs dimension each produced position holds.
        std::vector<int64_t> producedToLhs;
        for (int64_t k = 0; k < B; k++) producedToLhs.push_back(lb[k]);
        for (auto f : lf) producedToLhs.push_back(f);
        for (auto& pr : byRhs) producedToLhs.push_back(pr.second);
        if ((int64_t)producedToLhs.size() != lhsRank) {
            vjp_diag_ = "dot_general: lhs-gradient layout does not cover every lhs dimension";
            return false;
        }
        std::vector<int64_t> perm(lhsRank, -1);
        for (int64_t pos = 0; pos < lhsRank; pos++) perm[producedToLhs[pos]] = pos;
        bool identity = true;
        for (int64_t i = 0; i < lhsRank; i++) {
            if (perm[i] < 0) {
                vjp_diag_ = "dot_general: lhs-gradient layout is not a permutation";
                return false;
            }
            if (perm[i] != i) identity = false;
        }
        mlir::Value dLhs = identity ? prod : transposeBy(prod, perm);
        if (!dLhs) {
            vjp_diag_ = "dot_general: could not emit the lhs-gradient transpose";
            return false;
        }
        if (!accumulateGrad(grads, lhs, dLhs)) return false;
    }

    // ---- gradient w.r.t. the RHS ----
    if (isDifferentiableOperand(op, 1)) {
        auto prod = dotGeneral(g, lhs,
                               /*lhs_batch=*/gBatchPos, /*rhs_batch=*/lb,
                               /*lhs_contract=*/gLhsFreePos, /*rhs_contract=*/lf);
        if (!prod) {
            vjp_diag_ = "dot_general: could not emit the rhs-gradient contraction";
            return false;
        }
        std::vector<int64_t> producedToRhs;
        for (int64_t k = 0; k < B; k++) producedToRhs.push_back(rb[k]);
        for (auto f : rf) producedToRhs.push_back(f);
        for (auto& pr : byLhs) producedToRhs.push_back(pr.second);
        if ((int64_t)producedToRhs.size() != rhsRank) {
            vjp_diag_ = "dot_general: rhs-gradient layout does not cover every rhs dimension";
            return false;
        }
        std::vector<int64_t> perm(rhsRank, -1);
        for (int64_t pos = 0; pos < rhsRank; pos++) perm[producedToRhs[pos]] = pos;
        bool identity = true;
        for (int64_t i = 0; i < rhsRank; i++) {
            if (perm[i] < 0) {
                vjp_diag_ = "dot_general: rhs-gradient layout is not a permutation";
                return false;
            }
            if (perm[i] != i) identity = false;
        }
        mlir::Value dRhs = identity ? prod : transposeBy(prod, perm);
        if (!dRhs) {
            vjp_diag_ = "dot_general: could not emit the rhs-gradient transpose";
            return false;
        }
        if (!accumulateGrad(grads, rhs, dRhs)) return false;
    }
    return true;
}

/** @brief VJP of `stablehlo.broadcast_in_dim` — the op that carries ALL real
 *         broadcasting in an Eshkol graph, and therefore the rule that has to
 *         be right or every bias and every layer-norm scale trains on a wrong
 *         gradient.
 *
 *         Two distinct classes of output dimension must be summed away:
 *           (a) output dims the operand never mapped onto (a [3] bias
 *               broadcast to [2,3] never spans axis 0 — sum over it), and
 *           (b) output dims the operand DID map onto but with extent 1 while
 *               the output is wider (a [1,3] broadcast to [2,3] maps onto axis
 *               0 with extent 1 — also a sum, and this is the case a naive
 *               implementation misses, because forgetting (a) produces a shape
 *               error the compiler catches while forgetting (b) produces the
 *               right shape and wrong numbers).
 *         Then, because broadcast_dimensions is a mapping and is not required
 *         to be increasing, the surviving dims come out in output-dim order
 *         and are transposed back into operand-dim order; finally a reshape
 *         restores the extent-1 dims dropped by the reduce. */
bool StableHLOEmitter::Impl::vjpBroadcastInDim(mlir::stablehlo::BroadcastInDimOp op,
                                                mlir::Value g, GradMap& grads) {
    auto operand = op->getOperand(0);
    auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto gType = mlir::dyn_cast<mlir::RankedTensorType>(g.getType());
    if (!operandType || !resultType || !gType) {
        vjp_diag_ = "broadcast_in_dim: operands must be ranked tensors";
        return false;
    }
    if (gType.getShape() != resultType.getShape()) {
        vjp_diag_ = "broadcast_in_dim: cotangent shape does not match the broadcast result";
        return false;
    }

    auto bcast = op.getBroadcastDimensions();
    const int64_t opRank = operandType.getRank();
    const int64_t resRank = resultType.getRank();
    if ((int64_t)bcast.size() != opRank) {
        vjp_diag_ = "broadcast_in_dim: broadcast_dimensions has one entry per operand dimension";
        return false;
    }

    auto opShape = operandType.getShape();
    auto resShape = resultType.getShape();

    // (result dim, operand dim) pairs, ordered by result dim — which is the
    // order the surviving dimensions come out of the reduce in.
    std::vector<std::pair<int64_t, int64_t>> mapped;
    std::vector<bool> spanned(resRank, false);
    for (int64_t i = 0; i < opRank; i++) {
        const int64_t d = bcast[i];
        if (d < 0 || d >= resRank) {
            vjp_diag_ = "broadcast_in_dim: broadcast dimension out of range";
            return false;
        }
        if (spanned[d]) {
            vjp_diag_ = "broadcast_in_dim: broadcast_dimensions are not unique";
            return false;
        }
        spanned[d] = true;
        mapped.emplace_back(d, i);
    }
    std::sort(mapped.begin(), mapped.end());

    std::vector<int64_t> axes;             // output dims summed away
    std::vector<int64_t> keptOperandDims;  // operand dims that survive, in output-dim order
    for (auto& pr : mapped) {
        const int64_t d = pr.first, i = pr.second;
        if (opShape[i] == 1 && resShape[d] != 1) {
            axes.push_back(d);             // case (b): stretched from extent 1
        } else if (opShape[i] != resShape[d]) {
            vjp_diag_ = "broadcast_in_dim: operand extent is neither 1 nor equal to the result's";
            return false;
        } else {
            keptOperandDims.push_back(i);
        }
    }
    for (int64_t d = 0; d < resRank; d++) if (!spanned[d]) axes.push_back(d);  // case (a)
    std::sort(axes.begin(), axes.end());

    mlir::Value reduced = axes.empty() ? g : reduceWithBody(g, axes, StableHLOOp::REDUCE_SUM);
    if (!reduced) {
        vjp_diag_ = "broadcast_in_dim: could not emit the sum over the broadcast dimensions";
        return false;
    }

    // The surviving dims are in output-dim order; put them in operand-dim order.
    std::vector<int64_t> sortedKept = keptOperandDims;
    std::sort(sortedKept.begin(), sortedKept.end());
    bool identity = (sortedKept == keptOperandDims);
    if (!identity) {
        std::vector<int64_t> perm(sortedKept.size());
        for (size_t j = 0; j < sortedKept.size(); j++) {
            auto it = std::find(keptOperandDims.begin(), keptOperandDims.end(), sortedKept[j]);
            perm[j] = (int64_t)std::distance(keptOperandDims.begin(), it);
        }
        reduced = transposeBy(reduced, perm);
        if (!reduced) {
            vjp_diag_ = "broadcast_in_dim: could not emit the un-permute transpose";
            return false;
        }
    }

    // Restore the extent-1 operand dimensions the reduce dropped.
    auto rType = mlir::cast<mlir::RankedTensorType>(reduced.getType());
    if (rType.getShape() != opShape) {
        reduced = reshapeTo(reduced, opShape);
        if (!reduced) {
            vjp_diag_ = "broadcast_in_dim: could not reshape the gradient back to the operand shape";
            return false;
        }
    }
    return accumulateGrad(grads, operand, reduced);
}

/** @brief VJP of `stablehlo.reduce`, dispatching on the combiner in the
 *         reduction body (the op itself carries no "which reduction" tag, so
 *         the body's single binary op IS the tag).
 *
 *         SUM: the cotangent is broadcast back over the reduced axes —
 *              every input element contributed exactly once.
 *         MAX/MIN: the cotangent goes ONLY to the positions that attained the
 *              extremum. The mask is built by comparing the input against the
 *              broadcast result, and the cotangent is divided by the number of
 *              tied positions before being scattered onto them, so a tie
 *              splits the gradient evenly rather than duplicating it once per
 *              tie (which is what a bare mask multiply does, and it inflates
 *              the gradient of any tensor with repeated maxima — very common
 *              after a ReLU, where a whole block of zeros ties). This matches
 *              JAX's even-split convention for max/min.
 *         PROD: deliberately NOT implemented. The textbook rule
 *              (grad * out / input) divides by the input and silently yields
 *              inf/NaN whenever any element is zero, and the numerically safe
 *              form needs a zero-count pass. Failing closed is better than a
 *              rule that is right except on the inputs people actually hit. */
bool StableHLOEmitter::Impl::vjpReduce(mlir::stablehlo::ReduceOp op, mlir::Value g,
                                        GradMap& grads) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
        vjp_diag_ = "reduce: variadic (multi-input) reductions have no VJP rule here";
        return false;
    }
    auto input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType) {
        vjp_diag_ = "reduce: input must be a ranked tensor";
        return false;
    }

    // Identify the combiner: the single op in the body block before its return.
    mlir::Operation* combiner = nullptr;
    auto& region = op.getBody();
    if (region.hasOneBlock()) {
        auto& blk = region.front();
        if (!blk.empty()) combiner = &blk.front();
    }
    if (!combiner) {
        vjp_diag_ = "reduce: could not inspect the reduction body to identify the combiner";
        return false;
    }

    auto axes = op.getDimensions();
    const int64_t rank = inputType.getRank();
    std::vector<int64_t> kept;  // input dims that survive the reduce, ascending
    for (int64_t d = 0; d < rank; d++) {
        bool reduced = false;
        for (auto ax : axes) if (ax == d) { reduced = true; break; }
        if (!reduced) kept.push_back(d);
    }
    auto inShape = inputType.getShape();

    if (mlir::isa<mlir::stablehlo::AddOp>(combiner)) {
        auto contrib = broadcastInDim(g, inShape, kept);
        if (!contrib) {
            vjp_diag_ = "reduce(sum): could not broadcast the cotangent back over the reduced axes";
            return false;
        }
        return accumulateGrad(grads, input, contrib);
    }

    const bool isMax = mlir::isa<mlir::stablehlo::MaxOp>(combiner);
    const bool isMin = mlir::isa<mlir::stablehlo::MinOp>(combiner);
    if (isMax || isMin) {
        auto elemType = inputType.getElementType();
        auto outB = broadcastInDim(op->getResult(0), inShape, kept);
        if (!outB) {
            vjp_diag_ = "reduce(max/min): could not broadcast the result for the argmax mask";
            return false;
        }
        // mask = 1 where this element attained the extremum, 0 elsewhere.
        auto maskPred = builder_->create<mlir::stablehlo::CompareOp>(
            loc(), input, outB, mlir::stablehlo::ComparisonDirection::EQ,
            mlir::stablehlo::ComparisonType::NOTYPE).getResult();
        auto mask = convertElem(maskPred, elemType);
        auto counts = mask ? reduceWithBody(mask, axes, StableHLOOp::REDUCE_SUM) : nullptr;
        auto scaled = counts ? divV(g, counts) : nullptr;   // split ties evenly
        auto gB = scaled ? broadcastInDim(scaled, inShape, kept) : nullptr;
        auto contrib = gB ? mulV(gB, mask) : nullptr;
        if (!contrib) {
            vjp_diag_ = "reduce(max/min): could not emit the tie-split argmax routing";
            return false;
        }
        (void)isMin;  // the mask construction is identical for min
        return accumulateGrad(grads, input, contrib);
    }

    vjp_diag_ = "reduce: unsupported combiner (only sum/max/min have VJP rules; "
                "product is intentionally refused because grad*out/input is "
                "undefined wherever the input contains a zero)";
    return false;
}

/** @brief Apply the VJP rule for one forward op: given the cotangent `g` of
 *         its result, emit the StableHLO that computes each differentiable
 *         operand's contribution and accumulate it.
 *
 *         Every elementwise contribution is routed through
 *         reduceGradToShape() — see its comment for why that is a guard and
 *         not the broadcasting mechanism. Returns false (with vjp_diag_ set)
 *         for any op without a rule; the caller aborts the whole request
 *         rather than returning a gradient that is missing a term. */
bool StableHLOEmitter::Impl::vjpForOp(mlir::Operation* op, mlir::Value g, GradMap& grads) {
    auto& b = *builder_;
    auto l = loc();

    // ----- values with no inputs: nothing to propagate into -----
    if (mlir::isa<mlir::stablehlo::ConstantOp>(op) || mlir::isa<mlir::stablehlo::IotaOp>(op)) {
        return true;
    }

    auto typeOf = [](mlir::Value v) {
        return mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    };
    // True if every operand of this elementwise op already has the result's
    // shape, i.e. the op is a well-formed StableHLO elementwise op. The
    // rules that build a contribution FROM an operand (multiply, divide,
    // max/min) need this: reduceGradToShape() can un-broadcast a contribution
    // after the fact, but it cannot rescue a `g * b` whose two sides had
    // different shapes to begin with — that product is already malformed. So
    // those rules refuse rather than emit it. add/subtract are unaffected,
    // since their contribution is the cotangent itself.
    auto operandsMatchResult = [&]() {
        auto resType = typeOf(op->getResult(0));
        if (!resType) return false;
        for (unsigned i = 0; i < op->getNumOperands(); i++) {
            auto t = typeOf(op->getOperand(i));
            if (!t || t.getShape() != resType.getShape()) return false;
        }
        return true;
    };
    // Accumulate `contrib` onto operand `idx`, un-broadcasting first.
    auto push = [&](unsigned idx, mlir::Value contrib, const char* what) -> bool {
        if (!isDifferentiableOperand(op, idx)) return true;
        if (!contrib) { vjp_diag_ = std::string("failed to emit ") + what; return false; }
        auto target = typeOf(op->getOperand(idx));
        if (!target) { vjp_diag_ = std::string("non-ranked operand in ") + what; return false; }
        auto reduced = reduceGradToShape(contrib, target);
        if (!reduced) {
            vjp_diag_ = std::string("could not reconcile the cotangent shape in ") + what;
            return false;
        }
        return accumulateGrad(grads, op->getOperand(idx), reduced);
    };

    // ----- 1. elementwise arithmetic -----
    if (mlir::isa<mlir::stablehlo::AddOp>(op)) {
        // d(a+b)/da = g, d(a+b)/db = g
        return push(0, g, "add lhs gradient") && push(1, g, "add rhs gradient");
    }
    if (mlir::isa<mlir::stablehlo::SubtractOp>(op)) {
        // d(a-b)/da = g, d(a-b)/db = -g
        return push(0, g, "subtract lhs gradient") &&
               push(1, negate(g), "subtract rhs gradient");
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(op)) {
        // d(a*b)/da = g*b, d(a*b)/db = g*a
        if (!operandsMatchResult()) {
            vjp_diag_ = "multiply: operand shapes differ from the result shape; the gradient "
                        "needs the operands broadcast explicitly (emit a broadcast_in_dim in "
                        "the forward pass) rather than an implicitly broadcast product";
            return false;
        }
        return push(0, mulV(g, op->getOperand(1)), "multiply lhs gradient") &&
               push(1, mulV(g, op->getOperand(0)), "multiply rhs gradient");
    }
    if (mlir::isa<mlir::stablehlo::DivOp>(op)) {
        // d(a/b)/da = g/b, d(a/b)/db = -g*a/b^2, computed as -(g * (a/b)) / b
        // reusing the forward quotient so the square is never formed.
        if (!operandsMatchResult()) {
            vjp_diag_ = "divide: operand shapes differ from the result shape; the gradient "
                        "needs the operands broadcast explicitly (emit a broadcast_in_dim in "
                        "the forward pass) rather than an implicitly broadcast quotient";
            return false;
        }
        auto out = op->getResult(0);
        auto db = negate(divV(mulV(g, out), op->getOperand(1)));
        return push(0, divV(g, op->getOperand(1)), "divide lhs gradient") &&
               push(1, db, "divide rhs gradient");
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(op) || mlir::isa<mlir::stablehlo::MinOp>(op)) {
        // Elementwise max/min: the gradient goes to whichever operand won, and
        // a tie is SPLIT EVENLY (weight 0.5 each) rather than handed entirely
        // to the lhs. This matches JAX's convention and, more importantly,
        // matches the tie handling in vjpReduce() above — max(x, 0) as a ReLU
        // ties on a whole block of zeros, and the two rules disagreeing there
        // would be a gradient that depends on how the program was spelled.
        if (!operandsMatchResult()) {
            vjp_diag_ = "max/min: operand shapes differ from the result shape; the gradient "
                        "needs the operands broadcast explicitly in the forward pass";
            return false;
        }
        const bool isMax = mlir::isa<mlir::stablehlo::MaxOp>(op);
        auto a = op->getOperand(0);
        auto bb = op->getOperand(1);
        auto aType = typeOf(a);
        if (!aType) { vjp_diag_ = "max/min: operands must be ranked tensors"; return false; }
        auto winPred = b.create<mlir::stablehlo::CompareOp>(
            l, a, bb,
            isMax ? mlir::stablehlo::ComparisonDirection::GT
                  : mlir::stablehlo::ComparisonDirection::LT,
            mlir::stablehlo::ComparisonType::NOTYPE).getResult();
        auto eqPred = b.create<mlir::stablehlo::CompareOp>(
            l, a, bb, mlir::stablehlo::ComparisonDirection::EQ,
            mlir::stablehlo::ComparisonType::NOTYPE).getResult();
        auto elemType = aType.getElementType();
        auto win = convertElem(winPred, elemType);
        auto eq = convertElem(eqPred, elemType);
        auto half = constantSplat(aType, 0.5);
        auto one = constantSplat(aType, 1.0);
        auto wA = (win && eq && half) ? addV(win, mulV(eq, half)) : nullptr;   // 1 / 0.5 / 0
        auto wB = (wA && one) ? subV(one, wA) : nullptr;
        if (!wA || !wB) { vjp_diag_ = "max/min: could not emit the tie-split weights"; return false; }
        return push(0, mulV(g, wA), "max/min lhs gradient") &&
               push(1, mulV(g, wB), "max/min rhs gradient");
    }

    // ----- 2. contraction -----
    if (auto dot = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
        return vjpDotGeneral(dot, g, grads);
    }

    // ----- 3. reduction -----
    if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
        return vjpReduce(red, g, grads);
    }

    // ----- 4. transpose -----
    if (auto tr = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
        // The VJP of a permutation is the INVERSE permutation, not the same
        // one: inv[perm[i]] = i. They coincide only for rank 2 (and other
        // self-inverse permutations), which is exactly why a 2-D-only
        // implementation looks correct until the first attention transpose.
        auto perm = tr.getPermutation();
        std::vector<int64_t> inv(perm.size(), -1);
        for (size_t i = 0; i < perm.size(); i++) {
            if (perm[i] < 0 || perm[i] >= (int64_t)perm.size()) {
                vjp_diag_ = "transpose: permutation out of range";
                return false;
            }
            inv[perm[i]] = (int64_t)i;
        }
        return push(0, transposeBy(g, inv), "transpose gradient");
    }

    // ----- 5. shape -----
    if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
        auto inType = typeOf(op->getOperand(0));
        if (!inType) { vjp_diag_ = "reshape: operand must be a ranked tensor"; return false; }
        return push(0, reshapeTo(g, inType.getShape()), "reshape gradient");
    }
    if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
        return vjpBroadcastInDim(bc, g, grads);
    }

    // ----- 6. transcendental (chain rule) -----
    if (mlir::isa<mlir::stablehlo::ExpOp>(op)) {
        // d(exp x) = g * exp(x); reuse the forward result rather than
        // recomputing the exponential.
        return push(0, mulV(g, op->getResult(0)), "exp gradient");
    }
    if (mlir::isa<mlir::stablehlo::LogOp>(op)) {
        // d(log x) = g / x
        return push(0, divV(g, op->getOperand(0)), "log gradient");
    }
    if (mlir::isa<mlir::stablehlo::TanhOp>(op)) {
        // d(tanh x) = g * (1 - tanh(x)^2), again from the forward result.
        auto out = op->getResult(0);
        auto outType = typeOf(out);
        if (!outType) { vjp_diag_ = "tanh: result must be a ranked tensor"; return false; }
        auto one = constantSplat(outType, 1.0);
        auto deriv = one ? subV(one, mulV(out, out)) : nullptr;
        return push(0, mulV(g, deriv), "tanh gradient");
    }
    if (mlir::isa<mlir::stablehlo::SineOp>(op)) {
        // d(sin x) = g * cos(x)
        auto x = op->getOperand(0);
        auto cosx = b.create<mlir::stablehlo::CosineOp>(l, x.getType(), x).getResult();
        return push(0, mulV(g, cosx), "sin gradient");
    }
    if (mlir::isa<mlir::stablehlo::CosineOp>(op)) {
        // d(cos x) = -g * sin(x)
        auto x = op->getOperand(0);
        auto sinx = b.create<mlir::stablehlo::SineOp>(l, x.getType(), x).getResult();
        return push(0, negate(mulV(g, sinx)), "cos gradient");
    }

    // ----- 7. gather -> scatter-add: THE EMBEDDING GRADIENT -----
    if (auto gath = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op)) {
        // The transpose of a gather is a scatter-ADD into a zero tensor of the
        // table's shape, with the dimension numbers carried straight across:
        //   update_window_dims          <- offset_dims
        //   inserted_window_dims        <- collapsed_slice_dims
        //   scatter_dims_to_operand_dims<- start_index_map
        //   index_vector_dim            <- index_vector_dim
        // and the cotangent, which already has the gather's result shape, as
        // the updates. The combiner MUST be add: a token repeated in a batch
        // is gathered once per occurrence and its row has to receive the SUM
        // of those cotangents. A replace-style scatter keeps only the last
        // occurrence, which trains embeddings on a fraction of their gradient
        // and looks like a slightly-too-low learning rate rather than a bug.
        //
        // CAVEAT (exactness): gather CLAMPS out-of-range start indices while
        // scatter DROPS out-of-range updates, so for a graph that actually
        // indexes out of bounds this transpose is not exact — the clamped
        // read contributes to the forward value but its cotangent is
        // discarded. Every in-bounds index (which is what a vocabulary lookup
        // is) transposes exactly.
        auto operand = op->getOperand(0);
        auto indices = op->getOperand(1);
        auto gdn = gath.getDimensionNumbers();
        auto sdn = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
            ctx_.get(),
            /*updateWindowDims=*/gdn.getOffsetDims(),
            /*insertedWindowDims=*/gdn.getCollapsedSliceDims(),
            /*inputBatchingDims=*/{},
            /*scatterIndicesBatchingDims=*/{},
            /*scatterDimsToOperandDims=*/gdn.getStartIndexMap(),
            /*indexVectorDim=*/gdn.getIndexVectorDim());
        auto zeros = zerosLike(operand);
        auto contrib = zeros ? scatterAdd(zeros, indices, g, sdn) : nullptr;
        if (!contrib) {
            vjp_diag_ = "gather: could not emit the scatter-add embedding gradient";
            return false;
        }
        return accumulateGrad(grads, operand, contrib);
    }

    // ----- 8. slicing -----
    if (auto sl = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
        // The transpose of a static slice is a pad: drop the cotangent back
        // into a zero tensor of the operand's shape at the position it was
        // taken from. A strided slice needs INTERIOR padding of stride-1 to
        // put the elements back on their original lattice; ignoring the
        // stride packs the gradient into the first rows of the operand, which
        // is silently wrong for anything that strides (e.g. de-interleaving
        // heads).
        auto operand = op->getOperand(0);
        auto inType = typeOf(operand);
        auto gType = typeOf(g);
        if (!inType || !gType) { vjp_diag_ = "slice: operands must be ranked tensors"; return false; }
        auto start = sl.getStartIndices();
        auto strides = sl.getStrides();
        auto inShape = inType.getShape();
        auto gShape = gType.getShape();
        const size_t rank = (size_t)inType.getRank();
        if (start.size() != rank || strides.size() != rank || gShape.size() != rank) {
            vjp_diag_ = "slice: rank mismatch between the slice bounds and the operand";
            return false;
        }
        std::vector<int64_t> low(rank), high(rank), interior(rank);
        for (size_t i = 0; i < rank; i++) {
            if (strides[i] <= 0) { vjp_diag_ = "slice: non-positive stride"; return false; }
            low[i] = start[i];
            interior[i] = strides[i] - 1;
            const int64_t last = start[i] + (gShape[i] - 1) * strides[i];
            high[i] = inShape[i] - last - 1;
            if (high[i] < 0 || low[i] < 0) {
                vjp_diag_ = "slice: cotangent does not fit back inside the operand shape";
                return false;
            }
        }
        auto zero = constantScalar(inType.getElementType(), 0.0);
        if (!zero) { vjp_diag_ = "slice: could not build the zero pad value"; return false; }
        auto contrib = b.create<mlir::stablehlo::PadOp>(
            l, inType, g, zero,
            b.getDenseI64ArrayAttr(low),
            b.getDenseI64ArrayAttr(high),
            b.getDenseI64ArrayAttr(interior)).getResult();
        return accumulateGrad(grads, operand, contrib);
    }
    if (auto ds = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op)) {
        // The transpose of a runtime-indexed read is a runtime-indexed write
        // of the cotangent into a zero tensor. Both ops clamp their start
        // indices the same way, so the round trip is consistent.
        auto operand = op->getOperand(0);
        llvm::SmallVector<mlir::Value> starts;
        for (unsigned i = 1; i < op->getNumOperands(); i++) starts.push_back(op->getOperand(i));
        auto inType = typeOf(operand);
        if (!inType) { vjp_diag_ = "dynamic_slice: operand must be a ranked tensor"; return false; }
        auto zeros = zerosLike(operand);
        if (!zeros) { vjp_diag_ = "dynamic_slice: could not build the zero tensor"; return false; }
        auto contrib = b.create<mlir::stablehlo::DynamicUpdateSliceOp>(
            l, inType, zeros, g, mlir::ValueRange(starts)).getResult();
        (void)ds;
        return accumulateGrad(grads, operand, contrib);
    }
    if (mlir::isa<mlir::stablehlo::DynamicUpdateSliceOp>(op)) {
        // operand: the cotangent everywhere EXCEPT the written window, which
        //          the write overwrote and so contributes nothing there.
        // update:  the cotangent restricted to the written window.
        auto operand = op->getOperand(0);
        auto update = op->getOperand(1);
        llvm::SmallVector<mlir::Value> starts;
        for (unsigned i = 2; i < op->getNumOperands(); i++) starts.push_back(op->getOperand(i));
        auto inType = typeOf(operand);
        auto upType = typeOf(update);
        if (!inType || !upType) {
            vjp_diag_ = "dynamic_update_slice: operands must be ranked tensors";
            return false;
        }
        if (isDifferentiableOperand(op, 0)) {
            auto zeroWindow = zerosLike(update);
            if (!zeroWindow) {
                vjp_diag_ = "dynamic_update_slice: could not build the zero window";
                return false;
            }
            auto contrib = b.create<mlir::stablehlo::DynamicUpdateSliceOp>(
                l, inType, g, zeroWindow, mlir::ValueRange(starts)).getResult();
            if (!accumulateGrad(grads, operand, contrib)) return false;
        }
        if (isDifferentiableOperand(op, 1)) {
            auto contrib = b.create<mlir::stablehlo::DynamicSliceOp>(
                l, upType, g, mlir::ValueRange(starts),
                b.getDenseI64ArrayAttr(upType.getShape())).getResult();
            if (!accumulateGrad(grads, update, contrib)) return false;
        }
        return true;
    }
    if (auto cat = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
        // Each input gets the slab of the cotangent it contributed. Needed for
        // any model that splits and rejoins attention heads.
        const int64_t dim = (int64_t)cat.getDimension();
        auto gType = typeOf(g);
        if (!gType) { vjp_diag_ = "concatenate: cotangent must be a ranked tensor"; return false; }
        const int64_t rank = gType.getRank();
        if (dim < 0 || dim >= rank) {
            vjp_diag_ = "concatenate: dimension out of range";
            return false;
        }
        int64_t offset = 0;
        for (unsigned i = 0; i < op->getNumOperands(); i++) {
            auto inType = typeOf(op->getOperand(i));
            if (!inType || inType.getRank() != rank) {
                vjp_diag_ = "concatenate: input rank does not match the result";
                return false;
            }
            std::vector<int64_t> start(rank, 0), limit(gType.getShape().begin(),
                                                       gType.getShape().end());
            std::vector<int64_t> strides(rank, 1);
            start[dim] = offset;
            limit[dim] = offset + inType.getShape()[dim];
            offset += inType.getShape()[dim];
            if (!isDifferentiableOperand(op, i)) continue;
            auto contrib = sliceOf(g, start, limit, strides);
            if (!contrib) {
                vjp_diag_ = "concatenate: could not slice the cotangent for an input";
                return false;
            }
            if (!accumulateGrad(grads, op->getOperand(i), contrib)) return false;
        }
        return true;
    }
    if (auto pad = mlir::dyn_cast<mlir::stablehlo::PadOp>(op)) {
        // The transpose of a pad is a slice back out of the padded region,
        // stepping by interior+1 to skip the inserted elements.
        auto operand = op->getOperand(0);
        auto padValue = op->getOperand(1);
        auto inType = typeOf(operand);
        if (!inType) { vjp_diag_ = "pad: operand must be a ranked tensor"; return false; }
        auto low = pad.getEdgePaddingLow();
        auto high = pad.getEdgePaddingHigh();
        auto interior = pad.getInteriorPadding();
        const size_t rank = (size_t)inType.getRank();
        if (low.size() != rank || high.size() != rank || interior.size() != rank) {
            vjp_diag_ = "pad: padding arrays do not match the operand rank";
            return false;
        }
        auto inShape = inType.getShape();
        std::vector<int64_t> start(rank), limit(rank), strides(rank);
        for (size_t i = 0; i < rank; i++) {
            if (low[i] < 0 || high[i] < 0) {
                // StableHLO permits NEGATIVE edge padding, which crops rather
                // than grows. The cotangent for cropped-away elements is zero,
                // but the slice-back below cannot express that, so refuse
                // instead of emitting a plausible wrong answer.
                vjp_diag_ = "pad: negative (cropping) edge padding has no VJP rule here";
                return false;
            }
            if (interior[i] < 0) { vjp_diag_ = "pad: negative interior padding"; return false; }
            strides[i] = interior[i] + 1;
            start[i] = low[i];
            limit[i] = low[i] + (inShape[i] > 0 ? (inShape[i] - 1) * strides[i] + 1 : 0);
        }
        auto inner = sliceOf(g, start, limit, strides);
        if (!inner) { vjp_diag_ = "pad: could not slice the cotangent back out"; return false; }
        if (isDifferentiableOperand(op, 0)) {
            if (!accumulateGrad(grads, operand, inner)) return false;
        }
        if (isDifferentiableOperand(op, 1)) {
            // The pad value was replicated into every position the operand did
            // not occupy, so its gradient is the sum of the cotangent over
            // exactly those positions: total - (the part that came from the
            // operand). It is a constant in every graph Eshkol emits today, so
            // this is dead code XLA will remove — but computing it means a
            // future learned pad value cannot silently receive zero.
            std::vector<int64_t> allDims;
            auto gType = typeOf(g);
            if (!gType) { vjp_diag_ = "pad: cotangent must be a ranked tensor"; return false; }
            for (int64_t d = 0; d < gType.getRank(); d++) allDims.push_back(d);
            std::vector<int64_t> innerDims;
            for (size_t d = 0; d < rank; d++) innerDims.push_back((int64_t)d);
            auto total = reduceWithBody(g, allDims, StableHLOOp::REDUCE_SUM);
            auto fromOperand = reduceWithBody(inner, innerDims, StableHLOOp::REDUCE_SUM);
            auto contrib = (total && fromOperand) ? subV(total, fromOperand) : nullptr;
            if (!contrib) { vjp_diag_ = "pad: could not emit the pad-value gradient"; return false; }
            if (!accumulateGrad(grads, padValue, contrib)) return false;
        }
        return true;
    }

    // ----- 9. select -----
    if (mlir::isa<mlir::stablehlo::SelectOp>(op)) {
        // Route the cotangent down the branch that was taken and zero the
        // other; the predicate itself is boolean and carries no gradient.
        auto pred = op->getOperand(0);
        auto onTrue = op->getOperand(1);
        auto onFalse = op->getOperand(2);
        auto zeros = zerosLike(onTrue);
        if (!zeros) { vjp_diag_ = "select: could not build the zero branch"; return false; }
        if (isDifferentiableOperand(op, 1)) {
            auto contrib = b.create<mlir::stablehlo::SelectOp>(
                l, onTrue.getType(), pred, g, zeros).getResult();
            if (!accumulateGrad(grads, onTrue, contrib)) return false;
        }
        if (isDifferentiableOperand(op, 2)) {
            auto contrib = b.create<mlir::stablehlo::SelectOp>(
                l, onFalse.getType(), pred, zeros, g).getResult();
            if (!accumulateGrad(grads, onFalse, contrib)) return false;
        }
        return true;
    }

    // ----- 10. dtype conversion -----
    if (mlir::isa<mlir::stablehlo::ConvertOp>(op)) {
        // Convert the cotangent back to the input's dtype. This is the op that
        // makes mixed precision differentiable: a bf16 weight converted to f32
        // for the matmul gets an f32 cotangent that must come back as bf16.
        auto operand = op->getOperand(0);
        auto inType = typeOf(operand);
        if (!inType) { vjp_diag_ = "convert: operand must be a ranked tensor"; return false; }
        if (!mlir::isa<mlir::FloatType>(inType.getElementType())) {
            // A float->int convert is a step function: zero gradient almost
            // everywhere, undefined on the steps. isDifferentiableOperand()
            // already stops us here; the check is kept explicit.
            return true;
        }
        return push(0, convertElem(g, inType.getElementType()), "convert gradient");
    }

    vjp_diag_ = "no VJP rule for '" + op->getName().getStringRef().str() +
                "' (refusing to emit an incomplete gradient)";
    return false;
}

/** @brief Run one reverse-mode sweep: collect the backward cone from
 *         `output`, seed it, walk it in reverse topological order applying
 *         VJP rules, and read off the cotangent of each `wrt` value.
 *
 *         Three passes, each doing one thing:
 *          1. CONE — iterative post-order DFS over differentiable operands
 *             only, so an index or predicate computation is never entered.
 *             Post-order on a DAG is a topological order.
 *          2. RELEVANCE — mark the ops that are downstream of some `wrt`
 *             value. Ops upstream of every parameter can still be in the cone
 *             (they feed the loss) but no cotangent through them can reach a
 *             parameter, so emitting their backward ops would be pure waste.
 *          3. BACKWARD — reverse walk, applying vjpForOp to each relevant op
 *             that has an accumulated cotangent.
 *
 *         A `wrt` value the output does not depend on yields an explicit zero
 *         tensor of its shape rather than a null: a training step wants a
 *         well-shaped zero it can feed to an optimizer, and the shape contract
 *         "one gradient per parameter, same shape" holds unconditionally. */
bool StableHLOEmitter::Impl::runVJP(mlir::Value output, llvm::ArrayRef<mlir::Value> wrt,
                                     mlir::Value seed,
                                     llvm::SmallVectorImpl<mlir::Value>& out) {
    vjp_diag_.clear();
    if (!isFloatTensor(output)) {
        vjp_diag_ = "output value is not a float-element ranked tensor and cannot be differentiated";
        return false;
    }
    auto outType = mlir::cast<mlir::RankedTensorType>(output.getType());
    auto seedType = mlir::dyn_cast<mlir::RankedTensorType>(seed.getType());
    if (!seedType || seedType.getShape() != outType.getShape()) {
        vjp_diag_ = "seed cotangent shape does not match the output shape";
        return false;
    }
    for (size_t i = 0; i < wrt.size(); i++) {
        if (!isFloatTensor(wrt[i])) {
            vjp_diag_ = "wrt value #" + std::to_string(i) +
                        " is not a float-element ranked tensor";
            return false;
        }
    }

    // ---- 1. cone: post-order DFS from the output ----
    llvm::SmallVector<mlir::Operation*, 64> topo;
    llvm::DenseSet<mlir::Operation*> visited;
    llvm::SmallVector<std::pair<mlir::Operation*, bool>, 64> stack;
    if (auto* def = output.getDefiningOp()) stack.push_back({def, false});
    while (!stack.empty()) {
        auto entry = stack.back();
        stack.pop_back();
        if (entry.second) { topo.push_back(entry.first); continue; }
        if (!visited.insert(entry.first).second) continue;
        stack.push_back({entry.first, true});
        for (unsigned i = 0; i < entry.first->getNumOperands(); i++) {
            if (!isDifferentiableOperand(entry.first, i)) continue;
            if (auto* d = entry.first->getOperand(i).getDefiningOp()) {
                if (!visited.count(d)) stack.push_back({d, false});
            }
        }
    }

    // ---- 2. relevance: which ops are downstream of a parameter ----
    llvm::DenseSet<mlir::Value> wrtSet;
    for (auto v : wrt) wrtSet.insert(v);
    llvm::DenseSet<mlir::Operation*> relevant;
    for (auto* op : topo) {  // forward (topological) order: producers first
        for (unsigned i = 0; i < op->getNumOperands(); i++) {
            if (!isDifferentiableOperand(op, i)) continue;
            auto v = op->getOperand(i);
            if (wrtSet.count(v) || (v.getDefiningOp() && relevant.count(v.getDefiningOp()))) {
                relevant.insert(op);
                break;
            }
        }
    }

    // ---- 3. backward sweep ----
    GradMap grads;
    grads[output] = seed;
    for (int64_t i = (int64_t)topo.size() - 1; i >= 0; i--) {
        auto* op = topo[i];
        if (!relevant.count(op)) continue;          // nothing downstream needs it
        if (op->getNumResults() != 1) {
            vjp_diag_ = "multi-result op '" + op->getName().getStringRef().str() +
                        "' on the differentiable path has no VJP rule";
            return false;
        }
        auto it = grads.find(op->getResult(0));
        if (it == grads.end()) continue;            // no cotangent reached this result
        if (!vjpForOp(op, it->second, grads)) return false;
    }

    // ---- 4. read off the requested gradients ----
    out.clear();
    for (auto v : wrt) {
        auto it = grads.find(v);
        if (it != grads.end()) { out.push_back(it->second); continue; }
        auto z = zerosLike(v);
        if (!z) {
            vjp_diag_ = "could not build a zero gradient for an unreachable wrt value";
            return false;
        }
        out.push_back(z);
    }
    return true;
}


#endif  // ESHKOL_XLA_FULL_MLIR

// ===== Constant / Shape Helpers (public) =====

/** @brief Emit a `stablehlo.broadcast_in_dim` with an explicit result shape —
 *         the shape-changing broadcast that emitBroadcast() cannot express
 *         (it reuses the input type as the result type). Returns nullptr on a
 *         broadcast_dims/rank mismatch or if MLIR support isn't available. */
void* StableHLOEmitter::emitBroadcastInDim(void* input, const std::vector<int64_t>& result_shape,
                                            const std::vector<int64_t>& broadcast_dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !input) return nullptr;
    auto result = impl_->broadcastInDim(impl_->toValue(input), result_shape, broadcast_dims);
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)input; (void)result_shape; (void)broadcast_dims;
    return nullptr;
#endif
}

/** @brief Emit a zero constant with the same type as `value`. Returns nullptr
 *         for an unsupported element type or if MLIR support isn't available. */
void* StableHLOEmitter::emitZerosLike(void* value) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !value) return nullptr;
    auto result = impl_->zerosLike(impl_->toValue(value));
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)value;
    return nullptr;
#endif
}

/** @brief Emit a ones constant with the same type as `value` — the default
 *         seed cotangent. Returns nullptr for an unsupported element type or
 *         if MLIR support isn't available. */
void* StableHLOEmitter::emitOnesLike(void* value) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !value) return nullptr;
    auto result = impl_->onesLike(impl_->toValue(value));
    if (!result) return nullptr;
    return impl_->storeValue(result);
#else
    (void)value;
    return nullptr;
#endif
}

// ===== Reverse-Mode Gradients (public entry point) =====

/** @brief Emit the reverse-mode VJP of `output` w.r.t. `wrt` as StableHLO ops
 *         in the same module as the forward pass. This is the op that makes
 *         Eshkol able to TRAIN on XLA hardware rather than only run a forward
 *         pass: the pre-existing XLACodegen::emitGradient() emitted LLVM IR,
 *         which XLA never sees, and covered matmul only.
 *
 *         Fails closed. If any op on the backward path has no rule, the
 *         result is `complete == false` with an EMPTY gradient vector and a
 *         diagnostic naming the op — never a partially-populated vector,
 *         because a caller that skips the check and trains on it gets a model
 *         that silently converges to garbage. */
VJPResult StableHLOEmitter::emitVJP(void* output, const std::vector<void*>& wrt, void* seed) {
    VJPResult result;
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) {
        result.diagnostic = "StableHLO emitter was built without MLIR support";
        return result;
    }
    if (!output) {
        result.diagnostic = "null output value";
        return result;
    }
    if (wrt.empty()) {
        result.diagnostic = "no wrt values were given";
        return result;
    }
    llvm::SmallVector<mlir::Value, 8> wrtVals;
    for (auto* w : wrt) {
        if (!w) {
            result.diagnostic = "null entry in the wrt list";
            return result;
        }
        wrtVals.push_back(impl_->toValue(w));
    }
    auto outputVal = impl_->toValue(output);
    mlir::Value seedVal = seed ? impl_->toValue(seed) : impl_->onesLike(outputVal);
    if (!seedVal) {
        result.diagnostic = "could not build a ones seed for the output";
        return result;
    }
    llvm::SmallVector<mlir::Value, 8> grads;
    if (!impl_->runVJP(outputVal, wrtVals, seedVal, grads)) {
        result.diagnostic = impl_->vjp_diag_.empty() ? "reverse-mode sweep failed"
                                                     : impl_->vjp_diag_;
        return result;  // gradients deliberately left empty
    }
    result.gradients.reserve(grads.size());
    for (auto v : grads) result.gradients.push_back(impl_->storeValue(v));
    result.complete = true;
    return result;
#else
    (void)output; (void)wrt; (void)seed;
    result.diagnostic = "StableHLO emitter was built without MLIR support";
    return result;
#endif
}

// ===== Module Management =====

/** @brief Return the underlying MLIR module as an opaque `Operation*`
 *         (stable across the module's lifetime, owned by the emitter).
 *         Returns nullptr if MLIR support isn't available. */
void* StableHLOEmitter::getModule() const {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !impl_->module_) return nullptr;
    // Return the underlying Operation* (stable, owned by OwningOpRef)
    return static_cast<void*>(impl_->module_.get().getOperation());
#else
    return nullptr;
#endif
}

/** @brief Serialize the current MLIR module to its textual IR form. Returns
 *         an empty string if MLIR support isn't available. */
std::string StableHLOEmitter::serializeToString() const {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !impl_->module_) return "";
    std::string result;
    llvm::raw_string_ostream os(result);
    impl_->module_.get().print(os);
    return result;
#else
    return "";
#endif
}

/** @brief Discard the emitted-value pool and start a fresh empty module,
 *         releasing all previously returned opaque mlir::Value handles.
 *         No-op if MLIR support isn't available. */
void StableHLOEmitter::reset() {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return;
    impl_->value_pool_.clear();
    impl_->module_ = mlir::ModuleOp::create(impl_->builder_->getUnknownLoc());
    // The old module (and the block the builder pointed into) is gone; without
    // re-anchoring, every op emitted after a reset() would be built detached.
    impl_->builder_->setInsertionPointToEnd(impl_->module_->getBody());
#endif
}

} // namespace xla
} // namespace eshkol
