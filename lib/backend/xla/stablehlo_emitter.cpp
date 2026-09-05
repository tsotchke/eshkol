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
#include <llvm/Support/raw_ostream.h>
#include <cmath>
#include <deque>
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
 *         shape from `dims`' batching/contracting dimensions (fast path for
 *         plain 2-D [M,K]x[K,N] matmul, general path otherwise). Returns
 *         nullptr if MLIR support isn't available. */
void* StableHLOEmitter::emitMatmul(void* lhs, void* rhs, const DotDimensionNumbers& dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto lhsVal = impl_->toValue(lhs);
    auto rhsVal = impl_->toValue(rhs);

    // Convert DotDimensionNumbers to MLIR attribute
    auto dotDimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
        impl_->ctx_.get(),
        dims.lhs_batching_dims,
        dims.rhs_batching_dims,
        dims.lhs_contracting_dims,
        dims.rhs_contracting_dims);

    // Infer output type from input types
    auto lhsType = mlir::cast<mlir::RankedTensorType>(lhsVal.getType());
    auto rhsType = mlir::cast<mlir::RankedTensorType>(rhsVal.getType());
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    // For standard 2D matmul: [M,K] x [K,N] -> [M,N]
    // General case: remove contracting dims, keep batch dims
    std::vector<int64_t> outShape;
    if (lhsShape.size() == 2 && rhsShape.size() == 2) {
        outShape = {lhsShape[0], rhsShape[1]};
    } else {
        // General case: batch dims + non-contracting dims
        for (auto d : dims.lhs_batching_dims)
            outShape.push_back(lhsShape[d]);
        for (int64_t i = 0; i < (int64_t)lhsShape.size(); i++) {
            bool is_batch = false, is_contract = false;
            for (auto d : dims.lhs_batching_dims) if (d == i) is_batch = true;
            for (auto d : dims.lhs_contracting_dims) if (d == i) is_contract = true;
            if (!is_batch && !is_contract) outShape.push_back(lhsShape[i]);
        }
        for (int64_t i = 0; i < (int64_t)rhsShape.size(); i++) {
            bool is_batch = false, is_contract = false;
            for (auto d : dims.rhs_batching_dims) if (d == i) is_batch = true;
            for (auto d : dims.rhs_contracting_dims) if (d == i) is_contract = true;
            if (!is_batch && !is_contract) outShape.push_back(rhsShape[i]);
        }
    }

    auto outType = mlir::RankedTensorType::get(outShape, lhsType.getElementType());
    auto dotOp = b.create<mlir::stablehlo::DotGeneralOp>(
        impl_->loc(), outType, lhsVal, rhsVal, dotDimNumbers,
        /*precision_config=*/nullptr,
        /*algorithm=*/nullptr);
    return impl_->storeValue(dotOp.getResult());
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
 *         (sum/prod/max/min): builds the appropriate identity-element
 *         constant, constructs the reduction body region (a single binary
 *         op matching `op`), and returns the reduced-shape result. Returns
 *         nullptr for an unsupported element type/op or if MLIR support
 *         isn't available. */
void* StableHLOEmitter::emitReduce(void* input, const std::vector<int64_t>& axes, StableHLOOp op) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto l = impl_->loc();
    auto inputVal = impl_->toValue(input);
    auto inputType = mlir::cast<mlir::RankedTensorType>(inputVal.getType());
    auto elemType = inputType.getElementType();

    // Create scalar tensor type for the reduction body
    auto scalarType = mlir::RankedTensorType::get({}, elemType);

    // Create identity element for reduction
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
        auto attr = mlir::DenseElementsAttr::get(
            scalarType, llvm::ArrayRef<double>{identity});
        initValue = b.create<mlir::stablehlo::ConstantOp>(l, attr);
    } else if (mlir::isa<mlir::IntegerType>(elemType)) {
        int64_t identity;
        switch (op) {
            case StableHLOOp::REDUCE_SUM:  identity = 0; break;
            case StableHLOOp::REDUCE_PROD: identity = 1; break;
            case StableHLOOp::REDUCE_MAX:  identity = INT64_MIN; break;
            case StableHLOOp::REDUCE_MIN:  identity = INT64_MAX; break;
            default: return nullptr;
        }
        auto attr = mlir::DenseElementsAttr::get(
            scalarType, llvm::ArrayRef<int64_t>{identity});
        initValue = b.create<mlir::stablehlo::ConstantOp>(l, attr);
    } else {
        return nullptr;  // Unsupported element type for reduction
    }

    // Compute output shape (remove reduced dimensions)
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

    // Create ReduceOp
    auto reduceOp = b.create<mlir::stablehlo::ReduceOp>(
        l, mlir::TypeRange{outType}, mlir::ValueRange{inputVal},
        mlir::ValueRange{initValue}, b.getDenseI64ArrayAttr(axes));

    // Build computation body region
    auto& body = reduceOp.getBody();
    auto* bodyBlock = b.createBlock(&body);
    bodyBlock->addArgument(scalarType, l);
    bodyBlock->addArgument(scalarType, l);

    // Save current insertion point, build body, restore
    auto savedInsertionPoint = b.saveInsertionPoint();
    b.setInsertionPointToStart(bodyBlock);
    auto arg0 = bodyBlock->getArgument(0);
    auto arg1 = bodyBlock->getArgument(1);

    mlir::Value bodyResult;
    switch (op) {
        case StableHLOOp::REDUCE_SUM:
            bodyResult = b.create<mlir::stablehlo::AddOp>(
                l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_PROD:
            bodyResult = b.create<mlir::stablehlo::MulOp>(
                l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_MAX:
            bodyResult = b.create<mlir::stablehlo::MaxOp>(
                l, scalarType, arg0, arg1).getResult();
            break;
        case StableHLOOp::REDUCE_MIN:
            bodyResult = b.create<mlir::stablehlo::MinOp>(
                l, scalarType, arg0, arg1).getResult();
            break;
        default:
            b.restoreInsertionPoint(savedInsertionPoint);
            return nullptr;
    }

    b.create<mlir::stablehlo::ReturnOp>(l, mlir::ValueRange{bodyResult});
    b.restoreInsertionPoint(savedInsertionPoint);

    return impl_->storeValue(reduceOp.getResult(0));
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
 *         were gathered more than once. Builds the update_computation
 *         region as a single `stablehlo.add`, mirroring how emitReduce
 *         builds its reduction body. Only the additive combiner is
 *         implemented (the embedding-gradient use case); a replace-style
 *         scatter would need a different body. Returns nullptr for an
 *         unsupported element type or if MLIR support isn't available. */
void* StableHLOEmitter::emitScatter(void* operand, void* scatter_indices, void* updates,
                                     const ScatterDimensionNumbers& dims) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return nullptr;
    auto& b = *impl_->builder_;
    auto l = impl_->loc();
    auto operandVal = impl_->toValue(operand);
    auto scatterIndicesVal = impl_->toValue(scatter_indices);
    auto updatesVal = impl_->toValue(updates);
    auto operandType = mlir::cast<mlir::RankedTensorType>(operandVal.getType());
    auto elemType = operandType.getElementType();

    if (!mlir::isa<mlir::FloatType>(elemType) && !mlir::isa<mlir::IntegerType>(elemType)) {
        return nullptr;  // Unsupported element type for add-combine
    }

    // Scatter's result has the same shape/type as `operand` — it is a
    // functional "updated copy", not a shape-changing op.
    auto outType = operandType;
    auto scalarType = mlir::RankedTensorType::get({}, elemType);

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

    // UNCERTAIN: same caveat as GatherOp above — ScatterOp has no
    // hand-declared `let builders`, so this uses the ODS default builder in
    // declared-argument order (results, inputs, scatter_indices, updates,
    // scatter_dimension_numbers, indices_are_sorted, unique_indices). I
    // could not check the generated header to confirm the two trailing
    // DefaultValuedOptionalAttr<BoolAttr> parameters are passed this way
    // (vs. an elided overload, vs. raw `bool`) — verify before compiling.
    auto scatterOp = b.create<mlir::stablehlo::ScatterOp>(
        l, mlir::TypeRange{outType}, mlir::ValueRange{operandVal},
        scatterIndicesVal, mlir::ValueRange{updatesVal}, scatterDimNumbers,
        /*indices_are_sorted=*/b.getBoolAttr(false),
        /*unique_indices=*/b.getBoolAttr(false));

    // Build the update_computation region: result = current + update, i.e.
    // scatter-add (accumulate), matching the embedding-gradient use case.
    auto& region = scatterOp.getUpdateComputation();
    auto* bodyBlock = b.createBlock(&region);
    bodyBlock->addArgument(scalarType, l);
    bodyBlock->addArgument(scalarType, l);

    auto savedInsertionPoint = b.saveInsertionPoint();
    b.setInsertionPointToStart(bodyBlock);
    auto arg0 = bodyBlock->getArgument(0);
    auto arg1 = bodyBlock->getArgument(1);
    auto sum = b.create<mlir::stablehlo::AddOp>(l, scalarType, arg0, arg1).getResult();
    b.create<mlir::stablehlo::ReturnOp>(l, mlir::ValueRange{sum});
    b.restoreInsertionPoint(savedInsertionPoint);

    return impl_->storeValue(scatterOp.getResult(0));
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
#endif
}

} // namespace xla
} // namespace eshkol
