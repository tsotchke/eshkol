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

StableHLOEmitter::StableHLOEmitter()
    : impl_(std::make_unique<Impl>()) {}

StableHLOEmitter::~StableHLOEmitter() = default;

bool StableHLOEmitter::isAvailable() const {
    return impl_->available_;
}

// ===== Arithmetic Operations =====

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

// ===== Module Management =====

void* StableHLOEmitter::getModule() const {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_ || !impl_->module_) return nullptr;
    // Return the underlying Operation* (stable, owned by OwningOpRef)
    return static_cast<void*>(impl_->module_.get().getOperation());
#else
    return nullptr;
#endif
}

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

void StableHLOEmitter::reset() {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) return;
    impl_->value_pool_.clear();
    impl_->module_ = mlir::ModuleOp::create(impl_->builder_->getUnknownLoc());
#endif
}

} // namespace xla
} // namespace eshkol
