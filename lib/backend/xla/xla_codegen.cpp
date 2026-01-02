/*
 * XLA Backend Codegen Implementation for Eshkol
 *
 * Provides accelerated tensor operations via XLA/StableHLO for massive tensors.
 * Falls back to BLAS/SIMD/scalar for smaller tensors.
 *
 * Dispatch hierarchy: XLA (≥100K) → cBLAS (≥4K) → SIMD (≥64) → scalar
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/codegen_context.h"
#include <cstdlib>
#include <sstream>
#include <vector>
#include <unordered_map>

// MLIR includes (conditional compilation)
// Only include when BOTH MLIR and StableHLO are available
// (no point initializing MLIR without StableHLO for actual XLA ops)
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
#endif

// LLVM includes for IR generation
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Constants.h>

namespace eshkol {
namespace xla {

// ===== Global Threshold =====

// Default threshold: 100000 elements (massive tensors only)
// XLA is reserved for very large tensors due to compilation overhead
// Dispatch hierarchy: XLA (≥100K) → cBLAS (≥4K) → SIMD (≥64) → scalar
// Can be overridden via ESHKOL_XLA_THRESHOLD environment variable
size_t g_xla_threshold = 100000;

void xla_set_threshold(size_t threshold) {
    g_xla_threshold = threshold;
}

size_t xla_get_threshold() {
    return g_xla_threshold;
}

// Initialize threshold from environment on startup
namespace {
    struct ThresholdInitializer {
        ThresholdInitializer() {
            if (const char* env = std::getenv("ESHKOL_XLA_THRESHOLD")) {
                g_xla_threshold = static_cast<size_t>(std::atol(env));
            }
        }
    };
    static ThresholdInitializer init;
}

// ===== XLACodegen Implementation =====

class XLACodegen::Impl {
public:
    CodegenContext* ctx_ = nullptr;
    size_t threshold_ = 100000;
    bool available_ = false;
    size_t op_counter_ = 0;  // For generating unique function names

#ifdef ESHKOL_XLA_FULL_MLIR
    // Full MLIR + StableHLO available - enable XLA
    std::unique_ptr<mlir::MLIRContext> mlir_ctx_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;

    // Cache of compiled operations for reuse
    std::unordered_map<std::string, llvm::Function*> compiled_ops_;

    explicit Impl(CodegenContext& ctx) : ctx_(&ctx), threshold_(g_xla_threshold) {
        // Initialize MLIR context
        mlir_ctx_ = std::make_unique<mlir::MLIRContext>();

        // Load required dialects
        mlir_ctx_->loadDialect<mlir::func::FuncDialect>();
        mlir_ctx_->loadDialect<mlir::arith::ArithDialect>();
        mlir_ctx_->loadDialect<mlir::stablehlo::StablehloDialect>();

        // Create builder
        builder_ = std::make_unique<mlir::OpBuilder>(mlir_ctx_.get());

        // Create module
        module_ = mlir::ModuleOp::create(builder_->getUnknownLoc());

        // Full XLA available with StableHLO
        available_ = true;
    }

    mlir::MLIRContext* getMLIRContext() { return mlir_ctx_.get(); }
    mlir::OpBuilder* getBuilder() { return builder_.get(); }
    mlir::ModuleOp getModule() { return module_.get(); }

    // Build a StableHLO matmul function
    // Returns the MLIR function for the operation
    mlir::func::FuncOp buildMatmulFunc(
        const std::vector<int64_t>& a_shape,
        const std::vector<int64_t>& b_shape,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();

        // Create tensor types for inputs and output
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto aType = mlir::RankedTensorType::get(a_shape, f64Type);
        auto bType = mlir::RankedTensorType::get(b_shape, f64Type);

        // Output shape: [a_shape[0], b_shape[1]] for 2D matmul
        std::vector<int64_t> out_shape;
        if (a_shape.size() == 2 && b_shape.size() == 2) {
            out_shape = {a_shape[0], b_shape[1]};
        } else {
            // General case: last dim of a contracts with second-to-last of b
            out_shape = a_shape;
            out_shape.back() = b_shape.back();
        }
        auto outType = mlir::RankedTensorType::get(out_shape, f64Type);

        // Create function type: (tensor<...>, tensor<...>) -> tensor<...>
        auto funcType = mlir::FunctionType::get(
            mlir_ctx_.get(),
            {aType, bType},
            {outType});

        // Create function
        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        // Create entry block
        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        // Get function arguments
        auto aArg = entryBlock->getArgument(0);
        auto bArg = entryBlock->getArgument(1);

        // Create dot_general operation for matrix multiplication
        // For 2D matmul: contract dimension 1 of a with dimension 0 of b
        auto dotDimNumbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
            mlir_ctx_.get(),
            /*lhsBatchingDimensions=*/{},
            /*rhsBatchingDimensions=*/{},
            /*lhsContractingDimensions=*/{static_cast<int64_t>(a_shape.size() - 1)},
            /*rhsContractingDimensions=*/{0});

        auto dotOp = builder.create<mlir::stablehlo::DotGeneralOp>(
            loc,
            outType,
            aArg,
            bArg,
            dotDimNumbers,
            /*precision_config=*/nullptr,
            /*algorithm=*/nullptr);

        // Return result
        builder.create<mlir::func::ReturnOp>(loc, dotOp.getResult());

        return funcOp;
    }

    // Get or create the XLA matmul runtime function declaration
    llvm::Function* getOrCreateMatmulRuntime() {
        auto& llvm_ctx = ctx_->context();
        auto& module = ctx_->module();

        const char* funcName = "eshkol_xla_matmul";

        // Check if already declared
        if (auto* existing = module.getFunction(funcName)) {
            return existing;
        }

        // Create function type: void* xla_matmul(void* a, void* b, i64* a_shape, i64* b_shape, i64 a_rank, i64 b_rank)
        // Returns pointer to result tensor
        auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
        auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

        std::vector<llvm::Type*> paramTypes = {
            ptrTy,  // arena
            ptrTy,  // a_data
            ptrTy,  // b_data
            ptrTy,  // a_shape
            ptrTy,  // b_shape
            i64Ty,  // a_rank
            i64Ty   // b_rank
        };

        auto* funcTy = llvm::FunctionType::get(ptrTy, paramTypes, false);
        auto* func = llvm::Function::Create(
            funcTy,
            llvm::GlobalValue::ExternalLinkage,
            funcName,
            &module);

        return func;
    }

#else
    // Fallback mode - MLIR or StableHLO not available
    // XLA backend will be stubs only, falls back to BLAS/SIMD/scalar
    explicit Impl(CodegenContext& ctx) : ctx_(&ctx), threshold_(g_xla_threshold) {
        available_ = false;
    }
#endif
};

XLACodegen::XLACodegen(CodegenContext& ctx)
    : impl_(std::make_unique<Impl>(ctx)) {}

XLACodegen::~XLACodegen() = default;

XLACodegen::XLACodegen(XLACodegen&&) noexcept = default;
XLACodegen& XLACodegen::operator=(XLACodegen&&) noexcept = default;

// ===== Backend Selection =====

void XLACodegen::setThreshold(size_t min_elements) {
    impl_->threshold_ = min_elements;
}

bool XLACodegen::shouldUseXLA(size_t num_elements) const {
    // Use XLA only when:
    // 1. MLIR/StableHLO is available (impl_->available_)
    // 2. Tensor is large enough to justify XLA overhead
    return impl_->available_ && num_elements >= impl_->threshold_;
}

// ===== Tensor Operations =====

llvm::Value* XLACodegen::emitMatmul(llvm::Value* a, llvm::Value* b) {
#ifdef ESHKOL_XLA_FULL_MLIR
    if (!impl_->available_) {
        return nullptr;
    }

    // Get LLVM IR builder
    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();

    // Get runtime function
    auto* matmulFunc = impl_->getOrCreateMatmulRuntime();

    // Get arena pointer
    auto* arenaPtrPtr = impl_->ctx_->globalArena();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* arenaPtr = builder.CreateLoad(ptrTy, arenaPtrPtr, "arena_ptr");

    // Tensor struct layout:
    // struct tensor { i64 num_dims; i64* dims; double* data; }
    // a and b are pointers to tensor structs

    // Extract data and shape from tensor a
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Get num_dims from tensor a (first field)
    auto* aNumDimsPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        a, 0, "a_num_dims_ptr");
    auto* aNumDims = builder.CreateLoad(i64Ty, aNumDimsPtr, "a_num_dims");

    // Get dims pointer from tensor a (second field)
    auto* aDimsPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        a, 1, "a_dims_ptr");
    auto* aDims = builder.CreateLoad(ptrTy, aDimsPtr, "a_dims");

    // Get data pointer from tensor a (third field)
    auto* aDataPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        a, 2, "a_data_ptr");
    auto* aData = builder.CreateLoad(ptrTy, aDataPtr, "a_data");

    // Extract data and shape from tensor b
    auto* bNumDimsPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        b, 0, "b_num_dims_ptr");
    auto* bNumDims = builder.CreateLoad(i64Ty, bNumDimsPtr, "b_num_dims");

    auto* bDimsPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        b, 1, "b_dims_ptr");
    auto* bDims = builder.CreateLoad(ptrTy, bDimsPtr, "b_dims");

    auto* bDataPtr = builder.CreateStructGEP(
        llvm::StructType::get(llvm_ctx, {i64Ty, ptrTy, ptrTy}),
        b, 2, "b_data_ptr");
    auto* bData = builder.CreateLoad(ptrTy, bDataPtr, "b_data");

    // Call runtime function
    std::vector<llvm::Value*> args = {
        arenaPtr,
        aData,
        bData,
        aDims,
        bDims,
        aNumDims,
        bNumDims
    };

    auto* result = builder.CreateCall(matmulFunc, args, "matmul_result");

    return result;
#else
    (void)a;
    (void)b;
    return nullptr;
#endif
}

llvm::Value* XLACodegen::emitElementwise(llvm::Value* a, llvm::Value* b, ElementwiseOp op) {
    // STUB: Not implemented yet
    (void)a;
    (void)b;
    (void)op;
    return nullptr;
}

llvm::Value* XLACodegen::emitReduce(llvm::Value* input, int64_t axis, ReduceOp op) {
    // STUB: Not implemented yet
    (void)input;
    (void)axis;
    (void)op;
    return nullptr;
}

// ===== Autodiff Integration (Stub) =====

llvm::Value* XLACodegen::emitGradient(llvm::Value* output_node,
                                       const std::vector<llvm::Value*>& wrt_vars) {
    // STUB: Not implemented yet
    (void)output_node;
    (void)wrt_vars;
    return nullptr;
}

// ===== Compilation (Stubs) =====

void XLACodegen::compile(Target target) {
    // STUB: Not implemented yet
    (void)target;
}

void* XLACodegen::getExecutable() const {
    // STUB: Not implemented yet
    return nullptr;
}

// ===== Memory Integration (Stubs) =====

llvm::Value* XLACodegen::wrapArenaBuffer(llvm::Value* arena_ptr, llvm::Value* tensor_ptr) {
    // STUB: Not implemented yet
    (void)arena_ptr;
    (void)tensor_ptr;
    return nullptr;
}

llvm::Value* XLACodegen::ensureDevice(llvm::Value* tensor_ptr, Target target) {
    // STUB: Not implemented yet
    (void)tensor_ptr;
    (void)target;
    return nullptr;
}

// ===== Status =====

bool XLACodegen::isAvailable() const {
    return impl_->available_;
}

std::string XLACodegen::getDescription() const {
    std::ostringstream oss;
    oss << "XLA Backend (";
    if (impl_->available_) {
        oss << "StableHLO available, threshold=" << impl_->threshold_ << " elements";
    } else {
#ifdef ESHKOL_XLA_FULL_MLIR
        oss << "initialized but disabled";
#elif defined(ESHKOL_MLIR_AVAILABLE)
        oss << "MLIR available, StableHLO not found - using fallback";
#else
        oss << "MLIR not available - using stubs";
#endif
    }
    oss << ")";
    return oss.str();
}

} // namespace xla
} // namespace eshkol
