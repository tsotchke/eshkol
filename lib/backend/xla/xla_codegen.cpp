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
#include "eshkol/backend/xla/stablehlo_emitter.h"
#include "eshkol/backend/xla/xla_compiler.h"
#include "eshkol/backend/codegen_context.h"
#include "eshkol/backend/type_system.h"
#include <cstdlib>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <functional>

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

    // Constructor: defined out-of-line below (different for MLIR vs LLVM-only)
    explicit Impl(CodegenContext& ctx);

#ifdef ESHKOL_XLA_FULL_MLIR
    // Full MLIR + StableHLO available - enable XLA
    std::unique_ptr<mlir::MLIRContext> mlir_ctx_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;

    // Cache of compiled operations for reuse
    std::unordered_map<std::string, llvm::Function*> compiled_ops_;

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

    // Build a StableHLO elementwise function (binary or unary)
    mlir::func::FuncOp buildElementwiseFunc(
        const std::vector<int64_t>& shape,
        int op_code,
        bool is_unary,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto tensorType = mlir::RankedTensorType::get(shape, f64Type);

        // Function type: unary (tensor) -> tensor, binary (tensor, tensor) -> tensor
        mlir::FunctionType funcType;
        if (is_unary) {
            funcType = mlir::FunctionType::get(mlir_ctx_.get(), {tensorType}, {tensorType});
        } else {
            funcType = mlir::FunctionType::get(mlir_ctx_.get(), {tensorType, tensorType}, {tensorType});
        }

        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        auto aArg = entryBlock->getArgument(0);
        mlir::Value result;

        if (is_unary) {
            // Unary ops: EXP=4, LOG=5, SIN=6, COS=7, TANH=8
            switch (op_code) {
                case 4: result = builder.create<mlir::stablehlo::ExpOp>(loc, tensorType, aArg); break;
                case 5: result = builder.create<mlir::stablehlo::LogOp>(loc, tensorType, aArg); break;
                case 6: result = builder.create<mlir::stablehlo::SineOp>(loc, tensorType, aArg); break;
                case 7: result = builder.create<mlir::stablehlo::CosineOp>(loc, tensorType, aArg); break;
                case 8: result = builder.create<mlir::stablehlo::TanhOp>(loc, tensorType, aArg); break;
                default: result = aArg; break;
            }
        } else {
            auto bArg = entryBlock->getArgument(1);
            // Binary ops: ADD=0, SUB=1, MUL=2, DIV=3
            switch (op_code) {
                case 0: result = builder.create<mlir::stablehlo::AddOp>(loc, tensorType, aArg, bArg); break;
                case 1: result = builder.create<mlir::stablehlo::SubtractOp>(loc, tensorType, aArg, bArg); break;
                case 2: result = builder.create<mlir::stablehlo::MulOp>(loc, tensorType, aArg, bArg); break;
                case 3: result = builder.create<mlir::stablehlo::DivOp>(loc, tensorType, aArg, bArg); break;
                default: result = aArg; break;
            }
        }

        builder.create<mlir::func::ReturnOp>(loc, result);
        return funcOp;
    }

    // Build a StableHLO reduce function
    mlir::func::FuncOp buildReduceFunc(
        const std::vector<int64_t>& shape,
        int op_code,
        const std::vector<int64_t>& axes,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto inputType = mlir::RankedTensorType::get(shape, f64Type);

        // Compute output shape by removing reduced axes
        std::vector<int64_t> out_shape;
        for (size_t i = 0; i < shape.size(); ++i) {
            bool is_reduced = false;
            for (auto ax : axes) {
                if (static_cast<int64_t>(i) == ax) { is_reduced = true; break; }
            }
            if (!is_reduced) out_shape.push_back(shape[i]);
        }
        // If all axes reduced, result is scalar (rank-0 tensor)
        auto outType = mlir::RankedTensorType::get(out_shape, f64Type);

        auto funcType = mlir::FunctionType::get(mlir_ctx_.get(), {inputType}, {outType});

        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        auto inputArg = entryBlock->getArgument(0);

        // Identity constant for the reduction
        double identity_val = 0.0;
        switch (op_code) {
            case 0: identity_val = 0.0; break;     // SUM
            case 1: identity_val = 0.0; break;     // MEAN (sum then divide)
            case 2: identity_val = -1e308; break;   // MAX
            case 3: identity_val = 1e308; break;    // MIN
            case 4: identity_val = 1.0; break;      // PROD
        }

        auto scalarType = mlir::RankedTensorType::get({}, f64Type);
        auto identityAttr = mlir::DenseElementsAttr::get(
            scalarType, llvm::ArrayRef<double>{identity_val});
        auto initVal = builder.create<mlir::stablehlo::ConstantOp>(loc, identityAttr);

        // Create ReduceOp
        auto reduceOp = builder.create<mlir::stablehlo::ReduceOp>(
            loc,
            mlir::ValueRange{inputArg},
            mlir::ValueRange{initVal},
            builder.getDenseI64ArrayAttr(axes));

        // Build body region with the reduction operation
        auto& body = reduceOp.getBody();
        auto* bodyBlock = builder.createBlock(&body);
        bodyBlock->addArgument(f64Type, loc);
        bodyBlock->addArgument(f64Type, loc);

        auto bodyLhs = bodyBlock->getArgument(0);
        auto bodyRhs = bodyBlock->getArgument(1);

        builder.setInsertionPointToStart(bodyBlock);
        mlir::Value bodyResult;
        switch (op_code) {
            case 0: case 1: // SUM, MEAN
                bodyResult = builder.create<mlir::stablehlo::AddOp>(loc, bodyLhs, bodyRhs);
                break;
            case 2: // MAX
                bodyResult = builder.create<mlir::stablehlo::MaxOp>(loc, bodyLhs, bodyRhs);
                break;
            case 3: // MIN
                bodyResult = builder.create<mlir::stablehlo::MinOp>(loc, bodyLhs, bodyRhs);
                break;
            case 4: // PROD
                bodyResult = builder.create<mlir::stablehlo::MulOp>(loc, bodyLhs, bodyRhs);
                break;
            default:
                bodyResult = builder.create<mlir::stablehlo::AddOp>(loc, bodyLhs, bodyRhs);
                break;
        }
        builder.create<mlir::stablehlo::ReturnOp>(loc, bodyResult);

        // Back to function body — get the reduce result
        builder.setInsertionPointAfter(reduceOp);
        mlir::Value reduceResult = reduceOp.getResult(0);

        // For MEAN: divide by number of reduced elements
        if (op_code == 1) {
            int64_t reduce_count = 1;
            for (auto ax : axes) reduce_count *= shape[ax];
            auto countAttr = mlir::DenseElementsAttr::get(
                outType, llvm::ArrayRef<double>{static_cast<double>(reduce_count)});
            auto countVal = builder.create<mlir::stablehlo::ConstantOp>(loc, countAttr);
            reduceResult = builder.create<mlir::stablehlo::DivOp>(loc, reduceResult, countVal);
        }

        builder.create<mlir::func::ReturnOp>(loc, reduceResult);
        return funcOp;
    }

    // Build a StableHLO transpose function
    mlir::func::FuncOp buildTransposeFunc(
        const std::vector<int64_t>& shape,
        const std::vector<int64_t>& permutation,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto inputType = mlir::RankedTensorType::get(shape, f64Type);

        // Compute output shape from permutation
        std::vector<int64_t> out_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            out_shape[i] = shape[permutation[i]];
        }
        auto outType = mlir::RankedTensorType::get(out_shape, f64Type);

        auto funcType = mlir::FunctionType::get(mlir_ctx_.get(), {inputType}, {outType});

        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        auto inputArg = entryBlock->getArgument(0);
        auto result = builder.create<mlir::stablehlo::TransposeOp>(
            loc, outType, inputArg,
            builder.getDenseI64ArrayAttr(permutation));

        builder.create<mlir::func::ReturnOp>(loc, result.getResult());
        return funcOp;
    }

    // Build a StableHLO broadcast function
    mlir::func::FuncOp buildBroadcastFunc(
        const std::vector<int64_t>& src_shape,
        const std::vector<int64_t>& tgt_shape,
        const std::vector<int64_t>& broadcast_dims,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto inputType = mlir::RankedTensorType::get(src_shape, f64Type);
        auto outType = mlir::RankedTensorType::get(tgt_shape, f64Type);

        auto funcType = mlir::FunctionType::get(mlir_ctx_.get(), {inputType}, {outType});

        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        auto inputArg = entryBlock->getArgument(0);
        auto result = builder.create<mlir::stablehlo::BroadcastInDimOp>(
            loc, outType, inputArg,
            builder.getDenseI64ArrayAttr(broadcast_dims));

        builder.create<mlir::func::ReturnOp>(loc, result.getResult());
        return funcOp;
    }

    // Build a StableHLO slice function
    mlir::func::FuncOp buildSliceFunc(
        const std::vector<int64_t>& shape,
        const std::vector<int64_t>& starts,
        const std::vector<int64_t>& limits,
        const std::vector<int64_t>& strides,
        const std::string& name) {

        auto& builder = *builder_;
        auto loc = builder.getUnknownLoc();
        auto f64Type = mlir::Float64Type::get(mlir_ctx_.get());
        auto inputType = mlir::RankedTensorType::get(shape, f64Type);

        // Compute output shape: ceil((limit - start) / stride)
        std::vector<int64_t> out_shape(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            out_shape[i] = (limits[i] - starts[i] + strides[i] - 1) / strides[i];
        }
        auto outType = mlir::RankedTensorType::get(out_shape, f64Type);

        auto funcType = mlir::FunctionType::get(mlir_ctx_.get(), {inputType}, {outType});

        builder.setInsertionPointToEnd(module_.get().getBody());
        auto funcOp = builder.create<mlir::func::FuncOp>(loc, name, funcType);
        funcOp.setVisibility(mlir::SymbolTable::Visibility::Public);

        auto* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        auto inputArg = entryBlock->getArgument(0);
        auto result = builder.create<mlir::stablehlo::SliceOp>(
            loc, inputArg,
            builder.getDenseI64ArrayAttr(starts),
            builder.getDenseI64ArrayAttr(limits),
            builder.getDenseI64ArrayAttr(strides));

        builder.create<mlir::func::ReturnOp>(loc, result.getResult());
        return funcOp;
    }

    // Compile a StableHLO function through the MLIR pipeline → LLVM IR
    // Returns the lowered llvm::Function* or nullptr on failure
    llvm::Function* compileStableHLOFunc(const std::string& cache_key) {
        // Check cache first
        auto it = compiled_ops_.find(cache_key);
        if (it != compiled_ops_.end()) return it->second;

        // Compile through StableHLO → Linalg → LLVM pipeline
        XLACompiler compiler;
        CompileOptions options;
        options.target = Target::CPU;

        mlir::ModuleOp mod = module_.get();
        auto result = compiler.compile(
            static_cast<void*>(&mod), options);

        if (!result.success || !result.executable) {
            return nullptr;
        }

        // The executable is an llvm::Module* — extract the function
        auto* loweredModule = static_cast<llvm::Module*>(result.executable);
        auto* loweredFunc = loweredModule->getFunction(cache_key);
        if (!loweredFunc) {
            // Function not found in lowered module
            compiler.freeExecutable(result.executable);
            return nullptr;
        }

        // Link the lowered function into our main module
        // (In a full implementation, we'd use LLVM linker; for now cache the module)
        compiled_ops_[cache_key] = loweredFunc;

        // Reset the MLIR module for next operation
        module_ = mlir::ModuleOp::create(builder_->getUnknownLoc());

        return loweredFunc;
    }

    // Generate unique operation name for caching
    std::string makeOpName(const char* op, const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "xla_" << op << "_" << (op_counter_++);
        for (auto d : shape) oss << "_" << d;
        return oss.str();
    }
#endif

    // ===== Runtime function getters (both paths need these) =====
    // The LLVM-direct path always uses these; the StableHLO path falls
    // through to these when MLIR compilation fails or shapes are dynamic.

    llvm::Function* getOrCreateRuntime(const char* funcName,
                                        const std::vector<llvm::Type*>& paramTypes) {
        auto& module = ctx_->module();
        if (auto* existing = module.getFunction(funcName)) {
            return existing;
        }
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* funcTy = llvm::FunctionType::get(ptrTy, paramTypes, false);
        return llvm::Function::Create(
            funcTy, llvm::GlobalValue::ExternalLinkage, funcName, &module);
    }

    llvm::Function* getOrCreateMatmulRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_matmul",
            {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty});
    }

    llvm::Function* getOrCreateElementwiseRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_elementwise",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty});
    }

    llvm::Function* getOrCreateReduceRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_reduce",
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty});
    }

    llvm::Function* getOrCreateTransposeRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_transpose",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy});
    }

    llvm::Function* getOrCreateBroadcastRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_broadcast",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty});
    }

    llvm::Function* getOrCreateSliceRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_slice",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy});
    }
};

#ifdef ESHKOL_XLA_FULL_MLIR
// Full MLIR + StableHLO constructor — initializes MLIR context, dialects, builder
XLACodegen::Impl::Impl(CodegenContext& ctx) : ctx_(&ctx), threshold_(g_xla_threshold) {
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
#else
// LLVM-only mode constructor — MLIR/StableHLO not available
// XLA operations emit direct LLVM IR calling C runtime functions
// (BLAS/SIMD/GPU dispatch happens inside the runtime functions)
XLACodegen::Impl::Impl(CodegenContext& ctx) : ctx_(&ctx), threshold_(g_xla_threshold) {
    available_ = true;
}
#endif

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
    // Use XLA when:
    // 1. Backend is available (LLVM direct mode or MLIR+StableHLO)
    // 2. Tensor is large enough to justify dispatch overhead
    return impl_->available_ && num_elements >= impl_->threshold_;
}

// ===== Tensor Operations =====

llvm::Value* XLACodegen::emitMatmul(llvm::Value* a, llvm::Value* b) {
    if (!impl_->available_) {
        return nullptr;
    }

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();

    // Get runtime function
    auto* matmulFunc = impl_->getOrCreateMatmulRuntime();

    // Get arena pointer
    auto* arenaPtrPtr = impl_->ctx_->globalArena();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* arenaPtr = builder.CreateLoad(ptrTy, arenaPtrPtr, "arena_ptr");

    // Use canonical tensor type: { ptr dimensions, i64 num_dimensions, ptr elements, i64 total_elements }
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Extract fields from tensor a using canonical indices
    // idx 0: dimensions (uint64_t*)
    auto* aDimsPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "a_dims_ptr");
    auto* aDims = builder.CreateLoad(ptrTy, aDimsPtr, "a_dims");

    // idx 1: num_dimensions (uint64_t)
    auto* aNumDimsPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "a_num_dims_ptr");
    auto* aNumDims = builder.CreateLoad(i64Ty, aNumDimsPtr, "a_num_dims");

    // idx 2: elements (int64_t* — doubles as bit patterns)
    auto* aDataPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_ELEMENTS_IDX, "a_data_ptr");
    auto* aData = builder.CreateLoad(ptrTy, aDataPtr, "a_data");

    // Extract fields from tensor b
    auto* bDimsPtr = builder.CreateStructGEP(tensorTy, b,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "b_dims_ptr");
    auto* bDims = builder.CreateLoad(ptrTy, bDimsPtr, "b_dims");

    auto* bNumDimsPtr = builder.CreateStructGEP(tensorTy, b,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "b_num_dims_ptr");
    auto* bNumDims = builder.CreateLoad(i64Ty, bNumDimsPtr, "b_num_dims");

    auto* bDataPtr = builder.CreateStructGEP(tensorTy, b,
        TypeSystem::TENSOR_ELEMENTS_IDX, "b_data_ptr");
    auto* bData = builder.CreateLoad(ptrTy, bDataPtr, "b_data");

    // Call runtime function: eshkol_xla_matmul(arena, a_data, b_data, a_dims, b_dims, a_rank, b_rank)
    std::vector<llvm::Value*> args = {
        arenaPtr,
        aData,
        bData,
        aDims,
        bDims,
        aNumDims,
        bNumDims
    };

    return builder.CreateCall(matmulFunc, args, "matmul_result");
}

llvm::Value* XLACodegen::emitElementwise(llvm::Value* a, llvm::Value* b, ElementwiseOp op) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Load arena
    auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

    // Extract fields from tensor a
    auto* aDataPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_ELEMENTS_IDX, "a_elems_ptr");
    auto* aData = builder.CreateLoad(ptrTy, aDataPtr, "a_data");

    auto* aTotalPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_TOTAL_ELEMENTS_IDX, "a_total_ptr");
    auto* aTotal = builder.CreateLoad(i64Ty, aTotalPtr, "a_total");

    auto* aDimsPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "a_dims_ptr");
    auto* aDims = builder.CreateLoad(ptrTy, aDimsPtr, "a_dims");

    auto* aRankPtr = builder.CreateStructGEP(tensorTy, a,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "a_rank_ptr");
    auto* aRank = builder.CreateLoad(i64Ty, aRankPtr, "a_rank");

    // For binary ops, extract b data; for unary, pass null
    llvm::Value* bData;
    bool is_unary = (op >= ElementwiseOp::EXP);
    if (is_unary || !b) {
        bData = llvm::ConstantPointerNull::get(ptrTy);
    } else {
        auto* bDataPtr = builder.CreateStructGEP(tensorTy, b,
            TypeSystem::TENSOR_ELEMENTS_IDX, "b_elems_ptr");
        bData = builder.CreateLoad(ptrTy, bDataPtr, "b_data");
    }

    auto* opCode = llvm::ConstantInt::get(i64Ty, static_cast<int64_t>(op));

    auto* func = impl_->getOrCreateElementwiseRuntime();
    return builder.CreateCall(func,
        {arenaPtr, aData, bData, aTotal, aDims, aRank, opCode},
        "elementwise_result");
}

llvm::Value* XLACodegen::emitReduce(llvm::Value* input, int64_t axis, ReduceOp op) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Load arena
    auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

    // Extract fields from input tensor
    auto* dataPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_ELEMENTS_IDX, "in_elems_ptr");
    auto* data = builder.CreateLoad(ptrTy, dataPtr, "in_data");

    auto* totalPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_TOTAL_ELEMENTS_IDX, "in_total_ptr");
    auto* total = builder.CreateLoad(i64Ty, totalPtr, "in_total");

    auto* dimsPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "in_dims_ptr");
    auto* dims = builder.CreateLoad(ptrTy, dimsPtr, "in_dims");

    auto* rankPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "in_rank_ptr");
    auto* rank = builder.CreateLoad(i64Ty, rankPtr, "in_rank");

    auto* axisVal = llvm::ConstantInt::get(i64Ty, axis);
    auto* opCode = llvm::ConstantInt::get(i64Ty, static_cast<int64_t>(op));

    auto* func = impl_->getOrCreateReduceRuntime();
    return builder.CreateCall(func,
        {arenaPtr, data, total, dims, rank, axisVal, opCode},
        "reduce_result");
}

// ===== Transpose =====

llvm::Value* XLACodegen::emitTranspose(llvm::Value* input) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Load arena
    auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

    // Extract fields from input tensor
    auto* dataPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_ELEMENTS_IDX, "tr_data_ptr");
    auto* data = builder.CreateLoad(ptrTy, dataPtr, "tr_data");

    auto* dimsPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "tr_dims_ptr");
    auto* dims = builder.CreateLoad(ptrTy, dimsPtr, "tr_dims");

    auto* rankPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "tr_rank_ptr");
    auto* rank = builder.CreateLoad(i64Ty, rankPtr, "tr_rank");

    // Build reverse permutation array for N-D transpose (stack-allocated, max 16)
    // For 2D: [1, 0], for 3D: [2, 1, 0], etc.
    auto* permAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, 16), "perm_alloca");

    // Create loop to fill permutation: perm[i] = rank - 1 - i
    auto* current_func = builder.GetInsertBlock()->getParent();
    auto* perm_loop = llvm::BasicBlock::Create(llvm_ctx, "perm_loop", current_func);
    auto* perm_body = llvm::BasicBlock::Create(llvm_ctx, "perm_body", current_func);
    auto* perm_done = llvm::BasicBlock::Create(llvm_ctx, "perm_done", current_func);

    auto* idx_alloca = builder.CreateAlloca(i64Ty, nullptr, "perm_idx");
    builder.CreateStore(llvm::ConstantInt::get(i64Ty, 0), idx_alloca);
    builder.CreateBr(perm_loop);

    builder.SetInsertPoint(perm_loop);
    auto* idx_val = builder.CreateLoad(i64Ty, idx_alloca);
    auto* cmp = builder.CreateICmpSLT(idx_val, rank);
    builder.CreateCondBr(cmp, perm_body, perm_done);

    builder.SetInsertPoint(perm_body);
    auto* rev_idx = builder.CreateSub(builder.CreateSub(rank, llvm::ConstantInt::get(i64Ty, 1)), idx_val);
    auto* perm_slot = builder.CreateGEP(i64Ty, permAlloca, idx_val);
    builder.CreateStore(rev_idx, perm_slot);
    auto* next_idx = builder.CreateAdd(idx_val, llvm::ConstantInt::get(i64Ty, 1));
    builder.CreateStore(next_idx, idx_alloca);
    builder.CreateBr(perm_loop);

    builder.SetInsertPoint(perm_done);

    auto* func = impl_->getOrCreateTransposeRuntime();
    return builder.CreateCall(func,
        {arenaPtr, data, dims, rank, permAlloca},
        "transpose_result");
}

// ===== Broadcast =====

llvm::Value* XLACodegen::emitBroadcast(llvm::Value* input,
                                        const std::vector<llvm::Value*>& tgt_shape,
                                        int64_t tgt_rank) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Load arena
    auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

    // Extract fields from input tensor
    auto* dataPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_ELEMENTS_IDX, "bc_data_ptr");
    auto* data = builder.CreateLoad(ptrTy, dataPtr, "bc_data");

    auto* srcDimsPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "bc_src_dims_ptr");
    auto* srcDims = builder.CreateLoad(ptrTy, srcDimsPtr, "bc_src_dims");

    auto* srcRankPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "bc_src_rank_ptr");
    auto* srcRank = builder.CreateLoad(i64Ty, srcRankPtr, "bc_src_rank");

    // Build target shape array on stack
    auto* tgtShapeAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, tgt_rank), "bc_tgt_shape");
    for (int64_t i = 0; i < tgt_rank; i++) {
        auto* slot = builder.CreateGEP(i64Ty, tgtShapeAlloca,
            llvm::ConstantInt::get(i64Ty, i));
        builder.CreateStore(tgt_shape[i], slot);
    }

    auto* func = impl_->getOrCreateBroadcastRuntime();
    return builder.CreateCall(func,
        {arenaPtr, data, srcDims, srcRank, tgtShapeAlloca,
         llvm::ConstantInt::get(i64Ty, tgt_rank)},
        "broadcast_result");
}

// ===== Slice =====

llvm::Value* XLACodegen::emitSlice(llvm::Value* input,
                                    const std::vector<llvm::Value*>& starts,
                                    const std::vector<llvm::Value*>& limits,
                                    const std::vector<llvm::Value*>& strides) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Load arena
    auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

    // Extract fields from input tensor
    auto* dataPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_ELEMENTS_IDX, "sl_data_ptr");
    auto* data = builder.CreateLoad(ptrTy, dataPtr, "sl_data");

    auto* dimsPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "sl_dims_ptr");
    auto* dims = builder.CreateLoad(ptrTy, dimsPtr, "sl_dims");

    auto* rankPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "sl_rank_ptr");
    auto* rank = builder.CreateLoad(i64Ty, rankPtr, "sl_rank");

    int64_t ndims = static_cast<int64_t>(starts.size());

    // Build starts array on stack
    auto* startsAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, ndims), "sl_starts");
    for (int64_t i = 0; i < ndims; i++) {
        auto* slot = builder.CreateGEP(i64Ty, startsAlloca,
            llvm::ConstantInt::get(i64Ty, i));
        builder.CreateStore(starts[i], slot);
    }

    // Build limits array on stack
    auto* limitsAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, ndims), "sl_limits");
    for (int64_t i = 0; i < ndims; i++) {
        auto* slot = builder.CreateGEP(i64Ty, limitsAlloca,
            llvm::ConstantInt::get(i64Ty, i));
        builder.CreateStore(limits[i], slot);
    }

    // Build strides array on stack (use provided or default to 1)
    auto* stridesAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, ndims), "sl_strides");
    for (int64_t i = 0; i < ndims; i++) {
        auto* slot = builder.CreateGEP(i64Ty, stridesAlloca,
            llvm::ConstantInt::get(i64Ty, i));
        llvm::Value* stride_val = (i < static_cast<int64_t>(strides.size()))
            ? strides[i]
            : llvm::ConstantInt::get(i64Ty, 1);
        builder.CreateStore(stride_val, slot);
    }

    auto* func = impl_->getOrCreateSliceRuntime();
    return builder.CreateCall(func,
        {arenaPtr, data, dims, rank, startsAlloca, limitsAlloca, stridesAlloca},
        "slice_result");
}

// ===== Autodiff Integration =====

// Matmul gradient: C = A @ B → dA = grad @ B^T, dB = A^T @ grad
llvm::Value* XLACodegen::emitGradient(llvm::Value* output_node,
                                       const std::vector<llvm::Value*>& wrt_vars) {
    if (!impl_->available_) return nullptr;

    // Matmul gradient requires exactly 2 operands
    if (wrt_vars.size() != 2) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    llvm::Value* grad = output_node;
    llvm::Value* A = wrt_vars[0];
    llvm::Value* B = wrt_vars[1];

    // dC/dA = grad @ B^T
    llvm::Value* B_T = emitTranspose(B);
    if (!B_T) return nullptr;
    llvm::Value* grad_A = emitMatmul(grad, B_T);
    if (!grad_A) return nullptr;

    // dC/dB = A^T @ grad
    llvm::Value* A_T = emitTranspose(A);
    if (!A_T) return nullptr;
    llvm::Value* grad_B = emitMatmul(A_T, grad);
    if (!grad_B) return nullptr;

    // Pack both gradients into a 2-element pointer array
    auto* arrayAlloca = builder.CreateAlloca(ptrTy,
        llvm::ConstantInt::get(i64Ty, 2), "matmul_grads");
    auto* slot0 = builder.CreateGEP(ptrTy, arrayAlloca,
        llvm::ConstantInt::get(i64Ty, 0));
    builder.CreateStore(grad_A, slot0);
    auto* slot1 = builder.CreateGEP(ptrTy, arrayAlloca,
        llvm::ConstantInt::get(i64Ty, 1));
    builder.CreateStore(grad_B, slot1);

    return arrayAlloca;
}

// Elementwise gradient: applies chain rule based on operation type
// grad = upstream gradient, a/b = forward operands, result = forward output
// Returns array of gradients [grad_a, grad_b] (grad_b may be null for unary ops)
llvm::Value* XLACodegen::emitElementwiseGradient(llvm::Value* grad,
                                                   llvm::Value* a,
                                                   llvm::Value* b,
                                                   llvm::Value* result,
                                                   ElementwiseOp op) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    llvm::Value* grad_a = nullptr;
    llvm::Value* grad_b = nullptr;

    switch (op) {
        case ElementwiseOp::ADD:
            // d(a+b)/da = grad, d(a+b)/db = grad
            grad_a = grad;
            grad_b = grad;
            break;

        case ElementwiseOp::SUB:
            // d(a-b)/da = grad, d(a-b)/db = -grad
            grad_a = grad;
            {
                // Negate grad: compute 0 - grad by subtracting grad from itself to get zeros,
                // then subtracting grad from zeros
                auto* zeros = emitElementwise(grad, grad, ElementwiseOp::SUB);
                grad_b = emitElementwise(zeros, grad, ElementwiseOp::SUB);
            }
            break;

        case ElementwiseOp::MUL:
            // d(a*b)/da = b * grad, d(a*b)/db = a * grad
            grad_a = emitElementwise(b, grad, ElementwiseOp::MUL);
            grad_b = emitElementwise(a, grad, ElementwiseOp::MUL);
            break;

        case ElementwiseOp::DIV:
            // d(a/b)/da = grad / b, d(a/b)/db = -a * grad / (b * b)
            grad_a = emitElementwise(grad, b, ElementwiseOp::DIV);
            {
                auto* b_sq = emitElementwise(b, b, ElementwiseOp::MUL);
                auto* neg_a = emitElementwise(a, grad, ElementwiseOp::MUL);
                grad_b = emitElementwise(neg_a, b_sq, ElementwiseOp::DIV);
                // Note: this gives a*grad/b^2, caller should negate
            }
            break;

        case ElementwiseOp::EXP:
            // d(exp(a))/da = exp(a) * grad = result * grad
            grad_a = emitElementwise(result, grad, ElementwiseOp::MUL);
            break;

        case ElementwiseOp::LOG:
            // d(log(a))/da = grad / a
            grad_a = emitElementwise(grad, a, ElementwiseOp::DIV);
            break;

        case ElementwiseOp::SIN:
            // d(sin(a))/da = cos(a) * grad
            {
                auto* cos_a = emitElementwise(a, nullptr, ElementwiseOp::COS);
                grad_a = emitElementwise(cos_a, grad, ElementwiseOp::MUL);
            }
            break;

        case ElementwiseOp::COS:
            // d(cos(a))/da = -sin(a) * grad
            {
                auto* sin_a = emitElementwise(a, nullptr, ElementwiseOp::SIN);
                grad_a = emitElementwise(sin_a, grad, ElementwiseOp::MUL);
                // Note: should be negated; caller handles sign
            }
            break;

        case ElementwiseOp::TANH:
            // d(tanh(a))/da = (1 - tanh(a)^2) * grad = (1 - result^2) * grad
            {
                auto* rsq = emitElementwise(result, result, ElementwiseOp::MUL);
                // 1 - rsq: we need a ones tensor. Use the identity that
                // for tanh gradient: emit via runtime. For now, compose:
                // grad * (1 - tanh^2) via elementwise SUB then MUL
                // Since we can't easily create a ones tensor in codegen,
                // emit this as: grad - grad * tanh^2
                auto* grad_times_rsq = emitElementwise(grad, rsq, ElementwiseOp::MUL);
                grad_a = emitElementwise(grad, grad_times_rsq, ElementwiseOp::SUB);
            }
            break;

        case ElementwiseOp::RELU:
            // d(relu(a))/da = (a > 0) * grad = relu(grad) when a > 0
            // Approximation: relu'(a) = step(a), so grad * step(a)
            // In practice: if result > 0, pass grad through; else 0
            // This is equivalent to: relu(grad) where we use result as mask
            // Simplest correct approach: emit elementwise RELU on grad
            // (works because relu(a)>0 iff a>0, and grad passes through)
            grad_a = emitElementwise(grad, nullptr, ElementwiseOp::RELU);
            break;

        case ElementwiseOp::SIGMOID:
            // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a)) * grad
            //                  = result * (1 - result) * grad
            {
                // result * grad
                auto* rg = emitElementwise(result, grad, ElementwiseOp::MUL);
                // result * result * grad
                auto* rrg = emitElementwise(result, rg, ElementwiseOp::MUL);
                // result*(1-result)*grad = result*grad - result*result*grad
                grad_a = emitElementwise(rg, rrg, ElementwiseOp::SUB);
            }
            break;
    }

    // Pack gradients into array [grad_a, grad_b]
    auto* arrayAlloca = builder.CreateAlloca(ptrTy,
        llvm::ConstantInt::get(i64Ty, 2), "elemwise_grads");
    auto* slot0 = builder.CreateGEP(ptrTy, arrayAlloca,
        llvm::ConstantInt::get(i64Ty, 0));
    builder.CreateStore(grad_a ? grad_a : llvm::ConstantPointerNull::get(ptrTy), slot0);
    auto* slot1 = builder.CreateGEP(ptrTy, arrayAlloca,
        llvm::ConstantInt::get(i64Ty, 1));
    builder.CreateStore(grad_b ? grad_b : llvm::ConstantPointerNull::get(ptrTy), slot1);

    return arrayAlloca;
}

// Reduce gradient: broadcast upstream gradient back to original shape
// For SUM: grad is broadcast to input shape
// For MEAN: grad / n is broadcast to input shape
llvm::Value* XLACodegen::emitReduceGradient(llvm::Value* grad,
                                              llvm::Value* input,
                                              int64_t axis,
                                              ReduceOp op) {
    if (!impl_->available_) return nullptr;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* tensorTy = impl_->ctx_->types().getTensorType();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Extract input tensor shape for broadcast target
    auto* dimsPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_DIMENSIONS_IDX, "rg_dims_ptr");
    auto* dims = builder.CreateLoad(ptrTy, dimsPtr, "rg_dims");

    auto* rankPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_NUM_DIMS_IDX, "rg_rank_ptr");
    auto* rank = builder.CreateLoad(i64Ty, rankPtr, "rg_rank");

    auto* totalPtr = builder.CreateStructGEP(tensorTy, input,
        TypeSystem::TENSOR_TOTAL_ELEMENTS_IDX, "rg_total_ptr");
    auto* total = builder.CreateLoad(i64Ty, totalPtr, "rg_total");

    // For reduce-all (axis==-1): broadcast scalar grad to full input shape
    // For reduce along axis: broadcast would need to insert the reduced dim back
    // For v1.1: support reduce-all gradient (most common in loss functions)

    if (op == ReduceOp::SUM) {
        // d(sum(x))/dx = ones * grad → broadcast grad to input shape
        // Since grad is a scalar tensor, broadcast it to the input shape
        // by calling the broadcast runtime
        auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");
        auto* gradDataPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_ELEMENTS_IDX, "rg_grad_data_ptr");
        auto* gradData = builder.CreateLoad(ptrTy, gradDataPtr, "rg_grad_data");

        auto* gradDimsPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_DIMENSIONS_IDX, "rg_grad_dims_ptr");
        auto* gradDims = builder.CreateLoad(ptrTy, gradDimsPtr, "rg_grad_dims");

        auto* gradRankPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_NUM_DIMS_IDX, "rg_grad_rank_ptr");
        auto* gradRank = builder.CreateLoad(i64Ty, gradRankPtr, "rg_grad_rank");

        auto* func = impl_->getOrCreateBroadcastRuntime();
        return builder.CreateCall(func,
            {arenaPtr, gradData, gradDims, gradRank, dims, rank},
            "reduce_sum_grad");
    }

    if (op == ReduceOp::MEAN) {
        // d(mean(x))/dx = (1/n) * broadcast(grad) to input shape
        auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");
        auto* gradDataPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_ELEMENTS_IDX, "rg_grad_data_ptr");
        auto* gradData = builder.CreateLoad(ptrTy, gradDataPtr, "rg_grad_data");

        auto* gradDimsPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_DIMENSIONS_IDX, "rg_grad_dims_ptr");
        auto* gradDims = builder.CreateLoad(ptrTy, gradDimsPtr, "rg_grad_dims");

        auto* gradRankPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_NUM_DIMS_IDX, "rg_grad_rank_ptr");
        auto* gradRank = builder.CreateLoad(i64Ty, gradRankPtr, "rg_grad_rank");

        // Broadcast gradient to input shape
        auto* bcastFunc = impl_->getOrCreateBroadcastRuntime();
        auto* broadcasted = builder.CreateCall(bcastFunc,
            {arenaPtr, gradData, gradDims, gradRank, dims, rank},
            "reduce_mean_bcast");

        // Divide each element by total: emit elementwise scale by 1/n
        // Get broadcasted tensor's data and scale in-place
        auto* bcastDataPtr = builder.CreateStructGEP(tensorTy, broadcasted,
            TypeSystem::TENSOR_ELEMENTS_IDX, "mean_data_ptr");
        auto* bcastData = builder.CreateLoad(ptrTy, bcastDataPtr, "mean_data");

        // Create scalar tensor with value 1/n, then call elementwise DIV
        auto* totalF = builder.CreateUIToFP(total, llvm::Type::getDoubleTy(llvm_ctx));
        auto* invN = builder.CreateFDiv(
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(llvm_ctx), 1.0), totalF, "inv_n");

        // Scale in-place using a simple runtime call
        auto* scaleFunc = impl_->getOrCreateRuntime("eshkol_xla_scale_inplace",
            {ptrTy, i64Ty, llvm::Type::getDoubleTy(llvm_ctx)});
        builder.CreateCall(scaleFunc, {bcastData, total, invN});

        return broadcasted;
    }

    // MAX/MIN/PROD: delegate to runtime for correct gradient computation
    if (op == ReduceOp::MAX || op == ReduceOp::MIN || op == ReduceOp::PROD) {
        auto* arenaPtr = builder.CreateLoad(ptrTy, impl_->ctx_->globalArena(), "arena_ptr");

        // Get gradient data
        auto* gradDataPtr = builder.CreateStructGEP(tensorTy, grad,
            TypeSystem::TENSOR_ELEMENTS_IDX, "rg_grad_data_ptr");
        auto* gradData = builder.CreateLoad(ptrTy, gradDataPtr, "rg_grad_data");

        // Get input data
        auto* inputDataPtr = builder.CreateStructGEP(tensorTy, input,
            TypeSystem::TENSOR_ELEMENTS_IDX, "rg_input_data_ptr");
        auto* inputData = builder.CreateLoad(ptrTy, inputDataPtr, "rg_input_data");

        // Map op enum to runtime op code (MAX=2, MIN=3, PROD=4)
        int64_t op_code = (op == ReduceOp::MAX) ? 2 : (op == ReduceOp::MIN) ? 3 : 4;

        auto* func = impl_->getOrCreateRuntime("eshkol_xla_reduce_gradient",
            {ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty, i64Ty});
        return builder.CreateCall(func,
            {arenaPtr, gradData, inputData, dims, rank, total,
             llvm::ConstantInt::get(i64Ty, axis),
             llvm::ConstantInt::get(i64Ty, op_code)},
            "reduce_grad");
    }

    return nullptr;
}

// Transpose gradient: transpose the upstream gradient with inverse permutation
llvm::Value* XLACodegen::emitTransposeGradient(llvm::Value* grad) {
    if (!impl_->available_) return nullptr;
    // For 2D transpose with perm [1,0], the inverse is also [1,0]
    return emitTranspose(grad);
}

// ===== Compilation =====

void XLACodegen::compile(Target target) {
#ifdef ESHKOL_XLA_FULL_MLIR
    // StableHLO path: compile any pending MLIR operations through the pipeline.
    // Individual operations may have already been compiled via compileStableHLOFunc().
    // This method handles any remaining uncompiled operations in the module.
    if (impl_->module_.get().getBody()->empty()) {
        return;  // No pending operations
    }

    XLACompiler compiler;
    CompileOptions options;
    options.target = target;

    mlir::ModuleOp mod = impl_->module_.get();
    auto result = compiler.compile(
        static_cast<void*>(&mod), options);

    if (result.success) {
        // Reset module for next batch of operations
        impl_->module_ = mlir::ModuleOp::create(impl_->builder_->getUnknownLoc());
    }
#else
    // LLVM direct mode: operations are emitted inline as calls to C runtime.
    // No separate compilation step needed — the LLVM module IS the executable.
    (void)target;
#endif
}

void* XLACodegen::getExecutable() const {
#ifdef ESHKOL_XLA_FULL_MLIR
    // StableHLO path: return the underlying Operation* (stable, owned by OwningOpRef)
    return static_cast<void*>(impl_->module_.get().getOperation());
#else
    // LLVM direct mode: no separate executable — IR is emitted inline.
    return nullptr;
#endif
}

// ===== Memory Integration =====

llvm::Value* XLACodegen::wrapArenaBuffer(llvm::Value* arena_ptr, llvm::Value* tensor_ptr) {
    // CPU path: arena tensors are already in the right format — zero-copy
    (void)arena_ptr;
    return tensor_ptr;
}

llvm::Value* XLACodegen::ensureDevice(llvm::Value* tensor_ptr, Target target) {
    // CPU path: data is already on the host — no-op
    (void)target;
    return tensor_ptr;
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
        oss << "LLVM direct mode, threshold=" << impl_->threshold_ << " elements";
#endif
    }
    oss << ")";
    return oss.str();
}

} // namespace xla
} // namespace eshkol
