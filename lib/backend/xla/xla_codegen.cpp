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
#include "eshkol/backend/type_system.h"
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
    // LLVM-only mode — MLIR/StableHLO not available
    // XLA operations emit direct LLVM IR calling C runtime functions
    // (BLAS/SIMD/GPU dispatch happens inside the runtime functions)
    explicit Impl(CodegenContext& ctx) : ctx_(&ctx), threshold_(g_xla_threshold) {
        available_ = true;
    }

    // Get or create runtime function declarations (LLVM-only path)
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

    // void* eshkol_xla_elementwise(void* arena, void* a_data, void* b_data,
    //   i64 total_elements, void* shape, i64 rank, i64 op_code)
    llvm::Function* getOrCreateElementwiseRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_elementwise",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty});
    }

    // void* eshkol_xla_reduce(void* arena, void* data, i64 total_elements,
    //   void* shape, i64 rank, i64 axis, i64 op_code)
    llvm::Function* getOrCreateReduceRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_reduce",
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty});
    }

    // void* eshkol_xla_transpose(void* arena, void* data, void* shape, i64 rank, void* perm)
    llvm::Function* getOrCreateTransposeRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_transpose",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy});
    }

    // void* eshkol_xla_broadcast(void* arena, void* data, void* src_shape,
    //   i64 src_rank, void* tgt_shape, i64 tgt_rank)
    llvm::Function* getOrCreateBroadcastRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_broadcast",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty});
    }

    // void* eshkol_xla_slice(void* arena, void* data, void* shape,
    //   i64 rank, void* starts, void* limits, void* strides)
    llvm::Function* getOrCreateSliceRuntime() {
        auto* ptrTy = llvm::PointerType::get(ctx_->context(), 0);
        auto* i64Ty = llvm::Type::getInt64Ty(ctx_->context());
        return getOrCreateRuntime("eshkol_xla_slice",
            {ptrTy, ptrTy, ptrTy, i64Ty, ptrTy, ptrTy, ptrTy});
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

    // Build permutation array [1, 0] for 2D transpose (stack-allocated)
    auto* permAlloca = builder.CreateAlloca(i64Ty,
        llvm::ConstantInt::get(i64Ty, 2), "perm_alloca");
    auto* perm0 = builder.CreateGEP(i64Ty, permAlloca,
        llvm::ConstantInt::get(i64Ty, 0));
    builder.CreateStore(llvm::ConstantInt::get(i64Ty, 1), perm0);
    auto* perm1 = builder.CreateGEP(i64Ty, permAlloca,
        llvm::ConstantInt::get(i64Ty, 1));
    builder.CreateStore(llvm::ConstantInt::get(i64Ty, 0), perm1);

    auto* func = impl_->getOrCreateTransposeRuntime();
    return builder.CreateCall(func,
        {arenaPtr, data, dims, rank, permAlloca},
        "transpose_result");
}

// ===== Autodiff Integration =====

llvm::Value* XLACodegen::emitGradient(llvm::Value* output_node,
                                       const std::vector<llvm::Value*>& wrt_vars) {
    if (!impl_->available_) return nullptr;

    // Matmul gradient: C = A @ B
    //   dC/dA = grad @ B^T
    //   dC/dB = A^T @ grad
    // Convention: output_node = upstream gradient tensor (grad)
    //             wrt_vars[0] = A (left operand)
    //             wrt_vars[1] = B (right operand)
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
    // Caller loads [0] for dC/dA and [1] for dC/dB
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

// ===== Compilation =====

void XLACodegen::compile(Target target) {
    // LLVM direct mode: operations are emitted inline as calls to C runtime.
    // No separate compilation step needed — the LLVM module IS the executable.
    // MLIR path (when available): would lower StableHLO → LLVM here.
    (void)target;
}

void* XLACodegen::getExecutable() const {
    // LLVM direct mode: no separate executable — IR is emitted inline.
    return nullptr;
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
