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
#include <cstdlib>
#include <sstream>

// MLIR includes (conditional compilation)
// Only include when BOTH MLIR and StableHLO are available
// (no point initializing MLIR without StableHLO for actual XLA ops)
#if defined(ESHKOL_MLIR_AVAILABLE) && defined(ESHKOL_STABLEHLO_AVAILABLE)
#define ESHKOL_XLA_FULL_MLIR 1
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif

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

#ifdef ESHKOL_XLA_FULL_MLIR
    // Full MLIR + StableHLO available - enable XLA
    std::unique_ptr<mlir::MLIRContext> mlir_ctx_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::OwningOpRef<mlir::ModuleOp> module_;

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

// ===== Tensor Operations (Stubs) =====

llvm::Value* XLACodegen::emitMatmul(llvm::Value* a, llvm::Value* b) {
    // STUB: Not implemented yet
    (void)a;
    (void)b;
    return nullptr;
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
